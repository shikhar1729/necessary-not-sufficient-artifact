"""No-hint (neutral) traces: each main-task question with the reference note removed, one answer per model.

Input: --data-dir/traces/{source}_traces.jsonl (the W prompts). Output: --data-dir/neutral_accuracy/
{model}_neutral_traces.jsonl (read by main_task.py no_hint_diagnostics, analysis/noninternal_baselines.py and
analysis/no_hint_diagnostic_lenient.py)
and neutral_accuracy_by_dataset.csv. Backend "hf" (local GPU) or "openrouter"; the paper's run did not record which.
Run: python -m scripts.pipeline.no_hint_traces --data-dir <store> --models mistral_7b gemma_9b llama_8b
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from scripts.pipeline.common.config import load_project_config
from src.core.normalize import answers_match, parse_response
from src.core.slice import NOTE_BLOCK_RE

DATASET_ORDER = ["triviaqa", "hotpotqa", "gpqa", "mmlu_pro"]
OPENROUTER_MODEL_IDS = {
    "mistral": "mistralai/mistral-7b-instruct-v0.1",
    "gemma": "google/gemma-2-9b-it",
    "llama": "meta-llama/llama-3.1-8b-instruct",
}


def strip_note_block(prompt: str) -> str:
    """Remove the reference-note block from a W/G/C prompt."""
    neutral = NOTE_BLOCK_RE.sub("\n\n", prompt, count=1)
    return re.sub(r"\n{3,}", "\n\n", neutral)


def infer_answer_type(dataset: str) -> str:
    return "mcq" if dataset in {"gpqa", "mmlu_pro"} else "span"


def load_questions_from_traces(data_dir: Path, source_model: str) -> list[dict]:
    """One neutral prompt per unique question, built from the source model's W traces."""
    trace_path = data_dir / "traces" / f"{source_model}_traces.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing source trace file: {trace_path}")

    seen = set()
    rows: list[dict] = []
    with trace_path.open() as f:
        for line in f:
            trace = json.loads(line)
            if trace.get("condition") != "W":
                continue
            key = (trace["dataset"], str(trace["question_id"]))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "dataset": trace["dataset"],
                    "question_id": str(trace["question_id"]),
                    "split": trace.get("split", ""),
                    "answer_type": infer_answer_type(trace["dataset"]),
                    "gold_answer": trace["gold_answer"],
                    "gold_aliases": [trace["gold_answer"]],
                    "prompt": strip_note_block(trace["raw_prompt"]),
                }
            )

    rows.sort(key=lambda row: (DATASET_ORDER.index(row["dataset"]), row["question_id"]))
    return rows


def augment_triviaqa_aliases(records: list[dict]) -> None:
    """Score TriviaQA against all of its aliases when the dataset can be loaded."""
    trivia_ids = {row["question_id"] for row in records if row["dataset"] == "triviaqa"}
    if not trivia_ids:
        return
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets is not installed; using canonical TriviaQA answers only")
        return
    try:
        ds = load_dataset("trivia_qa", "rc.nocontext", split="train")
    except Exception as exc:
        print(f"Could not load TriviaQA aliases ({exc}); using canonical answers only")
        return

    alias_map = {}
    for row in ds:
        qid = str(row.get("question_id", ""))
        if qid in trivia_ids:
            alias_map[qid] = row["answer"].get("aliases", [])
    for row in records:
        if row["dataset"] == "triviaqa" and alias_map.get(row["question_id"]):
            row["gold_aliases"] = alias_map[row["question_id"]]


def parse_and_score(row: dict, response: str) -> dict:
    parsed = parse_response(response, row["answer_type"])
    is_correct = parsed.final_answer_norm is not None and answers_match(
        parsed.final_answer_norm, row["gold_answer"], row.get("gold_aliases")
    )
    return {
        "dataset": row["dataset"],
        "question_id": row["question_id"],
        "split": row["split"],
        "answer_type": row["answer_type"],
        "gold_answer": row["gold_answer"],
        "raw_prompt": row["prompt"],
        "raw_response": response,
        "final_answer_raw": parsed.final_answer_raw,
        "normalized_final": parsed.final_answer_norm,
        "parse_ok": parsed.parse_mode != "fail",
        "parse_mode": parsed.parse_mode,
        "is_correct": bool(is_correct),
    }


def load_done_keys(out_path: Path) -> set[tuple[str, str]]:
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                row = json.loads(line)
                done.add((row["dataset"], str(row["question_id"])))
    return done


def generate_for_model_hf(model_cfg, records: list[dict], output_dir: Path) -> None:
    import torch

    from scripts.pipeline.common.model_loader import ChatFormatter, load_model

    out_path = output_dir / f"{model_cfg.short_name}_neutral_traces.jsonl"
    done = load_done_keys(out_path)
    remaining = [row for row in records if (row["dataset"], str(row["question_id"])) not in done]
    print(f"[{model_cfg.short_name}] {len(remaining)} neutral examples to generate ({len(done)} done)")
    if not remaining:
        return

    model, tokenizer = load_model(model_cfg)
    formatter = ChatFormatter(tokenizer, model_cfg)
    dec = model_cfg.canonical_decoding

    with out_path.open("a") as f:
        for idx, row in enumerate(remaining, start=1):
            input_ids = formatter.format(row["prompt"])
            do_sample = dec.temperature > 0
            gen_kwargs = {
                "max_new_tokens": dec.max_new_tokens,
                "do_sample": do_sample,
                "top_p": dec.top_p,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = dec.temperature
            with torch.no_grad():
                output_ids = model.generate(input_ids, **gen_kwargs)
            response = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True).strip()
            out = parse_and_score(row, response)
            out.update({"model": model_cfg.short_name, "backend": "hf", "generation_model": model_cfg.model_id})
            f.write(json.dumps(out) + "\n")
            f.flush()
            if idx % 25 == 0:
                print(f"[{model_cfg.short_name}] {idx}/{len(remaining)}")


def generate_for_model_openrouter(
    model_cfg, records: list[dict], output_dir: Path, max_workers: int, max_retries: int = 4
) -> None:
    from openai import OpenAI
    from tqdm import tqdm

    from scripts.pipeline.common.model_loader import chat_messages

    short_name = model_cfg.short_name
    openrouter_model = OPENROUTER_MODEL_IDS.get(short_name)
    if not openrouter_model:
        raise ValueError(f"No OpenRouter model mapping for {short_name}")
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    out_path = output_dir / f"{short_name}_neutral_traces.jsonl"
    done = load_done_keys(out_path)
    remaining = [row for row in records if (row["dataset"], str(row["question_id"])) not in done]
    print(f"[{short_name}] {len(remaining)} neutral examples to generate via OpenRouter ({len(done)} done)")
    if not remaining:
        return

    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    dec = model_cfg.canonical_decoding

    def _call(row: dict) -> dict:
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=openrouter_model,
                    messages=chat_messages(model_cfg, row["prompt"]),
                    temperature=dec.temperature,
                    top_p=dec.top_p,
                    max_tokens=dec.max_new_tokens,
                )
                out = parse_and_score(row, (response.choices[0].message.content or "").strip())
                out.update(
                    {
                        "model": short_name,
                        "backend": "openrouter",
                        "generation_model": openrouter_model,
                        "output_name": short_name,
                    }
                )
                return out
            except Exception:
                if attempt == max_retries - 1:
                    raise
                time.sleep((2 ** attempt) + random.uniform(0, 1))
        raise RuntimeError("unreachable")

    with out_path.open("a") as f:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_call, row) for row in remaining]
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{short_name} neutral"):
                f.write(json.dumps(future.result()) + "\n")
                f.flush()


def summarize(output_dir: Path) -> None:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(output_dir.glob("*_neutral_traces.jsonl")):
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                grouped[(row["model"], row["dataset"])].append(row)

    csv_path = output_dir / "neutral_accuracy_by_dataset.csv"
    with csv_path.open("w") as f:
        f.write("model,dataset,n,parsed,correct,accuracy,parse_rate\n")
        for (model, dataset), items in sorted(grouped.items()):
            n = len(items)
            parsed = sum(bool(row["parse_ok"]) for row in items)
            correct = sum(bool(row["is_correct"]) for row in items)
            accuracy = correct / n if n else 0.0
            parse_rate = parsed / n if n else 0.0
            f.write(f"{model},{dataset},{n},{parsed},{correct},{accuracy:.6f},{parse_rate:.6f}\n")
    print(f"wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--data-dir", required=True, help="trace store with traces/")
    parser.add_argument("--models", nargs="+", default=["mistral_7b", "gemma_9b", "llama_8b"])
    parser.add_argument("--source-model", default="mistral", help="trace file whose W prompts are used")
    parser.add_argument(
        "--backend",
        choices=["auto", "openrouter", "hf"],
        default="auto",
        help="auto uses HF on CUDA and OpenRouter otherwise",
    )
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = data_dir / "neutral_accuracy"
    output_dir.mkdir(parents=True, exist_ok=True)

    project_cfg = load_project_config(Path(args.config))
    records = load_questions_from_traces(data_dir, args.source_model)
    augment_triviaqa_aliases(records)
    print(f"loaded {len(records)} neutral prompts from {args.source_model} W traces")

    backend = args.backend
    if backend in ("auto", "hf"):
        import torch

        if backend == "auto":
            backend = "hf" if torch.cuda.is_available() else "openrouter"
        elif not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for HF generation; use --backend openrouter.")
    print(f"using backend={backend}")

    for model_name in args.models:
        model_cfg = project_cfg.models[model_name]
        if backend == "openrouter":
            generate_for_model_openrouter(model_cfg, records, output_dir, args.max_workers)
        else:
            generate_for_model_hf(model_cfg, records, output_dir)

    summarize(output_dir)


if __name__ == "__main__":
    main()
