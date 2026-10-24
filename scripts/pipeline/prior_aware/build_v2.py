"""Prior-aware v2 rebuild (tab:prior-aware): an OpenRouter model proposes wrong candidates for TriviaQA/HotpotQA
(OPENROUTER_API_KEY; sampled at temperature 0.7, so a rerun gives a different pool), each model scores them by
no-hint log-probability, and the pair with the smallest prior gap (at most 0.25, among the --pair-top-k most likely
candidates) fills W/G/C/neutral prompts; traces and prompt-end states follow (GPU, HF_TOKEN). Output: candidate_pool,
candidate_prior_scores, selected_prior_matched_pairs.jsonl, conditions/, traces/, features/. The recorded pool holds
2,997 unique span questions, more than the default --items-per-dataset 1000 allows, so the run used a larger value
(at least 1500); the value was not recorded. Llama's traces were finished with the optional vLLM backend (vllm 0.23.0,
run with VLLM_WORKER_MULTIPROC_METHOD=spawn). Commands, otherwise at the defaults:

    python -m scripts.pipeline.prior_aware.build_v2 --output-dir <v2> --build-candidates --score-priors --build-conditions \
        --items-per-dataset <not recorded>
    python -m scripts.pipeline.prior_aware.build_v2 --output-dir <v2> --generate-traces --models mistral_7b gemma_9b
    VLLM_WORKER_MULTIPROC_METHOD=spawn python -m scripts.pipeline.prior_aware.build_v2 --output-dir <v2> \
        --generate-traces --models llama_8b --backend vllm
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.common.model_loader import DTYPE_MAP, format_chat, check_hf_access
from scripts.pipeline.dataset.prepare_hotpotqa import prepare_hotpotqa
from scripts.pipeline.dataset.prepare_triviaqa import prepare_triviaqa
from scripts.pipeline.prior_aware.batched_generation import (
    build_trace,
    done_keys,
    generate_with_transformers,
    generate_with_vllm,
    load_condition_records,
    load_jsonl_rows as load_jsonl,
    save_prompt_end_hidden_states,
    stable_int,
)
from src.core.normalize import answers_match, normalize_answer
from src.core.types import ConditionRecord, DatasetItem


DATASET_LOADERS = {
    "triviaqa": prepare_triviaqa,
    "hotpotqa": prepare_hotpotqa,
}


def dedupe_question_rows(rows: list[dict[str, Any]], *, min_candidates: int | None = None) -> list[dict[str, Any]]:
    """Keep one row per dataset/question key (the pool file is appended to across resumed runs)."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        if min_candidates is not None and len(row.get("wrong_candidates") or []) < min_candidates:
            continue
        key = (row["dataset"], str(row["question_id"]))
        current = best.get(key)
        row_score = (
            len(row.get("wrong_candidates") or []),
            1 if row.get("generation_error") is None else 0,
            -idx,
        )
        if current is None:
            best[key] = {**row, "_dedupe_score": row_score}
            continue
        if row_score > current["_dedupe_score"]:
            best[key] = {**row, "_dedupe_score": row_score}
    out = []
    for row in best.values():
        row.pop("_dedupe_score", None)
        out.append(row)
    return sorted(out, key=lambda r: (r["dataset"], str(r["question_id"])))


def dedupe_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        key = (row["model"], row["dataset"], str(row["question_id"]))
        scores = row.get("wrong_candidate_scores") or []
        finite = sum(math.isfinite(float(s.get("prior_logp", float("nan")))) for s in scores)
        row_score = (finite, len(scores), -idx)
        current = best.get(key)
        if current is None or row_score > current["_dedupe_score"]:
            best[key] = {**row, "_dedupe_score": row_score}
    out = []
    for row in best.values():
        row.pop("_dedupe_score", None)
        out.append(row)
    return sorted(out, key=lambda r: (r["model"], r["dataset"], str(r["question_id"])))


def get_openrouter_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required for candidate generation")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1", timeout=60.0)


def clean_candidate(text: str) -> str:
    text = text.strip().strip("-*•·")
    text = text.strip()
    while text and text[0].isdigit():
        text = text[1:].lstrip(".): \t")
    return text.strip().strip("\"'`")


def valid_wrong_candidate(candidate: str, item: DatasetItem, max_words: int) -> bool:
    if not candidate or len(candidate) > 80:
        return False
    if len(candidate.split()) > max_words:
        return False
    if answers_match(candidate, item.gold_answer, item.gold_aliases):
        return False
    norm = normalize_answer(candidate)
    if not norm:
        return False
    for alias in item.gold_aliases:
        if normalize_answer(alias) == norm:
            return False
    return True


def generate_candidates_for_item(
    client: OpenAI,
    item: DatasetItem,
    n_candidates: int,
    model: str,
    max_retries: int,
    max_words: int,
) -> tuple[list[str], str | None]:
    aliases = ", ".join(item.gold_aliases[:8])
    prompt = (
        f"Generate {n_candidates + 4} plausible but incorrect answers to this question.\n"
        "Each answer should be a short span, not a sentence. "
        "Make the wrong answers plausible competitors to the correct answer, "
        "not jokes and not obviously impossible.\n"
        "Reply with one answer per line and no explanation.\n\n"
        f"Question: {item.question}\n"
        f"Correct answer: {item.gold_answer}\n"
        f"Also avoid aliases/variants: {aliases}\n"
    )
    last_error: str | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=220,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.choices[0].message.content or ""
            out: list[str] = []
            seen: set[str] = set()
            for raw_line in text.splitlines():
                candidate = clean_candidate(raw_line)
                norm = normalize_answer(candidate)
                if norm in seen:
                    continue
                if valid_wrong_candidate(candidate, item, max_words):
                    out.append(candidate)
                    seen.add(norm)
                if len(out) >= n_candidates:
                    break
            return out, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2.0 * (attempt + 1))
    return [], last_error


def load_span_items(args: argparse.Namespace) -> list[DatasetItem]:
    cfg = load_project_config(Path(args.config))
    items: list[DatasetItem] = []
    for dataset in args.datasets:
        if dataset not in DATASET_LOADERS:
            raise ValueError(f"Prior-aware rebuild is span-only; unsupported dataset={dataset}")
        ds_cfg = cfg.datasets[dataset]
        ds_cfg = replace(ds_cfg, pilot_size=args.items_per_dataset, main_size=None)
        items.extend(DATASET_LOADERS[dataset](ds_cfg, seed=args.seed))
    return items


def build_candidate_pool(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "candidate_pool.jsonl"
    done = {
        (row["dataset"], str(row["question_id"]))
        for row in load_jsonl(pool_path)
        if len(row.get("wrong_candidates") or []) >= args.min_candidates
    }
    items = [
        item for item in load_span_items(args)
        if (item.dataset, str(item.question_id)) not in done
    ]
    print(f"candidate generation: {len(items)} remaining ({len(done)} already done)", flush=True)
    if not items:
        return

    client = get_openrouter_client()

    def task(item: DatasetItem) -> dict[str, Any]:
        candidates, error = generate_candidates_for_item(
            client,
            item,
            args.n_candidates,
            args.candidate_model,
            args.max_retries,
            args.max_candidate_words,
        )
        return {
            "dataset": item.dataset,
            "question_id": str(item.question_id),
            "question": item.question,
            "gold_answer": item.gold_answer,
            "gold_aliases": item.gold_aliases,
            "answer_type": item.answer_type,
            "wrong_candidates": candidates,
            "generation_error": error,
            "metadata": item.metadata,
        }

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor, pool_path.open("a") as f:
        futures = [executor.submit(task, item) for item in items]
        for future in tqdm(as_completed(futures), total=len(futures), desc="candidate_pool"):
            row = future.result()
            f.write(json.dumps(row) + "\n")
            f.flush()


def answer_variants(answer: str) -> list[str]:
    answer = str(answer or "").strip()
    if not answer:
        return []
    return list(dict.fromkeys([answer, f" {answer}"]))


@torch.inference_mode()
def score_answer(model, tokenizer, prompt_text: str, answer: str) -> float:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    if not prompt_ids:
        return float("-inf")
    device = next(model.parameters()).device
    best = float("-inf")
    for variant in answer_variants(answer):
        answer_ids = tokenizer(variant, add_special_tokens=False).input_ids
        if not answer_ids:
            continue
        full = torch.tensor([prompt_ids + answer_ids], device=device)
        out = model(input_ids=full[:, :-1], use_cache=False)
        log_probs = F.log_softmax(out.logits[0].float(), dim=-1)
        token_scores = []
        start = len(prompt_ids)
        for offset, target_id in enumerate(answer_ids):
            pred_pos = start + offset - 1
            if 0 <= pred_pos < log_probs.shape[0]:
                token_scores.append(float(log_probs[pred_pos, target_id].item()))
        if token_scores:
            best = max(best, float(np.mean(token_scores)))
    return best


def load_done_prior_scores(path: Path) -> set[tuple[str, str, str]]:
    done = set()
    for row in load_jsonl(path):
        done.add((row["model"], row["dataset"], str(row["question_id"])))
    return done


def score_candidate_priors(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_pool_rows = load_jsonl(output_dir / "candidate_pool.jsonl")
    pool_rows = dedupe_question_rows(raw_pool_rows, min_candidates=args.min_candidates)
    skipped = len(raw_pool_rows) - len(pool_rows)
    if skipped:
        print(f"candidate pool dedupe/filter skipped={skipped} raw rows", flush=True)
    score_path = output_dir / "candidate_prior_scores.jsonl"
    done = load_done_prior_scores(score_path)
    cfg = load_project_config(Path(args.config))
    neutral_template = cfg.prompt_templates["neutral"]

    for model_key in args.models:
        model_cfg = cfg.models[model_key]
        short = model_cfg.short_name
        rows = [
            row for row in pool_rows
            if (short, row["dataset"], str(row["question_id"])) not in done
        ]
        print(f"[{short}] prior scoring rows={len(rows)}", flush=True)
        if not rows:
            continue
        hf_token = os.environ.get("HF_TOKEN")
        check_hf_access(model_cfg.model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.model_id,
            torch_dtype=DTYPE_MAP.get(model_cfg.dtype, torch.bfloat16),
            device_map="auto",
            token=hf_token,
        )
        model.eval()
        with score_path.open("a") as f:
            for idx, row in enumerate(rows, start=1):
                neutral_prompt = neutral_template.template.format(question=row["question"])
                prompt_text = format_chat(tokenizer, model_cfg, neutral_prompt)
                if args.prior_answer_prefix:
                    prompt_text = prompt_text + args.prior_answer_prefix
                scores = []
                for candidate in row["wrong_candidates"]:
                    scores.append(
                        {
                            "candidate": candidate,
                            "prior_logp": score_answer(model, tokenizer, prompt_text, candidate),
                        }
                    )
                out = {
                    "model": short,
                    "model_key": model_key,
                    "dataset": row["dataset"],
                    "question_id": str(row["question_id"]),
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "gold_aliases": row["gold_aliases"],
                    "answer_type": row["answer_type"],
                    "wrong_candidate_scores": scores,
                    "prior_answer_prefix": args.prior_answer_prefix,
                }
                f.write(json.dumps(out) + "\n")
                f.flush()
                done.add((short, row["dataset"], str(row["question_id"])))
                if idx % 50 == 0:
                    print(f"[{short}] {idx}/{len(rows)}", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


def best_matched_pair(
    scores: list[dict[str, Any]],
    seed_parts: tuple[Any, ...],
    pair_top_k: int,
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    valid = [
        row for row in scores
        if math.isfinite(float(row.get("prior_logp", float("nan"))))
    ]
    valid = sorted(valid, key=lambda row: float(row["prior_logp"]), reverse=True)
    if pair_top_k > 0:
        valid = valid[:pair_top_k]
    if len(valid) < 2:
        return None
    best = None
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            gap = abs(float(valid[i]["prior_logp"]) - float(valid[j]["prior_logp"]))
            mean_prior = 0.5 * (float(valid[i]["prior_logp"]) + float(valid[j]["prior_logp"]))
            # Match by prior gap first, then prefer candidates that are both
            # plausible under the no-note model.
            objective = (gap, -mean_prior)
            if best is None or objective < best[0]:
                best = (objective, gap, valid[i], valid[j])
    assert best is not None
    _, gap, a, b = best
    if stable_int(*seed_parts) % 2:
        a, b = b, a
    return a, b, float(gap)


def build_condition_records(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    cfg = load_project_config(Path(args.config))
    note_template = cfg.prompt_templates["note_block_v1"]
    neutral_template = cfg.prompt_templates["neutral"]
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)
    selected_rows = []
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in dedupe_score_rows(load_jsonl(output_dir / "candidate_prior_scores.jsonl")):
        if len(row.get("wrong_candidate_scores") or []) < args.min_candidates:
            continue
        by_model.setdefault(row["model"], []).append(row)

    for model, rows in sorted(by_model.items()):
        condition_path = condition_dir / f"{model}_conditions.jsonl"
        with condition_path.open("w") as f:
            n_selected = 0
            for row in rows:
                pair = best_matched_pair(
                    row["wrong_candidate_scores"],
                    (args.seed, model, row["dataset"], row["question_id"]),
                    args.pair_top_k,
                )
                if pair is None:
                    continue
                h1_row, h2_row, prior_gap = pair
                if args.max_prior_gap is not None and prior_gap > args.max_prior_gap:
                    continue
                h1 = h1_row["candidate"]
                h2 = h2_row["candidate"]
                common = {
                    "dataset": row["dataset"],
                    "question_id": str(row["question_id"]),
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "gold_aliases": row["gold_aliases"],
                    "answer_type": "span",
                    "options": None,
                    "options_formatted": "",
                    "metadata": {
                        "prior_aware": True,
                        "h1_prior_logp": h1_row["prior_logp"],
                        "h2_prior_logp": h2_row["prior_logp"],
                        "prior_abs_gap": prior_gap,
                        "candidate_pool_size": len(row["wrong_candidate_scores"]),
                        "pair_top_k": args.pair_top_k,
                    },
                }
                for condition, candidate, alt, template in [
                    ("W", h1, h2, note_template),
                    ("G", h2, h1, note_template),
                    ("C", row["gold_answer"], None, note_template),
                    ("neutral", "", None, neutral_template),
                ]:
                    prompt = (
                        template.template.format(question=row["question"], candidate=candidate)
                        if condition != "neutral"
                        else template.template.format(question=row["question"])
                    )
                    record = ConditionRecord(
                        **common,
                        condition=condition,
                        candidate=candidate,
                        alt_wrong_candidate=alt,
                        prompt=prompt,
                        template_id=template.id,
                    )
                    f.write(json.dumps(record.__dict__) + "\n")
                selected_rows.append(
                    {
                        "model": model,
                        "dataset": row["dataset"],
                        "question_id": str(row["question_id"]),
                        "h1": h1,
                        "h2": h2,
                        "h1_prior_logp": h1_row["prior_logp"],
                        "h2_prior_logp": h2_row["prior_logp"],
                        "prior_abs_gap": prior_gap,
                        "candidate_pool_size": len(row["wrong_candidate_scores"]),
                        "pair_top_k": args.pair_top_k,
                    }
                )
                n_selected += 1
        print(f"[{model}] selected prior-aware pairs={n_selected} -> {condition_path}", flush=True)

    with (output_dir / "selected_prior_matched_pairs.jsonl").open("w") as f:
        for row in selected_rows:
            f.write(json.dumps(row) + "\n")


def generate_traces(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    cfg = load_project_config(Path(args.config))
    for model_key in args.models:
        model_cfg = cfg.models[model_key]
        if args.max_new_tokens_override is not None:
            model_cfg.canonical_decoding.max_new_tokens = args.max_new_tokens_override
        short = model_cfg.short_name
        condition_path = output_dir / "conditions" / f"{short}_conditions.jsonl"
        records = load_condition_records(condition_path)
        trace_path = output_dir / "traces" / f"{short}_traces.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        existing = done_keys(trace_path)
        records = [
            record for record in records
            if (record.dataset, str(record.question_id), record.condition) not in existing
        ]
        print(f"[{short}] trace generation records={len(records)} ({len(existing)} already done)", flush=True)
        if not records:
            continue
        if args.backend == "vllm":
            responses = generate_with_vllm(records, model_cfg, args.generate_batch_size)
        else:
            responses = generate_with_transformers(records, model_cfg, args.generate_batch_size)
        hs_paths = save_prompt_end_hidden_states(
            records,
            model_cfg,
            output_dir,
            args.extract_batch_size,
        )
        with trace_path.open("a") as f:
            for record, response, hs_path in zip(records, responses, hs_paths):
                trace = build_trace(record, model_cfg, response, hs_path, args.seed)
                f.write(json.dumps(trace.to_dict()) + "\n")
        print(f"[{short}] wrote traces -> {trace_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--output-dir", default="outputs/prior_aware_v2")
    parser.add_argument("--datasets", nargs="+", default=["triviaqa", "hotpotqa"])
    parser.add_argument("--items-per-dataset", type=int, default=1000)
    parser.add_argument("--models", nargs="+", default=["mistral_7b", "gemma_9b", "llama_8b"],
                        help="config keys in data/prompts/models.yaml (the other prior_aware scripts take short names)")
    parser.add_argument("--candidate-model", default="openai/gpt-4.1-mini")
    parser.add_argument("--n-candidates", type=int, default=6)
    parser.add_argument("--min-candidates", type=int, default=4)
    parser.add_argument("--max-candidate-words", type=int, default=6)
    parser.add_argument("--max-workers", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-prior-gap", type=float, default=0.25)
    parser.add_argument("--pair-top-k", type=int, default=4)
    parser.add_argument("--prior-answer-prefix", default="\nFinal answer:")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--max-new-tokens-override", type=int, default=None)
    parser.add_argument("--generate-batch-size", type=int, default=64)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--build-candidates", action="store_true")
    parser.add_argument("--score-priors", action="store_true")
    parser.add_argument("--build-conditions", action="store_true")
    parser.add_argument("--generate-traces", action="store_true")
    args = parser.parse_args()

    if not any([args.build_candidates, args.score_priors, args.build_conditions, args.generate_traces]):
        parser.error("Specify at least one phase")
    if args.build_candidates:
        build_candidate_pool(args)
    if args.score_priors:
        score_candidate_priors(args)
    if args.build_conditions:
        build_condition_records(args)
    if args.generate_traces:
        generate_traces(args)


if __name__ == "__main__":
    main()
