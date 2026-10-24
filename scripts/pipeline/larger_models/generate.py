"""Run a larger or reasoning model on the main task's existing W/G/C prompts (same questions, candidates, wording).

Input: --data-dir/traces/{source}_traces.jsonl; samples --questions-per-dataset questions per dataset (seed 2026).
Output: traces/{model}_traces.jsonl and features/ in the same store, read by larger_models/analyze.py and the judges.
The defaults give the DeepSeek-R1-Distill 200-question pilot's question selection; the Qwen/QwQ pilot settings were
not recorded. Decoding comes from models.yaml: DeepSeek samples at temperature 0.6 and no torch seed is set, so its
traces cannot be regenerated exactly. The source script's override flags (such as --temperature-override) are not
ported; whether the pilots used them was not recorded.
Run: python -m scripts.pipeline.larger_models.generate --data-dir <store> --model deepseek_r1_8b
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from scripts.pipeline.common.config import load_project_config
from src.core.types import ConditionRecord

MCQ_DATASETS = {"gpqa", "mmlu_pro"}


def infer_answer_type(dataset: str) -> str:
    return "mcq" if dataset in MCQ_DATASETS else "span"


def extract_question(raw_prompt: str) -> str:
    """Question text for metadata only; generation uses the stored raw_prompt."""
    marker = "\n\nReference note from an earlier pass:"
    return raw_prompt.split(marker, 1)[0].replace("Question: ", "", 1).strip()


def load_existing_records(
    data_dir: Path,
    source_model: str,
    datasets: list[str],
    questions_per_dataset: int | None,
    seed: int,
) -> list[ConditionRecord]:
    trace_path = data_dir / "traces" / f"{source_model}_traces.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing source traces: {trace_path}")

    by_key: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    with trace_path.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("dataset") not in datasets or row.get("condition") not in {"W", "G", "C"}:
                continue
            by_key[(row["dataset"], str(row["question_id"]))][row["condition"]] = row

    rng = random.Random(seed)
    selected_keys: list[tuple[str, str]] = []
    for dataset in datasets:
        keys = sorted(key for key, conds in by_key.items() if key[0] == dataset and {"W", "G", "C"}.issubset(conds))
        rng.shuffle(keys)
        if questions_per_dataset is not None:
            keys = keys[:questions_per_dataset]
        selected_keys.extend(sorted(keys))

    records: list[ConditionRecord] = []
    for key in selected_keys:
        for condition in ["W", "G", "C"]:
            row = by_key[key][condition]
            records.append(
                ConditionRecord(
                    dataset=row["dataset"],
                    question_id=str(row["question_id"]),
                    question=extract_question(row["raw_prompt"]),
                    gold_answer=row["gold_answer"],
                    gold_aliases=[row["gold_answer"]],
                    answer_type=infer_answer_type(row["dataset"]),
                    condition=condition,
                    candidate=row["candidate"],
                    alt_wrong_candidate=row.get("alt_wrong_candidate"),
                    prompt=row["raw_prompt"],
                    template_id=row.get("template_id", "note_block_v1"),
                    split=row.get("split", ""),
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--data-dir", required=True, help="trace store with the source model's traces/")
    parser.add_argument("--source-model", default="mistral")
    parser.add_argument("--model", required=True, help="model key in models.yaml, e.g. deepseek_r1_8b, qwen3_14b")
    parser.add_argument("--datasets", nargs="+", default=["triviaqa", "hotpotqa", "gpqa", "mmlu_pro"])
    parser.add_argument("--questions-per-dataset", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    from scripts.pipeline.common.trace_generation import generate_traces_for_model

    model_cfg = load_project_config(Path(args.config)).models[args.model]
    data_dir = Path(args.data_dir)
    records = load_existing_records(data_dir, args.source_model, args.datasets, args.questions_per_dataset, args.seed)
    n_questions = len({(r.dataset, r.question_id) for r in records})
    print(f"Running {model_cfg.short_name} on {len(records)} traces ({n_questions} questions, source={args.source_model})", flush=True)
    generate_traces_for_model(
        model_cfg=model_cfg,
        records=records,
        layers=[],
        output_dir=data_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
