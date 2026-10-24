"""Select the adaptive-v4 balanced slices (tab:adaptive-v4) from prescreen.py output: per model, the
dataset/template group with the largest minority class, then equal numbers of dependent and independent retained
pair directions (seeded shuffle, at most --target-per-class each). Output: selected_pairs.csv and conditions/.

    python -m scripts.pipeline.prior_aware.select_slices --source-dir <v4> --output-dir <v4_selected> --target-per-class 100
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.pipeline.prior_aware.batched_generation import truthy, write_jsonl
from src.core.slice import load_jsonl


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def choose_group(rows: list[dict[str, str]]) -> tuple[str, str, str]:
    counts: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    for row in rows:
        if not truthy(row.get("retained")):
            continue
        key = (row["model"], row["dataset"], row["template_id"])
        counts[key][row["label"]] += 1
    if not counts:
        raise ValueError("No retained rows found")
    return max(
        counts,
        key=lambda key: (
            min(counts[key]["dependent"], counts[key]["independent"]),
            counts[key]["dependent"] + counts[key]["independent"],
            key,
        ),
    )


def select_for_model(
    rows: list[dict[str, str]],
    *,
    model: str,
    target_per_class: int | None,
    seed: int,
) -> list[dict[str, str]]:
    candidates = [
        row for row in rows
        if row["model"] == model
        and truthy(row.get("retained"))
        and row["label"] in {"dependent", "independent"}
    ]
    if not candidates:
        return []
    chosen_model, chosen_dataset, chosen_template = choose_group(candidates)
    candidates = [
        row for row in candidates
        if row["model"] == chosen_model
        and row["dataset"] == chosen_dataset
        and row["template_id"] == chosen_template
    ]
    rng = random.Random(seed)
    by_label = {
        label: [row for row in candidates if row["label"] == label]
        for label in ["dependent", "independent"]
    }
    for label_rows in by_label.values():
        rng.shuffle(label_rows)
    n = min(len(by_label["dependent"]), len(by_label["independent"]))
    if target_per_class is not None:
        n = min(n, target_per_class)
    selected = by_label["dependent"][:n] + by_label["independent"][:n]
    selected = sorted(selected, key=lambda row: (row["dataset"], row["question_id"], row["label"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True, help="prescreen.py output")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"], choices=["mistral", "gemma", "llama"])
    parser.add_argument("--target-per-class", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    pair_rows = load_csv(args.source_dir / "analysis" / "balanced_v3_prescreen_pairs.csv")
    selected_rows: list[dict[str, Any]] = []
    selected_qids_by_model: dict[str, set[str]] = defaultdict(set)
    for model in args.models:
        model_rows = select_for_model(
            pair_rows,
            model=model,
            target_per_class=args.target_per_class,
            seed=args.seed,
        )
        counts = Counter(row["label"] for row in model_rows)
        print(
            f"[{model}] selected={len(model_rows)} "
            f"dep={counts['dependent']} indep={counts['independent']}",
            flush=True,
        )
        for row in model_rows:
            row = {**row, "selected_for_hidden_extraction": True}
            selected_rows.append(row)
            selected_qids_by_model[model].add(row["question_id"])

    for model in args.models:
        conditions = load_jsonl(args.source_dir / "conditions" / f"{model}_conditions.jsonl")
        keep = [
            row for row in conditions
            if str(row["question_id"]) in selected_qids_by_model[model]
            and row["condition"] in {"W", "G", "N"}
        ]
        write_jsonl(args.output_dir / "conditions" / f"{model}_conditions.jsonl", keep)
        print(f"[{model}] wrote {len(keep)} selected conditions", flush=True)
    write_csv(args.output_dir / "selected_pairs.csv", selected_rows)


if __name__ == "__main__":
    main()
