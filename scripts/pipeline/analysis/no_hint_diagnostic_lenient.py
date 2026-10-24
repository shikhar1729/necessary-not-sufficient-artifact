"""No-hint prior diagnostic on the lenient-rule slice (tab:no-hint-diagnostic; results: 41/68/63% vs 12/18/19%).

Lenient-rule slice: the lenient-rule main-task builder (stored label_behavior == "hint_follow", containment matching also for
option letters, empty G answer counted as dependent): 352/326/196 rows. Rows with a parsed no-hint trace (no_hint_traces.py)
give N = 349/316/188. Output: no_hint_diagnostic_{summary,examples}.csv, same columns as main_task.py; the printed
AUROC is 1 - prior_h1_auroc (independent direction). Run: python -m scripts.pipeline.analysis.no_hint_diagnostic_lenient --data-dir <store>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.pipeline.main_task import cluster_bootstrap_metric
from src.core.normalize import normalize_answer
from src.core.slice import load_jsonl


def lenient_answers_match(answer: str, gold: str, aliases: list[str] | None = None) -> bool:
    """The lenient-rule matcher: exact or containment either way, option letters included."""
    norm_answer = normalize_answer(answer)
    if not norm_answer:
        return False
    targets = [gold]
    if aliases:
        targets.extend(aliases)
    for target in targets:
        norm_target = normalize_answer(target)
        if not norm_target:
            continue
        if norm_answer == norm_target:
            return True
        if norm_answer in norm_target or norm_target in norm_answer:
            return True
    return False


def build_lenient_records(model: str, data_dir: Path) -> list[dict[str, Any]]:
    """The lenient-rule main-task selection (its feature-file check is dropped; every row had states)."""
    by_q: dict[str, dict[str, dict]] = defaultdict(dict)
    with open(data_dir / "traces" / f"{model}_traces.jsonl") as f:
        for line in f:
            d = json.loads(line)
            by_q[d["question_id"]][d["condition"]] = d

    records = []
    for qid, conds in by_q.items():
        w = conds.get("W")
        g = conds.get("G")
        if not w or not g or not w["parse_ok"] or not g["parse_ok"]:
            continue
        w_ans = w["normalized_final"]
        g_ans = g["normalized_final"]
        h1 = w["candidate"]
        if not w_ans or not lenient_answers_match(w_ans, h1):
            continue
        if w["label_behavior"] != "hint_follow":
            continue
        if lenient_answers_match(w_ans, w["gold_answer"]):
            continue
        dep = not lenient_answers_match(g_ans, h1) if g_ans else True
        records.append(
            {
                "model": model,
                "dataset": w["dataset"],
                "question_id": str(qid),
                "group": f"{w['dataset']}::{qid}",
                "label": int(dep),
                "h1": h1,
            }
        )
    return records


def no_hint_diagnostics(all_records: dict[str, list[dict]], data_dir: Path, n_bootstrap: int, seed: int, output_dir: Path) -> pd.DataFrame:
    """Same computation as main_task.no_hint_diagnostics, with the lenient matcher."""
    summary_rows = []
    detail_rows = []
    for model, records in all_records.items():
        neutral_path = data_dir / "neutral_accuracy" / f"{model}_neutral_traces.jsonl"
        if not neutral_path.exists():
            continue
        neutral = {(row["dataset"], str(row["question_id"])): row for row in load_jsonl(neutral_path)}
        rows = []
        for record in records:
            nrow = neutral.get((record["dataset"], record["question_id"]))
            if not nrow or not nrow.get("parse_ok"):
                continue
            item = {
                "model": model,
                "dataset": record["dataset"],
                "question_id": record["question_id"],
                "label": record["label"],
                "no_hint_answer": nrow.get("normalized_final"),
                "no_hint_correct": int(bool(nrow.get("is_correct"))),
                "no_hint_matches_h1": int(lenient_answers_match(nrow.get("normalized_final"), record["h1"])),
            }
            detail_rows.append(item)
            rows.append(item)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        y = df["label"].to_numpy(dtype=int)
        score = df["no_hint_matches_h1"].to_numpy(dtype=float)
        groups = np.array([f"{r['dataset']}::{r['question_id']}" for r in rows])
        metric = cluster_bootstrap_metric(y, score, groups, "auroc", n_bootstrap, seed)
        for label_name, label_value in [("dependent", 1), ("independent", 0), ("all", None)]:
            sub = df if label_value is None else df[df["label"] == label_value]
            summary_rows.append(
                {
                    "model": model,
                    "subset": label_name,
                    "n": len(sub),
                    "no_hint_h1_rate": float(sub["no_hint_matches_h1"].mean()) if len(sub) else np.nan,
                    "no_hint_correct_rate": float(sub["no_hint_correct"].mean()) if len(sub) else np.nan,
                    "prior_h1_auroc": metric["point"] if label_value is None else np.nan,
                    "prior_h1_auroc_ci_low": metric["ci_low"] if label_value is None else np.nan,
                    "prior_h1_auroc_ci_high": metric["ci_high"] if label_value is None else np.nan,
                }
            )
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["model", "subset"])
    summary.to_csv(output_dir / "no_hint_diagnostic_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "no_hint_diagnostic_examples.csv", index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="trace store with traces/ and neutral_accuracy/")
    parser.add_argument("--output-dir", default="outputs/no_hint_diagnostic_lenient")
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_records = {model: build_lenient_records(model, data_dir) for model in args.models}
    for model, records in all_records.items():
        n_dep = sum(r["label"] for r in records)
        print(f"{model}: lenient slice {len(records)} ({n_dep}/{len(records) - n_dep})")
    print(no_hint_diagnostics(all_records, data_dir, args.n_bootstrap, args.seed, output_dir).to_string(index=False))


if __name__ == "__main__":
    main()
