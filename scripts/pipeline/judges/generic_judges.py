"""Generic single-trace LLM judges (answer only; CoT and answer) on the retained slice; they never see the note.

Inputs: --data-dir traces, the judges.yaml views, OPENROUTER_API_KEY. Outputs: judge_{view}_{model}.csv and
generic_judge_summary.csv; the paper's Sonnet-4 values are the judge_answer_only and judge_cot_answer rows of
data/scores/controls/strict_rule_summary.csv (per-row scores not kept). data/scores/monitors/haiku35_judge_rows.csv
comes from a claude-3.5-haiku run on the lenient-rule slice (352/326/196); this script's strict-slice loader does
not give those rows, and that run's code is not included.
Run: python -m scripts.pipeline.judges.generic_judges --data-dir <store>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.common.judge_monitor import judge_batch
from src.core.normalize import answers_match
from src.core.slice import extract_task_text


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error with equal-width bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() / len(y_true) * abs(y_prob[mask].mean() - y_true[mask].mean())
    return ece


def compute_all_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    results = {}
    try:
        results["auroc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        results["auroc"] = 0.5
    try:
        results["auprc"] = average_precision_score(y_true, y_prob)
    except ValueError:
        results["auprc"] = y_true.mean()
    results["brier"] = brier_score_loss(y_true, y_prob)
    results["ece"] = compute_ece(y_true, y_prob)
    return results


def load_retained_examples(traces_path: Path) -> list[dict]:
    """Retained W traces (W answers h1, not gold; G parses) with the dependence label."""
    by_q = defaultdict(dict)
    with open(traces_path) as f:
        for line in f:
            d = json.loads(line)
            by_q[(d.get("dataset"), str(d["question_id"]))][d["condition"]] = d

    examples = []
    for (dataset, qid), conds in by_q.items():
        w = conds.get("W")
        g = conds.get("G")
        if not w or not g or not w["parse_ok"] or not g["parse_ok"]:
            continue
        w_ans = w["normalized_final"]
        g_ans = g["normalized_final"]
        h1 = w["candidate"]
        if not w_ans or not answers_match(w_ans, h1):
            continue
        if answers_match(w_ans, w["gold_answer"]):
            continue
        if not g_ans:
            continue
        dep = not answers_match(g_ans, h1)

        raw = w["raw_response"]
        fa_idx = raw.lower().rfind("final answer:")
        reasoning = raw[:fa_idx].strip() if fa_idx > 0 else raw

        examples.append({
            "question_id": f"{dataset}::{qid}",
            "dataset": dataset,
            "raw_question_id": qid,
            "question": w.get("question") or extract_task_text(w.get("raw_prompt", "")),
            "reasoning_text": reasoning,
            "final_answer": w_ans,
            "label_dep": 1 if dep else 0,
            "condition": "W",
            "candidate": h1,
            "gold_answer": w["gold_answer"],
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--data-dir", required=True, help="trace store with traces/")
    parser.add_argument("--output-dir", default="outputs/generic_judges")
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-4", help="overrides judges.yaml")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set; cannot run LLM judges.")
    cfg = load_project_config(args.config)
    cfg.judge.model_id = args.judge_model
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Judge model: {cfg.judge.model_id} (max_workers={cfg.judge.max_workers}, temperature={cfg.judge.temperature})")

    all_summaries = []
    for model in args.models:
        traces_path = data_dir / f"traces/{model}_traces.jsonl"
        if not traces_path.exists():
            print(f"Skipping {model}")
            continue
        examples = load_retained_examples(traces_path)
        labels = np.array([e["label_dep"] for e in examples])
        print(f"\n{model}: {len(examples)} retained examples (dep={labels.sum()}, indep={len(labels)-labels.sum()})")
        if len(examples) < 10:
            print("Not enough examples, skipping")
            continue

        for view_name in ["answer_only", "cot_answer"]:
            view = cfg.judge_views[view_name]
            results = judge_batch(cfg.judge, view, examples)
            result_map = {r.question_id: r for r in results}

            matched_labels, matched_probs, matched_details = [], [], []
            for e in examples:
                r = result_map.get(e["question_id"])
                if r:
                    matched_labels.append(e["label_dep"])
                    matched_probs.append(r.probability / 100.0)
                    matched_details.append({
                        "question_id": e["question_id"],
                        "dataset": e["dataset"],
                        "raw_question_id": e["raw_question_id"],
                        "question": e["question"],
                        "label_dep": e["label_dep"],
                        "judge_prob": r.probability,
                        "judge_verdict": r.binary,
                        "judge_reason": r.reason,
                        "model": model,
                        "view": view_name,
                    })
            if len(matched_labels) < 5:
                print(f"Only {len(matched_labels)} matched, skipping")
                continue

            y = np.array(matched_labels)
            metrics = compute_all_metrics(y, np.array(matched_probs))
            print(f"  {view_name}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")
            all_summaries.append({
                "model": model,
                "method": f"judge_{view_name}",
                "auroc": metrics["auroc"],
                "auprc": metrics["auprc"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "n_total": len(y),
                "n_pos": int(y.sum()),
                "n_neg": int(len(y) - y.sum()),
            })
            pd.DataFrame(matched_details).to_csv(output_dir / f"judge_{view_name}_{model}.csv", index=False)

    if all_summaries:
        df = pd.DataFrame(all_summaries)
        df.to_csv(output_dir / "generic_judge_summary.csv", index=False)
        print(df[["model", "method", "auroc", "auprc", "n_total"]].to_string(index=False))


if __name__ == "__main__":
    main()
