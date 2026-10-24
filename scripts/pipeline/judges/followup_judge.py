"""CoT + follow-up LLM judge (the CoT+FU column): CoT, answer and the F1 reply, on retained rows with a follow-up.

Inputs: --data-dir with traces/ and judges/{model}_followups_*.jsonl (followups.py), OPENROUTER_API_KEY. Outputs:
judge_cot_followup_{model}.csv and judge_followup_results.csv; the paper's values (N = 346/313/196) are the
judge_cot_followup rows of data/scores/controls/strict_rule_summary.csv, judge family unresolved (see the paper).
Run: python -m scripts.pipeline.judges.followup_judge --data-dir <store>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.common.judge_monitor import judge_batch
from scripts.pipeline.judges.generic_judges import compute_all_metrics
from src.core.normalize import answers_match
from src.core.slice import extract_task_text


def load_followup_examples(data_dir: Path, model: str) -> list[dict]:
    """Retained rows that have a follow-up; labels are recomputed from W/G, follow-ups only add text."""
    followups = {}
    for path in data_dir.glob(f"judges/{model}_followups_*.jsonl"):
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                followups[(d.get("dataset"), str(d["question_id"]))] = d
    if not followups:
        return []

    by_q = defaultdict(dict)
    with open(data_dir / f"traces/{model}_traces.jsonl") as f:
        for line in f:
            d = json.loads(line)
            by_q[(d.get("dataset"), str(d["question_id"]))][d["condition"]] = d

    examples = []
    for (dataset, qid), fu in followups.items():
        conds = by_q.get((dataset, qid), {})
        w = conds.get("W")
        g = conds.get("G")
        if not w or not g or not w.get("parse_ok") or not g.get("parse_ok"):
            continue
        h1 = w.get("candidate")
        w_ans = w.get("normalized_final")
        g_ans = g.get("normalized_final")
        if not w_ans or not answers_match(w_ans, h1):
            continue
        if answers_match(w_ans, w.get("gold_answer")):
            continue
        if not g_ans:
            continue
        dep = int(not answers_match(g_ans, h1))

        raw = w["raw_response"]
        fa_idx = raw.lower().rfind("final answer:")
        reasoning = raw[:fa_idx].strip() if fa_idx > 0 else raw

        examples.append({
            "question_id": f"{dataset}::{qid}",
            "dataset": dataset,
            "raw_question_id": qid,
            "question": w.get("question") or extract_task_text(w.get("raw_prompt", "")),
            "reasoning_text": reasoning,
            "final_answer": w.get("normalized_final", ""),
            "followup_response": fu.get("f1_response", ""),
            "label_dep": dep,
        })
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="trace store with traces/ and judges/")
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--output-dir", default="outputs/followup_judge")
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-4", help="overrides judges.yaml")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is not set; cannot run LLM judges.")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_project_config(args.config)
    cfg.judge.model_id = args.judge_model
    view = cfg.judge_views["cot_followup_answer"]
    print(f"Judge model: {cfg.judge.model_id} (max_workers={cfg.judge.max_workers}, temperature={cfg.judge.temperature})")

    all_results = []
    for model in args.models:
        examples = load_followup_examples(data_dir, model)
        if not examples:
            print(f"{model}: no follow-up data")
            continue
        labels = np.array([e["label_dep"] for e in examples])
        print(f"\n{model}: {len(examples)} examples (dep={labels.sum()}, indep={len(labels)-labels.sum()})")

        result_map = {r.question_id: r for r in judge_batch(cfg.judge, view, examples)}
        ml, mp, details = [], [], []
        for e in examples:
            r = result_map.get(e["question_id"])
            if r:
                ml.append(e["label_dep"])
                mp.append(r.probability / 100.0)
                details.append({
                    "question_id": e["question_id"], "dataset": e["dataset"], "raw_question_id": e["raw_question_id"],
                    "question": e["question"], "label_dep": e["label_dep"],
                    "judge_prob": r.probability, "model": model, "view": "cot_followup_answer",
                })

        if len(ml) >= 5:
            m = compute_all_metrics(np.array(ml), np.array(mp))
            print(f"  Judge (CoT+follow-up): AUROC={m['auroc']:.4f}")
            all_results.append({"model": model, "method": "judge_cot_followup", "auroc": m["auroc"], "n": len(ml)})
            pd.DataFrame(details).to_csv(output_dir / f"judge_cot_followup_{model}.csv", index=False)

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(output_dir / "judge_followup_results.csv", index=False)
        print(f"\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
