"""Per-row out-of-fold TF-IDF text-baseline scores on the retained slice, four text views (tab:monitor-intervals):
the nested GroupKFold pipeline of main_task.text_discriminative_baselines. Input: the trace store (--trace-root).
Output: --out-dir/tfidf_oof_predictions.csv (released as data/scores/monitors/tfidf_oof.csv) and fold choices.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.pipeline.main_task import (
    build_main_task_records,
    make_text,
    nested_text_cv,
)

DATA_DIR = Path()  # set from --trace-root in main()
OUT_DIR = Path("outputs/monitors")
MODELS = ["mistral", "gemma", "llama"]
VIEWS = ["answer_only", "cot_answer", "prompt_answer", "prompt_cot_answer"]


def main() -> None:
    global DATA_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", type=Path, required=True, help="trace store with traces/{model}_traces.jsonl")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    DATA_DIR, OUT_DIR = args.trace_root, args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    fold_rows = []
    for model in MODELS:
        records = build_main_task_records(model, DATA_DIR)
        labels = pd.Series([int(r["label"]) for r in records])
        groups = pd.Series([r["group"] for r in records])
        print(f"{model}: n={len(records)} dep={int(labels.sum())}")
        for view in VIEWS:
            texts = [make_text(r, view) for r in records]
            probs, used, folds = nested_text_cv(texts, labels.to_numpy(), groups.to_numpy())
            folds["model"] = model
            folds["view"] = view
            fold_rows.append(folds)
            for rec, p, u in zip(records, probs, used):
                if not u:
                    continue
                rows.append(
                    {
                        "model": model,
                        "view": view,
                        "dataset": rec["dataset"],
                        "question_id": rec["question_id"],
                        "group": rec["group"],
                        "label": int(rec["label"]),
                        "score": float(p),
                    }
                )
            sub = pd.DataFrame(
                [
                    {"label": int(r["label"]), "score": float(p)}
                    for r, p, u in zip(records, probs, used)
                    if u
                ]
            )
            print(
                f"  {view:18s} n={len(sub)} auroc={roc_auc_score(sub.label, sub.score):.4f}"
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "tfidf_oof_predictions.csv", index=False)
    pd.concat(fold_rows, ignore_index=True).to_csv(OUT_DIR / "tfidf_oof_folds.csv", index=False)
    print(f"wrote {len(out)} rows -> {OUT_DIR / 'tfidf_oof_predictions.csv'}")


if __name__ == "__main__":
    main()
