"""Nested prompt-end probe for one larger or reasoning model (tab:larger-models-full), pooled and per dataset.

Input: --data-dir with traces/{model}_traces.jsonl and features/ (larger_models/generate.py). Outputs
(--output-dir/<model>): funnel.csv, retained_records.csv, nested_summary.csv, nested_folds.csv,
nested_oof_predictions.csv, per_dataset_nested_summary.csv. The judge columns come from judges/generic_judges.py and
judges/debate_judge.py run on the same store. The DeepSeek pilot used --n-bootstrap 500 (seed 19 is fixed);
the Qwen/QwQ settings were not recorded.
Run: python -m scripts.pipeline.larger_models.analyze --model deepseek --data-dir <store> --n-bootstrap 500
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.pipeline.main_task import (
    C_GRID,
    cluster_bootstrap_metric,
    fit_dense_predict,
    load_state_matrix,
    resolve_feature_path,
)
from src.core.slice import build_main_task_records, load_jsonl


def summarize_funnel(traces: list[dict[str, Any]]) -> pd.DataFrame:
    """Per condition: rows, parsed rows and the stored behavioural label counts."""
    by_model_condition = defaultdict(list)
    for row in traces:
        by_model_condition[(row.get("model"), row.get("condition"))].append(row)
    rows = []
    for (model, condition), rows_for_key in sorted(by_model_condition.items()):
        behavior = Counter(row.get("label_behavior") for row in rows_for_key)
        rows.append(
            {
                "model": model,
                "condition": condition,
                "n": len(rows_for_key),
                "parse_ok": sum(bool(row.get("parse_ok")) for row in rows_for_key),
                "hint_follow": behavior.get("hint_follow", 0),
                "resist": behavior.get("resist", 0),
                "other_error": behavior.get("other_error", 0),
            }
        )
    return pd.DataFrame(rows)


def choose_valid_group_splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray, max_splits: int) -> int:
    """Largest GroupKFold count with both classes in every test fold (small slices)."""
    upper = min(max_splits, len(np.unique(groups)))
    for n_splits in range(upper, 1, -1):
        if all(len(np.unique(y[test_idx])) >= 2 for _, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups)):
            return n_splits
    return 2


def inner_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray, train_idx: np.ndarray, c_value: float) -> float:
    n_splits = choose_valid_group_splits(x[train_idx], y[train_idx], groups[train_idx], max_splits=3)
    if n_splits < 2:
        return 0.5
    scores = []
    for inner_train_rel, inner_val_rel in GroupKFold(n_splits=n_splits).split(x[train_idx], y[train_idx], groups[train_idx]):
        inner_train = train_idx[inner_train_rel]
        inner_val = train_idx[inner_val_rel]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        probs = fit_dense_predict(x[inner_train], y[inner_train], x[inner_val], c_value)
        scores.append(roc_auc_score(y[inner_val], probs))
    return float(np.mean(scores)) if scores else 0.5


def nested_cv(x_by_layer, layers, y, groups) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Outer GroupKFold (up to 5) picks the layer and C on inner folds of its training questions."""
    outer = GroupKFold(n_splits=choose_valid_group_splits(x_by_layer[layers[0]], y, groups, max_splits=5))
    probs = np.zeros(len(y), dtype=float)
    used = np.zeros(len(y), dtype=bool)
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(x_by_layer[layers[0]], y, groups)):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        best = (-np.inf, layers[0], 1.0)
        for layer in layers:
            for c_value in C_GRID:
                score = inner_score(x_by_layer[layer], y, groups, train_idx, c_value)
                if score > best[0]:
                    best = (score, layer, c_value)
        inner_auroc, layer, c_value = best
        probs[test_idx] = fit_dense_predict(
            x_by_layer[layer][train_idx], y[train_idx], x_by_layer[layer][test_idx], c_value
        )
        used[test_idx] = True
        fold_rows.append(
            {
                "fold": fold,
                "selected_layer": layer,
                "selected_c": c_value,
                "inner_auroc": inner_auroc,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "n_test_dep": int(y[test_idx].sum()),
                "n_test_indep": int(len(test_idx) - y[test_idx].sum()),
            }
        )
    return pd.DataFrame(fold_rows), probs, used


def summarize_nested_records(model: str, records: list[dict], data_dir: Path, n_bootstrap: int, seed: int, min_records: int = 20):
    if len(records) < min_records or len({int(row["label"]) for row in records}) < 2:
        return None, None, None
    layers, x_by_layer, _ = load_state_matrix(records, "w", data_dir)
    y = np.array([row["label"] for row in records], dtype=int)
    groups = np.array([row["group"] for row in records])
    fold_df, probs, used = nested_cv(x_by_layer, layers, y, groups)
    if not used.any():
        return None, None, None

    ci = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auroc", n_bootstrap, seed)
    summary = {
        "model": model,
        "n": int(used.sum()),
        "n_dep": int(y[used].sum()),
        "n_indep": int((1 - y[used]).sum()),
        "auroc": ci["point"],
        "auprc": average_precision_score(y[used], probs[used]),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "se": ci["se"],
        "selected_layers": ",".join(map(str, fold_df["selected_layer"].tolist())),
        "selected_cs": ",".join(map(str, fold_df["selected_c"].tolist())),
    }
    task_df = pd.DataFrame([{key: row[key] for key in ["model", "dataset", "question_id", "group", "label"]} for row in records])
    pred_df = task_df.loc[used].copy()
    pred_df["oof_prob"] = probs[used]
    return summary, fold_df, pred_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="short name, e.g. deepseek, qwen3_14b")
    parser.add_argument("--data-dir", required=True, help="trace store with traces/ and features/")
    parser.add_argument("--output-dir", default="outputs/larger_models")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    funnel = summarize_funnel(load_jsonl(data_dir / "traces" / f"{args.model}_traces.jsonl"))
    funnel.to_csv(out_dir / "funnel.csv", index=False)

    # Retained slice of record, keeping rows whose W prompt-end features exist.
    records = [r for r in build_main_task_records(args.model, data_dir) if resolve_feature_path(r["w"], data_dir)]
    task_df = pd.DataFrame([{key: row[key] for key in ["model", "dataset", "question_id", "group", "label"]} for row in records])
    task_df.to_csv(out_dir / "retained_records.csv", index=False)
    if len(records) < 20 or task_df["label"].nunique() < 2:
        print(f"Retained slice too small: N={len(records)}")
        print(funnel.to_string(index=False))
        return

    summary, fold_df, pred_df = summarize_nested_records(args.model, records, data_dir, args.n_bootstrap, seed=19)
    pd.DataFrame([summary]).to_csv(out_dir / "nested_summary.csv", index=False)
    fold_df.to_csv(out_dir / "nested_folds.csv", index=False)
    pred_df.to_csv(out_dir / "nested_oof_predictions.csv", index=False)

    per_dataset = []
    for dataset in sorted(task_df["dataset"].unique()):
        ds_summary, _, _ = summarize_nested_records(
            args.model, [r for r in records if r["dataset"] == dataset], data_dir, args.n_bootstrap, seed=19, min_records=10
        )
        if ds_summary is not None:
            ds_summary["dataset"] = dataset
            per_dataset.append(ds_summary)
    if per_dataset:
        pd.DataFrame(per_dataset).to_csv(out_dir / "per_dataset_nested_summary.csv", index=False)

    print(funnel.to_string(index=False))
    print(pd.DataFrame([summary]).to_string(index=False))


if __name__ == "__main__":
    main()
