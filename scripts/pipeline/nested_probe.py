"""Headline nested prompt-end probe and the 200-permutation shuffled-label control on the retained slice.

Inputs: --data-dir with traces/{model}_traces.jsonl and features/*.npz (prompt-end states, all layers).
Outputs (--output-dir): nested_probe_oof.csv and nested_probe_layers.csv (released as data/scores/probes/; the
layers file without its auroc, auprc, se and B = 2000 interval columns; the AUROC and interval are the hidden_probe rows of
data/scores/controls/strict_rule_summary.csv) and shuffled_label_baseline.csv (released as data/scores/controls/).
The paper's printed interval comes from analysis/monitor_intervals.py (seed 0), as does scripts/reproduce.py's.
Run: python -m scripts.pipeline.nested_probe --data-dir <store>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from scripts.pipeline.main_task import (
    cluster_bootstrap_metric,
    fit_dense_predict,
    load_state_matrix,
    nested_dense_layer_cv,
)
from src.core.slice import answers_match, group_traces, load_jsonl

MODELS = ["mistral", "gemma", "llama"]


def load_retained_records(model: str, data_dir: Path) -> list[dict]:
    """Retained W traces with the dependence label, in trace-file order.

    Same rule as src.core.slice.build_main_task_records, but unsorted: GroupKFold folds depend on
    record order, and the released folds come from this order.
    """
    records = []
    for (dataset, _), conds in group_traces(load_jsonl(data_dir / "traces" / f"{model}_traces.jsonl")).items():
        w_row = conds.get("W")
        g_row = conds.get("G")
        if not w_row or not g_row:
            continue
        if not w_row.get("parse_ok") or not g_row.get("parse_ok"):
            continue
        h1 = w_row["candidate"]
        w_ans = w_row.get("normalized_final")
        g_ans = g_row.get("normalized_final")
        if not w_ans or not answers_match(w_ans, h1):
            continue
        if answers_match(w_ans, w_row.get("gold_answer")):
            continue
        if not g_ans:
            continue
        records.append(
            {
                "model": model,
                "dataset": dataset,
                "question_id": w_row["question_id"],
                "group": f"{dataset}::{w_row['question_id']}",
                "label": int(not answers_match(g_ans, h1)),
                "w": w_row,
            }
        )
    return records


def load_model_inputs(model: str, data_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray, list[int], dict]:
    records = load_retained_records(model, data_dir)
    layers, x_by_layer, kept = load_state_matrix(records, "w", data_dir)
    if len(kept) != len(records):
        raise FileNotFoundError(f"{model}: {len(records) - len(kept)} retained rows have no feature file")
    y = np.array([r["label"] for r in records], dtype=int)
    groups = np.array([r["group"] for r in records])
    return records, y, groups, layers, x_by_layer


def nested_probe(models: list[str], data_dir: Path, output_dir: Path) -> pd.DataFrame:
    pred_rows = []
    layer_rows = []
    for model in models:
        print(f"nested probe: {model}", flush=True)
        records, y, groups, layers, x_by_layer = load_model_inputs(model, data_dir)
        probs, used, folds = nested_dense_layer_cv(x_by_layer, layers, y, groups)
        ci = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auroc", 2000, 31 + len(layer_rows))
        layer_rows.append(
            {
                "model": model,
                "auroc": roc_auc_score(y[used], probs[used]),
                "auprc": average_precision_score(y[used], probs[used]),
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "se": ci["se"],
                "n": int(used.sum()),
                "n_dep": int(y[used].sum()),
                "n_indep": int((1 - y[used]).sum()),
                "selected_layers": ",".join(map(str, folds["selected_layer"].tolist())),
                "selected_cs": ",".join(map(str, folds["C"].tolist())),
            }
        )
        for row, label, prob, used_flag in zip(records, y, probs, used):
            if used_flag:
                pred_rows.append(
                    {
                        "model": model,
                        "dataset": row["dataset"],
                        "question_id": row["question_id"],
                        "group": row["group"],
                        "label": int(label),
                        "oof_prob": float(prob),
                    }
                )
    layers_df = pd.DataFrame(layer_rows)
    layers_df.to_csv(output_dir / "nested_probe_layers.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "nested_probe_oof.csv", index=False)
    print(layers_df, flush=True)
    return layers_df


def shuffled_label_baseline(
    models: list[str], data_dir: Path, layers_csv: Path, output_dir: Path, n_shuffles: int, seed: int
) -> pd.DataFrame:
    """Refit the probe on permuted labels, reusing the headline outer folds, layers and Cs."""
    chosen = pd.read_csv(layers_csv).set_index("model")
    rng = np.random.default_rng(seed)
    rows = []
    for model in models:
        print(f"shuffled-label baseline: {model}", flush=True)
        records, y, groups, layers, x_by_layer = load_model_inputs(model, data_dir)
        selected_layers = [int(v) for v in str(chosen.loc[model, "selected_layers"]).split(",")]
        selected_cs = [float(v) for v in str(chosen.loc[model, "selected_cs"]).split(",")]
        outer_splits = list(GroupKFold(n_splits=5).split(x_by_layer[layers[0]], y, groups))
        if len(selected_layers) != len(outer_splits):
            raise ValueError(f"{model}: selected layer count does not match outer folds.")

        scores = []
        for _ in range(n_shuffles):
            y_shuf = rng.permutation(y)
            probs = np.zeros(len(y_shuf), dtype=float)
            used = np.zeros(len(y_shuf), dtype=bool)
            for (train_idx, test_idx), layer, c_value in zip(outer_splits, selected_layers, selected_cs):
                if len(np.unique(y_shuf[train_idx])) < 2 or len(np.unique(y_shuf[test_idx])) < 2:
                    continue
                x = x_by_layer[layer]
                probs[test_idx] = fit_dense_predict(x[train_idx], y_shuf[train_idx], x[test_idx], c_value)
                used[test_idx] = True
            if used.sum() and len(np.unique(y_shuf[used])) == 2:
                scores.append(roc_auc_score(y_shuf[used], probs[used]))

        score_arr = np.array(scores, dtype=float)
        rows.append(
            {
                "model": model,
                "n_shuffles": len(score_arr),
                "mean_auroc": float(score_arr.mean()),
                "ci_low": float(np.percentile(score_arr, 2.5)),
                "ci_high": float(np.percentile(score_arr, 97.5)),
                "min_auroc": float(score_arr.min()),
                "max_auroc": float(score_arr.max()),
                "n": int(len(y)),
                "n_dep": int(y.sum()),
                "n_indep": int((1 - y).sum()),
                "selected_layers": ",".join(map(str, selected_layers)),
                "selected_cs": ",".join(map(str, selected_cs)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "shuffled_label_baseline.csv", index=False)
    print(df, flush=True)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="trace store: traces/{model}_traces.jsonl, features/")
    parser.add_argument("--output-dir", default="outputs/nested_probe")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--n-shuffles", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    nested_probe(args.models, data_dir, output_dir)
    shuffled_label_baseline(
        args.models, data_dir, output_dir / "nested_probe_layers.csv", output_dir, args.n_shuffles, args.seed
    )


if __name__ == "__main__":
    main()
