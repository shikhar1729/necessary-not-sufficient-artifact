"""Main-task probes and baselines on the retained same-answer slice (strict label rule).
Input: --data-dir (traces/, features/, and neutral_accuracy/{model}_neutral_traces.jsonl from the no-hint run,
needed by the no-hint diagnostics only) and the nested-probe files in data/scores/probes/. Output (--output-dir):
per-dataset probe CIs, TF-IDF, control-state and paired-difference probes, no-hint diagnostics, selected layers.
Intervals: cluster bootstrap, B=1000, seed 1729 (the defaults; the released TF-IDF intervals in
data/scores/controls/strict_rule_summary.csv reproduce exactly with them; the command line was not recorded).
no_hint_diagnostics gives the strict-slice version; the printed table (tab:no-hint-diagnostic) comes from
analysis/no_hint_diagnostic_lenient.py.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.core.slice import (
    answers_match,
    build_main_task_records,
    load_jsonl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
C_GRID = [0.01, 0.1, 1.0, 10.0]
TEXT_CONFIGS = [
    {
        "name": "word12",
        "analyzer": "word",
        "ngram_range": (1, 2),
        "max_features": 50000,
    },
    {
        "name": "char35",
        "analyzer": "char_wb",
        "ngram_range": (3, 5),
        "max_features": 75000,
    },
]


def resolve_feature_path(row: dict[str, Any], data_dir: Path) -> Path | None:
    raw = Path(row.get("hidden_state_path", ""))
    candidates = [
        raw,
        PROJECT_ROOT / raw,
        data_dir / "features" / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def cluster_bootstrap_metric(
    y: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: str,
    n_bootstrap: int,
    seed: int,
) -> dict[str, float]:
    if len(np.unique(y)) < 2:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "se": np.nan}
    fn = roc_auc_score if metric == "auroc" else average_precision_score
    point = float(fn(y, scores))
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(float(fn(y[idx], scores[idx])))
    if not values:
        return {"point": point, "ci_low": point, "ci_high": point, "se": 0.0}
    arr = np.asarray(values)
    return {
        "point": point,
        "ci_low": float(np.percentile(arr, 2.5)),
        "ci_high": float(np.percentile(arr, 97.5)),
        "se": float(np.std(arr, ddof=1)),
    }


def per_dataset_probe_ci(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    rows = []
    df = pd.read_csv(Path(args.probe_oof))
    df = df[df["model"].isin(args.models)]
    for (model, dataset), sub in df.groupby(["model", "dataset"]):
        y = sub["label"].to_numpy(dtype=int)
        scores = sub["oof_prob"].to_numpy(dtype=float)
        groups = sub["group"].astype(str).to_numpy()
        auroc = cluster_bootstrap_metric(y, scores, groups, "auroc", args.n_bootstrap, args.seed)
        auprc = cluster_bootstrap_metric(y, scores, groups, "auprc", args.n_bootstrap, args.seed + 1)
        rows.append(
            {
                "model": model,
                "dataset": dataset,
                "n": len(sub),
                "n_dep": int(y.sum()),
                "n_indep": int(len(y) - y.sum()),
                "auroc": auroc["point"],
                "auroc_ci_low": auroc["ci_low"],
                "auroc_ci_high": auroc["ci_high"],
                "auprc": auprc["point"],
                "auprc_ci_low": auprc["ci_low"],
                "auprc_ci_high": auprc["ci_high"],
            }
        )
    out = pd.DataFrame(rows).sort_values(["model", "dataset"])
    out.to_csv(output_dir / "per_dataset_nested_probe_ci.csv", index=False)
    return out


def make_text(record: dict[str, Any], view: str) -> str:
    q = record["question"]
    ans = record["final_answer"]
    cot = record["reasoning_text"]
    prompt = record["w"].get("raw_prompt", "")
    if view == "answer_only":
        return f"Question:\n{q}\n\nFinal answer:\n{ans}"
    if view == "cot_answer":
        return f"Question:\n{q}\n\nReasoning:\n{cot}\n\nFinal answer:\n{ans}"
    if view == "prompt_answer":
        return f"Prompt:\n{prompt}\n\nFinal answer:\n{ans}"
    if view == "prompt_cot_answer":
        return f"Prompt:\n{prompt}\n\nReasoning:\n{cot}\n\nFinal answer:\n{ans}"
    raise ValueError(f"unknown view={view!r}")


def build_text_pipeline(config: dict[str, Any], c_value: float) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=config["analyzer"],
                    ngram_range=config["ngram_range"],
                    max_features=config["max_features"],
                    min_df=2,
                    lowercase=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=0,
                    solver="liblinear",
                ),
            ),
        ]
    )


def inner_text_score(
    texts: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    config: dict[str, Any],
    c_value: float,
) -> float:
    train_groups = np.unique(groups[train_idx])
    n_splits = min(3, len(train_groups))
    if n_splits < 2:
        return 0.5
    scores = []
    splitter = GroupKFold(n_splits=n_splits)
    for inner_train_rel, inner_val_rel in splitter.split(texts[train_idx], y[train_idx], groups[train_idx]):
        inner_train = train_idx[inner_train_rel]
        inner_val = train_idx[inner_val_rel]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        clf = build_text_pipeline(config, c_value)
        clf.fit(texts[inner_train], y[inner_train])
        probs = clf.predict_proba(texts[inner_val])[:, 1]
        scores.append(float(roc_auc_score(y[inner_val], probs)))
    return float(np.mean(scores)) if scores else 0.5


def nested_text_cv(
    texts: list[str],
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    texts_arr = np.asarray(texts, dtype=object)
    probs = np.zeros(len(y), dtype=float)
    used = np.zeros(len(y), dtype=bool)
    fold_rows = []
    outer = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (train_idx, test_idx) in enumerate(outer.split(texts_arr, y, groups)):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        best = (-math.inf, TEXT_CONFIGS[0], 1.0)
        for config in TEXT_CONFIGS:
            for c_value in C_GRID:
                score = inner_text_score(texts_arr, y, groups, train_idx, config, c_value)
                if score > best[0]:
                    best = (score, config, c_value)
        inner_score, config, c_value = best
        clf = build_text_pipeline(config, c_value)
        clf.fit(texts_arr[train_idx], y[train_idx])
        probs[test_idx] = clf.predict_proba(texts_arr[test_idx])[:, 1]
        used[test_idx] = True
        fold_rows.append(
            {
                "fold": fold,
                "feature_config": config["name"],
                "C": c_value,
                "inner_auroc": inner_score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
    return probs, used, pd.DataFrame(fold_rows)


def text_discriminative_baselines(
    all_records: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    views = ["answer_only", "cot_answer", "prompt_answer", "prompt_cot_answer"]
    summary_rows = []
    pred_rows = []
    fold_rows = []
    for model, records in all_records.items():
        y = np.array([row["label"] for row in records], dtype=int)
        groups = np.array([row["group"] for row in records], dtype=str)
        for view in views:
            texts = [make_text(row, view) for row in records]
            probs, used, folds = nested_text_cv(texts, y, groups)
            auroc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auroc", args.n_bootstrap, args.seed)
            auprc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auprc", args.n_bootstrap, args.seed + 1)
            summary_rows.append(
                {
                    "model": model,
                    "view": view,
                    "n": int(used.sum()),
                    "n_dep": int(y[used].sum()),
                    "n_indep": int(used.sum() - y[used].sum()),
                    "auroc": auroc["point"],
                    "auroc_ci_low": auroc["ci_low"],
                    "auroc_ci_high": auroc["ci_high"],
                    "auprc": auprc["point"],
                    "auprc_ci_low": auprc["ci_low"],
                    "auprc_ci_high": auprc["ci_high"],
                }
            )
            for row, prob in zip(records, probs):
                pred_rows.append(
                    {
                        "model": model,
                        "view": view,
                        "dataset": row["dataset"],
                        "question_id": row["question_id"],
                        "group": row["group"],
                        "label": row["label"],
                        "prob": prob,
                    }
                )
            folds = folds.copy()
            folds.insert(0, "view", view)
            folds.insert(0, "model", model)
            fold_rows.extend(folds.to_dict("records"))
    summary = pd.DataFrame(summary_rows).sort_values(["model", "view"])
    summary.to_csv(output_dir / "text_discriminative_baselines.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "text_discriminative_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "text_discriminative_folds.csv", index=False)
    return summary


def load_state_matrix(
    records: list[dict[str, Any]],
    source: str,
    data_dir: Path,
    only_layers: list[int] | None = None,
) -> tuple[list[int], dict[int, np.ndarray], list[int]]:
    paths = []
    kept = []
    for idx, row in enumerate(records):
        trace = row[source]
        if trace is None:
            continue
        path = resolve_feature_path(trace, data_dir)
        if path is None:
            continue
        paths.append(path)
        kept.append(idx)
    if not paths:
        raise RuntimeError(f"No feature paths found for source={source}")
    if only_layers is not None:
        layers = sorted(set(only_layers))
    else:
        with np.load(paths[0]) as z:
            layers = sorted(
                int(key.rsplit("_", 1)[1])
                for key in z.files
                if key.startswith("pos_-1_layer_")
            )
    by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for path in paths:
        with np.load(path) as z:
            for layer in layers:
                key = f"pos_-1_layer_{layer}"
                if key not in z.files:
                    raise RuntimeError(f"Missing {key} in {path}")
                by_layer[layer].append(z[key].astype(np.float32))
    return layers, {layer: np.stack(values, axis=0) for layer, values in by_layer.items()}, kept


def fit_dense_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, c_value: float) -> np.ndarray:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    clf = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    clf.fit(x_train_s, y_train)
    return clf.predict_proba(x_test_s)[:, 1]


def inner_dense_score(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    c_value: float,
) -> float:
    train_groups = np.unique(groups[train_idx])
    n_splits = min(3, len(train_groups))
    if n_splits < 2:
        return 0.5
    scores = []
    splitter = GroupKFold(n_splits=n_splits)
    for inner_train_rel, inner_val_rel in splitter.split(x[train_idx], y[train_idx], groups[train_idx]):
        inner_train = train_idx[inner_train_rel]
        inner_val = train_idx[inner_val_rel]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        probs = fit_dense_predict(x[inner_train], y[inner_train], x[inner_val], c_value)
        scores.append(float(roc_auc_score(y[inner_val], probs)))
    return float(np.mean(scores)) if scores else 0.5


def nested_dense_layer_cv(
    x_by_layer: dict[int, np.ndarray],
    layers: list[int],
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    probs = np.zeros(len(y), dtype=float)
    used = np.zeros(len(y), dtype=bool)
    fold_rows = []
    outer = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (train_idx, test_idx) in enumerate(outer.split(x_by_layer[layers[0]], y, groups)):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        best = (-math.inf, layers[0], 1.0)
        for layer in layers:
            x = x_by_layer[layer]
            for c_value in C_GRID:
                score = inner_dense_score(x, y, groups, train_idx, c_value)
                if score > best[0]:
                    best = (score, layer, c_value)
        inner_score, layer, c_value = best
        probs[test_idx] = fit_dense_predict(
            x_by_layer[layer][train_idx],
            y[train_idx],
            x_by_layer[layer][test_idx],
            c_value,
        )
        used[test_idx] = True
        fold_rows.append(
            {
                "fold": fold,
                "selected_layer": layer,
                "C": c_value,
                "inner_auroc": inner_score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
    return probs, used, pd.DataFrame(fold_rows)


def make_paired_layer_features(
    records: list[dict[str, Any]],
    data_dir: Path,
    left_source: str,
    right_source: str,
    only_layers: list[int] | None = None,
) -> tuple[list[int], dict[int, np.ndarray], list[int]]:
    left_layers, left_by_layer, left_kept = load_state_matrix(
        records,
        left_source,
        data_dir,
        only_layers=only_layers,
    )
    right_layers, right_by_layer, right_kept = load_state_matrix(
        records,
        right_source,
        data_dir,
        only_layers=only_layers,
    )
    if left_kept != right_kept:
        common = sorted(set(left_kept).intersection(right_kept))
        left_pos = {idx: pos for pos, idx in enumerate(left_kept)}
        right_pos = {idx: pos for pos, idx in enumerate(right_kept)}
        layers = sorted(set(left_layers).intersection(right_layers))
        diff_by_layer = {
            layer: np.stack(
                [
                    left_by_layer[layer][left_pos[idx]] - right_by_layer[layer][right_pos[idx]]
                    for idx in common
                ],
                axis=0,
            )
            for layer in layers
        }
        return layers, diff_by_layer, common

    layers = sorted(set(left_layers).intersection(right_layers))
    diff_by_layer = {
        layer: left_by_layer[layer] - right_by_layer[layer]
        for layer in layers
    }
    return layers, diff_by_layer, left_kept


def residualize_train_test(
    target_train: np.ndarray,
    control_train: np.ndarray,
    target_test: np.ndarray,
    control_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove the control-predictable part of target states without test leakage."""
    control_scaler = StandardScaler()
    target_scaler = StandardScaler()
    control_train_s = control_scaler.fit_transform(control_train)
    control_test_s = control_scaler.transform(control_test)
    target_train_s = target_scaler.fit_transform(target_train)
    target_test_s = target_scaler.transform(target_test)
    # Dual ridge is exact for standardized features and is much faster here
    # because n_examples is far smaller than the hidden-state dimension.
    kernel = control_train_s @ control_train_s.T
    kernel_reg = kernel.copy()
    kernel_reg.flat[:: kernel_reg.shape[0] + 1] += alpha
    dual = np.linalg.solve(kernel_reg, target_train_s)
    train_pred = kernel @ dual
    test_pred = (control_test_s @ control_train_s.T) @ dual
    return (
        target_train_s - train_pred,
        target_test_s - test_pred,
    )


def residualized_inner_score(
    target_x: np.ndarray,
    control_x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    c_value: float,
    residual_alpha: float,
) -> float:
    train_groups = np.unique(groups[train_idx])
    n_splits = min(3, len(train_groups))
    if n_splits < 2:
        return 0.5
    scores = []
    splitter = GroupKFold(n_splits=n_splits)
    for inner_train_rel, inner_val_rel in splitter.split(
        target_x[train_idx],
        y[train_idx],
        groups[train_idx],
    ):
        inner_train = train_idx[inner_train_rel]
        inner_val = train_idx[inner_val_rel]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        resid_train, resid_val = residualize_train_test(
            target_x[inner_train],
            control_x[inner_train],
            target_x[inner_val],
            control_x[inner_val],
            alpha=residual_alpha,
        )
        probs = fit_dense_predict(resid_train, y[inner_train], resid_val, c_value)
        scores.append(float(roc_auc_score(y[inner_val], probs)))
    return float(np.mean(scores)) if scores else 0.5


def nested_residualized_layer_cv(
    target_by_layer: dict[int, np.ndarray],
    control_by_layer: dict[int, np.ndarray],
    layers: list[int],
    y: np.ndarray,
    groups: np.ndarray,
    residual_alpha: float,
    c_values: list[float],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    probs = np.zeros(len(y), dtype=float)
    used = np.zeros(len(y), dtype=bool)
    fold_rows = []
    outer = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    for fold, (train_idx, test_idx) in enumerate(outer.split(target_by_layer[layers[0]], y, groups)):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        if len(layers) == 1 and len(c_values) == 1:
            inner_score, layer, c_value = np.nan, layers[0], c_values[0]
        else:
            best = (-math.inf, layers[0], c_values[0])
            for layer in layers:
                target_x = target_by_layer[layer]
                control_x = control_by_layer[layer]
                for c_value in c_values:
                    score = residualized_inner_score(
                        target_x,
                        control_x,
                        y,
                        groups,
                        train_idx,
                        c_value,
                        residual_alpha,
                    )
                    if score > best[0]:
                        best = (score, layer, c_value)
            inner_score, layer, c_value = best
        resid_train, resid_test = residualize_train_test(
            target_by_layer[layer][train_idx],
            control_by_layer[layer][train_idx],
            target_by_layer[layer][test_idx],
            control_by_layer[layer][test_idx],
            alpha=residual_alpha,
        )
        probs[test_idx] = fit_dense_predict(resid_train, y[train_idx], resid_test, c_value)
        used[test_idx] = True
        fold_rows.append(
            {
                "fold": fold,
                "selected_layer": layer,
                "C": c_value,
                "residual_alpha": residual_alpha,
                "inner_auroc": inner_score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
    return probs, used, pd.DataFrame(fold_rows)


def paired_difference_probes(
    all_records: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    fixed_layers = selected_layer_map(Path(args.probe_layers))
    summary_rows = []
    pred_rows = []
    fold_rows = []
    data_dir = Path(args.data_dir)
    direct_specs = [
        ("W_minus_G", "w", "g"),
        ("W_minus_C", "w", "c"),
    ]
    residual_specs = [] if args.skip_residual_paired else [
        ("W_resid_G", "w", "g"),
        ("W_resid_C", "w", "c"),
    ]

    for model, records in all_records.items():
        requested_layers = None
        if args.paired_layer_mode == "fixed":
            layer = fixed_layers.get(model)
            if layer is None:
                raise RuntimeError(f"No fixed selected layer found for model={model}")
            requested_layers = [layer]

        y_all = np.array([row["label"] for row in records], dtype=int)
        groups_all = np.array([row["group"] for row in records], dtype=str)

        for method, left_source, right_source in direct_specs:
            layers, x_by_layer, kept = make_paired_layer_features(
                records,
                data_dir,
                left_source,
                right_source,
                only_layers=requested_layers,
            )
            y = y_all[kept]
            groups = groups_all[kept]
            probs, used, folds = nested_dense_layer_cv(x_by_layer, layers, y, groups)
            auroc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auroc", args.n_bootstrap, args.seed)
            auprc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auprc", args.n_bootstrap, args.seed + 1)
            summary_rows.append(
                {
                    "model": model,
                    "method": method,
                    "layer_mode": args.paired_layer_mode,
                    "residual_alpha": np.nan,
                    "layers_evaluated": ",".join(map(str, layers)),
                    "n": int(used.sum()),
                    "n_dep": int(y[used].sum()),
                    "n_indep": int(used.sum() - y[used].sum()),
                    "auroc": auroc["point"],
                    "auroc_ci_low": auroc["ci_low"],
                    "auroc_ci_high": auroc["ci_high"],
                    "auprc": auprc["point"],
                    "auprc_ci_low": auprc["ci_low"],
                    "auprc_ci_high": auprc["ci_high"],
                }
            )
            for idx, prob in zip(np.array(kept)[used], probs[used]):
                row = records[int(idx)]
                pred_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "dataset": row["dataset"],
                        "question_id": row["question_id"],
                        "group": row["group"],
                        "label": row["label"],
                        "prob": prob,
                    }
                )
            folds = folds.copy()
            folds.insert(0, "method", method)
            folds.insert(0, "model", model)
            fold_rows.extend(folds.to_dict("records"))

        for method, target_source, control_source in residual_specs:
            target_layers, target_by_layer, target_kept = load_state_matrix(
                records,
                target_source,
                data_dir,
                only_layers=requested_layers,
            )
            control_layers, control_by_layer, control_kept = load_state_matrix(
                records,
                control_source,
                data_dir,
                only_layers=requested_layers,
            )
            common = sorted(set(target_kept).intersection(control_kept))
            target_pos = {idx: pos for pos, idx in enumerate(target_kept)}
            control_pos = {idx: pos for pos, idx in enumerate(control_kept)}
            layers = sorted(set(target_layers).intersection(control_layers))
            target_aligned = {
                layer: np.stack([target_by_layer[layer][target_pos[idx]] for idx in common], axis=0)
                for layer in layers
            }
            control_aligned = {
                layer: np.stack([control_by_layer[layer][control_pos[idx]] for idx in common], axis=0)
                for layer in layers
            }
            y = y_all[common]
            groups = groups_all[common]
            for residual_alpha in args.residual_alphas:
                probs, used, folds = nested_residualized_layer_cv(
                    target_aligned,
                    control_aligned,
                    layers,
                    y,
                    groups,
                    residual_alpha=residual_alpha,
                    c_values=args.residual_probe_c_values,
                )
                auroc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auroc", args.n_bootstrap, args.seed)
                auprc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auprc", args.n_bootstrap, args.seed + 1)
                summary_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "layer_mode": args.paired_layer_mode,
                        "residual_alpha": residual_alpha,
                        "layers_evaluated": ",".join(map(str, layers)),
                        "n": int(used.sum()),
                        "n_dep": int(y[used].sum()),
                        "n_indep": int(used.sum() - y[used].sum()),
                        "auroc": auroc["point"],
                        "auroc_ci_low": auroc["ci_low"],
                        "auroc_ci_high": auroc["ci_high"],
                        "auprc": auprc["point"],
                        "auprc_ci_low": auprc["ci_low"],
                        "auprc_ci_high": auprc["ci_high"],
                    }
                )
                for idx, prob in zip(np.array(common)[used], probs[used]):
                    row = records[int(idx)]
                    pred_rows.append(
                        {
                            "model": model,
                            "method": method,
                            "residual_alpha": residual_alpha,
                            "dataset": row["dataset"],
                            "question_id": row["question_id"],
                            "group": row["group"],
                            "label": row["label"],
                            "prob": prob,
                        }
                    )
                folds = folds.copy()
                if "residual_alpha" not in folds.columns:
                    folds.insert(0, "residual_alpha", residual_alpha)
                folds.insert(0, "method", method)
                folds.insert(0, "model", model)
                fold_rows.extend(folds.to_dict("records"))

    summary = pd.DataFrame(summary_rows).sort_values(["model", "method", "residual_alpha"], na_position="first")
    summary.to_csv(output_dir / "paired_difference_probe_baselines.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "paired_difference_probe_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "paired_difference_probe_folds.csv", index=False)
    return summary


def selected_layer_map(layers_csv: Path) -> dict[str, int]:
    ci_path = layers_csv
    out: dict[str, int] = {}
    if ci_path.exists():
        for _, row in pd.read_csv(ci_path).iterrows():
            layers = [int(x) for x in str(row["selected_layers"]).split(",") if str(x).strip()]
            if not layers:
                continue
            counts = Counter(layers)
            out[str(row["model"])] = counts.most_common(1)[0][0]
    return out


def control_state_probes(
    all_records: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    sources = {"W_prompt": "w", "G_prompt": "g", "C_prompt": "c"}
    fixed_layers = selected_layer_map(Path(args.probe_layers))
    summary_rows = []
    pred_rows = []
    fold_rows = []
    for model, records in all_records.items():
        for source_name, source_key in sources.items():
            requested_layers = None
            if args.control_state_mode == "fixed":
                layer = fixed_layers.get(model)
                if layer is None:
                    raise RuntimeError(f"No fixed selected layer found for model={model}")
                requested_layers = [layer]
            layers, x_by_layer, kept = load_state_matrix(
                records,
                source_key,
                Path(args.data_dir),
                only_layers=requested_layers,
            )
            sub_records = [records[idx] for idx in kept]
            y = np.array([row["label"] for row in sub_records], dtype=int)
            groups = np.array([row["group"] for row in sub_records], dtype=str)
            probs, used, folds = nested_dense_layer_cv(x_by_layer, layers, y, groups)
            auroc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auroc", args.n_bootstrap, args.seed)
            auprc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auprc", args.n_bootstrap, args.seed + 1)
            summary_rows.append(
                {
                    "model": model,
                    "source": source_name,
                    "layer_mode": args.control_state_mode,
                    "layers_evaluated": ",".join(map(str, layers)),
                    "n": int(used.sum()),
                    "n_dep": int(y[used].sum()),
                    "n_indep": int(used.sum() - y[used].sum()),
                    "auroc": auroc["point"],
                    "auroc_ci_low": auroc["ci_low"],
                    "auroc_ci_high": auroc["ci_high"],
                    "auprc": auprc["point"],
                    "auprc_ci_low": auprc["ci_low"],
                    "auprc_ci_high": auprc["ci_high"],
                }
            )
            for row, prob in zip(sub_records, probs):
                pred_rows.append(
                    {
                        "model": model,
                        "source": source_name,
                        "dataset": row["dataset"],
                        "question_id": row["question_id"],
                        "group": row["group"],
                        "label": row["label"],
                        "prob": prob,
                    }
                )
            folds = folds.copy()
            folds.insert(0, "source", source_name)
            folds.insert(0, "model", model)
            fold_rows.extend(folds.to_dict("records"))
    summary = pd.DataFrame(summary_rows).sort_values(["model", "source"])
    summary.to_csv(output_dir / "control_state_probe_baselines.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "control_state_probe_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "control_state_probe_folds.csv", index=False)
    return summary


def no_hint_diagnostics(
    all_records: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    summary_rows = []
    detail_rows = []
    neutral_dir = Path(args.data_dir) / "neutral_accuracy"
    for model, records in all_records.items():
        neutral_path = neutral_dir / f"{model}_neutral_traces.jsonl"
        if not neutral_path.exists():
            continue
        neutral = {
            (row["dataset"], str(row["question_id"])): row
            for row in load_jsonl(neutral_path)
        }
        rows = []
        for record in records:
            nrow = neutral.get((record["dataset"], record["question_id"]))
            if not nrow or not nrow.get("parse_ok"):
                continue
            prior_h1 = int(answers_match(nrow.get("normalized_final"), record["h1"]))
            item = {
                "model": model,
                "dataset": record["dataset"],
                "question_id": record["question_id"],
                "label": record["label"],
                "no_hint_answer": nrow.get("normalized_final"),
                "no_hint_correct": int(bool(nrow.get("is_correct"))),
                "no_hint_matches_h1": prior_h1,
            }
            detail_rows.append(item)
            rows.append(item)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        y = df["label"].to_numpy(dtype=int)
        score = df["no_hint_matches_h1"].to_numpy(dtype=float)
        groups = np.array([f"{r['dataset']}::{r['question_id']}" for r in rows])
        metric = cluster_bootstrap_metric(y, score, groups, "auroc", args.n_bootstrap, args.seed)
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
    if summary_rows:
        summary = pd.DataFrame(summary_rows).sort_values(["model", "subset"])
    else:
        summary = pd.DataFrame(
            columns=[
                "model",
                "subset",
                "n",
                "no_hint_h1_rate",
                "no_hint_correct_rate",
                "prior_h1_auroc",
                "prior_h1_auroc_ci_low",
                "prior_h1_auroc_ci_high",
            ]
        )
    summary.to_csv(output_dir / "no_hint_diagnostic_summary.csv", index=False)
    pd.DataFrame(detail_rows).to_csv(output_dir / "no_hint_diagnostic_examples.csv", index=False)
    return summary


def selected_layer_summary(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    rows = []
    df = pd.read_csv(Path(args.probe_layers))
    df = df[df["model"].isin(args.models)]
    for _, row in df.iterrows():
        layers = [int(x) for x in str(row["selected_layers"]).split(",") if str(x).strip()]
        counts = Counter(layers)
        rows.append(
            {
                "model": row["model"],
                "selected_layers": ",".join(map(str, layers)),
                "modal_layer": counts.most_common(1)[0][0] if layers else np.nan,
                "min_layer": min(layers) if layers else np.nan,
                "median_layer": float(np.median(layers)) if layers else np.nan,
                "max_layer": max(layers) if layers else np.nan,
                "n_outer_folds": len(layers),
                "layer_counts": "; ".join(f"{layer}:{count}" for layer, count in sorted(counts.items())),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "selected_layer_summary.csv", index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="trace store: traces/{model}_traces.jsonl, features/")
    parser.add_argument("--probe-oof", default="data/scores/probes/nested_probe_oof.csv")
    parser.add_argument("--probe-layers", default="data/scores/probes/nested_probe_layers.csv")
    parser.add_argument("--output-dir", default="outputs/main_task")
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument(
        "--paired-only",
        action="store_true",
        help="Only run paired-difference/residualized prompt-state probes.",
    )
    parser.add_argument(
        "--skip-paired",
        action="store_true",
        help="Skip paired-difference/residualized prompt-state probes.",
    )
    parser.add_argument(
        "--skip-residual-paired",
        action="store_true",
        help="Run W-G/W-C paired differences but skip residualized paired probes.",
    )
    parser.add_argument(
        "--skip-per-dataset",
        action="store_true",
        help="Skip the per-dataset probe CI aggregation.",
    )
    parser.add_argument(
        "--paired-layer-mode",
        choices=["fixed", "sweep"],
        default="fixed",
        help=(
            "Use each model's headline selected layer for paired diagnostics "
            "or run the expensive all-layer nested sweep."
        ),
    )
    parser.add_argument(
        "--residual-alphas",
        nargs="+",
        type=float,
        default=[1.0, 10.0, 100.0],
        help="Ridge alphas for fold-local W-on-control residualization.",
    )
    parser.add_argument(
        "--residual-probe-c-values",
        nargs="+",
        type=float,
        default=[1.0],
        help="Logistic C values considered for residualized probes.",
    )
    parser.add_argument(
        "--control-state-mode",
        choices=["fixed", "sweep"],
        default="fixed",
        help=(
            "Use each model's headline selected layer for control-state probes "
            "or run the expensive all-layer nested sweep."
        ),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = {
        model: build_main_task_records(model, Path(args.data_dir))
        for model in args.models
    }
    pd.DataFrame(
        [
            {
                "model": model,
                "n": len(records),
                "n_dep": int(sum(row["label"] for row in records)),
                "n_indep": int(len(records) - sum(row["label"] for row in records)),
            }
            for model, records in all_records.items()
        ]
    ).to_csv(output_dir / "main_task_counts.csv", index=False)

    if args.paired_only:
        paired_difference_probes(all_records, args, output_dir)
        print(f"wrote paired diagnostics to {output_dir}")
        return

    if not args.skip_per_dataset:
        per_dataset_probe_ci(args, output_dir)
    text_discriminative_baselines(all_records, args, output_dir)
    control_state_probes(all_records, args, output_dir)
    if not args.skip_paired:
        paired_difference_probes(all_records, args, output_dir)
    no_hint_diagnostics(all_records, args, output_dir)
    selected_layer_summary(args, output_dir)
    print(f"wrote main-task analyses to {output_dir}")


if __name__ == "__main__":
    main()
