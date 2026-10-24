"""Probes and baselines on the adaptive-v4 balanced slices (tab:adaptive-v4): nested grouped CV (groups are the
original question, so pair directions and wordings of one question never cross folds) for the W state, the G state
and W-G (layer and C chosen inside each outer fold), prior scalars, and TF-IDF text with and without priors.
Input: extract_states.py output. Output: balanced_v3_probe_baseline_{summary,predictions}.csv in --output-dir.

    python -m scripts.pipeline.prior_aware.probe_baselines --data-dir <v4_selected> --n-bootstrap 1000
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.pipeline.prior_aware.batched_generation import load_jsonl_rows, truthy


C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def trace_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (str(row["dataset"]), str(row["question_id"]), str(row["condition"]))


def resolve_feature_path(raw_path: str | None, data_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = [
        path,
        data_dir / path,
        data_dir / "features" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_rows(model: str, data_dir: Path) -> list[dict[str, Any]]:
    selected = [
        row for row in load_csv(data_dir / "selected_pairs.csv")
        if row.get("model") == model and truthy(row.get("retained"))
    ]
    traces = load_jsonl_rows(data_dir / "traces" / f"{model}_traces.jsonl")
    traces_by_key = {trace_key(row): row for row in traces}

    rows: list[dict[str, Any]] = []
    for row in selected:
        key_base = (str(row["dataset"]), str(row["question_id"]))
        w = traces_by_key.get((*key_base, "W"))
        g = traces_by_key.get((*key_base, "G"))
        if not w or not g:
            continue
        w_path = resolve_feature_path(w.get("hidden_state_path"), data_dir)
        g_path = resolve_feature_path(g.get("hidden_state_path"), data_dir)
        if w_path is None or g_path is None:
            continue
        h1_prior = as_float(row.get("h1_prior_logp"))
        h2_prior = as_float(row.get("h2_prior_logp"))
        prior_gap = as_float(row.get("prior_abs_gap"), abs(h1_prior - h2_prior))
        prior_signed = h1_prior - h2_prior
        neutral_matches_h1 = float(truthy(row.get("neutral_matches_h1")))
        public_text = "\n\n".join(
            str(part or "")
            for part in [
                w.get("raw_prompt"),
                "MODEL_REASONING:",
                w.get("think_text"),
                "MODEL_ANSWER:",
                w.get("final_text") or w.get("normalized_final"),
            ]
        )
        rows.append(
            {
                "model": model,
                "dataset": row["dataset"],
                "question_id": row["question_id"],
                "original_question_id": row.get("original_question_id", row["question_id"]),
                "group": f"{row['dataset']}::{row.get('original_question_id', row['question_id'])}",
                "template_id": row.get("template_id"),
                "pair_strategy": row.get("pair_strategy"),
                "label": 1 if row.get("label") == "dependent" else 0,
                "w_path": w_path,
                "g_path": g_path,
                "h1_prior_logp": h1_prior,
                "h2_prior_logp": h2_prior,
                "prior_abs_gap": prior_gap,
                "prior_signed_h1_minus_h2": prior_signed,
                "neutral_matches_h1": neutral_matches_h1,
                "w_public_text": public_text,
                "w_answer_text": str(w.get("final_text") or w.get("normalized_final") or ""),
            }
        )
    return rows


def layers_from_npz(path: Path) -> list[int]:
    with np.load(path) as z:
        return sorted(int(k.rsplit("_", 1)[1]) for k in z.files if k.startswith("pos_-1_layer_"))


def load_hidden_by_layer(rows: list[dict[str, Any]], source: str) -> tuple[list[int], dict[int, np.ndarray]]:
    layers = layers_from_npz(rows[0]["w_path"])
    values: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for row in rows:
        with np.load(row["w_path"]) as zw, np.load(row["g_path"]) as zg:
            for layer in layers:
                w = zw[f"pos_-1_layer_{layer}"].astype(np.float32)
                g = zg[f"pos_-1_layer_{layer}"].astype(np.float32)
                if source == "w":
                    x = w
                elif source == "g":
                    x = g
                elif source == "w_minus_g":
                    x = w - g
                else:
                    raise ValueError(source)
                values[layer].append(x)
    return layers, {layer: np.stack(items, axis=0) for layer, items in values.items()}


def choose_valid_group_splits(y: np.ndarray, groups: np.ndarray, max_splits: int) -> int:
    upper = min(max_splits, len(np.unique(groups)))
    for n_splits in range(upper, 1, -1):
        valid = True
        for _, test_idx in GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups):
            if len(np.unique(y[test_idx])) < 2:
                valid = False
                break
        if valid:
            return n_splits
    return 2


def fit_dense(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, c_value: float) -> np.ndarray:
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)
    clf = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
        solver="liblinear",
    )
    clf.fit(x_train_s, y_train)
    return clf.predict_proba(x_test_s)[:, 1]


def fit_text(
    train_texts: list[str],
    y_train: np.ndarray,
    test_texts: list[str],
    c_value: float,
    train_numeric: np.ndarray | None = None,
    test_numeric: np.ndarray | None = None,
) -> np.ndarray:
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=20000)
    x_train = vectorizer.fit_transform(train_texts)
    x_test = vectorizer.transform(test_texts)
    if train_numeric is not None and test_numeric is not None:
        scaler = StandardScaler()
        train_num = scaler.fit_transform(train_numeric)
        test_num = scaler.transform(test_numeric)
        x_train = sparse.hstack([x_train, sparse.csr_matrix(train_num)], format="csr")
        x_test = sparse.hstack([x_test, sparse.csr_matrix(test_num)], format="csr")
    clf = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        random_state=0,
        solver="liblinear",
    )
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_test)[:, 1]


def inner_dense_score(x: np.ndarray, y: np.ndarray, groups: np.ndarray, train_idx: np.ndarray, c_value: float) -> float:
    n_splits = choose_valid_group_splits(y[train_idx], groups[train_idx], max_splits=3)
    scores = []
    for inner_train_rel, inner_val_rel in GroupKFold(n_splits=n_splits).split(x[train_idx], y[train_idx], groups[train_idx]):
        inner_train = train_idx[inner_train_rel]
        inner_val = train_idx[inner_val_rel]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        scores.append(roc_auc_score(y[inner_val], fit_dense(x[inner_train], y[inner_train], x[inner_val], c_value)))
    return float(np.mean(scores)) if scores else 0.5


def nested_hidden(
    x_by_layer: dict[int, np.ndarray],
    layers: list[int],
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int], list[float]]:
    n_splits = choose_valid_group_splits(y, groups, max_splits=5)
    probs = np.zeros(len(y), dtype=float)
    used = np.zeros(len(y), dtype=bool)
    selected_layers: list[int] = []
    selected_cs: list[float] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x_by_layer[layers[0]], y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        best = (-np.inf, layers[0], 1.0)
        for layer in layers:
            for c_value in C_GRID:
                score = inner_dense_score(x_by_layer[layer], y, groups, train_idx, c_value)
                if score > best[0]:
                    best = (score, layer, c_value)
        _, layer, c_value = best
        probs[test_idx] = fit_dense(x_by_layer[layer][train_idx], y[train_idx], x_by_layer[layer][test_idx], c_value)
        used[test_idx] = True
        selected_layers.append(layer)
        selected_cs.append(c_value)
    return probs, used, selected_layers, selected_cs


def nested_dense_fixed(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[float]]:
    n_splits = choose_valid_group_splits(y, groups, max_splits=5)
    probs = np.zeros(len(y), dtype=float)
    used = np.zeros(len(y), dtype=bool)
    selected_cs: list[float] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        best = (-np.inf, 1.0)
        for c_value in C_GRID:
            score = inner_dense_score(x, y, groups, train_idx, c_value)
            if score > best[0]:
                best = (score, c_value)
        _, c_value = best
        probs[test_idx] = fit_dense(x[train_idx], y[train_idx], x[test_idx], c_value)
        used[test_idx] = True
        selected_cs.append(c_value)
    return probs, used, selected_cs


def inner_text_score(
    texts: list[str],
    numeric: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    c_value: float,
) -> float:
    n_splits = choose_valid_group_splits(y[train_idx], groups[train_idx], max_splits=3)
    scores = []
    for inner_train_rel, inner_val_rel in GroupKFold(n_splits=n_splits).split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx]):
        inner_train = train_idx[inner_train_rel]
        inner_val = train_idx[inner_val_rel]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        probs = fit_text(
            [texts[i] for i in inner_train],
            y[inner_train],
            [texts[i] for i in inner_val],
            c_value,
            None if numeric is None else numeric[inner_train],
            None if numeric is None else numeric[inner_val],
        )
        scores.append(roc_auc_score(y[inner_val], probs))
    return float(np.mean(scores)) if scores else 0.5


def nested_text(
    texts: list[str],
    numeric: np.ndarray | None,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    n_splits = choose_valid_group_splits(y, groups, max_splits=5)
    probs = np.zeros(len(y), dtype=float)
    used = np.zeros(len(y), dtype=bool)
    selected_cs: list[float] = []
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        best = (-np.inf, 1.0)
        for c_value in C_GRID:
            score = inner_text_score(texts, numeric, y, groups, train_idx, c_value)
            if score > best[0]:
                best = (score, c_value)
        _, c_value = best
        probs[test_idx] = fit_text(
            [texts[i] for i in train_idx],
            y[train_idx],
            [texts[i] for i in test_idx],
            c_value,
            None if numeric is None else numeric[train_idx],
            None if numeric is None else numeric[test_idx],
        )
        used[test_idx] = True
        selected_cs.append(c_value)
    return probs, used, selected_cs


def cluster_bootstrap(y: np.ndarray, probs: np.ndarray, groups: np.ndarray, n_bootstrap: int, seed: int) -> dict[str, float]:
    point = roc_auc_score(y, probs)
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    vals = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], probs[idx]))
    if not vals:
        return {"auroc": point, "ci_low": point, "ci_high": point, "se": 0.0}
    vals_arr = np.asarray(vals)
    return {
        "auroc": point,
        "ci_low": float(np.percentile(vals_arr, 2.5)),
        "ci_high": float(np.percentile(vals_arr, 97.5)),
        "se": float(vals_arr.std(ddof=1)),
    }


def summarize_method(
    model: str,
    method: str,
    y: np.ndarray,
    probs: np.ndarray,
    used: np.ndarray,
    groups: np.ndarray,
    rows: list[dict[str, Any]],
    n_bootstrap: int,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_u = y[used]
    p_u = probs[used]
    g_u = groups[used]
    ci = cluster_bootstrap(y_u, p_u, g_u, n_bootstrap, seed)
    summary = {
        "model": model,
        "method": method,
        "n": int(used.sum()),
        "n_dep": int(y_u.sum()),
        "n_indep": int((1 - y_u).sum()),
        "auroc": ci["auroc"],
        "auprc": average_precision_score(y_u, p_u),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "se": ci["se"],
    }
    if extra:
        summary.update(extra)
    pred_rows = []
    for row, prob, keep in zip(rows, probs, used):
        if not keep:
            continue
        pred_rows.append(
            {
                "model": model,
                "method": method,
                "dataset": row["dataset"],
                "question_id": row["question_id"],
                "original_question_id": row["original_question_id"],
                "template_id": row["template_id"],
                "pair_strategy": row["pair_strategy"],
                "label": row["label"],
                "oof_prob": float(prob),
            }
        )
    return summary, pred_rows


def run_model(model: str, data_dir: Path, n_bootstrap: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = build_rows(model, data_dir)
    if len(rows) < 10 or len({row["label"] for row in rows}) < 2:
        print(f"[{model}] insufficient selected rows: {len(rows)}", flush=True)
        return [], []
    y = np.array([row["label"] for row in rows], dtype=int)
    groups = np.array([row["group"] for row in rows])
    numeric = np.array(
        [
            [
                row["h1_prior_logp"],
                row["h2_prior_logp"],
                row["prior_abs_gap"],
                row["prior_signed_h1_minus_h2"],
                row["neutral_matches_h1"],
            ]
            for row in rows
        ],
        dtype=float,
    )
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)

    summary_rows: list[dict[str, Any]] = []
    pred_rows: list[dict[str, Any]] = []

    for source, method in [
        ("w", "hidden_w"),
        ("g", "hidden_g_control"),
        ("w_minus_g", "hidden_w_minus_g"),
    ]:
        layers, x_by_layer = load_hidden_by_layer(rows, source)
        probs, used, selected_layers, selected_cs = nested_hidden(x_by_layer, layers, y, groups)
        if used.any() and len(np.unique(y[used])) == 2:
            summary, preds = summarize_method(
                model,
                method,
                y,
                probs,
                used,
                groups,
                rows,
                n_bootstrap,
                seed,
                {
                    "selected_layers": ",".join(map(str, selected_layers)),
                    "selected_cs": ",".join(map(str, selected_cs)),
                },
            )
            summary_rows.append(summary)
            pred_rows.extend(preds)

    for method, texts, num in [
        ("prior_numeric", None, numeric),
        ("w_answer_text_tfidf", [row["w_answer_text"] for row in rows], None),
        ("w_public_text_tfidf", [row["w_public_text"] for row in rows], None),
        ("w_public_text_plus_prior", [row["w_public_text"] for row in rows], numeric),
    ]:
        if texts is None:
            probs, used, selected_cs = nested_dense_fixed(num, y, groups)
        else:
            probs, used, selected_cs = nested_text(texts, num, y, groups)
        if used.any() and len(np.unique(y[used])) == 2:
            summary, preds = summarize_method(
                model,
                method,
                y,
                probs,
                used,
                groups,
                rows,
                n_bootstrap,
                seed,
                {"selected_cs": ",".join(map(str, selected_cs))},
            )
            summary_rows.append(summary)
            pred_rows.extend(preds)

    print(
        f"[{model}] rows={len(rows)} dep={int(y.sum())} indep={int((1-y).sum())} "
        f"methods={len(summary_rows)}",
        flush=True,
    )
    return summary_rows, pred_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True, help="extract_states.py output with selected_pairs.csv")
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"], choices=["mistral", "gemma", "llama"])
    parser.add_argument("--output-dir", type=Path, default=None, help="default: <data-dir>/analysis")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    output_dir = args.output_dir or (args.data_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_summary: list[dict[str, Any]] = []
    all_preds: list[dict[str, Any]] = []
    for model in args.models:
        summary, preds = run_model(model, args.data_dir, args.n_bootstrap, args.seed)
        all_summary.extend(summary)
        all_preds.extend(preds)
    pd.DataFrame(all_summary).to_csv(output_dir / "balanced_v3_probe_baseline_summary.csv", index=False)
    pd.DataFrame(all_preds).to_csv(output_dir / "balanced_v3_probe_baseline_predictions.csv", index=False)
    if all_summary:
        print(pd.DataFrame(all_summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
