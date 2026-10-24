"""Prior-aware v2 audit (tab:prior-aware): retained same-answer slice of the v2 rebuild (strict label rule) and
nested-CV AUROC of the W state at the strict fixed layer (11/13/13), the G state, W-G, prior scalars and public text.
Input: build_v2.py output (traces/, features/, selected_prior_matched_pairs.jsonl). Output: prior_aware_audit_{summary,
predictions,folds}.csv; the paper prints slice=all, method=hidden_w_selected_layer and its n_dep/n_indep.

    python -m scripts.pipeline.prior_aware.analyze_v2 --data-dir <v2> --output-dir <v2>/audit_full
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.pipeline.main_task import (
    TEXT_CONFIGS,
    cluster_bootstrap_metric,
    load_state_matrix,
    make_text,
    selected_layer_map,
)
from scripts.pipeline.analysis.noninternal_baselines import token_count
from src.core.slice import build_main_task_records, load_jsonl


DATASETS = ["triviaqa", "hotpotqa"]
TEXT_CONFIGS_FAST = [
    config for config in TEXT_CONFIGS
    if config["name"] == "word12"
]
C_GRID_FAST = [0.1, 1.0, 10.0]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    use_text: bool
    use_numeric: bool
    use_hidden: bool
    hidden_source: str | None = None


METHODS = [
    MethodSpec("prior_scalar", False, True, False),
    MethodSpec("w_public_text", True, False, False),
    MethodSpec("text_plus_prior", True, True, False),
    MethodSpec("hidden_w_selected_layer", False, False, True, "w"),
    MethodSpec("hidden_g_control_layer", False, False, True, "g"),
    MethodSpec("hidden_w_minus_g_layer", False, False, True, "w_minus_g"),
    MethodSpec("text_prior_plus_w_hidden", True, True, True, "w"),
]


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def load_pair_map(data_dir: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in load_jsonl(data_dir / "selected_prior_matched_pairs.jsonl"):
        out[(row["model"], row["dataset"], str(row["question_id"]))] = row
    return out


def attach_pair_fields(
    model: str,
    records: list[dict[str, Any]],
    pair_map: dict[tuple[str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    enriched = []
    for row in records:
        pair = pair_map.get((model, row["dataset"], str(row["question_id"])))
        if pair is None:
            continue
        item = dict(row)
        h1_prior = safe_float(pair.get("h1_prior_logp"))
        h2_prior = safe_float(pair.get("h2_prior_logp"))
        item.update(
            {
                "pair_strategy": str(pair.get("pair_strategy", "single")),
                "h1_prior_logp": h1_prior,
                "h2_prior_logp": h2_prior,
                "prior_signed_gap_h1_minus_h2": h1_prior - h2_prior,
                "prior_abs_gap": abs(h1_prior - h2_prior),
                "prior_mean_logp": 0.5 * (h1_prior + h2_prior),
                "prior_min_logp": min(h1_prior, h2_prior),
                "prior_max_logp": max(h1_prior, h2_prior),
                "h1_len_chars": len(str(row.get("h1") or "")),
                "h2_len_chars": len(str(row.get("h2") or "")),
                "h1_token_count": token_count(row.get("h1")),
                "h2_token_count": token_count(row.get("h2")),
                "candidate_char_delta": abs(len(str(row.get("h1") or "")) - len(str(row.get("h2") or ""))),
                "candidate_token_delta": abs(token_count(row.get("h1")) - token_count(row.get("h2"))),
                "dataset_is_triviaqa": float(row["dataset"] == "triviaqa"),
                "dataset_is_hotpotqa": float(row["dataset"] == "hotpotqa"),
                "strategy_is_closest": float(pair.get("pair_strategy") == "closest"),
                "strategy_is_dominant": float(pair.get("pair_strategy") == "dominant"),
                "pair_top_k": safe_float(pair.get("pair_top_k")),
                "candidate_pool_size": safe_float(pair.get("candidate_pool_size")),
            }
        )
        enriched.append(item)
    return enriched


NUMERIC_KEYS = [
    "h1_prior_logp",
    "h2_prior_logp",
    "prior_signed_gap_h1_minus_h2",
    "prior_abs_gap",
    "prior_mean_logp",
    "prior_min_logp",
    "prior_max_logp",
    "h1_len_chars",
    "h2_len_chars",
    "h1_token_count",
    "h2_token_count",
    "candidate_char_delta",
    "candidate_token_delta",
    "dataset_is_triviaqa",
    "dataset_is_hotpotqa",
    "strategy_is_closest",
    "strategy_is_dominant",
    "pair_top_k",
    "candidate_pool_size",
]


def numeric_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[safe_float(row.get(key)) for key in NUMERIC_KEYS] for row in records],
        dtype=np.float32,
    )


def build_slices(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {"all": list(range(len(records)))}
    for threshold in [0.25, 0.5, 0.75, 1.0]:
        label = str(threshold).replace(".", "p")
        out[f"prior_gap_le_{label}"] = [
            i for i, row in enumerate(records)
            if safe_float(row.get("prior_abs_gap")) <= threshold
        ]
    for dataset in DATASETS:
        out[f"dataset_{dataset}"] = [
            i for i, row in enumerate(records)
            if row["dataset"] == dataset
        ]
    strategies = sorted({str(row.get("pair_strategy")) for row in records if row.get("pair_strategy")})
    for strategy in strategies:
        out[f"strategy_{strategy}"] = [
            i for i, row in enumerate(records)
            if str(row.get("pair_strategy")) == strategy
        ]
    return out


def transform_numeric(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    imputer = SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)
    scaler = StandardScaler()
    train_i = imputer.fit_transform(train)
    test_i = imputer.transform(test)
    return sparse.csr_matrix(scaler.fit_transform(train_i)), sparse.csr_matrix(scaler.transform(test_i))


def build_blocks(
    records: list[dict[str, Any]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    spec: MethodSpec,
    text_config: dict[str, Any] | None,
    numeric: np.ndarray,
    hidden: np.ndarray | None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    train_blocks = []
    test_blocks = []
    if spec.use_text:
        assert text_config is not None
        texts = np.asarray([make_text(row, "prompt_cot_answer") for row in records], dtype=object)
        vectorizer = TfidfVectorizer(
            analyzer=text_config["analyzer"],
            ngram_range=text_config["ngram_range"],
            max_features=text_config["max_features"],
            min_df=2,
            lowercase=True,
            strip_accents="unicode",
        )
        train_blocks.append(vectorizer.fit_transform(texts[train_idx]))
        test_blocks.append(vectorizer.transform(texts[test_idx]))
    if spec.use_numeric:
        train_num, test_num = transform_numeric(numeric[train_idx], numeric[test_idx])
        train_blocks.append(train_num)
        test_blocks.append(test_num)
    if spec.use_hidden:
        if hidden is None:
            raise RuntimeError("hidden requested but missing")
        train_hid, test_hid = transform_numeric(hidden[train_idx], hidden[test_idx])
        train_blocks.append(train_hid)
        test_blocks.append(test_hid)
    return sparse.hstack(train_blocks, format="csr"), sparse.hstack(test_blocks, format="csr")


def fit_predict(
    records: list[dict[str, Any]],
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    y: np.ndarray,
    spec: MethodSpec,
    text_config: dict[str, Any] | None,
    c_value: float,
    numeric: np.ndarray,
    hidden: np.ndarray | None,
    seed: int,
) -> np.ndarray:
    x_train, x_test = build_blocks(records, train_idx, test_idx, spec, text_config, numeric, hidden)
    clf = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=5000,
        random_state=seed,
        solver="liblinear",
    )
    clf.fit(x_train, y[train_idx])
    return clf.predict_proba(x_test)[:, 1]


def inner_score(
    records: list[dict[str, Any]],
    train_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    spec: MethodSpec,
    text_config: dict[str, Any] | None,
    c_value: float,
    numeric: np.ndarray,
    hidden: np.ndarray | None,
    seed: int,
) -> float:
    train_groups = np.unique(groups[train_idx])
    n_splits = min(3, len(train_groups))
    if n_splits < 2:
        return 0.5
    scores = []
    splitter = GroupKFold(n_splits=n_splits)
    for inner_train_rel, inner_val_rel in splitter.split(train_idx, y[train_idx], groups[train_idx]):
        inner_train = train_idx[inner_train_rel]
        inner_val = train_idx[inner_val_rel]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        probs = fit_predict(
            records,
            inner_train,
            inner_val,
            y,
            spec,
            text_config,
            c_value,
            numeric,
            hidden,
            seed,
        )
        scores.append(float(roc_auc_score(y[inner_val], probs)))
    return float(np.mean(scores)) if scores else 0.5


def candidate_grid(spec: MethodSpec) -> list[tuple[dict[str, Any] | None, float]]:
    configs = TEXT_CONFIGS_FAST if spec.use_text else [None]
    return [(config, c_value) for config in configs for c_value in C_GRID_FAST]


def nested_oof(
    records: list[dict[str, Any]],
    y: np.ndarray,
    groups: np.ndarray,
    spec: MethodSpec,
    numeric: np.ndarray,
    hidden: np.ndarray | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    probs = np.zeros(len(y), dtype=np.float32)
    used = np.zeros(len(y), dtype=bool)
    fold_rows = []
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return probs, used, pd.DataFrame()
    outer = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(outer.split(np.zeros(len(y)), y, groups)):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        best_score = -math.inf
        best_config: dict[str, Any] | None = None
        best_c = 1.0
        for text_config, c_value in candidate_grid(spec):
            score = inner_score(
                records,
                train_idx,
                y,
                groups,
                spec,
                text_config,
                c_value,
                numeric,
                hidden,
                seed + fold,
            )
            if score > best_score:
                best_score = score
                best_config = text_config
                best_c = c_value
        probs[test_idx] = fit_predict(
            records,
            train_idx,
            test_idx,
            y,
            spec,
            best_config,
            best_c,
            numeric,
            hidden,
            seed + fold,
        )
        used[test_idx] = True
        fold_rows.append(
            {
                "fold": fold,
                "C": best_c,
                "text_config": best_config["name"] if best_config else "",
                "inner_auroc": best_score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
    return probs, used, pd.DataFrame(fold_rows)


def summarize(
    y: np.ndarray,
    probs: np.ndarray,
    used: np.ndarray,
    groups: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    if not used.any() or len(np.unique(y[used])) < 2:
        return {
            "n_used": int(used.sum()),
            "auroc": math.nan,
            "auroc_ci_low": math.nan,
            "auroc_ci_high": math.nan,
            "auprc": math.nan,
            "auprc_ci_low": math.nan,
            "auprc_ci_high": math.nan,
        }
    auroc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auroc", n_bootstrap, seed)
    auprc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auprc", n_bootstrap, seed + 1)
    return {
        "n_used": int(used.sum()),
        "auroc": auroc["point"],
        "auroc_ci_low": auroc["ci_low"],
        "auroc_ci_high": auroc["ci_high"],
        "auprc": auprc["point"],
        "auprc_ci_low": auprc["ci_low"],
        "auprc_ci_high": auprc["ci_high"],
    }


def run_model(model: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    data_dir = Path(args.data_dir)
    pair_map = load_pair_map(data_dir)
    records = attach_pair_fields(model, build_main_task_records(model, data_dir), pair_map)
    if not records:
        return [], [], []

    layer_map = selected_layer_map(Path(args.probe_layers))
    if model not in layer_map:
        raise RuntimeError(f"No selected layer for {model} in {args.probe_layers}")
    selected_layer = layer_map[model]
    w_hidden_by_record: dict[int, np.ndarray] = {}
    g_hidden_by_record: dict[int, np.ndarray] = {}
    _, hidden_by_layer, kept = load_state_matrix(records, "w", data_dir, only_layers=[selected_layer])
    for pos, original_idx in enumerate(kept):
        w_hidden_by_record[original_idx] = hidden_by_layer[selected_layer][pos]
    _, g_hidden_by_layer, g_kept = load_state_matrix(records, "g", data_dir, only_layers=[selected_layer])
    for pos, original_idx in enumerate(g_kept):
        g_hidden_by_record[original_idx] = g_hidden_by_layer[selected_layer][pos]

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for slice_name, indices in build_slices(records).items():
        slice_labels = np.asarray([records[i]["label"] for i in indices], dtype=int)
        n_dep = int(slice_labels.sum())
        n_indep = int(len(slice_labels) - slice_labels.sum())
        if n_dep < args.min_class_count or n_indep < args.min_class_count:
            continue
        slice_records = [records[i] for i in indices]
        y = np.asarray([row["label"] for row in slice_records], dtype=int)
        groups = np.asarray([row["group"] for row in slice_records], dtype=str)
        numeric = numeric_matrix(slice_records)
        hidden_sources: dict[str, np.ndarray] = {}
        if all(i in w_hidden_by_record for i in indices):
            hidden_sources["w"] = np.stack([w_hidden_by_record[i] for i in indices], axis=0)
        if all(i in g_hidden_by_record for i in indices):
            hidden_sources["g"] = np.stack([g_hidden_by_record[i] for i in indices], axis=0)
        if "w" in hidden_sources and "g" in hidden_sources:
            hidden_sources["w_minus_g"] = hidden_sources["w"] - hidden_sources["g"]

        for spec in METHODS:
            hidden = hidden_sources.get(spec.hidden_source) if spec.use_hidden else None
            if spec.use_hidden and hidden is None:
                continue
            probs, used, folds = nested_oof(
                slice_records,
                y,
                groups,
                spec,
                numeric,
                hidden,
                args.seed,
            )
            metrics = summarize(y, probs, used, groups, args.n_bootstrap, args.seed)
            row = {
                "model": model,
                "slice": slice_name,
                "method": spec.name,
                "selected_layer": selected_layer if spec.use_hidden else math.nan,
                "n": len(slice_records),
                "n_dep": n_dep,
                "n_indep": n_indep,
                "mean_prior_abs_gap": float(np.nanmean([r["prior_abs_gap"] for r in slice_records])),
                **metrics,
            }
            summary_rows.append(row)
            print(
                f"{model:7s} {slice_name:20s} {spec.name:24s} "
                f"AUROC={metrics['auroc']:.3f} N={metrics['n_used']}",
                flush=True,
            )
            for local_idx, record in enumerate(slice_records):
                prediction_rows.append(
                    {
                        "model": model,
                        "slice": slice_name,
                        "method": spec.name,
                        "dataset": record["dataset"],
                        "question_id": record["question_id"],
                        "group": record["group"],
                        "label": record["label"],
                        "prior_abs_gap": record["prior_abs_gap"],
                        "prob": float(probs[local_idx]),
                        "used": bool(used[local_idx]),
                    }
                )
            if not folds.empty:
                folds = folds.copy()
                folds.insert(0, "method", spec.name)
                folds.insert(0, "slice", slice_name)
                folds.insert(0, "model", model)
                fold_rows.extend(folds.to_dict("records"))
    return summary_rows, prediction_rows, fold_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="build_v2.py output: traces/, features/, selected pairs")
    parser.add_argument("--probe-layers", default="data/scores/probes/nested_probe_layers.csv")
    parser.add_argument("--output-dir", default=None, help="default: <data-dir>/audit")
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"], choices=["mistral", "gemma", "llama"])
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.data_dir) / "audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for model in args.models:
        model_summary, model_predictions, model_folds = run_model(model, args)
        summary_rows.extend(model_summary)
        prediction_rows.extend(model_predictions)
        fold_rows.extend(model_folds)

    summary = pd.DataFrame(summary_rows).sort_values(["model", "slice", "method"])
    predictions = pd.DataFrame(prediction_rows).sort_values(["model", "slice", "method", "dataset", "question_id"])
    folds = pd.DataFrame(fold_rows).sort_values(["model", "slice", "method", "fold"])
    summary.to_csv(output_dir / "prior_aware_audit_summary.csv", index=False)
    predictions.to_csv(output_dir / "prior_aware_audit_predictions.csv", index=False)
    folds.to_csv(output_dir / "prior_aware_audit_folds.csv", index=False)
    print(f"Wrote {output_dir / 'prior_aware_audit_summary.csv'}")


if __name__ == "__main__":
    main()
