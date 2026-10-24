"""Combined non-internal baselines (tab:noninternal): public text, no-hint prior, candidate and design features,
alone and with the fixed-layer W state, all with question-grouped nested CV.

Inputs: --data-dir with traces/, features/ (*.npz and *_scores.json) and neutral_accuracy/ (no_hint_traces.py), plus
nested_probe_layers.csv for the fixed layer. Outputs: combined_baseline_{summary,predictions,folds}.csv; the
summary's all / neutral_not_h1 / neutral_neither_h1_nor_h2 AUROCs are the combined_noninternal rows of
data/scores/controls/strict_rule_summary.csv. Run: python -m scripts.pipeline.analysis.noninternal_baselines --data-dir <store>
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
    PROJECT_ROOT,
    TEXT_CONFIGS,
    cluster_bootstrap_metric,
    load_state_matrix,
    make_text,
    selected_layer_map,
)
from src.core.slice import answers_match, build_main_task_records, load_jsonl

DATASET_ORDER = ["triviaqa", "hotpotqa", "gpqa", "mmlu_pro"]
# One public-text view (the fullest W-trace text) and one TF-IDF config keep the nested grid tractable.
TEXT_VIEWS = ["prompt_cot_answer"]
TEXT_CONFIGS_FAST = [config for config in TEXT_CONFIGS if config["name"] == "word12"]
C_GRID_FAST = [0.1, 1.0, 10.0]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    use_text: bool
    use_public_numeric: bool
    use_design_numeric: bool
    use_hidden: bool


METHODS = [
    MethodSpec("prior_candidate", False, True, False, False),
    MethodSpec("w_public_text", True, False, False, False),
    MethodSpec("w_text_prior_candidate", True, True, False, False),
    MethodSpec("design_scalar", False, True, True, False),
    MethodSpec("w_text_prior_candidate_design", True, True, True, False),
    MethodSpec("hidden_fixed_layer", False, False, False, True),
    MethodSpec("w_text_prior_candidate_plus_hidden", True, True, False, True),
    MethodSpec("w_text_prior_candidate_design_plus_hidden", True, True, True, True),
]


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out if math.isfinite(out) else math.nan


def normalize_for_substring(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def token_count(text: Any) -> int:
    return len(str(text or "").split())


def load_neutral_map(data_dir: Path, model: str) -> dict[tuple[str, str], dict[str, Any]]:
    path = data_dir / "neutral_accuracy" / f"{model}_neutral_traces.jsonl"
    if not path.exists():
        return {}
    return {(row["dataset"], str(row["question_id"])): row for row in load_jsonl(path)}


def resolve_score_path(trace: dict[str, Any], data_dir: Path) -> Path | None:
    raw = trace.get("answer_score_path")
    if not raw:
        return None
    path = Path(raw)
    for candidate in [path, PROJECT_ROOT / path, data_dir / "features" / path.name]:
        if candidate.exists():
            return candidate
    return None


def load_scores(trace: dict[str, Any], data_dir: Path) -> dict[str, float]:
    """Commitment scores written at generation time (C, L_hint, L_gold); empty if missing."""
    path = resolve_score_path(trace, data_dir)
    if path is None:
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return {
        key: float(value)
        for key, value in data.items()
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    }


def commitment_fields(record: dict[str, Any], data_dir: Path) -> dict[str, float | None]:
    treatment = load_scores(record["w"], data_dir)
    control = load_scores(record["g"], data_dir)
    out = {
        "treatment_commitment": treatment.get("C"),
        "control_commitment": control.get("C"),
        "commitment_delta_treatment_minus_control": None,
        "treatment_l_hint": treatment.get("L_hint"),
        "treatment_l_correct": treatment.get("L_gold"),
        "control_l_hint": control.get("L_hint"),
        "control_l_correct": control.get("L_gold"),
    }
    if out["treatment_commitment"] is not None and out["control_commitment"] is not None:
        out["commitment_delta_treatment_minus_control"] = (
            float(out["treatment_commitment"]) - float(out["control_commitment"])
        )
    return out


def build_slices(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    return {
        "all": list(range(len(records))),
        "span_answer_only": [i for i, row in enumerate(records) if row["dataset"] in {"triviaqa", "hotpotqa"}],
        "mcq_only": [i for i, row in enumerate(records) if row["dataset"] in {"gpqa", "mmlu_pro"}],
        "neutral_available": [i for i, row in enumerate(records) if row.get("neutral_available")],
        "neutral_not_h1": [
            i for i, row in enumerate(records)
            if row.get("neutral_available") and not row.get("neutral_matches_h1")
        ],
        "neutral_neither_h1_nor_h2": [
            i for i, row in enumerate(records)
            if row.get("neutral_available") and not row.get("neutral_matches_h1") and not row.get("neutral_matches_h2")
        ],
    }


def attach_auxiliary_fields(records: list[dict[str, Any]], data_dir: Path, model: str) -> list[dict[str, Any]]:
    neutral = load_neutral_map(data_dir, model)
    all_h1 = [normalize_for_substring(row["h1"]) for row in records]
    all_h2 = [normalize_for_substring(row["h2"]) for row in records]
    h1_freq = {value: all_h1.count(value) for value in set(all_h1)}
    h2_freq = {value: all_h2.count(value) for value in set(all_h2)}
    enriched = []
    for row in records:
        item = dict(row)
        nrow = neutral.get((row["dataset"], str(row["question_id"])))
        q = normalize_for_substring(row.get("question"))
        cot = normalize_for_substring(row.get("reasoning_text"))
        h1 = normalize_for_substring(row.get("h1"))
        h2 = normalize_for_substring(row.get("h2"))

        item["neutral_available"] = bool(nrow and nrow.get("parse_ok"))
        item["neutral_answer"] = nrow.get("normalized_final") if nrow else None
        item["neutral_correct"] = float(bool(nrow.get("is_correct"))) if item["neutral_available"] else math.nan
        item["neutral_matches_h1"] = float(
            bool(item["neutral_available"] and answers_match(nrow.get("normalized_final"), row["h1"]))
        )
        item["neutral_matches_h2"] = float(
            bool(item["neutral_available"] and answers_match(nrow.get("normalized_final"), row["h2"]))
        )

        item["h1_len_chars"] = len(str(row.get("h1") or ""))
        item["h2_len_chars"] = len(str(row.get("h2") or ""))
        item["candidate_char_delta"] = abs(item["h1_len_chars"] - item["h2_len_chars"])
        item["h1_token_count"] = token_count(row.get("h1"))
        item["h2_token_count"] = token_count(row.get("h2"))
        item["candidate_token_delta"] = abs(item["h1_token_count"] - item["h2_token_count"])
        item["h1_in_question"] = float(bool(h1 and h1 in q))
        item["h2_in_question"] = float(bool(h2 and h2 in q))
        item["h1_in_cot"] = float(bool(h1 and h1 in cot))
        item["h2_in_cot"] = float(bool(h2 and h2 in cot))
        item["h1_string_frequency"] = h1_freq.get(h1, 0)
        item["h2_string_frequency"] = h2_freq.get(h2, 0)
        item["is_span_answer"] = float(row["dataset"] in {"triviaqa", "hotpotqa"})
        item["is_mcq"] = float(row["dataset"] in {"gpqa", "mmlu_pro"})
        for dataset in DATASET_ORDER:
            item[f"dataset_is_{dataset}"] = float(row["dataset"] == dataset)
        for key_name, value in commitment_fields(row, data_dir).items():
            item[key_name] = safe_float(value)
        enriched.append(item)
    return enriched


PUBLIC_NUMERIC_KEYS = [
    "neutral_available",
    "neutral_matches_h1",
    "neutral_correct",
    "h1_len_chars",
    "h1_token_count",
    "h1_in_question",
    "h1_in_cot",
    "h1_string_frequency",
    "is_span_answer",
    "is_mcq",
    "dataset_is_triviaqa",
    "dataset_is_hotpotqa",
    "dataset_is_gpqa",
    "dataset_is_mmlu_pro",
]

DESIGN_NUMERIC_KEYS = [
    "neutral_matches_h2",
    "h2_len_chars",
    "candidate_char_delta",
    "h2_token_count",
    "candidate_token_delta",
    "h2_in_question",
    "h2_in_cot",
    "h2_string_frequency",
    "treatment_commitment",
    "control_commitment",
    "commitment_delta_treatment_minus_control",
    "treatment_l_hint",
    "treatment_l_correct",
    "control_l_hint",
    "control_l_correct",
]


def numeric_matrix(records: list[dict[str, Any]], keys: list[str]) -> np.ndarray:
    if not keys:
        return np.zeros((len(records), 0), dtype=np.float32)
    return np.asarray([[safe_float(row.get(key)) for key in keys] for row in records], dtype=np.float32)


def fit_transform_dense(train: np.ndarray, test: np.ndarray) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    if train.shape[1] == 0:
        return sparse.csr_matrix((len(train), 0)), sparse.csr_matrix((len(test), 0))
    imputer = SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)
    scaler = StandardScaler()
    train_s = scaler.fit_transform(imputer.fit_transform(train))
    test_s = scaler.transform(imputer.transform(test))
    return sparse.csr_matrix(train_s), sparse.csr_matrix(test_s)


def build_feature_blocks(
    records: list[dict[str, Any]],
    indices_train: np.ndarray,
    indices_test: np.ndarray,
    spec: MethodSpec,
    text_view: str | None,
    text_config: dict[str, Any] | None,
    numeric: np.ndarray,
    hidden: np.ndarray | None,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    train_blocks = []
    test_blocks = []
    if spec.use_text:
        texts = np.asarray([make_text(row, text_view) for row in records], dtype=object)
        vec = TfidfVectorizer(
            analyzer=text_config["analyzer"],
            ngram_range=text_config["ngram_range"],
            max_features=text_config["max_features"],
            min_df=2,
            lowercase=True,
            strip_accents="unicode",
        )
        train_blocks.append(vec.fit_transform(texts[indices_train]))
        test_blocks.append(vec.transform(texts[indices_test]))
    if spec.use_public_numeric or spec.use_design_numeric:
        num_train, num_test = fit_transform_dense(numeric[indices_train], numeric[indices_test])
        train_blocks.append(num_train)
        test_blocks.append(num_test)
    if spec.use_hidden:
        if hidden is None:
            raise RuntimeError(f"Method {spec.name} requested hidden features but none were loaded")
        hid_train, hid_test = fit_transform_dense(hidden[indices_train], hidden[indices_test])
        train_blocks.append(hid_train)
        test_blocks.append(hid_test)
    return sparse.hstack(train_blocks, format="csr"), sparse.hstack(test_blocks, format="csr")


def fit_predict(records, train_idx, test_idx, y, spec, text_view, text_config, c_value, numeric, hidden, seed) -> np.ndarray:
    x_train, x_test = build_feature_blocks(records, train_idx, test_idx, spec, text_view, text_config, numeric, hidden)
    clf = LogisticRegression(C=c_value, class_weight="balanced", max_iter=5000, random_state=seed, solver="liblinear")
    clf.fit(x_train, y[train_idx])
    return clf.predict_proba(x_test)[:, 1]


def inner_score(records, train_idx, y, groups, spec, text_view, text_config, c_value, numeric, hidden, seed) -> float:
    n_splits = min(3, len(np.unique(groups[train_idx])))
    if n_splits < 2:
        return 0.5
    scores = []
    for inner_train_rel, inner_val_rel in GroupKFold(n_splits=n_splits).split(train_idx, y[train_idx], groups[train_idx]):
        inner_train = train_idx[inner_train_rel]
        inner_val = train_idx[inner_val_rel]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        probs = fit_predict(records, inner_train, inner_val, y, spec, text_view, text_config, c_value, numeric, hidden, seed)
        scores.append(float(roc_auc_score(y[inner_val], probs)))
    return float(np.mean(scores)) if scores else 0.5


def candidate_grid(spec: MethodSpec) -> list[tuple[str | None, dict[str, Any] | None, float]]:
    if spec.use_text:
        return [(view, config, c) for view in TEXT_VIEWS for config in TEXT_CONFIGS_FAST for c in C_GRID_FAST]
    return [(None, None, c) for c in C_GRID_FAST]


def nested_group_cv(records, y, groups, spec, numeric, hidden, seed) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    probs = np.zeros(len(y), dtype=np.float32)
    used = np.zeros(len(y), dtype=bool)
    fold_rows = []
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return probs, used, pd.DataFrame()
    for fold, (train_idx, test_idx) in enumerate(GroupKFold(n_splits=n_splits).split(np.zeros(len(y)), y, groups)):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        best_score = -math.inf
        best = None
        for text_view, text_config, c_value in candidate_grid(spec):
            score = inner_score(
                records, train_idx, y, groups, spec, text_view, text_config, c_value, numeric, hidden, seed + fold
            )
            if score > best_score:
                best_score = score
                best = (text_view, text_config, c_value)
        text_view, text_config, c_value = best
        probs[test_idx] = fit_predict(
            records, train_idx, test_idx, y, spec, text_view, text_config, c_value, numeric, hidden, seed + fold
        )
        used[test_idx] = True
        fold_rows.append(
            {
                "fold": fold,
                "text_view": text_view or "",
                "text_config": text_config["name"] if text_config else "",
                "C": c_value,
                "inner_auroc": best_score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
    return probs, used, pd.DataFrame(fold_rows)


def summarize_predictions(y, probs, used, groups, n_bootstrap: int, seed: int) -> dict[str, Any]:
    if int(used.sum()) == 0 or len(np.unique(y[used])) < 2:
        return {
            "n_used": int(used.sum()),
            "auroc": math.nan, "auroc_ci_low": math.nan, "auroc_ci_high": math.nan,
            "auprc": math.nan, "auprc_ci_low": math.nan, "auprc_ci_high": math.nan,
        }
    auroc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auroc", n_bootstrap, seed)
    auprc = cluster_bootstrap_metric(y[used], probs[used], groups[used], "auprc", n_bootstrap, seed + 1000)
    return {
        "n_used": int(used.sum()),
        "auroc": auroc["point"],
        "auroc_ci_low": auroc["ci_low"],
        "auroc_ci_high": auroc["ci_high"],
        "auprc": auprc["point"],
        "auprc_ci_low": auprc["ci_low"],
        "auprc_ci_high": auprc["ci_high"],
    }


def run_model(model: str, args: argparse.Namespace) -> tuple[list[dict], list[dict], list[dict]]:
    data_dir = Path(args.data_dir)
    records = attach_auxiliary_fields(build_main_task_records(model, data_dir), data_dir, model)
    slices = build_slices(records)
    layer = selected_layer_map(Path(args.probe_layers)).get(model)
    hidden_by_record: dict[int, np.ndarray] = {}
    if layer is not None:
        _, hidden_by_layer, kept = load_state_matrix(records, "w", data_dir, only_layers=[layer])
        for pos, original_idx in enumerate(kept):
            hidden_by_record[original_idx] = hidden_by_layer[layer][pos]

    summary_rows, pred_rows, fold_rows = [], [], []
    for slice_name, indices in slices.items():
        labels = np.asarray([records[i]["label"] for i in indices], dtype=int)
        n_dep = int(labels.sum())
        n_indep = int(len(labels) - labels.sum())
        if n_dep < args.min_class_count or n_indep < args.min_class_count:
            continue
        slice_records = [records[i] for i in indices]
        y = np.asarray([row["label"] for row in slice_records], dtype=int)
        groups = np.asarray([row["group"] for row in slice_records], dtype=str)
        public_numeric = numeric_matrix(slice_records, PUBLIC_NUMERIC_KEYS)
        design_numeric = numeric_matrix(slice_records, PUBLIC_NUMERIC_KEYS + DESIGN_NUMERIC_KEYS)
        hidden = None
        if layer is not None and all(i in hidden_by_record for i in indices):
            hidden = np.stack([hidden_by_record[i] for i in indices], axis=0)

        for spec in METHODS:
            if spec.use_hidden and hidden is None:
                continue
            numeric = design_numeric if spec.use_design_numeric else public_numeric
            probs, used, folds = nested_group_cv(slice_records, y, groups, spec, numeric, hidden, args.seed)
            metrics = summarize_predictions(y, probs, used, groups, args.n_bootstrap, args.seed)
            summary_rows.append(
                {
                    "model": model,
                    "slice": slice_name,
                    "method": spec.name,
                    "selected_layer": layer if spec.use_hidden else math.nan,
                    "n": len(slice_records),
                    "n_dep": n_dep,
                    "n_indep": n_indep,
                    **metrics,
                }
            )
            for local_idx, row in enumerate(slice_records):
                pred_rows.append(
                    {
                        "model": model,
                        "slice": slice_name,
                        "method": spec.name,
                        "dataset": row["dataset"],
                        "question_id": row["question_id"],
                        "group": row["group"],
                        "label": row["label"],
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
            print(f"{model:7s} {slice_name:26s} {spec.name:42s} AUROC={metrics['auroc']:.3f} N={metrics['n_used']}", flush=True)
    return summary_rows, pred_rows, fold_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="trace store: traces/, features/, neutral_accuracy/")
    parser.add_argument("--probe-layers", default="data/scores/probes/nested_probe_layers.csv")
    parser.add_argument("--output-dir", default="outputs/noninternal_baselines")
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--min-class-count", type=int, default=10)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    if not Path(args.probe_layers).exists():
        parser.error(f"--probe-layers {args.probe_layers} not found; without it every hidden-state method is skipped")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows, pred_rows, fold_rows = [], [], []
    for model in args.models:
        model_summary, model_pred, model_folds = run_model(model, args)
        summary_rows.extend(model_summary)
        pred_rows.extend(model_pred)
        fold_rows.extend(model_folds)

    pd.DataFrame(summary_rows).sort_values(["model", "slice", "method"]).to_csv(
        output_dir / "combined_baseline_summary.csv", index=False
    )
    pd.DataFrame(pred_rows).sort_values(["model", "slice", "method", "dataset", "question_id"]).to_csv(
        output_dir / "combined_baseline_predictions.csv", index=False
    )
    pd.DataFrame(fold_rows).sort_values(["model", "slice", "method", "fold"]).to_csv(
        output_dir / "combined_baseline_folds.csv", index=False
    )
    print(f"Wrote {output_dir / 'combined_baseline_summary.csv'}")


if __name__ == "__main__":
    main()
