"""Does the W-state probe score add to public features? (results: Mistral +0.032 [+0.003, +0.062]; app:corrected).

Score-level stack: out-of-fold base scores, logit-transformed, in a second GroupKFold(5) logistic stack (balanced,
C=1), with a 1000-resample question-cluster bootstrap (seed 1729) for AUROCs and paired deltas. Inputs:
combined_baseline_predictions.csv (analysis/noninternal_baselines.py) and control_state_probe_predictions.csv
(main_task.py). Outputs: incremental_hidden_signal_{summary,deltas}.csv; the printed value is the slice=all delta
noninternal_plus_w minus strong_noninternal. Run: python -m scripts.pipeline.analysis.incremental_hidden_signal
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class FeatureSet:
    name: str
    features: tuple[str, ...]


FEATURE_SETS = [
    FeatureSet("w_hidden", ("w_prompt_hidden",)),
    FeatureSet("hidden_fixed_layer", ("hidden_fixed_layer",)),
    FeatureSet("g_control", ("g_control_hidden",)),
    FeatureSet("c_control", ("c_control_hidden",)),
    FeatureSet("public_text", ("w_public_text",)),
    FeatureSet("prior_candidate", ("prior_candidate",)),
    FeatureSet("strong_noninternal", ("w_text_prior_candidate_design",)),
    FeatureSet("noninternal_plus_w", ("w_text_prior_candidate_design", "w_prompt_hidden")),
    FeatureSet("noninternal_plus_g", ("w_text_prior_candidate_design", "g_control_hidden")),
    FeatureSet("noninternal_plus_g_plus_w", ("w_text_prior_candidate_design", "g_control_hidden", "w_prompt_hidden")),
    FeatureSet("noninternal_plus_c", ("w_text_prior_candidate_design", "c_control_hidden")),
    FeatureSet("noninternal_plus_c_plus_w", ("w_text_prior_candidate_design", "c_control_hidden", "w_prompt_hidden")),
    FeatureSet("g_plus_w", ("g_control_hidden", "w_prompt_hidden")),
]
DELTA_PAIRS = [
    ("noninternal_plus_w", "strong_noninternal"),
    ("noninternal_plus_g_plus_w", "noninternal_plus_g"),
    ("noninternal_plus_c_plus_w", "noninternal_plus_c"),
    ("g_plus_w", "g_control"),
]


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), 1e-5, 1 - 1e-5)
    return np.log(values / (1 - values))


def grouped_stack_scores(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Grouped-CV logistic stack over score features."""
    scores = np.full(len(y), np.nan, dtype=float)
    used = np.zeros(len(y), dtype=bool)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits < 2:
        return scores, used
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(x, y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train_idx])
        x_test = scaler.transform(x[test_idx])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=0, solver="liblinear")
        clf.fit(x_train, y[train_idx])
        scores[test_idx] = clf.predict_proba(x_test)[:, 1]
        used[test_idx] = True
    return scores, used


def cluster_bootstrap(
    y: np.ndarray,
    score_a: np.ndarray,
    groups: np.ndarray,
    *,
    score_b: np.ndarray | None = None,
    metric: str = "auroc",
    n_bootstrap: int = 1000,
    seed: int = 1729,
) -> dict[str, float]:
    """Metric of score_a, or the paired difference score_a - score_b, with a question-cluster bootstrap."""
    used = np.isfinite(score_a)
    if score_b is not None:
        used &= np.isfinite(score_b)
    y = y[used]
    score_a = score_a[used]
    groups = groups[used]
    if score_b is not None:
        score_b = score_b[used]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"point": math.nan, "ci_low": math.nan, "ci_high": math.nan, "se": math.nan}

    def metric_fn(labels: np.ndarray, scores: np.ndarray) -> float:
        if len(np.unique(labels)) < 2:
            return math.nan
        if metric == "auprc":
            return float(average_precision_score(labels, scores))
        return float(roc_auc_score(labels, scores))

    point = metric_fn(y, score_a) if score_b is None else metric_fn(y, score_a) - metric_fn(y, score_b)
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_to_idx = {group: np.flatnonzero(groups == group) for group in unique_groups}
    boot = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_idx[group] for group in sampled])
        labels = y[idx]
        if len(np.unique(labels)) < 2:
            continue
        if score_b is None:
            value = metric_fn(labels, score_a[idx])
        else:
            value = metric_fn(labels, score_a[idx]) - metric_fn(labels, score_b[idx])
        if math.isfinite(value):
            boot.append(value)
    if not boot:
        return {"point": point, "ci_low": math.nan, "ci_high": math.nan, "se": math.nan}
    values = np.asarray(boot)
    return {
        "point": point,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "se": float(values.std(ddof=1)) if len(values) > 1 else math.nan,
    }


def stack_feature_set(frame: pd.DataFrame, feature_set: FeatureSet) -> tuple[np.ndarray, np.ndarray]:
    """One feature: its score as is. Several: the grouped logistic stack over their logits."""
    x = np.column_stack([logit(frame[name].to_numpy(dtype=float)) for name in feature_set.features])
    if len(feature_set.features) == 1:
        scores = frame[feature_set.features[0]].to_numpy(dtype=float)
        return scores, np.isfinite(scores)
    return grouped_stack_scores(x, frame["label"].to_numpy(dtype=int), frame["group"].astype(str).to_numpy())


def score_frame(combined_predictions: Path, control_predictions: Path, model: str, slice_name: str) -> pd.DataFrame:
    """Join the used out-of-fold scores of the non-internal methods with the W/G/C control-state probe scores."""
    combined = pd.read_csv(combined_predictions)
    combined["used"] = combined["used"].astype(bool)
    key = ["model", "slice", "dataset", "question_id", "group", "label"]
    wanted = ["hidden_fixed_layer", "w_public_text", "prior_candidate", "w_text_prior_candidate", "w_text_prior_candidate_design"]
    pieces = []
    for method in wanted:
        part = combined[
            (combined["model"] == model)
            & (combined["slice"] == slice_name)
            & (combined["method"] == method)
            & (combined["used"])
        ][key + ["prob"]].rename(columns={"prob": method})
        pieces.append(part)
    if not pieces or pieces[0].empty:
        return pd.DataFrame()
    out = pieces[0]
    for part in pieces[1:]:
        out = out.merge(part, on=key, how="inner")

    controls = pd.read_csv(control_predictions)
    control_key = ["model", "dataset", "question_id", "group", "label"]
    for source, name in [("W_prompt", "w_prompt_hidden"), ("G_prompt", "g_control_hidden"), ("C_prompt", "c_control_hidden")]:
        part = controls[(controls["model"] == model) & (controls["source"] == source)][control_key + ["prob"]]
        out = out.merge(part.rename(columns={"prob": name}), on=control_key, how="inner")
    return out


def run_slice(frame: pd.DataFrame, model: str, slice_name: str, n_bootstrap: int, seed: int) -> tuple[list[dict], list[dict]]:
    if frame.empty or frame["label"].nunique() < 2:
        return [], []
    y = frame["label"].to_numpy(dtype=int)
    groups = frame["group"].astype(str).to_numpy()
    n_dep = int(y.sum())
    summaries = []
    score_columns = {}
    for feature_set in FEATURE_SETS:
        if any(name not in frame.columns for name in feature_set.features):
            continue
        scores, used = stack_feature_set(frame, feature_set)
        score_columns[feature_set.name] = scores
        auc = cluster_bootstrap(y[used], scores[used], groups[used], metric="auroc", n_bootstrap=n_bootstrap, seed=seed)
        ap = cluster_bootstrap(y[used], scores[used], groups[used], metric="auprc", n_bootstrap=n_bootstrap, seed=seed)
        summaries.append(
            {
                "experiment": "corrected_main",
                "model": model,
                "slice": slice_name,
                "feature_set": feature_set.name,
                "features": "+".join(feature_set.features),
                "n": len(frame),
                "n_dep": n_dep,
                "n_indep": int(len(frame) - n_dep),
                "n_used": int(used.sum()),
                "auroc": auc["point"],
                "auroc_ci_low": auc["ci_low"],
                "auroc_ci_high": auc["ci_high"],
                "auroc_se": auc["se"],
                "auprc": ap["point"],
                "auprc_ci_low": ap["ci_low"],
                "auprc_ci_high": ap["ci_high"],
                "auprc_se": ap["se"],
            }
        )

    deltas = []
    for augmented, baseline in DELTA_PAIRS:
        if augmented not in score_columns or baseline not in score_columns:
            continue
        used = np.isfinite(score_columns[augmented]) & np.isfinite(score_columns[baseline])
        if used.sum() == 0 or len(np.unique(y[used])) < 2:
            continue
        out = cluster_bootstrap(
            y[used],
            score_columns[augmented][used],
            groups[used],
            score_b=score_columns[baseline][used],
            metric="auroc",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        deltas.append(
            {
                "experiment": "corrected_main",
                "model": model,
                "slice": slice_name,
                "augmented_feature_set": augmented,
                "baseline_feature_set": baseline,
                "n_used": int(used.sum()),
                "delta_auroc": out["point"],
                "delta_ci_low": out["ci_low"],
                "delta_ci_high": out["ci_high"],
                "delta_se": out["se"],
            }
        )
    return summaries, deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--combined-predictions", default="outputs/noninternal_baselines/combined_baseline_predictions.csv")
    parser.add_argument("--control-predictions", default="outputs/main_task/control_state_probe_predictions.csv")
    parser.add_argument("--output-dir", default="outputs/incremental_hidden_signal")
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--slices", nargs="+", default=["all", "span_answer_only", "neutral_not_h1", "neutral_neither_h1_nor_h2"])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []
    deltas: list[dict] = []
    for model in args.models:
        for slice_name in args.slices:
            frame = score_frame(Path(args.combined_predictions), Path(args.control_predictions), model, slice_name)
            new_summaries, new_deltas = run_slice(frame, model, slice_name, args.n_bootstrap, args.seed)
            summaries.extend(new_summaries)
            deltas.extend(new_deltas)
            if new_summaries:
                print(f"{model} {slice_name}: n={new_summaries[0]['n']}", flush=True)

    summary = pd.DataFrame(summaries).sort_values(["experiment", "model", "slice", "feature_set"])
    delta = pd.DataFrame(deltas).sort_values(["experiment", "model", "slice", "augmented_feature_set"])
    summary.to_csv(output_dir / "incremental_hidden_signal_summary.csv", index=False)
    delta.to_csv(output_dir / "incremental_hidden_signal_deltas.csv", index=False)
    print(delta.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
