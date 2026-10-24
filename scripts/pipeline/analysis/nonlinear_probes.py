"""Nonlinear probes (MLP, RBF-SVC, RFM kernel) against the linear probe on W, G, C, W-G and W-C prompt-end states
(tab:nonlinear-probes): same nested GroupKFold protocol and fixed layers 11/13/13 as the linear probes. CPU only.
Input: re-extracted states (--data-dir). Output (--out-dir): summary, paired deltas, folds and per-row OOF scores
(released as data/scores/controls/nonlinear_probe_oof.csv.gz). Recorded runs: the defaults, then three nulls
(concatenated as data/scores/controls/nonlinear_probe_permutation_null.csv):

    python -m scripts.pipeline.analysis.nonlinear_probes --data-dir <states> --permute 40 --models mistral \
        --featuresets W_minus_G --families rfm --tag perm_mistral_WmG_rfm
    (likewise gemma/W_minus_G/rfm --tag perm_gemma_WmG_rfm, mistral/W_minus_C/rbf_svc --tag perm_mistral_WmC_svc)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scripts.pipeline import main_task as RV

STRICT_LAYERS = {"mistral": 11, "gemma": 13, "llama": 13}
C_GRID = [0.01, 0.1, 1.0, 10.0]  # paper's linear grid
MLP_GRID = [
    {"hidden_layer_sizes": h, "alpha": a}
    for h in [(32,), (128,)]
    for a in [1e-3, 1e-1, 1e1, 1e3]
]
SVC_C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
SVC_GAMMA_SCALES = [0.01, 0.1, 1.0, 10.0]  # multiples of 1/d after standardization
RFM_REG_GRID = [1e-3, 1e-2, 1e-1, 1.0, 1e1, 1e2]
RFM_POWERS = [1.0, 0.5]
RFM_ITERS = 5
RFM_MAX_PCA = 256

# Apple Accelerate BLAS raises spurious divide/overflow FP flags inside matmul
# on finite inputs; results are verified finite downstream.
np.seterr(divide="ignore", over="ignore", invalid="ignore")


# ---------------------------------------------------------------- feature sets


def feature_sets(model: str, data_dir: Path) -> dict[str, dict[str, Any]]:
    """Return {name: {X, y, groups}} at the strict fixed layer for one model."""
    layer = STRICT_LAYERS[model]
    records = RV.build_main_task_records(model, data_dir)
    y_all = np.array([r["label"] for r in records], dtype=int)
    groups_all = np.array([r["group"] for r in records], dtype=str)

    out: dict[str, dict[str, Any]] = {}
    for name, source in [("W", "w"), ("G", "g"), ("C", "c")]:
        _, by_layer, kept = RV.load_state_matrix(records, source, data_dir, only_layers=[layer])
        out[name] = {
            "X": by_layer[layer].astype(np.float64),
            "y": y_all[kept],
            "groups": groups_all[kept],
        }
    for name, left, right in [("W_minus_G", "w", "g"), ("W_minus_C", "w", "c")]:
        _, by_layer, kept = RV.make_paired_layer_features(
            records, data_dir, left, right, only_layers=[layer]
        )
        out[name] = {
            "X": by_layer[layer].astype(np.float64),
            "y": y_all[kept],
            "groups": groups_all[kept],
        }
    return out


# ------------------------------------------------------------------ estimators


def balanced_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weight = {c: len(y) / (len(classes) * n) for c, n in zip(classes, counts)}
    return np.array([weight[v] for v in y], dtype=float)


def fit_linear(x_tr, y_tr, x_te, params):
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    clf = LogisticRegression(
        C=params["C"],
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    clf.fit(x_tr_s, y_tr)
    return clf.predict_proba(x_te_s)[:, 1]


def fit_mlp(x_tr, y_tr, x_te, params):
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    # MLPClassifier has no class_weight / sample_weight: balance by seeded
    # oversampling of the minority class inside the training fold.
    rng = np.random.default_rng(0)
    classes, counts = np.unique(y_tr, return_counts=True)
    target = counts.max()
    idx_parts = []
    for cls, cnt in zip(classes, counts):
        idx = np.flatnonzero(y_tr == cls)
        idx_parts.append(idx)
        if cnt < target:
            idx_parts.append(rng.choice(idx, size=target - cnt, replace=True))
    idx = np.concatenate(idx_parts)
    rng.shuffle(idx)
    clf = MLPClassifier(
        hidden_layer_sizes=params["hidden_layer_sizes"],
        alpha=params["alpha"],
        activation="relu",
        solver="adam",
        learning_rate_init=1e-3,
        batch_size=32,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=0,
    )
    clf.fit(x_tr_s[idx], y_tr[idx])
    return clf.predict_proba(x_te_s)[:, 1]


def _sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def fit_rbf_svc(x_tr, y_tr, x_te, params, cache: dict | None = None):
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    key = "d2"
    if cache is not None and key in cache:
        d2_tr, d2_te = cache[key]
    else:
        d2_tr = _sq_dists(x_tr_s, x_tr_s)
        d2_te = _sq_dists(x_te_s, x_tr_s)
        if cache is not None:
            cache[key] = (d2_tr, d2_te)
    gamma = params["gamma_scale"] / x_tr_s.shape[1]
    k_tr = np.exp(-gamma * d2_tr)
    k_te = np.exp(-gamma * d2_te)
    clf = SVC(C=params["C"], kernel="precomputed", class_weight="balanced")
    clf.fit(k_tr, y_tr)
    return clf.decision_function(k_te)


def fit_rfm(x_tr, y_tr, x_te, params, cache: dict | None = None):
    """RFM-style Mahalanobis-Laplace kernel ridge classifier.

    K_M(x, z) = exp(-sqrt((x-z)^T M (x-z)) / L), M initialised to I/k and
    updated to (a power of) the average gradient outer product (AGOP) of the
    current predictor. Bandwidth L is the median pairwise Mahalanobis distance,
    recomputed each iteration. Features are standardized then projected onto
    the training PCA basis (<= min(n_train - 1, 256) components) so that the
    k x k AGOP stays tractable; the training data span is retained.
    """
    scaler = StandardScaler()
    x_tr_s = scaler.fit_transform(x_tr)
    x_te_s = scaler.transform(x_te)
    n_comp = int(min(x_tr_s.shape[0] - 1, x_tr_s.shape[1], RFM_MAX_PCA))
    if cache is not None and "pca" in cache:
        z_tr, z_te = cache["pca"]
    else:
        pca = PCA(n_components=n_comp, random_state=0)
        z_tr = pca.fit_transform(x_tr_s)
        z_te = pca.transform(x_te_s)
        if cache is not None:
            cache["pca"] = (z_tr, z_te)
    k = z_tr.shape[1]
    y_pm = np.where(y_tr == 1, 1.0, -1.0)
    w = balanced_weights(y_tr)
    m = np.eye(k) / k
    reg = params["reg"]
    power = params["power"]
    scores = np.zeros(z_te.shape[0])
    for it in range(RFM_ITERS + 1):
        zm = z_tr @ m
        sq = np.einsum("ij,ij->i", z_tr, zm)
        d2 = sq[:, None] + sq[None, :] - 2.0 * (z_tr @ zm.T)
        dist = np.sqrt(np.maximum(d2, 0.0))
        off = dist[~np.eye(len(dist), dtype=bool)]
        bandwidth = float(np.median(off)) if off.size else 1.0
        if not np.isfinite(bandwidth) or bandwidth <= 1e-12:
            bandwidth = 1.0
        k_tr = np.exp(-dist / bandwidth)
        a = np.linalg.solve(k_tr + reg * np.diag(1.0 / w), y_pm)

        sq_te = np.einsum("ij,ij->i", z_te, z_te @ m)
        d2_te = sq_te[:, None] + sq[None, :] - 2.0 * (z_te @ zm.T)
        k_te = np.exp(-np.sqrt(np.maximum(d2_te, 0.0)) / bandwidth)
        scores = k_te @ a

        if it == RFM_ITERS:
            break
        # AGOP of f(x) = sum_i a_i K_M(x, x_i):
        #   grad f(x) = -(1/L) sum_i a_i K(x,x_i) M (x - x_i) / dist(x,x_i)
        safe = np.where(dist > 1e-12, dist, np.inf)
        coef = (a[None, :] * k_tr) / safe
        np.fill_diagonal(coef, 0.0)
        grads = -(1.0 / bandwidth) * ((coef.sum(1)[:, None] * z_tr - coef @ z_tr) @ m)
        agop = (grads.T @ grads) / len(z_tr)
        agop = 0.5 * (agop + agop.T)
        if power != 1.0:
            vals, vecs = np.linalg.eigh(agop)
            vals = np.clip(vals, 0.0, None) ** power
            agop = (vecs * vals) @ vecs.T
        tr = float(np.trace(agop))
        if not np.isfinite(tr) or tr <= 1e-24:
            break
        m = agop * (k / tr)  # keep trace(M) = trace(I) = k
    if not np.isfinite(scores).all():
        scores = np.where(np.isfinite(scores), scores, np.nan)
    return scores


FAMILIES = {
    "linear": {
        "grid": [{"C": c} for c in C_GRID],
        "fit": fit_linear,
        "cacheable": False,
    },
    "mlp": {
        "grid": MLP_GRID,
        "fit": fit_mlp,
        "cacheable": False,
    },
    "rbf_svc": {
        "grid": [
            {"C": c, "gamma_scale": g} for g in SVC_GAMMA_SCALES for c in SVC_C_GRID
        ],
        "fit": fit_rbf_svc,
        "cacheable": True,
    },
    "rfm": {
        "grid": [{"reg": r, "power": p} for p in RFM_POWERS for r in RFM_REG_GRID],
        "fit": fit_rfm,
        "cacheable": True,
    },
}


# ------------------------------------------------------------------ nested CV


def nested_oof(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    family: str,
    verbose: bool = False,
) -> tuple[np.ndarray, pd.DataFrame]:
    spec = FAMILIES[family]
    fit = spec["fit"]
    grid = spec["grid"]
    cacheable = spec["cacheable"]
    scores = np.full(len(y), np.nan)
    fold_rows = []
    outer = GroupKFold(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups)):
        inner_groups = np.unique(groups[train_idx])
        n_inner = min(3, len(inner_groups))
        best = (-np.inf, grid[0])
        if len(grid) > 1 and n_inner >= 2:
            inner = GroupKFold(n_splits=n_inner)
            splits = list(inner.split(x[train_idx], y[train_idx], groups[train_idx]))
            caches = [{} if cacheable else None for _ in splits]
            for params in grid:
                vals = []
                for (tr_rel, va_rel), cache in zip(splits, caches):
                    tr = train_idx[tr_rel]
                    va = train_idx[va_rel]
                    if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                        continue
                    kwargs = {"cache": cache} if cacheable else {}
                    s = fit(x[tr], y[tr], x[va], params, **kwargs)
                    vals.append(float(roc_auc_score(y[va], s)))
                score = float(np.mean(vals)) if vals else 0.5
                if score > best[0]:
                    best = (score, params)
        inner_score, params = best
        kwargs = {"cache": {}} if cacheable else {}
        scores[test_idx] = fit(x[train_idx], y[train_idx], x[test_idx], params, **kwargs)
        fold_rows.append(
            {
                "fold": fold,
                "inner_auroc": inner_score if np.isfinite(inner_score) else np.nan,
                "params": json.dumps({k: str(v) for k, v in params.items()}),
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )
        if verbose:
            print(f"      fold {fold}: params={params} inner={inner_score:.3f}", flush=True)
    return scores, pd.DataFrame(fold_rows)


# ------------------------------------------------------------------ bootstrap


def bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    uniq = np.unique(groups)
    where = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        out.append(np.concatenate([where[g] for g in sampled]))
    return out


def auroc_ci(y, s, boots) -> dict[str, float]:
    point = float(roc_auc_score(y, s))
    vals = [
        float(roc_auc_score(y[idx], s[idx]))
        for idx in boots
        if len(np.unique(y[idx])) >= 2
    ]
    return {
        "auroc": point,
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
        "se": float(np.std(vals, ddof=1)),
    }


def delta_ci(y, s_nl, s_lin, boots) -> dict[str, float]:
    point = float(roc_auc_score(y, s_nl) - roc_auc_score(y, s_lin))
    vals = []
    for idx in boots:
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(
            float(roc_auc_score(y[idx], s_nl[idx]) - roc_auc_score(y[idx], s_lin[idx]))
        )
    vals = np.array(vals)
    return {
        "delta": point,
        "delta_ci_low": float(np.percentile(vals, 2.5)),
        "delta_ci_high": float(np.percentile(vals, 97.5)),
        "delta_se": float(np.std(vals, ddof=1)),
        "p_two_sided_sign": float(2 * min((vals <= 0).mean(), (vals >= 0).mean())),
    }


# ----------------------------------------------------- permutation null (opt.)


def permutation_null(args) -> None:
    """Label-permutation null for selected cells.

    Folds depend only on `groups`, so permuting labels leaves the CV geometry
    untouched; any systematic AUROC above 0.5 would indicate optimism in the
    nested protocol rather than signal.
    """
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in args.models:
        sets = feature_sets(model, data_dir)
        for fs_name in args.featuresets:
            fs = sets[fs_name]
            x, y, groups = fs["X"], fs["y"], fs["groups"]
            for family in args.families:
                rng = np.random.default_rng(args.seed)
                for rep in range(args.permute):
                    y_perm = y.copy()
                    rng.shuffle(y_perm)
                    scores, _ = nested_oof(x, y_perm, groups, family)
                    ok = np.isfinite(scores)
                    auroc = float(roc_auc_score(y_perm[ok], scores[ok]))
                    rows.append(
                        {
                            "model": model,
                            "featureset": fs_name,
                            "family": family,
                            "rep": rep,
                            "auroc": auroc,
                        }
                    )
                    print(f"  perm {model}/{fs_name}/{family} rep={rep} auroc={auroc:.3f}", flush=True)
    path = out_dir / f"nonlinear_probe_permutation_null_{args.tag}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    print("wrote", path)


# ----------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        required=True,
        help="re-extracted prompt-end states (traces/ and features/)",
    )
    ap.add_argument(
        "--out-dir",
        default="outputs/controls",
    )
    ap.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    ap.add_argument(
        "--featuresets",
        nargs="+",
        default=["W", "G", "C", "W_minus_G", "W_minus_C"],
    )
    ap.add_argument("--families", nargs="+", default=["linear", "mlp", "rbf_svc", "rfm"])
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="strict")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--permute",
        type=int,
        default=0,
        help="If > 0, run this many label-permutation replicates instead of the "
        "main analysis, writing a permutation-null AUROC distribution for the "
        "requested model/featureset/family cells.",
    )
    args = ap.parse_args()

    if args.permute:
        permutation_null(args)
        return

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    fold_frames: list[pd.DataFrame] = []
    score_rows: list[dict[str, Any]] = []

    for model in args.models:
        t0 = time.time()
        sets = feature_sets(model, data_dir)
        print(f"[{model}] layer={STRICT_LAYERS[model]} loaded in {time.time()-t0:.1f}s", flush=True)
        for fs_name in args.featuresets:
            fs = sets[fs_name]
            x, y, groups = fs["X"], fs["y"], fs["groups"]
            print(
                f"  [{model}/{fs_name}] n={len(y)} n_dep={int(y.sum())} d={x.shape[1]}",
                flush=True,
            )
            per_family_scores: dict[str, np.ndarray] = {}
            timings: dict[str, float] = {}
            for family in args.families:
                t1 = time.time()
                scores, folds = nested_oof(x, y, groups, family, verbose=args.verbose)
                per_family_scores[family] = scores
                timings[family] = time.time() - t1
                folds = folds.copy()
                folds.insert(0, "family", family)
                folds.insert(0, "featureset", fs_name)
                folds.insert(0, "model", model)
                fold_frames.append(folds)
                for g, lab, s in zip(groups, y, scores):
                    score_rows.append(
                        {
                            "model": model,
                            "featureset": fs_name,
                            "family": family,
                            "group": g,
                            "label": int(lab),
                            "score": float(s),
                        }
                    )

            # Restrict to rows scored by every family so AUROCs and paired
            # deltas share one set of cluster-bootstrap resamples.
            mask = np.ones(len(y), dtype=bool)
            for scores in per_family_scores.values():
                mask &= np.isfinite(scores)
            if not mask.all():
                print(f"    WARNING: {int((~mask).sum())} rows unscored, dropped", flush=True)
            y_m, groups_m = y[mask], groups[mask]
            boots = bootstrap_indices(groups_m, args.n_bootstrap, args.seed)
            for family, scores in per_family_scores.items():
                stats = auroc_ci(y_m, scores[mask], boots)
                summary_rows.append(
                    {
                        "model": model,
                        "featureset": fs_name,
                        "family": family,
                        "layer": STRICT_LAYERS[model],
                        "n": int(mask.sum()),
                        "n_dep": int(y_m.sum()),
                        "n_indep": int(mask.sum() - y_m.sum()),
                        **stats,
                        "seconds": round(timings[family], 1),
                    }
                )
                print(
                    f"    {family:8s} AUROC={stats['auroc']:.3f} "
                    f"[{stats['ci_low']:.3f},{stats['ci_high']:.3f}] "
                    f"({timings[family]:.0f}s)",
                    flush=True,
                )
            if "linear" in per_family_scores:
                lin = per_family_scores["linear"][mask]
                for family, scores in per_family_scores.items():
                    if family == "linear":
                        continue
                    d = delta_ci(y_m, scores[mask], lin, boots)
                    delta_rows.append(
                        {
                            "model": model,
                            "featureset": fs_name,
                            "family": family,
                            "n": int(mask.sum()),
                            **d,
                        }
                    )
                    print(
                        f"    delta {family:8s} {d['delta']:+.3f} "
                        f"[{d['delta_ci_low']:+.3f},{d['delta_ci_high']:+.3f}]",
                        flush=True,
                    )

    tag = args.tag
    pd.DataFrame(summary_rows).to_csv(out_dir / f"nonlinear_probes_{tag}.csv", index=False)
    pd.DataFrame(delta_rows).to_csv(out_dir / f"nonlinear_probe_deltas_{tag}.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(
        out_dir / f"nonlinear_probe_folds_{tag}.csv", index=False
    )
    pd.DataFrame(score_rows).to_csv(
        out_dir / f"nonlinear_probe_oof_scores_{tag}.csv.gz", index=False, compression="gzip"
    )
    print("wrote", out_dir / f"nonlinear_probes_{tag}.csv")


if __name__ == "__main__":
    main()
