"""Nested layer search for the paired-difference and residual probes (tab:diff-layer-search): logistic regression,
outer GroupKFold(5), inner GroupKFold(3) over layer, C and ridge alpha, cluster bootstrap B=2000, seed 0. CPU only.
Input: re-extracted prompt-end states (--data-dir with traces/ and features/). Output (--out-dir), released in
data/scores/controls/: difference_probe_layer_search_summary.csv as difference_probe_summary.csv,
difference_probe_oof_predictions.csv as difference_probe_oof.csv, difference_probe_layer_curves.csv. The release
joins a run with the default feature sets and a second run with --feature-sets G_minus_C (commands not recorded).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scripts.pipeline import main_task as RRA  # noqa: E402

MODELS = ["mistral", "gemma", "llama"]
C_GRID = [0.01, 0.1, 1.0, 10.0]
RESIDUAL_ALPHAS = [1.0, 10.0, 100.0]
N_OUTER = 5
N_INNER = 3
N_BOOTSTRAP = 2000
BOOT_SEED = 0

# Strict-rule fixed layers used by the fixed-layer paired/residual rows (tab:paired-diff).
FIXED_LAYER = {"mistral": 11, "gemma": 13, "llama": 13}


# --------------------------------------------------------------------------
# verbatim copies of main_task's fit / residualize helpers
# (kept local so joblib workers import only this module; checked against RRA at start)
# --------------------------------------------------------------------------
def fit_dense_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, c_value: float
) -> np.ndarray:
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
    kernel = control_train_s @ control_train_s.T
    kernel_reg = kernel.copy()
    kernel_reg.flat[:: kernel_reg.shape[0] + 1] += alpha
    dual = np.linalg.solve(kernel_reg, target_train_s)
    train_pred = kernel @ dual
    test_pred = (control_test_s @ control_train_s.T) @ dual
    return target_train_s - train_pred, target_test_s - test_pred


def _check_helpers_match_main_task() -> None:
    rng = np.random.default_rng(0)
    xt, xv = rng.normal(size=(40, 25)), rng.normal(size=(12, 25))
    ct, cv = rng.normal(size=(40, 25)), rng.normal(size=(12, 25))
    yt = (rng.random(40) > 0.5).astype(int)
    assert np.allclose(
        fit_dense_predict(xt, yt, xv, 1.0), RRA.fit_dense_predict(xt, yt, xv, 1.0)
    ), "local fit_dense_predict diverged from main_task"
    a = residualize_train_test(xt, ct, xv, cv, 10.0)
    b = RRA.residualize_train_test(xt, ct, xv, cv, 10.0)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]), (
        "local residualize_train_test diverged from main_task"
    )


# --------------------------------------------------------------------------
# feature assembly
# --------------------------------------------------------------------------
def load_sources(
    records: list[dict[str, Any]], data_dir: Path, sources: tuple[str, ...]
) -> tuple[list[int], dict[str, dict[int, np.ndarray]], list[int]]:
    """Load the named trace sources and align them on a common kept-row set."""
    per_source: dict[str, tuple[list[int], dict[int, np.ndarray], list[int]]] = {}
    for source in sources:
        per_source[source] = RRA.load_state_matrix(records, source, data_dir)

    layers = sorted(
        set.intersection(*[set(v[0]) for v in per_source.values()])
    )
    kept_sets = [set(v[2]) for v in per_source.values()]
    common = sorted(set.intersection(*kept_sets))

    aligned: dict[str, dict[int, np.ndarray]] = {}
    for source, (_, by_layer, kept) in per_source.items():
        pos = {idx: p for p, idx in enumerate(kept)}
        take = np.array([pos[idx] for idx in common], dtype=int)
        aligned[source] = {layer: by_layer[layer][take] for layer in layers}
    return layers, aligned, common


# --------------------------------------------------------------------------
# per-(fold, layer) workers
# --------------------------------------------------------------------------
def _inner_splits(y: np.ndarray, groups: np.ndarray, train_idx: np.ndarray):
    train_groups = np.unique(groups[train_idx])
    n_splits = min(N_INNER, len(train_groups))
    if n_splits < 2:
        return []
    splitter = GroupKFold(n_splits=n_splits)
    out = []
    for itr, iva in splitter.split(train_idx, y[train_idx], groups[train_idx]):
        inner_train = train_idx[itr]
        inner_val = train_idx[iva]
        if len(np.unique(y[inner_train])) < 2 or len(np.unique(y[inner_val])) < 2:
            continue
        out.append((inner_train, inner_val))
    return out


def direct_cell(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """Inner scores plus held-out test scores for every C at one layer/fold."""
    inner = _inner_splits(y, groups, train_idx)
    inner_scores: dict[float, float] = {}
    for c_value in C_GRID:
        vals = []
        for inner_train, inner_val in inner:
            probs = fit_dense_predict(
                x[inner_train], y[inner_train], x[inner_val], c_value
            )
            vals.append(float(roc_auc_score(y[inner_val], probs)))
        inner_scores[c_value] = float(np.mean(vals)) if vals else 0.5
    test_scores = {
        c_value: fit_dense_predict(x[train_idx], y[train_idx], x[test_idx], c_value)
        for c_value in C_GRID
    }
    return {"inner": inner_scores, "test": test_scores}


def residual_cell(
    target: np.ndarray,
    control: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    """Same as direct_cell but over (alpha, C) with fold-local residualization."""
    inner = _inner_splits(y, groups, train_idx)
    inner_scores: dict[tuple[float, float], float] = {}
    test_scores: dict[tuple[float, float], np.ndarray] = {}
    for alpha in RESIDUAL_ALPHAS:
        per_c: dict[float, list[float]] = {c: [] for c in C_GRID}
        for inner_train, inner_val in inner:
            r_tr, r_va = residualize_train_test(
                target[inner_train],
                control[inner_train],
                target[inner_val],
                control[inner_val],
                alpha=alpha,
            )
            for c_value in C_GRID:
                probs = fit_dense_predict(r_tr, y[inner_train], r_va, c_value)
                per_c[c_value].append(float(roc_auc_score(y[inner_val], probs)))
        for c_value in C_GRID:
            vals = per_c[c_value]
            inner_scores[(alpha, c_value)] = float(np.mean(vals)) if vals else 0.5
        r_tr, r_te = residualize_train_test(
            target[train_idx],
            control[train_idx],
            target[test_idx],
            control[test_idx],
            alpha=alpha,
        )
        for c_value in C_GRID:
            test_scores[(alpha, c_value)] = fit_dense_predict(
                r_tr, y[train_idx], r_te, c_value
            )
    return {"inner": inner_scores, "test": test_scores}


# --------------------------------------------------------------------------
# driver for one feature set
# --------------------------------------------------------------------------
def run_feature_set(
    name: str,
    model: str,
    layers: list[int],
    y: np.ndarray,
    groups: np.ndarray,
    target_by_layer: dict[int, np.ndarray],
    control_by_layer: dict[int, np.ndarray] | None,
    n_jobs: int,
) -> dict[str, Any]:
    """Nested layer(+C[+alpha]) search and the full per-layer AUROC curve."""
    outer = GroupKFold(n_splits=min(N_OUTER, len(np.unique(groups))))
    folds = []
    for train_idx, test_idx in outer.split(target_by_layer[layers[0]], y, groups):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            continue
        folds.append((train_idx, test_idx))

    tasks = [(f, layer) for f in range(len(folds)) for layer in layers]
    if control_by_layer is None:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(direct_cell)(
                target_by_layer[layer], y, groups, folds[f][0], folds[f][1]
            )
            for f, layer in tasks
        )
        keys = [(np.nan, c) for c in C_GRID]
        cell_key = lambda k: k[1]  # noqa: E731
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
            delayed(residual_cell)(
                target_by_layer[layer],
                control_by_layer[layer],
                y,
                groups,
                folds[f][0],
                folds[f][1],
            )
            for f, layer in tasks
        )
        keys = [(a, c) for a in RESIDUAL_ALPHAS for c in C_GRID]
        cell_key = lambda k: k  # noqa: E731

    cells = {t: r for t, r in zip(tasks, results)}

    # ---- nested: pick (layer, alpha, C) per outer fold by inner AUROC ----
    nested_scores = np.full(len(y), np.nan)
    fold_rows = []
    for f, (train_idx, test_idx) in enumerate(folds):
        best = (-math.inf, layers[0], keys[0])
        for layer in layers:
            inner = cells[(f, layer)]["inner"]
            for key in keys:
                score = inner[cell_key(key)]
                if score > best[0]:
                    best = (score, layer, key)
        inner_score, layer, key = best
        nested_scores[test_idx] = cells[(f, layer)]["test"][cell_key(key)]
        fold_rows.append(
            {
                "feature_set": name,
                "model": model,
                "fold": f,
                "selected_layer": layer,
                "residual_alpha": key[0],
                "C": key[1],
                "inner_auroc": inner_score,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
            }
        )

    mask = ~np.isnan(nested_scores)
    nested_ci = RRA.cluster_bootstrap_metric(
        y[mask], nested_scores[mask], groups[mask], "auroc", N_BOOTSTRAP, BOOT_SEED
    )

    # ---- per-layer curve: honest OOF AUROC with C(+alpha) chosen inside folds ----
    curve_rows = []
    for layer in layers:
        scores = np.full(len(y), np.nan)
        picks = []
        for f, (train_idx, test_idx) in enumerate(folds):
            inner = cells[(f, layer)]["inner"]
            best_key = max(keys, key=lambda k: inner[cell_key(k)])
            scores[test_idx] = cells[(f, layer)]["test"][cell_key(best_key)]
            picks.append(best_key)
        m = ~np.isnan(scores)
        curve_rows.append(
            {
                "feature_set": name,
                "model": model,
                "layer": layer,
                "auroc": float(roc_auc_score(y[m], scores[m])),
                "selected_alphas": ",".join(
                    "" if np.isnan(p[0]) else str(p[0]) for p in picks
                ),
                "selected_Cs": ",".join(str(p[1]) for p in picks),
            }
        )
    curve = pd.DataFrame(curve_rows)
    best_row = curve.loc[curve["auroc"].idxmax()]

    # cluster-bootstrap CI for the best single layer (post-hoc max; optimistic)
    best_layer = int(best_row["layer"])
    best_scores = np.full(len(y), np.nan)
    for f, (train_idx, test_idx) in enumerate(folds):
        inner = cells[(f, best_layer)]["inner"]
        bk = max(keys, key=lambda k: inner[cell_key(k)])
        best_scores[test_idx] = cells[(f, best_layer)]["test"][cell_key(bk)]
    bm = ~np.isnan(best_scores)
    best_ci = RRA.cluster_bootstrap_metric(
        y[bm], best_scores[bm], groups[bm], "auroc", N_BOOTSTRAP, BOOT_SEED
    )

    # value at the strict-rule fixed layer, for comparison with tab:paired-diff
    fixed_layer = FIXED_LAYER[model]
    fixed_auroc = float("nan")
    if fixed_layer in set(curve["layer"]):
        fixed_auroc = float(
            curve.loc[curve["layer"] == fixed_layer, "auroc"].iloc[0]
        )

    summary = {
        "feature_set": name,
        "model": model,
        "n": int(mask.sum()),
        "n_dep": int(y[mask].sum()),
        "n_layers": len(layers),
        "n_folds": len(folds),
        "nested_auroc": nested_ci["point"],
        "nested_ci_low": nested_ci["ci_low"],
        "nested_ci_high": nested_ci["ci_high"],
        "selected_layers": ",".join(str(r["selected_layer"]) for r in fold_rows),
        "selected_Cs": ",".join(str(r["C"]) for r in fold_rows),
        "selected_alphas": ",".join(
            "" if np.isnan(r["residual_alpha"]) else str(r["residual_alpha"])
            for r in fold_rows
        ),
        "best_layer": best_layer,
        "best_layer_auroc": float(best_row["auroc"]),
        "best_layer_ci_low": best_ci["ci_low"],
        "best_layer_ci_high": best_ci["ci_high"],
        "fixed_layer": fixed_layer,
        "fixed_layer_auroc": fixed_auroc,
    }
    oof = pd.DataFrame(
        {
            "feature_set": name,
            "model": model,
            "group": groups[mask],
            "label": y[mask],
            "nested_score": nested_scores[mask],
        }
    )
    return {
        "summary": summary,
        "folds": pd.DataFrame(fold_rows),
        "curve": curve,
        "oof": oof,
    }


# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["raw_W", "W_minus_G", "W_minus_C", "W_resid_G", "W_resid_C"],
    )
    args = parser.parse_args()

    _check_helpers_match_main_task()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, fold_frames, curve_frames, oof_frames = [], [], [], []

    for model in args.models:
        records = RRA.build_main_task_records(model, data_dir)
        layers, aligned, common = load_sources(records, data_dir, ("w", "g", "c"))
        y = np.array([records[i]["label"] for i in common], dtype=int)
        groups = np.array([records[i]["group"] for i in common], dtype=str)
        print(
            f"[{model}] N={len(common)} n_dep={int(y.sum())} "
            f"layers={len(layers)} (full records={len(records)})",
            flush=True,
        )

        specs: list[tuple[str, dict[int, np.ndarray], dict[int, np.ndarray] | None]] = []
        for name in args.feature_sets:
            if name == "raw_W":
                specs.append((name, aligned["w"], None))
            elif name == "raw_G":
                specs.append((name, aligned["g"], None))
            elif name == "raw_C":
                specs.append((name, aligned["c"], None))
            elif name == "W_minus_G":
                specs.append(
                    (name, {l: aligned["w"][l] - aligned["g"][l] for l in layers}, None)
                )
            elif name == "W_minus_C":
                specs.append(
                    (name, {l: aligned["w"][l] - aligned["c"][l] for l in layers}, None)
                )
            elif name == "G_minus_C":
                # treatment-free analogue of W_minus_C: neither state is the
                # hinted-with-h1 treatment, so any signal here cannot be a
                # record of h1 having influenced the answer
                specs.append(
                    (name, {l: aligned["g"][l] - aligned["c"][l] for l in layers}, None)
                )
            elif name == "W_resid_G":
                specs.append((name, aligned["w"], aligned["g"]))
            elif name == "W_resid_C":
                specs.append((name, aligned["w"], aligned["c"]))
            else:
                raise ValueError(f"unknown feature set {name}")

        for name, target, control in specs:
            res = run_feature_set(
                name, model, layers, y, groups, target, control, args.n_jobs
            )
            summaries.append(res["summary"])
            fold_frames.append(res["folds"])
            curve_frames.append(res["curve"])
            oof_frames.append(res["oof"])
            s = res["summary"]
            print(
                f"  {name:11s} nested {s['nested_auroc']:.3f} "
                f"[{s['nested_ci_low']:.3f},{s['nested_ci_high']:.3f}] "
                f"layers={s['selected_layers']} | best L{s['best_layer']} "
                f"{s['best_layer_auroc']:.3f} | fixed L{s['fixed_layer']} {s['fixed_layer_auroc']:.3f}",
                flush=True,
            )
            pd.DataFrame(summaries).to_csv(
                out_dir / "difference_probe_layer_search_summary.csv", index=False
            )
            pd.concat(fold_frames).to_csv(
                out_dir / "difference_probe_layer_search_folds.csv", index=False
            )
            pd.concat(curve_frames).to_csv(
                out_dir / "difference_probe_layer_curves.csv", index=False
            )
            pd.concat(oof_frames).to_csv(
                out_dir / "difference_probe_oof_predictions.csv", index=False
            )

    meta = {
        "data_dir": str(data_dir),
        "models": args.models,
        "feature_sets": args.feature_sets,
        "c_grid": C_GRID,
        "residual_alphas": RESIDUAL_ALPHAS,
        "n_outer": N_OUTER,
        "n_inner": N_INNER,
        "n_bootstrap": N_BOOTSTRAP,
        "bootstrap_seed": BOOT_SEED,
        "fixed_layers": FIXED_LAYER,
    }
    (out_dir / "difference_probe_layer_search_meta.json").write_text(
        json.dumps(meta, indent=2)
    )
    print(f"\nwrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
