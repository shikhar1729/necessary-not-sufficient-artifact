"""Do W-state and G-state probes read the same information? (app:corrected) Cross-condition transfer (a probe fit on
S scored on T), within-class Spearman of W and G probe scores, coefficient cosines; protocol of positive_control.py,
cluster bootstrap B=2000, seed 0. Needs scikit-learn < 1.9 (fold ties). Input: re-extracted states (--data-dir).
Output (--out-dir): wg_transfer<suffix>.csv and wg_transfer_oof<suffix>.csv.gz; the three OOF files are
concatenated as data/scores/controls/wg_transfer_oof.csv.gz. Recorded run, once per model:

    for M in mistral gemma llama; do
      python -m scripts.pipeline.analysis.wg_transfer --data-dir <states> --out-dir <out> --models $M --suffix _$M
    done
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import sklearn
import warnings

if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 9):
    warnings.warn("scikit-learn >= 1.9 changes GroupKFold folds; use < 1.9 to reproduce")

from scripts.pipeline.main_task import (
    build_main_task_records,
    resolve_feature_path,
)

MODELS = ["mistral", "gemma", "llama"]
# Strict-rule fixed probe layers (mode of the strict nested run's selected layers).
FIXED_LAYER = {"mistral": 11, "gemma": 13, "llama": 13}
SOURCES = ["W", "G", "C"]
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_OUTER = 5
N_INNER = 3
N_BOOT = 2000
SEED = 0


def load_layer_states(
    records: list[dict[str, Any]], source: str, data_dir: Path, layer: int
) -> np.ndarray:
    key = f"pos_-1_layer_{layer}"
    rows = []
    for rec in records:
        path = resolve_feature_path(rec[source], data_dir)
        if path is None:
            raise RuntimeError(f"no feature file for {source} {rec['group']}")
        with np.load(path) as z:
            rows.append(z[key].astype(np.float32))
    return np.stack(rows, axis=0)


def make_clf(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=1000, random_state=0, solver="liblinear"
    )


def select_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    best_c, best_auc = C_GRID[0], -np.inf
    inner = list(GroupKFold(n_splits=N_INNER).split(x, y, groups))
    for c_value in C_GRID:
        scores = np.zeros(len(y))
        for tr, te in inner:
            scaler = StandardScaler().fit(x[tr])
            clf = make_clf(c_value).fit(scaler.transform(x[tr]), y[tr])
            scores[te] = clf.decision_function(scaler.transform(x[te]))
        auc = roc_auc_score(y, scores)
        if auc > best_auc:
            best_c, best_auc = c_value, auc
    return best_c


def within_class_spearman(a: np.ndarray, b: np.ndarray, y: np.ndarray, cls: int) -> float:
    mask = y == cls
    return float(spearmanr(a[mask], b[mask]).statistic)


def run_model(model: str, data_dir: Path) -> tuple[list[dict], list[dict]]:
    layer = FIXED_LAYER[model]
    recs = build_main_task_records(model, data_dir)
    y = np.array([r["label"] for r in recs])
    groups = np.array([r["group"] for r in recs])
    print(f"[{model}] N={len(y)} n_dep={int(y.sum())} layer={layer}", flush=True)
    states = {s: load_layer_states(recs, s.lower(), data_dir, layer) for s in SOURCES}

    oof = {(s, t): np.zeros(len(y)) for s in SOURCES for t in SOURCES}
    coefs: dict[str, list[np.ndarray]] = {s: [] for s in SOURCES}
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=N_OUTER).split(y, y, groups)):
        for s in SOURCES:
            c_value = select_c(states[s][tr], y[tr], groups[tr])
            scaler = StandardScaler().fit(states[s][tr])
            clf = make_clf(c_value).fit(scaler.transform(states[s][tr]), y[tr])
            coef = clf.coef_.ravel()
            coefs[s].append(coef / np.linalg.norm(coef))
            for t in SOURCES:
                oof[(s, t)][te] = clf.decision_function(scaler.transform(states[t][te]))
        print(f"  fold {fold} done", flush=True)

    # point estimates
    def stats(idx: np.ndarray) -> dict[str, float]:
        yy = y[idx]
        out: dict[str, float] = {}
        for s in SOURCES:
            for t in SOURCES:
                out[f"auc_{s}to{t}"] = roc_auc_score(yy, oof[(s, t)][idx])
        for s in SOURCES:
            for t in SOURCES:
                if s != t:
                    out[f"transfer_gap_{s}to{s}_minus_{s}to{t}"] = (
                        out[f"auc_{s}to{s}"] - out[f"auc_{s}to{t}"]
                    )
        for a, b in [("W", "G"), ("W", "C"), ("G", "C")]:
            sa, sb = oof[(a, a)][idx], oof[(b, b)][idx]
            out[f"spearman_{a}{a}_{b}{b}_all"] = float(spearmanr(sa, sb).statistic)
            out[f"spearman_{a}{a}_{b}{b}_dep"] = within_class_spearman(sa, sb, yy, 1)
            out[f"spearman_{a}{a}_{b}{b}_indep"] = within_class_spearman(sa, sb, yy, 0)
        return out

    point = stats(np.arange(len(y)))

    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}
    boot: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        for k, v in stats(idx).items():
            boot[k].append(v)

    rows = []
    for k, v in point.items():
        lo, hi = np.percentile(boot[k], [2.5, 97.5])
        rows.append({"model": model, "layer": layer, "metric": k, "estimate": v,
                     "ci_low": lo, "ci_high": hi, "n": len(y), "n_dep": int(y.sum())})

    mean_dir = {s: np.mean(coefs[s], axis=0) for s in SOURCES}
    for a, b in [("W", "G"), ("W", "C"), ("G", "C")]:
        cos = float(mean_dir[a] @ mean_dir[b] / (np.linalg.norm(mean_dir[a]) * np.linalg.norm(mean_dir[b])))
        rows.append({"model": model, "layer": layer, "metric": f"cosine_coef_{a}_{b}",
                     "estimate": cos, "ci_low": np.nan, "ci_high": np.nan,
                     "n": len(y), "n_dep": int(y.sum())})
    # fold-to-fold stability of each probe's own direction, as the reference scale
    for s in SOURCES:
        cs = [float(coefs[s][i] @ coefs[s][j]) for i in range(N_OUTER) for j in range(i + 1, N_OUTER)]
        rows.append({"model": model, "layer": layer, "metric": f"cosine_coef_{s}_across_folds_mean",
                     "estimate": float(np.mean(cs)), "ci_low": np.nan, "ci_high": np.nan,
                     "n": len(y), "n_dep": int(y.sum())})

    oof_rows = [
        {"model": model, "group": groups[i], "label": int(y[i]),
         **{f"score_{s}to{t}": float(oof[(s, t)][i]) for s in SOURCES for t in SOURCES}}
        for i in range(len(y))
    ]
    return rows, oof_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, oof_rows = [], []
    for model in args.models:
        r, o = run_model(model, data_dir)
        rows += r
        oof_rows += o
    pd.DataFrame(rows).to_csv(out_dir / f"wg_transfer{args.suffix}.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(
        out_dir / f"wg_transfer_oof{args.suffix}.csv.gz", index=False
    )


if __name__ == "__main__":
    main()
