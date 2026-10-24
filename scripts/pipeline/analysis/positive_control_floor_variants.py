"""Floor variants of the positive-control sign task (app:corrected): in-sample fit, ungrouped folds, grouped folds
and a sign-shuffled control, with the estimator and layers of positive_control.py.
Input: re-extracted states (--data-dir). Output: --out-dir/positive_control_floor_variants.csv, released in
data/scores/controls/. CPU only.
"""

from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from scripts.pipeline.analysis.positive_control import (
    C_GRID,
    FIXED_LAYER,
    MODELS,
    N_BOOT,
    N_INNER,
    N_OUTER,
    SEED,
    cluster_bootstrap,
    fit_predict_proba,
    load_layer_states,
    splits,
)
from src.core.slice import build_main_task_records

warnings.filterwarnings("ignore")


def pick_c(x: np.ndarray, y: np.ndarray, groups: np.ndarray, tr: np.ndarray) -> float:
    best = (-math.inf, C_GRID[0])
    for c_value in C_GRID:
        scores = []
        for itr_rel, iva_rel in splits(x[tr], y[tr], groups[tr], N_INNER):
            itr, iva = tr[itr_rel], tr[iva_rel]
            if len(np.unique(y[itr])) < 2 or len(np.unique(y[iva])) < 2:
                continue
            p, _ = fit_predict_proba(x[itr], y[itr], x[iva], c_value)
            scores.append(float(roc_auc_score(y[iva], p[:, 1])))
        score = float(np.mean(scores)) if scores else -math.inf
        if score > best[0]:
            best = (score, c_value)
    return best[1]


def scored_row(
    model: str, task: str, feature: str, y: np.ndarray, p: np.ndarray, g: np.ndarray, n_q: int
) -> dict[str, Any]:
    auroc = float(roc_auc_score(y, p))
    acc = float(np.mean((p >= 0.5).astype(int) == y))

    def boot(idx: np.ndarray) -> float:
        if len(np.unique(y[idx])) < 2:
            return float("nan")
        return float(roc_auc_score(y[idx], p[idx]))

    ci = cluster_bootstrap(g, boot, N_BOOT, SEED)
    return {
        "model": model,
        "task": task,
        "feature": feature,
        "layer": FIXED_LAYER[model],
        "n_rows": int(len(y)),
        "n_questions": n_q,
        "auroc": auroc,
        "auroc_ci_low": ci["ci_low"],
        "auroc_ci_high": ci["ci_high"],
        "acc": acc,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for model in args.models:
        layer = FIXED_LAYER[model]
        recs = build_main_task_records(model, data_dir)
        w = load_layer_states(recs, "w", data_dir, layer)
        g_s = load_layer_states(recs, "g", data_dir, layer)
        c_s = load_layer_states(recs, "c", data_dir, layer)
        n_q = len(recs)
        base_groups = np.array([r["group"] for r in recs], dtype=str)
        print(f"[{model}] N={n_q} layer={layer}", flush=True)

        for feat, mat in [("W-G", w - g_s), ("W-C", w - c_s)]:
            x = np.concatenate([mat, -mat], axis=0)
            y = np.array([1] * n_q + [0] * n_q, dtype=int)
            groups = np.concatenate([base_groups, base_groups])

            # 1) in-sample fit
            p_in, _ = fit_predict_proba(x, y, x, 1.0)
            rows.append(scored_row(model, "floor_fit_insample", feat, y, p_in[:, 1], groups, n_q))
            print(f"  insample {feat}: {rows[-1]['auroc']:.4f} acc={rows[-1]['acc']:.4f}", flush=True)

            # 2) random (ungrouped) 5-fold
            probs = np.zeros(len(y))
            used = np.zeros(len(y), dtype=bool)
            for tr, te in KFold(n_splits=N_OUTER, shuffle=True, random_state=0).split(x):
                p, _ = fit_predict_proba(x[tr], y[tr], x[te], 1.0)
                probs[te] = p[:, 1]
                used[te] = True
            rows.append(
                scored_row(model, "floor_random_split", feat, y[used], probs[used], groups[used], n_q)
            )
            print(f"  randsplit {feat}: {rows[-1]['auroc']:.4f} acc={rows[-1]['acc']:.4f}", flush=True)

            # 3) grouped 5-fold (cross-question), with nested C selection
            probs = np.zeros(len(y))
            used = np.zeros(len(y), dtype=bool)
            for tr, te in splits(x, y, groups, N_OUTER):
                c_value = pick_c(x, y, groups, tr)
                p, _ = fit_predict_proba(x[tr], y[tr], x[te], c_value)
                probs[te] = p[:, 1]
                used[te] = True
            rows.append(
                scored_row(model, "floor_grouped", feat, y[used], probs[used], groups[used], n_q)
            )
            print(f"  grouped {feat}: {rows[-1]['auroc']:.4f}", flush=True)

            # 4) shuffled-sign control: permute which member of each pair is "1"
            rng = np.random.default_rng(0)
            flip = rng.integers(0, 2, size=n_q).astype(bool)
            x_sh = x.copy()
            x_sh[: n_q][flip] = -mat[flip]
            x_sh[n_q:][flip] = mat[flip]
            probs = np.zeros(len(y))
            used = np.zeros(len(y), dtype=bool)
            for tr, te in splits(x_sh, y, groups, N_OUTER):
                p, _ = fit_predict_proba(x_sh[tr], y[tr], x_sh[te], 1.0)
                probs[te] = p[:, 1]
                used[te] = True
            rows.append(
                scored_row(
                    model, "floor_grouped_signshuffled", feat, y[used], probs[used], groups[used], n_q
                )
            )
            print(f"  signshuffled {feat}: {rows[-1]['auroc']:.4f}", flush=True)

        # ---- W-C specific contrast: h1 vs the CORRECT letter -------------- #
        # W names h1, C names the gold answer. The ordered relation between h1
        # and the gold letter is the contrast W-C actually carries, so the
        # h1-vs-h2 relation is the wrong target for W-C and is reported only
        # for symmetry in the main script.
        mc = [
            i
            for i, r in enumerate(recs)
            if r["dataset"] in {"mmlu_pro", "gpqa"}
            and len(str(r["h1"])) == 1
            and len(str(r["correct_answer"] or "")) == 1
            and str(r["h1"]) != str(r["correct_answer"])
        ]
        if len(mc) >= 40:
            x_mc = {"W": w[mc], "G": g_s[mc], "C": c_s[mc], "W-G": (w - g_s)[mc], "W-C": (w - c_s)[mc]}
            y_mc = np.array(
                [int(str(recs[i]["h1"]) > str(recs[i]["correct_answer"])) for i in mc], dtype=int
            )
            grp_mc = np.array([recs[i]["group"] for i in mc], dtype=str)
            for feat, mat in x_mc.items():
                probs = np.zeros(len(y_mc))
                used = np.zeros(len(y_mc), dtype=bool)
                for tr, te in splits(mat, y_mc, grp_mc, N_OUTER):
                    if len(np.unique(y_mc[tr])) < 2 or len(np.unique(y_mc[te])) < 2:
                        continue
                    c_value = pick_c(mat, y_mc, grp_mc, tr)
                    p, _ = fit_predict_proba(mat[tr], y_mc[tr], mat[te], c_value)
                    probs[te] = p[:, 1]
                    used[te] = True
                rows.append(
                    scored_row(
                        model,
                        "mc_h1_after_correct_alphabetically",
                        feat,
                        y_mc[used],
                        probs[used],
                        grp_mc[used],
                        int(used.sum()),
                    )
                )
                print(f"  h1_vs_correct {feat}: {rows[-1]['auroc']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(
        out_dir / f"positive_control_floor_variants{args.suffix}.csv", index=False
    )
    print("done")


if __name__ == "__main__":
    main()
