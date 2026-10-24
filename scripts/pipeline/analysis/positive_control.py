"""Positive control for the paired W-G and W-C difference estimator (tab:positive-control): decode quantities
known to differ between the prompts (which candidate the note named, its letter, length, the sign task) with
the same estimator, layers 11/13/13, folds and preprocessing as the dependence probes. Input: re-extracted
states (--data-dir). Output (--out-dir): positive_control.csv (released as data/scores/controls/), slice sizes,
settings. Needs scikit-learn==1.7.2: it fits multiclass liblinear models, which scikit-learn 1.8 rejects.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.pipeline.main_task import (
    build_main_task_records,
    resolve_feature_path,
)

MODELS = ["mistral", "gemma", "llama"]
# Strict-rule fixed probe layers (mode of the strict nested run's selected layers).
FIXED_LAYER = {"mistral": 11, "gemma": 13, "llama": 13}
C_GRID = [0.01, 0.1, 1.0, 10.0]
N_OUTER = 5
N_INNER = 3
N_BOOT = 2000
SEED = 0
MC_DATASETS = {"mmlu_pro", "gpqa"}
SPAN_DATASETS = {"hotpotqa", "triviaqa"}


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def load_layer_states(
    records: list[dict[str, Any]], source: str, data_dir: Path, layer: int
) -> np.ndarray:
    key = f"pos_-1_layer_{layer}"
    rows = []
    for rec in records:
        trace = rec[source]
        if trace is None:
            raise RuntimeError(f"missing {source} trace for {rec['group']}")
        path = resolve_feature_path(trace, data_dir)
        if path is None:
            raise RuntimeError(f"no feature file for {source} {rec['group']}")
        with np.load(path) as z:
            if key not in z.files:
                raise RuntimeError(f"{key} missing in {path}")
            rows.append(z[key].astype(np.float32))
    return np.stack(rows, axis=0)


# --------------------------------------------------------------------------- #
# estimator
# --------------------------------------------------------------------------- #
def make_clf(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",  # OvR for multi-class; same base estimator as paper
    )


def fit_predict_proba(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    c_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize inside the training fold only, fit, return (proba, classes)."""
    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xte = scaler.transform(x_test)
    clf = make_clf(c_value)
    clf.fit(xtr, y_train)
    return clf.predict_proba(xte), clf.classes_


def splits(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n: int):
    n_eff = min(n, len(np.unique(groups)))
    if n_eff < 2:
        return []
    return list(GroupKFold(n_splits=n_eff).split(x, y, groups))


def oof_multiclass(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[Any], list[dict[str, Any]]]:
    """Nested-CV out-of-fold predictions. Returns (pred, proba, classes, folds)."""
    all_classes = sorted(set(y.tolist()))
    cls_index = {c: i for i, c in enumerate(all_classes)}
    proba = np.zeros((len(y), len(all_classes)), dtype=float)
    pred = np.empty(len(y), dtype=object)
    fold_rows: list[dict[str, Any]] = []
    for fold, (tr, te) in enumerate(splits(x, y, groups, N_OUTER)):
        if len(set(y[tr].tolist())) < 2:
            continue
        best = (-math.inf, C_GRID[0])
        for c_value in C_GRID:
            scores = []
            for itr_rel, iva_rel in splits(x[tr], y[tr], groups[tr], N_INNER):
                itr, iva = tr[itr_rel], tr[iva_rel]
                if len(set(y[itr].tolist())) < 2:
                    continue
                p, classes = fit_predict_proba(x[itr], y[itr], x[iva], c_value)
                guess = classes[p.argmax(axis=1)]
                scores.append(float(np.mean(guess == y[iva])))
            score = float(np.mean(scores)) if scores else -math.inf
            if score > best[0]:
                best = (score, c_value)
        c_value = best[1]
        p, classes = fit_predict_proba(x[tr], y[tr], x[te], c_value)
        for j, c in enumerate(classes):
            proba[te, cls_index[c]] = p[:, j]
        pred[te] = classes[p.argmax(axis=1)]
        fold_rows.append(
            {
                "fold": fold,
                "C": c_value,
                "inner_acc": best[0],
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "n_train_classes": int(len(classes)),
            }
        )
    return pred, proba, all_classes, fold_rows


def macro_auroc(y: np.ndarray, proba: np.ndarray, classes: list[Any]) -> float:
    """One-vs-rest macro AUROC over classes with both labels present."""
    vals = []
    for j, c in enumerate(classes):
        pos = (y == c).astype(int)
        if 0 < pos.sum() < len(pos):
            vals.append(float(roc_auc_score(pos, proba[:, j])))
    return float(np.mean(vals)) if vals else float("nan")


def cluster_bootstrap(
    groups: np.ndarray, metric_fn: Callable[[np.ndarray], float], n_boot: int, seed: int
) -> dict[str, float]:
    """Bootstrap over unique question groups; metric_fn takes row indices."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    index_of = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        sampled = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index_of[g] for g in sampled])
        v = metric_fn(idx)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals.append(v)
    if not vals:
        return {"ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "ci_low": float(np.percentile(vals, 2.5)),
        "ci_high": float(np.percentile(vals, 97.5)),
    }


# --------------------------------------------------------------------------- #
# task runners
# --------------------------------------------------------------------------- #
def run_multiclass_task(
    name: str,
    feature: str,
    model: str,
    x: np.ndarray,
    labels: list[Any],
    groups_list: list[str],
    note: str,
) -> dict[str, Any]:
    y = np.array(labels, dtype=object)
    groups = np.array(groups_list, dtype=str)
    pred, proba, classes, folds = oof_multiclass(x, y, groups)
    scored = np.array([p is not None for p in pred])
    y_s, pred_s, proba_s, groups_s = y[scored], pred[scored], proba[scored], groups[scored]
    acc = float(np.mean(pred_s == y_s))
    mauroc = macro_auroc(y_s, proba_s, classes)
    counts = pd.Series(list(y_s)).value_counts()
    majority = float(counts.iloc[0] / len(y_s))
    uniform = 1.0 / len(classes)

    acc_ci = cluster_bootstrap(
        groups_s, lambda idx: float(np.mean(pred_s[idx] == y_s[idx])), N_BOOT, SEED
    )
    auc_ci = cluster_bootstrap(
        groups_s, lambda idx: macro_auroc(y_s[idx], proba_s[idx], classes), N_BOOT, SEED
    )
    return {
        "model": model,
        "task": name,
        "feature": feature,
        "kind": "multiclass" if len(classes) > 2 else "binary",
        "layer": FIXED_LAYER[model],
        "n": int(scored.sum()),
        "n_classes": len(classes),
        "metric_acc": acc,
        "acc_ci_low": acc_ci["ci_low"],
        "acc_ci_high": acc_ci["ci_high"],
        "chance_majority": majority,
        "chance_uniform": uniform,
        "macro_auroc": mauroc,
        "macro_auroc_ci_low": auc_ci["ci_low"],
        "macro_auroc_ci_high": auc_ci["ci_high"],
        "selected_C": ",".join(str(f["C"]) for f in folds),
        "note": note,
    }


def run_binary_task(
    name: str,
    feature: str,
    model: str,
    x: np.ndarray,
    labels: list[int],
    groups_list: list[str],
    note: str,
) -> dict[str, Any]:
    """Binary AUROC task -- the exact shape of the dependence probes."""
    y = np.array(labels, dtype=int)
    groups = np.array(groups_list, dtype=str)
    probs = np.zeros(len(y), dtype=float)
    used = np.zeros(len(y), dtype=bool)
    cs = []
    for tr, te in splits(x, y, groups, N_OUTER):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
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
        c_value = best[1]
        p, _ = fit_predict_proba(x[tr], y[tr], x[te], c_value)
        probs[te] = p[:, 1]
        used[te] = True
        cs.append(c_value)
    y_s, p_s, g_s = y[used], probs[used], groups[used]
    auroc = float(roc_auc_score(y_s, p_s))
    acc = float(np.mean((p_s >= 0.5).astype(int) == y_s))

    def boot_auc(idx: np.ndarray) -> float:
        if len(np.unique(y_s[idx])) < 2:
            return float("nan")
        return float(roc_auc_score(y_s[idx], p_s[idx]))

    auc_ci = cluster_bootstrap(g_s, boot_auc, N_BOOT, SEED)
    acc_ci = cluster_bootstrap(
        g_s, lambda idx: float(np.mean((p_s[idx] >= 0.5).astype(int) == y_s[idx])), N_BOOT, SEED
    )
    base = float(max(y_s.mean(), 1 - y_s.mean()))
    return {
        "model": model,
        "task": name,
        "feature": feature,
        "kind": "binary",
        "layer": FIXED_LAYER[model],
        "n": int(used.sum()),
        "n_classes": 2,
        "metric_acc": acc,
        "acc_ci_low": acc_ci["ci_low"],
        "acc_ci_high": acc_ci["ci_high"],
        "chance_majority": base,
        "chance_uniform": 0.5,
        "macro_auroc": auroc,
        "macro_auroc_ci_low": auc_ci["ci_low"],
        "macro_auroc_ci_high": auc_ci["ci_high"],
        "selected_C": ",".join(str(c) for c in cs),
        "note": note,
    }


# --------------------------------------------------------------------------- #
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
    sfx = args.suffix

    rows: list[dict[str, Any]] = []
    slice_rows: list[dict[str, Any]] = []

    for model in args.models:
        layer = FIXED_LAYER[model]
        recs = build_main_task_records(model, data_dir)
        print(f"[{model}] N={len(recs)} n_dep={sum(r['label'] for r in recs)} layer={layer}",
              flush=True)
        w = load_layer_states(recs, "w", data_dir, layer)
        g = load_layer_states(recs, "g", data_dir, layer)
        c = load_layer_states(recs, "c", data_dir, layer)
        d_wg = w - g
        d_wc = w - c
        groups_all = [r["group"] for r in recs]

        # sanity: the dependence label itself, same estimator / layer
        for feat, mat in [("W", w), ("G", g), ("C", c), ("W-G", d_wg), ("W-C", d_wc)]:
            rows.append(
                run_binary_task(
                    "ref_dependence_label",
                    feat,
                    model,
                    mat,
                    [r["label"] for r in recs],
                    groups_all,
                    "reference: the dependence label on the re-extracted states",
                )
            )
            print(f"  ref dep {feat}: {rows[-1]['macro_auroc']:.3f}", flush=True)

        # ---------------- (c) FLOOR ---------------- #
        for feat, mat in [("W-G", d_wg), ("W-C", d_wc)]:
            x_floor = np.concatenate([mat, -mat], axis=0)
            y_floor = [1] * len(mat) + [0] * len(mat)
            g_floor = groups_all + groups_all
            rows.append(
                run_binary_task(
                    "floor_sign_of_difference",
                    feat,
                    model,
                    x_floor,
                    y_floor,
                    g_floor,
                    "sign of the difference (+d vs -d); separable in sample but not across unseen "
                    "questions under grouped folds",
                )
            )
            print(f"  floor {feat}: {rows[-1]['macro_auroc']:.4f}", flush=True)

        # ---------------- (a)/(d) PRIMARY: multiple-choice subset ---------------- #
        mc = [i for i, r in enumerate(recs) if r["dataset"] in MC_DATASETS]
        mc_groups = [groups_all[i] for i in mc]
        h1 = [recs[i]["h1"] for i in mc]
        h2 = [recs[i]["h2"] for i in mc]
        pair = [f"{a}|{b}" for a, b in zip(h1, h2)]
        slice_rows.append(
            {
                "model": model,
                "slice": "multiple_choice",
                "n": len(mc),
                "n_h1_classes": len(set(h1)),
                "n_pair_classes": len(set(pair)),
            }
        )
        feats_mc = {
            "W": w[mc],
            "G": g[mc],
            "C": c[mc],
            "W-G": d_wg[mc],
            "W-C": d_wc[mc],
        }
        for feat, mat in feats_mc.items():
            rows.append(
                run_multiclass_task(
                    "mc_h1_letter",
                    feat,
                    model,
                    mat,
                    h1,
                    mc_groups,
                    "h1 letter is named in the W note and absent from the G note",
                )
            )
            print(f"  mc_h1 {feat}: acc={rows[-1]['metric_acc']:.3f} "
                  f"(maj {rows[-1]['chance_majority']:.3f}) "
                  f"mAUROC={rows[-1]['macro_auroc']:.3f}", flush=True)
            rows.append(
                run_multiclass_task(
                    "mc_h2_letter",
                    feat,
                    model,
                    mat,
                    h2,
                    mc_groups,
                    "h2 letter is named in the G note and absent from the W note",
                )
            )
            rows.append(
                run_multiclass_task(
                    "mc_ordered_pair",
                    feat,
                    model,
                    mat,
                    pair,
                    mc_groups,
                    "ordered (h1,h2); very many classes, small n -- read with care",
                )
            )
        # balanced binary version of the same contrast
        order_y = [int(a > b) for a, b in zip(h1, h2)]
        for feat, mat in feats_mc.items():
            rows.append(
                run_binary_task(
                    "mc_h1_after_h2_alphabetically",
                    feat,
                    model,
                    mat,
                    order_y,
                    mc_groups,
                    "balanced binary form of the letter-swap contrast",
                )
            )
            print(f"  mc_order {feat}: AUROC={rows[-1]['macro_auroc']:.3f}", flush=True)

        # ---------------- (b)/(d) SECONDARY: surface properties ---------------- #
        # b1: h1 lexically longer than h2 (characters). MC letters always tie, so
        # the non-tie rows are the span datasets plus any unequal MC strings.
        keep = [i for i, r in enumerate(recs) if len(str(r["h1"])) != len(str(r["h2"]))]
        y_len = [int(len(str(recs[i]["h1"])) > len(str(recs[i]["h2"]))) for i in keep]
        slice_rows.append(
            {
                "model": model,
                "slice": "length_nontie_all_datasets",
                "n": len(keep),
                "n_h1_classes": 2,
                "n_pair_classes": 0,
            }
        )
        for feat, mat in [("W", w), ("G", g), ("C", c), ("W-G", d_wg), ("W-C", d_wc)]:
            rows.append(
                run_binary_task(
                    "surface_h1_longer_than_h2",
                    feat,
                    model,
                    mat[keep],
                    y_len,
                    [groups_all[i] for i in keep],
                    "char length of the candidate string; a surface property of the swapped span",
                )
            )
            print(f"  len {feat}: AUROC={rows[-1]['macro_auroc']:.3f}", flush=True)

        # b2: span subset, first character of h1
        span = [i for i, r in enumerate(recs) if r["dataset"] in SPAN_DATASETS]
        span_groups = [groups_all[i] for i in span]
        first = [str(recs[i]["h1"])[:1].upper() for i in span]
        half = [int(ch >= "N") for ch in first]
        nwords = [int(len(str(recs[i]["h1"]).split()) > 1) for i in span]
        slice_rows.append(
            {
                "model": model,
                "slice": "span_subset",
                "n": len(span),
                "n_h1_classes": len(set(first)),
                "n_pair_classes": 0,
            }
        )
        for feat, mat in [("W", w), ("G", g), ("C", c), ("W-G", d_wg), ("W-C", d_wc)]:
            rows.append(
                run_multiclass_task(
                    "span_h1_first_char",
                    feat,
                    model,
                    mat[span],
                    first,
                    span_groups,
                    "first character of h1; literally the first token of the swapped span",
                )
            )
            rows.append(
                run_binary_task(
                    "span_h1_first_char_half",
                    feat,
                    model,
                    mat[span],
                    half,
                    span_groups,
                    "balanced binary: first char of h1 in N-Z vs A-M",
                )
            )
            rows.append(
                run_binary_task(
                    "span_h1_multiword",
                    feat,
                    model,
                    mat[span],
                    nwords,
                    span_groups,
                    "h1 is multi-word vs single-word",
                )
            )
            print(f"  span firstchar {feat}: acc={rows[-3]['metric_acc']:.3f} "
                  f"(maj {rows[-3]['chance_majority']:.3f}); half AUROC="
                  f"{rows[-2]['macro_auroc']:.3f}; multiword AUROC="
                  f"{rows[-1]['macro_auroc']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"positive_control{sfx}.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(
        out_dir / f"positive_control_slices{sfx}.csv", index=False
    )
    (out_dir / f"positive_control_settings{sfx}.json").write_text(
        json.dumps(
            {
                "data_dir": str(data_dir),
                "fixed_layers": FIXED_LAYER,
                "C_grid": C_GRID,
                "n_outer": N_OUTER,
                "n_inner": N_INNER,
                "n_bootstrap": N_BOOT,
                "seed": SEED,
                "mc_datasets": sorted(MC_DATASETS),
                "span_datasets": sorted(SPAN_DATASETS),
            },
            indent=2,
        )
    )
    print("\nwrote", out_dir / f"positive_control{sfx}.csv")


if __name__ == "__main__":
    main()
