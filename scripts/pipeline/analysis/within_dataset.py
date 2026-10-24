"""Within-dataset, leave-one-dataset-out and pooled W-state probes on the retained slice, plus the dataset-identity
floor (tab:within-dataset-strict). Fixed layers 11/13/13, GroupKFold(5), inner C search, cluster bootstrap B=2000,
seed 0. Input: re-extracted states (--data-dir). Output: --out-dir CSVs; within_dataset_probe_strict.csv,
lodo_probe_strict.csv and within_dataset_macro_strict.csv are released in data/scores/controls/ without the
_strict suffix (the first two without the status column; within_dataset_probe also without n_scored).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from scripts.pipeline.main_task import (
    build_main_task_records,
    load_state_matrix,
)

OUT_DIR = Path("outputs/controls")
DATA_DIR = Path()  # set from --data-dir in main()
FIXED_LAYER = {"mistral": 11, "gemma": 13, "llama": 13}
MODELS = ["mistral", "gemma", "llama"]
DATASETS = ["triviaqa", "hotpotqa", "gpqa", "mmlu_pro"]
C_GRID = [0.01, 0.1, 1.0, 10.0]
# numpy built against Apple Accelerate raises spurious fp-flag RuntimeWarnings
# from matmul. Verified harmless here: all predicted probabilities are finite
# and the analysis asserts that below.
warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
N_BOOTSTRAP = 2000
SEED = 0


# --------------------------------------------------------------------------- #
# core probe primitives (mirroring main_task.py)
# --------------------------------------------------------------------------- #
def fit_predict(x_train, y_train, x_test, c_value):
    # float64 for the scaler/solver: small within-dataset folds contain
    # near-constant dimensions whose float32 variance underflows, which makes
    # liblinear overflow. The features themselves are the stored float32 states.
    scaler = StandardScaler()
    xs_train = scaler.fit_transform(np.asarray(x_train, dtype=np.float64))
    xs_test = scaler.transform(np.asarray(x_test, dtype=np.float64))
    clf = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=1000,
        random_state=0,
        solver="liblinear",
    )
    clf.fit(xs_train, y_train)
    out = clf.predict_proba(xs_test)[:, 1]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite predicted probabilities")
    return out


def inner_select_c(x, y, groups, train_idx):
    """Pick C by inner GroupKFold AUROC on the training rows only."""
    train_groups = np.unique(groups[train_idx])
    n_splits = min(3, len(train_groups))
    best = (-np.inf, 1.0)
    if n_splits < 2:
        return 1.0, np.nan
    splitter = GroupKFold(n_splits=n_splits)
    splits = list(splitter.split(x[train_idx], y[train_idx], groups[train_idx]))
    for c_value in C_GRID:
        scores = []
        for inner_tr_rel, inner_va_rel in splits:
            inner_tr = train_idx[inner_tr_rel]
            inner_va = train_idx[inner_va_rel]
            if len(np.unique(y[inner_tr])) < 2 or len(np.unique(y[inner_va])) < 2:
                continue
            probs = fit_predict(x[inner_tr], y[inner_tr], x[inner_va], c_value)
            scores.append(float(roc_auc_score(y[inner_va], probs)))
        score = float(np.mean(scores)) if scores else 0.5
        if score > best[0]:
            best = (score, c_value)
    return best[1], best[0]


def oof_probs(x, y, groups, n_splits=5):
    """Out-of-fold probabilities with inner-CV C selection per outer fold."""
    probs = np.full(len(y), np.nan, dtype=float)
    used = np.zeros(len(y), dtype=bool)
    folds = np.full(len(y), -1, dtype=int)
    rows = []
    n_groups = len(np.unique(groups))
    splits = min(n_splits, n_groups)
    if splits < 2:
        return probs, used, folds, pd.DataFrame(rows)
    outer = GroupKFold(n_splits=splits)
    for fold, (train_idx, test_idx) in enumerate(outer.split(x, y, groups)):
        folds[test_idx] = fold
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            rows.append(
                {
                    "fold": fold,
                    "status": "skipped_class_balance",
                    "C": np.nan,
                    "inner_auroc": np.nan,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "test_indep_rate": float(1 - y[test_idx].mean()),
                }
            )
            continue
        c_value, inner = inner_select_c(x, y, groups, train_idx)
        probs[test_idx] = fit_predict(x[train_idx], y[train_idx], x[test_idx], c_value)
        used[test_idx] = True
        rows.append(
            {
                "fold": fold,
                "status": "ok",
                "C": c_value,
                "inner_auroc": inner,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "test_indep_rate": float(1 - y[test_idx].mean()),
            }
        )
    return probs, used, folds, pd.DataFrame(rows)


def cluster_bootstrap_auroc(y, scores, groups, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    if len(np.unique(y)) < 2:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}
    point = float(roc_auc_score(y, scores))
    unique_groups = np.unique(groups)
    index_by_group = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([index_by_group[g] for g in sampled])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(float(roc_auc_score(y[idx], scores[idx])))
    if not values:
        return {"point": point, "ci_low": np.nan, "ci_high": np.nan, "n_boot": 0}
    return {
        "point": point,
        "ci_low": float(np.percentile(values, 2.5)),
        "ci_high": float(np.percentile(values, 97.5)),
        "n_boot": len(values),
    }


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load_model(model):
    records = build_main_task_records(model, DATA_DIR)
    layer = FIXED_LAYER[model]
    layers, x_by_layer, kept = load_state_matrix(
        records, "w", DATA_DIR, only_layers=[layer]
    )
    sub = [records[i] for i in kept]
    x = x_by_layer[layer]
    frame = pd.DataFrame(
        {
            "dataset": [r["dataset"] for r in sub],
            "question_id": [r["question_id"] for r in sub],
            "group": [r["group"] for r in sub],
            "label": [r["label"] for r in sub],
        }
    )
    return frame, x, layer


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def run_pooled(model, frame, x, layer):
    y = frame["label"].to_numpy(int)
    groups = frame["group"].to_numpy(str)
    probs, used, folds, fold_df = oof_probs(x, y, groups)
    ci = cluster_bootstrap_auroc(y[used], probs[used], groups[used])
    summary = {
        "model": model,
        "scope": "pooled",
        "layer": layer,
        "n": int(used.sum()),
        "n_dep": int(y[used].sum()),
        "n_indep": int(used.sum() - y[used].sum()),
        "indep_rate": float(1 - y[used].mean()),
        "auroc": ci["point"],
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
    }

    # dataset-base-rate floor on exactly the same folds: each test row is scored
    # by the dep rate of its own dataset estimated on that fold's training rows.
    floor = np.full(len(y), np.nan)
    for fold in sorted(set(folds[folds >= 0])):
        test_mask = folds == fold
        train_mask = (~test_mask) & (folds >= 0)
        rates = (
            frame.loc[train_mask]
            .groupby("dataset")["label"]
            .mean()
            .to_dict()
        )
        overall = float(y[train_mask].mean())
        floor[test_mask] = [
            rates.get(d, overall) for d in frame.loc[test_mask, "dataset"]
        ]
    floor_mask = used & ~np.isnan(floor)
    fci = cluster_bootstrap_auroc(y[floor_mask], floor[floor_mask], groups[floor_mask])
    floor_summary = [
        {
            "model": model,
            "scope": "pooled",
            "floor_variant": "out_of_fold",
            "dataset": "ALL",
            "n": int(floor_mask.sum()),
            "auroc": fci["point"],
            "ci_low": fci["ci_low"],
            "ci_high": fci["ci_high"],
        }
    ]
    # Oracle variant: dataset dep rates estimated on the FULL retained slice.
    # This is the in-sample oracle printed in the paper (0.632/0.620/0.672).
    rates_all = frame.groupby("dataset")["label"].mean().to_dict()
    oracle = frame["dataset"].map(rates_all).to_numpy(float)
    oci = cluster_bootstrap_auroc(y, oracle, groups)
    floor_summary.append(
        {
            "model": model,
            "scope": "pooled",
            "floor_variant": "in_sample_oracle",
            "dataset": "ALL",
            "n": int(len(y)),
            "auroc": oci["point"],
            "ci_low": oci["ci_low"],
            "ci_high": oci["ci_high"],
        }
    )
    for ds, rate in sorted(rates_all.items()):
        floor_summary.append(
            {
                "model": model,
                "scope": "within_dataset",
                "floor_variant": "constant_by_construction",
                "dataset": ds,
                "n": int((frame["dataset"] == ds).sum()),
                "auroc": 0.5,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "dataset_dep_rate": float(rate),
            }
        )

    fold_df = fold_df.copy()
    fold_df.insert(0, "model", model)
    # per-fold dataset composition and label balance
    comp_rows = []
    for fold in sorted(set(folds[folds >= 0])):
        mask = folds == fold
        row = {
            "model": model,
            "fold": int(fold),
            "n_test": int(mask.sum()),
            "n_dep": int(y[mask].sum()),
            "n_indep": int(mask.sum() - y[mask].sum()),
            "indep_rate": float(1 - y[mask].mean()),
        }
        counts = frame.loc[mask, "dataset"].value_counts()
        for ds in DATASETS:
            row[f"n_{ds}"] = int(counts.get(ds, 0))
            sub = frame.loc[mask & (frame["dataset"] == ds).to_numpy()]
            row[f"indep_rate_{ds}"] = (
                float(1 - sub["label"].mean()) if len(sub) else np.nan
            )
        comp_rows.append(row)

    # pooled out-of-fold scores restricted to each dataset ("pooled, scored
    # within dataset"); the paper prints its macro-average.
    restricted_rows = []
    for ds in DATASETS:
        mask = used & (frame["dataset"] == ds).to_numpy()
        if mask.sum() == 0 or len(np.unique(y[mask])) < 2:
            continue
        ci = cluster_bootstrap_auroc(y[mask], probs[mask], groups[mask])
        # same-dataset base-rate floor is a constant => AUROC 0.5 by construction
        restricted_rows.append(
            {
                "model": model,
                "dataset": ds,
                "n": int(mask.sum()),
                "n_dep": int(y[mask].sum()),
                "n_indep": int(mask.sum() - y[mask].sum()),
                "indep_rate": float(1 - y[mask].mean()),
                "auroc": ci["point"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    return (
        summary,
        floor_summary,
        fold_df,
        pd.DataFrame(comp_rows),
        pd.DataFrame(restricted_rows),
    )


def run_within_dataset(model, frame, x, layer):
    rows = []
    for ds in DATASETS:
        mask = (frame["dataset"] == ds).to_numpy()
        if mask.sum() == 0:
            continue
        sub = frame.loc[mask]
        y = sub["label"].to_numpy(int)
        groups = sub["group"].to_numpy(str)
        xs = x[mask]
        base = {
            "model": model,
            "dataset": ds,
            "layer": layer,
            "n": int(mask.sum()),
            "n_dep": int(y.sum()),
            "n_indep": int(len(y) - y.sum()),
            "indep_rate": float(1 - y.mean()),
        }
        if len(np.unique(y)) < 2:
            rows.append({**base, "status": "single_class", "auroc": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "n_scored": 0})
            continue
        probs, used, _folds, _fold_df = oof_probs(xs, y, groups)
        if used.sum() == 0 or len(np.unique(y[used])) < 2:
            rows.append({**base, "status": "no_usable_fold", "auroc": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan, "n_scored": 0})
            continue
        ci = cluster_bootstrap_auroc(y[used], probs[used], groups[used])
        rows.append(
            {
                **base,
                "status": "ok",
                "n_scored": int(used.sum()),
                "auroc": ci["point"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def run_lodo(model, frame, x, layer):
    y_all = frame["label"].to_numpy(int)
    groups_all = frame["group"].to_numpy(str)
    rows = []
    for ds in DATASETS:
        test_mask = (frame["dataset"] == ds).to_numpy()
        train_mask = ~test_mask
        if test_mask.sum() == 0:
            continue
        y_tr, y_te = y_all[train_mask], y_all[test_mask]
        base = {
            "model": model,
            "heldout_dataset": ds,
            "layer": layer,
            "n_train": int(train_mask.sum()),
            "n_train_dep": int(y_tr.sum()),
            "n_train_indep": int(len(y_tr) - y_tr.sum()),
            "n_test": int(test_mask.sum()),
            "n_test_dep": int(y_te.sum()),
            "n_test_indep": int(len(y_te) - y_te.sum()),
            "test_indep_rate": float(1 - y_te.mean()),
        }
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            rows.append({**base, "status": "skipped_class_balance", "C": np.nan,
                         "inner_auroc": np.nan, "auroc": np.nan,
                         "ci_low": np.nan, "ci_high": np.nan})
            continue
        train_idx = np.flatnonzero(train_mask)
        c_value, inner = inner_select_c(x, y_all, groups_all, train_idx)
        probs = fit_predict(x[train_mask], y_tr, x[test_mask], c_value)
        ci = cluster_bootstrap_auroc(y_te, probs, groups_all[test_mask])
        rows.append(
            {
                **base,
                "status": "ok",
                "C": c_value,
                "inner_auroc": inner,
                "auroc": ci["point"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
            }
        )
    return pd.DataFrame(rows)


def main():
    global DATA_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="re-extracted states (traces/ and features/)")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--tag", default="strict")
    args = parser.parse_args()
    DATA_DIR = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pooled_rows, floor_rows = [], []
    within_frames, lodo_frames, fold_frames, comp_frames = [], [], [], []
    restricted_frames = []
    for model in MODELS:
        frame, x, layer = load_model(model)
        print(f"[{model}] layer={layer} n={len(frame)} dep={int(frame['label'].sum())}",
              flush=True)
        pooled, floor, fold_df, comp, restricted = run_pooled(model, frame, x, layer)
        pooled_rows.append(pooled)
        floor_rows.extend(floor)
        fold_frames.append(fold_df)
        comp_frames.append(comp)
        restricted_frames.append(restricted)
        within = run_within_dataset(model, frame, x, layer)
        within_frames.append(within)
        lodo_frames.append(run_lodo(model, frame, x, layer))
        print(within.to_string(index=False), flush=True)

    within_all = pd.concat(within_frames, ignore_index=True)
    lodo_all = pd.concat(lodo_frames, ignore_index=True)
    pooled_all = pd.DataFrame(pooled_rows)
    floor_all = pd.DataFrame(floor_rows)
    folds_all = pd.concat(fold_frames, ignore_index=True)
    comp_all = pd.concat(comp_frames, ignore_index=True)
    restricted_all = pd.concat(restricted_frames, ignore_index=True)

    macro_rows = []
    for model, sub in within_all.groupby("model", sort=False):
        ok = sub[sub["status"] == "ok"]
        rsub = restricted_all[restricted_all["model"] == model]
        macro_rows.append(
            {
                "model": model,
                "n_datasets": len(ok),
                "pooled_restricted_macro_auroc": float(rsub["auroc"].mean()),
                "macro_auroc": float(ok["auroc"].mean()) if len(ok) else np.nan,
                "weighted_auroc": (
                    float((ok["auroc"] * ok["n_scored"]).sum() / ok["n_scored"].sum())
                    if len(ok)
                    else np.nan
                ),
                "min_auroc": float(ok["auroc"].min()) if len(ok) else np.nan,
                "max_auroc": float(ok["auroc"].max()) if len(ok) else np.nan,
                "lodo_macro_auroc": float(
                    lodo_all[(lodo_all["model"] == model)
                             & (lodo_all["status"] == "ok")]["auroc"].mean()
                ),
            }
        )
    macro_all = pd.DataFrame(macro_rows)

    tag = args.tag
    within_all.to_csv(out_dir / f"within_dataset_probe_{tag}.csv", index=False)
    lodo_all.to_csv(out_dir / f"lodo_probe_{tag}.csv", index=False)
    pooled_all.to_csv(out_dir / f"pooled_probe_{tag}.csv", index=False)
    floor_all.to_csv(out_dir / f"dataset_base_rate_floor_{tag}.csv", index=False)
    folds_all.to_csv(out_dir / f"pooled_fold_detail_{tag}.csv", index=False)
    comp_all.to_csv(out_dir / f"pooled_fold_composition_{tag}.csv", index=False)
    macro_all.to_csv(out_dir / f"within_dataset_macro_{tag}.csv", index=False)
    restricted_all.to_csv(
        out_dir / f"pooled_probe_per_dataset_{tag}.csv", index=False
    )

    print("\n=== pooled restricted per dataset ===")
    print(restricted_all.to_string(index=False))
    print("\n=== within dataset ===")
    print(within_all.to_string(index=False))
    print("\n=== pooled ===")
    print(pooled_all.to_string(index=False))
    print("\n=== dataset-base-rate floor (pooled folds) ===")
    print(floor_all.to_string(index=False))
    print("\n=== macro ===")
    print(macro_all.to_string(index=False))
    print("\n=== lodo ===")
    print(lodo_all.to_string(index=False))
    print("\n=== fold detail ===")
    print(folds_all.to_string(index=False))
    print("\n=== fold composition ===")
    print(comp_all.to_string(index=False))
    print(json.dumps({"out_dir": str(out_dir), "tag": tag}))


if __name__ == "__main__":
    main()
