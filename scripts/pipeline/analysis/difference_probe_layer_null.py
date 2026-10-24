"""Label-permutation null for the best-single-layer AUROC of the difference probes (app:corrected); the defaults
are the recorded run (30 shuffles; W-G, W-C, raw W). C is held at 0.01, so the null slightly under-states
post-hoc optimism. Input: re-extracted prompt-end states (--data-dir). Output (--out-dir): null summary and
per-layer null curves, released as data/scores/controls/difference_probe_null_{summary,curves}.csv. CPU only.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from scripts.pipeline.analysis import difference_probe_layer_search as DPL  # noqa: E402

FIXED_C = 0.01


def layer_fold_scores(
    x: np.ndarray,
    y_variants: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> np.ndarray:
    """Test-fold probabilities for every label permutation at one layer/fold."""
    out = np.zeros((y_variants.shape[0], len(test_idx)), dtype=float)
    for s in range(y_variants.shape[0]):
        y = y_variants[s]
        if len(np.unique(y[train_idx])) < 2:
            out[s] = 0.5
            continue
        out[s] = DPL.fit_dense_predict(x[train_idx], y[train_idx], x[test_idx], FIXED_C)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--models", nargs="+", default=DPL.MODELS)
    parser.add_argument("--feature-sets", nargs="+", default=["W_minus_G", "W_minus_C", "raw_W"])
    parser.add_argument("--n-shuffles", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    curve_rows = []
    for model in args.models:
        records = DPL.RRA.build_main_task_records(model, data_dir)
        layers, aligned, common = DPL.load_sources(records, data_dir, ("w", "g", "c"))
        y_true = np.array([records[i]["label"] for i in common], dtype=int)
        groups = np.array([records[i]["group"] for i in common], dtype=str)

        rng = np.random.default_rng(args.seed)
        y_variants = np.stack(
            [y_true] + [rng.permutation(y_true) for _ in range(args.n_shuffles)]
        )

        outer = GroupKFold(n_splits=min(DPL.N_OUTER, len(np.unique(groups))))
        folds = list(outer.split(aligned["w"][layers[0]], y_true, groups))

        feature_mats = {}
        for name in args.feature_sets:
            if name == "raw_W":
                feature_mats[name] = aligned["w"]
            elif name == "W_minus_G":
                feature_mats[name] = {
                    l: aligned["w"][l] - aligned["g"][l] for l in layers
                }
            elif name == "W_minus_C":
                feature_mats[name] = {
                    l: aligned["w"][l] - aligned["c"][l] for l in layers
                }
            else:
                raise ValueError(name)

        for name, by_layer in feature_mats.items():
            tasks = [(f, layer) for f in range(len(folds)) for layer in layers]
            results = Parallel(n_jobs=args.n_jobs, backend="loky")(
                delayed(layer_fold_scores)(
                    by_layer[layer], y_variants, folds[f][0], folds[f][1]
                )
                for f, layer in tasks
            )
            cells = {t: r for t, r in zip(tasks, results)}

            # per-layer AUROC for every label variant
            auroc = np.zeros((len(y_variants), len(layers)))
            for li, layer in enumerate(layers):
                scores = np.zeros((len(y_variants), len(y_true)))
                for f, (_, test_idx) in enumerate(folds):
                    scores[:, test_idx] = cells[(f, layer)]
                for s in range(len(y_variants)):
                    auroc[s, li] = float(roc_auc_score(y_variants[s], scores[s]))

            observed = auroc[0]
            null = auroc[1:]
            null_max = null.max(axis=1)
            obs_max = float(observed.max())
            rows.append(
                {
                    "model": model,
                    "feature_set": name,
                    "n": len(y_true),
                    "n_layers": len(layers),
                    "n_shuffles": args.n_shuffles,
                    "fixed_C": FIXED_C,
                    "observed_best_layer": int(layers[int(np.argmax(observed))]),
                    "observed_best_auroc": obs_max,
                    "observed_mean_auroc": float(observed.mean()),
                    "null_mean_per_layer": float(null.mean()),
                    "null_sd_per_layer": float(null.std(ddof=1)),
                    "null_max_mean": float(null_max.mean()),
                    "null_max_p95": float(np.percentile(null_max, 95)),
                    "null_max_max": float(null_max.max()),
                    "p_value_best_layer": float((null_max >= obs_max).mean()),
                }
            )
            for li, layer in enumerate(layers):
                curve_rows.append(
                    {
                        "model": model,
                        "feature_set": name,
                        "layer": layer,
                        "observed_auroc_fixedC": float(observed[li]),
                        "null_mean": float(null[:, li].mean()),
                        "null_p95": float(np.percentile(null[:, li], 95)),
                    }
                )
            print(f"[{model}] {name}: {rows[-1]}", flush=True)
            pd.DataFrame(rows).to_csv(
                out_dir / "difference_probe_layer_null_summary.csv", index=False
            )
            pd.DataFrame(curve_rows).to_csv(
                out_dir / "difference_probe_layer_null_curves.csv", index=False
            )

    (out_dir / "difference_probe_layer_null_meta.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )
    print(f"wrote null calibration to {out_dir}")


if __name__ == "__main__":
    main()
