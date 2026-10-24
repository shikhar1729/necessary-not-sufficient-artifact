"""Role-swap candidate-identity holdout (tab:roleswap-holdout): W-G probe and public baselines under base_id grouping and
under the partial letter-cell holdout (12 correct-letter x unordered-pair cells), with cluster-bootstrap CIs over base_id.

Input: a generate_benchmark.py output dir (manifest, letter scores, hidden states, <model>/role_swapped_effect_rows.csv,
and role_swapped_predictions.csv for the base_id check). Uses its fixed-C estimators (the path behind tab:roleswap-main); TF-IDF is fit on all rows before CV.
Outputs in --out-dir: {tag}_summary.csv, {tag}_predictions.csv.gz, {tag}_checks.csv, {tag}_headline.csv (not released).
The paper's run used scikit-learn 1.9.0 (use 1.9.x so the base_id folds match the stored predictions). Paper runs:

    python -m scripts.pipeline.roleswap.letter_cell_holdout --data-dir <dir> --out-dir <out>
    python -m scripts.pipeline.roleswap.letter_cell_holdout --data-dir <dir_qwen14> --out-dir <out> \
        --models qwen3_14b --tag letter_cell_qwen14
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from scripts.pipeline.roleswap.bootstrap_cis import cluster_bootstrap_auroc
from scripts.pipeline.roleswap.generate_benchmark import (
    build_effect_rows,
    grouped_logistic_oof_fixed,
    grouped_ridge_oof_fixed,
    hidden_features_from_loaded,
    hidden_index,
    load_hidden,
    numeric_features,
    summarize_binary,
    summarize_regression,
)

# Best W-G layer per model in tab:roleswap-main; Qwen3-14B uses L39 as in tab:roleswap-holdout.
WG_LAYER = {"mistral": 31, "gemma": 31, "llama": 24, "qwen3_14b": 39}
PUBLIC_METHODS = ["numeric_public", "tfidf_w_prompt", "numeric_plus_tfidf"]


def make_groups(effect_rows: pd.DataFrame, scheme: str) -> np.ndarray:
    if scheme == "base_id":
        return effect_rows["base_id"].astype(str).to_numpy()
    if scheme == "letter_cell":
        return (
            effect_rows["correct_letter"].astype(str)
            + "::"
            + effect_rows["unordered_pair"].astype(str)
        ).to_numpy()
    raise ValueError(f"unknown group scheme: {scheme}")


def build_methods(effect_rows: pd.DataFrame, model_dir: Path) -> dict[str, np.ndarray | sparse.spmatrix]:
    x_num = numeric_features(effect_rows)
    x_text = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=4000).fit_transform(
        effect_rows["w_prompt"].astype(str).tolist()
    )
    methods: dict[str, np.ndarray | sparse.spmatrix] = {
        "numeric_public": x_num,
        "tfidf_w_prompt": x_text,
        "numeric_plus_tfidf": sparse.hstack([sparse.csr_matrix(x_num), x_text]).tocsr(),
        "neutral_logp_features": effect_rows[
            ["neutral_logp_h1", "neutral_logp_h2", "neutral_gap_h1_minus_h2"]
        ].to_numpy(dtype=np.float32),
    }
    meta, arrays = load_hidden(model_dir)
    index = hidden_index(meta)
    for layer in sorted(arrays):
        hidden = arrays[layer]
        for source in ["W_minus_G", "W_hidden", "G_hidden"]:
            methods[f"{source}_L{layer}"] = hidden_features_from_loaded(effect_rows, hidden, index, source)
    return methods


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, required=True, help="generate_benchmark.py --output-dir")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/roleswap_letter_cell"))
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--schemes", nargs="+", default=["base_id", "letter_cell"])
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tag", default="letter_cell_7_9b")
    args = parser.parse_args()

    print(f"scikit-learn {sklearn.__version__}", flush=True)
    stored_pred = pd.read_csv(args.data_dir / "role_swapped_predictions.csv", float_precision="round_trip")
    summary_rows: list[dict] = []
    pred_rows: list[pd.DataFrame] = []
    checks: list[dict] = []

    for model in args.models:
        model_dir = args.data_dir / model
        effect_rows = build_effect_rows(args.data_dir, model)
        stored = pd.read_csv(model_dir / "role_swapped_effect_rows.csv")
        same_order = list(stored["orientation_id"]) == list(effect_rows["orientation_id"])
        same_label = bool((stored["binary_label"].to_numpy() == effect_rows["binary_label"].to_numpy()).all())
        max_target = float(np.max(np.abs(stored["pair_centered_effect"].to_numpy() - effect_rows["pair_centered_effect"].to_numpy())))
        checks.append({"model": model, "check": "effect_rows_match_stored", "value": float(same_order and same_label), "detail": f"max|target diff|={max_target:.2e}"})
        print(f"[{model}] rows={len(effect_rows)} bases={effect_rows['base_id'].nunique()} "
              f"pairs={effect_rows['swap_pair_id'].nunique()} order={same_order} labels={same_label} "
              f"max|target diff|={max_target:.2e}", flush=True)

        y_bin = effect_rows["binary_label"].to_numpy(dtype=int)
        y_reg = effect_rows["pair_centered_effect"].to_numpy(dtype=float)
        clusters = effect_rows["base_id"].astype(str).to_numpy()
        methods = build_methods(effect_rows, model_dir)

        for scheme in args.schemes:
            groups = make_groups(effect_rows, scheme)
            n_groups = len(set(groups.tolist()))
            for method, x in methods.items():
                pred_b, used_b = grouped_logistic_oof_fixed(x, y_bin, groups)
                pred_r, used_r = grouped_ridge_oof_fixed(x, y_reg, groups)
                b = summarize_binary(y_bin, pred_b, used_b)
                r = summarize_regression(y_reg, pred_r, used_r)
                mask = used_b & np.isfinite(pred_b)
                point, lo, hi, n_cl, n_used = cluster_bootstrap_auroc(
                    y_bin[mask], pred_b[mask], clusters[mask], args.n_boot, args.seed
                )
                layer = int(method.rsplit("_L", 1)[1]) if "_L" in method else math.nan
                summary_rows.append(
                    {
                        "model": model,
                        "group_scheme": scheme,
                        "n_groups": n_groups,
                        "method": method,
                        "layer": layer,
                        "auroc": b["auroc"],
                        "ci_low": lo,
                        "ci_high": hi,
                        "auprc": b["auprc"],
                        "pearson": r["pearson"],
                        "n_rows": int(mask.sum()),
                        "n_clusters": n_cl,
                        "n_boot": args.n_boot,
                        "n_boot_used": n_used,
                        "seed": args.seed,
                    }
                )
                assert abs(point - b["auroc"]) < 1e-12
                print(f"[{model}/{scheme}] {method:24s} AUROC {b['auroc']:.4f} [{lo:.4f}, {hi:.4f}] "
                      f"Pearson {r['pearson']:.4f}", flush=True)
                pred_rows.append(
                    pd.DataFrame(
                        {
                            "model": model,
                            "group_scheme": scheme,
                            "method": method,
                            "base_id": effect_rows["base_id"].to_numpy(),
                            "orientation_id": effect_rows["orientation_id"].to_numpy(),
                            "binary_label": y_bin,
                            "pair_centered_effect": y_reg,
                            "pred_b": pred_b,
                            "used_b": used_b,
                            "pred_r": pred_r,
                        }
                    )
                )
                if scheme == "base_id":
                    ref = stored_pred[(stored_pred["model"] == model) & (stored_pred["method"] == method)]
                    if len(ref):
                        ref = ref.set_index("orientation_id").loc[effect_rows["orientation_id"]]
                        diff = float(np.nanmax(np.abs(ref["pred_b"].to_numpy(dtype=float) - pred_b)))
                        checks.append({"model": model, "check": f"base_id_pred_b_matches_stored::{method}", "value": diff, "detail": "max|pred_b diff|"})

    summary = pd.DataFrame(summary_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / f"{args.tag}_summary.csv", index=False, float_format="%.6f")
    pd.concat(pred_rows, ignore_index=True).to_csv(
        args.out_dir / f"{args.tag}_predictions.csv.gz", index=False, float_format="%.17g", compression="gzip"
    )
    pd.DataFrame(checks).to_csv(args.out_dir / f"{args.tag}_checks.csv", index=False)

    print("\nCHECKS (max |pred_b diff| vs stored base_id predictions)")
    chk = pd.DataFrame(checks)
    print(chk.groupby(["model"]).apply(lambda d: d["value"][d["check"].str.startswith("base_id")].max()).to_string())

    print("\nHEADLINE (W-G at the tab:roleswap-main layer; best public of numeric/tfidf/numeric+tfidf)")
    lines = []
    for model in args.models:
        for scheme in args.schemes:
            sub = summary[(summary["model"] == model) & (summary["group_scheme"] == scheme)]
            wg = sub[sub["method"] == f"W_minus_G_L{WG_LAYER[model]}"].iloc[0]
            wg_best = sub[sub["method"].str.startswith("W_minus_G_L")].sort_values("auroc").iloc[-1]
            pub = sub[sub["method"].isin(PUBLIC_METHODS)].sort_values("auroc").iloc[-1]
            pub_r = sub[sub["method"].isin(PUBLIC_METHODS)].sort_values("pearson").iloc[-1]
            raw = sub[sub["method"].str.match(r"^[WG]_hidden_L")].sort_values("auroc").iloc[-1]
            neu = sub[sub["method"] == "neutral_logp_features"].iloc[0]
            lines.append(
                {
                    "model": model,
                    "scheme": scheme,
                    "wg_method": wg["method"],
                    "wg_auroc": wg["auroc"],
                    "wg_ci": f"[{wg['ci_low']:.3f}, {wg['ci_high']:.3f}]",
                    "wg_pearson": wg["pearson"],
                    "wg_best4_method": wg_best["method"],
                    "wg_best4_auroc": wg_best["auroc"],
                    "public_method": pub["method"],
                    "public_auroc": pub["auroc"],
                    "public_ci": f"[{pub['ci_low']:.3f}, {pub['ci_high']:.3f}]",
                    "public_best_pearson_method": pub_r["method"],
                    "public_best_pearson": pub_r["pearson"],
                    "raw_best_method": raw["method"],
                    "raw_best_auroc": raw["auroc"],
                    "neutral_auroc": neu["auroc"],
                }
            )
    head = pd.DataFrame(lines)
    head.to_csv(args.out_dir / f"{args.tag}_headline.csv", index=False, float_format="%.4f")
    print(head.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
