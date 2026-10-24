"""Calibration of the judge-robustness cells (tab:judge-robustness): Brier, ECE with 10 equal-width bins, mean
predicted probability, for each cell and the two mean-probability ensembles. Input: --rows (default
data/scores/monitors/judge_robustness_rows.csv.gz). Output: --out-dir/judge_cell_metrics.csv. Its AUROC
intervals (seed 1729) are not the printed ones: the paper's intervals come from monitor_intervals.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

MODELS = ["mistral", "gemma", "llama"]

# first-run Sonnet-4 values printed in tab:corrected-main, for the re-run comparison below
FIRST_RUN = {
    "sonnet4_cot_generic": {"mistral": 0.555, "gemma": 0.577, "llama": 0.619},
    "sonnet4_cot_cf": {"mistral": 0.655, "gemma": 0.652, "llama": 0.667},
}


def bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not m.any():
            continue
        rows.append({"bin": f"[{lo:.1f},{hi:.1f})", "n": int(m.sum()), "mean_pred": p[m].mean(), "frac_dep": y[m].mean()})
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    b = bins(y, p, n_bins)
    if b.empty:
        return float("nan")
    return float((b["n"] / len(y) * (b["mean_pred"] - b["frac_dep"]).abs()).sum())


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "n": len(y),
        "base_rate": float(y.mean()),
        "auroc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(np.mean((p - y) ** 2)),
        "brier_baserate": float(np.mean((y.mean() - y) ** 2)),
        "ece": ece(y, p),
        "mean_pred": float(p.mean()),
        "mean_pred_dep": float(p[y == 1].mean()),
        "mean_pred_indep": float(p[y == 0].mean()),
    }


def boot_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 1729) -> tuple[float, float]:
    """Cluster bootstrap over question-id groups (seed 1729; the paper's intervals come from monitor_intervals.py)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in pick])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) < 2:
            continue
        vals.append(roc_auc_score(yy, pp))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, default=Path("data/scores/monitors/judge_robustness_rows.csv.gz"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/monitors"))
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    df = pd.read_csv(args.rows)
    df = df[df["parse_ok"].astype(bool) & df["prob_dependent"].notna()].copy()

    print(f"rows scored: {len(df)}  cells: {df['cell'].nunique()}  total cost US${pd.read_csv(args.rows)['cost_usd'].sum():.4f}\n")

    # ---- per-cell metric table -------------------------------------------
    recs = []
    for (cell, model), sub in df.groupby(["cell", "model"]):
        y = sub["label"].to_numpy(int)
        p = sub["prob_dependent"].to_numpy(float)
        m = metrics(y, p)
        lo, hi = boot_ci(y, p, sub["group"].to_numpy(), args.n_boot)
        recs.append(
            {
                "cell": cell,
                "model": model,
                "judge_model": sub["judge_model"].iloc[0],
                **m,
                "auroc_lo": lo,
                "auroc_hi": hi,
                "cost_usd": float(sub["cost_usd"].sum()),
                "mean_tin": float(sub["prompt_tokens"].mean()),
                "mean_tout": float(sub["completion_tokens"].mean()),
            }
        )
    cells = pd.DataFrame(recs)

    # ---- ensembles --------------------------------------------------------
    def ensemble(cell_list: list[str], name: str) -> list[dict]:
        sub = df[df["cell"].isin(cell_list)]
        wide = sub.pivot_table(index=["model", "group", "label"], columns="cell", values="prob_dependent").dropna()
        out = []
        for model, part in wide.reset_index().groupby("model"):
            y = part["label"].to_numpy(int)
            p = part[cell_list].to_numpy(float).mean(axis=1)
            m = metrics(y, p)
            lo, hi = boot_ci(y, p, part["group"].to_numpy(), args.n_boot)
            out.append({"cell": name, "model": model, "judge_model": "mean-prob ensemble", **m, "auroc_lo": lo, "auroc_hi": hi, "cost_usd": 0.0, "mean_tin": np.nan, "mean_tout": np.nan})
        return out

    para = ["sonnet4_cot_cf", "sonnet4_cot_cf_p1", "sonnet4_cot_cf_p2"]
    fam = ["sonnet4_cot_cf", "gpt41mini_cot_cf", "gemini25flash_cot_cf"]
    extra = []
    if set(para) <= set(df["cell"]):
        extra += ensemble(para, "ens_sonnet4_cf_paraphrase")
    if set(fam) <= set(df["cell"]):
        extra += ensemble(fam, "ens_crossfamily_cf")
    if extra:
        cells = pd.concat([cells, pd.DataFrame(extra)], ignore_index=True)

    cells = cells.sort_values(["cell", "model"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(args.out_dir / "judge_cell_metrics.csv", index=False)

    print("## Per-cell metrics\n")
    print("| cell | judge | model | N | AUROC [95% CI] | AUPRC | Brier | Brier(base rate) | ECE | mean p | mean p (dep) | mean p (indep) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, r in cells.iterrows():
        print(
            f"| {r['cell']} | {r['judge_model']} | {r['model']} | {int(r['n'])} | "
            f"{r['auroc']:.3f} [{r['auroc_lo']:.3f}, {r['auroc_hi']:.3f}] | {r['auprc']:.3f} | "
            f"{r['brier']:.3f} | {r['brier_baserate']:.3f} | {r['ece']:.3f} | {r['mean_pred']:.3f} | "
            f"{r['mean_pred_dep']:.3f} | {r['mean_pred_indep']:.3f} |"
        )

    # ---- paraphrase spread ------------------------------------------------
    if set(para) <= set(df["cell"]):
        print("\n## Paraphrase spread (Sonnet-4, CoT-CF view)\n")
        print("| model | P0 (original prompt) | P1 | P2 | range | mean-prob ensemble | pairwise Spearman (P0-P1/P0-P2/P1-P2) |")
        print("|---|---|---|---|---|---|---|")
        wide = df[df["cell"].isin(para)].pivot_table(index=["model", "group", "label"], columns="cell", values="prob_dependent").dropna().reset_index()
        for model in MODELS:
            part = wide[wide["model"] == model]
            a = cells[(cells["model"] == model)]
            vals = {c: float(a[a["cell"] == c]["auroc"].iloc[0]) for c in para}
            ens = float(a[a["cell"] == "ens_sonnet4_cf_paraphrase"]["auroc"].iloc[0])
            rho = part[para].corr(method="spearman")
            print(
                f"| {model} | {vals['sonnet4_cot_cf']:.3f} | {vals['sonnet4_cot_cf_p1']:.3f} | {vals['sonnet4_cot_cf_p2']:.3f} | "
                f"{max(vals.values()) - min(vals.values()):.3f} | {ens:.3f} | "
                f"{rho.loc[para[0], para[1]]:.2f}/{rho.loc[para[0], para[2]]:.2f}/{rho.loc[para[1], para[2]]:.2f} |"
            )

    # ---- reliability ------------------------------------------------------
    print("\n## Reliability bins (pooled over the three models, per cell)\n")
    for cell, sub in df.groupby("cell"):
        y = sub["label"].to_numpy(int)
        p = sub["prob_dependent"].to_numpy(float)
        b = bins(y, p)
        line = "; ".join(f"{r['bin']} n={int(r['n'])} p̄={r['mean_pred']:.2f} dep={r['frac_dep']:.2f}" for _, r in b.iterrows())
        print(f"- **{cell}**: {line}")

    # ---- reproduction check ----------------------------------------------
    print("\n## Re-run versus first-run Sonnet-4 cells\n")
    print("| view | model | first run | this run | delta |")
    print("|---|---|---|---|---|")
    for cell, key in [("sonnet4_cot_generic", "sonnet4_cot_generic"), ("sonnet4_cot_cf", "sonnet4_cot_cf")]:
        for model in MODELS:
            sel = cells[(cells["cell"] == cell) & (cells["model"] == model)]
            if sel.empty:
                continue
            got = float(sel["auroc"].iloc[0])
            pub = FIRST_RUN[key][model]
            print(f"| {cell} | {model} | {pub:.3f} | {got:.3f} | {got - pub:+.3f} |")

    print(f"\nwrote {args.out_dir / 'judge_cell_metrics.csv'}")


if __name__ == "__main__":
    main()
