"""Cluster-bootstrap 95% CIs for the role-swap binary AUROCs (tab:roleswap, tab:roleswap-main), resampling base_id.

Input: role_swapped_predictions.csv from generate_benchmark.py --analyze (the logistic OOF score pred_b, used_b,
binary_label, base_id). Output: --out CSV, one row per (model, method), B=2000, seed 0; the intervals are printed in
the paper and not released as a file. Paper run, once per output dir (7-9B and Qwen3-14B), then --check with --expect:

    python -m scripts.pipeline.roleswap.bootstrap_cis --pred <dir>/role_swapped_predictions.csv \
        --out <dir>/roleswap_auroc_cis_all.csv
    python -m scripts.pipeline.roleswap.bootstrap_cis --pred <dir>/role_swapped_predictions.csv \
        --out <dir>/roleswap_auroc_cis_all.csv --check --expect mistral:W_minus_G_L31=0.695 \
        --expect gemma:W_minus_G_L31=0.961 --expect llama:W_minus_G_L24=0.993
"""

from __future__ import annotations

import argparse
import glob
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCORE_COL = "pred_b"
GROUP_COL = "base_id"
LABEL_CANDIDATES = ("label_b", "label", "binary_label")
USED_CANDIDATES = ("used_b", "used")
TOL = 0.0005  # |computed - expected| that counts as a match under --check (equal after rounding to 3 decimals)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        metavar="CSV_OR_GLOB",
        help="prediction CSV path or glob; repeat to pass several files",
    )
    parser.add_argument("--out", required=True, help="output CSV (one row per model x method)")
    parser.add_argument("--methods", nargs="+", default=None, metavar="METHOD", help="only these methods (exact match)")
    parser.add_argument("--models", nargs="+", default=None, metavar="MODEL", help="only these models (exact match)")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare the recomputed point AUROC with --expect values; exit 1 on any mismatch",
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="MODEL:METHOD=VALUE",
        help="expected point AUROC from the paper, e.g. gemma:W_minus_G_L31=0.961 (repeatable)",
    )
    return parser.parse_args(argv)


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if not hits and Path(pattern).exists():
            hits = [pattern]
        if not hits:
            sys.exit(f"error: no file matches --pred {pattern!r}")
        paths.extend(Path(h) for h in hits)
    return paths


def pick_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def to_binary(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=int)
    if series.dtype == object:
        mapped = series.astype(str).str.strip().str.lower().map(
            {"1": 1, "0": 0, "true": 1, "false": 0, "1.0": 1, "0.0": 0}
        )
        if mapped.isna().any():
            bad = series[mapped.isna()].unique()[:5]
            sys.exit(f"error: label column has non-binary values, e.g. {list(bad)}")
        return mapped.to_numpy(dtype=int)
    values = series.to_numpy(dtype=float)
    if not np.all(np.isin(values, (0.0, 1.0))):
        sys.exit(f"error: label column has values outside {{0, 1}}: {np.unique(values)[:5]}")
    return values.astype(int)


def to_bool(series: pd.Series) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy()
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"}).to_numpy()


def load_predictions(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    for path in expand_paths(args.pred):
        # round_trip: pred_b is written with %.17g; the default float parser can perturb the last bit and
        # create or break ties, which moves the recomputed AUROC away from the summary value.
        df = pd.read_csv(path, float_precision="round_trip")
        for col in ("model", "method", GROUP_COL, SCORE_COL):
            if col not in df.columns:
                sys.exit(f"error: {path}: required column {col!r} missing (columns: {list(df.columns)})")
        label_col = pick_column(df, LABEL_CANDIDATES)
        if label_col is None:
            sys.exit(f"error: {path}: no label column (tried {LABEL_CANDIDATES})")
        used_col = pick_column(df, USED_CANDIDATES)

        tidy = pd.DataFrame(
            {
                # parent dir + basename: the 7-9B and Qwen3-14B runs both write role_swapped_predictions.csv.
                "source": "/".join(path.resolve().parts[-2:]),
                "model": df["model"].astype(str),
                "method": df["method"].astype(str),
                "cluster": df[GROUP_COL].astype(str),
                "label": to_binary(df[label_col]),
                "score": pd.to_numeric(df[SCORE_COL], errors="coerce").to_numpy(dtype=float),
            }
        )
        keep = np.isfinite(tidy["score"].to_numpy())
        if used_col is not None:
            keep &= to_bool(df[used_col])
        dropped = int((~keep).sum())
        print(
            f"[load] {path}: {len(df)} rows; label={label_col!r} score={SCORE_COL!r} "
            f"used={used_col!r}; dropped {dropped} rows (unused or non-finite score)",
            file=sys.stderr,
        )
        frames.append(tidy[keep].reset_index(drop=True))
    data = pd.concat(frames, ignore_index=True)
    if args.models:
        data = data[data["model"].isin(args.models)]
    if args.methods:
        data = data[data["method"].isin(args.methods)]
        missing = sorted(set(args.methods) - set(data["method"].unique()))
        if missing:
            print(f"[warn] methods not found in input: {missing}", file=sys.stderr)
    if data.empty:
        sys.exit("error: no rows left after filtering")
    return data.reset_index(drop=True)


def cluster_bootstrap_auroc(
    y: np.ndarray, score: np.ndarray, clusters: np.ndarray, n_boot: int, seed: int
) -> tuple[float, float, float, int, int]:
    """Return (point, ci_low, ci_high, n_clusters, n_boot_used).

    Point AUROC uses all rows. Each replicate draws n_clusters cluster ids with
    replacement and concatenates their rows. Replicates with one class are skipped.
    """
    point = float(roc_auc_score(y, score))
    ids, inverse = np.unique(clusters, return_inverse=True)
    n_clusters = len(ids)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(n_clusters + 1))
    members = [order[bounds[i] : bounds[i + 1]] for i in range(n_clusters)]

    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        idx = np.concatenate([members[c] for c in draw])
        y_b = y[idx]
        if y_b.min() == y_b.max():
            continue
        stats.append(roc_auc_score(y_b, score[idx]))
    if not stats:
        return point, math.nan, math.nan, n_clusters, 0
    low, high = np.percentile(stats, [2.5, 97.5])
    return point, float(low), float(high), n_clusters, len(stats)


def parse_expect(items: list[str]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for item in items:
        try:
            key, value = item.rsplit("=", 1)
            model, method = key.split(":", 1)
            out[(model.strip(), method.strip())] = float(value)
        except ValueError:
            sys.exit(f"error: --expect must look like MODEL:METHOD=VALUE, got {item!r}")
    return out


def format_latex(row: pd.Series) -> str:
    return f"{row['model']} & {row['method']} & {row['auroc']:.3f} [{row['ci_low']:.2f}, {row['ci_high']:.2f}] \\\\"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_boot < 1:
        sys.exit("error: --n-boot must be >= 1")
    data = load_predictions(args)

    results = []
    for (source, model, method), grp in data.groupby(["source", "model", "method"], sort=True):
        y = grp["label"].to_numpy(dtype=int)
        if y.min() == y.max():
            print(f"[warn] {model}/{method}: single class, skipped", file=sys.stderr)
            continue
        point, low, high, n_clusters, n_used = cluster_bootstrap_auroc(
            y, grp["score"].to_numpy(dtype=float), grp["cluster"].to_numpy(), args.n_boot, args.seed
        )
        results.append(
            {
                "model": model,
                "method": method,
                "auroc": point,
                "ci_low": low,
                "ci_high": high,
                "n_rows": int(len(grp)),
                "n_clusters": n_clusters,
                "n_boot": args.n_boot,
                "n_boot_used": n_used,
                "seed": args.seed,
                "source": source,
            }
        )
    if not results:
        sys.exit("error: no (model, method) group had two classes")
    table = pd.DataFrame(results)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    print(f"\n# Grouped bootstrap over {GROUP_COL}: B={args.n_boot}, seed={args.seed}, 95% percentile CI")
    print(table.drop(columns=["seed"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n# LaTeX (model & method & AUROC [95% CI]):")
    for _, row in table.iterrows():
        print(format_latex(row))

    exit_code = 0
    if args.check:
        expected = parse_expect(args.expect)
        print("\n# Check: recomputed point AUROC vs. expected (paper) value")
        lines = []
        for _, row in table.iterrows():
            key = (row["model"], row["method"])
            if key in expected:
                exp = expected.pop(key)
                diff = row["auroc"] - exp
                status = "OK" if abs(diff) <= TOL else "MISMATCH"
                if status == "MISMATCH":
                    exit_code = 1
            else:
                exp, diff, status = math.nan, math.nan, "no --expect"
            lines.append(
                {"model": key[0], "method": key[1], "computed": row["auroc"], "expected": exp, "diff": diff, "status": status}
            )
        for (model, method), exp in expected.items():
            exit_code = 1
            lines.append(
                {"model": model, "method": method, "computed": math.nan, "expected": exp, "diff": math.nan, "status": "NOT IN INPUT"}
            )
        print(pd.DataFrame(lines).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        sys.stdout.flush()
        if exit_code:
            print("\ncheck FAILED: at least one point AUROC does not match; do not use these CIs yet.", file=sys.stderr)
        else:
            print("\ncheck passed.")

    sys.stdout.flush()
    print(f"\nwrote {out_path}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
