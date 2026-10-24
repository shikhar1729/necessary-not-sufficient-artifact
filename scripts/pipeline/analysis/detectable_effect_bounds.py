"""Detectable-effect bounds for the small-sample negative results: exact McNemar floors and 80%-power bounds for
paired rate changes, Fisher power for steering, and Hanley-McNeil AUROC-vs-chance thresholds (e.g. the
adaptive-v4 0.704/0.682/0.701). Pure arithmetic from the sample sizes; prints the tables. No data needed.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

Z_ALPHA = 1.959963985  # two-sided alpha = 0.05
Z_BETA = 0.841621234  # 80% power


# --------------------------------------------------------------------------
# exact binomial helpers
# --------------------------------------------------------------------------
def binom_pmf(k: int, n: int, p: float) -> float:
    return math.comb(n, k) * (p**k) * ((1.0 - p) ** (n - k))


def binom_sf_ge(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Bin(n, p)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return sum(binom_pmf(j, n, p) for j in range(k, n + 1))


# --------------------------------------------------------------------------
# 1. McNemar exact: paired binary rate change
# --------------------------------------------------------------------------
def mcnemar_min_discordant(alpha: float = 0.05) -> int:
    """Smallest number of discordant pairs d that can reach two-sided p <= alpha
    under the exact conditional (binomial, p=1/2) test, in the most favourable
    case where every discordant pair moves in the same direction.

    Exact two-sided p for d successes out of d trials = 2 * (1/2)^d.
    Independent of n: n only limits how large d can be.
    """
    d = 1
    while 2.0 * (0.5**d) > alpha:
        d += 1
    return d


def mcnemar_mde(n: int, alpha: float = 0.05, power: float = 0.80) -> dict:
    """Minimum detectable paired rate change for n pairs.

    Most favourable alternative: every pair that changes moves in the
    hypothesised direction, so the discordant count d ~ Bin(n, pi) and the
    observed |Delta| = d / n with E|Delta| = pi.  Reject iff d >= d_min.
    Returns pi at `power`, i.e. a LOWER BOUND on the true MDE (any
    wrong-direction discordance makes it worse).
    """
    d_min = mcnemar_min_discordant(alpha)
    out = {
        "n": n,
        "d_min": d_min,
        "min_significant_delta": d_min / n if d_min <= n else float("nan"),
    }
    if d_min > n:
        out["mde_80"] = float("nan")
        return out
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if binom_sf_ge(d_min, n, mid) >= power:
            hi = mid
        else:
            lo = mid
    out["mde_80"] = hi
    out["power_at_mde"] = binom_sf_ge(d_min, n, hi)
    return out


# --------------------------------------------------------------------------
# 2. Fisher exact: unpaired two proportions, exact power by enumeration
# --------------------------------------------------------------------------
def fisher_reject_table(n1: int, n2: int, alpha: float = 0.05):
    """Boolean table reject[x][y] for Fisher's exact two-sided test.

    x successes of n1, y successes of n2.  Two-sided p = sum of
    hypergeometric probabilities <= observed, conditional on the margin.
    Computed exactly with Fractions, cached per margin.
    """
    reject = [[False] * (n2 + 1) for _ in range(n1 + 1)]
    n = n1 + n2
    for m in range(n + 1):  # total successes
        lo = max(0, m - n2)
        hi = min(n1, m)
        probs = {}
        denom = math.comb(n, m)
        for x in range(lo, hi + 1):
            probs[x] = Fraction(math.comb(n1, x) * math.comb(n2, m - x), denom)
        for x in range(lo, hi + 1):
            obs = probs[x]
            p = sum(v for v in probs.values() if v <= obs)
            if float(p) <= alpha:
                reject[x][m - x] = True
    return reject


def fisher_power(p1: float, p2: float, n1: int, n2: int, reject) -> float:
    b1 = [binom_pmf(x, n1, p1) for x in range(n1 + 1)]
    b2 = [binom_pmf(y, n2, p2) for y in range(n2 + 1)]
    tot = 0.0
    for x in range(n1 + 1):
        if b1[x] == 0.0:
            continue
        row = reject[x]
        s = 0.0
        for y in range(n2 + 1):
            if row[y]:
                s += b2[y]
        tot += b1[x] * s
    return tot


def fisher_mde(p1: float, n1: int, n2: int, direction: int, power: float = 0.80,
               alpha: float = 0.05) -> float:
    """Smallest |p2 - p1| reaching `power` under Fisher's exact test.

    direction=+1 searches upward from p1, -1 downward.  Grid 0.005.
    """
    reject = fisher_reject_table(n1, n2, alpha)
    step = 0.005
    d = step
    while d <= 1.0:
        p2 = p1 + direction * d
        if not (0.0 <= p2 <= 1.0):
            return float("nan")
        if fisher_power(p1, p2, n1, n2, reject) >= power:
            return d
        d += step
    return float("nan")


# --------------------------------------------------------------------------
# 3. AUROC vs chance: Hanley-McNeil
# --------------------------------------------------------------------------
def hanley_mcneil_var(a: float, n_pos: int, n_neg: int) -> float:
    q1 = a / (2.0 - a)
    q2 = 2.0 * a * a / (1.0 + a)
    return (a * (1 - a) + (n_pos - 1) * (q1 - a * a) + (n_neg - 1) * (q2 - a * a)) / (
        n_pos * n_neg
    )


def hanley_mcneil_se(a: float, n_pos: int, n_neg: int) -> float:
    return math.sqrt(max(hanley_mcneil_var(a, n_pos, n_neg), 0.0))


def auroc_mde(n_pos: int, n_neg: int) -> float:
    """Smallest AUROC > 0.5 detectable as different from 0.5 at two-sided
    alpha=0.05 and 80% power, using the Hanley-McNeil SE under the alternative
    for the power term and under the null for the size term.
    """
    se0 = hanley_mcneil_se(0.5, n_pos, n_neg)
    a = 0.5
    for _ in range(500):
        se1 = hanley_mcneil_se(a, n_pos, n_neg)
        a_new = 0.5 + Z_ALPHA * se0 + Z_BETA * se1
        if abs(a_new - a) < 1e-10:
            a = a_new
            break
        a = a_new
    return min(a, 1.0)


def auroc_ci_halfwidth(n_pos: int, n_neg: int, a: float = 0.5) -> float:
    return Z_ALPHA * hanley_mcneil_se(a, n_pos, n_neg)


# --------------------------------------------------------------------------
# 4. Cluster bootstrap on a tiny minority class (null scores)
# --------------------------------------------------------------------------
def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney with ties handled by midranks
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    allv = np.concatenate([pos, neg])[order]
    ranks = np.empty(len(allv), dtype=float)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[i : j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    inv = np.empty(len(allv), dtype=int)
    inv[order] = np.arange(len(allv))
    r_pos = ranks[inv[: len(pos)]]
    return (r_pos.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def cluster_bootstrap_null_width(n_pos: int, n_neg: int, b: int = 4000,
                                 seed: int = 0) -> dict:
    """95% percentile-interval width of AUROC under exchangeable (null) scores,
    resampling CLUSTERS with replacement, one row per cluster (the regime of the
    prior-aware slices, where rows come from distinct questions).  Clusters that
    share a question would only widen this.
    """
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    vals = []
    undef = 0
    for _ in range(b):
        scores = rng.random(n)  # null: scores independent of label
        idx = rng.integers(0, n, size=n)
        a = auroc(scores[idx], labels[idx])
        if math.isnan(a):
            undef += 1
        else:
            vals.append(a)
    v = np.asarray(vals)
    lo, hi = np.percentile(v, [2.5, 97.5])
    return {
        "lo": float(lo),
        "hi": float(hi),
        "width": float(hi - lo),
        "undefined_frac": undef / b,
        "b": b,
    }


# --------------------------------------------------------------------------
def main() -> None:
    print("=" * 78)
    print("1. PAIRED BINARY RATE CHANGE (McNemar exact, two-sided alpha=0.05)")
    print("=" * 78)
    print(f"minimum discordant pairs for significance: d_min = "
          f"{mcnemar_min_discordant()}  (independent of n)")
    for label, n, n_cells in [
        ("adaptive-v4 paired activation patching", 12, 12),
        ("direction-only patching", 15, 32),
        ("component patching", 16, 16),
        ("matched-control steering (strong-alpha surviving pairs)", 28, 10),
        ("matched-control steering (full holdout)", 40, 10),
    ]:
        r = mcnemar_mde(n)
        adj = mcnemar_mde(n, alpha=0.05 / n_cells)
        print(f"  n={n:3d}  min significant |D| = {r['min_significant_delta']:.3f} "
              f"({r['d_min']}/{n})   MDE@80% = {r['mde_80']:.3f}   | "
              f"max over {n_cells:2d} cells (Bonferroni): d_min={adj['d_min']}, "
              f"min sig |D| = {adj['min_significant_delta']:.3f}   ({label})")

    print()
    print("=" * 78)
    print("2. UNPAIRED TWO PROPORTIONS (Fisher exact, two-sided alpha=0.05)")
    print("=" * 78)
    for label, p1, n in [
        ("steering, dependence rate (base ~0.50), n=40/arm", 0.50, 40),
        ("steering, dependence rate (base ~0.50), n=28/arm", 0.50, 28),
        ("steering, W->h rate (base ~0.87), n=40/arm", 0.87, 40),
    ]:
        up = fisher_mde(p1, n, n, +1)
        dn = fisher_mde(p1, n, n, -1)
        print(f"  {label}: MDE@80% = +{up:.3f} / -{dn:.3f}")

    print()
    print("=" * 78)
    print("3. AUROC vs CHANCE (Hanley-McNeil, two-sided alpha=0.05, 80% power)")
    print("=" * 78)
    rows = [
        ("Injection  Mistral", 21, 57),
        ("Injection  Gemma", 22, 54),
        ("Injection  Llama", 19, 26),
        ("Injection  Qwen3-14B", 15, 47),
        ("Adaptive-v4 Mistral", 30, 30),
        ("Adaptive-v4 Gemma", 38, 38),
        ("Adaptive-v4 Llama", 31, 31),
        ("Qwen2.5-7B", 21, 14),
        ("Qwen3-14B (main task)", 36, 18),
        ("Qwen3-32B", 23, 19),
        ("QwQ-32B", 22, 13),
        ("DeepSeek-R1-Distill-8B", 37, 6),
    ]
    for label, npos, nneg in rows:
        mde = auroc_mde(npos, nneg)
        hw = auroc_ci_halfwidth(npos, nneg)
        print(f"  {label:26s} n={npos+nneg:3d} ({npos}/{nneg})  "
              f"MDE AUROC = {mde:.3f}  (|A-0.5| >= {mde-0.5:.3f})   "
              f"null 95% band = [{0.5-hw:.3f}, {0.5+hw:.3f}]")

    print()
    print("=" * 78)
    print("4. AUROC INTERVAL WIDTH ON TINY MINORITY CLASSES")
    print("=" * 78)
    rows = [
        ("prior-aware v2 Mistral", 649, 20),
        ("prior-aware v2 Gemma", 374, 15),
        ("prior-aware v2 Llama", 191, 11),
        ("balanced v3 Mistral", 118, 5),
        ("balanced v3 Gemma", 104, 6),
        ("balanced v3 Llama", 44, 4),
    ]
    for label, npos, nneg in rows:
        hw = auroc_ci_halfwidth(npos, nneg)
        cb = cluster_bootstrap_null_width(npos, nneg)
        print(f"  {label:24s} ({npos}/{nneg})  HM half-width = +/-{hw:.3f} "
              f"(width {2*hw:.3f})   cluster-boot null 95% = "
              f"[{cb['lo']:.3f}, {cb['hi']:.3f}] width {cb['width']:.3f}  "
              f"undef {cb['undefined_frac']*100:.1f}%")


if __name__ == "__main__":
    main()
