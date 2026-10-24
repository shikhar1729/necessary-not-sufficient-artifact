"""Probe plus text-monitor ensembles (app:ensemble): does pooling the probe with a judge beat both members?
Input: per-row scores in data/scores/ (probe OOF, note-aware, CoT-CF paraphrases, candidate prior, TF-IDF).
Output (outputs/ensemble/), released in data/scores/ensemble/: ensemble_per_row_scores.csv as per_row_scores.csv,
stack_lr_fold_weights.csv as stack_weights.csv, probe_increment_over_text.csv, best_with_probe_vs_best_text_only.csv
as best_with_vs_without_probe.csv. Paired row bootstrap B=2000, seed 0, as in monitor_intervals.py. No GPU, no API.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.pipeline.analysis.monitor_intervals import auroc, ci, resamples

ROOT = Path(__file__).resolve().parents[3]
SCORES = ROOT / "data" / "scores"
OUT = ROOT / "outputs" / "ensemble"

MODELS = ["mistral", "gemma", "llama"]
B = 2000
SEED = 0
N_SPLITS = 5


# ------------------------------------------------------------------ loading

MEMBER_LABEL = {
    "probe": "Probe (nested, best layer; OOF)",
    "na_cot": "Sonnet-4 note-aware CoT (no h')",
    "cf_para": "Sonnet-4 CoT-CF 3-paraphrase mean",
    "prior": "Sonnet-4 candidate-prior (no model output)",
    "tfidf": "TF-IDF best view per model (OOF)",
}


def load_members() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Per-model wide frames, one row per group, one column per member."""
    probe = pd.read_csv(SCORES / "probes" / "nested_probe_oof.csv")
    probe = probe.rename(columns={"oof_prob": "probe"})

    na = pd.read_csv(SCORES / "monitors" / "note_aware_judge_rows.csv.gz")
    na = na[na["parse_ok"].astype(bool)].copy()
    na["group"] = na["dataset"].astype(str) + "::" + na["question_id"].astype(str)
    na_cot = na[na["view"] == "note_aware_no_hprime"][["model", "group", "prob_dependent"]]
    na_cot = na_cot.rename(columns={"prob_dependent": "na_cot"})

    j19 = pd.read_csv(SCORES / "monitors" / "judge_robustness_rows.csv.gz")
    j19 = j19[j19["parse_ok"].astype(bool)]
    para = j19[j19["cell"].isin(["sonnet4_cot_cf", "sonnet4_cot_cf_p1", "sonnet4_cot_cf_p2"])]
    para = para.pivot_table(index=["model", "group"], columns="cell", values="prob_dependent").dropna()
    cf_para = para.mean(axis=1).rename("cf_para").reset_index()

    nj = pd.read_csv(SCORES / "monitors" / "prior_and_answer_cf_judge_rows.csv.gz")
    nj = nj[nj["parse_ok"].astype(bool)].copy()
    nj["group"] = nj["dataset"].astype(str) + "::" + nj["question_id"].astype(str)
    prior = nj[nj["view"] == "candidate_prior"][["model", "group", "prob_dependent"]]
    prior = prior.rename(columns={"prob_dependent": "prior"})

    # TF-IDF: the same "best view per model, selected on the outcome" cell as monitor_intervals.py
    tf = pd.read_csv(SCORES / "monitors" / "tfidf_oof.csv")
    best_view: dict[str, str] = {}
    tf_rows = []
    for m in MODELS:
        sub = tf[tf["model"] == m]
        scores = {v: auroc(g["label"].to_numpy(int), g["score"].to_numpy(float)) for v, g in sub.groupby("view")}
        bv = max(scores, key=lambda v: scores[v])
        best_view[m] = bv
        tf_rows.append(sub[sub["view"] == bv][["model", "group", "score"]].rename(columns={"score": "tfidf"}))
    tfidf = pd.concat(tf_rows, ignore_index=True)

    frames: dict[str, pd.DataFrame] = {}
    for m in MODELS:
        d = probe[probe["model"] == m][["group", "label", "probe"]].copy()
        for other in (na_cot, cf_para, prior, tfidf):
            d = d.merge(other[other["model"] == m].drop(columns="model"), on="group", how="left")
        frames[m] = d.reset_index(drop=True)
    return frames, best_view


def fold_ids(groups: np.ndarray) -> np.ndarray:
    """Reconstruct the probe's GroupKFold(5) assignment; assert order independence."""
    def assign(g: np.ndarray) -> dict[str, int]:
        out: dict[str, int] = {}
        splitter = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(g))))
        dummy = np.zeros((len(g), 1))
        for k, (_, test_idx) in enumerate(splitter.split(dummy, np.zeros(len(g)), g)):
            for i in test_idx:
                out[g[i]] = k
        return out

    a = assign(groups)
    shuffled = groups.copy()
    np.random.default_rng(123).shuffle(shuffled)
    b = assign(shuffled)
    assert a == b, "GroupKFold assignment is not order-independent; folds cannot be reconstructed"
    return np.array([a[g] for g in groups], dtype=int)


# ------------------------------------------------------------------ combiners


def _rank01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        r[order[i : j + 1]] = 0.5 * (i + j)
        i = j + 1
    return r / max(len(x) - 1, 1)


def rank_avg_global(X: np.ndarray) -> np.ndarray:
    return np.mean([_rank01(X[:, j]) for j in range(X.shape[1])], axis=0)


def z_avg_global(X: np.ndarray) -> np.ndarray:
    cols = []
    for j in range(X.shape[1]):
        c = X[:, j]
        sd = c.std()
        cols.append((c - c.mean()) / (sd if sd > 0 else 1.0))
    return np.mean(cols, axis=0)


def z_avg_foldfit(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Standardize with training-fold mean/sd only, then average."""
    out = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        cols = []
        for j in range(X.shape[1]):
            mu, sd = X[tr, j].mean(), X[tr, j].std()
            cols.append((X[te, j] - mu) / (sd if sd > 0 else 1.0))
        out[te] = np.mean(cols, axis=0)
    return out


def rank_avg_foldfit(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """Map test scores through the training-fold empirical CDF, then average."""
    out = np.full(len(y), np.nan)
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        cols = []
        for j in range(X.shape[1]):
            ref = np.sort(X[tr, j])
            lo = np.searchsorted(ref, X[te, j], side="left")
            hi = np.searchsorted(ref, X[te, j], side="right")
            cols.append(0.5 * (lo + hi) / max(len(ref), 1))
        out[te] = np.mean(cols, axis=0)
    return out


def stack_lr(X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Logistic meta-learner: weights fit inside training folds only."""
    out = np.full(len(y), np.nan)
    coefs = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if len(np.unique(y[tr])) < 2:
            continue
        clf = Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=SEED)),
            ]
        )
        clf.fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
        coefs.append({"fold": int(k), **{f"w{j}": float(w) for j, w in enumerate(clf[-1].coef_[0])}})
    return out, coefs


# ------------------------------------------------------------------ pairings

PAIRINGS: list[tuple[str, list[str]]] = [
    ("probe+note_aware_cot", ["probe", "na_cot"]),
    ("probe+cot_cf_paraphrase", ["probe", "cf_para"]),
    ("probe+candidate_prior", ["probe", "prior"]),
    ("probe+tfidf_best", ["probe", "tfidf"]),
    ("probe+note_aware_cot+candidate_prior", ["probe", "na_cot", "prior"]),
    # text-only control: do two text monitors combine any better than probe+text?
    ("note_aware_cot+candidate_prior", ["na_cot", "prior"]),
]


# -------------------------------------------- incremental value of the probe

# Combining any two noisy monitors helps.  So the decisive question is not
# "does probe+text beat each member" but "does ADDING the probe to a text stack
# help more than adding another text monitor would".  Each base below is scored
# with and without the probe on identical rows, with a paired bootstrap CI and a
# permutation null in which the probe column is shuffled across rows (keeping its
# marginal distribution, destroying its link to the label).
TEXT_BASES: list[tuple[str, list[str]]] = [
    ("na_cot", ["na_cot"]),
    ("prior", ["prior"]),
    ("cf_para", ["cf_para"]),
    ("tfidf", ["tfidf"]),
    ("na_cot+prior", ["na_cot", "prior"]),
    ("na_cot+cf_para", ["na_cot", "cf_para"]),
    ("prior+cf_para", ["prior", "cf_para"]),
    ("na_cot+prior+cf_para", ["na_cot", "prior", "cf_para"]),
    ("na_cot+prior+cf_para+tfidf", ["na_cot", "prior", "cf_para", "tfidf"]),
]
R_PERM = 1000
R_PERM_STACK = 300


def combine(name: str, X: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    if name == "rank_avg":
        return rank_avg_global(X)
    if name == "z_avg":
        return z_avg_global(X)
    if name == "stack_lr":
        return stack_lr(X, y, folds)[0]
    raise ValueError(name)


def increment_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        df = frames[model]
        for base_name, base in TEXT_BASES:
            sub = df[["group", "label", "probe", *base]].dropna().reset_index(drop=True)
            if len(sub) < 20:
                continue
            y = sub["label"].to_numpy(int)
            groups = sub["group"].to_numpy(dtype=object)
            folds = fold_ids(groups)
            Xb = sub[base].to_numpy(float)
            probe = sub["probe"].to_numpy(float)
            Xp = np.column_stack([probe, Xb])
            idx = resamples(len(y))

            for comb in ["rank_avg", "z_avg", "stack_lr"]:
                s_base = combine(comb, Xb, y, folds) if len(base) > 1 else Xb[:, 0]
                base_is_raw = len(base) == 1
                s_with = combine(comb, Xp, y, folds)
                a_base, a_with = auroc(y, s_base), auroc(y, s_with)
                deltas = []
                for i in idx:
                    va, vb = auroc(y[i], s_with[i]), auroc(y[i], s_base[i])
                    if np.isfinite(va) and np.isfinite(vb):
                        deltas.append(va - vb)
                lo, hi, _ = ci(deltas)
                arr = np.asarray(deltas)

                # permutation null: shuffle the probe column only
                r_perm = R_PERM_STACK if comb == "stack_lr" else R_PERM
                rng = np.random.default_rng(SEED + 7)
                null = []
                for _ in range(r_perm):
                    pp = rng.permutation(probe)
                    s_null = combine(comb, np.column_stack([pp, Xb]), y, folds)
                    null.append(auroc(y, s_null) - a_base)
                null_arr = np.asarray(null)
                obs = a_with - a_base
                rows.append(
                    {
                        "model": model,
                        "text_base": base_name,
                        "base_size": len(base),
                        "combiner": comb,
                        "base_is_raw_single_monitor": base_is_raw,
                        "n": len(y),
                        "auroc_base": a_base,
                        "auroc_base_plus_probe": a_with,
                        "delta": obs,
                        "ci_low": lo,
                        "ci_high": hi,
                        "includes_zero": bool(lo <= 0.0 <= hi),
                        "p_two_sided_boot": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
                        "perm_null_mean_delta": float(null_arr.mean()),
                        "perm_null_p95_delta": float(np.percentile(null_arr, 95)),
                        "perm_p_one_sided": float((null_arr >= obs).mean()),
                        "r_perm": r_perm,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frames, best_view = load_members()
    print("TF-IDF best view per model:", best_view)

    auroc_rows: list[dict] = []
    paired_rows: list[dict] = []
    corr_rows: list[dict] = []
    coef_rows: list[dict] = []
    score_rows: list[pd.DataFrame] = []

    for model in MODELS:
        df = frames[model]
        print(f"\n===== {model} (N={len(df)}) =====")
        print("member coverage:", {c: int(df[c].notna().sum()) for c in MEMBER_LABEL})

        # ---- correlation structure, on each pair's own common rows
        for a, b in itertools.combinations(MEMBER_LABEL, 2):
            sub = df[[a, b, "label"]].dropna()
            if len(sub) < 20:
                continue
            x, z = sub[a].to_numpy(float), sub[b].to_numpy(float)
            sp = spearmanr(x, z)
            pe = pearsonr(x, z)
            n = len(sub)
            k = max(int(round(0.1 * n)), 1)
            rx, rz = _rank01(x), _rank01(z)
            top_x = set(np.argsort(-rx, kind="mergesort")[:k])
            top_z = set(np.argsort(-rz, kind="mergesort")[:k])
            bot_x = set(np.argsort(rx, kind="mergesort")[:k])
            bot_z = set(np.argsort(rz, kind="mergesort")[:k])
            corr_rows.append(
                {
                    "model": model,
                    "a": a,
                    "b": b,
                    "n": n,
                    "spearman": float(sp.statistic),
                    "spearman_p": float(sp.pvalue),
                    "pearson": float(pe.statistic),
                    "pearson_p": float(pe.pvalue),
                    "decile_k": k,
                    "top_decile_overlap": len(top_x & top_z),
                    "top_decile_jaccard": len(top_x & top_z) / len(top_x | top_z),
                    "top_decile_expected_by_chance": k * k / n,
                    "bottom_decile_overlap": len(bot_x & bot_z),
                    "bottom_decile_jaccard": len(bot_x & bot_z) / len(bot_x | bot_z),
                    "auroc_a": auroc(sub["label"].to_numpy(int), x),
                    "auroc_b": auroc(sub["label"].to_numpy(int), z),
                }
            )

        # ---- ensembles
        for pair_name, members in PAIRINGS:
            sub = df[["group", "label", *members]].dropna().reset_index(drop=True)
            if len(sub) < 20:
                print(f"  skip {pair_name}: n={len(sub)}")
                continue
            y = sub["label"].to_numpy(int)
            groups = sub["group"].to_numpy(dtype=object)
            folds = fold_ids(groups)
            X = sub[members].to_numpy(float)
            idx = resamples(len(y))

            combos: dict[str, np.ndarray] = {
                "rank_avg": rank_avg_global(X),
                "z_avg": z_avg_global(X),
                "rank_avg_foldfit": rank_avg_foldfit(X, y, folds),
                "z_avg_foldfit": z_avg_foldfit(X, y, folds),
            }
            st, coefs = stack_lr(X, y, folds)
            combos["stack_lr"] = st
            for c in coefs:
                coef_rows.append({"model": model, "pairing": pair_name, "members": "|".join(members), **c})

            # members as their own "combos" so they get the same CI treatment
            member_scores = {f"member::{mm}": X[:, j] for j, mm in enumerate(members)}

            out_scores = sub[["group", "label"]].copy()
            out_scores["model"] = model
            out_scores["pairing"] = pair_name
            for mm, s in {**member_scores, **combos}.items():
                out_scores[mm.replace("member::", "m_")] = s
            score_rows.append(out_scores)

            for name, s in {**member_scores, **combos}.items():
                ok = np.isfinite(s)
                if ok.sum() != len(s):
                    print(f"  WARNING {pair_name}/{name}: {len(s)-ok.sum()} non-finite scores")
                point = auroc(y, s)
                boot = [auroc(y[i], s[i]) for i in idx]
                lo, hi, nb = ci(boot)
                auroc_rows.append(
                    {
                        "model": model,
                        "pairing": pair_name,
                        "members": "|".join(members),
                        "scorer": name,
                        "kind": "member" if name.startswith("member::") else "ensemble",
                        "n": len(y),
                        "n_dep": int(y.sum()),
                        "auroc": point,
                        "ci_low": lo,
                        "ci_high": hi,
                        "ci_half_width": (hi - lo) / 2,
                        "n_boot_valid": nb,
                    }
                )

            # ---- paired differences: each ensemble against each member
            for ens_name, es in combos.items():
                for j, mm in enumerate(members):
                    ms = X[:, j]
                    pa, pb = auroc(y, es), auroc(y, ms)
                    deltas = []
                    for i in idx:
                        va, vb = auroc(y[i], es[i]), auroc(y[i], ms[i])
                        if np.isfinite(va) and np.isfinite(vb):
                            deltas.append(va - vb)
                    lo, hi, nb = ci(deltas)
                    arr = np.asarray(deltas)
                    paired_rows.append(
                        {
                            "model": model,
                            "pairing": pair_name,
                            "ensemble": ens_name,
                            "member": mm,
                            "n": len(y),
                            "auroc_ensemble": pa,
                            "auroc_member": pb,
                            "delta": pa - pb,
                            "ci_low": lo,
                            "ci_high": hi,
                            "half_width": (hi - lo) / 2,
                            "includes_zero": bool(lo <= 0.0 <= hi),
                            "p_two_sided_boot": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
                            "is_best_member": bool(pb == max(auroc(y, X[:, k]) for k in range(X.shape[1]))),
                            "n_boot_valid": nb,
                        }
                    )

    a_tbl = pd.DataFrame(auroc_rows)
    p_tbl = pd.DataFrame(paired_rows)
    c_tbl = pd.DataFrame(corr_rows)
    a_tbl.to_csv(OUT / "ensemble_auroc_ci.csv", index=False)
    p_tbl.to_csv(OUT / "ensemble_paired_vs_members.csv", index=False)
    c_tbl.to_csv(OUT / "monitor_pair_correlations.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(OUT / "stack_lr_fold_weights.csv", index=False)
    pd.concat(score_rows, ignore_index=True).to_csv(OUT / "ensemble_per_row_scores.csv", index=False)
    (OUT / "meta.json").write_text(
        json.dumps({"B": B, "seed": SEED, "n_splits": N_SPLITS, "tfidf_best_view": best_view}, indent=2)
    )

    # ------------------------------------------------------------ console report
    print("\n\n================ AUROC: members vs ensembles ================")
    for model in MODELS:
        print(f"\n-- {model} --")
        for pair_name, _ in PAIRINGS:
            sub = a_tbl[(a_tbl["model"] == model) & (a_tbl["pairing"] == pair_name)]
            if sub.empty:
                continue
            print(f"  {pair_name}  (n={int(sub['n'].iloc[0])})")
            for r in sub.itertuples():
                tag = "MEMBER " if r.kind == "member" else "ens    "
                print(
                    f"    {tag}{r.scorer.replace('member::',''):18s} {r.auroc:.4f} "
                    f"[{r.ci_low:.4f}, {r.ci_high:.4f}]  hw={r.ci_half_width:.4f}"
                )

    print("\n\n======== paired: ensemble minus member (same rows, same resamples) ========")
    for r in p_tbl.sort_values(["model", "pairing", "ensemble", "member"]).itertuples():
        star = "" if r.includes_zero else "  *** EXCLUDES ZERO ***"
        best = " [best member]" if r.is_best_member else ""
        print(
            f"  {r.model:8s} {r.pairing:38s} {r.ensemble:16s} - {r.member:8s} "
            f"{r.auroc_ensemble:.4f}-{r.auroc_member:.4f}={r.delta:+.4f} "
            f"[{r.ci_low:+.4f},{r.ci_high:+.4f}] p={r.p_two_sided_boot:.3f}{best}{star}"
        )

    vs_best = p_tbl[p_tbl["is_best_member"]]
    print("\n\n======== HEADLINE: ensemble minus its BEST member ========")
    for r in vs_best.sort_values("delta", ascending=False).itertuples():
        star = "" if r.includes_zero else "  *** EXCLUDES ZERO ***"
        print(
            f"  {r.model:8s} {r.pairing:38s} {r.ensemble:16s} vs {r.member:8s} "
            f"delta={r.delta:+.4f} [{r.ci_low:+.4f},{r.ci_high:+.4f}]{star}"
        )
    n_sig = int((~vs_best["includes_zero"]).sum())
    print(f"\n  ensembles beating their best member with an interval excluding zero: {n_sig} / {len(vs_best)}")
    n_sig_all = int((~p_tbl["includes_zero"]).sum())
    print(f"  all ensemble-vs-member comparisons excluding zero: {n_sig_all} / {len(p_tbl)}")
    print(f"  median paired half-width (vs best member): {vs_best['half_width'].median():.4f}")
    print(f"  median paired half-width (all comparisons): {p_tbl['half_width'].median():.4f}")

    print("\n\n======== correlation structure ========")
    for r in c_tbl.sort_values(["model", "a", "b"]).itertuples():
        print(
            f"  {r.model:8s} {r.a:8s}/{r.b:8s} n={r.n:4d} spearman={r.spearman:+.3f} "
            f"pearson={r.pearson:+.3f} topdec={r.top_decile_overlap:3d}/{r.decile_k} "
            f"(chance {r.top_decile_expected_by_chance:.1f}) botdec={r.bottom_decile_overlap:3d}/{r.decile_k}"
        )

    # ---------------------------------------- incremental value of the probe
    print("\n\n======== INCREMENTAL: text base, then the same base + probe ========")
    print("(permutation null shuffles the probe column; perm_p is one-sided)")
    inc = increment_table(frames)
    inc.to_csv(OUT / "probe_increment_over_text.csv", index=False)
    for r in inc.sort_values(["model", "combiner", "base_size", "text_base"]).itertuples():
        star = "" if r.includes_zero else "  *** CI EXCLUDES ZERO ***"
        perm = "  PERM-SIG" if r.perm_p_one_sided < 0.05 else ""
        print(
            f"  {r.model:8s} {r.combiner:9s} {r.text_base:28s} "
            f"{r.auroc_base:.4f} -> {r.auroc_base_plus_probe:.4f}  d={r.delta:+.4f} "
            f"[{r.ci_low:+.4f},{r.ci_high:+.4f}] perm_p={r.perm_p_one_sided:.3f} "
            f"(null mean {r.perm_null_mean_delta:+.4f}, p95 {r.perm_null_p95_delta:+.4f})"
            f"{star}{perm}"
        )

    print("\n-- summary of the incremental test --")
    for comb in ["rank_avg", "z_avg", "stack_lr"]:
        sub = inc[inc["combiner"] == comb]
        print(
            f"  {comb:9s} CI excludes zero: {int((~sub['includes_zero']).sum())}/{len(sub)}   "
            f"perm p<0.05: {int((sub['perm_p_one_sided'] < 0.05).sum())}/{len(sub)}   "
            f"median delta {sub['delta'].median():+.4f}"
        )
    multi = inc[inc["base_size"] >= 2]
    print(
        f"  multi-monitor text bases only (base_size>=2): CI excludes zero "
        f"{int((~multi['includes_zero']).sum())}/{len(multi)}, perm p<0.05 "
        f"{int((multi['perm_p_one_sided'] < 0.05).sum())}/{len(multi)}"
    )

    print("\n-- best probe-including vs best text-only ensemble, per model --")
    bests = []
    for model in MODELS:
        sub = inc[inc["model"] == model]
        best_with = sub.loc[sub["auroc_base_plus_probe"].idxmax()]
        best_text = sub.loc[sub["auroc_base"].idxmax()]
        bests.append(
            {
                "model": model,
                "best_with_probe": f"{best_with['combiner']}({best_with['text_base']}+probe)",
                "auroc_with_probe": best_with["auroc_base_plus_probe"],
                "best_text_only": f"{best_text['combiner']}({best_text['text_base']})",
                "auroc_text_only": best_text["auroc_base"],
                "gap": best_with["auroc_base_plus_probe"] - best_text["auroc_base"],
            }
        )
        print(
            f"  {model:8s} with-probe {best_with['auroc_base_plus_probe']:.4f} "
            f"({best_with['combiner']}, {best_with['text_base']}+probe)  vs  "
            f"text-only {best_text['auroc_base']:.4f} ({best_text['combiner']}, {best_text['text_base']})  "
            f"gap {best_with['auroc_base_plus_probe'] - best_text['auroc_base']:+.4f}"
        )
    pd.DataFrame(bests).to_csv(OUT / "best_with_probe_vs_best_text_only.csv", index=False)

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
