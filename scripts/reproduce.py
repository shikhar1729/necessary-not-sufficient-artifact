"""Recompute the paper's numbers from data/ (CPU only, no network, about one minute).

    python scripts/reproduce.py        # works from any directory

Each check names where the numbers are printed (a LaTeX label such as tab:monitor-intervals; search
the paper source for it), the values printed in the paper, and the values recomputed from data/.
A value passes when it rounds to the printed value at the printed number of decimals. A few yes/no
checks test statements the paper makes in words (for example, which cell is the exception). One
precondition comes first: GroupKFold(5), rebuilt here, must give the stored fold column, which needs
scikit-learn < 1.9. The script exits with status 1 if a check or the precondition fails.

Intervals are 95% percentile bootstrap intervals with B = 2000 draws from numpy's default_rng(0),
started afresh for each interval, as in the scripts that produced them. The exception is the
span-label block (app:corrected, tab:span-relabel): one default_rng(1729) stream, used in a fixed
order, gives the parser-precision intervals (B = 4000) and then the probe intervals on the surviving
rows (B = 2000). Monitor and ensemble intervals resample rows, probe intervals resample question
clusters. The paper calls both question-cluster intervals; they are equivalent in distribution here
because every (dataset, question) cluster holds one row (the draws differ).
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, norm, rankdata, spearmanr
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.normalize import answers_match  # noqa: E402

S = ROOT / "data" / "scores"
EXAMPLES = ROOT / "data" / "examples"
M = ["mistral", "gemma", "llama"]
RESULTS: list[tuple[str, bool, int]] = []  # (kind, passed, printed numbers compared); kind is "number" or "yes/no"
PRECONDITIONS: list[bool] = []


def check(where, what, paper, got):
    """Pass when every recomputed value rounds to the printed one at its printed decimals."""
    toks = paper.split()
    want, got = np.array([float(t) for t in toks]), np.asarray(got, float).ravel()
    if want.shape != got.shape:
        RESULTS.append(("number", False, len(toks)))
        return print(f"[FAIL] {where} | {what}: expected {want.size} values, got {got.size}")
    dps = [len(t.split(".")[1]) if "." in t else 0 for t in toks]
    ok = all(abs(g - w) <= 0.5 * 10.0**-d + 1e-9 for g, w, d in zip(got, want, dps))
    RESULTS.append(("number", ok, len(toks)))
    print(f"[{'PASS' if ok else 'FAIL'}] {where} | {what}\n    paper: {' '.join(toks)}\n    data:  "
          + " ".join(f"{g:.{d + 1}f}" for g, d in zip(got, dps)))


def statement(where, what, ok):
    """A yes/no check of a statement the paper makes in words (for example, which cell is the exception)."""
    RESULTS.append(("yes/no", bool(ok), 0))
    print(f"[{'PASS' if ok else 'FAIL'}] {where} | {what}: {'yes' if ok else 'no'}")


def precondition(what, ok, detail):
    """A setup condition, not a printed number; a failure makes every later fold-based result suspect."""
    PRECONDITIONS.append(bool(ok))
    print(f"[{'OK' if ok else 'PRECONDITION FAILED'}] {what}" + ("" if ok else f"\n    {detail}"))


def table(text):  # "key v1 v2 ..." lines -> {key: "v1 v2 ..."}
    return dict(line.split(None, 1) for line in text.strip().splitlines())


def read(rel):
    return pd.read_csv(S / rel, dtype={"question_id": str}, low_memory=False)


# ------------------------------------------------------------- AUROC, bootstrap and tests
def auc_rows(Y, X):
    """AUROC of each row (ties get average ranks); NaN where a row has one class."""
    r, n1 = rankdata(X, axis=1), Y.sum(1)
    n0 = Y.shape[1] - n1
    with np.errstate(all="ignore"):
        a = ((r * Y).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.where((n1 > 0) & (n0 > 0), a, np.nan)


def auc(y, s):
    return float(auc_rows(np.asarray(y, int)[None], np.asarray(s, float)[None])[0])


def ci(v):  # 95% percentile interval over the finite draws
    v = np.asarray(v, float)
    return tuple(np.percentile(v[np.isfinite(v)], [2.5, 97.5]))


def row_draws(n):  # row resampling: monitors and ensembles
    return np.random.default_rng(0).integers(0, n, size=(2000, n))


def cluster_draws(groups, rng=None):
    """Question-cluster resampling: probes, injection, W/G transfer. Clusters are drawn from the
    sorted cluster ids; every cluster holds one row, so each draw maps to one row."""
    g = np.asarray(groups).astype(str)
    assert len(set(g)) == len(g), "expected one row per cluster"
    rng = np.random.default_rng(0) if rng is None else rng
    order = np.argsort(g)
    return np.stack([order[rng.choice(len(g), size=len(g))] for _ in range(2000)])


def boot(y, s, draws):  # AUROC on every resample
    return auc_rows(np.asarray(y, int)[draws], np.asarray(s, float)[draws])


def auc_ci(y, s, draws):
    return (auc(y, s), *ci(boot(y, s, draws)))


def delta_draws(y, a, b, draws):  # AUROC(a) - AUROC(b) on the same resamples, finite draws only
    d = boot(y, a, draws) - boot(y, b, draws)
    return d[np.isfinite(d)]


def paired_auc(y, a, b, draws):
    """AUROC(a) - AUROC(b) on identical rows and resamples: (delta, CI low, CI high, two-sided p)."""
    d = delta_draws(y, a, b, draws)
    return (auc(y, a) - auc(y, b), *ci(d), 2 * min((d <= 0).mean(), (d >= 0).mean()))


def ece(y, p):  # expected calibration error, 10 equal-width bins
    e = np.linspace(0, 1, 11)
    bins = [(p >= e[i]) & ((p < e[i + 1]) if i < 9 else (p <= e[i + 1])) for i in range(10)]
    return sum(b.sum() / len(y) * abs(p[b].mean() - y[b].mean()) for b in bins if b.any())


def holm_rejections(p, alpha=0.05):
    p, k = np.sort(p), 0
    while k < len(p) and p[k] <= alpha / (len(p) - k):
        k += 1
    return k


def mcnemar(b, c):  # exact two-sided McNemar p from the discordant counts
    return 1.0 if b + c == 0 else min(1.0, 2.0 * sum(comb(b + c, i) for i in range(min(b, c) + 1)) / 2.0 ** (b + c))


def power80(n, d):  # smallest rate pi with P(Bin(n, pi) >= d) >= 0.8
    lo, hi = 0.0, 1.0
    for _ in range(60):
        lo, hi = (lo, (lo + hi) / 2) if binom.sf(d - 1, n, (lo + hi) / 2) >= 0.8 else ((lo + hi) / 2, hi)
    return hi


def group_folds(d):  # GroupKFold(5) fold of every row, rebuilt
    f = np.full(len(d), -1)
    for k, (_, test) in enumerate(GroupKFold(5).split(np.zeros((len(d), 1)), d.label, d.group)):
        f[test] = k
    return f


# ------------------------------------------------------------- 1. retained examples (sec:methodology)
def slices(ex):
    check("tab:corrected-main", "N and dependent examples", "421 384 260 340 291 184",
          [len(ex[m]) for m in M] + [ex[m].label.sum() for m in M])
    check("tab:dataset-slices", "examples per dataset, TriviaQA/HotpotQA/GPQA/MMLU-Pro",
          "120 179 41 81 98 150 67 69 57 110 45 48",
          [(ex[m].dataset == d).sum() for m in M for d in ["triviaqa", "hotpotqa", "gpqa", "mmlu_pro"]])
    shares = []
    for m in M:
        d = ex[m][ex[m].label == 1]
        hp = np.array([answers_match(str(a), str(h)) for a, h in zip(d.final_answer_G, d.h2)])
        co = np.array([answers_match(str(a), str(c)) for a, c in zip(d.final_answer_G, d.correct_answer)])
        shares += [100 * hp.mean(), 100 * (co & ~hp).mean(), 100 * (~co & ~hp).mean()]
    check("sec:methodology", "dependent examples under G: % giving h', the correct answer, another answer",
          "54 9 38 46 20 35 40 26 34", shares)
    f = pd.read_csv(EXAMPLES / "w_trace_funnel.csv").set_index("model").loc[M]
    check("fig:task-funnel", "W traces per model; parse failures + other errors + resists + hint-follows; retained",
          "1728 1728 1728 1728 1728 1728 421 384 260",
          [*f.w_traces, *f[["parse_fail", "other_error", "resist", "hint_follow"]].sum(axis=1), *f.retained_same_answer])
    moved = [int((group_folds(ex[m]) != ex[m].fold.to_numpy()).sum()) for m in M]
    precondition("GroupKFold(5), rebuilt here, gives the fold column of data/examples (app:reproducibility)",
                 moved == [0, 0, 0], f"rows that move to another fold: {moved}. Install scikit-learn < 1.9 "
                 "(requirements.txt); 1.9 breaks GroupKFold ties differently.")


# ------------------------------------------------------------- 2. monitors (app:corrected)
JUDGES = ["sonnet4_cot_generic", "gpt41mini_cot_generic", "gemini25flash_cot_generic", "sonnet4_cot_cf",
          "sonnet4_cot_cf_p1", "sonnet4_cot_cf_p2", "gpt41mini_cot_cf", "gpt41mini_cot_cf_fewshot", "gemini25flash_cot_cf"]
MEAN_ENS = {"ens_sonnet4_cf_paraphrase": ["sonnet4_cot_cf", "sonnet4_cot_cf_p1", "sonnet4_cot_cf_p2"],
            "ens_crossfamily_cf": ["sonnet4_cot_cf", "gpt41mini_cot_cf", "gemini25flash_cot_cf"],
            "ens_crossfamily_generic": ["sonnet4_cot_generic", "gpt41mini_cot_generic", "gemini25flash_cot_generic"]}
NOTE_VIEWS = {"sonnet4_note_aware_cot": "note_aware_no_hprime", "sonnet4_note_aware_answer": "answer_note_aware_no_hprime"}
CF_VIEWS = {"sonnet4_candidate_prior": "candidate_prior", "sonnet4_answer_cf": "answer_counterfactual"}
HAIKU_VIEWS = {"haiku_answer_only": "answer_only", "haiku_cot_answer": "cot_answer", "haiku_cot_followup": "cot_followup_answer"}

# tab:monitor-intervals: AUROC, CI low, CI high for Mistral, Gemma, Llama
TAB_MONITOR_INTERVALS = table("""
probe_nested              0.637 0.569 0.702 0.653 0.592 0.720 0.610 0.533 0.686
sonnet4_candidate_prior   0.610 0.544 0.676 0.714 0.654 0.769 0.658 0.589 0.729
sonnet4_cot_generic       0.576 0.512 0.638 0.589 0.533 0.649 0.623 0.555 0.686
gpt41mini_cot_generic     0.570 0.502 0.641 0.614 0.549 0.676 0.593 0.522 0.664
gemini25flash_cot_generic 0.498 0.433 0.563 0.582 0.519 0.642 0.629 0.558 0.703
ens_crossfamily_generic   0.570 0.499 0.642 0.617 0.557 0.679 0.651 0.575 0.724
sonnet4_note_aware_cot    0.627 0.571 0.683 0.619 0.563 0.674 0.657 0.596 0.716
sonnet4_note_aware_answer 0.608 0.555 0.664 0.568 0.519 0.619 0.558 0.501 0.615
sonnet4_cot_cf            0.588 0.535 0.642 0.626 0.568 0.687 0.657 0.591 0.719
sonnet4_cot_cf_p1         0.674 0.617 0.726 0.636 0.595 0.676 0.620 0.573 0.664
sonnet4_cot_cf_p2         0.626 0.567 0.686 0.642 0.585 0.699 0.627 0.558 0.693
ens_sonnet4_cf_paraphrase 0.692 0.634 0.749 0.697 0.640 0.754 0.698 0.628 0.765
gpt41mini_cot_cf          0.668 0.606 0.727 0.593 0.531 0.653 0.630 0.563 0.693
gemini25flash_cot_cf      0.648 0.589 0.706 0.599 0.537 0.660 0.598 0.525 0.665
ens_crossfamily_cf        0.694 0.635 0.757 0.662 0.603 0.722 0.685 0.618 0.749
gpt41mini_cot_cf_fewshot  0.547 0.478 0.614 0.544 0.481 0.605 0.553 0.481 0.627
sonnet4_answer_cf         0.571 0.524 0.623 0.594 0.549 0.640 0.541 0.496 0.590
tfidf_answer_only         0.636 0.560 0.705 0.541 0.473 0.610 0.645 0.567 0.723
tfidf_cot_answer          0.578 0.499 0.652 0.586 0.518 0.659 0.643 0.574 0.720
tfidf_prompt_answer       0.649 0.580 0.717 0.539 0.469 0.608 0.633 0.553 0.718
tfidf_prompt_cot_answer   0.580 0.508 0.647 0.569 0.498 0.636 0.597 0.520 0.676
haiku_answer_only         0.605 0.522 0.681 0.716 0.647 0.775 0.637 0.546 0.724
haiku_cot_answer          0.606 0.520 0.688 0.589 0.504 0.659 0.588 0.485 0.688
haiku_cot_followup        0.595 0.520 0.667 0.598 0.518 0.677 0.532 0.437 0.629""")
# tab:monitor-paired: "a:b" -> AUROC(a) - AUROC(b), CI low, CI high for Mistral, Gemma, Llama
TAB_MONITOR_PAIRED = table("""
probe_nested:ens_crossfamily_cf                   -0.057 -0.149 0.033 -0.009 -0.097 0.075 -0.075 -0.170 0.027
probe_nested:ens_sonnet4_cf_paraphrase            -0.055 -0.148 0.035 -0.044 -0.130 0.035 -0.088 -0.180 0.010
probe_nested:sonnet4_cot_cf                        0.049 -0.046 0.138  0.027 -0.058 0.106 -0.047 -0.134 0.045
probe_nested:sonnet4_note_aware_cot                0.010 -0.076 0.095  0.034 -0.051 0.115 -0.047 -0.138 0.050
probe_nested:sonnet4_note_aware_answer             0.029 -0.059 0.113  0.085  0.002 0.165  0.052 -0.039 0.150
probe_nested:sonnet4_candidate_prior               0.027 -0.073 0.120 -0.061 -0.143 0.024 -0.048 -0.136 0.042
probe_nested:sonnet4_cot_generic                   0.061 -0.039 0.153  0.071 -0.018 0.151 -0.017 -0.117 0.087
probe_nested:sonnet4_answer_cf                     0.066 -0.014 0.145  0.059 -0.019 0.138  0.069 -0.012 0.151
probe_nested:tfidf_best                           -0.012 -0.098 0.070  0.068 -0.020 0.147 -0.035 -0.130 0.062
sonnet4_note_aware_cot:sonnet4_cot_generic         0.051 -0.018 0.126  0.029 -0.038 0.101  0.033 -0.032 0.095
sonnet4_note_aware_cot:sonnet4_note_aware_answer   0.019 -0.042 0.083  0.051 -0.014 0.113  0.099  0.029 0.170
sonnet4_cot_cf:sonnet4_note_aware_cot             -0.039 -0.099 0.022  0.007 -0.041 0.054  0.000 -0.063 0.061
ens_sonnet4_cf_paraphrase:sonnet4_note_aware_cot   0.065  0.009 0.125  0.079  0.031 0.126  0.040 -0.016 0.097
sonnet4_candidate_prior:sonnet4_cot_generic        0.034 -0.046 0.109  0.123  0.047 0.200  0.034 -0.054 0.125
sonnet4_cot_cf:sonnet4_candidate_prior            -0.023 -0.103 0.060 -0.088 -0.157 -0.019 -0.001 -0.081 0.075""")


def parsed(rel):  # judge rows that returned a probability, with the probability named "score"
    d = read(rel)
    return d[d.parse_ok.astype(bool)].rename(columns={"prob_dependent": "score"})


def monitor_cells(ex):
    """Per-row scores of every monitor cell, assembled as scripts/pipeline/analysis/monitor_intervals.py does."""
    out = []

    def add(cell, d):
        out.append(d[["model", "group", "label", "score"]].dropna(subset=["score"]).assign(cell=cell))

    add("probe_nested", read("probes/nested_probe_oof.csv").rename(columns={"oof_prob": "score"}))
    j = parsed("monitors/judge_robustness_rows.csv.gz")
    for c in JUDGES:
        add(c, j[j.cell == c])
    for c, members in MEAN_ENS.items():
        w = j[j.cell.isin(members)].pivot_table(index=["model", "group", "label"], columns="cell", values="score")
        add(c, w.dropna().mean(axis=1).rename("score").reset_index())
    for rel, views in [("note_aware_judge_rows", NOTE_VIEWS), ("prior_and_answer_cf_judge_rows", CF_VIEWS)]:
        d = parsed(f"monitors/{rel}.csv.gz")
        d["group"] = d.dataset + "::" + d.question_id
        for c, v in views.items():
            add(c, d[d.view == v])
    strict = pd.concat(ex.values())[["model", "group", "label"]]
    h = read("monitors/haiku35_judge_rows.csv").rename(columns={"judge_prob": "score"})  # 0-100 scale
    for c, v in HAIKU_VIEWS.items():  # lenient-slice rows: keep those in the strict slice, with the strict label
        add(c, h[h.view == v][["model", "group", "score"]].merge(strict, on=["model", "group"]))
    tf = read("monitors/tfidf_oof.csv")
    for v in ["answer_only", "cot_answer", "prompt_answer", "prompt_cot_answer"]:
        add(f"tfidf_{v}", tf[tf.view == v])
    best = [max((g for _, g in tf[tf.model == m].groupby("view")), key=lambda g: auc(g.label, g.score)) for m in M]
    add("tfidf_best", pd.concat(best))
    return pd.concat(out, ignore_index=True)


def monitors(ex):
    cells = monitor_cells(ex)
    st = {k: auc_ci(g.label, g.score, row_draws(len(g))) for k, g in cells.groupby(["cell", "model"], sort=False)}
    for c, v in TAB_MONITOR_INTERVALS.items():
        check("tab:monitor-intervals", f"{c}: AUROC [95% CI]", v, [st[(c, m)] for m in M])
    check("app:corrected", "best TF-IDF view per model, 4 dp", "0.6489 0.5856 0.6454", [st[("tfidf_best", m)][0] for m in M])
    shown = [v for (c, _), v in st.items() if c in TAB_MONITOR_INTERVALS]
    check("app:corrected", "of the 72 cells of tab:monitor-intervals: intervals containing 0.5; median half-width",
          "12 0.065", [sum(lo <= 0.5 <= hi for _, lo, hi in shown), np.median([(hi - lo) / 2 for _, lo, hi in shown])])

    def paired(a, b, m):  # on the rows both cells score
        da, db = (cells[(cells.cell == c) & (cells.model == m)].set_index("group") for c in (a, b))
        common = sorted(set(da.index) & set(db.index))
        y, sa = da.loc[common, "label"].to_numpy(int), da.loc[common, "score"]
        return dict(r=paired_auc(y, sa, db.loc[common, "score"], row_draws(len(y))), n=len(y), auc_a=auc(y, sa))

    for k, v in TAB_MONITOR_PAIRED.items():
        a, b = k.split(":")
        check("tab:monitor-paired", f"{a} minus {b} [95% CI]", v, [paired(a, b, m)["r"][:3] for m in M])
    others = [c for c in TAB_MONITOR_INTERVALS if c != "probe_nested"] + ["tfidf_best"]
    pm = [paired("probe_nested", c, m)["r"] for c in others for m in M]
    check("app:corrected", "probe-minus-monitor comparisons (of 72) whose interval contains zero; median half-width",
          "67 0.090", [sum(lo <= 0 <= hi for _, lo, hi, _ in pm), np.median([(hi - lo) / 2 for _, lo, hi, _ in pm])])
    check("app:corrected", "probe-minus-monitor comparisons left significant by a Holm correction", "0",
          [holm_rejections([p for *_, p in pm])])
    gen = [paired("probe_nested", "sonnet4_cot_generic", m) for m in M]
    check("tab:monitor-intervals", "rows the Sonnet-4 generic CoT judge scored, and the probe on them (caption)",
          "421 382 259 0.637 0.659 0.606", [r["n"] for r in gen] + [r["auc_a"] for r in gen])
    return cells


# ------------------------------------------------------------- 3. judge calibration
TAB_JUDGE_ROBUSTNESS = table("""
sonnet4_cot_generic       0.255 0.267 0.258 0.199 0.219 0.195 0.668 0.682 0.632
gpt41mini_cot_generic     0.196 0.250 0.294 0.090 0.175 0.233 0.740 0.634 0.543
gemini25flash_cot_generic 0.239 0.267 0.277 0.205 0.215 0.227 0.792 0.713 0.609
sonnet4_cot_cf            0.217 0.263 0.275 0.137 0.197 0.216 0.710 0.587 0.520
sonnet4_cot_cf_p1         0.304 0.393 0.404 0.317 0.428 0.419 0.493 0.330 0.290
sonnet4_cot_cf_p2         0.233 0.284 0.332 0.161 0.235 0.308 0.647 0.536 0.437
gpt41mini_cot_cf          0.185 0.259 0.289 0.081 0.177 0.229 0.751 0.641 0.568
gemini25flash_cot_cf      0.165 0.208 0.243 0.083 0.120 0.172 0.837 0.785 0.776
gpt41mini_cot_cf_fewshot  0.201 0.215 0.302 0.096 0.119 0.247 0.740 0.740 0.568
ens_sonnet4_cf_paraphrase 0.203 0.258 0.288 0.191 0.275 0.299 0.616 0.484 0.415
ens_crossfamily_cf        0.160 0.199 0.214 0.070 0.111 0.104 0.766 0.671 0.621""")


def calibration():
    j = read("monitors/judge_robustness_rows.csv.gz")
    ok, gen = j.parse_ok.astype(bool), j.cell == "sonnet4_cot_generic"
    check("app:corrected", "parse rate of the Sonnet-4 generic CoT cell, its unparsed rows; parse rate of every other cell",
          "0.9972 3 1.000", [ok[gen].mean(), (~ok[gen]).sum(), ok[~gen].mean()])
    j = j[ok & j.prob_dependent.notna()]
    beat = under = 0
    for c, v in TAB_JUDGE_ROBUSTNESS.items():
        got = []
        for m in M:  # an ensemble averages its members' probabilities
            cs = MEAN_ENS.get(c, [c])
            w = j[j.cell.isin(cs) & (j.model == m)].pivot_table(index=["group", "label"], columns="cell",
                                                                 values="prob_dependent").dropna().reset_index()
            y, p = w.label.to_numpy(int), w[cs].to_numpy(float).mean(axis=1)
            got.append((np.mean((p - y) ** 2), ece(y, p), p.mean()))
            beat += got[-1][0] < np.mean((y.mean() - y) ** 2)
            under += p.mean() < y.mean()
        check("tab:judge-robustness", f"{c}: Brier x3, ECE x3, mean p x3", v, np.array(got).T)
    check("app:corrected", "cells (of 33) beating the constant base-rate Brier score; with mean p below the base rate",
          "0 30", [beat, under])


# ------------------------------------------------------------- 4. ensembles (app:ensemble)
MEMBERS = ["probe", "na_cot", "cf_para", "prior", "tfidf"]
PAIRINGS = {"probe+note_aware_cot": ["probe", "na_cot"], "probe+cot_cf_paraphrase": ["probe", "cf_para"],
            "probe+candidate_prior": ["probe", "prior"], "probe+tfidf_best": ["probe", "tfidf"],
            "probe+note_aware_cot+candidate_prior": ["probe", "na_cot", "prior"],
            "note_aware_cot+candidate_prior": ["na_cot", "prior"]}
# tab:ensemble members (AUROC x3), then per pairing: rank-averaged AUROC x3 and the paired difference
# against the ensemble's nominally best member [CI] x3
TAB_ENSEMBLE_MEMBERS = table("""
probe   0.637 0.653 0.610
na_cot  0.627 0.619 0.657
cf_para 0.692 0.697 0.698
prior   0.610 0.714 0.658
tfidf   0.649 0.586 0.645""")
TAB_ENSEMBLE = table("""
probe+note_aware_cot                 0.691 0.686 0.697  0.053  0.013 0.096  0.032 -0.009 0.076  0.039 -0.015 0.096
probe+cot_cf_paraphrase              0.709 0.728 0.701  0.016 -0.032 0.067  0.030 -0.013 0.076  0.003 -0.049 0.054
probe+candidate_prior                0.671 0.745 0.675  0.034 -0.018 0.086  0.031 -0.011 0.074  0.016 -0.031 0.062
probe+tfidf_best                     0.674 0.654 0.659  0.025 -0.017 0.068  0.001 -0.047 0.046  0.013 -0.034 0.068
probe+note_aware_cot+candidate_prior 0.709 0.747 0.713  0.071  0.012 0.130  0.033 -0.013 0.079  0.054  0.009 0.099
note_aware_cot+candidate_prior       0.659 0.716 0.702  0.032 -0.022 0.087  0.002 -0.039 0.043  0.044 0.0005 0.087""")
# The best-achievable block maximizes over the nine text bases (the four text members and five of their
# combinations, as in scripts/pipeline/analysis/ensemble_probe_text.py, in its member order) and all combiners.
# Every maximum is a rank average (app:ensemble), so rank averages over these bases reproduce it.
TEXT_BASES = [["na_cot"], ["prior"], ["cf_para"], ["tfidf"], ["na_cot", "prior"], ["na_cot", "cf_para"],
              ["prior", "cf_para"], ["na_cot", "prior", "cf_para"], ["na_cot", "prior", "cf_para", "tfidf"]]


def ensembles():
    er = read("ensemble/per_row_scores.csv")
    member, rank_avg, cells = {}, {}, []  # cells: (model, has probe, beats best member, beats every member)
    for (m, pair), d in er.groupby(["model", "pairing"], sort=False):
        y, draws = d.label.to_numpy(int), row_draws(len(d))
        mem = {k: d[f"m_{k}"] for k in PAIRINGS[pair]}
        member.update({(m, k): auc(y, s) for k, s in mem.items()})
        best = max(mem, key=lambda k: member[(m, k)])
        for combiner in ["rank_avg", "z_avg", "rank_avg_foldfit", "z_avg_foldfit", "stack_lr"]:
            vs = {k: paired_auc(y, d[combiner], s, draws) for k, s in mem.items()}
            cells.append((m, "probe" in pair, vs[best][1] > 0, all(r[1] > 0 for r in vs.values())))
            if combiner == "rank_avg":
                rank_avg[(m, pair)] = [auc(y, d[combiner]), *vs[best][:3]]
    for k, v in TAB_ENSEMBLE_MEMBERS.items():
        check("tab:ensemble", f"member {k}: AUROC x3", v, [member[(m, k)] for m in M])
    for pair, v in TAB_ENSEMBLE.items():
        check("tab:ensemble", f"{pair}: rank-averaged AUROC x3; paired difference vs the best member [CI] x3", v,
              [rank_avg[(m, pair)][0] for m in M] + [x for m in M for x in rank_avg[(m, pair)][1:]])
    probe_cells = [c for c in cells if c[1]]
    check("app:ensemble", "of the 75 probe cells: beat the best member, beat every member, of those Mistral; "
          "no-probe cells among the 14 of 90 that beat every member", "14 10 10 4",
          [sum(c[2] for c in probe_cells), sum(c[3] for c in probe_cells),
           sum(c[3] and c[0] == "mistral" for c in probe_cells), sum(c[3] and not c[1] for c in cells)])

    best, cor = [], {}
    for m in M:  # one row per retained example with all five member scores
        d = er[er.model == m]
        w = d[d.pairing == "probe+note_aware_cot"][["group", "label"]]
        for k in MEMBERS:
            w = w.merge(d[d[f"m_{k}"].notna()].drop_duplicates("group")[["group", f"m_{k}"]], on="group", how="left")
        r01 = {k: (rankdata(w[f"m_{k}"]) - 1) / (len(w) - 1) for k in MEMBERS}  # normalised ranks

        def rank_mean(ks):
            return np.mean([r01[k] for k in ks], axis=0)

        # best achievable: the best rank-averaged text base, and the best one with the probe added
        text = max((rank_mean(b) for b in TEXT_BASES), key=lambda s: auc(w.label, s))
        probe = max((rank_mean(["probe", *b]) for b in TEXT_BASES), key=lambda s: auc(w.label, s))
        best.append([auc(w.label, text), auc(w.label, probe), *paired_auc(w.label, probe, text, row_draws(len(w)))[:3]])
        k10 = max(round(0.1 * len(w)), 1)
        top = {k: set(np.argsort(-r01[k], kind="mergesort")[:k10]) for k in MEMBERS}
        for a, b in combinations(MEMBERS, 2):  # Spearman and top-decile overlap
            cor[(m, a, b)] = (spearmanr(w[f"m_{a}"], w[f"m_{b}"]).statistic, len(top[a] & top[b]), k10, k10**2 / len(w))
    check("tab:ensemble", "best achievable: text only x3, allowed the probe x3, paired increment [CI] x3",
          "0.741 0.744 0.758 0.750 0.768 0.756 0.009 -0.014 0.032 0.024 -0.004 0.055 -0.002 -0.025 0.020",
          [r[0] for r in best] + [r[1] for r in best] + [x for r in best for x in r[2:]])
    check("app:ensemble", "Spearman of the probe with note-aware CoT x3, CoT-CF x3, candidate prior x3",
          "0.158 0.187 0.032 0.185 0.193 0.125 0.082 0.149 0.232",
          [cor[(m, "probe", k)][0] for k in ["na_cot", "cf_para", "prior"] for m in M])
    judges = [cor[(m, a, b)][0] for m in M for a, b in [("na_cot", "cf_para"), ("na_cot", "prior"), ("cf_para", "prior")]]
    tfidf = [cor[(m, "probe", "tfidf")][0] for m in M]
    check("app:ensemble", "Spearman among the three judges, range; probe with TF-IDF, range", "0.32 0.68 0.28 0.36",
          [min(judges), max(judges), min(tfidf), max(tfidf)])
    lt, mp = cor[("llama", "probe", "na_cot")], cor[("mistral", "probe", "prior")]
    check("app:ensemble", "top-decile overlap: Llama probe vs note-aware (of 26), Mistral probe vs prior (of 42); "
          "overlap expected by chance", "0 26 0 42 2.6 4.2", [lt[1], lt[2], mp[1], mp[2], lt[3], mp[3]])
    sw = read("ensemble/stack_weights.csv")
    sw = sw[sw.members.str.split("|").map(lambda ms: "probe" in ms)]
    sw["w_probe"] = [r[f"w{r.members.split('|').index('probe')}"] for _, r in sw.iterrows()]
    fold_mean = sw.groupby(["model", "pairing"]).w_probe.mean()
    check("app:ensemble", "stacked combiner's standardized probe weight, fold mean, range", "0.25 0.52",
          [fold_mean.min(), fold_mean.max()])


# ------------------------------------------------------------- 5. probe controls (app:corrected)
# tab:diff-layer-search: nested AUROC [CI] for Mistral, Gemma, Llama (AUROC at 3 dp, CI at 2 dp)
TAB_DIFF_LAYER_SEARCH = table("""
raw_W     0.639 0.57 0.71 0.656 0.59 0.72 0.610 0.54 0.69
W_minus_G 0.470 0.40 0.54 0.573 0.51 0.64 0.476 0.40 0.55
W_minus_C 0.558 0.49 0.63 0.515 0.45 0.59 0.609 0.54 0.69
G_minus_C 0.506 0.44 0.57 0.530 0.46 0.59 0.565 0.48 0.64
W_resid_G 0.550 0.48 0.62 0.489 0.42 0.56 0.484 0.41 0.55
W_resid_C 0.519 0.45 0.59 0.524 0.45 0.59 0.476 0.40 0.55""")
# tab:nonlinear-probes: linear, MLP, RBF-SVC, RFM for Mistral, then Gemma, then Llama
TAB_NONLINEAR = table("""
W         0.698 0.672 0.732 0.702 0.684 0.654 0.685 0.710 0.610 0.648 0.660 0.644
G         0.681 0.685 0.723 0.705 0.684 0.664 0.698 0.710 0.643 0.683 0.658 0.668
C         0.651 0.654 0.675 0.711 0.695 0.666 0.699 0.691 0.635 0.608 0.621 0.639
W_minus_G 0.540 0.560 0.542 0.575 0.566 0.554 0.546 0.574 0.514 0.472 0.477 0.453
W_minus_C 0.520 0.503 0.625 0.576 0.579 0.609 0.562 0.589 0.574 0.538 0.495 0.547""")


def difference_probes():
    do = read("controls/difference_probe_oof.csv")
    res = {k: auc_ci(g.label, g.nested_score, cluster_draws(g.group)) for k, g in do.groupby(["feature_set", "model"])}
    for fs, v in TAB_DIFF_LAYER_SEARCH.items():
        check("tab:diff-layer-search", f"{fs}: nested AUROC [95% CI]", v, [res[(fs, m)] for m in M])
    check("app:corrected", "Gemma W-G and Llama W-C [95% CI]", "0.573 0.507 0.637 0.609 0.535 0.685",
          [res[("W_minus_G", "gemma")], res[("W_minus_C", "llama")]])
    cur = read("controls/difference_probe_layer_curves.csv")
    g = cur[(cur.feature_set == "W_minus_G") & (cur.model == "gemma")]
    check("app:corrected", "Gemma W-G: layers with AUROC above 0.5, of all layers", "39 42", [(g.auroc > 0.5).sum(), len(g)])


def nonlinear():
    nl, fams, res, dl = read("controls/nonlinear_probe_oof.csv.gz"), ["linear", "mlp", "rbf_svc", "rfm"], {}, {}
    for (m, fs), g in nl.groupby(["model", "featureset"], sort=False):
        X = np.column_stack([g[g.family == f].score.to_numpy(float) for f in fams])  # same row order per family
        lab, ok = g[g.family == "linear"], np.isfinite(X).all(axis=1)
        y, draws = lab.label.to_numpy(int)[ok], cluster_draws(lab.group.to_numpy()[ok])
        for i, f in enumerate(fams):  # delta against the linear probe on the same resamples
            res[(m, fs, f)] = auc(y, X[ok, i])
            d = delta_draws(y, X[ok, i], X[ok, 0], draws)
            dl[(m, fs, f)] = (res[(m, fs, f)] - auc(y, X[ok, 0]), *ci(d), np.std(d, ddof=1))
    for fs, v in TAB_NONLINEAR.items():  # the linear column is also the re-extracted W/G/C, W-G, W-C of app:corrected
        check("tab:nonlinear-probes", f"{fs}: linear, MLP, RBF-SVC, RFM x3", v, [res[(m, fs, f)] for m in M for f in fams])
    cells = [("mistral", "W_minus_G", "rfm"), ("gemma", "W_minus_G", "rfm"), ("llama", "W_minus_G", "rbf_svc"),
             ("mistral", "C", "rfm"), ("mistral", "W_minus_C", "rbf_svc")]
    check("app:corrected", "deltas vs linear [CI]: best W-G per model; Mistral C RFM and W-C RBF-SVC",
          "0.035 -0.029 0.102 0.009 -0.066 0.079 -0.036 -0.132 0.061 0.060 0.010 0.112 0.105 0.037 0.180",
          [dl[c][:3] for c in cells])

    def pooled(fs, f):  # fixed-effect inverse-variance pooling of the per-model deltas
        w = np.array([dl[(m, fs, f)][3] ** -2 for m in M])
        return (w * [dl[(m, fs, f)][0] for m in M]).sum() / w.sum()

    nonzero = sum(not v[1] <= 0 <= v[2] for k, v in dl.items() if k[2] != "linear")
    z80 = norm.ppf(0.975) + norm.ppf(0.8)  # two-sided 5% test at 80% power
    mde = [z80 * dl[(m, "W_minus_G", f)][3] for m in M for f in fams[1:]]
    check("app:corrected", "deltas excluding zero (of 45); pooled W-G MLP, RBF-SVC, RFM; pooled G RBF-SVC, RFM; "
          "W-G delta detectable at 80% power, range", "2 -0.013 -0.019 -0.002 0.023 0.025 0.093 0.145",
          [nonzero, *(pooled("W_minus_G", f) for f in fams[1:]), pooled("G", "rbf_svc"), pooled("G", "rfm"), min(mde), max(mde)])
    pn, got = read("controls/nonlinear_probe_permutation_null.csv"), []
    for m, fs, f in [cells[0], cells[1], cells[4]]:
        null = pn[(pn.model == m) & (pn.featureset == fs) & (pn.family == f)].auroc.to_numpy()
        got.append((null.max(), (1 + (null >= res[(m, fs, f)]).sum()) / (1 + len(null))))
    check("app:corrected", "label-permutation null: maximum and p for Mistral and Gemma W-G RFM; p for Mistral W-C RBF-SVC",
          "0.575 0.049 0.617 0.098 0.024", [*got[0], *got[1], got[2][1]])
    return res


def within_dataset(ex, nl):
    oof, oracle = [], []
    for m in M:
        d, floor = ex[m], np.full(len(ex[m]), np.nan)
        for k in sorted(d.fold.unique()):  # score each test row by its dataset's dependent rate on the training folds
            test = (d.fold == k).to_numpy()
            floor[test] = d.dataset[test].map(d[~test].groupby("dataset").label.mean()).fillna(d.label[~test].mean())
        oof.append(auc(d.label, floor))
        oracle.append(auc(d.label, d.dataset.map(d.groupby("dataset").label.mean())))
    check("app:corrected", "dataset-identity floor out of fold x3, in-sample oracle x3", "0.564 0.574 0.636 0.632 0.620 0.672",
          oof + oracle)
    check("app:corrected", "pooled W probe minus the out-of-fold floor", "0.134 0.110 -0.026",
          [nl[(m, "W", "linear")] - o for m, o in zip(M, oof)])
    rate = [1 - ex[m].groupby("fold").label.mean() for m in M]
    check("app:corrected", "per-fold independent-class rate, min-max x3", "0.141 0.274 0.182 0.286 0.173 0.346",
          [(r.min(), r.max()) for r in rate])
    h = ex["llama"][ex["llama"].dataset == "hotpotqa"]  # within-dataset folds: GroupKFold(5) on that dataset
    f = group_folds(h)
    r = [1 - h.label.to_numpy()[f == k].mean() for k in range(5)]
    check("app:corrected", "Llama HotpotQA within-dataset fold independent-class rate, min-max", "0.045 0.318",
          [min(r), max(r)])
    p = read("probes/nested_probe_oof.csv")
    a = {k: auc(g.label, g.oof_prob) for k, g in p.groupby(["model", "dataset"])}
    check("app:diagnostics", "nested probe within each dataset: HotpotQA x3; GPQA x3; Llama MMLU-Pro",
          "0.68 0.76 0.71 0.48 0.48 0.41 0.40",
          [a[(m, "hotpotqa")] for m in M] + [a[(m, "gpqa")] for m in M] + [a[("llama", "mmlu_pro")]])
    layers = read("probes/nested_probe_layers.csv").set_index("model").loc[M].selected_layers
    check("app:diagnostics", "layers the nested probe selects in its five outer folds, x3",
          "11 9 2 12 17 13 13 9 14 36 13 7 30 30 13", [int(x) for s in layers for x in s.split(",")])


def wg_transfer():
    w, gap, auroc, sp = read("controls/wg_transfer_oof.csv.gz"), {}, {}, []
    for m in M:
        d = w[w.model == m]
        y, draws = d.label.to_numpy(int), cluster_draws(d.group)
        sc = {s + t: d[f"score_{s}to{t}"].to_numpy(float) for s in "WGC" for t in "WGC"}
        for s in "WGC":  # (s probe on s states) - (s probe on t states)
            for t in "WGC".replace(s, ""):
                gap[(m, s, t)] = paired_auc(y, sc[s + s], sc[s + t], draws)[:3]
        auroc[m] = (auc(y, sc["WW"]), auc(y, sc["WG"]))
        sp.append([spearmanr(sc["WW"][y == c], sc["GG"][y == c]).statistic for c in (1, 0)])
    nonzero = {k: v[0] for k, v in gap.items() if not v[1] <= 0 <= v[2]}
    check("app:corrected", "W probe on W states x3, on G states x3", "0.698 0.684 0.613 0.682 0.687 0.631",
          [auroc[m][0] for m in M] + [auroc[m][1] for m in M])
    check("app:corrected", "(W->W) - (W->G) [CI] x3", "0.016 -0.013 0.043 -0.002 -0.015 0.008 -0.018 -0.041 0.004",
          [gap[(m, "W", "G")] for m in M])
    check("app:corrected", "Spearman of W and G probe scores, dependent x3, independent x3", "0.92 0.98 0.76 0.90 0.98 0.84",
          np.array(sp).T)
    check("app:corrected", "gaps excluding zero (of 18): Mistral W->C [CI], Llama G->W, Llama C->W",
          "3 0.030 0.002 0.058 0.026 0.035",
          [len(nonzero), *gap[("mistral", "W", "C")], nonzero.get(("llama", "G", "W"), 0), nonzero.get(("llama", "C", "W"), 0)])


# ------------------------------------------------------------- 6. span-answer labels (app:corrected)
STRATA = ["exact", "contain_ans_in_h", "contain_h_in_ans", "negation_suspect"]


def span_labels(ex, cells):
    f = read("labels/span_label_features.csv")
    sp = f[f.answer_kind == "span"]
    check("app:corrected", "span rows of 1065; containment-only, six or more words, negation cue, one-word candidate",
          "714 163 46 15 106", [len(sp), sp.containment_only.sum(), (sp.n_words_answer >= 6).sum(),
                                sp.negation_suspect.sum(), (sp.n_words_h1 == 1).sum()])
    # One random stream for the whole block, used in the order of scripts/pipeline/analysis/span_label_analysis.py.
    pool, rng, prec, adj = read("labels/adjudication_pool.csv"), np.random.default_rng(1729), {}, {}
    for tag in ["gpt41", "gemini25flash"]:
        a = read(f"labels/adjudications_{tag}.csv").drop_duplicates("job_id", keep="last")
        a["asserts"] = a.verdict.eq("asserts").astype(int)
        adj[tag] = a.set_index("job_id")
        for scope in ["ALL"] + M:
            sub, ps = (a, pool) if scope == "ALL" else (a[a.model == scope], pool[pool.model == scope])
            pos, neg = {s: (ps.stratum == s).sum() for s in STRATA}, (ps.stratum == "parser_negative").sum()

            def frac(s, resample):  # share of adjudications in stratum s saying the answer asserts h
                v = sub[sub.stratum == s].asserts.to_numpy()
                v = v[rng.integers(0, len(v), len(v))] if resample and len(v) else v
                return v.mean() if len(v) else np.nan

            def precision_recall(resample):  # stratum-weighted over the span population
                tp = sum(pos[s] * frac(s, resample) for s in STRATA if pos[s] > 0)
                fn = neg * frac("parser_negative", resample) if neg else 0
                return tp / sum(pos.values()), tp / (tp + fn), sum(pos.values()) - tp

            prec[(tag, scope)] = (*precision_recall(False), *ci([precision_recall(True)[0] for _ in range(4000)]))
    g, e = prec[("gpt41", "ALL")], prec[("gemini25flash", "ALL")]
    check("app:corrected", "parser precision [CI], GPT-4.1 and Gemini-2.5-flash", "0.899 0.884 0.916 0.881 0.864 0.898",
          [g[0], *g[3:], e[0], *e[3:]])
    check("app:corrected", "estimated false-positive labels x2; exact matches and parser-negatives in the pool",
          "72 85 551 1702", [g[2], e[2], (pool.stratum == "exact").sum(), (pool.stratum == "parser_negative").sum()])
    check("app:corrected", "label noise, % of 1065 rows and % of 714 span rows", "6.8 8.0 10.1 11.9",
          [100 * g[2] / 1065, 100 * e[2] / 1065, 100 * g[2] / 714, 100 * e[2] / 714])
    check("app:corrected", "pooled parser recall (GPT-4.1)", "0.94", [g[1]])
    ag, ge = adj["gpt41"], adj["gemini25flash"]
    both = ag.index.intersection(ge.index)
    x, z = ag.loc[both].asserts, ge.loc[both].asserts
    pe = x.mean() * z.mean() + (1 - x.mean()) * (1 - z.mean())
    check("app:corrected", "adjudicated rows, agreement, Cohen kappa", "258 0.888 0.775",
          [len(both), (x == z).mean(), ((x == z).mean() - pe) / (1 - pe)])
    contain = ["contain_ans_in_h", "contain_h_in_ans"]
    exact = [a[a.stratum == "exact"] for a in (ag, ge)]
    negation = [a[a.stratum == "negation_suspect"] for a in (ag, ge)]
    check("app:corrected", "exact rows asserting h, of all exact adjudications; containment rows; of them asserting h, "
          "GPT-4.1 and Gemini; negation rows judged wrong of 15, x2", "100 100 148 90 77 14 14",
          [sum(d.asserts.sum() for d in exact), sum(len(d) for d in exact), ag.stratum.isin(contain).sum(),
           ag[ag.stratum.isin(contain)].asserts.sum(), ge[ge.stratum.isin(contain)].asserts.sum(),
           *[len(d) - d.asserts.sum() for d in negation]])

    new = read("labels/exact_match_slice.csv")
    newmap, comp, probe = dict(zip(new.key, new.label)), [], []
    oof = read("probes/nested_probe_oof.csv").assign(key=lambda d: d.model + "||" + d.group)
    for m in M:
        old, n = ex[m].set_index(ex[m].model + "||" + ex[m].group).label, new[new.model == m].set_index("key").label
        shared = old.index.intersection(n.index)
        comp += [len(n), 100 * n.mean(), len(old.index.difference(n.index)), len(n.index.difference(old.index)),
                 (old[shared] != n[shared]).sum()]
        sub = oof[oof.model == m]
        surv = sub[sub.key.isin(newmap)]
        # full slice, survivors with the old label, survivors with the new label (the full-slice
        # interval is not printed, but it draws from the stream before the survivor intervals)
        for y, s, gr in [(sub.label, sub.oof_prob, sub.group), (surv.label, surv.oof_prob, surv.group),
                         (surv.key.map(newmap), surv.oof_prob, surv.group)]:
            probe.append((*auc_ci(y, s, cluster_draws(gr, rng)), len(y)))
    check("tab:span-relabel", "exact-only N, dependent share %, dropped, added, flipped x3",
          "340 85.3 84 3 12 342 79.8 46 4 13 232 75.0 33 5 10", comp)
    check("tab:span-relabel", "probe on the full slice x3; surviving rows x3 (caption)", "0.637 0.653 0.610 337 338 227",
          [probe[3 * i][0] for i in range(3)] + [probe[3 * i + 1][3] for i in range(3)])
    check("tab:span-relabel", "probe on survivors, old label -> new label [CI] x3",
          "0.654 0.694 0.615 0.774 0.663 0.664 0.587 0.740 0.609 0.635 0.552 0.718",
          [(probe[3 * i + 1][0], *probe[3 * i + 2][:3]) for i in range(3)])
    judges, best = [*NOTE_VIEWS, *CF_VIEWS, *JUDGES, *MEAN_ENS], []
    for m in M:  # the 16 full-slice judge cells, rescored on the surviving rows with the new labels
        a = {}
        for c in judges + ["probe_nested"]:
            g_ = cells[(cells.cell == c) & (cells.model == m)]
            k = g_.model + "||" + g_.group
            a[c] = auc(k[k.isin(newmap)].map(newmap), g_.score[k.isin(newmap)])
        b = max(judges, key=a.get)
        best.append((a[b], a["probe_nested"], b == ("sonnet4_candidate_prior" if m == "gemma" else "ens_crossfamily_cf")))
    check("app:corrected", "after relabelling: best judge cell x3 (CoT-CF cross-family mean; Gemma: candidate prior), "
          "probe x3", "0.695 0.657 0.687 0.694 0.664 0.635", np.array(best)[:, :2].T)
    statement("app:corrected", "after relabelling, the best judge cell is the CoT-CF cross-family mean for Mistral and "
              "Llama and the candidate prior for Gemma", all(b[2] for b in best))


# ------------------------------------------------------------- 7. injection factorial (app:injection)
CELLS4 = ["adv_warn", "adv_nowarn", "ben_warn", "ben_nowarn"]
# tab:injection-estimand, Sonnet 4.5 column: AUROC [CI] for Mistral, Gemma, Llama
TAB_INJECTION_SONNET = table("""
adv_warn_vs_ben_nowarn   0.616 0.519 0.709 0.645 0.561 0.729 0.571 0.450 0.683
adv_nowarn_vs_ben_nowarn 0.596 0.516 0.678 0.634 0.550 0.720 0.534 0.438 0.631
adv_warn_vs_adv_nowarn   0.556 0.452 0.658 0.583 0.467 0.693 0.507 0.382 0.635""")


def injection():
    po, hj, out, dep, nind = read("injection/probe_oof.csv"), read("injection/haiku45_judge_scores.csv"), {}, [], []
    for m in M:
        for c in CELLS4:
            p, j = po[(po.model == m) & (po.cell == c)], hj[(hj.model == m) & (hj.cell == c)]
            dep, nind = dep + [j.label.mean()], nind + [(j.label == 0).sum()]
            out[f"{m}/{c}/probe"] = auc_ci(p.label, p.probe_oof_prob, cluster_draws(p.group))
            out[f"{m}/{c}/judge"] = auc_ci(j.label, j.judge_prob, cluster_draws(j.group))
    excl = {k for k, v in out.items() if not v[1] <= 0.5 <= v[2]}
    check("app:injection", "dependent rate per cell, range; independent rows per cell, range", "0.76 0.95 7 19",
          [min(dep), max(dep), min(nind), max(nind)])
    check("app:injection", "intervals containing 0.5 (of 24); the exceptions are Mistral adv_nowarn probe [CI], "
          "Mistral ben_warn judge, Llama adv_nowarn probe", "21 0.332 0.199 0.470 0.717 0.655",
          [24 - len(excl), *out["mistral/adv_nowarn/probe"], out["mistral/ben_warn/judge"][0],
           out["llama/adv_nowarn/probe"][0]])
    statement("app:injection", "the three intervals that exclude 0.5 are the Mistral adv_nowarn probe, the Mistral "
              "ben_warn judge and the Llama adv_nowarn probe",
              excl == {"mistral/adv_nowarn/probe", "mistral/ben_warn/judge", "llama/adv_nowarn/probe"})
    so = read("injection/sonnet45_judge_scores.csv")
    traces = so.assign(w=so.unit.str.split("_vs_").str[0]).drop_duplicates(["model", "w", "group"])
    res = {k: auc_ci(d.label, d.judge_prob, cluster_draws(d.group))
           for k, d in so[so.judge_prob.notna()].groupby(["level", "model", "unit"])}
    for u, v in TAB_INJECTION_SONNET.items():
        check("tab:injection-estimand", f"Sonnet 4.5 judge, {u}: AUROC [CI] x3", v, [res[("cross", m, u)] for m in M])
    excl = {(m, u) for (lv, m, u), v in res.items() if lv == "cross" and not v[1] <= 0.5 <= v[2]}
    in_cell = [v for k, v in res.items() if k[0] == "cell"]
    contrasts = [*TAB_INJECTION_SONNET]
    check("tab:injection-estimand", "traces the Sonnet 4.5 judge scored (of 1,731); intervals excluding 0.5 (Mistral and "
          "Gemma, original and adversarial-only)", "1715 4", [traces.judge_prob.notna().sum(), len(excl)])
    statement("tab:injection-estimand", "the four Sonnet 4.5 intervals that exclude 0.5 are Mistral's and Gemma's "
              "original and adversarial-only rows", excl == {(m, u) for m in M[:2] for u in contrasts[:2]})
    check("app:injection", "Sonnet 4.5 on the main-task estimand: cell intervals containing 0.5 (of 12); AUROC range",
          "12 0.406 0.632", [sum(lo <= 0.5 <= hi for _, lo, hi in in_cell), min(v[0] for v in in_cell), max(v[0] for v in in_cell)])
    r = read("injection/marker_rates.csv").query("subset == 'all'").set_index(["model", "cell"])
    effects = []
    for col in ["ctx_words_rate", "injection_flag_words_rate"]:
        for m in M:
            aw, an, bw, bn = (r.loc[(m, c), col] for c in CELLS4)
            effects += [(aw + bw - an - bn) / 2, (aw + an - bw - bn) / 2, aw - an - bw + bn]  # warning, adversarial, interaction
    check("tab:injection-factorial", "warning, adversarial and interaction effects from the cell rates x3; "
          "context words, then injection naming", "-0.049 0.127 -0.037 -0.039 0.057 0.003 -0.098 0.195 0.124 "
          "0.100 0.076 0.141 0.012 0.010 -0.001 0.128 0.165 0.224", effects)
    fn = read("injection/funnel.csv").query("cell != 'main_task'")
    check("app:injection", "question pairs per cell; pairs where both answers parse, range over the 12 cells", "399 282 386",
          [*fn.pairs.unique(), fn.both_parse_ok.min(), fn.both_parse_ok.max()])
    rs = json.loads((S / "injection" / "run_settings.json").read_text())
    check("app:injection", "max_new_tokens x3; seed; questions per dataset; traces in all", "512 512 640 2027 100 9576",
          [*(rs[m]["max_new_tokens"] for m in M), *{rs[m]["seed"] for m in M},
           *{rs[m]["questions_per_dataset"] for m in M}, sum(rs[m]["n_records"] for m in M)])


# ------------------------------------------------------------- 8. powered patching (app:patching)
def patching():
    det, key = read("patching/powered_patching_rows.csv.gz"), ["model", "dataset", "question_id", "label"]
    base = det[det.condition_name == "baseline"]
    piv = base.pivot_table(index=key, columns="recipient", values="matches_h1", aggfunc="first").astype("boolean").fillna(False)
    repro = (piv.W & (piv.G == (piv.index.get_level_values("label") == 0))).astype(bool).rename("rep")  # baseline reproduces the label
    b0 = base.set_index(key + ["recipient"]).matches_h1.rename("b0")
    pat = det[det.condition_name != "baseline"].join(b0, on=key + ["recipient"]).join(repro, on=key).dropna(subset=["b0"])
    rows = []
    for k, s in pat.groupby(["model", "label", "condition_name", "layer", "position"]):
        for pop, t in [("all", s), ("cond", s[s.rep.astype("boolean").fillna(False).astype(bool)])]:
            b, a = t.b0.astype(bool).to_numpy(), t.matches_h1.astype(bool).to_numpy()
            dn, up = int((b & ~a).sum()), int((~b & a).sum())
            rows.append(dict(zip(["model", "label", "cond", "layer", "pos"], k), pop=pop, n=len(t), dn=dn, up=up,
                             delta=a.mean() - b.mean(), p=mcnemar(dn, up), base=b.mean(), parse=t.parse_ok.mean()))
    r = pd.DataFrame(rows)
    A, C = r[r["pop"] == "all"], r[r["pop"] == "cond"]
    check("app:patching", "selected examples per model", "181 193 176", (base.groupby("model").size() // 2)[M])
    tab = []
    for m in M:
        for lab in (1, 0):
            g = A[(A.model == m) & (A.label == lab)]
            gw, wg = g[g.cond == "patch_G_to_W"], g[g.cond == "patch_W_to_G"]
            tab += [g.n.max(), 6 / g.n.max(), max(0, -gw.delta.min()), gw.p.min(), max(0, wg.delta.max()), wg.p.min()]
    check("tab:patching", "n, 6/n, G->W down (p), W->G up (p); per model, dependent then independent",
          "100 0.060 0.010 0.375 0.020 0.500 81 0.074 0.037 0.125 0.037 0.375 100 0.060 0.000 0.125 0.020 0.500 "
          "93 0.065 0.032 0.453 0.054 0.063 100 0.060 0.040 0.454 0.010 0.063 76 0.079 0.053 0.219 0.013 0.344", tab)
    top = A.loc[A.delta.abs().idxmax()]
    check("app:patching", "cells, n range; largest |delta|: n, down, up, |delta|, p; cells at p <= 0.05",
          "108 76 100 81 1 6 0.062 0.125 0",
          [len(A), A.n.min(), A.n.max(), top.n, top.dn, top.up, abs(top.delta), top.p, (A.p <= 0.05).sum()])
    statement("app:patching", "the largest |delta| is Mistral, independent class, G->W, layer 6, final eight tokens",
              (top.model, top.label, top.cond, top.layer, top.pos) == ("mistral", 0, "patch_G_to_W", 6, "last8"))
    check("app:patching", "smallest exact McNemar p over the 108 cells", "0.0625", [A.p.min()])
    check("app:patching", "significance floor 6/n and 13/n at n = 100 and 76; 80%-power bound for d = 6 and 13 at "
          "n = 100 and 76; earlier families at n = 12, 15, 16", "0.060 0.079 0.130 0.171 0.078 0.102 0.16 0.20 0.578 0.476 0.449",
          [6 / 100, 6 / 76, 13 / 100, 13 / 76, power80(100, 6), power80(76, 6), power80(100, 13), power80(76, 13),
           *(power80(n, 6) for n in (12, 15, 16))])
    check("app:patching", "smallest one-way discordant count with p <= 0.05, and with p <= 0.05/108", "6 13",
          [next(d for d in range(1, 50) if mcnemar(d, 0) <= q) for q in (0.05, 0.05 / 108)])
    dep_w = base[(base.recipient == "W") & (base.label == 1)].groupby("model").matches_h1
    share, count = dep_w.mean()[M], dep_w.sum()[M].astype(int)
    check("app:patching", "dependent W baselines still answering h; 80%-power bound on them, and after Bonferroni",
          "0.75 0.61 0.47 0.103 0.127 0.163 0.206 0.252 0.324",
          [*share, *(power80(n, 6) for n in count), *(power80(n, 13) for n in count)])
    rr = repro.groupby(level=["model", "label"]).mean()
    check("app:patching", "share of selected examples whose re-generated baseline reproduces the label, range", "0.42 0.74",
          [rr.min(), rr.max()])
    bwr, bgd = A[(A.cond == "patch_G_to_W") | (A.label == 0)].base, A[(A.cond == "patch_W_to_G") & (A.label == 1)].base
    check("app:patching", "unconditioned baseline h rate, range; dependent G-recipient baseline, range; "
          "patched parse rate, range", "0.47 0.86 0.02 0.12 0.82 1.00",
          [bwr.min(), bwr.max(), bgd.min(), bgd.max(), A.parse.min(), A.parse.max()])
    sig = C[C.p <= 0.05]
    check("app:patching", "conditioned population: n range; nominal cells, of them Llama; largest one-way count",
          "35 74 7 7 10", [C.n.min(), C.n.max(), len(sig), (sig.model == "llama").sum(), sig[["dn", "up"]].max(axis=1).max()])
    check("app:patching", "conditioned: largest delta and its p; matched independent control delta and p",
          "-0.238 0.002 -0.171 0.031", [sig.delta.min(), sig.p.min(), sig[sig.label == 0].delta.min(), sig[sig.label == 0].p.min()])


# ------------------------------------------------------------- 9. role-swap benchmark (app:roleswap)
def roleswap():
    oof = read("roleswap/dense_probe_oof.csv.gz")
    a = {k: auc(g.binary_label, g.oof_prob) for k, g in oof.groupby(["model", "position", "method", "selection", "slot"])}
    nest = [a[(m, "prompt_end", "W_minus_G", s, -1)] for m in M for s in ("nested_dense_layer_and_C", "nested_four_layer_and_C")]
    check("app:roleswap", "nested W-G at the prompt end, all slots vs four layers x3", "0.6955 0.6985 0.9565 0.9598 0.9906 0.9886",
          nest)
    check("app:roleswap", "largest all-slot vs four-layer difference at the prompt end, of the printed 4-dp values "
          "(0.0032 unrounded)", "0.0033", [max(abs(round(nest[i], 4) - round(nest[i + 1], 4)) for i in (0, 2, 4))])
    fixed = {(k[0], k[1], k[4]): v for k, v in a.items() if k[2:4] == ("W_minus_G", "fixed_C1.0")}
    check("tab:roleswap-lens", "best W-G probe over all slots: prompt end x3, answer position x3",
          "0.703 0.960 0.993 0.690 0.966 0.996",
          [max(v for k, v in fixed.items() if k[:2] == (m, p)) for p in ["prompt_end", "answer_pos"] for m in M])
    check("app:roleswap", "Mistral prompt end: post-norm slot 32 vs the four-layer (8/16/24/31) maximum", "0.7030 0.6816",
          [fixed[("mistral", "prompt_end", 32)], max(fixed[("mistral", "prompt_end", s)] for s in (8, 16, 24, 31))])
    e, got = read("roleswap/effect_rows.csv.gz"), []
    for m in M:
        g = e[e.model == m]
        ef, pair = g.effect_h1_w_minus_g, g.groupby("swap_pair_id").effect_h1_w_minus_g
        pm, gap = pair.transform("mean"), pair.agg(lambda v: abs(v.iloc[0] - v.iloc[-1])).mean()
        got.append([100 * (ef > 0).mean(), 100 * pair.agg(lambda v: (v > 0).all()).mean(),
                    100 * np.var(pm) / (np.var(pm) + np.var(ef - pm)), (ef < 0).sum(), 100 * gap / ef.mean(),
                    g.pair_centered_effect.std()])
    got = np.array(got)
    check("app:roleswap", "uncentered effect positive %, both orientations positive %, shared variance %, x3",
          "97.7 99.4 98.0 95.5 98.7 95.9 84.5 98.5 70.0", got[:, :3].T)
    check("app:roleswap", "negatives in 1,728 rows, range; within-pair gap as % of the shared boost, range", "11 39 16 57",
          [got[:, 3].min(), got[:, 3].max(), got[:, 4].min(), got[:, 4].max()])
    check("tab:roleswap-main", "target std x3", "0.093 0.396 0.278", got[:, 5])


def main():
    ex = {m: pd.read_csv(EXAMPLES / f"{m}.csv", dtype={"question_id": str}) for m in M}
    slices(ex)
    cells = monitors(ex)
    calibration()
    ensembles()
    difference_probes()
    nl = nonlinear()
    within_dataset(ex, nl)
    wg_transfer()
    span_labels(ex, cells)
    injection()
    patching()
    roleswap()
    numbers = [(ok, n) for kind, ok, n in RESULTS if kind == "number"]
    yes_no = [ok for kind, ok, _ in RESULTS if kind == "yes/no"]
    print(f"\n{sum(ok for ok, _ in numbers)}/{len(numbers)} checks PASS, comparing {sum(n for _, n in numbers)} "
          f"printed numbers; {sum(yes_no)}/{len(yes_no)} yes/no checks of statements PASS")
    if not all(PRECONDITIONS):
        print("PRECONDITION FAILED: see the first lines of the output.")
    sys.exit(0 if all(ok for _, ok, _ in RESULTS) and all(PRECONDITIONS) else 1)


if __name__ == "__main__":
    main()
