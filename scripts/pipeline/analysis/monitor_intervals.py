"""AUROC intervals for every monitor cell and paired probe-minus-monitor differences on identical rows
(tab:monitor-intervals, tab:monitor-paired). Input: per-row scores in data/scores/ and --trace-root (traces/ for
the slice, and the Claude 3.5 Haiku files judge_{view}_{model}.csv, released together as
data/scores/monitors/haiku35_judge_rows.csv). Output: --out-dir/monitor_cell_auroc_ci.csv, paired_auroc_deltas.csv.
Row resampling, B=2000, seed 0; every cluster holds one row, so this equals question-cluster resampling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.slice import build_main_task_records

ROOT = Path(__file__).resolve().parents[3]
SCORES = ROOT / "data" / "scores"
RESULTS = Path()  # set from --trace-root in main()
OUT = ROOT / "outputs" / "monitors"

MODELS = ["mistral", "gemma", "llama"]
B = 2000
SEED = 0

# ---------------------------------------------------------------- fast AUROC


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Tie-aware AUROC via the rank-sum identity."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    order = np.argsort(s, kind="mergesort")
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def resamples(n: int, b: int = B, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def ci(values: list[float]) -> tuple[float, float, int]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan, 0)
    return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)), len(arr))


# ---------------------------------------------------------------- cell loading

# cell key -> (family, access level, display label)
META: dict[str, tuple[str, str, str]] = {}


def register(key: str, family: str, access: str, label: str) -> None:
    META[key] = (family, access, label)


def load_cells() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    def add(key: str, df: pd.DataFrame, family: str, access: str, label: str) -> None:
        d = df[["model", "group", "label", "score"]].dropna(subset=["score"]).copy()
        d["cell"] = key
        register(key, family, access, label)
        frames.append(d)

    # ---- the strict slice, for validation and for the label of record
    slice_rows = []
    for m in MODELS:
        for r in build_main_task_records(m, RESULTS):
            slice_rows.append(
                {"model": m, "group": r["group"], "question_id": r["question_id"], "label": int(r["label"])}
            )
    sl = pd.DataFrame(slice_rows)

    # ---- internal probe (the paper's headline probe)
    p = pd.read_csv(SCORES / "probes" / "nested_probe_oof.csv")
    p = p.rename(columns={"oof_prob": "score"})
    add("probe_nested", p, "internal probe", "residual stream", "Probe (nested, best layer)")

    # ---- judge-robustness cells
    j19 = pd.read_csv(SCORES / "monitors" / "judge_robustness_rows.csv.gz")
    j19 = j19[j19["parse_ok"].astype(bool)]
    j19 = j19.rename(columns={"prob_dependent": "score"})
    fam = {
        "sonnet4_cot_generic": ("generic text judge", "CoT+answer", "Sonnet-4 CoT (generic, re-run)"),
        "gpt41mini_cot_generic": ("generic text judge", "CoT+answer", "GPT-4.1-mini CoT (generic)"),
        "gemini25flash_cot_generic": ("generic text judge", "CoT+answer", "Gemini-2.5-flash CoT (generic)"),
        "sonnet4_cot_cf": ("counterfactual judge", "CoT+answer+h'", "Sonnet-4 CoT-CF (original prompt, re-run)"),
        "sonnet4_cot_cf_p1": ("counterfactual judge", "CoT+answer+h'", "Sonnet-4 CoT-CF paraphrase P1"),
        "sonnet4_cot_cf_p2": ("counterfactual judge", "CoT+answer+h'", "Sonnet-4 CoT-CF paraphrase P2"),
        "gpt41mini_cot_cf": ("counterfactual judge", "CoT+answer+h'", "GPT-4.1-mini CoT-CF"),
        "gpt41mini_cot_cf_fewshot": ("counterfactual judge", "CoT+answer+h'+demos", "GPT-4.1-mini CoT-CF few-shot"),
        "gemini25flash_cot_cf": ("counterfactual judge", "CoT+answer+h'", "Gemini-2.5-flash CoT-CF"),
    }
    for cell, (family, access, label) in fam.items():
        add(cell, j19[j19["cell"] == cell], family, access, label)

    # ---- judge-robustness ensembles (mean probability, recomputed here)
    def mean_ens(cells: list[str], key: str, family: str, access: str, label: str) -> None:
        sub = j19[j19["cell"].isin(cells)]
        piv = sub.pivot_table(index=["model", "group", "label"], columns="cell", values="score")
        piv = piv.dropna()
        d = piv.mean(axis=1).rename("score").reset_index()
        add(key, d, family, access, label)

    mean_ens(
        ["sonnet4_cot_cf", "sonnet4_cot_cf_p1", "sonnet4_cot_cf_p2"],
        "ens_sonnet4_cf_paraphrase",
        "counterfactual judge",
        "CoT+answer+h'",
        "Sonnet-4 CoT-CF 3-paraphrase mean",
    )
    mean_ens(
        ["sonnet4_cot_cf", "gpt41mini_cot_cf", "gemini25flash_cot_cf"],
        "ens_crossfamily_cf",
        "counterfactual judge",
        "CoT+answer+h'",
        "CoT-CF cross-family mean (3 judges)",
    )
    mean_ens(
        ["sonnet4_cot_generic", "gpt41mini_cot_generic", "gemini25flash_cot_generic"],
        "ens_crossfamily_generic",
        "generic text judge",
        "CoT+answer",
        "Generic CoT cross-family mean (3 judges)",
    )

    # ---- note-aware judges
    na = pd.read_csv(SCORES / "monitors" / "note_aware_judge_rows.csv.gz")
    na = na[na["parse_ok"].astype(bool)].rename(columns={"prob_dependent": "score"})
    na["group"] = na["dataset"].astype(str) + "::" + na["question_id"].astype(str)
    add(
        "sonnet4_note_aware_cot",
        na[na["view"] == "note_aware_no_hprime"],
        "note-aware judge",
        "note+CoT+answer",
        "Sonnet-4 note-aware CoT (no h')",
    )
    add(
        "sonnet4_note_aware_answer",
        na[na["view"] == "answer_note_aware_no_hprime"],
        "note-aware judge",
        "note+answer",
        "Sonnet-4 note-aware answer-only (no h')",
    )

    # ---- Sonnet-4 re-run cells: candidate prior + answer-CF
    nj = pd.read_csv(SCORES / "monitors" / "prior_and_answer_cf_judge_rows.csv.gz")
    nj = nj[nj["parse_ok"].astype(bool)].rename(columns={"prob_dependent": "score"})
    nj["group"] = nj["dataset"].astype(str) + "::" + nj["question_id"].astype(str)
    add(
        "sonnet4_candidate_prior",
        nj[nj["view"] == "candidate_prior"],
        "prior-only judge",
        "question+h+h' (no model output)",
        "Sonnet-4 candidate-prior (re-run)",
    )
    add(
        "sonnet4_answer_cf",
        nj[nj["view"] == "answer_counterfactual"],
        "counterfactual judge",
        "answer+h'",
        "Sonnet-4 answer-CF (re-run)",
    )

    # ---- Haiku judge family (anthropic/claude-3.5-haiku)
    qid_to_group = {
        (r.model, r.question_id): r.group for r in sl.itertuples()
    }
    for view, label in [
        ("answer_only", "Haiku answer-only"),
        ("cot_answer", "Haiku CoT+answer"),
        ("cot_followup", "Haiku CoT+follow-up"),
    ]:
        rows = []
        for m in MODELS:
            h = pd.read_csv(RESULTS / f"judge_{view}_{m}.csv")
            h["model"] = m
            h["group"] = [qid_to_group.get((m, str(q))) for q in h["question_id"]]
            rows.append(h.dropna(subset=["group"]))
        d = pd.concat(rows, ignore_index=True).rename(columns={"judge_prob": "score"})
        # relabel with the corrected strict labels
        d = d.drop(columns=["label"], errors="ignore").merge(
            sl[["model", "group", "label"]], on=["model", "group"], how="inner"
        )
        access = "CoT+answer+follow-up" if view == "cot_followup" else (
            "answer" if view == "answer_only" else "CoT+answer"
        )
        add(f"haiku_{view}", d, "Haiku text judge", access, label)

    # ---- TF-IDF text baselines (per-view, plus best-view-per-model)
    tf = pd.read_csv(SCORES / "monitors" / "tfidf_oof.csv")
    for view in ["answer_only", "cot_answer", "prompt_answer", "prompt_cot_answer"]:
        add(
            f"tfidf_{view}",
            tf[tf["view"] == view],
            "TF-IDF text baseline",
            view.replace("_", "+"),
            f"TF-IDF {view}",
        )
    best_rows = []
    for m in MODELS:
        sub = tf[tf["model"] == m]
        scores = {v: auroc(g["label"].to_numpy(int), g["score"].to_numpy(float)) for v, g in sub.groupby("view")}
        best = max(scores, key=lambda v: scores[v])
        print(f"  TF-IDF best view for {m}: {best} ({scores[best]:.4f})")
        best_rows.append(sub[sub["view"] == best])
    add(
        "tfidf_best",
        pd.concat(best_rows, ignore_index=True),
        "TF-IDF text baseline",
        "best of 4 views",
        "TF-IDF best view per model (selected on outcome)",
    )

    cells = pd.concat(frames, ignore_index=True)
    # validate labels against the slice of record
    chk = cells.merge(sl[["model", "group", "label"]], on=["model", "group"], suffixes=("", "_ref"))
    bad = int((chk["label"] != chk["label_ref"]).sum())
    print(f"label mismatches vs strict slice: {bad} / {len(chk)}")
    orphans = len(cells) - len(chk)
    print(f"rows not in strict slice: {orphans}")
    return cells, sl


# ---------------------------------------------------------------- metrics


def cell_table(cells: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cell, model), g in cells.groupby(["cell", "model"]):
        y = g["label"].to_numpy(int)
        s = g["score"].to_numpy(float)
        point = auroc(y, s)
        idx = resamples(len(y))
        boot = [auroc(y[i], s[i]) for i in idx]
        lo, hi, nb = ci(boot)
        family, access, label = META[cell]
        rows.append(
            {
                "cell": cell,
                "label": label,
                "family": family,
                "access": access,
                "model": model,
                "n": len(y),
                "n_dep": int(y.sum()),
                "base_rate": float(y.mean()),
                "auroc": point,
                "ci_low": lo,
                "ci_high": hi,
                "ci_width": hi - lo,
                "excludes_0.5": bool(lo > 0.5),
                "n_boot_valid": nb,
                "n_distinct_scores": int(pd.Series(s).nunique()),
            }
        )
    return pd.DataFrame(rows)


def paired(cells: pd.DataFrame, a: str, b: str, model: str, restrict: set | None = None) -> dict:
    """AUROC(a) - AUROC(b) on the intersection of their rows, paired bootstrap."""
    da = cells[(cells["cell"] == a) & (cells["model"] == model)].set_index("group")
    db = cells[(cells["cell"] == b) & (cells["model"] == model)].set_index("group")
    common = sorted(set(da.index) & set(db.index))
    if restrict is not None:
        common = sorted(set(common) & restrict)
    if len(common) < 10:
        return {}
    y = da.loc[common, "label"].to_numpy(int)
    sa = da.loc[common, "score"].to_numpy(float)
    sb = db.loc[common, "score"].to_numpy(float)
    pa, pb = auroc(y, sa), auroc(y, sb)
    idx = resamples(len(common))
    deltas = []
    for i in idx:
        va, vb = auroc(y[i], sa[i]), auroc(y[i], sb[i])
        if np.isfinite(va) and np.isfinite(vb):
            deltas.append(va - vb)
    lo, hi, nb = ci(deltas)
    arr = np.asarray(deltas)
    return {
        "model": model,
        "cell_a": a,
        "cell_b": b,
        "label_a": META[a][2],
        "label_b": META[b][2],
        "n_common": len(common),
        "auroc_a": pa,
        "auroc_b": pb,
        "delta": pa - pb,
        "ci_low": lo,
        "ci_high": hi,
        "includes_zero": bool(lo <= 0.0 <= hi),
        "p_two_sided_boot": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())) if len(arr) else np.nan,
        "n_boot_valid": nb,
    }


def main() -> None:
    global RESULTS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", type=Path, required=True,
                    help="trace store with traces/ and judge_{view}_{model}.csv")
    ap.add_argument("--out-dir", type=Path, default=OUT)
    args = ap.parse_args()
    RESULTS, OUT = args.trace_root, args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    cells, sl = load_cells()
    print("\ncells loaded:")
    print(cells.groupby(["cell", "model"]).size().unstack().to_string())

    tbl = cell_table(cells)
    tbl = tbl.sort_values(["family", "cell", "model"])
    tbl.to_csv(OUT / "monitor_cell_auroc_ci.csv", index=False)
    print("\n== per-cell AUROC with cluster-bootstrap 95% CI ==")
    print(
        tbl.pivot_table(index=["family", "cell"], columns="model", values="auroc").round(3).to_string()
    )

    judge_cells = [c for c in META if c != "probe_nested"]
    rows = []
    # 1. probe minus every monitor
    for c in judge_cells:
        for m in MODELS:
            r = paired(cells, "probe_nested", c, m)
            if r:
                r["comparison"] = "probe_minus_monitor"
                rows.append(r)
    # 2. note-aware minus generic (same judge family: Sonnet-4)
    for m in MODELS:
        for a, b, tag in [
            ("sonnet4_note_aware_cot", "sonnet4_cot_generic", "note_aware_minus_generic"),
            ("sonnet4_note_aware_answer", "haiku_answer_only", "note_aware_answer_minus_haiku_answer"),
            ("sonnet4_cot_cf", "sonnet4_note_aware_cot", "counterfactual_minus_note_aware"),
            ("ens_sonnet4_cf_paraphrase", "sonnet4_note_aware_cot", "cf_paraphrase_ens_minus_note_aware"),
            ("sonnet4_candidate_prior", "sonnet4_cot_generic", "prior_minus_generic"),
            ("sonnet4_cot_cf", "sonnet4_candidate_prior", "counterfactual_minus_prior"),
            ("sonnet4_note_aware_cot", "sonnet4_candidate_prior", "note_aware_minus_prior"),
            ("sonnet4_note_aware_cot", "sonnet4_note_aware_answer", "note_aware_cot_minus_answer"),
            ("tfidf_best", "sonnet4_cot_generic", "tfidf_best_minus_generic"),
        ]:
            r = paired(cells, a, b, m)
            if r:
                r["comparison"] = tag
                rows.append(r)
    paired_tbl = pd.DataFrame(rows)
    paired_tbl.to_csv(OUT / "paired_auroc_deltas.csv", index=False)

    print("\n== probe minus monitor (paired, same rows, same resamples) ==")
    pm = paired_tbl[paired_tbl["comparison"] == "probe_minus_monitor"]
    for m in MODELS:
        print(f"\n-- {m} --")
        sub = pm[pm["model"] == m].sort_values("delta")
        for r in sub.itertuples():
            star = "" if r.includes_zero else "  *SIGNIFICANT*"
            print(
                f"  {r.cell_b:30s} n={r.n_common:4d} probe={r.auroc_a:.3f} judge={r.auroc_b:.3f} "
                f"delta={r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]{star}"
            )

    print("\n== within-text-monitor contrasts ==")
    other = paired_tbl[paired_tbl["comparison"] != "probe_minus_monitor"]
    for r in other.sort_values(["comparison", "model"]).itertuples():
        star = "" if r.includes_zero else "  *SIGNIFICANT*"
        print(
            f"  {r.comparison:38s} {r.model:8s} n={r.n_common:4d} "
            f"{r.auroc_a:.3f} - {r.auroc_b:.3f} = {r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}]{star}"
        )

    # headline check: does the CF judge beat the probe?
    print("\n== CoT-CF judge minus probe ==")
    for m in MODELS:
        for c in ["sonnet4_cot_cf", "ens_sonnet4_cf_paraphrase", "ens_crossfamily_cf"]:
            r = paired(cells, c, "probe_nested", m)
            if r:
                print(
                    f"  {m:8s} {c:28s} {r['auroc_a']:.3f} - {r['auroc_b']:.3f} = {r['delta']:+.3f} "
                    f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] zero_in={r['includes_zero']}"
                )

    # best non-internal monitor per model, paired against the probe
    print("\n== best non-internal monitor per model, paired vs probe ==")
    for m in MODELS:
        sub = tbl[(tbl["model"] == m) & (tbl["cell"] != "probe_nested")]
        best = sub.loc[sub["auroc"].idxmax()]
        r = paired(cells, "probe_nested", best["cell"], m)
        print(
            f"  {m:8s} best={best['cell']:28s} {best['auroc']:.3f} "
            f"[{best['ci_low']:.3f},{best['ci_high']:.3f}]  probe-minus-best={r['delta']:+.3f} "
            f"[{r['ci_low']:+.3f},{r['ci_high']:+.3f}] zero_in={r['includes_zero']}"
        )

    print(f"\nwrote {OUT / 'monitor_cell_auroc_ci.csv'}")
    print(f"wrote {OUT / 'paired_auroc_deltas.csv'}")


if __name__ == "__main__":
    main()
