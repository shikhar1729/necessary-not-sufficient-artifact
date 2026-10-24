"""Parser precision and recall against the two adjudicators, and the exact-match relabelling test (app:corrected,
tab:span-relabel). Input: --adjudication-dir (outputs of judges/span_label_adjudication.py, released as
data/scores/labels/adjudications_*.csv and adjudication_pool.csv), --trace-root, per-row scores in data/scores/.
Output: --out-dir CSVs; span_label_exactmatch_slice.csv is released as data/scores/labels/exact_match_slice.csv.
One RNG stream (seed 1729) is consumed in a fixed order, so the intervals depend on running the parts in sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.core.normalize import answers_match, normalize_answer
from src.core.slice import group_traces, load_jsonl

ROOT = Path(__file__).resolve().parents[3]
SCORES = ROOT / "data" / "scores"
DATA_DIR = Path()  # set from --trace-root in main()
ADJ = Path("outputs/labels")
OUT_DIR = Path("outputs/labels")
MODELS = ["mistral", "gemma", "llama"]
SPAN = {"triviaqa", "hotpotqa"}
STRATA = ["exact", "contain_ans_in_h", "contain_h_in_ans", "negation_suspect"]
RNG = np.random.default_rng(1729)
NBOOT = 4000


# ───────────────────────── (b) precision / recall ─────────────────────────────
def load_adj(tag: str) -> pd.DataFrame:
    df = pd.read_csv(ADJ / f"span_label_adjudications_{tag}.csv")
    df = df.drop_duplicates(subset=["job_id"], keep="last")
    df["asserts"] = df.verdict.eq("asserts").astype(int)
    return df


def weighted_pr(df: pd.DataFrame, pos_sizes: dict[str, int], neg_size: int,
                boot: bool = False) -> dict[str, float]:
    """Stratum-weighted parser precision and recall over the span population."""
    def frac(stratum: str) -> float:
        sub = df[df.stratum == stratum]
        a = sub.asserts.to_numpy()
        if boot and len(a):
            a = a[RNG.integers(0, len(a), len(a))]
        return float(a.mean()) if len(a) else np.nan

    npos = sum(pos_sizes.values())
    tp = sum(pos_sizes[s] * frac(s) for s in STRATA if pos_sizes[s] > 0)
    precision = tp / npos if npos else np.nan
    q_neg = frac("parser_negative") if neg_size > 0 else 0.0
    fn = neg_size * q_neg if neg_size > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    return {"precision": precision, "recall": recall, "tp": tp,
            "fp": npos - tp, "fn": fn, "q_neg": q_neg}


def ci(vals: list[float]) -> tuple[float, float]:
    arr = np.asarray([v for v in vals if np.isfinite(v)])
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def part_b() -> dict:
    pool = pd.read_csv(ADJ / "span_label_adjudication_pool.csv")
    out: dict = {"judges": {}}
    for tag in ("gpt-41", "gemini-25-flash"):
        adj = load_adj(tag)
        rows = []
        for scope in ["ALL"] + MODELS:
            sub = adj if scope == "ALL" else adj[adj.model == scope]
            psub = pool if scope == "ALL" else pool[pool.model == scope]
            pos_sizes = {s: int((psub.stratum == s).sum()) for s in STRATA}
            neg_size = int((psub.stratum == "parser_negative").sum())
            point = weighted_pr(sub, pos_sizes, neg_size)
            boots = [weighted_pr(sub, pos_sizes, neg_size, boot=True) for _ in range(NBOOT)]
            pl, ph = ci([b["precision"] for b in boots])
            rl, rh = ci([b["recall"] for b in boots])
            rows.append({
                "judge": tag, "scope": scope,
                "n_adjudicated": len(sub), "n_span_positives": sum(pos_sizes.values()),
                "n_span_negatives": neg_size,
                "precision": point["precision"], "precision_lo": pl, "precision_hi": ph,
                "recall": point["recall"], "recall_lo": rl, "recall_hi": rh,
                "est_false_positives": point["fp"], "est_false_negatives": point["fn"],
            })
        tbl = pd.DataFrame(rows)
        out["judges"][tag] = tbl
        # per-stratum assert rates
        st = (adj.groupby("stratum")
                 .agg(n=("asserts", "size"), n_asserts=("asserts", "sum"),
                      assert_rate=("asserts", "mean"))
                 .reindex(STRATA + ["parser_negative"]))
        st["pool_size"] = [int((pool.stratum == s).sum()) for s in st.index]
        out[f"strata_{tag}"] = st
        out[f"verdicts_{tag}"] = pd.crosstab(adj.stratum, adj.verdict)

    a = load_adj("gpt-41").set_index("job_id")
    b = load_adj("gemini-25-flash").set_index("job_id")
    common = a.index.intersection(b.index)
    agree_v = float((a.loc[common].verdict == b.loc[common].verdict).mean())
    agree_a = float((a.loc[common].asserts == b.loc[common].asserts).mean())
    pa, pb = a.loc[common].asserts.mean(), b.loc[common].asserts.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    out["agreement"] = {"n": len(common), "verdict_agreement": agree_v,
                        "asserts_agreement": agree_a,
                        "cohen_kappa_asserts": (agree_a - pe) / (1 - pe)}
    out["disagreements"] = pd.DataFrame({
        "stratum": a.loc[common].stratum, "h1": a.loc[common].h1,
        "final_answer_raw": a.loc[common].final_answer_raw,
        "gpt41": a.loc[common].verdict, "gemini": b.loc[common].verdict,
    }).query("gpt41 != gemini")
    return out


# ───────────────────────── (c) relabel sensitivity ────────────────────────────
def match_strict_span(answer, target, dataset: str) -> bool:
    """Exact-match-only for span datasets; unchanged MCQ-letter rule elsewhere."""
    if dataset in SPAN:
        a, t = normalize_answer(answer or ""), normalize_answer(target or "")
        return bool(a) and bool(t) and a == t
    return answers_match(answer or "", target or "")


def rebuild(exact_span: bool) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        traces = group_traces(load_jsonl(DATA_DIR / "traces" / f"{model}_traces.jsonl"))
        for (dataset, qid), conds in sorted(traces.items()):
            w, g = conds.get("W"), conds.get("G")
            if not w or not g or not w.get("parse_ok") or not g.get("parse_ok"):
                continue
            h1 = w.get("candidate")
            w_ans, g_ans = w.get("normalized_final"), g.get("normalized_final")
            m = (lambda a, t: match_strict_span(a, t, dataset)) if exact_span else \
                (lambda a, t: answers_match(a or "", t or ""))
            if not w_ans or not m(w_ans, h1):
                continue
            if m(w_ans, w.get("gold_answer")):
                continue
            if not g_ans:
                continue
            rows.append({"model": model, "dataset": dataset, "question_id": str(qid),
                         "group": f"{dataset}::{qid}", "label": int(not m(g_ans, h1)),
                         "answer_kind": "span" if dataset in SPAN else "mcq"})
    return pd.DataFrame(rows)


def boot_auroc(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> tuple[float, float, float]:
    uniq = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in uniq}
    point = roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    vals = []
    for _ in range(2000):
        pick = RNG.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy, pp = y[idx], p[idx]
        if len(np.unique(yy)) > 1:
            vals.append(roc_auc_score(yy, pp))
    lo, hi = (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)
    return point, float(lo), float(hi)


def part_c() -> dict:
    old = rebuild(exact_span=False)
    new = rebuild(exact_span=True)
    oof = pd.read_csv(SCORES / "probes" / "nested_probe_oof.csv")
    oof["key"] = oof.model + "||" + oof.group

    old["key"] = old.model + "||" + old.group
    new["key"] = new.model + "||" + new.group
    newmap = new.set_index("key").label.to_dict()

    comp = []
    for scope in MODELS:
        o, n = old[old.model == scope], new[new.model == scope]
        comp.append({"model": scope,
                     "N_lenient_span_rule": len(o), "dep_lenient": int(o.label.sum()),
                     "indep_lenient": int((1 - o.label).sum()),
                     "N_exact_span_rule": len(n), "dep_exact": int(n.label.sum()),
                     "indep_exact": int((1 - n.label).sum()),
                     "span_rows_exact": int((n.answer_kind == "span").sum()),
                     "mcq_rows_exact": int((n.answer_kind == "mcq").sum()),
                     "rows_dropped": len(set(o.key) - set(n.key)),
                     "rows_added": len(set(n.key) - set(o.key)),
                     "labels_flipped_on_survivors": int(sum(
                         1 for k in set(o.key) & set(n.key)
                         if o.set_index("key").label[k] != newmap[k]))})
    comp = pd.DataFrame(comp)

    rows = []
    for scope in MODELS:
        sub = oof[oof.model == scope].copy()
        y_old = sub.label.to_numpy(int)
        p = sub.oof_prob.to_numpy(float)
        g = sub.group.to_numpy()
        pt, lo, hi = boot_auroc(y_old, p, g)
        rows.append({"model": scope, "slice": "full strict slice (all rows, old label)",
                     "n": len(sub), "n_dep": int(y_old.sum()), "auroc": pt, "lo": lo, "hi": hi})

        surv = sub[sub.key.isin(newmap)].copy()
        surv["new_label"] = surv.key.map(newmap)
        for name, col in (("survivors, OLD label", "label"), ("survivors, NEW label", "new_label")):
            y = surv[col].to_numpy(int)
            pt, lo, hi = boot_auroc(y, surv.oof_prob.to_numpy(float), surv.group.to_numpy())
            rows.append({"model": scope, "slice": name, "n": len(surv),
                         "n_dep": int(y.sum()), "auroc": pt, "lo": lo, "hi": hi})
    auroc = pd.DataFrame(rows)

    # ---- the same relabel test applied to existing per-row text monitors ----
    monitors: list[pd.DataFrame] = []
    na = SCORES / "monitors" / "note_aware_judge_rows.csv.gz"
    if na.exists():
        d = pd.read_csv(na)
        d["monitor"] = d.judge_model.str.split("/").str[-1] + " / " + d.view
        monitors.append(d[["model", "dataset", "question_id", "label", "prob_dependent",
                           "parse_ok", "monitor"]])
    rob = SCORES / "monitors" / "judge_robustness_rows.csv.gz"
    if rob.exists():
        d = pd.read_csv(rob)
        d["monitor"] = d.judge_model.str.split("/").str[-1] + " / " + d.view
        monitors.append(d[["model", "dataset", "question_id", "label", "prob_dependent",
                           "parse_ok", "monitor"]])

    mrows = []
    if monitors:
        md = pd.concat(monitors, ignore_index=True)
        md = md[md.parse_ok.astype(bool)].copy()
        md["group"] = md.dataset + "::" + md.question_id.astype(str)
        md["key"] = md.model + "||" + md.group
        for (mon, scope), sub in md.groupby(["monitor", "model"]):
            if len(sub) < 100:
                continue
            pt, lo, hi = boot_auroc(sub.label.to_numpy(int),
                                    sub.prob_dependent.to_numpy(float), sub.group.to_numpy())
            mrows.append({"monitor": mon, "model": scope, "slice": "full strict slice",
                          "n": len(sub), "auroc": pt, "lo": lo, "hi": hi})
            surv = sub[sub.key.isin(newmap)].copy()
            surv["new_label"] = surv.key.map(newmap)
            for name, col in (("survivors, OLD label", "label"),
                              ("survivors, NEW label", "new_label")):
                pt, lo, hi = boot_auroc(surv[col].to_numpy(int),
                                        surv.prob_dependent.to_numpy(float),
                                        surv.group.to_numpy())
                mrows.append({"monitor": mon, "model": scope, "slice": name,
                              "n": len(surv), "auroc": pt, "lo": lo, "hi": hi})
    monitor_tbl = pd.DataFrame(mrows)
    return {"composition": comp, "auroc": auroc, "monitors": monitor_tbl,
            "new_slice": new, "old_slice": old}


def main() -> None:
    global DATA_DIR, ADJ, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", type=Path, required=True, help="trace store with traces/{model}_traces.jsonl")
    ap.add_argument("--adjudication-dir", type=Path, default=ADJ, help="outputs of span_label_adjudication.py")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    DATA_DIR, ADJ, OUT_DIR = args.trace_root, args.adjudication_dir, args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    b = part_b()
    print("=" * 78, "\n(b) PARSER PRECISION / RECALL\n")
    for tag in ("gpt-41", "gemini-25-flash"):
        print(f"--- per-stratum assert rate [{tag}] ---")
        print(b[f"strata_{tag}"].to_string())
        print(f"\n--- verdict breakdown [{tag}] ---")
        print(b[f"verdicts_{tag}"].to_string())
        print(f"\n--- weighted precision / recall [{tag}] ---")
        print(b["judges"][tag].round(4).to_string(index=False))
        print()
    print("--- adjudicator agreement ---")
    print(json.dumps(b["agreement"], indent=2))
    print(f"\n--- {len(b['disagreements'])} disagreements ---")
    print(b["disagreements"].to_string(max_colwidth=70))

    pd.concat([b["judges"][t] for t in ("gpt-41", "gemini-25-flash")], ignore_index=True) \
        .to_csv(OUT_DIR / "span_label_parser_precision_recall.csv", index=False)
    pd.concat([b[f"strata_{t}"].assign(judge=t) for t in ("gpt-41", "gemini-25-flash")]) \
        .to_csv(OUT_DIR / "span_label_stratum_assert_rates.csv")
    b["disagreements"].to_csv(OUT_DIR / "span_label_adjudicator_disagreements.csv")

    c = part_c()
    print("\n" + "=" * 78, "\n(c) RELABEL SENSITIVITY (exact-match-only for span answers)\n")
    print(c["composition"].to_string(index=False))
    print()
    print(c["auroc"].round(4).to_string(index=False))
    if not c["monitors"].empty:
        print("\n--- same relabel test on existing per-row text monitors ---")
        piv = c["monitors"].pivot_table(index=["monitor", "model"], columns="slice",
                                       values="auroc").round(4)
        print(piv.to_string())
        c["monitors"].to_csv(OUT_DIR / "span_label_relabel_monitor_auroc.csv", index=False)
    c["composition"].to_csv(OUT_DIR / "span_label_relabel_composition.csv", index=False)
    c["auroc"].to_csv(OUT_DIR / "span_label_relabel_probe_auroc.csv", index=False)
    c["new_slice"].to_csv(OUT_DIR / "span_label_exactmatch_slice.csv", index=False)


if __name__ == "__main__":
    main()
