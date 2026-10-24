"""Size-matched control for the ensembles (app:ensemble): is adding the probe to a text stack worth more than adding
one more text monitor? Same inputs and paired row bootstrap (B=2000, seed 0) as ensemble_probe_text.py.
Output: outputs/ensemble/probe_vs_text_addition_matched.csv, released as
data/scores/ensemble/probe_vs_text_addition.csv. No GPU, no API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.pipeline.analysis.ensemble_probe_text import (
    MODELS,
    OUT,
    TEXT_BASES,
    auroc,
    ci,
    combine,
    fold_ids,
    load_members,
    resamples,
)

TEXT_MONITORS = ["na_cot", "prior", "cf_para", "tfidf"]
COMBINERS = ["rank_avg", "z_avg", "stack_lr"]


def main() -> None:
    frames, best_view = load_members()
    rows = []
    for model in MODELS:
        df = frames[model]
        for base_name, base in TEXT_BASES:
            additions = [t for t in TEXT_MONITORS if t not in base]
            if not additions:
                continue
            cols = sorted(set(base) | set(additions) | {"probe"})
            sub = df[["group", "label", *cols]].dropna().reset_index(drop=True)
            if len(sub) < 20:
                continue
            y = sub["label"].to_numpy(int)
            folds = fold_ids(sub["group"].to_numpy(dtype=object))
            idx = resamples(len(y))
            Xb = sub[base].to_numpy(float)

            for comb in COMBINERS:
                s_probe = combine(comb, np.column_stack([sub["probe"].to_numpy(float), Xb]), y, folds)
                a_probe = auroc(y, s_probe)
                for t in additions:
                    s_text = combine(comb, np.column_stack([sub[t].to_numpy(float), Xb]), y, folds)
                    a_text = auroc(y, s_text)
                    deltas = []
                    for i in idx:
                        va, vb = auroc(y[i], s_probe[i]), auroc(y[i], s_text[i])
                        if np.isfinite(va) and np.isfinite(vb):
                            deltas.append(va - vb)
                    lo, hi, _ = ci(deltas)
                    arr = np.asarray(deltas)
                    rows.append(
                        {
                            "model": model,
                            "text_base": base_name,
                            "base_size": len(base),
                            "combiner": comb,
                            "text_addition": t,
                            "n": len(y),
                            "auroc_base_plus_probe": a_probe,
                            "auroc_base_plus_text": a_text,
                            "delta_probe_minus_text": a_probe - a_text,
                            "ci_low": lo,
                            "ci_high": hi,
                            "half_width": (hi - lo) / 2,
                            "includes_zero": bool(lo <= 0.0 <= hi),
                            "p_two_sided_boot": float(2 * min((arr <= 0).mean(), (arr >= 0).mean())),
                        }
                    )

    tbl = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    tbl.to_csv(OUT / "probe_vs_text_addition_matched.csv", index=False)

    print("==== size-matched: base+probe  minus  base+<text monitor> ====")
    for r in tbl.sort_values(["model", "combiner", "base_size", "text_base", "text_addition"]).itertuples():
        star = "" if r.includes_zero else "  *** EXCLUDES ZERO ***"
        print(
            f"  {r.model:8s} {r.combiner:9s} base={r.text_base:28s} +probe {r.auroc_base_plus_probe:.4f} "
            f"vs +{r.text_addition:8s} {r.auroc_base_plus_text:.4f}  d={r.delta_probe_minus_text:+.4f} "
            f"[{r.ci_low:+.4f},{r.ci_high:+.4f}]{star}"
        )

    print("\n==== summary ====")
    print(f"  comparisons: {len(tbl)}")
    print(f"  probe addition BETTER with interval excluding zero: {int(((~tbl['includes_zero']) & (tbl['delta_probe_minus_text'] > 0)).sum())}")
    print(f"  probe addition WORSE with interval excluding zero:  {int(((~tbl['includes_zero']) & (tbl['delta_probe_minus_text'] < 0)).sum())}")
    print(f"  intervals including zero: {int(tbl['includes_zero'].sum())}")
    print(f"  median delta (probe minus text addition): {tbl['delta_probe_minus_text'].median():+.4f}")
    print(f"  median paired half-width: {tbl['half_width'].median():.4f}")
    for model in MODELS:
        s = tbl[tbl["model"] == model]
        print(
            f"  {model:8s} median delta {s['delta_probe_minus_text'].median():+.4f}  "
            f"positive {int((s['delta_probe_minus_text'] > 0).sum())}/{len(s)}  "
            f"excluding zero {int((~s['includes_zero']).sum())}"
        )
    print(f"\nwrote {OUT / 'probe_vs_text_addition_matched.csv'}")


if __name__ == "__main__":
    main()
