"""Per-row span-label features of the retained slice (app:corrected): exact vs containment-only matches, answer
length, negation cues. Input: --trace-root (trace store). Output: --out-dir/span_label_retained_slice_features.csv,
released as data/scores/labels/span_label_features.csv without norm_answer (it repeats normalized_final) and exact_match
(the negation of containment_only) and without the question and reasoning text.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.core.normalize import normalize_answer
from src.core.slice import build_main_task_records

DATA_DIR = Path()  # set from --trace-root in main()
OUT_DIR = Path("outputs/labels")
MODELS = ["mistral", "gemma", "llama"]
SPAN_DATASETS = {"triviaqa", "hotpotqa"}
MCQ_DATASETS = {"mmlu_pro", "gpqa"}

NEG_PATTERNS = [
    r"\bnot\b", r"\bno\b", r"\bnone\b", r"\bneither\b", r"\bisn't\b", r"\bis not\b",
    r"\baren't\b", r"\bdoesn't\b", r"\bdoes not\b", r"\bcannot\b", r"\bcan't\b",
    r"\bincorrect\b", r"\bfalse\b", r"\bwrong\b", r"\bnever\b", r"\bnor\b",
    r"\bother than\b", r"\brather than\b", r"\bunlike\b", r"\bexcept\b",
]
NEG_RE = re.compile("|".join(NEG_PATTERNS), re.IGNORECASE)


def exact_match(answer: str, target: str) -> bool:
    a, t = normalize_answer(answer), normalize_answer(target)
    return bool(a) and bool(t) and a == t


def main() -> None:
    global DATA_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-root", type=Path, required=True, help="trace store with traces/{model}_traces.jsonl")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    DATA_DIR, OUT_DIR = args.trace_root, args.out_dir
    all_rows = []
    for model in MODELS:
        recs = build_main_task_records(model, DATA_DIR)
        n_dep = sum(r["label"] for r in recs)
        print(f"{model}: N={len(recs)} n_dep={n_dep} n_indep={len(recs)-n_dep}")
        for r in recs:
            w = r["w"]
            ans = r["final_answer"]
            h1 = r["h1"]
            is_span = r["dataset"] in SPAN_DATASETS
            exact = exact_match(ans, h1)
            na, nh = normalize_answer(ans), normalize_answer(h1)
            all_rows.append({
                "model": model,
                "dataset": r["dataset"],
                "question_id": r["question_id"],
                "group": r["group"],
                "label": r["label"],
                "answer_kind": "span" if is_span else "mcq",
                "h1": h1,
                "h2": r["h2"],
                "correct_answer": r["correct_answer"],
                "final_answer_raw": w.get("final_text", "").split("Final answer:")[-1].strip()
                                     if "Final answer:" in (w.get("final_text") or "") else ans,
                "normalized_final": ans,
                "norm_answer": na,
                "norm_h1": nh,
                "exact_match": exact,
                "containment_only": (not exact) and (na in nh or nh in na),
                "answer_longer_than_h": len(na.split()) > len(nh.split()),
                "n_words_answer": len(na.split()),
                "n_words_h1": len(nh.split()),
                "sentence_length": len(na.split()) >= 8,
                "negation_suspect": bool(NEG_RE.search(ans or "")),
                "question": r["question"],
                "reasoning_text": r["reasoning_text"],
            })

    import pandas as pd
    df = pd.DataFrame(all_rows)
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "span_label_retained_slice_features.csv", index=False)

    print("\n== slice composition ==")
    print(df.groupby(["model", "answer_kind"]).size())
    span = df[df.answer_kind == "span"]
    print(f"\nTotal retained rows: {len(df)}; span rows: {len(span)}; mcq rows: {len(df)-len(span)}")
    print("\n== span row diagnostics (retained slice) ==")
    print(f"exact match        : {int(span.exact_match.sum())}")
    co = int(span.containment_only.sum())
    print(f"containment only   : {co} ({100*co/len(span):.1f}%)")
    for thr in (6, 7, 8, 9, 10, 12):
        n = int((span.n_words_answer >= thr).sum())
        print(f"  answer >= {thr:2d} words : {n} ({100*n/len(span):.1f}%)")
    ns = int(span.negation_suspect.sum())
    print(f"negation regex hit : {ns} ({100*ns/len(span):.1f}%)")
    print(f"negation hit among containment-only: {int(span[span.containment_only].negation_suspect.sum())}")

    print("\n== sanity: do MCQ rows ever match by containment? ==")
    mcq = df[df.answer_kind == "mcq"]
    print(f"mcq exact {int(mcq.exact_match.sum())}/{len(mcq)}; containment-only {int(mcq.containment_only.sum())}")
    print("\n== longest span answers ==")
    for _, r in span.sort_values("n_words_answer", ascending=False).head(10).iterrows():
        print(f"  [{r.model}/{r.dataset}] h1={r.h1!r} -> {r.normalized_final[:170]!r}")


if __name__ == "__main__":
    main()
