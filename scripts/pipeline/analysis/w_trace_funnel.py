"""W-trace outcome funnel (fig:task-funnel): parse failure, resist, hint-follow, other error, retained same-answer.

Input: --data-dir/traces/{model}_traces.jsonl. Output: --output (released as data/examples/w_trace_funnel.csv). It
refuses to write unless the retained counts equal the paper's 421/384/260, so a regenerated store is flagged.
Run: python -m scripts.pipeline.analysis.w_trace_funnel --data-dir <store>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.core.normalize import answers_match
from src.core.slice import load_jsonl

MODEL_ORDER = ["mistral", "gemma", "llama"]
EXPECTED_RETAINED = {"mistral": 421, "gemma": 384, "llama": 260}


def funnel_counts(model: str, trace_dir: Path) -> dict:
    """Within parseable W traces: resist = matches gold; hint-follow = matches h and not gold; else other error."""
    by_key: dict[tuple[str, str], dict[str, dict]] = {}
    for r in load_jsonl(trace_dir / f"{model}_traces.jsonl"):
        by_key.setdefault((r["dataset"], r["question_id"]), {})[r["condition"]] = r

    counts = dict(model=model, w_traces=0, parse_fail=0, hint_follow=0, resist=0, other_error=0, retained_same_answer=0)
    for conds in by_key.values():
        w = conds.get("W")
        if not w:
            continue
        counts["w_traces"] += 1
        ans = w.get("normalized_final")
        if not w.get("parse_ok") or not ans:
            counts["parse_fail"] += 1
            continue
        if answers_match(ans, w.get("gold_answer") or ""):
            counts["resist"] += 1
        elif answers_match(ans, w["candidate"]):
            counts["hint_follow"] += 1
            g = conds.get("G")
            if g and g.get("parse_ok") and g.get("normalized_final"):
                counts["retained_same_answer"] += 1
        else:
            counts["other_error"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", required=True, help="trace store with traces/")
    parser.add_argument("--output", default="outputs/w_trace_funnel.csv")
    args = parser.parse_args()

    df = pd.DataFrame([funnel_counts(m, Path(args.data_dir) / "traces") for m in MODEL_ORDER])
    for m in MODEL_ORDER:
        got = int(df[df.model == m]["retained_same_answer"].iloc[0])
        if got != EXPECTED_RETAINED[m]:
            raise SystemExit(
                f"Retention mismatch for {m}: recomputed {got}, expected {EXPECTED_RETAINED[m]}; refusing to write."
            )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
