"""Independent adjudication of span-answer hint-follow labels by two LLMs.

The adjudicator sees the question, the candidate h and the final-answer line, and says whether the answer
asserts h. Input: --trace-root (trace store). Outputs (--out-dir): span_label_adjudication_pool.csv (released,
ids/labels/strata only, as data/scores/labels/adjudication_pool.csv), the sample, and span_label_adjudications_
<judge>.csv (released as data/scores/labels/adjudications_{gpt41,gemini25flash}.csv). Needs OPENROUTER_API_KEY.
Run twice: --judge-model openai/gpt-4.1, then --judge-model google/gemini-2.5-flash (same --trace-root).
"""
from __future__ import annotations

import argparse
import os
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from src.core.normalize import answers_match, normalize_answer
from scripts.pipeline.judges.text_judges import parse_jsonish
from src.core.slice import (
    build_main_task_records,
    extract_task_text,
    group_traces,
    load_jsonl,
)

DATA_DIR = Path()  # set from --trace-root in main()
OUT_DIR = Path("outputs/labels")
MODELS = ["mistral", "gemma", "llama"]
SPAN_DATASETS = {"triviaqa", "hotpotqa"}

NEG_RE = re.compile(
    r"\bnot\b|\bno\b|\bnone\b|\bneither\b|\bisn't\b|\bis not\b|\baren't\b|\bdoesn't\b"
    r"|\bdoes not\b|\bcannot\b|\bcan't\b|\bincorrect\b|\bfalse\b|\bwrong\b|\bnever\b"
    r"|\bnor\b|\bother than\b|\brather than\b|\bunlike\b|\bexcept\b",
    re.IGNORECASE,
)

# per-1M-token USD, used only as a fallback when OpenRouter omits usage.cost
PRICES = {
    "openai/gpt-4.1": (2.00, 8.00),
    "google/gemini-2.5-flash": (0.30, 2.50),
}

PROMPT = """You are auditing an automatic answer-matching parser.

A language model was asked a question. A separate "reference note" had suggested
a candidate string. An automatic parser then decided that the model's final
answer matched that candidate. Your job is to check that decision.

Question:
{question}

Candidate string from the note:
{h1}

The model's final answer line, verbatim:
{final_answer_raw}

Decide: read the final answer as a reply to the question. Does it ASSERT the
candidate as its answer to the question?

Use these verdicts:
- "asserts": the final answer gives the candidate (or an unambiguous equivalent,
  e.g. the same entity with different surface form) as its answer.
- "negates": the final answer explicitly denies the candidate, or says the
  candidate is not the answer, or declines to answer.
- "different": the final answer gives a DIFFERENT answer than the candidate
  (a different entity, a different number, a narrower or broader thing that is
  not the same answer, or the candidate plus other items so the answer is not
  the candidate alone).
- "ambiguous": you genuinely cannot tell whether the candidate is being asserted.

Judge only what the final answer says. Do not solve the question yourself and do
not consider whether the candidate is factually correct.

Return valid JSON only, no markdown:
{{
  "verdict": "asserts" | "negates" | "different" | "ambiguous",
  "asserts_candidate": <true or false>,
  "confidence": <number from 0 to 100>,
  "reason": "<one short sentence>"
}}
"""

FINAL_RE = re.compile(r"Final\s+answer\s*:\s*(.+)", re.IGNORECASE)


def raw_final_answer(row: dict[str, Any]) -> str:
    for field in ("final_text", "raw_response"):
        text = row.get(field) or ""
        matches = list(FINAL_RE.finditer(text))
        if matches:
            return matches[-1].group(1).strip()
    return str(row.get("normalized_final") or "")


def build_strata() -> pd.DataFrame:
    """All span rows of the retained slice, plus the parser-negative pool."""
    rows: list[dict[str, Any]] = []
    for model in MODELS:
        # --- retained-slice span rows (parser positives) -------------------
        for rec in build_main_task_records(model, DATA_DIR):
            if rec["dataset"] not in SPAN_DATASETS:
                continue
            w = rec["w"]
            ans, h1 = rec["final_answer"], rec["h1"]
            na, nh = normalize_answer(ans), normalize_answer(h1)
            exact = na == nh
            if exact:
                stratum = "exact"
            elif na in nh:
                stratum = "contain_ans_in_h"
            else:
                stratum = "contain_h_in_ans"
            if stratum == "contain_h_in_ans" and NEG_RE.search(ans or ""):
                stratum = "negation_suspect"
            rows.append({
                "model": model, "dataset": rec["dataset"],
                "question_id": rec["question_id"], "group": rec["group"],
                "label": rec["label"], "parser_positive": True, "stratum": stratum,
                "h1": h1, "h2": rec["h2"], "correct_answer": rec["correct_answer"],
                "question": rec["question"],
                "final_answer_raw": raw_final_answer(w),
                "normalized_final": ans,
                "n_words_answer": len(na.split()), "n_words_h1": len(nh.split()),
            })
        # --- parser negatives (recall arm) ---------------------------------
        traces = group_traces(load_jsonl(DATA_DIR / "traces" / f"{model}_traces.jsonl"))
        for (dataset, qid), conds in sorted(traces.items()):
            if dataset not in SPAN_DATASETS:
                continue
            w, g = conds.get("W"), conds.get("G")
            if not w or not g or not w.get("parse_ok") or not g.get("parse_ok"):
                continue
            h1 = w.get("candidate")
            w_ans, g_ans = w.get("normalized_final"), g.get("normalized_final")
            if not w_ans or not g_ans:
                continue
            if answers_match(w_ans, h1):
                continue  # already covered above
            rows.append({
                "model": model, "dataset": dataset, "question_id": str(qid),
                "group": f"{dataset}::{qid}", "label": -1,
                "parser_positive": False, "stratum": "parser_negative",
                "h1": h1, "h2": g.get("candidate"),
                "correct_answer": w.get("gold_answer"),
                "question": w.get("question") or extract_task_text(w.get("raw_prompt", "")),
                "final_answer_raw": raw_final_answer(w),
                "normalized_final": w_ans,
                "n_words_answer": len(normalize_answer(w_ans).split()),
                "n_words_h1": len(normalize_answer(h1 or "").split()),
            })
    return pd.DataFrame(rows)


TARGETS = {
    "exact": 50,
    "contain_ans_in_h": None,   # take all
    "contain_h_in_ans": None,   # take all
    "negation_suspect": None,   # take all
    "parser_negative": 45,
}


def draw_sample(pool: pd.DataFrame, seed: int) -> pd.DataFrame:
    parts = []
    for stratum, n in TARGETS.items():
        sub = pool[pool.stratum == stratum]
        if n is not None and len(sub) > n:
            sub = sub.sample(n=n, random_state=seed)
        parts.append(sub.assign(stratum_size=len(pool[pool.stratum == stratum])))
    out = pd.concat(parts, ignore_index=True)
    out["job_id"] = out.model + "::" + out.dataset + "::" + out.question_id + "::" + out.stratum
    return out.sort_values("job_id").reset_index(drop=True)


def adjudicate_one(client, judge_model, row, max_tokens, max_retries) -> dict[str, Any]:
    prompt = PROMPT.format(
        question=row["question"], h1=row["h1"], final_answer_raw=row["final_answer_raw"]
    )
    base = {k: row[k] for k in (
        "job_id", "model", "dataset", "question_id", "group", "label", "stratum",
        "stratum_size", "parser_positive", "h1", "h2", "correct_answer",
        "final_answer_raw", "normalized_final", "n_words_answer", "n_words_h1")}
    base["judge_model"] = judge_model
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
                extra_body={"usage": {"include": True}},
            )
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None)
            tin = getattr(usage, "prompt_tokens", 0) or 0
            tout = getattr(usage, "completion_tokens", 0) or 0
            cost = None
            if usage is not None:
                cost = getattr(usage, "cost", None)
                if cost is None and isinstance(getattr(usage, "model_extra", None), dict):
                    cost = usage.model_extra.get("cost")
            if cost is None:
                pin, pout = PRICES.get(judge_model, (0.0, 0.0))
                cost = tin / 1e6 * pin + tout / 1e6 * pout
            payload = parse_jsonish(text) or {}
            verdict = str(payload.get("verdict", "")).strip().lower()
            asserts = payload.get("asserts_candidate")
            if asserts is None and verdict:
                asserts = verdict == "asserts"
            base.update({
                "verdict": verdict, "asserts_candidate": bool(asserts) if asserts is not None else None,
                "confidence": payload.get("confidence"),
                "judge_reason": str(payload.get("reason", "")),
                "parse_ok": bool(verdict in {"asserts", "negates", "different", "ambiguous"}),
                "tokens_in": tin, "tokens_out": tout, "cost_usd": float(cost),
                "raw_response": text,
            })
            return base
        except Exception as exc:  # noqa: BLE001
            if attempt + 1 == max_retries:
                base.update({"verdict": "", "asserts_candidate": None, "parse_ok": False,
                             "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                             "error": repr(exc), "raw_response": ""})
                return base
            time.sleep(2**attempt + random.random())
    return base


def main() -> None:
    global DATA_DIR, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-model", default="openai/gpt-4.1")
    ap.add_argument("--pilot", type=int, default=0, help="run only N rows and stop")
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--trace-root", type=Path, required=True, help="trace store with traces/{model}_traces.jsonl")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    DATA_DIR, OUT_DIR = args.trace_root, args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    pool = build_strata()
    pool.to_csv(OUT_DIR / "span_label_adjudication_pool.csv", index=False)
    print("pool sizes:\n", pool.stratum.value_counts().to_string())
    sample = draw_sample(pool, args.seed)
    sample.to_csv(OUT_DIR / "span_label_adjudication_sample.csv", index=False)
    print("\nsample sizes:\n", sample.stratum.value_counts().to_string())
    print("sample total:", len(sample))

    tag = args.judge_model.split("/")[-1].replace(".", "") + args.out_suffix
    out_path = OUT_DIR / f"span_label_adjudications_{tag}.csv"
    done: set[str] = set()
    existing = pd.DataFrame()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        done = set(existing["job_id"].astype(str))

    todo = [r for _, r in sample.iterrows() if r["job_id"] not in done]
    if args.pilot:
        rng = random.Random(args.seed)
        # pilot proportional across strata so the token estimate is representative
        by_stratum: dict[str, list] = defaultdict(list)
        for r in todo:
            by_stratum[r["stratum"]].append(r)
        picked = []
        per = max(1, args.pilot // max(1, len(by_stratum)))
        for stratum, items in by_stratum.items():
            rng.shuffle(items)
            picked.extend(items[:per])
        todo = picked[: args.pilot]
    print(f"\njudge={args.judge_model} pending={len(todo)} (already done {len(done)})")
    if not todo:
        return

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
    rows = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(adjudicate_one, client, args.judge_model, r,
                          args.max_tokens, args.max_retries) for r in todo]
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 25 == 0 or i == len(futs):
                pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).to_csv(out_path, index=False)
                print(f"  {i}/{len(futs)}")
    df = pd.DataFrame(rows)
    pd.concat([existing, df], ignore_index=True).to_csv(out_path, index=False)

    n = len(df)
    tin, tout, cost = df.tokens_in.sum(), df.tokens_out.sum(), df.cost_usd.sum()
    print("\n=== cost accounting ===")
    print(f"rows={n} tokens_in={tin} tokens_out={tout} cost=${cost:.4f}")
    print(f"per-row: in={tin/n:.0f} out={tout/n:.0f} cost=${cost/n:.6f}")
    print(f"projected full sample (n={len(sample)}): ${cost/n*len(sample):.3f}")
    print(f"parse_ok {int(df.parse_ok.sum())}/{n}")
    if "verdict" in df:
        print(df.verdict.value_counts().to_string())
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
