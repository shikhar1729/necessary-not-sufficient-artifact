"""Re-score the injection factorial's cot_answer judge with anthropic/claude-sonnet-4.5 (Sonnet 4, the original
run's judge, is no longer served): same view, temperature 0, each retained W trace judged once and cached. It
first checks its records against data/scores/injection/haiku45_judge_scores.csv. Input: --factorial-dir.
Output (--out-dir): summary and injection_factorial_claude-sonnet-4.5_scores.csv, released as
data/scores/injection/sonnet45_judge_scores.csv without the constant judge_model column. Needs OPENROUTER_API_KEY.
Recorded run: once with the defaults, then once with --retry-unparsed --max-tokens 800.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.pipeline.injection.analyze import (
    CROSS_CELL_PAIRS,
    build_cross_cell_records,
    build_records,
)
from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.main_task import cluster_bootstrap_metric
from src.core.slice import load_jsonl

MODELS = ["mistral", "gemma", "llama"]
CELLS = ["adv_warn", "adv_nowarn", "ben_warn", "ben_nowarn"]
ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factorial-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot", type=int, default=0, help="judge only this many traces")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--judge-model", default="anthropic/claude-sonnet-4.5")
    ap.add_argument("--max-tokens", type=int, default=0, help="0 = config value (200)")
    ap.add_argument("--retry-unparsed", action="store_true",
                    help="re-judge cached rows whose reply had no PROBABILITY line")
    args = ap.parse_args()
    fdir, out = Path(args.factorial_dir), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- records -------------------------------------------------------------
    traces: dict[str, dict] = {}  # trace key -> judge example
    cell_recs, cross_recs = [], []
    for m in MODELS:
        rows = load_jsonl(fdir / "traces" / f"{m}_traces.jsonl")
        for cell in CELLS:
            for r in build_records(rows, cell):
                k = f"{m}|{cell}::{r['group']}"
                traces[k] = {"question_id": k, "question": r["question"],
                             "reasoning_text": r["reasoning_text"], "final_answer": r["final_answer"]}
                cell_recs.append({"model": m, "unit": cell, "group": r["group"],
                                  "label": int(r["label"]), "key": k})
        for t, c, _ in CROSS_CELL_PAIRS:
            for r in build_cross_cell_records(rows, t, c):
                k = f"{m}|{t}::{r['group']}"
                ex = {"question_id": k, "question": r["question"],
                      "reasoning_text": r["reasoning_text"], "final_answer": r["final_answer"]}
                if k in traces:
                    assert traces[k]["reasoning_text"] == ex["reasoning_text"], k
                traces[k] = ex
                cross_recs.append({"model": m, "unit": f"{t}_vs_{c}", "group": r["group"],
                                   "label": int(r["label"]), "key": k})
    cell_df, cross_df = pd.DataFrame(cell_recs), pd.DataFrame(cross_recs)

    # ---- check against the released Haiku 4.5 per-row scores --------------------
    haiku = pd.read_csv(ROOT / "data" / "scores" / "injection" / "haiku45_judge_scores.csv")
    merged = cell_df.merge(haiku, left_on=["model", "unit", "group"],
                           right_on=["model", "cell", "group"], how="outer", indicator=True)
    n_mis = int((merged["_merge"] != "both").sum())
    n_lab = int((merged.dropna(subset=["label_x", "label_y"])
                 .pipe(lambda d: (d.label_x != d.label_y).sum())))
    print(f"per-cell records {len(cell_df)} vs released Haiku 4.5 rows {len(haiku)}: "
          f"{n_mis} unmatched, {n_lab} label disagreements", flush=True)
    assert n_mis == 0 and n_lab == 0, "record rebuild does not match the released Haiku 4.5 rows"
    print(f"cross-cell records {len(cross_df)}; unique traces to judge {len(traces)}", flush=True)
    est_in = sum(len(e["reasoning_text"]) + len(e["question"]) + 900 for e in traces.values()) / 3.6
    print(f"approx input tokens {est_in:,.0f}; est. cost ~${est_in*3e-6 + len(traces)*120*15e-6:.2f}",
          flush=True)
    if args.dry_run:
        return

    # ---- judge (cached) ---------------------------------------------------------
    JUDGE = args.judge_model
    tag = JUDGE.split("/")[-1]
    cache_path = out / f"{tag}_judge_cache.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text().splitlines():
            d = json.loads(line)
            cache[d["key"]] = d
    if args.retry_unparsed:
        todo = [e for k, e in sorted(traces.items()) if k in cache and not cache[k]["parse_ok"]]
    else:
        todo = [e for k, e in sorted(traces.items()) if k not in cache]
    if args.pilot:
        todo = todo[: args.pilot]
    print(f"cached {len(cache)}, to judge {len(todo)}", flush=True)
    if todo:
        import os
        import re
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from openai import OpenAI

        cfg = load_project_config(ROOT / "data" / "prompts")
        view = cfg.judge_views["cot_answer"]
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise SystemExit("OPENROUTER_API_KEY is not set; it is needed to call the judge.")
        client = OpenAI(api_key=key, base_url=cfg.judge.base_url)
        errors: list[str] = []

        def one(ex: dict) -> dict | None:
            prompt = view.prompt.format(**ex)
            for attempt in range(4):
                try:
                    r = client.chat.completions.create(
                        model=JUDGE, max_tokens=args.max_tokens or cfg.judge.max_tokens,
                        temperature=cfg.judge.temperature,
                        messages=[{"role": "user", "content": prompt}])
                    text = (r.choices[0].message.content or "").strip()
                    m = re.search(r"PROBABILITY\s*:\s*(\d+)", text, re.IGNORECASE)
                    return {"key": ex["question_id"], "judge_model": JUDGE, "view": "cot_answer",
                            "max_tokens": args.max_tokens or cfg.judge.max_tokens,
                            "parse_ok": bool(m),
                            "prob": (max(0.0, min(100.0, float(m.group(1)))) / 100.0) if m else None,
                            "raw": text}
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{type(e).__name__}: {str(e)[:160]}")
                    time.sleep(2 * 2 ** attempt)
            return None

        with ThreadPoolExecutor(max_workers=16) as pool, cache_path.open("a") as fh:
            futs = [pool.submit(one, ex) for ex in todo]
            for i, f in enumerate(as_completed(futs), 1):
                d = f.result()
                if d:
                    fh.write(json.dumps(d) + "\n")
                    fh.flush()
                    cache[d["key"]] = d
                if i % 200 == 0:
                    print(f"  {i}/{len(todo)}", flush=True)
        if errors:
            print(f"{len(errors)} call errors; first: {errors[0]}", flush=True)
        n_ok = sum(1 for e in todo if cache.get(e["question_id"], {}).get("parse_ok"))
        print(f"returned {sum(1 for e in todo if e['question_id'] in cache)}/{len(todo)}, "
              f"parsed {n_ok}", flush=True)
    if args.pilot:
        for k in [e["question_id"] for e in todo][:2]:
            print("sample:", json.dumps(cache.get(k, {}))[:300], flush=True)
        print("pilot done", flush=True)
        return

    # ---- AUROCs -----------------------------------------------------------------
    def score(df: pd.DataFrame, probs: dict[str, float]) -> list[dict]:
        rows = []
        for (m, u), d in df.groupby(["model", "unit"], sort=False):
            have = d["key"].isin(probs)
            dd = d[have]
            res = {"model": m, "unit": u, "n": len(d), "n_scored": len(dd),
                   "n_dep": int(d.label.sum()), "n_indep": int((1 - d.label).sum())}
            if len(dd) >= 8 and dd.label.nunique() > 1:
                b = cluster_bootstrap_metric(dd.label.to_numpy(), dd["key"].map(probs).to_numpy(),
                                             dd.group.to_numpy(), "auroc", 2000, 0)
                res.update(auroc=b["point"], ci_low=b["ci_low"], ci_high=b["ci_high"])
            rows.append(res)
        return rows

    sonnet = {k: v["prob"] for k, v in cache.items() if v.get("parse_ok")}
    haiku_probs = {f"{r.model}|{r.cell}::{r.group}": r.judge_prob for r in haiku.itertuples()}
    out_rows = []
    for name, df in [("cell", cell_df), ("cross", cross_df)]:
        for judge_name, probs in [(tag, sonnet), ("claude-haiku-4.5", haiku_probs)]:
            if judge_name == "claude-haiku-4.5" and name == "cross":
                continue  # Haiku cross-cell per-row scores were never written; see crosscell_summary.csv
            for r in score(df, probs):
                out_rows.append({"level": name, "judge": judge_name, **r})
    summary = pd.DataFrame(out_rows)
    summary.to_csv(out / f"injection_factorial_{tag}_summary.csv", index=False)
    pd.concat([cell_df.assign(level="cell"), cross_df.assign(level="cross")]).assign(
        judge_prob=lambda d: d["key"].map(sonnet), judge_model=JUDGE
    ).drop(columns="key").to_csv(out / f"injection_factorial_{tag}_scores.csv", index=False)
    print(summary.round(3).to_string(index=False), flush=True)
    print(f"parse coverage: {len(sonnet)}/{len(traces)}", flush=True)


if __name__ == "__main__":
    main()
