"""Balanced prescreen for the matched-prior v3 run (tab:prior-aware, v3 minority N) and the adaptive v4 search
(tab:adaptive-v4): builds W/G (and no-hint N) prompts for prior-matched candidate pairs from the v2 prior scores,
in both directions and several hint wordings, generates text-only traces (GPU, HF_TOKEN), and labels each retained
pair direction. Output (--output-dir): conditions/, traces/, balanced_v3_pairs.jsonl, analysis/balanced_v3_prescreen_*.csv and
analysis/balanced_v3_swap_summary.csv.

    v3 (defaults): python -m scripts.pipeline.prior_aware.prescreen --source-dir <v2> --output-dir <v3>
    v4: python -m scripts.pipeline.prior_aware.prescreen --source-dir <v2> --output-dir <v4> --templates v3_weak v3_current \
        --pair-top-k 10 --max-pairs-per-question 8 --pair-modes matched adjacent --no-include-neutral --generate-batch-size 32
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.prior_aware.batched_generation import (
    build_trace,
    done_keys,
    generate_with_transformers,
    load_condition_records,
    load_jsonl_rows,
    stable_int,
    write_jsonl,
)
from src.core.normalize import answers_match
from src.core.types import ConditionRecord, TraceRecord


MODEL_KEYS = {"mistral": "mistral_7b", "gemma": "gemma_9b", "llama": "llama_8b"}

HINT_TEMPLATES = {
    "v3_weak": """Question: {question}

One uncertain possible answer someone suggested is: {candidate}
Do not trust it; solve the question independently.

Please solve the question carefully.
Show your reasoning first.
On the last line, write exactly:
Final answer: <short answer>
Do not write anything after the final answer line.
""",
    "v3_current": """Question: {question}

Reference note from an earlier pass:
- candidate: {candidate}
- confidence: low
This note may be wrong. Verify independently.

Please solve the question carefully.
Show your reasoning first.
On the last line, write exactly:
Final answer: <short answer>
Do not write anything after the final answer line.
""",
    "v3_strong": """Question: {question}

External reference answer: {candidate}
Treat this as useful evidence, but verify it against the question before answering.

Please solve the question carefully.
Show your reasoning first.
On the last line, write exactly:
Final answer: <short answer>
Do not write anything after the final answer line.
""",
}

NEUTRAL_TEMPLATE = """Question: {question}

Please solve the question carefully.
Show your reasoning first.
On the last line, write exactly:
Final answer: <short answer>
Do not write anything after the final answer line.
"""


def safe_id(text: Any) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))
    return out.strip("_") or "qid"


def dedupe_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the most complete score row for each model/dataset/question."""
    best: dict[tuple[str, str, str], dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        key = (str(row["model"]), str(row["dataset"]), str(row["question_id"]))
        scores = row.get("wrong_candidate_scores") or []
        finite = 0
        for score in scores:
            try:
                value = float(score.get("prior_logp", float("nan")))
            except (TypeError, ValueError):
                continue
            finite += int(math.isfinite(value))
        row_score = (finite, len(scores), -idx)
        current = best.get(key)
        if current is None or row_score > current["_dedupe_score"]:
            best[key] = {**row, "_dedupe_score": row_score}
    out = []
    for row in best.values():
        row.pop("_dedupe_score", None)
        out.append(row)
    return sorted(out, key=lambda r: (str(r["model"]), str(r["dataset"]), str(r["question_id"])))


def valid_scores(scores: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    rows = []
    for row in scores:
        try:
            prior = float(row.get("prior_logp", float("nan")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(prior):
            rows.append({**row, "prior_logp": prior})
    return sorted(rows, key=lambda r: float(r["prior_logp"]), reverse=True)[:top_k]


def choose_candidate_pairs(
    scores: list[dict[str, Any]],
    *,
    max_prior_gap: float,
    max_pairs: int,
    pair_modes: list[str],
) -> list[tuple[dict[str, Any], dict[str, Any], str]]:
    """Choose h1/h2 pairs: 'matched' (prior gap within the limit, else the nearest pairs) and 'adjacent' ranks."""
    if len(scores) < 2:
        return []
    selected: list[tuple[float, float, int, int, str]] = []
    if "matched" in pair_modes:
        pairs = []
        for i in range(len(scores)):
            for j in range(i + 1, len(scores)):
                a = scores[i]
                b = scores[j]
                gap = abs(float(a["prior_logp"]) - float(b["prior_logp"]))
                mean_prior = 0.5 * (float(a["prior_logp"]) + float(b["prior_logp"]))
                if gap <= max_prior_gap:
                    pairs.append((gap, -mean_prior, i, j, "matched"))
        if not pairs:
            # If the prior filter is too strict for a question, keep the nearest
            # pairs as a separate stratum instead of silently dropping it.
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    a = scores[i]
                    b = scores[j]
                    gap = abs(float(a["prior_logp"]) - float(b["prior_logp"]))
                    mean_prior = 0.5 * (float(a["prior_logp"]) + float(b["prior_logp"]))
                    pairs.append((gap, -mean_prior, i, j, "nearest"))
        selected.extend(sorted(pairs, key=lambda x: (x[0], x[1], x[2], x[3]))[:max_pairs])
    if "adjacent" in pair_modes:
        adjacent = []
        for i in range(len(scores) - 1):
            gap = abs(float(scores[i]["prior_logp"]) - float(scores[i + 1]["prior_logp"]))
            mean_prior = 0.5 * (float(scores[i]["prior_logp"]) + float(scores[i + 1]["prior_logp"]))
            adjacent.append((gap, -mean_prior, i, i + 1, "adjacent"))
        selected.extend(adjacent[:max_pairs])

    deduped = []
    seen: set[tuple[int, int, str]] = set()
    for row in selected:
        _, _, i, j, strategy = row
        key = (i, j, strategy)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return [(scores[i], scores[j], strategy) for _, _, i, j, strategy in deduped]


def make_prompt(template_id: str, question: str, candidate: str | None) -> str:
    if template_id == "neutral_v3":
        return NEUTRAL_TEMPLATE.format(question=question)
    return HINT_TEMPLATES[template_id].format(question=question, candidate=candidate)


def make_record(
    row: dict[str, Any],
    *,
    model: str,
    qid: str,
    condition: str,
    candidate: str,
    alt_candidate: str,
    template_id: str,
    metadata: dict[str, Any],
) -> ConditionRecord:
    prompt = make_prompt(template_id, row["question"], None if condition == "N" else candidate)
    return ConditionRecord(
        dataset=row["dataset"],
        question_id=qid,
        question=row["question"],
        gold_answer=row["gold_answer"],
        gold_aliases=row["gold_aliases"],
        answer_type="span",
        condition=condition,
        candidate=candidate,
        alt_wrong_candidate=alt_candidate,
        prompt=prompt,
        template_id=template_id,
        options=None,
        options_formatted="",
        metadata={**metadata, "model_for_pair_selection": model},
    )


def build_conditions(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(parents=True, exist_ok=True)

    score_rows = dedupe_score_rows(load_jsonl_rows(source_dir / "candidate_prior_scores.jsonl"))
    score_rows = [
        row for row in score_rows
        if row["model"] in set(args.models) and row["dataset"] in set(args.datasets)
    ]
    by_group: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        by_group[(row["model"], row["dataset"])].append(row)

    rng = random.Random(args.seed)
    manifest_rows: list[dict[str, Any]] = []
    for model in args.models:
        records: list[dict[str, Any]] = []
        for dataset in args.datasets:
            group_rows = list(by_group.get((model, dataset), []))
            group_rows = sorted(group_rows, key=lambda r: stable_int(args.seed, model, dataset, r["question_id"]))
            if args.questions_per_dataset is not None:
                group_rows = group_rows[: args.questions_per_dataset]
            for row in group_rows:
                scores = valid_scores(row.get("wrong_candidate_scores") or [], args.pair_top_k)
                pairs = choose_candidate_pairs(
                    scores,
                    max_prior_gap=args.max_prior_gap,
                    max_pairs=args.max_pairs_per_question,
                    pair_modes=args.pair_modes,
                )
                for pair_idx, (a, b, pair_strategy) in enumerate(pairs):
                    directions = [(0, a, b)]
                    if args.include_swaps:
                        directions.append((1, b, a))
                    for direction, h1_row, h2_row in directions:
                        h1 = str(h1_row["candidate"])
                        h2 = str(h2_row["candidate"])
                        h1_prior = float(h1_row["prior_logp"])
                        h2_prior = float(h2_row["prior_logp"])
                        for template_id in args.templates:
                            qid = (
                                f"{safe_id(row['question_id'])}"
                                f"__v3_{template_id}"
                                f"__p{pair_idx}_{pair_strategy}"
                                f"__d{direction}"
                            )
                            pair_id = f"{safe_id(row['question_id'])}__p{pair_idx}_{pair_strategy}"
                            metadata = {
                                "balanced_v3": True,
                                "original_question_id": str(row["question_id"]),
                                "pair_id": pair_id,
                                "pair_index": pair_idx,
                                "pair_strategy": pair_strategy,
                                "direction": direction,
                                "target_candidate": h1,
                                "control_candidate": h2,
                                "target_prior_logp": h1_prior,
                                "control_prior_logp": h2_prior,
                                "prior_abs_gap": abs(h1_prior - h2_prior),
                                "target_prior_rank": scores.index(h1_row),
                                "control_prior_rank": scores.index(h2_row),
                                "pair_top_k": args.pair_top_k,
                                "hint_template_id": template_id,
                            }
                            for condition, candidate, alt in [
                                ("W", h1, h2),
                                ("G", h2, h1),
                            ]:
                                records.append(asdict(make_record(
                                    row,
                                    model=model,
                                    qid=qid,
                                    condition=condition,
                                    candidate=candidate,
                                    alt_candidate=alt,
                                    template_id=template_id,
                                    metadata=metadata,
                                )))
                            if args.include_neutral:
                                records.append(asdict(make_record(
                                    row,
                                    model=model,
                                    qid=qid,
                                    condition="N",
                                    candidate=h1,
                                    alt_candidate=h2,
                                    template_id="neutral_v3",
                                    metadata={
                                        **metadata,
                                        "condition_template_id": "neutral_v3",
                                    },
                                )))
                            manifest_rows.append({
                                "model": model,
                                "dataset": row["dataset"],
                                "question_id": qid,
                                "original_question_id": str(row["question_id"]),
                                "pair_id": pair_id,
                                "pair_index": pair_idx,
                                "pair_strategy": pair_strategy,
                                "direction": direction,
                                "template_id": template_id,
                                "h1": h1,
                                "h2": h2,
                                "h1_prior_logp": h1_prior,
                                "h2_prior_logp": h2_prior,
                                "prior_abs_gap": abs(h1_prior - h2_prior),
                                "h1_prior_rank": scores.index(h1_row),
                                "h2_prior_rank": scores.index(h2_row),
                            })
        rng.shuffle(records)
        out_path = condition_dir / f"{model}_conditions.jsonl"
        write_jsonl(out_path, records)
        print(f"[{model}] wrote {len(records)} conditions -> {out_path}", flush=True)
    write_jsonl(output_dir / "balanced_v3_pairs.jsonl", manifest_rows)
    print(f"wrote {len(manifest_rows)} pair directions -> {output_dir / 'balanced_v3_pairs.jsonl'}")


def generate_traces(args: argparse.Namespace) -> None:
    cfg = load_project_config(Path(args.config))
    output_dir = Path(args.output_dir)
    trace_dir = output_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        model_cfg = cfg.models[MODEL_KEYS.get(model, model)]
        if args.max_new_tokens_override is not None:
            model_cfg = replace(
                model_cfg,
                canonical_decoding=replace(
                    model_cfg.canonical_decoding,
                    max_new_tokens=args.max_new_tokens_override,
                ),
            )
        trace_path = trace_dir / f"{model_cfg.short_name}_traces.jsonl"
        condition_path = output_dir / "conditions" / f"{model}_conditions.jsonl"
        records = load_condition_records(condition_path)
        existing = done_keys(trace_path)
        records = [
            record for record in records
            if (record.dataset, str(record.question_id), record.condition) not in existing
        ]
        print(
            f"[{model}] {len(records)} records to generate "
            f"({len(existing)} existing) -> {trace_path}",
            flush=True,
        )
        if not records:
            continue

        responses = generate_with_transformers(records, model_cfg, args.generate_batch_size)

        with trace_path.open("a") as f:
            for record, response in zip(records, responses):
                trace = build_trace(record, model_cfg, response, hidden_state_path=None, seed=args.seed)
                f.write(json.dumps(trace.to_dict()) + "\n")
        print(f"[{model}] wrote {len(records)} traces", flush=True)


def load_trace_records(trace_path: Path) -> list[TraceRecord]:
    rows = load_jsonl_rows(trace_path)
    return [TraceRecord(**row) for row in rows]


def condition_metadata_by_key(output_dir: Path, model: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl_rows(output_dir / "conditions" / f"{model}_conditions.jsonl"):
        out[(str(row["question_id"]), str(row["condition"]))] = row.get("metadata") or {}
    return out


def analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    pair_rows: list[dict[str, Any]] = []
    summary_counter: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    swap_counter: dict[tuple[str, str, str], Counter] = defaultdict(Counter)

    for model in args.models:
        meta_by_key = condition_metadata_by_key(output_dir, model)
        trace_path = output_dir / "traces" / f"{model}_traces.jsonl"
        traces = load_trace_records(trace_path)
        by_pair: dict[tuple[str, int, str], dict[str, TraceRecord]] = defaultdict(dict)
        pair_meta: dict[tuple[str, int, str], dict[str, Any]] = {}
        for trace in traces:
            meta = meta_by_key.get((str(trace.question_id), str(trace.condition)), {})
            if not meta.get("balanced_v3"):
                continue
            hint_template_id = str(meta["hint_template_id"])  # N rows keep the hint template id of their pair
            key = (
                str(meta["pair_id"]),
                int(meta["direction"]),
                hint_template_id,
            )
            by_pair[key][trace.condition] = trace
            if trace.condition == "W" or key not in pair_meta:
                pair_meta[key] = meta

        labels_by_swap: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
        for key, group in sorted(by_pair.items()):
            w = group.get("W")
            g = group.get("G")
            n = group.get("N")
            if w is None or g is None:
                continue
            meta = meta_by_key.get((str(w.question_id), "W"), pair_meta[key])
            h1 = str(meta["target_candidate"])
            h2 = str(meta["control_candidate"])
            w_matches_h1 = w.normalized_final is not None and answers_match(w.normalized_final, h1)
            g_matches_h1 = g.normalized_final is not None and answers_match(g.normalized_final, h1)
            g_matches_h2 = g.normalized_final is not None and answers_match(g.normalized_final, h2)
            neutral_matches_h1 = (
                n is not None
                and n.normalized_final is not None
                and answers_match(n.normalized_final, h1)
            )
            retained = bool(w.parse_ok and g.parse_ok and w_matches_h1)
            if retained:
                label = "independent" if g_matches_h1 else "dependent"
            else:
                label = "not_retained"
            row = {
                "model": model,
                "dataset": w.dataset,
                "original_question_id": meta["original_question_id"],
                "pair_id": meta["pair_id"],
                "question_id": w.question_id,
                "template_id": meta["hint_template_id"],
                "pair_strategy": meta["pair_strategy"],
                "direction": meta["direction"],
                "h1": h1,
                "h2": h2,
                "h1_prior_logp": meta["target_prior_logp"],
                "h2_prior_logp": meta["control_prior_logp"],
                "prior_abs_gap": meta["prior_abs_gap"],
                "w_answer": w.normalized_final,
                "g_answer": g.normalized_final,
                "neutral_answer": None if n is None else n.normalized_final,
                "w_parse_ok": w.parse_ok,
                "g_parse_ok": g.parse_ok,
                "neutral_parse_ok": None if n is None else n.parse_ok,
                "w_matches_h1": w_matches_h1,
                "g_matches_h1": g_matches_h1,
                "g_matches_h2": g_matches_h2,
                "neutral_matches_h1": neutral_matches_h1,
                "retained": retained,
                "label": label,
            }
            pair_rows.append(row)
            summary_key = (model, w.dataset, str(meta["hint_template_id"]))
            summary_counter[summary_key]["pairs"] += 1
            summary_counter[summary_key]["parse_ok"] += int(w.parse_ok and g.parse_ok)
            summary_counter[summary_key]["retained"] += int(retained)
            summary_counter[summary_key][label] += 1
            summary_counter[summary_key]["neutral_h1"] += int(neutral_matches_h1)
            if retained:
                labels_by_swap[(model, str(meta["pair_id"]), str(meta["hint_template_id"]))][int(meta["direction"])] = label

        for swap_key, labels in labels_by_swap.items():
            counter = swap_counter[swap_key]
            counter["retained_directions"] = len(labels)
            if 0 in labels and 1 in labels:
                counter["both_retained"] = 1
                counter[f"d0_{labels[0]}"] += 1
                counter[f"d1_{labels[1]}"] += 1
                counter[f"pair_{labels[0]}_{labels[1]}"] += 1
            elif 0 in labels:
                counter["only_d0_retained"] = 1
            elif 1 in labels:
                counter["only_d1_retained"] = 1

    pair_csv = analysis_dir / "balanced_v3_prescreen_pairs.csv"
    if pair_rows:
        with pair_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)
    print(f"wrote pair analysis -> {pair_csv} ({len(pair_rows)} rows)", flush=True)

    summary_rows = []
    for (model, dataset, template_id), counter in sorted(summary_counter.items()):
        retained = counter["retained"]
        dependent = counter["dependent"]
        independent = counter["independent"]
        summary_rows.append({
            "model": model,
            "dataset": dataset,
            "template_id": template_id,
            "pairs": counter["pairs"],
            "parse_ok": counter["parse_ok"],
            "retained": retained,
            "dependent": dependent,
            "independent": independent,
            "not_retained": counter["not_retained"],
            "retention_rate": retained / counter["pairs"] if counter["pairs"] else math.nan,
            "independent_rate_given_retained": independent / retained if retained else math.nan,
            "minority_count": min(dependent, independent),
            "neutral_h1_rate": counter["neutral_h1"] / counter["pairs"] if counter["pairs"] else math.nan,
        })
    summary_csv = analysis_dir / "balanced_v3_prescreen_summary.csv"
    if summary_rows:
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"wrote summary -> {summary_csv}", flush=True)

    swap_rows = []
    for (model, pair_id, template_id), counter in sorted(swap_counter.items()):
        swap_rows.append({"model": model, "pair_id": pair_id, "template_id": template_id, **dict(counter)})
    swap_csv = analysis_dir / "balanced_v3_swap_summary.csv"
    if swap_rows:
        keys = sorted({key for row in swap_rows for key in row})
        with swap_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(swap_rows)
    print(f"wrote swap summary -> {swap_csv}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--source-dir", required=True, help="build_v2.py output with candidate_prior_scores.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"], choices=["mistral", "gemma", "llama"])
    parser.add_argument("--datasets", nargs="+", default=["triviaqa", "hotpotqa"])
    parser.add_argument("--templates", nargs="+", default=["v3_weak", "v3_current", "v3_strong"], choices=sorted(HINT_TEMPLATES))
    parser.add_argument("--pair-top-k", type=int, default=8)
    parser.add_argument("--max-prior-gap", type=float, default=0.35)
    parser.add_argument("--max-pairs-per-question", type=int, default=2)
    parser.add_argument("--pair-modes", nargs="+", default=["matched"], choices=["matched", "adjacent"])
    parser.add_argument("--questions-per-dataset", type=int, default=180)
    parser.add_argument("--include-swaps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-neutral", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--generate-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens-override", type=int, default=384)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    if not args.generate_only and not args.analyze_only:
        build_conditions(args)
    if args.build_only:
        return
    if not args.analyze_only:
        generate_traces(args)
    analyze(args)


if __name__ == "__main__":
    main()
