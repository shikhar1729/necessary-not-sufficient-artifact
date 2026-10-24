"""Powered paired activation patching with exact McNemar tests (tab:patching); the defaults are the recorded run.
Input: --data-dir with traces/{model}_traces.jsonl. Output (--output-dir): per-row JSONL, the McNemar summary and
<run-id>_detail.csv, released as data/scores/patching/powered_patching_rows.csv.gz (without cell_key, run_id,
donor, interp, position_tokens). Needs a GPU and HF_TOKEN; --summarize-only re-reads the JSONL, no --data-dir.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from collections import defaultdict
from contextlib import contextmanager
from math import comb
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from transformers import StoppingCriteriaList

from scripts.pipeline.common.config import ModelConfig, load_project_config
from scripts.pipeline.common.model_loader import DTYPE_MAP, AnswerLineStopping, format_chat, get_layers, check_hf_access
from src.core.normalize import answers_match, parse_response

MCQ_DATASETS = {"gpqa", "mmlu_pro"}
# early / middle (= the 13/24/30 layer of the earlier patching runs) / late, per model
DEFAULT_LAYER_SETS = {
    "mistral": [6, 13, 28],
    "gemma": [8, 24, 38],
    "llama": [8, 20, 30],
}
# position specs: "last1" patches only the final prompt token, "prev1" only the second-to-last token,
# "last8" the final 8 tokens.
DEFAULT_POSITIONS = ["last1", "prev1", "last8"]


def answer_type(dataset: str) -> str:
    return "mcq" if dataset in MCQ_DATASETS else "span"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact (binomial) McNemar p-value for discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * tail))


def min_discordant_for_significance(alpha: float = 0.05) -> int:
    """Smallest one-directional discordant count with exact p <= alpha."""
    d = 1
    while d < 200:
        if exact_mcnemar_p(d, 0) <= alpha:
            return d
        d += 1
    return 200


MIN_DISCORDANT = min_discordant_for_significance(0.05)


# ---------------------------------------------------------------------------
# example construction
# ---------------------------------------------------------------------------
def load_traces(path: Path) -> dict[tuple[str, str], dict[str, dict[str, Any]]]:
    by_key: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            by_key[(row["dataset"], str(row["question_id"]))][row["condition"]] = row
    return by_key


def build_main_task_examples(data_dir: Path, model_short: str) -> list[dict[str, Any]]:
    """Strict same-answer main-task slice: a_W == h, a_W != y*, label = 1[a_G != h]."""
    by_key = load_traces(data_dir / "traces" / f"{model_short}_traces.jsonl")
    examples: list[dict[str, Any]] = []
    for (dataset, question_id), conds in sorted(by_key.items()):
        w = conds.get("W")
        g = conds.get("G")
        if not w or not g or not w.get("parse_ok") or not g.get("parse_ok"):
            continue
        h1 = w["candidate"]
        w_answer = w.get("normalized_final")
        g_answer = g.get("normalized_final")
        if not w_answer or not answers_match(w_answer, h1):
            continue
        if answers_match(w_answer, w.get("gold_answer")):
            continue
        if not g_answer:
            continue
        examples.append(
            {
                "dataset": dataset,
                "question_id": question_id,
                "w": w,
                "g": g,
                "label": int(not answers_match(g_answer, h1)),
                "answer_type": answer_type(dataset),
            }
        )
    return examples


def select_examples(
    examples: list[dict[str, Any]], per_class: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for label in (1, 0):
        pool = [ex for ex in examples if ex["label"] == label]
        rng.shuffle(pool)
        out.extend(pool[:per_class])
    out.sort(key=lambda ex: (ex["label"], ex["dataset"], str(ex["question_id"])))
    return out


# ---------------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------------
def position_slice(spec: str) -> tuple[int, int]:
    """Return (start_offset, end_offset) as negative python slice bounds."""
    if spec == "last1":
        return (-1, 0)
    if spec == "prev1":
        return (-2, -1)
    if spec.startswith("last") and spec[4:].isdigit():
        return (-int(spec[4:]), 0)
    raise ValueError(f"unknown position spec {spec!r}")


@contextmanager
def capture_hook(
    model: Any,
    model_cfg: ModelConfig,
    layers: Iterable[int],
    store: dict[int, torch.Tensor],
    keep_last: int = 8,
):
    """Record the pre-hook input (= residual stream entering each layer) on the prefill.

    Only the final `keep_last` positions are kept: with left padding those are the
    last real prompt tokens, and they are the only ones any position spec needs.
    """
    module_list = get_layers(model, model_cfg)
    handles = []
    seen: dict[int, int] = {layer: 0 for layer in layers}

    def make(layer: int):
        def hook_fn(module, args):
            seen[layer] += 1
            if seen[layer] != 1 or not args:
                return None
            store[layer] = args[0][:, -keep_last:, :].detach().clone()
            return None

        return hook_fn

    for layer in layers:
        handles.append(module_list[layer].register_forward_pre_hook(make(layer)))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def patch_hook(
    model: Any,
    model_cfg: ModelConfig,
    layer: int,
    donor: torch.Tensor | None,
    spec: str,
    interp: float,
):
    """Overwrite a slice of the recipient's prefill residual stream with donor rows."""
    if donor is None or interp <= 0:
        yield
        return
    module_list = get_layers(model, model_cfg)
    start, end = position_slice(spec)
    calls = 0

    def hook_fn(module, args):
        nonlocal calls
        calls += 1
        if calls != 1 or not args:
            return None
        hidden = args[0]
        if hidden.shape[1] < abs(start):
            return None
        src = donor.to(device=hidden.device, dtype=hidden.dtype)
        patched = hidden.clone()
        hi = hidden.shape[1] + end if end else hidden.shape[1]
        lo = hidden.shape[1] + start
        patched[:, lo:hi, :] = patched[:, lo:hi, :] + interp * (src - patched[:, lo:hi, :])
        return (patched,) + tuple(args[1:])

    handle = module_list[layer].register_forward_pre_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


@torch.inference_mode()
def generate_batch(
    model: Any,
    tokenizer: Any,
    prompt_texts: list[str],
    max_new_tokens: int,
    layer: int | None = None,
    donor: torch.Tensor | None = None,
    spec: str = "last1",
    interp: float = 1.0,
    model_cfg: ModelConfig | None = None,
    capture_layers: list[int] | None = None,
    early_stop: bool = True,
    keep_last: int = 8,
) -> tuple[list[str], dict[int, torch.Tensor]]:
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)
    store: dict[int, torch.Tensor] = {}
    ctx_patch = (
        patch_hook(model, model_cfg, layer, donor, spec, interp)
        if layer is not None and donor is not None
        else patch_hook(model, model_cfg, 0, None, spec, 0.0)
    )
    ctx_capture = (
        capture_hook(model, model_cfg, capture_layers, store, keep_last=keep_last)
        if capture_layers
        else capture_hook(model, model_cfg, [], store, keep_last=keep_last)
    )
    criteria = None
    if early_stop:
        criteria = StoppingCriteriaList(
            [AnswerLineStopping(tokenizer, encoded["input_ids"].shape[1])]
        )
    with ctx_capture, ctx_patch:
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=criteria,
        )
    width = encoded["input_ids"].shape[1]
    texts = [
        tokenizer.decode(seq[width:], skip_special_tokens=True).strip() for seq in generated
    ]
    del encoded, generated
    return texts, store


def score(example: dict[str, Any], text: str) -> dict[str, Any]:
    parsed = parse_response(text, example["answer_type"])
    norm = parsed.final_answer_norm
    h1 = example["w"]["candidate"]
    h2 = example["g"]["candidate"]
    gold = example["w"]["gold_answer"]
    return {
        "normalized_final": norm,
        "parse_ok": parsed.parse_mode != "fail",
        "parse_mode": parsed.parse_mode,
        "matches_h1": bool(norm and answers_match(norm, h1)),
        "matches_h2": bool(norm and answers_match(norm, h2)),
        "matches_correct": bool(norm and answers_match(norm, gold)),
        "response_chars": len(text),
    }


# ---------------------------------------------------------------------------
# main run
# ---------------------------------------------------------------------------
def run_model(args: argparse.Namespace, model_key: str) -> None:
    cfg = load_project_config(Path(args.config))
    model_cfg = cfg.models[model_key]
    short = model_cfg.short_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{short}_{args.run_id}_rows.jsonl"

    layers = args.layer_map.get(short, DEFAULT_LAYER_SETS.get(short, [8, 16, 24]))
    positions = args.positions
    keep_last = max(abs(position_slice(spec)[0]) for spec in positions)
    examples_all = build_main_task_examples(Path(args.data_dir), short)
    examples = select_examples(examples_all, args.per_class, args.seed)
    n_dep = sum(ex["label"] for ex in examples)
    print(
        f"[{short}] slice={len(examples_all)} selected={len(examples)} "
        f"(dep={n_dep}, indep={len(examples) - n_dep}) layers={layers} positions={positions}",
        flush=True,
    )
    if args.plan_only:
        return

    done: set[str] = set()
    if out_path.exists():
        with out_path.open() as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add(row["cell_key"])

    hf_token = os.environ.get("HF_TOKEN")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    check_hf_access(model_cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, token=hf_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        torch_dtype=DTYPE_MAP.get(model_cfg.dtype, torch.bfloat16),
        device_map="auto",
        token=hf_token,
    )
    model.eval()

    w_texts = [format_chat(tokenizer, model_cfg, ex["w"]["raw_prompt"]) for ex in examples]
    g_texts = [format_chat(tokenizer, model_cfg, ex["g"]["raw_prompt"]) for ex in examples]

    t_start = time.time()
    n_written = 0
    with out_path.open("a") as sink:

        def emit(rows: list[dict[str, Any]]) -> None:
            nonlocal n_written
            for row in rows:
                sink.write(json.dumps(row) + "\n")
                n_written += 1
            sink.flush()

        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            bw = w_texts[start : start + args.batch_size]
            bg = g_texts[start : start + args.batch_size]

            # one batched forward per recipient condition: baseline text + donor states
            donors: dict[str, dict[int, torch.Tensor]] = {}
            baselines: dict[str, list[dict[str, Any]]] = {}
            for cond, texts in [("W", bw), ("G", bg)]:
                out_texts, store = generate_batch(
                    model,
                    tokenizer,
                    texts,
                    args.max_new_tokens,
                    model_cfg=model_cfg,
                    capture_layers=layers,
                    early_stop=not args.no_early_stop,
                    keep_last=keep_last,
                )
                donors[cond] = store
                baselines[cond] = [score(ex, text) for ex, text in zip(batch, out_texts)]
                rows = []
                for ex, text, scored in zip(batch, out_texts, baselines[cond]):
                    key = f"{ex['dataset']}::{ex['question_id']}::{cond}::baseline::none::none::0"
                    if key in done:
                        continue
                    rows.append(
                        {
                            "cell_key": key,
                            "run_id": args.run_id,
                            "model": short,
                            "dataset": ex["dataset"],
                            "question_id": str(ex["question_id"]),
                            "label": int(ex["label"]),
                            "recipient": cond,
                            "donor": "none",
                            "condition_name": "baseline",
                            "layer": -1,
                            "position": "none",
                            "interp": 0.0,
                            "h1": ex["w"]["candidate"],
                            "h2": ex["g"]["candidate"],
                            "correct_answer": ex["w"]["gold_answer"],
                            "raw_response": text if args.save_text else "",
                            **scored,
                        }
                    )
                emit(rows)

            # patched generations
            for recipient, donor_cond in [("G", "W"), ("W", "G")]:
                texts = bg if recipient == "G" else bw
                for layer in layers:
                    donor_full = donors[donor_cond].get(layer)
                    if donor_full is None:
                        continue
                    for spec in positions:
                        lo, hi = position_slice(spec)
                        length = (hi if hi else 0) - lo
                        if donor_full.shape[1] < abs(lo):
                            continue
                        end = donor_full.shape[1] + hi if hi else donor_full.shape[1]
                        begin = donor_full.shape[1] + lo
                        donor_slice = donor_full[:, begin:end, :]
                        for interp in args.interps:
                            keys = [
                                f"{ex['dataset']}::{ex['question_id']}::{recipient}::patch_"
                                f"{donor_cond}_to_{recipient}::L{layer}::{spec}::{interp}"
                                for ex in batch
                            ]
                            if all(k in done for k in keys):
                                continue
                            out_texts, _ = generate_batch(
                                model,
                                tokenizer,
                                texts,
                                args.max_new_tokens,
                                layer=layer,
                                donor=donor_slice,
                                spec=spec,
                                interp=float(interp),
                                model_cfg=model_cfg,
                                early_stop=not args.no_early_stop,
                            )
                            rows = []
                            for ex, text, key in zip(batch, out_texts, keys):
                                if key in done:
                                    continue
                                scored = score(ex, text)
                                rows.append(
                                    {
                                        "cell_key": key,
                                        "run_id": args.run_id,
                                        "model": short,
                                        "dataset": ex["dataset"],
                                        "question_id": str(ex["question_id"]),
                                        "label": int(ex["label"]),
                                        "recipient": recipient,
                                        "donor": donor_cond,
                                        "condition_name": f"patch_{donor_cond}_to_{recipient}",
                                        "layer": int(layer),
                                        "position": spec,
                                        "position_tokens": int(length),
                                        "interp": float(interp),
                                        "h1": ex["w"]["candidate"],
                                        "h2": ex["g"]["candidate"],
                                        "correct_answer": ex["w"]["gold_answer"],
                                        "raw_response": text if args.save_text else "",
                                        **scored,
                                    }
                                )
                            emit(rows)
            for store in donors.values():
                store.clear()
            del donors
            gc.collect()
            torch.cuda.empty_cache()
            elapsed = time.time() - t_start
            done_examples = min(start + args.batch_size, len(examples))
            print(
                f"[{short}] {done_examples}/{len(examples)} examples, {n_written} rows, "
                f"{elapsed / 60:.1f} min",
                flush=True,
            )

    print(f"[{short}] wrote {n_written} rows -> {out_path}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.glob(f"*_{args.run_id}_rows.jsonl")):
        with path.open() as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        print("no rows to summarize", flush=True)
        return
    df = pd.DataFrame(rows)
    df.drop(columns=[c for c in ["raw_response"] if c in df.columns]).to_csv(
        output_dir / f"{args.run_id}_detail.csv", index=False
    )

    base = df[df["condition_name"] == "baseline"][
        ["model", "dataset", "question_id", "label", "recipient", "matches_h1", "parse_ok"]
    ].rename(columns={"matches_h1": "baseline_h1", "parse_ok": "baseline_parse_ok"})

    # Batched greedy regeneration does not always reproduce the unbatched greedy
    # answer stored in the source traces (bf16 + left padding). Flag the examples
    # whose W and G baselines DO reproduce the defining properties of the strict
    # slice (a_W == h, and a_G == h iff the stored label is `indep`) so every cell
    # can be reported on both populations.
    pivot = base.pivot_table(
        index=["model", "dataset", "question_id", "label"],
        columns="recipient",
        values="baseline_h1",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    for col in ("W", "G"):
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot["slice_reproduced"] = (
        pivot["W"].astype("boolean").fillna(False)
        & (pivot["G"].astype("boolean").fillna(False) == (pivot["label"] == 0))
    )
    repro = pivot[["model", "dataset", "question_id", "label", "slice_reproduced"]]
    repro_rates = (
        pivot.groupby(["model", "label"], as_index=False)["slice_reproduced"]
        .agg(n="count", n_reproduced="sum")
        .assign(reproduction_rate=lambda d: d["n_reproduced"] / d["n"])
    )
    repro_rates["label_name"] = np.where(repro_rates["label"] == 1, "dependent", "independent")
    repro_rates.to_csv(output_dir / f"{args.run_id}_baseline_reproduction.csv", index=False)
    print("BASELINE REPRODUCTION OF THE STRICT SLICE", flush=True)
    print(repro_rates.round(4).to_string(index=False), flush=True)

    patched = (
        df[df["condition_name"] != "baseline"]
        .merge(base, on=["model", "dataset", "question_id", "label", "recipient"], how="left")
        .merge(repro, on=["model", "dataset", "question_id", "label"], how="left")
    )

    summary: list[dict[str, Any]] = []
    group_cols = [
        "model",
        "label",
        "recipient",
        "donor",
        "condition_name",
        "layer",
        "position",
        "interp",
    ]
    for keys, sub_all in patched.groupby(group_cols, dropna=False):
        for population in ("all_selected", "baseline_consistent"):
            sub = sub_all.dropna(subset=["baseline_h1"])
            if population == "baseline_consistent":
                sub = sub[sub["slice_reproduced"].astype("boolean").fillna(False)]
            if sub.empty:
                continue
            b0 = sub["baseline_h1"].astype(bool).to_numpy()
            b1 = sub["matches_h1"].astype(bool).to_numpy()
            n = int(len(sub))
            flips_down = int(np.sum(b0 & ~b1))
            flips_up = int(np.sum(~b0 & b1))
            p_value = exact_mcnemar_p(flips_down, flips_up)
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "population": population,
                    "label_name": "dependent" if row["label"] == 1 else "independent",
                    "n": n,
                    "baseline_h1_rate": float(b0.mean()),
                    "patched_h1_rate": float(b1.mean()),
                    "delta_h1_rate": float(b1.mean() - b0.mean()),
                    "discordant_down": flips_down,
                    "discordant_up": flips_up,
                    "n_discordant": flips_down + flips_up,
                    "mcnemar_exact_p": p_value,
                    "min_discordant_for_p05": MIN_DISCORDANT,
                    "detectable_abs_delta_at_n": float(MIN_DISCORDANT / n) if n else math.nan,
                    "patched_h2_rate": float(sub["matches_h2"].astype(bool).mean()),
                    "patched_correct_rate": float(sub["matches_correct"].astype(bool).mean()),
                    "patched_parse_ok_rate": float(sub["parse_ok"].astype(bool).mean()),
                    "baseline_parse_ok_rate": float(sub["baseline_parse_ok"].astype(bool).mean()),
                }
            )
            summary.append(row)

    summary_df = pd.DataFrame(summary).sort_values(
        ["population", "model", "label", "condition_name", "layer", "position", "interp"]
    )
    summary_df.to_csv(output_dir / f"{args.run_id}_mcnemar_summary.csv", index=False)
    print(f"wrote {output_dir / f'{args.run_id}_mcnemar_summary.csv'}", flush=True)
    show = [
        "population",
        "model",
        "label_name",
        "condition_name",
        "layer",
        "position",
        "interp",
        "n",
        "baseline_h1_rate",
        "patched_h1_rate",
        "delta_h1_rate",
        "discordant_down",
        "discordant_up",
        "mcnemar_exact_p",
        "detectable_abs_delta_at_n",
    ]
    print(
        summary_df[summary_df["population"] == "baseline_consistent"][show]
        .round(4)
        .to_string(index=False),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--data-dir", help="trace store with traces/{model}_traces.jsonl (not needed with --summarize-only)")
    parser.add_argument("--output-dir", default="outputs/powered_patching")
    parser.add_argument("--models", nargs="+", default=["mistral_7b", "gemma_9b", "llama_8b"])
    parser.add_argument("--run-id", default="powered_patch_v1")
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--positions", nargs="+", default=DEFAULT_POSITIONS)
    parser.add_argument("--interps", nargs="+", type=float, default=[1.0])
    parser.add_argument(
        "--layers",
        nargs="*",
        default=[],
        help="per-model layer sets, e.g. mistral:6,13,28 gemma:8,24,38 llama:8,20,30",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--save-text", action="store_true")
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="disable the 'Final answer:' stopping criterion (slower, pure max_new_tokens)",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if not args.summarize_only and not args.data_dir:
        parser.error("--data-dir is required unless --summarize-only is given")

    args.layer_map = {}
    for entry in args.layers:
        short, values = entry.split(":", 1)
        args.layer_map[short] = [int(v) for v in values.split(",") if v]

    if args.summarize_only:
        summarize(args)
        return
    for model_key in args.models:
        run_model(args, model_key)
    summarize(args)


if __name__ == "__main__":
    main()
