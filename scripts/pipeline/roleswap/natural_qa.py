"""Natural-QA role swap (tab:roleswap-natural; "public features 0.840 vs W-G 0.620" in the results).

--build-manifest: per model and span dataset, 160 questions in seeded hash order, each with its 5 highest-prior wrong
candidates (input: candidate_prior_scores.jsonl from prior_aware/build_v2.py). --score (GPU, HF_TOKEN with
--use-hf-token): teacher-forced log p(candidate | note naming each candidate, "\\nFinal answer:") and the last-token
states. --analyze: pair-centred effect per orientation, ridge (alpha 10) OOF predictions grouped by question, binary
AUROC of their sign. Paper run: the three stages with these defaults (the analysis was run with no text baseline).
    python -m scripts.pipeline.roleswap.natural_qa --source-dir <v2> --output-dir <dir> --build-manifest --score --analyze --use-hf-token
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.common.model_loader import DTYPE_MAP, format_chat, check_hf_access
from scripts.pipeline.prior_aware.batched_generation import load_jsonl_rows
from scripts.pipeline.prior_aware.build_v2 import answer_variants, dedupe_score_rows
from scripts.pipeline.roleswap.generate_benchmark import (
    grouped_ridge_oof_fixed,
    summarize_binary,
    summarize_regression,
)


def stable_float(*parts: Any) -> float:
    text = "::".join(str(part) for part in parts)
    value = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)
    return value / float(16**12)


def finite_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def tok_count(text: str) -> int:
    return len(str(text or "").split())


def parse_layer_map(items: list[str]) -> dict[str, int]:
    out = {}
    for item in items:
        model, layer = item.split(":", 1)
        out[model] = int(layer)
    return out


def load_manifest(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, converters={"candidates_json": json.loads, "priors_json": json.loads})


# ── Stage 1: manifest ─────────────────────────────────────────────────────────

def build_manifest(args: argparse.Namespace) -> Path:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = dedupe_score_rows(load_jsonl_rows(source_dir / "candidate_prior_scores.jsonl"))

    manifest_rows: list[dict[str, Any]] = []
    for model, model_rows in sorted(pd.DataFrame(rows).groupby("model")):
        for dataset, dataset_rows in sorted(model_rows.groupby("dataset")):
            records = sorted(
                dataset_rows.to_dict("records"),
                key=lambda row: stable_float(args.seed, model, dataset, row["question_id"]),
            )
            if args.questions_per_dataset is not None:
                records = records[: args.questions_per_dataset]
            for row in records:
                scored = []
                for item in row.get("wrong_candidate_scores") or []:
                    prior = finite_float(item.get("prior_logp"))
                    candidate = str(item.get("candidate") or "").strip()
                    if candidate and math.isfinite(prior):
                        scored.append({"candidate": candidate, "prior": prior})
                scored = sorted(scored, key=lambda item: item["prior"], reverse=True)
                if len(scored) < args.top_k:
                    continue
                scored = scored[: args.top_k]
                manifest_rows.append(
                    {
                        "model": model,
                        "dataset": row["dataset"],
                        "question_id": str(row["question_id"]),
                        "question": row["question"],
                        "gold_answer": row["gold_answer"],
                        "candidates_json": json.dumps([item["candidate"] for item in scored]),
                        "priors_json": json.dumps([item["prior"] for item in scored]),
                        "top_k": len(scored),
                    },
                )

    manifest = pd.DataFrame(manifest_rows).sort_values(["model", "dataset", "question_id"])
    path = output_dir / "multicandidate_manifest.csv"
    manifest.to_csv(path, index=False)
    summary = manifest.groupby(["model", "dataset"], as_index=False).size().rename(columns={"size": "n_questions"})
    summary.to_csv(output_dir / "multicandidate_manifest_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return path


# ── Stage 2: candidate log-probs and last-token states (GPU) ─────────────────

def make_prompt_text(tokenizer: Any, model_cfg: Any, template: Any, question: str, candidate: str, answer_prefix: str) -> str:
    user_prompt = template.template.format(question=question, candidate=candidate)
    return format_chat(tokenizer, model_cfg, user_prompt) + answer_prefix


def encode_score_items(tokenizer: Any, prompt_text: str, answer: str) -> list[dict[str, Any]]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    out = []
    for variant in answer_variants(answer):
        answer_ids = tokenizer(variant, add_special_tokens=False).input_ids
        if prompt_ids and answer_ids:
            out.append({"prompt_ids": prompt_ids, "answer_ids": answer_ids, "variant": variant})
    return out


@torch.inference_mode()
def score_batch(model: Any, tokenizer: Any, items: list[dict[str, Any]]) -> list[float]:
    """Mean answer-token log-prob of one surface variant per item (right padding)."""
    if not items:
        return []
    device = next(model.parameters()).device
    seqs = [item["prompt_ids"] + item["answer_ids"] for item in items]
    max_len = max(len(seq) for seq in seqs)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    input_ids = torch.full((len(seqs), max_len - 1), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros_like(input_ids)
    for row, seq in enumerate(seqs):
        trimmed = seq[:-1]
        input_ids[row, : len(trimmed)] = torch.tensor(trimmed, dtype=torch.long, device=device)
        attention_mask[row, : len(trimmed)] = 1
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    log_probs = torch.nn.functional.log_softmax(out.logits.float(), dim=-1)
    scores = []
    for row, item in enumerate(items):
        start = len(item["prompt_ids"])
        token_scores = []
        for offset, target_id in enumerate(item["answer_ids"]):
            pred_pos = start + offset - 1
            if 0 <= pred_pos < log_probs.shape[1]:
                token_scores.append(float(log_probs[row, pred_pos, target_id].item()))
        scores.append(float(np.mean(token_scores)) if token_scores else float("-inf"))
    return scores


@torch.inference_mode()
def hidden_batch(model: Any, tokenizer: Any, prompts: list[str], layers: list[int]) -> dict[int, np.ndarray]:
    """Last-token output_hidden_states[L] of each prompt (slot L; slot 0 is the embeddings)."""
    device = next(model.parameters()).device
    # padding_side stays at the tokenizer's default, as in the recorded run; last_positions assumes right padding.
    encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True, use_cache=False)
    last_positions = attention_mask.sum(dim=1) - 1
    result = {}
    for layer in layers:
        hs = out.hidden_states[layer]
        result[layer] = hs[torch.arange(hs.shape[0], device=device), last_positions].float().cpu().numpy()
    return result


def score_model(args: argparse.Namespace, model_key: str) -> None:
    cfg = load_project_config(Path(args.config))
    model_cfg = cfg.models[model_key]
    model_short = model_cfg.short_name
    output_dir = Path(args.output_dir)
    score_dir = output_dir / "scores"
    feature_dir = output_dir / "features"
    score_dir.mkdir(parents=True, exist_ok=True)
    feature_dir.mkdir(parents=True, exist_ok=True)
    score_path = score_dir / f"{model_short}_multicandidate_scores.csv"
    feature_path = feature_dir / f"{model_short}_prompt_hidden.npz"
    feature_meta_path = feature_dir / f"{model_short}_prompt_hidden_rows.csv"

    manifest = load_manifest(output_dir / "multicandidate_manifest.csv")
    manifest = manifest[manifest["model"] == model_short].copy()
    if manifest.empty:
        print(f"[{model_short}] no manifest rows", flush=True)
        return

    done: set[tuple[str, str, int, int]] = set()
    if score_path.exists():
        prev = pd.read_csv(score_path)
        done = set(
            zip(
                prev["dataset"].astype(str),
                prev["question_id"].astype(str),
                prev["target_idx"].astype(int),
                prev["prompt_idx"].astype(int),
            ),
        )

    hf_token = None
    if args.use_hf_token:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise RuntimeError("--use-hf-token was given but HF_TOKEN is not set")
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
    template = cfg.prompt_templates[args.prompt_template]

    score_rows = []
    variant_items = []
    variant_owner: list[int] = []
    hidden_prompts = []
    hidden_rows = []
    for row in manifest.to_dict("records"):
        candidates = row["candidates_json"]
        priors = row["priors_json"]
        prompts = [
            make_prompt_text(tokenizer, model_cfg, template, row["question"], candidate, args.answer_prefix)
            for candidate in candidates
        ]
        for target_idx, target in enumerate(candidates):
            for prompt_idx, prompt_candidate in enumerate(candidates):
                if (str(row["dataset"]), str(row["question_id"]), target_idx, prompt_idx) in done:
                    continue
                owner = len(score_rows)
                score_rows.append(
                    {
                        "model": model_short,
                        "dataset": row["dataset"],
                        "question_id": str(row["question_id"]),
                        "target_idx": target_idx,
                        "prompt_idx": prompt_idx,
                        "target_candidate": target,
                        "prompt_candidate": prompt_candidate,
                        "target_prior_logp": priors[target_idx],
                        "prompt_prior_logp": priors[prompt_idx],
                        "target_rank": target_idx + 1,
                        "prompt_rank": prompt_idx + 1,
                        "is_self_prompt": int(target_idx == prompt_idx),
                    },
                )
                for item in encode_score_items(tokenizer, prompts[prompt_idx], target):
                    variant_items.append(item)
                    variant_owner.append(owner)
            hidden_prompts.append(prompts[target_idx])
            hidden_rows.append(
                {
                    "model": model_short,
                    "dataset": row["dataset"],
                    "question_id": str(row["question_id"]),
                    "target_idx": target_idx,
                    "target_candidate": target,
                    "target_prior_logp": priors[target_idx],
                    "target_rank": target_idx + 1,
                    "group": f"{model_short}::{row['dataset']}::{row['question_id']}",
                    "text": f"{row['question']}\nCandidate: {target}",
                },
            )

    if score_rows:
        best = np.full(len(score_rows), -np.inf, dtype=float)
        for start in range(0, len(variant_items), args.score_batch_size):
            batch = variant_items[start : start + args.score_batch_size]
            owners = variant_owner[start : start + args.score_batch_size]
            for owner, score in zip(owners, score_batch(model, tokenizer, batch)):
                if score > best[owner]:
                    best[owner] = score
            if start and (start // args.score_batch_size) % 20 == 0:
                print(f"[{model_short}] scored variants {start}/{len(variant_items)}", flush=True)
        for row, score in zip(score_rows, best):
            row["logp"] = float(score)
        pd.DataFrame(score_rows).to_csv(score_path, mode="a", header=not score_path.exists(), index=False)
        print(f"[{model_short}] wrote scores -> {score_path}", flush=True)

    layers = [parse_layer_map(args.layer_map)[model_short]]
    if args.include_last_layer:
        n_layers = getattr(model.config, "num_hidden_layers", None)
        if n_layers is not None:
            layers.append(int(n_layers))
    layers = sorted(set(layers))
    hidden_arrays = {layer: [] for layer in layers}
    if hidden_prompts and not feature_path.exists():
        for start in range(0, len(hidden_prompts), args.hidden_batch_size):
            values = hidden_batch(model, tokenizer, hidden_prompts[start : start + args.hidden_batch_size], layers)
            for layer in layers:
                hidden_arrays[layer].append(values[layer])
            if start and (start // args.hidden_batch_size) % 20 == 0:
                print(f"[{model_short}] hidden prompts {start}/{len(hidden_prompts)}", flush=True)
        arrays = {f"layer_{layer}": np.concatenate(chunks, axis=0) for layer, chunks in hidden_arrays.items()}
        arrays["layers"] = np.asarray(layers, dtype=np.int32)
        np.savez_compressed(feature_path, **arrays)
        pd.DataFrame(hidden_rows).to_csv(feature_meta_path, index=False)
        print(f"[{model_short}] wrote hidden -> {feature_path}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── Stage 3: pair-centred role-swap analysis ──────────────────────────────────

def score_lookup(scores: pd.DataFrame) -> dict[tuple[str, int, int], float]:
    return {(str(row.question_id), int(row.target_idx), int(row.prompt_idx)): float(row.logp) for row in scores.itertuples()}


def build_effect_rows(source_dir: Path, model: str) -> pd.DataFrame:
    """For each unordered candidate pair: effect(A) = log p(A | note A) - log p(A | note B), centred on the pair mean."""
    manifest = load_manifest(source_dir / "multicandidate_manifest.csv")
    manifest = manifest[manifest["model"] == model].copy()
    scores = pd.read_csv(source_dir / "scores" / f"{model}_multicandidate_scores.csv")
    scores = scores[scores["model"] == model].copy()
    lookup = score_lookup(scores)
    rows: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        qid = str(row["question_id"])
        candidates = list(row["candidates_json"])
        priors = [float(x) for x in row["priors_json"]]
        for left_idx, right_idx in combinations(range(len(candidates)), 2):
            pair_id = f"{model}::{row['dataset']}::{qid}::{left_idx}_{right_idx}"
            effects: dict[int, float] = {}
            for target_idx, control_idx in [(left_idx, right_idx), (right_idx, left_idx)]:
                self_key = (qid, target_idx, target_idx)
                control_key = (qid, target_idx, control_idx)
                if self_key not in lookup or control_key not in lookup:
                    continue
                effects[target_idx] = lookup[self_key] - lookup[control_key]
            if len(effects) != 2:
                continue
            pair_mean = float(np.mean(list(effects.values())))
            pair_abs_gap = float(abs(effects[left_idx] - effects[right_idx]))
            for target_idx, control_idx in [(left_idx, right_idx), (right_idx, left_idx)]:
                target = str(candidates[target_idx])
                control = str(candidates[control_idx])
                centered = float(effects[target_idx] - pair_mean)
                rows.append(
                    {
                        "model": model,
                        "dataset": row["dataset"],
                        "question_id": qid,
                        "pair_id": pair_id,
                        "target_idx": target_idx,
                        "control_idx": control_idx,
                        "target_candidate": target,
                        "control_candidate": control,
                        "target_prior_logp": priors[target_idx],
                        "control_prior_logp": priors[control_idx],
                        "prior_gap_target_minus_control": priors[target_idx] - priors[control_idx],
                        "target_rank": target_idx + 1,
                        "control_rank": control_idx + 1,
                        "target_len_chars": len(target),
                        "control_len_chars": len(control),
                        "target_tok_count": tok_count(target),
                        "control_tok_count": tok_count(control),
                        "self_logp": lookup[(qid, target_idx, target_idx)],
                        "control_logp": lookup[(qid, target_idx, control_idx)],
                        "effect_self_minus_control": effects[target_idx],
                        "pair_mean_effect": pair_mean,
                        "pair_centered_effect": centered,
                        "pair_abs_gap": pair_abs_gap,
                        "binary_label": int(centered > 0),
                        "group": f"{model}::{row['dataset']}::{qid}",
                        "text": f"Question: {row['question']}\nTarget candidate: {target}\nControl candidate: {control}",
                    }
                )
    return pd.DataFrame(rows)


def load_hidden(source_dir: Path, model: str) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    meta = pd.read_csv(source_dir / "features" / f"{model}_prompt_hidden_rows.csv")
    data = np.load(source_dir / "features" / f"{model}_prompt_hidden.npz")
    layers = [int(x) for x in data["layers"].tolist()]
    return meta, {layer: data[f"layer_{layer}"] for layer in layers}


def hidden_features(effect_rows: pd.DataFrame, meta: pd.DataFrame, hidden: np.ndarray, source: str) -> np.ndarray:
    """W: the note names the target; G: it names the other candidate."""
    index = {(str(row.question_id), int(row.target_idx)): idx for idx, row in enumerate(meta.itertuples())}
    feats = []
    for row in effect_rows.itertuples():
        w = hidden[index[(str(row.question_id), int(row.target_idx))]]
        g = hidden[index[(str(row.question_id), int(row.control_idx))]]
        if source == "W_hidden":
            feats.append(w)
        elif source == "G_hidden":
            feats.append(g)
        elif source == "W_minus_G":
            feats.append(w - g)
        else:
            raise ValueError(source)
    return np.asarray(feats, dtype=np.float32)


def public_numeric(df: pd.DataFrame) -> np.ndarray:
    """11 public features: priors and gap, ranks, lengths (chars, words), dataset indicators."""
    dense = pd.DataFrame(
        {
            "target_prior_logp": df["target_prior_logp"].astype(float),
            "control_prior_logp": df["control_prior_logp"].astype(float),
            "prior_gap_target_minus_control": df["prior_gap_target_minus_control"].astype(float),
            "target_rank": df["target_rank"].astype(float),
            "control_rank": df["control_rank"].astype(float),
            "target_len_chars": df["target_len_chars"].astype(float),
            "control_len_chars": df["control_len_chars"].astype(float),
            "target_tok_count": df["target_tok_count"].astype(float),
            "control_tok_count": df["control_tok_count"].astype(float),
            "dataset_is_hotpotqa": (df["dataset"] == "hotpotqa").astype(float),
            "dataset_is_triviaqa": (df["dataset"] == "triviaqa").astype(float),
        }
    )
    return dense.to_numpy(dtype=np.float32)


def analyze(args: argparse.Namespace) -> None:
    """Binary AUROC scores the ridge OOF predictions of the continuous effect against its sign."""
    source_dir = Path(args.output_dir)
    output_dir = source_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    reg_rows, bin_rows, pred_rows, effect_frames = [], [], [], []

    for model in args.model_shorts:
        effects = build_effect_rows(source_dir, model)
        if effects.empty:
            continue
        effects.to_csv(output_dir / f"{model}_natural_role_swapped_effect_rows.csv", index=False)
        effect_frames.append(effects)
        groups = effects["group"].astype(str).to_numpy()
        y_reg = effects["pair_centered_effect"].to_numpy(dtype=float)
        y_bin = effects["binary_label"].to_numpy(dtype=int)

        methods: dict[str, np.ndarray] = {"numeric_public": public_numeric(effects)}
        meta, arrays = load_hidden(source_dir, model)
        for layer, hidden in arrays.items():
            for source in ["W_hidden", "G_hidden", "W_minus_G"]:
                methods[f"{source}_L{layer}"] = hidden_features(effects, meta, hidden, source)

        records = effects.to_dict("records")
        for method, x in methods.items():
            print(f"[{model}] fitting {method} shape={x.shape}", flush=True)
            layer = float(method.rsplit("_L", 1)[1]) if "_L" in method else math.nan
            pred_r, used_r = grouped_ridge_oof_fixed(x, y_reg, groups, alpha=args.ridge_alpha)
            reg_rows.append(
                {
                    "model": model,
                    "method": method,
                    "layer": layer,
                    "n": len(effects),
                    "target_mean": float(np.mean(y_reg)),
                    "target_std": float(np.std(y_reg)),
                    **summarize_regression(y_reg, pred_r, used_r),
                }
            )
            bin_rows.append(
                {
                    "model": model,
                    "method": method,
                    "layer": layer,
                    "n": len(effects),
                    "positives": int(y_bin.sum()),
                    "negatives": int((1 - y_bin).sum()),
                    **summarize_binary(y_bin, pred_r, used_r),
                }
            )
            for row, score, used in zip(records, pred_r, used_r):
                pred_rows.append(
                    {
                        "model": model,
                        "method": method,
                        "layer": layer,
                        "dataset": row["dataset"],
                        "question_id": row["question_id"],
                        "pair_id": row["pair_id"],
                        "target_idx": row["target_idx"],
                        "control_idx": row["control_idx"],
                        "pair_centered_effect": row["pair_centered_effect"],
                        "binary_label": row["binary_label"],
                        "prediction": float(score) if np.isfinite(score) else math.nan,
                        "used": bool(used),
                    }
                )

    reg = pd.DataFrame(reg_rows)
    binary = pd.DataFrame(bin_rows)
    reg.to_csv(output_dir / "natural_role_swapped_regression_summary.csv", index=False)
    binary.to_csv(output_dir / "natural_role_swapped_binary_summary.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(output_dir / "natural_role_swapped_predictions.csv", index=False)
    if effect_frames:
        pd.concat(effect_frames, ignore_index=True).to_csv(output_dir / "natural_role_swapped_effect_rows_all.csv", index=False)
    if not binary.empty:
        print(binary.sort_values(["model", "auroc"], ascending=[True, False]).round(4).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--source-dir", help="prior-aware v2 dir with candidate_prior_scores.jsonl (--build-manifest)")
    parser.add_argument("--output-dir", required=True, help="benchmark dir; the analysis is written to <dir>/analysis")
    parser.add_argument("--models", nargs="+", default=["mistral_7b", "gemma_9b", "llama_8b"])
    parser.add_argument("--model-shorts", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--prompt-template", default="note_block_v1")
    parser.add_argument("--answer-prefix", default="\nFinal answer:")
    parser.add_argument("--questions-per-dataset", type=int, default=160)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--use-hf-token", action="store_true")
    parser.add_argument("--layer-map", nargs="+", default=["mistral:11", "gemma:13", "llama:13"])
    parser.add_argument("--include-last-layer", action="store_true", help="also save the last slot (not recorded for the paper run)")
    parser.add_argument("--score-batch-size", type=int, default=16)
    parser.add_argument("--hidden-batch-size", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    if not (args.build_manifest or args.score or args.analyze):
        parser.error("Specify --build-manifest, --score and/or --analyze")
    if args.build_manifest:
        if not args.source_dir:
            parser.error("--build-manifest needs --source-dir")
        build_manifest(args)
    if args.score:
        for model_key in args.models:
            score_model(args, model_key)
    if args.analyze:
        analyze(args)


if __name__ == "__main__":
    main()
