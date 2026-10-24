"""Cross-question donor control for full-state role-swap patching (tab:roleswap-patch-controls; results text).

Recipient: question q, note names B. Matched donor: same q, note names C. Cross-base donor: another question with
the same ordered note letters. Input: a generate_benchmark.py output dir (manifest, and
role_swapped_effect_rows_all.csv, which only generate_benchmark.py --analyze writes). Output:
{model}_L{layer}_crossbase_full_patch_results.csv and crossbase_full_patch_summary.csv. GPU, HF_TOKEN.
Paper run (Qwen3-32B, defaults: layer 63, 24 swap pairs x 2 directions, seed 0, prompt end):
    python -m scripts.pipeline.roleswap.crossbase_patch_control --data-dir <qwen32_dir> --output-dir <out> --models qwen3_32b --use-hf-token
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.common.model_loader import DTYPE_MAP, format_chat, check_hf_access
from scripts.pipeline.roleswap.generate_benchmark import build_effect_rows


# ── Hooks and letter scoring ──────────────────────────────────────────────────

def get_layers_module(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise TypeError(f"Could not find transformer layers for {type(model)}")


def get_output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def replace_output_tensor(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


def answer_variants(letter: str) -> list[str]:
    return [f" {letter}", letter, f" {letter})"]


def encode_variants(tokenizer: Any, prompt_text: str, candidate_letters: list[str]) -> tuple[list[str], list[int], list[list[int]], int]:
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    labels = []
    rows = []
    for letter in candidate_letters:
        for text in answer_variants(letter):
            answer_ids = tokenizer(text, add_special_tokens=False).input_ids
            if answer_ids:
                labels.append(letter)
                rows.append(prompt_ids + answer_ids)
    return labels, prompt_ids, rows, len(prompt_ids) - 1


@torch.inference_mode()
def score_letters(
    model: Any,
    tokenizer: Any,
    layer: int,
    prompt_text: str,
    candidate_letters: list[str],
    donor_vec: torch.Tensor | None = None,
) -> dict[str, float]:
    """Mean token log-prob of each letter (best of three surface forms); optionally patch the last prompt token."""
    labels, prompt_ids, rows, patch_pos = encode_variants(tokenizer, prompt_text, candidate_letters)
    if not rows:
        return {letter: float("-inf") for letter in candidate_letters}
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    max_len = max(len(row) for row in rows)
    input_ids = torch.full((len(rows), max_len), int(pad_id), dtype=torch.long, device=model.device)
    attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=model.device)
    for idx, row in enumerate(rows):
        input_ids[idx, : len(row)] = torch.tensor(row, dtype=torch.long, device=model.device)
        attention_mask[idx, : len(row)] = 1

    handle = None
    if donor_vec is not None:
        layers = get_layers_module(model)
        donor = donor_vec.to(device=model.device, dtype=next(model.parameters()).dtype)

        def hook(_module, _inputs, output):
            hidden = get_output_tensor(output).clone()
            hidden[:, patch_pos, :] = donor
            return replace_output_tensor(output, hidden)

        handle = layers[layer].register_forward_hook(hook)
    try:
        out = model(input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=False)
    finally:
        if handle is not None:
            handle.remove()

    log_probs = torch.nn.functional.log_softmax(out.logits.float(), dim=-1)
    best = {letter: float("-inf") for letter in candidate_letters}
    start = len(prompt_ids)
    for idx, (letter, full_row) in enumerate(zip(labels, rows)):
        answer_ids = full_row[len(prompt_ids):]
        token_scores = []
        for offset, target_id in enumerate(answer_ids):
            pred_pos = start + offset - 1
            if 0 <= pred_pos < log_probs.shape[1]:
                token_scores.append(float(log_probs[idx, pred_pos, target_id].item()))
        if token_scores:
            best[letter] = max(best[letter], float(np.mean(token_scores)))
    return best


def load_patch_cases(data_dir: Path, model_short: str, n_pairs: int) -> pd.DataFrame:
    """Both directions of the n_pairs swap pairs with the largest pair gap; donor = the other orientation."""
    effects = pd.read_csv(data_dir / "role_swapped_effect_rows_all.csv")
    effects = effects[effects["model"] == model_short].copy()
    manifest = pd.read_csv(data_dir / "role_swapped_manifest.csv")
    manifest = manifest[(manifest["condition"] == "W")].copy()
    prompt_map = manifest.set_index("orientation_id")["prompt"].to_dict()
    rows = []
    for pair_id, sub in effects.groupby("swap_pair_id"):
        if len(sub) != 2:
            continue
        sub = sub.sort_values("pair_centered_effect", ascending=False)
        high = sub.iloc[0].to_dict()
        low = sub.iloc[1].to_dict()
        if float(high["pair_centered_effect"]) <= float(low["pair_centered_effect"]):
            continue
        for donor, recipient, direction in [(high, low, "high_to_low"), (low, high, "low_to_high")]:
            rows.append(
                {
                    "model": model_short,
                    "swap_pair_id": pair_id,
                    "direction": direction,
                    "pair_abs_gap": float(high["pair_abs_gap"]),
                    "donor_orientation_id": donor["orientation_id"],
                    "recipient_orientation_id": recipient["orientation_id"],
                    "donor_candidate": str(donor["h1"]),
                    "recipient_candidate": str(recipient["h1"]),
                    "donor_prompt": prompt_map[str(donor["orientation_id"])],
                    "recipient_prompt": prompt_map[str(recipient["orientation_id"])],
                    "donor_centered_effect": float(donor["pair_centered_effect"]),
                    "recipient_centered_effect": float(recipient["pair_centered_effect"]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    selected_pairs = (
        out.drop_duplicates("swap_pair_id").sort_values("pair_abs_gap", ascending=False).head(n_pairs)["swap_pair_id"]
    )
    return out[out["swap_pair_id"].isin(set(selected_pairs))].copy()


def build_prompt_maps(manifest: pd.DataFrame, tokenizer: Any, model_cfg: Any, answer_prefix: str) -> dict[tuple[str, str], str]:
    out: dict[tuple[str, str], str] = {}
    for row in manifest.to_dict("records"):
        key = (str(row["orientation_id"]), str(row["condition"]))
        out[key] = format_chat(tokenizer, model_cfg, str(row["prompt"])) + answer_prefix
    return out


@torch.inference_mode()
def capture_prompt_vectors(
    model: Any,
    tokenizer: Any,
    layer: int,
    items: list[tuple[str, str]],
    prompt_map: dict[tuple[str, str], str],
    batch_size: int,
) -> dict[tuple[str, str], np.ndarray]:
    """Last-prompt-token output of one layer, captured through the same hook used for patching (right padding)."""
    layers = get_layers_module(model)
    result: dict[tuple[str, str], np.ndarray] = {}
    old_padding = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in tqdm(range(0, len(items), batch_size), desc=f"capture L{layer}"):
            batch_items = items[start : start + batch_size]
            prompts = [prompt_map[item] for item in batch_items]
            encoded = tokenizer(prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(model.device)
            last_positions = encoded["attention_mask"].sum(dim=1) - 1
            captured: dict[str, torch.Tensor] = {}

            def hook(_module, _inputs, output):
                hidden = get_output_tensor(output)
                captured["vecs"] = torch.stack(
                    [hidden[i, int(pos.item())].detach().clone() for i, pos in enumerate(last_positions)]
                )
                return output

            handle = layers[layer].register_forward_hook(hook)
            try:
                model(**encoded, use_cache=False)
            finally:
                handle.remove()
            arr = captured["vecs"].float().cpu().numpy()
            for item, vec in zip(batch_items, arr):
                result[item] = vec.astype(np.float32)
    finally:
        tokenizer.padding_side = old_padding
    return result


# ── Cross-base control ────────────────────────────────────────────────────────

def _orientation_lookup(effect_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows = effect_rows.drop_duplicates("orientation_id").to_dict("records")
    return {str(row["orientation_id"]): row for row in rows}


def choose_crossbase_donor(
    effect_rows: pd.DataFrame,
    case: dict[str, Any],
    orientation_info: dict[str, dict[str, Any]],
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    """A different-question donor with the same ordered note candidates, drawn at random (seeded)."""
    donor_info = orientation_info[str(case["donor_orientation_id"])]
    recipient_info = orientation_info[str(case["recipient_orientation_id"])]
    blocked_base_ids = {str(donor_info["base_id"]), str(recipient_info["base_id"])}
    blocked_pair_ids = {str(donor_info["swap_pair_id"]), str(recipient_info["swap_pair_id"])}

    candidates = effect_rows[
        (effect_rows["h1"].astype(str) == str(case["donor_candidate"]))
        & (effect_rows["h2"].astype(str) == str(case["recipient_candidate"]))
        & (~effect_rows["base_id"].astype(str).isin(blocked_base_ids))
        & (~effect_rows["swap_pair_id"].astype(str).isin(blocked_pair_ids))
    ].copy()
    if candidates.empty:
        return None
    idx = int(rng.integers(0, len(candidates)))
    return candidates.iloc[idx].to_dict()


def summarize(output_dir: Path) -> None:
    frames = [pd.read_csv(path) for path in output_dir.glob("*_crossbase_full_patch_results.csv")]
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    summary = (
        df.groupby(["model", "layer", "donor_source"], as_index=False)
        .agg(
            n=("delta_margin_toward_donor", "size"),
            mean_delta_margin_toward_donor=("delta_margin_toward_donor", "mean"),
            median_delta_margin_toward_donor=("delta_margin_toward_donor", "median"),
            frac_positive_delta=("delta_margin_toward_donor", lambda x: float((x > 0).mean())),
            mean_abs_delta=("delta_margin_toward_donor", lambda x: float(np.abs(x).mean())),
        )
        .sort_values(["model", "layer", "donor_source"])
    )
    summary.to_csv(output_dir / "crossbase_full_patch_summary.csv", index=False)
    print(summary.round(4).to_string(index=False), flush=True)


def run_model(args: argparse.Namespace, model_key: str) -> None:
    model_cfg = load_project_config(Path(args.config)).models[model_key]
    model_short = model_cfg.short_name
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    effect_rows = build_effect_rows(data_dir, model_short)
    if effect_rows.empty:
        print(f"[{model_short}] no effect rows", flush=True)
        return
    cases = load_patch_cases(data_dir, model_short, args.n_pairs)
    if cases.empty:
        print(f"[{model_short}] no patch cases", flush=True)
        return

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

    layer = args.layer
    if layer < 0:
        layer = len(get_layers_module(model)) + layer

    manifest = pd.read_csv(data_dir / "role_swapped_manifest.csv")
    prompt_map = build_prompt_maps(manifest, tokenizer, model_cfg, args.answer_prefix)
    orientation_info = _orientation_lookup(effect_rows)
    rng = np.random.default_rng(args.seed)

    chosen_sources: list[dict[str, Any]] = []
    needed: set[tuple[str, str]] = set()
    for case in cases.to_dict("records"):
        donor_oid = str(case["donor_orientation_id"])
        recipient_oid = str(case["recipient_orientation_id"])
        needed.add((donor_oid, "W"))
        needed.add((recipient_oid, "W"))
        chosen_sources.append(
            {
                "case": case,
                "donor_source": "matched_same_question",
                "source_orientation_id": donor_oid,
                "source_base_id": orientation_info[donor_oid]["base_id"],
                "source_swap_pair_id": orientation_info[donor_oid]["swap_pair_id"],
            }
        )
        cross = choose_crossbase_donor(effect_rows, case, orientation_info, rng)
        if cross is not None:
            cross_oid = str(cross["orientation_id"])
            needed.add((cross_oid, "W"))
            chosen_sources.append(
                {
                    "case": case,
                    "donor_source": "crossbase_same_order_letters",
                    "source_orientation_id": cross_oid,
                    "source_base_id": cross["base_id"],
                    "source_swap_pair_id": cross["swap_pair_id"],
                }
            )

    print(f"[{model_short}] cross-base capture layer={layer} prompts={len(needed)} cases={len(cases)} sources={len(chosen_sources)}", flush=True)
    live = capture_prompt_vectors(model, tokenizer, layer, sorted(needed), prompt_map, args.capture_batch_size)

    rows: list[dict[str, Any]] = []
    for item in tqdm(chosen_sources, desc=f"{model_short} cross-base patch"):
        case = item["case"]
        recipient_prompt = prompt_map[(str(case["recipient_orientation_id"]), "W")]
        donor_candidate = str(case["donor_candidate"])
        recipient_candidate = str(case["recipient_candidate"])
        candidates = list(dict.fromkeys([donor_candidate, recipient_candidate]))
        base_scores = score_letters(model, tokenizer, layer, recipient_prompt, candidates, donor_vec=None)
        base_margin = base_scores[donor_candidate] - base_scores[recipient_candidate]
        source_vec = live[(str(item["source_orientation_id"]), "W")]
        patched_scores = score_letters(
            model, tokenizer, layer, recipient_prompt, candidates, donor_vec=torch.tensor(source_vec, dtype=torch.float32)
        )
        patched_margin = patched_scores[donor_candidate] - patched_scores[recipient_candidate]
        donor_info = orientation_info[str(case["donor_orientation_id"])]
        recipient_info = orientation_info[str(case["recipient_orientation_id"])]
        rows.append(
            {
                "model": model_short,
                "layer": int(layer),
                "swap_pair_id": case["swap_pair_id"],
                "direction": case["direction"],
                "pair_abs_gap": float(case["pair_abs_gap"]),
                "donor_orientation_id": case["donor_orientation_id"],
                "recipient_orientation_id": case["recipient_orientation_id"],
                "donor_base_id": donor_info["base_id"],
                "recipient_base_id": recipient_info["base_id"],
                "donor_candidate": donor_candidate,
                "recipient_candidate": recipient_candidate,
                "donor_source": item["donor_source"],
                "source_orientation_id": item["source_orientation_id"],
                "source_base_id": item["source_base_id"],
                "source_swap_pair_id": item["source_swap_pair_id"],
                "base_margin_donor_minus_recipient": float(base_margin),
                "patched_margin_donor_minus_recipient": float(patched_margin),
                "delta_margin_toward_donor": float(patched_margin - base_margin),
            }
        )

    out_path = output_dir / f"{model_short}_L{layer}_crossbase_full_patch_results.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[{model_short}] wrote {out_path}", flush=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--data-dir", required=True, help="generate_benchmark.py output dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", nargs="+", default=["qwen3_32b"])
    parser.add_argument("--layer", type=int, default=63)
    parser.add_argument("--n-pairs", type=int, default=24)
    parser.add_argument("--capture-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--answer-prefix", default="", help="empty = patch at the prompt end")
    parser.add_argument("--use-hf-token", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="rebuild the summary from the per-case files")
    args = parser.parse_args()

    if not args.summarize_only:
        for model_key in args.models:
            run_model(args, model_key)
    summarize(Path(args.output_dir))


if __name__ == "__main__":
    main()
