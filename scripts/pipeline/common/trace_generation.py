"""Generate one trace per condition record, saving prompt-end hidden states and answer commitment scores.

Batch size 1, no padding, KV cache; hooks keep hidden[0, -1, :] of each decoder layer (no final norm). Writes
traces/{model}_traces.jsonl, features/*.npz (pos_-1_layer_L = prompt end, pos_<last>_layer_L = last generated token)
and, for W and G, features/*_scores.json. Used by generate_traces.py and larger_models/generate.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from scripts.pipeline.common.config import ModelConfig
from scripts.pipeline.common.hook_manager import HookManager
from scripts.pipeline.common.model_loader import ChatFormatter, load_model
from scripts.pipeline.common.sentence_boundaries import segment_sentences
from src.core.labels import assign_behavioral_label
from src.core.normalize import parse_response
from src.core.types import ConditionRecord, GenerationResult, TraceRecord


# ── Decoding ──────────────────────────────────────────────────────────────────

@torch.inference_mode()
def generate_with_hidden_states(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    layers: list[int],
    layer_accessor: str = "model.model.layers",
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> GenerationResult:
    """Token-by-token decode; keeps hidden states at the prompt end (-1) and at the last generated token."""
    device = input_ids.device
    eos_id = tokenizer.eos_token_id

    hook_mgr = HookManager(model, layers, layer_accessor)
    hidden_states: dict[int, dict[int, torch.Tensor]] = {}
    generated_ids: list[int] = []
    past_key_values = None

    try:
        # Prefill pass: prompt-end states.
        with hook_mgr.extract_step():
            outputs = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
            )
        past_key_values = outputs.past_key_values
        hidden_states[-1] = hook_mgr.get_and_clear()

        next_token_id = _sample_token(outputs.logits[0, -1, :], temperature, top_p)
        generated_ids.append(next_token_id)
        if next_token_id == eos_id:
            return _build_result(tokenizer, generated_ids, hidden_states, past_key_values)

        for _ in range(1, max_new_tokens):
            cur_token = torch.tensor([[generated_ids[-1]]], device=device)
            outputs = model(
                input_ids=cur_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            next_token_id = _sample_token(outputs.logits[0, -1, :], temperature, top_p)
            generated_ids.append(next_token_id)
            if next_token_id == eos_id:
                break

        # Pre-answer position: run the last generated token once more with extraction on.
        last_pos = len(generated_ids) - 1
        if last_pos not in hidden_states and last_pos > 0:
            cur_token = torch.tensor([[generated_ids[-1]]], device=device)
            with hook_mgr.extract_step():
                outputs = model(
                    input_ids=cur_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            hidden_states[last_pos] = hook_mgr.get_and_clear()
            past_key_values = outputs.past_key_values

    finally:
        hook_mgr.remove_all()

    return _build_result(tokenizer, generated_ids, hidden_states, past_key_values)


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Greedy at temperature 0, otherwise nucleus sampling."""
    if temperature <= 0.0:
        return logits.argmax(dim=-1).item()

    logits = logits / temperature

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
        sorted_indices_to_remove[0] = False
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = float("-inf")

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def _build_result(tokenizer, generated_ids: list[int], hidden_states: dict, past_key_values) -> GenerationResult:
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return GenerationResult(
        token_ids=generated_ids,
        text=text,
        hidden_states=hidden_states,
        kv_cache=past_key_values,
    )


def clone_kv_cache(kv_cache: DynamicCache) -> DynamicCache:
    """Deep-clone a DynamicCache (the cache every model here returns under the pinned transformers)."""
    if not isinstance(kv_cache, DynamicCache):
        raise TypeError(f"expected a DynamicCache, got {type(kv_cache).__name__}")
    cloned = DynamicCache()
    for layer_idx in range(len(kv_cache)):
        layer = kv_cache.layers[layer_idx]
        cloned.update(layer.keys.clone(), layer.values.clone(), layer_idx)
    return cloned


# ── Commitment scores (forced decoding after the generated trace) ─────────────

@torch.inference_mode()
def compute_commitment_scores(model, tokenizer, past_key_values, answer_strings: list[str]) -> dict[str, float]:
    """Length-normalised log P(answer | trace) per answer string; the first answer token is not scored."""
    device = next(model.parameters()).device
    scores = {}

    for answer_str in answer_strings:
        if not answer_str.strip():
            scores[answer_str] = float("-inf")
            continue

        kv = clone_kv_cache(past_key_values)
        answer_tokens = tokenizer.encode(answer_str, add_special_tokens=False)
        if not answer_tokens:
            scores[answer_str] = float("-inf")
            continue

        total_log_prob = 0.0
        cur_kv = kv
        for i, target_id in enumerate(answer_tokens):
            if i == 0:
                # No logits for the first answer token are kept, so feed it and skip its score.
                outputs = model(
                    input_ids=torch.tensor([[target_id]], device=device),
                    past_key_values=cur_kv,
                    use_cache=True,
                )
                cur_kv = outputs.past_key_values
                continue

            outputs = model(
                input_ids=torch.tensor([[answer_tokens[i - 1]]], device=device),
                past_key_values=cur_kv,
                use_cache=True,
            )
            cur_kv = outputs.past_key_values
            log_probs = F.log_softmax(outputs.logits[0, -1, :], dim=-1)
            total_log_prob += log_probs[target_id].item()

        n_scored = max(len(answer_tokens) - 1, 1)
        scores[answer_str] = total_log_prob / n_scored

    return scores


def commitment_score(
    model,
    tokenizer,
    past_key_values,
    gold_answer: str,
    hint_answer: str,
    gold_aliases: list[str] | None = None,
) -> dict[str, float]:
    """C = L_hint - L_gold (max over gold aliases); positive means more committed to the hint."""
    gold_candidates = [gold_answer] + (gold_aliases or [])
    all_answers = list(set([hint_answer] + gold_candidates))
    scores = compute_commitment_scores(model, tokenizer, past_key_values, all_answers)

    l_hint = scores.get(hint_answer, float("-inf"))
    l_gold = max(scores.get(g, float("-inf")) for g in gold_candidates)
    return {"L_gold": l_gold, "L_hint": l_hint, "C": l_hint - l_gold}


def commitment_score_mcq(model, tokenizer, past_key_values, gold_letter: str, hint_letter: str) -> dict[str, float]:
    """MCQ commitment: next-token log-prob of each option letter after one space token."""
    device = next(model.parameters()).device

    space_id = tokenizer.encode(" ", add_special_tokens=False)
    if not space_id:
        space_id = [tokenizer.eos_token_id]

    kv = clone_kv_cache(past_key_values)
    outputs = model(input_ids=torch.tensor([[space_id[0]]], device=device), past_key_values=kv, use_cache=True)
    log_probs = F.log_softmax(outputs.logits[0, -1, :], dim=-1)

    def _letter_logprob(letter: str) -> float:
        token_id = tokenizer.encode(letter, add_special_tokens=False)
        if token_id:
            return log_probs[token_id[0]].item()
        return float("-inf")

    l_gold = _letter_logprob(gold_letter)
    l_hint = _letter_logprob(hint_letter)
    return {"L_gold": l_gold, "L_hint": l_hint, "C": l_hint - l_gold}


# ── One trace, one model ──────────────────────────────────────────────────────

def _safe(part) -> str:
    """An id used in a file name; refuse path separators and '..'."""
    text = str(part)
    if "/" in text or "\\" in text or ".." in text:
        raise ValueError(f"unsafe id for a file name: {text!r}")
    return text


def generate_trace(
    model,
    tokenizer,
    formatter: ChatFormatter,
    record: ConditionRecord,
    model_cfg: ModelConfig,
    layers: list[int],
    output_dir: Path,
    seed: int = 42,
) -> TraceRecord:
    """Generate a single trace, save its hidden states and commitment scores, return the trace row."""
    dec = model_cfg.canonical_decoding
    input_ids = formatter.format(record.prompt)

    result = generate_with_hidden_states(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        max_new_tokens=dec.max_new_tokens,
        layers=layers,
        layer_accessor=model_cfg.layer_accessor,
        temperature=dec.temperature,
        top_p=dec.top_p,
    )

    parse_result = parse_response(result.text, record.answer_type)
    sentence_spans = segment_sentences(parse_result.reasoning_text)
    label = assign_behavioral_label(
        normalized_answer=parse_result.final_answer_norm,
        hint_candidate=record.candidate,
        gold_answer=record.gold_answer,
        gold_aliases=record.gold_aliases,
    )

    hs_path = None
    if result.hidden_states:
        hs_filename = f"{_safe(record.dataset)}_{_safe(record.question_id)}_{record.condition}_{model_cfg.short_name}.npz"
        hs_path = str(output_dir / "features" / hs_filename)
        Path(hs_path).parent.mkdir(parents=True, exist_ok=True)
        hs_arrays = {}
        for pos, layer_dict in result.hidden_states.items():
            for layer_idx, tensor in layer_dict.items():
                key = f"pos_{pos}_layer_{layer_idx}"
                hs_arrays[key] = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
        np.savez_compressed(hs_path, **hs_arrays)

    score_path = None
    if result.kv_cache and record.condition in ("W", "G"):
        try:
            if record.answer_type == "mcq":
                scores = commitment_score_mcq(
                    model, tokenizer, result.kv_cache,
                    gold_letter=record.gold_answer,
                    hint_letter=record.candidate,
                )
            else:
                scores = commitment_score(
                    model, tokenizer, result.kv_cache,
                    gold_answer=record.gold_answer,
                    hint_answer=record.candidate,
                    gold_aliases=record.gold_aliases,
                )
            score_filename = (
                f"{_safe(record.dataset)}_{_safe(record.question_id)}_{record.condition}_{model_cfg.short_name}_scores.json"
            )
            score_path = str(output_dir / "features" / score_filename)
            with open(score_path, "w") as f:
                json.dump(scores, f)
        except Exception as e:
            print(f"  Warning: commitment score failed for {record.question_id}: {e}")

    return TraceRecord(
        dataset=record.dataset,
        question_id=record.question_id,
        split=record.split,
        model=model_cfg.short_name,
        model_mode="thinking" if model_cfg.enable_thinking else "instruct",
        template_id=record.template_id,
        condition=record.condition,
        candidate=record.candidate,
        alt_wrong_candidate=record.alt_wrong_candidate,
        gold_answer=record.gold_answer,
        seed=seed,
        decoding_params={"temperature": dec.temperature, "top_p": dec.top_p, "max_new_tokens": dec.max_new_tokens},
        raw_prompt=record.prompt,
        raw_response=result.text,
        think_text=None,
        final_text=parse_result.reasoning_text + ("\nFinal answer: " + parse_result.final_answer_raw if parse_result.final_answer_raw else ""),
        normalized_final=parse_result.final_answer_norm,
        label_behavior=label,
        parse_ok=parse_result.parse_mode != "fail",
        parse_mode=parse_result.parse_mode,
        truncated=len(result.token_ids) >= dec.max_new_tokens,
        sentence_spans=[{"start_char": s.start_char, "end_char": s.end_char, "text": s.text} for s in sentence_spans],
        hidden_state_path=hs_path,
        answer_score_path=score_path,
    )


def generate_traces_for_model(
    model_cfg: ModelConfig,
    records: list[ConditionRecord],
    layers: list[int],
    output_dir: Path,
    seed: int = 42,
) -> list[TraceRecord]:
    """Generate all traces for one model; resumes from an existing traces file. Empty layers = all layers."""
    model, tokenizer = load_model(model_cfg)
    formatter = ChatFormatter(tokenizer, model_cfg)
    if not layers:
        layers = list(range(model.config.num_hidden_layers))

    traces_path = output_dir / "traces" / f"{model_cfg.short_name}_traces.jsonl"
    traces_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys = set()
    if traces_path.exists():
        with open(traces_path) as f:
            for line in f:
                try:
                    d = json.loads(line.strip())
                    done_keys.add((d["dataset"], d["question_id"], d["condition"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"  [{model_cfg.short_name}] Resuming: {len(done_keys)} traces already done")

    remaining = [r for r in records if (r.dataset, r.question_id, r.condition) not in done_keys]
    print(f"  [{model_cfg.short_name}] {len(remaining)} records to generate ({len(records) - len(remaining)} skipped)")

    traces = []
    with open(traces_path, "a") as f:
        for i, record in enumerate(remaining):
            trace = generate_trace(model, tokenizer, formatter, record, model_cfg, layers, output_dir, seed)
            f.write(json.dumps(trace.to_dict()) + "\n")
            f.flush()
            traces.append(trace)
            if (i + 1) % 25 == 0:
                print(f"  [{model_cfg.short_name}] {i+1}/{len(remaining)}")

    print(f"  [{model_cfg.short_name}] Done. {len(traces)} new traces saved to {traces_path}")
    return traces
