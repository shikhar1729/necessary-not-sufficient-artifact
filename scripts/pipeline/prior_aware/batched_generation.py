"""Batched generation and prompt-end state extraction shared by build_v2.py, prescreen.py and extract_states.py,
plus the small row helpers the prior_aware scripts share.

Generation is greedy and batched with left padding (transformers), or vLLM (imported only when asked for).
Prompt-end states come from a separate right-padded forward pass with output_hidden_states: layer L is
slot L+1, so the last layer is after the final norm. One NPZ per row: features/{dataset}_{qid}_{cond}_{model}.npz.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from scripts.pipeline.common.config import ModelConfig
from scripts.pipeline.common.model_loader import DTYPE_MAP, format_chat, check_hf_access
from scripts.pipeline.common.sentence_boundaries import segment_sentences
from src.core.labels import assign_behavioral_label
from src.core.normalize import parse_response
from src.core.types import ConditionRecord, TraceRecord


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def truthy(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def stable_int(*parts: Any) -> int:
    text = "::".join(str(p) for p in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:12], 16)


def load_condition_records(path: Path) -> list[ConditionRecord]:
    return [ConditionRecord(**row) for row in load_jsonl_rows(path)]


def done_keys(trace_path: Path) -> set[tuple[str, str, str]]:
    if not trace_path.exists():
        return set()
    keys = set()
    with trace_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((row["dataset"], str(row["question_id"]), row["condition"]))
    return keys


def generate_with_transformers(
    records: list[ConditionRecord],
    model_cfg: ModelConfig,
    batch_size: int,
) -> list[str]:
    check_hf_access(model_cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        torch_dtype=DTYPE_MAP.get(model_cfg.dtype, torch.bfloat16),
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    prompts = [format_chat(tokenizer, model_cfg, record.prompt) for record in records]
    outputs: list[str] = []
    dec = model_cfg.canonical_decoding
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate(transformers)"):
        batch_prompts = prompts[start:start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=dec.temperature > 0,
                temperature=dec.temperature if dec.temperature > 0 else None,
                top_p=dec.top_p,
                max_new_tokens=dec.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        for seq in generated:
            new_tokens = seq[prompt_width:]
            outputs.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def generate_with_vllm(
    records: list[ConditionRecord],
    model_cfg: ModelConfig,
    batch_size: int,
) -> list[str]:
    from vllm import LLM, SamplingParams

    check_hf_access(model_cfg.model_id)

    dtype = "bfloat16" if model_cfg.dtype == "bfloat16" else "float16"
    llm = LLM(
        model=model_cfg.model_id,
        dtype=dtype,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        enforce_eager=True,
        hf_token=True if os.environ.get("HF_TOKEN") else None,
    )
    tokenizer = llm.get_tokenizer()
    prompts = [format_chat(tokenizer, model_cfg, record.prompt) for record in records]
    dec = model_cfg.canonical_decoding
    sampling = SamplingParams(
        temperature=dec.temperature,
        top_p=dec.top_p,
        max_tokens=dec.max_new_tokens,
    )
    outputs: list[str] = []
    for start in tqdm(range(0, len(prompts), batch_size), desc="generate(vllm)"):
        batch_prompts = prompts[start:start + batch_size]
        batch_outputs = llm.generate(batch_prompts, sampling)
        outputs.extend(out.outputs[0].text.strip() for out in batch_outputs)
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return outputs


def save_prompt_end_hidden_states(
    records: list[ConditionRecord],
    model_cfg: ModelConfig,
    output_dir: Path,
    batch_size: int,
) -> list[str]:
    check_hf_access(model_cfg.model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id, token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_id,
        torch_dtype=DTYPE_MAP.get(model_cfg.dtype, torch.bfloat16),
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    prompt_texts = [format_chat(tokenizer, model_cfg, record.prompt) for record in records]
    paths: list[str] = []
    n_layers = model.config.num_hidden_layers

    for start in tqdm(range(0, len(records), batch_size), desc="extract prompt_end"):
        batch_records = records[start:start + batch_size]
        batch_prompts = prompt_texts[start:start + batch_size]
        encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        last_positions = encoded["attention_mask"].sum(dim=1) - 1
        with torch.inference_mode():
            out = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden_states = out.hidden_states
        for b, record in enumerate(batch_records):
            pos = int(last_positions[b].item())
            arrays = {}
            for layer in range(n_layers):
                vec = hidden_states[layer + 1][b, pos].detach().float().cpu().numpy()
                arrays[f"pos_-1_layer_{layer}"] = vec
            filename = f"{record.dataset}_{record.question_id}_{record.condition}_{model_cfg.short_name}.npz"
            path = feature_dir / filename
            np.savez_compressed(path, **arrays)
            paths.append(str(path))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return paths


def build_trace(
    record: ConditionRecord,
    model_cfg: ModelConfig,
    response: str,
    hidden_state_path: str | None,
    seed: int,
) -> TraceRecord:
    parse = parse_response(response, record.answer_type)
    label = assign_behavioral_label(
        normalized_answer=parse.final_answer_norm,
        hint_candidate=record.candidate,
        gold_answer=record.gold_answer,
        gold_aliases=record.gold_aliases,
    )
    sentence_spans = segment_sentences(parse.reasoning_text)
    dec = model_cfg.canonical_decoding
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
        raw_response=response,
        think_text=None,
        final_text=parse.reasoning_text + ("\nFinal answer: " + parse.final_answer_raw if parse.final_answer_raw else ""),
        normalized_final=parse.final_answer_norm,
        label_behavior=label,
        parse_ok=parse.parse_mode != "fail",
        parse_mode=parse.parse_mode,
        truncated=False,
        sentence_spans=[{"start_char": s.start_char, "end_char": s.end_char, "text": s.text} for s in sentence_spans],
        hidden_state_path=hidden_state_path,
        answer_score_path=None,
    )
