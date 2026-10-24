"""Extract prompt-end hidden states (all layers) for the W/G/C traces without regenerating text. Batch size 1, no
padding, forward hooks on the decoder layers taking hidden[0, -1, :] (raw layer output, no final norm).
Input: --trace-dir with {model}_traces.jsonl (rows keep raw_prompt). Output (--output-dir): features/*.npz
(float32, keys pos_-1_layer_{L}; not released, 4.2 GB) and traces/ with hidden_state_path repointed. Needs a GPU
and HF_TOKEN. Recorded run: one model at a time, --models mistral, then gemma, then llama.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.common.model_loader import ChatFormatter, load_model
from scripts.pipeline.common.hook_manager import HookManager
from src.core.slice import load_jsonl

MODEL_KEYS = {"mistral": "mistral_7b", "gemma": "gemma_9b", "llama": "llama_8b"}


def trace_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["dataset"], str(row["question_id"]), row["condition"])


def feature_filename(key: tuple[str, str, str], short_name: str) -> str:
    dataset, question_id, condition = key
    for part in (dataset, question_id, condition):
        if "/" in part or "\\" in part or ".." in part:
            raise ValueError(f"unsafe id for a file name: {part!r}")
    return f"{dataset}_{question_id}_{condition}_{short_name}.npz"


def extract_model(
    model_name: str,
    cfg,
    trace_dir: Path,
    output_dir: Path,
    conditions: set[str],
    limit: int | None,
    progress_every: int,
    overwrite: bool,
) -> dict[str, Any]:
    model_cfg = cfg.models[MODEL_KEYS.get(model_name, model_name)]
    short = model_cfg.short_name
    rows = load_jsonl(trace_dir / f"{model_name}_traces.jsonl")

    # One NPZ per (dataset, question_id, condition). The natural trace files
    # contain one duplicated question (triviaqa qb_1578), so dedup on the key
    # that determines the output filename and keep the first occurrence.
    first_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    order: list[tuple[str, str, str]] = []
    for row in rows:
        if row["condition"] not in conditions:
            continue
        key = trace_key(row)
        if key in first_by_key:
            continue
        first_by_key[key] = row
        order.append(key)
    if limit is not None:
        order = order[:limit]

    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"progress_{model_name}.json"

    todo = []
    for key in order:
        path = feature_dir / feature_filename(key, short)
        if not overwrite and path.exists() and path.stat().st_size > 0:
            continue
        todo.append(key)
    print(
        f"[{model_name}] {len(order)} target NPZ files, {len(todo)} to extract "
        f"({len(order) - len(todo)} already present)",
        flush=True,
    )

    stats: dict[str, Any] = {
        "model": model_name,
        "model_id": model_cfg.model_id,
        "short_name": short,
        "n_target": len(order),
        "n_extracted": 0,
        "n_layers": None,
        "hidden_dim": None,
        "started": time.time(),
    }
    if not todo:
        stats["elapsed_sec"] = 0.0
        return stats

    model, tokenizer = load_model(model_cfg)
    n_layers = int(model.config.num_hidden_layers)
    layers = list(range(n_layers))
    formatter = ChatFormatter(tokenizer, model_cfg)
    hook_mgr = HookManager(model, layers, model_cfg.layer_accessor)
    stats["n_layers"] = n_layers

    start = time.time()
    try:
        for i, key in enumerate(todo):
            row = first_by_key[key]
            input_ids = formatter.format(row["raw_prompt"])
            with torch.inference_mode():
                with hook_mgr.extract_step():
                    model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=True,
                    )
            collected = hook_mgr.get_and_clear()
            if len(collected) != n_layers:
                raise RuntimeError(
                    f"{key}: collected {len(collected)} layers, expected {n_layers}"
                )
            arrays = {
                f"pos_-1_layer_{layer}": collected[layer].numpy()
                for layer in layers
            }
            if stats["hidden_dim"] is None:
                stats["hidden_dim"] = int(arrays["pos_-1_layer_0"].shape[0])
            np.savez_compressed(feature_dir / feature_filename(key, short), **arrays)
            stats["n_extracted"] += 1

            if (i + 1) % progress_every == 0 or (i + 1) == len(todo):
                elapsed = time.time() - start
                rate = (i + 1) / max(elapsed, 1e-6)
                payload = {
                    "model": model_name,
                    "done": i + 1,
                    "todo": len(todo),
                    "target": len(order),
                    "elapsed_sec": round(elapsed, 1),
                    "rate_per_sec": round(rate, 2),
                    "eta_sec": round((len(todo) - i - 1) / max(rate, 1e-6), 1),
                }
                progress_path.write_text(json.dumps(payload))
                print(f"[{model_name}] {json.dumps(payload)}", flush=True)
    finally:
        hook_mgr.remove_all()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stats["elapsed_sec"] = round(time.time() - start, 1)

    # Trace copy with hidden_state_path repointed at the recomputed features,
    # plus a manifest for downstream verification.
    trace_out = output_dir / "traces" / f"{model_name}_traces.jsonl"
    trace_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out = output_dir / f"manifest_{model_name}.jsonl"
    with trace_out.open("w") as tf, manifest_out.open("w") as mf:
        for row in rows:
            key = trace_key(row)
            name = feature_filename(key, short)
            path = feature_dir / name
            new_row = dict(row)
            new_row["hidden_state_path"] = f"data/features/{name}"
            new_row["answer_score_path"] = None
            tf.write(json.dumps(new_row) + "\n")
            mf.write(
                json.dumps(
                    {
                        "dataset": key[0],
                        "question_id": key[1],
                        "condition": key[2],
                        "npz": name,
                        "exists": path.exists(),
                        "bytes": path.stat().st_size if path.exists() else 0,
                    }
                )
                + "\n"
            )
    print(f"[{model_name}] wrote {trace_out} and {manifest_out}", flush=True)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"])
    parser.add_argument("--conditions", nargs="+", default=["W", "G", "C"])
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test row cap per model.")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_project_config(Path(args.config))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for model_name in args.models:
        all_stats.append(
            extract_model(
                model_name=model_name,
                cfg=cfg,
                trace_dir=args.trace_dir,
                output_dir=args.output_dir,
                conditions=set(args.conditions),
                limit=args.limit,
                progress_every=args.progress_every,
                overwrite=args.overwrite,
            )
        )
        (args.output_dir / "extraction_stats.json").write_text(json.dumps(all_stats, indent=2))
    print(json.dumps(all_stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
