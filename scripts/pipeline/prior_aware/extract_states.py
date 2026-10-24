"""Prompt-end states for the selected adaptive-v4 rows (tab:adaptive-v4), without regenerating text: one batched
forward pass over the selected W/G/N prompts (GPU, HF_TOKEN), then the prescreen traces are copied with
hidden_state_path filled in. Input: prescreen.py traces and select_slices.py conditions. Output: features/, traces/.

    python -m scripts.pipeline.prior_aware.extract_states --source-trace-dir <v4> --selected-dir <v4_selected> \
        --output-dir <v4_selected>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.pipeline.common.config import load_project_config
from scripts.pipeline.prior_aware.batched_generation import (
    load_condition_records,
    load_jsonl_rows,
    save_prompt_end_hidden_states,
    write_jsonl,
)

MODEL_KEYS = {"mistral": "mistral_7b", "gemma": "gemma_9b", "llama": "llama_8b"}


def trace_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (row["dataset"], str(row["question_id"]), row["condition"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--source-trace-dir", type=Path, required=True, help="prescreen.py output (traces/)")
    parser.add_argument("--selected-dir", type=Path, required=True, help="select_slices.py output (conditions/)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["mistral", "gemma", "llama"], choices=["mistral", "gemma", "llama"])
    parser.add_argument("--extract-batch-size", type=int, default=32)
    args = parser.parse_args()

    cfg = load_project_config(Path(args.config))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        model_cfg = cfg.models[MODEL_KEYS.get(model, model)]
        records = load_condition_records(args.selected_dir / "conditions" / f"{model}_conditions.jsonl")
        traces = load_jsonl_rows(args.source_trace_dir / "traces" / f"{model}_traces.jsonl")
        traces_by_key = {trace_key(row): row for row in traces}
        records = [record for record in records if (record.dataset, str(record.question_id), record.condition) in traces_by_key]
        print(f"[{model}] extracting hidden states for {len(records)} selected traces", flush=True)
        if not records:
            continue
        hs_paths = save_prompt_end_hidden_states(
            records,
            model_cfg,
            args.output_dir,
            args.extract_batch_size,
        )
        selected_traces = []
        for record, hs_path in zip(records, hs_paths):
            row = dict(traces_by_key[(record.dataset, str(record.question_id), record.condition)])
            row["hidden_state_path"] = hs_path
            selected_traces.append(row)
        write_jsonl(args.output_dir / "traces" / f"{model}_traces.jsonl", selected_traces)
        print(f"[{model}] wrote selected traces -> {args.output_dir / 'traces' / f'{model}_traces.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
