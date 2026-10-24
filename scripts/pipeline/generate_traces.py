"""Build the main-task W/G/C conditions and generate the traces, prompt-end states and commitment scores.

Inputs: data/prompts/*.yaml, the four Hugging Face datasets (HF_TOKEN for GPQA and gated models),
OPENROUTER_API_KEY for span candidates. Outputs (--data-dir): processed/{dataset}_{items,conditions}.jsonl,
traces/{model}_traces.jsonl, features/*.npz and *_scores.json (the trace store every main-task script reads).
Run it to completion: each start draws new candidates, so it stops if traces/ already has rows for a model.
Run: python -m scripts.pipeline.generate_traces --data-dir <store> --models mistral_7b gemma_9b llama_8b --gpus 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Process
from pathlib import Path

from scripts.pipeline.common.config import ModelConfig, load_project_config
from scripts.pipeline.dataset.build_conditions import build_conditions
from scripts.pipeline.dataset.generate_wrong_candidates import generate_wrong_candidates
from scripts.pipeline.dataset.prepare_gpqa import prepare_gpqa
from scripts.pipeline.dataset.prepare_hotpotqa import prepare_hotpotqa
from scripts.pipeline.dataset.prepare_mmlu_pro import prepare_mmlu_pro
from scripts.pipeline.dataset.prepare_triviaqa import prepare_triviaqa
from scripts.pipeline.dataset.split import assign_splits
from src.core.types import ConditionRecord

DATASET_LOADERS = {
    "triviaqa": prepare_triviaqa,
    "hotpotqa": prepare_hotpotqa,
    "gpqa": prepare_gpqa,
    "mmlu_pro": prepare_mmlu_pro,
}


def prepare_all_datasets(project_cfg, dataset_names: list[str], data_dir: Path) -> list[ConditionRecord]:
    """Load each dataset, attach two wrong candidates, build W/G/C conditions and assign splits."""
    all_records = []
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for ds_name in dataset_names:
        print(f"\n{'='*40} {ds_name.upper()} {'='*40}")
        items = DATASET_LOADERS[ds_name](project_cfg.datasets[ds_name])
        items = generate_wrong_candidates(items)

        with open(processed_dir / f"{ds_name}_items.jsonl", "w") as f:
            for item in items:
                f.write(json.dumps({
                    "dataset": item.dataset,
                    "question_id": item.question_id,
                    "question": item.question,
                    "gold_answer": item.gold_answer,
                    "answer_type": item.answer_type,
                    "wrong_candidates": item.wrong_candidates,
                }) + "\n")

        template = project_cfg.prompt_templates["note_block_v1"]
        mcq_template = project_cfg.prompt_templates.get("note_block_v1_mcq")
        records = assign_splits(build_conditions(items, template, mcq_template))
        all_records.extend(records)

        with open(processed_dir / f"{ds_name}_conditions.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps({
                    "dataset": r.dataset, "question_id": r.question_id,
                    "condition": r.condition, "candidate": r.candidate,
                    "gold_answer": r.gold_answer, "split": r.split,
                    "answer_type": r.answer_type,
                }) + "\n")

    print(f"\nTotal: {len(all_records)} condition records across {len(dataset_names)} datasets")
    return all_records


def _has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open() as f:
        return any(line.strip() for line in f)


def _run_model_process(
    cfg: ModelConfig,
    records: list[ConditionRecord],
    output_dir: str,
    gpu_id: int,
) -> None:
    """One process per model; torch is imported only after the GPU is pinned."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    from scripts.pipeline.common.trace_generation import generate_traces_for_model

    generate_traces_for_model(
        model_cfg=cfg,
        records=records,
        layers=[],  # all layers
        output_dir=Path(output_dir),
    )


def run_parallel(
    model_configs: list[tuple[ModelConfig, int]],
    records: list[ConditionRecord],
    output_dir: Path,
) -> None:
    """Launch one process per (model, GPU) pair; every model gets the same records."""
    output_dir.mkdir(parents=True, exist_ok=True)
    processes = []
    for cfg, gpu_id in model_configs:
        p = Process(target=_run_model_process, args=(cfg, records, str(output_dir), gpu_id))
        p.start()
        processes.append((cfg.short_name, p))
        print(f"Started {cfg.short_name} on GPU {gpu_id}")

    for name, p in processes:
        p.join()
        print(f"{name} finished (exit code {p.exitcode})")
    print("All models done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="data/prompts/")
    parser.add_argument("--data-dir", required=True, help="output trace store")
    parser.add_argument("--models", nargs="+", default=["mistral_7b", "gemma_9b", "llama_8b"])
    parser.add_argument("--datasets", nargs="+", default=["triviaqa", "hotpotqa", "gpqa", "mmlu_pro"])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--resume-existing-traces",
        action="store_true",
        help="keep existing traces and generate only the missing ones; candidates are drawn again on every start, "
        "so resumed W and G traces can come from different candidates",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    project_cfg = load_project_config(args.config)

    started = [
        name for name in args.models
        if _has_rows(data_dir / "traces" / f"{project_cfg.models[name].short_name}_traces.jsonl")
    ]
    if started and not args.resume_existing_traces:
        raise SystemExit(
            f"traces already exist for {', '.join(started)} in {data_dir / 'traces'}. Every start draws new wrong "
            "candidates, so a restart can pair W and G traces built from different candidates. Delete traces/ and "
            "run again, or pass --resume-existing-traces to keep them anyway."
        )

    records = prepare_all_datasets(project_cfg, args.datasets, data_dir)
    if not records:
        raise SystemExit("No records to process.")

    print(f"\nGENERATION: {len(args.models)} models x {len(records)} records")
    model_gpu_pairs = [
        (project_cfg.models[name], args.gpus[i % len(args.gpus)]) for i, name in enumerate(args.models)
    ]
    run_parallel(model_gpu_pairs, records, data_dir)


if __name__ == "__main__":
    main()
