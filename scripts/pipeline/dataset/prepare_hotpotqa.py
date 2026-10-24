"""Load HotpotQA (distractor, validation), drop yes/no questions, return span DatasetItems (seed 42)."""

from __future__ import annotations

import random

from datasets import load_dataset

from scripts.pipeline.common.config import DatasetConfig
from src.core.types import DatasetItem


def prepare_hotpotqa(cfg: DatasetConfig, seed: int = 42) -> list[DatasetItem]:
    """Load HotpotQA distractor/dev split, exclude yes/no, keep span-only."""
    print(f"Loading HotpotQA ({cfg.hf_path}, {cfg.hf_name}, split={cfg.split})...")
    ds = load_dataset(cfg.hf_path, cfg.hf_name, split=cfg.split)

    items = []
    for row in ds:
        question = row["question"].strip()
        if len(question) < cfg.filter.get("min_question_len", 10):
            continue

        answer = row["answer"].strip()
        if not answer:
            continue

        # Exclude yes/no questions
        if cfg.filter.get("exclude_yes_no", True):
            if answer.lower() in ("yes", "no"):
                continue

        if len(answer) > cfg.max_answer_len:
            continue

        items.append(DatasetItem(
            dataset="hotpotqa",
            question_id=row.get("id", f"hotpotqa_{len(items)}"),
            question=question,
            gold_answer=answer,
            gold_aliases=[answer],
            answer_type="span",
            metadata={"type": row.get("type", ""), "level": row.get("level", "")},
        ))

    random.seed(seed)
    random.shuffle(items)

    target = cfg.pilot_size or cfg.main_size or len(items)
    items = items[:target]
    print(f"  HotpotQA: {len(items)} items after filtering")
    return items
