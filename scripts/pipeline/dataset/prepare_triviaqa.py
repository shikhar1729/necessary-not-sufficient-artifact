"""Load TriviaQA (rc.nocontext, train) and return short-answer DatasetItems (seed 42)."""

from __future__ import annotations

import random

from datasets import load_dataset

from scripts.pipeline.common.config import DatasetConfig
from src.core.types import DatasetItem


def prepare_triviaqa(cfg: DatasetConfig, seed: int = 42) -> list[DatasetItem]:
    """Load TriviaQA, filter to short-answer items, return DatasetItems."""
    print(f"Loading TriviaQA ({cfg.hf_path}, {cfg.hf_name}, split={cfg.split})...")
    ds = load_dataset(cfg.hf_path, cfg.hf_name, split=cfg.split)

    items = []
    for row in ds:
        question = row["question"].strip()
        if len(question) < cfg.filter.get("min_question_len", 10):
            continue

        aliases = row["answer"]["aliases"]
        if not aliases or not question:
            continue

        # Use longest alias as canonical (most descriptive), capped by max_answer_len
        canonical = max(aliases, key=len)
        if len(canonical) > cfg.max_answer_len:
            sorted_by_len = sorted(aliases, key=len)
            canonical = sorted_by_len[len(sorted_by_len) // 2]
        if len(canonical) > cfg.max_answer_len:
            continue

        normalized_aliases = [a.lower().strip() for a in aliases]

        items.append(DatasetItem(
            dataset="triviaqa",
            question_id=row.get("question_id", f"triviaqa_{len(items)}"),
            question=question,
            gold_answer=canonical,
            gold_aliases=aliases,
            answer_type="span",
            metadata={"normalized_aliases": normalized_aliases},
        ))

    random.seed(seed)
    random.shuffle(items)

    target = cfg.pilot_size or cfg.main_size or len(items)
    items = items[:target]
    print(f"  TriviaQA: {len(items)} items after filtering")
    return items
