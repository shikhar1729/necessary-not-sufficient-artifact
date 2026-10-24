"""Load GPQA-Diamond (gated; needs HF_TOKEN) and return multiple-choice DatasetItems with shuffled options.

The option shuffle seeds random with hash(question) + 42; str hashing is salted per process, so the
realised option order is not re-derivable. data/examples/ records the letters used for the retained rows.
"""

from __future__ import annotations

import random

from datasets import load_dataset

from scripts.pipeline.common.config import DatasetConfig
from src.core.types import DatasetItem


def _format_options(options: list[str]) -> str:
    """Format MCQ options as 'A) option1\\nB) option2\\n...'"""
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))


def prepare_gpqa(cfg: DatasetConfig, seed: int = 42) -> list[DatasetItem]:
    """Load GPQA Diamond, extract MCQ options."""
    print(f"Loading GPQA ({cfg.hf_path}, {cfg.hf_name}, split={cfg.split})...")
    ds = load_dataset(cfg.hf_path, cfg.hf_name, split=cfg.split)

    items = []
    for row in ds:
        question = row.get("Question", "").strip()
        if len(question) < cfg.filter.get("min_question_len", 10):
            continue

        correct = row.get("Correct Answer", "").strip()
        if not correct:
            continue

        # Collect all options
        options = []
        for key in ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"]:
            val = row.get(key, "").strip()
            if val:
                options.append(val)

        if len(options) < 2:
            continue

        # Shuffle options and find correct letter
        random.seed(hash(question) + seed)
        random.shuffle(options)
        correct_idx = options.index(correct)
        correct_letter = "ABCDEFGHIJ"[correct_idx]

        items.append(DatasetItem(
            dataset="gpqa",
            question_id=f"gpqa_{len(items)}",
            question=question,
            gold_answer=correct_letter,
            gold_aliases=[correct_letter, correct],
            answer_type="mcq",
            options=options,
            options_formatted=_format_options(options),
            metadata={"correct_text": correct, "correct_idx": correct_idx},
        ))

    random.seed(seed)
    random.shuffle(items)

    target = cfg.pilot_size or cfg.main_size or len(items)
    if target:
        items = items[:target]
    print(f"  GPQA: {len(items)} items after filtering")
    return items
