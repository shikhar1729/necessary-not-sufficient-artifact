"""Load MMLU-Pro (test) and return multiple-choice DatasetItems, stratified by category.

Each category is shuffled with random.seed(seed + hash(category)). Python salts str hashes per process unless
PYTHONHASHSEED is set, and it was not set for the paper's run, so the sampled questions cannot be re-derived;
data/examples/ records the question ids of the retained rows.
"""

from __future__ import annotations

import random
from collections import defaultdict

from datasets import load_dataset

from scripts.pipeline.common.config import DatasetConfig
from src.core.types import DatasetItem


def _format_options(options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}) {opt}" for i, opt in enumerate(options))


def prepare_mmlu_pro(cfg: DatasetConfig, seed: int = 42) -> list[DatasetItem]:
    """Load MMLU-Pro, stratified sample across domains."""
    print(f"Loading MMLU-Pro ({cfg.hf_path}, split={cfg.split})...")
    ds = load_dataset(cfg.hf_path, split=cfg.split)

    # Group by category
    by_category: dict[str, list] = defaultdict(list)
    for row in ds:
        question = row.get("question", "").strip()
        if len(question) < cfg.filter.get("min_question_len", 10):
            continue

        options = row.get("options", [])
        if not options or len(options) < 2:
            continue

        answer_idx = row.get("answer_index")
        answer_str = row.get("answer", "").strip()

        # Determine correct letter
        if answer_idx is not None and 0 <= answer_idx < len(options):
            correct_letter = "ABCDEFGHIJ"[answer_idx]
            correct_text = options[answer_idx]
        elif answer_str and len(answer_str) == 1 and answer_str.upper() in "ABCDEFGHIJ":
            correct_letter = answer_str.upper()
            idx = ord(correct_letter) - ord("A")
            correct_text = options[idx] if idx < len(options) else answer_str
        else:
            continue

        category = row.get("category", "unknown")
        by_category[category].append(DatasetItem(
            dataset="mmlu_pro",
            question_id=row.get("question_id", f"mmlu_pro_{row.get('id', len(by_category[category]))}"),
            question=question,
            gold_answer=correct_letter,
            gold_aliases=[correct_letter, correct_text],
            answer_type="mcq",
            options=options,
            options_formatted=_format_options(options),
            metadata={"category": category, "correct_text": correct_text},
        ))

    # Stratified sampling
    target = cfg.pilot_size or cfg.main_size
    if target and len(by_category) > 0:
        per_domain = target // len(by_category)
        items = []
        for cat, cat_items in sorted(by_category.items()):
            random.seed(seed + hash(cat))
            random.shuffle(cat_items)
            items.extend(cat_items[:per_domain])
    else:
        items = [item for cat_items in by_category.values() for item in cat_items]

    random.seed(seed)
    random.shuffle(items)
    print(f"  MMLU-Pro: {len(items)} items across {len(by_category)} categories")
    return items
