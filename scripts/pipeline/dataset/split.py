"""Deterministic 60/20/20 train/val/test split by SHA-256 of "dataset:question_id".

Recorded in each trace row's `split` field; the paper's analyses do not use it (they use GroupKFold over the
whole retained slice). All conditions of one question land in the same split.
"""

from __future__ import annotations

import hashlib

from src.core.types import ConditionRecord


def _hash_split(key: str, train_frac: float = 0.6, val_frac: float = 0.2) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000
    if h < train_frac * 1000:
        return "train"
    elif h < (train_frac + val_frac) * 1000:
        return "val"
    else:
        return "test"


def assign_splits(
    records: list[ConditionRecord],
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> list[ConditionRecord]:
    """Set record.split from (dataset, question_id)."""
    split_counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        split = _hash_split(f"{record.dataset}:{record.question_id}", train_frac, val_frac)
        record.split = split
        split_counts[split] += 1
    print(f"  Splits: {split_counts}")
    return records
