"""Shared dataclasses used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Dataset & Conditions ──────────────────────────────────────────────────────

@dataclass
class DatasetItem:
    dataset: str
    question_id: str
    question: str
    gold_answer: str
    gold_aliases: list[str]
    answer_type: Literal["span", "mcq"]
    options: list[str] | None = None          # MCQ option strings (A, B, C, ...)
    options_formatted: str = ""               # Pre-formatted "A) ... B) ..."
    metadata: dict = field(default_factory=dict)
    wrong_candidates: list[str] = field(default_factory=list)  # [h1, h2]


@dataclass
class ConditionRecord:
    """A single (item, condition) ready for generation."""
    dataset: str
    question_id: str
    question: str
    gold_answer: str
    gold_aliases: list[str]
    answer_type: Literal["span", "mcq"]
    condition: Literal["W", "G", "C", "H", "neutral"]
    candidate: str                            # h1 / h2 / y*
    alt_wrong_candidate: str | None = None    # the other wrong candidate
    prompt: str = ""
    template_id: str = "note_block_v1"
    options: list[str] | None = None
    options_formatted: str = ""
    metadata: dict = field(default_factory=dict)
    split: str = ""                           # train / val / test


# ── Parsing ───────────────────────────────────────────────────────────────────

@dataclass
class ParseResult:
    reasoning_text: str
    final_answer_raw: str | None
    final_answer_norm: str | None
    parse_mode: Literal["direct_marker", "extractor_mcq", "extractor_freeform", "fail"]
    answer_marker_start_char: int | None = None
    answer_marker_end_char: int | None = None


# ── Sentence Spans ────────────────────────────────────────────────────────────

@dataclass
class SentenceSpan:
    start_char: int
    end_char: int
    text: str
    token_start: int | None = None
    token_end: int | None = None              # last token index of this sentence


# ── Generation Results ────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    hidden_states: dict[int, dict[int, object]]   # position -> layer -> tensor
    kv_cache: object | None = None
    parse_result: ParseResult | None = None
    sentence_spans: list[SentenceSpan] = field(default_factory=list)


# ── Trace Record (JSONL serializable) ─────────────────────────────────────────

@dataclass
class TraceRecord:
    dataset: str
    question_id: str
    split: str
    model: str
    model_mode: str                           # "instruct" / "thinking"
    template_id: str
    condition: str                            # W / G / C / neutral
    candidate: str
    alt_wrong_candidate: str | None
    gold_answer: str
    seed: int
    decoding_params: dict
    raw_prompt: str
    raw_response: str
    think_text: str | None                    # for thinking models
    final_text: str
    normalized_final: str | None
    label_behavior: str | None                # hint_follow / resist / other_error
    parse_ok: bool
    parse_mode: str
    truncated: bool
    sentence_spans: list[dict]                # serialized SentenceSpan dicts
    hidden_state_path: str | None
    answer_score_path: str | None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ── Judge Results ─────────────────────────────────────────────────────────────

@dataclass
class JudgeResult:
    question_id: str
    input_view: str                           # answer_only / cot_answer / cot_followup_answer
    probability: float                        # 0-100
    binary: bool
    reason: str
