"""Tiered answer parser and normalizer.

Parsing priority:
  1. direct_marker — look for "Final answer:" line
  2. extractor_mcq — for MCQ, extract single letter
  3. extractor_freeform — regex fallback for free-form
  4. fail — no answer found
"""

from __future__ import annotations

import re
import unicodedata

from .types import ParseResult


# ── Normalization ─────────────────────────────────────────────────────────────

# Common articles/filler to strip for fuzzy matching
_ARTICLES = {"a", "an", "the", "of"}


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, remove articles."""
    text = text.lower().strip()
    # Unicode normalize
    text = unicodedata.normalize("NFKD", text)
    # Strip leading/trailing quotes
    text = text.strip("\"'`")
    # Remove punctuation (keep alphanumeric and spaces)
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove articles
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens) if tokens else text


def _matches_mcq_letter(answer_norm: str, target_norm: str) -> bool:
    """Exact matching for option-letter targets.

    Fuzzy containment is useful for span answers, but it is unsafe for MCQ
    letters: target "D" should not match "cannot be determined".
    """
    if len(target_norm) != 1 or target_norm not in "abcdefghij":
        return False
    return answer_norm == target_norm or answer_norm == f"option {target_norm}"


def answers_match(answer: str, gold: str, aliases: list[str] | None = None) -> bool:
    """Check if answer matches gold or any alias (normalized fuzzy match)."""
    norm_answer = normalize_answer(answer)
    if not norm_answer:
        return False

    targets = [gold]
    if aliases:
        targets.extend(aliases)

    for target in targets:
        norm_target = normalize_answer(target)
        if not norm_target:
            continue
        # MCQ option letters require exact option-letter matching.
        if len(norm_target) == 1 and norm_target in "abcdefghij":
            if _matches_mcq_letter(norm_answer, norm_target):
                return True
            continue
        # Exact match
        if norm_answer == norm_target:
            return True
        # Containment (either direction)
        if norm_answer in norm_target or norm_target in norm_answer:
            return True

    return False


# ── Parsing ───────────────────────────────────────────────────────────────────

_FINAL_ANSWER_RE = re.compile(
    r"^[*\s]*Final\s+answer\s*:\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)

_MCQ_LETTER_RE = re.compile(r"^[*\s]*([A-J])[.)\s]", re.IGNORECASE)

_FREEFORM_PATTERNS = [
    re.compile(r"(?:the|my)\s+answer\s+(?:is|to the question is)[:\s]+(.+?)[\.\n]", re.IGNORECASE),
    re.compile(r"(?:Therefore|So|Thus|Hence),?\s+the\s+answer\s+(?:is|to the question is)[:\s]+(.+?)[\.\n]", re.IGNORECASE),
    re.compile(r"ANSWER\s*:\s*(.+?)$", re.IGNORECASE | re.MULTILINE),
]


def parse_response(text: str, answer_type: str = "span") -> ParseResult:
    """Tiered parser: direct_marker -> extractor_mcq/freeform -> fail.

    Returns the LAST match for direct_marker (models sometimes restate).
    """
    # ── Tier 1: Direct marker ("Final answer: ...") ──────────────────────
    matches = list(_FINAL_ANSWER_RE.finditer(text))
    if matches:
        m = matches[-1]  # use LAST occurrence
        raw = m.group(1).strip().rstrip(".")
        reasoning = text[:m.start()].rstrip()
        return ParseResult(
            reasoning_text=reasoning,
            final_answer_raw=raw,
            final_answer_norm=normalize_answer(raw),
            parse_mode="direct_marker",
            answer_marker_start_char=m.start(),
            answer_marker_end_char=m.end(),
        )

    # ── Tier 2: MCQ extractor ────────────────────────────────────────────
    if answer_type == "mcq":
        # Look for standalone letter at end of text
        lines = text.strip().split("\n")
        for line in reversed(lines):
            line_stripped = line.strip().replace("*", "")
            m = _MCQ_LETTER_RE.match(line_stripped)
            if m:
                letter = m.group(1).upper()
                reasoning = text[:text.rfind(line)].rstrip()
                return ParseResult(
                    reasoning_text=reasoning,
                    final_answer_raw=letter,
                    final_answer_norm=letter.lower(),
                    parse_mode="extractor_mcq",
                )
            # Also check if line is just a single letter
            if len(line_stripped) == 1 and line_stripped.upper() in "ABCDEFGHIJ":
                reasoning = text[:text.rfind(line)].rstrip()
                return ParseResult(
                    reasoning_text=reasoning,
                    final_answer_raw=line_stripped.upper(),
                    final_answer_norm=line_stripped.lower(),
                    parse_mode="extractor_mcq",
                )

    # ── Tier 3: Free-form extractor ──────────────────────────────────────
    for pattern in _FREEFORM_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            m = matches[-1]
            raw = m.group(1).strip().strip("\"'").rstrip(".")
            reasoning = text[:m.start()].rstrip()
            return ParseResult(
                reasoning_text=reasoning,
                final_answer_raw=raw,
                final_answer_norm=normalize_answer(raw),
                parse_mode="extractor_freeform",
            )

    # ── Tier 4: Fail ─────────────────────────────────────────────────────
    return ParseResult(
        reasoning_text=text,
        final_answer_raw=None,
        final_answer_norm=None,
        parse_mode="fail",
    )
