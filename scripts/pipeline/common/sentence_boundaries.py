"""Sentence boundaries in chain-of-thought text (pysbd, with a regex fallback)."""

from __future__ import annotations

import re

from src.core.types import SentenceSpan

try:
    import pysbd
    _HAS_PYSBD = True
except ImportError:
    _HAS_PYSBD = False


# ── Fallback regex segmenter ─────────────────────────────────────────────────

_SENTENCE_END_RE = re.compile(r'([.!?])\s+(?=[A-Z])')


def _regex_segment(text: str) -> list[str]:
    """Simple regex sentence splitter as fallback."""
    splits = _SENTENCE_END_RE.split(text)
    sentences = []
    i = 0
    while i < len(splits):
        s = splits[i]
        # Re-attach punctuation
        if i + 1 < len(splits) and len(splits[i + 1]) == 1 and splits[i + 1] in ".!?":
            s += splits[i + 1]
            i += 2
        else:
            i += 1
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences if sentences else [text.strip()]


# ── Main API ──────────────────────────────────────────────────────────────────

def segment_sentences(text: str) -> list[SentenceSpan]:
    """Segment text into sentences, returning character-level spans."""
    if not text.strip():
        return []

    if _HAS_PYSBD:
        seg = pysbd.Segmenter(language="en", clean=False)
        sentences = seg.segment(text)
    else:
        sentences = _regex_segment(text)

    spans = []
    search_start = 0
    for sent in sentences:
        sent_stripped = sent.strip()
        if not sent_stripped:
            continue
        # Find this sentence in the original text
        idx = text.find(sent_stripped, search_start)
        if idx == -1:
            # Fallback: approximate position
            idx = search_start
        start = idx
        end = start + len(sent_stripped)
        spans.append(SentenceSpan(
            start_char=start,
            end_char=end,
            text=sent_stripped,
        ))
        search_start = end

    return spans
