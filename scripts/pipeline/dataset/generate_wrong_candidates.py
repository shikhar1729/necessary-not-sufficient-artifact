"""Attach two wrong candidates [h1, h2] to each DatasetItem.

Span items: anthropic/claude-3.5-haiku via OpenRouter (OPENROUTER_API_KEY) proposes 5, filtered as in
data/prompts/candidate_generation.md. MCQ items: two incorrect option letters. For the retained rows the realised h1/h2 are in
data/examples/ (neither step is re-derivable: sampling LLM, salted str hash).
"""

from __future__ import annotations

import os
import random
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

from src.core.types import DatasetItem


def _get_client() -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set; it is needed for span candidate generation.")
    return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")


def _generate_wrong_span(
    client: OpenAI,
    item: DatasetItem,
    n_candidates: int = 5,
    model: str = "anthropic/claude-3.5-haiku",
    max_retries: int = 3,
) -> list[str]:
    """Generate plausible wrong answers for a free-form question via LLM."""
    aliases_str = ", ".join(item.gold_aliases[:5])
    prompt = (
        f"Generate {n_candidates} plausible but INCORRECT answers to this question. "
        f"Each answer should be a short span (1-5 words), similar in style to the correct answer. "
        f"Reply with one answer per line, nothing else.\n\n"
        f"Question: {item.question}\n"
        f"(The correct answer is: {item.gold_answer})\n"
        f"(Also avoid these variants: {aliases_str})"
    )

    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=model, max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            text = r.choices[0].message.content.strip()
            candidates = []
            norm_aliases = {a.lower().strip() for a in item.gold_aliases}
            for line in text.split("\n"):
                c = line.strip().strip("-•·0123456789.)")
                c = c.strip().strip("\"'")
                if not c or len(c) > 80:
                    continue
                if c.lower().strip() in norm_aliases:
                    continue
                candidates.append(c)
            return candidates[:n_candidates]
        except Exception:
            time.sleep(2 * (attempt + 1))

    return []


def _pick_mcq_wrong(item: DatasetItem, seed: int = 42) -> list[str]:
    """Pick 2 incorrect MCQ options as wrong candidates."""
    if not item.options:
        return []

    correct_idx = ord(item.gold_answer.upper()) - ord("A") if len(item.gold_answer) == 1 else -1
    wrong_options = []
    for i in range(len(item.options)):
        if i != correct_idx:
            letter = "ABCDEFGHIJ"[i]
            wrong_options.append(letter)

    random.seed(seed + hash(item.question_id))
    random.shuffle(wrong_options)
    return wrong_options[:2]


def generate_wrong_candidates(
    items: list[DatasetItem],
    max_workers: int = 20,
    seed: int = 42,
) -> list[DatasetItem]:
    """Add wrong_candidates [h1, h2] to each item.

    For MCQ: deterministic, no API call.
    For span: LLM-generated, concurrent.
    """
    mcq_items = [it for it in items if it.answer_type == "mcq"]
    span_items = [it for it in items if it.answer_type == "span"]

    # MCQ: deterministic
    for it in mcq_items:
        it.wrong_candidates = _pick_mcq_wrong(it, seed)

    # Span: LLM-generated
    if span_items:
        client = _get_client()
        print(f"Generating wrong candidates for {len(span_items)} span items...")

        def _task(item: DatasetItem) -> tuple[DatasetItem, list[str]]:
            candidates = _generate_wrong_span(client, item)
            return item, candidates

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_task, it) for it in span_items]
            for f in tqdm(as_completed(futures), total=len(futures), desc="wrong candidates"):
                item, candidates = f.result()
                if len(candidates) >= 2:
                    item.wrong_candidates = candidates[:2]
                elif len(candidates) == 1:
                    item.wrong_candidates = [candidates[0], candidates[0]]

    # Filter to items with 2 candidates
    valid = [it for it in items if len(it.wrong_candidates) >= 2]
    print(f"  {len(valid)}/{len(items)} items have 2 wrong candidates")
    return valid
