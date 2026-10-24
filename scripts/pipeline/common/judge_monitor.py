"""OpenRouter text judge used by judges/generic_judges.py, judges/followup_judge.py and injection/analyze.py
(--run-judge). Input: judge config and records with question, reasoning and answer. Output: one probability per
record on the judge's 0-100 scale (the callers divide by 100 for AUROC). Needs OPENROUTER_API_KEY; stops if unset.
"""

from __future__ import annotations

import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

from scripts.pipeline.common.config import JudgeConfig, JudgeViewConfig
from src.core.types import JudgeResult


def _get_client(cfg: JudgeConfig) -> OpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    return OpenAI(api_key=key, base_url=cfg.base_url)


def _parse_judge_response(text: str) -> tuple[float, bool, str]:
    """Parse structured judge output: PROBABILITY, VERDICT, REASON."""
    prob = 50.0
    verdict = False
    reason = ""

    m = re.search(r"PROBABILITY\s*:\s*(\d+)", text, re.IGNORECASE)
    if m:
        prob = float(m.group(1))
        prob = max(0.0, min(100.0, prob))

    m = re.search(r"VERDICT\s*:\s*(YES|NO)", text, re.IGNORECASE)
    if m:
        verdict = m.group(1).upper() == "YES"

    m = re.search(r"REASON\s*:\s*(.+)", text, re.IGNORECASE)
    if m:
        reason = m.group(1).strip()

    return prob, verdict, reason


def judge_single(
    client: OpenAI,
    cfg: JudgeConfig,
    view: JudgeViewConfig,
    format_kwargs: dict,
    question_id: str,
) -> JudgeResult | None:
    """Judge a single example with retry."""
    prompt = view.prompt.format(**format_kwargs)

    for attempt in range(cfg.max_retries):
        try:
            r = client.chat.completions.create(
                model=cfg.model_id,
                max_tokens=cfg.max_tokens,
                temperature=cfg.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = r.choices[0].message.content.strip()
            prob, verdict, reason = _parse_judge_response(text)
            return JudgeResult(
                question_id=question_id,
                input_view=view.id,
                probability=prob,
                binary=verdict,
                reason=reason,
            )
        except Exception:
            delay = cfg.retry_base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)

    return None


def judge_batch(
    cfg: JudgeConfig,
    view: JudgeViewConfig,
    examples: list[dict],
    question_id_key: str = "question_id",
) -> list[JudgeResult]:
    """Judge a batch of examples concurrently.

    Each example dict must contain keys matching the view's prompt placeholders.
    """
    client = _get_client(cfg)
    results = []

    def _task(ex: dict) -> JudgeResult | None:
        return judge_single(client, cfg, view, ex, ex[question_id_key])

    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = [executor.submit(_task, ex) for ex in examples]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Judging ({view.id})"):
            result = f.result()
            if result:
                results.append(result)

    print(f"  [{view.id}] {len(results)}/{len(examples)} judged successfully")
    return results
