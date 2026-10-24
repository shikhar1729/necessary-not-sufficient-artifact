"""Build the W(h1), G(h2) and C(gold) condition records from DatasetItems.

All three use the same note template (data/prompts/prompts.yaml: note_block_v1, or note_block_v1_mcq for multiple
choice); only the candidate changes.
"""

from __future__ import annotations

from scripts.pipeline.common.config import PromptTemplate
from src.core.types import ConditionRecord, DatasetItem


def build_conditions(
    items: list[DatasetItem],
    template: PromptTemplate,
    mcq_template: PromptTemplate | None = None,
) -> list[ConditionRecord]:
    """Build condition records for all items.

    For each item with wrong_candidates [h1, h2]:
      W(h1): candidate = h1 (wrong hint)
      G(h2): candidate = h2 (ghost/control wrong)
      C(y*): candidate = gold_answer (correct hint)
    """
    records = []

    for item in items:
        if len(item.wrong_candidates) < 2:
            continue

        h1, h2 = item.wrong_candidates[0], item.wrong_candidates[1]
        tmpl = mcq_template if item.answer_type == "mcq" and mcq_template else template

        # Common kwargs for template formatting
        fmt_kwargs = {
            "question": item.question,
            "options_formatted": item.options_formatted,
        }

        for condition, candidate, alt in [
            ("W", h1, h2),
            ("G", h2, h1),
            ("C", item.gold_answer, None),
        ]:
            prompt = tmpl.template.format(candidate=candidate, **fmt_kwargs)
            records.append(ConditionRecord(
                dataset=item.dataset,
                question_id=item.question_id,
                question=item.question,
                gold_answer=item.gold_answer,
                gold_aliases=item.gold_aliases,
                answer_type=item.answer_type,
                condition=condition,
                candidate=candidate,
                alt_wrong_candidate=alt,
                prompt=prompt,
                template_id=tmpl.id,
                options=item.options,
                options_formatted=item.options_formatted,
                metadata=item.metadata,
            ))

    print(f"  Built {len(records)} condition records from {len(items)} items")
    return records
