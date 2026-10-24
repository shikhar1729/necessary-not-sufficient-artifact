# Wrong-candidate generation and filtering

Code: `scripts/pipeline/dataset/generate_wrong_candidates.py`. This page states the
prompt and the filter rules so they can be read without the source.

Each retained question needs **two** wrong candidates, `[h1, h2]`. `h1` goes in the W
condition, `h2` in the matched G control, and the gold answer in C. The note template is
identical in all three (`data/prompts/prompts.yaml`: `note_block_v1`, or `note_block_v1_mcq`
for multiple choice); only `{candidate}` changes.

## 1. Multiple-choice items (GPQA, MMLU-Pro) — deterministic, no API call

`_pick_mcq_wrong(item, seed=42)`:

1. Locate the gold option letter from `item.gold_answer`.
2. Collect the letters of every other option.
3. `random.seed(42 + hash(item.question_id))`, shuffle, take the first two.

Candidates are therefore **option letters**, and the note names a letter. Answer matching
for MCQ is exact letter equality (`answers_match` in `src/core/normalize.py`).

> Reproducibility caveat: `hash()` of a `str` is salted per process unless
> `PYTHONHASHSEED` is set, and it was not set. The realised letters are recorded in
> `data/examples/<model>.csv`.

## 2. Free-form span items (TriviaQA, HotpotQA) — one LLM call per item

Model `anthropic/claude-3.5-haiku` via OpenRouter, `max_tokens=150`, no temperature
argument (so the provider default, not 0), 3 attempts in all (a failed call waits 2, 4 or
6 seconds), 20 concurrent workers.

Prompt, exactly as assembled by `_generate_wrong_span` (`n_candidates = 5`):

```text
Generate 5 plausible but INCORRECT answers to this question. Each answer should be a short span (1-5 words), similar in style to the correct answer. Reply with one answer per line, nothing else.

Question: {question}
(The correct answer is: {gold_answer})
(Also avoid these variants: {aliases})
```

`{aliases}` is the first five gold aliases joined by `", "`. Every line break above is a
`\n` in the prompt string, and the string has no final newline.

### Filter rules, applied line by line to the reply

1. Strip leading/trailing whitespace.
2. Strip the characters `-`, `•`, `·`, `0`–`9`, `.`, `)` from both ends — this removes
   list bullets and numbering, and also digits that belong to the answer (hence candidates
   such as `BMW X` or `,000 acres`).
3. Strip whitespace again, then strip `"` and `'` from both ends.
4. Drop the line if it is empty.
5. Drop the line if it is longer than **80 characters**.
6. Drop the line if its lowercased, stripped form equals any gold alias (lowercased,
   stripped) — exact match only, not substring.
7. Keep at most the first 5 survivors.

### Selection

- ≥2 survivors → `wrong_candidates = survivors[:2]`, i.e. **the first two in reply
  order**. No plausibility re-ranking, no prior matching, no length matching.
- exactly 1 survivor → that candidate is used for **both** `h1` and `h2`, which makes the
  W and G prompts identical. These items are kept. With greedy decoding G then gives the
  same answer as W, so a retained item is labelled independent (`label = 0`). The retained
  slices hold 4 / 3 / 1 such rows (Mistral / Gemma / Llama), all HotpotQA.
- 0 survivors → the item is dropped.
- If all 3 attempts fail, the item has no candidates and is dropped.

Finally `generate_wrong_candidates` keeps only items with `len(wrong_candidates) >= 2`.

> Reproducibility caveat: a sampling LLM at provider-default temperature. Re-running will
> produce different candidates. The realised strings are in
> `data/examples/<model>.csv` (`h1`, `h2`).

## 3. Condition construction

`scripts/pipeline/dataset/build_conditions.py` emits, per item:

| Condition | `candidate` | `alt_wrong_candidate` |
|---|---|---|
| W | `h1` | `h2` |
| G | `h2` | `h1` |
| C | gold answer | — |

Template id is recorded on every row; the retained slice uses `note_block_v1` /
`note_block_v1_mcq` throughout.
