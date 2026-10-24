# Necessary but Not Sufficient: Auditing Same-Answer Counterfactuals for Chain-of-Thought Faithfulness

Shikhar Shiromani; Leo Richter (University College London)

NeurIPS 2026 workshops: Interpreting Agent Behavior (IAB) and Trustworthy AI for Good (AI4GOOD). Paper: [`paper.pdf`](paper.pdf).

A note in the prompt suggests a wrong answer; a matched prompt suggests a different wrong answer. Among traces
that adopt the first suggestion, we label whether the second would have changed the answer. Linear probes on the
prompt-end hidden state predict this label at AUROC 0.61-0.65, about as well as LLM judges once the judges see
the note. Our controls do not show that the probe reads why the model answered: features that never look inside
the model predict the label better, probes do as well on prompts that never contained the hint, and editing the
states does not detectably change the answer, within limits set by sample size. This repository holds the
per-example data and summary tables behind the paper's numbers, a script that recomputes them, and the code that
produced the data.

**Terms.** Each question has three prompts with the same note template: W names a wrong candidate `h1` (the
hint), G names a different wrong candidate `h2` (the matched control), and C names the correct answer.

## Repository layout

```
paper.pdf
data/
  examples/   one row per retained example and model (mistral, gemma, llama), with folds; the W-trace funnel
  scores/     per-example scores and summary tables behind each result, by paper section:
              monitors/ probes/ controls/ ensemble/ labels/ injection/ patching/ roleswap/
  prompts/    task, injection, role-swap and follow-up prompts, candidate generation, every judge prompt, configs
src/core/     answer parsing and matching (normalize.py) and the retained slice and label (slice.py)
scripts/
  reproduce.py  recomputes the paper's numbers from data/
  pipeline/     the code that produced data/; its README.md lists the steps
```

## Reproduce the paper's numbers

```bash
pip install -r requirements.txt     # Python 3.12 or newer
python scripts/reproduce.py         # about one minute on a laptop CPU, no network, from any directory
```

The script first checks that `GroupKFold(5)` rebuilds the stored folds, which needs scikit-learn below 1.9. It
then runs 143 checks comparing 1,105 printed numbers, plus 4 yes/no checks of statements in the text. Each check
names its paper label (for example `tab:monitor-intervals`) and prints the paper's and the recomputed values; a
value passes when it rounds to the printed one. The script exits with status 1 on any failure. Its docstring gives
the bootstrap settings.

## Data

- `data/examples/{model}.csv`: the retained examples (N = 421 / 384 / 260 for Mistral / Gemma / Llama), keyed by
  `group` (`dataset::question_id`, the unit of every split and bootstrap), with the W and G candidates `h1`, `h2`,
  the normalised answers `final_answer_*`, and the outer `GroupKFold(5)` `fold`. No single script writes them: they
  are the retained slice of `src/core/slice.py` plus that fold, which `nested_probe.py` rebuilds row for row.
  `w_trace_funnel.csv` counts the outcomes of all 1,728 W traces per model.
- **Label.** A W trace is kept when W and G both parse, G gives an answer, and the W answer matches `h1` but not
  the gold answer. `label = 1` (dependent) if the G answer does not match `h1`, else 0 (`src/core/slice.py`).
- `data/scores/`: rows are keyed by `model` and `group` (or `dataset` and `question_id`), plus `job_id` in the
  API-judge files, `unit` and `level` in `injection/sonnet45_judge_scores.csv`, and `base_id` and `orientation_id`
  in `roleswap/` (label `binary_label`). Probe files hold out-of-fold scores. `monitors/` judge files hold the
  probability of dependence (0-1), verdict, reason and `judge_model`; token counts are in the API-judge and
  adjudication files, US$ cost in `judge_robustness_rows` and the adjudication files. `haiku35_judge_rows.csv`
  (Claude 3.5 Haiku) uses a 0-100 scale and lenient-slice rows and labels (`label_lenient`), with no verdict or
  reason on its 874 `cot_followup_answer` rows; `reproduce.py` scores its strict-slice rows with the strict label.
  `injection/` judge files hold only the probability. Summary tables cover results whose inputs are not released:
  most of `controls/` (including the hand-assembled `strict_rule_summary.csv`), `ensemble/` and `injection/`.

## Regenerating from scratch

[`scripts/pipeline/README.md`](scripts/pipeline/README.md) lists every step with its script and released file. The
scripts need `requirements-pipeline.txt` (Python 3.11 or 3.12), GPUs, `HF_TOKEN` and `OPENROUTER_API_KEY`. The
October 2026 re-extraction, injection, patching and lens runs used one A100 80GB; the role-swap regeneration and
the prior-aware runs used an H100; the hardware of the main-task traces was not recorded. Models:
Mistral-7B-Instruct-v0.1, Gemma-2-9B-it, Llama-3.1-8B-Instruct, and the larger models in `data/prompts/models.yaml`.

Not included: hidden states and raw generations, because of their size (the generations also contain GPQA
text), and the code for some appendix-only results: the lenient-rule diagnostics, the Claude 3.5 Haiku judge run,
the attention probe and strict-slice layer sweep, the h3, candidate-swap, steering, small-sample patching,
repeatability, trajectory and branching analyses, the original injection run, some role-swap and prior-aware side
runs, the route-disclosure pilot and the figure code. The pipeline README lists each with its paper label.

## Notes

- Judge model names are moving API aliases, not snapshots. `anthropic/claude-sonnet-4` is no longer served on
  OpenRouter (the injection re-score used `anthropic/claude-sonnet-4.5`), and `anthropic/claude-3.5-haiku` is
  retired. Rerunning a judge may not give the same scores. Hugging Face model revisions were not pinned.
- GPQA question and option text is withheld at the GPQA authors' request (GPQA itself is CC BY 4.0). On GPQA
  rows, model reasoning, judge reasons and answers that are not a bare option letter read `[GPQA text withheld]`.
- The strict slices are used throughout; the lenient-rule slices (352 / 326 / 196: stored labels and looser letter
  matching) appear only where the paper says so.
- `data/prompts/` holds every prompt except the prior-aware candidate and hint prompts (`prior_aware/build_v2.py`,
  `prior_aware/prescreen.py`) and the natural-QA role-swap prompt (`roleswap/natural_qa.py`).

## Citation

```bibtex
@inproceedings{shiromani2026necessary,
  title     = {Necessary but Not Sufficient: Auditing Same-Answer Counterfactuals for Chain-of-Thought Faithfulness},
  author    = {Shiromani, Shikhar and Richter, Leo},
  booktitle = {NeurIPS 2026 Workshops (IAB, AI4GOOD)},
  year      = {2026}
}
```

## License

- Code: MIT (`LICENSE`).
- Data: our labels, scores, judge outputs and prompt files are CC BY 4.0 (`data/LICENSE`). Question and answer
  text keeps the licence of its source dataset: TriviaQA (Apache-2.0, per its GitHub repository), HotpotQA
  (CC BY-SA 4.0) and MMLU-Pro (MIT). GPQA (CC BY 4.0) text is not included.
- Model outputs in `data/` come from Mistral-7B-Instruct-v0.1 (Apache-2.0), Gemma-2-9B-it (Gemma Terms of Use)
  and Llama-3.1-8B-Instruct (Llama 3.1 Community License). Judge, candidate and adjudication outputs come from
  Claude Sonnet 4, Claude 3.5 Haiku, Claude Haiku 4.5 and Claude Sonnet 4.5 (Anthropic), GPT-4.1 and GPT-4.1-mini
  (OpenAI) and Gemini 2.5 Flash (Google), accessed through OpenRouter, and are subject to those providers' terms.
