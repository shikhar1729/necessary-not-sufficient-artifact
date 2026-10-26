# Necessary but Not Sufficient: Auditing Same-Answer Counterfactuals for Chain-of-Thought Faithfulness

NeurIPS 2026 workshops: Interpreting Agent Behavior (IAB) and Trustworthy AI for Good (AI4GOOD).

We test whether linear probes on a model's hidden states can tell when a hint in the prompt caused its answer.
The probes reach AUROC 0.61-0.65, about as well as LLM judges that see the hint, but our controls do not show
that they read why the model answered.

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

| Folder | Contents |
|---|---|
| `data/examples/` | `{model}.csv`: the retained examples (N = 421 / 384 / 260 for Mistral / Gemma / Llama). `w_trace_funnel.csv`: the outcomes of all 1,728 W traces per model. |
| `data/scores/` | Per-example scores and summary tables, one subfolder per result. |
| `data/prompts/` | Every prompt template and the YAML configs (model, dataset, judge). |

**Label.** A W trace is kept when W and G both parse, G gives an answer, and the W answer matches `h1` but not
the gold answer. `label = 1` (dependent) if the G answer does not match `h1`, else 0 (`src/core/slice.py`).

**Columns.**
- Keys: `model` plus `group` (`dataset::question_id`, the unit of every split and bootstrap) or `dataset` and
  `question_id`; `roleswap/` uses `base_id` and `orientation_id`.
- `examples/`: the candidates `h1` and `h2`, the normalised answers `final_answer_*`, `label` and the outer
  `GroupKFold(5)` `fold`.
- Probe files: out-of-fold scores.
- `monitors/` judge files: the judge's probability of dependence (0-1), verdict, reason and `judge_model`.
  Exception: `haiku35_judge_rows.csv` uses a 0-100 scale and the lenient-rule labels (`label_lenient`).
  `injection/` judge files hold only the probability.
- Summary tables: most of `controls/`, `ensemble/` and `injection/`, because their inputs are not released.
- GPQA question and option text is withheld: those cells read `[GPQA text withheld]`.

## Regenerating from scratch

[`scripts/pipeline/README.md`](scripts/pipeline/README.md) lists every step with its script and released file,
the hardware and versions, and the results that have no script here. The scripts need `requirements-pipeline.txt`
(Python 3.11 or 3.12), GPUs, `HF_TOKEN` and `OPENROUTER_API_KEY`. Hidden states and raw generations are not
included because of their size.

## Citation

```bibtex
@inproceedings{shiromani2026necessary,
  title     = {Necessary but Not Sufficient: Auditing Same-Answer Counterfactuals for Chain-of-Thought Faithfulness},
  author    = {Shiromani, Shikhar and Richter, Leo},
  booktitle = {NeurIPS 2026 Workshops (IAB, AI4GOOD)},
  year      = {2026}
}
```
