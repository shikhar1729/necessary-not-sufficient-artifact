# Pipeline: from datasets to `data/`

These scripts produced the released data, except where noted below. Install `requirements-pipeline.txt` (Python
3.11 or 3.12), set `HF_TOKEN` (gated models and GPQA) and `OPENROUTER_API_KEY` (candidates and judges), and run each
script from the repository root as a module, for example `python -m scripts.pipeline.nested_probe --help`.
Configs are in `data/prompts/`; set `HF_HOME` in the environment to choose the model cache.

Every script starts with a short header: inputs, outputs and the `data/` file they were released as. Most headers
also give the run command; where the paper used several runs or non-default flags, they list the exact commands
(for example `judges/text_judges.py`, `judges/span_label_adjudication.py`, `injection/generate.py`,
`injection/analyze.py`, `injection/sonnet_rescore.py`, `analysis/nonlinear_probes.py`, `analysis/wg_transfer.py`,
the `prior_aware/` and `roleswap/` scripts). `<store>` below is the trace store that `generate_traces.py` writes.

Shared code: `src/core/` (answer matching and the retained slice), `common/` (configs, model loading and chat
formatting, hooks, sentence splitting, trace generation, the OpenRouter judge), `dataset/` (dataset preparation,
wrong candidates, W/G/C conditions) and `prior_aware/batched_generation.py` (batched generation and row helpers).

## Steps

| Step | Script | Released as (or paper table) |
|---|---|---|
| **Main-task data** | | |
| Datasets, candidates, W/G/C prompts, traces, prompt-end states | `generate_traces.py` | `<store>`, not released |
| Retained examples with folds | no single script: `src/core/slice.py` plus `GroupKFold(5)`; `nested_probe.py` rebuilds the same rows, order and folds | `data/examples/{model}.csv` |
| Re-extracted prompt-end states (float32, all layers) | `extract_prompt_end_states.py` | not released (4.2 GB) |
| No-hint traces | `no_hint_traces.py` | not released |
| Follow-up turns | `followups.py` | not released |
| W-trace funnel | `analysis/w_trace_funnel.py` | `data/examples/w_trace_funnel.csv` |
| **Probes and baselines** | | |
| Headline nested probe; shuffled-label control | `nested_probe.py` | `probes/nested_probe_{oof,layers}.csv`; `controls/shuffled_label_baseline.csv`; `hidden_probe`, `sanity` rows of `strict_rule_summary.csv` |
| Per-dataset probe, TF-IDF, control-state and paired-difference probes, strict no-hint diagnostic | `main_task.py` | `public_text_discriminative`, `control_state_probe`, `paired_difference_or_residual` rows of `controls/strict_rule_summary.csv` |
| TF-IDF out-of-fold scores | `analysis/tfidf_oof.py` | `monitors/tfidf_oof.csv` |
| Non-internal baselines | `analysis/noninternal_baselines.py` | `combined_noninternal` rows of `strict_rule_summary.csv` |
| Added value of the W state over public features | `analysis/incremental_hidden_signal.py` | no released file (app:corrected) |
| No-hint diagnostic, lenient-rule slice | `analysis/no_hint_diagnostic_lenient.py` | no released file (tab:no-hint-diagnostic) |
| Difference-probe layer search | `analysis/difference_probe_layer_search.py` | `controls/difference_probe_{summary,oof,layer_curves}.csv` |
| Difference-probe label-permutation null | `analysis/difference_probe_layer_null.py` | `controls/difference_probe_null_{summary,curves}.csv` |
| Positive control (needs scikit-learn 1.7.2) | `analysis/positive_control.py` | `controls/positive_control.csv` |
| Positive-control floor variants | `analysis/positive_control_floor_variants.py` | `controls/positive_control_floor_variants.csv` |
| Nonlinear probes and their nulls | `analysis/nonlinear_probes.py` | `controls/nonlinear_probe_oof.csv.gz`, `nonlinear_probe_permutation_null.csv` |
| Within-dataset and leave-one-dataset-out probes | `analysis/within_dataset.py` | `controls/within_dataset_{probe,macro}.csv`, `lodo_probe.csv` |
| W/G transfer | `analysis/wg_transfer.py` | `controls/wg_transfer_oof.csv.gz` |
| Strict vs lenient-matching counts | `analysis/reconcile_counts.py` | printed (app:corrected) |
| **Text monitors** | | |
| Generic judges (answer; CoT and answer) | `judges/generic_judges.py` | `judge_answer_only`, `judge_cot_answer` rows of `strict_rule_summary.csv` |
| Claude 3.5 Haiku judges, lenient-rule slice | no script | `monitors/haiku35_judge_rows.csv` (Haiku rows of tab:monitor-intervals) |
| CoT + follow-up judge | `judges/followup_judge.py` | `judge_cot_followup` rows of `strict_rule_summary.csv` |
| Debate judge | `judges/debate_judge.py` | `debate_judge` rows of `strict_rule_summary.csv` |
| Counterfactual, candidate-prior and note-aware judges | `judges/text_judges.py` | `monitors/{prior_and_answer_cf,note_aware}_judge_rows.csv.gz` |
| Cross-family judges, paraphrases, few-shot | `judges/judge_robustness.py` | `monitors/judge_robustness_rows.csv.gz` |
| Monitor intervals and paired differences | `analysis/monitor_intervals.py` | no released file (tab:monitor-intervals, tab:monitor-paired) |
| Judge calibration | `analysis/judge_calibration.py` | no released file (tab:judge-robustness) |
| Probe-plus-text ensembles | `analysis/ensemble_probe_text.py` | `ensemble/{per_row_scores,stack_weights,probe_increment_over_text,best_with_vs_without_probe}.csv` |
| Size-matched ensemble control | `analysis/ensemble_probe_vs_text_addition.py` | `ensemble/probe_vs_text_addition.csv` |
| **Span-answer labels** | | |
| Span-label features | `analysis/span_label_features.py` | `labels/span_label_features.csv` |
| Two-model adjudication | `judges/span_label_adjudication.py` | `labels/adjudication_pool.csv`, `adjudications_{gpt41,gemini25flash}.csv` |
| Parser precision and exact-match relabelling | `analysis/span_label_analysis.py` | `labels/exact_match_slice.csv` |
| **Interventions and boundary conditions** | | |
| Powered activation patching | `patching/powered_patching.py` | `patching/powered_patching_rows.csv.gz` |
| Detectable-effect bounds | `analysis/detectable_effect_bounds.py` | printed (app:patching, app:prioraware) |
| Injection factorial: traces and states | `injection/generate.py` | `injection/run_settings.json` |
| Injection factorial: analysis and Haiku 4.5 judge | `injection/analyze.py` | `injection/{funnel,marker_rates,crosscell_summary,probe_oof,haiku45_judge_scores}.csv` |
| Injection factorial: Sonnet 4.5 re-score | `injection/sonnet_rescore.py` | `injection/sonnet45_judge_scores.csv` |
| Prior-aware v2 rebuild and audit | `prior_aware/build_v2.py`, `prior_aware/analyze_v2.py` | no released file (tab:prior-aware) |
| Matched-prior v3 and adaptive-v4 prescreen | `prior_aware/prescreen.py` | no released file (tab:prior-aware, tab:adaptive-v4) |
| Adaptive-v4 slices, states and probes | `prior_aware/{select_slices,extract_states,probe_baselines}.py` | no released file (tab:adaptive-v4) |
| Larger and reasoning models | `larger_models/generate.py`, `larger_models/analyze.py` | no released file (tab:larger-models-full) |
| **Role-swapped synthetic benchmark** | | |
| Prompts, letter scores, states and probes | `roleswap/generate_benchmark.py` | `roleswap/effect_rows.csv.gz` |
| Interval for each binary AUROC | `roleswap/bootstrap_cis.py` | no released file (tab:roleswap, tab:roleswap-main) |
| Letter-cell holdout | `roleswap/letter_cell_holdout.py` | no released file (tab:roleswap-holdout) |
| Logit lens and dense layer sweep | `roleswap/logit_lens_dense_sweep.py` | `roleswap/logit_lens_summary.csv`, `dense_probe_oof.csv.gz` |
| Natural-QA role swap | `roleswap/natural_qa.py` | no released file (tab:roleswap-natural) |
| Cross-question donor patch control | `roleswap/crossbase_patch_control.py` | no released file (donor rows of tab:roleswap-patch-controls) |

Paths in the last column are under `data/scores/` unless they start with `data/`. No script writes
`controls/strict_rule_summary.csv`: it is a hand-assembled table of the outputs named above. Its attention-probe
and strict-slice layer-sweep rows have no script here.

## Not included

- Hidden states and raw generations, because of their size (the generations also contain GPQA text).
- Code for these results:
  - the lenient-rule diagnostics: tab:per-dataset-auroc, tab:text-discriminative, tab:control-state,
    tab:selected-layers, tab:lodo-transfer, fig:helpful-harmful-transfer, and the lenient-rule probe
    (0.65/0.70/0.73), generic-judge and design-scalar numbers in app:corrected (tab:no-hint-diagnostic has a script);
  - the Claude 3.5 Haiku judge run behind the Haiku rows of tab:monitor-intervals (its rows are released);
  - the attention probe and the strict-slice layer sweep (tab:strict-neutral);
  - the h3 and candidate-swap controls (tab:h3, tab:h3-probes, tab:candidate-swap, tab:candidate-swap-diag);
  - steering (tab:matched-steering) and the small-sample patching families of app:patching (adaptive-v4,
    direction-only and component patching);
  - repeatability (tab:repeatability), and the trajectory and branching analyses (fig:signal-timing,
    fig:sentence-trajectory, tab:sentence-trajectory, tab:branching);
  - the original scaled injection run (tab:injection-full);
  - role-swap side runs: the nonce variant (tab:roleswap-nonce), the Qwen3-32B row of tab:roleswap, the
    full-state patching variants of tab:roleswap-patch and the subspace rows of tab:roleswap-patch-controls;
  - a few prior-aware side analyses in app:prioraware (the v2 side checks and the dominant-prior v3 run);
  - the route-disclosure pilot (app:pilot-route-disclosure: fig:pilot-route, tab:pilot-route);
  - the code that draws the figures (their numbers are in the tables above or in `data/`).

## Settings, versions and hardware

- Main-task generation is greedy (temperature 0, up to 2048 new tokens, batch size 1, no padding). The injection,
  patching and prior-aware runs batch with left padding, which does not always repeat the unbatched answers.
- Hardware: the October 2026 re-extraction, injection, patching and lens runs used one A100 80GB (Python 3.11.10,
  torch 2.4.1, transformers 4.57.6). The role-swap regeneration used an H100 (torch 2.4.1, transformers 4.57.6,
  scikit-learn 1.9.0), and so did the prior-aware runs (adaptive v4: torch 2.8.0, transformers 4.46.3). The
  hardware of the main-task traces was not recorded.
- scikit-learn 1.7.2 reproduces the paper's `GroupKFold` folds on the main task. The role-swap analyses
  (`roleswap/generate_benchmark.py --analyze`, `roleswap/letter_cell_holdout.py`, `roleswap/logit_lens_dense_sweep.py
  --analyze`) and `injection/analyze.py` ran under scikit-learn 1.9.x and reproduce their folds only there.
- vLLM is optional: `prior_aware/build_v2.py --backend vllm` (the Llama v2 traces used vllm 0.23.0 with
  `VLLM_WORKER_MULTIPROC_METHOD=spawn`); it is a commented line in `requirements-pipeline.txt`.
- Regenerated data will differ in places. Candidates came from Python `hash()` without a fixed `PYTHONHASHSEED`
  (MCQ picks, and also the MMLU-Pro sample and the GPQA option order) or from a sampling LLM, so `h1` and `h2` in
  `data/examples/` are the record. `generate_traces.py` draws them again on every start, so it refuses to resume.
  Follow-up selection reads the stored `label_behavior` field, so regenerated traces select differently. The
  lenient-rule no-hint diagnostic keeps the lenient-rule matcher by design.
- Larger models: the DeepSeek interval used `--n-bootstrap 500` (seed 19); the Qwen/QwQ pilot settings were not
  recorded. Judges: `anthropic/claude-sonnet-4`, the default of the generic, follow-up and debate judges, is no
  longer served on OpenRouter.
