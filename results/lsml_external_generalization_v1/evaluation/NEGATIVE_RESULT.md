# Negative result - answer-local token L-SML external transfer

**Hypothesis (one sentence):** Answer-local unlabeled token fitting improves official external metrics over matched fixed-weight controls.
**Date / step:** 2026-09-24, Step 433.
**Benchmark / population:** Pinned Hard2Verify 200/200 answers and Socratic 2995/2995 answers on each of two backbones; 6,190 answer/backbone records, 53,970 steps. Dataset revisions: ../SOURCES.json. Source-only calibration and frozen source folds.

## Arms compared

All numbers are percentages from `METRICS.json`; primary metrics differ across columns and must not be averaged.

| Arm | What it is | Hard2 Balanced F1 | Socratic/Qwen3 PRMScore | Socratic/QwQ PRMScore | vs reference |
|---|---|---:|---:|---:|---|
| CT7 reference | Fixed seven-view reference |37.751|58.753|60.166|Incumbent reference|
| Frozen L-SML | Source-fit step fusion |43.670|63.221|64.238|Leading learned variant|
| Local equal | Matched token equal control |42.212|63.247|63.240|Diagnostic control only|
| Local partition equal | Same local grouping, equal group/within weights |42.931|63.020|61.612|Diagnostic control only|
| Local L-SML | Per-answer token fit, step Top10 readout |42.805|61.961|62.153|Fails to add value over ordinary equal on Socratic|

**Win/loss record:** Local L-SML versus matched ordinary equal: point estimates 1W-2L over the three cells; corrected evidence 0 wins, 2 losses, 1 inconclusive. Differences and paired corrected intervals from `CONTRASTS.json`: Hard2 +0.594pp [-2.744,+3.877] (includes zero: yes); Socratic/Qwen3 -1.285pp [-1.894,-0.665] (includes zero: no); Socratic/QwQ -1.087pp [-1.854,-0.313] (includes zero: no). Local L-SML does improve over CT7 in all three cells, so the negative finding concerns incremental learned weighting against its matched controls, not absence of useful bank11 signal.

## Three plausible reasons it failed

1. Unlabeled covariance inside one answer can reflect token style and feature dependence rather than correctness reliability. Numerical/algebra checks passed, but they do not establish the estimator assumptions on answer-local observations.
2. Local fitting uses token observations while evaluation uses Top10 aggregated step scores. Its observation axis and ordering of standardization/fusion/readout differ from the stronger frozen step method; this is not a clean fitting-scope-only ablation.
3. The fixed source threshold meets a different external distribution of answer-normalized risks. Better mean within-answer ranking need not improve the official thresholded metric. Per-answer decision-budget diagnostics expose a limitation but do not prove calibration is the sole cause.

These are hypotheses, not established causal explanations. Fallback is not a plausible dominant explanation: native fits cover 2984/2995 Socratic answers per backbone, and matched-native ordering agrees (`METRICS.json`, `independent_coverage/AUDIT.json`).

## Verdict

- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open

The current local implementation is not the leading candidate; frozen L-SML supplies positive external evidence, while the local implementation has not beaten its matched controls on the primary endpoints.

## What would reopen it
A single source-only, CPU-only revision with matched fit/readout and calibration controls must improve official PRMScore in separated source validation before a fresh external test; these already-inspected benchmarks cannot supply a new untouched-test claim.

**Source files:** `METRICS.json`, `METRICS.csv`, `CONTRASTS.json`, `RED_TEAM.md`, `CALIBRATION_DIAGNOSTICS.json`; report `docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md`. Commands: `scripts/run_external_local_cpu.py --workers 4`, then `scripts/evaluate_external_locked_scores.py --root results/lsml_external_generalization_v1/evaluation --inputs scratch/external_generalization_private/inputs`. Coverage 6190/6190.
