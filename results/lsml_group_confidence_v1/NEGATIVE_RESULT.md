# Negative result — binary latent-group confidence

**Hypothesis (one sentence):** accumulating nonduplicated member likelihood evidence
will improve PRMScore over averaging that evidence and over same-bank means.
**Date / step:** 2026-09-24; dated Codex HISTORY entry, stage `lsml_group_confidence_v1`.
**Benchmark / population:** corrected v3 labels/v2 source folds;13,769 answers and
145,597 steps; PRMScore6,211 noncontrol answers/83,371steps, within-AUC6,030,
PB erroneous-answer localization4,442 over8cells. Five3fit/1cal/1test rotations.

## Arms compared

| arm | what it is | PRMScore | PRMB within-AUC | PB localization macro8 | vs candidate |
|---|---|---:|---:|---:|---|
| reference | binary-fitted family15 spectral L-SML, continuous application |0.641834|0.762283|0.362135|higher on all3 endpoints|
| same-bank raw equal |28 feature means|0.640876|0.761683|0.360701|higher on all3 endpoints|
| same-bank family equal |15 standardized family means|0.643323|0.765854|0.365084|higher on all3 endpoints|
| matched evidence average |same fitted latent model; mean group log evidence|0.632473|0.755771|0.343707|higher PRMScore; lower secondary points|
| candidate |summed member log evidence, continuous continuation|0.631878|0.757599|0.348029|—|

Numbers: `METRICS.csv`. PB is erroneous-answer localization, not PB macro-F1;
this explicit endpoint supersedes the generic negative-result template's heading.

**Win/loss record:** versus family15equal:0wins/9losses across the8PB localization
cells and1PRMB PRMScore cell (`PER_CELL.csv`). PRMScore delta−0.011444,
corrected paired CI[−0.015013,−0.007668], excludes0. Versus evidence-average,
delta−0.000595,corrected CI[−0.002642,+0.001508], includes0. `CONTRASTS.csv`:
20,000 source-question bootstrap draws,707groups,Bonferroni over4primary contrasts.

## Three plausible reasons it failed

1. The named families need not meet the latent-tree conditional-independence
   assumptions. Model off-block covariance relative residual is0.407–0.419.
   Sensitivity/specificity estimates are imperfect against held-out PRMB truth
   (mean absolute discrepancies0.1491/0.0857 across140fold-feature estimates).
   This is diagnostic evidence, not proof of a sole cause.
2. Per-answer q80 binary indicators are proxies for risk, not observed error
   classifiers with a known stable conditional-error model. Their shared ranks,
   varying answer lengths, ties and domain differences can distort interpretation
   of the inferred global latent variable.
3. Applying the binary likelihood to real-valued standardized telemetry is an
   unproved extension. Sum-vs-average alters evidence temperature including the
   intercept; singleton extensions are unbounded. However, binary-only scoring
   also loses and linearization is similarly weak, so extrapolation alone cannot
   be asserted as the explanation.

## Verdict

- [ ] closes the DIRECTION
- [x] closes this IMPLEMENTATION only

Do not promote this fixed-family, q80, EM-based continuous-likelihood variant to
external evaluation. The latent model works on its controlled simulation, and
this result does not close confidence-aware group learning in other settings.

## What would reopen it

A separately declared source experiment with defensible family-error structure
and more faithful reliability estimation that beats both matched equal controls
and its own ablation on PRMScore before external testing.

**Source files:** `METRICS.csv`, `PER_CELL.csv`, `CONTRASTS.csv`, `FITS.json`,
`MODEL_DIAGNOSTIC.json`, `RED_TEAM.md`, sealed `PREDICTIONS.npz` and
`CALIBRATION.npz`; full report
`docs/experiments/LSML_GROUP_CONFIDENCE_RESULTS_20260924_HE.md`.
Commands: `python scripts/run_lsml_group_confidence.py fit`, then `evaluate`;
posthoc diagnostic `python scripts/analyze_lsml_group_confidence.py`.
Checked13,769/13,769 answers; all10arms complete, no fallback or missing predictions.
