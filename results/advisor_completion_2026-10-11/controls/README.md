# Advisor v11: corrected-input controls and registered decisions

This folder completes the CPU arithmetic for Claude tasks 1–4, 5א and 10.
Task 5 provenance and task 9 literature assessment are handled by the report stream.
The development population is 6,969 answers / 94,203 steps in 707 authenticated
canonical source groups; pooled PRMScore excludes the 758 synthetic correct
controls. Within-answer AUC has 6,030 eligible answers; top1-on-error has 6,035.
Original 1-based error membership is preserved, including the 100 rows with
out-of-range annotations. No labels select signs or rule thresholds here.

## Reproduction

From the repository root, after the completion package's input acquisition:

```sh
.venv/bin/python results/advisor_completion_2026-10-11/controls/run_controls.py --stage all
.venv/bin/python results/advisor_completion_2026-10-11/controls/validate_controls.py
```

The individual stages are `exports`, `bank_scores`, `prefix`, `telemetry`,
`controls`, `decisions`, and `budgets`. Each records a status with the executed
script hash. Raw telemetry is loaded once into the thin `NUISANCE_INPUTS.npz`;
resume validates telemetry, token spans, metadata and answer order. No model
inference, package installation, GPU or cluster execution occurs.

## Files and conventions

- `ALL_BANK_TOP1_AUC.csv` and `ALL_BANK_PER_ANSWER.npz`: all 14 finite, defined
  exported scorers on each of the ten banks. Native PRM and later-step index are
  separate references. Cached LR is handled by the LR stream. The exported
  `DS_FILTER_EQ` is **DS filtering followed by equal averaging**. The frozen
  exports explicitly contain no `DS_SIGNED`; these are never renamed or conflated.
- `ALL_BANK_Q80_PRMSCORE.csv`: all ten-bank matrices, plus native PRM, with the
  nine type scores and full three-category means. Multi-solutions uses valid F1
  alone. Overall is pooled macro-F1, not an average of nine subtype scores.
- `PREFIX_MASK_ATTRIBUTION.csv`, `PREFIX_MASK_CONTRASTS.csv`,
  `PREFIX_PER_ANSWER.npz`: old, prefix-corrected and prefix-plus-mask1 comparisons,
  position-matched AUC, exclusion of step 0, and weighted contributions of pair
  earliest positions. Changes are input-ablation associations after frozen
  scorer refits, not isolated causal effects of individual feature changes.
- `NUISANCE_CONTROLS.csv`, `PAIRED_AUC_CONTRASTS.csv`,
  `CONTROL_PER_ANSWER.npz`: unrestricted and fixed-position / length / repetition
  controls. Primary matching uses five relative-position bins and three global
  label-free token-length bins within each answer. The supplementary triple
  match also conditions on repetition bins. For each matched comparison,
  unrestricted AUC on exactly the same eligible answers is reported. Bin cuts,
  coverage and estimand limits are in `CONTROL_PROTOCOL.json`.
- `EXACT_DECISION_RULES.csv`, `DECISION_FLAGS.npz`, `DECISION_COUNTS.npz`,
  `PER_ANSWER_COUNTS.csv`: exact registered Q80, raw-survivor R2 and corrected
  entropy gate, with controls, counts, biases and MAE by type. TOP1/2/3 are labelled
  diagnostics. Native PRM rejects reward below 0.5; a reward of exactly 0.5 is valid.
- `EXACT_RULE_DIAGNOSTICS.json`: hash and AST adapter of the pinned canonical
  `rules`, `answer_z` and `top_flags` functions. The functions execute verbatim,
  while unrelated historical Windows imports are excluded. Actual oriented raw
  survivor channels reconstruct the saved global G with max error 8.88e-16, so
  R2 here is a certified rule replay rather than the earlier export-only
  reconstruction. The entropy gate reads the corrected mask1 token entropy.
- `PAIRED_PRMSCORE_CONTRASTS.csv`: pooled PRMScore intervals for the three fixed
  rules and native PRM. `BUDGET_CURVES.csv` and `DIAGNOSTIC_CEILINGS.json` record
  threshold, true-count and perfect-placement diagnostics. All ceilings use
  labels and intervene on different components; their gains are not additive.
- `DATASET_EXAMPLES.json`: three genuine NR and three NCL examples, selected as
  the first examples in immutable bank order rather than by our success.
- `ARITHMETIC_VALIDATION.json`: independent numerical algorithms check 11,200
  sklearn AUC cases and 250 repository official-evaluator F1 cases. All pass
  within 2.3e-16. Tolerant versus exact top1 selection agrees in every exported
  bank/method/answer. This does not replace the final methodological reviewer.

The figures `prefix_mask_ablation`, `nuisance_controls`, and `budget_curves` have
both SVG and PNG copies. Every plot uses the machine-readable tables above.

## Findings and limits

The prefix correction accounts for most of the N16 AUC reduction: −1.586 points
for L-SML and −1.177 for IU-PCR. Mask-only changes are −0.075 and −0.013 points,
with paired source-group intervals spanning zero. The prefix decline remains
after coarse position matching and after excluding step 0; the evidence does
not support an explanation based solely on top1 choosing the first step.

After position-plus-length matching, NR retains a large advantage over PRM
(L-SML +0.1974, CI [0.1589, 0.2364]; IU-PCR +0.1749, [0.1362, 0.2151]). NCL
point estimates are positive, but both intervals include zero. The additional
repetition control retains the NR advantage and leaves NCL uncertain. Overall
matched AUC differences versus PRM and the simple average include zero.

The corrected entropy gate lowers L-SML/IU pooled PRMScore to 0.64870/0.64832
from Q80 0.65381/0.65359, while reducing synthetic-control false flagging from
99.47%/99.34% to 58.18%/59.23%. Native PRM is 0.65457, with 19.00% control false
flagging. R2 changes the tradeoff but does not improve the pooled score.

All results are on previously inspected development data. The 5,000 paired
canonical-source-group bootstrap draws use seed 20261011 and pointwise 95%
percentile intervals; they do not correct historical method selection or
multiple comparisons. Coarse matching changes eligibility and pair weights
and does not remove every nuisance difference. Repetition is also part of the
NR construct, so conditioning on it can remove real task signal.
