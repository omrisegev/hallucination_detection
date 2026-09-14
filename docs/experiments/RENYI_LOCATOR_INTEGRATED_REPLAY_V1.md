# Renyi locator integrated replay v1

Status before execution: **FROZEN / NOT YET REPLAYED**

## Purpose

Independently reconstruct and evaluate the Experiment-3 minimum-regret
candidate and the current frozen locator from raw source rows. The replay may
not tune a feature, support, transform, readout, weight, gate, or threshold.

## Frozen candidate and reference

- Candidate: `ve1q50__h10__hinf0__scale_step_equal`.
- Current reference: `ve1q15__h10__hinf0__raw_step_equal`.
- Both use the frozen tail15 whole-answer Top10 gate with one q=.33 for every
  ProcessBench cell, and earliest argmax for the localized step.
- PRMBench uses no answer gate; PRMScore q=.8 thresholds are calculated from
  the other frozen source-group folds.

The candidate retains q15 H0lim/VE0/VE0.75, replaces q15 VE1 with q50 VE1,
divides each token view by its within-answer standard deviation without
centering, applies Top10 separately within each official step, and then takes
an equal mean. Native H1, q15 Hinf, local IU and time-varying weights are not
included because Experiment 3 did not support their promotion.

## Promotion rule

The candidate is promoted only if its point ProcessBench all-eight score is no
lower than the current reference and its PRMB-within AUROC is no more than
.002 lower. This uses the margins frozen in Experiment 3. If it fails, the
integrated recommendation is the current q15/raw/per-view-Top10 locator plus
the already frozen tail15 Top10 q=.33 gate.

## Required checks

- Independently reread all 13,769 raw answers and 145,597 official steps.
- Recompute both locator streams without reading Experiment-3 scores.
- Verify the recomputed streams against the score archives only after scoring.
- Freeze and hash the replay streams before aggregate evaluation.
- Report the two complete gate+locator methods, deltas, 10,000 paired
  whole-source-group bootstrap draws, and a machine-readable promotion result.

This remains a development replay, not external confirmation.
