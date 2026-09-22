# Tail15 readout head-to-head v1

Status before execution: **PROTOCOL FROZEN / NOT YET RUN**

Frozen on: 2026-09-14

## Question

Does the newly leading missing-top15-mass gate benefit from the transferable
whole-answer Top10 readout, or from the simpler whole-answer mean that led the
earlier ProcessBench-only screen?

## Frozen comparison

- `tail15_mass__token_top10` with its threshold selected on the 15-cell math
  panel;
- `tail15_mass__token_mean` with its independently selected threshold on that
  same math panel.

Both candidates use the exact same raw signal:
`1 - sum(exp(top15_logprobs))` at every token. Only the whole-answer readout is
different. Each candidate is converted to a label-free within-cell mid-rank
percentile. No feature, readout, or threshold is selected or recalibrated on
ProcessBench.

Both gates are applied to the same frozen
`selected_q15_raw_per_view_top10` locator in all eight ProcessBench cells. The
existing mean-entropy q=.3 gate is retained as an operational reference. The
historical tail15-mean q=.3 row is not a candidate in this head-to-head because
its q was fixed/inspected on ProcessBench rather than selected on math.

## Evaluation

Report math and ProcessBench total-answer clean/error metrics, official
ProcessBench exact-localization macro-F1, and clean/error decision trade-offs.
Use 10,000 paired whole-source-group bootstrap draws and a family-wise 98.333%
interval for Top10-minus-mean and each candidate-minus-operational-baseline.

This remains development evidence. A method may be retained as a candidate but
cannot be called externally confirmed.
