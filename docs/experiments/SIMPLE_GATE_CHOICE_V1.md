# Simple answer-gate choice v1

Status before execution: **PROTOCOL FROZEN / NOT YET RUN**
Frozen on: 2026-09-14

## Question

Can the whole-answer gate be reduced to one of two interpretable choices,
without promoting the higher-complexity three-feature development winner?

## Frozen candidates

1. `entropy_native__token_top10`: the mean of the ten largest native cached
   Shannon-H1 token entropies in the complete answer.
2. `q15_raw4_mean__token_top10`: reuse the frozen q15 token-level static
   localization fusion -- the natural-unit equal mean of oriented `H0lim`,
   `VE0`, `VE0.75`, and `VE1` at every token -- and then take one Top10 mean
   over the complete answer.

The second candidate must pass an identity audit against the already frozen
`original_static_fusion_before_top10` localization stream. Specifically, after
applying the registered per-step Top10 readout to the candidate token stream,
the mean of those frozen step scores for every ProcessBench answer must equal
`q15_raw4_mean__mean_step_top10` from the independently frozen detector bank.

This candidate is the exact fusion-before-Top10 static localization definition.
It is not the later selected `q15_raw_per_view_top10` locator: Top10 and equal
fusion do not commute, and the latter has no single fused token stream to reuse.

## Math-panel decision

- Population: all 18,614 answers from the 15 historical GSM8K/MATH500 cells.
- Each raw score is converted to a mid-rank empirical percentile inside its
  cell, without labels.
- One common threshold `q` is selected from `.05, .10, ..., .95` by equal-family
  answer-level clean/error macro-F1.
- Prefer `entropy_native__token_top10` whenever it is not worse on the primary
  metric, because it uses one existing signal and no additional fusion.
- Do not consider or promote the three-feature equal-mean gate in this choice.

## ProcessBench transfer

Freeze the selected method and its math-selected `q` before opening the
ProcessBench target. Transfer it unchanged to all eight ProcessBench cells with
the already frozen q15 per-view-Top10 locator. Within-cell mid-rank conversion
is label-free score calibration; there is no ProcessBench threshold, feature,
fusion, or method selection.

Compare with the existing q15 locator plus mean-entropy q=.3 gate. Report
answer detection, official localization macro-F1, clean/error trade-offs, and a
paired whole-source-group interval. This remains development evidence, not an
external or new-model confirmation.
