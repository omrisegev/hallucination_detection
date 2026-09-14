# Tail15 localization-aware operating point v1

Status before execution: **PROTOCOL FROZEN / NOT YET RUN**

Frozen on: 2026-09-14

## Question

With the signal and readout fixed to `tail15_mass__token_top10`, does selecting
one uniform gate threshold for the actual end-to-end localization objective
recover the improvement that answer-level macro-F1 selection missed?

## Frozen components

- Locator: `selected_q15_raw_per_view_top10`, unchanged.
- Gate signal: raw mass outside the saved top-15 probabilities at each token.
- Answer readout: mean of the ten largest token values.
- Score calibration: label-free within-cell mid-rank percentile.
- Threshold grid: one q shared by all eight ProcessBench cells, `.01-.99` in
  increments of `.01`.

No feature, readout, fusion weight, locator, cell-specific threshold, or
benchmark-specific method may change.

## Development selection

Select q on all eight ProcessBench development cells by maximum official
all-eight exact-localization macro-F1. Ties are resolved by higher minimum of
q4/q8 macro-F1, then higher answer-level family-macro F1, then proximity to the
previous math-selected q=.40, then lower q.

This deliberately uses ProcessBench labels for operating-point development.
The selected q is not an unbiased ProcessBench estimate and must later be
confirmed on new data or a new model.

## Immediate integration replay

After q selection, replay and compare:

1. starting method: original static fusion-before-Top10 locator plus entropy
   mean q=.3 gate;
2. locator-only update: selected q15 per-view-Top10 locator plus the same old
   entropy gate;
3. gate-only update: original static locator plus tail15 Top10 at selected q;
4. complete method: selected q15 locator plus tail15 Top10 at selected q;
5. the same complete method at math-selected q=.40;
6. historical q15 locator plus PB-developed tail15 mean q=.3 diagnostic.

Use 10,000 paired whole-source-group bootstrap draws and a family-wise 99%
interval across five registered comparisons. Report the unchanged PRMBench
metrics of the selected locator alongside the ProcessBench result so the final
development specification is presented as a whole.
