# Integrated q15 finalist plus tail15 gate replay v1

Status before execution: **PROTOCOL FROZEN / NOT YET REPLAYED**
Frozen on: 2026-09-14
Population: the frozen 13,769-answer localization development contract.

## Purpose

Test the cumulative algorithm obtained by combining the already frozen q15
locator decisions with the gate-feature decision selected in
`GATE_FEATURE_READOUT_SELECTION_V1.md`.

This is the requested same-benchmark integration checkpoint. It checks whether
the selected components compose positively; it is not independent confirmation
or a new selection step.

## Frozen integrated algorithm

- Locator: q15 `{H0lim, VE0, VE0.75, VE1}`, frozen label-free orientation,
  Top10 separately per view, raw equal step fusion.
- Peak: earliest step argmax.
- Gate detector: whole-answer mean mass missing outside the saved top-15 head,
  exactly `tail15_mass__token_mean` from the completed gate selection.
- Gate threshold: other-fold detector quantile q=.3, using the exact five
  thresholds frozen by the selection experiment.
- Open gate returns the locator peak; closed gate returns clean (`-1`).
- PRMB scores and q=.8 calibration are unchanged because this gate is PB-only.

No feature, readout, support, quantile or threshold is reselected here.

## Fixed decomposition

Evaluate four complete algorithms:

1. original static fusion-before-Top10 + original frozen entropy gate;
2. q15 finalist locator + original frozen entropy gate;
3. original static locator + selected tail15 gate;
4. q15 finalist locator + selected tail15 gate (integrated proposal).

The first two must reproduce their prior metrics exactly. The fourth must
reproduce the gate-selection winner exactly. Four paired whole-source-group PB
contrasts use 10,000 draws and a family-wise 98.75% interval:

- integrated minus q15 finalist + entropy (incremental gate decision);
- integrated minus original static + entropy (cumulative decisions);
- original static + tail15 minus original static + entropy (gate portability);
- integrated minus original static + tail15 (locator contribution under the
  selected gate).

Promotion requires positive point deltas for both the incremental and
cumulative comparisons. Intervals and the development-selection boundary must
be reported; passing the point gate does not imply model-transfer confirmation.
