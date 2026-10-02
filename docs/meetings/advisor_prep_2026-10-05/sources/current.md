# per_dataset_fit_v1: the method fitted per model per dataset

Omri, 2026-09-29: "why would we learn shared weights? I want the method learned each time per model per dataset - that is how it
will be used: ask a model n questions, get answers, run the method." Approved: build, run, have a subagent check, present.

- Protocol `PROTOCOL.json` frozen at 355db88f5; A1 (42decb9c4, after a two-cell smoke): cross-fit halves along the label-free
  partition; A2 (55e79c4b2, after the independent pre-run review, GO WITH FIXES): completed smoke disclosure, cross-fit failures
  non-fatal, clip reporting, extra checks. Run `run_20260929` (ba0ef880d): COMPLETE, 452 s; the per-cell fit function reproduces
  Steps 457/460/461 on their own rows to 8.9e-16. Red team `run_20260929/RED_TEAM.md`.
- Contract: each of the 9 cells (PRMBench; 8 ProcessBench cells, q4/q8 = two telemetry versions of the same answers) is fitted on
  all of its own answers without labels and scores the same answers: label-free signs, DS filter, partition, weights, position
  prior, slopes, PRMScore threshold. ProcessBench also got a first-error readout from the same latent model,
  P(first error at t) = q_t prod_{s<t}(1 - q_s). Secondary: cross-fitted slopes.

## Results

PRMBench within-answer AUC (per-dataset / pooled fit as before):

| Bank | plain average + position channel | grouped + position prior | plain average + prior, cross-fitted slope |
|---|---|---|---|
| 16 (13+d) | 0.8002 / 0.8002 | 0.8054 / 0.8056 | **0.8089** |
| 23 (20+d) | 0.7751 / 0.7751 | 0.7699 / 0.7697 | 0.7677 |
| 35 (32+d) | 0.7712 / 0.7712 | 0.7844 / 0.7853 | 0.7972 |
| 54 (51+d) | 0.7688 / 0.7688 | 0.7808 / 0.7765 | 0.7733 |

ProcessBench first-error accuracy (macro-8), per-dataset fits:

| Bank | content, no position (plain / grouped) | + position channel (plain) | grouped + position prior | first-error readout (grouped, with prior) | pooled content (plain / grouped) |
|---|---|---|---|---|---|
| 16 | 0.4131 / 0.3964 | 0.4131 | 0.3133 | 0.2227 | 0.4126 / 0.4092 |
| 23 | 0.3960 / 0.3822 | 0.3960 | 0.2230 | 0.1786 | 0.3957 / 0.3793 |
| 35 | 0.3650 / 0.3737 | 0.3650 | 0.2410 | 0.2138 | 0.3734 / 0.3665 |
| 54 | 0.3679 / 0.3694 | 0.3697 | 0.2158 | 0.1833 | 0.3730 / 0.3790 |

References: ct7 0.7724 / 0.3989, fam421 0.7801 / 0.3980 (within-AUC / ProcessBench).

## Reading
- Per-dataset fitting is free on PRMBench (the plain average is bitwise identical; the grouped version within 0.001 on three
  banks and +0.004 on B54), so it can be the method's contract without loss.
- On ProcessBench the per-cell fit decides by itself not to use the position channel (dropped in 32/32 cell-banks, kept on
  PRMBench), so the plain average with the position channel no longer costs ProcessBench (31/32 identical to no position).
- The per-cell position PRIOR is dangerous: on ProcessBench the channels are high on the first step of every answer, clean ones
  included (a telemetry artefact), the latent class latches onto it, and in 17/32 grouped cell-banks the method then always
  predicts step 0 (-23 points there). A prior estimated from the channels' agreement cannot tell errors from where the
  channels are systematically high.
- The first-error readout fails (10-20 points below the argmax, also without any prior): with a per-step latent prevalence of
  0.3-0.5 the product favours the earliest high step. The plain argmax is already the right readout for ProcessBench.
- The cross-fitted slope gives the highest PRMBench value so far (0.8089 on 13+d; 0.7972 on 32+d) by giving position 1.4-2.2x
  more weight, but loses on 23+d (it re-weights the content) and costs ProcessBench: not adopted.
- Open: a position term that cannot latch onto a telemetry artefact (e.g. estimated only from answers' relative change, or
  removed when the content score itself carries the same position profile); the external check with per-dataset fitting.
