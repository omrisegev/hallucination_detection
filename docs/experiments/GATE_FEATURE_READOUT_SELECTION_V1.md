# Gate feature and readout selection v1

Status before execution: **PROTOCOL FROZEN / NOT YET SCORED**
Frozen on: 2026-09-14
Population: all 6,800 ProcessBench answers in the frozen localization
development contract; the q15 finalist locator is fixed.

## Question

Can replacing the current whole-answer mean-entropy detector, while keeping its
label-free foldwise q=.3 threshold rule fixed, improve the final first-error
decision produced by the selected q15 locator?

This experiment selects a gate feature/readout on development. It is followed
by a separate cumulative integration replay on the same benchmark, as requested;
neither stage is untouched model-transfer confirmation.

## Fixed locator and decision rule

- Locator: the bitwise-frozen `selected_q15_raw_per_view_top10` step vector.
- Peak: earliest step argmax.
- Gate: open the peak when the answer detector is at or above the q=.3 quantile
  fitted on all other source-group folds; otherwise predict clean (`-1`).
- One detector definition and one threshold rule are used in all eight PB
  cells. No feature, readout, sign or threshold depends on dataset/model/cell.
- q=.3 is not swept in this experiment. This isolates feature/readout choice.

## Frozen token signals

All signals are computed from saved token telemetry; higher means more likely
erroneous under the declared natural or frozen label-free orientation.

1. native stored top-15 Shannon token entropy (renormalized head; current gate source);
2. oriented q15 `H0lim`;
3. oriented q15 `VE0`;
4. oriented q15 `VE0.75`;
5. oriented q15 `VE1`;
6. q15 Shannon entropy `H1`;
7. q15 min-entropy `Hinf = -log(max q)`;
8. raw top-1 surprisal `-log p1`;
9. probability mass missing outside top-15;
10. probability mass missing outside top-50;
11. the raw arithmetic mean of the four frozen finalist token views.

`H1`, `Hinf` and `-log p1` are separate. The first two use the normalized q15
head; `-log p1` preserves the raw top-1 probability level.

## Frozen readouts

Every token signal is reduced to one answer detector in three ways:

- `token_mean`: mean over all answer tokens;
- `token_top10`: mean of the ten highest-risk tokens in the complete answer;
- `mean_step_top10`: Top10 within each official step, followed by an equal mean
  over the answer's steps.

This gives 33 candidates. The existing gate is exactly
`entropy_native__token_mean` and must replay bitwise.

## Selection and reporting

For every candidate, report PB all-eight, q4 and q8 macro F1; every cell's F1;
clean accuracy; erroneous-answer exact accuracy; fold thresholds; and
clean/error separability AUC. Rank by all-eight macro F1. Exact ties prefer the
higher worst-cell F1, then `token_top10`, `mean_step_top10`, `token_mean`, then
lexical name. Promote the winner only if its all-eight point estimate exceeds
the entropy baseline; otherwise retain the baseline.

The selected-versus-baseline paired whole-source-group bootstrap uses 10,000
draws and a descriptive 95% interval. Because the same development labels
select the candidate, this interval is diagnostic and not a selection-adjusted
confirmation interval.

## Required gates

- deterministic unit tests and an all-cell/all-fold real-data smoke;
- exact detector/record alignment and finite PB coverage;
- bitwise entropy-mean reproduction;
- detector archive frozen and hashed before target evaluation;
- no labels accepted by the signal/readout API;
- explicit selection-optimism boundary;
- machine-readable metrics, selection, error analysis and review.
