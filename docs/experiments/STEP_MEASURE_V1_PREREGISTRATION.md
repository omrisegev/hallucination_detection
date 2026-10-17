# Pre-registration — replacing the step-measurement stage

Claude, 2026-09-19. Branch `claude/token-probability-fusion-v1`. Written while the sweep
was running and **before any of its numbers were read**. Development-only.

## Scope, fixed by Omri

The localizer is three stages: a token feature bank, a fusion that collapses the bank to
one token series, and a **step measurement** that collapses the tokens inside a step to
one number.

**Only the third stage is in scope.** The eleven-channel bank, the pooled standardizer and
the L-SML weights are taken as given and are not reopened — the input to this experiment
is the finished fused token series cached in `C1_TOKEN_SERIES.npz`, whose Top-10 step
readout replays the published arm to 0.000e+00. Anything this stage fits is fitted **per
model**, over all of that model's steps: q4 has 3,400 answers / 25,697 steps / 2.19M
tokens, q8 has 10,369 / 119,900 / 4.78M.

Directions of interest, in Omri's order: self-supervised, graph, DSP. This document covers
the **DSP** arm, which is run first because it is the only one whose defect is already
measured rather than hypothesised, and because it is an hour of CPU.

## Why this stage rather than any other

| lever | measured range in gate-free SLA |
|---|---|
| **step-measurement width** (K=1 → K=20) | **20.42 → 37.01, a 16.6 pp range** |
| decision rule on top of it (argmax / first-crossing / CUSUM / BOCPD) | 3 to 8 pp, and never above argmax on the step series |
| fitting scope of the fusion (pooled / per model / per cell) | −0.22 to −1.85 pp, never positive |
| feature-bank composition (the energy pair) | −0.03 to −1.69 pp |

The width of one statistic is worth more than every other axis tried, and it was never
designed — 10 is an inherited constant.

## The two defects, and the 2x2 that crosses them

**Defect 1 — a Top-K mean assumes white noise.** Measured on the fused token series:
lag-1 autocorrelation **0.500**, decaying through 0.371 / 0.272 / 0.168 / 0.109 at lags
2 / 3 / 5 / 8 and reaching zero only past lag 80. A flat K-window therefore carries about
`(1+r)/(1-r) = 3x` the variance it would on independent samples. Independently:
`n_eff` per step is **22–29 in every one of the eight cells** while raw tokens per step
range **55–93**, so the extra tokens in a long step are very nearly redundant. Nothing in
the pipeline prewhitens. The textbook response to a known signal in coloured noise is to
prewhiten first and match afterwards.

**Defect 2 — a Top-K mean throws away time.** It takes the K largest values *anywhere* in
the step, unordered. If the error is a contiguous burst, the matched statistic is the best
contiguous run of tokens, not the best unordered set.

| | unordered (Top-K mean) | contiguous (best window of width w) |
|---|---|---|
| **raw series** | the incumbent, K=10 → 35.92 | |
| **whitened series** | | |

Whitener: AR(8) via Yule–Walker on pooled *within-answer* autocorrelations (never across
an answer boundary), fitted per model on training folds only, applied causally with the
first `order` samples padded by the answer's own first value so that step 0 stays
scoreable — 12.25% of first errors are at step 0. Grids: K ∈ {1,3,5,10,20,40,80},
w ∈ {1,2,4,8,16,32,64}. A single pooled whitener is included as a control on the scope.

## Predictions and falsification

**P1 — the width is buying noise suppression.** If the K=10-to-20 optimum exists *because*
the noise is correlated, then after whitening **the best width moves DOWN toward 1 and the
peak moves UP**. Falsified if whitening leaves the optimum where it is, or lowers the peak.
This is the load-bearing prediction: it is what distinguishes "the readout is a badly
tuned filter" from "the readout is doing something the filter framing does not describe".

**P2 — the error is a contiguous burst.** The best contiguous window beats the Top-K mean
at matched effective width. Falsified if it does not, in which case the informative tokens
inside a step are scattered rather than adjacent, and the temporal axis inside a step
carries nothing.

**P3 — per-model fitting earns its scope.** The per-model whitener beats the single pooled
one. Falsified if the pooled whitener is at or above it — which would repeat what per-model
scope already did for the fusion stage (−0.22 short / −0.65 long, both intervals covering
zero), and would mean the two models' token dynamics are not different enough to matter.

## Protocol

Primary endpoint: **gate-free SLA** — exact first-error hit on the 4,442 erroneous
ProcessBench answers, per cell, mean over the eight cells. Secondary: `within1`, and
per-subset. Uncertainty: paired source-group bootstrap, 10,000 draws, seed 20260919,
`_group_draws` from `stage_b_2x2_v1.py`.

Anchor: `raw x Top-10` must replay **35.92** before any other number is printed.

**Label-free selection** reuses criterion 1 from `readout_calibration_v1.py`: split-half
odd/even token reproducibility, with the count proportionally halved on each half so that
each half computes the same statistic as the whole rather than a denser one. The
label-using best of the sweep is reported as a **ceiling, never as a candidate** — the same
discipline the LOCO-5 threshold sweep is held to.

**Prior belief, recorded so the result can contradict it.** I expect P1 to hold in
direction and to be small — the `n_eff` evidence is strong but the optimal K does *not*
track `n_eff` (n_eff is flat at 22–29 while the best K ranges 9–56), which already says
something other than pure noise-averaging is setting the width. I expect P2 to fail,
because the error step profile is an isolated impulse at step resolution and nothing yet
suggests a multi-token burst at token resolution. I give P3 the lowest odds of the three.

## What this is not

Not a candidate for the method of record. Not a confirmation — this population has been
inspected before, and no untouched test is involved. Not a change to CT7, to the gate, or
to any frozen release.
