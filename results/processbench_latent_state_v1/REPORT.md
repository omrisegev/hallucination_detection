# IU-PCR latent-state localization on ProcessBench

## Result

**Decision: NOT PROMOTED.** The reversible IU-HMM changes all-cell ProcessBench
F1 by **-1.69 points** and exact localization by
**-1.49 points** relative to the frozen core-five
DUFS-LIU localizer. On the six cells not used by the earlier GL-LIU component
selection, the differences are **-1.58 F1 points** and
**-1.54 exact-localization points**.

The direct mechanism control is also negative: relative to the ordinary IU-PCR
sequence that initializes it, the HMM changes PB-F1 by
**-1.63 points** and raw exact localization by
**-1.42 points**. The six non-selection cells are held-out
method-selection cells from four dataset families and two scorer models; they
are not six independent datasets.

This comparison is exploratory: these ProcessBench labels were already opened
in earlier project experiments. Labels were not used to fit IU-PCR, either HMM,
the global detector, or any score. A separate process froze and hashed every
score before the evaluation command read labels or step spans.

## What the metrics mean

- **Exact localization** is the percentage of erroneous traces whose predicted
  token maps to the annotated first erroneous reasoning step.
- **Within one step** also accepts the adjacent step.
- **Clean accuracy** is the percentage of fully correct traces on which the
  global detector abstains.
- **ProcessBench F1** is the harmonic mean of exact localization on erroneous
  traces and clean accuracy. It therefore tests both detection and placement.

All headline tables are equal-cell macro averages. They do not pool every trace
as though the two scorer-model views of a dataset were independent samples.

## Methods

Every method except Mind the Gap uses the same frozen mixed-v2 DUFS-LIU global
detector. The local input is always the same five token curves: entropy,
entropy sliding variance, absolute entropy CUSUM, spilled-energy sliding
variance, and absolute spilled-energy CUSUM.

Ordinary two-component IU-PCR fuses these curves into one scalar token-risk
sequence. The primary HMM has two reversible latent states. Its output at token
`t` is the posterior probability of entering the higher-IU-risk state at `t`.
The absorbing HMM is a falsification control for the stronger assumption that
the risk state persists after the first error. Both HMMs share one emission
variance, use three deterministic starts, and select the valid start with the
largest label-free likelihood. Failed state-separation or collapse guards fall
back exactly to the IU-PCR argmax.

## End-to-end results

| System | All 8 PB-F1 (%) | Erroneous exact (%) | Clean accuracy (%) | Within-one SLA (%) | Non-selection 6 PB-F1 (%) |
|---|---|---|---|---|---|
| IU-PCR core | 31.67 | 22.00 | 58.66 | 46.29 | 31.41 |
| Temporal LIU core | 31.36 | 21.79 | 57.99 | 46.76 | 30.76 |
| DUFS-LIU core | 31.72 | 22.05 | 58.71 | 46.28 | 31.41 |
| IU-HMM reversible | 30.03 | 20.91 | 55.94 | 43.03 | 29.83 |
| IU-HMM absorbing | 12.64 | 7.62 | 47.99 | 23.49 | 12.44 |
| Mind the Gap | 25.71 | 17.84 | 48.63 | 39.35 | 24.74 |

![End-to-end F1](figures/end_to_end_f1_per_cell.png)

## Localizer results

| Localizer | All 8 exact (%) | Within one step (%) | Mean signed step error | Normalized token distance (%) | Non-selection 6 exact (%) |
|---|---|---|---|---|---|
| IU-PCR core | 26.62 | 57.09 | 0.13 | 28.98 | 25.75 |
| Temporal LIU core | 26.41 | 57.18 | 0.07 | 28.56 | 25.14 |
| DUFS-LIU core | 26.70 | 57.10 | 0.14 | 28.95 | 25.78 |
| IU-HMM reversible | 25.20 | 52.26 | -0.29 | 30.18 | 24.23 |
| IU-HMM absorbing | 8.73 | 28.23 | 2.61 | 55.70 | 8.43 |
| Mind the Gap | 22.25 | 49.68 | 0.78 | — | 21.04 |

![Local exact](figures/local_exact_macro.png)

![Paired F1 delta](figures/paired_f1_delta.png)

The 95% range across the existing repeated calibration splits is
[-3.71, +0.76] F1 points.
This is split variability, not an independent-data confidence interval.

## Mechanism diagnostics

![HMM diagnostics](figures/hmm_diagnostics.png)

![Posterior diagnostics](figures/posterior_diagnostics.png)

| Cell | HMM | Fallback | Separation (SD) | High occupancy (%) | Variance | P(0 to 1) (%) | P(1 to 0) (%) | Mean peak (%) | Entry entropy (%) | No credible entry (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| 4b/gsm8k | reversible | no | 2.29 | 31.24 | 0.47 | 1.22 | 2.96 | 68.95 | 42.20 | 4.75 |
| 4b/gsm8k | absorbing | no | 1.56 | 11.36 | 0.80 | 0.05 | 0.00 | 3.96 | 9.17 | 88.50 |
| 4b/math | reversible | no | 3.26 | 14.52 | 0.43 | 0.38 | 2.24 | 40.10 | 34.78 | 44.70 |
| 4b/math | absorbing | no | 3.55 | 6.44 | 0.57 | 0.01 | 0.00 | 1.33 | 0.79 | 97.20 |
| 4b/olympiadbench | reversible | no | 2.70 | 27.07 | 0.41 | 0.75 | 2.01 | 68.43 | 40.62 | 8.50 |
| 4b/olympiadbench | absorbing | no | 1.18 | 37.90 | 0.75 | 0.08 | 0.00 | 8.04 | 19.13 | 68.00 |
| 4b/omnimath | reversible | no | 2.89 | 14.89 | 0.48 | 0.33 | 1.86 | 42.27 | 42.74 | 33.70 |
| 4b/omnimath | absorbing | no | 4.13 | 4.09 | 0.60 | 0.01 | 0.00 | 1.58 | 0.82 | 96.50 |
| 8b/gsm8k | reversible | no | 2.26 | 37.22 | 0.45 | 1.19 | 2.54 | 81.38 | 37.30 | 3.50 |
| 8b/gsm8k | absorbing | no | 1.61 | 10.38 | 0.80 | 0.04 | 0.00 | 3.54 | 8.10 | 89.25 |
| 8b/math | reversible | no | 3.17 | 15.74 | 0.43 | 0.49 | 2.59 | 44.66 | 36.33 | 40.00 |
| 8b/math | absorbing | no | 3.43 | 6.48 | 0.58 | 0.01 | 0.00 | 1.55 | 1.09 | 97.20 |
| 8b/olympiadbench | reversible | no | 2.66 | 27.40 | 0.42 | 0.79 | 2.09 | 69.42 | 41.71 | 7.80 |
| 8b/olympiadbench | absorbing | no | 1.23 | 32.82 | 0.75 | 0.07 | 0.00 | 8.57 | 17.63 | 68.60 |
| 8b/omnimath | reversible | no | 2.74 | 17.48 | 0.48 | 0.44 | 2.07 | 47.90 | 44.15 | 26.00 |
| 8b/omnimath | absorbing | no | 3.53 | 5.11 | 0.62 | 0.01 | 0.00 | 2.42 | 1.16 | 95.10 |

![Error-aligned entry probability](figures/error_aligned_entry.png)

The error-aligned plot includes only cells that produced a true posterior entry
curve. A guarded fallback returns the IU-PCR risk curve and is never averaged
with probabilities. Its sharp mean peak at the annotated boundary is an
interesting mechanism observation, not a performance claim: every annotated
error begins at a reasoning-step boundary, so a matched non-error-boundary
control is still required to distinguish error onset from generic step syntax.

![Signed step error](figures/signed_step_error.png)

![Position-length diagnostic](figures/position_length.png)

## Pre-declared decision panel

| Condition | Result |
|---|---|
| no reversible HMM fallback | PASS |
| all-cell exact localization improves | FAIL |
| non-selection exact localization improves | FAIL |
| all-cell ProcessBench F1 improves | FAIL |
| non-selection ProcessBench F1 improves | FAIL |
| no cell loses more than one F1 point | FAIL |

## Interpretation boundary

An HMM gain would show that explicit temporal state transitions improve the
same IU-PCR signal. It would not show that the hidden state is literally
correctness. A likelihood gain without localization gain means the HMM found a
stable temporal regime unrelated to the benchmark target. An absorbing-model
failure would specifically reject persistent post-error telemetry, not the
reversible onset model.
