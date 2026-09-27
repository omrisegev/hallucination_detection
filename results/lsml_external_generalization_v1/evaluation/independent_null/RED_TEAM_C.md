# Red-team C: independent math and null audit

The locked predictions contain signal beyond randomized step labels, but these
checks do **not** confirm a universal learned-fusion advantage. Local L-SML has
lower observed PRMScore than matched equal weighting on both Socratic backbones.
Hard2Verify's frozen-minus-equal gain remains within the global-shuffle contrast
range. Null contrasts are descriptive diagnostics, not source-group-valid tests.

## Evidence and access

`LABEL_NULL.json`: all 6,190/6,190 sealed answer/backbone records, comprising 200
Hard2Verify answers/1,860 steps and 2,995 Socratic answers/26,055 steps on each
backbone. Every cell seal and the current source-bundle hash passed before any
annotation was opened. Independent direct confusion algebra computes the metrics;
no root metrics, contrasts, reports or other agents' outputs were read.

1,000 global label permutations preserve each dataset's class counts. A separate
200-draw diagnostic permutes included-step labels within each answer, preserving
its class composition. The same label-permutation stream is used on both Socratic
backbones. Predictions stay fixed because labels enter neither fitting nor source
calibration. These shuffles do not assume exchangeability for inferential claims.

## Independently computed observed metrics

Values are fractions. Hard2Verify uses the harmonic mean of correct/error recall
(Balanced F1); Socratic uses the mean of the two class F1 scores (PRMScore).

| Arm | Hard2Verify/Qwen3 | Socratic/Qwen3 | Socratic/QwQ |
|---|---:|---:|---:|
| Frozen L-SML | 0.436695 | 0.632213 | 0.642382 |
| Frozen equal | 0.408822 | 0.607912 | 0.615034 |
| Frozen partition equal | 0.397576 | 0.612697 | 0.621098 |
| Local L-SML | 0.428054 | 0.619613 | 0.621526 |
| Local equal | 0.422117 | 0.632468 | 0.632401 |
| Local partition equal | 0.429312 | 0.630196 | 0.616116 |
| CT7 | 0.377510 | 0.587529 | 0.601656 |

All 21 observed arm/cell scores exceed their respective global and within-answer
shuffle 97.5th percentiles. This is a descriptive signal check, not 21 adjusted
hypothesis tests. Full distributions, standard deviations and all 18 contrasts
are in `LABEL_NULL.json`.

## Learned-minus-equal contrasts

Numbers below are percentage points. Brackets are descriptive 2.5th–97.5th
percentiles of the null contrast, not confidence intervals for the real effect.

| Contrast | Observed | Global null mean [range] | Within-answer null mean [range] |
|---|---:|---:|---:|
| Hard2Verify frozen − equal | +2.787 | +1.177 [−0.645, +2.949] | +1.197 [−0.127, +2.181] |
| Hard2Verify local − equal | +0.594 | +0.207 [−1.569, +2.054] | +0.193 [−1.409, +1.714] |
| Socratic/Qwen3 frozen − equal | +2.430 | +0.417 [+0.065, +0.764] | +0.578 [+0.247, +0.939] |
| Socratic/Qwen3 local − equal | −1.285 | −0.026 [−0.414, +0.346] | −0.047 [−0.403, +0.251] |
| Socratic/QwQ frozen − equal | +2.735 | +0.496 [+0.128, +0.860] | +0.702 [+0.350, +1.015] |
| Socratic/QwQ local − equal | −1.087 | +0.028 [−0.431, +0.451] | −0.041 [−0.471, +0.356] |

Prediction prevalence produces nonzero null advantages: frozen L-SML predicts
correctness less often than frozen equal. Global-shuffle mean Hard2Verify BF1 is
0.322921 versus 0.311147; Socratic/Qwen3 PRMScore is 0.488741 versus 0.484572;
Socratic/QwQ is 0.489791 versus 0.484830. Chance performance is therefore not
universally 0.5, and positive observed differences should not be attributed wholly
to better ranking. The observed Socratic frozen gains exceed both shuffle ranges;
Hard2Verify requires the separate registered source-group uncertainty assessment.

## Math and failure accounting

`ACTUAL_MATH.json` verifies all 6,190 answers and 377,790 step/arm decisions:
finite-score masks, zero decisions for empty steps, exact source-threshold
application, within-answer zero mean/unit variance, finite local weights with L1
norm one, required fit-row/channel counts, and identical local fallback curves.
Largest absolute standardized-score mean is 1.74e-15; largest standard-deviation
error 3.33e-16; largest weight L1 error 4.44e-16.

Native local coverage is 200/200, 2,984/2,995 and 2,984/2,995. All 22 non-native
records report insufficient fitting observations. No recorded native grouping is
degenerate and no native entropy-anchor correlation equals zero. Guarded
three-unit stages occur in 197, 2,959 and 2,850 answers; flagged four-unit stages
occur in 80, 1,208 and 1,251. Guards are disclosed parts of the recipe, not hidden
numerical failure fallback. Fallback scores match across local arms, but separately
source-calibrated thresholds can yield different fallback decisions.

Earlier independent synthetic checks verified continuous L-SML flattening
`w_j = cross_g * within_gj` to 3.33e-16 and paired group-bootstrap intervals
exactly on 1,000 draws over four answers/three groups. No mathematical identity
error was found. Current numerical-failure observation rejects silently substituted
estimator outputs at the external wrapper; source strict replay was handled by
the parent workflow, not independently repeated here.

## Feature permutation: bounded FEASIBILITY only

`FEATURE_NULL.json` checks **36/6,190**, 12 token-length ranks per cell including
extrema. Exactly bank11 column 0, `q15_H1`, is permuted across the union of valid
step-token positions within each answer, **crossing step boundaries**. It is not
a within-step permutation (which would preserve the frozen channel's Top10).

Actual inputs change in 36/36 answers. Frozen L-SML scores change in 36/36 and
13 decisions change. Local refits change scores/weights in 34/34 native examples,
partitions in 33/34, and 73 decisions. The two fallback examples are unchanged;
CT7 is exactly unchanged in 36/36 because its raw telemetry inputs are untouched.
This proves the perturbation is consumed. It does not test quality degradation,
establish a full-population feature null, or demonstrate fusion superiority.

Reproduction scripts: `audit_nulls.py` (feature and authorized post-seal label
modes), `actual_math.py`. JSON outputs carry source/seal/script hashes and seeds.
