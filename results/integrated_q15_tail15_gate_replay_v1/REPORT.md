# Integrated q15 finalist plus tail15 gate replay v1

Status: **COMPLETE / REVIEW PASS** on 13,769 answers, including all 6,800
ProcessBench rows and 145,597 official steps. This is the requested cumulative
same-development integration check, not independent confirmation.

## Integrated algorithm

`q15 {H0lim, VE0, VE0.75, VE1} -> per-view step Top10 -> raw equal locator -> earliest argmax -> whole-answer mean missing top-15 mass gate -> other-fold q=.3 threshold`

The gate is ProcessBench-only, so PRMB scores remain the frozen q15 finalist
values: within .753436, fold-pooled .722708 and PRMScore .634412.

## Decomposition

| Complete algorithm | PB all-8 | PB q4 | PB q8 | Clean accuracy | Error exact |
|---|---:|---:|---:|---:|---:|
| Original static locator + entropy | 36.1674% | **36.9200%** | 35.4148% | 50.0848% | 27.0599% |
| q15 finalist locator + entropy | 36.6201% | **37.4205%** | 35.8196% | 50.0848% | 27.6677% |
| Original static locator + tail15 | 36.5937% | 35.6788% | 37.5085% | **57.9304%** | 27.4651% |
| **Integrated q15 finalist + tail15** | **36.8818%** | 36.0180% | **37.7456%** | **57.9304%** | **27.9604%** |

The integration is positive in the requested additive sense:

- locator decisions alone: +0.453 PB points over the original algorithm;
- tail15 gate added to the frozen finalist: +0.262 points;
- all accepted decisions together: +0.714 points over the original algorithm.

## Family-wise paired intervals

Intervals are 98.75% whole-source-group bootstrap intervals over the four
frozen decomposition contrasts. Values are PB percentage points.

| Contrast | Point delta | Interval |
|---|---:|---:|
| Integrated - q15 finalist + entropy | +0.262 | [-1.413, +1.962] |
| Integrated - original static + entropy | **+0.714** | [-1.085, +2.518] |
| Original static + tail15 - original static + entropy | +0.426 | [-1.223, +2.081] |
| Integrated - original static + tail15 | +0.288 | [-0.425, +1.003] |

All point deltas are positive, so the preregistered composition gate passes.
Every interval crosses zero, so the evidence does not establish a reliable F1
gain. The correct distinction is: **the selected decisions compose without an
observed regression, but the added gate benefit remains uncertain**.

## Decision

`RETAIN_TAIL15_MEAN_AS_NEXT_GATE_CANDIDATE; DO_NOT_YET_REPLACE_ENTROPY`.

The next bounded gate question should be whether the fixed q=.3 operating point
is suppressing the stronger tail-mass ranking signal. Any selected quantile or
calibration rule must again be inserted into the complete q15 algorithm and
rerun cumulatively. The final new-model confirmation remains later.

## Integrity

- Original static + entropy and q15 finalist + entropy replay exactly.
- Integrated q15 + tail15 replays the selected gate result exactly.
- No feature/readout/threshold was reselected in the integration runner.
- Four mechanism checks and all source result reviews PASS.
- Full coverage: 13,769 answers, 6,800 PB decisions and 145,597 steps.
- No fitting, package installation, GPU/cluster action, Drive mutation, commit
  or push was performed by the integration replay.
