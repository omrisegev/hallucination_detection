# Renyi position-temporal fusion v1

Status: **COMPLETE / REVIEW PASS** on the frozen 13,769-answer development
population. This is development evidence, not untouched confirmation.

## Result in one sentence

The answer-local position prior learned the hypothesized early-to-late shift
from `VE_0.75`/`VE_1` toward `H0lim`/`VE_0` and produced a small, family-wise
significant PRMB within-answer gain over its scale-only control, but no
ProcessBench gain; the fully external position-IU arm regressed strongly.
No fused arm dominates the best single alpha views.

## Frozen headline results

| Method | PB all-8 | PRMB within AUC | PRMB pooled AUC | PRMScore q=.8 |
|---|---:|---:|---:|---:|
| Single `H0lim` | 35.5349% | .743972 | .716235 | .633422 |
| Single `VE_0` | 35.5662% | **.753406** | **.723092** | **.635477** |
| Single `VE_0.75` | **36.7648%** | .732302 | .704314 | .618679 |
| Single `VE_1` | 35.9610% | .737786 | .710138 | .625781 |
| Four-view equal | 36.2041% | .746602 | .675153 | .589883 |
| Answer-local IU | 36.3500% | .745703 | .671526 | .587001 |
| External IU + position mean | 36.3544% | .738733 | .662449 | .583014 |
| External position IU | 34.6521% | .739485 | .663602 | .583683 |
| Local shrinkage IU + position | 35.9189% | .747381 | .671582 | .587410 |
| Local position scale-only | 35.7549% | .746325 | .672558 | .588215 |
| Local position shuffled | 36.4198% | .745555 | .670458 | .587007 |

The single-view leaders remain task-dependent: `VE_0.75` is the ProcessBench
leader, while `VE_0` is the PRMB leader on all three registered endpoints.
This reproduces the motivation for the experiment but does not make the
label-using union of their hits a deployable fusion.

## Primary contrasts

Primary confidence intervals are the frozen 98.333% grouped intervals with
10,000 canonical-source-group bootstrap draws.

- External position IU minus stationary IU with matched position mean:
  ProcessBench **-1.7024 percentage points**, CI **[-2.6533, -0.7953]**;
  PRMB within **+0.000751**, CI **[-0.000140, +0.001692]**. The coefficient
  trajectory is harmful on ProcessBench and uncertain on PRMB.
- Local shrinkage IU + position minus scale-only: ProcessBench
  **+0.1640 percentage points**, CI **[-0.1729, +0.4966]**; PRMB within
  **+0.001056**, CI **[+0.000357, +0.001770]**. The position-dependent
  direction contributes a real but small within-answer ranking gain; the PB
  effect is unresolved. PRMScore is slightly lower (.587410 versus .588215).

Secondary controls sharpen the boundary. Relative to static answer-local IU,
the local position arm gains +.001678 PRMB within AUC (95% CI
[+.001086,+.002271]) but changes PB by -0.4311 points (95% CI
[-1.0121,+.1530]). Relative to shuffled position it gains +.001826 within AUC
(95% CI [+.001283,+.002363]) while PB changes by -0.5009 points (95% CI
[-1.0725,+.0779]). The external position arm loses both to its shuffled
control: -1.6378 PB points and -.006806 within AUC, with both 95% intervals
strictly below zero.

## What the learned maps did

The label-free mean local position map follows the post-hoc hypothesis:

| Fractional position | `H0lim` | `VE_0` | `VE_0.75` | `VE_1` |
|---|---:|---:|---:|---:|
| First region | .1451 | .1170 | .1678 | .1705 |
| Last region | .1663 | .1512 | .1550 | .1546 |

Thus late tokens receive more `H0lim`/`VE_0`, while early tokens receive more
`VE_0.75`/`VE_1`. The mean per-answer position drift is .01447, compared with
.00448 for scale-only and .00132 for the shuffled control. This is a measured
mechanism result, not evidence that the true error position entered training:
all fits used only token features plus source-group/fold identity.

## Interpretation and presentation decision

The useful story is not a new universal fused winner. It is a three-part
result:

1. Alpha controls *where* uncertainty evidence is strongest: `VE_0.75` leads
   ProcessBench, while `VE_0` leads PRMB.
2. An unsupervised answer-local position prior can recover the expected
   early/late reweighting and measurably improve within-answer ranking over
   direction-free controls.
3. Position-dependent coefficients transferred wholly from other answers are
   too brittle here; they reduce ProcessBench sharply, and all fusion scores
   remain poorly cross-answer calibrated for PRMScore.

For a paper or talk, show the single-alpha frontier first, then the learned
early/late weight shift, then the primary control intervals. Do not headline
the external position method or claim that temporal fusion solved the
cross-benchmark tradeoff. The next bounded candidate, if pursued, should keep
the answer-local direction update and address score calibration or a
ProcessBench-safe gate; it should not add another alpha sweep on this same
development population.

## Integrity record

- Real-data smoke: 27/27 answers, no failed outer or nested vectors.
- Full run: 13,769/13,769 answers and 145,597 official steps.
- 55 outer/nested exclusion-safe external models, all independently replayed.
- Label-firewall perturbations for PB, PRMB outer and PRMB two-fold nested fits
  left fitted bytes and provenance unchanged.
- `VE_1` step scores reproduce the frozen varentropy15 reference.
- No score fallback and no performance threshold were applied.
- One scoring checkpoint resume followed a numerical assertion tolerance fix;
  evaluation was then rerun with the protocol's stricter 98.333% primary
  interval. These driver-only changes and hashes are retained in `MANIFEST.json`.

Canonical machine-readable artifacts: `METRICS.json`, `CONTRASTS.json`,
`CALIBRATION.json`, `COMPARISON.csv`, `RESULT_REVIEW.json`,
`EXCLUSION_REVIEW.json`, `LABEL_FIREWALL_REVIEW.json`, and `SCORE_REVIEW.json`.
The large resumable SQLite and token/step score archive remain local.
