# Probability normalization and mass ablation v1

Status: **COMPLETE / REVIEW PASS** on the frozen 13,769-answer development
population. This is diagnostic development evidence, not untouched
confirmation.

## Result in one sentence

Top-K renormalization is not the standalone-to-fusion failure: proper
escort-varentropy is invariant to `p -> q` on a fixed support and the complete
run verifies that identity numerically. Support width matters, while
within-answer mean subtraction is the clear cause of the large pooled and
PRMScore calibration loss. A single coarse tail bucket does not repair the
representation.

## Frozen results

| Method | PB all-8 | PB raw exact | PRMB within | PRMB pooled | PRMScore |
|---|---:|---:|---:|---:|---:|
| `VE.75 q15` raw | **36.7648%** | 33.7461% | .732302 | .704314 | .618679 |
| `VE.75 q50` raw | 36.1581% | 33.0482% | .749937 | **.721936** | **.633719** |
| `VE.75` coarse tail-bucket-15 | 36.7180% | **33.8586%** | .731818 | .703037 | .618099 |
| `VE1 q15` raw | 35.9610% | 32.6204% | .737786 | .710138 | .625781 |
| `VE1 q50` raw | 35.6755% | 32.2152% | .742465 | .715783 | .632777 |
| `VE1` coarse tail-bucket-15 | 36.0756% | 32.7780% | .738432 | .710165 | .625775 |
| Residual tail mass alone | 33.0643% | 28.2080% | .732226 | .710768 | .620627 |
| Four-view equal, raw | 36.1674% | 32.6655% | **.751115** | **.721407** | **.634805** |
| Four-view equal, center only | 36.1674% | 32.6655% | **.751115** | .661668 | .576951 |
| Four-view equal, scale only | 36.2041% | 32.9356% | .746602 | .715708 | .630093 |
| Four-view equal, answer z-score | 36.2041% | 32.9356% | .746602 | .675153 | .589883 |
| Four-view equal, fold-global z-score | **36.3769%** | **33.0707%** | .746098 | .716879 | .629195 |
| Five-view fold-global z-score + tail | 35.3296% | 31.3372% | .742357 | .716869 | .631445 |

Bold values identify the leader inside each local comparison block, not a
globally selected deployable winner.

## Registered primary contrasts

Intervals are the frozen family-wise 99.1667% source-group bootstrap intervals.
PB deltas are percentage points below; within deltas remain AUROC units.

| Contrast | PB delta [interval] | PRMB-within delta [interval] | Reading |
|---|---:|---:|---|
| `VE.75 q50 - q15` | -0.607 [-1.669, +0.368] | +.017634 [.013837, .021295] | Strong PRMB gain, PB tradeoff |
| `VE.75 tail bucket - q15` | -0.047 [-0.431, +0.356] | -.000485 [-.001617, .000616] | No useful change |
| `VE1 q50 - q15` | -0.285 [-1.214, +0.614] | +.004678 [.002115, .007423] | Passes frozen promotion rule |
| `VE1 tail bucket - q15` | +0.115 [-0.297, +0.505] | +.000646 [-.000579, .001882] | No useful change |
| Fold-global z - answer z | +0.173 [-0.032, +0.403] | -.000504 [-.001110, .000113] | Same localization; calibration restored |
| Global z + tail - global z | -1.047 [-2.260, +0.152] | -.003741 [-.006303, -.001041] | Tail addition is harmful |

## What this answers

### 1. Renormalizing the retained head is not the VarEntropy discrepancy

The maximum token-level discrepancy between a proper calculation from raw
`p_i` and the equivalent calculation from conditional `q_i` was
`8.73e-10` across the complete population. The maximum step-score difference
against four frozen standalone references was `7.05e-12`. The hypothesis that
`p -> q` alone changed the successful varentropy is therefore rejected.

This does not say the retained support is sufficient. It says only that scaling
all probabilities on one fixed support cannot change a properly centered
escort variance.

### 2. Support width matters; total missing mass alone is not enough

Moving from 15 to 50 ranks materially improves PRMB. For `VE1`, the +.004678
within-answer gain is positive under the corrected interval and the PB loss is
inside the frozen 0.5-point noninferiority margin, so `VE1 q50` is the one
representation arm promoted to the next feature-bank experiment.

For `VE.75`, q50 gains much more on PRMB (+.017634) but loses 0.607 PB points,
just beyond the registered point-estimate margin. Keep q15 for the PB-facing
single-view frontier and treat q50 as a task-specific sensitivity, not a
replacement.

The residual mass outside top 15 is small: mean .00363, median .00144 and 90th
percentile .00790 at the answer-mean level. Collapsing it into one category
does not preserve how that mass is distributed across low-probability tokens.
That explains why q50 can help while the coarse bucket and tail-alone arms do
not. The data support tail *shape/support*, not tail mass as one extra scalar.

### 3. Within-answer centering destroys cross-answer calibration

The registered affine controls isolate this cleanly:

- raw and center-only fusion have identical PB and within-answer AUROC, but
  centering drops pooled AUROC from .721407 to .661668 and PRMScore from
  .634805 to .576951;
- scale-only and full answer z-score also have identical PB and within AUROC,
  but centering drops pooled AUROC from .715708 to .675153 and PRMScore from
  .630093 to .589883;
- fold-global standardization retains PB/within behavior while recovering
  pooled AUROC to .716879 and PRMScore to .629195.

Thus the earlier fusion did compute the standalone features correctly and then
discard their between-answer levels during preprocessing. Future calibrated
fusion should not subtract a separate mean inside every answer.

The raw equal fusion is not a balanced four-expert result. Average external
feature scale is approximately 62.3% `VE0`, 30.1% `H0lim`, 5.2% `VE.75`, and
2.5% `VE1`; its mean within-answer correlation with `VE0` is .9918. Its good
PRMB numbers therefore mostly recover the low-alpha leader rather than prove a
successful general fusion.

## Decision

- `REJECT_P_TO_Q_AS_CAUSE`
- `PROMOTE_VE1_Q50_SUPPORT_VARIANT`
- `RETAIN_VE075_Q15_FOR_PROCESSBENCH`
- `REJECT_COARSE_TAIL_BUCKET_AS_FUSION_INPUT`
- `AVOID_WITHIN_ANSWER_CENTERING_FOR_CALIBRATED_FUSION`
- `CARRY_FOLD_GLOBAL_OR_SCALE_ONLY_PREPROCESSING_AS_CONTROLS`

Experiment 2 remains the gate study and has not been started. Its first screen
will use Top10 mean as agreed; this experiment did not alter the frozen
mean-entropy q=.3 gate.

## Integrity record

- 5/5 unit tests pass; real-data smoke passes on 27/27 answers.
- Full coverage: 13,769 answers and 145,597 official steps.
- 41,645 outer/nested standardizer contexts reviewed; held source groups are
  excluded in every context.
- Scores were frozen before evaluation at SHA256
  `402af091e82a10147de659cac22befbd7f8fd3a5c8c3427082935368ab50daef`.
- Evaluation-only review status: PASS; no performance fallback or threshold was
  applied.

Canonical artifacts: `FROZEN_SCORES.json`, `SCORE_REVIEW.json`, `METRICS.json`,
`CONTRASTS.json`, `COMPARISON.csv`, `POSTHOC_PREPROCESSING_REVIEW.json`, and
`RESULT_REVIEW.json`.

