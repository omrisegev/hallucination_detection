# CIW cross-scale localization v1

## Question

Can the CIW-DEEM input idea be moved from completed-response features to the
joint response/token setting, so that whole-answer structure removes nuisance
variation from the 29 token streams before step localization?

## Input model

The fit worker receives only the already prepared, target-free localization
matrix and the separately frozen CIW response score.  For token coordinate
`j`, it uses two response-level predictors: the mean of coordinate `j` over
the complete answer and the CIW-DEEM answer risk.  Five deterministic
row-held-out folds estimate predictability:

```text
reliability_j = clip(R2_oof_j, 0, 1)
alpha_j       = 0.5 * reliability_j
innovation_j  = (token_j - predicted_j) / sd(oof residual_j)
token'_j      = (1 - alpha_j) * standardized_token_j
                + alpha_j * innovation_j
```

The unchanged two-component token IU-PCR head fuses `token'`, and the maximum
token risk in each supplied step is combined with the response-risk rank by
the existing geometric-mean adapter.  `alpha=0` reproduces the frozen token
IU-PCR score to floating-point tolerance (`<= 2.3e-16` on the real smoke
cell).  No first-error or step-correctness target is accepted by the fit code.

One prespecified solver ablation replaces the token IU-PCR head by the existing
SU-PCR low-rank-plus-sparse reproduction.  It leaves the cross-scale gate and
all other inputs fixed.

## Structural finding

The whole-answer predictors explain a modest portion of token variation.  In
the frozen 13-cell IU run, mean clipped OOF R2 ranges from `0.0733` to `0.0901`
and the mean gate ranges from `0.0366` to `0.0451`.  Thus most token variation
is local; the result does not justify replacing token coordinates by response
coordinates.

## Frozen primary results

The primary arm below uses the CIW response head, matching the previous CIW
application adapter.

| Method | ProcessBench macro F1 | PRMBench step AUROC | PRMBench step AUPRC |
|---|---:|---:|---:|
| Previous CIW response + frozen token IU29 | 0.309136 | 0.581138 | not promoted |
| Cross-scale input + token IU-PCR | 0.308301 | 0.582489 | 0.196327 |
| Cross-scale input + token SU-PCR | 0.306202 | 0.582516 | see frozen report |
| B3 response + frozen token IU29 | 0.310228 | 0.584218 | 0.197104 |
| IU-PCR response + frozen token IU29 | 0.308194 | 0.598834 | 0.208690 |
| DUFS-LIU response + frozen token IU29 | 0.309731 | 0.600431 | 0.209774 |

The primary cross-scale arm therefore does not improve both tasks.  It loses
`0.000835` ProcessBench macro F1 versus the previous CIW adapter, while gaining
`0.001351` PRMBench AUROC.

## Response-head diagnostic after the primary opening

After the primary labels had been opened, the already frozen corrected token
score was paired with three other frozen response heads.  This is a
retrospective mechanism diagnostic, not a second clean selection test:

| Response head | Frozen-token PRM AUROC | Corrected-token PRM AUROC | Delta |
|---|---:|---:|---:|
| B3 | 0.584218 | 0.585575 | +0.001356 |
| IU-PCR | 0.598834 | 0.600109 | +0.001275 |
| DUFS-LIU | 0.600431 | 0.601699 | +0.001268 |

The same corrected token head loses `0.0006` to `0.0009` ProcessBench macro F1
for all three response heads.  Preserving each answer's original token-score
mean and variance removes that ProcessBench loss for B3 (`0.310259` versus
`0.310228`) but also removes the PRMBench gain (`0.584028` versus `0.584218`).

## Decision

The cross-scale innovation layer is a supported **PRMBench step-ranking
ablation**, not a replacement localization method.  It consistently improves
PRMBench with three independent frozen response heads, but the unsupervised
statistic used here does not tell us when the correction is safe for
ProcessBench's first-error-plus-clean-abstention objective.  SU-PCR does not
resolve that mismatch: on a real cell its ordering is almost identical to the
IU head (Spearman `0.99994`) and its ProcessBench score is worse.

The next distinct question is a task-policy/readout question, not another
covariance solver: learn or derive an unlabeled reliability statistic that
separates absolute answer calibration from relative step evidence.  Until
that statistic is validated, retain the frozen task incumbents.

## Files

- `spectral_utils/ciw_cross_scale_localization.py`
- `scripts/reconstruction_benchmark/run_ciw_cross_scale_localization.py`
- `scripts/test_ciw_cross_scale_localization.py`
- `local_cache/ciw_cross_scale_localization_v1/`
- `local_cache/ciw_cross_scale_su_localization_v1/`
