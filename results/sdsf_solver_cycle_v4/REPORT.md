# SDSF solver cycle v4 — fixed-stable real artifact

Decision: **ABANDON_FULL_INVERSE_SDSF**.

This is a mechanism study on the retrospective 24-cell artifact. It does not promote a tuned solver.

## Full-data factorial

| contrast | cell mean | family macro [95% CI] | W/L |
|---|---:|---:|---:|
| `head rescaling only` | -0.00 | -0.01 [-0.02, +0.00] | 10/11 |
| `tail addition only` | -3.28 | -2.01 [-3.47, -0.77] | 3/21 |
| `full inverse vs PCR` | -3.32 | -2.04 [-3.52, -0.78] | 3/21 |
| `direct two-channel CCA vs PCR` | -1.48 | -1.96 [-3.67, -0.46] | 7/17 |

## Condition path

Family-weighted tail-effect slope: **-0.617 points per log(kappa)**. Negative means weaker regularization admits a more harmful tail.

## Held-out sample-size test

| training fraction | tail effect | head-rescaling effect |
|---:|---:|---:|
| 0.25 | -4.44 | +0.00 |
| 0.50 | -3.86 | +0.00 |
| 0.75 | -3.34 | -0.00 |

## Conclusion

The low-eigenvalue tail, not top-two rescaling, carries the loss; it remains harmful on held-out rows at the largest training fraction. Direct channel CCA also fails to beat PCR. The supported action is to abandon full-inverse SDSF for these features, retain the low-dimensional PCR solver, and investigate dependency information only as a reliability correction on genuinely new families.
