# Dependency-aware U-PCR reliability cycle v5

Decision: **STOP_SYNTHETIC_HYPOTHESIS_REJECTED**.

This cycle changes only the pair-equation reliability solve. Every arm uses the same sparse decomposition, g2 interval, and two-component PCR final solver.

## Development convergence

| step | candidate | utility | rho error reduction | AUROC delta | running best | promoted |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | `diag_var` | -0.0292 | -2.92% | -0.001 pp | -0.0292 | yes |
| 2 | `diag_mad` | -0.0078 | -0.78% | -0.001 pp | -0.0078 | yes |
| 3 | `gaussian_gls` | -0.0908 | -9.08% | +0.003 pp | -0.0078 |  |
| 4 | `lw_gls` | -0.0496 | -4.96% | +0.001 pp | -0.0078 | yes |
| 5 | `hybrid_gls` | -0.0531 | -5.31% | +0.002 pp | -0.0078 |  |

![Development convergence](convergence.svg)

## Sealed synthetic validation

| candidate | rho reduction [95% CI] | AUROC delta [95% CI] | clean | p05 | result |
|---|---:|---:|---:|---:|:---:|
| `diag_mad` | -0.35% [-2.51, +1.48] | -0.000 [-0.001, +0.000] pp | +0.000 pp | -0.004 pp | **FAIL** |
| `diag_var` | -2.47% [-6.22, +0.68] | +0.000 [-0.001, +0.002] pp | -0.000 pp | -0.006 pp | **FAIL** |
| `lw_gls` | -6.69% [-18.60, +2.63] | +0.000 [-0.004, +0.005] pp | -0.002 pp | -0.030 pp | **FAIL** |

## Real replay

Not run: no candidate crossed all frozen synthetic gates. This is the registered stop rule, not missing output.

## Scientific conclusion

Covariance-aware moment weighting did not satisfy the preregistered mechanism-and-no-harm criteria on disjoint synthetic validation. The correlated pair equations are real, but correcting their sampling covariance is not a demonstrated improvement to U-PCR under these worlds. The real artifact was intentionally not opened for candidate selection.
