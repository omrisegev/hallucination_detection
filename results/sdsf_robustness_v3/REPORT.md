# SDSF robustness research loop v3

Decision: **ADVANCE_BEST_TO_REAL_DATA**.

The loop compares every candidate with fixed-orientation SU-PCR. Positive deltas mean higher held-out AUROC. Development selected candidates by a frozen robust utility; only those candidates were opened on disjoint validation seeds.

## Development ledger

| step | candidate | utility | primary mean | primary p05 | clean | promoted |
|---:|---|---:|---:|---:|---:|:---:|
| 1 | `sdsf_cond100` | +0.00775 | +0.00698 | +0.00222 | +0.00001 | yes |
| 2 | `sdsf_cond50` | +0.00775 | +0.00698 | +0.00222 | +0.00001 | yes |
| 3 | `sdsf_cond20` | +0.00775 | +0.00698 | +0.00222 | +0.00001 |  |
| 4 | `diag25_cond50` | +0.00726 | +0.00651 | +0.00305 | +0.00000 |  |
| 5 | `diag50_cond50` | +0.00602 | +0.00539 | +0.00273 | +0.00000 |  |
| 6 | `rho_boot_tau0.5` | +0.00805 | +0.00718 | +0.00327 | -0.00000 | yes |
| 7 | `rho_boot_tau1` | +0.00767 | +0.00689 | +0.00318 | -0.00000 |  |
| 8 | `rho_boot_tau2` | +0.00642 | +0.00588 | +0.00252 | +0.00000 |  |
| 9 | `joint_tau1_diag25` | +0.00657 | +0.00592 | +0.00281 | +0.00000 |  |
| 10 | `blend_half_joint_su` | +0.00437 | +0.00396 | +0.00196 | -0.00000 |  |

## Sealed validation

| candidate | vs SU-PCR [95% CI] | vs current SDSF [95% CI] | primary p05 | clean | decision |
|---|---:|---:|---:|---:|:---:|
| `rho_boot_tau0.5` | +0.00802 [+0.00727, +0.00874] | +0.00040 [+0.00013, +0.00070] | +0.00227 | +0.00000 | **PASS** |
| `sdsf_cond100` | +0.00762 [+0.00677, +0.00848] | +0.00000 [+0.00000, +0.00000] | +0.00144 | +0.00001 | **FAIL** |
| `sdsf_cond50` | +0.00762 [+0.00675, +0.00846] | +0.00000 [+0.00000, +0.00000] | +0.00144 | +0.00001 | **FAIL** |

## Frozen validation gates

A candidate passes only if it improves both SU-PCR and the current SDSF, both paired 95% CI lower bounds are non-negative, its 5th percentile is at least -2 AUROC points, and its clean-world mean is at least -0.5 points. Dense dependence and small-sample results are mandatory stress reports, not promotion gates.

## Interpretation boundary

Synthetic success establishes only that the stabilization mechanism works in the declared covariance worlds. It does not establish improvement on hallucination detection. A failed candidate must not be repaired using validation labels; a new hypothesis requires a new version and seed namespace.
