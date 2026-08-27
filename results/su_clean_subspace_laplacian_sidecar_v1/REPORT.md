# SU clean-subspace + DUFS-Laplacian sidecar

Retrospective mechanism study on the frozen 24-cell mixed-v2 roster. It does not change the canonical baseline.

## Formulation boundary

Literal `L` has rank at most two and need not be positive semidefinite, so it cannot define an identifiable K>2 PCR basis. The full-K clean-subspace arms therefore use `C-S` with observed diagonal preserved; the covariance-clean arm PSD-projects it. Literal-L K=1,2 arms are retained as diagnostics only.

## Baselines

| method | cell AUROC | family AUROC | family delta vs IU (pp) |
|---|---:|---:|---:|
| iu_pcr | 0.776087 | 0.742347 | +0.0000 |
| dufs_liu | 0.776560 | 0.743044 | +0.0697 |
| su_pcr | 0.771381 | 0.739274 | -0.3073 |

## Retrospective best surface points

| method | K | lambda | family delta vs IU (pp) | graph gain vs matched lambda=0 (pp) |
|---|---:|---:|---:|---:|
| uc_obs | 2 | 0.3 | -0.1167 | +0.1906 |
| uclean_obs | 2 | 0.1 | +0.0450 | +0.0674 |
| uclean_clean | 2 | 0.1 | +0.0637 | +0.0631 |
| ul_obs | 1 | 0 | -0.2403 | +0.0000 |
| ul_clean | 1 | 0 | -0.2403 | +0.0000 |
| ul_lraw | 2 | 0.3 | +0.0770 | +0.1015 |

## LOFO selection

| method | family delta vs IU (pp) [95% CI] | wins |
|---|---:|---:|
| uc_obs | -0.5252 [-1.2794, +0.1086] | 3/8 |
| uclean_obs | -0.1465 [-0.4431, +0.1404] | 3/8 |
| uclean_clean | -0.0965 [-0.3501, +0.1548] | 4/8 |
| ul_obs | -0.5809 [-1.4052, +0.0777] | 3/8 |
| ul_clean | -0.8590 [-2.2750, +0.0910] | 3/8 |
| ul_lraw | +0.0238 [-0.1080, +0.1580] | 3/8 |

## Diagnostics

- SU decompositions converged in 24/24 cells.
- Literal L was indefinite in 2/24 cells and has K>2 identifiable = false by construction.
- Raw C-S had negative eigenvalues in 7/24 cells; median PSD rank 28.5.
- `uc_obs, K=2, lambda=0` reproduces SU-PCR to numerical precision in every cell.
