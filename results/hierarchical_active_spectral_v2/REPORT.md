# Hierarchical and active spectral correction v2

Decision: **STOP_AND_REVISE**.

The real replay is leave-one-family-out: no target-family labels enter the donor head. Active and uniform arms use identical target-label budgets.

## Registered gates

| gate | observed | rule | result |
|---|---:|---:|:---:|
| `real_mean_vs_upcr` | -0.001873 | >= 0.010000 | **FAIL** |
| `real_ci_low_vs_upcr` | -0.004076 | > 0.000000 | **FAIL** |
| `real_vs_local_uniform` | -0.001390 | >= 0.000000 | **FAIL** |
| `real_active_vs_hybrid_uniform` | 0.007794 | >= 0.000000 | **PASS** |
| `real_qa_vs_upcr` | -0.003147 | >= -0.005000 | **PASS** |
| `real_math_vs_upcr` | -0.001109 | >= -0.005000 | **PASS** |
| `real_catastrophic_losses` | 0.000000 | <= 2.000000 | **PASS** |
| `synthetic_shared_mean` | 0.476320 | >= 0.010000 | **PASS** |
| `synthetic_shared_ci_low` | 0.467986 | > 0.000000 | **PASS** |
| `synthetic_sufficient_no_harm` | -0.000619 | >= -0.005000 | **PASS** |
| `synthetic_family_shift_no_harm` | -0.004349 | >= -0.010000 | **PASS** |
| `synthetic_shared_active_vs_uniform` | 0.001848 | >= 0.000000 | **PASS** |

## Real-cell learning curves

Cell-macro held-out AUROC after averaging split repetitions.

| target labels | U-PCR | local uniform | local active | pooled LOFO | hybrid uniform | hybrid active |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7734 | 0.7734 | 0.7734 | 0.7590 | 0.7590 | 0.7590 |
| 5 | 0.7734 | 0.7736 | 0.7714 | 0.7590 | 0.7606 | 0.7674 |
| 10 | 0.7734 | 0.7733 | 0.7709 | 0.7590 | 0.7618 | 0.7697 |
| 20 | 0.7734 | 0.7729 | 0.7716 | 0.7590 | 0.7637 | 0.7715 |
| 40 | 0.7734 | 0.7729 | 0.7726 | 0.7590 | 0.7664 | 0.7723 |
| 80 | 0.7734 | 0.7731 | 0.7741 | 0.7590 | 0.7692 | 0.7729 |

## Mechanism decomposition at 20 target labels

| mechanism | contrast | mean [95% CI] | W/L |
|---|---|---:|---:|
| local acquisition | `local_uniform2` -> `local_active2` | -0.13pp [-0.30, +0.04] | 10/14 |
| LOFO transfer | `local_uniform2` -> `hybrid_domain_uniform` | -0.92pp [-1.30, -0.59] | 1/23 |
| hybrid acquisition | `hybrid_domain_uniform` -> `hybrid_domain_active` | +0.78pp [+0.53, +1.08] | 23/1 |
| combined candidate | `upcr` -> `hybrid_domain_active` | -0.19pp [-0.41, +0.02] | 8/16 |
| pooled, no target labels | `upcr` -> `pooled_domain_lofo` | -1.44pp [-2.01, -0.92] | 1/23 |
| all-domain pooling | `upcr` -> `pooled_all_lofo` | -0.96pp [-1.36, -0.57] | 3/21 |

## Synthetic transfer boundary at 20 target labels

| world | pooled LOFO vs U-PCR | hybrid active vs U-PCR | active vs uniform |
|---|---:|---:|---:|
| `upcr_sufficient` | -0.39pp | -0.06pp | +0.16pp |
| `shared_correction` | +47.28pp | +47.63pp | +0.18pp |
| `family_shift` | -0.43pp | -0.08pp | +0.17pp |

## Acquisition validity at 20 labels

| policy | selected sets containing both classes |
|---|---:|
| `local_controlled2` | 100.0% |
| `local_uniform2` | 97.3% |
| `local_active2` | 97.1% |
| `hybrid_domain_uniform` | 97.3% |
| `hybrid_domain_active` | 93.5% |

## Protocol notes

- Feature schema: `confidence-orientation-v1`; common named features: 16.
- Real bundle validity macro: `0.773527902891`.
- Repetitions: 20 real and 20 per synthetic meta-world.
- Donor acquisition: 20 uniform labels per eligible donor cell.
- Donor cost is historical supervision and is not equal to the target-only label budget; only active-vs-uniform contrasts are equal-cost acquisition comparisons.
- Test labels are read only after every target score is frozen.
