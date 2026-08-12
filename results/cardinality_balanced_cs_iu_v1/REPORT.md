# Cardinality-balanced contribution-space IU: selection audit

**Status:** retrospective cross-domain selection evidence; not a prospective confirmation of CB-CS-IU.

The family-cardinality rule is positive in both domains and is the current frozen non-supervised candidate. ProcessBench selected the pivot from leverage to cardinality, so a new untouched benchmark is still required.

| domain | contrast | cell delta | equal-group delta (95% CI) | W/L | worst |
|---|---|---:|---:|---:|---:|
| original_23_cell | `cardinality - iu` | +0.467pp | +0.442pp [+0.204, +0.682] | 17/6 | -0.595pp |
| original_23_cell | `leverage_balanced - iu` | +0.569pp | +0.633pp [+0.314, +0.949] | 19/4 | -0.606pp |
| original_23_cell | `dufs_liu - iu` | +0.047pp | +0.069pp [-0.031, +0.181] | 17/6 | -0.247pp |
| original_23_cell | `cardinality - leverage_balanced` | -0.103pp | -0.191pp [-0.435, +0.045] | 9/14 | -1.004pp |
| original_23_cell | `cardinality - dufs_liu` | +0.420pp | +0.374pp [+0.197, +0.536] | 18/5 | -0.629pp |
| processbench_confirmation | `cardinality - iu` | +0.895pp | +0.864pp [+0.721, +1.101] | 6/0 | +0.570pp |
| processbench_confirmation | `leverage_balanced - iu` | +0.340pp | +0.267pp [+0.045, +0.582] | 5/1 | -0.018pp |
| processbench_confirmation | `dufs_liu - iu` | +0.221pp | +0.227pp [+0.164, +0.291] | 6/0 | +0.096pp |
| processbench_confirmation | `cardinality - leverage_balanced` | +0.556pp | +0.597pp [+0.472, +0.728] | 6/0 | +0.402pp |
| processbench_confirmation | `cardinality - dufs_liu` | +0.674pp | +0.637pp [+0.510, +0.842] | 6/0 | +0.391pp |

## Interpretation

On the original cells, cardinality balancing improved equal-family AUROC by +0.442pp. On the frozen ProcessBench confirmation slice it improved equal-subset AUROC by +0.864pp and won all 6 cells.

On ProcessBench it also beat leverage balancing by +0.597pp, with interval [+0.472, +0.728]pp. This supports family multiplicity as the more transferable nuisance observable. It does not erase selection bias: this contrast motivated the pivot.

## Claim boundary

The score computation was label-free and the cardinality score had already been frozen as a control before both reports. However, promoting it to the primary method happened after report inspection. CB-CS-IU is therefore ready for a pristine confirmation, not yet prospectively confirmed.
