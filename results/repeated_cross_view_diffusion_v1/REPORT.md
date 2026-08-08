# Repeated cross-view alternating-diffusion experiment

**Decision: STOP OR REVISE CROSS-VIEW DIFFUSION.**

## Terms

- **Partition:** two disjoint feature sets that together contain the full pool.
- **Dependency block:** near-duplicate rank-correlated features that cannot be split across views.
- **Alternating diffusion:** a two-step sample transition through both view graphs.
- **Consensus graph:** the average alternating graph across frozen partitions, reduced back to k neighbours.
- **pp:** AUROC percentage points.

## Headline

| method | macro AUROC | change vs IU | family change | family interval | wins/losses | worst |
|---|---:|---:|---:|---:|---:|---:|
| deployed_upcr | 0.7735 | -0.053pp | -0.016pp | [-0.150,+0.139] | 11/13 | -1.842pp |
| iu_pcr | 0.7741 | +0.000pp | +0.000pp | [+0.000,+0.000] | 0/0 | +0.000pp |
| dufs_liu | 0.7741 | +0.008pp | +0.008pp | [-0.064,+0.096] | 13/10 | -0.317pp |
| raw_uniform_liu | 0.7739 | -0.021pp | -0.023pp | [-0.052,+0.005] | 9/14 | -0.361pp |
| atomic_random__lambda_0p1 | 0.7742 | +0.018pp | +0.008pp | [-0.017,+0.036] | 11/11 | -0.071pp |
| family_blocked__lambda_0p1 | 0.7743 | +0.019pp | +0.016pp | [-0.025,+0.060] | 11/11 | -0.267pp |
| dependency_blocked_t4__lambda_0p1 | 0.7740 | -0.008pp | -0.004pp | [-0.036,+0.028] | 10/14 | -0.179pp |
| dependency_blocked_t8__lambda_0p1 | 0.7740 | -0.010pp | -0.014pp | [-0.055,+0.025] | 10/14 | -0.178pp |
| dependency_direct__lambda_0p1 | 0.7737 | -0.033pp | -0.035pp | [-0.077,+0.010] | 8/15 | -0.222pp |
| dependency_node_permuted__lambda_0p1 | 0.7741 | +0.001pp | +0.002pp | [-0.003,+0.009] | 9/14 | -0.022pp |
| dependency_blocked_k5__lambda_0p1 | 0.7741 | +0.005pp | -0.007pp | [-0.039,+0.026] | 9/15 | -0.130pp |
| dependency_blocked_k11__lambda_0p1 | 0.7741 | +0.005pp | -0.000pp | [-0.023,+0.025] | 9/15 | -0.159pp |
| dependency_blocked__lambda_0p1 | 0.7741 | +0.004pp | -0.010pp | [-0.052,+0.029] | 10/14 | -0.133pp |

## Label-free convergence

- **atomic_random:** median graph CKA 0.650; median partition-score Spearman 1.000; median edge Jaccard 0.298.
- **dependency_blocked:** median graph CKA 0.536; median partition-score Spearman 1.000; median edge Jaccard 0.229.
- **family_blocked:** median graph CKA 0.622; median partition-score Spearman 1.000; median edge Jaccard 0.313.

## Continuation gates

| gate | observed | pass |
|---|---:|:---:|
| primary mean improvement is at least +0.20pp | 0.0042 | no |
| primary equal-family lower bound is above zero | -0.0523 | no |
| primary improves at least 14 of 24 cells | 10.0000 | no |
| primary worst loss is no worse than -2pp | -0.1326 | yes |
| primary beats atomic-random splitting | -0.0134 | no |
| primary beats family-blocked splitting | -0.0150 | no |
| primary beats direct arithmetic view averaging | 0.0369 | yes |
| primary beats the node-permuted graph | 0.0031 | yes |
| primary beats frozen DUFS-LIU | -0.0034 | no |
| median partition-to-consensus graph CKA is at least 0.50 | 0.5361 | yes |
| median T=8 versus T=16 score Spearman is at least 0.95 | 1.0000 | yes |

## Interpretation

The registered common-manifold mechanism did not pass every gate. Stable partitions alone must not be interpreted as correctness evidence. Use the random, family, direct-average, and permuted controls to identify whether the failure is duplicate leakage, missing cross-view signal, ordinary shrinkage, or stable target-irrelevant geometry.

## Audit

Run fingerprint: `75688ad173434232a516d06a9646bd9cd6238ed55581e1808ce47abf0f495923`. All registered sources, inputs, reference scores, and new score artifacts were verified before correctness labels were opened. The 24 cells are retrospective development evidence.
