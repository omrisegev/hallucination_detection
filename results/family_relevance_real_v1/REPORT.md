# Graph-coupled family relevance diagnostic

**Decision: STOP BEFORE LEARNED MIXTURE.**

## Terms

- **Family gate:** a sample-specific weight shared by related features.
- **Family graph:** prior knowledge about which measurement families are related.
- **Conditional headroom:** the optimistic gain from choosing a different fixed family expert in each frozen context stratum.
- **pp:** AUROC percentage points.

## Synthetic boundary

The selected path improved IU-PCR by +0.773pp with 20/20 wins when inactive family members had independent noise. It lost 9.272pp with 0/20 wins when inactive members shared a coherent nuisance. The gate can detect inconsistency; it cannot detect a consistently wrong family.

## Real-data headline

| method | macro AUROC | change vs IU | family change | wins/losses | worst |
|---|---:|---:|---:|---:|---:|
| deployed_upcr | 0.7735 | -0.053pp | -0.016pp | 11/13 | -1.842pp |
| iu_pcr | 0.7741 | +0.000pp | +0.000pp | 0/0 | +0.000pp |
| dufs_liu | 0.7741 | +0.008pp | +0.008pp | 13/10 | -0.317pp |
| manual_graph__beta_0__blend_1 | 0.7751 | +0.108pp | +0.170pp | 14/10 | -0.692pp |
| permuted_graph__beta_3__blend_1 | 0.7728 | -0.122pp | -0.075pp | 8/16 | -0.605pp |
| global_gate__beta_3__blend_1 | 0.7735 | -0.056pp | -0.032pp | 11/13 | -0.520pp |
| sample_permuted_gate__beta_3__blend_1 | 0.7735 | -0.053pp | -0.007pp | 10/14 | -0.811pp |
| manual_graph__beta_3__blend_1 | 0.7727 | -0.135pp | -0.096pp | 8/16 | -1.016pp |

## Conditional specialization

| context | valid cells | headroom | permutation p | Holm p | support |
|---|---:|---:|---:|---:|:---:|
| context_trace_length | 20 | +0.802pp | 0.5848 | 0.5848 | no |
| context_family_disagreement | 23 | +0.990pp | 0.1158 | 0.2315 | no |
| context_iu_rank | 23 | +2.833pp | 0.0020 | 0.0060 | yes |

## Continuation gates

| gate | observed | pass |
|---|---:|:---:|
| primary mean change versus IU-PCR > 0 | -0.1354 | no |
| primary equal-family lower bound > 0 | -0.2933 | no |
| primary improves at least 14 of 24 cells | 8.0000 | no |
| primary worst loss no worse than -2pp | -1.0156 | yes |
| primary beats beta=0 family gate | -0.2433 | no |
| primary beats permuted family graph | -0.0134 | no |
| primary beats global family gate | -0.0793 | no |
| primary beats sample-permuted local gate | -0.0829 | no |
| primary beats frozen DUFS-LIU | -0.1430 | no |
| at least one frozen context supports specialization | 1.0000 | yes |

## Parameter diagnosis

The best descriptive frozen path was beta=0, alpha=1, with +0.108pp. The registered primary remains beta=3, alpha=1 regardless of this result.
`beta` controls how strongly related family gates are smoothed. `alpha` controls how much the local gate replaces ordinary IU-PCR. A better post-label grid point is not a promoted method.

## Interpretation

The complete mechanism was not shown. Do not build a more flexible learned family mixture from these labels. Use the failed controls to decide whether the missing part is family prior, local routing, or target information.

## Audit

Run fingerprint: `3297076c0faa88de042ec586f91dd5288ee16ee2b54579a8fbcc0b8999e4dfa0`. All sources, inputs, frozen reference scores, and new score artifacts were verified before correctness labels were read. These 24 cells are retrospective development evidence.
