# SU-aware pooled-graph adaptation — canonical conclusion (2026-08-23)

## Scope

This note records the bounded conclusion of the isolated SU-PCR adaptation
sidecar. It does not modify the frozen IU-PCR baseline, Family-NRM, or the
canonical pooled graph-roughness lineage.

The experiment asked whether SU-style covariance cleaning or the SU estimate
of the feature--target correlation could add value to the family-residual
pooled graph direction. The primary comparison held the canonical graph
construction and selector fixed and changed only the covariance/correlation
component under study.

## Protocol repair

The first sidecar (`V1`) searched graph topology and neighborhood size and used
a max-mean selector. It did not exactly reproduce the canonical fixed
union-kNN `k=7`, one-standard-error/tail-guard result and produced optimistic
estimates (+0.452pp to +0.502pp versus IU). It is retained only as a sensitivity
analysis and must not be cited as the primary result.

Conservative `V2` fixed union-kNN `k=7` and used the canonical selector. It
reproduced the current observed-IU pooled graph result exactly.

## Development result

All numbers below are equal-dataset-family AUROC changes in percentage points
relative to IU-PCR under nested leave-dataset-family-out selection.

| arm | delta vs IU | 95% family-bootstrap CI | positive families |
|---|---:|---:|---:|
| current observed-IU pooled graph | +0.251 | [+0.027,+0.458] | 6/8 |
| IU + all-sparse covariance cleaning | +0.262 | [+0.053,+0.450] | 7/8 |
| IU + cross-family sparse cleaning (prespecified) | +0.260 | [+0.045,+0.454] | 6/8 |
| IU + shared cross-family cleaning | +0.245 | [+0.031,+0.443] | 6/8 |
| SU-rho + observed covariance | -0.198 | [-1.198,+0.408] | 6/8 |
| SU-rho + cross-family cleaning | -0.077 | [-0.881,+0.433] | 6/8 |

The paired primary contrast, cross-family cleaning minus the current method,
was only **+0.009pp**, with 95% family-bootstrap CI
**[-0.012,+0.037]** and 4/8 positive families. Cross-family cleaning without
the graph contributed +0.005pp; the graph increment was +0.255pp. Thus almost
all of the useful signal remains attributable to the pooled graph correction,
not to SU covariance cleaning.

The SU estimate of rho was the consistently harmful intervention. The data do
not support replacing the IU rho estimate with SU rho in this pipeline.

## Frozen retrospective transfer

The prespecified cross-family-cleaned arm and the current arm were frozen on
the original development families before scoring the transfer labels. These
surfaces were historically known, so this is a retrospective stress test, not
prospective confirmation.

| domain | current vs IU | cross-clean vs IU | cross-clean minus current |
|---|---:|---:|---:|
| ProcessBench Llama | +0.588pp | +0.560pp | -0.028pp |
| ProcessBench Qwen | +0.137pp | +0.121pp | -0.015pp |
| SemGrad | +0.257pp | +0.251pp | -0.006pp |

Cross-family cleaning was worse than the current method in 12/14 external
cells. It therefore failed the incremental-value/generalization test.

## Decision and interpretation boundary

Decision: **`CLOSE_SU_ADAPTATION_NO_INCREMENTAL_VALUE`**.

- Retain the observed-IU pooled graph direction as the candidate under study.
- Do not promote any SU-rho or SU-cleaned variant.
- Do not spend new PRMBench/HLE confirmation budget on these SU variants.
- Reopen SU only if a new, independently justified block/hierarchical model
  makes a prediction that differs from the tested element-wise decomposition.
- The result rejects incremental value from this SU adaptation; it does not
  reject the pooled graph mechanism itself.

The pooled graph result is still discovery-level. Its registered mechanism
controls do not uniquely attribute the gain to residualization: the
contribution graph reached +0.299pp, and the complete graph-attribution gate
failed, although the real graph beat the mean of 20 node permutations by
+0.411pp (randomization p=0.0476). These controls must accompany any claim
about why the graph works.

## Canonical artifacts

- `results/su_pooled_graph_adaptation_conservative_v2/REPORT.md`
- `results/su_pooled_graph_adaptation_transfer_v1/REPORT.md`
- `results/pooled_graph_roughness_direction_v2/controls/REPORT.md`
- `docs/experiments/SU_POOLED_GRAPH_ADAPTATION_CONSERVATIVE_V2.md`
