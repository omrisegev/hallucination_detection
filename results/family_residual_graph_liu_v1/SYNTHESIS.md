# Family-residual graph LIU v1 — synthesis

## Decision

`CLOSE_FAMILY_RESIDUAL_GRAPH_LIU_NO_TRANSFER_VALUE`

The family residuals contain a small amount of answer-local neighbourhood
information, but neither the registered graph construction nor the registered
Laplacian actuators turn it into a stable correctness improvement over
ordinary IU-PCR. The nested development estimate is indistinguishable from
zero, and the frozen finalist is harmful on both external-to-development
stress tests. This direction is not promoted.

## What was tested

Ordinary mixed-v2 IU-PCR was decomposed into six frozen provenance-family
contributions. Each contribution was standardized and residualized against
the IU score. The resulting per-answer vector `R` entered a block-balanced
sample metric together with optional DUFS and IU-score coordinates. No family
covariance eigenvector was selected and no semantic meaning was assigned to an
eigenvalue near one.

The label-free fit froze 1,766 scores in each of 24 cells before any labels
were opened. The selectable grid contained 1,152 graph/readout candidates.
Nested leave-one-dataset-family-out selection evaluated the complete
hyperparameter procedure across eight original dataset families. The final
all-development selector chose a pure residual graph (`eta=1`, `beta=0`),
`k=7`, contribution-space actuation, `lambda=.03`, and correction cap `1/G`.
Thus the most direct version of the proposed reframe—not a diluted hybrid—was
actually tested.

## Main results

| surface | estimator | AUROC change vs IU | uncertainty / interpretation |
|---|---|---:|---|
| original 8 families | nested HPO procedure | +0.010pp | 95% family bootstrap [-0.027,+0.061]; 3/8 positive |
| original 8 families | fixed default | +0.006pp | [-0.027,+0.040] |
| original 8 families | historical DUFS-LIU | +0.069pp | comparator only |
| original 8 families | frozen Family-NRM | +0.277pp | reference gain |
| PRMBench/Qwen3-8B | frozen finalist | -0.192pp | source-group CI [-0.233,-0.150] |
| PRMBench/Qwen3-8B | fixed default | +0.029pp | CI [-0.009,+0.067] |
| PRMBench/Qwen3-8B | frozen Family-NRM | +0.460pp | reference gain |
| HLE/Qwen2.5-72B | frozen finalist | -0.476pp | stratified CI [-0.789,-0.163] |
| HLE/Qwen2.5-72B | fixed default | -0.637pp | secondary, interim judge |
| HLE/Qwen2.5-72B | frozen Family-NRM | +0.345pp | CI crosses zero; reference only |

The nested procedure recovered only 3.6% of Family-NRM's original point gain.
Its `D_0.5` was -0.129pp and its `D_0.3` was -0.073pp, so both the half-recovery
target and continuation floor failed. On PRMBench the finalist recovered
-41.7% of NRM's point gain; on HLE, -137.7%.

## Mechanism isolation

When the all-development finalist is evaluated descriptively on the same 23
cells used to select it, the pure residual graph plus contribution-space
actuator gains +0.043pp. The matched U2 actuator gains only +0.001pp, so the
small in-sample effect requires the family contribution-space actuator. But
the same readout on the ordinary DUFS graph is +0.010pp, and the nested
selected arm is actually -0.031pp below its matched DUFS-readout arm.

The fixed controls show that the residual graph is not entirely arbitrary:

| fixed control | equal-family change vs IU |
|---|---:|
| pure residual graph | +0.043pp |
| raw contribution (`H`) graph | +0.013pp |
| row-permuted residual graph | +0.005pp |
| node-permuted graph | +0.003pp |
| random cardinality-matched families | +0.011pp |
| direct score diffusion | -0.003pp |
| length-only graph | +0.049pp |

Residualization therefore improves the descriptive graph effect relative to
`H` and the permutation controls. That is evidence for weak local structure,
not for a transferable correctness mechanism: length-only geometry is at
least as strong in development, only four of eight families benefit from the
residual graph, nested selection collapses to near zero, and both frozen
external transfers are negative.

## Claim boundary

The experiment supports: “family residuals encode some non-random local
neighbourhood structure.” It does not support: “this structure provides
incremental, generalizable AUROC over IU-PCR through Laplacian smoothing.”
The latter is the user-facing and research-relevant target, and it failed.

PRMBench and HLE were not used for graph-LIU HPO, but their labels had been
opened historically for other methods. They are frozen retrospective transfer
tests, not prospective confirmation. Given their negative results, acquiring
a new sealed dataset solely to confirm this candidate is not justified.

## Artifacts

- `REPORT.md`: nested development result and promotion gates.
- `nested_outer.csv`: held-family selection trace.
- `FROZEN_SELECTION.json`: final configuration and selection hash.
- `controls/REPORT.md`: fixed mechanism controls.
- `../family_residual_graph_liu_prmbench_v1/REPORT.md`: PRMBench transfer.
- `../family_residual_graph_liu_hle_v1/REPORT.md`: HLE stress test.
