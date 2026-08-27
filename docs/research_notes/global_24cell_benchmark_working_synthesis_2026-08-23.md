# Working synthesis: which methods belong in the next 24-cell benchmark?

**Status (2026-08-23): design note, not a result.** No new score was computed
for this note. The purpose is to decide what should be rerun under one common
contract before preparing the next advisor update.

## Short answer

The next benchmark should not start from one declared winner. It should rerun
the methods below on the same 24 dataset-model cells, rows, measurements,
orientation, and macro definition.

The main list is deliberately inclusive:

1. continuous L-SML, plus the DUFS and GroupFS selector-to-L-SML pipelines;
2. IU-PCR, deployed U-PCR, and U-PCR with `sign(rho)` orientation as the
   linear covariance references;
3. DUFS-LIU and balanced-atomic CA-SpecRaGE as the main graph methods;
4. SU-PCR as the sparse-correlated-error extension of U-PCR;
5. continuous additive DEEM (DEEM-B3) as the main nonlinear method;
6. new within-cell versions of Family-NRM and PGRD;
7. simple mean, best single measurement, GOOD_6, LOCO_5, and Unified-28 as
   interpretation references.

Family-NRM and PGRD do **not** inherently require other datasets. Their new
within-cell variants belong in the main leaderboard. They are new ablations,
however, and have not yet been run. The historical cross-dataset algorithms
remain the canonical completed experiments and must not be renamed as the new
within-cell variants.

## What the existing numbers do and do not say

Two completed 24-cell packages currently use different contracts:

| completed package | feature contract | reported aggregation | result |
|---|---|---|---|
| Frozen graph benchmark v1 | `fixed_stable_v1` | cell-macro | IU-PCR 0.77406; DUFS-LIU 0.77414; balanced-atomic CA-SpecRaGE 0.77429 |
| DEEM vs IU-PCR v1, branch `f7f7801` | each cell's full present inventory (19-30 measurements) | registered primary: equal-family macro | DEEM-B3 0.74850 vs matched IU-PCR 0.74281; it passed the not-worse rule but not the full superiority rule |

The DEEM artifact also gives a 24-cell macro of **0.78182** for B3 and
**0.77485** for its matched IU-PCR. This is encouraging, but it is not yet a
win over the 0.774 graph table: the feature inventories, IU implementation
contract, and primary macro differ. The fresh rebuild of the DEEM package is
byte-identical, so the result is reproducible within its own contract.

Residual-Graph DEEM is a different experiment. It stopped at its synthetic
specificity gate and never entered the natural-label leaderboard. It must not
be described as a failure of graph-free DEEM-B3.

The old Family-NRM and PGRD numbers are also not entries in the new main
leaderboard. Historical Family-NRM fitted its direction without labels, but
label counts affected which donor cells were admitted; under the new strict
accounting it is a legacy label-touched calibration, not a clean regime B run.
Historical PGRD pooled graph moments across donor families and used donor
labels to choose correction settings, so it is regime C. The new local
variants use only the target cell and need fresh scores.

## One method family, three selection regimes

Every Family-NRM and PGRD table will use the same axis:

| regime | data used to construct the correction | label use | role |
|---|---|---|---|
| **A. within-cell fully unsupervised** | target cell only | none | primary 24-cell leaderboard |
| **B. donor-unsupervised** | target-free donor cells may choose a stable direction, geometry, or fixed setting | no donor or target labels | secondary ablation |
| **C. donor-label selection** | target-free donor cells; donor labels may choose direction, sign, graph, or strength | supervised model selection | secondary ceiling |

A held-target-label oracle is outside A/B/C and is shown only as diagnostic
headroom. Donor data are a regime choice, not a requirement of either method
family.

The comparison is diagnostic, but a raw `B-A` or `C-A` difference is not by
itself causal because the data source also changes. The benchmark therefore
needs matched slices: keep the residuals, candidate directions, graph,
actuator, and correction strength fixed, and change only the selection source
or its label access. Separate rows may then vary graph or strength explicitly.

## New within-cell variants

For one cell, IU-PCR gives a standardized base score `b`. Its feature
contributions are summed inside six fixed measurement families, giving
`h_1,...,h_G`. Each family is residualized against the base score:

\[
u_g=z(h_g)-\operatorname{Proj}_{b}(z(h_g)),\qquad r_g=z(u_g),\qquad
R=[r_1,\ldots,r_G].
\]

### Family-NRM-WC

Compute `C_R=R^T R/n` in the target cell. Select the eigenspace whose
eigenvalue is closest to one. If several eigenvalues tie within the frozen
tolerance, project the equal-family vector into their joint eigenspace; this
avoids choosing an arbitrary eigenvector basis. The same equal-family vector
fixes the sign. The correction is standardized and limited to standard
deviation `1/G`:

\[
S=\operatorname{eigspace}_{\min|\lambda-1|}(C_R),\qquad
v_*=\frac{SS^T\mathbf 1}{\|SS^T\mathbf 1\|},\qquad
s=b+\frac{Rv_*}{G\,\operatorname{sd}(Rv_*)}.
\]

This is the direct local version of the original Family-NRM rule. It is not a
claim that the eigenvalue-near-one mode is correctness; the benchmark tests
that assumption.

### PGRD-WC

Build the fixed duplicate-safe `k=7` answer graph in `R`, with normalized
Laplacian `L`. First compute `A_0=R^TLR/n` and `c_0=R^TLb/n`, then use the same
trace scaling as the existing PGRD primitive:

\[
\alpha=G/\operatorname{tr}(A_0),\qquad A=\alpha A_0,\qquad
c=\alpha c_0,\qquad d=-c,\qquad
s=b+\frac{Rd}{G\,\operatorname{sd}(Rd)}.
\]

The sign follows graph-roughness descent and the strength is fixed. The
scaling cancels in the standardized cross-only score, but it matters for
pooled and full-quadratic arms and is therefore part of the contract. The
cross-only form is primary because the historical quadratic preconditioner
added about 0.006 percentage points and its `lambda` was selected with donor
labels. The full quadratic rule can remain a named sensitivity arm.

Both methods fall back exactly to IU-PCR if fewer than three families are present
or the correction is numerically constant. Every missing-family, tie, graph,
normalization, sign, and fallback rule must be frozen before labels are read.

## What should be compared after the rerun

The main output is cell-macro AUROC over all 24 cells, followed by family-macro,
QA, math, correctness-positive AUPRC, wins/losses, worst-cell change, and paired
uncertainty versus the exact matched IU-PCR score. Every score is oriented so
higher means more likely correct. A hallucination-positive AUPRC, if useful,
must be a separately named metric rather than mixed into this table. A full
per-cell table and heatmap are required; the macro alone can hide opposite
behavior across QA and mathematics.

Published methods should appear beside our scores only at the dataset-model
cell level, with access and protocol marked. A published number on different
generated answers is an external anchor, not a paired comparison. The exact-row
subset should be shown separately.

Localization, prefix prediction, and stopping remain separate benchmark lanes.
Their metrics are not averaged into the global 24-cell leaderboard. The same
method roster can be projected into those tasks only when its score is causal
at the required token or step boundary.

The complete population and task inventory is now in
`docs/experiments/MULTI_POPULATION_METHOD_BENCHMARK_V1.md`. It also covers
external response transfer, RAG, white-box access, repeated generations,
published comparisons, and negative-scope datasets. These lanes share a
method registry but not one cross-task macro.

## Decisions still needed before implementation

1. Choose one primary feature contract for every method: the full present
   inventory (my current recommendation because it contains every available
   measurement and reproduces the DEEM input) or `fixed_stable_v1`. The other
   may be a sensitivity lane, but the two must not be mixed in one ranking.
2. Freeze the exact IU-PCR solver and orientation used by every residual method.
3. Confirm the local Family-NRM tie/missing-family rules and the PGRD graph
   constructor and trust value.
4. Decide whether continuous L-SML/GOOD_6 is only a label-informed reference or
   also a rerun arm.
5. Freeze the main and secondary tables before implementing or opening labels.

The detailed draft protocol is
`docs/experiments/GLOBAL_24CELL_METHOD_BENCHMARK_V2.md`; the machine-readable
soft roster is `configs/global_24cell_method_benchmark_v2_registry.csv`.
`configs/global_24cell_method_benchmark_v2_run_registry_schema.csv` defines
the provenance fields that must be filled and frozen before any score can enter
the aligned table.

## Evidence used for this synthesis

- fixed-stable IU/DUFS/CA results:
  `results/frozen_24cell_benchmark/headline_summary.csv`;
- DEEM-B3 decision, per-cell metrics, and rebuild evidence on commit `f7f7801`:
  `results/deem_vs_iupcr_24cell_v1/evaluation/B199/DECISION.json`,
  `PER_CELL_METRICS.csv`, and `rebuild/REBUILD_VERIFICATION.json`;
- historical Family-NRM:
  `results/neutral_residual_mode_cs_iu_v1/REPORT.md`;
- historical PGRD and selection audit:
  `results/pooled_graph_roughness_direction_v2/SYNTHESIS.md` and
  `results/graph_geometry_selection_research_v1/FINAL_REPORT.md`.
