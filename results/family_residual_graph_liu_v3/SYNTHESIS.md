# Family-residual graph LIU v3 — canonical synthesis

## Decision

`CLOSE_FAMILY_RESIDUAL_GRAPH_LIU_NO_INCREMENTAL_VALUE_V3`

Using the six Family-NRM residual contributions as graph coordinates does not
provide a reliable improvement over ordinary IU-PCR, and changing the graph
topology does not rescue the mechanism.  The successful Family-NRM correction
cannot be recovered by this graph/Laplacian route without its original spectral
orientation rule.

V1 is withdrawn because its historical kNN builder was not self-safe under
duplicate coordinates and could collapse its bandwidth.  V2 repaired the graph
builder but stopped label-free when its every-cell 95% largest-component filter
left no eligible graph.  V3 is the canonical bug-repaired lineage: graph
validity is an eligibility requirement, while connectivity is a separate
mechanism diagnostic.

Two independent final audits found no artifact blocker. All 37 integrity checks
passed across bundle/source/config/score/diagnostic/manifest hashes, the score
files contain no labels, the freeze barrier predates label access, and the
external IU scores match the Family-NRM IU comparator to numerical precision.

## Development result

All 24 score banks and diagnostics were fitted and hashed before labels were
opened.  The primary selector was union-kNN only; adaptive-k was kept as a
separate topology sensitivity.  Nested leave-dataset-family-out evaluation over
eight families gave:

- Family-residual graph LIU vs IU-PCR: **+0.018pp AUROC**, 95% equal-family
  bootstrap **[-0.041,+0.080]pp**; 4/8 families positive.
- NRM gain recovery: **6.6%** of the frozen Family-NRM +0.277pp point gain.
  Both the 30% and 50% recovery contrasts failed.
- Adaptive-only selector: **+0.015pp** [-0.005,+0.050]pp.
- Union-plus-adaptive selector: **+0.018pp** [-0.042,+0.081]pp.
- Corrected DUFS-coordinate LIU comparator: **+0.068pp** versus IU-PCR.
- The primary arm minus the same readout on a DUFS graph: **-0.014pp**.

The frozen all-family finalist is pure residual coordinates, union-kNN with
`k=7`, historical U2 actuation, and `lambda=0.03`.  Its graph passed the
necessary connectivity diagnostic in 22/24 cells, but utility promotion failed.

## Mechanism isolation

The fixed post-selection controls are descriptive rather than nested estimates,
but they are consistent with the primary failure:

- selected residual graph: +0.001pp;
- node-permuted graph: +0.002pp;
- random family graph: +0.004pp;
- mutual-kNN graph: +0.002pp;
- direct score diffusion: -0.006pp.

The selected graph does not outperform topology-destroying controls by a
meaningful margin.  A cardinality-balanced contribution score reached +0.442pp
in development, showing that contribution-space signal can exist without the
graph, but it reversed on both external stress tests and is not a transferable
replacement.

## Retrospective external stress tests

These datasets had known Family-NRM outcomes before V3 and therefore are not
prospective confirmation.

- PRMBench: finalist **-0.0068pp** vs IU-PCR, source-group 95% CI
  **[-0.0100,-0.0037]pp**. Family-NRM was +0.460pp. Cardinality was -0.864pp.
- HLE: finalist **-0.0190pp**, stratified 95% CI **[-0.0471,+0.0084]pp**.
  Family-NRM was +0.345pp. HLE has only 68 judged-correct answers and is a weak
  stress test.

## Interpretation boundary

This result closes the tested family-residual graph construction, not every
possible use of contribution residuals.  It specifically rejects the claim
that answer-level proximity in the tested DUFS/baseline/family-residual metric,
followed by normalized-Laplacian U2 or contribution-space actuation, recovers a
stable part of Family-NRM's gain.  A future reopening requires a different
target-orientation or observation model, not another search over kNN topology
on the same coordinates.

Canonical artifacts: `RESULT.json`, `REPORT.md`, `controls/REPORT.md`, and the
two V3 retrospective external reports.
