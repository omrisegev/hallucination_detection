# Family-residual graph LIU v2 — audited rebuild and topology sensitivity

## Status and claim boundary

This is a post-audit retrospective rebuild. V1 exposed duplicate-coordinate
bugs in self removal and bandwidth estimation, and its graph-health filter was
added after labels were opened. V2 is a fresh score lineage with a frozen
constructor and a pre-fit topology axis. V1 artifacts are preserved but are
withdrawn from scientific interpretation.

PRMBench and HLE outcomes were already visible before V2 was specified. They
may be rerun only as labelled retrospective bug-repair sensitivities; they are
not transfer confirmation and cannot select V2.

## Representation

For each cell, ordinary mixed-v2 IU-PCR produces standardized baseline score
`b`. Its atomic feature contributions are summed inside the six frozen
provenance families and residualized linearly against `b`, producing
standardized answer-by-family coordinates `R`. No covariance eigenvector is
selected and no eigenvalue-near-one semantic claim is used.

The graph coordinates are the block-balanced metric

```text
(1-eta) DUFS + eta [beta b + (1-beta) R],
```

with each active block divided by a deterministic median non-zero squared
pair distance. The pair subsample and seed (`1729`) are identical for every
`eta/beta/topology` setting in a cell, so metric-weight contrasts cannot absorb
Monte-Carlo normalization noise.

## Audited graph axis

All constructors use the reviewed target-blind routines in
`spectral_utils/graph_topology.py`. Self is removed by row identity. Distance
ties are resolved by a unique row key after expanding the query through the
boundary tie. Bandwidth is the distance to the kth strictly-positive distinct
sample location, so duplicate multiplicity cannot collapse it to zero.

The strict bug-repair primary is:

1. symmetric self-safe union-kNN at `k in {5,7,15}`;

The separately reported topology-rescue sensitivity is:

2. adaptive union-kNN with exact directed mean 7, range `[3,25]`, scale-k 7,
   and density-rank power 8.

The primary selector is union-only. Adaptive-only and union-plus-adaptive
selectors are secondary retrospective rescue analyses, so an adaptive result
cannot silently redefine the v1 estimand after the earlier outcomes were seen.

Mutual-kNN is not selectable: the earlier topology audit found systematic
fragmentation and no utility rescue. It is computed only for the final fixed
control. Radius and diffusion graph constructors are excluded because the
prior audit found radius unhealthy and no distinct topology rescue.

Before labels open, a graph setting is eligible only if every one of the 24
development cells is symmetric, finite, has positive mean degree, isolated
fraction below 5%, and largest connected component at least 95%. This health
rule is validity filtering, not HPO.

## Readout and registered grid

The two selectable actuators are historical two-PC U2-LIU and an IU-anchored
family contribution-space correction `b + R delta`. Direct score diffusion is
diagnostic-only.

- `eta in {0,.25,.5,.75,1}`;
- `beta in {0,.25,.5,.75,1}` (`beta=.5` only at `eta=0`);
- union `k in {5,7,15}` or the one fixed adaptive topology;
- `lambda in {0,.03,.1,.3,1,3,10}`;
- contribution correction cap factor in `{.5,1,2}` times `1/G`.

Only configurations with a non-zero family-residual graph weight and
`lambda>0` may be selected. The fixed default is union, `eta=beta=.5`, `k=7`,
`lambda=.1`, contribution-space readout, and cap `1/G`.

## Fit/evaluation barrier

The fit process must not index labels. It freezes every score, diagnostic,
source hash, configuration, and a self-consistent completion manifest. The
report verifies all source and score hashes, writes the score-freeze manifest
once without later rewriting it, and only then reads labels.

## Nested selection and mechanism isolation

The original fewer-than-20-positive exclusion is retained. Nested outer
leave-one-dataset-family-out evaluation across eight families estimates the
entire HPO procedure. Inner selection maximizes equal-family AUROC change over
IU, uses a one-standard-error rule, then frozen tail-safety and complexity
tie-breaks.

Fixed controls after selection are: same graph with U2 and contribution-space
actuation, same readout on DUFS coordinates, raw contributions `H`, row-
permuted `R`, node-permuted graph, one deterministic cardinality-matched random
family partition, length-only, baseline-only, R-only, mutual-kNN, cardinality
CS-IU, and direct diffusion.

## Decision gates

Promotion requires all of:

- equal-family bootstrap lower bound above zero versus IU;
- point gain at least +0.10pp;
- at least 6/8 positive held-out families;
- worst-family loss at least -0.50pp;
- non-zero Laplacian in at least six outer folds;
- non-negative lower bound for `D_0.3 = delta_new - .3 delta_NRM`.

Half of the original Family-NRM gain is the point recovery target. AUPRC and
the graph/readout controls are secondary. A graph-mechanism claim additionally
requires improvement over matched actuation and permutation controls. Because
external outcomes are already known, V2 can close the mechanism but cannot
create a new confirmation claim.
