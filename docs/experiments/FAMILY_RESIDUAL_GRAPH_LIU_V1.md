# Family-residual graph LIU v1

## Status and question

This protocol is a retrospective development experiment on the original
completed-answer cells.  It asks whether the manually registered provenance
families that made Family-NRM useful also define a better answer-by-answer
graph for Laplacian regularization, without selecting a covariance eigenvector
or assigning semantic meaning to an eigenvalue near one.

The primary comparator is ordinary mixed-v2 IU-PCR.  Frozen Family-NRM and the
historical mixed-v2 DUFS-LIU score are comparators; neither may select a graph,
an actuation space, or a hyperparameter.

## Label-free representation and fit boundary

For each cell, ordinary IU-PCR produces score `b`.  Its feature contributions
are summed inside the six frozen provenance families to form `H`, and every
standardized contribution is residualized against standardized `b`, yielding
`R`.  These operations use no correctness label.

DUFS is fitted once on the mixed-v2 atomic features.  It is not refitted on
`R`.  For each sample pair, the graph metric is

```text
d² = (1-eta) d²_DUFS + eta [beta d²_b + (1-beta) d²_R],
```

where every block is divided by a deterministic estimate of its median
non-zero squared pair distance.  A self-safe symmetric self-tuning union-kNN
graph is built from the resulting coordinates.

The fit command cannot access `__labels`.  It writes every registered score,
diagnostic, configuration, and hash before the report command reads labels.

## Factorial mechanisms

Two deployable actuation spaces are compared on matched graphs:

1. historical U2-LIU, whose weights remain in ordinary IU-PCR's top-two
   covariance eigenspace;
2. contribution-space LIU, whose score is `b + R delta` and whose anchored
   closed-form correction minimizes family-residual graph roughness plus an
   identity prior on `delta`.

Sample-space diffusion is computed only for the frozen finalist as a
graph-quality diagnostic and cannot be selected as the deployable candidate.

The registered grid is:

- `eta, beta in {0,.25,.5,.75,1}` (`beta=.5` only when `eta=0`);
- `k in {5,7,15}`;
- `lambda in {0,.03,.1,.3,1,3,10}`;
- contribution correction SD caps `{1/(2G),1/G,2/G}`.

The fixed-default candidate is `eta=beta=.5`, `k=7`, `lambda=.1`, cap `1/G`.

## Hyperparameter selection

The report excludes the same fewer-than-20-positive cell as Family-NRM and
uses the remaining original-23 cells in eight dataset-family blocks.  Every
outer leave-one-dataset-family-out fold selects its configuration using only
the other seven families.  Selection maximizes equal-family AUROC change over
IU, applies a one-standard-error rule, and then uses frozen tail-safety and
complexity tie-breaks.  Only the held-family score enters the estimate of the
selection procedure.

This nested meta-selection uses development correctness labels.  Therefore it
is distinct from the fixed-default label-free line.  Inside a target cell, all
fits remain label-free.

## Required attribution and controls

After the development finalist is frozen, matched controls must distinguish:

- graph-only (`hybrid graph x U2`);
- actuator-only (`DUFS graph x contribution-space`);
- full interaction (`hybrid graph x contribution-space`);
- standardized contributions `H` instead of residuals `R`;
- row-permuted `R`, node-permuted graph, length-only graph, and a
  cardinality-matched random family partition;
- exact `lambda=0` identity, projected ridge, historical DUFS-LIU,
  cardinality CS-IU, and frozen Family-NRM;
- sample-space diffusion as a non-deployable graph ceiling.

A graph-mechanism claim requires improvement over the matched actuation arm
and the row/node-permuted controls.  Improvement that survives only as generic
ridge or diffusion is not called family-residual graph identification.

## Evaluation and success ladder

The primary metric is equal-dataset-family AUROC change versus IU-PCR with a
paired family bootstrap.  AUPRC, wins/losses, worst cell/family, graph health,
rank displacement, and target/length smoothness are secondary.

Development promotion requires a positive paired lower bound versus IU, at
least `+0.10pp` point gain, at least six of eight positive dataset families,
worst-family loss no worse than `-0.50pp`, and a non-zero Laplacian selected in
at least six outer folds.  Recovery of Family-NRM is reported using
`D_q = delta_candidate - q * delta_NRM`: 50% is the point target and a
non-negative lower bound for `q=.3` is the continuation floor.

All currently available ProcessBench, SemGrad, PRMBench, HLE, AQuA, CoQA, and
RAGTruth labels have been opened historically.  They can supply a frozen
external-to-development transfer audit but not prospective confirmation.
Prospective confirmation requires at least two newly sealed dataset families,
including one unseen model family, after this development finalist is frozen.
