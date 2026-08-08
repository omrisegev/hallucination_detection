# Frozen 24-cell view-fusion benchmark: conclusion

**Date:** 2026-08-07
**Status:** completed, frozen negative result
**Decision:** do not promote CA-SpecRaGE or LOCO micro-views

This note is the post-run research record. The score-producing source files and
the registered protocol are intentionally unchanged because their SHA-256
hashes are part of the immutable run manifest.

## Question

The old SpecRaGE experiment used six manually defined feature families. A
read-only diagnosis showed that these families create different graphs, but
family agreement does not predict which family helps fusion. We therefore
asked:

> Is local reliability represented better by manual feature families,
> individual features, or small groups learned from their effect on IU-PCR?

The experiment also tested a stronger claim:

> Does sample-specific cross-view reliability improve IU-PCR, or is a global
> weighting sufficient?

## Leakage boundary

All 24 cells used the same samples, `fixed_stable_v1` features, fixed feature
directions, two-component IU-PCR head, seeds, and Laplacian path. The fitting
program had no label argument and never read a label array.

It first wrote and hashed every score and diagnostic file. The separate report
program verified all hashes, exclusively created
`SCORE_FREEZE_MANIFEST.json`, and only then read labels. No score file contains
labels, targets, correctness values, or observed metrics.

These 24 cells are retrospective development data. They are not an unseen
confirmation set because earlier feature and method work already used them.

## How to read the results

- A **cell** is one dataset/model pair.
- **AUROC** measures ranking quality: 0.5 is random and 1.0 is perfect.
- One **percentage point (pp)** is 0.01 AUROC. A change from 0.770 to 0.775 is
  +0.5pp.
- A **cell-macro** value gives every cell equal weight, regardless of its
  sample count.
- A **paired interval** resamples cells or dataset families while preserving
  the candidate and baseline result from the same unit. An interval containing
  zero does not support a consistent gain.
- **Wins / ties / losses** count cells where the AUROC change is positive,
  effectively zero, or negative.
- **Adjusted Rand index (ARI)** measures whether two clusterings agree. Zero
  is chance-level agreement and one is identical grouping.

## Compared view definitions

### Manual views

The six historical semantic/provenance families. They are a baseline, not a
learned part of the method.

### Balanced atomic views

Each feature is one view. Features that belong to the same learned duplicate
group divide that group's prior mass, so repeated variants cannot gain more
total influence merely because there are several of them.

### LOCO micro-views

For feature `j`, construct a one-feature graph with Laplacian `L_j` and compute

\[
R_j=F L_jF^\top/n,
\qquad
S_j=\frac{U^\top R_jU}{\operatorname{tr}(U^\top R_jU)}.
\]

Features are compared using the basis-invariant distance

\[
d_{jk}=\|S_j-S_k\|_F/\sqrt{2}.
\]

For each held cell, the partition was learned from the other 23 cells only.
Candidate cluster counts 3--8 were compared with the same bootstrap
perturbations. Selection used only silhouette, adjusted-Rand stability,
singleton rate, and group-size imbalance.

## Compared graph interfaces and controls

For every view definition, the benchmark fitted:

- the adapted SpecRaGE embedding graph;
- the CA-trained embedding graph;
- the CA sample-specific alpha graph;
- an end-to-end uniform-fusion embedding;
- the exact prior-alpha graph made from the same CA base graphs;
- a global-alpha control;
- a sample-permuted-alpha control.

The exact prior, global, and permuted controls are necessary. They separate
the value of the base graphs and marginal weights from the value of assigning
different weights to different samples.

## Frozen headline results

| method | cell-macro AUROC | change versus IU-PCR |
|---|---:|---:|
| Deployed U-PCR | 0.7735 | -0.054 pp |
| IU-PCR | 0.7741 | reference |
| DUFS-LIU, lambda 0.1 | 0.7741 | +0.008 pp |
| CA-alpha, manual views, lambda 10 | 0.7721 | -0.193 pp |
| CA-alpha, balanced atomic views, lambda 10 | 0.7743 | +0.023 pp |
| CA-alpha, LOCO micro-views, lambda 10 | 0.7704 | -0.363 pp |

Balanced atomic views were the safest new schema, but their confidence
interval crossed zero and their 24-cell record was 11 wins, 1 tie, and 12
losses versus IU-PCR. This is a tie, not an improvement.

LOCO micro-views won only 5 of 24 cells, lost 19, and had a worst loss of
-2.855 pp versus IU-PCR. They passed 0 of 8 preregistered promotion gates.

## The controls identify the failed mechanism

At lambda 10, sample-specific alpha minus its controls was:

| schema | versus prior alpha | versus global alpha | versus permuted alpha |
|---|---:|---:|---:|
| manual | -0.075 pp | -0.070 pp | -0.063 pp |
| atomic | +0.095 pp | -0.030 pp | -0.019 pp |
| micro | -0.010 pp | -0.059 pp | -0.056 pp |

The atomic learner improved over its prior graph, but global and permuted alpha
were slightly better than sample-specific alpha. Global and permuted controls
were nearly identical. Therefore the experiment found no value in deciding
which sample receives which weight. Any small atomic effect comes from the
base geometry or marginal/global weighting.

The full lambda path does not hide a useful setting. Atomic sample alpha peaks
at only +0.056 pp at lambda 1; atomic global alpha peaks at +0.072 pp. Manual
peaks near zero. Micro's best value is lambda 0, which is ordinary IU-PCR.
These are sensitivity results, not settings to promote after seeing labels.

## The micro-view result is informative

Micro partitions were not noisy:

- every cell selected three groups;
- bootstrap adjusted-Rand scores ranged from 0.84 to 0.94;
- no selected partition contained a singleton;
- projected-roughness distances were larger for micro than for manual or
  atomic views.

The method therefore succeeded at finding stable groups that affect the
IU-PCR subspace differently. Those groups were still harmful for correctness.
This separates two ideas that had previously been conflated:

1. a stable fusion geometry exists;
2. that geometry is relevant to hallucination correctness.

Only the first is supported.

## Numerical and computational checks

The headline alpha graphs did not collapse: all cell/schema graphs had one
connected component, acceptable degree tails, and projected condition numbers
between about 1.1 and 3.8. Alpha moved away from its prior and followed the
agreement target. This is not the old uniform-alpha smoke failure.

There were 135 unavailable algebraic-connectivity estimates across eight
cells. All were ARPACK diagnostics for secondary embedding-Y graphs, repeated
along their lambda paths. They are stored as JSON `null` with exact paths. They
do not affect scores, AUROC, alpha diagnostics, graph component counts, or the
headline conclusion.

The full fit took about 75 minutes. Atomic neural fitting dominated the cost.
The extra complexity produced no measurable accuracy gain.

## Independent review

An independent post-run review classified this as a valid frozen negative
result. It agreed that neither CA-SpecRaGE nor LOCO micro-views should be
promoted. Its main recommendation was to remove clustering and sample-specific
alpha, and consider global atomic operators only after testing whether a
label-free diagnostic predicts operator usefulness. That recommendation is a
research proposal, not evidence that the next method will work; the Phase-0
audit below is designed to falsify it cheaply.

## Claims after this experiment

### Supported

- Manual semantic families are not necessary. Balanced atomic views have a
  higher cell-macro result inside this interface, but family-level uncertainty
  prevents a general claim that they are better on new families.
- Atomic feature granularity avoids some destructive averaging.
- Stable micro-view grouping can be learned without labels.
- The frozen evaluation and controls can identify whether local alpha matters.

### Rejected for the current method

- Cross-view agreement provides useful sample-local hallucination reliability.
- Projected-roughness LOCO groups are useful views for CA-alpha fusion.
- CA-SpecRaGE improves real-data IU-PCR at the synthetic transfer setting.
- A different value on the observed lambda path is enough to rescue the claim.

### Still open

- A simple global weighting of atomic roughness operators may be safer than
  manual grouping or sample-specific alpha.
- It is not known whether any label-free statistic can identify which atomic
  operator helps correctness.
- It is not known whether a useful graph regularizer exists beyond generic
  trace-matched ridge shrinkage.

## Decision

Stop developing sample-specific CA-SpecRaGE and do not tune the micro-view
partition further. Keep deployed U-PCR, IU-PCR, and DUFS-LIU as baselines.

The next experiment must test the missing premise before building a new model:

> Does a label-free stability statistic for an atomic roughness operator
> predict that operator's held-out contribution to IU-PCR?

If the answer is no, close this graph-regularization line instead of trying a
new grouping rule.

## Artifacts

- Full report: `results/frozen_24cell_benchmark/REPORT.md`
- Immutable definition: `results/frozen_24cell_benchmark/RUN_DEFINITION.json`
- Score freeze: `results/frozen_24cell_benchmark/SCORE_FREEZE_MANIFEST.json`
- Headline table: `results/frozen_24cell_benchmark/headline_summary.csv`
- Paired table: `results/frozen_24cell_benchmark/paired_comparisons.csv`
- Diagnostics and histories: `results/frozen_24cell_benchmark/diagnostics.csv`
  and `training_history.csv`
- Figures: `results/frozen_24cell_benchmark/figures/`
- Independent reviewer instructions:
  `results/frozen_24cell_benchmark/REVIEWER_GUIDE.md`

The interrupted pre-fix run is preserved separately in
`results/frozen_24cell_benchmark_failed_nan_diagnostic/`. It is an execution
audit, not a scientific result, and should not be committed with the final
benchmark unless that failure record is explicitly wanted.
