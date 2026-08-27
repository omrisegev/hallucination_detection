# Pooled Graph-Roughness Direction — final synthesis

**Decision: `RETROSPECTIVE_RECOVERY_WITH_DOMAIN_DEPENDENT_TRANSFER`.**

The experiment succeeded at its first objective: it reconstructed most of the
Family-NRM development gain without selecting an eigenvector whose eigenvalue
is near one. It did not satisfy the stronger objective of uniform transfer.
The frozen primary improves IU-PCR on the historical development families and
on ProcessBench, SemGrad, and HLE, but it is significantly harmful on
PRMBench. It is therefore a useful discovery and mechanism result, not a
deployable replacement for Family-NRM or an independent generalization claim.

## Method that worked

For each source cell, let `b` be the standardized IU score, `R` the standardized
six-family IU-contribution residuals, and `L` the symmetric normalized
Laplacian of a duplicate-safe union-kNN graph built on `R` (`k=7`). The fitted
cell moments are

\[
A_e = R_e^\top L_eR_e/n_e,\qquad
c_e = R_e^\top L_eb_e/n_e.
\]

Both are trace-normalized, embedded into the same six-family coordinate system,
averaged equally within dataset family, and then averaged equally across source
families. The correction direction is

\[
d_\lambda=-\lambda(I+\lambda\bar A)^{-1}\bar c,
\]

and a target cell is scored as

\[
s=b+\frac{t}{G}\frac{Rd_\lambda}{\operatorname{sd}(Rd_\lambda)}.
\]

This is graph-roughness descent: `c` is the local derivative of graph roughness
at IU, while `A` is a ridge/preconditioning term. There is no eigendecomposition
and no semantic claim attached to an eigenvalue near one. Conditional on the
donor-selected hyperparameters and direction, target scoring is label-free;
the whole procedure is nevertheless meta-supervised because donor labels select
`lambda` and trust. In transfer, the graph is a **source-side calibration
device**: the frozen six-family direction is applied to target `R`, but no
target graph is built or smoothed.

## Development reconstruction

Strict double leave-one-dataset-family-out HPO selected `lambda=.03` and
`trust=.5` in all eight outer folds.

| procedure | equal-family delta vs IU (pp) | 95% CI (pp) | positive families | worst family | Family-NRM recovery |
|---|---:|---:|---:|---:|---:|
| primary one-SE | **+0.251** | **[+0.027, +0.458]** | 6/8 | -0.277 | 90.8% |
| nested max-mean sensitivity | +0.450 | not promoted | 6/8 | -0.468 | 162.3% |
| frozen Family-NRM reference | +0.277 | — | — | — | 100% |

The eight outer directions are stable (minimum pairwise cosine 0.929; mean
0.978). The per-family effect profile also resembles Family-NRM (Pearson 0.779,
Spearman 0.857, sign agreement 7/8). The registered `D_0.30` point estimate is
+0.168pp, but its interval is [-0.004,+0.322]pp, so the complete predictive
gate narrowly fails. With only eight dataset families, the positive percentile
bootstrap interval must also be read alongside the two-sided t interval, which
crosses zero.

## What the controls isolate

| matched arm | arm vs IU (pp) | real residual graph minus arm (pp) | 95% CI (pp) |
|---|---:|---:|---:|
| DUFS graph | +0.011 | +0.240 | [-0.162,+0.620] |
| contribution graph | +0.299 | -0.048 | [-0.329,+0.216] |
| equal-cell rather than equal-family pooling | +0.127 | +0.124 | [-0.041,+0.288] |
| cross-only (`d=-c`) | +0.245 | +0.006 | **[+0.002,+0.012]** |
| family-axis permutation | +0.009 | +0.242 | [-0.061,+0.534] |

Across 20 matched node permutations, the null mean is -0.159pp. The real graph
beats that mean by +0.411pp [+.185,+.618], with the coarse 20-permutation
randomization p-value 1/21 = 0.0476. This is positive but coarse, borderline
evidence for graph-to-sample alignment on the retrospective development panel.
However, the full registered
graph-attribution gate fails because the real-minus-DUFS interval crosses zero.

The most important mechanism qualification is that the selected direction is
almost entirely the pooled cross-roughness gradient `-c`: the `A` term adds only
+0.006pp. Moreover, a graph built from unresidualized family contributions is
at least as strong on development. The supported claim is therefore a
**family-structured graph cross-roughness orientation**, not a unique effect of
family residuals and not evidence for a special Laplacian eigenmode.

## Retrospective transfer

All target scores and hashes were frozen before each report indexed target
labels. Every panel below had nevertheless already been inspected elsewhere in
the project, so these are known-outcome stress tests rather than prospective
confirmation.

| panel | primary one-SE delta (pp) | max-mean delta (pp) | Family-NRM delta (pp) | primary recovery |
|---|---:|---:|---:|---:|
| ProcessBench / Llama | +0.588 | +1.219 | +1.580 | 37.2% |
| ProcessBench / Qwen | +0.137 | +0.215 | +0.557 | 24.6% |
| SemGrad | +0.257 | +0.155 | +1.310 | 19.6% |
| PRMBench | **-0.420** `[-0.621,-0.226]` | -0.950 | **+0.460** | -91.2% |
| HLE | **+0.912** `[+0.248,+1.512]` | +1.624 | +0.345 | 264.0% |

HLE is encouraging but fragile as a benchmark: it contains only 68
judged-correct answers and uses one interim judge. Its 264% recovery ratio is
also unstable because the Family-NRM denominator is small and uncertain.
PRMBench is the decisive
counterexample to a general replacement claim: its larger clustered evaluation
finds a significant loss while the original frozen Family-NRM improves.

The transfer controls reinforce the domain-dependence. On HLE, cross-only is
essentially identical to primary (+0.915pp), whereas the DUFS graph is harmful
(-1.138pp). On PRMBench, both primary and cross-only are harmful (about
-0.419pp), while the DUFS and contribution controls are small and uncertain.
Thus changing the kNN constructor again is not supported as the missing fix;
the unresolved problem is whether the source-derived signed family orientation
is compatible with the target domain.

## Claim boundary and next valid test

The defensible conclusion is:

1. A pooled family-graph roughness gradient can reconstruct 90.8% of the
   historical Family-NRM development gain without the eigenvalue-one rule.
2. Graph alignment has borderline positive support; equal-dataset-family
   pooling has a positive but uncertain increment. Residual specificity and
   superiority to the DUFS graph are not identified.
3. Transfer is heterogeneous: positive on several retrospective panels and
   significantly negative on PRMBench. No universal generalization claim is
   justified.

Do not tune further on PRMBench or HLE. A clean next test would freeze the
one-SE estimator exactly as recorded here, produce target scores on at least
one genuinely unopened dataset family (preferably across two model families),
hash them before labels are exposed, and use IU-PCR as the primary comparator.
Until that test, retain PGRD as a discovery-level candidate and keep the frozen
Family-NRM result separate.

## Integrity audit

Three independent post-run audits reproduced the development selector,
mechanism contrasts, all external AUROCs/AUPRCs, the 200,000-draw development
bootstrap, and the PRMBench/HLE bootstraps. They also verified every current
source/input/score hash, all 26 frozen external directions, exact comparator
row/IU alignment, and the HLE judge-manifest binding. The mechanical unit suite,
Python compilation, and `git diff --check` pass.

| frozen artifact | manifest hash |
|---|---|
| development fit | `10ab5790fcd2545892d1f6353cc7620b8a5e94cd80fffddf381872341326376b` |
| mechanism controls | `96aeeb3d9c33f0a13fea2f0b9e151548e571dd564db6c36868e443e242984938` |
| ProcessBench/SemGrad | `f0903d23b4279ee38760ff9c3b5d1f1dcdb3f75cc3a2f1e2c1857142d39eb930` |
| PRMBench | `ec0e34243492ec1a08b031c3bfe77d5d3ce976224bcb4bbe2d09433f6fddb97d` |
| HLE | `9629a23e98a0473fb8c18d84464791d8c57f4acb0affd0f9d6c757d8b4fe26ea` |

## Canonical artifacts

- Development: `RESULT.json`, `REPORT.md`, `FROZEN_SELECTION.json`, and
  `FIT_COMPLETE.json` in this directory.
- Mechanism controls: `controls/RESULT.json` and `controls/REPORT.md`.
- ProcessBench/SemGrad:
  `../pooled_graph_roughness_external_v1/process_semgrad/RESULT.json`.
- PRMBench: `../pooled_graph_roughness_external_v1/prmbench/RESULT.json`.
- HLE: `../pooled_graph_roughness_external_v1/hle/RESULT.json`.
