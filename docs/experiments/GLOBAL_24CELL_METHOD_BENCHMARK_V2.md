# Global 24-cell method benchmark v2

**Status:** design draft; roster is intentionally inclusive; no run is
authorized by this document.

## 1. Question

On the same 24 dataset-model cells and one common measurement contract, which
of our label-free response-level fusion methods ranks correct answers most
reliably? For Family-NRM and PGRD, a second question is whether any gain comes
from the residual/graph construction or from cross-dataset selection.

This is retrospective development benchmarking. These cells have already
influenced method and feature design, so a positive result can choose a
candidate for new-data validation but cannot itself be external confirmation.
"Fully unsupervised" below describes score fitting in a cell under a frozen
feature/family/orientation contract; it does not erase the earlier labelled
development history of that contract.

## 2. Shared population and evaluation contract

- Exact roster: the existing 24 cells in `scripts/inscope_cells.py`; 48,607
  rows; `inside_coqa_llama7b` remains excluded for its documented generation
  defect.
- Primary metric: mean of the 24 per-cell AUROCs (**cell-macro**).
- Secondary summaries: equal-dataset-family macro, QA macro, math macro,
  correctness-positive cell-macro AUPRC, prevalence, wins/ties/losses, worst
  cell, and paired intervals versus matched IU-PCR. Orient every score so that
  a larger value means more likely correct. If hallucination-positive AUPRC is
  useful, report it under a separate name; do not mix the two conventions.
- Per-cell output: AUROC, AUPRC, score variance, rank correlation with IU-PCR,
  feature count, present families, runtime, and all method-specific health
  diagnostics.
- No pooled-row AUROC and no average across response detection, localization,
  prefix prediction, or stopping.

Before implementation, freeze exactly one primary measurement contract for all
methods. The current draft preference is the full present inventory because it
retains every available measurement and matches the completed DEEM input; this
is a recommendation, not yet a frozen decision:

- `fixed_stable_v1`, used by Frozen 24-cell benchmark v1; or
- the full present inventory (19-30 measurements by cell), used by the DEEM v1
  package.

The unchosen contract may be a prespecified sensitivity lane. Results from the
two lanes may not be combined or ranked together. Every method must receive the
same transformed matrix, feature order, signs, row IDs, and missing-feature
policy inside a lane.

## 3. Main candidate roster

The soft roster lives in
`configs/global_24cell_method_benchmark_v2_registry.csv`. No uncertain method
is removed at the design stage. The required executable-manifest fields are in
`configs/global_24cell_method_benchmark_v2_run_registry_schema.csv`; its current
example row is deliberately non-executable.

### Main label-free leaderboard

1. equal-family mean (label-free floor);
2. continuous L-SML on the full aligned pool, plus the parameter-free DUFS,
   tuned DUFS, and GroupFS selector wrappers followed by the same continuous
   L-SML fusion;
3. deployed U-PCR, U-PCR with `sign(rho)` orientation, and keep-all IU-PCR
   (linear covariance references);
4. DUFS-LIU at its frozen graph strength;
5. SU-PCR with sparse correlated errors;
6. balanced-atomic CA-SpecRaGE;
7. continuous additive DEEM (B3), five-seed ensemble;
8. Family-NRM-WC, the new within-cell neutral-residual ablation; and
9. PGRD-WC-cross, the new within-cell graph-cross-gradient ablation.

The L-SML selector rows are included under Omri's soft-decision rule: if their
aligned implementation or exact role is uncertain, retain them now and decide
at the pre-run review. Continuous L-SML with GOOD_6 and LOCO_5 remain
label-informed references rather than label-free methods. Unified-28 remains a
cross-task reference arm. Historical mixed-v2 feature transformations remain a
separate contract sensitivity because their mapping was selected after
inspecting these development cells.

The best single measurement is also a label-informed diagnostic ceiling when
"best" is chosen from these outcomes. It is reported beside the references,
not ranked as a label-free method.

### Mechanism and completeness controls

- hard and repaired-soft packaged DEEM adapters (B1/B2), with collapse health
  shown beside their scores;
- PGRD-WC full quadratic actuator;
- Family-NRM-WC with the equal-family residual direction instead of the
  eigenmode;
- PGRD-WC with a DUFS-coordinate graph, a node-permuted graph, and an optional
  preregistered permutation gate that returns exactly to IU-PCR when the local
  cross-gradient is indistinguishable from its graph-permutation null;
- manual and micro-view CA-SpecRaGE;
- Family-residual Graph-LIU;
- leverage- and cardinality-balanced contribution-space IU;
- Atomic-NRM and SU-aware PGRD sidecars;
- node-permuted, uniform, and zero-correction graph controls where relevant.

Residual-Graph DEEM is not a natural-data leaderboard row. Its graph arms
stopped at the synthetic specificity gate; graph-free DEEM-B3 is separate.

## 4. Common residual representation

For feature matrix `X`, IU-PCR gives weights `w` and base score `s_IU=Xw`.
The fixed family registry partitions the present features. For family `g`,

\[
h_g=\sum_{i\in g}w_iX_i,\qquad b=z(s_{IU}),\qquad
u_g=z(h_g)-b\frac{b^Tz(h_g)}{b^Tb},\qquad r_g=z(u_g).
\]

The final `z` uses population standard deviation (`ddof=0`). A family whose
contribution or residual standard deviation is at or below the frozen
numerical tolerance is dropped and logged; it is not silently standardized
with scale one. Write `R=[r_1,...,r_G]`, where `G` is counted after this rule.
Only usable families present in the target cell enter the local solve. The
current roster includes cells with only five present families, so
missing-family support is required; absence alone may not exclude a cell. The
family registry is a fixed prior, not learned structure.

## 5. Family-NRM and PGRD regimes

The method family and the selection regime are separate factors.

| regime | target data | donor data | correctness labels | report location |
|---|---|---|---|---|
| **A — within-cell fully unsupervised** | constructs everything from the target cell | none | none | main leaderboard |
| **B — donor-unsupervised** | target scoring only | may select stable direction, geometry, or settings without labels | none | secondary factorial |
| **C — donor-label selection** | target scoring only | labels may select direction, sign, graph, or strength | donor labels only; target family held out | supervised ceiling |

Target-label oracles are outside A/B/C. They are diagnostic ceilings and may
never appear in the unsupervised leaderboard.

### 5.1 Family-NRM-A: within-cell neutral mode

Compute `C_R=R^T R/n` in the target cell. Let `S` contain all eigenvectors whose
distance from eigenvalue one is within the frozen tolerance of the minimum.
Project the equal-family vector onto this eigenspace and normalize it:

\[
v_*=\frac{SS^T\mathbf 1}{\|SS^T\mathbf 1\|}.
\]

This is basis-invariant when eigenvalues tie and orients the direction toward
the fixed equal-family anchor. Score

\[
s_A=b+\frac{Rv_*}{G\,\operatorname{sd}(Rv_*)}.
\]

Use exact IU fallback when `G<3`, the tied-space projection is below the frozen
numerical floor, or the correction SD is below that floor.
This is a new benchmark ablation, not the completed canonical Family-NRM
experiment.

### 5.2 PGRD-A: within-cell cross-gradient

Build the duplicate-safe symmetric union-kNN graph on `R`, with `k=7`, stable
row-ID tie keys, and a symmetric normalized Laplacian `L`. Let

\[
A_0=R^TLR/n,\qquad c_0=R^TLb/n,\qquad
\alpha=G/\operatorname{tr}(A_0),\qquad A=\alpha A_0,\qquad
c=\alpha c_0,\qquad d=-c.
\]

The primary score is

\[
s_A=b+\frac{Rd}{G\,\operatorname{sd}(Rd)}.
\]

The sign is fixed by descent (`c^Td <= 0`); no orientation label is used. The
trace scaling cancels after standardizing the primary cross-only correction,
but changes pooled and full-quadratic variants, so it is mandatory in every
arm. Use exact IU fallback for an invalid graph, non-positive roughness trace,
zero direction, or constant correction. The full quadratic
`d=-lambda(I+lambda A)^(-1)c` is a secondary actuator and needs a fixed lambda
chosen without reusing donor-label tuning.

### 5.3 B and C variants

Apply the same three-regime labels to both families:

- Family-NRM-B pools target-free donor residual covariance and chooses the
  neutral mode without donor labels. Family-NRM-C allows donor labels to select
  the mode, orientation, or correction strength.
- PGRD-B pools donor graph moments and fixes geometry/direction/strength using
  target-free diagnostics only. PGRD-C allows donor labels to select graph,
  sign, actuator, or strength under outer leave-dataset-family-out evaluation.

The historical Family-NRM direction fit did not read labels, but its donor
roster used a label-dependent minimum-positive rule. Under this strict axis it
is a legacy C calibration, not B. The historical PGRD one-SE/max-mean results
are C. The Step-286 "label-free geometry selector" also belongs in legacy C:
its geometry choice was label-free, but its correction strength was inherited
from donor-label selection. Clean B versions require a fresh rerun with an
availability-only roster and structurally fixed strength.

### 5.4 Matched diagnostic slices

The three regimes are not one causal contrast unless the other factors are
held fixed. The registered factorial therefore has two layers:

1. a matched selector slice fixes `R`, the candidate-direction/graph bank, the
   cross-only actuator, normalization, and `1/G` correction strength, then
   changes only whether the direction is constructed in the target cell,
   pooled from unlabeled donors, or selected with donor labels;
2. separate mechanism rows vary graph, actuator, sign, or strength one at a
   time and are not interpreted as the A/B/C selection effect.

For Family-NRM, a supervised residual teacher is a different estimator, not a
regime-C mode selector. Keep it as a separately named supervised ceiling.

## 6. Other method definitions

- **IU-PCR:** infer a two-component linear score from feature covariance and
  keep the full common inventory.
- **U-PCR:** the deployed version that excludes weak features and refits.
- **U-PCR + sign(rho):** the historical label-free polarity variant that uses
  the estimated feature-target covariance signs, while retaining one global
  anchor bit. Keep it separate from the current fixed-orientation deployment.
- **DUFS-LIU:** learn a label-free sample graph with DUFS and regularize the
  IU-PCR solve; `lambda=0` must reproduce IU-PCR exactly.
- **SU-PCR:** decompose observed covariance into low-rank shared signal plus
  sparse correlated errors before estimating the U-PCR score.
- **CA-SpecRaGE atomic:** treat each measurement as a view, balance duplicates,
  learn sample-specific view agreement, and regularize IU-PCR.
- **DEEM-B3:** a graph-free nonlinear energy model with a bounded additive
  network inside each fixed measurement family. Its score fit uses no labels.

## 7. Label firewall and reproducibility

Use three physically separated fitting/evaluation stages:

1. **A/B fitting:** receives label-free target/donor bundles and cannot import
   a label loader. It writes one score vector and diagnostic record per
   method/cell/seed.
2. **C selection:** may read donor labels only, with the entire target dataset
   family physically excluded. It freezes the selected rule and target scores.
3. **Evaluation:** opens target labels only after every score and configuration
   hash is frozen.

A fresh rebuild must match frozen score artifacts semantically and all
deterministic artifacts byte-for-byte.

The current CSV is a soft candidate roster, not an executable run manifest.
Before any run, its `method_id` must join one-to-one to a frozen machine-readable
run registry recording: population and row-policy hashes, feature contract,
IU implementation and orientation, selection regime, family prior, graph,
normalization, strength, hyperparameters, seed ensemble, label access,
fallback, implementation commit, and configuration hash. A result lacking
that record cannot enter the aligned leaderboard.

## 8. Required tables and figures

1. Main 24-cell leaderboard with cell-macro first and one matched IU-PCR row.
2. Pairwise deltas versus IU-PCR with cell and dataset-family blocked intervals.
3. Full per-cell AUROC/AUPRC table and method-by-cell heatmap.
4. QA/math and eight-family profiles; no single macro without these views.
5. Family-NRM and PGRD A/B/C factorial table. C and target-label oracles use a
   different color and never enter the unsupervised rank order.
6. Rank correlation and disagreement plots between methods, including cells
   where the overall leader loses.
7. Graph diagnostics: edge overlap, operator cosine, degree/connectivity,
   roughness, length association, and their relationship to paired AUROC
   changes. Diagnostics explain differences; they do not select a winner after
   labels are opened.
8. Runtime and memory versus row/feature count.

## 9. Published and application comparisons

Published response detectors are per-cell external anchors unless row identity
is proven. Report model/dataset, access class, sampling count, and whether the
number is paper-exact, adapted, or proxy. Do not compute a paired delta or a
24-cell published macro from unmatched generated answers.

This 24-cell benchmark is one lane of the broader protocol in
`docs/experiments/MULTI_POPULATION_METHOD_BENCHMARK_V1.md`. That protocol also
registers the external response, localization, prefix, stopping, RAG,
white-box, repeated-generation, published-comparator, and negative-stress
panels. Each receives its own roster and metric. A response method enters a
step, prefix, or evidence lane only through a frozen task adapter, and no
cross-task average is allowed.

## 10. Pre-run decisions

This draft becomes executable only after Omri confirms:

1. the primary feature contract and its sensitivity contract;
2. the exact IU-PCR implementation/orientation shared by all residual methods;
3. Family-NRM-A missing-mode tolerance and fallback;
4. PGRD-A graph construction and fixed trust;
5. which soft-roster controls remain in the full run; and
6. the bootstrap units, seeds, and promotion/noninferiority thresholds.
