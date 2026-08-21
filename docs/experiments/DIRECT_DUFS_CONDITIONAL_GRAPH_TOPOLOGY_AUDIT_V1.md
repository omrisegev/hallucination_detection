# Direct DUFS conditional graph-topology audit v1

## Status and claim boundary

This protocol was frozen before any result from the new radius, adaptive, or
diffusion graphs was evaluated, then amended before execution in response to
an independent no-edit implementation review. No real-cache candidate-graph
outcome was opened before these amendments. The experiment is a retrospective
closure audit on already opened caches. It cannot promote a detector or
establish a physical hallucination manifold. Its narrow question is whether
any fixed, label-free graph construction exposes target geometry that remains
after conditioning on answer length and that is useful to frozen LIU.

The three tasks remain separate throughout:

1. Global completed-answer hallucination (`global24`).
2. ProcessBench presence of a reasoning-process error (`processbench`).
3. RAGTruth response hallucination (`ragtruth`).

No target metric is pooled across these lanes. The review amendments replace a
coarse-bin null vulnerable to residual within-bin length ordering, explicitly
remove self neighbours under duplicate rows, and require three predeclared
target-blind resolutions of tied distances.

## Validation populations

- `global24`: the 21 frozen cells whose registered matrix contains an explicit
  `trace_length` coordinate. Each cell is held out when fitting the train-only
  length residualizer.
- `processbench`: Qwen3-4B on GSM8K, MATH, OlympiadBench, and OmniMath fits the
  train-only residualizer. The matching four Qwen3-8B cells are validation.
- `ragtruth`: development fits the residualizer; test `original30_full` is
  validation. This is one split diagnostic, not an across-cell estimate.

All feature signs and mixed-v2 transformations are inherited unchanged.
Labels are unavailable to representation construction, DUFS fitting, graph
construction, IU-PCR, and LIU. Labels enter only target evaluation and the
predeclared conditional-null procedures.

## Frozen representation axis

Every validation cell is evaluated under:

1. `original`: registered matrix and canonical DUFS gates.
2. `drop_length`: delete every registered feature whose name contains
   `length`; refit DUFS with seeds `(11,23,37)` for 80 epochs.
3. `train_residualized`: start from `drop_length`, fit the registered
   equal-cell-weighted cubic model of `log1p(length)` on training cells only,
   subtract its prediction, standardize without labels, and refit DUFS.

The residualizer is a falsification arm, not a candidate detector. Its known
performance loss is retained rather than repaired post hoc.

## Frozen graph axis

All candidate graphs use the same soft-gated coordinates within a
representation. Unless noted, affinity for edge `(i,j)` is

`exp(-||z_i-z_j||^2 / (sigma_i sigma_j))`,

where `sigma_i` is distance to the seventh strictly-positive **distinct sample
location** (or the farthest available positive distinct location when fewer
than seven exist; `1` only when every row is identical). Duplicate multiplicity
therefore cannot collapse bandwidth to zero, while every duplicate sample still
remains a graph node. Self is removed by row identity, never by assuming the
first returned neighbour is self. This matters because several registered
matrices contain many exact duplicates.

Distance ties are resolved lexicographically by a target-blind unique key after
an expanding query proves that every distance tied at the selected boundary is
present. No coordinate jitter is applied. The complete candidate audit is
repeated for frozen tie seeds `(101,211,307)`; every gate must pass all three.
Duplicate exposure, unique-row candidate depth, and the tie rule are reported.

The roster is:

1. `union_knn_k7_self_safe`: corrected symmetric union-kNN and edge baseline.
2. `radius_edge_matched_k7`: globally shortest undirected pairs with exactly
   the corrected union edge count. Sparse expansion must prove completeness.
3. `adaptive_knn_mean7_k3_25`: density ranks allocate directed degrees in
   `[3,25]` with mean exactly seven; sparse regions receive more neighbours.
4. `diffusion_edge_matched_base25_t2`: base-25 walk, two steps, top-25 row
   truncation, and corrected-union edge budget.
5. `diffusion_edge_matched_base25_t4`: registered four-step sensitivity; it
   remains in the five-member Holm family but is decision-ineligible and cannot
   rescue a failed two-step primary by itself.
6. `deployed_union_knn_k7`: historical implementation, used only to reproduce
   frozen LIU. It is not a candidate because its self assumption is unsafe
   under duplicates.
7. `mutual_knn_k7`: reciprocal-kNN control; fragmentation is not repaired.
8. `length_only_knn_k7`: positive nuisance control using held length only.
9. `permuted_self_safe_union_knn_k7`: deterministic node relabelling of the
   corrected union, preserving topology and weights, as a negative control.

Radius expansion is capped at 512 neighbours per row. A cell fails closed if
the exact shortest-pair boundary cannot be proved within that cap. No graph,
`k`, diffusion time, edge budget, or representation is chosen by AUROC.

## Length-conditional target nulls

The raw test permutes binary target over all rows. It cannot support a
hallucination-specific interpretation.

The earlier coarse contiguous-bin null is forbidden: a target monotone in
length inside every bin can pass it despite being generated by length alone.

Two separately predeclared primary nulls must both pass; neither rescues the
other post hoc:

1. **Exact-length swaps.** Labels move only among rows with exactly equal held
   length. Eligibility requires movable fraction at least `0.20`, at least 20
   movable rows, and at least five mixed exact-length strata.
2. **Cross-fitted flexible propensity CRT.** A five-fold OOF histogram
   gradient-boosting classifier predicts target from held `log1p(length)` only:
   100 iterations, learning rate `0.05`, at most 15 leaves, L2 `1`, no early
   stopping, and a frozen sample-size leaf rule. Null targets are independent
   Bernoulli draws from clipped OOF propensities. Eligibility requires overlap
   fraction at least `0.20` in `[0.05,0.95]`, Brier no worse than the constant
   model by more than `0.01`, weighted decile calibration MAE at most `0.10`,
   and both classes in every draw.

A third **adjacent length-matched-pair swap** is sensitivity only. Rows sorted
by held `log1p(length)` are paired deterministically without target access and
swapped independently. Diagnostics include discordant pairs, movable fraction,
exact-tie fraction, and median/95th-percentile/maximum gaps. Eligibility needs
ten discordant pairs, movable fraction `0.20`, p95 gap at most `log(1.25)`, and
maximum gap at most `log(2)`. It cannot rescue either primary null.

The same 199 unconditional, exact, CRT, and pair draws are reused across all
representations, graphs, and tie seeds of a cell. The statistic is the
symmetric-normalized-Laplacian Rayleigh quotient of centered target. Positive
effect means lower energy than the null mean. Weighted neighbour purity is
evaluated against every permutation family. Held length uses only unconditional
permutations.

Within each cell, representation, and tie seed, exact, CRT, and pair p-values
are Holm adjusted separately over the five fixed candidates. If any member is
non-finite, the full family fails closed instead of silently shrinking.

## Graph health and LIU utility

Every graph reports edges, components, isolated fraction, largest component,
degree quantiles, effective neighbours, and algebraic connectivity when
connected. The sparse eigensolver uses a frozen deterministic start vector so
this diagnostic is rebuild-stable. Ordinary IU-PCR (`lambda=0`) and LIU
(`lambda=0.1`) use the same
feature matrix. Target AUROC and paired delta are reported. Equal-cell mean
delta intervals use 5,000 deterministic paired bootstraps in `global24` and
`processbench`; RAGTruth is descriptive because it has one validation split.

## Predeclared interpretation gates

A candidate has `CONDITIONAL_GEOMETRY` in a lane only when:

1. At least two thirds of cells are eligible separately for exact and CRT.
   Within each eligible set, effect is positive in at least two thirds of
   `global24` and `processbench` cells and Holm `p<=0.05` in at least half. The
   single RAGTruth split must be eligible, positive, and significant under both.
2. The result holds in `original` and `drop_length`; `train_residualized` is a
   registered falsification arm.
3. It holds under all three tie seeds.
4. At least 90% of cells have largest component at least `0.90` and isolated
   fraction at most `0.05`.

A candidate has `DETECTOR_UTILITY` only when, under both required
representations and every tie seed, LIU-minus-IU mean AUROC is positive in both
multi-cell lanes, the global 95% paired interval is strictly positive, and
RAGTruth is nonnegative.

Before interpretation, a fail-closed control gate requires: a strong length
effect on length-only; within each multi-cell lane, at most 15% unadjusted false
positives under either primary conditional null on that graph and at most 15%
raw false positives on permuted union; the single RAGTruth lane permits no
false positive under any of those controls; sufficient null eligibility in
every lane; exact edge budgets;
proved radius boundaries; exact adaptive mean-k; and historical frozen-score
reproduction. Failure yields `CONTROL_FAILURE_INVALIDATES_GEOMETRY_AUDIT`.

Otherwise the decision is assigned without choosing a best graph post hoc:

- `NO_GRAPH_REVEALS_LENGTH_CONDITIONAL_TARGET_GEOMETRY`.
- `CONDITIONAL_GEOMETRY_WITHOUT_DETECTOR_UTILITY`.
- `ROBUST_LENGTH_CONDITIONAL_GEOMETRY_AND_UTILITY`.

Raw smoothness without both conditional gates is never called evidence of a
hallucination-specific manifold. AUROC gain without conditional geometry is
graph regularization, not semantic identification.

## Required outputs

- `RUN_DEFINITION.json`: source/input hashes and frozen constants.
- `CELL_GRAPH_METRICS.csv`: all nulls, purity, health, AUROC, and eligibility.
- `LANE_GRAPH_SUMMARY.csv`: separate lane/representation/tie summaries.
- `PAIRED_INTERVALS.csv`: paired cell-bootstrap intervals by tie seed.
- `CONTROL_CHECKS.json`: all fail-closed controls and invariants.
- `REPRESENTATION_DIAGNOSTICS.json`, `DECISION.json`, and `REPORT.md`.
- `REBUILD_VERIFICATION.json`: isolated deterministic rebuild hashes.
- Static figures derived only after tables are complete.

Checkpoints are atomic per cell and representation. Fresh rows pass through the
same JSON round trip as resumed rows, so non-finite diagnostics cannot make
final tables depend on checkpoint use.
