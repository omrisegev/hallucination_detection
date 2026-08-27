# Graph geometry and selector identification research V1

**Status:** launch brief for a separate research thread (2026-08-23).

This is a bounded reopening of graph research motivated by a specific protocol
anomaly. It is not permission to replace frozen baselines, relabel previously
opened data as confirmation, or report the largest value found in a geometry
sweep as a validated result.

## 1. Goal

Determine whether the pooled family-residual roughness method contains a
transferable graph-geometry signal beyond the fixed duplicate-safe union-kNN
graph with `k=7`, and whether that geometry can be selected without correctness
labels.

The central deliverables are:

1. an exact decomposition of the apparent +0.251pp to +0.452pp gap into
   selector, graph-search, and trust-grid effects;
2. a compact, nonredundant bank of plausible graph geometries;
3. supervised-oracle and genuinely label-free geometry selectors kept as
   different methods;
4. nested held-family estimates and frozen retrospective transfer;
5. a clear decision about whether geometry is informative, selection is
   merely optimistic, or a useful geometry exists but remains unidentified
   without labels.

## 2. Starting evidence and the critical correction

The canonical pooled graph-roughness reconstruction reports:

- fixed residual union-kNN, `k=7`, conservative one-SE/tail guard:
  **+0.251pp** versus IU-PCR, 95% equal-family bootstrap
  **[+0.027,+0.458]**, 6/8 positive families;
- the same canonical lineage's nested max-mean HPO sensitivity:
  **+0.450pp**, 6/8 positive families.

The SU sidecar V1 searched four graph settings, used max-mean selection, and
changed the trust grid. Its observed-IU control was **+0.452pp**. Therefore the
near equality `+0.450 ~= +0.452` is strong preliminary evidence that most of
the apparent +0.201pp increase may come from the selector rather than graph
choice. It is not currently valid to call the difference an oracle graph gap.

The conservative SU V2 restored fixed union-`k=7`, the one-SE/tail-guard
selector, and the canonical trust grid, exactly reproducing +0.251pp. SU
cross-family covariance cleaning then added only +0.009pp
[-0.012,+0.037], and SU-rho was harmful. SU is closed for this research and
must not be mixed into the geometry experiment.

The current graph mechanism also has an attribution warning. The contribution
graph reached +0.299pp; real residual graph minus cross-only was only +0.006pp
[+0.002,+0.012]; and the complete registered graph-attribution gate failed.
Twenty node permutations were nevertheless informative: real minus their mean
was +0.411pp [+0.185,+0.618], randomization `p=0.0476`. The new study must
retain these controls and must not presuppose that residualization is the
unique source of value.

## 3. Questions and competing hypotheses

### H1 — selector effect

The one-SE/tail guard deliberately trades mean development AUROC for stability
and simplicity. Max-mean may select a genuinely better transferable correction
strength, or may capitalize on seven inner families. The +0.199pp difference
between the two known reports must be estimated as a selector effect before
geometry is interpreted.

### H2 — geometry effect

Under identical candidate scores, trust grid, and selector, allowing graph
geometry to vary may improve held-family AUROC over fixed union-`k=7`.

### H3 — unidentified geometry

A supervised donor-label selector may choose useful graphs on held families,
while intrinsic label-free graph criteria fail. This would establish useful
geometry headroom but not an unsupervised solution.

### H4 — label-free geometry identification

Stability, operator consistency, and predicted roughness descent may select a
graph with low regret relative to the supervised oracle and improve IU-PCR on
held families without using correctness labels in graph selection.

### H5 — selection optimism

The larger hypothesis class may improve inner training-family AUROC but not
outer-family or transfer AUROC. In this case the +0.452pp observation is model
selection variance rather than a reusable mechanism.

## 4. Phase A: exact protocol-factor decomposition

Before adding a new graph, build or reuse one label-free, hashed score bank and
run a matched factorial. All compared arms must use the same source omissions,
lambda candidates, direction fitting, target scoring, and equal-family
aggregation.

### Factors

1. **Geometry capacity**
   - fixed: duplicate-safe residual union-kNN, `k=7`;
   - searched: `union_k5`, `union_k7`, `union_k15`, `adaptive_k7`.
2. **Selector**
   - canonical one-SE plus worst-inner-family guard `-0.005 AUROC`;
   - max inner-family mean.
3. **Trust class**
   - canonical: `{0.5, 1, 2}`;
   - V1: `{0.25, 0.5, 1}`;
   - expanded diagnostic: `{0.25, 0.5, 1, 2}`.
4. **Lambda**
   - hold fixed across arms at
     `{0.03, 0.1, 0.3, 1, 3, 10, 30, 100}`.

Run strict nested leave-dataset-family-out. Report paired outer-family main
effects and interactions. Required contrasts include:

- max-mean minus one-SE for fixed union-`k=7` under the same trust class;
- searched geometry minus fixed union-`k=7` under the same selector and trust
  class;
- V1 trust class minus canonical trust class under fixed geometry/selector;
- the exact V1 protocol and exact canonical protocol as reproduction anchors.

No interpretation of geometry is allowed until both known anchors reproduce
within numerical tolerance. If +0.450 appears under fixed union-`k=7`, the
research report must state that the prior +0.452 value is not evidence of a
graph-search gain.

## 5. Phase B: compact geometry bank

Audit `HISTORY.md` and the existing graph/topology reports before freezing the
bank. Do not repeat already closed broad sweeps without a new discriminating
hypothesis. Prefer a factorial whose axes have mechanistic meanings.

At minimum preserve these coordinate controls:

- family residual coordinates `R`;
- standardized unresidualized family contributions;
- historical DUFS coordinates;
- a block-balanced DUFS/family hybrid only if it can be compared at matched
  capacity.

Candidate geometry axes may include, after auditing existing implementations:

- topology: symmetric union, mutual, and adaptive kNN;
- neighborhood scale: a small prespecified set such as `{5,7,15}`;
- metric: standardized Euclidean, cosine/correlation, and a shrinkage
  Mahalanobis metric fitted without labels;
- edge weighting: existing reviewed weighting and one self-tuning heat-kernel
  alternative;
- representation: residual versus contribution coordinates.

Do not take the unrestricted Cartesian product merely because it is cheap.
Remove duplicate or nearly identical graphs using edge overlap/operator
similarity before outcome scoring, and report the effective hypothesis-class
size. Every graph must use deterministic duplicate-safe tie handling.

## 6. Three selectors that must remain separate

### 6.1 Fixed label-free selector

Choose a graph using only prespecified intrinsic diagnostics. Candidate
diagnostics include:

- connectedness, isolated-node fraction, degree-tail balance, and numerical
  validity;
- graph stability under deterministic row subsampling, coordinate jitter, and
  feature/family leave-one-out perturbations;
- cosine stability of the pooled direction under leave-one-source-family-out
  fits;
- dispersion of trace-normalized `(A_e,c_e)` across source families;
- predicted quadratic roughness decrease under the fitted direction;
- guard against a graph making answer length substantially smoother than the
  IU baseline/target-free family coordinates.

The rule combining diagnostics must be frozen before labels score it. Prefer a
hard validity filter followed by a simple lexicographic/Pareto rule. If weights
are tuned using AUROC, the method is no longer label-free and belongs below.

### 6.2 Supervised donor-label meta-selector

Use labels only from the nested training families to choose graph geometry and
calibration. This is statistically legitimate under strict outer LOFO, but it
must be named supervised meta-selection. It answers whether known labeled
domains can select a graph that transfers to a new domain.

### 6.3 Oracle diagnostic

For diagnosis only, identify the best geometry separately for each held family
using that family's labels. Never report it as a deployable method. Use it to
measure geometry headroom and selector regret.

For each selector report selection agreement, rank correlation with held-family
geometry performance, and regret relative to the held-family oracle. A
label-free selector is interesting only if low regret accompanies real AUROC
gain; agreement alone is insufficient when many graphs are equivalent.

## 7. Evaluation and claim boundaries

### Development

- Original eight dataset families, strict nested LOFO.
- Equal weight to dataset families; cells average only within family.
- Primary endpoint: AUROC delta versus exact IU-PCR.
- Required paired comparisons: new selector versus IU-PCR, canonical fixed
  graph, and capacity-matched graph-only/selector-only controls.
- Secondary: AUPRC, wins, worst family, direction cosine, graph health,
  effective number of distinct graphs, and Family-NRM gain recovery.

### Transfer

Freeze geometry class, selector, lambda/trust rule, directions, hashes, and
scores before reading transfer labels. ProcessBench Llama/Qwen, SemGrad,
PRMBench, and HLE are historically opened and may be used only as
retrospective stress tests. A confirmatory claim requires a new sealed dataset
family and preferably a new model family.

### Interpretation

- If only max-mean improves, conclude selector/correction-strength sensitivity,
  not graph-geometry discovery.
- If supervised graph selection improves but the label-free selector does not,
  conclude geometry headroom without label-free identification.
- If geometry search loses on outer folds, conclude selection optimism.
- Promote a label-free geometry claim only when it beats fixed union-`k=7`
  under matched capacity and transfers after freeze.

## 8. Controls and safety checks

- exact IU-PCR identity at zero correction;
- exact reproduction of +0.251pp canonical and approximately +0.450pp
  fixed-graph max-mean anchors;
- at least 20 deterministic node-permuted graph controls;
- contribution, DUFS, cross-only, equal-cell pooling, and family-axis
  permutation controls;
- row identity, family registry, missing-family, and duplicate-coordinate
  assertions;
- fit/report separation: graph construction and candidate scores are hashed
  without labels before outcome reporting;
- no SU covariance or SU-rho arms in this study;
- no overwrite of frozen canonical artifacts.

## 9. Required plots

1. factorial effect forest plot for selector, geometry capacity, trust class,
   and their interactions;
2. held-family paired line plot for canonical, max-mean fixed graph,
   supervised geometry selector, and label-free geometry selector;
3. geometry-by-family AUROC heatmap with nested selections overlaid;
4. intrinsic diagnostic versus held-family graph performance scatterplots;
5. selector regret and oracle-gap plot;
6. effective graph diversity/edge-overlap heatmap;
7. frozen transfer comparison against IU-PCR, canonical pooled graph, and
   Family-NRM.

Each plot must distinguish retrospective discovery, supervised selection,
label-free selection, oracle diagnostics, and external stress tests.

## 10. Canonical starting artifacts

| artifact | SHA-256 at launch |
|---|---|
| `docs/experiments/POOLED_GRAPH_ROUGHNESS_DIRECTION_V1.md` | `56d0f90ade36a7e5f2df31b286a64e0cfd4dce14c58a009b339147d735f93af8` |
| `results/pooled_graph_roughness_direction_v2/REPORT.md` | `ad15df53a5f399d5727665a02aa6dd13230ffdb14064c6810a7cb50f2da05185` |
| `results/pooled_graph_roughness_direction_v2/RESULT.json` | `afd81cf5f3bf50fce2d7e4e312c194604928aa06300c922ea8014c960000484b` |
| `results/pooled_graph_roughness_direction_v2/FROZEN_SELECTION.json` | `ff0b6e824d0140b7e5fbdab0d10f97b7a32ff80217d6b740915436c5ce8d1aa3` |
| `results/pooled_graph_roughness_direction_v2/controls/REPORT.md` | `8f4263a73017a0179995acdaaadb7fb3852011af406336f139b285c6d5e5982a` |
| `docs/experiments/SU_POOLED_GRAPH_ADAPTATION_SIDECAR_V1.md` | `814ab6a0e9d7babd21cc17dfafe547d1c5eaac403142b8a16c3f9e857bf51ef9` |
| `results/su_pooled_graph_adaptation_sidecar_v1/REPORT.md` | `f48f57864141b31abe2809d8b1a26d45ae05485533940b5e4dae56f2f981a77a` |
| `docs/experiments/SU_POOLED_GRAPH_ADAPTATION_CONSERVATIVE_V2.md` | `ca8d80dc92466016ce1623710eed1680aca3ebe173efc55c9493abf0dad8e365` |
| `results/su_pooled_graph_adaptation_conservative_v2/REPORT.md` | `7d213a47d06a7b2fedbfbfb833b8579f9450438bca2de4f88548346e503af6d8` |
| `spectral_utils/pooled_graph_roughness.py` | `d33dd89e61fb44d56f4c6e26b89a4e4835e542c17d8f7c8d7deefa99d6c1eb61` |
| `spectral_utils/family_residual_graph.py` | `f07f05e41fe8de275045fe6ae018e9a1254398c2c615a5cefb9bb060f9f38ba9` |

Also read:

- `CLAUDE.md`, `PROGRESS.md`, and `HISTORY.md`;
- `docs/research_notes/su_pooled_graph_adaptation_conclusion_2026-08-23.md`;
- `results/family_residual_graph_liu_v3/SYNTHESIS.md`;
- `results/direct_dufs_conditional_graph_topology_audit_v1/REPORT.md`;
- `docs/research_notes/research_status_consolidated_2026-08-19.md`.

## 11. Decision vocabulary

The final report must choose one bounded decision:

- `SELECTOR_EFFECT_WITHOUT_GEOMETRY_GAIN`;
- `SUPERVISED_GEOMETRY_HEADROOM_ONLY`;
- `LABEL_FREE_GEOMETRY_SELECTION_SUPPORTED`;
- `GEOMETRY_SEARCH_SELECTION_OPTIMISM`;
- `INCONCLUSIVE_GEOMETRY_IDENTIFICATION`.

Do not use “optimal graph” unless performance is measured under strict held-
family selection and the scope (supervised, label-free, or oracle) is stated.
