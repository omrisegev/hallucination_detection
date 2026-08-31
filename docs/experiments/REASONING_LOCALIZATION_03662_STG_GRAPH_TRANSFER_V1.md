# Reasoning Localization 0.3662 — STG-SU / Temporal-Graph Transfer V1

Status: `PARTIALLY_CLOSED_BY_PREMISE`; the feature-support arm remains a
survivor-gated Phase-3 direction, while the registered within-answer temporal
graph arm is `NOT_RUN_BY_GATE` after the STEP-CUT premise audit.

## Historical premise and claim boundary

The exact source is the side worktree `stg_su_input_v1`. Its corrected
final-answer hallucination-detection arm, `STG_SU_STABLE`, learns a sparse
off-diagonal covariance support with stochastic gates, retains edges appearing
in at least four of five grouped folds, and then performs a deterministic
additive-plus-sparse SU-PCR refit. It changes SU-PCR's support extractor only;
`rho`, `g2`, and the final two-PC predictor remain fixed.

On the corrected oriented 24-cell evaluation, STG-SU-PCR reached cell/equal-
family AUROC `0.776120/0.742875`, versus IU-PCR `0.776087/0.742347`, canonical
SU-PCR `0.771381/0.739274`, and DUFS-LIU `0.776556/0.743035`. The gain over IU
was `+0.053pp` (`p=0.2109`) and over a cardinality-matched random support
`+0.043pp` (`p=0.2695`). The defensible premise is recovery of canonical SU
to near IU/DUFS-LIU parity, not supported superiority and not localization.

The early orientation-inverted report is excluded. The experiment and code
are untracked in that side worktree and absent from HEAD `66abed7`; provenance
must cite exact artifact hashes rather than the branch commit alone.

## Survivor-gated localization ladder

This branch opens only after Phase 2C freezes a compact eligible roster.

1. `P3G_T0_PARENT`: exact strongest compact reference and frozen reducer.
2. `P3G_F1_STG_FEATURE_SUPPORT`: STG learns sparse support/Laplacian among
   eligible feature or family blocks; outer fusion remains frozen.
3. `P3G_T1_TEMPORAL_GRAPH`: nodes are tokens, fixed masked bins, or annotated
   steps within one response; donor-only lag/similarity/covariance edges define
   one frozen graph operator before the same reducer.
4. At most one combined feature-by-time graph arm may open only if F1 and T1
   each pass their exact-parent premise gates.

## STEP-CUT premise audit (development-only)

The exploratory `STEP-CUT` screen froze donor-only graph scores across all
twelve ProcessBench cells before importing labels, but the population labels
had already been opened elsewhere. It is therefore development evidence, not
fresh confirmation and not a Phase-3 localization run.

The frozen candidate used five token-local family axes (entropy level,
entropy dynamics, sampled-token energy, partition energy, and top-k
distribution), top-ten pooling within each annotated step, a temporal chain
plus mutual two-nearest-neighbour weighted graph, donor-only bandwidth,
scaling and null centering, and a negative-log-conductance boundary score.
Structural trace length was excluded because it is constant within a response.

On Qwen8 late-error rows, the full graph reached Hit@1 `0.22036` versus
length-matched uniform chance `0.19708`, a grouped-bootstrap delta of
`+0.02328 [+0.00445,+0.04230]`. That apparent lift does not establish a graph-
content premise because all required negative controls contradict it:

- full graph versus chain-only: Hit@1 `-0.02329
  [-0.04004,-0.00689]`;
- full graph versus step-permuted features: `-0.02230
  [-0.04419,-0.00024]`;
- full graph versus random edges: `-0.02297
  [-0.03541,-0.01043]`;
- full-graph MRR versus chance: `-0.01475
  [-0.02652,-0.00268]`.

Equal-rank entropy-plus-graph fusion also causes supported directional harm on
the opened development population. Against entropy-top-ten, Qwen all-error
Hit@1 changes by `-0.03342 [-0.05499,-0.01309]` and MRR by `-0.06075
[-0.07397,-0.04762]`; the Llama scorer transfer repeats the Hit@1 harm at
`-0.02793 [-0.05136,-0.00493]`. The graph score is not merely an entropy
alias (Spearman `-0.1166`).

The correct interpretation is a failed **graph-content premise**: the positive
Hit@1 comparison with uniform chance is explained by topology/position bias,
because destroyed or random graph controls perform as well or better. This is
not a generic rejection caused by an interval crossing zero; the decisive
control and fusion intervals are directionally negative and exclude zero.

Consequences:

- `P3G_T1_TEMPORAL_GRAPH` is `NOT_RUN_BY_GATE / NO_PROMOTION`; the exact
  Phase-3 arm was not executed and must not receive a fabricated numeric score.
- No combined feature-by-time graph arm may open from this premise.
- The separate `P3G_F1_STG_FEATURE_SUPPORT` arm remains planned because it
  concerns sparse support among feature/family blocks rather than an
  within-answer temporal graph.
- A chain-edge change score is diagnostic only. Its MRR advantage over chance
  is inconclusive (`+0.00547 [-0.00802,+0.01942]`), so it requires a new frozen
  protocol and independent premise before any execution or fusion use.

Evidence artifact:
`outputs/step_cut_exploratory_v1/STEP_CUT_EXPLORATORY.html`, SHA-256
`b40b7791b55c8d3d2405b15eba5b603564dfae3c921f15deb74dbfbd68cf1bd2`.

## Fit and leakage contract

- All scaling, adjacency, penalties, strength, support and orientation use
  donor/calibration rows only; held responses are projection-only.
- All scorer copies of a question remain in one fold. Bootstrap units are
  whole source questions; within-answer nodes are structured observations,
  never independent samples.
- Padding, masks, binning and variable-length rules freeze before labels.
- Graph construction and step-reducer selection are separate factors.
- Every learned graph has an exact zero-strength parent alias.
- Required controls: cardinality-matched random support/graph, time
  permutation, feature permutation, and matched equal-family plus ordinary IU.
- No ProcessBench or PRMBench label selects adjacency, penalty, strength,
  support, orientation, or reducer.
- Early-detection derivatives are prefix-only and suffix-invariant; no future
  tokens, full-answer normalization, or bidirectional temporal edges.

For the still-eligible feature-support arm, promotion requires a
multiplicity-valid improvement over both the exact
parent and strongest compact reference, no material worst-cell/exact/clean
regression, and separately frozen PRMBench transfer. ProcessBench and
PRMBench metrics are never averaged.
