# Reasoning Localization 0.3662 — STG-SU / Temporal-Graph Transfer V1

Status: `PLANNED`; survivor-gated Phase-3 direction; no localization result.

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

Promotion requires a multiplicity-valid improvement over both the exact
parent and strongest compact reference, no material worst-cell/exact/clean
regression, and separately frozen PRMBench transfer. ProcessBench and
PRMBench metrics are never averaged.
