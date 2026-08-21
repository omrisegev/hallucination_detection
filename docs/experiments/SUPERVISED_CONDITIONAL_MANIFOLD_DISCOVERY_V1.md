# Supervised conditional manifold discovery v1

## Status and claim boundary

This protocol is frozen before the new supervised representation search is
evaluated.  It is a **supervised discovery diagnostic** on outcome-opened data,
not DUFS, not a label-free detector, and not external confirmation.  Its purpose
is to nominate exactly one frozen representation for later evaluation on new
dataset/model cells that do not participate in this search.

The previous conditional graph audit keeps its registered decision
`CONTROL_FAILURE_INVALIDATES_GEOMETRY_AUDIT`.  The present experiment does not
repair or reinterpret that decision.  It follows the descriptive observation
that corrected union-kNN, adaptive-k, and diffusion exposed similar recurrent
length-conditional geometry, while topology changes did not establish
specificity, graph health, or detector utility.

The primary estimand is Global completed-answer error.  Process-error
localization and RAG grounding hallucination remain separate estimands and are
not pooled into this discovery objective.  Future validation manifests name a
lane explicitly; a claim is lane-specific unless it passes a separately frozen
validation in every named lane.

## Discovery population and features

- Source: `results/dependency_fusion_raw/cells.npz`.
- Population: the 21 registered in-scope Global cells that contain an explicit
  `trace_length` coordinate.
- Outer grouping: the eight dataset families `triviaqa`, `hotpotqa`, `sciq`,
  `nq_open`, `squad_v2`, `truthfulqa`, `gsm8k`, and `math500`.
- Target: `1 - correctness_label`.
- Candidate features: the sorted intersection of registered `fixed_stable_v1`
  features across all 21 cells after excluding every feature whose name
  contains `length`.  Missing-feature imputation is forbidden.
- Stored cells are already standardized by their own unlabeled feature
  marginal.  Graph construction in an external validation cell is therefore a
  transductive batch operation on `X`, but validation labels never affect the
  representation or graph.

Copies from one cell never cross an outer fold because an entire dataset family
is held out.  A future manifest must provide a `group_id` when repeated source
questions or repeated generations could otherwise cross a fit/evaluation
boundary.

## Frozen representation family

The no-label baseline uses equal feature weights.  The supervised family fits a
nonnegative diagonal metric as follows:

1. Within each donor cell, cross-fit a cubic ridge model of target on held
   `log1p(trace_length)` and form target residuals.
2. Measure the absolute standardized covariance of every feature with the
   residual target.
3. Average relevance with equal dataset-family weight and equal cell weight
   inside a family.
4. Repeat with fit seeds `(17, 29, 43)` using frozen Bayesian family/cell
   weights; average the three simplex-normalized vectors for the deployed
   metric and retain the individual vectors as a stability diagnostic.
5. Evaluate fixed supports of size `{5, 10, 15, all}` obtained by target-blind
   feature-name tie breaking after donor-only relevance ranking.

Every support is a registered member of the discovery family.  No support,
weight, sign, transformation, or optimizer seed is selected using a held
family.  Per-feature nonlinear transforms, token-derived feature expansion,
and neural learners are outside v1 and require a new protocol.

## Frozen graph rule

The primary graph is the corrected self-safe local-scale union-kNN graph.  For
each representation and cell, select without labels

`k* = min {k in (3,5,7,10,15,25): largest_component >= .90 and isolated <= .05}`.

If no value passes by 25, the cell fails closed.  Health is a constraint, not a
reward.  Fixed-k sensitivities are `(5,7,10,15)`; conditional effect must be
positive for at least three of four values.  Radius, adaptive-k, and diffusion
are excluded from representation discovery because the prior audit found no
clear advantage and radius was unhealthy.

All evaluation is repeated for frozen target-blind tie seeds `(101,211,307)`.
Tie seeds are robustness dimensions, never independent statistical samples.

## Conditional geometry

Two primary length-conditional tests remain separate and co-required:

1. exact-length swaps;
2. a cross-fitted flexible propensity CRT for target given held length.

Eligibility and graph-health thresholds are inherited from the reviewed direct
audit.  The cell statistic is the normalized-Laplacian target smoothness effect;
positive values mean smoother target labels than the conditional null.  The
discovery score uses the worse of the exact and CRT effects.  AUROC never enters
feature weighting or candidate nomination.

## Grouped discovery evaluation

Primary evaluation is leave-one-dataset-family-out.  In each of eight outer
folds, all feature relevance, weights, supports, standardization beyond the
registered per-cell contract, and comparator fitting use donor families only.
Held-family labels are used once for evaluation.  Metrics are computed in every
held cell and then averaged first within family and then across families.

There is no row-random split and no concatenated out-of-fold AUROC.  Ranking
metrics are computed per held cell and averaged with equal family weight.

## Search-matched linear controls

Every support is compared with:

1. balanced L2 logistic regression fit on donor cells with equal
   cell-by-class weight;
2. `linear_score_graph`, a one-dimensional graph on that frozen logistic logit
   in the held cell, using the same label-free connectivity rule and conditional
   tests;
3. the equal-weight/no-label metric graph.

The metric representation must beat the search-matched linear-score graph to
support a local nonlinear-geometry claim.  Otherwise a positive result is
labelled `TRANSFERABLE_SUPERVISED_DIRECTION_ONLY`.

## Whole-search conditional null

Ordinary global label permutation is forbidden.  The entire outer-fold search
is rerun under both exact-length and CRT conditional-null worlds.  Each rerun
refits residual models, feature relevance, all three fit seeds, every support,
the logistic comparator, label-free `k*`, and the outer aggregation.

The search statistic is an equal-family aggregate of conditional-residual graph
smoothness, with unhealthy cells failing closed.  A deterministic target-blind
row subsample of at most 384 rows per cell is used for this max-statistic only;
the reported conditional effects use all rows.  The null statistic is the
maximum over all supervised supports, so

`p_maxT = (1 + count(max_null >= observed)) / (B + 1)`.

Discovery uses `B=199` separately for exact and CRT worlds.  A candidate that
would otherwise be sent to external validation is recalibrated with `B=999`
before validation labels are opened.  The whole-pipeline false-promotion rule
requires exact **and** CRT promotion; a constituent p-value alone cannot promote
or fail a planted control.

Planted tests include length-only target, target-independent row permutation,
and synthetic dataset/model-only worlds where the grouping is identifiable.

## Frozen internal gates

A support is an internally transferable geometry candidate only if:

- exact and CRT are eligible in at least two thirds of independent held
  environments;
- effect is positive in at least two thirds of eligible environments;
- equal-family median `min(exact_effect, crt_effect) >= .02`;
- family-bootstrap 95% lower bounds for both effects exceed zero;
- exact and CRT whole-search `p_maxT <= .05`;
- all three tie seeds pass, and at least three of four fixed-k sensitivities are
  positive;
- at least 90% of cells pass graph health;
- median cosine between fit-seed/outer-fold weight vectors is at least `.80`,
  median support Jaccard at least `.60`, and every final feature appears in at
  least 70% of outer folds.

A distinct local-manifold candidate additionally needs an equal-family
conditional-effect advantage of at least `.02` over `linear_score_graph`, with
a paired family-bootstrap lower bound above zero and maxT-adjusted `p<=.05`.

Detector utility is a separate secondary claim: LIU-minus-IU must be at least
`.005` AUROC in Global with a paired equal-family 95% lower bound above zero,
and no separately registered transfer lane may be worse than `-.005`.

## Decisions

- `CONTROL_FAILURE_INVALIDATES_SUPERVISED_DISCOVERY`
- `NO_STABLE_SUPERVISED_CONDITIONAL_GEOMETRY`
- `TRANSFERABLE_SUPERVISED_DIRECTION_ONLY`
- `INTERNAL_NONLINEAR_GEOMETRY_CANDIDATE_AWAITING_EXTERNAL_VALIDATION`

No v1 outcome may use the words universal, typical, confirmed, or validated.

## External validation contract

Discovery emits `FROZEN_CANDIDATE.json` containing one support, weights, feature
order, fit/source hashes, graph rule, tie seeds, thresholds, and comparator
parameters.  The external manifest supplies new cells with lane, dataset
family, model family, matrix, feature names, held length, binary target, and
optional group IDs.  The validator refuses a feature mismatch, a source-family
overlap declared forbidden by the manifest, or any change to the frozen
candidate.

External validation is one-shot.  Dataset-new/model-old supports only a
cross-dataset claim; model-new/dataset-old supports only cross-model transfer;
both new together is required for the strongest shared-geometry claim.

## Required outputs

- `RUN_DEFINITION.json` with source, protocol, input, and environment hashes;
- `OUTER_CELL_METRICS.csv` and `OUTER_FAMILY_SUMMARY.csv`;
- `WEIGHT_STABILITY.csv` and `CANDIDATE_SUMMARY.csv`;
- `WHOLE_SEARCH_NULL.json`, `CONTROLS.json`, and `DECISION.json`;
- `FROZEN_CANDIDATE.json` only when a candidate is nominatable;
- `REPORT.md`, static figures, and deterministic rebuild verification.
