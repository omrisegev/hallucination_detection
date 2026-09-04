# Joint L-SML localization optimization plan v1

Status: `DRAFT_RETROSPECTIVE_DEVELOPMENT_PLAN__NOT_REGISTERED__NOT_RUN`

Date: 2026-09-04

## Purpose and authority boundary

The frozen Joint L-SML candidate failed on both opened localization tasks. This
document specifies a fair method-development experiment to learn why and to
construct at most two successor candidates. It does not reinterpret the failed
candidate, authorize promotion, or make the ProcessBench/PRMBench populations
fresh again.

The latest user instruction authorizes retrospective optimization on the
already-opened data. It supersedes the earlier no-more-scoring restriction only
for a separately registered, nested source-group development study. It does not
authorize a new-leader claim, an unbounded sweep, cluster generation, or reuse
of the same population as confirmation. Confirmation still requires a new
population and a separately frozen protocol.

ProcessBench and PRMBench remain different objectives:

- ProcessBench: cross-fitted first-error macro-F1, with Qwen3-4B/Qwen3-8B
  source-question pairing and equal subset/model aggregation.
- PRMBench: step-ranking AUROC, with problem-level folds and official spans.

Their metrics will never be averaged to choose a method.

## Failure diagnosis that motivates the plan

The frozen diagnostic in
`results/joint_lsml_existing_localization_v1/failure_diagnostic_v1/` supports
two linked mechanisms.

### 1. Cross-cell scale failure on ProcessBench

The final hierarchical map is

\[
w_i = v_i c_{g(i)},
\]

where `v` is the fitted global loading and `c` is a second SML solution over
virtual group classifiers. The implementation orients this vector but does not
normalize its final norm or donor fused-score variance. Joint head L2 norms
therefore differ substantially across cells. The amendment also places the
unit-norm `G=[]` flat-SML fallback into the same Qwen3-4B threshold pool as
larger-norm Joint heads.

This explains the large ProcessBench collapse: q4/MATH and q8/GSM8K retain
high detector-rank correlation and locator agreement with fixed L-SML, but the
shared threshold activates only a small fraction of their responses.

### 2. Structural objective/deployed-head mismatch

The covariance model fits both the global factor `v` and group factors `u`:

\[
\widehat\Sigma = vv^\top + \sum_\ell
\operatorname{diag}(u^\ell)G^\ell\operatorname{diag}(u^\ell)
+ \operatorname{diag}(d).
\]

The deployed hierarchical head does not use the fitted `u` directly. They
affect the score only indirectly through the fitted `v`; a new SML stage then
constructs the cross-group weights. Consequently, lower covariance misfit is
not the objective of the scorer actually used at localization time.

PRMBench is important here: there is no pooled threshold, yet Joint still loses
slightly to IU-PCR and fixed-family L-SML. Scale normalization alone is
therefore necessary for ProcessBench comparability but is not a complete
method fix.

### 3. What is not yet identified

Every fitted cell chose INTERNAL K=3 and the learned partitions have low ARI
with the five provenance families. But the frozen run did not score ordinary
continuous L-SML with those exact INTERNAL groups. We therefore cannot tell
whether grouping or the hierarchical map is the main causal branch. The next
experiment must cross those two choices explicitly.

## Engineering invariant before any tuning

Every learned weight map must be put on one donor-frozen score scale before a
cross-cell threshold can be fitted.

Primary convention:

\[
\widetilde w = w / \operatorname{SD}_{\text{fit population}}(Xw).
\]

The sign is fixed by the existing confidence anchor. In ordinary outer source-
group CV, the fit population is the outer-training rows and the scale is
applied unchanged to held-out rows. The separate transductive subset-transfer
audit defines its target-free fit population explicitly below. If the fit-
population score standard deviation is non-finite or below `1e-8`, the fit is
inadmissible. This is an estimator invariant, not a label-selected
hyperparameter.

The donor-score-SD convention applies to every learned arm, including IU and
ordinary continuous L-SML. Unit-L2 normalization is retained only as one
prespecified scale-sensitivity diagnostic because fixed-family continuous
L-SML currently has that convention; it is never the deployed scale in this
study. Fallback and Joint heads may never share an absolute threshold without
the same donor-score normalizer.

## Stage 1: grouping x map mechanism experiment

This stage isolates the earliest unresolved branch with a 2 x 2 design:

1. Grouping:
   - `PROVENANCE5`: the frozen five feature families.
   - `INTERNAL`: label-free residual-graph grouping, K chosen only from
     outer-training source groups.
2. Weight map:
   - `CONT_LSML`: the maintained continuous L-SML map, followed by the common
     donor-score-SD normalization.
   - `HIERARCHICAL_JOINT`: the current global-loading x cross-group map, with
     the donor-score scale invariant above.

The four cells are supervised nested-CV mechanism diagnostics, not four
label-free outer-test candidates. Inner localization labels may estimate which
branch has task value; such a result is a method-development ceiling, not a
label-free selector.

The primary label-free successor is fixed before those outcomes:
`INTERNAL + CONT_LSML + TF_SR + donor_score_sd_1`. INTERNAL K is selected only
by the structural stability/null rule below. No structural statistic chooses
between CONT_LSML and HIERARCHICAL_JOINT, because the completed diagnostic has
already shown that covariance misfit is not an efficacy surrogate.

Required diagnostics per outer fold:

- partition K, group sizes, held-source stability, affinity energy and a
  degree/weight-matched null;
- covariance misfit and convergence, but never as the efficacy selector;
- minimum pairwise score-map Spearman;
- weight L1/L2/effective count, donor score SD and held-cell score SD;
- selected-map stability across inner folds;
- ProcessBench error/clean accuracy and activation by cell, or PRMBench ranking
  metrics, only at the registered evaluation stage.

K remains in `{3,4,6,8}`. It is selected by median held-source co-assignment
ARI, then mean ARI, then smaller K. A candidate is rejected when residual
affinity is numerically degenerate, when its observed stability does not exceed
the frozen rewiring-null gate, or when group-size/admissibility requirements
fail. Deterministic merging of small groups may be studied only as a separately
enumerated inner configuration; it may not be invoked after seeing a held-fold
failure.

## Stage 2: fusion order

Use equations in artifacts because earlier project prose used
"trajectory-first" inconsistently.

### Token-fuse then step-reduce (`TF_SR`)

This is the current path:

\[
s_t = f(X_{t,1},\ldots,X_{t,p}),\qquad
r_j = \operatorname{reduce}_{t\in\text{step }j}(s_t).
\]

ProcessBench reduction is the fixed top-`min(10, step_length)` mean, with
answer detector `max_t`; PRMBench is maximum fused risk inside the official
step span.

### Per-feature step-reduce then fuse (`FR_SF`)

\[
z_{j,i} = \operatorname{reduce}_{t\in\text{step }j}(X_{t,i}),\qquad
r_j = f(z_{j,1},\ldots,z_{j,p}).
\]

For ProcessBench, each feature is reduced to its fixed top-`min(10,
step_length)` mean inside each step; the fused step risks then define both
answer detector `max_j r_j` and locator `argmax_j r_j`. This intentionally
differs from the current max-token answer detector and is reported as a fusion-
order intervention, not an exact detector alias. For PRMBench, each feature is
reduced by max inside the official step span before fusion.

The reducer is applied identically to every feature before fusion. All
preprocessing, grouping, DUFS gates if present, and weights are fitted only on
outer-training steps. Held-out steps are projection-only. No response outcome
or first-error label enters an unsupervised fit.

The older project result described in HISTORY Step 253 fitted IU on token
trajectories and reduced afterward; it supports retaining `TF_SR`, but does not
answer this explicit order comparison for Joint L-SML.

## Equal optimization budgets

"Equal budget" means the same maximum number of preregistered configurations,
outer folds, inner folds, seeds, and candidate slots—not an unlimited parameter
sweep for either method.

### IU/U-PCR family: exact eight inner configurations

Use only repository-supported choices. Freeze these eight rows before the
runner exists; all unlisted gates/fallbacks remain off:

1. `(components=2, scale=.25, loss=l2, exclusion=off)` — deployed IU.
2. `(components=1, scale=.25, loss=l2, exclusion=off)`.
3. `(components=2, scale=.10, loss=l2, exclusion=off)`.
4. `(components=1, scale=.10, loss=l2, exclusion=off)`.
5. `(components=2, scale=.25, loss=l1, exclusion=off)`.
6. `(components=1, scale=.25, loss=l1, exclusion=off)`.
7. `(components=2, scale=.25, loss=l2, exclusion=on, refit=on)`.
8. `(components=1, scale=.25, loss=l2, exclusion=on, refit=on)`.

This is a balanced eight-configuration budget, not a claim that every
interaction has been exhaustively searched. No feature family, orientation, or
reducer may be selected from labels.

### Joint family: maximum eight inner configurations

- grouping in `{PROVENANCE5, INTERNAL}`;
- map in `{CONT_LSML, HIERARCHICAL_JOINT}`;
- order in `{TF_SR, FR_SF}`.

Group-size handling and K selection are parts of the registered INTERNAL rule,
not additional post-hoc variants. Covariance-inverse maps remain diagnostics
unless they replace one of these eight before registration.

### Baselines, diagnostics and candidate slots

The core study reports exactly:

1. equal weight over all active-23 features;
2. equal-family averaging;
3. deployed IU-PCR;
4. fixed-family continuous L-SML;
5. best nested-CV IU configuration as the equally tuned comparator;
6. the single fixed label-free Joint successor;
7. one inner-label-selected Joint ceiling, explicitly diagnostic and ineligible
   for promotion.

The first four are controls. Equal-all23 is new to this active-23 efficacy
comparison and must be frozen before evaluation; no historical score should be
invented for it. The ceiling is recorded in the cumulative exposure ledger but
does not occupy a promotion-candidate slot. The cumulative successor budget is
two: the fixed label-free core candidate and, only in a later registered study,
one optional DUFS candidate.

This seven-row retrospective comparison is a narrow override of the earlier
one-candidate/three-baseline opened-data ledger, explicitly authorized by the
latest user request for equal-all23, fusion-order and equally tuned IU/Joint
comparisons. It does not alter the frozen Step-347/348 result family.

## Nested source-group evaluation

Use five outer folds and five inner folds with fixed namespaces and seeds.

For each outer fold:

1. Hold out complete source-question/problem groups.
2. Recompute imputation, standardization, DUFS if used, covariance, grouping,
   weights, and donor-score scale on outer training only.
3. Use the fixed inner folds to select one IU configuration and one Joint
   ceiling separately for PB and PRM; the primary Joint route remains fixed.
4. Project the selected heads once onto the untouched outer fold.

For ProcessBench, q4/q8 copies of a source question must share their fold. In
addition, a prespecified leave-one-subset-pair-out scale-transfer audit withholds
both q4 and q8 cells of one subset from outcome-based threshold calibration.
The held pair may fit its weights, grouping and score-SD normalizer from its own
target-free telemetry, matching the project's transductive label-free fitting
contract; it may not use first-error outcomes. The threshold learned on the
other three subset pairs is then applied unchanged. This tests whether the
donor-SD convention makes scores comparable to an unseen labeled domain while
preventing q4/q8 copies from leaking across the holdout.

For PRMBench, use problem-level folds and official step spans. PB and PRM select
separate configurations; a joint winner exists only if the same frozen method
passes each task's gate independently.

The fixed primary label-free candidate is evaluated in the outer folds without
inner label selection. Separately, inner labels may select one of the eight
Joint configurations to estimate a `SUPERVISED_METHOD_DEVELOPMENT_CEILING`.
Fitting remains label-free at application time, but the ceiling's method
selection is not label-free and cannot become the reported unsupervised winner.

All outcomes on the present Qwen populations are
`RETROSPECTIVE_OPENED_DEVELOPMENT`, even under nested CV.

## DUFS: useful role and prohibited role

DUFS cannot select K. It returns one gate per feature; it does not estimate a
partition or the number of groups. Equating the number of open gates with K
would conflate feature support with dependence structure and contradict the
existing adaptive-K failure in Research_Directions.

The safest reusable adapter is
`spectral_utils.adapted_dufs.adapted_dufs_soft_gates`, which returns
parameter-free seed-averaged gates and diagnostics without importing the
historical label-prior controls. The exact historical hard rule is
`spectral_utils.selectors.a2_groupfs.dufs_pf_gates` with signed `mu > 0`.
Do not import the omnibus `a2_groupfs` runner into the localization fit.

The latest user request narrowly reopens DUFS for this one question despite the
Step-343 stop on further DUFS family/context searches. That override does not
reopen family sweeps, hard pruning, or PRMBench transfer of the old DUFS arms.

DUFS does not enter the core eight-versus-eight study. It may enter one
separately registered second-stage successor study only if the grouping x map
result leaves an INTERNAL-affinity stability problem worth testing:

- `SOFT_PF_AFFINITY`: compute parameter-free gates `q` inside each training
  fold and use
  \[
  A' = \operatorname{diag}(q)\,|R-vv^\top|\,\operatorname{diag}(q)
  \]
  for INTERNAL clustering. Do not delete features. K is still selected by the
  same stability/null rule.

This is a new graph heuristic, not an L-SML theorem. The second-stage study
must replace rather than append to one selected Joint route, retain the matched
plain-INTERNAL route, and keep an eight-configuration ceiling. It receives one
of the two Joint candidate slots and advances only if gate vectors are
seed-stable, affinity is nondegenerate, and every fold has admissible groups.

A hard PF-DUFS prefilter is a later ablation, not the primary path. If tested,
it must preserve the semantic anchor, retain at least nine features, fail closed
without repair/fallback, and apply the identical selected support to an IU arm
so feature selection is separated from Joint fusion.

Historical evidence makes DUFS secondary: matched localization DUFS arms were
effectively tied with IU, earlier hard filtering removed complementary
covariance, and parameter-free DUFS improved implementation hygiene rather than
established localization efficacy. Its value here is a stable relevance
diagnostic for the residual graph, not a shortcut around K selection.

## Decision gates

Before outer label evaluation, require in every fitted fold:

- finite scores and preprocessing parameters;
- donor fused-score SD exactly one within numeric tolerance;
- held ProcessBench subset-pair normalizers use only target-free telemetry and
  never its outcomes or threshold fit;
- grouping group sizes and held-source stability admissible;
- nondegenerate affinity and stability above the frozen null criterion;
- final weight-map score agreement and effective-count bounds fixed before the
  run;
- no fallback that changes the score scale or estimator identity.

Development support on ProcessBench requires the fixed label-free successor to
improve over the best nested-CV IU configuration and show no material loss to
deployed IU, fixed-family continuous L-SML, or either simple averaging control.
PRMBench requires the same logic using AUROC. The two task-specific primary
contrasts form one intersection-union decision: both must pass and the metrics
are never averaged. Paired uncertainty uses one shared source-group/problem
resampling stream. The supervised ceiling and all factorial contrasts are
secondary descriptive exposures. If the optional DUFS study later opens, its
new task contrasts are added to the cumulative multiplicity ledger before any
label access.

Failure on either task keeps the method in development. Passing the current
opened populations permits only a frozen fresh-data test; it is not promotion.

## Implementation order

1. Add a normalized weight-map adapter and exact donor/held projection tests.
2. Add the grouping x map 2 x 2 mechanism runner without labels.
3. Add `TF_SR` and `FR_SF` adapters with exact reducer aliases.
4. Add the fold-contained PF-DUFS soft-affinity adapter and stability tests.
5. Freeze the eight-versus-eight configuration budget, folds, seeds, candidate
   slots, power/multiplicity ledger, and all source/runtime hashes.
6. Run score freeze, independent audit, then the retrospective nested-CV
   evaluation.
7. If supported, freeze a separate fresh-population protocol before generation.

No optimization experiment has been run by this document.
