# Token-local fusion optimization v1

**Status:** the Phase-1 implementation and target-free deterministic fit are
complete. The corrected score freeze must pass an independent pre-label audit
before target opening. No ProcessBench or PRMBench targets had been opened
under this protocol as of the score-freeze boundary.

**Branch:** `codex/token-local-fusion-optimization-v1`

**Worktree:**
`local_cache/worktrees/token_local_fusion_optimization_v1`

## 1. Why this experiment exists

A read-only reconstruction on the eight Qwen ProcessBench cells isolated the
global and local fusion mechanisms while holding the feature contract and the
global-local decision rule fixed.

| Global head | Local token head | ProcessBench macro F1 |
|---|---|---:|
| IU-PCR | IU-PCR | 30.8570% |
| equal mean over 30 response features | IU-PCR | 30.7052% |
| equal mean over 30 response features | equal mean over 29 token streams | 26.2942% |
| n/a | equal token mean alone | about 23.38% |

With 20,000 paired source-question bootstrap draws:

- equal/equal minus IU/IU was `-4.5628` F1 points, 95% CI
  `[-6.0678, -3.0624]`;
- equal-global/IU-local minus IU/IU was `-0.1519` points, 95% CI
  `[-0.6132, +0.3005]`;
- equal/equal minus equal-global/IU-local was `-4.4110` points, 95% CI
  `[-5.8843, -2.9406]`.

The supported interpretation is narrow but important: on this population,
the response-level fusion is close to saturated, while the token-level fusion
still supplies about 4.4--4.6 F1 points. This is retrospective development
evidence from already opened populations, not confirmation.

## 2. Fixed task and data contract

### ProcessBench primary task

- First-error localization on the same eight Qwen cells used by the modern
  aligned reconstruction.
- Five group-safe cross-fitting folds. A source question and all of its scorer
  copies stay in one fold.
- The primary global head is the equal mean of the 30 standardized
  response-level mixed-v2 features. This intentionally removes global-fusion
  complexity from the local-fusion ablation.
- A token head produces one risk value per token. The primary step reducer is
  the maximum token risk within the supplied step boundary.
- Response and step risks are combined exactly as
  `sqrt(midrank(response_risk) * midrank(step_risk))`.
- The operating threshold is cross-fitted after score freeze. If the maximum
  step score is not strictly above the threshold, the prediction is no error.

### PRMBench secondary transfer task

- Every-step error ranking remains a separate estimand.
- Report step AUROC and AUPRC. Do not combine them with ProcessBench F1 or use
  PRMBench to select the ProcessBench method.

### Frozen token input

Every local method receives exactly the same 29 token streams from
`spectral_utils.fixed_application_pipelines`: 28 token-resolved mixed-v2 views
plus constant trace length. CUSUM magnitude and location are two answer-level
reductions of one token stream, so there are 29 streams rather than 30.

All methods must reuse the incumbent token cap, finite-value imputation,
nondegenerate-coordinate mask, donor-only standardization, token-to-step
reducer, response/local rank combination, and threshold evaluator. A method
may change only the token fusion or a declared token-input transform.

## 3. Exact comparators

### Equal-local comparator

Use the same fit rows, imputation, keep mask, and standardization as the token
IU incumbent. Replace the fitted IU vector by a uniform mean across the kept
standardized confidence streams. Token risk is the negative equal mean. Keep
the maximum step reducer, global-local combination, and threshold fit
unchanged.

### Local IU-PCR incumbent

Mirror `_fit_token_iu` in
`spectral_utils/reconstruction_benchmark/localization_fit.py` exactly:

- L2 U-PCR;
- two components;
- `g2_projection_k=1`;
- `scale_ratio=0.25`;
- no exclusion, difficulty gate, recomputation, or simple-average fallback;
- label-free orientation against the equal-mean confidence anchor.

This modern 29-stream token IU is not the historical GL-LIU five-view
temporal head. GL-LIU remains a separate reference row.

## 4. Registered local-fusion ladder

The experiment proceeds in this order. A later arm must not silently change
the input matrix, reducer, global head, or evaluator.

1. `LOCAL_EQUAL29`: exact equal-local comparator.
2. `LOCAL_EQUAL_FAMILY`: equal mass for source/provenance groups, with constant
   trace length treated as context rather than an equal expert.
3. `LOCAL_IU29`: exact incumbent alias.
4. `LOCAL_SU29`: the existing fixed low-rank-plus-sparse SU-PCR reproduction.
5. `LOCAL_STG_SU29`: an STG gate learns a stable sparse error-covariance
   support without labels; SU-PCR is then fitted on that support.
6. `LOCAL_DUFS_LIU29`: target-free DUFS soft gates followed by the same local
   LIU solver and fixed settings.
7. `LOCAL_TOKEN_B3`: a continuous B3 energy model fitted directly to token
   rows, with source-question-safe fitting and unchanged downstream reducer.
8. `LOCAL_TOKEN_CIW_B3`: a token-native CIW input layer followed by unchanged
   token B3. It must operate on token trajectories themselves and must not
   reuse completed-response CIW risk as a token gate.

### 4.1 Frozen Phase-1 details (added before any Phase-1 fit)

The six local heads share one immutable preparation object.  The stream order
is `SHARED_TOKEN_VIEWS`; the 60,000-token evenly spaced cap, fit medians,
nondegenerate mask, population mean/standard deviation, and application-time
fallback are reproduced from `_fit_token_iu`.  Step scores are always span
maxima.  No method may construct its own fit-token sample or standardization.

`LOCAL_EQUAL_FAMILY` uses the five non-structural `specrage_views` provenance
families: entropy level, entropy dynamics, sampled-token energy, partition
energy, and top-k distribution.  Each present family receives mass `1/5` and
its members share that mass equally.  The constant trace-length stream remains
in the common prepared matrix but receives zero *local-expert* weight in this
arm.  It is response context, already present in the fixed equal-30 global
head; counting it as a sixth local sensor would let one response-level scalar
compete with five token-resolved evidence sources.

`LOCAL_SU29` uses the existing `sparse_upcr_fit` defaults exactly: rank two,
two PCR components, projection `k=1`, `scale_ratio=0.25`, 300-point `g2` grid,
threshold multiplier one, 100 outer and 40 completion iterations, tolerance
`1e-8`, no sparse-fraction cap, and no label-bearing input.

`LOCAL_STG_SU29` learns gates over the upper-triangular covariance-error
pairs.  Five deterministic opaque-row folds and seeds `(11,23,37)` are used.
For each donor/held fold, the donor covariance is decomposed into a rank-two
completion plus its off-diagonal residual.  Stochastic hard-sigmoid gates
(`sigma=0.5`, Adam learning rate `0.05`, 120 epochs) predict the held covariance
from that donor low-rank matrix and residual.  The frozen penalty roster is
`(0.10, 1.0, 3.0, 4.0, 5.0)`; choose the lowest held-covariance-error penalty
whose *uncapped* consensus support already satisfies the SU sparse-support
theorem, and fail closed if none does.  A pair enters the consensus support
only when mean survival probability is at least `0.75` and at least three of
five fold means cross `0.75`.  No post-selection truncation is allowed. SU-PCR
is then refit with that topology fixed; no correctness label participates.
Before real fitting, a rank-two null must return empty support and a planted
three-pair world must recover at least two pairs deterministically.

This feasibility-constrained roster replaces an initial target-free smoke in
which penalties `(0, 0.01, 0.03, 0.10)` saturated the theorem cap in all nine
fit cells.  That smoke was rejected before any target import because truncating
a dense residual is not evidence of learned sparse support.  The rejected
score-freeze hashes are retained under `superseded_dense_support_*` and cannot
enter evaluation.

`LOCAL_DUFS_LIU29` freezes the historical local settings: parameter-free DUFS
soft gates with seeds `(11,23,37)` and 80 epochs, self-tuning symmetric `k=7`
graph, and Laplacian-IU `lambda=0.3`.  The IU moment estimate, full coordinate
pool, two-component projection, and all other IU settings remain unchanged.
At `lambda=0` the implementation must be a byte-exact IU weight alias.

The `LOCAL_IU29` end-to-end token-risk alias uses the repository's existing
cross-scale mechanical criterion: maximum absolute error at most `1e-12`.
Byte equality is also recorded.  This distinction permits harmless BLAS
reduction-order differences while rejecting any scientifically meaningful
change.

Two negative controls are frozen before targets open: a deterministic
feature-label permutation of the learned STG support (seed `2026082801`) and a
deterministic DUFS graph-node permutation (seed `2026082802`).  These controls
preserve the relevant support count or graph spectrum while breaking its
alignment to the token coordinates/rows.  They are diagnostic and cannot be
promoted.

The token-CIW primary input hypothesis uses the universal 3-by-3 core:

- sources: predictive entropy, sampled-token surprisal, raw partition energy;
- operators: level, sliding variance, absolute CUSUM;
- each coordinate is predicted only from the four coordinates sharing its
  source or operator;
- group-held-out R-squared estimates predictable structure;
- the innovation is the held residual divided by its donor residual scale;
- all non-core token streams bypass the transform unchanged.

This tests whether predictable shared trajectory should be separated from
local innovation before energy fusion. It is distinct from the completed
cross-scale CIW experiment, which used whole-answer means and response risk.

## 5. What “optimization” means here

No ProcessBench or PRMBench target may select a gate, support, penalty,
component count, blend, or reducer.

- Freeze a small candidate roster before target evaluation.
- Fit IU/SU/STG/DUFS/B3 parameters using target-free token matrices only.
- Select STG support and any regularization by grouped held-out covariance or
  reconstruction stability, plus prespecified synthetic recovery/null tests.
- Keep the CIW innovation blend bounded and nested: zero blend must reproduce
  its frozen base exactly. Any tested maximum blend values are separate frozen
  candidates, not an AUROC/F1-tuned path.
- Keep `max` as the primary step reducer. `top-k mean` and onset/persistence
  reducers are secondary, preregistered ablations and may not replace the
  primary after labels open.
- A supervised local logistic/PRM head may be reported only as an explicit
  ceiling and may never choose the unsupervised method.

## 6. Detector and locator must be reported separately

For ProcessBench report all of:

- official macro F1;
- exact first-error localization on error traces;
- within-one-step localization;
- clean-trace abstention;
- overall decision accuracy.

Exact and within-one measure the locator. Clean abstention mainly measures the
detector/calibration policy. A gain in one does not establish a gain in the
other.

## 7. Comparisons and decision rule

Every candidate is compared on identical rows with:

- equal-global + equal-local;
- equal-global + local IU incumbent;
- IU-global + local IU incumbent;
- Mind the Gap under the same historical split/threshold protocol;
- historical GL-LIU as a separate, non-identical localizer.

The primary promotion contrast is equal-global + candidate-local versus
equal-global + local IU. A candidate advances only if all of the following
hold on the registered ProcessBench analysis:

1. macro-F1 delta is at least `+0.005` absolute;
2. paired grouped 95% CI lower bound is above zero;
3. at least 6 of 8 cells are wins or ties (`0.0005` tie tolerance);
4. worst-cell F1 delta is at least `-0.02`;
5. exact and within-one each regress by no more than `0.005` absolute;
6. clean abstention regresses by no more than `0.01` absolute;
7. the method still beats equal/equal by at least 3 F1 points with a positive
   paired lower bound;
8. mechanical identity, determinism, score reconstruction, health, and label
   firewall checks all pass.

PRMBench is a secondary transfer guard: require step-AUROC delta versus its
frozen incumbent of at least `-0.002` and disclose AUPRC. It cannot rescue a
failed ProcessBench primary.

If no method passes, retain local IU29. A point-estimate winner below the
threshold is exploratory, not the new method of record.

## 8. Retrospective boundary and confirmation

All current ProcessBench and PRMBench labels have historical exposure. The
eight-cell screen can compare mechanisms and estimate effect size, but cannot
confirm a newly selected winner.

Before opening targets:

1. freeze the method roster, hyperparameters, reducers, metrics, and decision
   rule;
2. fit all candidates in a target-free process;
3. write immutable score artifacts with source, bundle, row-order, config,
   environment, and code hashes; the environment binding must be a canonical
   SHA-256 over Python/platform identity, dependency versions, and numerical
   thread-limit variables, repeated identically in every cell record;
4. complete an independent pre-label audit;
5. only then import labels in a separate evaluator.

Promotion beyond “retrospective challenger” requires a new model population or
new first-error localization dataset that was not used to design the method.
Leave-one-family-out analysis on opened data is a robustness check, not fresh
confirmation.

## 9. Implementation order for the next session

1. Read `CLAUDE.md`, `PROGRESS.md`, this protocol, and the incumbent source.
2. Verify the clean worktree and preserve the dirty CIW worktree untouched.
3. Build one shared token preparation/result interface and exact aliases for
   `LOCAL_EQUAL29` and `LOCAL_IU29`.
4. Add `LOCAL_SU29`, `LOCAL_STG_SU29`, and `LOCAL_DUFS_LIU29` one at a time,
   with mechanical tests and a target-free smoke after each method.
5. Freeze and run the linear/spectral ladder before implementing token B3.
6. Only after interpreting that result, implement `LOCAL_TOKEN_B3` and
   `LOCAL_TOKEN_CIW_B3` as the second stage.
7. Run an independent code/protocol audit before any label import.
8. Do not merge, push, or mutate Drive without explicit authorization.
