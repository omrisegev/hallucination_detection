# Conditional Shrinkage IU: position, local token graph, and coefficient GraphTV

Registered 2026-09-13 before fitting these families. User authorized completing all three,
an independent literature/code review, and separate AIRCC submissions. This extends the
preparation-only note CONDITIONAL_IU_FOLLOWUPS_V1.md. No performance claim is made here.

## Question and deliberate limits

Can answer-local fusion benefit from unlabeled statistics borrowed from other answers,
while recovering its original weights and scores exactly when the extension is off?
Use the frozen normalized RBM12 **feature bank**, not RBM predictions as the new inputs:
entropy15, varentropy15, moment3_15, selected_surprisal, selected_squared, selected_cubed,
moment4_15, selected_power4, moment5_15, selected_power5, moment6_15, selected_power6.
Normalization remains answer-local; inactive columns are zero and excluded from fitting.

All graph nodes are tokens of the **whole answer**. Graph windows cross official step
boundaries. Position means fractional position in the whole generation, with 16 regions.
Only the final readout uses benchmark steps. No new features, RBM refits, q/alpha search,
or first_near_max. Concurrent Claude jobs and the running answer-position experiment
are independent and their code, input files, and frozen results are not modified.

## Shared baseline and the precise IU extension

For the answer-standardized active feature matrix Z (T by P), calculate population
covariance C=Z'Z/T. The existing memory-bounded Ledoit-Wolf-style calculation supplies
beta for C_base=(1-beta)C+beta diag(C). Serial tokens do not satisfy an IID claim; beta
is the existing numerical shrinkage recipe, not a new proven optimality result.

Call the canonical upcr_fit_covariance(C_base, **IU_FIT_DEFAULTS): two components,
scale_ratio=.25, L2 additive fit, g2_projection_k=1, no feature exclusion, difficulty
abstention, simple-mean fallback or automatic component selection.

Keep its estimated marginal signal vector rho and its orthonormal two-component
subspace U fixed **within this answer**, while changing the covariance in the weight
solve. This is a solve-only, LIU-style extension. It is not a fresh full IU fit at each
token, PCA renamed IU, or an estimate of conditional Cov(X,Y|position).

For a token target covariance C_target,t:

    G_t = U'[(1-alpha) C_base + alpha C_target,t]U + 1e-12 I
    r = U' rho
    theta_t = G_t^-1 r
    w_t = orientation * U theta_t
    score_t = Z_t w_t

alpha=.25 is a frozen engineering constant, separate from beta. The baseline orientation
is anchored once per answer to varentropy15 and never re-estimated per token. At alpha=0,
return the original IU weights and token scores array-exactly. Also test the general
quadratic and GraphTV solutions at zero and continuity for small positive alpha.
The objective is a **fixed-signal quadratic surrogate**: covariance borrowing alone
does not establish that marginal rho is the correct conditional signal vector.

## Unlabeled training priors and fair controls

Each answer supplies 16 weighted population covariances, centered WITHIN its regions.
Token/region overlap is fractional, without zero padding. Average those covariances
equally across source groups and then equally across answers in each group. Fit separately
per benchmark cell and excluded fold set. The pooled control is the equal-region average
of exactly these same covariances; it must not add variance of region means.

The position arm uses the region covariance at each token's relative answer position,
with overlap weighting for tokens crossing a region boundary. Its scale-only control
projects each new weight vector onto the baseline direction:

    gain_t = (w_t' w_base)/(w_base' w_base)
    w_scale,t = gain_t w_base

Thus a position-dependent amplitude change is not mistaken for changed relative feature
weights. Report gains (including signs), orthogonal coefficient change, projected covariance
scale and condition numbers. The shuffled-position control uses a fixed UID-seeded token
position assignment permutation in training AND evaluation. Features and labels stay put.

## Three registered families

| Family | Arms | Primary contrast |
|---|---|---|
| Position-conditioned IU | baseline, pooled, position, position scale-only, shuffled position | position minus scale-only |
| Graph-local IU | baseline, pooled, sliding-window, graph-local, shuffled graph-local | graph-local minus sliding-window |
| Coefficient GraphTV | baseline, position, position scale-only, GraphTV, shuffled GraphTV | GraphTV minus position |

All panels also contain original answer-local RBM12 Logit/Top10 and the 13 frozen
reference rows. Secondary contrasts: pooled versus baseline, position versus pooled
where available, real versus shuffled graph/position, and each arm versus RBM12.

### Graph-local covariance

Candidate edges connect tokens within +/-16 in whole-answer token order. Affinity is
exp(-distance(i,j)^2/(sigma_i sigma_j)) in the frozen active standardized feature space.
sigma_i is distance to the eighth in-window nonself neighbor, or the furthest available
one for short answers; floor=1e-8. Duplicate neighbors count. Underflow edges are zero.
The sliding-window control uses the same candidate window with edge weights one.

The covariance neighborhood includes the token itself with weight one. Define the
weight-concentration count n_w=(sum weights)^2/sum(weights^2), not independent sample size.
For gamma=P/(P+n_w-1), the target is (1-gamma) C_neighborhood + gamma C_train,pooled.
Use the common conditional solve above, with alpha=.25. This tests a complete local
estimation recipe: affinity changes effective weight count AND shrinkage, so its contrast
does not isolate geometry alone. Shuffling node assignments preserves the graph's edge
weights/degree multiset but destroys locality as well as feature-neighborhood alignment.

### Graph regularization on coefficient vectors

Use the same position G_t and r as the unpenalized position arm. Minimize the convex
objective over two-dimensional coefficient coordinates:

    sum_t [ .5 theta_t' G_t theta_t - r' theta_t ]
       + eta sum_(i,j) edge_weight_(i,j) ||theta_i-theta_j||_2

eta=.1. Each undirected affinity edge occurs once; normalize edge weights to sum T/2.
This is the UNSQUARED L2 Network Lasso penalty. Because U is orthonormal and constant
within the answer, the distance between theta vectors equals feature-weight distance.
We regularize coefficients, not neighboring output scores. Spikes may still be weakened,
so gains/losses in first-error detection remain an explicit outcome.

Chambolle-Pock primal-dual solver: max_iter=3000; relative primal-dual gap tolerance=1e-6;
step sizes satisfy the incidence-operator norm bound. Report objectives, gap, iteration
count, and convergence. Finite capped solutions remain scored and clearly identified as
budget-limited. Invalid/numerical failures receive NaN scores; no silent replacement.
eta=0 must equal the position arm. alpha=0 gives the same constant baseline optimum
even at eta>0, tested through the general solver as well as the exact production bypass.

## Contract, calibration, failure policy, and reporting

Full population: 13,769 answers, PB 6,800 in eight cells and PRMB 6,969 in one cell.
Frozen v3 annotations, v2 source-group folds, step boundaries and IDs. Token scores feed
Top10 mean per step (all tokens for short steps), then argmax with earliest tie.
The external raw-entropy q=.3 gate and saved thresholds remain unchanged.

All priors are fit outside the test source-group fold. PRMScore q=.8 calibration uses
inner predictions: when calibrating outer fold f using answer fold h, its prior excludes
both f and h. Answer-local baseline/reference scores need no external inner refit.
Targets/labels never enter prior or graph fitting; sanitize metadata before training.
This is unlabeled fusion **with access to other answers**, distinct from the answer-local
baseline; external gate and PRMScore calibration are disclosed separately.

Report PB all-eight/Q4/Q8/per-cell, clean accuracy, exact first error, early/late and
gate-suppressed exact peaks. Report PRMB within-answer AUC, pooled AUC, mean fold AUC,
PRMScore, denominators and failures. Common-cohort uncertainty must state its coverage.
No calculation failure disappears: it remains invalid in fixed benchmark denominators;
undefined AUC/calibration is explicitly unavailable, never silently imputed as correct.

Use 10,000 paired canonical-source-group bootstrap draws. Three registered primary
comparisons receive 98.3333% intervals (1-.05/3) per endpoint; secondary intervals are 95%.
This does not give simultaneous coverage across all reported endpoints. The intervals are
conditional on saved fits/predictions and are development evidence, not a research-wide
selection correction or validation on untouched data. There is no performance cutoff.

## Literature: what is borrowed and what is our adaptation

| Primary source | Used here | Not claimed |
|---|---|---|
| [Multi-Target Shrinkage](https://arxiv.org/abs/1412.2041), Bartz et al. | Borrow a covariance target from related populations | We do not fit their optimal multi-target QP or inherit their asymptotic/IID guarantees |
| [Self-Tuning Spectral Clustering](https://proceedings.neurips.cc/paper_files/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf), Zelnik-Manor & Perona | Locally scaled affinity | No clustering, no proof this graph encodes correctness; local candidate window is our choice |
| [Network Lasso](https://web.stanford.edu/~boyd/papers/network_lasso.html), Hallac et al. | Convex node losses plus unsquared edge L2 coefficient penalty | We use a primal-dual solver rather than the paper's distributed ADMM |
| [Chambolle & Pock](https://doi.org/10.1007/s10851-010-0251-1) | Primal-dual updates and a step-size bound | Convergence is measured numerically; likelihood/task improvement is not guaranteed |

## Acceptance and execution

Before cluster submission: synthetic mathematical fixtures and independent code/literature
review. On the cluster, source-byte validation and actual-data smoke precede the full run.
Smoke 27 is feasibility only; the
supervisor starts the full population only after its smoke review passes. Full evaluation
includes independent endpoint reconstruction and all frozen-reference reproduction.
Three isolated jobs/output directories, frozen code, one BLAS thread each, CPU-only AIRCC,
checkpoint every answer and capped invocations resumed safely. Reuse the already verified
remote data read-only; do not duplicate large caches locally or change running job code.

Return chat first, then CSV/JSON and short HISTORY/PROGRESS/Research_Directions updates.
No HTML and no automatic move to another research family on completion.
