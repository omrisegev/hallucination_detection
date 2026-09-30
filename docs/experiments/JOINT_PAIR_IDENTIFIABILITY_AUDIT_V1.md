# Pair groups inside our Joint L-SML fusion

2026-09-07. Mathematical and unlabeled structural audit; no new quality claim.

## Decision question

Claude's v2 report proposes lowering the minimum feature-group size from
three to two. Does that preserve identifiable native covariance/weights,
and can an explicit pair representation make the extension well defined?
Do not change the frozen legacy kernel, its scores, or Claude's worktree.

## Distinctions to verify

In a pair, the residual off-diagonal covariance supplies one equation
r = u_i*u_j. Changing u_i -> t*u_i and u_j -> u_j/t preserves r.
Equal-magnitude loadings are a convention, not unique identification.
If both residual variances remain nonnegative, adjusting the diagonal
noise keeps the complete observed covariance unchanged. But the current
kernel clips negative diagonal noise, which can make the covariance and
native inverse depend on that arbitrary representative. The profiled
global-loading Jacobian and current multistart checks do not by themselves
test this full-head invariance.

For a fixed fitted global loading v, b_i = S_ii - v_i^2 and likewise b_j.
The pair can fit r with nonnegative diagonal noise exactly when b_i,b_j
are nonnegative and |r| <= sqrt(b_i*b_j). The proposed representative uses
u_i^2 = |r| sqrt(b_i/b_j) and u_j^2 = |r| sqrt(b_j/b_i), so equal fractions
of the available variances are allocated to the group factor. This is a
declared parameterization, not a recovered latent truth. Handle zero
budgets/product and numerical tolerances explicitly; reject infeasible
pairs rather than silently clipping a different model. Existing minimum-
three behavior stays unchanged. Global identifiability, convergence and
multistart checks remain required; add native-map/covariance agreement
across converged starts for pair fits. This does not validate a hierarchical
head automatically, nor does it guarantee useful localization.

## Frozen structural scope

Use both feature banks saved for all 110 answers in fusion_replication_v1,
with the existing cohort, preprocessing, four chronological blocks, seed,
K={3,4,6,8}, stability rule and inverse condition 1000. Only lower the
grouping minimum to two and use the pair-aware covariance construction.
Do not add the wider K or condition-number dose at the same time. Reuse
legacy fits exactly when grouping/fit inputs match a minimum-three case.
Read feature/normalization metadata, not EVALUATION.json or label arrays.
Record every block/failure, old/new K, group sizes, global-Jacobian status,
diagonal budgets, covariance/map agreement and runtime. A successful
structural result is eligibility for a later frozen quality experiment,
not a measured improvement over historical scores.

Before the structural audit, test a continuous pair-scale family showing
identical off-diagonals and varying clipped heads, feasible-pair invariance,
negative and zero residuals, infeasible budgets, feature permutation/sign
equivariance, minimum-three replay and a genuine pair fit. Test an
independent analytic construction and numerical finite-difference Jacobian.
Bind the scientific sources, tests, this protocol, the 110-answer parent
manifest and its frozen score files. Maximum three CPU workers, 1,200 seconds
per invocation with checkpointed completed answers and no killing of an
in-flight fit. Inspect counts and review representative numerical results
before writing the HTML/Markdown evidence report. Further graph and quality
evaluation is a subsequent bounded stage; the full fusion goal remains open.
