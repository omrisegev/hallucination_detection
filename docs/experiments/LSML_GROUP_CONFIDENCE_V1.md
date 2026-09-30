# Binary latent-group confidence experiment — 2026-09-24

## Question and pre-evaluation commitment

Does retaining the amount of **nonduplicated probabilistic evidence within a
group** improve transfer-oriented source quality over averaging that evidence?
User authorized this CPU experiment, then specifically suggested sensitivity and
specificity. This specifies that suggestion before any new quality evaluation.

Assumption: a binary latent-group model can distinguish member-to-group accuracy
from group-to-label accuracy. This is an assumption, not a certification that the
existing named families satisfy conditional independence.

Success: exact-duplicate invariance and complementary-measurement synthetic gates
pass; all 13,769 source answers receive finite predictions; the candidate improves
official PRMScore over both its matched confidence-disabled ablation and the two
same-bank averaging controls, with corrected paired uncertainty. Meaningful
departure from averaging is measured by score ranks, weight/contribution shares,
and changed decisions, never optimized for its own sake. Failure to clear these
criteria does not close latent-group modeling generally.

## One model, one continuous extension

Use the already fixed, development-label-selected 28 features / 15 named families
from `named_group_fusion_v1/run_20260924/DESIGN.json`. No feature reselection,
orientation search, group search, threshold sweep, or external quality evaluation.
Fit donors on three corrected source folds; calibrate on a fourth; evaluate on
the fifth. This is donor-fitted development research, not answer-only learning or
an unbiased test of the previously label-selected feature bank.

For each answer, threshold each oriented continuous step feature at its empirical
80th percentile, using a strict greater-than comparison (ties are not broken by
time; constant columns supply all-zero indicators). Binary training then uses
all steps in fitting folds. Model:

* Y is binary; alpha_g are independent conditional on Y.
* B_i are independent conditional on their own alpha_g.
* theta[i,0/1] = P(B_i=1 | alpha_g=0/1): member specificity=1-theta[i,0],
  sensitivity=theta[i,1].
* a[g,0/1] = P(alpha_g=1 | Y=0/1): group specificity=1-a[g,0],
  sensitivity=a[g,1].
* Singleton groups set alpha=B directly, avoiding an unidentifiable two-link
  singleton chain. Exact duplicate continuous columns are collapsed first, then
  identical binary columns within a family are one observed classifier. Their
  continuous extension averages distinct continuous columns, not duplicate copies.

Initialize from the off-block rank-one covariance structure, then perform
deterministic EM on this binary latent tree, with fixed symmetric half-count MAP
regularization, fixed convergence tolerance and iteration cap. No true labels
enter initialization, fitting, orientation, or calibration. Orient global Y to
the fixed entropy member; orient each alpha so its members tend to increase with
alpha. Record convergence, objective, covariance residuals, all parameters and
all duplicate equivalence classes. This is an implementation of the paper's
latent model with fixed families and EM refinement, not a claim to reproduce its
published L-SML estimator or its exact numerical results.

Within-group binary log evidence is
`e_g(b) = sum_i [b_i log(theta_i1/theta_i0) +
(1-b_i) log((1-theta_i1)/(1-theta_i0))]`.
The group contributes
`log[(1-a_g1)+a_g1 exp(e_g)] - log[(1-a_g0)+a_g0 exp(e_g)]`.
More complementary members increase confidence about alpha; the group's effect
still saturates at its reliability relative to Y. Member count never multiplies
the outer group weight.

Continuous extension: substitute `b_i = mu_i + sd_i*z_i`, where mu/sd come from
binary fitting observations and z is the answer-standardized continuous feature.
This recovers binary likelihood on standardized binary inputs, but extrapolation
to continuous telemetry is **our unproved extension**, not a Bernoulli likelihood
for real-valued observations and not a calibrated error probability. Singleton
contributions use the same affine Bernoulli-log-likelihood continuation directly.
Final answer z-score and pooled unlabeled calibration q80 match the source bundle.

Primary candidate: summed group evidence. Matched ablation: average each group's
log evidence over distinct binary members before the same outer likelihood.
Everything else, including learned parameters, is identical. Binary application
and first-order linearization of the continuous candidate are diagnostic arms.

Controls: raw28 equal; standardized-family15 equal; the existing continuous
L-SML implementation fitted on binary family indicators and applied to continuous
family scores; previously frozen bank11 L-SML/equal and CT7-z, replayed on identical
IDs, folds, calibration access and metrics. The existing two-stage spectral
implementation must not be mislabeled the exact paper likelihood.

## Tests and evaluation

Before benchmark quality: test probability normalization / exhaustive latent-state
likelihood, EM monotonicity and convergence, singleton handling, duplicate
invariance, continuous-to-binary identity, finite limiting evidence, and a binary
latent-tree simulation where extra conditionally independent measurements improve
held-out log loss. Include a duplicate intervention in that same simulation.

Freeze input/code/protocol hashes before source fitting; seal all predictions
before opening evaluation labels. Validate full identity/offset/fold alignment
against corrected v3 source data. Calibration and evaluation use the SAME fitted
model; only the evaluation fold writes to the OOF output. No calibration-overwrite
or calibration from another fold's fitted model.

Primary: official PRMBench PRMScore (6,211 non-control answers). Secondary:
within-answer AUC (6,030 eligible answers), PB error-answer exact first-error
localization macro over eight cells (4,442 answers). Also per-cell and per-fold,
coverage, runtime, score agreement, parameter and saturation diagnostics.
Primary contrasts: candidate minus evidence-average, raw28 equal, family15 equal,
and binary-fitted family L-SML. 20,000 paired source-question bootstrap draws,
Bonferroni across these four PRMScore contrasts; fixed seed 20260924. Other
comparisons are descriptive. No new hyperparameter variants after seeing quality.

Run scope: existing local CPU caches only, one fixed model fit per fold, at most
500 EM iterations per fold and 30 minutes CPU-stage wall time. If estimation or
a scientific gate fails, preserve the failure and report it rather than silently
substituting another algorithm. Existing results and other worktrees are read-only.

Theory source: Jaffe et al. (2016), sections 2–4,
https://proceedings.mlr.press/v51/jaffe16.pdf. The theoretical model distinguishes
member and group accuracy; duplicate removal and the continuous extension here
are declared engineering additions. Already inspected external benchmarks are
not reopened for tuning in this experiment.
