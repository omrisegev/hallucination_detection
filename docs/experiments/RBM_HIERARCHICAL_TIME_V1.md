# Frozen RBM12: hierarchical time fusion v1

Authorized 2026-09-13. Base 113c7eb79. Stop after this family and discuss full results.
This is a positive one-factor covariance model inspired by Factor Analysis, not IU-PCR.
The common factor is not assumed to be semantic error. RBM12 is never refitted.

## Frozen design

- Reconstruct oriented pre-sigmoid logits from saved normalization and RBM weights.
- 16 relative-position regions per step, piecewise-constant token overlap integration.
  Equal region weights exactly reproduce the all-token mean. No zero padding.
- Covariance uses answer-standardized token logits, region means, column centering,
  and S-1 denominator. One-step answers do not estimate training covariances.
- Local, shared and hierarchical fit the same positive Gaussian one-factor model
  Sigma=aa'+D, a>=0. Minimize (logdet(Sigma)+trace(inv(Sigma) C))/2.
  u=(a/D)/sum(a/D); final scores are weighted raw-logit region means.
- Shared covariance: per cell, equal source-group weight and equal answer weight
  within each eligible group. No token-count weighting.
- Hierarchy alpha=(S-1)/(S-1+16). 16 is a frozen engineering constant, not an
  estimate of independent observations. One-step local=uniform; hierarchy=shared.
- Float64 L-BFGS-B, maxiter1000, ftol1e-10, gtol1e-6. Noise floor
  max(1e-8,1e-3 trace(C)/16). Two deterministic starts: uniform loadings
  0.5 sqrt(mean(diag C)); abs leading eigenvector times
  sqrt(max(lambda1-mean(other eigenvalues),0.01 mean(diag C))). Initial noise
  max(diag(C)-a^2,floor). Choose smaller objective without labels.
  Zero covariance or sum(a/D)<=1e-12 yields named uniform rule. Numerical failures
  remain NaN; finite nonconverged fits remain scored and flagged.

## Arms and access

Top10; all-token mean; maximum contiguous-10 mean; local; shared; hierarchical;
hierarchical with deterministic independent token permutations within steps;
supervised positive temporal weights (separate labelled, other-answer panel).
Boundary diagnostics remove only the first token's contribution from the frozen
Top10 and hierarchical readouts. Hierarchical token weights are renormalized over
remaining tokens; regions and weights are not refitted. Singletons retained and
flagged; zero remaining mass is a named diagnostic failure.

Supervised fit uses standardized region profiles, positive logistic coefficients,
intercept, 0.01 L2 (objective adds .005 ||w||^2), and the same solver tolerances.
Group, answer and known-step weighting precedes class balancing. PB uses clean
prefix and first error only, excluding later steps. PRMB uses original labels.
After fitting, coefficients are normalized to sum one; raw weighted mean is the
reported score. The intercept is recorded but not used in the reported mean.

All 13,769 answers; v3 labels, v2 canonical source-group folds, fixed external
entropy q0.3 gate, original argmax with earliest tie. No first_near_max.
Outer model f excludes all source groups in f. PRMScore q0.8 calibration for shared,
hierarchical, shuffled and supervised arms uses inner h predictions from models
excluding both f and h. Local adaptation may see its own answer without labels.
Duplicate exclusion sets reuse identical saved models. Calibration failures and
coverage are reported. No replacement of missing answers.

Primary contrasts: shared-local, hierarchical-shared. 10,000 paired source-group
bootstrap draws, 97.5% intervals. Other contrasts 95%, descriptive. PB all8/Q4/Q8,
per-cell, clean accuracy, exact error and early/late; PRMB within, pooled continuity,
fold-average AUC, PRMScore, coverage. Intervals condition on saved predictions.
Compatible frozen entropy/varentropy/IU/RBM reference rows remain in the ledger.

## Execution and validation

Single process, BLAS1; wait for 4GiB free before loading a cache; 8h invocation cap.
Hash source caches, saved model state, contract, labels, gate and evaluator code.
No source copy, source modification, new extraction, RBM fitting, or HTML.
Smoke27 is feasibility only; a full 13,769 run supplies all performance conclusions.
Tests cover exact baseline reconstruction, overlap/mean identity, positive weights,
short/constant/missing/tied inputs, numerical gradient, synthetic factor recovery,
fold exclusions and held-out label perturbation, independent metrics, and mutually
exclusive lost-success attribution. Checkpoints fail closed on manifest drift.

Reference: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
Future tensor, conditional RBM and convolution families remain outside this run.
