# Step390: descriptive error profiles and token trajectories

User requests distributions and feature/predictor plots comparing errors that
the existing methods find with those they all miss. No new detector, fit,
threshold search, or flow training. Use frozen Step389 outputs.

Primary scope: all4442 PB model-answer records with annotated first error,
from the full13769-answer benchmark. Partition into found by at least one of
the32 combinations, missed by all32 with gate open, and gate closed. Also
report stricter misses after including the five individual predictor methods.
This is post-selection descriptive analysis. Cohort membership uses labels
and the historical methods; a high score at a detected step is partly true
by construction. Do not infer a new predictive rule or semantic cause.

Display answer-normalized histograms of length, error location, step fraction,
gate score, predictor disagreement and change relative to previous context.
Compute aligned feature/residual profiles in four-token bins from64 tokens
before to128 after the START of the first annotated error step. Each answer
has equal weight per available bin. Missing history is missing, not zero.
Report bin coverage. Step labels do not identify an exact erroneous token;
tokens after the step are not automatically annotated erroneous.

Six predetermined numeric diagnostics for found versus gate-open missed:
log2 token count, first error start/answer token count, erroneous-step token
fraction, L2 change in five standardized features (first16 error-step tokens
versus up to16 preceding tokens), mean predictor disagreement (population
standard deviation of their scalar predictions over error-step tokens), and
mean signed residual across predictors over the error step. Missing history
excludes the L2 diagnostic only; report denominators.

Report raw differences and standardized differences within common-support
strata: PB cell x global PB-error token-length quartile x relative first-error
token-position third. Retain strata with >=5 records per outcome, using
harmonic count weights. These weights and strata are descriptive and fixed
before outcome summaries. Bootstrap10000 source groups, keeping duplicate
model views together; CI99.583333% for6 diagnostics x2 comparisons. Intervals
describe these selected development cohorts, not causal effects or selection-
adjusted future detection gains. Use fixed standardization weights in draws.

Show one covariate-matched caught/missed pair selected near the missed cohort
median in the highest-overlap PB cell, plus the prior watermelon and closed-
gate locker examples. Selection uses covariates and category, not attractive
feature curves. Include full token traces, all five observed standardized
features, scalar observed target and all five predicted scalar means, signed
residuals, and frozen step scores. Scalar prediction is the mean of five
feature predictions, exactly reconstructed as observed mean minus stored
signed residual. It is not each model's complete vector output. Mark every
step boundary and shade only the annotated first-error step; do not smooth
or delete tokens. Link full step text where locally available.

Acceptance: identity/coverage checks, all32 outcome masks reproduce Step389,
gated plus early plus late plus exact counts reconcile, cached residual hash
matches the source artifact inventory, feature IDs and spans align, scalar
prediction minus observation reproduces residuals. Standalone figures and
Hebrew HTML with all outcomes, denominators, uncertainty and explicit limits.
