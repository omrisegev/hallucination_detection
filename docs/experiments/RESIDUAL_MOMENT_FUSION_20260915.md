# Residual-moment fusion: bounded synthetic check then full real data

Omri explicitly authorized old unused worktree cleanup and asked to advance
quickly from synthetic checks to the full real population. Freeze this before
new outcomes. Synthetic failure is diagnostic unless mechanics fail; it is not
a veto on the real benchmark. Preserve Step381 innovation5 and Step384 results.

## Question and fixed comparisons

Does fitting IU moments on residuals from the existing past16 ridge predictor
improve fusion, separately from scoring residuals themselves? Use innovation5
only. No bank search, new flow/CCA training, gate search or learned groups.

The core crossed comparison has moment inputs L/R (levels/residuals), score
inputs L/R, and heads native2PC/simplex tau1 eta.25. Eight policies. Evaluate
each with answer-local moments and with pooled source-excluded moments within
the same dataset/model cell (16 policies total, a fixed factorial, not tuning).
Include raw per-stream Top10 equal on L and R (2 controls), the unchanged
original4 and innovation5 anchors, and archived same-gate single/RBM anchors
where already available. These are development comparisons, not confirmation.

R_t = raw_x_t - (answer_mean + answer_scale * ridge_prediction_t). The predictor
uses all feature values in preceding16 tokens, mask and relative position, with
its original whole-answer normalization. It is offline, not causal-online.
Use the frozen Step379 innovation5 ridge fits; verify hashes and source IDs.
Tokens0..15 use the existing masked-history model, not deletion/static omission.

Centered covariance is fitted in coordinates scaled by LEVEL SD, shared by L
and R. Never independently standardize residuals for the crossed comparison.
Floor1e-6 I, fixed g2 grid ceiling .25*trace(C_L)/5 for both C_L and C_R,
canonical additive L2/full-pool moments and300 grid points. Native head keeps
two PCs; convert its coefficients to raw units and normalize absolute sum to1
for both representations. This is a declared score-scale convention; it does
not make native and simplex identical. Simplex uses the full covariance,
B=level_SD/mean(level_SD), tau1 and eta.25. No dynamic eta in the real roster.

Readout is Top10 on each ORIGINAL ORIENTED stream of the scoring representation,
then multiply the resulting stream summaries by the answer's constant weights
and sum. Negative native coefficients do not reverse token selection inside
Top10. This isolates weights at the established per-stream readout. No token
fusion-before-Top10 comparisons or extra sign fitting in this stage.

All13769 answers /145597 steps /6968779 tokens receive scores. Exact same
frozen tail15 percentile gate q=.33. No labels enter predictors/moments/weights.
Labels enter only the separated evaluator and PRMScore development calibration.

## Source separation and fitting budget

Pooled moments use up to16 uniform landmark tokens per reference answer,
including endpoints, source->answer->landmark balanced. Each outer fit excludes
the held fold; residuals of every reference fold also exclude THAT fold from
ridge fitting. Thus use existing two-fold-excluded ridge models for outer
reference residuals, never in-sample training residuals.

Nested PRM calibration excludes both outer and calibration folds. Its reference
residuals use three-fold-excluded ridge fits. Fit only the10 missing triple
models, using the original deterministic16384 sampling, ridge1, validation
split and no label access. Held query predictors use the existing single/pair
models. Evaluate actual nested calibration; do not reuse outer predictions
whose predictors learned from the outer held source. Existing inputs immutable.

Answer-local moments use the current answer's entire unlabeled L/R arrays,
while their predictor remains fitted on other source groups. Hence this is
answer-local covariance with an external predictor, not a strict answer-only
algorithm. This supplies the user's preferred local-information comparison.

## Short synthetic stage

20 seeds, independent train/test answers; five fixed time worlds: fast target
with slow nuisance (.97), fast target/fast nuisance, fast target/no nuisance,
slow target(.97)/fast nuisance, and both slow(.97).200 answers x64 tokens,
same loadings/noise as Claude's B. Use the SAME crossed head/readout coordinate
conventions (synthetic token ranking rather than real per-step Top10). Ridge
residuals for covariance fitting are out-of-answer-fold predictions. Test
predictor uses all training answers. Record paired seed differences, coverage,
noise/signal predictability and direction diagnostics. No tuning to outcomes.

Retain Claude's equal-variance and gate findings as calibrated/exploratory
evidence; add analytic equal-marginal-variance and constant-eta identity checks.
g2 near the ceiling makes its proposed gate approximately constant: no new real
eta policy until evidence distinguishes gating from raising a fixed eta.
No requirement that a model win worlds where desired signal is predictable.

## Evaluation, audit and decision

Primary: for each of pooled/local x native/simplex, residual-moment/residual-score
minus level-moment/residual-score (weight contribution), and residual-moment/
residual-score minus equal-residual (benefit over mean).8 contrasts x2 endpoints
PB/within =>16 endpoints,10000 paired SOURCE-GROUP bootstrap draws, Bonferroni
99.6875% intervals. Other contrasts descriptive95%; no adaptive promotion of
secondary to primary. Report full Pareto and PRMScore, per-cell, coverage,
early/late errors and runtime. The original5 and4 quality anchors must replay.

Do not promote on stability/NLL alone. Zero-norm unidentifiable native heads
use explicitly recorded equal weights; report pure failure and fallback counts.
Numerical/KKT failures halt; no silent recovery. Tests cover common units,
crossed identities, zero-predictor identity, finite masks, source exclusion,
canonical native agreement and independent optimizer/readout/PB/within replay.
Small runs check mechanics/cost only. Write resumable score arrays per excluded
fold set, fit moments/weights/fallbacks, manifest hashes and complete state.
Memory bounded per answer; no full token residual cache; output budget<500MB.
