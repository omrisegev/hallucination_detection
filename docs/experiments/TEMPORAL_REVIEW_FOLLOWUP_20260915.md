# Temporal review follow-up: frozen diagnostic contract

Registered before fitting position profiles or evaluating the new controls.
This follows Claude's review and Omri's approved research program. Development
data and the label-selected innovation5 bank remain development-exposed.

## Scope and stopping

Preserve all frozen scores, manifests and checkpoints. STOP_AFTER_JOB prevents
the next neural job from starting. Do not run more flow training in this stage.
Fold-0 model diagnostics are implementation/mechanism diagnostics only, not
quality comparisons or grounds to reject a model family.

## Selector and checkpoint audit

Audit all 55 DUFS selectors, including gate saturation and pairwise top-4 seed
overlap (intersection/4 and Jaccard separately). Dense gates do not establish
sparse selection; evaluated top-k scores remain historical observations.

For each completed, non-smoke BEST.pt available before this audit starts, use
16 batches of 256 source-group-balanced validation windows, fixed seed 20260915.
No labels or parameter updates. Report FM, repel and curve, active hinge
fractions, positive/negative error, transport-target norm, condition displacement
and separately weighted gradient norms. Apply identical diagnostic PGD to FM
as a diagnostic counterfactual, not a claim it was trained with DiFlo losses.
Report synthetic identity negatives separately. These are checkpoint snapshots;
unlogged historical losses cannot be recovered. Record paper deviations:
MLP width128 rather than512, <=50k rather than synthetic200k updates;
early stopping on held-out unlabeled objective; margins1/.9 retained without
an established task-scale justification. Do not change them in this stage.

## Full-population position control

All 13,769 answers /145,597 steps /6,968,779 tokens. Reuse frozen source groups,
outer folds, raw double-precision H0lim, spans and baseline gate. Fitting accepts
only the existing label-free metadata schema. No new inference or alpha search.

Compute I[t]=H0[t]-mean(H0[:t]); I[0]=0. Fit profiles using t>=1 only.
Position=(t+.5)/T; 16 fixed bins with linear interpolation between bin centers
and constant boundary extension. Condition on cell and length strata
T<=256, 257..512, 513..1024, >1024. Within each cell/stratum, give source groups
equal weight, answers within group equal weight, and eligible tokens within
answer equal weight. A stratum with <20 fitting source groups uses its cell's
profile, explicitly logged. Empty position bins use the cell bin. Near-zero
conditional std uses the global std (floor1e-8), explicitly logged.

Let mu(p),sigma(p) be training profiles and mu0,sigma0 their training population
mean/std. Keep original four-stream per-step Top10 scores B unchanged. New
fifth-stream readouts, all with the same (4B+Top10(stream))/5:

- profile_only: mu(p), deterministic position/length/cell information only.
- constant_profile: mu0, control for the fifth-stream denominator/offset.
- mean_detrended: I-mu(p)+mu0, preserving global level and residual amplitude.
- location_scale_detrended: mu0+(I-mu(p))*sigma0/sigma(p).

All streams retain zero at the first token (no history). Preserve original4
and unmodified innovation5 alongside the controls. This is a fixed nonlinear
location/scale diagnostic, not a claim to remove every position interaction.
No stochastic null is introduced. Profile-only does not mimic innovation's
temporal autocorrelation or full conditional distribution.

Five source-excluded fits for evaluation. Ten additional pair-excluded fits
produce PRMB calibration predictions: for test fold h, calibration fold g
must be scored with both h and g excluded. Same q=.8 PRMScore calibration;
same frozen transductive PB tail15 q=.33. Fitting, calibration and output
coverage asserted independently; no silent mean fallback.

Two primary comparisons: mean_detrended vs profile_only and
location_scale_detrended vs profile_only, each on PB and within-AUC.
10,000 paired source-group bootstrap draws; Bonferroni98.75% intervals for
these four endpoint comparisons. Other comparisons to original4/innovation5
and constant_profile are descriptive95% intervals. No correction claim for
all historical adaptive development. Do not promote a new winner in this stage.
Include fixed historical earlier-VE0/VE075-peak PB readout separately.

## Acceptance and interpretation

Exact replay of frozen innovation step scores from original float64 features;
synthetic constant-profile identity; known nonlinear trend removal; held-fold
feature/label mutation invariance of fitting; first-token/constant-feature
handling; independent scalar PB audit of every method, per-cell counts and
macro; independent pairwise within-AUC on all valid PRMB answers.
Save source/code/score hashes, fitting groups, profile fallback counts, scores,
predictions, calibration thresholds, paired intervals, length/first-error strata
and source-group access. A residual gain beyond profile supports answer-specific
information conditional on these controls; it does not establish optimal
temporal routing, causal correctness, or a novel learning algorithm.

Next design decision, after reviewing this result: whether the remaining
answer-specific signal justifies feature weighting varying along the answer.
Do not launch that model or retune the gate within this diagnostic stage.
