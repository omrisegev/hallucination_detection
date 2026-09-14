# Renyi locator feature-bank experiment v1

Status before execution: **PROTOCOL FROZEN / NOT YET SCORED**

## Questions

1. Does native H1 entropy add useful token-local information to the frozen
   q15 H0lim/VE0/VE0.75/VE1 locator bank?
2. Does q15 Hinf (`-log q1`) add useful top-probability information?
3. Does replacing VE1 q15 with the Experiment-1-promoted VE1 q50 support
   improve the uniform PB/PRMB frontier?
4. Are any gains retained by a deployable fusion, rather than appearing only
   in the label-using union of hits?
5. Do the accepted feature and solver decisions still help when integrated
   with the frozen tail15 Top10 q=.33 gate?

## Fixed population and evaluation

- Same 13,769 answers, 145,597 official steps, source groups and folds as the
  completed Renyi experiments.
- Native H1 is the cached full-vocabulary token entropy. Hinf is calculated on
  the conditional top-15 distribution. Raw `-log p1` is excluded from the
  primary roster because it was nearly redundant with Hinf in the math screen.
- H1 and Hinf use their natural high-is-risk orientation. Existing Renyi/VE
  columns retain the frozen answer-local orientation to q15 VE1.
- Every feature uses Top10 within each official step. Ties use earliest argmax.
- ProcessBench uses the already frozen tail15 Top10 gate and one uniform q=.33.
  PRMBench uses no answer gate and q=.8 calibration from other source-group
  folds.
- No benchmark label enters feature construction or fusion fitting.

## Factorial feature banks

Eight banks cross:

- VE1 support: q15 or q50, replacing rather than duplicating VE1;
- native H1: absent or present;
- q15 Hinf: absent or present.

The q15/no-H1/no-Hinf bank is the exact current locator bank.

## Solvers

1. `raw_step_equal`: Top10 each natural-unit feature, then equal mean. This is
   the current locator contract.
2. `scale_step_equal`: divide each token feature by its within-answer standard
   deviation without centering, then Top10 each feature and equal mean. This is
   a calibration/scale control.
3. `answer_z_local_iu`: within-answer z-score, static answer-local IU-PCR,
   fused token curve, then Top10. The covariance receives the same diagonal LW
   stabilization and frozen IU settings as the existing local-IU method.

No position-varying weights, external priors, shrinkage-position update, or
Joint L-SML are allowed in this experiment.

## Selection

Report PB raw exact, PB all-eight with the frozen gate, PRMB within, fold and
pooled OOF AUROC, and PRMScore. Selection uses one method for both benchmarks:
minimize the worst normalized regret from the best PB and best PRMB-within
scores, with normalization margins .002 PB and .002 AUROC. Ties prefer fewer
features, q15, raw equal, scale equal, then local IU.

The label-using union of correct peaks is diagnostic only. The selected method
is frozen before a separate final replay reconstructs it from raw source rows,
checks score identity, and evaluates the complete gate+locator integration.
This is development selection and replay, not external confirmation.
