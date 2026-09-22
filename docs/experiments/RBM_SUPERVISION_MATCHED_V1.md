# Matched supervision of RBM coefficient updates

Authorized 2026-09-11. Base f7203a8a6. Two alternatives only, plus the frozen
original RBM as a context reference. This tests supervised versus unsupervised
COEFFICIENT UPDATES, not replacement of the entire answer-local fitting stage.

Both alternatives retain the exact saved answer-local bank12 RBM, normalization,
active columns, orientation, visible means, and base parameters. Both learn
the SAME 13-dimensional correction (12 weights and one intercept), separately
per cell on four canonical source-group folds, then evaluate on the fifth.
Token risk logit = saved oriented logit + Z*delta_w + delta_b.
The correction is shared across answers in a cell; the base weights remain
answer-specific. Neither updated arm is strictly answer-only because its
correction uses other training answers. This common access is explicit.

Unsupervised update: exact mean answer Gaussian-RBM negative log likelihood.
For orientation o, update raw w by o*delta_w on active coordinates and raw b
by o*delta_b. Keep a and unit visible variance fixed. Include the exact
partition function of each answer's updated model.

Supervised update: class-balanced step binary cross entropy AFTER the unchanged
Top10 token-logit mean. Each answer first receives total unit known-step mass,
then the two classes each receive total .5 within the training fold. PRMB uses
its known step labels. PB clean answers have all-correct steps; erroneous
answers have a correct prefix and one first-error step, with all later labels
UNKNOWN and excluded from the loss. No token truth labels are fabricated.
All training answers are available to both objectives; only the objective and
its access to training correctness labels differ. No test labels select signs.

Same zero correction initialization, same parameterization, same penalty
0.01/2*||delta||^2 (including intercept), L-BFGS-B maxiter=200,
ftol=1e-10,gtol=1e-6,maxls=40. No hyperparameter search or position terms.
Nonconvergence is recorded; finite decreasing-loss outputs remain scored and
flagged (same rule both arms). Nonfinite/worse-loss fit is an explicit failure.

Same exact token matrix, logit score, Top10 mean, earliest argmax ties and
saved entropy q=.3 gate in both arms. Zero correction MUST replay the original
RBM12 Logit/Top10 scores. No step-feature averaging. Short steps use all tokens;
original shared boundary spans remain unchanged.

Use all13,769 answers. PB macro8,Q4/Q8,per-cell,early/late,clean/error successes
and coverage. PRMB within-answer AUC, mean held-out-fold AUC, PRMScore. For both
updated arms, q=.8 PRMScore calibration uses TRAINING-answer scores produced
by that same fold model, with no test groups. Do not pool cross-fold scores
for the headline pooled AUC; show the mean fold AUC for both arms and baseline.

Primary contrast: supervised update minus unsupervised update, PB and within
AUC,10,000 paired canonical-group bootstrap draws,97.5% CIs. Descriptive95%
contrasts to the unchanged RBM. Intervals condition on fitted predictions;
this remains development evidence and is not an oracle performance bound.

Before conclusions: synthetic/direct Top10 and gradient tests; exact zero
update replay on all answers; source/label/span/column/fold signatures; all
model outputs and calibration replay; independent metric arithmetic. Small
checks establish mechanics only. Stop for chat results, no next model/HTML.
