# Supervised position diagnostic v1

Authorized 2026-09-11. Base ce008fb1d. This is a supervised development
diagnostic, not an answer-only method and not an oracle bound for RBM.

## Fixed question and inputs

Does a change in feature weights between the first and second halves add
predictive information beyond a position-only term? Use all 13,769 answers,
the frozen canonical source folds, labels, spans, and raw entropy gate.
Use the saved bank12 normalized feature MEANS per step from the completed
data diagnostic. Inactive answer-local columns are zero. No new features.
The first ceil(S/2) steps have c=-1; the remainder have c=+1.

Three nested linear scores: static x*w+b; position prior x*w+b+beta*c;
conditional x*w+b+beta*c+c*x*d. The conditional versus prior contrast is
primary. Prior versus static and static versus existing RBM are explanatory.
All three use step argmax, never first_near_max. Means differ from the
historical Top10 readout: results cannot isolate training from representation
when comparing to historical RBM. Internal three-way comparisons are matched.

## Training

Fit separately in each of the nine cells, using four of the five frozen
source-group folds; score only the held-out fold. All cell rows of a source
group remain in that fold. Fit step-column scaling on training steps only,
then form the position interactions. No test-label sign selection.

ProcessBench: equal-answer listwise cross entropy for the annotated first
error among ALL steps of an erroneous answer. Clean answers do not train
the location head; their decision is handled by the unchanged entropy gate.
This uses the first-error target, not invented binary labels after it.
Intercept is fixed at zero because it cancels in the listwise objective.

PRMBench: binary cross entropy on known STEP labels, inverse labeled-step
count per answer and class-balanced aggregate weights. Unknown labels are
excluded from the loss, not from scoring. An unpenalized intercept is fitted.

All fits use mean loss + 0.01/2 * squared coefficient norm, zero initialization,
L-BFGS-B, maxiter=1000, ftol=1e-12, gtol=1e-7, maxls=40. No ridge search.
Nonconvergence or nonfinite values are explicit failures, never fallbacks.

## Metrics and calibration

Original eight-cell ProcessBench macro, Q4/Q8, each cell, raw exact hits,
early/late picks, gate-suppressed hits and coverage. PRMB within-answer AUC,
mean held-out-fold pooled AUC, PRMScore, denominators. Do NOT concatenate
different supervised models' probabilities for pooled AUC. Also calculate
fold-mean AUC for existing references for a matched reference column.

For PRMScore, the model that predicts fold f also scores all other folds.
Its q=.8 threshold is computed from those training-answer scores, with no
held-out groups or labels. Training-score calibration is explicitly declared;
it is not concatenated out-of-fold score calibration and not nested OOF.
The existing unsupervised references retain their original thresholds.

10,000 paired canonical-source-group bootstrap draws, 97.5% intervals for
conditional-minus-prior on the two benchmark endpoints; descriptive 95%
intervals otherwise. These are conditional on fitted out-of-fold predictions:
they exclude refit and previous model-selection uncertainty. PRMScore is
descriptive, not a localization-gain criterion.

## References, integrity and stopping

Keep all 35 configurations from the completed position experiment unchanged.
Add raw entropy step mean and equal normalized step-feature mean as controls.
Validate saved identities/columns/step counts, finite inputs, fold separation,
loss gradients, short and one-step answers, original reference metrics,
independent prediction/threshold/metric replay and failure denominators.
Small synthetic/preflight tests establish mechanics only. No full-data tuning.

Deliver chat-first results and machine-readable artifacts, update history and
direction, then stop. Do not automatically train separate state variances,
additional RBM units, CD, or temporal models. DUFS remains untouched.

Interpret a positive conditional-versus-prior result as evidence for this
joint supervised model; it does not prove unlabeled learnability. A null or
negative result only limits this fixed linear, step-mean, two-half diagnostic.
