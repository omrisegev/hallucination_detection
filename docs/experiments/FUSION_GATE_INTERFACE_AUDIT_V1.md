# Fusion normalization and no-error interface audit v1

Date: 2026-09-07. Adaptive diagnostic on the existing 58 development answers.
No new candidate, fitted correctness threshold, confirmation set, or inference.

## Decision question

Does the next supporting change need to address the binary no-error gate,
the location readout, or both? What information does answer normalization
remove, and would undoing centering itself change the existing mixture gate?

## Frozen scope

Use the exact release, IDs, width-8 moments, fits and scores from representation
v1 and reliability regularization v1. Audit equal, IU, Joint lambda zero,
Joint graph 0.1, permuted graph, entropy, and the previously tested IU graph
correction. Preserve all invalid fits. Freeze code, this protocol, parent
hashes and derived unlabeled diagnostics before reading labels in this stage.
These labels are already exposed development labels, not a fresh holdout.

For each answer, reconstruct the normalized matrix from saved means, scales,
signs and active columns. Apply a fixed positive affine transform separately
to each feature coordinate; rerun normalization/orientation and compare its
output. This is a coordinate counterfactual, not physically valid alternate
LLM telemetry. It proves an information invariance only within this pipeline.

For each valid frozen fusion core, reconstruct the score from saved weights.
Retain its uncentered projection `-(X / sd * signs) @ w` as a diagnostic.
For entropy use mean window entropy divided by its answer-local standard
deviation; retain the raw mean separately as an absolute-level probe. These projections use each
answer's own feature scales and weight convention; they are NOT calibrated
or directly comparable probabilities. Save their constant offset from the
original score; compare step ordering and pooled versus within-answer AUC.
Do not select a new score or its polarity from these diagnostic results.

Refit the exact one/two-component GMM to original scores and to scores shifted
by +10, including the same shift on step scores. Verify BIC difference,
two-component choice, threshold shift and prediction. Unit-variance fitting
is separately checked. A translation cannot supply missing correctness
semantics to a free-mean mixture. Do not assert general scale invariance with
the implementation's fixed `reg_covar` or prove all gray-box localization
impossible from this restricted invariance.

Record the fitted models' likelihoods, component parameters and lag-one score
correlation. Compute the BIC preference that exact duplication of observations
would give to these same fitted parameters, using the doubled nominal sample
size. This is an analytic pseudo-replication stress diagnostic, not new data,
not a refitted optimum, and not a proposed model-selection rule.

## Evaluation-only diagnostics

1. Replay the seven existing endpoints. For PB, show binary gate confusion,
   clean successes, exact first-error successes, and peak-before/exact/after
   counts on erroneous answers. Keep per-subset denominators and invalid fits.
2. Decompose PB macro-F1 using four explicitly labelled oracles: current gate
   and current peak; perfect binary gate with current peak; current gate with
   perfect locator; both perfect on valid fits only. None is an achieved method.
   Also replay the previously frozen parent-gate diagnostic for IU correction.
3. For PRMB, decompose pooled AUROC into within-answer and cross-answer
   positive-negative comparisons with pair-count weighting. Preserve the
   existing unweighted mean within-answer AUROC as a distinct estimand.
   Check the uncentered projection leaves within-answer comparisons unchanged.
4. Descriptive PB answer-level AUROCs for the fixed-direction probes: GMM
   BIC gain (larger = more error), raw window entropy mean (larger = more
   error), and raw spilled-energy mean (larger = more error). No sign flipping,
   threshold fitting, feature selection, or publication claim. Report each
   subset and valid class counts rather than only a pooled score.

No significance claim is made from diagnostic counts or probe AUROCs on this
small, length-stratified cohort. Existing paired candidate intervals remain
in parent reports. An oracle gain is not an expected achievable gain.

## Verification and next decision

Meaningful tests cover affine normalization, translation of the actual GMM,
the pair-count AUC identity, failure-preserving oracle accounting, and the
fixed-parameter duplicated-observation BIC identity. Independently rejoin all
labels by cell and row ID, reconstruct all displayed metrics, and verify
source/score/report hashes. Do not mutate any parent artifacts.

If a gate fix alone has a low oracle ceiling, retain feature/localizer work
as a priority instead of pursuing a gate-only sweep. A future pooled unlabeled
gate is explicitly hybrid and requires a new frozen protocol. Strict current-
answer fusion stays the primary reference. This audit does not close Joint
feature-bank/grouping research, temporal/sampling ideas, comparator coverage,
untouched confirmation, or the historical 24-cell transfer.
