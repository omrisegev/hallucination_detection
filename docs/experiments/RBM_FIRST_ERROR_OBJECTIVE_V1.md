# RBM first-error objective v1

Authorized 2026-09-11. Base f9d984266. Question: does first-error selection
training improve PB compared with the existing supervised step BCE update?

Keep the exact bank12 token matrices, saved answer-specific RBM logits and
models, active columns, normalization and orientation. Keep the same shared
13-coefficient additive correction, zero initialization, L2 .01/2||delta||^2,
L-BFGS-B maxiter200,ftol1e-10,gtol1e-6,maxls40, per-cell canonical source folds,
Top10 token-logit mean, earliest argmax ties and frozen entropy q.3 gate.
No position terms, new features, graph, threshold or hyperparameter search.

The old supervised arm used balanced step BCE with known prefixes/first errors
and unknown labels after the first error. The new arm uses equal-answer
categorical cross entropy: logsumexp(all step scores) minus the score at the
annotated first error. Later steps compete as locations without being assigned
binary correctness labels. Clean training answers have no first-error target
and contribute zero location loss; they remain available in the same training
fold and are retained in evaluation with the identical external gate.
This objective-specific label eligibility is explicit, not an identical
binary-label loss over the same set of supervised steps.

A shared logit intercept cancels analytically in the categorical loss; its
gradient is only the common penalty and it stays zero from the common zero
initialization. Keep the same13-parameter interface. Fit40 PB models only.
Copy the preceding PRMB supervised scores and q.8 thresholds unchanged; do
not claim a new PRMB result or transfer experiment. No test labels in fitting.
Both supervised arms use other training answers and are not answer-only.

Evaluate all6,800 PB answers, macro8,Q4/Q8,each cell,clean accuracy,raw/gated
first-error hits,early/late choices,gate-suppressed hits and coverage. Show
the unchanged6,969 PRMB answers for continuity. Retain the old supervised,
unchanged RBM, Varentropy15/IU, Varentropy50 and entropy reference scores.
Primary contrast: first-error objective minus step BCE,10,000 paired canonical
source-group draws,97.5%CI for PB. Other comparisons descriptive95%. PRMB
identity is an integrity check, not new evidence. Intervals condition on
fitted predictions and exclude refit/earlier selection uncertainty.

Preflight: direct loss and finite-difference gradients through Top10; one-step,
clean/error,unknown-after-error,short/shared-boundary spans and shift invariance.
Verify input hashes, full original zero-update replay from the prior study,
fold separation, old result replay and independent new-model/metric arithmetic.
Flag finite nonconverged fits under the SAME policy as the preceding run;
nonfinite/worse-loss models are explicit failures, no fallback. No pilot ranking.

Report chat-first, save CSV/JSON and concise HISTORY/PROGRESS/Research_Directions
updates with reasons. No HTML or next experiment. Source caches/DUFS unchanged.
