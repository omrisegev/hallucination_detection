# One position-conditioned RBM fusion candidate

Base a9cda144e; codex/rbm-position-fusion-v1. Full corrected v3 localization:
13,769 answers, same source groups/folds/gate and top10 step mean + argmax.
No first_near_max in any new arm. The prior 32 configurations are archived
comparators only; the active comparisons use original argmax.

Use the saved trained bank12 (moments through order6), its answer-local
normalization, visible mean a, hidden bias b, weight vector w0 and orientation.
These shared quantities remain frozen. Estimate one 12-dimensional correction
d from the same answer without correctness labels. Conditional energy:
E(z,h|c)=.5||z-a||^2-h[b+z'(w0+c*d)]. Context is -1 for the first ceil(S/2)
steps and +1 for the remainder. Exact conditional log partition, excluding
the Gaussian constant, is softplus(b+a'(w0+c*d)+.5||w0+c*d||^2).
Token risk is saved_orientation * [b+z'(w0+c*d)]. No per-half reorientation.

Minimize average exact conditional NLL plus .5*lambda*||d||^2, where
lambda=.1+P/max(1,min(T_early,T_late)). P is the active column count.
This fixed engineering rule reuses a .1 ridge floor and adds dimension/sample
shrinkage for short halves. It is not an estimated optimum or effective-N
guarantee for dependent tokens. No lambda, bank, position or feature search.
All tokens participate: tokens outside scoring spans follow the preceding
step, and any prefix follows the first step. Readout spans stay unchanged.
Single-step answers keep d=0 for all arms as a declared structural identity.

One candidate (chronological halves), two controls with the same number of
adjustable weights, ridge and optimizer budget: (1) shared correction c=+1
for all tokens; (2) permutation of the step-context labels, seeded by answer
UID, preserving step counts. Permutation changes token counts when steps have
different lengths; keep the chronological ridge for all three arms. This is
a specificity control, not a perfectly length-matched causal test.
Fit L-BFGS-B, maxiter100, ftol1e-10, gtol1e-6, maxls40, initialization d=0.
Finite non-converged fits are reported, following the historical convention;
invalid fits produce missing scores and count as failures. No hidden fallback.

Two primary comparisons: position versus frozen bank12 logit/max and position
versus shared-correction/max. PB macro8 and PRMB within-answer AUC, 10,000
paired source-group bootstrap draws, 97.5% intervals. Descriptive95% intervals
against permuted, Var15/IU, entropy, Var15, Var50 and bank6 posterior/max.
PRMScore uses new q=.8 other-fold thresholds; entropy gate stays unchanged.
Report cell metrics, Q4/Q8, within/pooled AUC, PRMScore, gains/losses, early/late
peaks, failures, runtime, correction sizes and whether improvements reflect
weights changing or just score shifts. PRMScore/pooled AUC are point estimates.

Before full scoring: finite-difference gradients, normalized Gaussian-mixture
density equivalence, d=0 saved-RBM equivalence, short/gap/negative-orientation
tests, fixed 27-case mechanics smoke and code freeze. Full independent metric
arithmetic plus saved-state score/objective audit after completion.
Stop with chat results and update HISTORY/PROGRESS/Research_Directions. No HTML.
All evidence remains development data. No capacity, CD or other model follows
automatically. Preserve prior worktrees, checkpoints and DUFS.
