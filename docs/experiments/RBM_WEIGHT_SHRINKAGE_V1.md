# Fixed six-moment RBM weight shrinkage

Authorized by Omri on 2026-09-11, implemented by a separate Codex agent.
Parent: codex/rbm-m3-powers-v1 at 295011d45567febb013e39b7fb4414768550cd03.
Branch: codex/rbm-weight-shrinkage-v1. No additional agent or HTML report.

## Question and dependency

Does one modest penalty toward the initial equal weights improve learned
fusion while preserving the six existing moment features?

The prerequisite m3/powers experiment is COMPLETE on all 13,769 answers and
its separate arithmetic RESULT_REVIEW is PASS. Removing m3 reduced within-answer
ranking (.73598 to .70635); raw48 learned RBM fell well below its initialization.
These are development findings, not a reason to choose features by labels here.
Keep the original six-column bank and its original mean6 orientation, as proposed
before inspecting the prerequisite's final metrics. No selection of a different
bank, shrinkage coefficient, graph, feature selector, whitening or readout.

## One change

For the existing exact single-hidden-unit, unit-visible-variance Gaussian RBM,
minimize average negative log likelihood plus lambda ||w-w0||^2.
Freeze lambda=0.1 and w0_j=2/P, with P the number of varying retained columns.
The penalty therefore has curvature 0.2 in the standardized weight coordinates.
This is one fixed, moderate engineering coefficient, not a proven optimum or a
label-selected value. No coefficient sweep. Because likelihood is averaged over
tokens, the relative penalty does not shrink automatically in longer answers.
This is an explicit regularized estimate, not an externally fitted prior.
Only w is penalized: visible means and hidden bias retain their original fit.
The weight penalty breaks latent-state sign symmetry toward positive initial
weights. Preserve the original post-fit mean6 orientation for all three arms;
record raw and oriented distances so sign changes cannot look like shrinkage.

Unchanged: H15, V15, m3_15, selected a, a^2, a^3; per-answer normalization and
constant-column filtering; L-BFGS-B maxiter100, ftol1e-10, gtol1e-6, maxls40;
initial visible means and bias zero; sigmoid at token level before Top10 mean.
Initial RBM and original trained RBM are recomputed and must replay saved step
scores exactly for every answer. Nonconvergence is flagged but finite fits remain
included, as before; no silent retries or fallbacks.

## Population and controls

Complete fixed cached localization development: 13,769 answers, 145,597 steps,
corrected v3 annotations and canonical source groups/folds. Same externally
calibrated mean-entropy q0.3 ProcessBench gate, and q0.8 PRMScore threshold
contract. Fit each answer alone. No newly generated traces or global24 test.

Three newly scored arms: original trained RBM, original initial RBM, shrinkage
RBM. Retain saved six-moment IU/mean and power references, all nine Varentropy/
entropy references and historical/Mind-the-Gap panels from the existing evaluator.
Import Claude's length and seeded random-step controls only after manifest input
hashes, aligned entropy/Varentropy arrays and recomputed metrics replay. They use
the same informative entropy gate: do not call them pure length/random detectors.
Do not import his exploratory first-near-max as the shared readout.

## Inference and review

Primary comparison: regularized minus original trained RBM. Report 10,000 paired
canonical-source-group bootstrap draws; use 97.5% intervals for the two primary
endpoints (ProcessBench eight-cell macro and PRMB within-answer AUC), consistent
with the preceding fixed protocols. Secondary comparisons to initialization,
entropy and Varentropy use descriptive95% intervals, not winner selection.
Show all PB cells/Q4/Q8, PRMB within/pooled AUC, PRMScore, valid denominators,
failures, convergence, raw/oriented weight changes, gain/loss cases and runtime.
No arbitrary advancement threshold or promise of improvement. Bootstrap does
not cover uncertainty from all previous research choices; untouched confirmation
would still be required.

Tests: numerical penalty gradient; lambda0 exact objective/fit/score replay;
both historical controls replay; constant/short input handling; no label input;
common orientation and state replay. Full-budget short/median/long-answer smoke
on every source cell is mechanics only. Freeze code/protocol before full run.
Runner uses WAL checkpoint, atomic JSON retry, two below-normal CPU workers and
the existing evaluator. It refuses the full run unless the prerequisite is
COMPLETE/PASS and unchanged. Separate arithmetic metric verifier runs after
evaluation. Preserve source results and all frozen files in other worktrees.
