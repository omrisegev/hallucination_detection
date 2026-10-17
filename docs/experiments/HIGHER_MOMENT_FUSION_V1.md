# Compact moments through order six: frozen localization development

User requested RBM and IU-PCR with moments through order6. Parent9e3d452ab;
isolated codex/higher-moment-fusion-v1. No global24 claim in this run.

One answer supplies T token rows. Retain original6 columns exactly:
H15, V15, m3_15, a, a^2, a^3. Here q is the frozen top15-normalized
probability convention, Y=-log(q+1e-12), mr=sum(q*Y^r), and a is frozen
selected-token surprisal. V is variance, not raw m2. This retains the previous
bank rather than adding a duplicate m2=V+H^2. For each order4,5,6 append
mr and a^r. Nested banks have6,8,10,12 columns, NOT15 rank contributions per
power. Constant columns removed by original zscore_columns, fitted only within
the answer. No new clipping, feature pruning, graph or covariance shrinkage.

Each bank has equal mean, IU-PCR, initial RBM and exact trained Gaussian RBM.
RBM is original unregularized one-hidden-unit/unit-visible-variance model,
100 L-BFGS iterations. Initial RBM weights2/P, zero biases, before training.
ALL orders orient toward the SAME mean of normalized original6 columns.
This prevents new features changing the definition of the latent risk sign.
For order3 this is the historical original anchor. Original order3 scores
must replay for all4 arms exactly on the full population. Keep failures and
nonconvergence visible; no retries to seek higher metrics or hidden fallback.

Freeze the code before full execution. Full13769 model-answer rows, same
v3 source groups/labels/folds/step spans; original raw-entropy q0.3 gate,
Top10 token mean step readout, first-max ties. Use existing evaluator unchanged.
Saved Varentropy15/50, entropy, direct-probability, power and moment references
replayed. Claude length/random controls imported only with matching manifests
and reference-array replay. Preserve historical comparator provenance and
Mind-the-Gap limitations. These are previously seen development data.

Primary comparisons: degree6 vsdegree3 for RBM and for IU-PCR. 10000 canonical
source-group bootstrap draws,97.5% intervals for each primary contrast.
Orders4/5 vs3 and each learned model vs its matched mean/initial control are
exploratory95%. No ranking on the27-answer mechanical smoke. No declaration
of an optimum or external confirmation from the development grid.

Report PB all8/Q4/Q8/per-cell, PRMB within-answer and pooled AUC, PRMScore,
coverage/failures/convergence/weights, paired intervals and PB gained/lost.
Different orders jointly change distribution and selected-token powers;
cannot attribute a gain to one of these two families in isolation.

Acceptance: direct formula/column nesting tests, finite high-order values,
order3 model/step reference replay, constant/short input handling, full metric
arithmetic verification, state/score replay. Source/input hashes immutable
on resume; WAL checkpoint and atomic replacement retries retained.
Chat-first findings; no HTML, no further automatic sweep.
