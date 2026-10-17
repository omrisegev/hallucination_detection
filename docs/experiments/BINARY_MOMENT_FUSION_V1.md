# Binary entropy / varentropy contribution fusion, v1

User requested adding 15 entropy contributions and executing the binary SML
comparison. Parent b7bf10bd, isolated codex/binary-moment-fusion-v1 worktree.

## Frozen question and roster

Does learned binary fusion help, and do entropy contributions add to the
previous 18-view proposal? Two banks: var18=[15 varentropy contributions,H,V,a];
both33=[15 entropy contributions,15 varentropy contributions,H,V,a].
K=15 ranked alternatives, not 15 temporal positions. Same frozen epsilon-based
Top-K conditional q as previous Varentropy scorer. e_k=q_k*(-log(q_k));
c_k=q_k*(-log(q_k)-H)^2; H=sum(e_k), V=sum(c_k).
a is original selected-token surprisal; it is NOT renormalized within Top-K.
No third moment, powers, graph, Joint, new window, threshold search or inference.

All bank preprocessing is answer-local, offline, and label-free. Orient EACH
varying feature by the sign of its Pearson covariance with H15 within that
answer (zero covariance: retain positive). This common risk convention applies
to ALL four solvers, not just SML. Entropy anchoring is an explicit prior, not
proof of correctness or a fully anchor-free method. Zscore continuous columns;
binary votes use strict > their signed answer-local median, ties negative.
Exact duplicate or constant binary votes are removed; retain first original
column in fixed bank order. Continuous mean retains all varying columns.
Report duplicates since discretization and deduplication both change that
comparison. Do not describe it as pure thresholding with no other change.

Four solvers per bank: normalized continuous mean; equal binary vote; signed
SML; existing binary L-SML with residual group selection, K=2..min(m-1,8),
loading_scale=unit. SML uses the existing majority-positive global sign rule;
ties use its deterministic first-nonzero convention. No post-hoc label sign.
SML final weights have L1 norm one; L-SML across-group weights likewise.
L-SML retains the existing within-group sign threshold (zero -> +1).
Do not call that output a calibrated posterior or exact published likelihood.
No small-m equal fallback. Need >=3 distinct votes for SML, >=5 for L-SML.
Failures stay failures in full denominators. Nonunique SML leading eigendirection
fails explicitly. L-SML near-tied partitions/constant virtual voters are
reported, not evidence of identified independent experts. No automatic promotion.

## Evaluation

Full13769 model-answer rows/145597 steps, frozen source groups/folds/v3 labels.
Same token-to-step spans/top10 mean and first tie rule. Same dual__iu external
entropy q=.3 ProcessBench gate; same held-source-group PRMScore q=.8 calibration.
No label-based detector threshold tuning. Fit remains answer-only but the
overall benchmark pipeline includes externally calibrated decisions.
Nine frozen references from varentropy_contribution_fusion_v1 must replay.
PB all8/Q4/Q8 and each cell, clean accuracy, exact/early/late/suppressed counts;
PRMB within-answer and pooled AUC, PRMScore, denominators and failures.
Primary contrasts: both33 SML-equal binary; both33 L-SML-SML. Source-group paired
bootstrap10000 draws,97.5% intervals for PB and within AUC. Other registered
bank/solver/incumbent contrasts are exploratory95%. Development data only.

Feasibility smoke on short/median/95th length in each of9 cells; no ranking from
that subset. Freeze code/protocol before full scoring. Retain checkpoints and
all per-answer thresholds, signs, grouping, weights, failures. Verify metrics
independently with existing arithmetic replay. Results in chat before any HTML.
