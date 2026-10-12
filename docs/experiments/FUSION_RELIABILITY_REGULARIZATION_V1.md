# Fusion reliability regularization v1 — frozen before scoring, 2026-09-07

Question: does reducing sensitivity of existing fusion scores to local block
perturbations improve localization? Preserve the same 58 exposed development
answers, release localization-cached-v1-20260907, width-8 moment bank, all
nonoverlapping fitting rows, tail scoring window, orientations and Joint
groups. No new inference, other-answer fit, labels or sampling of fitting rows.
This is adaptive development, not untouched confirmation. A stability win is
not a correctness win. Fusion remains the method.

## Perturbation sensitivity

Use the unchanged sampling helper to resample four two-token blocks within
each original eight-token fitting window, preserving all 29 telemetry columns
together. Recompute 27 moments, apply original answer-fit normalization and
orientation, and form delta Z relative to the original windows. Sixteen
SHA-seeded perturbations (replicates 0–15) form Q_train = mean(delta Z' delta Z)
over replicate-window observations. Eight distinct replicates (16–23) form
Q_validation. Use identity cell/row_id + '/reliability-v1'. No new centering
of perturbed values: the second moment includes perturbation bias. These
are sensitivity matrices, not established noise covariance or LOCA bursts.
Replicas neither increase effective sample size nor provide independent
answers. Serial dependence and the plausibility of this perturbation remain
limitations. The feature roster is fixed from the original answer.

## Weight regularization, three cores and five penalties

Joint: replay the parent's exact five-start fitted covariance C and global
loading v, using its stored groups, original normalized matrix and seed
2026090601. Recheck convergence, multistart, full profiled global Jacobian and
condition <=1e8. Native weight solve: (C + lambda R) w = v with the unchanged
PSD projection and analytic condition-1000 ridge helper. Check lambda-zero and
graph-0.1 score replay before trusting new results. No rescue of invalid Joint
fits and no hidden IU fallback.

IU and equal: retain their original unit-SD weights b, and solve
(I + lambda R) w = b. This is a supporting correction of their existing
weights, not a native IU covariance refit. Use them as matched contribution
controls; do not describe the three native matrices as identical. Lambda-zero
replays each core after the same final scale/orientation boundary.

Five penalties, each trace-matched to the core matrix:

1. isotropic I: generic conditioning control; scalar shrinking of equal/IU
   cancels under unit-SD scoring, an expected invariance.
2. dufs_graph: parent's feature-gated k=7 window graph roughness Z' L Z/N.
   Use saved DUFS gates where available, otherwise the same fixed 120-epoch
   0/1/2-seed learner for equal/IU. Joint graph-0.1 must match parent.
3. block_diag: diagonal of Q_train.
4. block_diag_permuted: one SHA-seeded permutation of that diagonal over
   feature coordinates; same marginal penalties, wrong feature assignment.
5. block_full: full PSD Q_train; permits correlated perturbation effects.

All recipes use lambda in {0,0.1,1,10}. Normalize/orient candidate weights
with the unchanged parent boundary. Validation loss is w' Q_validation w
after original-fit score SD is one. Choose the smallest lambda with loss
<= minimum loss + 0.01 * max(lambda-zero loss, 1e-12). Thus the tuning rule
uses only current-answer generated perturbations, not correctness labels.
Save all losses, penalties, candidate weights and selections. Invalid grid
members are explicit; selection requires at least one valid candidate.
The criterion can favor an unhelpful stable score: test that hypothesis.

## Arms and common decision contract

Primary: 15 automatically selected recipes (three cores x five penalties),
six exact parent-peak controls (equal, IU, Joint lambda-zero, Joint graph 0.1,
Joint permuted graph, entropy), plus explicit Joint graph lambda 1 and 10.
Total 23 arms. Do not label-select the grid or add an oracle winner. Keep the
complete grid as unlabeled diagnostics, not a retrospectively tuned headline.

Score all original windows. Overlap mean to tokens, maximum to official
steps. The unchanged one-vs-two GMM BIC gate is fitted to all original
nonoverlapping risks; if its threshold rule accepts an error, choose the peak
official step, else -1. The gate rule is fixed but its outcome can change.
Save a secondary fixed-parent-gate decision to isolate that effect. Missing
fits or readouts are failures; no parent substitution. Parent arms are exact
replays. Keep the declared negative-entropy global sign anchor; not anchor-free.

## Evaluation and audit

Freeze code, protocol, dependencies, all scores and decisions before decoding
labels in this version. Same parent PRMB pooled and within-answer AUROC,
coverage, PB four-subset macro harmonic exact-error/clean accuracy with failure
penalties. Compare PRMB on common valid IDs and PB on all fixed IDs. Paired
1,000 source-group bootstrap, same parent evaluator and seed. Missing-class
bootstrap draws remain undefined, with valid-draw counts reported.

Register 38 contrasts: each selected recipe vs its parent (15); block_diag
vs isotropic, permuted diagonal and full matrix per core (9); IU vs equal and
Joint vs IU for each penalty (10); Joint graph auto/1/10 vs parent graph-0.1
(3); Joint block_full vs Joint dufs_graph (1). All intervals exploratory and
unadjusted. Report chosen lambda counts, perturbation loss, score correlations,
fit failures, effective weight changes and runtime. No claim that greater
lambda or greater stability guarantees useful localization.

Tests: second-moment PSD and direct quadratic identity, held-perturbation
split, trace normalization, isotropic invariance, lambda-zero identity,
selection/tie/failure rules and actual Joint graph replay on cached inputs.
Independent review reconstructs scores from saved weights/normalization,
selected loss/rule, readout/mapping, label joins, all metrics and frozen hashes.
Maximum three CPU workers and resumable per-answer saves. Historical parents
and Claude's worktree remain intact. Do not close feature-bank/group discovery,
broader comparator replay, untouched confirmation or 24-cell transfer based
on this head-regularization pilot.
