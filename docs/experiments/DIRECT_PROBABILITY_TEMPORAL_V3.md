# Direct probability temporal fusion v3

Authorized 2026-09-10. Development experiment, not untouched confirmation.
Parent: direct-probability-fusion-v2, commit 438b03a54. No new inference.

## First completed comparison: temporal representation

Use every one of the frozen 13,769 localization rows, unchanged source groups,
annotations, token spans, mean-entropy q0.3 gate, and top-10 token mean readout.
PRMScore retains cross-fold q0.8 calibration. Calibration uses other answers;
normalization, covariance and fusion fit use only the current answer.

The 17 coordinates remain exactly v2: 1-p1, p2..p15, selected-token surprisal,
residual tail. Keep tail to avoid changing two decisions at once.

- Current: T x 17, v2 numerical replay on successful fits.
- Lag8: T x 136, [current, previous1, ..., previous7]. At answer start repeat
  the first token for missing history; retain and score every token. Cross step
  boundaries but never answer boundaries. This is relative context, not eight
  fixed positions in each variable-length reasoning step.
- Delta: T x 34, [current, current-previous], zero first-token difference.
- ShuffledLag8: retain current rows, randomly permute the seven-token historical
  context bundles between token positions, seed derived from answer UID. This
  preserves historical-column marginals but destroys their alignment to current
  observations. This is a negative control, not a causal model.

Current, Lag8 and Delta each use Equal, IU-PCR, full diagonal LW shrinkage,
and full singleton-group Joint-inspired LW shrinkage (same target as v2).
ShuffledLag8 uses Equal and IU. The automatic alpha is the existing heuristic,
not a proved optimum under dependent observations. No alpha search.

Two-axis IU on Lag8: (a) fuse 17 coordinates within each lag then fuse eight
lag streams; (b) fuse eight lags within each coordinate then fuse 17 streams.
Standardize intermediate streams within the answer. Save composed weights.

Temporal-graph controls use Current+IU, chronological chain versus node-permuted
chain, the existing LIU projected solve and normalized Laplacian, lambda=0.1.
No other graph or positional encoding is part of this first comparison.

## Continuing family comparison (not silently equated to this first stage)

The user's full mechanism roster remains: U/IU, LIU/DUFS-LIU, diagonal/block/
Joint shrinkage, CONT/L-SML, full Joint including gate/LIU/diagonal graph hooks,
continuous B3 and its raw/residual/family graph mechanisms. These require explicit
probability-space partitions and measured runtime before launching all-answer
nonlinear/multistart fits. Do not call shrinkage full Joint or omit B3 silently.
Historical24 is a separate continuing task: its v2 fitting scope is across
answers in each cell, unlike localization. A temporal adapter must declare its
aggregation order before that run. No claim that either continuation is complete.

## Evaluation and decision evidence

Evaluate only after the complete first-stage score matrix is present. A small
smoke is numerical/runtime feasibility only. Report all PB8 cells, Q4/Q8/all,
clean accuracy, exact/early/late locations and suppressed correct peaks. Report
PRMB pooled and within-answer AUC, valid denominators and PRMScore. Include saved
v2, token IU9, entropy, varentropy and adapted Mind the Gap as references.

Two primary contrasts: Lag8-IU minus Current-IU; Delta-IU minus Current-IU.
Use 10,000 paired bootstrap samples of canonical source groups, 97.5% intervals
for PB macro and within-answer AUC. Other contrasts are exploratory, with 95%
intervals. No performance pass/fail threshold and no promised winner.
Intervals condition on saved fits and calibration, not prior development choices.

Save per-answer failures and runtime; never drop failed rows from PB denominators.
Saved v2 scores are a separate exact reference retaining their original fallback
semantics. New pure methods never substitute IU or entropy on failure. Malformed
raw input or a source/hash mismatch stops the run as a provenance failure.
For PRMB show coverage and common-answer comparisons; an invalid prediction is
an explicit failure, not a clean step. Preserve original reports unchanged.
Check source hashes on fresh run/resume and bind checkpoints to code+protocol.
Save token scores/weights for a deterministic diagnostic sample and all step
scores for the full population. No large raw-cache copies or new HTML report.

Tests: known spike direction and token mapping; first seven tokens; one-token
answers; no cross-answer access; lambda-zero IU identity; hierarchy reconstruction;
current-v2 score equality; fold and ID equality; interrupted/resumed equivalence;
short steps/top-10 behavior; missing/nonfinite failures retained. Before scoring,
one clean-context preflight reviewer reports findings before any corrections.

EDIS context: local digest papers/digests/edis-paper.md describes burst/rebound
spikes and answer-level discrimination. It motivates temporal inputs, not a
guarantee that each entropy spike identifies the annotated first erroneous step.
