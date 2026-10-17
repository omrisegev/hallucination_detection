# Answer-position fusion implementation and review

Omri corrected the temporal coordinate: a token's position is measured across
the complete answer, not reset at each reasoning step. Steps remain the frozen
annotation/readout units. The previous within-step Gaussian factor run reached
smoke27 PASS only; it did not produce full-population scientific conclusions.

Implementation lives in branch codex/answer-position-fusion-v1, based on
6249db384. Modules/drivers are named answer_position_fusion. Original worktrees,
scores, caches and Claude's processes are untouched.

| Family | Learning | Position comparison |
|---|---|---|
| Gaussian factor | Marginal Gaussian likelihood, common diagonal noise | Fixed versus rank1/rank2 loadings |
| Gaussian RBM H1 | Exact continuous-visible/Bernoulli-hidden likelihood | Fixed versus rank1/rank2 loadings |
| IU-PCR | Canonical additive covariance fit and two-component PCR | Pooled versus regularized regional fits |

All learn from other answers without labels, under the same source-group folds.
All retain the 12-column bank, token scoring, Top10 step readout and fixed
entropy gate. Means/scales are learned only from allowed training answers;
within-family mean-only controls distinguish centering from weight changes.
Each temporal family has a position-permutation control that leaves tokens
and annotations in their original steps. Position-only early/late controls
and equal fusion are also present. The fixed original answer-local RBM12 and
the thirteen inherited reference rows retain their original access labels.

Review checks include exact historical H1 likelihood and gradient reduction,
an independent two-Gaussian derivation, finite-difference gradients at both
ranks, canonical IU equivalence, whole-answer/short-answer interval alignment,
stationary invariance to position permutation, and held-answer feature/label
mutation without changing actual training blocks or the fitted IU map.

A pre-data review caught an important control mismatch: regional IU shrinks
its training means toward the pooled mean, while the first control draft used
unshrunk regional means. The control now uses the identical shrinkage rule;
its coefficients stay fixed. An explicit fixture verifies this. The waiting
smoke had no extracted answers or fitted models; its artifacts were preserved
under results/answer_position_fusion_v1/smoke_pre_review_20260913.

The implementation is tested, but no full performance finding is available.
The sequential supervisor runs smoke -> full only after smoke review PASS,
resumes normal eight-hour caps, and stops on a failure. It waits for at least
4GiB available RAM before loading a cache. Never confuse waiting with fitting.
Live state: results/answer_position_fusion_v1/PROGRAM_STATE.json and the child
RUN_STATE.json (inside smoke/ during feasibility). No new model family starts
after this comparison. Report in chat before generating further documents.

Numerical mismatches/failures, iteration caps, partial coverage and matched
uncertainty remain visible. More data or better density is not proof of better
localization. Rank2 is an interaction model, not two hallucination labels.
