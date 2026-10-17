# Trajectory fusion history audit for Step318

Read directly on September7. This is a scoped implementation audit, not a
claim that no other ensemble exists in the project or literature.

| Existing work | Actual observations and fit scope | Relevance to the next test |
|---|---|---|
| Claude v2 Module B | `trajectory_reducer.step_order_statistics` builds step x top10 order-statistic matrices, sorted by risk. SML fits need at least50 complete steps; the v2 driver pools training answers. The3x3 feature/trajectory grid includes SML/IU/Joint trajectory heads. | Already tests both axes in an order-statistic representation, not chronological single-answer fusion. Its old labels/folds and fitting scope prevent using its headline as current corrected evidence. |
| Claude B2a/B2b/B3 | Maximum/top10-mean blend, relative-position bins, and explicitly supervised logistic regression. Alpha in B2a was selected using training labels. | A fixed blend is not a new idea. Do not import the selected alpha as a label-free answer-only parameter. |
| `FrozenOnlineGLIUEnsemble` | Global and local heads fitted on multiple supplied answers. Combines two completed-answer scalars at0.5/0.5 after donor-answer centering/scaling. Local scores are already collapsed by max/top-fraction. | Not a combination of complete current-answer IU and Joint trajectories. It does establish prior ensemble/calibration work. |
| `multitask_trajectory` | Nine raw streams and causal transforms (level, EWMA, onset, area, persistence, running max), with separate global/local/online heads. | Temporal features before fusion already exist. Do not call them new merely because the current bank differs. |
| `ciw_cross_scale_localization` | Predict token coordinates from answer means and a response-level CIW score; estimate cross-answer out-of-fold reliability and apply IU. | Cross-scale fusion exists, but its donor scope is different. Not the intended answer-only IMM component. |
| Step299 `fused_trajectory_readouts` | Scalar HMM, ordinary Kalman, actual two-mode IMM and BOCPD after each separate moment-bank fuser, on the original58 answers. Original binary gate retained; temporal curves held over the tail. | IMM-after-fusion is already implemented/tested. Step318 adds joint use of two existing fusion trajectories and tests native GMM beside a fixed-IU diagnostic, on corrected current110. This is an extension, not a new IMM invention or KalmanNet. |
| Step314/317 | Whole-trajectory and peak forensics; observation selection with fixed full GMM support. Step317 shows pooled AUROC can move mostly through answer location/scale. | Preserve full-score references, output normalization, within-answer ranking and exact error/no-error diagnostics. |

## Important mathematical distinction

With a common standardized feature matrix Z, two linear fusion outputs are
s_IU=Z w_IU and s_J=Z w_J. A fixed pointwise average is exactly
Z[(w_IU+w_J)/2]. It can change a peak before argmax but adds no chronological
learning. A covariance-weighted combination also remains linear once its
answer-specific weights are fitted. Both are useful controls; neither alone
completes the chronological trajectory-axis objective.

An IMM processes successive observations with a state and mode posterior.
The proposed supporting component combines the existing IU/Joint signals,
then filters the resulting sufficient statistic. It keeps feature fusion
as its input and compares the same component with simpler fusion inputs.

For the declared common-level Gaussian observation model y=1*x+noise with
fixed covariance R, the sufficient scalar is
z=(1'R^-1*y)/(1'R^-1*1), with variance r=1/(1'R^-1*1).
An IMM with identical measurement covariance across its process modes is
equivalent to applying the scalar IMM to z: the orthogonal observation
likelihood factor is common to all modes and cancels in their normalization.
Test against direct vector Gaussian/Joseph updates, rather than claiming
an extra multivariate information gain beyond that sufficient statistic.

R will be a within-answer heuristic estimated from first differences. It is
not identified semantic error noise. Identical/affine-duplicate source tracks
must collapse to one observation; duplicated fusion outputs are not additional
evidence. A time-permuted control tests whether chronological order matters.
Single-IU/Joint/equal IMM controls test whether combining the tracks helps.

## Sources inspected

- `C:/Users/omris/TAU/hd_jlsml_v2_wt/spectral_utils/trajectory_reducer.py`
- `C:/Users/omris/TAU/hd_jlsml_v2_wt/results/joint_lsml_optimization_v2/REPORT.md`
- `docs/experiments/LOCALIZATION_BENCHMARK_CONTINUITY_PLAN_20260906.md`,Section3A
- `spectral_utils/online_localization_fusion.py`,FrozenOnlineGLIUEnsemble and fit_frozen_online_gl_liu
- `spectral_utils/multitask_trajectory.py`,state/head definitions
- `spectral_utils/ciw_cross_scale_localization.py`,fit_cross_scale_token_head
- `spectral_utils/fused_trajectory_readouts.py`,imm_filter and score_readouts
- `docs/experiments/FUSED_TRAJECTORY_READOUT_PILOT_V1.md`
- `results/localization_history_bridge_v3/REPORT.md` (corrected old58 results)
- `results/fusion_sampling_replication_v1/REPORT.md` and DIAGNOSTICS.json

No Claude-worktree file is edited. Formal paper novelty, a complete literature
survey and a corrected multi-answer replication are not established here.
