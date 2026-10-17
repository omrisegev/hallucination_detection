# Minimax information refinement inside Joint

`spectral_utils.joint_minimax_membership.fit_minimax_joint` has the same training
and scoring contract as [Step410](JOINT_FEASIBLE_API.md). It changes only the
ranking used to select the next three eligible deletion proposals.

For the fixed initial factor matrix A and reference I, each proposal receives
`max(1 - information_after_deletion / I)` across relevant initial factors.
The lowest score is tried first. Conditional covariance identities calculate
these scores efficiently; independent reduced-system solves audit them.
`worst_initial_information_loss` is cumulative relative to the original set,
not incremental relative to the latest fit. `selection.ranking` records the rule.

After ranking, the unchanged information-budget check precedes group discovery
and the checked Joint refit. All accepted states retain95% of each relevant
original factor. Groups, current loadings and readout can change at every step,
but the information reference cannot. At most three proposals are attempted.
The last accepted state supplies weights, with the same orientation, exact-alias
expansion and explicit invalid-fit accounting. No label-selected count, protected
feature or gate adjustment is introduced. BOCPD is an ordinary eligible input.

The registered evaluation is hybrid five-source-fold fitting, not answer-only.
The external fixed gate and historical comparator scores remain separate.
See [the frozen protocol](JOINT_MINIMAX_REFINEMENT_V1.md) for quality requirements.
This minimax surrogate concerns information preservation, not true error risk;
admissibility and successful pruning alone cannot establish localization gains.
