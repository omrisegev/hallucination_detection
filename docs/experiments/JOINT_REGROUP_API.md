# Joint refinement with group rediscovery

`spectral_utils.joint_regroup_membership.fit_regroup_joint` accepts the same
training-only arguments as [staged membership](JOINT_STAGED_MEMBERSHIP_API.md).
Its staged membership is unchanged; only the information-refinement phase calls
the new `spectral_utils.joint_regroup_refinement.refine_with_regrouping`.
Score through `score_noise_aware_joint`, with explicit fallback if requested.
The method returns no gate decision; the registered fixed gate is external.

Each deletion proposal rediscover groups using the existing four inner-source-
fold deletion/stability calculation over K3/K4. The selected partition must pass
the ordinary checked Joint fit. A failed regrouping rejects the proposal; there
is no implicit use of the previous labels. The old three-proposal budget and
current-group min-two eligibility rule remain. The final selected fit uses its
own stored labels, not the labels from the initial bank restricted to its subset.

The95% stop always uses the original factor matrix/information. Changing groups
does not reset this reference. Inspect
`model['refinement']['selection']` for the full path, per-state labels/loadings,
initial reference, proposed partitions, failed proposals and first-crossing
selection. `group_seed` is fixed across proposals; `fit_seed` advances by the
number of accepted deletions, matching the registered protocol.

The initial/group loadings are reconstruction parameters, not semantic labels
for correct or hallucinated steps. These fits use training answers from other
source folds; do not present them as answer-only or as supervised learning.
The full [protocol](JOINT_REGROUP_REFINEMENT_V1.md) and results determine whether
regrouping improves quality and preserves the baseline. A score difference from
the previous base is expected and is separately evaluated under paired margins.
