# Information-feasible Joint refinement

`spectral_utils.joint_feasible_membership.fit_feasible_joint` follows the same
training/scoring contract as [regrouping Joint](JOINT_REGROUP_API.md). It changes
only how refinement handles a proposed information-budget violation.

The next three proposals are still ranked from the current Joint factors. Each
proposal is first checked against95% of every relevant INITIAL factor's
information. A failing proposal records `reason='INFORMATION_BUDGET'` and
`discovery=None`, then the next proposal is tried. Passing the information check
does not waive group discovery or native Joint identification checks.

All accepted states satisfy the original information constraint. The selected
state is the last accepted state; no violating crossing is accepted into the
path. Termination means no eligible top3 proposal passes all constraints, the
same min8 cap was reached, or no current group has a deletable member.
The information reference, grouping rule, proposal ranking, search budget and
readout are unchanged. The threshold is not adapted to held outcomes.

Inspect `model['refinement']['selection']['deletion_audit']` for per-proposal
retention and separate information/group/fit rejections. The gate is external;
the registered experiment is hybrid source-fold fitting with no digit inputs.
See [the frozen protocol](JOINT_FEASIBLE_REFINEMENT_V1.md) for the full quality
and surplus-feature preservation requirements. An admissible fit or smaller
feature bank does not by itself establish improved localization.
