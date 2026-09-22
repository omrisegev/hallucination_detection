# OG-SML Agent B T0 v1

This protocol implements only the retrospective falsification test T0 from the
attached OG-SML proposal.  It is frozen by
`PRIOR_ORDER_AUDIT_OG_SML_AGENT_B_V1.md` before execution.

The test reads the 18 label-free C-v2 lane records, reconstructs the graph family
selected by that run, evaluates Theorems 1--2 and `J`, and compares those values
with the already frozen C-v2 joint stability outcomes.  It never reads token rows,
targets, labels, fused score arrays, or outcome metrics.

Possible terminal states:

- `T0_CONFIRMED_CONTINUE_TO_STEPS_0_6`
- `T0_FALSIFIED_STOP_BEFORE_STEPS_0_6`

No other result permits implementation of OG-SML Steps 0--6 in this worktree.

