# Independent audit: OG-SML Agent B T0

Reviewer task: `structured_fusion_audit`  
Date: 2026-09-04  
Mode: read-only independent recomputation from the frozen C-v2 ledger

## Verdict

- Numerical T0 implementation and mandatory stop decision: **PASS**.
- General overlapping-group OG-SML mechanism: **NOT TESTED** because T0 failed.
- Immutable-final bundle before the final COMPLETE inventory: **INCOMPLETE**.

## Independent reproduction

The reviewer independently reproduced the graph primitives and all T0 numbers:

- prior primary pass: 0 admissible, 3 inadmissible;
- prior primary fail: 6 admissible, 9 inadmissible;
- prior-pass `J_selection`: `[0, 0, 0]`;
- admissible failed-lane J values:
  `0.0024740165, 0.0057010896, 0.0204873113, 0.0312513115,
  0.0468369779, 0.0770585588`;
- `min(pass J)=0 < max(fail J)=0.07705855880674317`.

Free/exclusive edge construction, connectivity, bipartiteness, weighted free-graph
Fiedler values, unweighted exclusive Fiedler values, and `J_selection` matched an
independent implementation.

## Source-lineage finding

All 18 C-v2 selected structures are single INTERNAL hard partitions.  The primary
C-v2 fit uses INTERNAL only; provenance is fitted separately as a reference.  The
attached proposal's Remark 6 therefore assumes an overlapping
INTERNAL+provenance primary fit that never occurred.

The literal T0 prediction is falsified.  The graph theorems and the general
overlapping-group hypothesis are not falsified.

## Gate interpretation

The primary comparison is not a pure optimizer-stability comparison:

- multistart PASS: 15/18;
- profiled-Jacobian PASS: 18/18;
- regularization/weight-sensitivity PASS: 3/18;
- `primary_gate_pass` equals the regularization verdict in all 18 lanes.

As an auxiliary diagnostic, all three multistart failures are graph-inadmissible,
but nine additional graph-inadmissible lanes still pass multistart.  This does not
rescue the preregistered T0 prediction.

## Specification issues before any revision

1. Connected non-bipartite support requires an odd cycle, not necessarily a
   triangle; the current `triangle_condition` name is too narrow semantically.
2. Theorem 1's conclusion can be repaired, but its proof overstates the magnitude
   ambiguity of disconnected non-bipartite components; those components retain
   independent sign gauges.
3. `K>=3` identifies the global loading for a hard partition, but full Theorem-2
   admissibility additionally requires adequate within-block exclusive support.
4. Log-domain WLS and masked power/SDP cannot generally be bit-exact aliases of
   the repository's eigenvector SML/hard L-SML on noisy sample covariance.
5. Failure of free-edge identification alone does not establish a continuum of
   minimizers for the full joint covariance objective.
6. Retrospective `J` uses the selected C-v2 loading and is therefore endogenous to
   the fit it is meant to explain.

## Integrity and firewall

Proposal and ledger hashes match.  Registered sources and current result-artifact
hashes match.  Seven graph tests pass.  The runner reads only the frozen structural
ledger; no token rows, labels, targets, fused score arrays, or outcome metrics are
loaded.  `POSTRUN_AMENDMENT.md` accurately discloses the registry rewrite and the
unchanged numerical result.

Final recommendation: keep terminal status
`T0_FALSIFIED_STOP_BEFORE_STEPS_0_6`; revise the mechanism, theorem wording, alias
requirements, and T0 target before any implementation of Steps 0--6.

