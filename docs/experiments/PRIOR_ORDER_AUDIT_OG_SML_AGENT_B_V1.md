# Prior-order audit: OG-SML Agent B v1

Date: 2026-09-04  
Base commit: `250e092e1a0f5b2e460e2fd0221bcbded28069dc`  
Branch: `codex/og-sml-agent-b-v1`  
Worktree: `/Users/osegev/Desktop/hallucination_detection/local_cache/worktrees/og_sml_agent_b_v1`

## Scope authorized by the current order

The current order asks Agent B to test and, only if the test succeeds, implement
the graph-identifiable extension of continuous L-SML described in the attached
OG-SML proposal.  This work is label-free and structural.  It does not authorize
benchmark scoring, model inference, cluster work, commit, push, or promotion.

## Mapping to repository rules

- `CLAUDE.md`: the work is isolated in a new worktree, uses frozen inputs, records
  hashes, and fails closed before labels.  No Git publication or external compute
  is performed.
- `PROGRESS.md`: earlier findings close additional scoring and residual/graph
  variants on the already opened localization populations.  The present order
  overrides that boundary only for a no-label identifiability falsification test;
  it does not reopen those efficacy claims or populations.
- `HISTORY.md`: continuous L-SML is established as a useful historical fusion
  method, but not as a validated official localization winner.  The present test
  keeps that distinction.
- `Research_Directions.md`: prior negative results for residual-guided selection,
  graph smoothing on the step axis, hard filtering, DUFS, and related sweeps remain
  in force.  OG-SML is evaluated only on its new claimed mechanism: identifiability
  of the feature-dependence factorization.
- Git lineage: C-v2 remains unchanged and is consumed only through its frozen
  label-free structural ledger.

## Mandatory stop rule

T0 is executed before OG-SML Steps 0--6.  The attached proposal predicts that the
three C-v2 lanes passing the joint stability gate are graph-admissible and have
larger identifiability score `J` than the 15 failed lanes.  This audit freezes the
following literal test:

1. the source ledger contains exactly 18 lanes and exactly three prior joint-gate
   passes;
2. every prior pass is admissible under Theorems 1--2; and
3. after inadmissible lanes are ranked below admissible lanes, the minimum `J`
   among prior passes exceeds the maximum `J` among prior failures.

If any item fails, T0 is `FALSIFIED`; Agent B stops before Steps 0--6 and reports
which premise failed.  No substitute gate or post-hoc weakening is allowed.

## T0 estimator choices frozen before execution

- The selected C-v2 structure is exactly `internal.groups` in each ledger record.
  It is one hard partition, not a union with `provenance_reference`.
- The loading used to weight the free graph is the selected C-v2
  `structured_fit.global_loading`.  This is a retrospective explanatory diagnostic,
  not a new estimate.
- The free graph contains every pair that shares no selected group.
- Each exclusive graph contains the within-group edges belonging to that group and
  no other group.
- Connectivity is evaluated on all free-graph vertices.  For an exclusive graph it
  is evaluated on non-isolated vertices, and at least three such vertices are
  required.
- Non-bipartiteness is evaluated by deterministic two-coloring.
- The weighted free-graph Fiedler value uses edge weight `|v_i v_j|`.  Exclusive
  Fiedler values are unweighted, as written in the proposal's definition of `J`.
- `J_raw` is the minimum of the free and exclusive Fiedler values.  `J_selection`
  equals `J_raw` only for an admissible structure and is zero otherwise.

## Input integrity

- Attached proposal SHA256:
  `8cda8508add50a424c26106f222d771ffbd44a6a1b5705889a7d7ae83f01f803`
- C-v2 `REAL_STRUCTURAL_LEDGER.json` SHA256:
  `027f2617dfc1d48732de9fe24d3b9809395021fb01f3e0b1391f0af68f4f5ae4`
- Required ledger claims: `labels_seen=false`, `targets_loaded=false`,
  `outcome_metrics_computed=false`, and no persisted fused score array.

