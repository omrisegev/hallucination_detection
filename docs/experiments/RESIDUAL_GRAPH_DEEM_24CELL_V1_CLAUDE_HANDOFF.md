# Claude handoff — Residual-Graph DEEM 24-cell v1

## Purpose and authority

This is an execution handoff for the frozen experiment in
`RESIDUAL_GRAPH_DEEM_24CELL_V1.md`.  The implementation is complete, but Codex
does not have AIRCC access.  Claude is authorized to synchronize the clean
branch, submit and monitor the frozen Slurm chains, retrieve compact evidence,
and commit those compact results.  This handoff does **not** authorize a method,
hyperparameter, population, source, or decision-rule change.

Phase 2/3, Localization, and Early Detection remain out of scope.

## Required Git state

- Branch: `codex/residual-graph-deem-24cell-v1`
- Required base: `0a631b28c61496cffb06b32972506cbadfc2cec1`
- Implementation/firewall commit that must be an ancestor of `HEAD`:
  `dec23f231bec1466de1908fd9184d1b7a8c988a1`
- Worktree used by Codex:
  `/Users/osegev/Desktop/hallucination_detection_residual_graph_deem_24cell_v1`

Before synchronization, verify:

```bash
git switch codex/residual-graph-deem-24cell-v1
git merge-base --is-ancestor dec23f231bec1466de1908fd9184d1b7a8c988a1 HEAD
test "$(git merge-base HEAD master)" = 0a631b28c61496cffb06b32972506cbadfc2cec1
test -z "$(git status --porcelain)"
python scripts/test_residual_graph_deem_protocol.py
python scripts/test_residual_graph_deem.py
```

Stop if any check fails.  Do not silently rebase, merge a newer `master`, or
select another base.

## Frozen inputs and current state

- Registry:
  `configs/residual_graph_deem_24cell_v1_registry.json`
- Registry content SHA-256:
  `84fa79e396672d9d7f930d385202c0270facb0a32da6b7e78b42926133d5b776`
- Protocol pre-amendment SHA-256:
  `ffb708a1e527b45caed245b783defa1b2cfe2f83f6a1e990ad00efb552c31e7d`
- AIRCC run manifest:
  `cluster/residual_graph_deem_24cell_v1_manifest.json`
- AIRCC run root:
  `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/residual_graph_deem_24cell_v1`
- Drive destination:
  `gdrive:hallucination_detection/cluster_results/residual_graph_deem_24cell_v1`

At handoff time the Drive destination did not exist.  No natural-label sidecar
was created, no natural label was opened, and no full Phase-0/Phase-1 result
exists.  The passing local Phase-0 smoke run is a software test only; Stage A
correctly refuses to consume it.

The raw source mapping is frozen in the AIRCC manifest: 18 cells use the
`repgrid` prefix and six use the declared `regen` overrides.  Do not copy the
approximately 7.82 GiB source corpus or large fit artifacts through the local
machine.

## Connectivity, synchronization, and submission

Follow `CLAUDE.md`, `cluster/README.md`, and the repository's `/aircc-submit`
workflow.  TAU VPN is required.  First run the bounded connectivity check:

```bash
ssh -o ConnectTimeout=5 aircc 'echo ok'
```

If it fails or hangs, report that the TAU VPN is unavailable and stop.

At a clean cluster boundary, synchronize this worktree using the required
tar-over-SSH path:

```bash
bash cluster/sync_code.sh
```

`cluster/sync_code.sh` creates a local `SYNC_COMMIT.json` stamp.  It must report
the handoff `HEAD` with `dirty=false`; do not commit that generated stamp.

Submit the frozen B=199 pipeline from `$SHARED/code`:

```bash
ssh aircc 'cd /shared/cycle2_tau_averbuch_prj/omrisegev1/code && bash cluster/submit_residual_graph_deem_chain_v1.sh'
```

Record every returned job ID.  The chain performs, in order:

1. full Phase 0;
2. one-cell-at-a-time target-free source/bundle construction;
3. checkpointed Stage A across all 24 cells and five seeds;
4. isolated sidecar construction only after the score freeze;
5. B=199 evaluation and report;
6. checkpoint-resume evaluation;
7. genuinely fresh Stage-A rebuild and evaluation;
8. deterministic rebuild verification and final report refresh.

The repeated Stage-A jobs are an intentional linear `afterany` chain.  They
continue immutable checkpoints across the eight-hour wall.  Do not parallelize
them against the same output directory.

## Monitoring and fail-closed rules

Use `/aircc-status` or the repository's cluster-ops workflow.  Do not run raw
SSH polling loops.  Preserve logs and partial objective histories on every
failure.

Stop the scientific run and report the exact evidence if any of these occurs:

- protocol, registry, source, admission, code, config, environment, or
  `RUN_IDENTITY.json` mismatch;
- the Drive prefix already exists with a different or missing run identity;
- inventory/order/sign/row-count mismatch;
- incomplete or unhealthy fit, missing seed, collapsed posterior, orientation
  ambiguity, graph/gate failure, or fold-artifact failure;
- non-bijective `row_id` label join;
- missing/tampered fit artifact or score-freeze prerequisite;
- attempt to open labels before a complete immutable
  `SCORE_FREEZE_MANIFEST.json`;
- resume/fresh summary, decision, or semantic-hash mismatch.

Do not repair a failed scientific run by changing seeds, epochs, learning
rates, lambda values, graph topology, health thresholds, source population, or
null logic.  A code defect may be fixed only on a new reviewed commit, followed
by a fresh run identity and a full restart from the appropriate pre-label
boundary.

## Conditional B=999 promotion

Do not submit B=999 merely because B=199 completed.  It is legal only when the
frozen B=199 `DECISION.json` contains:

```json
{"eligible_for_B999": true}
```

The promotion script independently enforces the gate.  If eligible, submit:

```bash
ssh aircc 'cd /shared/cycle2_tau_averbuch_prj/omrisegev1/code && bash cluster/submit_residual_graph_deem_promotion_v1.sh'
```

Monitor the B=999 evaluation, report, resume evaluation, fresh evaluation, and
final rebuild verification in the same fail-closed manner.  Stage-A scores and
graphs remain frozen; they are not refit for target-null draws.

## Required output and Git return

The AIRCC driver uploads large data, bundles, checkpoints, graphs, scores, and
sidecars directly to the registered Drive prefix.  Never fetch or commit those
large artifacts.

Return to Git only compact, reviewer-facing evidence:

- Phase-0 registry/completion and nominated-lambda manifests;
- run identity, run definition, fit/score-freeze manifests, and compact hashes;
- evaluation CSV/JSON summaries, `DECISION.json`, bootstrap/null summaries,
  controls and diagnostics;
- `REPORT.md`, `REVIEWER_GUIDE.md`, and the 12 registered visualizations;
- `REBUILD_VERIFICATION.json` and compact resume/fresh evidence;
- small logs needed to explain a fail-closed outcome.

Before committing results, confirm that no `.pkl`, raw source, bundle, label
sidecar, fit `.npz`, checkpoint, dense graph, or other large artifact is staged.
Use the final commit subject:

```text
experiment: record residual graph DEEM 24-cell results
```

Push the same branch.  Report the final primary decision, promotion status,
job IDs, Drive prefix, result commit, and whether rebuild verification passed.

## Claim boundary

B1/B2 are packaged `deem==0.2.0` adapter controls, not paper-exact.  B3 and
G0–G5 are continuous-visible adaptations and carry no DEEM theorem claim.  No
historical score may substitute for the fresh matched-inventory run.  No pooled
row AUROC is permitted, and no Phase 2/3, localization, early-detection, or
universal-manifold claim may be inferred from this experiment.
