# Claude handoff — DEEM vs IU-PCR 24-cell v1

The active research question has changed. Do not resume the archived
Residual-Graph Stage A chain. Phase 0 repaired the nuisance numerics but
closed the current graph extension on specificity; it did not test B3 on
natural targets.

## Frozen identity

- Branch: `codex/residual-graph-deem-24cell-v1`
- Required ancestor: `0a631b28c61496cffb06b32972506cbadfc2cec1`
- Experiment: `deem_vs_iupcr_24cell_v1`
- Active arms: B0, B1, B2, B3 only
- Protocol: `docs/experiments/DEEM_VS_IUPCR_24CELL_V1.md`
- AIRCC root: `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/deem_vs_iupcr_24cell_v1`
- Drive prefix: `gdrive:hallucination_detection/cluster_results/deem_vs_iupcr_24cell_v1`

Before submission, verify the checked-out HEAD matches the commit named in the
handoff message, the tree is clean, the required ancestor is present, and the
new Drive/shared prefixes are either absent or contain the identical
`RUN_IDENTITY.json`. Never attach this run to the old
`residual_graph_deem_24cell_v1` prefix.

Run locally before synchronization:

```bash
python scripts/test_deem_vs_iupcr_24cell_v1.py
python scripts/test_residual_graph_deem.py
python scripts/test_residual_graph_deem_protocol.py
python scripts/run_deem_vs_iupcr_24cell_v1.py preflight \
  --out-dir /tmp/deem_vs_iupcr_smoke --smoke
```

The smoke artifact is not a scientific preflight and Stage A refuses it.
Synchronize the clean commit with the existing `cluster/sync_code.sh` workflow,
then launch exactly:

```bash
bash cluster/submit_deem_vs_iupcr_chain_v1.sh
```

The chain runs a full label-free preflight, rebuilds/verifies all 24 bundles,
produces exactly 480 B0–B3 fits, freezes scores, opens sidecars only after that
freeze, evaluates B=199, reports, and performs resume plus fresh rebuilds.
Scientific boundaries use `afterok`; only continuation jobs within a long
Stage-A checkpoint chain use `afterany`.

Do not change models, seeds, thresholds, health gates, inventories, or
hyperparameters. Stop fail-closed on any identity/hash/schema/health mismatch.
Do not create a graph, graph sensitivity, DUFS gate, or G0–G5 fit.

Only when `evaluation/B199/DECISION.json` contains
`"eligible_for_B999": true`, launch:

```bash
bash cluster/submit_deem_vs_iupcr_promotion_v1.sh
```

Upload large artifacts directly from AIRCC to the new Drive prefix. Return to
Git only compact manifests, evaluation tables, report, reviewer guide, and
rebuild evidence.
