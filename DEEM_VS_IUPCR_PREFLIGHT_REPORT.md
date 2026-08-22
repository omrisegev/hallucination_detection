# DEEM vs IU-PCR 24-cell v1 — preflight execution report

Execution record for `deem_vs_iupcr_24cell_v1` on AIRCC. Two chain submissions
were made; both stopped at the label-free preflight and **Stage A was never
opened**. No natural label was accessed, no score freeze was created, and no
scientific output exists.

Branch `codex/residual-graph-deem-24cell-v1`, base commit `5973b5f`.

## Summary

| attempt | preflight job | failing gate | cause |
|---------|---------------|--------------|-------|
| 1 | 219646 | `adapter_boundary_pass` | environment: cuBLAS workspace unset |
| 2 | 219682 | `adapter_boundary_pass` | **mechanism: B2 score collapse on the 30-feature inventory** |

Attempt 1 was an infrastructure defect and is fixed (`8457893`). Attempt 2 is
not an infrastructure defect and is left for review.

---

## Attempt 1 — job 219646: cuBLAS workspace (fixed)

All 20 adapter boundary fits failed before producing a score:

    RuntimeError: Deterministic behavior was enabled with
    torch.use_deterministic_algorithms(True), but this operation is not
    deterministic because it uses CuBLAS and you have CUDA >= 10.2.
    ... you must set an environment variable before running your PyTorch
    application: CUBLAS_WORKSPACE_CONFIG=:4096:8

The frozen adapter config already specifies `device: cuda` and
`deterministic: True`. On CUDA >= 10.2 that combination raises on any cuBLAS
operation unless a fixed cuBLAS workspace is reserved, and the variable must be
set before the process starts, so it cannot be set from inside Python.

Fixed in `8457893` by exporting `CUBLAS_WORKSPACE_CONFIG=:4096:8` in
`cluster/submit_deem_vs_iupcr_24cell_v1.sbatch`. This makes the determinism the
config already requests achievable; it changes no model, seed, threshold, health
gate, inventory or hyperparameter. The sbatch is not in `CORE_SOURCES`, so
`code_sha256` was unchanged (`0599b682...`) and the run identity was preserved.

The wheelhouse was not implicated: the failing records report `deem: 0.2.0` in
their environment block.

---

## Attempt 2 — job 219682: B2 collapses on the wide inventory

With the cuBLAS fix the adapter boundary executed properly. Everything passed
except one thing.

Passing:

    adapter_boundary_fits        20
    status of all 20 fits        complete
    package_version of all 20    0.2.0
    deterministic_replay_exact   True
    seven_schema_fixtures        7/7 pass (b0_healthy, posterior_sd, reconstruction)
    natural_targets_opened       False
    graph_arms_executed          False

Failing: `adapter_boundary_pass = False`, from 5 unhealthy fits. The pattern is
completely clean:

| fixture | B1 (hard) | B2 (soft/rank) |
|---------|-----------|----------------|
| `schema_p19` (19 features) | 5/5 healthy | 5/5 healthy |
| `schema_p30` (30 features) | 5/5 healthy | **0/5 — all collapse** |

The health criterion is
`np.isfinite(score).all() and np.std(score) >= 1e-3`
(`scripts/deem_vs_iupcr_adapter_worker_v1.py:68`). The B2 scores are finite but
near-constant:

| seed | `score_sd` | `score_n_unique` |
|------|-----------|------------------|
| 0 | 1.27e-04 | 39 |
| 1 | 5.42e-05 | 128 |
| 2 | 2.26e-06 | 8 |
| 3 | 5.92e-06 | 12 |
| 4 | 1.14e-06 | 128 |

Between 8x and 440x below the threshold, on every seed. This is systematic and
deterministic, not seed noise.

**Reading:** the soft/rank adapter degenerates as inventory width grows. The
hard adapter (B1) survives both widths.

### This failure mode is already on record in this repository

`scripts/deem_soft_collapse_probe.py` exists as a registered diagnostic and its
docstring states:

> `deem_deep_soft` failed on 27 of 30 attempts in the registered sweep, every
> one with the same `ValueError: method returned a non-finite or constant
> score`.

B2 is named `deem_inventory_soft_rank_adapter020_repaired`. The repair does not
appear to hold at 30 features.

### Why no fix was attempted

The handoff states: *"Do not change models, seeds, thresholds, health gates,
inventories, or hyperparameters."* Every available route forward is one of
those:

- lowering the `1e-3` threshold — a health-gate change;
- changing the B2 adapter config (`hidden_dim=1`, `sparsemax` preprocessing,
  learning rate) — a hyperparameter change;
- dropping B2 — an arm change.

The gate behaved exactly as designed and refused to open Stage A. The decision
is left for review.

### An observation on the contrast, not a recommendation

If the soft/rank control cannot produce a score with variance on the wide
inventory, then the registered `B3-B2` contrast is being computed against a
degenerate comparator on those cells, and would not be informative there even if
it were forced to run. Relaxing the threshold would convert a visible failure
into a silent one inside one of the three preregistered contrasts.

---

## Secondary finding: the chain still fails open one link deeper

`afterok` worked at the boundaries it guards: when preflight failed, `bundles`
and `stage-a-1` were both cancelled.

But `stage-a-2` depends `afterany:stage-a-1`, and **a cancellation satisfies
`afterany`**. In attempt 1, `sacct` recorded job 219649 (`stage-a-2`) with
`Elapsed 00:00:08` — it started. It was cancelled manually before it did
anything; its log shows only the container image import and it wrote nothing.

The remaining eleven Stage-A jobs would each have allocated a node, installed
dependencies and failed. In practice `run_deem_vs_iupcr_24cell_v1.py:437-447`
verifies the preflight contract and includes `adapter_boundary_pass` explicitly,
so they would very likely have refused on their own — the exposure is wasted
allocation rather than corrupted science. Noted for the chain design; not
changed here.

---

## State

- Queue empty; both chains fully cancelled.
- `natural_targets_opened = False`, `graph_arms_executed = False`.
- No score freeze, no label sidecar, no evaluation output.
- Attempt 1 artifacts archived to
  `_failed_deem_vs_iupcr_24cell_v1_job219646` on both Drive and `$SHARED`.
- Attempt 2 artifacts remain at the live prefix as evidence. Because
  `code_sha256` is unchanged, that prefix holds an identical `RUN_IDENTITY.json`
  and satisfies the handoff's precondition; it will need clearing or archiving
  before any resubmission that changes code.
