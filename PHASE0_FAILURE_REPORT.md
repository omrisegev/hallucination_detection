# Residual-Graph DEEM 24-cell v1 — phase0 failure report

Run attempt: AIRCC jobs 217597-217627, 2026-08-22. Branch
`codex/residual-graph-deem-24cell-v1` @ 82ef3228c34c207ad0a4bdda0676c73decb3fb24.
Result: phase0 FAILED (exit 1); chain stopped; no scientific output produced.

## Symptom

    File "spectral_utils/residual_graph_deem.py", line 533, in __call__
        eigenvalues, eigenvectors = torch.linalg.eigh(
    torch._C._LinAlgError: linalg.eigh: The algorithm failed to converge because
    the input matrix is ill-conditioned or has too many repeated eigenvalues (error code: 2)

The error message is misleading. The matrix is not ill-conditioned.

## Root cause: nuisance head diverges to NaN

Instrumented reproduction (CPU, float64) captured the actual `eigh` input:

    shape: (3, 3)
    n_nan: 9   n_inf: 0
    all finite: False
    matrix: [[nan nan nan]
             [nan nan nan]
             [nan nan nan]]

Failure coordinates:

    n_fits: 45          (45th fit_continuous_deem call in phase0)
    seed:   0
    lambda: 1.0         (largest value in LAMBDA_GRID)
    mech:   nuisance
    X_finite: True      (input data is clean)

Input `X` is finite, so the data is fine. The nuisance-head parameters
(`W1, d1, W2, d2`) diverge to NaN during optimization; `torch.linalg.eigh` is
merely where the NaN first surfaces.

## Proposed mechanism

1. `W1` and `W2` are both initialised at `* 0.005`
   (`residual_graph_deem.py:515,519`), so the two-layer output
   `tanh(X @ W1.T + d1) @ W2.T + d2` is very small.
2. Its 3x3 covariance is smaller still. Measured at init: max abs entry
   **2.43e-07**, versus `whitening_ridge = 1e-6`. The ridge is **4.1x larger
   than the entire covariance**, so `covariance + ridge*I` is ridge-dominated.
3. The whitening applies `eigenvalues.clamp_min(ridge).rsqrt()`
   (line 537). With eigenvalues pinned near the 1e-6 ridge floor, this
   amplifies `U` by ~1e3.
4. Gradients flowing back through that ~1e3 amplification are correspondingly
   large. At `lambda = 1.0` the nuisance penalty term
   (`nuisance_smooth + gamma * nuisance_orth`, lines 628-634) dominates the
   objective, and the parameters blow up.
5. By fit 45 the head is NaN.

Consistent with the ordering: the run survives lambda in
{0.0, 0.01, 0.03, 0.1, 0.3} and dies at lambda = 1.0.

## Why the smoke run passed

`--smoke` does not exercise this path:

| setting      | --smoke      | full            |
|--------------|--------------|-----------------|
| `epochs`     | 3            | 100             |
| `dufs_epochs`| 3            | 120             |
| `n_rows`     | 160          | 1024            |
| seeds        | `(0,)`       | all 5           |
| lambdas      | `(0.0, 0.1)` | full 6-value grid, incl. **1.0** |

`lambda = 0.0` skips the nuisance branch entirely, so smoke touched this code
for 3 epochs at a single non-zero lambda, and never at lambda = 1.0.

## Reproduction (no cluster, no GPU, no Drive)

`run_phase0` fits only `generate_synthetic_worlds(registry)` with a fixed base
seed, and the fit is pinned to CPU float64 (`device: str = "cpu"`, enforced at
`residual_graph_deem.py:571`). Reproduced identically on the AIRCC NGC
container and on Windows CPU float64 — this is deterministic, not
platform-specific.

    python scripts/run_residual_graph_deem_24cell_v1.py phase0 --out-dir <tmp>

Takes roughly 10 minutes to reach the failure.

## Secondary defect: chain fails open after a phase0 failure

`bundles` depends `afterok:phase0` and was correctly cancelled. But
`stage-a-1` depends `afterany:bundles`, and `afterany` is satisfied by a
*cancellation*. All 12 Stage-A jobs therefore remained PENDING and eligible
after phase0 failed; they were cancelled manually. A failed phase0 should
collapse the chain, not release it.

## Secondary defect: deem/entmax silently not installed

AIRCC compute nodes have no outbound DNS. Every pip fetch failed with
`Name or service not known`; the login node does have internet, so this is not
visible from a login-node check. `deem==0.2.0` and its `entmax` dependency were
both skipped. The job still proceeded because the legacy `setup.py develop`
path treats dependency-fetch failure as non-fatal, so pip exited 0 and the
sbatch `set -e` never tripped.

This did not affect phase0, but `deem` is imported lazily at
`spectral_utils/deem_adapter.py:209`, reached only by the B1/B2 packaged
adapter controls in Stage A — so it would have failed many hours into the
chain, after Stage-A checkpoints were already being written.

Fix does not touch `CORE_SOURCES`: vendor the wheels from the login node into
`$SHARED` and add `--no-index --find-links` to the sbatch pip invocation.

## Note for whoever fixes the numerics

A ridge bump or an SVD fallback at line 533 would mask a NaN rather than fix
it. The divergence happens upstream of `eigh`. Note also that the protocol
already has a health-gate concept ("unhealthy fit cannot produce a
cell-complete checkpoint"); a NaN fit arguably belongs in that path rather than
crashing the job.

## State

No scientific output exists. No Drive prefix, no RUN_IDENTITY.json, no scores,
no label sidecar; the score-freeze firewall was never approached. Cluster is
clean for a restart. No method, hyperparameter, seed, lambda, threshold, or
population was changed.

## Addendum: verification of the proposed ridge-Cholesky whitening

Checked numerically before implementation (CPU float64), to confirm the
substitution is a stability fix and not a methodological change.

### 1. The penalty is exactly invariant to the choice of whitening

ZCA whitening (`Sigma^-1/2` via `eigh`) and Cholesky whitening
(`c @ L^-T`) differ by an orthogonal rotation `Q` of the whitened
coordinates. All three penalty terms are invariant under `U -> U Q`:

- `U.square().sum()` = Frobenius norm — invariant.
- `_sparse_quadratic(U, laplacian)` = `trace(U^T L U)` — invariant under
  cyclic trace.
- `cross.square().sum()` = `||U^T e||^2` — invariant.

Measured, identical to 10 decimal places including gradients:

    zca   |U|F2=3066.8718286541 smooth=0.9648821352 orth=1.272075e-03 gradnorm=1.281906e+00
    chol  |U|F2=3066.8718286541 smooth=0.9648821352 orth=1.272075e-03 gradnorm=1.281906e+00

So the objective and its gradients are unchanged. This is a numerically
stable route to the same quantity, not a different model.

### 2. The ridge does not separate near-identical eigenvalues

Confirmed: `Sigma + ridge*I` shifts every eigenvalue equally and leaves the
gaps `lambda_i - lambda_j` untouched. The `eigh` backward carries terms in
`1/(lambda_i - lambda_j)`, so the ridge cannot stabilise it.

Measured eigenvalue gap at the two init scales:

| init scale | min eigenvalue | min eigenvalue gap | grad norm |
|------------|----------------|--------------------|-----------|
| 0.05       | 3.552e-04      | 7.485e-04          | 1.78e-02  |
| **0.005** (frozen) | 1.041e-06 | **8.597e-08**  | 3.37e-01  |

The frozen init leaves the eigenvalues four orders of magnitude closer
together.

### 3. Cholesky stays finite exactly where eigh produces NaN

3x3 covariance driven to degeneracy, gradient through the whitening matrix:

| case                | ZCA (`eigh`) backward | Cholesky backward |
|---------------------|-----------------------|-------------------|
| exactly degenerate  | **NaN**               | finite, 3.62e+08  |
| near-degenerate     | finite, 3.62e+08      | finite, 3.62e+08  |

This reproduces the observed failure end to end: the optimiser drove the
nuisance covariance to degeneracy, the `eigh` backward emitted NaN into the
parameters, and the next forward pass handed an all-NaN covariance to `eigh`
— which is exactly the captured state (9/9 NaN, `X_finite: True`).

### 4. Caveat for the local validation run

Cholesky removes the NaN but not the magnitude: the gradient is still
**3.6e+08** in the degenerate case, because `whitening_ridge = 1e-6` sits
above the covariance scale (~2.4e-07). The fix makes the failure mode
finite and observable rather than catastrophic; it does not by itself
guarantee that lambda = 1.0 trains sensibly.

The full local Phase 0 (all ten worlds, five seeds, lambda up to 1.0,
100 epochs) should therefore be checked for sane convergence, not merely for
absence of NaN.

---

# Phase 0 re-run outcome (AIRCC job 218512)

Run of the fixed code (commit 645a405) as a standalone phase0 gate, before any
chain submission.

## The numerical fix is confirmed

    checkpoints   50 / 50   (all ten worlds, all five seeds)
    fits          3050
    unhealthy     0
    NaN / LinAlgError / FloatingPointError   none
    dependency check ok: deem=0.2.0          (wheelhouse, no outbound DNS)

G4/nuisance at lambda = 1.0 -- the exact arm, mechanism and lambda that
produced the NaN on job 217597 -- is healthy on all ten worlds.  The
nuisance_whitening health gate passes throughout, so the covariance does grow
above the ridge during training and the init-scale/ridge mismatch is a
transient at initialisation rather than a persistent defect.

## Phase 0's verdict is nonetheless STOP

No admissible lambda exists for the **target** mechanism:

| mechanism | surviving lambdas | nomination |
|-----------|-------------------|------------|
| target (G3)   | **none**  | **None** |
| nuisance (G4) | 0.01      | 0.01 |
| family (G5)   | 0.01      | 0.01 |

`passed` therefore evaluates False and the correct status is
`stop_before_natural_targets`.  The chain was not submitted.

### Why target fails: effect without specificity

At every lambda the target graph also improves worlds where it is required not
to (`length_target`, `class_permutation`, `pure_noise`, and others).  The
violation grows monotonically with lambda:

| lambda | mean delta (positive worlds) | violations | worst violation |
|--------|------------------------------|------------|-----------------|
| 0.01   | +0.00020 | 2 | +0.0033 |
| 0.03   | +0.00059 | 3 | +0.0098 |
| 0.1    | +0.00202 | 7 | +0.0339 |
| 0.3    | +0.00648 | 7 | +0.1069 |
| 1.0    | +0.02567 | 7 | +0.2670 |

### And the converse for the other two: specificity without effect

`nuisance` mean delta on its positive worlds is **negative** (-0.00006 at the
nominated lambda = 0.01).  `family` mean delta is **0.00000** at every lambda.
Both survive the admissibility filter largely by doing nothing.

No mechanism shows a positive and specific effect simultaneously.  This is a
result about the method, not a defect, and is left for review.

## Two further defects found (fixed here, not yet exercised)

1. **A scientific stop was recorded as a crash.**  `arm_auc[arm]` is set to NaN
   when an arm's mechanism nominated no lambda -- an expected outcome, and the
   winner logic already filters it with `np.isfinite`.  But the raw dict was
   still serialised, and `atomic_write_json` uses `allow_nan=False` by design,
   so the run died with `ValueError: Out of range float values are not JSON
   compliant: nan` at `PHASE0_COMPLETE.json`.  The verdict was computed
   correctly and then lost.  NaN is now carried to JSON as `null`; no decision
   logic changed.

2. **`source_hash()` was platform-dependent.**  It keyed on
   `str(path.relative_to(ROOT))`, which yields backslashes on Windows, so
   identical code hashed to `398ddf74...` locally and `ca8e1188...` on Linux.
   That value is recorded in `RUN_IDENTITY.json`, so cross-platform replay
   checks would fail spuriously.  Now uses `as_posix()`.

## Cost note for the restart

Phase-0 checkpoints validate `code_sha256` and `raise SystemExit` on mismatch
rather than recomputing.  Both fixes above change `source_hash()`, so the 50
existing checkpoints are invalidated and the next phase0 re-runs all fits
(~6 h at the observed 8.65 checkpoints/hour).  Re-running purely to emit
`PHASE0_COMPLETE.json` was therefore not worth it; `PHASE0_RESULTS.json`
(3050 rows) is the authoritative artifact and every number above derives from
it.  The next Phase 0 -- needed anyway if the target mechanism is revised --
will write the completion manifest properly.
