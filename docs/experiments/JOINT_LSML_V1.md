# Joint L-SML v1 — label-free structural protocol

## Inputs and populations

- Nine target-free donor cells: the eight frozen Qwen ProcessBench cells and the
  frozen Qwen3-8B PRMBench response cell.
- Two structural lanes per cell: `v2_active28` and `h2_24`.
- Exact sanitized C-v2 input manifest and cell NPZ hashes are inherited and
  verified.  Each NPZ exposes only `raw`, `token_offsets`, and `row_ids`.
- ProcessBench and PRMBench remain separate panels.  The 18 lanes are never
  treated as independent efficacy replications.

## Task 1: donor-score ridge stability

Reconstruct each frozen C-v2 lane matrix exactly.  From the frozen C-v2 ledger,
read `structured_fit.model_covariance` and `structured_fit.global_loading`.
Recompute `regularized_covariance_weights` at target conditions
`{1e2,1e3,1e4}` and calculate all three pairwise Spearman correlations of the
in-memory donor scores `Xw`.  A lane passes the requested replacement diagnostic
iff its minimum finite pairwise correlation is at least `0.99`.

Only correlations and weight diagnostics are persisted.  Donor score arrays are
not persisted.

## Orientation and pruning

Orientation is estimated in the absolute raw 29-feature coordinate system.
`trace_length` is nuisance-only.  For each cell, the leading eigenvector of the
off-diagonal covariance over the other 28 standardized raw streams is gauged by
the semantic rule `entropy_series=-1` (higher entropy means lower confidence).

For each active stream:

- `mean(|v_i|)<0.01` is weak and excluded;
- fewer than six matching sign votes among nine cells is unstable and excluded;
- weighted degree is `sum_{j!=i}|v_i v_j|`;
- the per-cell degree rule keeps a stream when its degree is at least `0.1` times
  that cell's median degree;
- the global Agent-A roster retains a stream only when it passes the degree rule
  in at least eight of nine cells and is neither weak nor unstable.

The entropy anchor is protected.  An anchor that fails any rule aborts rather
than being silently restored.  Excluded features retain a schema-only sign from
the current frozen confidence dictionary in the 29-entry registry, but are
explicitly inactive and cannot enter fusion.

For the H2 reference lane, the same raw-feature exclusions apply by name and
`C7_EDIS_ONSET` receives its fixed confidence direction.  A separate H2 degree
vote (same `tau=0.1`, kept in at least eight of nine cells) determines the common
H2 comparison roster.

## LOAO-consensus grouping

Candidate `K` is frozen to `{3,4,6,8}` subject to `K<p`.  For each K and each
held answer:

1. fit the covariance on all other answers;
2. compute `|R-v0 v0^T|` off diagonal;
3. run deterministic spectral clustering;
4. convert labels to a hard co-assignment matrix.

The mean held-answer co-assignment is clustered once at the same K to obtain the
LOAO-consensus partition.  Stability is the ARI of every held-answer partition
against that consensus.  Candidates with `K<3` or any group smaller than three
are rejected.  Selection maximizes median ARI, then mean ARI, then chooses smaller
K.  There is no residual-misfit selection and provenance is not a candidate.
If no candidate survives in a lane, that lane is recorded as
`BLOCKED_NO_ADMISSIBLE_PARTITION`; the other frozen lanes continue unchanged so
the requested 18-lane structural table remains complete. A blocked lane does
not receive a fitted model or weight map.

R2 is a post-failure engineering continuation of the hash-locked R1 run. R1
failed closed before materializing any result payload because one lane lacked an
admissible partition. R2 changes only failure handling so every lane is recorded;
it does not change a threshold, estimator, candidate K, or selection rule.

## Joint estimator

For the selected hard partition, fit

`Sigma_off = vv^T + blockdiag_g(u_g u_g^T)`

with the deterministic five-start Gauss-Seidel block-coordinate estimator from
C-v2: exact scalar least-squares updates, monotone objective, maximum 5,000
sweeps, and the same convergence tolerances.  The global loading is gauged so its
coefficient on the confidence-oriented entropy anchor is positive.

The model diagonal is the nonnegative observed diagonal residual.  Report
clipping count/mass, convergence, multistart fitted-model/loading agreement,
Jacobian diagnostics, and off-diagonal misfit versus the existing hard-L-SML
fit on the identical matrix and partition.

## Weight maps

All maps are computed and compared without labels:

1. `hierarchical_joint`: form each virtual group classifier as `X_g v_g`, run
   SML across the virtual classifiers, and set `w_i=v_i*a_group(i)`.  No covariance
   inverse or diagonal residual enters this map.
2. `model_inverse_1e3`: PSD-project the fitted model covariance and solve the
   condition-controlled system at frozen target condition `1e3`.
3. `sample_inverse_1e3`: apply the same solver and target to the observed sample
   covariance.
4. `continuous_lsml_reference`: the maintained repository implementation with
   the identical fixed consensus partition.

Report every pairwise Spearman correlation among the four in-memory donor scores,
as well as weight norms and finite-status diagnostics.  Scores are not persisted.

## Exact dispatch aliases

- `K=1` dispatches directly to maintained `sml_fuse_signed`; returned scores and
  weights must be bit-identical.
- Requesting `two_stage_alias` dispatches directly to maintained
  `lsml_continuous` with fixed groups; returned scores must be bit-identical.

No mathematical equivalence between different noisy-covariance estimators is
claimed.

## Outputs

- Task-1 18-lane ridge-score stability table and plot.
- Joint-L-SML 18-lane structural table and plots.
- Orientation cell ledger.
- Absolute raw-domain 29-entry orientation registry compatible with Agent A.
- Per-cell removal ledger and global pruned-roster registry compatible with Agent A.
- Tests, execution registry, report, claim audit, and final hash inventory.

The only terminal claim is structural and label-free.  No outcome-scoring protocol
is opened by this run.
