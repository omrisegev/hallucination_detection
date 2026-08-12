# Specification: neutral residual mode CS-IU v1

**Date:** 2026-08-13

**Status:** frozen label-free candidate.  The original 23-cell, ProcessBench,
and SemGrad results are retrospective.  HLE/Qwen2.5-72B is the first
candidate-frozen independent-example confirmation.

## Motivation and scope

The supervised HARP-inspired global contribution teacher established that a
single target-aligned direction exists in the six-family IU-PCR contribution
space and transfers to held-out examples.  Cardinality balancing failed on
SemGrad, so feature multiplicity is not an adequate nuisance identifier.

NRM-CS-IU replaces the supervised teacher with a spectral rule learned from
unlabelled source batches.  It treats large shared residual modes as
cross-family nuisance and near-zero modes as deterministic redundancy.  The
candidate retains the residual mode closest to the unit-variance null expected
for standardized independent residual variation.

The method changes fusion only.  It uses the existing mixed-v2 feature matrix,
one ordinary IU-PCR fit, and the frozen provenance-family registry.  It adds no
model inference, feature, label, hidden state, or white-box access.

## Per-cell contribution coordinates

For every calibration or target cell, independently and without labels:

1. fit ordinary mixed-v2 IU-PCR;
2. decompose its score into the exact family contributions
   `h_g(x) = sum_{i in g} w_i f_i(x)`;
3. standardize IU to `b` and every contribution to unit variance;
4. regress every standardized contribution on `b` and standardize the
   resulting residual column, yielding `R`.

The columns of `R` follow the fixed six-family `VIEW_ORDER`.

## Unlabelled cross-cell calibration

The calibration source is the 23 eligible original cells.  Correctness labels
must not be passed to the calibration API.

For every pair of families `(g,h)`, average `(R_c^T R_c / n_c)[g,h]` across
only source cells in which both families are present.  Symmetrize the resulting
six-by-six matrix `C_R`.  Let `(lambda_j, v_j)` be its ordered eigensystem.

Select

```text
j* = argmin_j |lambda_j - 1|.
```

Orient `v_j*` so its dot product with the all-ones equal-family anchor is
positive.  An exact-zero tie uses the sign of its largest-magnitude entry.
No eigenvalue range, component count, direction sign, or source cell is tuned
against correctness.

## Target score

On a target cell, retain the entries of the frozen six-dimensional direction
for the families that are present.  Let `q = R v`.  The score is

```text
s_NRM = b + q / (G * sd(q)),
```

where `G` is the number of present families.  A degenerate zero-variance mode
returns exact IU.  The `1/G` trust scale is the same fixed convention already
used by the earlier label-free contribution candidates; it is not selected on
HLE.

The entire target-cell operation is affine in the existing feature matrix.
The implementation must expose the equivalent per-feature weights and
intercept and numerically reconstruct the score.

## Evaluation protocol

### Retrospective evidence

- Original cells: leave one dataset family out, recalibrate from the remaining
  unlabelled cells, then evaluate every cell in the held-out family.
- External transfer: calibrate once on all 23 unlabelled original cells and
  transfer unchanged to Qwen ProcessBench, Llama ProcessBench, and SemGrad.

These results characterize the candidate but cannot confirm it because their
labels or prior method results were available during development.

### Frozen HLE confirmation

Use all 2,158 HLE Qwen2.5-72B rows in deterministic row-key order.  The score
phase may receive only the four whitelisted telemetry arrays used by mixed-v2.
It writes row IDs, IU, CB-CS-IU, and NRM-CS-IU scores plus source/data/code
hashes.  Verify those hashes before a separate report phase reads the interim
Codex-judge sidecar.

Correctness (`correct == "yes"`) is the positive class.  Report AUROC and a
20,000-draw paired stratified bootstrap of the NRM-minus-IU delta.  Also report
answer-type diagnostics, but do not make them primary because each has only
31--37 positives.

Pre-registered gates:

1. score/source/data hashes verify before labels are read;
2. the scoring payload and fitting API contain no target/label field;
3. all values are finite, the IU decomposition reconstructs, the correction is
   orthogonal to IU on the target batch, and effective weights reconstruct;
4. HLE NRM-minus-IU AUROC point delta is positive;
5. the paired 95% bootstrap lower bound is positive.

Failure of gate 5 is an inconclusive/noisy confirmation, not permission to tune
the mode or trust scale on HLE.  Failure of gate 4 rejects v1.

## Interpretation boundary

The source calibration is unsupervised but trans-environment: it learns one
fixed direction from multiple unlabelled batches.  It is not a per-cell-only
method.  The neutral-eigenvalue rule encodes a structural assumption, not an
identifiability theorem that guarantees separation for arbitrary data.

The HLE labels are complete but interim: they come from a single Codex judge,
not HLE's original GPT-4o protocol, and only 68/2,158 answers are marked
correct.  Any confirmation is therefore evidence of cross-example/model
transfer under that stated label protocol, not a paper-faithful HLE claim.
