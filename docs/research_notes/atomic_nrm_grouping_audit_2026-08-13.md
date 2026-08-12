# Atomic NRM grouping audit — bounded negative result

## Decision

The six provenance families cannot currently be removed from NRM-CS-IU.

The exact retained assumption is:

> Before estimating the residual covariance, IU feature contributions are
> aggregated by the frozen measurement-provenance registry in
> `spectral_utils/specrage_views.py` (`VIEW_ORDER` / `FEATURE_TO_VIEW`).

This is not claimed to be the unique or theoretically optimal grouping.  It is
the only grouping tested here that supplies a consistently useful label-free
orientation beyond the development data.

The frozen family NRM v1 and all its artifacts remain unchanged.

## What was built and frozen before candidate metrics

Atomic NRM decomposes the same ordinary IU score into one contribution per
existing `mixed_v2` feature.  It uses no labels, new inference, hidden state,
white-box quantity, or new feature.  The output reconstructs exactly as one
effective feature-weight vector plus intercept.

The development labels had already been opened in earlier project work, so
this is not a fresh-dataset blindness claim. The structural script itself used
only the frozen 23-cell source telemetry, and the candidate formula, direction,
scale, exclusions, and hashes were fixed before any Atomic NRM metric was
computed or inspected. Seventeen atoms were present and active in all cells;
13 incompletely covered atoms were
frozen out.  A 1,000-draw within-cell permutation null retained a
two-dimensional neutral subspace at eigenvalues 0.960685 and 1.025557.  The
candidate projects a symmetric inverse-absolute-dependence anchor into this
subspace and uses fixed target scale `1/sqrt(17)`.  Its direction, code,
covariance, exclusions, and input hashes were recorded in
`SPEC_ATOMIC_NEUTRAL_RESIDUAL_PROJECTOR_CS_IU_CANDIDATE_V1.md` before any
candidate metric was computed.

Structural checks passed:

- affine reconstruction error below `1e-10`;
- IU/correction covariance below `1e-10`;
- feature-order permutation score error `8.88e-16`;
- no provenance-family registry import in the atomic module;
- no inactive source atom; minimum relative IU weight 0.00531;
- positive-semidefinite common-feature covariance;
- leave-one-cell direction absolute cosine at least 0.975505.

The redesign addresses the main high-dimensional objection to family NRM.  It
does not carry over the fragile `argmin |lambda-1|` rule.

## Retrospective results

All calibration and scoring functions were label-free. The retrospective
loader also read already-opened labels in the same process solely to compute
AUROC; no fit API received them. Original data used
leave-one-dataset-family-out calibration. Qwen and Llama ProcessBench plus
SemGrad are retrospective transfer sets because their labels were already
opened in earlier project steps.

| domain | family NRM | frozen Atomic Projector | direct atomic minus family |
|---|---:|---:|---:|
| original 23 LOFO | +0.277pp | -0.667pp | -0.944pp [-1.654,-0.174] |
| ProcessBench Llama | +1.580pp | -1.106pp | -2.686pp [-3.214,-2.159] |
| ProcessBench Qwen | +0.557pp | -1.305pp | -1.862pp [-2.665,-0.878] |
| SemGrad | +1.310pp | -4.216pp | -5.526pp [-9.005,-2.047] |

The negative result is not explained by choosing the wrong vector inside the
neutral subspace:

- equal-anchor projector: -1.051 / -1.168 / -1.499 / -4.340pp;
- single closest-to-one mode: -0.996 / -1.275 / -1.729 / -4.327pp;
- dependence-learned five-group partition: -0.741 / -1.118 / -1.354 /
  -2.234pp;
- deterministic family refinement: +0.033 / +0.077 / -0.250 / +0.291pp;
- deterministic coarsening: -0.939 / +0.334 / -0.253 / -2.153pp.

Fifty random partitions matched the eligible provenance-family size profile
`[6,4,3,3,1]`.  Only 3/50 matched or beat family NRM on the original cells,
1/50 on Llama ProcessBench, 13/50 on Qwen ProcessBench, and 3/50 on SemGrad.
This shows that the result is not explained by group count or cardinality
alone.

## Supervised ceiling: information is present, orientation is missing

The suggested atomic supervised control used 30 stratified 60/40 splits per
cell, class-balanced anchored logistic loss, and four fixed ridge priors.  AUROC
was averaged per split within cell and then equally by dataset group; no global
out-of-fold concatenation was used.

At every prior, the atomic representation beat the family representation:

| prior | family vs IU | atomic vs IU | atomic minus family |
|---:|---:|---:|---:|
| 0.3 | +0.721pp | +1.298pp | +0.577pp [+0.102,+0.910] |
| 1 | +0.444pp | +1.042pp | +0.598pp [+0.385,+0.786] |
| 3 | +0.209pp | +0.599pp | +0.390pp [+0.264,+0.498] |
| 10 | +0.078pp | +0.231pp | +0.153pp [+0.079,+0.207] |

Thus aggregation does discard some supervised target resolution.  Atomic NRM
failed because the label-free null geometry and symmetric anchor point in the
wrong target direction, not because atomic coordinates lack signal.

## External-label decision

The candidate failed the frozen retrospective gate in every development and
transfer domain.  No genuinely untouched external labels were opened for it.
Consuming a clean target after a decisive development failure would not be a
confirmation; it would only spend held-out evidence on a rejected formula.
There was no post-label pivot.

## Consequence for the thesis

Retain family NRM-CS-IU v1 as the confirmed method and state its provenance
partition as an explicit inductive prior.  The evidence beyond original
development cells is Qwen ProcessBench, Llama ProcessBench, SemGrad, and the
already frozen PRMBench confirmation, all favoring the family rule; the atomic
and learned-group controls fail on the first three retrospective transfer
families.

Future de-grouping work is justified only if it introduces a new, label-free
target-orientation principle.  More ways to choose a vector from the
identity-like covariance bulk are closed by this result.

Canonical artifacts:

- `SPEC_ATOMIC_NEUTRAL_RESIDUAL_PROJECTOR_CS_IU_CANDIDATE_V1.md`;
- `spectral_utils/atomic_neutral_residual.py`;
- `results/atomic_nrm_structural_audit_v1/`;
- `results/atomic_nrm_retrospective_controls_v1/`;
- `results/atomic_contribution_supervised_ceiling_v1/`;
- `docs/research_notes/atomic_nrm_null_spectrum_literature_2026-08-13.md`.
