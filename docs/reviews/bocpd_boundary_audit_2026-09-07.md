# BOCPD boundary-convention audit

2026-09-07. Source-level finding during the fused-trajectory pilot. Historical
scores remain unchanged; this note does not recompute or repair old releases.

## Two legitimate conventions

The [original Adams–MacKay paper](https://arxiv.org/html/0710.3742v1), Algorithm
1 and equations (3)–(7), evaluates the current datum under each existing run
and then branches to a boundary after that datum. With constant hazard h,
the normalized r=0 mass can be h at every time. That is not sufficient evidence
of a coding bug: the run-length distribution, including growth hypotheses,
still changes. The old comment that this necessarily kills all change-point
statistics is too broad.

In a reset-before-current-observation model, a newly started segment instead
uses its prior predictive to score the current datum. Its posterior sufficient
statistics must then consume that datum. This gives a different convention
with a coherent probability model; it must be named explicitly.

## What the inspected old code does

`spectral_utils/temporal_models.py:bocpd_gaussian` scores the reset branch
using the prior predictive, but resets mu/kappa/alpha/beta to their unchanged
prior values. It therefore combines reset-before-observation prediction with
reset-after-observation sufficient statistics. The first observation also
does not enter the update loop; the default initialization takes a mean of
the first eight observations while the returned sequence starts earlier.
That initialization must not be called prefix-causal for those early outputs.

This is a source-level inconsistency, not a measured estimate of its effect
on old AUROCs. Existing historical outcomes cannot be relabelled as results
of the corrected model. A faithful replay remains available from unchanged
source/history.

The later `_BOCPDState` in `spectral_utils/unified_causal_iu.py` does update
the reset branch with the current value. Do not generalize the old-module
finding to every BOCPD implementation in the repository.

## New pilot's verification

`spectral_utils/fused_trajectory_readouts.py:bocpd_filter` implements a
reset-before-observation Gaussian product-partition model with known
observation variance and a conjugate Gaussian segment-mean prior. It uses
all nonoverlapping window observations from a frozen fusion trajectory and
does not truncate the run distribution in this small pilot.

The test enumerates every partition of each prefix of a five-observation
sequence. Segment marginal evidence is computed independently with the
Gaussian block covariance `R I + prior_variance 11^T`. Both reset posterior
and posterior mean match the recursion to 1e-12. The first-window onset score
is a separately declared heuristic; it is not a posterior probability of
hallucination. See `tests/test_fused_trajectory_readouts.py` and the frozen
`docs/experiments/FUSED_TRAJECTORY_READOUT_PILOT_V1.md`.

This verifies the stated recursion. It does not establish that telemetry
changes are reasoning errors, nor a universal advantage of one convention.
