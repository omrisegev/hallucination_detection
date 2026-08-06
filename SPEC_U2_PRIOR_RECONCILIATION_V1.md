# U2-prior reconciliation — development specification

Status: **frozen diagnostic checkpoint; consumed development data and saved
artifacts only**

## Why this checkpoint exists

Target-Anchored Laplacian IU-PCR (TA-LIU) showed that a few labels can identify
the target-relevant feature block, but it used those labels much less
effectively than ordinary logistic regression on the two leading covariance
directions.  The proposed response—fit a logistic head in U2 with a prior
centred on IU-PCR—may already be the method called `anchored_pcr2` in the
earlier semi-supervised study.

This checkpoint reconciles those definitions before another estimator is
invented.  It is not a method-confirmation experiment and cannot support a new
performance claim.

## Questions

1. Is current IU-prior logistic merely a change of coordinates of a
   two-dimensional U2 logistic head?
2. Is the historical `anchored_pcr2` also equivalent, or does its
   exclusion/recompute U-PCR configuration change the two-dimensional space?
3. Even with evaluation-label access, is there at least one AUROC point of
   usable headroom between IU-PCR and U2 logistic?
4. Do the saved 24-cell real-artifact results leave the U2-prior direction
   open?

## Frozen constructions

For a feature-by-sample matrix `F`, let `C = F F^T / n`.  The ordinary U2
basis contains the two leading eigenvectors of `C`.  Each score direction is
centred and divided by its transductive standard deviation, so its columns are
orthonormal in the `C` metric.

The following methods receive the same calibration indices and are evaluated
on the identical non-calibration complement:

1. `iu`: ordinary current IU-PCR.
2. `ta_liu`: the already frozen TA-LIU construction (`lambda=0.1`, `k=7`).
3. `u2_logistic`: the preregistered fixed-L2 logistic control on ordinary U2
   coordinates.
4. `anchored_pcr2_historical`: the v1 semi-supervised head.  Its first basis
   direction uses the historical exclusion/recompute U-PCR configuration and
   its coefficient prior is `(1, 0)` with strength 10.
5. `anchored_pcr2_current`: the same head, except that the first direction is
   current ordinary IU-PCR.
6. `anchored_pcr2_current_reparameterized`: method 5 represented in the
   covariance-normalized ordinary U2 basis.  Its prior is transformed by the
   measured basis map; its penalty and intercept penalty are unchanged.

The current IU-PCR configuration is frozen to `loss="l2"`,
`exclusion=False`, `difficulty_gate=False`, `simple_avg_fallback=False`,
`recompute_after_exclusion=False`, `n_components=2`,
`auto_components=False`, `g2_projection_k=1`, and `scale_ratio=0.25`.  The
historical configuration is separately frozen to `loss="l2"`,
`exclusion=True`, `difficulty_gate=False`, `simple_avg_fallback=True`,
`recompute_after_exclusion=True`, `g2_projection_k=1`, and
`scale_ratio=0.25`, with its original component defaults.

Methods 5 and 6 are equivalent only when their measured subspaces agree.  The
experiment must not assume this.  For current and historical anchored bases it
stores principal angles, Euclidean projector distance, covariance-metric
orthonormality, basis reconstruction error, and prior-coordinate maps.  For
method 6 it additionally stores objective, effective-weight, intercept, and
score equality against method 5.

Current-basis equivalence requires all of: maximum principal angle at most
`1e-7` radians, projector Frobenius distance at most `1e-7`, relative basis
reconstruction error at most `1e-7`, and covariance-orthonormality error at
most `1e-8`.  Conditional on those geometric gates, fitted equivalence requires
absolute objective difference at most `1e-7`, effective-weight and intercept
maximum absolute differences at most `1e-7`, and maximum absolute score
difference at most `1e-7 * max(1, std(score))`.  If any geometric gate fails,
method 6 is skipped and reported as an invalid reparameterization; a projection
into U2 may be diagnosed but must not be named an equivalent method.  The
historical basis is tested against the same geometry thresholds, but a failure
is an expected possible result of exclusion/recomputation rather than a harness
failure.

If a calibration prefix has one class, supervised logistic controls use their
already declared deterministic constant-score fallback.  The anchored heads
remain mathematically defined because their prior regularizes the coefficients;
the event is recorded and cannot be used to select a method.

## Synthetic replay

Only the already consumed paired selective-target development matrices are
regenerated:

- seed offset `40,000`;
- eight independent matrices;
- the identical `F` is scored once with target `g` and once with target `u`;
- `n=360`;
- 16 deterministic calibration permutations;
- nested budgets `{4, 8, 16, 32, 64}`.

No confirmation seed in the reserved `2,600,000` block may be generated.  The
matrix and calibration-index hashes must match across paired target views.

AUROC on the non-calibration complement is primary and AUPRC is secondary.
Calibration draws are averaged within each dataset before uncertainty is
computed; the eight matrices are the independent units.

## Optimistic closure oracles

For every evaluation set, IU and U2-logistic scores are separately centred and
scaled using all transductive samples without labels.  The following deliberately
optimistic controls are priced:

- the better endpoint (`max(AUC_IU, AUC_U2)`);
- a fixed 50/50 normalized score average;
- the best normalized interpolation on the grid `alpha = 0, .005, ..., 1`,
  selected and evaluated on the same evaluation labels.

These controls may reject the already tested candidates but may not close the
whole U2 span and may not open a new candidate.  In particular, the better
endpoint is **not** a ceiling on interior interpolation.  If an optimistic
control exceeds one point, a later independently selected or cross-fitted rule would still
need mean gain at least one point, lower confidence bound above zero, positive
equal-family macro performance, QA and math each no worse than -0.5 points,
and leave-one-family-out sensitivity before a new estimator cycle could open.

## Saved-real-artifact reconciliation

No raw hallucination features or labels are opened.  The checkpoint reads only
these frozen saved outputs:

- `results/semi_supervised_spectral_v1/replicates.csv`: exactly 39,600 data
  rows, including 32,400 real rows; exactly 5,760 real budget-20 rows covering
  24 cells, 30 repetitions, and eight methods.  This artifact uses
  `confidence-orientation-v1`, removes the four quarantined views, and uses the
  separately frozen historical U-PCR configuration above.
- `results/upcr_study/11_posthoc_controls/mix_splithalf.csv`: exactly 24 data
  rows, five split-half repetitions per cell.
- `results/upcr_study/11_posthoc_controls/summary.json`: `n_cells=24` and the
  corresponding `mix_splithalf` summary.

The Step-227 mixing artifact is explicitly **context only**: its generating
script is `scripts/upcr_study/exp11_posthoc_controls.py`, it sweeps 721 angles
over `[0, pi)` on one row half and evaluates the chosen angle on the other, but
uses the historical full canonical feature pool, per-split `sign(rho)`
orientation, and the historical deployed U-PCR configuration.  It is not
configuration-compatible with the current fixed-stable candidate and cannot
close that channel.

At budget 20 the current-schema semi-supervised artifact is averaged first over
30 repetitions within `(cell, method)`.  Only then are paired cell deltas,
wins/losses, group means, and 10,000-draw paired cell-bootstrap 95% intervals
computed.  It reconstructs:

- `gold_pcr2`, `anchored_pcr2`, and `anchored_pcr6` mean delta versus U-PCR;
- win/loss counts;
- the optimistic cell-level binary-switch ceiling;
- QA and math group summaries when identifiable;
- the saved historical held-out full-angle mixing control from Step 227 as
  incompatible contextual evidence, with its provenance flag preserved.

Normalized row-score interpolation cannot be reconstructed from AUROCs alone;
that missing information is reported rather than imputed.  A cell-level binary
endpoint switch receives its own paired cell-bootstrap interval and is named an
optimistic endpoint switch, never an interpolation ceiling.

## Decision rule

Stop developing the **already tested U2 logistic and anchored-head family** if
none of its current-schema saved methods improves U-PCR, and every optimistic
cell-level endpoint switch is below `+1.00` AUROC point.  This does not prove
that every angle or every future U2 estimator is closed.  Synthetic gains may
explain a mechanism but cannot override the target-application stop decision.

If the family stops, stop for discussion.  The next experiment is not chosen
automatically: first price the still-open few-label stability-selected subset
channel at the same 20-label budget; then compare its attainable headroom with
a current-schema, recycling-guarded FUSE-derived pseudo-target probe.

## Required outputs

- raw synthetic metrics and compressed row-level scores;
- dataset-level and method-level synthetic summaries;
- basis-geometry and reparameterization diagnostics;
- real-artifact replay table and decision JSON;
- plots of method deltas, basis geometry, optimistic oracle ceilings, and saved
  real-artifact performance;
- an explicit statement that no confirmation data were generated.

After implementation and execution, an independent read-only agent must audit
the code, geometry claims, aggregation unit, artifacts, decision, and plots.
Then stop for discussion regardless of the outcome.
