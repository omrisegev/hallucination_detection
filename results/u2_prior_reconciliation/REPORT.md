# U2-prior reconciliation — consumed development checkpoint

Version: `u2-prior-reconciliation-v1-2026-08-06`

## Scope

This run regenerated only the already consumed paired synthetic development
matrices and read saved CSV/JSON experiment results. It did not open raw real
hallucination features or labels. It generated no confirmation data and every
synthetic seed remained below the reserved `2,600,000` block.

## Reconciliation result

- Current anchored basis: max principal angle `1.379e-15` radians;
  all eight datasets pass the frozen geometry gates.
- Historical anchored basis: max principal angle `1.499e+00` radians;
  it differed under the frozen historical U-PCR configuration, which includes
  exclusion, recomputation, fallback, and component-setting differences. This
  checkpoint does not isolate which setting caused the difference.
- Current reparameterized fits passing all equality tolerances: `1280/1280`.

Therefore current IU-prior logistic in two dimensions is a coordinate change
of the ordinary U2 head on these matrices. It is not a distinct estimator class.
Historical `anchored_pcr2` must be judged separately when its excluded-feature
basis does not span full-matrix U2.

## Synthetic mechanism at 16 labels

All values below are AUROC points versus ordinary IU-PCR, after averaging the
16 calibration draws within each of eight independent matrices.

| target | TA-LIU | U2 logistic | historical anchored2 | current anchored2 | optimistic interpolation |
|---|---:|---:|---:|---:|---:|
| g | +1.267 | +19.523 | -5.473 | +9.544 | +20.956 |
| u | +0.840 | +0.740 | +2.187 | +1.468 | +2.492 |

The interpolation is selected and scored on the same evaluation labels. It is
an optimistic mechanism diagnostic, not a deployable result.

## Saved real artifacts at 20 labels

The current-schema semi-supervised CSV was averaged over 30 repetitions within
each cell and method before comparing 24 cells.

| tested method | mean delta | 95% cell bootstrap | W/L | endpoint switch mean (95% bootstrap) | maximum cell switch | cells >=1 point |
|---|---:|---:|---:|---:|---:|---:|
| gold_pcr2 | -4.279 | [-5.383, -3.206] | 1/23 | +0.003 ([+0.000, +0.008]) | +0.062 | 0 |
| anchored_pcr2 | -0.149 | [-0.253, -0.049] | 4/20 | +0.043 ([+0.001, +0.095]) | +0.416 | 0 |
| anchored_pcr6 | -0.355 | [-0.636, -0.062] | 8/16 | +0.149 ([+0.036, +0.301]) | +1.491 | 1 |

The historical split-half full-angle sweep was `+0.193`
points (95% interval `[-0.078, +0.513]`).
It is contextual only because it used the historical full feature pool,
per-split sign(rho) orientation, and the historical deployed U-PCR configuration.
It cannot close every angle under the current fixed-stable schema.

## Decision

`stop_tested_family = false`.

The frozen stop rule is literal at cell level. A false flag does not promote
a method: it means at least one cell-specific endpoint switch reached the
one-point threshold, even if the method's overall mean was negative. A true
flag would stop further variants of the tested family. Neither outcome proves
that every U2 angle or every future U2 estimator is closed. The next
branch remains a user decision after comparing few-label subset-adaptation
headroom with a current-schema, recycling-guarded FUSE pseudo-target probe.

## Plots

- `synthetic_method_deltas.png`
- `basis_geometry.png`
- `synthetic_optimistic_controls.png`
- `saved_real_reconciliation.png`

No confirmation experiment was run.
