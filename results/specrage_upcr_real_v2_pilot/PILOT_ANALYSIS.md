# CA-SpecRaGE–LIU ten-cell execution pilot

This is a deliberately non-selective execution pilot over the first ten cells
in the exported bundle. It uses one model seed, 30 epochs, and the synthetic
`agreement_k15` configuration. It is not leave-one-family-out calibration and
must not be cited as a real-data performance estimate.

## Result versus ordinary IU-PCR

Mean AUROC-point changes across the ten cells are:

| graph arm | lambda 0.1 | lambda 0.3 | lambda 1 | lambda 10 | wins at 0.1 |
|---|---:|---:|---:|---:|---:|
| CA-SpecRaGE agreement graph | +0.010 | +0.001 | -0.094 | -0.264 | 6/10 |
| CA-SpecRaGE fused-Y graph | +0.024 | +0.020 | -0.049 | -0.310 | 7/10 |
| end-to-end uniform fused-Y graph | -0.003 | -0.051 | -0.226 | -0.551 | 6/10 |
| raw uniform-view graph | -0.004 | -0.003 | -0.016 | -0.077 | 5/10 |
| DUFS-LIU | +0.019 | +0.006 | -0.035 | -0.112 | 6/10 |

The fixed deployed U-PCR arm is -0.231 points versus full-pool IU-PCR on this
nonrepresentative subset. That does not turn the approximately zero graph gains
into a contribution: CA-SpecRaGE does not separate from DUFS-LIU or ordinary
IU-PCR here.

## Interpretation

The synthetic mechanism transfers in direction only at very small lambda and
with negligible magnitude. The synthetic one-standard-error choice
`lambda=10` does not transfer. Increasing lambda monotonically worsens every
non-oracle real arm, while the synthetic dependent-error worlds improve and
saturate. This is evidence of a world-to-data geometry mismatch, not a reason
to tune lambda on these ten cells.

The agreement learner itself is active: normalized alpha entropy ranges from
0.886 to 0.988 rather than remaining exactly uniform, and the sample-arm loss
falls from 0.217 to 0.077 on average. Its raw spectral condition number remains
below 819 in this pilot and the SVD floor is inactive for the sample arm. The
failure is therefore downstream usefulness of the learned geometry, not the v1
failure in which alpha never moved.

## Decision

- Retain CA-SpecRaGE as a positive synthetic mechanism baseline.
- Do not promote it as the real-data method.
- Do not use this sorted ten-cell subset to tune hyperparameters.
- Before the full grouped development run, measure whether real provenance
  views possess a majority-shared target neighbourhood and whether the learned
  agreement graph changes the IU-PCR projected roughness in a direction that
  labels later confirm is useful.

Reproduce the execution pilot with:

```bash
python scripts/specrage_upcr_real.py \
  --stage smoke \
  --max-cells 10 \
  --out-dir results/specrage_upcr_real_v2_pilot
```
