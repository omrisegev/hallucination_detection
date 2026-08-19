# White-box NRM-CS-IU addendum

## Status

**PRELIMINARY / VALIDATION BLOCKED; retrospective post-v2 analysis.** The
white-box v2 result is not mutated or replaced. NRM scores were fitted without
correctness fields and frozen before evaluation, but the hypothesis was
proposed after v2 outcomes were historically visible. Corrected live Gate B
and the independent architecture-fidelity pilot also remain incomplete.

## Method

The exact anchor-oriented IU-PCR score is decomposed into either:

- four architecture-relative depth quartiles on `resid-core-L`; or
- twelve fixed module-by-metric families on `lens-96`.

Within every source cell, group contributions are standardized and
residualized against standardized IU. Source residual covariance matrices are
averaged equally by cell. NRM selects the eigenvector whose eigenvalue is
closest to one, orients it toward the equal-family risk direction, and applies
the frozen `1/G` correction scale to the unlabeled target cell.

The roster is not fully crossed, so simultaneous dataset-and-model exclusion
is not identifiable for GSM8K/Llama. Three explicit calibrations are reported
instead:

- leave-dataset-out (`LODO`);
- leave-model-out (`LOMO`); and
- leave-one-cell-out (`LOCO`) sensitivity.

## Main results

Equal-cell macro over 13 eligible cells:

| Method | AUROC | AUPRC |
|---|---:|---:|
| IU-PCR, residual core | 0.6206 | 0.4812 |
| Depth NRM, LODO | 0.6182 | 0.4769 |
| Depth NRM, LOMO | 0.6250 | 0.4851 |
| Depth NRM, LOCO | 0.6251 | 0.4852 |
| IU-PCR, lens-96 | 0.7202 | 0.5960 |
| Lens NRM, LODO | 0.7191 | 0.5938 |
| Lens NRM, LOMO | 0.7187 | 0.5931 |
| Final-layer NLL | 0.7298 | 0.5892 |

Key paired AUROC contrasts:

| Contrast | Delta [95% grouped-bootstrap CI] | W/T/L | Worst cell |
|---|---:|---:|---:|
| Depth NRM LODO − residual IU | −0.00244 [−0.00575,+0.00073] | 6/1/6 | −0.01670 |
| Depth NRM LOMO − residual IU | +0.00438 [+0.00126,+0.00760] | 8/1/4 | −0.01358 |
| Depth NRM LODO − final NLL | −0.11162 [−0.12450,−0.09903] | 3/0/10 | −0.20498 |
| Lens NRM LODO − lens IU | −0.00106 [−0.00212,+0.00002] | 5/0/8 | −0.00617 |
| Lens NRM LOMO − lens IU | −0.00152 [−0.00257,−0.00043] | 4/2/7 | −0.00589 |

For AUPRC, lens LODO is also negative versus IU: −0.00227
[−0.00380,−0.00045]. Holm-adjusted Wilcoxon p-values are all 1.0 and are
treated only as low-power supporting evidence.

## Decision

Do not adopt NRM as the white-box method. The small Depth-NRM improvement
under leave-model-out is a bounded new-data hypothesis, not a robust result:
it reverses under leave-dataset-out, the richer lens version is negative, and
final-layer NLL remains substantially stronger. The next honest test is the
exact frozen Depth-NRM LOMO rule on a new, fully crossed model-by-dataset
capture without retuning.

The full responsive report, per-cell metrics, paired deltas, calibration
directions, score hashes, and manifests are in [REPORT.html](REPORT.html).
