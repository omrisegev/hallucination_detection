# Target-anchored Laplacian IU-PCR synthetic study

Version: `target-anchored-liu-synthetic-v1-2026-08-06`

## Scope

This report contains the **development** split only. It used generated data; no
real hallucination features or cached benchmark data were opened.
Development uses an already-consumed seed block and is exploratory. It cannot
establish the preregistered claim or change the frozen primary budget k=16.

## Design

- Dataset replicates: 8
- Calibration permutations per dataset: 16
- Samples per dataset: 360; graph k-NN: 7; lambda: 0.1
- Nested label budgets: [4, 8, 16, 32, 64]; frozen primary budget: 16
- Calibration draws are averaged within each dataset. Dataset replicates are
  the uncertainty units in every confidence interval.
- The paired targets use identical F and identical calibration indices.
- Eligible for reserved confirmation: yes.

## Frozen k=16 results

Values are AUROC changes in percentage points versus ordinary IU-PCR,
reported as dataset mean +/- one SE.

| task | DUFS-LIU | Projected ridge | Pseudo-anchor | TA-LIU | U2 logistic | Full logistic | Oracle latent |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smooth signal | +0.613 +/- 0.157 | +0.462 +/- 0.118 | +0.612 +/- 0.157 | +0.579 +/- 0.149 | -1.142 +/- 0.575 | -2.794 +/- 0.529 | +0.633 +/- 0.162 |
| Broad nuisance | -0.456 +/- 0.086 | -0.242 +/- 0.012 | -0.974 +/- 0.086 | +1.668 +/- 0.057 | +13.382 +/- 0.757 | +11.636 +/- 0.837 | +2.038 +/- 0.062 |
| Paired target: g | -1.782 +/- 0.091 | -0.366 +/- 0.052 | -1.434 +/- 0.121 | +1.267 +/- 0.121 | +19.523 +/- 0.445 | +16.972 +/- 0.512 | +1.541 +/- 0.080 |
| Paired target: u | +0.813 +/- 0.073 | +0.164 +/- 0.024 | +0.676 +/- 0.078 | +0.840 +/- 0.081 | +0.740 +/- 0.331 | -1.057 +/- 0.351 | +0.979 +/- 0.080 |
| Correlated errors | -0.016 +/- 0.011 | -0.002 +/- 0.001 | -0.007 +/- 0.008 | -0.021 +/- 0.008 | -2.553 +/- 0.258 | -2.730 +/- 0.292 | +0.028 +/- 0.021 |
| Pure noise | +0.025 +/- 0.016 | +0.000 +/- 0.004 | +0.006 +/- 0.011 | +0.009 +/- 0.011 | +1.449 +/- 1.490 | +1.367 +/- 1.393 | +0.002 +/- 0.003 |

## Preregistered-gate diagnostic

Because this is development, the result below is exploratory only.

| gate | result |
|---|---:|
| selective nuisance rescue | PASS |
| label swap consistency | PASS |
| same label attribution | FAIL |
| smooth preservation | PASS |
| existing nuisance safety | PASS |
| correlated error safety | PASS |
| null safety | PASS |
| identifiability invariants | PASS |

Overall: **FAIL**

The raw CSV retains every calibration draw. `per_dataset.csv` first averages
those draws; `summary.csv`, confidence bounds, win counts, and leave-one-dataset-
out diagnostics all use the dataset replicate as the effective sample size.

## Plots

- `01_development_budget_curves.png`: sample efficiency on all tasks.
- `02_development_label_swap.png`: identical F under the two targets.
- `03_development_role_gates.png`: whether target gates switch planted roles.
- `04_development_primary_controls.png`: frozen k=16 against every control.

Stop for interpretation before opening the reserved confirmation split or any
real hallucination data.
