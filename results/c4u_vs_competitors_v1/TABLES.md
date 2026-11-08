# C4U (frozen, C4U-parisi-within-jaffe-cross-v1) vs PRISM, CROP, SLIDER, Mind the Gap

Input R1 in every cell (raw banks from AIRCC, see README). PRM and AUTO rows: RESULTS_BACKBONE.csv of telemetry_only_vs_competitors_v1 (same code; PRM rows re-computed here and parity-gated). C4U flags = its own DS-posterior count; threshold protocols use the C4U rank.

## PRISM: PRMBench PRMScore (Table 2)

| PRM | PRM own thr | AUTO | C4U | C4U TPR | C4U FPR | C4U @TPR 87.3 | AUTO @TPR 87.3 | PRM @TPR 87.3 |
|---|---|---|---|---|---|---|---|---|
| Qwen-PRM-7B | 65.49 | 69.87 | 71.44 | 88.6 | 42.5 | 69.05 | 67.46 | 68.26 |
| Qwen-PRM-72B | 68.18 | 70.18 | 72.58 | 89.1 | 40.7 | 70.12 | 68.02 | 69.22 |
| PURE | 65.31 | 69.06 | 69.94 | 89.5 | 48.4 | 67.52 | 66.81 | 66.41 |
| RLHFlow | 54.43 | 65.92 | 67.88 | 90.0 | 54.4 | 65.50 | 64.15 | 55.26 |
- PRISM paper (retrained Qwen-PRM-7B): PRMScore 68.00 at TPR 87.30 / FPR 47.13. Baseline Qwen-PRM-7B 65.50.
- "@TPR 87.3" = global threshold chosen with labels to match PRISM's TPR (fairness check, not a method).

## SLIDER: PRMBench redundancy class

| PRM | PRM acc | AUTO acc | C4U acc | C4U bal-acc (own flags) | AUTO bal-acc (GAP) |
|---|---|---|---|---|---|
| Qwen-PRM-7B | 50.44 | 75.84 | 74.40 | 72.30 | 72.33 |
| Qwen-PRM-72B | 69.83 | 76.28 | 76.61 | 75.27 | 75.25 |
| PURE | 59.28 | 76.40 | 74.71 | 72.36 | 73.65 |
| RLHFlow | 54.56 | 69.35 | 69.40 | 68.99 | 70.24 |
- acc = plain accuracy, class-balanced 1-D LR threshold, 10 trace-level 50/50 splits (SLIDER protocol). Paper Step-RRI 61.91.
- Majority class (always "not redundant") = 86.8 plain accuracy; bal-acc is the informative number.

## CROP Table 6, alpha 0.05: PRMBench

| PRM | PRM kept | AUTO kept | C4U kept | C4U risk | PRM AUROC | AUTO AUROC | C4U AUROC |
|---|---|---|---|---|---|---|---|
| Qwen-PRM-7B | 17.2 | 21.6 | 24.1 | 4.6 | 0.800 | 0.781 | 0.809 |
| Qwen-PRM-72B | 20.4 | 20.5 | 24.1 | 4.8 | 0.784 | 0.780 | 0.813 |
| PURE | 16.9 | 18.9 | 21.5 | 5.0 | 0.781 | 0.748 | 0.776 |
| RLHFlow | 8.7 | 15.1 | 18.5 | 4.5 | 0.631 | 0.709 | 0.732 |
- CROP paper (Qwen-PRM-7B): Trace+PRM LR 21.8, Direct PRM 4.8 (fixed grid), random 8.7. 201-quantile grid for every row; threshold calibrated with labels for every row.

## CROP Table 6, alpha 0.05: ProcessBench

| PRM | PRM kept | AUTO kept | C4U kept | C4U risk | PRM AUROC | AUTO AUROC | C4U AUROC |
|---|---|---|---|---|---|---|---|
| Qwen-PRM-7B | 44.2 | 13.8 | 18.9 | 5.5 | 0.883 | 0.771 | 0.824 |
| Qwen-PRM-72B | 49.6 | 14.4 | 21.3 | 4.6 | 0.907 | 0.783 | 0.859 |
| PURE | 28.1 | 11.2 | 12.6 | 5.5 | 0.765 | 0.740 | 0.765 |
| RLHFlow | 20.8 | 9.5 | 8.9 | 5.2 | 0.699 | 0.724 | 0.722 |
- CROP paper (Qwen-PRM-7B): Trace+PRM LR 42.4, Direct PRM 46.7, random 9.8. 201-quantile grid for every row; threshold calibrated with labels for every row.

## CROP Table 6, alpha 0.05: CRV-Arith

| PRM | PRM kept | AUTO kept | C4U kept | C4U risk | PRM AUROC | AUTO AUROC | C4U AUROC |
|---|---|---|---|---|---|---|---|
| Qwen-PRM-7B | 96.9 | 22.8 | 48.2 | 4.0 | 0.943 | 0.685 | 0.755 |
| Qwen-PRM-72B | 97.0 | 32.0 | 89.3 | 4.4 | 0.978 | 0.697 | 0.250 |
| PURE | 90.3 | 44.3 | 64.1 | 5.1 | 0.877 | 0.387 | 0.654 |
| RLHFlow | 95.1 | 50.6 | 43.7 | 4.8 | 0.860 | 0.424 | 0.621 |
- CROP paper (Qwen-PRM-7B): Trace+PRM LR 97.1, Direct PRM 97.5, random 62.0. 201-quantile grid for every row; threshold calibrated with labels for every row.

## CROP Table 6, alpha 0.05: CRV-GSM8K

| PRM | PRM kept | AUTO kept | C4U kept | C4U risk | PRM AUROC | AUTO AUROC | C4U AUROC |
|---|---|---|---|---|---|---|---|
| Qwen-PRM-7B | 80.9 | 29.2 | 44.3 | 5.0 | 0.849 | 0.582 | 0.662 |
| Qwen-PRM-72B | 81.4 | 28.7 | 39.0 | 4.5 | 0.879 | 0.588 | 0.671 |
| PURE | 63.0 | 31.1 | 34.0 | 4.7 | 0.847 | 0.527 | 0.555 |
| RLHFlow | 62.3 | 35.1 | 51.5 | 5.5 | 0.797 | 0.522 | 0.581 |
- CROP paper (Qwen-PRM-7B): Trace+PRM LR 63.1, Direct PRM 81.7, random 27.2. 201-quantile grid for every row; threshold calibrated with labels for every row.

## CROP Table 6, alpha 0.05: PRM800K

| PRM | PRM kept | AUTO kept | C4U kept | C4U risk | PRM AUROC | AUTO AUROC | C4U AUROC |
|---|---|---|---|---|---|---|---|
| Qwen-PRM-7B | 69.1 | 30.5 | 48.4 | 4.9 | 0.930 | 0.709 | 0.824 |
| Qwen-PRM-72B | 75.4 | 35.8 | 44.7 | 5.2 | 0.950 | 0.808 | 0.882 |
| PURE | 61.8 | 23.6 | 28.0 | 5.1 | 0.878 | 0.685 | 0.726 |
| RLHFlow | 37.3 | 13.6 | 22.5 | 5.5 | 0.748 | 0.563 | 0.336 |
- CROP paper (Qwen-PRM-7B): Trace+PRM LR 89.5, Direct PRM 67.2, random 20.5. 201-quantile grid for every row; threshold calibrated with labels for every row.

## CROP Table 17: cross-domain (risk % / kept %)

| PRM | direction | PRM | AUTO | C4U |
|---|---|---|---|---|
| Qwen-PRM-7B | crv_arith->crv_gsm8k | 14.5 / 94.8 | 9.1 / 48.0 | 16.2 / 95.1 |
| Qwen-PRM-7B | crv_gsm8k->crv_arith | 0.7 / 77.3 | 1.7 / 18.3 | 0.8 / 17.9 |
| Qwen-PRM-72B | crv_arith->crv_gsm8k | 14.2 / 94.7 | 12.8 / 59.5 | 19.6 / 98.6 |
| Qwen-PRM-72B | crv_gsm8k->crv_arith | 0.3 / 73.5 | 0.9 / 18.2 | 0.0 / 20.1 |
| PURE | crv_arith->crv_gsm8k | 10.9 / 88.9 | 6.5 / 38.2 | 5.4 / 37.6 |
| PURE | crv_gsm8k->crv_arith | 0.7 / 33.2 | 3.2 / 33.5 | 5.0 / 62.1 |
| RLHFlow | crv_arith->crv_gsm8k | 13.7 / 91.5 | 14.7 / 78.3 | 18.6 / 97.4 |
| RLHFlow | crv_gsm8k->crv_arith | 0.3 / 34.1 | 1.6 / 26.5 | 0.1 / 0.9 |
- CROP paper: Arith->GSM8K risk 14.8 / kept 69.9; GSM8K->Arith 5.1 / 69.0. Target risk 5%.

## Mind the Gap Table 3: ProcessBench SLA (2,221 error answers)

| PRM | PRM own thr | backbone TELE16 GAP | C4U macro [95% CI] | C4U overall | C4U > MtG 39.39 |
|---|---|---|---|---|---|
| Qwen-PRM-7B | 61.87 | 47.23 | 56.65 [54.4, 59.1] | 54.30 | yes |
| Qwen-PRM-72B | 71.22 | 50.64 | 61.97 [59.7, 64.2] | 60.20 | yes |
| PURE | 38.98 | 33.21 | 34.39 [32.2, 36.6] | 32.87 | no, below |
| RLHFlow | 19.09 | 32.52 | 17.32 [15.5, 19.2] | 14.68 | no, below |
- SLA = exact match of the first error step; prediction = first flagged step. MtG (Shannon Drop, Qwen3-8B, no PRM): macro 39.39 / overall 38.11.
- CI: 2,000 answer resamples stratified by subset, no refit.

## PRISM Table 6: ProcessBench average F1

| PRM | PRM own thr | AUTO (PRM gate, NC) | C4U (own count) | C4U correct-answer acc, GSM8K |
|---|---|---|---|---|
| Qwen-PRM-7B | 72.94 | 63.21 | 1.32 | 0.5 |
| Qwen-PRM-72B | 78.20 | 67.27 | 0.90 | 1.0 |
| PURE | 52.58 | 37.69 | 0.53 | 0.0 |
| RLHFlow | 29.16 | 33.17 | 0.00 | 0.0 |
- C4U flags at least one step in almost every answer, so its accuracy on correct answers is near 0 and F1 collapses. AUTO borrows the PRM gate (NC rule) for this metric.

## ARES
- Not run. ARES was dropped from deck v2 (Omri, 2026-10-29); no ARES protocol cell was asked here.

## ProcessBench with the PRM gate (NC rule), added 2026-11-08 (c4u_processbench_nc.py, RESULTS_PB_NC.csv)

NC: the PRM decides whether an answer has an error and how many steps; C4U's rank decides which steps. Same rule as AUTO_N16_NC.

| PRM | PRM own thr avg F1 | AUTO NC avg F1 | C4U NC avg F1 | C4U own count avg F1 | PRM own thr macro SLA | C4U NC macro SLA [95% CI] | C4U own count macro SLA |
|---|---|---|---|---|---|---|---|
| Qwen-PRM-7B | 72.94 | 63.21 | 69.68 | 1.32 | 61.87 | 57.40 [55.2, 59.6] | 56.65 |
| Qwen-PRM-72B | 78.20 | 67.27 | 75.64 | 0.90 | 71.22 | 67.08 [64.9, 69.1] | 61.97 |
| PURE | 52.58 | 37.69 | 42.62 | 0.53 | 38.98 | 28.74 [26.6, 30.8] | 34.39 |
| RLHFlow | 29.16 | 33.17 | 29.83 | 0.00 | 19.09 | 19.51 [17.7, 21.5] | 17.32 |
- MtG (Qwen3-8B, no PRM): macro SLA 39.39. PRM native F1 re-computed through eval_processbench, parity 4/4 vs RESULTS_BACKBONE.
