# Family representatives v1: label-free structure metrics for T(N56), 2026-10-20

Spec: `docs/experiments/FAMILY_REPRESENTATIVES_V1_HE.md` (sections 10-12). **No labels were read** in anything here.
Input: `results/advisor_completion_2026-10-11/inputs/bank_V_N56.npz` (PRMBench, thinkfix mask1, 6,969 answers,
94,203 steps, answer-z). Folds: `population.npz` source-group folds (question identity, not correctness).

## Files
- `redundancy_map.py` -> `REDUNDANCY.json`: copies, near-copies, per-family spectra, |r| clusters.
- `rank1_check.py` -> `RANK1_N56.json`: rank-one residual by F1-F7 partition (first look).
- `metrics_a.py`: S1-S7 (section 11-A) plus S1g. `run_baseline.py` -> `METRICS_A_N56.json`.
  Parity: L-SML inside the calculator reproduces the shipped `S__LSML_CONT` of `bank_N56.npz`, max |diff| 2.6e-14.
- `transforms.py`: T1 copies, T2 block compression (mean / PC1 / PC1+PC2 / PC1+residuals), T3 spline
  residualization on relative step position and log step token length, T4 rank-Gaussianization.
  `run_steps.py` -> `METRICS_A_STEPS.json`.

## Result (lower S1/S1g/S2 is better; higher S3/S7 is better)
| variant | m | S1 | S1g | S2 | S3 | K | groups<3 | pairs>=.95 | neg cross w | fold ARI |
|---|---|---|---|---|---|---|---|---|---|---|
| N56 | 56 | .338 | .522 | .035 | 3.94 | 8 | 0 | 12 | 2 | 1.00 |
| T1 copies | 52 | .384 | .532 | .040 | 4.10 | 8 | 1 | 5 | 2 | .96 |
| T2 PC1, r>=.8 | 39 | .430 | .397 | .103 | 6.28 | 8 | 1 | 0 | 0 | 1.00 |
| T2 PC1+PC2, r>=.8 | 44 | .523 | .414 | .135 | 4.69 | 6 | 0 | 0 | 0 | .59 |
| T2 PC1+residuals, r>=.8 | 45 | .589 | .581 | .075 | 6.80 | 8 | 0 | 0 | 0 | 1.00 |
| T2 PC1, r>=.5 | 21 | .361 | .238 | .302 | 4.93 | 6 | 2 | 0 | 0 | .65 |
| **T3 on T2 PC1 r>=.8** | 39 | **.242** | **.225** | .048 | 3.73 | 7 | 3 | 0 | 1 | 1.00 |
| T3 on T4+T2 | 38 | .186 | .182 | .087 | 3.78 | 7 | 3 | 0 | 2 | .70 |

## Reading
- S1 counts channel pairs, so removing copies (which fit the rank-one model well) raises it. S1g, on the K group
  scores that L-SML's cross stage actually sees, is the cleaner measure. S1g with different K is not strictly comparable.
- Compression to PC1 (or mean; identical at r>=.8) removes all |r|>=.95 pairs, raises the effective number of
  groups 3.9 -> 6.3 and removes the negative cross weights.
- **Contrast features break the structure**, as section 9 predicts: PC2 and residual-on-PC1 columns raise S1/S1g
  and PC2 makes the grouping unstable (ARI .59).
- **Step length is a shared nuisance:** group scores correlate up to .74 with answer-z log token length, all same sign.
  T3 lowers S1g .397 -> .225. But it removes 50-56% of the variance of the energy/top-1 channels (median 18.5%),
  and energy_level is the strongest LR channel; whether that variance is signal cannot be told without labels.

## One-time label evaluation (2026-10-20) — negative
`freeze.py` -> `FREEZE.json` (before labels); `evaluate.py predict` -> `evaluation/PREDICTIONS.npz` + `PREDICTION_SEAL.json`;
`evaluate.py evaluate` -> `evaluation/RESULTS.json`; `diagnose.py` -> `evaluation/DIAGNOSE.json`. See `NEGATIVE_RESULT.md`.
L-SML within-AUC N56 .7704 / A .7490 / B .7392 (PRMScore .6410 / .6282 / .6245); every paired interval below 0.
