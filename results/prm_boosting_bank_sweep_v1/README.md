# L-SML of PRM + telemetry banks, and the oracle-count ceiling (2026-10-18)

AIRCC CPU job **288319** (`prmboost_banksweep`; `-p cpu`, `owner_940`, 96G, 8 CPUs; 6 min 42 s, exit 0).
Script `lsml_bank_sweep.py` runs from `eval_mirror_prm_gen_v1`, and `run_sweep.sh` is the runner. The log is
`prmboost_banksweep_288319.out`. The tables below are built from `SWEEP_RESULTS_*`, `SWEEP_BOOTSTRAP_*` and
`SWEEP_DIAG_*` by `make_tables.py`; the same tables are also in `TABLES.md`. N = 6,969 answers per PRM for PRMScore and
6,030 for within-AUC. This is PRMBench development data, not external evidence. Every arm is label-free
except where noted:
LSML_CONT, S5 and S6 use label-chosen channels, and ORACLE uses the true error count, so it is a ceiling, not a method.
No fixed-weight averaged arm is reported here.

**Digit content of each bank.** N11 and N13 are digit-free. N14 and N16 contain the three digit channels. S5 and S6 are
label-selected and contain `digit_spread`.

## Parity gate: PASS

All 20 checks match `../prm_boosting_controls_v1/CONTROLS_RESULTS_*.csv` exactly (difference 0.0 in within-AUC,
PRMScore and all 9 category scores). The checks are PRMG_N16 = LSML_N16_PRMG, AUTO_N16 = LSML_N16_AUTO, LSML_CONT, and
PRM_native, for each of the 5 PRMs. For Qwen-7B the values are 0.8298/67.80, 0.8243/67.49 and 0.8495/69.48.
`skipped_banks` is empty for all 5 PRMs, so every bank ran on every PRM.

## What it shows (PRMScore, Q80; * = 95% source-group CI excludes 0)

1. **L-SML over PRM + a label-free bank beats the native PRM wherever the PRM is weak, and on large banks for the
   strong PRMs.**
   - RLHFlow and MathShepherd gain +7.5 to +14.7 with every nested bank and every arm. Qwen-7B gains +1.2 to +2.8.
   - Qwen-72B and PURE gain only with N16: PRMG +0.65* / +1.56* and CHAIN +0.73* / +1.97*.
   - With N11 these two lose or tie: PURE PRMG -1.79*, Qwen-72B CHAIN -1.10*.
   - With the digit-free banks, the best result for Qwen-72B and PURE is CHAIN_N13, at +0.63 / +0.27 (CIs include 0).
2. **The label-free nested banks mostly do not reach LSML_CONT.**
   - The exceptions are N16 on PURE (CHAIN +0.60*) and on RLHFlow (PRMG +0.81*, CHAIN +0.39*).
   - The label-selected S5 bank beats LSML_CONT on 4 of 5 PRMs (+0.5 to +2.7). That is a label-selected comparison,
     not a label-free result.
3. **CHAIN versus PRMG.**
   - Standardizing each group score before the cross stage raises the PRM's cross-stage share in almost every cell
     (Table 4).
   - On the 20 nested-bank cells it is better in 12, worse in 3 (Qwen-72B N11; RLHFlow and MathShepherd N16), and
     inconclusive in 5.
   - On S5/S6 it is mixed: 2 better, 3 worse, 5 inconclusive. CHAIN is undetermined (K+1 = 3) in 6 of those 10 cells,
     all of them S5/S6 on Qwen-7B, Qwen-72B, PURE and RLHFlow.
   - Effect size is small: -0.6 to +1.2 PRMScore.
4. **PRM weight share does not fall as the bank grows** (Table 3).
   - In PRMG it stays at about 0.08-0.15 from N11 through N16 on four PRMs, and about 0.02 on MathShepherd.
   - In 15 of 20 nested-bank cells, AUTO merges the PRM into a telemetry group ("merged"). Where it does, the PRM
     share is 0.004-0.10.
5. **Count headroom (ORACLE - Q80) on fused scores is +5.8 to +8.0 for every fused arm** (Table 5).
   - That is the same size as the telemetry-only arm, so adding the PRM does not use up the count headroom.
   - The native PRM's +16.8 on MathShepherd reflects a badly calibrated Q80 count for that PRM.
   - Highest ceilings: label-free nested CHAIN_N13 on Qwen-72B, 76.51; label-selected CHAIN_S5 on Qwen-72B, 78.76.
6. **Identical rows are real.**
   - On MathShepherd, N11 and N14 give the same numbers in every arm, and AUTO_N13 = AUTO_N16.
   - In those fits the digit group receives a cross-stage weight of exactly 0 (`PRMG_cross_weights` in
     `SWEEP_DIAG_math_shepherd.json`) and the rest of the grouping is unchanged. That is a fitted outcome, not a
     copying error.

### Table 1. within-AUC / PRMScore (Q80)

| arm | Qwen-7B | Qwen-72B | PURE | RLHFlow | MathShepherd |
|---|---|---|---|---|---|
| PRM_native | 0.8012 / 65.49 | 0.7948 / 68.18 | 0.7628 / 65.31 | 0.6624 / 54.43 | 0.6844 / 47.02 |
| LSML_CONT | 0.8495 / 69.48 | 0.8565 / 69.84 | 0.8107 / 66.69 | 0.7716 / 63.53 | 0.7589 / 63.68 |
| TELE_N11 | 0.8046 / 66.42 | 0.7989 / 65.49 | 0.7439 / 62.55 | 0.7337 / 61.88 | 0.6872 / 58.46 |
| PRMG_N11 | 0.8180 / 67.03 | 0.8266 / 67.65 | 0.7639 / 63.52 | 0.7418 / 61.97 | 0.6921 / 58.70 |
| AUTO_N11 | 0.8180 / 67.03 | 0.8265 / 67.63 | 0.7588 / 63.34 | 0.7413 / 62.01 | 0.7027 / 59.54 |
| CHAIN_N11 | 0.8274 / 67.49 | 0.8216 / 67.08 | 0.7736 / 64.22 | 0.7428 / 62.02 | 0.7019 / 59.48 |
| TELE_N13 | 0.7986 / 66.24 | 0.8196 / 67.17 | 0.7677 / 64.90 | 0.7491 / 63.09 | 0.7159 / 60.88 |
| PRMG_N13 | 0.8205 / 67.33 | 0.8342 / 68.37 | 0.7826 / 65.49 | 0.7555 / 63.11 | 0.7235 / 61.36 |
| AUTO_N13 | 0.8076 / 66.73 | 0.8311 / 68.08 | 0.7668 / 64.84 | 0.7552 / 63.23 | 0.7285 / 61.70 |
| CHAIN_N13 | 0.8288 / 67.82 | 0.8415 / 68.81 | 0.7887 / 65.59 | 0.7580 / 63.07 | 0.7269 / 61.56 |
| TELE_N14 | 0.8121 / 66.70 | 0.8060 / 66.04 | 0.7660 / 64.05 | 0.7446 / 62.67 | 0.6872 / 58.46 |
| PRMG_N14 | 0.8228 / 67.38 | 0.8252 / 67.45 | 0.7824 / 64.86 | 0.7522 / 62.81 | 0.6921 / 58.70 |
| AUTO_N14 | 0.8228 / 67.38 | 0.8176 / 66.83 | 0.7781 / 64.73 | 0.7533 / 62.97 | 0.7027 / 59.54 |
| CHAIN_N14 | 0.8311 / 67.85 | 0.8284 / 67.66 | 0.7911 / 65.45 | 0.7507 / 62.61 | 0.7019 / 59.48 |
| TELE_N16 | 0.8176 / 67.11 | 0.8230 / 67.53 | 0.7918 / 66.33 | 0.7668 / 64.28 | 0.7072 / 60.14 |
| PRMG_N16 | 0.8298 / 67.80 | 0.8400 / 68.83 | 0.8047 / 66.88 | 0.7753 / 64.34 | 0.7133 / 60.58 |
| AUTO_N16 | 0.8243 / 67.49 | 0.8278 / 67.83 | 0.7990 / 66.67 | 0.7647 / 63.95 | 0.7285 / 61.70 |
| CHAIN_N16 | 0.8377 / 68.25 | 0.8428 / 68.90 | 0.8156 / 67.29 | 0.7749 / 63.92 | 0.7119 / 60.31 |
| TELE_S5 | 0.8367 / 68.91 | 0.8428 / 69.01 | 0.7955 / 66.25 | 0.7931 / 65.63 | 0.7541 / 63.76 |
| PRMG_S5 | 0.8535 / 69.65 | 0.8644 / 70.71 | 0.8193 / 67.46 | 0.7990 / 66.18 | 0.7842 / 65.29 |
| AUTO_S5 | 0.8484 / 69.36 | 0.8609 / 70.51 | 0.8236 / 67.57 | 0.7950 / 65.94 | 0.7842 / 65.29 |
| CHAIN_S5 | 0.8542 / 69.63 | 0.8653 / 70.75 | 0.8164 / 67.19 | 0.7980 / 65.91 | 0.7891 / 65.40 |
| TELE_S6 | 0.8327 / 68.63 | 0.8407 / 68.89 | 0.8008 / 66.54 | 0.7790 / 65.33 | 0.7208 / 61.46 |
| PRMG_S6 | 0.8477 / 69.29 | 0.8600 / 70.38 | 0.8228 / 67.68 | 0.7764 / 64.62 | 0.7389 / 62.36 |
| AUTO_S6 | 0.8424 / 69.03 | 0.8547 / 69.90 | 0.8154 / 67.45 | 0.7841 / 65.34 | 0.7424 / 62.70 |
| CHAIN_S6 | 0.8497 / 69.37 | 0.8626 / 70.59 | 0.8265 / 67.61 | 0.7731 / 64.10 | 0.7586 / 63.57 |

### Table 2. PRMScore contrast vs PRM_native and vs LSML_CONT (Q80), 95% source-group CI

| arm | Qwen-7B | Qwen-72B | PURE | RLHFlow | MathShepherd |
|---|---|---|---|---|---|
| PRMG_N11 - PRM_native | +1.54 [+0.90, +2.20]* | -0.53 [-1.15, +0.13] | -1.79 [-2.44, -1.10]* | +7.54 [+6.89, +8.23]* | +11.69 [+10.10, +13.26]* |
| PRMG_N11 - LSML_CONT | -2.45 [-2.83, -2.07]* | -2.19 [-2.61, -1.77]* | -3.17 [-3.57, -2.77]* | -1.56 [-2.01, -1.10]* | -4.98 [-5.46, -4.51]* |
| AUTO_N11 - PRM_native | +1.54 [+0.90, +2.20]* | -0.55 [-1.18, +0.10] | -1.97 [-2.63, -1.28]* | +7.58 [+6.93, +8.26]* | +12.53 [+10.95, +14.02]* |
| AUTO_N11 - LSML_CONT | -2.45 [-2.83, -2.07]* | -2.22 [-2.62, -1.80]* | -3.35 [-3.76, -2.94]* | -1.52 [-1.97, -1.06]* | -4.14 [-4.58, -3.71]* |
| CHAIN_N11 - PRM_native | +2.00 [+1.36, +2.67]* | -1.10 [-1.73, -0.45]* | -1.10 [-1.77, -0.39]* | +7.59 [+6.89, +8.29]* | +12.46 [+10.86, +14.00]* |
| CHAIN_N11 - LSML_CONT | -1.99 [-2.37, -1.62]* | -2.76 [-3.18, -2.36]* | -2.47 [-2.87, -2.06]* | -1.52 [-1.95, -1.07]* | -4.21 [-4.67, -3.75]* |
| PRMG_N13 - PRM_native | +1.84 [+1.16, +2.54]* | +0.20 [-0.45, +0.86] | +0.18 [-0.49, +0.89] | +8.69 [+7.99, +9.38]* | +14.34 [+12.75, +15.84]* |
| PRMG_N13 - LSML_CONT | -2.16 [-2.54, -1.77]* | -1.47 [-1.87, -1.06]* | -1.20 [-1.52, -0.85]* | -0.42 [-0.81, -0.01]* | -2.33 [-2.77, -1.89]* |
| AUTO_N13 - PRM_native | +1.24 [+0.57, +1.95]* | -0.10 [-0.73, +0.56] | -0.48 [-1.13, +0.24] | +8.80 [+8.12, +9.50]* | +14.69 [+13.12, +16.17]* |
| AUTO_N13 - LSML_CONT | -2.75 [-3.14, -2.35]* | -1.76 [-2.17, -1.34]* | -1.85 [-2.22, -1.48]* | -0.30 [-0.70, +0.10] | -1.98 [-2.40, -1.54]* |
| CHAIN_N13 - PRM_native | +2.33 [+1.69, +3.00]* | +0.63 [-0.01, +1.26] | +0.27 [-0.37, +0.98] | +8.64 [+7.95, +9.34]* | +14.54 [+12.97, +16.01]* |
| CHAIN_N13 - LSML_CONT | -1.67 [-2.00, -1.31]* | -1.04 [-1.41, -0.67]* | -1.10 [-1.42, -0.77]* | -0.47 [-0.86, -0.06]* | -2.12 [-2.57, -1.69]* |
| PRMG_N14 - PRM_native | +1.90 [+1.26, +2.55]* | -0.73 [-1.38, -0.06]* | -0.45 [-1.13, +0.22] | +8.38 [+7.70, +9.06]* | +11.69 [+10.10, +13.26]* |
| PRMG_N14 - LSML_CONT | -2.10 [-2.50, -1.67]* | -2.39 [-2.85, -1.92]* | -1.83 [-2.21, -1.44]* | -0.72 [-1.16, -0.26]* | -4.98 [-5.46, -4.51]* |
| AUTO_N14 - PRM_native | +1.90 [+1.26, +2.55]* | -1.35 [-1.99, -0.68]* | -0.59 [-1.25, +0.09] | +8.54 [+7.87, +9.23]* | +12.53 [+10.95, +14.02]* |
| AUTO_N14 - LSML_CONT | -2.10 [-2.50, -1.67]* | -3.01 [-3.50, -2.53]* | -1.96 [-2.36, -1.56]* | -0.56 [-1.01, -0.09]* | -4.14 [-4.58, -3.71]* |
| CHAIN_N14 - PRM_native | +2.36 [+1.73, +3.03]* | -0.52 [-1.18, +0.16] | +0.13 [-0.52, +0.84] | +8.18 [+7.51, +8.86]* | +12.46 [+10.86, +14.00]* |
| CHAIN_N14 - LSML_CONT | -1.63 [-2.02, -1.23]* | -2.19 [-2.62, -1.75]* | -1.24 [-1.60, -0.86]* | -0.92 [-1.34, -0.48]* | -4.21 [-4.67, -3.75]* |
| PRMG_N16 - PRM_native | +2.31 [+1.66, +2.97]* | +0.65 [+0.02, +1.32]* | +1.56 [+0.87, +2.28]* | +9.91 [+9.20, +10.62]* | +13.57 [+11.98, +15.06]* |
| PRMG_N16 - LSML_CONT | -1.68 [-2.06, -1.27]* | -1.01 [-1.41, -0.60]* | +0.19 [-0.14, +0.52] | +0.81 [+0.45, +1.19]* | -3.10 [-3.54, -2.64]* |
| AUTO_N16 - PRM_native | +2.00 [+1.36, +2.65]* | -0.35 [-1.00, +0.32] | +1.36 [+0.68, +2.08]* | +9.52 [+8.82, +10.22]* | +14.69 [+13.12, +16.17]* |
| AUTO_N16 - LSML_CONT | -1.99 [-2.37, -1.58]* | -2.01 [-2.48, -1.54]* | -0.02 [-0.34, +0.33] | +0.42 [+0.03, +0.86]* | -1.98 [-2.40, -1.54]* |
| CHAIN_N16 - PRM_native | +2.76 [+2.13, +3.41]* | +0.73 [+0.10, +1.40]* | +1.97 [+1.30, +2.69]* | +9.49 [+8.82, +10.19]* | +13.29 [+11.73, +14.78]* |
| CHAIN_N16 - LSML_CONT | -1.23 [-1.59, -0.86]* | -0.94 [-1.32, -0.56]* | +0.60 [+0.28, +0.91]* | +0.39 [+0.05, +0.74]* | -3.37 [-3.85, -2.93]* |
| PRMG_S5 - PRM_native | +4.16 [+3.49, +4.88]* | +2.53 [+1.85, +3.20]* | +2.14 [+1.49, +2.86]* | +11.76 [+11.02, +12.49]* | +18.27 [+16.74, +19.69]* |
| PRMG_S5 - LSML_CONT | +0.17 [-0.12, +0.45] | +0.87 [+0.59, +1.16]* | +0.77 [+0.51, +1.04]* | +2.65 [+2.29, +3.04]* | +1.60 [+1.31, +1.88]* |
| AUTO_S5 - PRM_native | +3.87 [+3.21, +4.59]* | +2.33 [+1.65, +3.00]* | +2.26 [+1.62, +2.99]* | +11.51 [+10.75, +12.24]* | +18.27 [+16.74, +19.69]* |
| AUTO_S5 - LSML_CONT | -0.12 [-0.42, +0.19] | +0.67 [+0.35, +0.99]* | +0.88 [+0.55, +1.23]* | +2.41 [+2.05, +2.80]* | +1.60 [+1.31, +1.88]* |
| CHAIN_S5 - PRM_native | +4.14 [+3.48, +4.84]* | +2.57 [+1.93, +3.21]* | +1.87 [+1.23, +2.60]* | +11.48 [+10.75, +12.19]* | +18.38 [+16.87, +19.79]* |
| CHAIN_S5 - LSML_CONT | +0.15 [-0.11, +0.43] | +0.91 [+0.65, +1.17]* | +0.50 [+0.24, +0.77]* | +2.38 [+2.01, +2.76]* | +1.72 [+1.42, +2.01]* |
| PRMG_S6 - PRM_native | +3.80 [+3.12, +4.52]* | +2.20 [+1.55, +2.87]* | +2.37 [+1.71, +3.09]* | +10.19 [+9.49, +10.88]* | +15.34 [+13.79, +16.84]* |
| PRMG_S6 - LSML_CONT | -0.19 [-0.50, +0.11] | +0.54 [+0.24, +0.87]* | +0.99 [+0.70, +1.31]* | +1.09 [+0.71, +1.51]* | -1.33 [-1.70, -0.94]* |
| AUTO_S6 - PRM_native | +3.54 [+2.86, +4.25]* | +1.72 [+1.06, +2.38]* | +2.14 [+1.46, +2.87]* | +10.91 [+10.20, +11.63]* | +15.69 [+14.15, +17.19]* |
| AUTO_S6 - LSML_CONT | -0.45 [-0.76, -0.13]* | +0.06 [-0.29, +0.41] | +0.76 [+0.45, +1.11]* | +1.80 [+1.43, +2.21]* | -0.98 [-1.29, -0.64]* |
| CHAIN_S6 - PRM_native | +3.88 [+3.22, +4.58]* | +2.41 [+1.79, +3.05]* | +2.30 [+1.64, +3.03]* | +9.67 [+8.97, +10.37]* | +16.55 [+15.01, +18.03]* |
| CHAIN_S6 - LSML_CONT | -0.11 [-0.41, +0.18] | +0.75 [+0.48, +1.04]* | +0.92 [+0.62, +1.25]* | +0.57 [+0.19, +0.97]* | -0.11 [-0.50, +0.28] |

### Table 3. PRM weight share in the fused score (least squares, |w_PRM| / sum |w|)

| bank | arm | Qwen-7B | Qwen-72B | PURE | RLHFlow | MathShepherd |
|---|---|---|---|---|---|---|
| N11 | PRMG | 0.142 | 0.110 | 0.102 | 0.111 | 0.019 |
| N11 | AUTO | 0.142 (alone) | 0.115 (alone) | 0.076 (merged) | 0.098 (merged) | 0.090 (merged) |
| N11 | CHAIN | 0.207 | 0.145 | 0.161 | 0.172 | 0.045 |
| N13 | PRMG | 0.152 | 0.092 | 0.087 | 0.096 | 0.026 |
| N13 | AUTO | 0.071 (merged) | 0.067 (merged) | 0.004 (merged) | 0.085 (merged) | 0.032 (alone) |
| N13 | CHAIN | 0.291 | 0.153 | 0.148 | 0.141 | 0.053 |
| N14 | PRMG | 0.111 | 0.083 | 0.091 | 0.089 | 0.019 |
| N14 | AUTO | 0.111 (alone) | 0.057 (merged) | 0.062 (merged) | 0.076 (merged) | 0.090 (merged) |
| N14 | CHAIN | 0.196 | 0.134 | 0.158 | 0.149 | 0.045 |
| N16 | PRMG | 0.116 | 0.086 | 0.096 | 0.114 | 0.022 |
| N16 | AUTO | 0.058 (merged) | 0.005 (merged) | 0.047 (merged) | 0.070 (merged) | 0.032 (alone) |
| N16 | CHAIN | 0.237 | 0.150 | 0.206 | 0.237 | 0.033 |
| S5 | PRMG | 0.218 | 0.209 | 0.209 | 0.195 | 0.198 |
| S5 | AUTO | 0.156 (merged) | 0.175 (alone) | 0.219 (merged) | 0.209 (merged) | 0.198 (alone) |
| S5 | CHAIN | 0.277 | 0.272 | 0.234 | 0.206 | 0.228 |
| S6 | PRMG | 0.190 | 0.177 | 0.236 | 0.194 | 0.095 |
| S6 | AUTO | 0.123 (merged) | 0.140 (merged) | 0.168 (merged) | 0.173 (alone) | 0.144 (merged) |
| S6 | CHAIN | 0.270 | 0.254 | 0.301 | 0.267 | 0.125 |

### Table 4. CHAIN - PRMG (Q80) with the PRM cross-stage share in each

Cell: dPRMScore [CI] ; dAUC ; PRM cross share PRMG -> CHAIN ; K+1 (u = CHAIN_undetermined)

| bank | Qwen-7B | Qwen-72B | PURE | RLHFlow | MathShepherd |
|---|---|---|---|---|---|
| N11 | +0.46 [+0.26, +0.67]* ; +0.0093 ; 0.20 -> 0.22 ; 6 | -0.57 [-0.78, -0.35]* ; -0.0050 ; 0.16 -> 0.16 ; 6 | +0.70 [+0.50, +0.89]* ; +0.0097 ; 0.16 -> 0.18 ; 5 | +0.04 [-0.16, +0.24] ; +0.0010 ; 0.17 -> 0.19 ; 5 | +0.77 [+0.63, +0.93]* ; +0.0098 ; 0.04 -> 0.05 ; 4 |
| N13 | +0.49 [+0.30, +0.67]* ; +0.0083 ; 0.26 -> 0.32 ; 4 | +0.43 [+0.24, +0.62]* ; +0.0073 ; 0.14 -> 0.17 ; 6 | +0.09 [-0.06, +0.24] ; +0.0061 ; 0.13 -> 0.17 ; 6 | -0.05 [-0.25, +0.15] ; +0.0025 ; 0.14 -> 0.15 ; 7 | +0.20 [+0.00, +0.41]* ; +0.0035 ; 0.05 -> 0.06 ; 5 |
| N14 | +0.47 [+0.30, +0.64]* ; +0.0083 ; 0.17 -> 0.21 ; 6 | +0.20 [+0.03, +0.39]* ; +0.0032 ; 0.13 -> 0.14 ; 7 | +0.59 [+0.42, +0.77]* ; +0.0087 ; 0.14 -> 0.18 ; 6 | -0.20 [-0.39, +0.00] ; -0.0015 ; 0.14 -> 0.17 ; 6 | +0.77 [+0.63, +0.93]* ; +0.0098 ; 0.04 -> 0.05 ; 5 |
| N16 | +0.45 [+0.28, +0.61]* ; +0.0079 ; 0.20 -> 0.26 ; 5 | +0.08 [-0.13, +0.28] ; +0.0028 ; 0.14 -> 0.16 ; 7 | +0.41 [+0.24, +0.59]* ; +0.0109 ; 0.17 -> 0.23 ; 5 | -0.42 [-0.65, -0.18]* ; -0.0004 ; 0.21 -> 0.26 ; 5 | -0.27 [-0.48, -0.07]* ; -0.0014 ; 0.04 -> 0.04 ; 7 |
| S5 | -0.02 [-0.15, +0.10] ; +0.0007 ; 0.30 -> 0.32 ; 3 u | +0.03 [-0.11, +0.19] ; +0.0010 ; 0.29 -> 0.31 ; 3 u | -0.27 [-0.38, -0.14]* ; -0.0028 ; 0.25 -> 0.26 ; 4 | -0.27 [-0.39, -0.16]* ; -0.0010 ; 0.22 -> 0.21 ; 5 | +0.11 [-0.04, +0.26] ; +0.0050 ; 0.22 -> 0.24 ; 5 |
| S6 | +0.08 [-0.06, +0.22] ; +0.0020 ; 0.28 -> 0.31 ; 3 u | +0.21 [+0.03, +0.39]* ; +0.0026 ; 0.27 -> 0.30 ; 3 u | -0.07 [-0.27, +0.13] ; +0.0037 ; 0.35 -> 0.36 ; 3 u | -0.52 [-0.70, -0.34]* ; -0.0033 ; 0.29 -> 0.31 ; 3 u | +1.21 [+1.04, +1.39]* ; +0.0197 ; 0.11 -> 0.13 ; 6 |

### Table 5. ORACLE - Q80 PRMScore (count headroom; ORACLE uses labels)

| arm | Qwen-7B | Qwen-72B | PURE | RLHFlow | MathShepherd |
|---|---|---|---|---|---|
| PRM_native | 72.50 (+7.01) | 73.54 (+5.36) | 68.79 (+3.48) | 62.26 (+7.83) | 63.82 (+16.80) |
| LSML_CONT | 76.90 (+7.42) | 77.83 (+7.99) | 73.05 (+6.36) | 69.92 (+6.39) | 70.07 (+6.39) |
| TELE_N11 | 73.37 (+6.95) | 72.84 (+7.35) | 68.77 (+6.22) | 68.16 (+6.29) | 64.26 (+5.80) |
| PRMG_N11 | 74.14 (+7.11) | 75.05 (+7.40) | 70.03 (+6.51) | 68.38 (+6.41) | 64.61 (+5.91) |
| AUTO_N11 | 74.14 (+7.11) | 75.05 (+7.43) | 69.68 (+6.34) | 68.38 (+6.37) | 65.42 (+5.88) |
| CHAIN_N11 | 74.91 (+7.42) | 74.70 (+7.63) | 70.61 (+6.39) | 68.21 (+6.20) | 65.24 (+5.77) |
| TELE_N13 | 73.03 (+6.79) | 74.59 (+7.42) | 70.74 (+5.84) | 69.45 (+6.35) | 66.78 (+5.90) |
| PRMG_N13 | 74.46 (+7.14) | 75.89 (+7.52) | 71.50 (+6.01) | 69.70 (+6.58) | 67.34 (+5.99) |
| AUTO_N13 | 73.54 (+6.81) | 75.57 (+7.48) | 70.66 (+5.82) | 69.64 (+6.41) | 67.62 (+5.91) |
| CHAIN_N13 | 74.91 (+7.09) | 76.51 (+7.71) | 71.73 (+6.15) | 69.66 (+6.59) | 67.77 (+6.22) |
| TELE_N14 | 73.70 (+7.00) | 72.84 (+6.81) | 70.28 (+6.22) | 69.18 (+6.51) | 64.26 (+5.80) |
| PRMG_N14 | 74.22 (+6.83) | 74.38 (+6.93) | 71.31 (+6.44) | 69.34 (+6.53) | 64.61 (+5.91) |
| AUTO_N14 | 74.22 (+6.83) | 73.70 (+6.87) | 71.07 (+6.34) | 69.40 (+6.43) | 65.42 (+5.88) |
| CHAIN_N14 | 74.89 (+7.04) | 74.92 (+7.27) | 71.75 (+6.30) | 69.05 (+6.44) | 65.24 (+5.77) |
| TELE_N16 | 74.14 (+7.03) | 74.39 (+6.86) | 72.49 (+6.16) | 70.75 (+6.47) | 66.12 (+5.98) |
| PRMG_N16 | 74.91 (+7.11) | 76.17 (+7.34) | 73.13 (+6.25) | 71.07 (+6.73) | 66.49 (+5.91) |
| AUTO_N16 | 74.59 (+7.10) | 74.89 (+7.06) | 72.84 (+6.17) | 70.62 (+6.67) | 67.62 (+5.91) |
| CHAIN_N16 | 75.40 (+7.15) | 76.37 (+7.46) | 73.73 (+6.45) | 70.70 (+6.77) | 66.46 (+6.15) |
| TELE_S5 | 76.29 (+7.38) | 76.40 (+7.39) | 72.56 (+6.31) | 72.68 (+7.05) | 70.10 (+6.34) |
| PRMG_S5 | 77.31 (+7.66) | 78.59 (+7.88) | 74.01 (+6.55) | 72.98 (+6.80) | 71.79 (+6.51) |
| AUTO_S5 | 76.94 (+7.59) | 78.10 (+7.58) | 74.13 (+6.56) | 72.83 (+6.89) | 71.79 (+6.51) |
| CHAIN_S5 | 77.28 (+7.65) | 78.76 (+8.02) | 73.68 (+6.50) | 72.72 (+6.81) | 71.93 (+6.53) |
| TELE_S6 | 76.05 (+7.41) | 76.35 (+7.46) | 73.22 (+6.68) | 72.26 (+6.93) | 67.43 (+5.96) |
| PRMG_S6 | 76.86 (+7.57) | 78.22 (+7.84) | 74.41 (+6.73) | 71.46 (+6.84) | 68.61 (+6.25) |
| AUTO_S6 | 76.58 (+7.55) | 77.61 (+7.70) | 73.88 (+6.43) | 72.08 (+6.74) | 68.96 (+6.26) |
| CHAIN_S6 | 77.10 (+7.73) | 78.52 (+7.92) | 74.39 (+6.78) | 70.84 (+6.73) | 69.94 (+6.37) |

### Table 6. Groups found (K) and skipped banks

| bank | Qwen-7B | Qwen-72B | PURE | RLHFlow | MathShepherd |
|---|---|---|---|---|---|
| N11 | TELE 5, PRMG 6, AUTO 6 | TELE 5, PRMG 6, AUTO 5 | TELE 4, PRMG 5, AUTO 4 | TELE 4, PRMG 5, AUTO 4 | TELE 3, PRMG 4, AUTO 3 |
| N13 | TELE 3, PRMG 4, AUTO 3 | TELE 5, PRMG 6, AUTO 5 | TELE 5, PRMG 6, AUTO 5 | TELE 6, PRMG 7, AUTO 5 | TELE 4, PRMG 5, AUTO 4 |
| N14 | TELE 5, PRMG 6, AUTO 6 | TELE 6, PRMG 7, AUTO 5 | TELE 5, PRMG 6, AUTO 5 | TELE 5, PRMG 6, AUTO 5 | TELE 4, PRMG 5, AUTO 4 |
| N16 | TELE 4, PRMG 5, AUTO 4 | TELE 6, PRMG 7, AUTO 5 | TELE 4, PRMG 5, AUTO 4 | TELE 4, PRMG 5, AUTO 6 | TELE 6, PRMG 7, AUTO 5 |
| S5 | TELE 2, PRMG 3, AUTO 2 | TELE 2, PRMG 3, AUTO 5 | TELE 3, PRMG 4, AUTO 2 | TELE 4, PRMG 5, AUTO 2 | TELE 4, PRMG 5, AUTO 5 |
| S6 | TELE 2, PRMG 3, AUTO 2 | TELE 2, PRMG 3, AUTO 2 | TELE 2, PRMG 3, AUTO 3 | TELE 2, PRMG 3, AUTO 5 | TELE 5, PRMG 6, AUTO 4 |

skipped_banks: Qwen-7B: none; Qwen-72B: none; PURE: none; RLHFlow: none; MathShepherd: none
