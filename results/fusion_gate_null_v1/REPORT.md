# Controlled fusion-gate check - Step319

Review PASS. No new benchmark candidate or model inference.

768 synthetic source trials, five readouts,3840 valid outputs. All12 conditions retained. Warm uses256 additional synthetic observations as a startup diagnostic.
At N256/rho0/no jump: raw opens0/64, IMM cold26/64, IMM warm26/64. At N64:0/64,11/64,12/64. Input has one stationary Gaussian regime; output distribution need not be Gaussian.
At N64/rho0/+3SD jump: raw21/64, IMM cold/warm64/64. Sensitivity must be preserved in any future calibration.
BIC is model selection, not a5% semantic-error test. These are simulation frequencies, not real-answer false-positive rates.
Simulation106.04s; review12.06s. Three tests;768 source/1536 Kalman/3840 GMM algebra/48 vector IMM/120 GMM refits pass.
Next verify one procedure-matched calibration rule on independent simulations before a gate-only benchmark comparison. Preserve all176 anchors and full research scope.

| N | rho | Source | Readout | Valid | Failures | Open | Rate | 95% CI | Two components |
|---|---|---|---|---|---|---|---|---|---|
| 16 | 0.0 | stationary | raw | 64 | 0 | 15 | 23.44% | [13.75%, 35.69%] | 15 |
| 16 | 0.0 | stationary | kalman_cold | 64 | 0 | 16 | 25.00% | [15.02%, 37.40%] | 16 |
| 16 | 0.0 | stationary | imm_cold | 64 | 0 | 14 | 21.88% | [12.51%, 33.97%] | 14 |
| 16 | 0.0 | stationary | kalman_warm | 64 | 0 | 18 | 28.12% | [17.60%, 40.76%] | 18 |
| 16 | 0.0 | stationary | imm_warm | 64 | 0 | 14 | 21.88% | [12.51%, 33.97%] | 14 |
| 16 | 0.0 | jump | raw | 64 | 0 | 20 | 31.25% | [20.24%, 44.06%] | 20 |
| 16 | 0.0 | jump | kalman_cold | 64 | 0 | 23 | 35.94% | [24.32%, 48.90%] | 23 |
| 16 | 0.0 | jump | imm_cold | 64 | 0 | 46 | 71.88% | [59.24%, 82.40%] | 46 |
| 16 | 0.0 | jump | kalman_warm | 64 | 0 | 58 | 90.62% | [80.70%, 96.48%] | 58 |
| 16 | 0.0 | jump | imm_warm | 64 | 0 | 60 | 93.75% | [84.76%, 98.27%] | 60 |
| 16 | 0.6 | stationary | raw | 64 | 0 | 12 | 18.75% | [10.08%, 30.46%] | 12 |
| 16 | 0.6 | stationary | kalman_cold | 64 | 0 | 30 | 46.88% | [34.28%, 59.77%] | 30 |
| 16 | 0.6 | stationary | imm_cold | 64 | 0 | 26 | 40.62% | [28.51%, 53.63%] | 26 |
| 16 | 0.6 | stationary | kalman_warm | 64 | 0 | 30 | 46.88% | [34.28%, 59.77%] | 30 |
| 16 | 0.6 | stationary | imm_warm | 64 | 0 | 22 | 34.38% | [22.95%, 47.30%] | 22 |
| 16 | 0.9 | stationary | raw | 64 | 0 | 15 | 23.44% | [13.75%, 35.69%] | 15 |
| 16 | 0.9 | stationary | kalman_cold | 64 | 0 | 38 | 59.38% | [46.37%, 71.49%] | 38 |
| 16 | 0.9 | stationary | imm_cold | 64 | 0 | 34 | 53.12% | [40.23%, 65.72%] | 34 |
| 16 | 0.9 | stationary | kalman_warm | 64 | 0 | 33 | 51.56% | [38.73%, 64.25%] | 33 |
| 16 | 0.9 | stationary | imm_warm | 64 | 0 | 35 | 54.69% | [41.75%, 67.18%] | 35 |
| 64 | 0.0 | stationary | raw | 64 | 0 | 0 | 0.00% | [0.00%, 5.60%] | 0 |
| 64 | 0.0 | stationary | kalman_cold | 64 | 0 | 17 | 26.56% | [16.30%, 39.09%] | 17 |
| 64 | 0.0 | stationary | imm_cold | 64 | 0 | 11 | 17.19% | [8.90%, 28.68%] | 11 |
| 64 | 0.0 | stationary | kalman_warm | 64 | 0 | 8 | 12.50% | [5.55%, 23.15%] | 8 |
| 64 | 0.0 | stationary | imm_warm | 64 | 0 | 12 | 18.75% | [10.08%, 30.46%] | 12 |
| 64 | 0.0 | jump | raw | 64 | 0 | 21 | 32.81% | [21.59%, 45.69%] | 21 |
| 64 | 0.0 | jump | kalman_cold | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 64 | 0.0 | jump | imm_cold | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 64 | 0.0 | jump | kalman_warm | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 64 | 0.0 | jump | imm_warm | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 64 | 0.6 | stationary | raw | 64 | 0 | 0 | 0.00% | [0.00%, 5.60%] | 0 |
| 64 | 0.6 | stationary | kalman_cold | 64 | 0 | 28 | 43.75% | [31.37%, 56.72%] | 28 |
| 64 | 0.6 | stationary | imm_cold | 64 | 0 | 17 | 26.56% | [16.30%, 39.09%] | 17 |
| 64 | 0.6 | stationary | kalman_warm | 64 | 0 | 37 | 57.81% | [44.82%, 70.06%] | 37 |
| 64 | 0.6 | stationary | imm_warm | 64 | 0 | 16 | 25.00% | [15.02%, 37.40%] | 16 |
| 64 | 0.9 | stationary | raw | 64 | 0 | 8 | 12.50% | [5.55%, 23.15%] | 8 |
| 64 | 0.9 | stationary | kalman_cold | 64 | 0 | 53 | 82.81% | [71.32%, 91.10%] | 53 |
| 64 | 0.9 | stationary | imm_cold | 64 | 0 | 31 | 48.44% | [35.75%, 61.27%] | 31 |
| 64 | 0.9 | stationary | kalman_warm | 64 | 0 | 55 | 85.94% | [74.98%, 93.36%] | 55 |
| 64 | 0.9 | stationary | imm_warm | 64 | 0 | 31 | 48.44% | [35.75%, 61.27%] | 31 |
| 256 | 0.0 | stationary | raw | 64 | 0 | 0 | 0.00% | [0.00%, 5.60%] | 0 |
| 256 | 0.0 | stationary | kalman_cold | 64 | 0 | 16 | 25.00% | [15.02%, 37.40%] | 16 |
| 256 | 0.0 | stationary | imm_cold | 64 | 0 | 26 | 40.62% | [28.51%, 53.63%] | 26 |
| 256 | 0.0 | stationary | kalman_warm | 64 | 0 | 14 | 21.88% | [12.51%, 33.97%] | 14 |
| 256 | 0.0 | stationary | imm_warm | 64 | 0 | 26 | 40.62% | [28.51%, 53.63%] | 26 |
| 256 | 0.0 | jump | raw | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 256 | 0.0 | jump | kalman_cold | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 256 | 0.0 | jump | imm_cold | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 256 | 0.0 | jump | kalman_warm | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 256 | 0.0 | jump | imm_warm | 64 | 0 | 64 | 100.00% | [94.40%, 100.00%] | 64 |
| 256 | 0.6 | stationary | raw | 64 | 0 | 0 | 0.00% | [0.00%, 5.60%] | 0 |
| 256 | 0.6 | stationary | kalman_cold | 64 | 0 | 27 | 42.19% | [29.94%, 55.18%] | 27 |
| 256 | 0.6 | stationary | imm_cold | 64 | 0 | 9 | 14.06% | [6.64%, 25.02%] | 9 |
| 256 | 0.6 | stationary | kalman_warm | 64 | 0 | 25 | 39.06% | [27.10%, 52.07%] | 25 |
| 256 | 0.6 | stationary | imm_warm | 64 | 0 | 8 | 12.50% | [5.55%, 23.15%] | 8 |
| 256 | 0.9 | stationary | raw | 64 | 0 | 13 | 20.31% | [11.28%, 32.23%] | 13 |
| 256 | 0.9 | stationary | kalman_cold | 64 | 0 | 42 | 65.62% | [52.70%, 77.05%] | 42 |
| 256 | 0.9 | stationary | imm_cold | 64 | 0 | 21 | 32.81% | [21.59%, 45.69%] | 21 |
| 256 | 0.9 | stationary | kalman_warm | 64 | 0 | 49 | 76.56% | [64.31%, 86.25%] | 49 |
| 256 | 0.9 | stationary | imm_warm | 64 | 0 | 20 | 31.25% | [20.24%, 44.06%] | 20 |
