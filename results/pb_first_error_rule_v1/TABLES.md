# Tables (generated from RESULTS.csv)
## ProcessBench average F1 (4 subsets)
| arm | Qwen-7B | Qwen-72B | PURE | RLHFlow |
|---|---|---|---|---|
| REF PRM native (p>.5) | 72.94 | 78.20 | 52.58 | 29.16 |
| REF C4U own flags | 1.32 | 0.90 | 0.53 | 0.00 |
| REF C4U NC (PRM gate+count) | 69.68 | 75.64 | 42.62 | 29.83 |
| CP PRM (anchored) | 72.81 | 78.67 | 52.22 | 29.21 |
| CP PRM (EM to convergence) | 37.43 | 53.06 | 0.00 | 29.51 |
| CP PRM + drop (anchored) | 72.26 | 78.84 | 54.43 | 29.53 |
| CP C4U rank (PRM anchor) | 59.10 | 68.60 | 7.52 | 19.18 |
| CP PRM + C4U rank (anchored) | 72.22 | 78.81 | 55.08 | 29.48 |
| CP PRM + TELE16 (anchored) | 65.01 | 69.87 | 55.53 | 26.18 |
| CP TELE16 (PRM anchor only) | 56.97 | 59.19 | 46.98 | 20.96 |
| FREE backbone first-pass init | 47.35 | 44.47 | 37.13 | 28.75 |
| FREE CP TELE16 backbone (EM) | 0.00 | 0.00 | 0.00 | 0.00 |
| FREE CP TELE16 backbone (EM, pi=1/2) | 51.46 | 52.04 | 28.97 | 21.46 |
| FREE standalone first-pass init | 29.26 | 30.70 | 30.15 | 29.31 |
| FREE CP TELE16 standalone (EM) | 0.00 | 0.00 | 0.00 | 0.00 |
| FREE CP TELE16 standalone (EM, pi=1/2) | 17.92 | 32.21 | 22.78 | 24.10 |

## Macro SLA, official prediction (a -1 on an error answer is a miss)
| arm | Qwen-7B | Qwen-72B | PURE | RLHFlow |
|---|---|---|---|---|
| REF PRM native (p>.5) | 61.87 | 71.22 | 38.98 | 19.09 |
| REF C4U own flags | 56.65 | 61.97 | 34.39 | 17.32 |
| REF C4U NC (PRM gate+count) | 57.40 | 67.08 | 28.74 | 19.51 |
| CP PRM (anchored) | 61.31 | 70.02 | 38.52 | 19.13 |
| CP PRM (EM to convergence) | 24.89 | 38.55 | 33.13 | 19.98 |
| CP PRM + drop (anchored) | 61.04 | 70.36 | 42.30 | 19.51 |
| CP C4U rank (PRM anchor) | 48.79 | 65.00 | 4.14 | 18.00 |
| CP PRM + C4U rank (anchored) | 61.42 | 70.99 | 42.35 | 19.54 |
| CP PRM + TELE16 (anchored) | 52.27 | 60.39 | 41.99 | 16.56 |
| CP TELE16 (PRM anchor only) | 43.55 | 47.99 | 34.02 | 12.72 |
| FREE backbone first-pass init | 33.86 | 31.39 | 25.12 | 18.73 |
| FREE CP TELE16 backbone (EM) | 13.79 | 15.70 | 23.08 | 16.15 |
| FREE CP TELE16 backbone (EM, pi=1/2) | 38.03 | 39.47 | 18.63 | 12.92 |
| FREE standalone first-pass init | 23.11 | 23.74 | 22.47 | 20.98 |
| FREE CP TELE16 standalone (EM) | 37.55 | 42.10 | 35.37 | 37.04 |
| FREE CP TELE16 standalone (EM, pi=1/2) | 33.34 | 24.05 | 24.78 | 31.66 |

## Macro SLA, Mind the Gap protocol (a step is always predicted; CP = argmax of P(tau=t), t<=n; refs: argmax score if no flag). Mind the Gap Qwen3-8B = 39.39
| arm | Qwen-7B | Qwen-72B | PURE | RLHFlow |
|---|---|---|---|---|
| REF PRM native (p>.5) | 69.54 | 73.91 | 47.97 | 28.52 |
| REF C4U own flags | 56.65 | 61.97 | 34.39 | 17.32 |
| REF C4U NC (PRM gate+count) | 63.80 | 70.19 | 39.90 | 28.84 |
| CP PRM (anchored) | 65.91 | 72.79 | 42.57 | 22.97 |
| CP PRM (EM to convergence) | 33.01 | 44.19 | 33.13 | 24.54 |
| CP PRM + drop (anchored) | 67.12 | 73.46 | 48.92 | 24.62 |
| CP C4U rank (PRM anchor) | 56.87 | 68.28 | 40.31 | 23.25 |
| CP PRM + C4U rank (anchored) | 67.66 | 74.80 | 51.24 | 27.89 |
| CP PRM + TELE16 (anchored) | 60.84 | 64.43 | 52.59 | 25.16 |
| CP TELE16 (PRM anchor only) | 51.19 | 52.87 | 43.72 | 20.97 |
| FREE backbone first-pass init | 33.86 | 31.39 | 25.12 | 18.73 |
| FREE CP TELE16 backbone (EM) | 13.79 | 15.70 | 23.08 | 16.15 |
| FREE CP TELE16 backbone (EM, pi=1/2) | 44.45 | 44.23 | 27.34 | 21.06 |
| FREE standalone first-pass init | 23.11 | 23.74 | 22.47 | 20.98 |
| FREE CP TELE16 standalone (EM) | 37.55 | 42.10 | 35.37 | 37.04 |
| FREE CP TELE16 standalone (EM, pi=1/2) | 35.66 | 37.32 | 34.12 | 37.21 |

## Per-subset F1, C4U + change-point rule vs C4U NC vs PRM
| PRM | arm | GSM8K | MATH | Olympiad | Omni-MATH |
|---|---|---|---|---|---|
| qwen_prm_7b | REF PRM native (p>.5) | 81.77 | 77.23 | 66.69 | 66.05 |
| qwen_prm_7b | REF C4U NC (PRM gate+count) | 79.82 | 74.00 | 62.97 | 61.95 |
| qwen_prm_7b | CP PRM + C4U rank (anchored) | 81.90 | 76.71 | 64.63 | 65.63 |
| qwen_prm_72b | REF PRM native (p>.5) | 87.00 | 80.31 | 74.09 | 71.39 |
| qwen_prm_72b | REF C4U NC (PRM gate+count) | 84.87 | 77.55 | 71.10 | 69.02 |
| qwen_prm_72b | CP PRM + C4U rank (anchored) | 87.20 | 81.02 | 72.99 | 74.02 |
| pure_prm | REF PRM native (p>.5) | 63.35 | 62.94 | 45.32 | 38.70 |
| pure_prm | REF C4U NC (PRM gate+count) | 47.05 | 50.25 | 36.65 | 36.52 |
| pure_prm | CP PRM + C4U rank (anchored) | 66.06 | 62.12 | 47.69 | 44.45 |
| rlhflow_prm | REF PRM native (p>.5) | 51.86 | 34.43 | 14.25 | 16.11 |
| rlhflow_prm | REF C4U NC (PRM gate+count) | 48.65 | 35.38 | 16.33 | 18.97 |
| rlhflow_prm | CP PRM + C4U rank (anchored) | 53.48 | 33.97 | 14.06 | 16.41 |

## Paired bootstrap (1000 answer resamples within subset), see BOOTSTRAP.csv
| PRM | A | B | metric | A-B [95% CI] |
|---|---|---|---|---|
| qwen_prm_7b | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | avg F1 | +2.54 [+1.27, +3.72] |
| qwen_prm_7b | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | macro SLA | +4.02 [+2.38, +5.58] |
| qwen_prm_7b | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | avg F1 | -0.71 [-1.59, +0.23] |
| qwen_prm_7b | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | macro SLA | -0.44 [-1.67, +0.86] |
| qwen_prm_7b | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | macro SLA (forced) | +3.86 [+2.29, +5.78] |
| qwen_prm_7b | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | macro SLA (forced) | -1.87 [-3.31, -0.66] |
| qwen_prm_7b | CP PRM + TELE16 (anchored) | REF PRM native (p>.5) | avg F1 | -7.93 [-9.45, -6.43] |
| qwen_prm_7b | CP PRM + TELE16 (anchored) | REF PRM native (p>.5) | macro SLA | -9.60 [-11.67, -7.70] |
| qwen_prm_7b | CP PRM + drop (anchored) | REF PRM native (p>.5) | avg F1 | -0.68 [-1.66, +0.25] |
| qwen_prm_7b | CP PRM + drop (anchored) | REF PRM native (p>.5) | macro SLA | -0.83 [-2.10, +0.48] |
| qwen_prm_7b | FREE CP TELE16 standalone (EM) | Mind the Gap Qwen3-8B (39.39, published) | macro SLA (forced) | -1.84 [-4.15, +0.56] |
| qwen_prm_7b | FREE CP TELE16 backbone (EM, pi=1/2) | Mind the Gap Qwen3-8B (39.39, published) | macro SLA (forced) | +5.06 [+2.43, +7.44] |
| qwen_prm_72b | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | avg F1 | +3.17 [+2.14, +4.31] |
| qwen_prm_72b | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | macro SLA | +3.92 [+2.34, +5.71] |
| qwen_prm_72b | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | avg F1 | +0.61 [-0.41, +1.55] |
| qwen_prm_72b | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | macro SLA | -0.23 [-1.65, +1.10] |
| qwen_prm_72b | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | macro SLA (forced) | +4.62 [+3.01, +6.32] |
| qwen_prm_72b | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | macro SLA (forced) | +0.90 [-0.40, +2.31] |
| qwen_prm_72b | CP PRM + TELE16 (anchored) | REF PRM native (p>.5) | avg F1 | -8.33 [-9.76, -6.95] |
| qwen_prm_72b | CP PRM + TELE16 (anchored) | REF PRM native (p>.5) | macro SLA | -10.83 [-12.57, -8.89] |
| qwen_prm_72b | CP PRM + drop (anchored) | REF PRM native (p>.5) | avg F1 | +0.64 [-0.28, +1.60] |
| qwen_prm_72b | CP PRM + drop (anchored) | REF PRM native (p>.5) | macro SLA | -0.87 [-2.18, +0.50] |
| qwen_prm_72b | FREE CP TELE16 standalone (EM) | Mind the Gap Qwen3-8B (39.39, published) | macro SLA (forced) | +2.71 [+0.56, +4.93] |
| qwen_prm_72b | FREE CP TELE16 backbone (EM, pi=1/2) | Mind the Gap Qwen3-8B (39.39, published) | macro SLA (forced) | +4.84 [+2.42, +7.09] |
| pure_prm | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | avg F1 | +12.46 [+9.93, +15.04] |
| pure_prm | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | macro SLA | +13.62 [+11.08, +16.12] |
| pure_prm | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | avg F1 | +2.50 [+0.97, +3.83] |
| pure_prm | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | macro SLA | +3.37 [+1.79, +4.92] |
| pure_prm | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | macro SLA (forced) | +11.34 [+8.43, +14.28] |
| pure_prm | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | macro SLA (forced) | +3.27 [+1.41, +5.19] |
| pure_prm | CP PRM + TELE16 (anchored) | REF PRM native (p>.5) | avg F1 | +2.96 [+1.23, +4.66] |
| pure_prm | CP PRM + TELE16 (anchored) | REF PRM native (p>.5) | macro SLA | +3.01 [+1.17, +4.80] |
| pure_prm | CP PRM + drop (anchored) | REF PRM native (p>.5) | avg F1 | +1.86 [+0.52, +3.23] |
| pure_prm | CP PRM + drop (anchored) | REF PRM native (p>.5) | macro SLA | +3.32 [+1.87, +4.79] |
| pure_prm | FREE CP TELE16 standalone (EM) | Mind the Gap Qwen3-8B (39.39, published) | macro SLA (forced) | -4.02 [-6.18, -1.64] |
| pure_prm | FREE CP TELE16 backbone (EM, pi=1/2) | Mind the Gap Qwen3-8B (39.39, published) | macro SLA (forced) | -12.05 [-14.26, -9.90] |
| rlhflow_prm | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | avg F1 | -0.35 [-2.01, +1.33] |
| rlhflow_prm | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | macro SLA | +0.03 [-1.41, +1.50] |
| rlhflow_prm | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | avg F1 | +0.32 [-0.66, +1.28] |
| rlhflow_prm | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | macro SLA | +0.46 [-0.40, +1.34] |
| rlhflow_prm | CP PRM + C4U rank (anchored) | REF C4U NC (PRM gate+count) | macro SLA (forced) | -0.95 [-2.69, +0.77] |
| rlhflow_prm | CP PRM + C4U rank (anchored) | REF PRM native (p>.5) | macro SLA (forced) | -0.63 [-2.06, +0.85] |
| rlhflow_prm | CP PRM + TELE16 (anchored) | REF PRM native (p>.5) | avg F1 | -2.98 [-5.11, -0.87] |
| rlhflow_prm | CP PRM + TELE16 (anchored) | REF PRM native (p>.5) | macro SLA | -2.53 [-4.35, -0.78] |
| rlhflow_prm | CP PRM + drop (anchored) | REF PRM native (p>.5) | avg F1 | +0.37 [-0.69, +1.54] |
| rlhflow_prm | CP PRM + drop (anchored) | REF PRM native (p>.5) | macro SLA | +0.43 [-0.50, +1.45] |
| rlhflow_prm | FREE CP TELE16 standalone (EM) | Mind the Gap Qwen3-8B (39.39, published) | macro SLA (forced) | -2.35 [-4.66, -0.05] |
| rlhflow_prm | FREE CP TELE16 backbone (EM, pi=1/2) | Mind the Gap Qwen3-8B (39.39, published) | macro SLA (forced) | -18.33 [-20.46, -16.31] |
