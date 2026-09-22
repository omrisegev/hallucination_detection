# Renyi alpha sweep — selection stability (development evidence)

## renyi_entropy

| alpha | PB % | within | pooled | PRMScore |
|---|---:|---:|---:|---:|
| 0.0 | 35.53 | 0.7440 | 0.7162 | 0.6334 |
| 0.001 | 35.53 | 0.7440 | 0.7162 | 0.6334 |
| 0.01 | 35.50 | 0.7438 | 0.7161 | 0.6334 |
| 0.02 | 35.48 | 0.7437 | 0.7161 | 0.6334 |
| 0.05 | 35.53 | 0.7433 | 0.7158 | 0.6334 |
| 0.1 | 35.37 | 0.7425 | 0.7152 | 0.6331 |
| 0.15 | 35.44 | 0.7421 | 0.7146 | 0.6329 |
| 0.2 | 35.49 | 0.7421 | 0.7139 | 0.6325 |
| 0.25 | 35.52 | 0.7414 | 0.7131 | 0.6322 |
| 0.3 | 35.49 | 0.7408 | 0.7122 | 0.6318 |
| 0.4 | 35.49 | 0.7394 | 0.7105 | 0.6313 |
| 0.5 | 35.55 | 0.7371 | 0.7088 | 0.6296 |
| 0.75 | 35.42 | 0.7330 | 0.7053 | 0.6268 |
| 1.0 | 35.44 | 0.7301 | 0.7027 | 0.6254 |
| 1.5 | 35.46 | 0.7265 | 0.6992 | 0.6235 |
| 2.0 | 35.59 | 0.7242 | 0.6972 | 0.6213 |
| 3.0 | 35.78 | 0.7219 | 0.6951 | 0.6195 |
| 4.0 | 35.80 | 0.7207 | 0.6942 | 0.6177 |
| 8.0 | 35.88 | 0.7192 | 0.6930 | 0.6160 |
| inf | 35.77 | 0.7178 | 0.6924 | 0.6158 |

argmax: pb: view__a8, within: view__H0lim, pooled: view__H0lim, prmscore: view__a0.02
fold-wise within argmax: fold 0: view__a0.001 (0.7402 vs global-best 0.7402), fold 1: view__H0lim (0.7355 vs global-best 0.7355), fold 2: view__H0lim (0.7432 vs global-best 0.7432), fold 3: view__H0lim (0.7492 vs global-best 0.7492), fold 4: view__a0.001 (0.7516 vs global-best 0.7515)
cell-wise PB argmax: pb_gsm8k_q4: view__a0.05 (44.91 vs 43.01), pb_gsm8k_q8: view__a0.15 (41.01 vs 40.24), pb_math_q4: view__Hinf (35.73 vs 35.57), pb_math_q8: view__a8 (34.86 vs 34.86), pb_olympiadbench_q4: view__a8 (31.94 vs 31.94), pb_olympiadbench_q8: view__Hinf (32.58 vs 32.45), pb_omnimath_q4: view__a8 (35.78 vs 35.78), pb_omnimath_q8: view__a0.15 (34.09 vs 33.18)
cross-fitted within (choose on 4 folds, evaluate on 1): 0.7439 (oracle per fold 0.7439); choices: view__H0lim, view__H0lim, view__a0.001, view__a0.001, view__H0lim

## escort_varentropy

| alpha | PB % | within | pooled | PRMScore |
|---|---:|---:|---:|---:|
| 0.0 | 35.57 | 0.7534 | 0.7231 | 0.6355 |
| 0.1 | 35.77 | 0.7482 | 0.7201 | 0.6340 |
| 0.25 | 34.89 | 0.7352 | 0.7089 | 0.6316 |
| 0.5 | 36.20 | 0.6788 | 0.6341 | 0.5731 |
| 0.75 | 36.76 | 0.7323 | 0.7043 | 0.6187 |
| 1.0 | 35.96 | 0.7378 | 0.7101 | 0.6258 |
| 1.5 | 35.46 | 0.7333 | 0.7049 | 0.6248 |
| 2.0 | 35.01 | 0.7259 | 0.6986 | 0.6194 |
| 3.0 | 34.98 | 0.7136 | 0.6876 | 0.6113 |
| 4.0 | 34.06 | 0.7061 | 0.6790 | 0.6060 |
| 8.0 | 34.17 | 0.6824 | 0.6455 | 0.5909 |

argmax: pb: view__ve0.75, within: view__ve0, pooled: view__ve0, prmscore: view__ve0
fold-wise within argmax: fold 0: view__ve0 (0.7489 vs global-best 0.7489), fold 1: view__ve0 (0.7431 vs global-best 0.7431), fold 2: view__ve0 (0.7567 vs global-best 0.7567), fold 3: view__ve0 (0.7559 vs global-best 0.7559), fold 4: view__ve0 (0.7623 vs global-best 0.7623)
cell-wise PB argmax: pb_gsm8k_q4: view__ve1 (45.28 vs 44.91), pb_gsm8k_q8: view__ve0.1 (41.51 vs 40.50), pb_math_q4: view__ve4 (36.51 vs 36.20), pb_math_q8: view__ve0.5 (36.26 vs 35.88), pb_olympiadbench_q4: view__ve0.75 (35.60 vs 35.60), pb_olympiadbench_q8: view__ve0.75 (35.06 vs 35.06), pb_omnimath_q4: view__ve0.25 (35.56 vs 32.78), pb_omnimath_q8: view__ve0.1 (34.29 vs 33.18)
cross-fitted within (choose on 4 folds, evaluate on 1): 0.7534 (oracle per fold 0.7534); choices: view__ve0, view__ve0, view__ve0, view__ve0, view__ve0
