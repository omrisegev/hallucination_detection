# Cumulative-vote fusion v1 — long-chain diagnostics

## 1. Label-free fitted parameters (pooled scope, mean over 5 folds)

| localizer | SML w | L-SML w | psi = P(votes <=n | truly <=n) (not-late) | eta = P(votes >n | truly >n) (not-early) |
|---|---:|---:|---:|---:|
| family6 | +0.603 | +0.483 | 0.968 | 0.945 |
| unified28 | +0.152 | +0.160 | 0.755 | 0.289 |
| gl_liu | +0.502 | +0.483 | 0.730 | 0.791 |
| max_ent | +0.544 | +0.693 | 0.718 | 0.845 |
| mind_gap | -0.253 | -0.160 | 0.261 | 0.601 |

L-SML groups per fold (index = localizer order above): [[0, 1, 0, 2, 1], [0, 1, 0, 2, 1], [0, 1, 0, 2, 1], [0, 1, 0, 2, 1], [0, 1, 0, 2, 1]]

## 2. Signed offset (prediction − first error) per subset, erroneous answers

| subset | n | label median / p90 | localizer | early | exact | late | mean offset |
|---|---:|---|---|---:|---:|---:|---:|
| gsm8k | 207 | 2 / 4 | single:family6 | 0.29 | 0.44 | 0.27 | -0.07 |
| gsm8k | 207 | 2 / 4 | single:unified28 | 0.29 | 0.40 | 0.32 | -0.09 |
| gsm8k | 207 | 2 / 4 | single:gl_liu | 0.30 | 0.37 | 0.32 | -0.03 |
| gsm8k | 207 | 2 / 4 | single:max_ent | 0.24 | 0.43 | 0.33 | +0.19 |
| gsm8k | 207 | 2 / 4 | single:mind_gap | 0.25 | 0.29 | 0.46 | +0.53 |
| gsm8k | 207 | 2 / 4 | median | 0.27 | 0.43 | 0.30 | -0.01 |
| gsm8k | 207 | 2 / 4 | ds_mode | 0.28 | 0.45 | 0.27 | -0.06 |
| math | 594 | 2 / 5 | single:family6 | 0.26 | 0.31 | 0.43 | +0.57 |
| math | 594 | 2 / 5 | single:unified28 | 0.43 | 0.26 | 0.31 | -0.34 |
| math | 594 | 2 / 5 | single:gl_liu | 0.25 | 0.29 | 0.46 | +0.59 |
| math | 594 | 2 / 5 | single:max_ent | 0.25 | 0.30 | 0.44 | +0.69 |
| math | 594 | 2 / 5 | single:mind_gap | 0.23 | 0.26 | 0.51 | +0.96 |
| math | 594 | 2 / 5 | median | 0.25 | 0.32 | 0.43 | +0.55 |
| math | 594 | 2 / 5 | ds_mode | 0.26 | 0.31 | 0.43 | +0.56 |
| olympiadbench | 661 | 3 / 7 | single:family6 | 0.31 | 0.29 | 0.39 | +0.23 |
| olympiadbench | 661 | 3 / 7 | single:unified28 | 0.50 | 0.19 | 0.31 | -0.88 |
| olympiadbench | 661 | 3 / 7 | single:gl_liu | 0.30 | 0.25 | 0.45 | +0.53 |
| olympiadbench | 661 | 3 / 7 | single:max_ent | 0.29 | 0.28 | 0.43 | +0.51 |
| olympiadbench | 661 | 3 / 7 | single:mind_gap | 0.29 | 0.22 | 0.49 | +0.88 |
| olympiadbench | 661 | 3 / 7 | median | 0.30 | 0.29 | 0.42 | +0.33 |
| olympiadbench | 661 | 3 / 7 | ds_mode | 0.31 | 0.29 | 0.39 | +0.24 |
| omnimath | 759 | 2 / 6 | single:family6 | 0.25 | 0.26 | 0.49 | +0.72 |
| omnimath | 759 | 2 / 6 | single:unified28 | 0.41 | 0.17 | 0.41 | -0.09 |
| omnimath | 759 | 2 / 6 | single:gl_liu | 0.23 | 0.24 | 0.53 | +0.76 |
| omnimath | 759 | 2 / 6 | single:max_ent | 0.23 | 0.26 | 0.50 | +0.99 |
| omnimath | 759 | 2 / 6 | single:mind_gap | 0.23 | 0.21 | 0.55 | +1.27 |
| omnimath | 759 | 2 / 6 | median | 0.24 | 0.25 | 0.52 | +0.80 |
| omnimath | 759 | 2 / 6 | ds_mode | 0.25 | 0.26 | 0.49 | +0.74 |
| all | 2221 | 2 / 6 | single:family6 | 0.28 | 0.30 | 0.42 | +0.46 |
| all | 2221 | 2 / 6 | single:unified28 | 0.43 | 0.22 | 0.35 | -0.39 |
| all | 2221 | 2 / 6 | single:gl_liu | 0.26 | 0.27 | 0.47 | +0.57 |
| all | 2221 | 2 / 6 | single:max_ent | 0.26 | 0.29 | 0.45 | +0.69 |
| all | 2221 | 2 / 6 | single:mind_gap | 0.25 | 0.23 | 0.52 | +1.00 |
| all | 2221 | 2 / 6 | median | 0.26 | 0.29 | 0.44 | +0.52 |
| all | 2221 | 2 / 6 | ds_mode | 0.27 | 0.30 | 0.43 | +0.47 |

## 3. Is the lateness more than argmax noise? Mean offset by true position vs a uniform guess over [0, max locator]

Long subsets (OlympiadBench + Omni-MATH), erroneous answers.

| first error at | n | uniform guess | family6 | max_ent | gl_liu | mind_gap | unified28 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 120 | +2.37 | +2.45 | +2.96 | +2.81 | +3.53 | +1.80 |
| 1 | 319 | +1.19 | +1.67 | +1.79 | +1.78 | +2.30 | +0.84 |
| 2 | 321 | +0.40 | +1.14 | +1.35 | +1.29 | +1.69 | +0.45 |
| 3 | 197 | -0.38 | +0.42 | +0.63 | +0.50 | +1.00 | -0.23 |
| 4 | 147 | -1.11 | -0.32 | +0.04 | +0.07 | +0.08 | -0.65 |
| 5 | 105 | -1.94 | -0.79 | -0.66 | -0.61 | -0.22 | -1.98 |
| 6 | 60 | -2.42 | -0.73 | -0.07 | -0.32 | -0.73 | -2.72 |

## 4. Incumbent (family6) accuracy and lateness vs chain depth (max locator across the five)

| depth bucket | n | SLA % | late | early | label median |
|---|---:|---:|---:|---:|---:|
| 0–2 | 532 | 45.5 | 0.22 | 0.32 | 1 |
| 3–4 | 665 | 33.7 | 0.40 | 0.27 | 2 |
| 5–7 | 634 | 21.9 | 0.52 | 0.26 | 3 |
| 8–∞ | 390 | 16.4 | 0.60 | 0.24 | 4 |

## 5. By token length (prefix-lane `final_length`, available for 1105 of 2221 erroneous answers)

| subset | tercile (tokens) | n | family6 SLA | family6 late | mind_gap SLA | mind_gap late | median SLA |
|---|---|---:|---:|---:|---:|---:|---:|
| gsm8k | short (cuts 243/317) | 31 | 58.1 | 0.19 | 38.7 | 0.35 | 48.4 |
| gsm8k | mid (cuts 243/317) | 32 | 43.8 | 0.34 | 34.4 | 0.41 | 37.5 |
| gsm8k | long (cuts 243/317) | 31 | 45.2 | 0.42 | 25.8 | 0.58 | 38.7 |
| math | short (cuts 402/645) | 101 | 30.7 | 0.37 | 28.7 | 0.48 | 35.6 |
| math | mid (cuts 402/645) | 100 | 41.0 | 0.35 | 25.0 | 0.45 | 38.0 |
| math | long (cuts 402/645) | 102 | 12.7 | 0.59 | 16.7 | 0.60 | 15.7 |
| olympiadbench | short (cuts 616/841) | 109 | 32.1 | 0.34 | 24.8 | 0.40 | 30.3 |
| olympiadbench | mid (cuts 616/841) | 110 | 29.1 | 0.35 | 14.5 | 0.52 | 32.7 |
| olympiadbench | long (cuts 616/841) | 111 | 18.0 | 0.49 | 18.9 | 0.53 | 16.2 |
| omnimath | short (cuts 606/862) | 124 | 34.7 | 0.42 | 29.0 | 0.48 | 31.5 |
| omnimath | mid (cuts 606/862) | 128 | 29.7 | 0.47 | 21.9 | 0.54 | 27.3 |
| omnimath | long (cuts 606/862) | 126 | 14.3 | 0.61 | 23.8 | 0.49 | 16.7 |

## 6. Positional prior: where predictions land vs where errors are (all erroneous, %)

| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8+ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| label | 12.2 | 23.7 | 22.2 | 14.5 | 8.9 | 6.7 | 3.7 | 2.7 | 5.4 |
| family6 | 12.2 | 19.7 | 18.3 | 13.9 | 10.8 | 8.1 | 4.8 | 4.2 | 8.0 |
| unified28 | 34.3 | 17.4 | 12.7 | 9.3 | 7.1 | 5.7 | 4.3 | 2.4 | 6.9 |
| gl_liu | 5.9 | 21.0 | 20.7 | 15.3 | 11.5 | 9.4 | 5.9 | 3.2 | 7.2 |
| max_ent | 10.0 | 20.9 | 16.8 | 13.4 | 10.9 | 8.4 | 5.4 | 3.8 | 10.3 |
| mind_gap | 6.6 | 16.1 | 19.3 | 15.3 | 12.2 | 8.8 | 6.4 | 4.9 | 10.4 |

## 7. Lower-quantile readouts of the fused cumulative vote (DESCRIPTIVE sweep; q is label-selected here, not a candidate)

| readout | gsm8k | math | olympiadbench | omnimath | all | Δ vs family6 on long subsets (95% CI) | late / early on long |
|---|---:|---:|---:|---:|---:|---|---|
| min | 40.10 | 29.12 | 19.52 | 25.03 | 25.89 | -5.14 [-7.75, -2.54] | 0.18 / 0.60 |
| 2nd_min | 42.51 | 31.65 | 29.35 | 28.19 | 30.80 | +1.13 [-0.85, +3.10] | 0.34 / 0.37 |
| median | 42.51 | 31.99 | 28.59 | 24.51 | 29.40 | -1.20 [-2.82, +0.49] | 0.47 / 0.26 |
| sml_q0.2 | 42.03 | 31.65 | 28.29 | 28.59 | 30.57 | +0.85 [-0.99, +2.75] | 0.33 / 0.38 |
| sml_q0.3 | 43.96 | 32.32 | 28.90 | 27.14 | 30.62 | +0.35 [-1.27, +1.97] | 0.36 / 0.36 |
| sml_q0.4 | 44.44 | 32.15 | 28.74 | 25.43 | 29.99 | -0.63 [-1.62, +0.35] | 0.45 / 0.28 |
| sml_q0.5 | 43.48 | 31.82 | 29.20 | 24.90 | 29.76 | -0.70 [-1.97, +0.56] | 0.47 / 0.26 |
| family6 (reference) | 44.44 | 31.14 | 29.20 | 26.22 | 30.12 | — | 0.45 / 0.28 |
