# ProcessBench SLA: standalone telemetry (GAP rule) vs Mind the Gap

Metric: Mind the Gap Table 3 SLA. Exact match of the first-error step, on the 2,221 erroneous answers only. Macro = mean of 4 subsets.
Rule: GAP first flag (label-free). Standalone = base-model telemetry only, no PRM signal. Math-Shepherd is excluded.

## Main table (Macro SLA / Overall SLA)

| Base model (PRM pair) | Standalone TELE_N16 | Standalone MEAN_N16 | Backbone TELE_N16 (Step 483) | MtG Qwen3-8B |
|---|---|---|---|---|
| Qwen2.5-Math-7B-Instruct (Qwen-PRM-7B) | 38.47 / 35.70 | 38.56 / 36.06 | 47.23 / 44.35 | 39.39 / 38.11 |
| Qwen2.5-Math-72B-Instruct (Qwen-PRM-72B) | 38.10 / 36.43 | 39.29 / 37.64 | 50.64 / 49.08 | 39.39 / 38.11 |
| Qwen2.5-Math-7B (PURE) | 31.29 / 30.21 | 34.02 / 32.69 | 33.21 / 31.74 | 39.39 / 38.11 |
| Llama-3.1-8B-Instruct (RLHFlow) | 34.29 / 32.06 | 34.30 / 32.28 | 32.52 / 30.17 | 39.39 / 38.11 |

MtG Qwen3-4B: 39.14 / 38.43. Position controls: always first step 13.50 / 12.25; always last step 4.84 / 4.10.

## Bootstrap (2,000 answer resamples, stratified by subset), standalone GAP Macro SLA

| Pair | Arm | Macro | 95% CI | vs MtG 8B | CI includes 39.39 |
|---|---|---|---|---|---|
| Qwen-7B | TELE_N16 | 38.47 | 36.09-40.76 | -0.92 | yes |
| Qwen-7B | MEAN_N16 | 38.56 | 36.28-40.79 | -0.83 | yes |
| Qwen-72B | TELE_N16 | 38.10 | 35.76-40.46 | -1.29 | yes |
| Qwen-72B | MEAN_N16 | 39.29 | 36.97-41.59 | -0.10 | yes |
| PURE | TELE_N16 | 31.29 | 29.10-33.61 | -8.10 | no (below) |
| PURE | MEAN_N16 | 34.02 | 31.80-36.19 | -5.37 | no (below) |
| RLHFlow | TELE_N16 | 34.29 | 32.11-36.62 | -5.10 | no (below) |
| RLHFlow | MEAN_N16 | 34.30 | 31.96-36.63 | -5.09 | no (below) |

## Rule and bank ablation, standalone Macro SLA

| Pair | TELE_N16 GAP | TELE_N16 Argmax | MEAN_N16 Argmax | TELE_N13 GAP |
|---|---|---|---|---|
| Qwen-7B | 38.47 | 39.48 | 40.81 | 31.72 |
| Qwen-72B | 38.10 | 39.11 | 39.97 | 31.10 |
| PURE | 31.29 | 35.84 | 37.90 | 26.50 |
| RLHFlow | 34.29 | 37.06 | 37.51 | 29.67 |

## Findings

- Standalone ties Mind the Gap on the two Qwen-Math-Instruct models (-0.1 to -1.3, CIs include 39.39). It is not a win.
- Standalone is 5-8 points below Mind the Gap on Qwen2.5-Math-7B base (PURE) and Llama-3.1-8B (RLHFlow).
- The Step 483 gain (+7.8 / +11.3) comes from the PRM backbone hidden state: standalone loses 8.8 (7B) and 12.5 (72B) points against backbone.
- L-SML (TELE) does not beat the plain average (MEAN) here. MEAN >= TELE in 4/4.
- In standalone, Argmax beats GAP first-flag in 4/4 (+1.0 to +4.6). In backbone the order was the opposite (Step 483). Argmax MEAN_N16 on Qwen-7B (40.81) is the only standalone number above MtG; it is a post-hoc choice.
- Digit channels matter: N13 (no digits) is 4.6 to 7.0 points below N16.
- Models are not matched: MtG uses Qwen3-4B/8B; our base models are Qwen2.5-Math and Llama-3.1. This is an access-level comparison (telemetry only vs entropy only), not a same-model comparison.
