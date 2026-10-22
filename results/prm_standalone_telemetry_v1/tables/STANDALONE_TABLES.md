## Qwen2.5-Math-PRM-7B (standalone model: Qwen2.5-Math-7B-Instruct; PRM scale: PRM_Q80_PROB)

| arm | uses PRM | telemetry | within-AUC | Q80 | GAP | flag 2 | true count |
|---|---|---|---|---|---|---|---|
| PRM, native threshold | yes | - | 0.801 | 65.49 |  |  |  |
| PRM alone, same rule | yes | - | 0.801 | 67.32 | 68.01 | 68.16 | 72.50 |
| Backbone telemetry, plain mean | no | backbone | 0.805 | 66.44 | 68.60 | 69.10 | 73.18 |
| Backbone telemetry, L-SML | no | backbone | 0.818 | 67.11 | 69.26 | 69.79 | 74.14 |
| Standalone telemetry, plain mean | no | standalone | 0.782 | 65.42 | 67.14 | 68.12 | 71.86 |
| Standalone telemetry, L-SML | no | standalone | 0.793 | 66.01 | 68.26 | 68.66 | 72.84 |
| PRM + backbone telemetry, plain mean | yes | backbone | 0.815 | 67.00 | 69.32 | 69.72 | 73.93 |
| PRM + backbone telemetry, L-SML | yes | backbone | 0.838 | 68.25 | 70.40 | 70.80 | 75.40 |

| contrast (GAP, 95% CI) | PRMScore | within-AUC |
|---|---|---|
| Standalone vs backbone, L-SML | -0.99 [-1.41, -0.58] | -0.024 [-0.028, -0.021] |
| Standalone vs backbone, plain mean | -1.46 [-1.88, -1.06] | -0.023 [-0.026, -0.020] |
| Standalone: L-SML vs plain mean | +1.13 [+0.75, +1.52] | +0.012 [+0.009, +0.014] |
| Standalone L-SML vs the PRM alone | +0.26 [-0.36, +0.85] | -0.008 [-0.014, -0.001] |
| Backbone L-SML vs the PRM alone | +1.25 [+0.62, +1.86] | +0.016 [+0.011, +0.022] |

## Qwen2.5-Math-PRM-72B (standalone model: Qwen2.5-Math-72B-Instruct; PRM scale: PRM_Q80_PROB)

| arm | uses PRM | telemetry | within-AUC | Q80 | GAP | flag 2 | true count |
|---|---|---|---|---|---|---|---|
| PRM, native threshold | yes | - | 0.795 | 68.18 |  |  |  |
| PRM alone, same rule | yes | - | 0.795 | 69.04 | 69.49 | 69.38 | 73.54 |
| Backbone telemetry, plain mean | no | backbone | 0.805 | 66.17 | 68.11 | 68.96 | 72.90 |
| Backbone telemetry, L-SML | no | backbone | 0.823 | 67.53 | 69.68 | 70.29 | 74.39 |
| Standalone telemetry, plain mean | no | standalone | 0.780 | 65.20 | 67.39 | 68.04 | 71.86 |
| Standalone telemetry, L-SML | no | standalone | 0.781 | 65.63 | 67.80 | 68.17 | 71.97 |
| PRM + backbone telemetry, plain mean | yes | backbone | 0.819 | 67.11 | 69.01 | 70.09 | 74.22 |
| PRM + backbone telemetry, L-SML | yes | backbone | 0.843 | 68.90 | 71.09 | 71.90 | 76.37 |

| contrast (GAP, 95% CI) | PRMScore | within-AUC |
|---|---|---|
| Standalone vs backbone, L-SML | -1.88 [-2.37, -1.42] | -0.042 [-0.046, -0.038] |
| Standalone vs backbone, plain mean | -0.72 [-1.16, -0.29] | -0.025 [-0.029, -0.022] |
| Standalone: L-SML vs plain mean | +0.42 [+0.07, +0.75] | +0.002 [-0.001, +0.004] |
| Standalone L-SML vs the PRM alone | -1.69 [-2.36, -1.04] | -0.014 [-0.021, -0.006] |
| Backbone L-SML vs the PRM alone | +0.19 [-0.43, +0.81] | +0.028 [+0.022, +0.035] |

## PURE-PRM-7B (standalone model: Qwen2.5-Math-7B; PRM scale: PRM_Q80_PROB)

| arm | uses PRM | telemetry | within-AUC | Q80 | GAP | flag 2 | true count |
|---|---|---|---|---|---|---|---|
| PRM, native threshold | yes | - | 0.763 | 65.31 |  |  |  |
| PRM alone, same rule | yes | - | 0.763 | 64.72 | 66.24 | 64.56 | 68.79 |
| Backbone telemetry, plain mean | no | backbone | 0.785 | 65.83 | 67.98 | 68.26 | 71.80 |
| Backbone telemetry, L-SML | no | backbone | 0.792 | 66.33 | 68.71 | 68.64 | 72.49 |
| Standalone telemetry, plain mean | no | standalone | 0.769 | 64.93 | 67.06 | 67.70 | 71.18 |
| Standalone telemetry, L-SML | no | standalone | 0.770 | 65.27 | 67.63 | 67.90 | 71.27 |
| PRM + backbone telemetry, plain mean | yes | backbone | 0.795 | 66.39 | 68.52 | 68.82 | 72.43 |
| PRM + backbone telemetry, L-SML | yes | backbone | 0.816 | 67.29 | 69.51 | 69.60 | 73.73 |

| contrast (GAP, 95% CI) | PRMScore | within-AUC |
|---|---|---|
| Standalone vs backbone, L-SML | -1.08 [-1.39, -0.75] | -0.021 [-0.024, -0.019] |
| Standalone vs backbone, plain mean | -0.93 [-1.26, -0.59] | -0.016 [-0.019, -0.014] |
| Standalone: L-SML vs plain mean | +0.58 [+0.24, +0.90] | +0.001 [-0.001, +0.004] |
| Standalone L-SML vs the PRM alone | +1.40 [+0.78, +1.99] | +0.008 [+0.000, +0.015] |
| Backbone L-SML vs the PRM alone | +2.47 [+1.88, +3.05] | +0.029 [+0.022, +0.036] |

## RLHFlow-PRM-Mistral-8B (standalone model: Llama-3.1-8B-Instruct; PRM scale: PRM_Q80)

| arm | uses PRM | telemetry | within-AUC | Q80 | GAP | flag 2 | true count |
|---|---|---|---|---|---|---|---|
| PRM, native threshold | yes | - | 0.662 | 54.43 |  |  |  |
| PRM alone, same rule | yes | - | 0.662 | 56.46 | 59.39 | 58.92 | 62.26 |
| Backbone telemetry, plain mean | no | backbone | 0.756 | 63.70 | 65.57 | 66.74 | 70.00 |
| Backbone telemetry, L-SML | no | backbone | 0.767 | 64.28 | 66.17 | 67.40 | 70.75 |
| Standalone telemetry, plain mean | no | standalone | 0.783 | 65.42 | 67.45 | 68.20 | 71.74 |
| Standalone telemetry, L-SML | no | standalone | 0.788 | 66.10 | 68.20 | 68.48 | 72.29 |
| PRM + backbone telemetry, plain mean | yes | backbone | 0.762 | 63.86 | 65.82 | 66.95 | 70.14 |
| PRM + backbone telemetry, L-SML | yes | backbone | 0.775 | 63.92 | 66.25 | 67.43 | 70.70 |

| contrast (GAP, 95% CI) | PRMScore | within-AUC |
|---|---|---|
| Standalone vs backbone, L-SML | +2.03 [+1.61, +2.45] | +0.021 [+0.017, +0.025] |
| Standalone vs backbone, plain mean | +1.89 [+1.46, +2.30] | +0.027 [+0.023, +0.030] |
| Standalone: L-SML vs plain mean | +0.75 [+0.40, +1.11] | +0.005 [+0.002, +0.008] |
| Standalone L-SML vs the PRM alone | +8.81 [+8.10, +9.55] | +0.125 [+0.116, +0.135] |
| Backbone L-SML vs the PRM alone | +6.78 [+6.07, +7.51] | +0.104 [+0.095, +0.114] |

## Math-Shepherd-Mistral-7B (standalone model: Mistral-7B-v0.1; PRM scale: PRM_Q80)

| arm | uses PRM | telemetry | within-AUC | Q80 | GAP | flag 2 | true count |
|---|---|---|---|---|---|---|---|
| PRM, native threshold | yes | - | 0.684 | 47.02 |  |  |  |
| PRM alone, same rule | yes | - | 0.684 | 57.71 | 60.21 | 59.32 | 63.82 |
| Backbone telemetry, plain mean | no | backbone | 0.740 | 62.35 | 64.22 | 65.12 | 68.48 |
| Backbone telemetry, L-SML | no | backbone | 0.748 | 62.91 | 64.97 | 65.71 | 69.14 |
| Standalone telemetry, plain mean | no | standalone | 0.754 | 64.00 | 66.13 | 66.91 | 70.24 |
| Standalone telemetry, L-SML | no | standalone | 0.760 | 64.85 | 66.94 | 67.37 | 70.78 |
| PRM + backbone telemetry, plain mean | yes | backbone | 0.754 | 63.05 | 65.08 | 65.89 | 69.35 |
| PRM + backbone telemetry, L-SML | yes | backbone | 0.778 | 64.37 | 66.39 | 66.76 | 70.70 |

| contrast (GAP, 95% CI) | PRMScore | within-AUC |
|---|---|---|
| Standalone vs backbone, L-SML | +1.97 [+1.51, +2.42] | +0.012 [+0.008, +0.016] |
| Standalone vs backbone, plain mean | +1.91 [+1.51, +2.32] | +0.014 [+0.010, +0.017] |
| Standalone: L-SML vs plain mean | +0.81 [+0.45, +1.17] | +0.006 [+0.004, +0.008] |
| Standalone L-SML vs the PRM alone | +6.73 [+6.00, +7.53] | +0.076 [+0.066, +0.085] |
| Backbone L-SML vs the PRM alone | +4.76 [+4.00, +5.56] | +0.064 [+0.054, +0.074] |
