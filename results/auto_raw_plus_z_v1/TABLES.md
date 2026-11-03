### Table 1. Metrics per arm (PRMBench, N = 6,969 answers; within-AUC over 6,030 answers with an error)

| PRM | arm | within-AUC | pooled step AUC | answer AUC (max step) | PRMScore |
|---|---|---|---|---|---|
| Qwen-7B | PRM_native | 0.8012 | 0.8029 | 0.8215 | 65.49 |
| Qwen-7B | PRM_Q80_PROB | 0.8012 | 0.8029 | 0.8215 | 67.32 |
| Qwen-7B | Z17 | 0.8243 | 0.7837 | 0.5796 | 67.49 |
| Qwen-7B | CAT34 | 0.8366 | 0.8137 | 0.7691 | 68.26 |
| Qwen-72B | PRM_native | 0.7948 | 0.7913 | 0.8411 | 68.18 |
| Qwen-72B | PRM_Q80_PROB | 0.7948 | 0.7913 | 0.8411 | 69.04 |
| Qwen-72B | Z17 | 0.8278 | 0.7833 | 0.5682 | 67.83 |
| Qwen-72B | CAT34 | 0.8391 | 0.8145 | 0.7438 | 68.69 |
| PURE | PRM_native | 0.7628 | 0.7836 | 0.6944 | 65.31 |
| PURE | PRM_Q80_PROB | 0.7628 | 0.7836 | 0.6944 | 64.72 |
| PURE | Z17 | 0.7990 | 0.7545 | 0.5877 | 66.67 |
| PURE | CAT34 | 0.8002 | 0.7767 | 0.6804 | 66.50 |
| RLHFlow | PRM_native | 0.6624 | 0.6333 | 0.6373 | 54.43 |
| RLHFlow | PRM_Q80_PROB | 0.6624 | 0.6333 | 0.6373 | 56.54 |
| RLHFlow | Z17 | 0.7647 | 0.7160 | 0.6041 | 63.95 |
| RLHFlow | CAT34 | 0.7780 | 0.7430 | 0.7083 | 64.58 |

### Table 2. Difference vs Z17 (95% CI; answer-level bootstrap, 1,000 draws; * = CI excludes 0)

| PRM | contrast | within-AUC diff | PRMScore diff | PRMScore diff, source-group bootstrap |
|---|---|---|---|---|
| Qwen-7B | CAT34 - Z17 | +0.0123 [+0.0107, +0.0138]* | +0.77 [+0.58, +0.95]* | +0.77 [+0.57, +0.96]* |
| Qwen-7B | PRM_Q80_PROB - Z17 | -0.0231 [-0.0283, -0.0178]* | -0.17 [-0.64, +0.32] | -0.17 [-0.71, +0.36] |
| Qwen-72B | CAT34 - Z17 | +0.0113 [+0.0098, +0.0130]* | +0.86 [+0.68, +1.06]* | +0.86 [+0.64, +1.06]* |
| Qwen-72B | PRM_Q80_PROB - Z17 | -0.0330 [-0.0391, -0.0273]* | +1.21 [+0.71, +1.75]* | +1.21 [+0.59, +1.81]* |
| PURE | CAT34 - Z17 | +0.0013 [-0.0001, +0.0025] | -0.18 [-0.35, -0.00]* | -0.18 [-0.34, -0.00]* |
| PURE | PRM_Q80_PROB - Z17 | -0.0361 [-0.0423, -0.0294]* | -1.96 [-2.47, -1.42]* | -1.96 [-2.54, -1.31]* |
| RLHFlow | CAT34 - Z17 | +0.0132 [+0.0116, +0.0149]* | +0.63 [+0.46, +0.82]* | +0.63 [+0.43, +0.85]* |
| RLHFlow | PRM_Q80_PROB - Z17 | -0.1024 [-0.1093, -0.0956]* | -7.41 [-7.90, -6.90]* | -7.41 [-8.06, -6.70]* |

Pre-declared rule (CAT34 beats Z17 if PRMScore diff > 0 with CI excluding 0 in >= 3/4 PRMs): **3/4 -> MET**.

### Table 3. Grouping and spectrum

| PRM | Z17 K | CAT34 K | raw_i in same CAT34 group as z_i | lead eigenvalue share, Z17 inputs | lead eigenvalue share, CAT34 inputs |
|---|---|---|---|---|---|
| Qwen-7B | 4 | 8 | 15/17 | 0.353 | 0.328 |
| Qwen-72B | 5 | 7 | 17/17 | 0.346 | 0.323 |
| PURE | 4 | 7 | 16/17 | 0.340 | 0.320 |
| RLHFlow | 6 | 7 | 17/17 | 0.314 | 0.299 |

### Table 4. |corr(raw_i, z_i)| by step class (raw = global rank-normal; mean over the 4 PRMs)

| channel | clean steps | error steps | pairs grouped together (of 4 PRMs) |
|---|---|---|---|
| PRM | 0.75 | 0.73 | 4 |
| q15_H1 | 0.80 | 0.77 | 4 |
| q15_VE1 | 0.84 | 0.82 | 4 |
| chosen_surprisal | 0.79 | 0.77 | 2 |
| logprob_margin | 0.79 | 0.76 | 4 |
| true_tail50 | 0.72 | 0.67 | 3 |
| energy_level | 0.81 | 0.78 | 4 |
| energy_innovation | 0.86 | 0.85 | 4 |
| top15_turnover | 0.80 | 0.77 | 4 |
| top50_js | 0.65 | 0.59 | 4 |
| dominant_freq16 | 0.83 | 0.78 | 4 |
| bocpd_p0 | 0.82 | 0.82 | 4 |
| realized_z | 0.98 | 0.98 | 4 |
| realized_drv | 0.87 | 0.87 | 4 |
| digit_alternative | 0.75 | 0.75 | 4 |
| digit_spread | 0.77 | 0.78 | 4 |
| digit_alternative_innovation | 0.76 | 0.76 | 4 |

### Table 5. CAT34 groups per PRM

- **Qwen-7B**: G0 = z:PRM, raw:PRM; G1 = z:q15_H1, z:q15_VE1, z:logprob_margin, z:energy_level, raw:q15_H1, raw:q15_VE1, raw:chosen_surprisal, raw:logprob_margin, raw:true_tail50, raw:energy_level; G2 = z:digit_alternative, z:digit_spread, z:digit_alternative_innovation, raw:digit_alternative, raw:digit_spread, raw:digit_alternative_innovation; G3 = z:bocpd_p0, raw:bocpd_p0; G4 = z:dominant_freq16, raw:dominant_freq16; G5 = z:chosen_surprisal, z:realized_z, z:realized_drv, raw:realized_z, raw:realized_drv; G6 = z:top15_turnover, raw:top15_turnover; G7 = z:true_tail50, z:energy_innovation, z:top50_js, raw:energy_innovation, raw:top50_js
- **Qwen-72B**: G0 = z:energy_innovation, z:top15_turnover, z:top50_js, raw:energy_innovation, raw:top15_turnover, raw:top50_js; G1 = z:q15_H1, z:q15_VE1, z:chosen_surprisal, z:logprob_margin, z:true_tail50, z:energy_level, raw:q15_H1, raw:q15_VE1, raw:chosen_surprisal, raw:logprob_margin, raw:true_tail50, raw:energy_level; G2 = z:bocpd_p0, raw:bocpd_p0; G3 = z:digit_alternative, z:digit_spread, z:digit_alternative_innovation, raw:digit_alternative, raw:digit_spread, raw:digit_alternative_innovation; G4 = z:dominant_freq16, raw:dominant_freq16; G5 = z:realized_z, z:realized_drv, raw:realized_z, raw:realized_drv; G6 = z:PRM, raw:PRM
- **PURE**: G0 = z:PRM, z:true_tail50, z:energy_level, raw:PRM, raw:true_tail50, raw:energy_level; G1 = z:dominant_freq16, raw:dominant_freq16; G2 = z:q15_H1, z:q15_VE1, z:logprob_margin, raw:q15_H1, raw:q15_VE1, raw:chosen_surprisal, raw:logprob_margin; G3 = z:chosen_surprisal, z:realized_z, z:realized_drv, raw:realized_z, raw:realized_drv; G4 = z:bocpd_p0, raw:bocpd_p0; G5 = z:digit_alternative, z:digit_spread, z:digit_alternative_innovation, raw:digit_alternative, raw:digit_spread, raw:digit_alternative_innovation; G6 = z:energy_innovation, z:top15_turnover, z:top50_js, raw:energy_innovation, raw:top15_turnover, raw:top50_js
- **RLHFlow**: G0 = z:digit_alternative, z:digit_spread, z:digit_alternative_innovation, raw:digit_alternative, raw:digit_spread, raw:digit_alternative_innovation; G1 = z:q15_H1, z:q15_VE1, z:logprob_margin, z:true_tail50, z:energy_level, raw:q15_H1, raw:q15_VE1, raw:logprob_margin, raw:true_tail50, raw:energy_level; G2 = z:energy_innovation, z:top15_turnover, z:top50_js, raw:energy_innovation, raw:top15_turnover, raw:top50_js; G3 = z:PRM, raw:PRM; G4 = z:dominant_freq16, raw:dominant_freq16; G5 = z:chosen_surprisal, z:realized_z, z:realized_drv, raw:chosen_surprisal, raw:realized_z, raw:realized_drv; G6 = z:bocpd_p0, raw:bocpd_p0
