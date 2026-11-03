### Table 1. Socratic-PRMBench, 2,995 answers (PRMScore x100; within-AUC over included non-empty steps)

| PRM | arm | within-AUC | pooled step AUC | PRMScore |
|---|---|---|---|---|
| Qwen-7B | PRM_native | 0.8236 | 0.8202 | 68.09 |
| Qwen-7B | PRMPROB_Q80 | 0.8236 | 0.8202 | 67.69 |
| Qwen-7B | POS_ONLY_Q80 | 0.7307 | 0.7151 | 58.01 |
| Qwen-7B | Z17_Q80 | 0.7220 | 0.7033 | 64.28 |
| Qwen-7B | CAT34_Q80 | 0.7541 | 0.7450 | 65.37 |
| Qwen-72B | PRM_native | 0.8141 | 0.8222 | 71.96 |
| Qwen-72B | PRMPROB_Q80 | 0.8141 | 0.8222 | 67.35 |
| Qwen-72B | POS_ONLY_Q80 | 0.7307 | 0.7151 | 58.01 |
| Qwen-72B | Z17_Q80 | 0.6612 | 0.6450 | 59.57 |
| Qwen-72B | CAT34_Q80 | 0.6442 | 0.6307 | 58.10 |
| PURE | PRM_native | 0.7956 | 0.7378 | 63.71 |
| PURE | PRMPROB_Q80 | 0.7956 | 0.7378 | 63.49 |
| PURE | POS_ONLY_Q80 | 0.7307 | 0.7151 | 58.01 |
| PURE | Z17_Q80 | 0.6600 | 0.6396 | 63.39 |
| PURE | CAT34_Q80 | 0.6802 | 0.6683 | 64.26 |
| RLHFlow | PRM_native | 0.5494 | 0.5792 | 48.12 |
| RLHFlow | PRMPROB_Q80 | 0.5494 | 0.5792 | 49.29 |
| RLHFlow | POS_ONLY_Q80 | 0.7307 | 0.7151 | 58.01 |
| RLHFlow | Z17_Q80 | 0.5953 | 0.5800 | 57.56 |
| RLHFlow | CAT34_Q80 | 0.6024 | 0.5857 | 57.07 |

### Table 2. Difference vs Z17 (95% CI, answer-level bootstrap, 1,000 draws; * = CI excludes 0)

| PRM | contrast | PRMScore diff | within-AUC diff |
|---|---|---|---|
| Qwen-7B | CAT34_Q80 - Z17 | +1.10 [+0.82, +1.37]* | +0.0321 [+0.0295, +0.0344]* |
| Qwen-7B | PRMPROB_Q80 - Z17 | +3.42 [+2.68, +4.15]* | +0.1016 [+0.0934, +0.1094]* |
| Qwen-7B | POS_ONLY_Q80 - Z17 | -6.27 [-7.28, -5.20]* | +0.0088 [-0.0057, +0.0215] |
| Qwen-72B | CAT34_Q80 - Z17 | -1.47 [-1.73, -1.18]* | -0.0170 [-0.0192, -0.0148]* |
| Qwen-72B | PRMPROB_Q80 - Z17 | +7.78 [+6.91, +8.57]* | +0.1529 [+0.1427, +0.1631]* |
| Qwen-72B | POS_ONLY_Q80 - Z17 | -1.56 [-2.58, -0.51]* | +0.0695 [+0.0545, +0.0838]* |
| PURE | CAT34_Q80 - Z17 | +0.86 [+0.61, +1.12]* | +0.0201 [+0.0182, +0.0223]* |
| PURE | PRMPROB_Q80 - Z17 | +0.10 [-0.89, +1.05] | +0.1356 [+0.1228, +0.1468]* |
| PURE | POS_ONLY_Q80 - Z17 | -5.39 [-6.43, -4.28]* | +0.0707 [+0.0546, +0.0857]* |
| RLHFlow | CAT34_Q80 - Z17 | -0.49 [-0.82, -0.17]* | +0.0071 [+0.0045, +0.0094]* |
| RLHFlow | PRMPROB_Q80 - Z17 | -8.27 [-9.14, -7.33]* | -0.0459 [-0.0591, -0.0338]* |
| RLHFlow | POS_ONLY_Q80 - Z17 | +0.45 [-0.64, +1.55] | +0.1355 [+0.1190, +0.1507]* |

Pre-declared rule (PRMScore diff > 0 with CI excluding 0 in >= 3/4 PRMs): **2/4 -> NOT MET**.

### Table 3. Grouping and spectrum

| PRM | Z17 K | CAT34 K | raw_i in same group as z_i | lead eigenvalue share Z17 / CAT34 | constant views dropped |
|---|---|---|---|---|---|
| Qwen-7B | 5 | 8 | 17/17 | 0.374 / 0.349 | none |
| Qwen-72B | 6 | 7 | 17/17 | 0.392 / 0.359 | none |
| PURE | 4 | 8 | 16/17 | 0.373 / 0.346 | none |
| RLHFlow | 4 | 7 | 17/17 | 0.361 / 0.339 | none |

### Table 4. |corr(raw_i, z_i)| by step class (mean over the 4 PRMs)

| channel | clean steps | error steps | pairs grouped together (of 4) |
|---|---|---|---|
| PRM | 0.72 | 0.72 | 4 |
| q15_H1 | 0.81 | 0.80 | 4 |
| q15_VE1 | 0.84 | 0.87 | 4 |
| chosen_surprisal | 0.82 | 0.82 | 3 |
| logprob_margin | 0.78 | 0.80 | 4 |
| true_tail50 | 0.74 | 0.68 | 4 |
| energy_level | 0.76 | 0.77 | 4 |
| energy_innovation | 0.82 | 0.84 | 4 |
| top15_turnover | 0.80 | 0.83 | 4 |
| top50_js | 0.63 | 0.64 | 4 |
| dominant_freq16 | 0.83 | 0.85 | 4 |
| bocpd_p0 | 0.83 | 0.83 | 4 |
| realized_z | 0.98 | 0.98 | 4 |
| realized_drv | 0.90 | 0.91 | 4 |
| digit_alternative | 0.74 | 0.77 | 4 |
| digit_spread | 0.76 | 0.80 | 4 |
| digit_alternative_innovation | 0.75 | 0.77 | 4 |

### Table 5. CAT34 groups per PRM

- **Qwen-7B**: G0 = z:digit_alternative, z:digit_spread, z:digit_alternative_innovation, raw:digit_alternative, raw:digit_spread, raw:digit_alternative_innovation; G1 = z:q15_VE1, z:energy_level, raw:q15_VE1, raw:energy_level; G2 = z:q15_H1, z:chosen_surprisal, z:logprob_margin, z:true_tail50, raw:q15_H1, raw:chosen_surprisal, raw:logprob_margin, raw:true_tail50; G3 = z:dominant_freq16, raw:dominant_freq16; G4 = z:realized_z, z:realized_drv, raw:realized_z, raw:realized_drv; G5 = z:bocpd_p0, raw:bocpd_p0; G6 = z:PRM, raw:PRM; G7 = z:energy_innovation, z:top15_turnover, z:top50_js, raw:energy_innovation, raw:top15_turnover, raw:top50_js
- **Qwen-72B**: G0 = z:digit_alternative, z:digit_spread, z:digit_alternative_innovation, raw:digit_alternative, raw:digit_spread, raw:digit_alternative_innovation; G1 = z:q15_H1, z:q15_VE1, z:chosen_surprisal, z:logprob_margin, z:true_tail50, z:realized_drv, raw:q15_H1, raw:q15_VE1, raw:chosen_surprisal, raw:logprob_margin, raw:true_tail50, raw:realized_drv; G2 = z:energy_level, z:bocpd_p0, raw:energy_level, raw:bocpd_p0; G3 = z:realized_z, raw:realized_z; G4 = z:energy_innovation, z:top15_turnover, z:top50_js, raw:energy_innovation, raw:top15_turnover, raw:top50_js; G5 = z:PRM, raw:PRM; G6 = z:dominant_freq16, raw:dominant_freq16
- **PURE**: G0 = z:chosen_surprisal, z:realized_z, z:realized_drv, raw:realized_z, raw:realized_drv; G1 = z:dominant_freq16, raw:dominant_freq16; G2 = z:energy_innovation, z:top15_turnover, z:top50_js, raw:energy_innovation, raw:top15_turnover, raw:top50_js; G3 = z:digit_alternative, z:digit_spread, z:digit_alternative_innovation, raw:digit_alternative, raw:digit_spread, raw:digit_alternative_innovation; G4 = z:energy_level, raw:energy_level; G5 = z:bocpd_p0, raw:bocpd_p0; G6 = z:PRM, raw:PRM; G7 = z:q15_H1, z:q15_VE1, z:logprob_margin, z:true_tail50, raw:q15_H1, raw:q15_VE1, raw:chosen_surprisal, raw:logprob_margin, raw:true_tail50
- **RLHFlow**: G0 = z:digit_alternative, z:digit_spread, z:digit_alternative_innovation, raw:digit_alternative, raw:digit_spread, raw:digit_alternative_innovation; G1 = z:chosen_surprisal, z:realized_z, z:realized_drv, raw:chosen_surprisal, raw:realized_z, raw:realized_drv; G2 = z:q15_H1, z:q15_VE1, z:logprob_margin, z:true_tail50, z:energy_level, raw:q15_H1, raw:q15_VE1, raw:logprob_margin, raw:true_tail50, raw:energy_level; G3 = z:PRM, raw:PRM; G4 = z:dominant_freq16, raw:dominant_freq16; G5 = z:energy_innovation, z:top15_turnover, z:top50_js, raw:energy_innovation, raw:top15_turnover, raw:top50_js; G6 = z:bocpd_p0, raw:bocpd_p0
