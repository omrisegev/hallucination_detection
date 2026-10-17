# Learned rosters and weight allocation

Exact aliases count once canonically; expanded weights split their mass. Active membership can include zero readout weights. Frequencies and mean weights describe NATIVE folds only; failed folds are explicit and never counted as a selected roster.

Weights describe this fit, not a causal contribution or feature quality test.

Native fit counts: {'base': 5, 'duplicates': 5, 'noise': 2, 'near_copies': 3, 'structured_noise': 5}

| Original feature | Base retained folds | Near-copy retained folds |
|---|---:|---:|
| rank_1_risk | 5/5 | 3/3 |
| rank_2_risk | 5/5 | 3/3 |
| rank_3_risk | 2/5 | 1/3 |
| rank_4_risk | 2/5 | 2/3 |
| rank_5_risk | 2/5 | 2/3 |
| rank_6_risk | 2/5 | 1/3 |
| rank_7_risk | 1/5 | 1/3 |
| rank_8_risk | 1/5 | 0/3 |
| rank_9_risk | 1/5 | 0/3 |
| rank_10_risk | 0/5 | 0/3 |
| rank_11_risk | 0/5 | 0/3 |
| rank_12_risk | 2/5 | 1/3 |
| rank_13_risk | 2/5 | 2/3 |
| rank_14_risk | 0/5 | 0/3 |
| rank_15_risk | 0/5 | 0/3 |
| surprisal | 5/5 | 3/3 |
| top1_loggap | 5/5 | 3/3 |
| censored_rank50 | 5/5 | 3/3 |
| mass_above | 5/5 | 3/3 |
| tail15 | 5/5 | 3/3 |
| tail50 | 0/5 | 0/3 |
| a0.25 | 5/5 | 3/3 |
| a0.5 | 5/5 | 3/3 |
| H1 | 5/5 | 3/3 |
| a2 | 5/5 | 3/3 |
| a4 | 3/5 | 1/3 |
| Hinf | 4/5 | 3/3 |
| H0lim | 5/5 | 2/3 |
| ve0 | 5/5 | 3/3 |
| ve0.5 | 5/5 | 3/3 |
| ve0.75 | 5/5 | 3/3 |
| ve1 | 5/5 | 3/3 |
| ve2 | 5/5 | 3/3 |
| ve4 | 5/5 | 3/3 |
| top2_ratio | 3/5 | 1/3 |
| a0.25_prefix_innovation | 5/5 | 3/3 |
| a0.5_prefix_innovation | 5/5 | 3/3 |
| H1_prefix_innovation | 5/5 | 3/3 |
| a2_prefix_innovation | 5/5 | 3/3 |
| a4_prefix_innovation | 1/5 | 1/3 |
| Hinf_prefix_innovation | 4/5 | 2/3 |
| H0lim_prefix_innovation | 3/5 | 0/3 |
| ve0_prefix_innovation | 1/5 | 1/3 |
| ve0.5_prefix_innovation | 4/5 | 3/3 |
| ve0.75_prefix_innovation | 5/5 | 3/3 |
| ve1_prefix_innovation | 5/5 | 3/3 |
| ve2_prefix_innovation | 5/5 | 3/3 |
| ve4_prefix_innovation | 5/5 | 3/3 |
| top15_turnover | 5/5 | 3/3 |
| top50_truncated_js | 5/5 | 3/3 |
| bocpd_signed_residual | 5/5 | 3/3 |

| Family | Base mean absolute weight | Near-copy mean absolute weight |
|---|---:|---:|
| probability_ranks | 0.119093 | 0.136532 |
| provided_token_confidence | 0.189845 | 0.182516 |
| tail_mass | 0.012961 | 0.011552 |
| distribution_shape | 0.337417 | 0.336768 |
| top2_ratio | 0.014202 | 0.009416 |
| prefix_innovation | 0.265422 | 0.269685 |
| top15_turnover | 0.008238 | 0.007357 |
| top50_truncated_js | 0.001828 | 0.001760 |
| bocpd | 0.050995 | 0.044414 |
| added | 0.000000 | 0.000000 |
