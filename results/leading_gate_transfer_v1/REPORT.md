# Leading simple gates as localization gates v1 — report

Status: **COMPLETE / REVIEW PASS**

| Gate | Math q | Math answer F1 | PB answer F1 | PB AUROC | PB localization | Delta vs baseline | 99% interval |
|---|---:|---:|---:|---:|---:|---:|---:|
| `q15_VE1__token_top10` | 0.45 | 0.633562 | 0.687474 | 0.787407 | 34.9234% | -1.697pp | [-3.692, +0.267]pp |
| `entropy_native__token_top10` | 0.45 | 0.633271 | 0.693567 | 0.792620 | 35.5339% | -1.086pp | [-2.979, +0.895]pp |
| `q15_Hinf__token_top10` | 0.45 | 0.632930 | 0.681939 | 0.778047 | 35.0077% | -1.612pp | [-3.466, +0.363]pp |
| `q15_raw4_mean__token_top10` | 0.45 | 0.632755 | 0.697559 | 0.798699 | 35.5691% | -1.051pp | [-2.958, +0.895]pp |
| `tail15_mass__token_top10` | 0.40 | 0.632415 | 0.697932 | 0.799571 | 36.6736% | +0.054pp | [-1.632, +1.804]pp |
| Existing entropy mean q=.3 | 0.30 PB baseline | -- | 0.649999 | 0.742301 | 36.6201% | -- | -- |

The answer-detection and localization rankings are intentionally reported separately. No feature or q was selected on ProcessBench in this transfer; interpreting the displayed ranking as a development choice requires later external confirmation.

Answer-detection ranking: `tail15_mass__token_top10`, `q15_raw4_mean__token_top10`, `entropy_native__token_top10`, `q15_VE1__token_top10`, `q15_Hinf__token_top10`.

Localization ranking: `tail15_mass__token_top10`, `q15_raw4_mean__token_top10`, `entropy_native__token_top10`, `q15_Hinf__token_top10`, `q15_VE1__token_top10`.
