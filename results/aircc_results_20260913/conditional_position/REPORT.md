# Conditional IU follow-up: position

Full cached development benchmark. External covariance fitting excludes held source-group folds.
Native IU rho and its two-dimensional eigenspace stay fixed within each answer. No model-performance threshold is applied.

Primary contrast: position minus position_scale_only; 98.33333% interval; 10,000 grouped draws.

| Method | PB % | PRMB within AUC | PRMScore | Coverage |
|---|---:|---:|---:|---:|
| Original answer-local RBM12 | 36.27123 | 0.74520 | 0.62222 | 13769/13769 |
| Answer-local Shrinkage IU | 20.27275 | 0.69059 | 0.56861 | 13769/13769 |
| Shrinkage IU + pooled covariance prior | 20.32481 | 0.69430 | 0.56885 | 13769/13769 |
| Shrinkage IU + answer-position covariance prior | 35.80656 | 0.75690 | 0.58991 | 13769/13769 |
| Position prior: baseline direction, scale change only | 32.13622 | 0.74153 | 0.58285 | 13769/13769 |
| Position prior with shuffled position assignments | 20.28093 | 0.69052 | 0.56795 | 13769/13769 |

All thirteen frozen reference rows are in COMPARISON.csv. Failures remain in the coverage denominator.
Conditional endpoints and common-cohort uncertainty coverage are recorded explicitly in METRICS.json and ENDPOINT_CONTRASTS.json.
Per-answer outer weight maps and every outer/inner fit diagnostic are retained in the checkpoint SQLite.
