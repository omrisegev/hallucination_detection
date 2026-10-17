# Conditional IU follow-up: graph_local

Full cached development benchmark. External covariance fitting excludes held source-group folds.
Native IU rho and its two-dimensional eigenspace stay fixed within each answer. No model-performance threshold is applied.

Primary contrast: graph_local minus sliding_window; 98.33333% interval; 10,000 grouped draws.

| Method | PB % | PRMB within AUC | PRMScore | Coverage |
|---|---:|---:|---:|---:|
| Original answer-local RBM12 | 36.27123 | 0.74520 | 0.62222 | 13769/13769 |
| Answer-local Shrinkage IU | 20.27275 | 0.69059 | 0.56861 | 13769/13769 |
| Shrinkage IU + pooled covariance prior | 20.32481 | 0.69430 | 0.56885 | 13769/13769 |
| Sliding-window IU + pooled covariance prior | 35.54707 | 0.74182 | 0.58590 | 13769/13769 |
| Local token-graph IU + pooled covariance prior | 21.16468 | 0.70115 | 0.57174 | 13769/13769 |
| Local token-graph IU with shuffled graph | 36.26779 | 0.74945 | 0.58857 | 13769/13769 |

All thirteen frozen reference rows are in COMPARISON.csv. Failures remain in the coverage denominator.
Conditional endpoints and common-cohort uncertainty coverage are recorded explicitly in METRICS.json and ENDPOINT_CONTRASTS.json.
Per-answer outer weight maps and every outer/inner fit diagnostic are retained in the checkpoint SQLite.
