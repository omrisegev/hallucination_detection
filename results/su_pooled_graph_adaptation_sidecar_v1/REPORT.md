# SU-aware pooled graph adaptation sidecar v1

This is retrospective development evidence and does not alter a frozen baseline.

## Headline

- Reproduced observed-IU pooled graph: **+0.452pp** [-0.009,+0.915], 6/8 positive families.
- Prespecified IU + cross-family sparse cleaning: **+0.502pp** [+0.059,+0.942], graph increment +0.496pp.
- Direct primary-minus-current point contrast: **+0.050pp**.

## All arms

| arm | graph vs IU (pp) | 95% family bootstrap | wins | matched upstream vs IU | graph increment | independent no-graph |
|---|---:|---:|---:|---:|---:|---:|
| `iu_observed_mean` | +0.452 | [-0.009,+0.915] | 6/8 | +0.000 | +0.452 | +0.000 |
| `su_observed_mean` | -0.171 | [-1.423,+0.687] | 6/8 | -0.411 | +0.240 | -0.411 |
| `iu_all_sparse_mean` | +0.426 | [-0.059,+0.903] | 6/8 | -0.015 | +0.441 | -0.019 |
| `su_all_sparse_mean` | +0.403 | [-0.115,+0.878] | 6/8 | -0.011 | +0.415 | -0.011 |
| `iu_cross_sparse_mean` | +0.502 | [+0.059,+0.942] | 6/8 | +0.005 | +0.496 | +0.005 |
| `su_cross_sparse_mean` | +0.355 | [-0.193,+0.868] | 5/8 | -0.119 | +0.474 | -0.261 |
| `iu_shared_cross_mean` | +0.453 | [+0.014,+0.890] | 6/8 | +0.009 | +0.444 | +0.007 |
| `su_shared_cross_mean` | -0.159 | [-1.369,+0.635] | 6/8 | -0.395 | +0.237 | -0.394 |
| `iu_observed_geomedian` | +0.453 | [-0.000,+0.885] | 6/8 | +0.000 | +0.453 | +0.000 |
| `iu_cross_sparse_geomedian` | +0.475 | [+0.031,+0.903] | 6/8 | +0.005 | +0.470 | +0.005 |

## Interpretation boundary

The graph/covariance directions are label-free, but the hyperparameters are selected retrospectively inside nested folds. With eight development families, the bootstrap intervals are descriptive. A positive arm must still be frozen and transferred; the table cannot be used to choose an unregistered maximum and call it confirmation.
