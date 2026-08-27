# SU-aware pooled graph adaptation — conservative V2

V2 repairs the failed V1 reproduction by fixing union k=7 and using the canonical one-SE/tail-guard selector. V1 is an optimistic sensitivity only.

## Headline

- Current observed-IU pooled graph: **+0.251pp** [+0.027,+0.458], 6/8 wins.
- Prespecified IU + cross-family sparse cleaning: **+0.260pp** [+0.045,+0.454], graph increment +0.255pp.
- Direct primary minus current: **+0.009pp** [-0.012,+0.037], 4/8 families.

## Arms

| arm | graph vs IU | 95% CI | wins | clean/rho alone | graph increment | independently selected no-graph |
|---|---:|---:|---:|---:|---:|---:|
| `iu_observed_mean` | +0.251 | [+0.027,+0.458] | 6/8 | +0.000 | +0.251 | +0.000 |
| `su_observed_mean` | -0.198 | [-1.198,+0.408] | 6/8 | -0.411 | +0.213 | -0.411 |
| `iu_all_sparse_mean` | +0.262 | [+0.053,+0.450] | 7/8 | +0.017 | +0.246 | +0.017 |
| `su_all_sparse_mean` | +0.233 | [-0.060,+0.473] | 6/8 | -0.011 | +0.244 | -0.106 |
| `iu_cross_sparse_mean` | +0.260 | [+0.045,+0.454] | 6/8 | +0.005 | +0.255 | +0.005 |
| `su_cross_sparse_mean` | -0.077 | [-0.881,+0.433] | 6/8 | -0.350 | +0.273 | -0.350 |
| `iu_shared_cross_mean` | +0.245 | [+0.031,+0.443] | 6/8 | -0.002 | +0.247 | -0.001 |
| `su_shared_cross_mean` | -0.192 | [-1.195,+0.410] | 6/8 | -0.414 | +0.222 | -0.414 |
| `iu_observed_geomedian` | +0.252 | [+0.026,+0.459] | 6/8 | +0.000 | +0.252 | +0.000 |
| `iu_cross_sparse_geomedian` | +0.256 | [+0.041,+0.451] | 6/8 | +0.005 | +0.250 | +0.005 |

## Claim boundary

All directions and fit artifacts are label-free; lambda, trust, and alpha are retrospectively meta-selected in nested folds. The primary comparison is the paired clean-minus-current contrast, not whether each arm separately exceeds IU. No arm is confirmation until frozen transfer is run.
