# Claude real-data checks (development, label-free fitting)

| method | PB % | within | PRMScore (default cal.) |
|---|---:|---:|---:|
| replay_native_LL | 39.3825 | 0.757218 | 0.635103 |
| replay_simplex_LL | 39.3033 | 0.758262 | 0.638258 |
| cluster3_native_LL | 36.5267 | 0.749427 | 0.621376 |
| cluster3_simplex_LL | 39.3797 | 0.757860 | 0.637137 |
| innovation5 | 39.8314 | 0.760293 | 0.638830 |
| original4 | 37.4749 | 0.753436 | 0.634412 |

| contrast | PB delta | PB CI | within CI | level |
|---|---:|---|---|---|
| cluster3_native_LL_minus_replay_native_LL | -2.8558 | [-4.5397, -1.223] | [-0.011677, -0.003924] | 0.975 |
| cluster3_simplex_LL_minus_replay_simplex_LL | +0.0764 | [-0.2241, 0.3954] | [-0.000853, 3.5e-05] | 0.975 |
| cluster3_native_LL_minus_innovation5 | -3.3046 | [-4.5394, -2.0735] | [-0.013566, -0.008097] | 0.95 |
| cluster3_simplex_LL_minus_innovation5 | -0.4517 | [-0.8374, -0.0717] | [-0.00311, -0.001766] | 0.95 |
| replay_simplex_LL_minus_innovation5 | -0.5280 | [-0.9238, -0.1501] | [-0.002674, -0.001416] | 0.95 |

```
{
 "answers": 13769,
 "max_replay_delta": 0.0,
 "g2_over_var_y": {
  "min": 1.0,
  "median": 1.0,
  "at_ceiling_fraction": 1.0
 },
 "additive_residual_all_pairs": {
  "median": 0.09179166835057544,
  "q90": 0.12165362200006107
 },
 "additive_residual_cross_group": {
  "median": 0.006588499103810018,
  "q90": 0.014119166852087656
 },
 "rho_cosine_allpairs_vs_cluster3": {
  "median": 0.9929012844733968,
  "q10": 0.9866415091855868
 },
 "partition_top": [
  [
   "04|1|23",
   8848
  ],
  [
   "014|2|3",
   4535
  ],
  [
   "01|23|4",
   386
  ]
 ],
 "modal_partition": "04|1|23",
 "modal_share_by_fold": {
  "0": 0.6488138030194105,
  "1": 0.6343065693430657,
  "2": 0.6879120879120879,
  "3": 0.6195809248554913,
  "4": 0.6227719170607494
 },
 "modal_share_by_cell": {
  "pb_gsm8k_q4": 0.92,
  "pb_gsm8k_q8": 0.93,
  "pb_math_q4": 0.875,
  "pb_math_q8": 0.858,
  "pb_olympiadbench_q4": 0.861,
  "pb_olympiadbench_q8": 0.82,
  "pb_omnimath_q4": 0.744,
  "pb_omnimath_q8": 0.693,
  "prmbench_qwen3_8b": 0.46735543119529344
 },
 "feature_order": [
  "H0lim",
  "VE0",
  "VE075",
  "VE1",
  "innovation"
 ],
 "seconds": 98.12749530000292
}
```
