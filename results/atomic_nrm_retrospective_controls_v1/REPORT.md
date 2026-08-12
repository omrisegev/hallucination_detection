# Atomic NRM candidate v1 — retrospective controls

All calibration and scoring functions are label-free. The datasets' labels were already open historically and the retrospective loader reads them in the same process solely for AUROC; they are never passed to a candidate or control fit.

| domain | method | equal-group delta vs IU | 95% interval | W/L | worst |
|---|---|---:|---:|---:|---:|
| original_23 | `family_nrm` | +0.277pp | [+0.020, +0.534] | 15/8 | -1.804pp |
| original_23 | `atomic_projector_invabs` | -0.667pp | [-1.234, -0.033] | 6/17 | -2.287pp |
| original_23 | `atomic_projector_equal` | -1.051pp | [-1.607, -0.385] | 4/19 | -4.086pp |
| original_23 | `atomic_closest_one` | -0.996pp | [-1.577, -0.316] | 1/22 | -4.086pp |
| original_23 | `learned_partition` | -0.741pp | [-1.330, -0.141] | 3/20 | -4.171pp |
| original_23 | `refined_partition` | +0.033pp | [-0.241, +0.307] | 10/13 | -1.675pp |
| original_23 | `coarsened_partition` | -0.939pp | [-1.986, -0.175] | 5/18 | -4.173pp |
| original_23 | `random_partition_mean` | -0.182pp | [-0.383, +0.009] | 7/16 | -0.733pp |
| processbench_llama | `family_nrm` | +1.580pp | [+0.918, +2.346] | 4/0 | +0.725pp |
| processbench_llama | `atomic_projector_invabs` | -1.106pp | [-1.644, -0.605] | 0/4 | -1.881pp |
| processbench_llama | `atomic_projector_equal` | -1.168pp | [-1.649, -0.663] | 0/4 | -1.858pp |
| processbench_llama | `atomic_closest_one` | -1.275pp | [-1.696, -0.813] | 0/4 | -1.799pp |
| processbench_llama | `learned_partition` | -1.118pp | [-1.436, -0.822] | 0/4 | -1.550pp |
| processbench_llama | `refined_partition` | +0.077pp | [-0.511, +0.876] | 1/3 | -0.672pp |
| processbench_llama | `coarsened_partition` | +0.334pp | [-0.440, +1.219] | 2/2 | -0.747pp |
| processbench_llama | `random_partition_mean` | +0.136pp | [-0.176, +0.449] | 2/2 | -0.266pp |
| processbench_qwen | `family_nrm` | +0.557pp | [+0.236, +0.828] | 7/1 | -0.123pp |
| processbench_qwen | `atomic_projector_invabs` | -1.305pp | [-1.987, -0.629] | 1/7 | -2.961pp |
| processbench_qwen | `atomic_projector_equal` | -1.499pp | [-2.035, -0.884] | 1/7 | -2.953pp |
| processbench_qwen | `atomic_closest_one` | -1.729pp | [-2.121, -1.223] | 1/7 | -2.834pp |
| processbench_qwen | `learned_partition` | -1.354pp | [-2.003, -0.705] | 0/8 | -2.343pp |
| processbench_qwen | `refined_partition` | -0.250pp | [-0.327, -0.197] | 2/6 | -0.771pp |
| processbench_qwen | `coarsened_partition` | -0.253pp | [-1.519, +0.784] | 4/4 | -2.095pp |
| processbench_qwen | `random_partition_mean` | -0.022pp | [-0.216, +0.174] | 4/4 | -0.535pp |
| semgrad | `family_nrm` | +1.310pp | [+0.205, +2.415] | 2/0 | +0.205pp |
| semgrad | `atomic_projector_invabs` | -4.216pp | [-6.590, -1.842] | 0/2 | -6.590pp |
| semgrad | `atomic_projector_equal` | -4.340pp | [-6.840, -1.840] | 0/2 | -6.840pp |
| semgrad | `atomic_closest_one` | -4.327pp | [-6.881, -1.774] | 0/2 | -6.881pp |
| semgrad | `learned_partition` | -2.234pp | [-3.138, -1.330] | 0/2 | -3.138pp |
| semgrad | `refined_partition` | +0.291pp | [-0.407, +0.990] | 1/1 | -0.407pp |
| semgrad | `coarsened_partition` | -2.153pp | [-2.939, -1.366] | 0/2 | -2.939pp |
| semgrad | `random_partition_mean` | -0.674pp | [-0.837, -0.511] | 0/2 | -0.837pp |

## Matched random-partition distribution

| domain | p05 | median | p95 | positive | match/beat family NRM |
|---|---:|---:|---:|---:|---:|
| original_23 | -0.531pp | -0.254pp | +0.277pp | 16/50 | 3/50 |
| processbench_llama | -1.612pp | +0.226pp | +1.136pp | 34/50 | 1/50 |
| processbench_qwen | -1.193pp | +0.042pp | +1.038pp | 26/50 | 13/50 |
| semgrad | -2.701pp | -0.418pp | +1.418pp | 17/50 | 3/50 |

## Direct contrast with family NRM

| domain | method | delta vs family NRM | 95% interval |
|---|---|---:|---:|
| original_23 | `atomic_projector_invabs` | -0.944pp | [-1.654, -0.174] |
| original_23 | `atomic_projector_equal` | -1.328pp | [-2.073, -0.460] |
| original_23 | `atomic_closest_one` | -1.273pp | [-2.015, -0.416] |
| original_23 | `learned_partition` | -1.018pp | [-1.822, -0.188] |
| original_23 | `refined_partition` | -0.244pp | [-0.596, +0.078] |
| original_23 | `coarsened_partition` | -1.217pp | [-2.475, -0.245] |
| original_23 | `random_partition_mean` | -0.459pp | [-0.881, -0.040] |
| processbench_llama | `atomic_projector_invabs` | -2.686pp | [-3.214, -2.159] |
| processbench_llama | `atomic_projector_equal` | -2.748pp | [-3.215, -2.281] |
| processbench_llama | `atomic_closest_one` | -2.855pp | [-3.247, -2.463] |
| processbench_llama | `learned_partition` | -2.698pp | [-3.168, -2.355] |
| processbench_llama | `refined_partition` | -1.503pp | [-1.977, -0.988] |
| processbench_llama | `coarsened_partition` | -1.246pp | [-2.261, -0.231] |
| processbench_llama | `random_partition_mean` | -1.444pp | [-1.836, -1.060] |
| processbench_qwen | `atomic_projector_invabs` | -1.862pp | [-2.665, -0.878] |
| processbench_qwen | `atomic_projector_equal` | -2.056pp | [-2.718, -1.127] |
| processbench_qwen | `atomic_closest_one` | -2.285pp | [-2.735, -1.443] |
| processbench_qwen | `learned_partition` | -1.911pp | [-2.341, -1.401] |
| processbench_qwen | `refined_partition` | -0.806pp | [-1.025, -0.563] |
| processbench_qwen | `coarsened_partition` | -0.810pp | [-1.742, -0.057] |
| processbench_qwen | `random_partition_mean` | -0.579pp | [-0.669, -0.454] |
| semgrad | `atomic_projector_invabs` | -5.526pp | [-9.005, -2.047] |
| semgrad | `atomic_projector_equal` | -5.650pp | [-9.255, -2.045] |
| semgrad | `atomic_closest_one` | -5.637pp | [-9.296, -1.978] |
| semgrad | `learned_partition` | -3.544pp | [-5.553, -1.535] |
| semgrad | `refined_partition` | -1.018pp | [-1.425, -0.611] |
| semgrad | `coarsened_partition` | -3.462pp | [-5.354, -1.570] |
| semgrad | `random_partition_mean` | -1.984pp | [-3.252, -0.716] |

Feature-order invariance max error: 8.882e-16.
