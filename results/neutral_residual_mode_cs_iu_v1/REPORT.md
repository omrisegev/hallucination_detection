# Neutral residual mode CS-IU v1

**Status:** frozen label-free candidate; all results below are retrospective.

| domain | equal-group delta vs IU | 95% interval | W/L | worst |
|---|---:|---:|---:|---:|
| original_23 | +0.277pp | [+0.016, +0.533] | 15/8 | -1.804pp |
| processbench_llama | +1.580pp | [+0.918, +2.346] | 4/0 | +0.725pp |
| processbench_qwen | +0.557pp | [+0.236, +0.828] | 7/1 | -0.123pp |
| semgrad | +1.310pp | [+0.205, +2.415] | 2/0 | +0.205pp |

## Frozen source calibration

Selected eigenvalue: `1.035378`.

| family | direction |
|---|---:|
| `entropy_level` | +0.093928 |
| `entropy_dynamics` | -0.113808 |
| `sampled_token_energy` | -0.673995 |
| `partition_energy` | +0.714635 |
| `topk_distribution` | +0.112033 |
| `structural` | +0.026490 |

No HLE label or score is read by this audit.  The calibration above is the immutable input to the separate frozen HLE confirmation.
