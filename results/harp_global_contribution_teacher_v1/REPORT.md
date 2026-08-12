# HARP-inspired global contribution teacher

**Status:** supervised proof-of-feasibility; not a deployable unsupervised method.

One six-family correction was trained on the original 23 cells. External target labels never entered that fit.

| evaluation domain | method | equal-group delta vs IU | 95% interval | W/L | worst |
|---|---|---:|---:|---:|---:|
| original_23 | `global_teacher` | +0.410pp | [+0.252, +0.572] | 20/3 | -0.473pp |
| original_23 | `cardinality` | +0.442pp | [+0.207, +0.681] | 17/6 | -0.595pp |
| processbench_llama | `global_teacher` | +1.191pp | [+0.823, +1.498] | 4/0 | +0.653pp |
| processbench_llama | `cardinality` | +1.263pp | [+0.708, +1.692] | 4/0 | +0.476pp |
| processbench_qwen | `global_teacher` | +0.684pp | [+0.544, +0.821] | 8/0 | +0.451pp |
| processbench_qwen | `cardinality` | +0.853pp | [+0.676, +1.088] | 8/0 | +0.564pp |
| semgrad | `global_teacher` | +0.646pp | [+0.330, +0.961] | 2/0 | +0.330pp |
| semgrad | `cardinality` | -0.767pp | [-1.708, +0.175] | 1/1 | -1.708pp |

## Source-23 teacher coefficients

| family | delta |
|---|---:|
| `entropy_level` | +0.020845 |
| `entropy_dynamics` | -0.035412 |
| `sampled_token_energy` | -0.031546 |
| `partition_energy` | +0.051543 |
| `topk_distribution` | +0.025367 |
| `structural` | +0.057866 |

## Interpretation

If the global teacher transfers where cardinality balancing fails, then the target correction is present and reusable in contribution space, while the current label-free nuisance proxy is insufficient. That is evidence for continuing self-supervised target-direction research, not for deploying these supervised coefficients.
