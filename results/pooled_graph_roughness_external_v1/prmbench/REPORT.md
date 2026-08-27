# Pooled Graph-Roughness external — prmbench

**Retrospective known-outcome stress test; not independent confirmation.**

| method | AUROC | Δ vs IU (pp) | 95% CI (pp) | NRM recovery |
|---|---:|---:|---:|---:|
| `iu` | 0.720602 | +0.000 | — | 0.0% |
| `primary_one_se` | 0.716404 | -0.420 | [-0.621, -0.226] | -91.2% |
| `max_mean_sensitivity` | 0.711103 | -0.950 | [-1.315, -0.582] | -206.3% |
| `family_nrm` | 0.725206 | +0.460 | [+0.064, +0.865] | 100.0% |
| `dufs_graph` | 0.721999 | +0.140 | [-0.044, +0.318] | 30.3% |
| `contribution_graph` | 0.721139 | +0.054 | [-0.137, +0.241] | 11.7% |
| `cross_only` | 0.716417 | -0.418 | [-0.621, -0.225] | -90.9% |

All target scores and hashes were frozen before this report indexed target labels. Target transforms are transductive but label-free.
