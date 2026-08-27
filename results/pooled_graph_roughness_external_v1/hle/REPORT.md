# Pooled Graph-Roughness external — hle

**Retrospective known-outcome stress test; not independent confirmation.**

| method | AUROC | Δ vs IU (pp) | 95% CI (pp) | NRM recovery |
|---|---:|---:|---:|---:|
| `iu` | 0.516775 | +0.000 | — | 0.0% |
| `primary_one_se` | 0.525894 | +0.912 | [+0.248, +1.512] | 264.0% |
| `max_mean_sensitivity` | 0.533014 | +1.624 | [+0.243, +2.821] | 470.1% |
| `family_nrm` | 0.520229 | +0.345 | [-0.922, +1.601] | 100.0% |
| `dufs_graph` | 0.505397 | -1.138 | [-1.844, -0.452] | -329.3% |
| `contribution_graph` | 0.514333 | -0.244 | [-1.054, +0.571] | -70.7% |
| `cross_only` | 0.525929 | +0.915 | [+0.250, +1.514] | 265.0% |

All target scores and hashes were frozen before this report indexed target labels. Target transforms are transductive but label-free.
