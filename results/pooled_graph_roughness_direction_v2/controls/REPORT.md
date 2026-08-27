# Pooled Graph-Roughness V2 mechanism controls

**FAIL** for the complete registered graph-attribution gate.

Real residual graph: +0.251pp versus IU.

| matched control | control vs IU (pp) | real − control (pp) | 95% CI (pp) |
|---|---:|---:|---:|
| `dufs_graph` | +0.011 | +0.240 | [-0.162, +0.620] |
| `contribution_graph` | +0.299 | -0.048 | [-0.329, +0.216] |
| `equal_cell_pooling` | +0.127 | +0.124 | [-0.041, +0.288] |
| `cross_only` | +0.245 | +0.006 | [+0.002, +0.012] |
| `family_axis_permuted` | +0.009 | +0.242 | [-0.061, +0.534] |
| `node_permuted_00` | +0.074 | +0.178 | [-0.203, +0.531] |

Twenty matched node permutations average -0.159pp; real minus their mean is +0.411pp [+0.185, +0.618]pp, randomization p=0.0476.

Controls were fitted and hashed without row-level targets after the primary hyperparameters were frozen. They are retrospective mechanism tests, not independent validation.
