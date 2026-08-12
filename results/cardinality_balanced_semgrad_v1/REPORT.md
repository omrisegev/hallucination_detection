# Frozen CB-CS-IU transfer to SemGrad

**Status:** independent-example frozen transfer with historical label visibility disclosed.

Across SciQ and TruthfulQA with equal dataset weight, CB-CS-IU changed BEM-error AUROC by **-0.767pp** versus ordinary IU. The hierarchical paired interval is [-1.950, +0.435]pp.

## Per-dataset primary result

| dataset | IU AUROC | CB AUROC | delta | paired 95% interval |
|---|---:|---:|---:|---:|
| sciq | 0.7450 | 0.7467 | +0.175pp | [-0.393, +0.756] |
| truthfulqa | 0.6942 | 0.6771 | -1.708pp | [-2.225, -1.195] |

## Equal-dataset mechanism contrasts

| contrast | delta | hierarchical 95% interval |
|---|---:|---:|
| `cardinality - iu` | -0.767pp | [-1.950, +0.435] |
| `leverage - iu` | -0.396pp | [-1.064, +0.279] |
| `dufs_liu - iu` | +0.119pp | [-0.119, +0.348] |
| `cardinality - leverage` | -0.371pp | [-1.094, +0.282] |
| `cardinality - dufs_liu` | -0.885pp | [-2.256, +0.539] |
| `cardinality - uniform` | +0.325pp | [+0.026, +0.832] |
| `cardinality - reverse_cardinality` | -1.265pp | [-3.694, +1.144] |

## Frozen gates

- **FAIL — positive CB delta in both datasets:** `{'sciq': 0.1747692204029372, 'truthfulqa': -1.7080882817934628}`
- **FAIL — positive equal-dataset hierarchical interval:** `[-1.9502940618478515, 0.43481739774947625]`
- **FAIL — tail safety:** `-1.7080882817934628`
- **FAIL — CB beats reversed direction in both datasets:** `False`
- **PASS — numerical invariants:** `8.881784197001252e-16`

## Boundary

Fit received telemetry-only dictionaries and could not access `bem_correct`, `bem_score`, or the temporary ROUGE-L `label`. Data, BEM manifests, scores, source files, and row identities were verified before BEM-error evaluation.

These are independent answer-level examples and a different benchmark protocol from the development evidence. They are not pristine in the strongest historical sense because their labels and earlier IU/DUFS results already existed elsewhere in the repository.
