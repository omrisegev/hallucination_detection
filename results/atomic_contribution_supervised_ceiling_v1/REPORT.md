# Atomic contribution supervised ceiling

**Status:** supervised diagnostic only; not a label-free method.

| representation | prior | equal-group delta vs IU | 95% interval | W/L | worst |
|---|---:|---:|---:|---:|---:|
| family | 0.3 | +0.721pp | [+0.308, +1.113] | 21/2 | -0.238pp |
| family | 1 | +0.444pp | [+0.221, +0.654] | 21/2 | -0.050pp |
| family | 3 | +0.209pp | [+0.116, +0.298] | 22/1 | -0.031pp |
| family | 10 | +0.078pp | [+0.047, +0.105] | 21/2 | -0.009pp |
| atomic | 0.3 | +1.298pp | [+0.478, +1.958] | 20/3 | -1.158pp |
| atomic | 1 | +1.042pp | [+0.638, +1.397] | 22/1 | -0.072pp |
| atomic | 3 | +0.599pp | [+0.396, +0.777] | 23/0 | +0.056pp |
| atomic | 10 | +0.231pp | [+0.144, +0.303] | 22/1 | -0.016pp |

## Direct atomic-minus-family contrast

| prior | atomic minus family | 95% interval |
|---:|---:|---:|
| 0.3 | +0.577pp | [+0.102, +0.910] |
| 1 | +0.598pp | [+0.385, +0.786] |
| 3 | +0.390pp | [+0.264, +0.498] |
| 10 | +0.153pp | [+0.079, +0.207] |

Each AUROC is computed on one held-out split, then averaged within cell. No out-of-fold predictions are concatenated across cells. All heads use class-balanced loss.
