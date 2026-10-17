# Label-free choice between views (post hoc, exploratory)

## ProcessBench (gate open, erroneous, both views valid): 3579 answers

VE_0 and VE_0.75 point at different steps on 42.5% of them. When they disagree: VE_0 right 0.225, VE_0.75 right 0.268, neither 0.508.

| disagreement case | n | VE_0 right | VE_0.75 right | earlier argmax right | later argmax right |
|---|---:|---:|---:|---:|---:|
| VE_0 points earlier | 693 | 0.283 | 0.175 | 0.283 | 0.175 |
| VE_0.75 points earlier | 828 | 0.176 | 0.345 | 0.345 | 0.176 |

| n_steps (disagreements only) | n | VE_0 right | VE_0.75 right |
|---|---:|---:|---:|
| 1-4 | 177 | 0.282 | 0.463 |
| 5-8 | 808 | 0.265 | 0.265 |
| 9-12 | 397 | 0.154 | 0.229 |
| 13+ | 139 | 0.122 | 0.144 |

## Label-free combination rules on the pair (hit rate on the same answers; rules use no labels)

| rule | exact-hit rate |
|---|---:|
| VE_0 alone | 0.327 |
| VE_0.75 alone | 0.345 |
| later argmax of the pair | 0.306 |
| earlier argmax of the pair | 0.366 |
| VE_0.75 if its argmax is step 0, else VE_0 | 0.329 |
| mean of z-scored steps: VE_0 + VE_0.75 | 0.352 |
| mean of z-scored steps: VE_0 + VE_0.75 + VE_1 + H0lim | 0.345 |
| mean of z-scored steps: all five | 0.342 |
| max of z-scored steps: VE_0 + VE_0.75 | 0.348 |
| oracle: either of the pair (ceiling, uses labels) | 0.441 |

## PRMBench: does a label-free answer statistic predict (AUC VE_0 − AUC VE_0.75)?  n = 6030, mean diff +0.0211

| label-free statistic | Spearman with AUC diff | VE_0 better in lowest quartile | in highest quartile |
|---|---:|---:|---:|
| n_steps | +0.028 | 0.300 | 0.491 |
| n_tokens | +0.027 | 0.301 | 0.487 |
| mean token entropy (gate detector) | +nan | nan | nan |
| argmax position of VE_0 (relative) | -0.053 | 0.426 | 0.380 |
| argmax position of VE_0.75 (relative) | -0.010 | 0.464 | 0.395 |
| argmax disagreement (|pos diff|) | +0.002 | 0.318 | 0.479 |
| std of VE_0 step scores / std of VE_0.75 | +0.009 | 0.403 | 0.403 |

Reading rule: a |Spearman| near 0 means the statistic cannot tell which view to trust; the quartile columns show the fraction of answers where VE_0 has the higher AUC.