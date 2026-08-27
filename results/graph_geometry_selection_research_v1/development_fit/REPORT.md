# Graph Geometry Selection Research V1 — development

**Provisional decision: `GEOMETRY_SEARCH_SELECTION_OPTIMISM`.** the expanded class improved inner selected means but lost on outer families.

## Anchor decomposition

- Fixed union-k7 + one-SE/canonical: **+0.251477pp** (exact anchor).
- Fixed union-k7 + max-mean/canonical: **+0.449629pp** (exact anchor).
- Selector effect: **+0.198152pp**.
- Matched searched/max-mean/V1-trust with the common eight-lambda grid: **+0.451606pp**.
- Separate exact legacy V1 reproduction (five-lambda grid): **+0.451606pp**.
- Matched trust-grid effect (fixed/max-mean, V1 minus canonical): **+0.000000pp**.
- Matched geometry-capacity effect (max-mean/V1 trust): **+0.001977pp**.
- Legacy five-lambda minus common eight-lambda effect: **+0.000000pp**.

## Selector comparison

| method | mean ΔAUROC (pp) | 95% family bootstrap | oracle regret (pp) | oracle geometry agreement |
|---|---:|---:|---:|---:|
| `canonical_fixed_one_se` | +0.251 | [+0.027, +0.458] | +0.286 | 1/8 |
| `fixed_max_mean` | +0.450 | [-0.003, +0.896] | +0.087 | 1/8 |
| `supervised_geometry_one_se` | +0.224 | [-0.109, +0.531] | +0.313 | 3/8 |
| `supervised_geometry_max_mean` | +0.437 | [-0.010, +0.778] | +0.100 | 4/8 |
| `intrinsic_label_free` | +0.220 | [+0.023, +0.423] | +0.317 | 2/8 |
| `held_family_geometry_oracle` | +0.537 | [+0.311, +0.715] | +0.000 | 8/8 |
| `held_family_full_tuple_ceiling` | +1.041 | [+0.618, +1.494] | -0.504 | 5/8 |

## Actuator decomposition

`full` and `cross` were frozen and evaluated as separate arms; no selector could choose between them. Each paired full−cross row uses the full arm's selected trust for both directions.

Because every correction `R d` is normalized to a fixed requested SD, `cross = -cbar` has no lambda parameter and identifies direction only. Target-free diagnostics include `cosine(d_full,-cbar)`, leave-source stability and dispersion of `cbar`, plus 20 deterministic node permutations per geometry. If full approximately equals cross throughout the bank, the mechanism is a pooled graph cross-gradient rather than a quadratic graph solve.


## Boundaries

The new fit consumed a physically target-free archive and every candidate score hash was verified before this report opened outcomes. The canonical historical fit was logically label-whitelisted but not physically isolated; its score bank remains hash-consistent.

The inherited 20-node-permutation, contribution, DUFS, cross-only, equal-cell, and family-axis controls apply exactly to the fixed residual union-k7 arm. They do not validate a newly selected geometry. No SU covariance cleaning or SU-rho arm appears here.

These eight-family comparisons are retrospective and conditional on an already outcome-informed frozen feature contract. External datasets are also opened stress tests, not confirmation.
