# Graph Geometry Selection Research V1 — post-report audit

**Status: `PASS`.** This audit read only frozen development JSON/CSV artifacts; it opened neither raw labels nor the feature archive.

## Corrected oracle semantics

Held-family geometry oracles now use donor-selected correction strength under the same policy as the method being compared. One-SE and max-mean are separate estimands; the full-tuple held-label ceiling is not called geometry regret.

| policy | method | mean ΔAUROC (pp) | matched oracle regret (pp) | agreement |
|---|---|---:|---:|---:|
| `intrinsic_fixed_strength` | `canonical_fixed_strength` | +0.251477 | +0.200544 | 2/8 |
| `intrinsic_fixed_strength` | `held_family_geometry_oracle` | +0.452021 | +0.000000 | 8/8 |
| `intrinsic_fixed_strength` | `intrinsic_label_free` | +0.219821 | +0.232201 | 1/8 |
| `max_mean` | `fixed_residual_union_k7` | +0.449629 | +0.273210 | 1/8 |
| `max_mean` | `held_family_geometry_oracle` | +0.722839 | +0.000000 | 8/8 |
| `max_mean` | `supervised_geometry_selector` | +0.436682 | +0.286157 | 3/8 |
| `one_se` | `fixed_residual_union_k7` | +0.251477 | +0.285623 | 1/8 |
| `one_se` | `held_family_geometry_oracle` | +0.537100 | +0.000000 | 8/8 |
| `one_se` | `supervised_geometry_selector` | +0.223622 | +0.313478 | 3/8 |

The one-SE policy-matched geometry oracle is **+0.537100pp**; the max-mean policy-matched geometry oracle is **+0.722839pp**.
The intrinsic selector is **+0.219821pp** versus its fixed-strength geometry oracle at **+0.452021pp**.

All policy-matched regret values are nonnegative by construction and checked at runtime. The separately named full-tuple ceiling is **+1.040837pp**.

## Corrected selection optimism

The matched estimand is `(searched_inner − fixed_inner) − (searched_outer − fixed_outer)`.

For the Phase-B selector bank under max-mean it is **+0.163612pp** (5/8 positive families; exact one-sided sign-flip p=0.265625).
The earlier **+0.159140pp** number was the searched arm's raw inner-minus-outer gap, not the matched search-optimism estimand.

## Actuator CSV semantic correction

The legacy `lambda_is_a_cross_parameter` column is misnamed. Its values were verified on every row to mean `lambda_is_full_parameter`; cross has no lambda parameter. The corrected interpretation is emitted in `actuator_arms_semantic_correction.csv`.

## Finding

`GEOMETRY_SEARCH_SELECTION_OPTIMISM` remains supported after using the matched difference-in-differences. The geometry-oracle headroom must be quoted with its selector policy, and the held-label full-tuple result remains an optimism ceiling only.
