# Supervised conditional manifold discovery v1

**Decision: `TRANSFERABLE_SUPERVISED_DIRECTION_ONLY`**

This is supervised internal discovery on outcome-opened Global cells. It is not external validation and does not change the prior DUFS audit decision.

## Candidate gates

| candidate | support | geometry | distinct vs linear | utility | exact/CRT maxT |
|---|---:|---:|---:|---:|---:|
| `supervised_s10` | 10 | True | False | False | 0.005/0.005 |
| `supervised_s15` | 15 | True | False | False | 0.005/0.005 |
| `supervised_s5` | 5 | True | False | False | 0.005/0.005 |
| `supervised_sall` | 16 | True | False | False | 0.005/0.005 |

## Interpretation

A stable supervised representation transfers internally, but it does not establish local geometry beyond the search-matched linear direction.

Frozen candidate: `supervised_sall`. External validation remains unopened.

## Figures

![Conditional geometry and linear advantage](01_conditional_geometry_and_linear_advantage.png)

![Discovered feature weights](02_discovered_feature_weights.png)
