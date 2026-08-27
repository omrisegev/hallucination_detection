# Pooled Graph-Roughness Direction V1

**Retrospective reconstruction; mechanism controls pending.**

Primary strict nested one-SE: **+0.251pp** AUROC (equal-family bootstrap 95% CI [+0.027, +0.458]pp), 6/8 positive and 90.8% of the frozen Family-NRM gain recovered.

Nested max-mean HPO sensitivity: **+0.450pp**, 6/8 positive.

| held dataset family | primary ΔAUROC (pp) | max-mean ΔAUROC (pp) |
|---|---:|---:|
| `gsm8k` | +0.365 | +0.663 |
| `hotpotqa` | +0.134 | +0.086 |
| `math500` | +0.228 | +0.379 |
| `nq_open` | -0.277 | -0.468 |
| `sciq` | +0.570 | +1.481 |
| `squad_v2` | +0.591 | +0.902 |
| `triviaqa` | -0.157 | -0.433 |
| `truthfulqa` | +0.558 | +0.986 |

Outer-direction cosine: min 0.929, mean 0.978.
The registered `D_0.30` lower-bound gate is FAIL: +0.168pp [-0.004, +0.322]pp.

The direction and every candidate score were fitted and hashed before this report opened labels. However, the graph, pooling rule, and selector were designed after the development outcomes were already known, so the result is discovery-level. PRMBench/HLE and ProcessBench/SemGrad are also known-outcome stress tests.
