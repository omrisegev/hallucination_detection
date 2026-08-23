# DEEM vs IU-PCR — 24-cell frozen benchmark

**Decision:** `CONTINUOUS_DEEM_NONINFERIOR_TO_IUPCR`

The previous residual-graph hypothesis was closed because Phase 0 failed the frozen specificity gate. That result does not falsify graph-free continuous B3. This report evaluates only B0–B3.

## Equal-family results

| arm | AUROC | AUPRC | QA AUROC | math AUROC |
|---|---:|---:|---:|---:|
| B0 — IU-PCR | 0.742810 | 0.774869 | 0.759081 | 0.784317 |
| B1 — hard DEEM adapter | 0.708241 | 0.719450 | 0.722973 | 0.743370 |
| B2 — repaired soft/rank DEEM | 0.677975 | 0.716923 | 0.705292 | 0.754495 |
| B3 — continuous additive DEEM | 0.748501 | 0.777998 | 0.764035 | 0.792484 |

## Preregistered B3 contrasts

| contrast | Δ AUROC | 95% interval | Holm p | W/T/L |
|---|---:|---:|---:|---:|
| B3−B0 | +0.005691 | [+0.001258, +0.009752] | 0.0061 | 17/1/6 |
| B3−B1 | +0.040261 | [+0.026701, +0.054370] | 0.0003 | 24/0/0 |
| B3−B2 | +0.070526 | [+0.020070, +0.130140] | 0.0003 | 21/1/2 |

## Interpretation boundary

B1/B2 are pinned 0.2.0 adapter controls, not paper-exact DEEM. B3 is a continuous-visible adaptation. No graph arm, graph hyperparameter, Localization result, or Early Detection result is part of this benchmark.
