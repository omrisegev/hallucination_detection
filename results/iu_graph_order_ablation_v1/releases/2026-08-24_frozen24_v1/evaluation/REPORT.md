# IU graph-order ablation v1 — result

**Status:** D0 retrospective mechanism evidence; not independent validation.

The current roster's DEEM-B3 is graph-free. The exact residual/Laplacian
objective proposed for this ablation is reported as residual ridge correction,
not as DEEM-B3.

## Macro-24 paired results

All deltas are percentage points versus freshly recomputed IU-PCR; intervals
use 20,000 paired source-group bootstrap draws.

| arm | ΔAUROC [95% CI] | ΔAUPRC [95% CI] |
|---|---:|---:|
| `residual_ridge_correction__lam_0p03` | +0.0010 [-0.0223, +0.0198] | +0.0006 [-0.0113, +0.0112] |
| `residual_ridge_correction__lam_0p3` | +0.0050 [-0.0401, +0.0469] | -0.0079 [-0.0471, +0.0323] |
| `feature_smooth_residual_graph__lam_0p03` | -0.0014 [-0.0313, +0.0266] | -0.0070 [-0.0385, +0.0214] |
| `feature_smooth_raw_graph__lam_0p1` | +0.0010 [-0.0409, +0.0433] | -0.0549 [-0.1248, +0.0067] |
| `score_smooth_residual_graph__lam_0p03` | +0.0071 [-0.0230, +0.0353] | -0.0014 [-0.0352, +0.0290] |

## Interpretation

At weak regularization every graph arm is statistically tied with IU-PCR.
Increasing lambda makes the graph operation mechanically stronger but degrades
the macro result. The exact residual-ridge arm at lambda=.03 remains -0.515pp below signed DEEM-B3 [-0.782, -0.246] and -0.492pp below equal-family mean [-0.786, -0.194].

Therefore neither smoothing X before IU nor the exact constrained residual
correction explains the DEEM-B3/equal-family advantage on frozen24. The result
supports treating equal-family balancing as the simpler live explanation and
does not support a graph-guided residual-correction claim on this panel.

See `macro_response_curve.png` and `per_cell_delta_heatmap.png`. Every plotted
number is sourced from `contrasts_long.csv`.
