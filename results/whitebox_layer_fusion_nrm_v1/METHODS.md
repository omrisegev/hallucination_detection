# Methods — white-box NRM-CS-IU v1

This is a label-free fitting addendum over the frozen prepared feature bundles
from `results/whitebox_layer_fusion_v2/`. It does not rerun capture and does not
alter any v2 score or comparison.

For a standardized white-box feature matrix `F` and anchor-oriented IU-PCR
weights `w`, group contribution `g` is

`h_g(x) = sum_{i in g} w_i F_i(x)`.

The group contributions reconstruct the frozen v2 IU score to at most `1e-10`
in every cell. Each source cell standardizes `h_g`, residualizes it against
standardized IU, and contributes one equal-weight residual covariance matrix.
The calibration chooses the covariance eigenvector nearest eigenvalue one,
orients it to positive sum, and applies the correction at `1/G` standard
deviations. These are the frozen NRM-CS-IU choices; labels never choose the
mode, sign, grouping, or scale.

Depth groups are relative quartiles, allowing 32-, 36-, and 40-layer models to
share the ordered basis `depth_band_0` through `depth_band_3`. Lens groups are
the twelve fixed combinations of `{attn, mlp, resid}` and
`{entropy, target NLL, top-1 surprisal, KL-to-final}` on the existing spaced
layers.

Every target is scored three times:

- `LODO`: calibration excludes all cells of the target dataset;
- `LOMO`: calibration excludes all cells of the target exact model;
- `LOCO`: calibration excludes only the target cell.

Uncertainty uses 2,000 deterministic paired bootstrap draws with root seed
`20260812`, resampling complete problem groups within each cell and reusing the
same draws across methods. Headlines are equal-cell macros over the 13 eligible
cells; the rejected CoQA/Llama-1 cell remains appendix-only.
