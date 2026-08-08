# Feature-orientation contract

## Decision

`confidence-orientation-v1` is the fixed inference-time contract for the 30 raw in-scope
features. After orientation, every column has one meaning:

> A larger value means “more likely correct.”

The direction is an offline, versioned constant. It is never inferred from the labels—or from
`sign(rho)`—of the cell being scored. Unknown features fail closed instead of silently receiving
`+1`.

The historical mapping remains available as `LEGACY_FEATURE_SIGNS`; it is not the default for new
work.

## Corrections

Thirteen old directions were inconsistent with the earlier consensus audit and the current
equal-family audit:

| feature | legacy | v1 |
|---|---:|---:|
| `trace_length` | +1 | -1 |
| `high_band_power` | -1 | +1 |
| `hl_ratio` | -1 | +1 |
| `dominant_freq` | -1 | +1 |
| `spectral_centroid` | -1 | +1 |
| `hurst_exponent` | +1 | -1 |
| `cusum_shift_idx` | +1 | -1 |
| `epr_spilled` | +1 (implicit fallback) | -1 |
| `sw_var_peak_spilled` | +1 (implicit fallback) | -1 |
| `cusum_max_spilled` | +1 (implicit fallback) | -1 |
| `min_spilled` | +1 (implicit fallback) | -1 |
| `epr_energy` | -1 | +1 |
| `min_energy` | -1 | +1 |

All 30 contract directions match the equal-family empirical direction in the committed 24-cell
bundle. That is a retrospective consistency check, not external validation. A leave-one-family-out
audit gives the same direction for 28/30 features; the two exceptions are weak, unstable raw views
described below.

## Stable schema

`fixed_stable_v1` quarantines four raw views:

- `pe_mean`
- `stft_spectral_entropy`
- `cusum_shift_idx`
- `rpdi`

They are not declared useless. Prior work shows cell-specific non-monotonicity or weak directional
transfer. A later monotone replacement may reintroduce one, but it must **replace** its parent—not be
added beside it—because deterministic duplicates distort U-PCR's pair equations.

The stable schema is an experimental arm, not a blanket deletion from every legacy experiment.
This is deliberate: on the real artifact it helps SU-PCR/SDSF in aggregate, but it is fractionally
worse for plain U-PCR and does not eliminate SDSF's real-cell tails.

## DUFS-LIU mixed-v2 development contract

`fixed_stable_v1` remains the historical frozen baseline. The next external
DUFS-LIU run uses a separately versioned development candidate rather than
removing all four views:

| feature | operation |
|---|---|
| `pe_mean` | replace with `-z^2` |
| `stft_spectral_entropy` | replace with `-|rank(x)-mode_rank|` |
| `cusum_shift_idx` | keep raw under its frozen confidence direction |
| `rpdi` | keep raw under its frozen confidence direction |

The code registry is
`spectral_utils/dufs_liu_feature_contract.py` and its version is
`dufs-liu-mixed-v2-development-2026-08-07`. The mapping was selected from 256
contracts on the existing 24 development cells. It must not be changed after
seeing the next external-family results. The observed +0.242pp development
gain is selection-biased and is not a replacement for the historical headline.

## Evidence after the correction

The artifact-only replay in `results/fixed_orientation_validation/` shows:

- U-PCR `fixed_all_v1` versus deployable `sign(rho)`: **+0.07pp** cell macro and **+0.10pp**
  equal-family macro; worst-cell change **-0.40pp**.
- SDSF `fixed_stable_v1` versus `sign(rho)`: **+1.62pp** cell macro and **+2.36pp** equal-family
  macro, but worst-cell change remains **-13.07pp**.
- The consensus anchor and historical `epr` anchor choose the same global sign in **288/288**
  comparisons.

Therefore fixed direction solves the avoidable per-feature polarity seam for U-PCR. It does **not**
by itself establish SDSF on the real task.

The disjoint-seed synthetic v2 benchmark in
`results/synthetic_dependency_fusion_fixed_v2/` provides the mechanism result:

- Fixed SDSF beats fixed SU-PCR by **+0.845pp**, 95% bootstrap CI **[+0.747, +0.943]pp**, with
  **39/40** wins in the primary sparse-large world.
- Across both sparse worlds it wins **77/80** repetitions.
- Fixed SU-PCR versus fixed IU-PCR is effectively zero, so sparse covariance cleaning alone does
  not explain the gain.
- The full admission decision remains `STOP_AND_REVISE` because that SU-PCR-specific gate was
  preregistered and failed. The scientific claim must be narrowed before a real-data run.

This cleanly separates the two findings: the catastrophic synthetic tails came from the feature
orientation seam, while the useful synthetic gain comes from SDSF's full dependency/reliability
weighted solve.

## Reproduction

Run the dataset-free contract and fusion gates:

```bash
python scripts/test_feature_contract.py
python scripts/test_dependency_fusion.py
```

Replay the committed real artifact without the original data:

```bash
python scripts/fixed_orientation_validation.py
```

Run the full synthetic mechanism benchmark:

```bash
python scripts/synthetic_dependency_fusion_validation.py --skip-dufs
```

The synthetic script returns exit code 2 when a full admission gate fails. That is expected for
the current v2 result and is not an execution error.
