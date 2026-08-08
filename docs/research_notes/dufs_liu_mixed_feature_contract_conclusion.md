# DUFS-LIU mixed feature contract: development conclusion

## Question

The first frozen DUFS-LIU benchmark removed four raw features because their
relationship with correctness was not reliably monotone:

- `pe_mean`
- `stft_spectral_entropy`
- `cusum_shift_idx`
- `rpdi`

Earlier comparisons transformed all four features in the same way. That was an
unnecessary restriction. This experiment asked whether each feature should be
removed, kept raw, squared around zero, or folded around its label-free KDE mode.

## Protocol

The search evaluated all `4^4 = 256` global contracts. A transformed feature
replaced its parent; raw and transformed copies never coexisted. DUFS-LIU kept
its frozen settings: seeds 11, 23, and 37; 80 DUFS epochs; graph `k=7`; and
`lambda=0.1`.

The fit phase did not read labels. It wrote and hashed every score for every
contract and cell. The report opened labels only after the complete score bank
was frozen. Contract selection was then inspected in two ways:

1. the best contract on all 24 development cells;
2. leave-one-dataset-family-out (LOFO), where a held family could not choose its
   own contract.

This is still retrospective development. The same 24 cells influenced earlier
work, and the best of 256 contracts is expected to look too good on those cells.

## Selected development candidate

| feature | operation | meaning after transformation |
|---|---|---|
| `pe_mean` | `squared` | `-z^2`; values near the cell mean are larger |
| `stft_spectral_entropy` | `mode` | `-|rank-mode_rank|`; values near the label-free density mode are larger |
| `cusum_shift_idx` | `raw` | frozen confidence orientation, then z-score |
| `rpdi` | `raw` | frozen confidence orientation, then z-score |

The two transformed views replace their raw parents. The other two raw views
return to the pool with fixed directions. Missing features remain missing.

## Results

| method and contract | cell-macro AUROC | change from stable-only |
|---|---:|---:|
| IU-PCR, stable-only | 0.774063 | -- |
| IU-PCR, its retrospective mixed winner | 0.776261 | +0.220pp |
| DUFS-LIU, stable-only | 0.774139 | -- |
| DUFS-LIU, mixed-v2 candidate | 0.776562 | +0.242pp |

For DUFS-LIU the selected candidate improved 17 cells and hurt 7. Its worst
cell change was -0.279pp; its largest change was +3.201pp on
`math500_qwenmath7b`. Under the same mixed-v2 feature contract, DUFS-LIU was
+0.048pp above ordinary IU-PCR. The Laplacian effect is therefore still small,
but it is larger than the +0.008pp difference under stable-only features.

The LOFO contract-selection procedure scored 0.775367, or +0.123pp over the
stable-only DUFS-LIU scores. This transfer estimate is fragile. Removing the
single `math500_qwenmath7b` cell reduces the LOFO mean change to about +0.022pp.

Two decisions were stable across the eight held-family folds:

- `rpdi=raw` in 8/8 folds;
- `stft_spectral_entropy=mode` in 7/8 folds.

The other decisions were not stable:

- `pe_mean`: mode 4/8, squared 3/8, drop 1/8;
- `cusum_shift_idx`: raw 4/8, squared 4/8.

The exact retrospective winner is therefore not a proved universal contract.

## Decision

Freeze `dufs-liu-mixed-v2-development-2026-08-07` as the feature contract for
the next external DUFS-LIU run. Do not replace the published stable-only number
with 0.776562: that number helped select the contract and is optimistic.

The old `fixed_stable_v1` result remains the unbiased-with-respect-to-this-search
historical baseline. A new dataset/model family must compare stable-only and
mixed-v2 without changing any transformation, DUFS setting, or Laplacian
parameter. Promotion requires the mixed-v2 gain to persist outside
`math500_qwenmath7b` and outside the MATH-500 family.

This result refines the baseline. It does not reverse the broader conclusion
that static graph objectives have not yet identified sample-local correctness
reliability.

## Files

- Runner: `scripts/dufs_liu_feature_contract_search.py`
- Frozen contract: `spectral_utils/dufs_liu_feature_contract.py`
- Full report: `results/dufs_liu_feature_contract_search/REPORT.md`
- Score bank and freeze manifest: `results/dufs_liu_feature_contract_search/`
