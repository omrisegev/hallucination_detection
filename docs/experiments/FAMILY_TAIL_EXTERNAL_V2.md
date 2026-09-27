# Family-tail external transfer V2: the defect-corrected row

Date: 2026-09-24. Branch: `codex/lsml-external-generalization-v1`.

This follows up on `FAMILY_TAIL_EXTERNAL_V1.md` and its results (commit 1d3c23681). It adds one arm to that study and changes nothing else.

## Why a V2

Claude's Step 443 (branch `claude/ssl-pseudolabel-residual-v1`, `results/tail_threshold_calibration_v1/`) found a defect in the V1 candidate recipe.
- V1 passed the centred tail marks to `lsml_continuous` without z-scoring them. Their variance is ~0.18, and the function's documented input is z-scored.
- The default `loading_scale='unit'` K criterion is not scale invariant, so K collapsed to 2 on every source fold. The same continuous features multiplied by 0.4 also give K=2.
- With standardized marks, K=4 and the partition is stable.
- On source, the corrected row scores lower: 64.17 PRMScore against 64.52 for V1 and 64.33 for family equal. It is added because it is the correct recipe, not because it scored better.

## Contract

- **Lock**: `results/family_tail_transfer_v2/TRANSFER_LOCK_V2.json`, sha256 `0c5c55996503394edf2faeb1bc066e3fdec3c6eb315c79c0e7d59ca7d61c986c`.
  - It keeps every V1 row, recipe item, deployment weight and threshold. The rescoring script asserts this.
  - It adds `F15_tailstd_lsml`: the same marks, z-scored over the pooled fit rows, then `lsml_continuous` (unit, small_m_guard). Weights are applied to the continuous family features, followed by within-answer z.
  - The row is fitted on source folds 0-3; its q80 threshold comes from fold 4 and is 0.87716.
- **Exposure**: the lock was written before V1 external results were read. The V1 results already existed, and the lock records that. The corrected row is fully determined by the defect fix. Nothing was chosen with external data. The benchmarks were exposed before, so this is exploratory, not confirmation.
- **Features**: reused from the sealed V1 records. They are the same telemetry, masks and 48-channel extraction that passed the full source parity gate. There was no new extraction, GPU work or target fitting.
- **Replay gate**: before sealing, the ten V1 arms must reproduce every V1 score (to within 1e-12) and every decision. The observed difference was 0.0 on 6,190/6,190 records.
- **Analysis**: V1 evaluator logic, unchanged.
  - Primary family: the corrected row against the V1 tail row, family equal, family covariance L-SML and bank11 L-SML. That is 4 contrasts x 3 cells = 12, Bonferroni-corrected, with 100,000 source-question draws and seed 20260924.
  - The six V1 contrasts are replayed with their V1 family of 18. They match V1 exactly.

## Results

Scores are percentages. The Hard2Verify and Socratic metrics must not be averaged.

| Method | Hard2Verify balanced F1 | Socratic-Qwen3 PRMScore | Socratic-QwQ PRMScore |
|---|---:|---:|---:|
| bank11 L-SML | **43.67** | **63.22** | **64.24** |
| Family15 equal | 42.38 | 61.12 | 62.94 |
| **Family15 tail L-SML, corrected (V2)** | 42.28 | 61.00 | 62.30 |
| Family15 tail L-SML, V1 (K=2) | 41.02 | 59.92 | 62.69 |
| Family15 covariance L-SML | 42.02 | 60.02 | 61.15 |
| CT7 | 37.75 | 58.75 | 60.17 |

Differences are in points. Intervals are Bonferroni-corrected over 12 contrasts.

| Corrected minus | Hard2Verify | Socratic-Qwen3 | Socratic-QwQ |
|---|---|---|---|
| V1 tail row | +1.26 [-2.05, +4.69] | +1.08 [+0.50, +1.68] | -0.39 [-0.98, +0.20] |
| Family equal | -0.10 [-1.97, +1.78] | -0.12 [-0.44, +0.21] | -0.64 [-0.98, -0.30] |
| Family covariance L-SML | +0.26 [-1.24, +1.85] | +0.98 [+0.64, +1.33] | +1.15 [+0.77, +1.54] |
| bank11 L-SML | -1.39 [-4.01, +1.24] | -2.22 [-2.78, -1.66] | -1.94 [-2.55, -1.34] |

The source-disjoint sensitivity panel agrees in direction.

**Interpretation.**
- The fix brings the family-tail row up to the family-equal level, the same pattern as on source.
- Learning on tail marks beats continuous covariance L-SML on the same families, but it adds nothing over plain family averaging.
- bank11 L-SML leads every cell by 1.4-2.2 points. It is also the only representation where learned weights beat equal both on source and externally.
- Source rank did not predict external rank.

**Not done**: Codex's three independent audits (metric recompute, coverage, null) were not re-run on V2.

## Reproduction

```text
python -B .worktrees/ssl-pseudolabel-residual-v1/scripts/experiments/family_tail_transfer_lock_v2.py   # source lock (claude/ssl-pseudolabel-residual-v1)
python scripts/run_family_external_v2.py
python scripts/evaluate_family_external_v2.py --seal-only
python scripts/evaluate_family_external_v2.py
```

The copied lock must match the sha256 above. `results/family_tail_external_v2/IMPLEMENTATION_FREEZE.json` pins the code, inputs and V1 seals. Per-answer records, predictions and bootstrap arrays are git-ignored, as in V1. They can be regenerated from the V1 records in minutes.
