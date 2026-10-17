# Label-free fusion of PRM + 3 channels (2026-10-14)

This replaces the fixed ½ PRM weight with a fitted label-free fusion of 4 views: PRM risk, `energy_level`,
`hist_entropy_cusum_abs_series` and `realized_drv`. The three channels come from each PRM's own backbone. Each
model is fitted on its pooled PRMBench steps, and the output is oriented to correlate positively with PRM.

Arms:
- `LSML_CONT`: `spectral_utils.fusion_utils.lsml_continuous` with automatic group detection.
- `SML4`: every view in its own group.
- `EQ4`: the plain mean of the 4 views.
- `S3`: the earlier fixed ½ rule.

AIRCC job 283967. Script: `lsml_four.py`.

## Official PRMScore (ANSWER_Z_Q80; PRM native = its own rule)

| PRM | PRM native | S3 (fixed ½) | EQ4 | SML4 | **CONT L-SML** | L-SML − S3 [95% CI] |
|---|---|---|---|---|---|---|
| Qwen2.5-Math-PRM-7B | 65.49 | 69.38 | 68.86 | 69.15 | **69.48** | +0.10 [−0.10, +0.31] |
| Qwen2.5-Math-PRM-72B | 68.18 | 69.39 | 69.40 | 69.55 | **69.84** | +0.45 [+0.23, +0.67] |
| PURE-PRM-7B | 65.31 | **66.77** | 66.31 | 66.39 | 66.69 | −0.08 [−0.34, +0.15] |
| RLHFlow-8B | 54.43 | 61.00 | 63.28 | 61.53 | **63.53** | +2.53 [+2.13, +2.92] |
| MathShepherd-7B | 47.02 | 63.15 | 63.34 | **64.55** | 63.68 | +0.54 [+0.16, +0.92] |

CONT L-SML compared with PRM native:
- 7B: +3.99
- 72B: +1.66
- PURE: +1.38
- RLHFlow: +9.10
- MathShepherd: +16.67

Every one of these CIs excludes 0.

The 7B native value is 65.49 here, against 65.46 from the Step 242 labels. The difference comes from rebuilding the
labels with a strict `> 0.5`.

## Weights and groups found by L-SML

These are least-squares coefficients of the fused score on the 4 views, normalized so the absolute values sum to 1.

- **72B, 7B and PURE.** K=2: PRM forms one group and the 3 channels form the other. PRM gets a weight of about
  0.37 to 0.38.
- **RLHFlow.** PRM is grouped with energy and cusum, and `realized_drv` is alone. PRM's weight drops to 0.29.
- **MathShepherd.** PRM is grouped with cusum and drv. PRM's weight is 0.23.

L-SML sets PRM's weight by itself, and that weight is lower when the PRM is weaker. This is the behaviour the fixed
½ rule could not provide.
