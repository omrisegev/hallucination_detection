# Why PRM boosting fails on Socratic: supervised ceiling and channel directions (2026-10-19)

**LABEL-USING DIAGNOSTIC.** Nothing here is a method or a candidate. Everything in this directory uses the
Socratic gold labels: the LR arms are trained on them, and the channel signs are read from them.

## Short answer

The telemetry does carry step-error signal on Socratic, but only a little beyond the PRM. The label-free
fusion fails to recover even that little: it lands well **below** the PRM.

For Qwen-7B (within-answer AUC):

| arm | within-AUC |
|---|---|
| PRM alone | 0.824 |
| supervised LR on PRM + N16 (oracle) | 0.836 |
| label-free CHAIN_N16 | 0.776 |
| label-free TELE_N16 | 0.712 |

## Part A: supervised LR ceiling

**Setup:** balanced LR, C=1, standardized. 5-fold GroupKFold by source question, out-of-fold. N = 2,995
answers. Official metric and paired bootstrap come from `../prm_boost_external_v1/eval_external.py`.
Files: `eval/RESULTS_socratic.csv`, `eval/BOOTSTRAP_socratic.csv`, `LR_CEILING_withinauc.csv`.

| PRM | PRM native: PRMScore / AUC | LR PRM+N13 (digit-free): Δ PRMScore / Δ AUC | LR PRM+N16: Δ PRMScore / Δ AUC | LR telemetry only (N16): AUC |
|---|---|---|---|---|
| Qwen-7B | 68.09 / 0.824 | −0.68 [−1.40, +0.09] / +0.009 [+0.005, +0.014] | −0.33 [−1.05, +0.42] / +0.013 | 0.806 |
| Qwen-72B | 71.96 / 0.814 | −4.15* / +0.012* | −3.80* / +0.024* | 0.809 |
| PURE | 63.71 / 0.796 | +3.97* / +0.035* | +4.01* / +0.034* | 0.782 |
| RLHFlow | 48.12 / 0.549 | +12.60* / +0.173* | +13.64* / +0.186* | 0.735 |
| MathShepherd | 64.39 / 0.718 | −1.04* / +0.043* | −1.04* / +0.043* | 0.697 |

\* means the 95% CI excludes 0. The Δ columns are against the PRM native. N13 and N16 are identical on
MathShepherd because its digit channels are inactive (amendment A2).

**Reading:**
1. **There is real but small headroom for the strong PRMs.** The supervised ceiling raises within-AUC by
   +0.009 to +0.024 on Qwen-7B and Qwen-72B, and by +0.035 to +0.043 on PURE and MathShepherd. On PRMBench,
   label-free CHAIN_N16 alone gained +0.028 to +0.113 (Table 1 of `../prm_boosting_bank_sweep_v1/README.md`).
   Socratic has less room to start with.
2. **Part of the PRMScore loss is the Q80 decision rule, not the score.** With the PRM's own score, Q80 is
   already below the PRM's native threshold:
   - Qwen-72B −5.10, MathShepherd −4.86, Qwen-7B −0.73 (`PRM_Q80` rows).
   - Even the supervised ranking cannot recover that under Q80.
3. **On RLHFlow the telemetry alone is much better than the PRM** (0.735 vs 0.549). That is why boosting
   "works" there.

## Part B: channel directions, PRMBench vs Socratic (`CHANNEL_DIRECTION_socratic.csv`)

Each channel is taken as L-SML sees it. Its within-answer AUC against the error label is compared between
Socratic and PRMBench. The PRMBench reference covers Qwen-7B only:
`results/prm_boosting_v1/CHANNEL_ATLAS_prm_backbone.csv`.

Qwen-7B, oriented AUC (supervised sign in parentheses):

| channel | PRMBench | Socratic |
|---|---|---|
| prm_risk | 0.801 (+) | **0.824 (+)** |
| energy_level | **0.799 (+)** | 0.683 (+) |
| realized_drv | 0.736 (+) | 0.623 (+) |
| logprob_margin | 0.744 (+) | 0.662 (+) |
| q15_H1 / q15_VE1 | 0.765 / 0.775 | 0.736 / 0.722 |
| true_tail50 | 0.755 | 0.771 |
| energy_innovation | 0.611 (−) | 0.636 (−) |
| top50_js | 0.586 (−) | 0.653 (−) |
| top15_turnover | 0.588 (+) | 0.560 (**−**) |
| realized_z | 0.572 (+) | 0.561 (**−**) |

**Reading:**
1. **The balance between PRM and telemetry flips.**
   - On PRMBench the best telemetry channel (energy_level, 0.799) is as strong as the PRM (0.801), so a
     near-equal fusion helps.
   - On Socratic the PRM is stronger (0.824), while energy_level falls to 0.683 and realized_drv to 0.623.
     The same near-equal fusion now dilutes the PRM.
   - This fits the hypothesis that part of the PRMBench gain comes from how PRMBench builds its errors
     (inserted edits that change the text's "energy"). It does not prove that hypothesis.
2. **The label-free TELE fit gives almost every channel a positive weight.**
   - `tele_n16_weight` > 0 on 76 of 80 channel-PRM pairs. The exceptions: realized_z on Qwen-72B (−0.28),
     and three near-zero weights on RLHFlow.
   - So channels whose error direction is negative get the wrong direction on both benchmarks. On
     Qwen-7B: energy_innovation, top50_js, and on Socratic also top15_turnover and realized_z (4 of 16).
   - Across the 5 PRMs, 3 to 6 channels of 16 have a fitted direction opposite to their supervised one
     (`tele_n16_uses_right_direction`).
   - The wrong-direction channels are weak (AUC 0.56–0.65), but they pull the fused score down.
3. **The direction errors are only part of the gap.**
   - On Qwen-7B the label-free TELE_N16 reaches 0.712 and the supervised LR on N16 reaches 0.806, a gap of
     0.094.
   - Wrong signs on 4 weak channels cannot explain all of that. The weights also do not follow channel
     strength.

## What this means for the next step

- **Do not tune a label-free variant on Socratic.** It is now inspected data, so any fix chosen from these
  tables would be label-selected.
- A fix can be designed and frozen, then tested on an untouched benchmark. The mechanisms point to two
  candidates:
  - learn each channel's sign rather than forcing all weights positive;
  - weight the PRM by its estimated reliability rather than giving it a near-equal share.
  ProcessBench, now being prepared for AIRCC, can serve as that test.
- **The ceiling is small for strong PRMs on Socratic** (≤ +0.024 AUC). Even a perfect label-free fusion
  would add little there. RLHFlow, a weak PRM, is where the telemetry helps.

## Gaps

- The PRMBench side of Part B covers Qwen-7B only. Per-channel atlases for the other 4 PRMs on PRMBench
  need their PRMBench FEATURES.npz files, which exist only on AIRCC.
- A PRMBench LR ceiling on PRM + N13/N16 with within-answer AUC was not computed here, for the same reason.
  `../prmbench_paper_metric_parity_v1/` has a pooled-AUROC LR analog on CROP's subset: 0.829.

## Reproduce

```
.venv/bin/python -B results/prm_boost_socratic_diagnosis_v1/diagnose.py <banks root> <socratic_gold.json> <scores dir>
.venv/bin/python -B results/prm_boost_external_v1/eval_external.py socratic <scores dir> <gold dir> results/prm_boost_socratic_diagnosis_v1/eval
```

- Banks root: `full_<prm>_socratic/bank_v2` from `gdrive:hallucination_detection/cluster_results/prm_boost_external_v1/`.
- Gold: `results/remote_claims_replay_2026-10-15/inputs/socratic_gold.json`.
- The OOF scores (npz) are not committed.
