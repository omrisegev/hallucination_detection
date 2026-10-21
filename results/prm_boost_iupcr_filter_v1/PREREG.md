# IU-PCR-style channel filter against the PRM: pre-registration (2026-10-21)

Request: Omri, 2026-10-21 ("ablation in the style of IU-PCR against the PRM; filter features like U-PCR").
This file is committed and pushed **before** any result of this experiment exists.

## Question

Can a label-free, PRM-referenced channel filter make the simple fusion of PRM + telemetry better than the PRM alone,
and better than the unfiltered mean (EQ_N16)?

## Inputs (no new inference, CPU only)

- Stage-1 exports of `results/prm_boost_collection_v1` (data root `COLLECTION_DATA`):
  `prmbench/ABL5_SCORES_<prm>.npz` + `ABL5_CHANNELS_<prm>.npz`; `socratic/SCORES_<prm>.npz` + `_CHANNELS.npz`.
- Bank N16 (answer-z, label-free-oriented channels, exactly the channels of EQ_N16 / CHAIN_N16).
  Math-Shepherd uses the digit-fixed bank (the collection exports already hold it).
- 5 PRMs: Qwen2.5-Math-PRM-7B, Qwen2.5-Math-PRM-72B, PURE-PRM-7B, RLHFlow-PRM-Mistral-8B, Math-Shepherd-Mistral-7B.
- 2 benchmarks: PRMBench (6,969 rows, official pool without the controls) and Socratic-PRMBench.
- Labels are read only by the evaluator (`evaluate_collection.py`), after every score is fixed.

## Method (label-free)

The PRM channel is `zp = answer_z(logit(1 - reward))`, the same channel that EQ_N16 and CHAIN_N16 use.

1. **IU-PCR estimate.** Stack `F = [zp, c_1..c_16]`, z-score each row over all steps, and run
   `spectral_utils.upcr.upcr_fit(F, **laplacian_upcr.IU_FIT_DEFAULTS)` (two-component L2 IU-PCR, no built-in
   exclusion). `rho_hat_full[j]` is the label-free estimate of Cov(f_j, error).
   Constant channels are removed first.
2. **Ranking (IU, primary).** Iterative drop: fit, remove the channel (never the PRM) with the smallest `rho_hat`,
   refit on the rest; repeat until 2 channels remain; order those 2 by the last fit. This gives a rank of the 16
   channels.
   **Ranking (CORR, secondary).** Pooled Pearson correlation of each channel with `zp` (agreement with the PRM).
3. **Auto-k (the pre-registered single arm, PRM as reference).** Start from all channels. Fit IU-PCR on
   `[zp] + kept`. Remove every kept channel with `rho_j < rho_PRM / 3` or `rho_j < 0.05 * var_y`
   (Algorithm 1 thresholds of Dror et al., with `rho_PRM` in the place of `rho_max`). Refit and repeat until nothing
   is removed or fewer than 2 channels remain.
4. **Fusion of `zp` + kept channels.**
   - `MEAN`: equal-weight mean of `zp` and the kept channels (k = 16 is exactly EQ_N16).
   - `CHAIN`: the CHAIN_N16 construction on the kept channels only (TELE L-SML on the kept channels, answer-z
     virtual classifiers, cross L-SML with `zp` as its own group, sign anchored to `zp`).
     k = 16 is exactly CHAIN_N16. k = 2: plain SML over the 3 views. k = 1: not defined (not reported).
   - k = 0 is the PRM alone (= PRM_Q80).
5. **Sweep.** k = 0..16 for both rankings and both fusions. The sweep is descriptive. No k is picked with labels.
   The best k per cell is reported only as a label-chosen reading (marked).

## Metrics and rules

- Official metric: PRMBench PRMScore; Socratic PRMScore (x100). Within-answer AUC.
- Main rule: **Q80** (each benchmark keeps its own code, as in the collection). Secondary: GAP.
- Paired bootstrap, 2,000 draws (PRMBench by source group, Socratic by answer), seeds of `evaluate_collection.py`.
- References: `PRM_Q80_PROB` (the stronger PRM scale under Q80), `PRM_Q80` (logit), `EQ_N16`, `CHAIN_N16`.

## Parity gates (must pass before any result is read)

- MEAN at k = 16 equals the exported `EQ_N16` (max abs diff < 1e-9) in all 10 cells.
- CHAIN at k = 16 equals the exported `CHAIN_N16` (max abs diff < 1e-6) in all 10 cells.
- `PRM_Q80_PROB`, `EQ_N16`, `CHAIN_N16` Q80 PRMScore equal the committed `ABL5_RESULTS` / collection values.

## Success rule (fixed now)

Primary arm: **IUF_MEAN_AUTO** (auto-k, MEAN fusion), rule Q80.

- **S1 (beats the PRM, PRMBench).** `IUF_MEAN_AUTO - PRM_Q80_PROB` PRMScore has 95% CI > 0 on at least 3 of 5 PRMs,
  and at least one of them is a Qwen PRM.
- **S2 (filter helps, PRMBench).** `IUF_MEAN_AUTO - EQ_N16` PRMScore point >= 0 on at least 4 of 5 PRMs, and 95% CI > 0
  on at least 2 of 5.
- **S3 (no damage, Socratic).** `IUF_MEAN_AUTO - PRM_Q80_PROB` Socratic PRMScore has 95% CI < 0 on at most 2 of 5 PRMs.

Verdict: "filter works" only if S1, S2 and S3 all hold. S1 + S3 without S2 = "PRM + telemetry works, the filter adds
nothing over EQ". Any other outcome = "fails". IUF_CHAIN_AUTO and the CORR ranking are secondary and do not change the
verdict. Within-AUC is reported next to every PRMScore and does not change the verdict.

## Caveat fixed in advance

PRMBench is development data: it was inspected many times. Socratic was inspected too. A pass here is not untouched
confirmation.
