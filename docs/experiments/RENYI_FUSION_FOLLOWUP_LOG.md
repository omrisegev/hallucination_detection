# Renyi / varentropy fusion follow-up experiment log

This is the short, append-only decision log for the ordered follow-up ladder.
Detailed protocols and machine-readable results live in their linked folders.
Every experiment stops for review before the next one begins.

## Experiment 1 — probability normalization, retained mass and preprocessing

- Date opened: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Question: did probability-head conditioning, discarded tail mass, or
  within-answer feature standardization cause the standalone-to-fusion loss?
- Fixed readout: Top10 token mean per official step.
- Fixed gate: mean entropy, training-fold quantile 0.3.
- Protocol: `PROBABILITY_NORMALIZATION_ABLATION_V1.md`
- Results: proper raw-head and conditional-head escort-varentropy agree to
  `8.73e-10`; four frozen score references replay to `7.05e-12`. `VE1 q50`
  gains .004678 PRMB within AUROC over q15 under the corrected interval while
  losing 0.285 PB points. A coarse tail bucket adds no useful signal. Removing
  within-answer centering restores roughly .04-.06 pooled/PRMScore without
  changing the registered within-answer ordering controls.
- Decision: `REJECT_P_TO_Q_AS_CAUSE`; promote `VE1 q50` as the support-width
  candidate; retain `VE.75 q15` for ProcessBench; do not add a scalar tail
  bucket; do not center every answer separately in calibrated fusion.
- Report: `../../results/probability_normalization_ablation_v1/REPORT.md`
- Frozen scores SHA256:
  `402af091e82a10147de659cac22befbd7f8fd3a5c8c3427082935368ab50daef`.

## Planned later experiments

1. Gate feature/readout optimization, beginning with Top10 mean.
2. Locator feature-bank expansion with H1, one top-1 view and retained mass.
3. Sparse position-varying simplex fusion.
4. One frozen new-model confirmation.

No later experiment is authorized by the execution of Experiment 1.

## Experiment 1B — benchmark-uniform support and weighting

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Question: can one q15/q50/multiscale representation and one weighting rule be
  used unchanged across ProcessBench and PRMBench?
- Protocol: `UNIFORM_MULTISCALE_FUSION_V1.md`
- Uniformity: one feature definition and one shared cross-fitted PB+PRMB model;
  no cell-, model- or benchmark-specific weights or method selection.
- Results: the frozen worst-regret rule selects q15 raw equal after per-view
  Top10 (`PB=.366201`, `PRMB-within=.753436`). q50 raw gains only .001407
  within while losing .634 PB points; q15+q50 raw adds only .000312 within and
  loses .368 PB points. Global scaling removes useful natural scale ratios.
  The supervised simplex zeroes both low-alpha q15 views and collapses the
  eight-view bank to q50 `VE.75/VE1`, then loses on the localization endpoints.
  Centering leaves PB/within invariant but costs .006-.012 fold-pooled AUROC and
  .0075-.0096 PRMScore.
- Decision: `SELECT_Q15_RAW_EQUAL_UNIFORMLY_ON_DEVELOPMENT`; this supersedes
  Experiment 1's benchmark-specific wording about retaining q15 for PB and q50
  for PRMB. Do not add static q15+q50 duplication; retain natural scale ratios;
  reject this fixed global step-BCE simplex; preserve answer-level means.
- Report: `../../results/uniform_multiscale_fusion_v1/REPORT.md`
- Frozen OOF scores SHA256:
  `2e87a3d11f4ee77594b64775457b1e99e864efb7d5eaca8350f9688b4adc14f4`.

## Updated continuation after Experiment 1B

1. Gate feature/readout optimization, beginning with Top10 mean and the q15
   locator held fixed.
2. Locator expansion with H1 and one top-1 view only under a separately frozen
   uniform protocol; do not reopen q50 or a scalar tail bucket by default.
3. Sparse position-varying fusion only with an objective aligned to the actual
   localization endpoints, not the rejected global step-BCE surrogate.
4. One frozen new-model confirmation.

## Finalist replay after Experiment 1B

- Date opened/completed: 2026-09-14
- Status: **COMPLETE / REVIEW PASS**
- Question: what does the one deployable specification selected by Experiments
  1/1B score when replayed by itself against the original static and
  position-temporal starting points?
- Protocol: `SELECTED_Q15_FINALIST_REPLAY_V1.md`
- Frozen finalist: q15 `{H0lim, VE0, VE0.75, VE1}`, label-free orientation,
  per-view Top10, raw equal step fusion, no centering/scaling, no supervised
  weights, no q50 duplication, and one method for every benchmark. The entropy
  q=.3 gate and PRMScore q=.8 calibration remain unchanged.
- Results: PB all-8 36.6201%, PRMB within .753436, fold-pooled .722708,
  PRMScore .634412. Versus the original fusion-before-Top10, PB is +.453pp and
  within is +.002321 (family-wise 98.333% CI [.001322,.003367]); PRMScore is
  essentially unchanged (-.000392). Versus the original local shrinkage +
  position method, PB is +.701pp, within +.006055
  [.003317,.008850], and PRMScore +.047003.
- Decision: `FREEZE_Q15_RAW_PER_VIEW_TOP10_AS_CURRENT_DEVELOPMENT_FINALIST`.
  The tested position-varying standardized fusion remains mechanism evidence,
  not the current winner. The next separate experiment may change the gate;
  this replay does not.
- Report: `../../results/selected_q15_finalist_replay_v1/REPORT.md`
- Frozen score SHA256:
  `3bd5c5b95474d75366b97b012b018d168cacafb3ccea178d26170207e220c7e6`.
