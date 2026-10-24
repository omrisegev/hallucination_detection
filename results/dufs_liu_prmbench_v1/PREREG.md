# DUFS-LIU on PRMBench localization: label calibration, PRM pseudo-labels, and replacing continuous L-SML (pre-registration, 2026-10-24)

Request: Omri, 2026-10-24.
1. "Can some label-using calibration identify better whether DUFS-LIU-PCR sits on a nuisance manifold or on the
   hallucination manifold?"
2. "Can DUFS-LIU-PCR be used for hallucination localization on PRMBench instead of continuous L-SML?"
3. (follow-up the same day) "Let the PRM itself give a pseudo-label that we use to overcome the DUFS-LIU-PCR problems."

This file is committed **before** any score of the new arms exists. Only the evaluator parity check
(published CHAIN_N16 / TELE_N16 / EQ_N16 / PRM numbers re-computed in this worktree) was run before writing it.

## Background (what is already known)

- DUFS-LIU = DUFS-gated Laplacian-regularized IU-PCR (`docs/methods/dufs_liu.md`, `spectral_utils/laplacian_upcr.py`).
  Its documented failure: DUFS can learn a smooth nuisance manifold that is unrelated to correctness, and IU-PCR's
  latent is whatever the channels share most, which need not be the error.
- TA-LIU (target-anchored LIU, `spectral_utils/target_anchored_laplacian.py`, `SPEC_TARGET_ANCHORED_LIU_V1.md`) is
  the existing label-using variant: calibration labels set the graph gates. Synthetic development only
  (`results/target_anchored_laplacian_synthetic/`): it rescued the nuisance world (+1.27 AUROC points over IU-PCR)
  but plain logistic regression with the same 16 labels gained +19.5, so the "same-label attribution" gate failed.
  It was never run on real data.
- PRMBench, current line: CHAIN_N16 (continuous L-SML over the PRM and the N16 bank) beats the PRM under GAP for
  all 5 PRMs (`results/prm_boost_collection_v1/`). On telemetry alone, IU-PCR and continuous L-SML tie
  (`configs/current_line_registry.json`). Supervised LR beats both on telemetry (Step 464).

## Population and inputs (no new inference, CPU only)

- PRMBench, all 6,969 answers / 94,203 steps; 5 PRMs: Qwen2.5-Math-PRM-7B, Qwen2.5-Math-PRM-72B, PURE-PRM-7B,
  RLHFlow-PRM-Mistral-8B, Math-Shepherd-Mistral-7B. Development data (inspected many times); nothing here is
  untouched confirmation.
- Stage-1 exports of `results/prm_boost_collection_v1` (data root `COLLECTION_DATA=/Users/osegev/Desktop/hd_collection_data`):
  `prmbench/ABL5_SCORES_<prm>.npz`, `ABL5_CHANNELS_<prm>.npz`. Labels and metadata through
  `results/prm_boost_collection_v1/evaluate_collection.py` (`load_prmbench`).
- Channels: bank N16 (answer-z, label-free-oriented), exactly the channels of CHAIN_N16 / EQ_N16.
- PRM channel: `zp = answer_z(score__PRM_Q80)` = answer-z of logit(1 - reward), the channel CHAIN_N16 uses.
- `X17 = [zp, N16]`, `X16 = N16`. Every column is z-scored over all steps (as `monday_cont_baselines_v1/run.py:fit_iu`).
  Constant columns are dropped and recorded.

## Fixed DUFS-LIU recipe (inherited, not tuned)

Registered DUFS-LIU (`spectral_utils/historical_multitask_baselines.py:fit_registered_dufs_global`):
`adapted_dufs_soft_gates(F, seeds=(11, 23, 37), epochs=80)`, `build_graph_from_features(F, gates, k=7)`,
`laplacian_iu_path(F, (0.1,), graph=graph)` with `IU_FIT_DEFAULTS` (two-component L2 IU-PCR, no exclusion).
IU-PCR is the same call at lambda 0. Fits are pooled over all steps of one PRM cell (transductive), like L-SML.
Score orientation: telemetry-only arms are anchored to the channel mean (as TELE_N16); arms that contain the PRM
are anchored to `zp` (as CHAIN_N16 / PRMG_N16). Every flip is recorded.

## Arms

Label-free (fit once, all steps):
- `IU_N16`, `DUFSLIU_N16`: IU-PCR and DUFS-LIU on X16.
- `IU_PRM16`, `DUFSLIU_PRM16`: IU-PCR and DUFS-LIU on X17. **`DUFSLIU_PRM16` is the primary replacement arm.**

PRM pseudo-label (no PRMBench label is read; fit once, all steps):
- pseudo-label `yp` = the PRM's own GAP flags: `topk_flags(zp, off, gap_counts(zp, off))` (the adopted count rule,
  answer-local, applied to the PRM score).
- `PALIU_PRM16` (**primary pseudo-label arm**): TA-LIU with `yp` in place of labels:
  gates `q_i = max(r_i, 0) / RMS(max(r, 0))`, `r_i = Corr(X17_i, yp)` over all steps
  (`target_anchored_laplacian.pseudo_anchor_laplacian_fit`), graph k=7, lambda 0.1, IU-PCR rho and subspace unchanged.
- `PRHOLIU_PRM16` (secondary): DUFS-LIU with IU-PCR's estimated `rho` replaced by `rho_i = Cov(X17_i, yp)`
  (DUFS graph unchanged, lambda 0.1). Known caveat, fixed now: the `zp` entry is self-referential (yp is a function
  of zp), so its rho is inflated.
- `PRHOPCR_PRM16` (secondary): the same target at lambda 0 (no Laplacian), to separate target from graph.

Labels (cross-fitted; only calibration labels enter any fit):
- Folds: 5 folds by source group (unique `source_group` values of all answers, `default_rng(20261021).permutation`,
  fold = position mod 5, the collection's ARES fold rule). Every answer is scored by the fit that held out its fold.
- Label budgets: k = 64 and k = 512 calibration answers sampled from the training folds
  (`default_rng([20261024, k, draw, fold])`, 5 draws each), and FULL = every training-fold answer (1 draw).
  All steps of a calibration answer are used.
- `TALIU_PRM16@k` (**primary label arm at k = 512**): TA-LIU, gates from `Corr(X17_i, y)` on calibration steps
  (`target_anchored_laplacian.target_anchored_laplacian_fit`), lambda 0.1.
- `RHOLIU_PRM16@k` (secondary): DUFS-LIU with `rho_i = Cov(X17_i, y)` on calibration steps (DUFS graph unchanged).
- `RHOPCR_PRM16@k` (secondary): the same target at lambda 0.
- `LR_PRM16@k` (attribution control, same labels): `LogisticRegression(C=1, class_weight='balanced', lbfgs,
  max_iter=1000)` on X17, decision function.

References (exported, unchanged): CHAIN_N16 (current method), PRMG_N16, TELE_N16, EQ_N16, MEAN_N16, the PRM under
the same rule (better scale under GAP, a label-chosen reading as in the collection), LSML_CONT (different,
label-selected channels; reference only: its third channel is not in the local exports, so no like-for-like
DUFS-LIU version of it is possible here).

## Metrics

- Within-answer AUC (`evaluate_collection.within`) and official PRMScore under GAP (`measure(..., 'GAP')`);
  Q80 reported as secondary. Paired bootstrap by source group, 2,000 draws, `boot_weights` (seed 20261013),
  one weight matrix per PRM shared by every contrast.
- Multi-draw arms (k = 64, 512): point = mean of the per-draw deltas; CI = percentiles of the per-replicate mean
  over draws (same bootstrap weights for every draw).

## Manifold diagnostics (the identification question, measured directly)

- D1 nuisance alignment: pooled Pearson correlation of answer-z(score) with answer-z(step position) and with
  answer-z(log step tokens).
- D2 content-only ranking: within-AUC after removing position and log length from answer-z(score) (pooled OLS).
- D3 gate profiles: DUFS, PRM-pseudo-label and label (FULL, fold mean) gates per channel, next to each channel's
  |corr| with position and length (label-free) and its label point-biserial correlation (diagnostic only).
- D4 weight geometry: cosine of each arm's weights with (a) the supervised direction (LR on X17 with all labels,
  in-sample, diagnostic only) and (b) the nuisance direction (OLS of answer-z position and of answer-z log length
  on X17).

## Success rules (fixed now)

A. **Replacement** (primary `DUFSLIU_PRM16` vs CHAIN_N16; primary metric GAP PRMScore):
- "can replace" (non-inferior): on all 5 PRMs the PRMScore CI lower bound > -0.5 AND the within-AUC CI lower
  bound > -0.005;
- "replaces and improves": non-inferior AND PRMScore CI > 0 on at least 3 of 5 PRMs;
- otherwise "cannot replace".

B. **Label calibration** (primary `TALIU_PRM16@512`; primary metric within-AUC):
- "labels help identification": `TALIU@512 - DUFSLIU_PRM16` CI > 0 on at least 3 of 5 PRMs;
- "efficient use of labels": `TALIU@512 - LR@512` CI lower bound > -0.005 on at least 3 of 5 PRMs.
The same two tests are reported for `RHOLIU@512` (secondary; does not change the verdict).

C. **PRM pseudo-label** (primary `PALIU_PRM16`; primary metric within-AUC):
- "pseudo-label helps": `PALIU - DUFSLIU_PRM16` within-AUC CI > 0 on at least 3 of 5 PRMs AND GAP PRMScore point
  >= 0 on at least 3 of 5;
- replacement test of rule A applied to `PALIU_PRM16` vs CHAIN_N16.
`PRHOLIU` / `PRHOPCR` are secondary.

## Parity gates (must pass before any new result is read)

- The evaluator reproduces `tables/STATUS_MAIN.csv` within-AUC and GAP PRMScore for CHAIN_N16, TELE_N16, EQ_N16,
  PRM_Q80, PRM_Q80_PROB on all 5 PRMs (|diff| < 1e-9).
- The external-rho LIU solve with IU-PCR's own `rho_hat` reproduces `laplacian_iu_path` weights (unit test).
- `pseudo_anchor_laplacian_fit` / `target_anchored_laplacian_fit` are called unchanged.

## Caveats fixed in advance

- PRMBench is development data; a pass is development evidence, not confirmation.
- The label arms use labels; they answer "how much do labels help", they are not label-free methods.
- The pseudo-label arms read no PRMBench label, but the PRM was trained by its authors with labels on other data.
- The nuisance diagnostics name position and length only; other nuisances are not measured.
