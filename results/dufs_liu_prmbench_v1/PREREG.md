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

## Amendment 1 (2026-10-24, written before any 56-channel score exists): the 56-channel bank

Omri asked the same day to test the method with more than 16 channels, "maybe the 56". Scope chosen by Omri: the
local Qwen3-8B telemetry, where the supervised gap is largest (thinkfix_mask1 within-AUC: N56 continuous L-SML
.7705, IU-PCR .7640, versus N16 .7792 / .7805; supervised LR about .82 on N56, Step 464). N56 on the five PRM
backbones is not built here (it needs their FEATURES files from AIRCC).

Inputs and construction:
- `results/advisor_completion_2026-10-11/inputs/features_thinkfix_mask1.npz` of the main checkout (sha256 recorded
  in the outputs), cell `prmbench_qwen3_8b` (6,969 answers), joined to the PRMBench evaluator by row id.
- Banks N16 and N56 from `results/thinkfix_rescore_v1/thinkfix_mask1/nested_banks/BANKS.json`, built exactly as
  `monday_external_panel_v1/score.py:build_bank` (answer standardization; the three digit channels standardized
  over their active steps). These banks are not sign-oriented per channel.
- PRM for the fused arms: Qwen2.5-Math-PRM-7B, `zp` = answer-z of `score__PRM_Q80` from `ABL5_SCORES_qwen_prm_7b`.

Parity gates (before any new 56-channel number is read):
- Canonical `fit_iu` and continuous L-SML on the rebuilt banks reproduce the published thinkfix_mask1
  `cont_baselines/ABSOLUTE.csv` within-AUC of IU_PCR and LSML_CONT on N16 and N56 (|diff| < 1e-9).
- The re-implemented CHAIN construction (ablation5: TELE L-SML virtual classifiers, answer-z, cross L-SML with the
  PRM as its own group, anchored to `zp`) reproduces the exported CHAIN_N16 of the PRM-backbone Qwen-7B cell
  (max |diff| < 1e-6).

Orientation: for every IU-family arm in this amendment, each channel is first oriented without labels by the sign of
its pooled correlation with the line's level score (mean answer-z of q15_H1, q15_VE1, logprob_margin, true_tail50,
energy_level), the orientation of ALL_EQ_ORIENTED. The canonical unoriented IU_PCR is kept as the parity row.

Arms, for B in {N16, N56}, same recipes as above:
- telemetry only: `IU_B` (oriented), `DUFSLIU_B`, `LSML_B` (continuous L-SML, published recipe), `EQ_B` (oriented
  equal mean, control); labels: `TALIU_B`, `RHOLIU_B`, `RHOPCR_B`, `LR_B` at k = 64, 512, FULL.
- PRM + telemetry: `IU_PRM_B`, `DUFSLIU_PRM_B`, `CHAIN_B` (L-SML reference), `EQ_PRM_B` (control); PRM pseudo-label:
  `PALIU_PRM_B`, `PRHOLIU_PRM_B`; labels: `TALIU_PRM_B`, `RHOLIU_PRM_B`, `RHOPCR_PRM_B`, `LR_PRM_B` at k = 64, 512, FULL.

Questions and rules (one PRM, so "3 of 5" becomes the single paired CI):
- A56: `DUFSLIU_PRM_N56 - CHAIN_N56` and `DUFSLIU_N56 - LSML_N56`, with rule A's margins (PRMScore lower bound
  > -0.5 and within-AUC lower bound > -0.005 = can replace).
- B56: `TALIU@512 - DUFSLIU` CI > 0 = labels help; `TALIU@512 - LR@512` lower bound > -0.005 = efficient; for both
  settings at N56. RHOLIU secondary.
- C56: `PALIU_PRM_N56 - DUFSLIU_PRM_N56` within-AUC CI > 0 = pseudo-label helps.
- D56 (more channels): the N56 - N16 difference of each arm, reported with its paired CI; the question is whether the
  label-free arms lose and the label arms gain, as in Step 464.

## Amendment 2 (2026-10-25, written before any score of these arms): the PRM defines the IU-PCR subspace

Request (Omri, 2026-10-25): "Test one label-free variant where the PRM defines the IU-PCR subspace, pre-registered
the same way, against CHAIN_N16 on the 5 PRMs."

The subspace is `U = orthonormal basis of span{e_PRM, (0, c)}`, with `c_j = Cov(zp, x_j)` over all steps on the
z-scored X17. In words: the PRM's own axis, plus the telemetry direction that agrees most with the PRM (the first
partial-least-squares direction of the telemetry, with the PRM as the response). No label enters.

Arms (PRM-backbone N16, the same 5 PRMs, inputs and evaluator as above):
- **`PPLIU_PRM16` (primary).** Registered DUFS-LIU with only its two-component subspace replaced by `U`:
  `w = U [U^T (C + 0.1 R_bar) U]^-1 U^T rho_hat`. It keeps IU-PCR's own `rho_hat` and the registered DUFS graph of
  X17 (the graph of DUFSLIU_PRM16), with `R_bar` trace-matched inside `U`. Anchored to `zp`.
- **`PPREF_PRM16` (secondary diagnostic; it does not enter the verdict).** The same plane and graph, with the PRM as
  the reference rater of the additive model (`a_PRM = 0`): `rho_j = Cov(x_j, zp)` for every channel, and
  `rho_PRM = Var(Y)`, estimated by the PRM's squared multiple correlation with the 16 channels (the classic
  communality lower bound).

Label-free facts computed before writing this (scratch computation on the unlabeled matrices, no label read):
- IU-PCR's estimate of `Var(Y)` (`g2_hat`) is 0.175 / 0.217 / 0.176 / 0.181 / 0.249 (Qwen-7B, Qwen-72B, PURE,
  RLHFlow, Math-Shepherd). It is capped at 0.25 by `IU_FIT_DEFAULTS` (scale_ratio 0.25).
- The PRM's agreement with the telemetry alone implies a reliability of at least
  `R^2(zp ~ PRM-aligned telemetry composite)` = 0.327 / 0.279 / 0.254 / 0.236 / 0.112. So on 4 of 5 PRMs the IU
  model trusts the PRM less than that agreement allows. IU-PCR's `rho_hat` ranks the PRM 11th to 17th of 17 views.
- With IU-PCR's `rho_hat`, the PRM's weight in this plane is negative on 5 of 5 PRMs: -0.024 / -0.057 / -0.024 /
  -0.025 / -0.051 (at lambda 0).
- With the PRM as reference (the secondary arm), the PRM weight is positive on 5 of 5 (+0.044 / +0.080 / +0.062 /
  +0.120 / +0.195; share 0.05-0.28), and the score correlates 0.88-1.00 with the PRM-aligned telemetry composite.

Pre-registered predictions:
- the primary arm cannot replace CHAIN_N16 (rule A) and does not beat DUFSLIU_PRM16;
- the secondary arm behaves like the PRM-aligned telemetry composite.

Rules:
- **Replacement:** rule A, with the same margins, for `PPLIU_PRM16 - CHAIN_N16`.
- **"The PRM subspace helps DUFS-LIU":** `PPLIU_PRM16 - DUFSLIU_PRM16` within-AUC CI > 0 on at least 3 of 5 PRMs.
- The secondary arm is reported with the same contrasts, without a verdict.

Implementation and parity:
- `liu_external_target` gains an optional `subspace`, unit-tested: the top-2 eigenvectors reproduce the default
  solve.
- The DUFS graph is rebuilt with the registered seeds through the split call. It must reproduce the saved
  DUFSLIU_PRM16 scores of all 5 PRMs (max |diff| < 1e-8) before any new score is read.

## Caveats fixed in advance

- PRMBench is development data; a pass is development evidence, not confirmation.
- The label arms use labels; they answer "how much do labels help", they are not label-free methods.
- The pseudo-label arms read no PRMBench label, but the PRM was trained by its authors with labels on other data.
- The nuisance diagnostics name position and length only; other nuisances are not measured.
