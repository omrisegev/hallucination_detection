# Pre-registration: label-free clean-answer detection from raw (not answer-standardized) continuous scores (CPU, 2026-10-29)

Written and committed before the first run. Development data; labels only score.
Branch: `claude/answer-mixture-gate-2026-10-29`. Results directory: `results/answer_mixture_gate_v1/`.

## Question

Without labels, can a two-component mixture fitted to answer-level **raw** continuous scores do two things:
1. estimate the share of answers that contain an error;
2. mark clean answers (k = 0)?

**Why raw.** The DS-gate experiments on branch `claude/ds-gate-2026-10-29` (E1, E1 fixed, E1b) failed for a
structural reason. GAP votes and per-answer z-scores remove each answer's own level, so a clean answer looks like an
erroneous one. Raw scores keep that level.

**What the theory says.**
- FRAMING_THEORY P6: from second moments of binary votes, the prevalence is not identified.
- A continuous two-component mixture is identified, but only under its distributional assumption.
- The earlier third-moment estimates were 1.5–3× too high (Steps 459, 464).

Omri approved this test on 2026-10-29.

## Data and cells

Socratic is excluded: only 0.2% of its answers are clean.

**Track P: the PRM's raw score, no telemetry.** The answer-level score comes from the PRM's raw step risk,
logit(1 − reward), stored unstandardized.
- **PRMBench:** all 6,969 answers, the 758 label-known control answers included. Clean share .134. The official
  non-control pool is 97% erroneous and cannot test a clean gate.
- **Hard2Verify:** the 177-answer common set (`collection_parity.h2v_common_uids`). Clean share .23.
- **ProcessBench:** 3,400 answers (`results/prm_boost_processbench_v1/local/processbench_gold.json`). Clean share
  about .35.
- **PRMs:** qwen_prm_7b, qwen_prm_72b, pure_prm and rlhflow_prm, so 12 cells. Math-Shepherd is out of the line.

**Track T: raw telemetry, ProcessBench × qwen_prm_7b only.**
- **Source:** the raw backbone telemetry is downloaded from
  `gdrive:hallucination_detection/cluster_results/prm_boost_processbench_v1/full_qwen_prm/external_telemetry.pkl`
  (about 950 MB) to `/Users/osegev/Desktop/hd_collection_data/processbench_raw/qwen_prm/`, outside the repo.
- **Channels:** the N16 channels per step, rebuilt with `results/prm_boost_external_v1/build_bank_external.py` and
  `scripts/rebuild_source_bank_v4.py`. For Qwen the frozen digit ids 15..24 are correct.
- **Parity, required before any result is read:** TELE_N16 and CHAIN_N16 recomputed from the rebuilt bank, after
  the usual answer-z, must equal the stored `processbench_scores/SCORES_CHAIN_qwen_prm.npz`
  (`score__TELE_N16`, `score__CHAIN_N16`) to 1e-6. Otherwise stop and report.

## Answer-level score (fixed)

1. **Step scores**, every one oriented so that higher means more likely wrong:
   - **P:** the PRM's raw risk.
   - **T-MEAN:** each raw channel standardized over all steps of the cell, not per answer, then the plain mean of the
     16.
   - **T-LSML:** TELE_N16's own fitted weights (its usual answer-z fit), applied to the same globally standardized
     raw channels.
   - **T-CHAIN:** CHAIN_N16's fitted cross weights, applied to [the PRM's global-z raw risk, the T-LSML group
     scores computed the same way].
2. **Answer score.** The primary is s_a = the maximum step score in the answer. The secondary is the mean of the top
   2 steps (or the single step, for 1-step answers).
3. **Covariate adjustment.** Pooled OLS of s_a on [log number of steps, log answer length in tokens where available,
   else log number of steps only]. The residual is used. No labels.

## Mixture and readouts

**Models.**
- **Primary:** a 1-D two-component Gaussian mixture (`sklearn.mixture.GaussianMixture`, full covariance, n_init 10,
  seed 20261029) on the residual.
- **Secondary:** the same mixture after a Yeo–Johnson transform (`sklearn.preprocessing.PowerTransformer`), to
  reduce skew.

**Readouts.**
- **π̂:** the weight of the component with the higher mean, i.e. the estimated share of answers with an error.
- **Clean decision:** an answer is called clean when its posterior P(error component) < .5.
- **Mixture support:** BIC(2 components) against BIC(1).

**Synthetic checks, run and reported before the real data:**
- **(a) No true mixture.** 3,000 unimodal heavy-tailed draws (Student-t with 3 degrees of freedom; and a skewed
  log-normal). Report π̂, and whether BIC prefers 2 components. This measures how easily the method invents a
  second group.
- **(b) A true mixture.** Prevalence .65 and .20, separation d = 1 and 2, Gaussian and t₅ noise. Report the recovery
  of π̂.

## Endpoints (labels only score)

1. **|π̂ − π|** per cell, where π is the true share of answers with an error.
2. **The clean gate:**
   - the share of clean answers it closes;
   - the share of erroneous answers it keeps.
3. **Ranking:** the AUROC of s_a for "answer has an error". This shows whether the raw score orders answers at
   all, independent of the cut.
4. **Context:** the DS-gate estimates from `claude/ds-gate-2026-10-29`, and the truth.

## Pass rule (fixed now)

- **Track P passes** if |π̂ − π| ≤ .05 in at least 3 of 4 PRMs on ProcessBench **and** on Hard2Verify, with the
  primary score and model.
- **Track T passes** if |π̂ − π| ≤ .05 for at least one of T-MEAN, T-LSML and T-CHAIN, and it is closer to π than
  Track P's estimate for the same cell.
- **The gate is useful (secondary)** if it closes at least 30% of clean answers while keeping at least 85% of
  erroneous answers. This is the DS-gate D4 bar.
- **If both tracks fail,** the verdict is that label-free clean-answer detection by an answer-level mixture is not
  supported on this data. It is then recorded as a negative result with its reasons.
- **Nothing is tuned on these outputs.** Secondary variants never replace the primary.

## Verification

An independent subagent re-implements the endpoints from the raw files with its own code. It does not import this
experiment's scripts. It checks:
- that no label enters any fit;
- the populations;
- the parity claim;
- the synthetic checks.

It must reproduce π̂, the gate rates and the AUROC within 1e-6 for the deterministic parts. For the mixture fits it
must reproduce them within .005 for π̂, or explain the difference.

## Limits stated in advance

- **ProcessBench and Hard2Verify were inspected before,** and ProcessBench is out of focus for the thesis. It is used
  here only because it has many clean answers.
- **Track T covers one PRM.** The other backbones' raw telemetry would need another download, of about 2.8 GB.
- **A mixture fit is transductive per cell,** and it assumes the clean and erroneous answers form two groups.
