# external_lr_prm_v1: supervised comparators on Socratic-PRMBench and Hard2Verify

Requested 2026-10-12 for advisor page section 5 ("Does it transfer outside PRMBench?").
Base commit of the checkout: `cbb498ef4` (branch `codex/lsml-external-generalization-v1`).
Nothing here is committed or pushed yet. Files are uncommitted additions.

## Status
- **Step 1, LR transfer: DONE** (CPU, local, 51 s). Results are in `RESULTS.csv`, with detail in `LR_TRANSFER.csv` and `LR_TRANSFER_LOG.json`.
- **Step 2, Qwen2.5-Math-PRM-7B on the externals: NOT RUN.** AIRCC is reachable and the PRM checkpoint is cached there. The driver (`cluster/run_external_prm.py`) and the sbatch file (`cluster/submit_external_prm.sbatch`) are written. The mandatory `/preflight` gate failed at its first rows:
  - the local C: drive has 4.1 GB free (the gate needs 10 GB);
  - the main checkout is dirty (517 porcelain lines), and a dirty tree is only allowed for a smoke or pilot run.
  
  Nothing was submitted.

## Step 1 recipe
This mirrors `results/advisor_completion_2026-10-11/lr` on branch `codex/advisor-v11-completion` (`spectral_utils/advisor_lr.py`).
- **Features.** Bank N16, 16 columns, per-answer z-score. The 3 digit channels are z-scored over their active positions only. This is the `prepare_population.py` adapter.
- **Telemetry.** think-fix mask1:
  - PRMBench source banks: `results/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz` (8B) and `results/thinkfix_rescore_4b_v1/source_bank_thinkfix_mask1_4b/FEATURES.npz` (4B), on AIRCC.
  - External banks: `external_banks_v4/{socratic,hard2verify}_qwen3_8b/FEATURES.npz` from the AIRCC chain copies `thinkfix_ext_thinkfix_mask1` (8B) and `thinkfix_ext_thinkfix_mask1_4b` (4B). The local copies, with SHA-256 hashes in the log, are in `scratch/external_lr_prm_v1/inputs/`.
- **Model.** StandardScaler + LogisticRegression(class_weight='balanced', lbfgs). C was chosen from {1e-3, ..., 1e3} by mean class-balanced log-loss over the 5 PRMBench FOLDS_V2 source-group folds, then the model was refit on all training steps. No target labels were used.
- **Training data.** Primary: PRMBench noncontrol answers (6,211 answers, `classification != correct`). Sensitivity: all 6,969 answers, the v11 population. The two give the same results to about 0.002.
- **Chosen C.** 8B: 10 (noncontrol) / 100 (all). 4B: 100 / 1000.
- **Decisions.** ANSWER_Z_Q80, the formula of `monday_cont_baselines_v1/score_ext.py`. Evaluation uses the same frozen external evaluator code path as `evaluate_ext.py`: Socratic is scored with PRMScore and Hard2Verify with Balanced F1. Within-answer AUC is the mean over answers.

## Checks (all pass)
- The PRMBench 8B adapter reproduces the frozen `prmscore_inputs/bank_N16.npz` ALL_EQ score to within 1.3e-15.
- The external ALL_EQ / N16 / ANSWER_Z_Q80 rows of `qwen3_4b_vs_8b__external_nested.csv` are reproduced exactly (difference below 1e-9) for all four cells: {Socratic, Hard2Verify} x {8B, 4B}.

## Caveats
- Within-answer AUC is not threshold-dependent. The LR gain on within-AUC (about +0.06 Socratic, +0.08 Hard2Verify over IU-PCR) is larger than its gain on the official metric (about +0.02 to +0.03).
- Hard2Verify has 200 answers. No confidence intervals were computed.
- External labels had already been opened in earlier rounds, so this is exploratory transfer evidence, not an untouched test.
- The step-index control on Hard2Verify has a within-AUC of 0.865 (in `external_nested.csv`). Position alone is a strong Hard2Verify baseline.

## Step 2 plan (ready, not run)
- One GPU job per benchmark on account `cycle3_tau_averbuch_prj`, QOS `owner_940`, partition `power-gpu`.
- Input: `$S/data/lsml_external_20260924/{socratic,hard2verify}/answers.json`. These have the same SHA-256 hashes as the local inputs, so the uids and `steps` lists are the ones our telemetry was scored on.
- Scoring: the same `spectral_utils/prm_scorer.score_steps` as `cluster/run_prmbench_prm.py`, with one `<extra_0>` after each step.
- Offline evaluation, two rules:
  - the native rule, reward >= 0.5 means the step is valid;
  - ANSWER_Z_Q80 on the risk (1 - reward).
- Rows over 4,096 input tokens are counted in the job manifest. Long Hard2Verify proofs may exceed the PRM's context.
