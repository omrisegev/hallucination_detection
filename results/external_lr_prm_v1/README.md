# external_lr_prm_v1: supervised comparators on Socratic-PRMBench and Hard2Verify

Requested 2026-10-12 for advisor page section 5 ("Does it transfer outside PRMBench?").
Base commit of the checkout: `cbb498ef4` (branch `codex/lsml-external-generalization-v1`).
Nothing here is committed or pushed yet. Files are uncommitted additions.

## Status
- **Step 1, LR transfer: DONE** (CPU, local, 51 s). Results are in `RESULTS.csv`, with detail in `LR_TRANSFER.csv` and `LR_TRANSFER_LOG.json`.
- **Step 2, PRM: DONE** (see below; submitted on Omri's go-ahead, 2026-10-12, which waived the local-disk preflight row). Original note: AIRCC is reachable and the PRM checkpoint is cached there. The driver (`cluster/run_external_prm.py`) and the sbatch file (`cluster/submit_external_prm.sbatch`) are written. The mandatory `/preflight` gate failed at its first rows:
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

## Step 2 results: Qwen2.5-Math-PRM-7B (supervised PRM)
AIRCC jobs 282806 (Socratic, 6.4 min) and 282807 (Hard2Verify, 4.8 min) both COMPLETED. Code snapshot: commit d6c449a4f, copied to `code_external_prm_v1`. Raw rewards: `prm_raw/<bench>/external_prm.pkl`. Evaluator: `prm_evaluate.py`; its rows are in `PRM_EXTERNAL.csv` and were merged into `RESULTS.csv`.

| benchmark | rule | official metric | within-answer AUC |
|---|---|---|---|
| Socratic | NATIVE_0.5 | 0.680 | 0.824 |
| Socratic | ANSWER_Z_Q80 | 0.677 | 0.824 |
| Hard2Verify | NATIVE_0.5 | 0.234 | 0.634 |
| Hard2Verify | ANSWER_Z_Q80 | 0.391 | 0.634 |

- **Input format.** Same answers.json as our telemetry (SHA-256 identical), using the model card's template with one `<extra_0>` after each step.
- **Rules.** NATIVE_0.5: reward >= 0.5 means valid, on every step. ANSWER_Z_Q80 is applied to 1 - reward over the non-empty steps; empty steps are predicted incorrect, exactly as for our methods.
- **Long inputs.** 19 of the 200 Hard2Verify inputs exceed 4,096 tokens (max 11,719). Their within-AUC is not lower: 0.636 on 12 defined answers, vs 0.633 on 136 shorter ones.
- **Hard2Verify at 0.5.** The PRM's native rule flags very few errors there, which gives Balanced F1 0.234.
- **No confidence intervals.**
