# Line status: claude/decision-rule-v1

Status: PAUSED-FOR-DECISION

Owner / last updated: Claude session 41041b9e (Steps 453-454, 459-460), 2026-09-30

## Steps on this branch

Steps 453 and 454 were committed on `claude/ssl-pseudolabel-residual-v1` (they are ancestors of this branch); 459 and 460 are on this branch only.

- Step 453 [Claude, calfix of six older runners] - The evaluation/calibration score overwrite was fixed in six runners (bank20, indbank, declared_joint, error_cluster, core_virtual, partition_ceiling) and all were re-scored; no cited conclusion changed (largest move: err50 L-SML within-AUC 0.5733 -> 0.5902). Source: `results/*/run_20260927_calfix/CALFIX_VERIFY.json` and `BEFORE_AFTER.csv` (on the ssl branch).
- Step 454 [Claude, PRMScore decomposition] - On frozen predictions, the DS filter's PRMScore gain (0.0038) and the fused row's tie with realized_drv (0.6565 vs 0.6590) are mostly position structure, and most of the matched-rule gap to Qwen PRM is how many steps are flagged per answer. Source: `results/expectation_realization_v1/prmscore_decomposition_v1/P1_OVERALL.json`, `RED_TEAM.md`.
- Step 459 [Claude, decision_rule_v1] - Letting the flag count vary per answer from raw channel levels raises PRMScore 0.6565 -> 0.6635 (Holm interval [0.0012, 0.0132]); the threshold chosen from Dawid-Skene sensitivity/specificity loses because the estimates are biased. Source: `results/decision_rule_v1/run_20260929/METRICS.json`, `HOLM_PRIMARY_FAMILY.json`, `RED_TEAM.md`.
- Step 460 [Claude, answer_gate_v1] - The project's answer-level detectors (U-PCR full pool, original L-SML GOOD_5) detect erroneous ProcessBench answers (U-PCR 0.773 AUROC; its edge over mean entropy is answer length); used as the flag decision they lower PRMScore and lift ProcessBench official F1 mostly generically (random gate 0.272, pure gate 0.362); PRMBench has a fixed-error-count construction prior (flagging the top-2 steps per answer gives PRMScore 0.6808, post hoc). Source: `results/answer_gate_v1/run_20260930/METRICS.json`, `ANSWER_DETECTION.json`, `RED_TEAM.md`.

## Paused for decision

Questions for Omri:

1. Does a gate enter the frozen candidate for the external benchmarks (Hard2Verify, Socratic-PRMBench), and which one?
   - (a) no gate: keep the frozen answer-z q80 rule;
   - (b) the Step-459 count rule R2 (flag count from raw channel levels), calibrated per benchmark;
   - (c) a pure answer gate on mean token entropy (leave the least suspicious answers unflagged, frozen rule inside), which helps ProcessBench official F1 but lowers PRMScore.
   The choice must be frozen before any external result is seen.
2. Should every PRMScore report from now on include the fixed-count baseline (top-k steps per answer) and the error-count-per-answer structure, given the PRMBench construction prior found in Step 460?

What blocks further work: both questions change what is sent to the external benchmarks; no new experiment is started until they are answered.

## Outside git but needed

All `.npz` files are ignored by git (`results/**/*.npz`).

| Path | Size | Status |
|---|---:|---|
| `.worktrees/decision-rule-v1/results/decision_rule_v1/run_20260929/DECISIONS.npz` | 0.9 MB | regenerable: `scripts/experiments/decision_rule_run.py run_20260929` (about 40 s) |
| `.worktrees/decision-rule-v1/results/decision_rule_v1/run_20260929_a1/DECISIONS.npz` | 0.9 MB | regenerable: same script with `--ds-population prm` |
| `.worktrees/decision-rule-v1/results/answer_gate_v1/ANSWER_FEATURES.npz` | 2.5 MB | regenerable: `scripts/experiments/answer_features_extract.py` (about 7 min, CPU) |
| `.worktrees/decision-rule-v1/results/answer_gate_v1/run_20260930/DECISIONS.npz` | 0.6 MB | regenerable: `scripts/experiments/answer_gate_run.py run_20260930` (needs the features and the decision_rule_v1 decisions) |
| `.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz` | 47 MB | frozen stage-B scores (input to Steps 454, 459, 460); regenerable by `er_stage_b_run.py` (about 4 min); byte-identical re-run exists in `run_20260927_stage_b_thr/` |
| `.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b_thr/CHANNELS.npz` | 13 MB | regenerable by the same re-run |
| `.worktrees/ssl-pseudolabel-residual-v1/results/*/run_20260927_calfix/` (six folders, STEP_SCORES and CAL_SCORES npz) | 24-43 MB each | regenerable by the fixed runners (Step 453); partition ceiling takes about 45 min |
| `.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz` | 294 MB | upstream input (token telemetry for all 13,769 answers); not produced on this line; not archived by this line |
| `.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz` | 19 MB | upstream input; not produced on this line |

Nothing on this line is pushed. Not running: no background job, no AIRCC job.
