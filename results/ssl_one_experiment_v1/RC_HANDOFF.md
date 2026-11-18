# RC HANDOFF: run the one SSL experiment on AIRCC (cycle4 spot), 2026-11-18

Who: the Claude Code session on Omri's PC (Remote Control). Requested by the cloud thread "Unify post-meeting research into one plan".
Omri's request (2026-11-18 16:24 Israel time): run our experiment on AIRCC through cycle4, updated by the Mac's corrections, and upload the findings to Drive.
Design and decisions: ANSWERS_TO_MAC_2026-11-18.md (same Drive folder). This file is the run procedure only.

## Rules
- Never switch branches in C:\Users\omris\TAU\hallucination_detection. Use a new worktree.
- Never stop or cancel processes or jobs that you did not start. Delete nothing.
- Push only the new experiment branch, with `git push --no-verify` (LFS budget). No merge. Never commit pkl files.
- Do not change the analysis decisions. If a step fails, stop that step and report what failed, with the output.
- Report to the thread in Hebrew, tables + bullets.

## Files from the cloud (Drive)
gdrive:hallucination_detection/cluster_results/ssl_one_experiment_v1/code_from_cloud/
- make_variants.py   split by corrected question groups + V1/V2 inputs (answers 1-10)
- extract_steps.py   per-run step table: sorted top-48 token values of the 11 token channels + bank step channels (gate: top10 == bank)
- smoke_check.py     12-row checks: manifest, equal prefix (stop rule), C vs stored top10 (Spearman)
- RC_HANDOFF.md      this file
- ANSWERS_TO_MAC_2026-11-18.md

## A. Branch (PC)
1. `git fetch origin` (GIT_LFS_SKIP_SMUDGE=1). New worktree, new branch `claude/ssl-one-experiment-2026-11-18` from `origin/claude/simes-answer-gate-2026-11-15` (43b2886e). Do NOT base it on claude/shuffle-gate-design-2026-11-16 (its commit deletes most of the tree).
2. Copy in: `git checkout origin/claude/shuffle-gate-design-2026-11-16 -- results/shuffle_gate_design_v1`; `git checkout origin/claude/pb-first-error-rule-2026-11-10 -- results/pb_first_error_rule_v1`; `git checkout origin/claude/sml-readout-calibration-2026-11-12 -- results/sml_readout_calibration_v1`; `git checkout origin/claude/sa-readouts-2026-11-12 -- results/bench_gap_diagnosis_v1/readout_alternatives data_export/readout_steps_sa_v1`.
3. Put the 3 .py files + the 2 .md files in `results/ssl_one_experiment_v1/`. Add your AIRCC job scripts there (section C). Commit, push --no-verify. Report the commit hash.

## B. AIRCC setup
1. `ssh aircc`. Paths as in results/prm_standalone_telemetry_v1/aircc/common.sh: S=/shared/cycle2_tau_averbuch_prj/omrisegev1, R=$S/results, H=$S/hf_cache/hub, pyenv `$S/pyenv_native_312_20261011/activate.sh`. Check that $S is still writable; if not, use the cycle4 user dir (`/shared/cycle4_..._spot_prj/users/omrisegev1`) and say so.
2. W=$S/results/ssl_one_experiment_v1 (new; refuse to overwrite if it exists). Frozen code dir: a checkout of the pushed branch commit at $S/code_ssl_one_v1 (GIT_LFS_SKIP_SMUDGE=1), as the earlier frozen CODE_DIRs. Set HD_ROOT=$S/code_ssl_one_v1 for the 3 scripts.
3. Model: the cached snapshot of Qwen/Qwen2.5-Math-7B-Instruct under $H (the same one the stored standalone runs used; report the snapshot hash; expected ef9926d7...). No new download unless it is missing.
4. Account: the cycle4 spot account (it accepted a 1-GPU `--test-only` on 2026-11-18 16:00). Partition power-gpu (or compute-gpu if it is faster). cycle2/cycle3 accounts: do not use.

## C. Inputs (login node, pyenv; light CPU, about 1 min)
`python3 make_variants.py --pb-answers results/prm_boost_processbench_v1/inputs/answers.json --pb-gold results/prm_boost_processbench_v1/local/processbench_gold.json --prmb-meta $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl --folds results/localization_source_group_audit_v1/FOLDS_V2.json --out $W/inputs`
- Must print `GROUP GATE ... "status": "PASS"` (707 PRMBench groups, every id in FOLDS_V2.json). If FAIL: stop and report the line.
- If the login node refuses the memory, run it as the first step of a GPU job instead.
- Report VARIANTS_REPORT.json (rows, skips, relax levels) and the SPLIT counts.

## D. GPU jobs: 6 independent jobs, one per input, each with its own preflight (one queue wait)
Inputs (all `--mode instruct --model <snapshot> --prompt-style qwen_cot_chat --max-batch 1`):
| Job | Driver input |
|---|---|
| processbench_C | `--answers results/prm_boost_processbench_v1/inputs/answers.json` |
| processbench_V1 / _V2 | `--answers $W/inputs/pb_V1_answers.json` / `pb_V2_answers.json` |
| prmbench_C | `--meta-pkl $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl` |
| prmbench_V1 / _V2 | `--meta-pkl $W/inputs/prmbench_V1_meta.pkl` / `prmbench_V2_meta.pkl` |
Driver = cluster/run_prmbench_prm_telemetry.py of the frozen code dir, through the existing cluster/submit_prm_telemetry_offline.sbatch pattern (see prm_standalone_telemetry_v1/aircc/submit_gpu.sh). Batch 1 for all 6 (answer 14).

Each job, in order:
1. Preflight (12 rows, `--n-samples 12`, into $W/pre/<bench>_<V>/): run the benchmark's C input AND this job's input (for a C job, only C). Then `smoke_check.py --runs $W/pre/<bench>_<V> --variants $W/inputs --stored-readouts $HD_ROOT/data_export/readout_steps_sa_v1 --out $W/pre/<bench>_<V>/SMOKE.json`. Exit code != 0 -> stop the job (no full run).
   The 12 V rows are the children of the first 12 C rows, so the equal-prefix check has its pairs.
2. Full run into $W/runs/<bench>_<V>/ (resumes from --out; on exit 85 or preemption, requeue with the same arguments).
3. Bank, with the SAME frozen builder as the stored standalone bank of that benchmark:
   - PRMBench: as results/prm_standalone_telemetry_v1/aircc/downstream.sbatch: build_bank.py with `--base $R/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz --mask-code $S/code_prm_boost_v1/mask_fixed --tokenizer <7B-Instruct snapshot>/tokenizer.json`, run from $S/code_thinkfix_bank_553de79.
   - ProcessBench: the 2-step build of the stored standalone PB bank: graft_rewards.py then prm_boost_build_bank_external_v2.py (Drive prm_boost_crop_ares_benchmarks_v1/code/). The builder uses rewards only for REWARDS.npz, so graft dummy (zero) rewards; FEATURES must not depend on them (check on the C preflight rows: FEATURES with zero rewards == FEATURES with the PRM rewards).
   Output $W/banks/<bench>_<V>/FEATURES.npz.
4. `extract_steps.py $W/runs/<bench>_<V>/<pkl> $W/banks/<bench>_<V>/FEATURES.npz $W/steps/STEPS_<bench>_<V>.npz --workers <cpus>`. Must print `GATE PASS`.
5. Request about 6 h wall, 16 CPUs, 96 GB RAM (PRMBench bank build). Write a short JOB_<bench>_<V>.json (job id, node, GPU type, s/row, peak GPU memory, wall time, exit codes).

Expected time on one A100/B200: PB about 1 h per input, PRMBench about 2 h per input (batch 1), plus about 30 min CPU.

## E. When you have submitted the jobs
Report to the thread: branch + commit, AIRCC paths, snapshot hash, account/partition, the 6 job ids, the Slurm start estimates, the make_variants gate and counts. Then end the session. Do not wait for the jobs.

## F. Later (a new session will ask for this): collect
1. For each job: JOB_*.json, SMOKE.json, manifest.json, extract GATE.
2. Parity of run C vs the stored standalone banks (report, no stop): Spearman per channel of the 16 TELE16 raw columns, our C bank vs $R/prm_standalone_telemetry_v1/bank_qwen_prm_7b/FEATURES.npz (PRMBench) and vs the stored PB standalone bank (hd_collection_data/c4u_r1_inputs/crop_ares_all/full_sa_qwen_prm_processbench/bank_v3/FEATURES.npz on the PC).
3. `rclone copy $W gdrive:hallucination_detection/cluster_results/ssl_one_experiment_v1/aircc/` (all, including raw pkls), then `rclone check`.
4. Copy to the PC folder C:\Users\omris\TAU\hd_collection_data\ssl_one_experiment_v1\ only: steps/*.npz, inputs/ (without the large pkl/json inputs), pre/*/SMOKE.json, runs/*/manifest.json, JOB_*.json, parity json. Check free disk first (need about 1 GB).
