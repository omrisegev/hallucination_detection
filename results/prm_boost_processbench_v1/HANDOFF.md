# HANDOFF for the Windows agent: PRM boosting on ProcessBench (AIRCC)

From the Mac session on 2026-10-19. This Mac has no AIRCC access, so nothing below has been submitted.

Goal: collect PRM-backbone telemetry and run label-free PRM boosting on ProcessBench for the 5 PRMs, the same way the Socratic run did (`../prm_boost_external_v1/README.md`). Then evaluate locally with ProcessBench's official metric plus the PRISM and CROP views. Background and metric definitions are in [README.md](README.md).

## Rules (CLAUDE.md, apply to every step)

- **Allowed:** teacher-forced PRM scoring of the existing ProcessBench answers.
- **Not allowed:** training, answer regeneration, or truncation.
- **Labels never go to AIRCC.** `local/` is never copied there, and `ship.sh` sends only `inputs/` and `aircc/`.
- **No sbatch without a `/preflight` PASS** from the same session.
- **Timing first:** at most 12 examples and at most 1 GPU-hour per PRM. Then **stop and report to Omri**. The full run needs his budget decision.
- **Chains are linear 2-link chains** (`afterany`, never a fan-out); `submit_full_chains.sh` builds them that way. A clean `exit 85` is not requeued, which is why link 2 exists.
- **Cluster polling** goes through `/aircc-status` or the `cluster-ops` sub-agent, never raw ssh loops.
- **Label digit channels.** The N16 arms (HALF_N16, LSML_N16_PRMG, *_N16) contain 3 digit channels, and every table must say so. N13 is the digit-free bank. LSML_CONT uses 3 PRMBench label-chosen channels.

## Step 0: get the code

```bash
git fetch origin +refs/heads/claude/integration-2026-10-17:refs/remotes/origin/claude/integration-2026-10-17
```

Check out or rebase to that tip. Confirm `results/prm_boost_processbench_v1/` exists.

Run the local tests (CPU, seconds):

```bash
python -m unittest tests.test_prm_boost_processbench_metrics
```

Expect `Ran 11 tests ... OK`.

## Step 1: ship inputs and code (from the repo root, Git Bash)

```bash
bash results/prm_boost_processbench_v1/ship.sh aircc
```

This does the following:
- checks the local sha256 of `answers.json` and `answers_timing12.json` against `inputs/INPUT_MANIFEST.json` (`1f0acbb3…`, `4a1d1ff3…`);
- refuses if `answers.json` contains a label field;
- copies `aircc/*` plus the two frozen downstream files (as `*.repo.py`) to `$SHARED/results/prm_boost_processbench_v1/code/`;
- copies the inputs to `$SHARED/data/prm_boost_processbench_v1/`;
- prints the remote sha256. Compare it with the local lines.

Do not re-ship while a `pb_down` job is running.

## Step 2: confirm the telemetry code dir is the Socratic one

The GPU jobs run from `$SHARED/code_integration_7534b8bf4`, the same dir as Socratic. These files are byte-identical between commit 7534b8bf4 and the current tip. Verify the remote copies against your checkout:

```bash
for f in cluster/run_prmbench_prm_telemetry.py cluster/submit_prm_telemetry_offline.sbatch cluster/backfill_views.py cluster/run_teacher_forced.py cluster/install_offline_tf_thinkfix.sh spectral_utils/prm_adapters.py spectral_utils/prm_scorer.py spectral_utils/prmbench.py; do echo "$(sha256sum $f | cut -c1-16) $(ssh aircc sha256sum /shared/cycle2_tau_averbuch_prj/omrisegev1/code_integration_7534b8bf4/$f | cut -c1-16) $f"; done
```

The two hashes on every line must be equal. If one differs, stop and report it; do not sync over that dir.

The downstream reuses the same dirs as Socratic: `code_thinkfix_bank_553de79`, `code_thinkfix_rescore_v1`, `eval_mirror_prm_gen_v1`, and `results/prm_boost_v1/bank_prm/FEATURES.npz`. `pb_down.sh` checks with `cmp` that the AIRCC bank builder and `score_external` copies equal the repo versions, and refuses otherwise.

## Step 3: /preflight, then the context census (CPU, no GPU)

Run `/preflight`. For this pipeline:
- **The tree:** your checkout at the pushed tip.
- **Disk:** the full run writes about 1.1 GB of telemetry per PRM, about 6 GB in total.
- **The CPU smoke:** the unit tests from step 0, plus the census below.
- **The label-sanity gate:** `answers.json` has no labels, which `ship.sh` checks.

Then:

```bash
ssh aircc 'bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/submit_census.sh'
```

The output is `census/CENSUS_SUMMARY.json`, plus one `CENSUS_<prm>.json` per PRM. Per PRM it gives the context limit, the number of answers that would overflow (by subset), step-text mismatches and build errors.
- `n_build_errors` must be 0. A non-zero count means the driver would crash on those rows: stop and report.
- Overflow rows are expected to be few or none (the longest answer is 7,947 characters). They become counted failures, not truncations.
- Check `n_rows_step_text_mismatch` for `rlhflow_prm`, which had 3 on Hard2Verify. If it is non-zero, report it with the uids before the full run.

## Step 4: timing jobs (5 × 12 answers, 1 GPU × 1 h each)

```bash
ssh aircc 'bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/submit_timing.sh'
```

The script prints `<prm> <jobid>` for each of the 5 PRMs. It uses the same sbatch and code dir as the Socratic timing (jobs 287911-287920). Outputs go to `$O/timing_<prm>/`.

Exit code 2 (`COMPLETE_WITH_FAILURES`) means counted context failures. Record it in the table; it is not a crash.

Optional pipeline smoke on the 12-row cells (CPU, minutes), after the 5 timing jobs end:

```bash
ssh aircc 'RUN=timing bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/submit_down.sh'
```

This writes `$O/scores_timing/`. It is a pipeline check only. Do not report or evaluate those numbers as results.

## Step 5: STOP and report the timing to Omri

Copy each `timing_<prm>/manifest.json` into `results/prm_boost_processbench_v1/timing_<prm>/manifest.json`. Add the table below to README.md, in the same columns as `../prm_boost_external_v1/README.md`:

| adapter | rows | elapsed s | s/row | full-run GPU-h | max input tokens | context overflow (census) | step-text mismatch | status |

How to fill the columns:
- **full-run GPU-h** = s/row × 3,400 × 1.13 / 3,600. The 1.13 factor is the mean answer length of all answers over that of the 12 timing answers (2,071 / 1,829 characters, from INPUT_MANIFEST). Add about 1.5 minutes of setup per job.
- **Proposed wall per link:** about 1.5 × the estimated hours, rounded up, with at least 1 h. Two links give twice that capacity.
- **Rough guess before timing:** 9–13 GPU-h in total, most of it qwen_prm_72b.

Commit and push the manifests, the census summary and the README table. Then wait.

Also put the README's open question 1 (the first-error rule for the fused arms) to Omri. It should be decided before results exist.

## Step 6: full run (only after Omri approves the budget)

```bash
ssh aircc 'BUDGET_OK=1 WALL_7B=<hh:mm:ss> WALL_72B=<hh:mm:ss> bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/submit_full_chains.sh'
```

For each of the 5 PRMs, the script prints `<prm> <link1> <link2>`. Link 2 is `--dependency=afterany:<link1>` with the same `--out`, so it resumes from link 1's checkpoint.

Record all 10 job ids in README.md (the same table as the Socratic "Full run" section). Then commit and push. Outputs go to `$O/full_<prm>/external_telemetry.pkl` and `manifest.json`.

## Step 7: CPU downstream (bank + all scoring arms, label-free)

**GATE: do not run this step yet.** Omri approved the pre-registration on 2026-10-19
([PREREGISTRATION_HE.md](PREREGISTRATION_HE.md)). Step 7 runs only after every code change in its section 8 is committed
and its commit hash is recorded there:
1. Mistral digit ids verified on AIRCC (`../prm_boost_ablation5_v1/HANDOFF_MS_DIGIT_FIX_2026-10-19.md`, step 2).
2. `pb_down.sh` uses the v2 bank builder with `--tokenizer`, and `lsml_continuous_guarded` in both scorers.
3. The added control arms in `score_chain_pb.py`.
4. The NC rule and the question-clustered bootstrap in `local/eval_processbench.py`, with tests.
5. Bank parity: the new builder gives the old banks exactly for Qwen, PURE and RLHFlow.

Until then, stop after step 6, including the `DEP=afterany:...` shortcut below. Steps 0–6 (GPU telemetry) do not depend on
the recipe and may run as written.

When all 10 jobs have ended (`/aircc-status`), submit:

```bash
ssh aircc 'bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/submit_down.sh'
```

Alternatively, submit right after step 6 with `DEP=afterany:<the 5 link-2 ids, colon-separated>`.

For each PRM, `pb_down.sh` does the following and stops on the first failure:
1. It checks that the manifest is `COMPLETE`, or `COMPLETE_WITH_FAILURES` with only context-overflow failures.
2. It builds `full_<prm>/bank_v2/`.
3. It writes `scores/SCORES_<prm>.npz` (frozen arms) and `scores/SCORES_CHAIN_<prm>.npz` (CHAIN/TELE/PRMG for N13 and N16).

`ONLY="qwen_prm pure_prm"` limits it to some PRMs, for example to re-run one after a fix. `bank_v2` is skipped when it already exists.

## Step 8: upload to Drive (rclone on the cluster)

```bash
ssh aircc 'bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/upload_drive.sh'
```

This copies `full_*/bank_v2/*` (FEATURES.npz, REWARDS.npz, MANIFEST.json), the telemetry manifests, `census/` and `scores/` to `gdrive:hallucination_detection/cluster_results/prm_boost_processbench_v1/`. The ~1 GB telemetry pkls stay on AIRCC. For a slow upload, run it detached with `nohup` and poll the log.

## Step 9: local evaluation (gold labels stay on this machine)

1. **Fetch the scores.** They are a few MB and git-ignored by `results/**/*.npz`:

   ```bash
   mkdir -p results/prm_boost_processbench_v1/scores
   ```
   ```bash
   scp 'aircc:/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/scores/SCORES_*' results/prm_boost_processbench_v1/scores/
   ```
   ```bash
   scp aircc:/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/full_qwen_prm/bank_v2/REWARDS.npz results/prm_boost_processbench_v1/scores/REWARDS_qwen_prm.npz
   ```

2. **Evaluate:**

   ```bash
   python results/prm_boost_processbench_v1/local/eval_processbench.py results/prm_boost_processbench_v1/scores results/prm_boost_processbench_v1/eval
   ```

   This takes about a minute. It writes `eval/RESULTS_processbench.csv` (one row per PRM × arm × rule × population), `eval/SUBSETS_processbench.csv` (per-subset official and PRISM columns), `eval/BOOTSTRAP_processbench.csv` and `eval/EVAL_MANIFEST.json`.
   - It stops if `score__PRM` or LSML_N16_PRMG differs between a PRM's two score files.

3. **Parity anchor (qwen_prm).** `PRM_native` for qwen_prm should be close to the job-177776 values: average F1 72.94; GSM8K 81.77, MATH 77.23, OlympiadBench 66.69, Omni-MATH 66.05; AUROC first_only 0.883. If the job-177776 pkls are on this machine, also run:

   ```bash
   python results/prm_boost_processbench_v1/local/parity_qwen_prm_rewards.py results/prm_boost_processbench_v1/scores/REWARDS_qwen_prm.npz --pkl-dir <dataset_cache/four_localization/pb_prm_qwen25math7b_full>
   ```

   It reports the max reward gap and how many first-error predictions changed. A gap of more than 0.5 F1 on any subset means: stop and investigate before reporting.

4. **Commit** the `eval/` CSVs, the `scores/*.json` diagnostics and a README result section. Lead with the main conclusion for Omri, followed by a table in the Socratic layout:
   - **official metric:** PRM native vs each arm;
   - **within-AUC:** Δ with its CI;
   - **pooled AUROC first_only:** beside CROP's 0.880 / 0.894;
   - the PRISM TPR/TNR from `SUBSETS_processbench.csv`.

   Mark CHAIN as a post-registration arm. Do not headline HALF_N16 (Omri, see the results map).

## What to send back

- the census summary;
- the timing table with the proposed walls (then stop);
- after approval: the job ids;
- the downstream status;
- the parity numbers;
- the commit hash with the evaluation.

If something fails, quote the log line and stop. Do not patch the frozen driver or bank code to make it pass.
