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
python -m unittest tests.test_prm_boost_processbench_metrics tests.test_prm_boost_processbench_prereg
```

Expect `Ran 32 tests ... OK` (the `PARITY y FAIL` lines it prints are an expected failure case inside a test).

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

The downstream reuses the same dirs as Socratic: `code_thinkfix_bank_553de79`, `code_thinkfix_rescore_v1`, `eval_mirror_prm_gen_v1`, and `results/prm_boost_v1/bank_prm/FEATURES.npz`. Since the section 8 changes it uses the v2 files under new names (step 7), and never overwrites the old copies.

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

## Step 7: CPU downstream (bank parity, then bank + all scoring arms, label-free)

Omri approved the pre-registration on 2026-10-19 ([PREREGISTRATION_HE.md](PREREGISTRATION_HE.md)). The section 8 code
changes are in the commit whose hash is recorded on the last line of its section 8. Status of the gate:

| # | section 8 item | status |
|---|---|---|
| 1 | Mistral digit ids | Verified in AIRCC job 288932 (`claude/prm-boosting` c2c33946, `../prm_boost_ablation5_v1/digitfix_v1/README.md`): 28734, 28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774. `bank_parity.sh` re-checks all 5 tokenizers (7b) and `pb_down.sh` checks each one before its build. PURE's tokenizer has not been resolved on AIRCC yet; the same check covers it. |
| 2 | v2 builder with `--tokenizer`, guarded L-SML in both scorers | In code: `pb_down.sh` builds `full_<prm>/bank_v3` with `prm_boost_build_bank_external_v2.py --tokenizer`, then `score_external_v2.py` and `score_chain_pb.py` (both `lsml_continuous_guarded`). |
| 3 | control arms | In code: `score_chain_pb.py` adds MEAN/HALF/EQ on N13 and N16, POS_ONLY, LEN_ONLY. |
| 4 | NC rule, clustered bootstrap, tests | In code: `local/eval_processbench.py`, `local/pb_metrics.py`, `tests/test_prm_boost_processbench_prereg.py`. |
| 5 | bank parity | **Open, AIRCC only:** step 7b. `pb_down.sh` (RUN=full) refuses to run until it passes. |

Do not run 7b or 7c before step 6 has finished (all 10 chain jobs ended), except as noted in 7b.

### 7a. Ship the new code

Re-run step 1 (`ship.sh`) from the pushed tip, with LF line endings in the checkout. New files in `$O/code/`:
`build_bank_external_v2.repo.py`, `digit_token_ids.repo.py`, `score_external_v2.repo.py`, `bank_parity.sh`,
`bank_parity.py`, `submit_parity.sh`. The old `build_bank_external.repo.py` and `score_external.repo.py` on AIRCC are
not re-sent and stay as they are.

On AIRCC the v2 files get new names. They are installed when absent; an existing copy must be byte-identical, or the script stops (nothing is overwritten):
- `$BANK_CODE/scripts/prm_boost_build_bank_external_v2.py` and `$BANK_CODE/scripts/digit_token_ids.py`. Job 288932 already installed both from the same source, so `cmp` should pass.
- `$MIRROR/prm_boost_score_external_v2.py`.

The mirror's `spectral_utils/fusion_utils.py` must contain `lsml_continuous_guarded`. Job 288932 appended it (`guarded_append.py`), and `pb_down.sh` checks the import.

Tokenizers come from `$S/hf_cache/hub` (`tok_for` in `env.sh`). Math-Shepherd uses the Mistral-7B-v0.1 `tokenizer.json`, the same tokenizer as in job 288932. If a lookup fails, export `TOK_<prm>=<path to tokenizer.json>` before the submit; the jobs use `--export=ALL`.

### 7b. Bank parity (section 8 step 5; CPU job, 12 bank builds, about 1-2 h)

```bash
ssh aircc 'bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/submit_parity.sh'
```

Optionally, submit it right after step 6 with `DEP=afterany:<the 5 link-2 ids, colon-separated>`.

For qwen_prm, pure_prm, rlhflow_prm and qwen_prm_72b (digit ids 15..24), it checks two cells into `$O/parity/`:
- `socratic_<prm>`: the v2 builder on the Socratic raw telemetry, against the existing Socratic `bank_v2` (old builder);
- `pb_<prm>`: the old and the v2 builder, both on the ProcessBench raw telemetry from step 6.

It also verifies that the old builder on AIRCC is the frozen one (sha256 in `env.sh`).

Expected log, `logs/pb_parity_<jobid>.out`:
- 5 lines `digit ids OK <prm> [...]`. Math-Shepherd must show the Mistral ids above, all others 15..24.
- 8 lines `PARITY <cell> OK max_abs 0.0 diffs 0`.
- `PARITY SUMMARY OK`. It writes `parity/PARITY_SUMMARY.json`.

Any `FAIL` means stop and report the line. Do not patch a builder.

The job can run before step 6 ends. The pb cells then print `NOTE ... not checked`, and `pb_down.sh` still refuses. Re-run it after step 6; finished banks are reused.

### 7c. Downstream

Submit only after `PARITY SUMMARY OK`:

```bash
ssh aircc 'bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/submit_down.sh'
```

You can also chain it behind 7b with `DEP=afterok:<parity job id>`.

For each PRM, `pb_down.sh` stops on the first failure:
1. It checks `parity/PARITY_SUMMARY.json`: every `socratic_`/`pb_` pair must be exact (RUN=full only).
2. It checks that the manifest is `COMPLETE`, or `COMPLETE_WITH_FAILURES` with only context-overflow failures.
3. It checks the digit ids of the PRM's tokenizer.
4. It builds `full_<prm>/bank_v3/` with the v2 builder and checks the ids recorded in its MANIFEST.
5. It writes `scores/SCORES_<prm>.npz` with `score_external_v2.py` (the frozen arms, guarded L-SML).
6. It writes `scores/SCORES_CHAIN_<prm>.npz` with `score_chain_pb.py`: CHAIN/TELE/PRMG and MEAN/HALF/EQ for N13 and N16, POS_ONLY, and LEN_ONLY from the raw step-token spans.

`ONLY="qwen_prm pure_prm"` limits it to some PRMs, for example to re-run one after a fix. `bank_v3` is skipped when it already exists. A `bank_v2` from an older run (for example the step 4 smoke) is never reused.

Send back the parity log lines, `parity/PARITY_SUMMARY.json`, each `bank_v3/MANIFEST.json` `digit_ids` block, and `scores/SCORES_CHAIN_<prm>.json` (`zero_var_channels`, `tele_dropped_constant`). Expect empty lists: with the digit fix, no channel should be constant.

## Step 8: upload to Drive (rclone on the cluster)

```bash
ssh aircc 'bash /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/code/upload_drive.sh'
```

This copies `full_*/bank_v3/*` (FEATURES.npz, REWARDS.npz, MANIFEST.json), the parity summary and manifests, the telemetry manifests, `census/` and `scores/` to `gdrive:hallucination_detection/cluster_results/prm_boost_processbench_v1/`. The ~1 GB telemetry pkls stay on AIRCC. For a slow upload, run it detached with `nohup` and poll the log.

## Step 9: local evaluation (gold labels stay on this machine)

1. **Fetch the scores.** They are a few MB and git-ignored by `results/**/*.npz`:

   ```bash
   mkdir -p results/prm_boost_processbench_v1/scores
   ```
   ```bash
   scp 'aircc:/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/scores/SCORES_*' results/prm_boost_processbench_v1/scores/
   ```
   ```bash
   scp aircc:/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_processbench_v1/full_qwen_prm/bank_v3/REWARDS.npz results/prm_boost_processbench_v1/scores/REWARDS_qwen_prm.npz
   ```

2. **Evaluate:**

   ```bash
   python results/prm_boost_processbench_v1/local/eval_processbench.py results/prm_boost_processbench_v1/scores results/prm_boost_processbench_v1/eval
   ```

   This takes about two minutes. It writes:
   - `eval/RESULTS_processbench.csv`: one row per PRM × arm × rule × population, with rules Q80, NC and native;
   - `eval/SUBSETS_processbench.csv`: per-subset official and PRISM columns;
   - `eval/BOOTSTRAP_processbench.csv`: question-clustered, stratified by subset;
   - `eval/VERDICT_processbench.json`: E1, E2 and the verdict of section 4;
   - `eval/EVAL_MANIFEST.json`. Its `clusters` should read 2,842 clusters and 988 answers in shared questions.
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

   On ProcessBench, CHAIN_N16 with NC is the pre-registered primary arm. Lead with its verdict line, then E1 and E2 per PRM. All other arms and Q80 are secondary. LSML_CONT is marked as using PRMBench label-chosen channels. HALF and EQ are controls. Do not headline HALF_N16 (Omri, see the results map).

## What to send back

- the census summary;
- the timing table with the proposed walls (then stop);
- after approval: the job ids;
- the downstream status;
- the parity numbers;
- the commit hash with the evaluation.

If something fails, quote the log line and stop. Do not patch the frozen driver or bank code to make it pass.
