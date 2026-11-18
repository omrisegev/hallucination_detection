# RC HANDOFF V34: realism check of 2 new synthetic errors on AIRCC (dev only), 2026-11-18

Who: the Claude Code session on Omri's PC (Remote Control). Requested by the cloud thread "Unify post-meeting research into one plan".
Omri's decision (2026-11-18 22:24 Israel time, decision card): "Numeric edit + on-topic donor": one dev-only GPU pass with two
new variants, then only the realism check for both. Background: `analysis_v1/DIAG_REALISM.md` section 4 (options A and B).
- V3 numeric edit: in step k, the last number changes by +-1 (decimal: last digit +-1). Later steps unchanged.
- V4 on-topic donor: step k is replaced by a step of another answer to the same question, from another position.
Code: `make_variants_v34.py` (inputs), `realism_v34.py` (check). Pipeline files extended to V3/V4: `aircc/common.sh`,
`aircc/job.sbatch`, `aircc/submit.sh`, `aircc/prmb_variant_bank.py`, `smoke_check.py`, `ssl_data.py` (V1/V2 behavior unchanged).

## Rules (same as RC_HANDOFF.md)
- Never switch branches in C:\Users\omris\TAU\hallucination_detection, no new worktree (C: has little space). Use `git fetch` +
  `git archive` only.
- Never stop or cancel jobs or processes that you did not start. Delete nothing (also not the old files in $W).
- Do not change any rule. If a step fails, stop that step and report the output. Never commit pkl files.
- Push only to `claude/ssl-one-experiment-2026-11-18`, by git plumbing, with `--no-verify`.

## 1. Code to AIRCC (only the experiment folder of the frozen checkout)
S=/shared/cycle2_tau_averbuch_prj/omrisegev1, W=$S/results/ssl_one_experiment_v1, X=$S/code_ssl_one_v1/results/ssl_one_experiment_v1.
1. PC: `git fetch origin claude/ssl-one-experiment-2026-11-18` (GIT_LFS_SKIP_SMUDGE=1). Commit = the branch head (report it).
2. AIRCC: back up the current folder first: `tar -czf $W/code_backup_pre_v34.tgz -C $S/code_ssl_one_v1 results/ssl_one_experiment_v1`.
3. Copy the new versions: `git archive <commit> results/ssl_one_experiment_v1/*.py results/ssl_one_experiment_v1/aircc results/ssl_one_experiment_v1/*.md`
   piped (or as a file) to `tar -x -C $S/code_ssl_one_v1` on AIRCC. Check: `grep -c V3 $X/aircc/common.sh` > 0.

## 2. Inputs (AIRCC login node, the same pyenv as make_variants.py; about 1 min)
```
cd $X && python3 make_variants_v34.py --pb-answers $S/code_ssl_one_v1/results/prm_boost_processbench_v1/inputs/answers.json \
  --pb-gold $S/code_ssl_one_v1/results/prm_boost_processbench_v1/local/processbench_gold.json \
  --prmb-meta $S/results/prm_boost_v1/full_prm/prmbench_telemetry.pkl \
  --folds $S/code_ssl_one_v1/results/localization_source_group_audit_v1/FOLDS_V2.json --split $W/inputs/SPLIT.json --out $W/inputs
```
- Must print `SPLIT GATE PASS`. If not: stop and report. It refuses to overwrite files (it writes only new V3/V4 files).
- Expected (local test on the real PB answers): PB V3 about 1,690 rows, PB V4 about 420 rows (most PB questions have one answer).
- Report the 4 lines `<bench> <V3|V4> {...}` (rows, skips, relax levels).

## 3. GPU jobs (4 jobs, the same job.sbatch; dev answers only, so about 5-15 min each on B200)
`bash $X/aircc/submit.sh processbench_V3 processbench_V4 prmbench_V3 prmbench_V4`
Each job: preflight 12 rows (C rows of the parents of the first 12 variant rows + 12 variant rows) -> smoke_check (strict
prefix tolerance 1e-4, as pre-registered) -> full run -> bank (frozen builders) -> extract_steps (GATE PASS) -> JOB_*.json.
- If a smoke check fails: do NOT resubmit and do NOT change the tolerance. Report SMOKE.json (S2 failures). If the only
  failure is the numeric tolerance (ids_equal and spans_equal true), you may run smoke_check.py again on the login node with
  `--prefix-tol 0.05 --out $W/pre/<job>/SMOKE_tol05.json` as information only, and report it. Omri decides about the tolerance.
- Poll the jobs at most every 5 minutes. All other jobs continue.

## 4. Upload and copy
1. Drive: the same rclone remote and folder as the first upload (gdrive:hallucination_detection/cluster_results/ssl_one_experiment_v1/aircc/):
   copy the new files only: inputs/*V3*, inputs/*V4*, inputs/VARIANTS_V34_REPORT.json, code_backup_pre_v34.tgz,
   pre/*_V3/SMOKE.json, pre/*_V4/SMOKE.json, runs/*_V3/manifest.json, runs/*_V4/manifest.json, steps/STEPS_*_V3.npz,
   steps/STEPS_*_V4.npz, JOB_*_V3.json, JOB_*_V4.json. Do not upload the raw telemetry pkl files.
2. PC: copy steps/STEPS_*_V3.npz, steps/STEPS_*_V4.npz, inputs/SIDECAR_*_V3.jsonl, inputs/SIDECAR_*_V4.jsonl and
   inputs/VARIANTS_V34_REPORT.json to C:\Users\omris\TAU\hd_collection_data\ssl_one_experiment_v1\ (steps\, inputs\).
   Check free disk first (need well under 1 GB).

## 5. Realism check (PC, CPU, about 1-2 min)
1. Code: `git archive <commit> results/ssl_one_experiment_v1/*.py` into a new folder
   C:\Users\omris\TAU\hd_collection_data\ssl_one_experiment_v1\analysis_v34\code\ (no checkout).
2. Run with the same Python 3.13 and the same --gold file as the analysis run of analysis_v1:
   `python realism_v34.py --W C:\Users\omris\TAU\hd_collection_data\ssl_one_experiment_v1 --gold <processbench_gold.json> --out ...\analysis_v1\REALISM_V34`
3. Check the lines `code check`: PRMBench step V2 .8209, answer V2 .7232, answer V1 .806; PB answer V1 .898. If a code check
   fails: report it (the comparison is then not valid).
4. Push `analysis_v1/REALISM_V34/` (REALISM_V34.json, R1_realism_v34.png, realism_v34.log) to the branch under
   results/ssl_one_experiment_v1/analysis_v1/REALISM_V34/ by git plumbing (as for analysis_v1), --no-verify.
   Upload the same folder to Drive cluster_results/ssl_one_experiment_v1/analysis_v1/REALISM_V34/ (only this new folder).

## 6. Report to the thread (Hebrew, tables + bullets)
Commit, input counts, job ids and states, SMOKE results, GATE PASS lines, the realism table (variant x step / answer AUROC,
agreement, pass), the code-check line, the push commit and the Drive paths.
