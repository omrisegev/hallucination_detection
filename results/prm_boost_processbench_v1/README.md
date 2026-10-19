# PRM boosting on ProcessBench (2026-10-19): code and inputs, nothing run yet

This directory runs the same frozen PRM-boosting recipe as the Socratic transfer (`../prm_boost_external_v1/`) on ProcessBench. That recipe is the PRM step reward plus telemetry from the same PRM's backbone, fused label-free. PRISM (arXiv 2606.09078) and CROP (arXiv 2605.30085) both report ProcessBench, so these results can be compared with both papers.

**Status:** code, inputs and tests only. No GPU job has run, no label has been read by any scoring step, and nothing has been scored. The run order is in [HANDOFF.md](HANDOFF.md). The full run waits for Omri's budget decision after the timing jobs.

## Layout

| path | goes to AIRCC? | what |
|---|---|---|
| `make_inputs.py` | no | converter: local ProcessBench rows → `inputs/` and `local/processbench_gold.json` |
| `inputs/answers.json` | yes | 3,400 answers `{uid, question, steps, benchmark, source_id}`, no label field |
| `inputs/answers_timing12.json` | yes | 3 deterministic answers per subset (seed 20261018) |
| `inputs/INPUT_MANIFEST.json` | yes | counts, sha256 of every input, ordered-id hash |
| `aircc/` | yes (`ship.sh`) | census, timing, full chains, CPU downstream, Drive upload |
| `local/` | **never** | gold labels, evaluator, metric module, parity scripts |
| `ship.sh` | runs on Windows | sends `inputs/` and `aircc/` only; checks sha256 on both ends |

AIRCC paths: inputs `$SHARED/data/prm_boost_processbench_v1/`, everything else `$SHARED/results/prm_boost_processbench_v1/` (`code/`, `census/`, `timing_<prm>/`, `full_<prm>/`, `scores/`).

## Inputs (`make_inputs.py`)

- **Source.** The dataset fields (id, generator, problem, steps, label) of the 3,400 Qwen/ProcessBench rows carried in the 2026-08-10 Qwen2.5-Math-PRM-7B run (job 177776, `dataset_cache/four_localization/pb_prm_qwen25math7b_full/`). The source pkl hashes are in the manifest. `--hf` reads the same rows from the HF cache instead.
- **Integrity.** The ordered-id hash `7a7edc7a…d20e10` equals the repo's official-3,400 audit (`results/fair_paper_exact_comparisons_v1/PROCESSBENCH_SOURCE_AUDIT.json`). The converter stops if it differs.
- **Counts:**

| subset | answers | erroneous | correct | steps |
|---|---|---|---|---|
| GSM8K | 400 | 207 | 193 | 2,082 |
| MATH | 1,000 | 594 | 406 | 6,505 |
| OlympiadBench | 1,000 | 661 | 339 | 8,819 |
| Omni-MATH | 1,000 | 759 | 241 | 8,291 |
| **total** | 3,400 | 2,221 | 1,179 | 25,697 |

  25,697 steps is the same count CROP gives in its Table 4. 272 answers have their first error at step 0.
- **Label convention.** `label` is the 0-based first erroneous step, and -1 means a correct solution. It must not go through `spectral_utils.prm_label_contract`, which is PRMBench's one-based contract.
- **Hashes.** `answers.json` `1f0acbb3…c9f85fe`, `answers_timing12.json` `4a1d1ff3…a344d9e`, `processbench_gold.json` `54d9d0c6…c53406ac`. `results/** -text` in `.gitattributes` keeps them byte-exact on a Windows checkout.
- **Lengths.** The longest answer is 7,947 characters and none has an empty step. Overflow of the 4,096-token Qwen2.5-Math context is unlikely, but `aircc/length_census.py` counts it exactly per PRM before any GPU job runs.
  - Over-long answers are **not truncated**: the driver records them as counted failures (`failure='context_overflow'`).
  - The evaluator reports coverage, a `common` population (answers every PRM read), and `avg_f1_failures_wrong`, which counts every unscored answer as a miss.

## Pipeline (identical to Socratic, frozen)

1. **Telemetry.** `cluster/run_prmbench_prm_telemetry.py --answers`, submitted with `cluster/submit_prm_telemetry_offline.sbatch`.
   - It runs from `$SHARED/code_integration_7534b8bf4`, the code dir of the Socratic run.
   - The driver, adapters, scorer and sbatch files are byte-identical between that commit and this one (`git diff 7534b8bf4 HEAD` is empty for those files).
2. **Bank.** `../prm_boost_external_v1/build_bank_external.py` builds the frozen 56-channel bank with mask1 and empty steps dropped.
3. **Frozen arms.** `../prm_boost_external_v1/score_external.py` computes PRM, HALF_N16, LSML_N16_PRMG and LSML_CONT.
4. **Post-registration arms.** `aircc/score_chain_pb.py` computes CHAIN/TELE/PRMG for N13 (digit-free) and N16 (3 digit channels).
   - Its scoring body is byte-identical to `../prm_boost_external_v1/chain_socratic/score_chain_external.py`; only the module paths differ, so it can run on the AIRCC mirror.
   - CHAIN is a post-registration arm, as on Socratic.
5. **Evaluation.** `local/eval_processbench.py` runs on the local machine only.

## Metrics (`local/pb_metrics.py`, `local/eval_processbench.py`)

- **Official ProcessBench metric.** Predicted first error = first flagged step, or -1 if none. error_acc, correct_acc and their harmonic-mean F1 are computed per subset; the average is the mean of the four subset F1s.
- **PRISM.** PRISM's ProcessBench table (Table 6) reports TPR/TNR/F1. Its Baseline row equals our job-177776 PRM-7B run to 0.1: GSM8K 71.0 / 96.4 / 81.8 and MATH 67.0 / 91.1 / 77.2. So PRISM's TPR/TNR are the official error_acc/correct_acc, and they are exported as `prism_tpr/prism_tnr/prism_f1`.
  - PRISM's OlympiadBench and Omni-MATH cells look column-swapped in their table.
- **Step confusion.** Step-level TPR/TNR/F1 on the labelled domain is our own definition and is labelled that way. The domain is all steps of correct answers plus steps 0..label of erroneous answers.
- **Within-answer AUC.** In each erroneous answer with label ≥ 1, the AUC of the first-error step against the steps before it.
- **Pooled step AUROC (CROP).** It has three declared targets: `prefix` (the domain above), `first_only` (all steps; only the first error is positive) and `cumulative` (every step at or after the first error is positive).
  - On the job-177776 PRM-7B rewards, `first_only` gives **0.883**. CROP reports Direct PRM at 0.8801 ± 0.0059, so `first_only` is the target to quote beside CROP.
  - `prefix` gives 0.954 and `cumulative` 0.825.
- **Decision rules.** `<arm>_Q80` uses the answer-z pooled 80th percentile, as on PRMBench and Socratic. `PRM_native` is reward < 0.5, ProcessBench's tie rule. `PRM_native_adapter` is the adapters' `native_valid` (reward > 0.5 is valid), kept for continuity.
  - The two native rules differ only on bf16 rewards of exactly 0.5 (4 of 3,400 answers for PRM-7B). `PRM_native` reproduces job 177776's recorded F1 exactly, while `PRM_native_adapter` does not (+0.05 average).
- **Bootstrap.** Paired, stratified by subset, 1,000 draws, seed 20261018. Each arm/rule is compared with PRM_native and with PRM_Q80 on average F1, within-AUC and pooled AUROC (prefix and first_only).

**PRM-7B reference from job 177776** (`local/reference_prm7b_0810.py`; the evaluator's real-data check):

| rule | avg F1 | GSM8K | MATH | OlympiadBench | Omni-MATH | within-AUC | AUROC first_only | AUROC prefix |
|---|---|---|---|---|---|---|---|---|
| PRM_native | 72.94 | 81.77 | 77.23 | 66.69 | 66.05 | 0.939 (1,949 answers) | 0.883 | 0.954 |
| PRM_Q80 | 34.76 | | | | | 0.939 | 0.883 | 0.954 |

## Open questions for Omri (before the full run)

1. **First-error rule for the fused arms.** The answer-z Q80 rule flags steps in almost every answer, including correct ones. On ProcessBench's official metric it therefore collapses: the PRM alone scores 34.8 under Q80 against 72.9 under its native rule.
   - So the fused arms, which have only Q80, cannot compete on official F1 by construction. They can only be judged on within-AUC and pooled AUROC.
   - A label-free answer-level rule would have to be pre-registered before results arrive. One example: flag the same fraction of steps that the PRM's native rule flags.
2. **CROP's target and access.** `../prmbench_paper_metric_parity_v1/` (commit 419bb905) reads PRISM's ProcessBench
   Table 6 the same way as here, and it has a CROP split-protocol reproduction for PRMBench (`crop_metrics.py`). Its ProcessBench row
   can be added once these scores exist. **Target:** CROP's stated 6.6% erroneous-step rate does not match `first_only` (8.6%) or any other target we tried. Their split protocol (10 × 60/20/20, supervised LR fusion) also differs from our full-population, label-free scoring.
3. **Budget.** This is a guess before timing: the Socratic per-row rates × 3,400 rows, scaled for the longer answers, give about 9–13 GPU-h. The timing jobs replace this guess.

## Census and timing on AIRCC (2026-10-19, Windows session)

Shipped at commit 9ce261d80 (hub `claude/prm-boosting` + the pre-registration commit). The remote sha256 of both
input files matched the manifest, and all 8 telemetry files in `code_integration_7534b8bf4` matched this checkout.
`/preflight` passed.

**Census** (job 289024, CPU, all 3,400 answers; files in `census/`): 0 build errors and 0 context overflows on all
5 PRMs. rlhflow_prm has **87 rows with a step-text mismatch** (gsm8k 2, math 19, olympiadbench 30, omnimath 36). The
uids are in `census/MISMATCH_rlhflow_prm.json` (diagnostic job 289052, script `census/mismatch_uids.py`). None of the
12 timing rows is among them. The cause is not yet diagnosed; it is not explained by non-ASCII text.

Operational note: `submit_census.sh` submits to power-gpu without a GPU, so Slurm moves it to `cpu`, where the NVIDIA
container hook fails (`nvidia-container-cli` not found, job 289021). Resubmitting with `-p cpu` and
`--export=...,NVIDIA_VISIBLE_DEVICES=void` works (jobs 289024, 289052).

**Timing** (jobs 289026-289030, 12 answers, 1 GPU, 1 h wall each; manifests in `timing_<prm>/`):

| adapter | job | rows | elapsed s | s/row | full-run GPU-h | max input tokens (census) | context overflow (census) | step-text mismatch | status |
|---|---|---|---|---|---|---|---|---|---|
| qwen_prm | 289026 | 12 | 9.0 | 0.75 | 0.80 | 2610 | 0 | 0 (timing: 0) | COMPLETE |
| pure_prm | 289027 | 12 | 6.0 | 0.50 | 0.53 | 2580 | 0 | 0 (timing: 0) | COMPLETE |
| rlhflow_prm | 289028 | 12 | 9.2 | 0.77 | 0.82 | 2899 | 0 | 87 (timing: 0) | COMPLETE |
| math_shepherd | 289029 | 12 | 6.9 | 0.57 | 0.61 | 2853 | 0 | 0 (timing: 0) | COMPLETE |
| qwen_prm_72b | 289030 | 12 | 48.9 | 4.08 | 4.35 | 2612 | 0 | 0 (timing: 0) | COMPLETE |

full-run GPU-h = s/row × 3,400 × 1.13 / 3,600. Total about **7.1 GPU-h** of compute, plus about 1.5 minutes of
setup per job (about 7.4 GPU-h with 10 links). Each timing job took 1.4-2.1 minutes of wall in total.

**Proposed walls per link** (1.5 × estimate, rounded up): `WALL_7B=02:00:00`, `WALL_72B=07:00:00`. Two links give
twice that capacity. Waiting for Omri's budget decision; nothing beyond timing has been submitted.

## Full run (Omri approved all 5 PRMs, 2026-10-19)

Submitted with `BUDGET_OK=1 WALL_7B=02:00:00 WALL_72B=07:00:00 submit_full_chains.sh`; linear 2-link chains (link 2 =
`afterany:<link 1>`, same `--out`).

| adapter | link 1 | link 2 | wall per link |
|---|---|---|---|
| qwen_prm | 289058 | 289059 | 02:00:00 |
| pure_prm | 289060 | 289061 | 02:00:00 |
| rlhflow_prm | 289062 | 289063 | 02:00:00 |
| math_shepherd | 289064 | 289065 | 02:00:00 |
| qwen_prm_72b | 289066 | 289067 | 07:00:00 |

Step 7 (downstream) stays gated on the pre-registration section 8 code changes.

**Full-run outcome:** all 10 jobs COMPLETED 0:0; all 5 manifests (`full_<prm>/manifest.json`) are COMPLETE with
3,400/3,400 rows, 0 failures, 0 context overflow, `reward_path_max_abs_gap` 0.0. Link 1 did all the scoring
(elapsed_sec from the link-1 logs: qwen_prm 304, pure_prm 308, rlhflow_prm 370, math_shepherd 350, qwen_prm_72b 712);
link 2 found the work done and rewrote the manifest, so the saved manifest shows the link-2 job id and elapsed time.
rlhflow_prm: 87 rows (133 steps) with a step-text mismatch and 3,378 rows with boundary whitespace, as the census
predicted; the other four PRMs have 0. The ~0.9 GB telemetry pkls stay on AIRCC.

## Step 7 on AIRCC (2026-10-19/20, Windows session, code shipped from d6e9cb04a)

- **PURE tokenizer.** The offline cache for `jinachris/Qwen2.5-Math-7B-PRM800K` has `vocab.json` + `merges.txt` but no
  `tokenizer.json`, so parity job 289255 stopped at the digit check. Job 289257 (`parity_diag/derive_pure_tok.py`)
  saved PURE's own tokenizer as `$S/tok_derived/pure_prm/tokenizer.json`: digit ids 15..24, `vocab.json` identical to
  Qwen2.5-Math-7B-Instruct's. All later jobs run with `TOK_pure_prm` set to it. Digit ids OK on all 5 PRMs.
- **Bank parity (job 289258):** all 4 `pb_<prm>` pairs exact (max_abs 0.0). All 4 `socratic_<prm>` pairs FAIL against the
  existing Socratic `bank_v2` with max_abs 8.9e-10 to 5.4e-8 in `FEATURES.npz`.
- **Diagnostic (job 289294, `parity_diag/parity_diag.sh`):** the OLD frozen builder, re-run today on the Socratic qwen_prm
  telemetry, equals the v2 bank exactly and differs from the existing `bank_v2` by the same 3.26e-9. The gap is from
  rebuilding, not from the v2 builder.
- **Omri's decision (2026-10-19, option 1):** the Socratic reference is the old-builder rebuild. Job 289297
  (`parity_diag/parity_rebuiltref_then_down.sh`) builds the other 3 old-builder Socratic references, re-runs
  `bank_parity.py` (the failed summary is kept as `parity/PARITY_SUMMARY_existing_socratic_ref_FAIL.json`), then runs
  `pb_down.sh` unchanged. No builder or scorer code was changed.
