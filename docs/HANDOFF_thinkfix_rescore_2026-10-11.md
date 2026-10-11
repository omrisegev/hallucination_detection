# Handoff: Qwen3 think-block fix, GPU reruns and CPU re-score (2026-10-11)

Branch `claude/thinkfix-rescore-v1`, based on `claude/tf-think-block-fix` (fix commit 153589e43).
Full method notes are in `results/thinkfix_rescore_v1/README.md`.

## Done

1. **GPU reruns with the empty think block** (AIRCC, 2026-10-10). Every rerun went into a new
   `_thinkfix` directory; the old directories are untouched.
   - Gate B passed for Qwen3-8B (job 280415) and Qwen3-4B (job 280416). The empty block was stripped
     in 20/20 traces. Median |dH| was 0; median r was 0.9987 / 0.9982, which is within the driver's
     noise floor.
   - PRMBench pilot: job 280422. Decode check: job 280423.

   | Run | Jobs | Stats |
   |---|---|---|
   | PRMBench 8B | 280425, 280426 | 6969 rows, 3 misaligned (as before); first-token surprisal median 34.6 -> 24.1; `<think>` top-1 on 100% -> 0% of rows |
   | ProcessBench 8B | 280427, 280428 | 3400 rows; first token 34-37 -> 10-13 nats |
   | ProcessBench 4B | 280429, 280430 | 3400 rows; first token 34 -> 14-16 nats |
   | Hard2Verify Qwen3-8B | 280431 | 200 records, alignment gate PASS |
   | Socratic Qwen3-8B | 280432 | 2995 records, alignment gate PASS |

   - socratic_qwq32b was not rerun: QwQ's template has no non-thinking mode.
   - Per-cell `THINKFIX_STATS.json` and manifests are under `dataset_cache/...`. Their paths are
     listed under "Where the data lives".
2. **mask1 cells** (forced token 0 removed; Omri, 2026-10-10). Built by `scripts/build_mask1_cells.py`
   in job 280515: 0 failures in every cell. The MASK.json files are copied to
   `results/thinkfix_rescore_v1/mask1_cells/`.
3. **Feature bank rebuild** (`scripts/rebuild_source_bank_v4.py`):
   - The old cells reproduce the frozen bank to 6.1e-7 (job 280517). This was the gate for everything
     after it.
   - The thinkfix bank was built in job 280518 and the mask1 bank in job 280519.
4. **PRMBench/ProcessBench re-score** (dev_run + cont baselines) for old, thinkfix and mask1
   (jobs 280525, 280526, 280533):
   - The old replay reproduces the frozen numbers. The exceptions are 27 of 3633 PRMBench values in
     N40/N53/N56, all within 3.3e-4.
   - mask1 within-answer AUC is lower by 0.0088 on average. N16 MEAN_MOM_ga goes 0.7909 -> 0.7767.
   - The comparison tables are `results/thinkfix_rescore_v1/{PRMBENCH,PROCESSBENCH}_COMPARISON.csv`.
   - The PRMScore inputs (mask1) are in `results/thinkfix_rescore_v1/prmscore_inputs/`.

## Not finished

- **Supervised LR: DONE** (follow-up commit). The first attempt (jobs 280534-280536) failed in
  aggregate on `relative_to(MAIN)`. After the fix, jobs 280724-280726 completed.
  - N16 LR within-AUC: 0.8224 -> 0.8015 (mask1).
  - The old replay hits the frozen 0.822366 exactly.
  - The LR's lead over the best label-free method narrows from 0.026 to 0.023.
- **Drive verification** of every `_thinkfix`/`_mask1` dir plus `thinkfix_rescore_v1`: rclone copy +
  check loop started 2026-10-11 (`$SHARED/drive_verify.sh`). Results are in
  `$SHARED/logs/drive_verify_20261011.tsv`.
- **External step 3, part 1: DONE.** Locked L-SML evaluation. Working copies are
  `$SHARED/thinkfix_ext_{thinkfix,thinkfix_mask1}`; scoring jobs 280815/280817, evaluation jobs
  280820/280821. Results are in `results/thinkfix_rescore_v1/external/`. The Socratic mask1 cell was
  rebuilt after a mask-rule fix (job 280814); see README.
- **External step 3, part 2: DONE.** Jobs: thinkfix 280824/280825/280847-280854, mask1
  280828/280829/280855-280862. Outputs are in `results/thinkfix_rescore_v1/external/<variant>/panel_chain/`
  and `EXTERNAL_PANEL_COMPARISON.csv`.
- **Self-generated step 4: DONE.** The ProcessBench B16 reference was refit (job 280870) and
  `selfgen/run_b16_eval_thinkfix.py` was run locally. Results are in `results/thinkfix_rescore_v1/selfgen/`. Every script in the chain hardcodes the three
  cells, including QwQ, and its freeze hashes cover the scripts themselves. The re-run needs patched
  copies under a new freeze written before the labels are opened. Chain:
  1. `scripts/run_external_local_cpu.py`
  2. `scripts/run_family_external.py`
  3. ssl worktree `scripts/extract_external_banks_v4.py external`
  4. `results/monday_external_panel_v1/score.py` + `evaluate.py`
  5. the cont and nested `score_ext`/`evaluate_ext` scripts
  6. `scripts/evaluate_external_locked_scores.py`

  The mask1 records are at
  `$SHARED/results/lsml_external_generalization_v1/{hard2verify,socratic}_qwen3_8b_full_thinkfix_153589e_mask1/records`.
- **Self-generated check (step 4)**: done; see above. The original note follows. The generated `evdrop_*` traces themselves are
  clean. Their teacher-forced reference side reads the old PB cells and still needs a re-run:
  `scripts/self_generated_labels/evaluate_selfgen.py` and `b16_fit.py`/`b16_eval.py` on
  `claude/self-generated-step-labels-v1`, with `HD_MAIN_CHECKOUT` pointing at a tree that has the
  `_thinkfix` cells.

## Where the data lives

- **AIRCC**: `$SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1`.
  - Cells:
    - `results/prmbench_qwen3_8b_telemetry_thinkfix{,_mask1}`
    - `results/pb_qwen3_{8b,4b}_thinkfix{,_mask1}`
    - `results/lsml_external_generalization_v1/*_thinkfix_153589e{,_mask1}`
  - Re-score: `results/thinkfix_rescore_v1/`. It holds the banks, `FEATURES.npz`, `PER_ANSWER.npz`
    and the per-variant `prmscore_inputs`.
  - Code snapshots: `code_tf_thinkfix_153589e`, `code_thinkfix_rescore_v1`, `code_thinkfix_bank_553de79`.
  - Downstream mirror: `thinkfix_downstream/` (a directory literally named
    `C:\Users\omris\TAU\hallucination_detection`).
  - Native python env: `pyenv_native_312_20261011`.
  - Wheelhouses: `wheelhouse/{tf_thinkfix_20261010,native_312_20261011,downstream_20261011}`.
- **Drive**: `gdrive:hallucination_detection/cluster_results/<same names>`. The rclone upload of the
  GPU outputs was started 2026-10-10 (log `$SHARED/logs/upload_thinkfix_20261010.log`); check that it
  completed. The old/thinkfix `prmscore_inputs` go to `cluster_results/thinkfix_rescore_v1/`.
- **Local main checkout**:
  - `dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_thinkfix/` (pkl + stats)
  - `dataset_cache/repgrid/pb_qwen3_*_thinkfix/` (manifests/stats only)
  - `dataset_cache/lsml_external_thinkfix/`
  - `results/thinkfix_rescore_v1/`

## Resuming on another machine

1. `git fetch origin claude/thinkfix-rescore-v1` and read `results/thinkfix_rescore_v1/README.md`.
2. Large inputs (pkl, FEATURES.npz) are not in git. Fetch them from Drive or AIRCC.
3. Cluster rules learned this session:
   - Use account `cycle3_tau_averbuch_prj`, QoS `owner_940`; `owner_880` is rejected.
   - CPU-only jobs land on partition `cpu`, where the NGC container fails ("spank plugin stack").
     Use `cluster/native_job.sbatch` or `cluster/thinkfix_downstream.sbatch`; both use system
     python3.12 + `pyenv_native_312_20261011`.
   - GPU jobs need `cluster/install_offline_tf_thinkfix.sh`, because compute nodes have no DNS.
