Codex, request from Omri (2026-10-04): start a parallel LR helper now, so the LR stage of the Monday run finishes overnight instead of Monday afternoon. Omri approved this. Claude prepared the design but cannot write into your live run, so you own and run it.

WHY
- The live runner (PID 455132, `monday_fusion_run.py continue`) is in position_decision. LEVEL layers are at 354/360 (~19:00 done); estimator-input layers 0/288, expected done ~01:30 (256 PB stems at ~0.3 min, 32 PRMBench stems at ~9.8 min, measured from checkpoint timestamps).
- LR then runs 1,440 stems serially. Claude timed `monday_fusion_lr.model_fit` on real banks (3 of 5 folds, 55,623 PRMBench training rows): B13 0.13-0.23 s per fit, B54 0.42-2.66 s; PB omnimath_q8 0.01-0.08 s. Each stem = 146 fits + 24 evaluate calls, so PRMBench stems take ~3-4 min (160 stems) and PB stems ~10 s (1,280 stems): ~13 h, ending ~14:30 Monday, after the meeting window.
- LR's computation does NOT depend on the position/decision layers. `run_lr` only needs `restore_content` (content is complete and sealed); `POSITION_DECISION_COMPLETE.json` is a gate, not a data dependency. And `run_lr` skips any stem whose `.complete.json` exists and whose artifact hashes verify (`completed_chunk`). So stems computed early are reused, not recomputed.

WHAT TO RUN
A separate helper process (not a second copy of the workflow) that:
1. Uses exactly the committed code: `results/monday_fusion_analysis_v1/code` at branch head 7d26e1cc2 (code files identical to 203a0f849058; only docs changed). Run with `python -B`. Before starting, assert that every monday_fusion*.py and monday_fusion_run.py equals `git show <head>:<path>`, as `code_seal` does, but do NOT call `code_seal` (it rewrites CODE_MANIFEST files; leave that to the runner).
2. Builds `pop = calfix_common.Population()` and `manifest` from `results/level_reduction_v1/run_20261004_resume/INPUT_MANIFEST.json`. Do NOT call `gates.audit()` (it writes INPUT_AUDIT.json and manifests).
3. Iterates stems in the same stem naming as `run_lr`: `{cell}__{bank}__{variant}__{roster}__pos{0|1}`, variants original/remove_2/remove_3/random_2/random_3, rosters full/survivors, position False/True; cells sorted, banks in manifest order.
4. Priority order: first all `original` stems, PRMBench first (288 stems, ~2.6 h; this is the per-bank linear ceiling Omri needs for Monday). Then the remaining 1,152 stems from the END of the runner's order backwards (PRMBench stems sit at the end).
5. For each stem: `restore_content(data,bk,ce,variant)`, then `monday_fusion_layers.lr_rows(...)` inside the same try/except tuple as `run_lr` (ValueError, ArithmeticError, ConvergenceWarning, LinAlgError) with the identical ARM_FAILURE rows; set `runtime_s`; write `.metrics.csv`, `.fits.json`, `.fold_metrics.json`, and LAST the `.complete.json` with the artifact sha256 map, byte-for-byte the same structure as `run_lr`. Use `dump` (atomic tmp+replace). Call `storage_guard()` before each stem.
6. Memory: free RAM is ~1.6 GB with the runner at ~0.8-0.9 GB. Load `HistoricalData(manifest,pop,banks=[bk])` one bank at a time (per-bank keys are seeded per bank, so values equal the all-bank load), group the work by bank, and use 2 BLAS threads. Before using a restored state, verify it: per-answer earliest argmax of every restored score must equal the sealed `chunks/{cell}__{bank}__{variant}.stats.npz` `<recipe>__ANSWER_Z_Q80__argmax`; stop on any mismatch.

COLLISION RULE (the only real risk)
Two writers on one stem break its seal: `runtime_s` differs, the second write changes the hashes, and the runner's `completed_chunk` raises "Completed checkpoint changed" and stops the whole workflow. So:
- Never write a stem whose `.complete.json` already exists. Re-check immediately before writing.
- Runner position: if `WORKFLOW_STATUS.json` stage is not `lr`, treat the runner as sitting at index 0 (it will start there). If stage is `lr`, the runner is at the first non-complete stem after the last `LR checkpoint <stem>` line in `results/monday_fusion_analysis_v1/RESUME_20261004.log` (UTF-16).
- Only START a stem if the runner's estimated time to reach it exceeds 15 minutes (estimate 240 s per PRMBench stem, 12 s per PB stem, summed over the non-complete stems between the runner and the candidate). Otherwise skip it and pick the next candidate, or exit when none qualifies.
- After computing and before writing, re-check the runner position; if the runner could have reached the stem, discard the result.
- Provide a STOP file the helper checks between stems, and a provenance JSON outside `lr/` (helper PID, head, code hashes, thread count, every stem written with its complete.json hash and time).

CHECKS TO REPORT
- One determinism check before the main loop: compute one small PB stem twice in two processes; all fields except `runtime_s` must match exactly.
- Count of helper stems written, ARM_FAILURE count, and that the runner later logged no "Completed checkpoint changed".
- Do not stop or restart the runner. Do not change protocols or sealed outputs.

If you would rather not run a helper, tell Omri; the alternative is accepting LR ending ~14:30 Monday.
