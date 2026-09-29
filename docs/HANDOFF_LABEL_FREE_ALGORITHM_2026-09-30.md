# Handoff: the label-free algorithm line, Steps 447-462 (Claude, 2026-09-30)

Written for Omri and for any agent (or merge session) continuing this line. Worktree
`.worktrees/ssl-pseudolabel-residual-v1`, branch `claude/ssl-pseudolabel-residual-v1`, pushed at 716892a88 and later documentation
commits. The research state and next directions are in `Research_Directions.md`, section "2026-09-30 (Claude): the label-free
algorithm line". The narrative is in HISTORY.md Steps 447-462; lessons in LESSONS.md (entries dated 2026-09-28 and 2026-09-29).

## 1. What each step did (all on the matched 13,769-answer population, source_groups_v2 folds unless stated)

| Step | Experiment (results dir) | Question | Outcome | Verdict files |
|---|---|---|---|---|
| 447 | partition ceiling | Is the declared Joint partition near the best block-equal partition? | Bottom quartile; the whole space barely reaches CT7 | HISTORY |
| 448-449 | Mind the Gap audit + reproduction | Can the published ProcessBench detector be reproduced? | Released code is not the published method; published method reproduced in pattern, within 3.3 SLA points | HISTORY, memory |
| 450-452 | `expectation_realization_v1` (stages A, B, B2) | Does the written-token block help; can label-free SML/DS estimates weight channels? | Block +0.0119 under L-SML; estimates select (filter = anti-oriented channels) but cannot weight; L-SML = averaging | `run_20260927*/RED_TEAM.md` |
| 453 | calfix of six older runners | Did the eval/calibration overwrite change conclusions? | No cited conclusion changes | HISTORY |
| 454 | PRMScore decomposition | Where do PRMScore gains come from? | Mostly position structure; gap to Qwen PRM is flag-count allocation | HISTORY |
| 455 | `er_generality_v1` | Does the stage-B method generalize to 20/32/51 channels? | Filter selection yes; score gain, grouping, L-SML no | `run_20260927/RED_TEAM.md` |
| 456 | `lsml_merge_step_v1` | Does one label-free merge step fix L-SML's partition; Omri's DS band rule? | Merge fixes the split level family; L-SML still loses to averaging; band rule no gain | `run_20260928/RED_TEAM.md`, `REPORT_HE.html` |
| 457 | `algorithm_decisions_v1` | Decide every open component on 8 banks | Frozen candidate = DS filter + plain average (mean 0.7656); grouped DS weights help only on 32(+d); L-SML loses 8/8; digits help everywhere | `run_20260928/RED_TEAM.md`, `REPORT_HE.html` (artifact https://claude.ai/artifact/VRKmRKGMtpDrM5An177XK1) |
| 458 | `partition_switch_v1` | A label-free rule for when to use the grouping | Meets its criterion only through twin bank families; with the family held out it never switches on | `run_20260928/RED_TEAM.md` |
| 459 | `algorithm_external_v1` + `external_banks_v4` | The frozen candidates on Hard2Verify / Socratic (all digit banks) | Above ct7 on Socratic mostly through position; step index alone beats every method; exploratory (benchmarks exposed) | `run_20260929/RED_TEAM.md`, SEAL.json |
| 460 | `position_channel_v1` | Step position as one more channel | Adopted (+0.004 to +0.009, 4 banks); direction label-free, weight = 1/(p-1) | `run_20260929/RED_TEAM.md` |
| 461 | `position_prior_v1` | Position as a prior inside the DS model | Grouped version adopted (0.8056); weight still 2-5x below the optimum; the added value is the weight size | `run_20260929/RED_TEAM.md` |
| 462 | `per_dataset_fit_v1` | Omri's contract: fit per model per dataset | Free on PRMBench; ProcessBench drops position by itself (32/32); per-cell prior latches onto a step-0 artefact; first-error readout fails | `run_20260929/RED_TEAM.md` |

Every experiment from Step 456 on: frozen PROTOCOL.json committed before the run (amendments recorded inside it, each dated and
marked before/after the full run), fold-0 or two-cell smoke, independent pre-run review (from Step 460), full run, red team of three
agents (recomputation / coverage / nulls and math), SUMMARY.md, HISTORY/PROGRESS/LESSONS, commit and push.

## 2. Code (all under `scripts/experiments/` unless stated; tests under `tests/`)

- `ds_group_weights.py` (hem_fit, group_matrix, weighted_group_score, mle_group_weights) - `test_ds_group_weights.py` (8 tests)
- `lsml_merge_step.py` (canon, absorb_merge, band_select) - `test_lsml_merge_step.py` (9 tests)
- `position_prior_ds.py` (position_bins, fit_pds, content_log_odds, latent_slope) - `test_position_prior_ds.py` (7 tests)
- `spectral_utils/external_banks_v4.py` + `scripts/{verify_external_banks_v4_source,extract_external_banks_v4}.py` - `test_external_banks_v4.py` (6 tests)
- Runners: `lsml_merge_step_run.py`, `algorithm_decisions_run.py` (+ `algorithm_decisions_parallel.sh`: 4 processes, fit/assemble split,
  identical to sequential to 2.2e-13), `partition_switch_run.py`, `position_channel_run.py`, `position_prior_run.py`,
  `per_dataset_fit_run.py`; external: `scripts/{fit,score,evaluate}_algorithm_external_v1.py`.
- Each runner writes RUN_STATUS.json (COMPLETE / STOPPED / CRASHED), CODE_MANIFEST.json (script, protocol and helper hashes) and
  refuses to overwrite a completed run. Smoke: set `ER_FOLDS=0` (or `ER_CELLS=...` for `per_dataset_fit_run.py`),
  `ER_DRAWS=2000`, `ER_NULL_PERMS=5`, and pass a smoke directory name.
- `per_dataset_fit_run.py` contains a code-path replay: its per-cell fit function, called on fold 0's pooled rows, reproduces
  Steps 457/460/461 to 8.9e-16 (hard stop). Use it as the base for future per-dataset experiments.

## 3. Artefacts NOT in git (must be archived before this worktree is removed)

About 2.3 GB, all git-ignored or untracked, all reproducible from the runners except where noted:
- `STEP_SCORES.npz` of every run (e.g. `algorithm_decisions_v1/run_20260928` 351 MB, `per_dataset_fit_v1/run_20260929` 75 MB,
  `position_prior_v1/run_20260929` 51 MB, `position_channel_v1/run_20260929` 27 MB, `lsml_merge_step_v1/run_20260928` 103 MB,
  `expectation_realization_v1/run_20260927_stage_b2*` 154 MB each). Later runners read the Step 457/460/461 STEP_SCORES as replay
  references.
- `BOOTSTRAP_DELTAS.npz` (lsml_merge_step 142 MB, expectation_realization, er_generality).
- `algorithm_external_v1/run_20260929/*/PREDICTIONS_UNSEALED.json` (1 + 16 + 16 MB): their digest is in SEAL.json; losing them
  breaks the seal audit trail. Not reproducible without the private external inputs in
  `M/scratch/external_generalization_private/inputs`.
- `external_banks_v4/*/FEATURES.npz`, `source/FEATURES.npz`, `reference_arrays/*.npy` (hashes in `reference_arrays/HASHES.json`).
- **Pool inputs.** `algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json` points `pool_z` / `pool_names` to a temporary session
  scratchpad. A hash-verified backup is now in `results/algorithm_decisions_v1/inputs_backup/` (pool_z.npy git-ignored, 57 MB;
  HASHES.json and pool_names.json committed). They are built by `results/indbank_lsml_prmbench_v1/pool_structure.py`.
Suggested archive: `rclone copy` of these paths to `gdrive:hallucination_detection/claude_label_free_line_2026-09/` (Omri
authorized Drive storage for large outputs, CLAUDE.md 2026-09-24), then verify sizes and hashes before removing the worktree.

## 4. Open decisions for Omri
1. Digit features: CLAUDE.md (2026-09-17) excludes them; Omri approved them for the external runs; all banks in 457-462 contain them.
2. Which next direction to run (Research_Directions, 2026-09-30 section): the filter's position decision as the gate for the
   prior; a position weight that does not re-weight the content; artefact removal; the external check under the per-dataset
   contract; untouched confirmation (MedPRMBench).
3. The advisor report is paused until the findings from the other conversations are collected.

## 5. Known traps (from this line's red teams)
- Fit per (model, dataset) on its own answers; never share fitted quantities across datasets (feedback memory, LESSONS 2026-09-29).
- Every position-related gain carries the step-index row and the same-length swap null; a position channel gains more under the
  swap by construction, so the content share is descriptive only.
- "Kept by the filter" needs a random-channel floor and a flipped control; diagnostics use the rows the method fits on.
- An estimated weight needs a post-hoc dose curve beside it; a structured term needs a same-size scalar comparison.
- A stated failure mechanism needs the intervention that tests it before it is written.
- Load npz files once outside loops; write multi-line patches with the Write tool (a bash heredoc parse error runs nothing).
