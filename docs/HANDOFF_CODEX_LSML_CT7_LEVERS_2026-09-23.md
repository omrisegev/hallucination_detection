# Handoff to Codex — three L-SML levers against CT7's failures (2026-09-23)

Branch `claude/lsml-ct7-levers-v1` (cut from `claude/competition-sync-2026-09-23` @ `49787d46`).
Omri's decisions (2026-09-23): implement here, not on the token-axis branch; item 4 is measurement
plus gated fusion; Codex runs the three protocols on the data and reports; Claude reviews.

Background: `docs/research_notes/ct7_anatomy_2026-09-23.md` (what CT7 is and where it fails) and
`docs/research_notes/lsml_against_ct7_failures_2026-09-23.md` (the levers). Every run is development
evidence on the frozen 13,769-answer population; no promotion language in any report.

## What is in the branch (all synthetic-tested here; no real number exists yet)

| item | code | protocol (read first) | config |
|---|---|---|---|
| shared | `scripts/experiments/ct7_levers_common.py` (light dataset, anchors, endpoints, strata intervals, freeze) | — | — |
| 2 | `spectral_utils/family_equal_readout.py`, `scripts/experiments/ct7_family_equal_v1.py` | `docs/experiments/CT7_FAMILY_EQUAL_V1.md` | `configs/ct7_family_equal_v1.json` |
| 3 | `spectral_utils/ct7_token_streams.py`, `scripts/diagnostics/extract_ct7_token_streams_v1.py`, `scripts/experiments/ct7_token_lsml_v1.py` | `docs/experiments/CT7_TOKEN_LSML_V1.md` | `configs/ct7_token_lsml_v1.json` |
| 4 | `spectral_utils/window_moment_bank.py`, `scripts/diagnostics/window_pr_measurement_v1.py`, `scripts/experiments/window_answer_local_fusion_v1.py` | `docs/experiments/WINDOW_REPRESENTATION_B3_V1.md` | `configs/window_representation_b3_v1.json` |

Tests: `python -B -m pytest -q tests/test_family_equal_readout.py tests/test_ct7_token_streams.py tests/test_window_moment_bank.py`
(12 tests; the frozen `tests/test_cumulative_vote_v2.py` still passes; nothing under `scripts/experiments/cvf_v2/`
or in the five hashed `spectral_utils` modules was edited). Dry runs: each driver has `--dry-run`
(item 4's fusion also `--force`), building a synthetic population through the real scorers.

## Inputs (absolute paths in the configs point at Omri's machine; edit if the layout differs)

- `results/localization_full_benchmark_v3/evaluation/{JOINED.json,JOINED.npz}` and
  `results/localization_source_group_audit_v1/FOLDS_V2.json` (tracked).
- `CT7_DEV_SCORES.npz` (sha `9d10d2ff…b430`; `.worktrees/token-probability-fusion-v1/results/chosen_token_calibration_v1/`).
- `results/cumulative_vote_fusion_v2/ct7_profiles_v1/{profiles.npy,PROFILE_VALIDATION.json}` (sha `d564ba43…ff674`)
  and `results/cumulative_vote_fusion_v2/step_lengths.npy` (optional longest-step control).
- `TOKEN_MATRICES.npz` (`.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/`), items 3-4.
- Raw pickles under `dataset_cache/` (item 3 extraction), the temporal worktree's
  `results/temporal_context_data_v1/` (verbatim BOCPD; optional, fallback recomputes), and
  `results/length_explicit_ct7_v1/bank/` (gate i; optional but recommended).
- `dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl` (PRMScore; optional).

## Run order and expected cost

```
# item 2 — seconds for the arms, minutes for the 10,000-draw bootstrap
python -B scripts/experiments/ct7_family_equal_v1.py --config configs/ct7_family_equal_v1.json

# item 3 — extraction ~2 h CPU once, then minutes
python -B scripts/diagnostics/extract_ct7_token_streams_v1.py --source-root C:/Users/omris/TAU/hallucination_detection ^
    --temporal C:/Users/omris/TAU/hallucination_detection/.worktrees/temporal-research-20260915/results/temporal_context_data_v1 ^
    --bank-dir C:/Users/omris/TAU/hallucination_detection/.worktrees/token-probability-fusion-v1/results/length_explicit_ct7_v1/bank ^
    --profiles C:/Users/omris/TAU/hallucination_detection/results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy ^
    --out results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz
python -B scripts/experiments/ct7_token_lsml_v1.py --config configs/ct7_token_lsml_v1.json

# item 4 — measurement minutes; fusion only if WINDOW_PR.json says gate_passed
python -B scripts/diagnostics/window_pr_measurement_v1.py --config configs/window_representation_b3_v1.json
python -B scripts/experiments/window_answer_local_fusion_v1.py --config configs/window_representation_b3_v1.json
```

Use `--smoke 30` on the extraction first (30 answers per cell) to check the raw-pickle plumbing; a
smoke output must not be passed to the driver.

## Asserts that must pass before any new row exists (do not relax them)

- CT7 replay: macro-F1 .41188745848863717 and within-AUC .7723966352864217 on 6,030 eligible answers
  (`ct7_levers_common.replay_ct7`); `profiles.npy` sha; its mean equal to CT7 to 1e-12.
- Item 3 gates: (i) masked Top10 of the five bank columns equals the frozen bank extraction exactly
  after the float32 cast; (ii) the BOCPD column replays CT7's view 5 (1e-8 verbatim / 1e-6 fallback).
- Item 4 anchor: conditional PR of the CT7 profiles within 0.01 of 1.80.
- `RUN_FREEZE.json` in each output directory: a changed freeze refuses the directory; use a new one.

## What to report back (per item)

`RESULTS.json`, `SUMMARY.csv`, `UNCERTAINTY.json` (the frozen bootstrap's contrasts with Holm),
the depth-stratum intervals (in `RESULTS.json["strata"]`), coverage counts (zeroed families, 2-step
answers; invalid tokens; fallbacks / native rows / abstentions), runtime per stage, and for item 3 the
per-fold weights, K, groups and negative weights, plus the conditional PR of the seven Top10 views.
Read each result against the predictions and the decision language written in its protocol; say
which prediction held and which did not. No candidate is promoted, frozen or added to CT7 from these
runs. HISTORY blocks are tagged `[Codex]` with the date; never renumber; merge HISTORY/PROGRESS by
union only.

## Things that will bite

- `spectral_utils/__init__.py` imports torch; the drivers register a bare package if the real import
  fails (`ct7_levers_common.ensure_spectral_package`), so they run in a numpy/scipy/sklearn/pandas
  environment. Tests use `tests/lever_imports.py` for the same reason.
- The frozen `cvf_v2.uncertainty.bootstrap` packs exactly eight ProcessBench cells; the light dataset
  keeps the real cell set. Its `UNCERTAINTY.json` is cached by a fingerprint of the method dicts.
- Item 3's `TokenData` asserts the token counts of `CT7_TOKEN_MATRICES.npz` equal the roster's; the
  extraction builds them from the same records, so a mismatch means a stale roster.
- Item 4: dense windows are stride 1 for scoring only; fits use the non-overlapping grid; PRMBench
  answers will fall back often (median step 24 tokens) and that is reported, not hidden.
