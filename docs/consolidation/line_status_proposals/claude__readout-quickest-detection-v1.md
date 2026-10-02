# Line status proposal: claude/readout-quickest-detection-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- Readout, decision-rule and per-channel-evidence levers on the frozen eleven-channel step bank (own Steps 428-432): SATURATED
- The branch as a container of lineage B (the 2026-09-17 local `master` consolidation and the 2026-09-22 `consolidation/fusion-2026-09-22` merge): SUPERSEDED as a development line (by `lsml-ct7-levers-run`, which holds Steps 428-437 and all of this branch's result directories except the three noted below)
- Frozen step-level population and score files under `results/step_evidence_v1/` and `results/readout_family_v1/`: TOOL-ONLY (read by newer lines through this worktree's path)

Owner / last updated: no owner session. Tip cf70933d1 (2026-09-23). No origin ref: the push is blocked by the GitHub LFS budget (lineage B tracks `dataset_cache/**/*.pkl` in LFS). The branch head is one of the 7 heads in the Drive bundle `lfs_blocked_branches_2026-10-01.bundle` (verified 2026-10-01, `docs/consolidation/RECONCILIATION_2026-10-01.md`).

## Steps on this branch

Own steps (also present, as files and HISTORY blocks, on `lsml-ct7-levers-run`):
- Step 428 [Claude][readout-qd] - PRMBench errors persist only weakly; the long-chain collapse is an argmax competition effect, not signal loss; aggregation beats every spread statistic; a first-crossing rule never beats the argmax (0 of 69 locators). Source: `results/quickest_detection_diagnostics_v1/`.
- Step 429 [Claude][readout-qd] - Readout family on the frozen eleven-channel bank: the label-free consensus readout choice loses -5.67 points [-7.22, -4.10] against top-5 and falls below its shuffled-token null; label-selection over 17 readouts equals selection over 7; fitting on probability mass repairs the learned soft arms only up to equal weighting (+0.3 [-0.45, +1.07]); best new arm 3.5 points behind CT7. Source: `results/readout_family_v1/`.
- Step 430 [Claude][competition] - The channels drift towards the START of the answer, which explains the early misses but not the late ones; no readout family misses different answers than top-k beyond the shuffled null; no non-max decision rule beats the argmax (earliest-of-top-2 -6.7 points on CT7). Source: `results/competition_diagnostics_v1/`, `docs/research_notes/generation_drift_hypothesis_2026-09-22.md`.
- Step 432 [Claude][competition] - Per-channel likelihood-ratio evidence from pseudo-labels: the mechanism is real (half the decisions change, +17.5 points over the random-seed null) but never beats the linear fusion it is seeded from on ProcessBench (top-5 -0.60 [-2.35, +1.12]); its label-using ceiling is below label-free equal fusion; on PRMBench within-answer ranking top-5 position evidence reaches 0.7674 (+0.014 over the bank's frozen equal fusion, 0.005 below CT7) while PRMScore falls 5 points. Source: `results/step_evidence_v1/`. (Step 431 was planned and closed without running, see Step 430.)

Inherited from merged branches (each has, or is covered by, its own status file): 2026-09-17 local `master` (Steps 390-420 of lineage B, including `Step 418 [Claude] - Freeze development candidate CT7`); `claude/token-probability-fusion-v1` (Steps 423-427); `origin/claude/token-axis-fusion-sampling-3i9r2u` (Steps 423-424); `codex/cumulative-vote-fusion-v2` (`Step 421 [Codex]` and the untagged v2 block); `codex/token-local-fusion-optimization-v1`.

## Evidence

- `results/readout_family_v1/` and `results/step_evidence_v1/` @ `claude/readout-quickest-detection-v1` (the same directories, 27 and 28 tracked files, are on `origin/claude/lsml-ct7-levers-v1`): 300 outer + 320 inner jobs for Step 429 and 30 outer + 40 inner for Step 432; ProcessBench gate-free localization on the 4,442 erroneous answers over 8 cells; PRMBench within-answer AUC on 6,030 answers; anchor parity against Codex's run on 61 shared methods (Step 429) and on 25 shared rows (Step 432).
- PROGRESS.md on this branch, block "Steps 430-432": "the readout, readout-as-voter, non-max-rule and per-channel-evidence levers on the eleven-channel bank are closed (Steps 429-432)". This is the branch's own wording about these levers on this bank. The proposal reads it as SATURATED for the tested implementations, not as a closed direction.

## What failed (implementation) vs what is still open (direction)

Exhausted on current inputs: readout choice (17 readouts, label-free or label-selected), readout-as-voter fusion, non-max decision rules, first-crossing rules and per-channel likelihood-ratio evidence, all on one-pass entropy/probability channels with about 1.8 effective independent views. The record shows every lever at or below equal-weight fusion or CT7 on ProcessBench.

Still open, as this branch's own PROGRESS names it:
- Genuinely new independent views for the 1.8-effective-view ceiling (attention flow, layer-wise views from the white-box line). The white-box locator stage has since ended negative (`claude/whitebox-layer-views-v1` status file); its answer-gate stage is OUT OF SCOPE FOR NOW (Omri, 2026-10-02; not closed).
- The no-error gate.
- A position-conditional null is the recorded label-free fix for early misses if a future locator needs one.
- The PRMBench-only gain of position evidence (0.7674) foreshadows the ssl line's position work (Steps 460-462), where it is carried now.

## Reopening conditions

1. A new channel that does not come from the same one-pass output distribution and raises the conditional participation ratio while having useful single-channel ProcessBench accuracy.
2. A different decision component (gate, or a readout for late misses on a new series), judged on exact localizations gained against lost.
3. Drift-based late-bias corrections are not a reopening path on the current evidence: the late competing peaks are real excursions above a lower null (Step 430). New evidence against that finding would reopen them.

## Dependencies other lines have on it

Read by absolute path `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/` (OOF_ANSWERS.csv, OOF_STEP_SCORES.npz, INPUT_FREEZE.json) as the frozen source population by: 33 script files on the ssl line outside source snapshots (among them `calfix_common.py`, `expectation_realization_run.py`, `er_stage_b_run.py`, `er_generality_run.py`, `algorithm_decisions_run.py`, `lsml_merge_step_run.py`, `per_dataset_fit_run.py`), `tensor_mom_stage_a_run.py` (estimator line), the family-tail handoff (section 4) and `b16_fit.py` on `claude/self-generated-step-labels-v1`. Do not remove this worktree before those runners are pointed elsewhere.

The Step 432 profiles are hard links to `results/readout_family_v1/` (do not remove either, per this branch's PROGRESS).

## Outside git but needed

About 1.56 GB of git-ignored files in `.worktrees/readout-quickest-detection-v1` (2,093 ignored paths; figure from the 2026-10-01 consolidation review), including:
- `results/readout_family_v1/shuffled_full.npy` and `profiles_full.npy` (217.8 MB each), `OOF_STEP_SCORES.npz` (98.6 MB), `profiles.npy` (89.7 MB), `BOOTSTRAP_PRIMARY_DRAWS.npz` (29.2 MB), `channel_merits.npy` (20.6 MB), `shuffled_top5.npy` (12.8 MB).
- `results/step_evidence_v1/OOF_STEP_SCORES.npz` (53.9 MB) and `BOOTSTRAP_PRIMARY_DRAWS.npz` (11.9 MB).
- `results/quickest_detection_diagnostics_v1/B1_DELAY_CURVES.csv` (41.6 MB).

No Drive archive record for these was found in this check; the 2026-10-01 upload did not include other worktrees' ignored files (reconciliation: "not yet uploaded"). Regenerable by the Step 428-432 runners, which read `TOKEN_MATRICES.npz`, `CT7_DEV_SCORES.npz` and `OOF_SCORES.npz` from `.worktrees/token-probability-fusion-v1` and the frozen profiles from `.worktrees/cumulative-vote-fusion-v2` (this branch's PROGRESS, Steps 428-429 block). The frozen files and their hashes are the record that newer lines replay against.
