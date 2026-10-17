# Line status proposal: claude/er-generality-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status: SUPERSEDED (by `claude/ssl-pseudolabel-residual-v1`, which contains this branch and continued the same question in Steps 456-457)

Owner / last updated: no owner session. Tip 35f97a59e (2026-09-28), identical on origin and contained in `origin/claude/ssl-pseudolabel-residual-v1`. The worktree `.worktrees/er-generality-v1` still exists, although its SUMMARY says it was deleted after the merge.

## Steps on this branch

- Step 455 [Claude, er_generality] - Omri's concern was that the stage-B method was tailored to exactly 13 channels. On four pre-existing banks (13, 20, 32 and 51 channels) with the frozen stage-B/B2 chain and no parameter changed: the Dawid-Skene filter's channel selection transfers (it removed every channel with true balanced accuracy below 0.475 and never one at 0.52 or above), but its score gain, the discovered grouping and L-SML do not. Source: `results/er_generality_v1/SUMMARY.md`, `RED_TEAM.md`, `PROTOCOL.json`, `run_20260927/`.

## Evidence

`results/er_generality_v1/SUMMARY.md` @ `claude/er-generality-v1` (PRMBench within-answer AUC on 6,030 answers; PRMScore; ProcessBench gate-free localization macro over 8 cells; run COMPLETE with 0 failed fits; the 13-channel bank reproduces stage B to 8.9e-16 and the 20/32-channel banks reproduce Steps 438/439 exactly; three independent agents in `RED_TEAM.md` reproduce every number):
- Filter then average minus average of all channels: 20 channels +0.0008 (adjusted interval [-0.0018, +0.0033], no effect); 32 channels +0.0035 ([+0.0006, +0.0063], 86% of it from 61 answers); 51 channels +0.0128 (95% [+0.0103, +0.0152], about 70% positional).
- Equal weights per discovered group are below plain averaging after the filter: -0.0068 (20 channels), -0.0030 (32), -0.0109 (51).
- L-SML on all channels collapses on the 32-channel bank (0.6722 against 0.7440 for the average); after the filter it is still 0.0290 below plain averaging.
- Frozen interpretation rules (`PROTOCOL.json`): filter_generalizes not established; DS_beats_simple_filter not established; binary_grouping_helps no; binary_beats_continuous_partition yes; lsml_beats_averaging no.

## What failed (implementation) vs what is still open (direction)

Exhausted on these banks: the transfer of the stage-B grouping and of L-SML to larger banks. These are the same L-SML variants the ssl status file marks SATURATED (8 of 8 banks below plain averaging in Step 457).

Not failed: the label-free filter's selection, which became part of the frozen candidate (filter then plain average, Step 457).

Already followed up on the ssl branch: the "flip instead of drop" idea at the end of the SUMMARY was tested in Step 456 and hurt (32 channels -0.0157; provenance note section 4.1 row 6). The merge step (Step 456) and algorithm decisions (Step 457) continue this line.

Still open (carried by the ssl line, not this branch): label-free channel-quality estimates are biased (estimated prevalence 0.28-0.30 against a true 0.14 on every bank) and channels near the 0.5 cut have no margin.

## Reopening conditions

As for the L-SML variants in the ssl status file: only with channels that are conditionally independent given the label (for example a second model or internal layer states), since the current channels share one output distribution. For the filter's gain specifically: a bank where the reversed channels carry content beyond step position (the 51-channel gain is mostly positional).

## Dependencies other lines have on it

- The ssl line's Steps 456-457 build on its banks and runner; `scripts/experiments/er_generality_run.py` is snapshotted inside `results/lsml_merge_step_v1/*/SOURCE_SNAPSHOT/` on the ssl branch.
- No other line imports from this worktree.

## Outside git but needed

- `results/er_generality_v1/run_20260927/STEP_SCORES.npz` (46.9 MB) and `BOOTSTRAP_DELTAS.npz` (85.9 MB), git-ignored, exist in both `.worktrees/er-generality-v1` and `.worktrees/ssl-pseudolabel-residual-v1`. The ssl copies are listed with these sizes in the ssl Drive archive manifest (`docs/archives/DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.csv` on the ssl branch; MD5 f1abce77... and c368d30d...). This check compared sizes only, not hashes, against the er-generality worktree copies.
- `results/er_generality_v1/smoke_fold0/STEP_SCORES.npz` (10.2 MB) and `BOOTSTRAP_DELTAS.npz` (1.8 MB) exist only in the er-generality worktree; smoke run, regenerable, not a cited result.
- Inputs read by absolute paths in `scripts/experiments/er_generality_run.py`: the depth, token-probability, readout-quickest (`results/step_evidence_v1`) and cumulative-vote (`ct7_profiles_v1`, `cvf_v2`) worktrees; the source pool `pool_z.npy` / `pool_names.json` in session 2d14a8c9's temp scratchpad (line 124; archived copies are listed in the external-line status file); and the stage-B scores `.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz` (git-ignored, in the ssl Drive archive).
