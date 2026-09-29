# Line status: lsml-ct7-levers-run

Status: SATURATED

Owner / last updated: the Claude session working in `.worktrees/lsml-ct7-levers-run` (session 036ef7ec), 2026-09-30. This session owns Step 447 only. The other steps below were carried by earlier sessions and are summarized from this branch's HISTORY.md and result files.

## Steps on this branch

- **Step 433 [Claude][ct7-levers]** (earlier Claude session). Built the three L-SML levers against CT7's failures (family-equal weighting of CT7's seven views, token-level L-SML before the readout, and a window representation), with pre-registered protocols and synthetic tests only; no real-data number. Source: HISTORY.md Step 433; tests `tests/test_family_equal_readout.py`, `tests/test_ct7_token_streams.py`, `tests/test_window_moment_bank.py`.
- **Step 434 [local][ct7-levers]** (not this session). Family-equal weighting of CT7's seven views is a null on ProcessBench: the primary long-answer SLA contrast is +0.36 pp [-3.13, +3.75], and every family arm minus CT7 on SLA and F1 has Holm p = 1 over 54 contrasts. PRMBench within-answer AUROC gains +0.77 pp (Holm .005). Source: `results/ct7_family_equal_v1/UNCERTAINTY.json`, `SUMMARY.csv`.
- **Step 435 [Claude][ct7-levers]** (earlier Claude session). Review of Step 434 and gate correction A1 for the token extraction; no new number. Source: HISTORY.md Step 435.
- **Step 436 [local][ct7-levers]** (not this session). Token-level L-SML on CT7's streams is below equal weighting on the same streams (-0.83 pp SLA [-1.93, +0.25]) and below CT7 in every arm (-1.60 to -2.69 pp SLA). The window representation passes its participation-ratio gate (3.748), but every window arm is 10.19 to 10.59 pp SLA below CT7 (Holm .0045), and its learned weights buy nothing over equal weights. Source: `results/ct7_token_lsml_v1/UNCERTAINTY.json`, `results/window_representation_b3_v1/WINDOW_PR.json` and `results/window_representation_b3_v1/fusion/`.
- **Step 437 [local][prm-measure]** (not this session). The supervised Qwen2.5-Math-PRM-7B, measured beside CT7 on PRMBench, is ahead on within-answer AUROC (+2.88 pp [+2.14, +3.60]) and on PRMScore at q80 (.6804 vs .6457). It is below CT7 on the argmax first-error hit (57.78% vs 61.14%, 6,035 erroneous answers), and the two hit sets overlap on only 39.02% of answers. Source: `results/prm_vs_ct7_prmbench_v1/MEASUREMENT.json`, `PRMSCORE.json`.
- **Steps 441-442 [Claude, token-level tail L-SML artifacts]** (earlier Claude session; the narrative is on the ssl branch). Token-level tail-mark L-SML on CT7's streams is below equal weighting (-0.17 PRMScore [-0.27, -0.08]), and fusing tokens before the readout costs -0.92 PRMScore against reading out first. Reading out each stream and then averaging gives the best PRMBench PRMScore on the branch (64.88). Source: `results/ct7_token_tail_lsml_calfix_v1/run_20260924_1550/METRICS.csv`, `CONTRASTS.csv`.
- **Step 447 [Claude, PB tail weights]** (this session). The tested recipe learns L-SML weights from the binary top-20% tail marks and applies them to the continuous step signals. On ProcessBench it does not beat equal weighting on any representation: +0.08 pp (CT7 step readouts), -0.13 pp (family15) and -0.20 pp (bank11) first-error hit, all with Holm p = 1.0 over 10 contrasts. Fitting on ProcessBench rows only does not help either. Source: `results/pb_tail_weights_v1/run_20260927_0043/SUMMARY.json`, `SUMMARY_METRICS.csv`; report `results/pb_tail_weights_v1/REPORT_HE.md`.

## Why the line is saturated

**What was tried on this branch.** Every attempt kept CT7's streams (or the step banks) fixed and changed how they are weighted:
- declared family partitions of CT7's seven views (Step 434);
- token-level L-SML before the readout (Step 436);
- a window representation with learned weights (Step 436);
- token-level tail-mark L-SML (Steps 441-442);
- step-level tail-mark weights on the CT7 step readouts, family15 and bank11, fitted either on pooled rows or on ProcessBench rows only (Step 447).

**Evidence that it is exhausted:**
- **Step 434:** every family-weighting arm minus CT7 on ProcessBench SLA and F1 has Holm p = 1. That is 54 contrasts on 13,769 answers, 4,442 of them erroneous ProcessBench answers (`results/ct7_family_equal_v1/UNCERTAINTY.json`).
- **Step 436:** every token-level arm is below CT7, and every window arm is 10.19 to 10.59 pp below CT7 (`results/ct7_token_lsml_v1/UNCERTAINTY.json`, `results/window_representation_b3_v1/fusion/`).
- **Step 442:** the token-level tail recipe is below equal weighting on PRMBench PRMScore, on 6,211 non-control answers (`results/ct7_token_tail_lsml_calfix_v1/run_20260924_1550/CONTRASTS.csv`).
- **Step 447:** none of the 10 pre-declared contrasts favors the learned weights on the ProcessBench first-error hit (4,442 erroneous answers, 8 cells, `results/pb_tail_weights_v1/run_20260927_0043/SUMMARY.json`). The equal average of the per-stream readouts remains the best row on both benchmarks: 41.36 ProcessBench SLA and 64.88 PRMBench PRMScore.

**Summary.** On these one-pass streams, no learned weighting has beaten equal weighting on ProcessBench, whatever the learning object: covariance, tail marks, family partitions, token or step level, pooled or ProcessBench-only fitting.

**What new evidence would reopen it:**
1. **A new stream that carries independent information.** It must not be another function of the same one-pass entropy and probability block. It should raise the conditional participation ratio and also have useful single-stream ProcessBench accuracy.
2. **Work on a different decision component.** One candidate is the no-error gate. In Step 447, the frozen CT7 gate beat every step-score threshold gate by 8.5 to 12.5 F1 points, including a threshold chosen with labels. The other candidate is the readout for late misses.
3. **An untouched dataset where equal and learned weights disagree in ranking.**

The error-type split measured on 2026-09-27 could motivate a type-aware combination with a supervised PRM. In it, this line's methods are ahead of the supervised PRM on redundancy errors, and the PRM is ahead on semantic errors (artifact https://claude.ai/artifact/EArjBd9q2aXnvwoAimYPWr). That would be a new line, not a continuation of this one.

## Outside git but needed

**Inputs:**
- `C:/Users/omris/TAU/hallucination_detection/results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz` (447 MB, main checkout, git-ignored).
  - Input to Steps 436, 441-442 and 447.
  - Not archived: only its `CT7_TOKEN_MATRICES.MANIFEST.json` is in Codex's Drive archive.
  - Regenerable with `scripts/diagnostics/extract_ct7_token_streams_v1.py`; the manifest records 1,117 s.
- `pool_z.npy` (60.6 MB, sha256 `d9abccfb...`) and `pool_names.json`.
  - Input to the family15 and bank11 rows of Step 447.
  - They live in another session's temp scratchpad: `C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad/`.
  - Archived in Codex's Drive archive `gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/evaluation_archives/family_tail_external_v1_results_a1e2f683cc6932a9.tar.gz`, member `reproduction/source_pool/`. Restore steps: `results/family_tail_external_v1/RESTORE.md` in the main checkout.
- `C:/Users/omris/TAU/hallucination_detection/results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy` (8.2 MB, main checkout).
  - Input to Steps 434 and 437.
  - Copied from `.worktrees/cumulative-vote-fusion-v2/`; its sha matches the tracked `PROFILE_VALIDATION.json`.
- Step 447 replays its reference rows against `.worktrees/ssl-pseudolabel-residual-v1/results/tail1_transfer_v3/run_20260924_2259/SCORES.npz`, which is git-ignored on the ssl branch. It also imports the frozen evaluator `calfix_common.py`, `calfix_evaluate.py` and `tail_calib_common.py` from that branch; their hashes are checked at run time.

**Git-ignored outputs on this branch, all regenerable by rerunning their scripts:**
- `results/pb_tail_weights_v1/run_20260927_0043/SCORES.npz` (60.6 MB, about 5 min CPU) and `BOOTSTRAP_PRIMARY.npz` (1.3 MB).
- `results/ct7_token_tail_lsml_calfix_v1/run_20260924_1550/SCORES.npz` (37.3 MB, about 6 min; needs the token matrices) and `BOOTSTRAP_PRIMARY.npz` (0.8 MB).
- Bootstrap and decision files of 0.1 to 4.8 MB in `results/ct7_family_equal_v1/`, `results/ct7_token_lsml_v1/`, `results/ct7_token_tail_lsml_v1/`, `results/window_representation_b3_v1/fusion/` and `results/prm_vs_ct7_prmbench_v1/`.

**Not needed for the branch:** the scripts that built the PRMScore breakdown page (`breakdown_data.py`, `build_breakdown.py`) are in this session's temp scratchpad. The page itself is published.

**Unpushed commits:** `1a3446b6e`, `4cae70d8a`, `9ebe914c2` and the commit that adds this file are not pushed to `origin/claude/lsml-ct7-levers-v1`.
