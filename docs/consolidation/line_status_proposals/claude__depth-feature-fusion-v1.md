# Line status proposal: claude/depth-feature-fusion-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- Adding internal-layer depth summaries to the eleven-channel step bank (`bank_plus_depth_fusion_v1`): SATURATED for this implementation (together with the white-box Stage 1b locator result, the tested depth reductions add nothing to step localization; other reductions of the depth field are untested, so the direction is not closed)
- Gate-free step-localization engine (`spectral_utils/gate_free_sla.py`), the eleven-channel step-bank baseline (`step_level_bank_baseline_v1`, the origin of "bank11 L-SML") and this worktree's copy of the canonical fusion modules: TOOL-ONLY

Owner / last updated: no owner session. One own commit, 63100b109 (2026-09-19), on top of the 2026-09-17 local `master`. No origin ref: the push is blocked by the GitHub LFS budget; the head is one of the 7 heads in the verified Drive bundle `lfs_blocked_branches_2026-10-01.bundle` (`docs/consolidation/RECONCILIATION_2026-10-01.md`). The two experiments below were never committed on this branch; their scripts, results and score files were rescued in commit 755b562a2 on `rescue/main-checkout-loose-files-2026-10-01` (on origin).

## Steps on this branch

No HISTORY step numbers were assigned on this line.
- Commit 63100b109 "Gate-free SLA engine, validated against a reported number rather than a fixture" - Mind-the-Gap's native step-level localization accuracy (tolerance 0, erroneous answers only, within-answer argmax, no no-error decision). Replaying Step 422's saved out-of-fold scores returns 35.92 and 32.59 with maximum per-cell difference 0.0 on the identical 4,442 erroneous ProcessBench answers. Source: `spectral_utils/gate_free_sla.py`, `scripts/validate_gate_free_sla.py`.
- `step_level_bank_baseline_v1` (uncommitted here, rescued) - Eleven token channels read out per step and fused at step level, five donor source-fold fits: equal 34.95, continuous L-SML 34.78, CT7 39.89 (gate-free, mean of 8 ProcessBench cells); continuous minus equal is a tie in 8 of 9 cells and a loss on MATH q4 (-0.0286 [-0.0503, -0.0082]). The `continuous` arm is the "bank11 L-SML" later frozen and sent to the external benchmarks. Source: `results/step_level_bank_baseline_v1/RESULTS.json`.
- `bank_plus_depth_fusion_v1` (uncommitted here, rescued) - Adding one depth principal component per lens quantity (four columns from 424 eligible depth columns) to the bank: -6.04 points with equal weights (0 wins, 6 losses over 8 cells) and -7.07 under continuous L-SML (0 wins, 4 losses); the alternative "least redundant four columns" variant -1.21 (equal) and -4.14 (L-SML). `depth_adds: false`. Source: `results/bank_plus_depth_fusion_v1/RESULTS.json`.

## Evidence

- `results/bank_plus_depth_fusion_v1/RESULTS.json` @ `rescue/main-checkout-loose-files-2026-10-01`, key `kill_rule`: four pre-declared contrasts, all negative; population 13,769 answers / 145,597 steps / 10,477 erroneous / 3,483 source groups; 10,000 paired draws. The runner's docstring records that the depth columns are not redundant with the bank (median R-squared 0.49 against the eleven channels) and that adding them raises the participation ratio from 4.411 to 5.672, which it explicitly gives no authority.
- `results/step_level_bank_baseline_v1/RESULTS.json` @ same ref, keys `pb_mean` and `vs_equal`: same population and draws; effective views of the bank 4.411.
- White-box Stage 1b (`claude/whitebox-layer-views-v1`, `RESULTS_STAGE_1B.md`): label-selected best of 848 depth columns 35.46 against CT7 39.89.
- `docs/reviews/LSML_PRMBENCH_MONTH_REVIEW_20260923_HE.md` @ `rescue/main-checkout-loose-files-2026-10-01`, lines 7-9: identifies `step_level_bank_baseline_v1`, arm `continuous`, as the experiment the bank11 L-SML line continues, and notes the worktree was kept because its outputs were not in git.

## What failed (implementation) vs what is still open (direction)

Failed: depth summaries (first principal components per lens quantity, or the least redundant columns) appended to the eleven-channel step bank. Together with the white-box locator kill test, the depth field does not add step-localization signal in the current 864-column reduction.

Not failed: the engine and the baseline, which are measurement tools and the origin record of bank11 L-SML.

Still open: a different reduction of the raw depth field. The white-box answer-gate stage is OUT OF SCOPE FOR NOW (Omri, 2026-10-02; not closed, not a next step; see the white-box status file). The step-level baseline already showed continuous L-SML tying equal weighting on ProcessBench; its later PRMBench and external advantages are recorded on the external and ssl lines.

## Reopening conditions

A depth quantity outside the current reduction (or attention-based views) that beats the final-layer statistic as a single locator in a pre-registered contrast, before it is added to a bank. The same participation-ratio warning applies: new variance is not new signal.

## Dependencies other lines have on it

- Code imported by absolute path: 29 script files on `claude/ssl-pseudolabel-residual-v1` mention this worktree (excluding source snapshots; for example `algorithm_decisions_run.py`, `bank20_lsml_run.py`, `declared_joint_run.py`, `core_virtual_lsml_run.py`, `calfix_evaluate.py`), `tensor_mom_stage_a_run.py` on the estimator line and `er_generality_run.py` put `.worktrees/depth-feature-fusion-v1` on the Python path to import `spectral_utils.lsml_gate_locator_research`, `spectral_utils.joint_lsml` and `spectral_utils.fusion_utils` (the copy with the small-sample guard). These three modules are byte-identical on `origin/claude/lsml-ct7-levers-v1` and the local `master`; `gate_free_sla.py` and `validate_gate_free_sla.py` exist only on this branch (and in the bundle).
- The rescued runners on the rescue branch import `gate_free_sla.py`, so they cannot run from the rescue branch alone.
- The external line's frozen bank11 bundle descends from the `continuous` arm here.
- `run_bank_plus_depth_fusion_v1.py` reads the white-box `STEP_LEVEL.npz` from `.worktrees/whitebox-layer-views-v1`.

## Outside git but needed

- `results/step_level_bank_baseline_v1/STEP_SCORES.npz` (3.6 MB) and `results/bank_plus_depth_fusion_v1/STEP_SCORES.npz` (8.0 MB): force-added in the rescue commit 755b562a2 (blobs 9cff3288... and b78b2080...) because they were the only copies; still present untracked in `.worktrees/depth-feature-fusion-v1`. Verified in this check: `git hash-object` of both worktree files equals the rescued blobs, so the review's "verify the rescued blobs" item is satisfied for these two files.
- Inputs of the two runners (`scripts/run_step_level_bank_baseline_v1.py` lines 79-85): `.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz` and `TOKEN_MATRICES.npz`, `.worktrees/token-probability-fusion-v1/results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz`, and from the main checkout `results/localization_full_benchmark_v3/evaluation/` and `results/localization_source_group_audit_v1/FOLDS_V2.json`; plus the white-box `STEP_LEVEL.npz` for the depth runner.
