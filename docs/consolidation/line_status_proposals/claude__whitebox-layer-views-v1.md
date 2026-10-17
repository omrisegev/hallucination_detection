# Line status proposal: claude/whitebox-layer-views-v1

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).
>
> How to read the status words: they describe the tested implementations and the branch, not the research direction. SATURATED = the implementations tested on current inputs gave no measured gain. IMPLEMENTATION-NEGATIVE = this implementation lost to its matched reference. SUPERSEDED = the branch's content is carried by a newer line. TOOL-ONLY = kept as code or data that other lines use. OUT OF SCOPE FOR NOW and DEFERRED are Omri's words (2026-10-02) and mean "not closed, not a next step". None of these words closes a research direction.
>
> Drafted 2026-10-01 by the consolidation session ae2dd164 (no owner session exists for this line); Omri's 2026-10-02 decisions applied on 2026-10-02.

Proposed status:
- Internal-layer depth field as a step locator (Stage 1b): SATURATED for the tested 864-column reduction (ended by its own pre-registered kill rule; even the label-selected best of 848 depth columns stays below the production locator; other depth quantities are untested, so the direction is not closed)
- Internal-layer geometry as an answer-level no-error gate (Stage 1a and the licensed gate-fusion stage): OUT OF SCOPE FOR NOW (Omri, 2026-10-02; not closed and not a next step; never run; it would also need a label-free orientation rule)
- Step-length follow-up requested in Stage 1b section 5: OUT OF SCOPE FOR NOW (Omri, 2026-10-02; not closed; requested, not run)
- Per-layer field extraction on AIRCC (Step 421) and the two reductions: TOOL-ONLY

Owner / last updated: no owner session. Tip f2f533795 (2026-09-21), identical on origin. Worktree `.worktrees/whitebox-layer-views-v1`.

## Steps on this branch

- Step 421 [Claude] - White-box per-layer field extracted on AIRCC for all 13,769 localization answers (ProcessBench x Qwen3-4B/8B, PRMBench x Qwen3-8B), teacher-forced; per token, per layer, per tap: lens entropy, lens log-probability of the provided token and of the top-1, KL to the final layer, residual norm; 0 failures, 5.5 GB, about 18.4 GPU-hours. Source: `docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md`. (This HISTORY block is also on the ssl branch.)
- Stage 1a (no HISTORY step number; commit af0d0c2b1) - Gate kill test: not killed, not cleared. One fixed column (`hidden_dist_adjacent.layer_32`) ties the production gate on ProcessBench and supplies a working detector on PRMBench, where the production gate is constant; but its orientation is label-derived. Source: `results/whitebox_layer_views_localization_v1/RESULTS_STAGE_1A.md`, `KILL_TEST_1A.json`, `AXIS3_ERROR_COMPLEMENTARITY.json`.
- Stage 1b (no HISTORY step number; commit f2f533795) - Locator kill test: depth beats the final-layer lens entropy in 0 of 9 cells; the locator branch ends as pre-registered. Source: `results/whitebox_layer_views_localization_v1/RESULTS_STAGE_1B.md`, `KILL_TEST_1B.json`.

## Evidence

All @ `claude/whitebox-layer-views-v1`, `results/whitebox_layer_views_localization_v1/`; population 13,769 answers / 145,597 steps / 9 cells / 10,477 erroneous answers / 3,483 source groups; 10,000 shared paired source-group draws; development evidence.
- `RESULTS_STAGE_1B.md` section 3: best label-free depth column minus final-layer lens entropy, gate-free step localization: ties in 7 cells, losses in 2 (omnimath q8 -0.049 [-0.088, -0.010]; PRMBench -0.138 [-0.154, -0.123]), wins in 0. Section 4 (mean over 8 ProcessBench cells): final-layer lens entropy 33.61, label-selected depth oracle (best of 848 per cell) 35.46, production CT7 locator 39.89.
- `RESULTS_STAGE_1A.md` section 3.2: fixed geometry column minus the production gate, answer-level AUROC: 8 of 8 ProcessBench intervals include zero (mean 0.727 against 0.710); PRMBench +0.160 [+0.150, +0.170], where the production gate opens 0 of 6,969 answers. Section 3.3: with the registered contract's orientation the same column scores 0.273; the 0.727 depends on a sign chosen with labels. Section 4, axis 3: failure correlation with the production gate 0.402 at matched opened fraction; 630 answers the production gate gets wrong and the geometry gets right (1,048 the other way).
- `RESULTS_STAGE_1B.md` section 5: within-answer rank residualization against step length moves final-layer lens entropy 33.61 to 16.79, the best depth column 31.56 to 16.99 and CT7 39.89 to 20.09 (chance 16.58); a crude control, ordering only.
- Related, on another line: adding four depth principal components to the eleven-channel step bank lowers ProcessBench gate-free localization by 6.04 points (0 wins, 6 losses over 8 cells), `results/bank_plus_depth_fusion_v1/RESULTS.json` @ `rescue/main-checkout-loose-files-2026-10-01` (see the `claude/depth-feature-fusion-v1` status file).

## What failed (implementation) vs what is still open (direction)

Exhausted on current inputs: depth as a step locator. This is the pre-registered verdict, it extends the earlier final-answer result (Steps 243-245b: per-layer lens fusion below the final-layer statistic), and the label-selected ceiling over 848 columns is still 4.43 points below CT7. Adding depth to the step bank also lost (above).

Not closed, but OUT OF SCOPE FOR NOW by Omri's decision of 2026-10-02 (Stage 1b section 7 had said: "What the white-box line is now: one open experiment, the answer-gate fusion"):
- Answer-gate fusion of the geometry with the production gate, judged on exact localizations gained against lost at a matched opened fraction, not AUROC.
- Blocking before any candidate: a label-free orientation rule for the geometry columns (Stage 1a section 5). The registered contract's declared direction for `distances_and_convergence` is wrong for this feature on this task and should be fixed upstream.
- The step-length follow-up (Stage 1b section 5), which "bears on every step-level readout in the project". The 2026-10-01 consolidation review keeps it as a separate checklist item and warns against reading the crude rank residualization as a causal estimate.
- Not computed: the participation ratio of the views (Stage 3 of the plan); moot for the locator, still relevant for the gate.

How this sits with other lines: the decision-rule line (`claude/decision-rule-v1`) has an open question to Omri about which answer gate enters the frozen candidate; the white-box gate would be one more option there. That choice is Omri's.

## Reopening conditions

- Locator: a depth quantity not available in the current 864-column reduction (for example attention flow or a different reduction of the raw field) that beats the final-layer statistic in a matched, pre-registered contrast. The raw per-token field is still on Drive, so a new reduction needs CPU, not new inference.
- Gate: only if Omri brings the stage back into scope; it would also need a label-free orientation rule first.

## Dependencies other lines have on it

- `scripts/run_bank_plus_depth_fusion_v1.py` (rescued onto `rescue/main-checkout-loose-files-2026-10-01`; lives in the depth worktree) reads `.worktrees/whitebox-layer-views-v1/results/whitebox_layer_views_localization_v1/STEP_LEVEL.npz` (lines 104-105).
- The readout-quickest PROGRESS names "layer-wise views from the white-box line" as one of the remaining independent-view levers.
- No ssl, ct7-levers or decision-rule runner imports from this worktree.

## Outside git but needed

- `.worktrees/whitebox-layer-views-v1/results/whitebox_layer_views_localization_v1/STEP_LEVEL.npz` (509.2 MB on disk; documented as 486 MB, sha256 f25c47f7..., cluster job 263494, code commit 463cd9b41) and `ANSWER_LEVEL.npz` (305.4 MB, sha256 909ac8b6...), git-ignored. No Drive archive record found in this check; the 2026-10-01 review lists 814,609,017 ignored bytes in this worktree as not yet uploaded. Regenerable on AIRCC from the raw field with `cluster/reduce_layer_views_step_level.py` and `cluster/reduce_layer_views_answer_level.py` (step reduction 296.6 s).
- Raw per-layer field (5.5 GB): `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/{pb_layer_views_qwen3_4b,pb_layer_views_qwen3_8b,prmbench_layer_views_qwen3_8b}/` on AIRCC, one `rows/<row_id>.npz` per answer, backed up to `gdrive:hallucination_detection/cluster_results/` under the same names (`docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md` section 2). The handoff warns the cycle-2 tree is likely to be retired; the Drive copy is the durable one.
