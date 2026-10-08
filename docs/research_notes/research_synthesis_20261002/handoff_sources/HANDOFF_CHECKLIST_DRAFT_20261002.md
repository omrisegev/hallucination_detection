# Research handoff checklist (draft, 2026-10-01 consolidation)

Draft built read-only on 2026-10-02 from the 21 rows of "Research handoff checklist" in
`docs/reviews/CONSOLIDATION_REVIEW_20261001.md` (lines 93-119) and from Omri's decisions in
`docs/consolidation/DECISIONS_2026-10-01.md`. Every source below was opened with `git show <ref>:<path>`
or read on disk. Nothing in the repository was changed. This is a draft for Omri to review; it does
not choose the next experiment.

Refs used (tip commit at the time of reading):

- `origin/claude/ssl-pseudolabel-residual-v1` at 553de79c6 (newest main line; worktree `.worktrees/ssl-pseudolabel-residual-v1`)
- `origin/claude/estimator-provenance-collection-2026-09-29` at aa8bf95d5 (worktree `.worktrees/tensor-mom-v1`; NOT an ancestor of the SSL tip)
- `claude/decision-rule-v1` at cbe345a4b (not in the SSL tip)
- `claude/self-generated-step-labels-v1` at 2465062e5 (not in the SSL tip)
- `lsml-ct7-levers-run` at 1503e529a (same commit as `origin/claude/lsml-ct7-levers-v1`; not in the SSL tip)
- `claude/whitebox-layer-views-v1` at f2f533795 (not in the SSL tip)
- `claude/readout-quickest-detection-v1` at cf70933d1 (not in the SSL tip)
- `codex/lsml-external-generalization-v1` at bef0b1de3 (main checkout; already an ancestor of the SSL tip)
- `origin/rescue/main-checkout-loose-files-2026-10-01` at a3ec90516 (its worktree has one uncommitted edit: a "Coverage checks" and "Pass 2" section in `docs/consolidation/RECONCILIATION_2026-10-01.md`, plus untracked `results/consolidation_2026-10-01/`)

Where the two input documents live:

- `docs/reviews/CONSOLIDATION_REVIEW_20261001.md`: committed on `origin/rescue/main-checkout-loose-files-2026-10-01`; an untracked copy sits in the main checkout.
- `docs/consolidation/DECISIONS_2026-10-01.md`: committed on `origin/rescue/main-checkout-loose-files-2026-10-01` (52 lines).

Drive locations referred to below:

- SSL worktree archive (verified 367/367 by size and MD5 on 2026-10-01): `gdrive:hallucination_detection/consolidated_results/local_backup_2026-09-30/ssl-pseudolabel-residual-v1/`, recorded in `docs/archives/DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.md` @ `origin/claude/ssl-pseudolabel-residual-v1`.
- This consolidation pass: `gdrive:hallucination_detection/consolidated_results/local_backup_2026-10-01/` (pass 1 partly verified, pass 2 in progress, per `docs/consolidation/RECONCILIATION_2026-10-01.md`).
- Cluster outputs: `gdrive:hallucination_detection/cluster_results/` (raw pickles, external telemetry archives, layer-view captures).

Cross-cutting facts that affect almost every entry (each checked on disk on 2026-10-02):

- The two pool inputs `pool_z.npy` and `pool_names.json` are GONE from the temporary scratchpad `C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad\`. `results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json` @ `origin/claude/ssl-pseudolabel-residual-v1` still points both entries there, and `per_dataset_fit_run.py`, `position_channel_run.py`, `position_prior_run.py`, `partition_switch_run.py` resolve their inputs through that manifest and hash-check them. They will hard-stop until the files are restored. A hash-verified local backup exists at `.worktrees/ssl-pseudolabel-residual-v1/results/algorithm_decisions_v1/inputs_backup/` (`pool_z.npy` 60,568,480 bytes, git-ignored; `pool_names.json` and `HASHES.json` committed), and the SSL Drive archive includes it.
- Only `origin/claude/ssl-pseudolabel-residual-v1` carries the 2026-09-30 (digits stay) and 2026-10-01 (averaging not final; Joint stopped) decisions in `CLAUDE.md`. The main checkout's `CLAUDE.md` and the decision-rule, self-generated-label, CT7-levers and estimator branches still open with the 2026-09-24 / 2026-09-17 instructions, including the digit exclusion and Joint continuation text.
- The upstream inputs used by nearly every runner are git-ignored files inside other worktrees: `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/{OOF_STEP_SCORES.npz, profiles_full.npy}` (OOF_ANSWERS.csv and INPUT_FREEZE.json are tracked), `.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/{TOKEN_MATRICES.npz, DERIVATIVE_CHANNELS.npz}`, `.worktrees/token-probability-fusion-v1/results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz`, `.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy`, `.worktrees/depth-feature-fusion-v1/results/step_level_bank_baseline_v1/STEP_SCORES.npz`. All exist locally today. None of these worktrees has a dedicated Drive archive record; pass 2 of the 2026-10-01 upload is meant to cover ignored files of 14 worktrees.

---

### 1. Final fusion and clustering

- **Source:**
  - `CLAUDE.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, lines 3-19 ("Omri decision update - 2026-10-01").
  - `Research_Directions.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, lines 3757-3768 ("Omri decisions 2026-10-01").
  - `results/algorithm_decisions_v1/SUMMARY.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, lines 22-29 (results table) and 34-45 (decisions).
  - `docs/consolidation/DECISIONS_2026-10-01.md` @ `origin/rescue/main-checkout-loose-files-2026-10-01`, item 1.
  - All four exist.
- **Current status:**
  - USER DECISION (Omri, 2026-10-01): "The plain average is not acceptable as the final fusion step." The frozen candidate (Dawid-Skene filter, then plain average) is still reported as the current result, also to the advisors.
  - COMPLETED EXPERIMENT (Step 457, `results/algorithm_decisions_v1/SUMMARY.md`): the frozen rule selected the filter plus plain average (8-bank mean within-answer AUC 0.7656). The runner-up, latent-group EM (HEM) weights within groups and Dawid-Skene weights between groups, has mean 0.7670, gains on 32 and 32+digits, and loses on 20+digits (-0.0097) and 51 (-0.0074). It therefore fails the frozen no-loss-on-any-bank rule. L-SML is below the plain average on 8 of 8 banks.
  - OPEN PROPOSAL (Omri's direction, recorded by Claude): replace the plain average with a combination that uses the label-free SML or method-of-moments estimates. The starting point is the Step 457 runner-up: filter, then label-free partition plus one absorption merge, then HEM within groups and Dawid-Skene between groups.
- **Next step (as stated):**
  - Develop that combination from the runner-up.
  - Address the known obstacle "rather than ignore" it: the estimates select channels but have not weighted them better than averaging (Steps 450-452, 457, 464).
  - The sources do not specify a protocol.
- **Dependencies:**
  - Code @ `origin/claude/ssl-pseudolabel-residual-v1`:
    - `scripts/experiments/ds_group_weights.py` (hem_fit, mle_group_weights; `tests/test_ds_group_weights.py`)
    - `scripts/experiments/lsml_merge_step.py` (absorb_merge)
    - `scripts/experiments/algorithm_decisions_run.py` and `algorithm_decisions_parallel.sh`
    - `scripts/experiments/per_dataset_fit_run.py` (the handoff names it as the base for future per-dataset experiments: `docs/HANDOFF_LABEL_FREE_ALGORITHM_2026-09-30.md` lines 42-43)
    - `spectral_utils/prmbench.py` (imported from `.worktrees/depth-feature-fusion-v1`)
  - The method-of-moments estimator `tensor_mom_estimate` exists only in `scripts/experiments/er_stage_a.py` @ `origin/claude/estimator-provenance-collection-2026-09-29`. The SSL copy lacks it.
  - Data:
    - `pool_z.npy` and `pool_names.json`: must be restored, see the cross-cutting note.
    - `digit_family_extension_v1/FEATURES.npz` and `indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv` in the SSL worktree.
    - The upstream worktree inputs listed above.
    - The Step 457/460/461 `STEP_SCORES.npz` replay references, git-ignored in the SSL worktree and in the SSL Drive archive.
  - Compute: CPU only. The Step 457 grid ran as 4 local processes.
  - Blocking by Omri: none for the direction itself. Per the SSL line status, the choice of the next run is to be made "after the merge, among all options on the table".
- **Known risks:**
  - The runner-up's 8-bank mean gain (+0.0014) is post hoc and comes with broad losses on two banks.
  - Dawid-Skene overestimates prevalence (0.28-0.41 against 0.14) and gives the level group the top weight in 40/40 bank-folds.
  - All of this evidence is PRMBench development data. The selection used PRMBench outcomes.
  - The runners read four other worktrees and the missing temporary scratchpad by absolute path.

### 2. Level-family removal

- **Source:**
  - `CLAUDE.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, lines 15-18.
  - `Research_Directions.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, lines 3766-3768.
  - `docs/consolidation/DECISIONS_2026-10-01.md` @ `origin/rescue/main-checkout-loose-files-2026-10-01`, item 3.
- **Current status:** USER DECISION (Omri, proposed 2026-09-27 and confirmed 2026-10-01): "drop 2-3 level-family channels to break the level group's dominance ... then re-run the filter, partition and estimates". Omri's caveat: going from 5 to 3 level channels need not leave two level groups. No experiment has run it. It is an unrun experiment, not a completed one.
- **Next step (as stated):**
  - Remove 2-3 level-family channels.
  - Re-run the Dawid-Skene filter, the label-free partition and the estimates.
  - Check the resulting partition, not only the score, with matched controls.
- **Dependencies:**
  - Same code and data as entry 1. The bank definitions (`MI['banks']`) are read from `results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json`.
  - The level channels come from `DERIVATIVE_CHANNELS.npz` (token-probability worktree, git-ignored).
  - CPU only. Not blocked; it waits for Omri to pick it among the options.
- **Known risks:**
  - The choice of which 2-3 channels to drop must be made label-free and declared before scoring.
  - Fewer level channels may simply change the partition rather than the weighting.

### 3. Tensor method of moments

- **Source:**
  - `results/tensor_mom_v1/NEGATIVE_RESULT.md` @ `origin/claude/estimator-provenance-collection-2026-09-29`: hypothesis on line 3, verdict on lines 26-29, reopening conditions on lines 31-32.
  - `PROGRESS.md` @ the same ref, section "Tensor MoM run: negative for estimation, candidate unchanged".
  - `HISTORY.md` @ the same ref, Step 464 at line 18852.
  - `docs/research_notes/ESTIMATOR_PROVENANCE_2026-09-29_HE.md` (estimator branch only).
- **Current status:** COMPLETED EXPERIMENT (Step 464), negative for this implementation only.
  - The third-moment estimator gives prevalence 0.23 against a truth of 0.14 (Dawid-Skene and HEM give about 0.28).
  - Per-channel sensitivity is worse: mean absolute error 0.179 against Dawid-Skene's 0.118.
  - It keeps the same channels as Dawid-Skene on 5/5 folds, and the bar fails on 0/5 folds for every estimator.
  - The verdict box marks "closes this IMPLEMENTATION only".
  - Separately, the fixed-20%-threshold self-consistency idea was "Deprioritized after discussion" (`PROGRESS.md` @ the estimator branch). It is a deprioritized idea, not a decision on the reopening questions.
- **Next step (as stated):** Reopen only with one of the following, evaluated against the same stage-A truth on the same folds:
  - votes whose dependence is removed before estimation (position-residualized channels, or one vote per declared block);
  - a marking rule that is not a uniform per-answer quantile.
- **Dependencies:**
  - Code @ `origin/claude/estimator-provenance-collection-2026-09-29` only: `scripts/experiments/tensor_mom_stage_a_run.py` and `er_stage_a.tensor_mom_estimate`, with `tests/test_er_stage_a.py` (10/10).
  - `er_stage_a.py` imports `cvf_v2` from `.worktrees/cumulative-vote-fusion-v2/scripts/experiments`. Per the collection handoff, 3 tests in `tests/test_er_stage_a.py` fail without that worktree.
  - Data: the stage-A inputs (`results/expectation_realization_v1/run_20260927/INPUT_MANIFEST.json`) and the upstream worktree arrays.
  - CPU only. The estimator branch must be merged; it is not in the SSL tip.
- **Known risks:**
  - The branch is not an ancestor of the SSL tip, so a merge that takes only SSL loses Steps 463-464 and the estimator.
  - Step number 463 collides with the self-generated-labels Step 463; keep both, tagged.

### 4. Position

- **Source:**
  - `Research_Directions.md` @ `origin/claude/ssl-pseudolabel-residual-v1`: lines 3739-3749 (next directions 1-3) and 3722 ("Decided or closed").
  - `HISTORY.md` @ the same ref, Step 462 at line 18824.
  - `docs/line_status/claude__ssl-pseudolabel-residual-v1.md` @ the same ref, lines 44-48.
  - Controls: `docs/HANDOFF_LABEL_FREE_ALGORITHM_2026-09-30.md` lines 73-74 and `LESSONS.md` lines 346-351 @ the same ref.
- **Current status:**
  - COMPLETED EXPERIMENTS: Steps 460-462, `results/{position_channel_v1,position_prior_v1,per_dataset_fit_v1}/SUMMARY.md`.
    - The position channel's direction is identified without labels; its weight is not.
    - The prior's weight is 2-5x below the PRMBench optimum.
    - Fitted per dataset, the filter keeps position on PRMBench (4/4) and drops it on ProcessBench (32/32).
    - The per-cell prior latches onto a step-0 artefact (17/32 grouped fits predict step 0).
  - OPEN PROPOSALS (Claude), none started. These are three distinct proposals:
    1. Use the filter's per-dataset position decision as the gate for the stronger prior.
    2. Set the position weight from the cross-fitted slope without re-weighting the content (plain average + logit(pi_b)/a_cf).
    3. Remove the per-dataset position profile of the content score before any position term. The source calls this a safeguard, not a method: -0.7 to +1.0 points against argmax.
- **Next step (as stated):**
  - Pick one or more of the three proposals.
  - Each needs a fresh frozen protocol, because proposal 1 was suggested by Step 462's outcome. A claim needs data other than PRMBench and ProcessBench.
  - Carry the step-index row, the same-length swap null and equal-share comparisons.
- **Dependencies:**
  - Code @ `origin/claude/ssl-pseudolabel-residual-v1`: `scripts/experiments/per_dataset_fit_run.py`, `position_prior_ds.py` (`tests/test_position_prior_ds.py`), `position_channel_run.py`, `position_prior_run.py`.
  - Data: everything in entry 1, including the missing pool files; `results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz` (git-ignored, exists locally, in the SSL Drive archive).
  - CPU only. Blocking: Omri's choice among the options.
- **Known risks:**
  - A position channel gains more under the swap null by construction, so the content share is descriptive only.
  - PRMBench rewards position and ProcessBench penalizes it.
  - The proposal-1 idea is post hoc.
  - The runners import from the depth worktree and run `git rev-parse` inside it to record its head.

### 5. Fitting access

- **Source:**
  - `Research_Directions.md` @ `origin/claude/ssl-pseudolabel-residual-v1`: lines 3687-3691 ("Fitting contract (Omri, 2026-09-29)") and lines 3-6 (pointer).
  - The same file, line 3647 ("Open, and never run despite being specified three times", in the Step421-422 block): the matched answer-local versus pooled comparison.
  - Source-fitted bank11 contract: `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1.md` lines 32-41 and `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` lines 16-21 (both @ `origin/claude/ssl-pseudolabel-residual-v1`).
- **Current status:**
  - USER DECISION (Omri, 2026-09-29): the method is fitted "per model per dataset, on that dataset's own unlabeled answers". Nothing fitted is shared across datasets.
  - COMPLETED EXPERIMENT (Step 462): this contract costs nothing on PRMBench. Steps 456-461 ProcessBench numbers are superseded by Step 462.
  - The frozen source-fitted bank11 L-SML (external lock) and the answer-local studies are separate, older contracts.
  - The matched answer-local versus pooled comparison on one fixed bank is recorded as open and never run.
- **Next step (as stated):**
  - Keep the three contracts (per-dataset transductive, source-fitted frozen, answer-local) labelled separately. Do not mix their metrics, and do not re-apply the new contract to historical results.
  - The matched answer-local versus pooled comparison remains an open specification.
- **Dependencies:**
  - None new. Documentation discipline in the consolidated `Research_Directions.md`, `PROGRESS.md` and `CLAUDE.md`.
  - The per-dataset runner is `per_dataset_fit_run.py` (entry 4).
- **Known risks:**
  - The main checkout `CLAUDE.md` still states answer-local fitting as primary (2026-09-23 and 2026-09-07 blocks) without the 2026-09-29 contract.
  - Historical tables in reports mix contracts unless labelled.

### 6. Digits and Joint

- **Source:**
  - `CLAUDE.md` @ `origin/claude/ssl-pseudolabel-residual-v1`: lines 21-29 ("2026-09-30: digit features stay"), lines 13-14 (Joint discontinued) and lines 85-107 (the older 2026-09-17 exclusion).
  - Commits f7ef911e8 (2026-09-30) and 680c537c6 (2026-10-01), both on `origin/claude/ssl-pseudolabel-residual-v1`.
  - `docs/consolidation/DECISIONS_2026-10-01.md` @ `origin/rescue/main-checkout-loose-files-2026-10-01`, items 2 and 4.
  - Historical Joint records:
    - `results/declared_joint_prmbench_v1/` @ `origin/claude/ssl-pseudolabel-residual-v1`
    - `docs/JOINT_LSML_METHOD_CARD_2026-09-17.md` @ `lsml-ct7-levers-run` (not on the SSL ref)
    - `HANDOFF_JOINT_LSML_REVIEW_TO_CODEX_2026_09_04.md` @ `origin/codex/og-sml-agent-b-v1`
- **Current status:**
  - USER DECISION (Omri, 2026-09-30): keep the three digit features (digit_alternative, digit_spread, digit_alternative_innovation) in the label-free fusion line. The banks are B16/B23/B35/B54. This "does not by itself reopen digit-specific gates, digit anchors or digit-oriented fusion signs".
  - USER DECISION (Omri, 2026-10-01): "Joint L-SML is discontinued". Do not add it as a new arm. Historical results stay as records.
- **Next step (as stated):**
  - No experiment. Record both decisions above the historical instructions in the consolidated `CLAUDE.md`.
  - Preserve Joint studies and reviewer concerns as records.
- **Dependencies:**
  - The merge must carry the SSL `CLAUDE.md` blocks.
  - Digit features come from `results/digit_family_extension_v1/FEATURES.npz` (SSL worktree, git-ignored, in the SSL Drive archive).
- **Known risks:**
  - On every other branch and in the main checkout, the 2026-09-17 digit exclusion and the 2026-09-17 "feature selection inside Joint" continuation are still the top instructions.
  - The 2026-09-17 exclusion still governs gates, anchors and signs; it must not be dropped wholesale.

### 7. Answer gate and decision rule

- **Source:**
  - `docs/line_status/claude__decision-rule-v1.md` @ `claude/decision-rule-v1`, lines 16-27.
  - `HISTORY.md` @ `claude/decision-rule-v1`, Step 459 at line 18800 and Step 460 at line 18808.
  - `results/decision_rule_v1/SUMMARY.md` and `results/answer_gate_v1/SUMMARY.md` @ `claude/decision-rule-v1`.
- **Current status:**
  - COMPLETED EXPERIMENTS:
    - Step 459: a per-answer flag count from raw channel levels (rule R2) raises PRMScore 0.6565 -> 0.6635 (Holm interval [0.0012, 0.0132]). A threshold built from Dawid-Skene sensitivity and specificity loses.
    - Step 460: answer-level detectors find erroneous ProcessBench answers (U-PCR 0.773 AUROC, but the edge over mean entropy is length). A gate lifts ProcessBench official F1 mostly generically: random gate 0.272, pure gate 0.362. On PRMBench it lowers PRMScore.
  - PAUSED-FOR-DECISION. Open question 1 for Omri, with three options:
    - (a) no gate, frozen answer-z q80 rule;
    - (b) the Step-459 count rule R2, calibrated per benchmark;
    - (c) a pure answer gate on mean token entropy.
- **Next step (as stated):**
  - Omri chooses (a), (b) or (c).
  - Freeze it "before any external result is seen", keeping wrong-answer detection, number of flags and localization as separate endpoints.
  - "No new experiment is started until they are answered."
- **Dependencies:**
  - Code @ `claude/decision-rule-v1` only: `scripts/experiments/decision_rule_run.py`, `answer_gate_run.py`, `answer_features_extract.py`.
  - Data:
    - `results/answer_gate_v1/ANSWER_FEATURES.npz` (2.6 MB, git-ignored, regenerable in about 7 min CPU)
    - the frozen stage-B scores `.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz` (exists, in the SSL Drive archive)
    - `TOKEN_MATRICES.npz` and `DERIVATIVE_CHANNELS.npz` (token-probability worktree; the status file says "not archived by this line")
  - CPU only. Blocking: Omri's decision.
- **Known risks:**
  - Hard2Verify and Socratic results were already inspected (Steps 433, 444, 446, 459), so no gate choice can make them untouched evidence.
  - The PRMBench construction prior (about 2 errors per answer) inflates any length-based count rule. The top-2 diagnostic (0.6808) is post hoc.
  - The branch is not in the SSL tip, and its runners read the SSL worktree by absolute path.

### 8. Whitebox answer gate

- **Source:**
  - `results/whitebox_layer_views_localization_v1/RESULTS_STAGE_1A.md` @ `claude/whitebox-layer-views-v1`: section 3.3 at line 84 (orientation is label-derived), section 5 at line 123, section 6 at line 148.
  - `results/whitebox_layer_views_localization_v1/RESULTS_STAGE_1B.md` @ the same ref, section 7 at line 124.
  - `docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md` @ the same ref.
- **Current status:**
  - COMPLETED EXPERIMENT, stage 1b: "The locator branch ends, as pre-registered."
  - COMPLETED EXPERIMENT, stage 1a: a tie against the production CT7 gate, 0.727 against 0.710 mean ProcessBench AUROC with overlapping intervals. Errors are complementary: failure phi 0.402, 630 answers rescued.
  - The 0.727 depends on a sign flip chosen by looking at labels; with the contract's own orientation the feature scores 0.273.
  - OPEN: the answer-gate fusion experiment, licensed by stage 1a.
- **Next step (as stated):**
  - First establish a label-free orientation rule ("Blocking before any candidate").
  - Then run a gate-fusion experiment judged on exact localizations gained versus lost at a matched opened fraction, not on AUROC alone.
- **Dependencies:**
  - Code @ `claude/whitebox-layer-views-v1`: `scripts/run_whitebox_kill_test_1a.py`, `scripts/run_whitebox_kill_test_1a_followup.py`, `cluster/reduce_layer_views_answer_level.py`, `scripts/land_layer_views_answer_level.py`.
  - Data:
    - `results/whitebox_layer_views_localization_v1/ANSWER_LEVEL.npz` (305 MB, git-ignored in the whitebox worktree, exists locally)
    - CT7's gate from `CT7_DEV_SCORES.npz`, read through the sibling path `ROOT.parent/token-probability-fusion-v1/...`
    - raw layer-view captures on AIRCC `cycle2` (possibly being retired) and on Drive `cluster_results/{pb_layer_views_qwen3_4b, pb_layer_views_qwen3_8b, prmbench_layer_views_qwen3_8b}` (per the whitebox handoff, lines 30-46)
  - CPU only for the gate fusion. Not blocked by Omri, but blocked by the orientation rule.
- **Known risks:**
  - The headline number is oracle-oriented, the `max(auc, 1-auc)` trap.
  - The branch is not in the SSL tip.
  - The whitebox worktree's ignored files (about 815 MB) are listed by the review as not yet archived.

### 9. Length confounding

- **Source:** `results/whitebox_layer_views_localization_v1/RESULTS_STAGE_1B.md` @ `claude/whitebox-layer-views-v1`, section 5, lines 95-113.
- **Current status:**
  - COMPLETED DIAGNOSTIC: after within-answer rank residualization against step length, ProcessBench scores fall as follows. CT7's locator keeps a margin of +3.5 over chance; the two single-statistic arms sit at chance.
    - final_lens_H: 33.61 -> 16.79
    - depth_contract: 31.56 -> 16.99
    - CT7 locator: 39.89 -> 20.09
  - OPEN PROPOSAL (Claude, whitebox stage 1b): "This is worth its own follow-up regardless of the white-box line, because it bears on every step-level readout in the project." No protocol exists.
- **Next step (as stated):** A separate step-length / readout follow-up. The sources give no design beyond that sentence.
- **Dependencies:**
  - `scripts/run_whitebox_kill_test_1b.py` and `results/whitebox_layer_views_localization_v1/STEP_LEVEL.npz` (486 MB on paper, 509,199,000 bytes on disk; git-ignored in the whitebox worktree).
  - For non-whitebox readouts: the step evidence and CT7 arrays in the readout-quickest and token-probability worktrees.
  - CPU only.
- **Known risks:**
  - The source warns that a rank difference "is a crude control, not a regression residual", so the residualized values are an upper bound on damage, not a causal estimate. Only the ordering survives.

### 10. External reevaluation (Hard2Verify, Socratic-PRMBench)

- **Source:**
  - `Research_Directions.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, line 3750 (next direction 4).
  - `docs/line_status/claude__ssl-pseudolabel-residual-v1.md` @ the same ref, line 48 (option 4).
  - `results/algorithm_external_v1/SUMMARY.md` @ the same ref (Step 459).
  - `LESSONS.md` @ the same ref, lines 346-351.
- **Current status:**
  - COMPLETED EXPERIMENT (Step 459, exploratory): the 13+digit plain average beats CT7 on Socratic (+0.044 PRMScore) mostly through step position. Step index alone beats every method there (within-answer AUC 0.73 Socratic, 0.86 Hard2Verify).
  - OPEN PROPOSAL (Claude): re-check under the per-dataset contract, with the step-index row, swap nulls and equal-share comparisons.
- **Next step (as stated):**
  - Ask whether the filter keeps the position channel on Hard2Verify and Socratic.
  - Compare the method with the step index alone.
  - Label it exploratory: these benchmarks are exposed.
- **Dependencies:**
  - Code @ `origin/claude/ssl-pseudolabel-residual-v1`: `scripts/fit_algorithm_external_v1.py`, `scripts/score_algorithm_external_v1.py`, `scripts/evaluate_algorithm_external_v1.py`, `scripts/extract_external_banks_v4.py`, `scripts/verify_external_banks_v4_source.py`, `spectral_utils/external_banks_v4.py` (`tests/test_external_banks_v4.py`).
  - Data:
    - `results/external_banks_v4/*/FEATURES.npz` and `reference_arrays/*.npy` (SSL worktree, git-ignored, in the SSL Drive archive)
    - `algorithm_external_v1/run_20260929/*/PREDICTIONS_UNSEALED.json` (in the SSL Drive archive; digests in SEAL.json)
    - private inputs `scratch/external_generalization_private/inputs` and `sources` in the main checkout (git-ignored)
  - Per the rescue worktree's uncommitted reconciliation, only 24 of 6,670 private files (2.24 GB) are matched on Drive. 6,646 files (7.64 GB) were queued for pass 2, which has not reported.
  - CPU only. Blocking: Omri's choice among the options.
- **Known risks:**
  - The labels have been inspected, so this cannot become untouched confirmation, and no tuning is allowed on them.
  - The scorer and evaluator read `MAIN/scratch/...` by absolute path.
  - The fitter reads the depth and readout-quickest worktrees.

### 11. Untouched confirmation

- **Source:**
  - `Research_Directions.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, line 3754 (next direction 6).
  - `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` @ the same ref, line 64.
  - External specification `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1.md` @ the same ref: line 4 ("MedPRMBench is deferred") and lines 63-81 (execution gates).
  - Runbook `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1_RUNBOOK.md` @ the same ref.
- **Current status:**
  - DEFERRED (MedPRMBench), as an OPEN PROPOSAL: the remaining untouched candidate.
  - PRMBench, ProcessBench, Hard2Verify and Socratic are all exposed, so no benchmark already examined can supply untouched confirmation.
- **Next step (as stated):**
  1. Lock the method and the decision rule first (entries 1, 4 and 7).
  2. Then apply the execution gates:
     - AIRCC preflight PASS in the same session;
     - at least 10 GB local/TMPDIR and 50 GB shared free;
     - a timing job of at most 12 deterministic examples per dataset/backbone, at most 1 allocated GPU-hour each;
     - Omri's full-run budget decision before full inference.
- **Dependencies:**
  - Collector code @ `origin/claude/ssl-pseudolabel-residual-v1` (also on `codex/lsml-external-generalization-v1`): `cluster/run_external_telemetry.py`, `cluster/submit_external_telemetry.sbatch`, `scripts/prepare_external_sources.py`, `scripts/prepare_external_inputs.py`.
  - No MedPRMBench adapter or data contract exists in `spectral_utils/external_generalization/` or `cluster/` on any inspected ref. A grep for "medprm" finds only "deferred" notes.
  - AIRCC GPU. The whitebox handoff warns that `cycle2` storage may be retired.
  - Blocking by Omri: method/decision lock and the budget decision.
- **Known risks:**
  - Source overlap with PRMBench and ProcessBench must be audited.
  - The adapter does not exist yet.
  - Any post-hoc choice made on the exposed benchmarks must be frozen before the test.

### 12. Source-side transfer proxy

- **Source:**
  - `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, section 3 items 1-2, lines 56-62.
  - `Research_Directions.md` @ the same ref, line 3677 ("Open direction").
- **Current status:**
  - OPEN PROPOSAL (Claude, 2026-09-24), "each needs Omri's go". Never run.
  - Motivation is a completed finding: source five-fold PRMScore rank did not predict external rank in three locks (V1-V3).
- **Next step (as stated):**
  1. Build a source split that mimics a domain shift. The options named are:
     - leave-one-benchmark-out (fit on ProcessBench, evaluate PRMBench);
     - leave-one-ProcessBench-cell-out;
     - PRMBench category holdout.
  2. Check retrospectively whether it would have ranked bank11 L-SML above the family variants and CT7. Label this proxy validation on exposed data. If no proxy tracks transfer, say so and prefer simpler, more constrained representations.
  3. Diagnose why bank11 transfers: partition and weight stability across folds, and external per-channel behaviour (diagnostic only).
- **Dependencies:**
  - Code @ `origin/claude/ssl-pseudolabel-residual-v1`: `scripts/experiments/family_tail_calfix_run.py`, `family_tail_calfix_eval.py`, `calfix_common.py`, `calfix_evaluate.py`, `tail_calib_common.py` (tests `scripts/experiments/test_calfix_common.py`, `test_tail_calib_common.py`).
  - Data:
    - pool files: missing from the temporary scratchpad; restore from the SSL `inputs_backup/` or from Codex's Drive archive `gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/evaluation_archives/family_tail_external_v1_results_a1e2f683cc6932a9.tar.gz`, member `reproduction/source_pool/`, per `results/family_tail_external_v1/RESTORE.md`
    - `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/`
    - `results/lsml_external_generalization_v1/evaluation/source/{VALIDATION_FITS,BUNDLE}.json` (main checkout)
  - CPU only. Blocking: Omri's go.
- **Known risks:**
  - Validating the proxy against the known external ranking uses exposed data.
  - The short merge-handoff outline omitted this item.
  - Every source runner hard-codes the temporary scratchpad path, which no longer holds the pool files.

### 13. Family-tail audits and library contract

- **Source:** `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, section 3 items 3 and 5 (lines 63 and 65) and section 5 (lines 85-87).
- **Current status:**
  - COMPLETED: the V1 independent audits (metric recompute, coverage, null) and the exact replays of V2/V3. Each V2/V3 run replays every earlier arm, difference 0.0.
  - OPEN PROPOSAL (Claude): independent audits of V2 and V3, which were never run.
  - Library contract: a stated rule, partly enforced. `tail_calib_common.lsml_fit_scaled` enforces it for new code.
- **Next step (as stated):**
  - Run Codex's three audit scripts on V2 and V3. The recompute script is in `results/family_tail_external_v1/independent_metrics/`.
  - Guard the library contract: callers of `lsml_continuous` pass z-scored inputs or use `loading_scale='complete'`.
  - Any hard guard in `spectral_utils/external_generalization` needs a new version; frozen modules are not edited in place.
- **Dependencies:**
  - Audit scripts `results/family_tail_external_v1/independent_metrics/` (on both `origin/claude/ssl-pseudolabel-residual-v1` and `codex/lsml-external-generalization-v1`).
  - Evaluators `scripts/run_family_external_v2.py`, `scripts/evaluate_family_external_v2.py`, `scripts/evaluate_family_external_v3.py`.
  - Per-answer records and bootstrap arrays of V1/V2/V3 (git-ignored; regenerable from V1 records "in minutes").
  - Private inputs in `scratch/external_generalization_private/` (partial Drive coverage, see entry 10). CPU only.
- **Known risks:**
  - `spectral_utils/external_generalization/_bank11/__init__.py` and `chosen_token_calibration.py` are committed with CRLF under an `eol=lf` rule. They are hash-locked and must not be renormalized (`docs/HANDOFF_COLLECTION_2026-09-29.md` lines 149-157 @ `origin/claude/estimator-provenance-collection-2026-09-29`).
  - The default `loading_scale='unit'` K criterion is scale-sensitive.

### 14. External published comparators

- **Source:**
  - `docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md` @ `origin/claude/ssl-pseudolabel-residual-v1`: line 25 ("published critic/PRM reproduction remains outstanding"), line 27, lines 163-167 (published context).
  - `Research_Directions.md` @ the same ref, line 18 ("Reproducing those comparators remains unfinished").
  - Specification `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1.md` @ the same ref, lines 18-23 and 92-93.
- **Current status:**
  - OPEN (unfinished part of an approved pipeline).
  - Comparator inference was authorized in the 2026-09-24 specification:
    - Hard2Verify: the same Qwen3-8B critic, Qwen2.5-Math-PRM-7B and Universal-PRM;
    - Socratic: the QwQ-32B critic and Qwen PRM-7B.
  - Only the published numbers are quoted, as context: Hard2Verify 53.51 / 42.37 / 60.27 Balanced F1; Socratic 68.0 / 73.8 PRMScore.
- **Next step (as stated):**
  - Reproduce the published critic/PRM comparators under explicit asset, access and inference requirements.
  - Keep target-calibrated PRM protocols separate from source-only calibration.
  - Make no SOTA claim.
- **Dependencies:**
  - AIRCC GPU inference with preflight, a timing job and a budget decision (same gates as entry 11).
  - External inputs in `scratch/external_generalization_private/inputs` (main checkout).
  - No comparator-inference runner was identified on the inspected refs; only the telemetry collector exists.
  - Blocking: Omri's budget decision.
- **Known risks:**
  - The published Hard2Verify PRM thresholds were tuned on 100 target responses, a different access condition.
  - No paired predictions exist.

### 15. New information from another model

- **Source:**
  - `docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, lines 19-25.
  - `docs/experiments/EXTERNAL_TEACHER_FORCING_REPEATABILITY_20260924.md` @ the same ref, lines 1-39.
  - Reopening condition for L-SML: `docs/line_status/claude__ssl-pseudolabel-residual-v1.md` lines 55-56.
- **Current status:**
  - COMPLETED DIAGNOSTICS:
    - Identical teacher-forcing repeats gave exactly equal saved telemetry for 36/36 answer/backbone pairs, so they supply no diversity.
    - Qwen3 and QwQ frozen decisions disagree on 6.72% of Socratic steps (mean within-answer Spearman 0.9003).
    - "No cross-backbone fusion was fitted or evaluated."
  - OPEN PROPOSAL: a bounded source-side cross-backbone or residual option. It is a hypothesis, not demonstrated fusion benefit.
- **Next step (as stated):**
  - Investigate any residual or cross-backbone extension on source development data first, using the retained telemetry and CPU.
  - Freeze before external evaluation.
  - Compare against matched fixed-weight controls.
  - Prompt variants are a separate, changed-conditioning option.
- **Dependencies:**
  - Source-side, the local `dataset_cache/repgrid/` holds ProcessBench telemetry from three scorers over the same answers: `pb_qwen3_4b`, `pb_qwen3_8b`, `pb_llama31_8b` (LFS; Qwen3 pickles matched on Drive `cluster_results/pb_qwen3_*` per `bplf_coverage.json` in the rescue worktree).
  - External: Socratic Qwen3/QwQ telemetry in `scratch/external_generalization_private/evaluation_archives` (9.5 GB local) and `gdrive:.../cluster_results/lsml_external_generalization_v1/`.
  - Descriptive analysis: `results/lsml_external_generalization_v1/evaluation/BACKBONE_COMPLEMENTARITY.json` and `scripts/audit_external_backbone_diversity.py`.
  - CPU only.
- **Known risks:**
  - The gold-assisted accuracy bounds in BACKBONE_COMPLEMENTARITY.json are not deployable.
  - External labels are exposed.
  - Temperature transforms or duplicated columns are not independent evidence.

### 16. Self-generated labels

- **Source:**
  - `docs/line_status/claude__self-generated-step-labels-v1.md` @ `claude/self-generated-step-labels-v1`, lines 17-31 (decision and options 1-3) and 38-50 (dependencies).
  - `docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md` @ the same ref.
  - `HISTORY.md` @ the same ref, Step 463 at line 18363.
- **Current status:**
  - COMPLETED EXPERIMENT (Step 463 [self-generated step labels]):
    - 890 own answers received final two-judge labels.
    - B16 ranks the first error as well on own text as on teacher-forced ProcessBench: pooled within-answer AUC 0.757 against 0.763, inside the equivalence margin.
    - Per cell the comparison is underpowered. GSM8K / Qwen3-4B is borderline: 0.712 against 0.801.
  - PAUSED-FOR-DECISION. All three options are open:
    1. Close with the underpowered-cell caveat.
    2. Collect sampled own answers (temperature 1, about 3 per question, Qwen3-4B and Qwen3-8B, GSM8K and MATH) on AIRCC and re-label them with the two-judge protocol.
    3. A paired test on existing answers, scoring each answer with both models (a short AIRCC teacher-forced run).
- **Next step (as stated):** Omri chooses option 1, 2 or 3.
  - Option 2 needs AIRCC preflight, a timing run of at most 12 examples per cell, Omri's budget decision, and judge runs on the GPT side.
  - Option 3 needs "minutes of GPU".
- **Dependencies:**
  - Code @ `claude/self-generated-step-labels-v1`: `scripts/self_generated_labels/{extract_own_answers.py, b16_fit.py, b16_eval.py}` and `.claude/agents/step-judge.md`.
  - Data:
    - `dataset_cache/repgrid/evdrop_{gsm8k,math}_qwen3_{4b,8b}/raw_*.pkl` in the main checkout. These are Git LFS. All four, plus the pilots, are matched by SHA-256 on Drive `cluster_results/evdrop_*` per `results/consolidation_2026-10-01/bplf_coverage.json` in the rescue worktree.
    - `results/self_generated_step_labels_v1/private/own_answers.jsonl` (15 MB, git-ignored, exists).
    - `b16/FEATURES.npz` (5.4 MB, untracked, about 65 min CPU to regenerate).
  - `b16_fit.py` also reads the SSL worktree (per-dataset code and Step 462 `STEP_SCORES.npz`) and the algorithm-decisions `INPUT_MANIFEST.json`.
  - Blocking: Omri's choice; for option 2, a budget.
- **Known risks:**
  - The quoted final merge plan omitted option 3.
  - Step number 463 collides with the estimator-branch Step 463.
  - The branch is not in the SSL tip.
  - `b16_fit.py` also reads `dataset_cache/repgrid/pb_qwen3_8b/processbench_gsm8k.pkl` via MAIN.

### 17. Supervised PRM by error type

- **Source:**
  - `docs/line_status/lsml-ct7-levers-run.md` @ `lsml-ct7-levers-run`, line 13 (Step 437) and line 39 (the error-type proposal).
  - `HISTORY.md` @ the same ref, Step 437 at line 58063 (error-type paragraph near line 58098, access note near line 58122).
  - `results/prm_vs_ct7_prmbench_v1/{MEASUREMENT,PRMSCORE,RUN_FREEZE}.json` @ the same ref.
- **Current status:**
  - COMPLETED MEASUREMENT (Step 437):
    - The supervised Qwen2.5-Math-PRM-7B is ahead of CT7 on within-answer AUROC (+2.88 pp) and PRMScore (0.6804 against 0.6457).
    - It is below CT7 on argmax hit (57.78% against 61.14%). The hit sets overlap on only 39.02%.
    - Split by error type: the PRM leads on confidence, counterfactual and deception errors; CT7 leads on redundancy, circular and domain_inconsistency.
  - OPEN PROPOSAL (Claude): a type-aware combination with a supervised PRM, "a new line, not a continuation".
- **Next step (as stated):**
  - A type-aware combination with matched access and error-type controls.
  - Complementary hit sets do not by themselves establish a label-free router.
- **Dependencies:**
  - PRM rewards: `dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl` (2 MB, git-ignored; the reconciliation says the 2026-08-20 Drive record covers 23 four_localization payloads, not re-verified here).
  - Other inputs: `results/localization_full_benchmark_v3/evaluation/JOINED.{json,npz}` (main checkout), `FOLDS_V2.json`, `CT7_DEV_SCORES.npz` (token-probability worktree).
  - The ProcessBench PRM run `dataset_cache/four_localization/pb_prm_qwen25math7b_full`.
  - CPU only.
- **Known risks:**
  - The PRM is a supervised high-access reference row, never a label-free arm (access note in Step 437).
  - The original error-type breakdown page is a claude.ai artifact (https://claude.ai/artifact/EArjBd9q2aXnvwoAimYPWr) and was not verified. Its builder scripts were left in a temporary scratchpad. The numbers survive in HISTORY Step 437.

### 18. PRMScore reporting

- **Source:**
  - `docs/line_status/claude__decision-rule-v1.md` @ `claude/decision-rule-v1`, line 25 (question 2).
  - `results/answer_gate_v1/SUMMARY.md` @ the same ref, lines 36-38.
  - `CLAUDE.md` (2026-09-23 block, every ref) ("Separate within-AUC, any-error hit and official PRMScore").
- **Current status:**
  - COMPLETED FINDING (Step 460, post hoc): PRMBench inserts about 2 errors per answer. Flagging the top-2 steps per answer, with no answer-level score, gives PRMScore 0.6808.
  - PAUSED-FOR-DECISION (question 2 for Omri): should every PRMScore report from now on include the fixed-count top-k baseline and the error-count-per-answer structure?
- **Next step (as stated):**
  - Omri answers question 2.
  - Per the review: retain fixed-count top-k baselines and the error-count analysis; say that k and the top-2 finding were post hoc; do not present top-k as needing no ranking method.
  - Report within-answer AUC, official PRMScore, first-error and no-error metrics separately.
- **Dependencies:** `scripts/experiments/answer_gate_run.py` @ `claude/decision-rule-v1`; reporting conventions in the consolidated `CLAUDE.md`. No compute.
- **Known risks:**
  - The top-2 rule does not transfer to ProcessBench.
  - Treating 0.6808 as a method would be selection on outcomes.

### 19. Saturated implementations

- **Source:**
  - The four line-status files:
    - `docs/line_status/claude__ssl-pseudolabel-residual-v1.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, lines 4-7 and 51-61
    - `docs/line_status/lsml-ct7-levers-run.md` @ `lsml-ct7-levers-run`, lines 17-39
    - `docs/line_status/claude__decision-rule-v1.md` @ `claude/decision-rule-v1`
    - `docs/line_status/claude__self-generated-step-labels-v1.md` @ `claude/self-generated-step-labels-v1`
  - `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` section 3 item 6, lines 66-71.
  - `results/tensor_mom_v1/NEGATIVE_RESULT.md` @ `origin/claude/estimator-provenance-collection-2026-09-29`.
  - `results/lsml_group_confidence_v1/NEGATIVE_RESULT.md` @ `origin/rescue/main-checkout-loose-files-2026-10-01`, verdict on lines 46-53 and reopening condition on lines 55-59.
  - Readout family: `HISTORY.md` Step 429 at line 57729 and `results/readout_family_v1/` @ `claude/readout-quickest-detection-v1`. That branch has no line-status file.
- **Current status:** COMPLETED EXPERIMENTS, each negative for its implementation. Every "SATURATED" label is a Claude/Codex proposal with evidence, not an Omri closure decision (DECISIONS item 8).
  - **CT7 weighting** (Steps 434, 436, 441-442, 447): no learned weighting beats equal on ProcessBench. Reopen with any of:
    - a new stream with independent information;
    - work on the no-error gate or the late-miss readout;
    - an untouched dataset where equal and learned disagree.
  - **Family/tail recipes** (Steps 441-446): no tail variant beats matched equal on source. Do not reopen tail tuning, per-family thresholds, 16 families, token-level tail fusion or FUSE objectives "without a new idea".
  - **L-SML variants on the step bank** (Steps 438-446, 456, 457): below plain average on 8/8 banks. Reopen only with channels conditionally independent given the label, such as a second model or internal layer states.
  - **First-error product readout** (Step 462): below argmax in 64/64 cells. Reopen only with a per-step error rate at the true scale (about 0.14).
  - **Partition switch** (Step 458): 0/80 switches with the family held out. Reopen only with banks that do not share channels.
  - **Readout family** (Step 429): the label-free consensus rule fails and the extended family is exhausted even label-selected. No reopening condition is written in a status file.
  - **Group confidence** (Codex, `lsml_group_confidence_v1`): 0 wins / 9 losses against family15-equal. Reopen with a declared source experiment with defensible family-error structure that beats both matched equal controls.
  - **Tensor MoM**: see entry 3.
- **Next step (as stated):**
  - No experiment.
  - Carry each implementation's evidence file and reopening condition into the consolidated handoff. Do not let a negative implementation close an estimator family.
- **Dependencies:** Documentation only. The group-confidence code `spectral_utils/lsml_group_confidence*.py`, `scripts/run_lsml_group_confidence.py` and `tests/test_lsml_group_confidence.py` exist only on `origin/rescue/main-checkout-loose-files-2026-10-01`.
- **Known risks:**
  - The readout-quickest line has no status file, so its reopening condition is undocumented.
  - The group-confidence ignored results (three files, 22,786,417 bytes) were omitted from upload pass 1 (review finding 2).

### 20. Historical transfer and supporting backlog

- **Source:**
  - Canonical research mandate: `CLAUDE.md` (all refs; 2026-09-07 blocks, "historical24" at lines 152, 737, 779, 806, 833 of the SSL copy).
  - `docs/experiments/LOCALIZATION_RESEARCH_MANDATE_20260907.md` @ `origin/rescue/main-checkout-loose-files-2026-10-01`. Its stage table has D, D2, E2, E rows, and rows F and G marked "Pending" at line 407.
  - Rescued early records @ `origin/rescue/main-checkout-loose-files-2026-10-01`: `docs/reviews/temporal_geometry_revisit_2026-09-07.md`, `docs/reviews/bocpd_boundary_audit_2026-09-07.md`, `docs/experiments/FULL_LOCALIZATION_SAMPLING_V3.md`, `HISTORICAL_FUSION_REFIT_V3.md`, `HISTORICAL_JOINT_REFIT_V3.md`, `papers/digests/{loca-local-conformal-autoencoder,task-based-graph-signal-compression}.md`.
  - The eight uncovered historical refs (all resolve): `origin/codex/reconstruction-science-results-v1`, `origin/codex/reconstruction-benchmark-v1`, `claude/advisor-letter-aug27`, `origin/codex/graph-geometry-selection-v1`, `origin/codex/iu-graph-smoothing-ablation-v1`, `origin/codex/deem-b3-moe-gating-v1`, `selector/a4-antigravity-unsupervised`, `origin/codex/og-sml-agent-b-v1`.
- **Current status:** Mixed. Each item needs an explicit disposition (completed / superseded / deferred), which none currently has.
  - **historical24 transfer**: OPEN requirement, never run (mandate stage G "Pending").
  - **Full token/window sampling**: COMPLETED. `results/localization_full_sampling_v3/RUN_STATE.json` in the main checkout reads phase "COMPLETE_REVIEWED_FULL_SAMPLING", 13,769/13,769, with `evaluation/REPORT.html`. Sparse/short-error retention: the 2026-09-07 records say "untested". I did not find it resolved in the full-sampling report.
  - **BOCPD/HMM/IMM, LOCA/Diverging Flows/KalmanNet, task-based sampling**: LOW-priority supporting backlog (CLAUDE.md 2026-09-07). Precursors only; the named methods were never implemented.
  - **Joint continuation proposals**: SUPERSEDED by the 2026-10-01 stop decision.
  - **RAG / general-task lines**: DEFERRED (outside the active claim since 2026-09-07).
- **Next step (as stated):**
  - Write the disposition per item: completed, superseded or deferred.
  - historical24 transfer runs only after the localization candidate is locked, with the historical contract and comparators.
  - For historical branches, Omri accepts "an archive with an index" that records what each line accomplished, its evidence, and unfinished or superseded next steps (DECISIONS item 5).
- **Dependencies:**
  - historical24 needs the historical cell caches. Their location was not verified in this pass.
  - `results/localization_full_sampling_v3/` is untracked (not ignored) in the main checkout and on no ref. Its 6.9 GB tar failed in upload pass 1 (DNS error) and is queued for retry in pass 2.
  - Index work: CPU / documentation.
- **Known risks:**
  - An unqualified "saturated" label could silently drop these lines.
  - Several runners on `lsml-ct7-levers-run` and `claude/readout-quickest-detection-v1` point at worktrees that no longer exist (`fusion-independence-atlas-v1`, `antigravity`) or at macOS paths (`/Users/osegev/Desktop/...`). See the final section.

### 21. Technical debt and advisor report

- **Source:**
  - MATH grader: `docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md` @ `claude/self-generated-step-labels-v1`, lines 81-91, and `docs/line_status/claude__self-generated-step-labels-v1.md` lines 33-36.
  - Portable paths and LFS: `docs/HANDOFF_COLLECTION_2026-09-29.md` @ `origin/claude/estimator-provenance-collection-2026-09-29`, lines 23-27, 131-139.
  - Drive client ID: `docs/archives/DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, lines 25-26.
  - Advisor report: `Research_Directions.md` @ `origin/claude/ssl-pseudolabel-residual-v1`, line 3755, and `docs/consolidation/DECISIONS_2026-10-01.md` item 1.
  - LFS publication: DECISIONS item 9.
- **Current status:**
  - DEFERRED (Omri, 2026-09-29): MATH regrading. `spectral_utils.data_loaders.is_correct_math` misgrades equivalent forms: 81 of 366 non-truncated grader-wrong MATH own answers are judged correct. Old self-generated MATH final-answer results remain unverified.
  - OPEN technical debt:
    - portable data paths: lineage A is "not reproducible from its own checkout";
    - LFS storage: over the GitHub budget. Omri accepts publishing without uploading LFS objects once all required objects are verified on Drive and restore is tested (DECISIONS item 9).
    - the rclone shared client_id will stop working during 2026; create a personal one.
  - USER DECISION: the advisor report is paused until cross-conversation collection. The current averaging result may be reported with Omri's stated limitation that it is not the final fusion step.
- **Next step (as stated):**
  - Regrade offline with a real equivalence checker (CPU), count flips per cell, and rescore old MATH results only if material.
  - Replace `MAIN / '.worktrees/<branch>/...'` with repo-relative paths in new runners only. Frozen runners keep recorded paths.
  - Verify LFS objects on Drive and test restore.
  - Create the client_id.
  - Write the advisor report after collection.
- **Dependencies:**
  - `spectral_utils/data_loaders.py`; affected caches `math500` via `cluster/run_inference.py`, AMC/AIME wrappers, `evdrop_math_*`.
  - Upload pass 2 results (`MANIFEST_v2.tsv`, not yet written into the reconciliation).
  - The two uncovered LFS objects: `dataset_cache/repgrid/gpqa_r1distill8b/raw_gpqa_T1.0.pkl` and `trace_gpqa_r1qwen7b/raw_gpqa_T1.0.pkl`, 6.17 GB.
- **Known risks:**
  - The Colab-era MATH-500 headline cells are not verified to use the same grader.
  - The 2026-09-17 local `master` consolidation was never pushed for exactly the LFS reason.

---

## Absolute worktree paths in runners

`MAIN` is the hard-coded `C:\Users\omris\TAU\hallucination_detection` (30 files under `scripts/experiments/` on `origin/claude/ssl-pseudolabel-residual-v1` define it). "Temporary scratchpad" means `C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad`. That folder still exists, but `pool_z.npy` and `pool_names.json` are missing from it. Short worktree names below stand for `MAIN/.worktrees/<name>`.

Worktree status on disk:

- Present: `depth-feature-fusion-v1`, `readout-quickest-detection-v1`, `token-probability-fusion-v1`, `cumulative-vote-fusion-v2`, `lsml-ct7-levers-run`, `ssl-pseudolabel-residual-v1`, `consolidation-fusion-2026-09-22`.
- Absent: `fusion-independence-atlas-v1`, `antigravity`, `reconstruction-science-run-v1`.

### origin/claude/ssl-pseudolabel-residual-v1, scripts/experiments/*.py

- `algorithm_decisions_run.py`:
  - depth-feature-fusion-v1 (sys.path import; hashes `spectral_utils/prmbench.py`)
  - token-probability-fusion-v1
  - readout-quickest-detection-v1/results/step_evidence_v1
  - cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1
  - cumulative-vote-fusion-v2/scripts/experiments/cvf_v2/{em,core}.py (hashed)
  - temporary scratchpad `pool_z.npy`, `pool_names.json` (MISSING)
- `bank20_lsml_run.py`:
  - depth-feature-fusion-v1 (import, plus `results/step_level_bank_baseline_v1/STEP_SCORES.npz`)
  - readout-quickest step_evidence_v1
  - token-probability-fusion-v1/results/token_probability_fusion_v1 (`DERIVATIVE_CHANNELS.npz`)
  - `MAIN/results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz`
  - `MAIN/results/localization_source_group_audit_v1/FOLDS_V2.json`
- `bank20_lsml_report.py`, `declared_joint_report.py`, `ssl_s0_report.py`: mention worktree paths in report text only.
- `calfix_common.py`: readout-quickest step_evidence_v1.
- `calfix_evaluate.py`: `C:\Users\omris\TAU\hallucination_detection\.worktrees\depth-feature-fusion-v1` (literal path).
- `core_virtual_lsml_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; temporary scratchpad.
- `declared_joint_run.py`:
  - depth-feature-fusion-v1; readout-quickest step_evidence_v1
  - token-probability-fusion-v1 results
  - `MAIN/results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz`
  - temporary scratchpad (`pool_z.npy`, `pool_names.json`)
- `er_digit_share.py`: token-probability-fusion-v1; lsml-ct7-levers-run; readout-quickest step_evidence_v1.
- `er_generality_run.py`:
  - depth-feature-fusion-v1; token-probability-fusion-v1; readout-quickest step_evidence_v1
  - cumulative-vote-fusion-v2 ct7_profiles_v1 and cvf_v2/{em,core}.py
  - ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz (its own worktree, by absolute path)
  - temporary scratchpad
- `er_prmscore_decomposition.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1.
- `er_stage_a.py`: cumulative-vote-fusion-v2/scripts/experiments (imports `cvf_v2`).
- `er_stage_b_run.py`, `er_stage_b2_run.py`: depth-feature-fusion-v1; token-probability-fusion-v1; readout-quickest step_evidence_v1; cumulative-vote-fusion-v2 ct7_profiles_v1 and cvf_v2/{em,core}.py.
- `error_cluster_lsml_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; temporary scratchpad.
- `expectation_realization_run.py`:
  - depth-feature-fusion-v1 (plus hash of `spectral_utils/prmbench.py`)
  - token-probability-fusion-v1 (plus `spectral_utils/derivative_step_channel_v1.py`)
  - readout-quickest step_evidence_v1
  - cumulative-vote-fusion-v2 ct7_profiles_v1 and cvf_v2/{em,core}.py
  - lsml-ct7-levers-run `spectral_utils/chosen_token_calibration.py`
- `family_tail_calfix_run.py`: depth-feature-fusion-v1; temporary scratchpad; `MAIN/results/lsml_external_generalization_v1/evaluation/source/VALIDATION_FITS.json`.
- `family_tail_calfix_eval.py`: `MAIN/results/lsml_external_generalization_v1/evaluation/source`.
- `family_tail_transfer_lock.py`: temporary scratchpad; `MAIN/results/lsml_external_generalization_v1/evaluation/source/BUNDLE.json`.
- `family_tail_transfer_lock_v2.py`: temporary scratchpad.
- `indbank_lsml_run.py`:
  - depth-feature-fusion-v1; readout-quickest step_evidence_v1; token-probability-fusion-v1 results
  - `MAIN/results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz`
  - temporary scratchpad
- `lsml_merge_step_run.py`:
  - depth-feature-fusion-v1; token-probability-fusion-v1; readout-quickest step_evidence_v1
  - cumulative-vote-fusion-v2 ct7_profiles_v1 and cvf_v2/{em,core}.py
  - temporary scratchpad
- `mtg_reproduction_extract.py`, `mtg_reproduction_score.py`: token-probability-fusion-v1 (sys.path import in the extractor).
- `named_group_fusion_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; temporary scratchpad.
- `partition_ceiling_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; token-probability-fusion-v1 results; `MAIN/results/localization_source_group_audit_v1/FOLDS_V2.json`.
- `partition_switch_run.py`, `per_dataset_fit_run.py`, `position_channel_run.py`, `position_prior_run.py`:
  - Direct reads:
    - depth-feature-fusion-v1 (sys.path import; the last three also run `git rev-parse HEAD` with cwd in that worktree)
    - readout-quickest step_evidence_v1
  - Indirect reads, through `results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json`:
    - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`
    - token-probability-fusion-v1 `results/localization_full_benchmark_v3/evaluation/JOINED.json`
    - cumulative-vote-fusion-v2 `ct7_profiles_v1/{profiles.npy, PROFILE_VALIDATION.json}`
    - readout-quickest `OOF_ANSWERS.csv` and `OOF_STEP_SCORES.npz`
    - ssl-pseudolabel-residual-v1 `results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv`, `results/digit_family_extension_v1/FEATURES.npz`, `lsml_merge_step_v1` and `stage_b2` `STEP_SCORES.npz`
    - temporary scratchpad `pool_z.npy` and `pool_names.json` (MISSING: hash check will hard-stop)
- `ssl_s0_audit.py`, `ssl_s0c_calibration.py`, `ssl_s0c_ct7_diagnostic.py`, `ssl_s1_run.py`: readout-quickest-detection-v1 (step_evidence_v1). `ssl_s0_audit.py` and `ssl_s1_run.py` also read `MAIN/results/localization_source_group_audit_v1/FOLDS_V2.json`.
- `ssl_s2_run.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`.
- `ssl_s5_run.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`; `MAIN/results/localization_source_group_audit_v1/FOLDS_V2.json`.
- `tail1_transfer_v3_run.py`, `tail_threshold_calibration_run.py`: temporary scratchpad.
- `tail_label_share_run.py`: readout-quickest step_evidence_v1; temporary scratchpad.
- `tail_lsml_banks_run.py`, `tail_weighted_fusion_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; temporary scratchpad.
- `test_calfix_common.py`: depth-feature-fusion-v1 (sys.path).

### origin/claude/ssl-pseudolabel-residual-v1, other scripts

- `scripts/fit_algorithm_external_v1.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1.
- `scripts/score_algorithm_external_v1.py`: depth-feature-fusion-v1; `MAIN/results/lsml_external_generalization_v1/evaluation`; `MAIN/scratch/external_generalization_private/inputs`.
- `scripts/evaluate_algorithm_external_v1.py`: `MAIN/scratch/external_generalization_private/{inputs,sources}`.
- `scripts/extract_external_banks_v4.py`: `MAIN/scratch/external_generalization_private`.
- `scripts/verify_external_banks_v4_source.py`:
  - readout-quickest `OOF_STEP_SCORES.npz`
  - cumulative-vote-fusion-v2 `ct7_profiles_v1`
  - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`
  - provenance strings pointing into cumulative-vote-fusion-v2 and token-probability-fusion-v1 scripts
- `scripts/fit_external_source_bundle.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`, `DERIVATIVE_CHANNELS.npz`, `chosen_token_calibration_v1/CT7_DEV_SCORES.npz`.
- `scripts/verify_external_source_strict.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz` and `DERIVATIVE_CHANNELS.npz`; depth-feature-fusion-v1 `results/step_level_bank_baseline_v1/RESULTS.json`.
- `scripts/audit_family_tail_handoff.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`; readout-quickest `OOF_STEP_SCORES.npz` and `profiles_full.npy`.
- `scripts/audit_prmbench_lsml_saved_scores.py`:
  - depth-feature-fusion-v1 results and source
  - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`
  - readout-quickest step_evidence_v1
  - consolidation-fusion-2026-09-22 `results/digitfree20_ladder_v1/RUN.json`
- `scripts/repair_runtime_fusion_reports.py`: ssl-pseudolabel-residual-v1 `results/ssl_pseudolabel_residual_v1/<stage>/run_20260923`.
- `scripts/review_digit_alternative_probability_v1.py`: lsml-ct7-levers-run `spectral_utils/fusion_signal_registry.py` (via `ROOT.parents[1]`).
- `scripts/run_digit_family_extension_v1.py`: token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`.
- `scripts/index_month_lsml_prm_results.py`: the whole `.worktrees` directory.
- `scripts/selector_deep_report.py`: `.worktrees/antigravity` (ABSENT).
- `scripts/localization/our_arm.py`: derives the main tree from the `.worktrees/<name>` layout.
- `scripts/window_answer_local_fusion_v2.py`: usage example `--source-root .worktrees/lsml-ct7-levers-run`.
- Read `scratch/external_generalization_private/` relative to their own checkout root, so they need that folder inside whichever checkout runs them:
  - `scripts/evaluate_family_external{,_v2,_v3}.py`
  - `scripts/run_family_external.py`
  - `scripts/run_external_local_cpu.py`
  - `scripts/verify_external_official_metrics.py`
  - `scripts/archive_external_evaluation.py`
  - `scripts/audit_external_backbone_diversity.py`

### origin/claude/estimator-provenance-collection-2026-09-29 (files differing from the SSL tip)

- `scripts/experiments/er_stage_a.py`: cumulative-vote-fusion-v2/scripts/experiments.
- `scripts/experiments/tensor_mom_stage_a_run.py`:
  - depth-feature-fusion-v1
  - token-probability-fusion-v1/results/token_probability_fusion_v1
  - cumulative-vote-fusion-v2 ct7_profiles_v1
  - readout-quickest step_evidence_v1

### claude/decision-rule-v1

- `scripts/experiments/answer_features_extract.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`; readout-quickest step_evidence_v1.
- `scripts/experiments/answer_gate_run.py`:
  - depth-feature-fusion-v1
  - ssl-pseudolabel-residual-v1
  - readout-quickest step_evidence_v1
  - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`
  - cumulative-vote-fusion-v2 `ct7_profiles_v1/{profiles.npy, PROFILE_VALIDATION.json}`
- `scripts/experiments/decision_rule_run.py`: the same as `answer_gate_run.py`, plus hashes of cumulative-vote-fusion-v2 `cvf_v2/{em,core}.py`.

### claude/self-generated-step-labels-v1

- `scripts/self_generated_labels/b16_fit.py`:
  - ssl-pseudolabel-residual-v1 (sys.path `scripts/experiments`; `results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json`; `results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz`)
  - readout-quickest step_evidence_v1
  - `MAIN/dataset_cache/repgrid/...` (own-answer pickles and `pb_qwen3_8b/processbench_gsm8k.pkl`)
- `scripts/self_generated_labels/b16_eval.py`: ssl-pseudolabel-residual-v1 (`per_dataset_fit_v1` `STEP_SCORES.npz`); readout-quickest `OOF_ANSWERS.csv`.
- `scripts/self_generated_labels/extract_own_answers.py`: `MAIN/dataset_cache/repgrid/<cell>/{manifest.json, raw_*.pkl}`.

### lsml-ct7-levers-run (files differing from the SSL tip)

- `scripts/experiments/ct7_token_tail_lsml_calfix_v1.py`: `C:\Users\omris\TAU\hallucination_detection\.worktrees\ssl-pseudolabel-residual-v1\scripts\experiments` (sys.path).
- `scripts/experiments/pb_tail_weights_v1.py`:
  - temporary scratchpad
  - ssl-pseudolabel-residual-v1 `results/tail_threshold_calibration_v1/TRANSFER_LOCK_V2.json` and `results/tail1_transfer_v3/run_20260924_2259` (via the imported SSL path)
- `scripts/diagnostics/evidence_domain_fusion_probe_v1.py`, `evidence_domain_independence_v1.py`, `position_length_vs_ct7_v1.py`: `.worktrees\fusion-independence-atlas-v1` (ABSENT). The same scripts on `claude/readout-quickest-detection-v1` read the same absent worktree.
- `scripts/joint_lsml_localization/{evaluate_existing_v1, _r1, _r2, evaluate_processbench_amendment_v1, run_existing_v1}.py`, `scripts/reasoning_localization/{register_h3_llama_transfer, run_phase1_baseline}.py`: `/Users/osegev/Desktop/hallucination_detection/.worktrees/` (a macOS path from another machine).
- `scripts/reasoning_localization/register_h3_historical_headtohead.py`: `/Users/osegev/Desktop/hallucination_detection/.worktrees/reconstruction-science-run-v1` (a macOS path; that worktree is ABSENT here too).

### claude/whitebox-layer-views-v1

- `scripts/run_whitebox_kill_test_1a.py`, `scripts/run_whitebox_kill_test_1a_followup.py`, `scripts/run_whitebox_kill_test_1b.py`: `ROOT.parent/token-probability-fusion-v1/results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz`. This is a sibling-worktree relative path, so it only works when run from inside `.worktrees/`.

### codex/lsml-external-generalization-v1 and claude/readout-quickest-detection-v1

- `codex/lsml-external-generalization-v1`: no runner differs from the SSL tip on these checks. The branch is an ancestor of SSL.
- `claude/readout-quickest-detection-v1`: only the two atlas diagnostics listed above.
