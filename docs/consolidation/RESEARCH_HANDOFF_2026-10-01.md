# Research handoff checklist, 2026-10-01 consolidation (verified 2026-10-02)

This file is the research-direction checklist for the 2026-10-01 consolidation. It covers the 21 directions in the review
`docs/reviews/CONSOLIDATION_REVIEW_20261001.md` (table rows on lines 99-119 @rescue) and applies Omri's decisions 1-21 in
`docs/consolidation/DECISIONS_2026-10-01.md` @rescue (83 lines at 6173c7354). It does not choose the next experiment.
**No experiment starts automatically (decision 21).**

## Summary

| Entry | Main tag | Next step | Blocked by |
|---|---|---|---|
| 1-2. Final fusion; level-family removal | OMRI DECISION | Build an SML/MoM-based combination from the Step 457 runner-up; separately, remove 2-3 level channels | Omri's go (decision 21); restore pool files; merge the estimator branch for MoM |
| 3. Tensor method of moments | COMPLETED FINDING (negative for this implementation) | Reopen only under the written reopening conditions | Estimator branch not in the newest line |
| 4. Position | PROPOSED EXPERIMENT (Claude) | Choose among all options after the merge | Omri's choice; restore pool files |
| 5-6. Fitting access; digits and Joint | OMRI DECISION | No experiment. Carry the decisions into the consolidated CLAUDE.md | The merge |
| 7, 18. Decision rule; PRMScore reporting | PROPOSED EXPERIMENT (Claude) | Omri answers questions 1 and 2 | Omri's answer |
| 8-9. Whitebox gate; length confounding | OUT OF SCOPE | None for now (decision 14) | Not applicable |
| 10. External reevaluation | PROPOSED EXPERIMENT (Claude) | Exploratory check under the per-dataset contract | Omri's choice |
| 11. Untouched confirmation | OMRI DECISION (MedPRMBench deferred) | Lock the method, then the AIRCC gates | Method/decision lock; budget; missing adapter |
| 12. Transfer proxy | OMRI DECISION (approved, decision 17) | Write the scope, protocol and success criterion, then run one bounded test | The protocol must be written first; Omri's go to start |
| 13. Family-tail V2/V3 audits | OMRI DECISION (approved, decision 16) | Run the three audit scripts on V2 and V3 | Omri's go to start (decision 21) |
| 14-15. Published comparators; second model | OMRI DECISION / PROPOSED EXPERIMENT | Comparator inference; source-side cross-backbone study | GPU budget; Omri's go |
| 16. Self-generated labels | PROPOSED EXPERIMENT (Claude) | Omri chooses option 1, 2 or 3 | Omri's choice; budget for option 2 |
| 17. Supervised PRM | COMPLETED FINDING + DEFERRED | (a) none; (b) deferred, not closed (decision 18) | Not applicable |
| 19-20. Saturated implementations; historical backlog | COMPLETED FINDING + PROPOSED dispositions | Write the archive index and dispositions | Documentation only |
| 21. Technical debt; advisor material | DEFERRED / OMRI DECISION | Monday material in parallel (decision 19); MATH regrade deferred | Three uploads still failing |

## How this was verified

A draft was built read-only on 2026-10-02 from the sources with `git show <ref>:<path>` and on-disk reads. Six independent
agents then checked the draft against the git sources, entry by entry, and returned OK / ERROR / GAP findings. The corrections
were applied on 2026-10-02. Omri's decisions 14-21, committed at `@rescue` 6173c7354, were also applied. Doubtful claims were
spot-checked with `git show` before they were used: the tensor stage-A bar, IMM being implemented, the group-confidence
reopening text, the PRM-7B scorer, the full-sampling short-error diagnostics, the readout Step 429 paragraph and the macOS path
list. Backup facts come from the upload manifests in the session scratchpad
(`...\ae2dd164-...\scratchpad\upload\MANIFEST_v2_run2.tsv`; an untracked copy is at
`results/consolidation_2026-10-01/MANIFEST_v2_run2.tsv` in the rescue worktree) and from the tar file lists in the same folder.
Anything that could not be confirmed is listed under "Unresolved verification gaps".

**Tag legend.** Every status line carries exactly one tag:

- COMPLETED FINDING: a measured result in a committed source.
- OMRI DECISION: Omri's own decision.
- PROPOSED EXPERIMENT (proposer named): also covers proposed maintenance or reporting tasks.
- DEFERRED.
- OUT OF SCOPE.
- UNRESOLVED GAP.

A negative result closes an *implementation*. Only Omri closes a *direction*. The one explicit direction stop is Joint L-SML
for new arms (decision 2).

## Refs

Tips were read on 2026-10-02. These short names are used below:

- `@ssl` = `origin/claude/ssl-pseudolabel-residual-v1` at 553de79c6. This is the newest main line (worktree `.worktrees/ssl-pseudolabel-residual-v1`).
- `@estimator` = `origin/claude/estimator-provenance-collection-2026-09-29` at aa8bf95d5 (worktree `.worktrees/tensor-mom-v1`). It is NOT an ancestor of `@ssl`.
- `@decision-rule` = `claude/decision-rule-v1` at cbe345a4b. Not in `@ssl`.
- `@self-labels` = `claude/self-generated-step-labels-v1` at 2465062e5. Not in `@ssl`.
- `@ct7-levers` = `lsml-ct7-levers-run` at 1503e529a, the same commit as `origin/claude/lsml-ct7-levers-v1`. Not in `@ssl`.
- `@whitebox` = `claude/whitebox-layer-views-v1` at f2f533795. Not in `@ssl`.
- `@readout` = `claude/readout-quickest-detection-v1` at cf70933d1. Not in `@ssl`.
- `@codex-external` = `codex/lsml-external-generalization-v1` at bef0b1de3 (main checkout). It is an ancestor of `@ssl`.
- `@rescue` = `origin/rescue/main-checkout-loose-files-2026-10-01` at 6173c7354 ("Consolidation decisions 15-21 (Omri, 2026-10-02)").
  The rescue worktree's uncommitted state: `docs/consolidation/RECONCILIATION_2026-10-01.md` is modified, and
  `docs/consolidation/line_status_proposals/` and the pass-2 manifests/scripts under `results/consolidation_2026-10-01/` are untracked.

## Cross-cutting facts

**Pool inputs: UNRESOLVED GAP (blocks runners until restored).**

- `pool_z.npy` and `pool_names.json` are gone from the temporary scratchpad
  `C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad\`.
  The folder still exists.
- `results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json`@ssl:33,38 still points there.
- Four runners resolve their inputs through that manifest and hard-stop on a SHA mismatch (`per_dataset_fit_run.py`@ssl:73-76 and
  its equivalents): `per_dataset_fit_run.py`, `position_channel_run.py`, `position_prior_run.py`, `partition_switch_run.py`.
- `algorithm_decisions_run.py`@ssl:115 also hard-codes the same scratchpad (`SCR = Path(r'C:\Users\DELL\...2d14a8c9...\scratchpad')`).
- Two verified restore sources exist:
  - the hash-verified local backup `.worktrees/ssl-pseudolabel-residual-v1/results/algorithm_decisions_v1/inputs_backup/`.
    `pool_z.npy` is 60,568,480 B, SHA-256 equal to the manifest, git-ignored. `pool_names.json` and `HASHES.json` are tracked @ssl.
    It is in the SSL Drive archive record (`docs/archives/DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.md`@ssl).
  - Codex's Drive archive `family_tail_external_v1_results_a1e2f683cc6932a9.tar.gz`, member `reproduction/source_pool/`
    (`results/family_tail_external_v1/RESTORE.md`@ssl; `ARCHIVE.json` upload_status "SHA256_VERIFIED_ON_GDRIVE").

**Decision text location: COMPLETED FINDING.**

- Only `@ssl` carries the 2026-09-30 block (digits stay) and the 2026-10-01 block (averaging is not final; Joint stopped) in
  `CLAUDE.md`.
- These open with the 2026-09-24 block, then 2026-09-23, then 2026-09-17: the main checkout, `@decision-rule`, `@self-labels`,
  `@estimator` and `@rescue`.
- These still open with the 2026-09-17 blocks (digit exclusion, feature selection inside Joint): `@ct7-levers`, `@readout` and
  `@whitebox`.

**Upstream inputs read by nearly every runner: COMPLETED FINDING (now on Drive).** These live in other worktrees and are not in git:

- `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/OOF_STEP_SCORES.npz` (ignored) and `profiles_full.npy`.
  `profiles_full.npy` is untracked and NOT ignored, so `git clean` would not protect it.
  `OOF_ANSWERS.csv` and `INPUT_FREEZE.json` are tracked.
- `.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/{TOKEN_MATRICES.npz, DERIVATIVE_CHANNELS.npz}` and
  `results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz` (ignored).
- `.worktrees/cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy`.
- `.worktrees/depth-feature-fusion-v1/results/step_level_bank_baseline_v1/STEP_SCORES.npz`.

All of these exist locally. Pass 2 uploaded each one inside its worktree's off-git tar, verified by local-stream MD5 = Drive MD5
and by member count (manifest rows `worktree_offgit_{readout-quickest-detection-v1, token-probability-fusion-v1,
cumulative-vote-fusion-v2, depth-feature-fusion-v1}`: 2,175 / 113 / 1,296 / 35 members). Each file name appears in the
corresponding tar list `wt_<worktree>.txt` in the upload scratchpad.

**Backup state after pass 2.** The Drive root is
`gdrive:hallucination_detection/consolidated_results/local_backup_2026-10-01/`.

- COMPLETED FINDING: completed and verified by local-stream MD5 = Drive MD5 and tar member count:
  - `localization_full_sampling_v3` (28,008 files)
  - `main_checkout_ignored_results` (36,143 files)
  - re-verification of the six pass-1 result tars: `lsml_external_generalization_v1`, `localization_full_benchmark_v3`,
    `localization_full_shortlist_v3`, `fusion_shrinkage_iu_v1`, `fusion_multiwidth_iu_v1`, `fusion_onset_innovation_iu_v1`
  - off-git files of 12 worktrees: readout-quickest-detection-v1, whitebox-layer-views-v1, token-probability-fusion-v1,
    cumulative-vote-fusion-v2, self-generated-step-labels-v1, decision-rule-v1, depth-feature-fusion-v1,
    digit-alternative-probability-v1, family15-tail20-transfer-v1, consolidation-fusion-2026-09-22, a6-s0b, tensor-mom-v1
  - the 6,646 previously missing files of `scratch/external_generalization_private` (7.65 GB)
  - the 2 missing backup-pre-lfs-fix LFS objects, the GPQA pickles (`rclone check`, 0 differences)
- COMPLETED FINDING: covered by a previously verified backup. The SSL worktree archive was verified 367/367 on 2026-09-30.
- UNRESOLVED GAP: three items FAILED and are not yet on Drive. The cause was the Drive 403 per-minute quota on rclone's shared
  client_id. The local archives are complete and a retry is running. Do not delete these locally:
  - `fusion_multiwidth_dense_v1` (27,546 members)
  - off-git files of `er-generality-v1` (37)
  - off-git files of `lsml-ct7-levers-run` (43; includes `results/prm_vs_ct7_prmbench_v1/PRM_OOF_DECISIONS.npz`)
- COMPLETED FINDING: the `lsml_group_confidence_v1` ignored results ARE inside `main_checkout_ignored_results`. Its list
  `main_ignored.txt` (36,143 lines = the verified member count) has `results/lsml_group_confidence_v1/{CALIBRATION, EVALUATION,
  PREDICTIONS}.npz` at lines 22245-22247.

---

### 1. Final fusion and clustering

- **Source:**
  - `CLAUDE.md`@ssl:3-19 ("Omri decision update - 2026-10-01").
  - `Research_Directions.md`@ssl:3757-3768.
  - `results/algorithm_decisions_v1/SUMMARY.md`@ssl:22-29 (table) and 34-50 (decisions).
  - `DECISIONS_2026-10-01.md`@rescue item 1.
  - The review row `CONSOLIDATION_REVIEW_20261001.md`@rescue:99.
- **Current status:**
  - OMRI DECISION (2026-10-01): "The plain average is not acceptable as the final fusion step" (`CLAUDE.md`@ssl:6).
    The frozen candidate (Dawid-Skene filter, then plain average) is still reported as the current result, also to the advisors.
    "The next direction for the fusion step is to replace the plain average with a combination that uses the label-free SML or
    method-of-moments (tensor) estimates" (`CLAUDE.md`@ssl:8-10). The starting point is the Step 457 runner-up: filter,
    label-free partition plus one absorption merge, latent-group EM (HEM) within groups, Dawid-Skene between groups
    (DECISIONS item 1). Only the wording "closest existing recipe" in `Research_Directions.md` is Claude's.
  - COMPLETED FINDING (Step 457): the frozen rule selected the filter plus plain average (8-bank mean within-answer AUC 0.7656,
    `SUMMARY.md`@ssl:24). The runner-up (HEM within, Dawid-Skene between) has a mean of 0.7670 (`SUMMARY.md`@ssl:25), +0.0014.
    It has broad content gains on 32 and 32+digits (+0.0145 / +0.0100) and thin gains on 13 and 51+digits, "concentrated in 1% of
    answers". It has broad losses on 20+digits (-0.0097) and 51 (-0.0074) (`SUMMARY.md`@ssl:38-41). Bank 20 is also below the
    candidate (0.7526 vs 0.7540, `SUMMARY.md`@ssl:24-25); the source does not call that loss significant. The frozen rule requires
    no *significant* loss on any bank: a bank loses when the 95% interval of (variant - base) is entirely below 0
    (`PROTOCOL.json`@ssl). "No variant avoided a significant loss on all 8 banks" (`SUMMARY.md`@ssl:38). L-SML is below the
    plain average on all 8 banks (`SUMMARY.md`@ssl:46).
- **Next step:** develop the SML/MoM combination from the runner-up. Address, "rather than ignore", the known obstacle: the
  estimates select channels but have not weighted them better than averaging (Steps 450-452, 457, 464; `CLAUDE.md`@ssl:10-12).
  The sources give no protocol.
- **Dependencies:**
  - Code @ssl:
    - `scripts/experiments/ds_group_weights.py` (`hem_fit` l.24, `mle_group_weights` l.86; `tests/test_ds_group_weights.py`)
    - `scripts/experiments/lsml_merge_step.py` (`absorb_merge` l.24)
    - `scripts/experiments/algorithm_decisions_run.py` and `algorithm_decisions_parallel.sh`
    - `scripts/experiments/per_dataset_fit_run.py`: "Use it as the base for future per-dataset experiments"
      (`docs/HANDOFF_LABEL_FREE_ALGORITHM_2026-09-30.md`@ssl:42-43)
    - `spectral_utils/prmbench.py`. It is tracked @ssl; its blob is identical to the depth-worktree copy that the runners put
      first on `sys.path`.
  - `tensor_mom_estimate` exists only in `scripts/experiments/er_stage_a.py`@estimator:80.
  - Data:
    - the pool files (cross-cutting; restore first)
    - `results/digit_family_extension_v1/FEATURES.npz` and `results/indbank_lsml_prmbench_v1/POOL_STRUCTURE.csv`, both tracked @ssl
    - the upstream worktree inputs (cross-cutting)
    - the Step 457/460/461 `STEP_SCORES.npz` replay references (ignored in the SSL worktree; in the SSL Drive archive record)
  - Compute: CPU. The Step 457 grid ran as 4 local processes (`algorithm_decisions_parallel.sh`@ssl).
  - Blocking: Omri's go to start (decision 21). The quote "after the merge, among all options on the table"
    (`Research_Directions.md`@ssl:3739) refers to the position option list in entry 4, not to this direction.
- **Known risks:**
  - The runner-up's +0.0014 is post hoc and has broad losses on two banks.
  - Dawid-Skene overestimates prevalence (0.28-0.41 on channels vs 0.14) and gives the level group the top weight in 40/40
    bank-folds (`SUMMARY.md`@ssl:44-45).
  - All the evidence is PRMBench development data; selection used PRMBench outcomes (`SUMMARY.md`@ssl:12).
  - The runners read four other worktrees and the missing scratchpad by absolute path (`algorithm_decisions_run.py`@ssl:18-44, 115).

### 2. Level-family removal

- **Source:**
  - `CLAUDE.md`@ssl:15-18 ("Proposed by Omri 2026-09-27; no experiment has tested it yet").
  - `Research_Directions.md`@ssl:3766-3768. The quoted wording is from here.
  - DECISIONS item 3 @rescue.
- **Current status:** OMRI DECISION (proposed 2026-09-27, kept on 2026-10-01): "drop 2-3 level-family channels to break the level
  group's dominance (...), then re-run the filter, partition and estimates" (`Research_Directions.md`@ssl:3767). Omri's caveat:
  going from 5 to 3 level channels need not leave two level groups (DECISIONS item 3). It has never been run.
- **Next step:**
  - Remove 2-3 level-family channels.
  - Re-run the Dawid-Skene filter, the label-free partition and the estimates.
  - "Check the resulting partition, not only the score" (`Research_Directions.md`@ssl:3768).
  - "Matched controls" is the review's addition (`CONSOLIDATION_REVIEW`@rescue:100).
- **Dependencies:**
  - The same code and data as entry 1. The banks come from `BN = MI['banks']` (`per_dataset_fit_run.py`@ssl:86).
  - The level bank is `DERIVATIVE_CHANNELS.npz` (token-probability worktree; ignored locally; on Drive since pass 2).
  - CPU.
  - Blocked by Omri's go (decision 21) and the pool restore.
- **Known risks:**
  - Which channels to drop must be chosen label-free and declared before scoring.
  - Fewer level channels may only change the partition, not the weighting.

### 3. Tensor method of moments

- **Source:**
  - `results/tensor_mom_v1/NEGATIVE_RESULT.md`@estimator: hypothesis l.3, verdict l.26-29, reopening l.31-32.
  - `PROGRESS.md`@estimator, section "Tensor MoM run: negative for estimation, candidate unchanged".
  - `HISTORY.md`@estimator:18852 (Step 464).
  - `docs/research_notes/ESTIMATOR_PROVENANCE_2026-09-29_HE.md` (@estimator only).
- **Current status:**
  - COMPLETED FINDING (Step 464), negative for this implementation only ("[x] closes this IMPLEMENTATION only").
    - Estimated prevalence is 0.231-0.232 against a truth of 0.137-0.141. Dawid-Skene and HEM give about 0.28.
    - Per-channel sensitivity MAE is 0.1785 against Dawid-Skene's 0.1178.
    - MoM keeps the same channels as Dawid-Skene on 5/5 folds.
    - The stage-A bar **passes on 0/5 folds** for every estimator, i.e. fails on 5/5 (`NEGATIVE_RESULT.md`@estimator table;
      `HISTORY.md`@estimator:18852). The PROGRESS line "Bar failed 0/5" is ambiguous and should not be copied.
  - DEFERRED: the fixed-20%-threshold self-consistency idea was "Deprioritized after discussion" (`PROGRESS.md`@estimator:9).
    This is not a decision on the reopening questions.
- **Next step:** reopen only with "votes whose dependence is removed before estimation (position-residualized channels, or one
  vote per declared block), or a marking rule that is not a uniform per-answer quantile; evaluated against the same stage-A truth
  on the same folds" (`NEGATIVE_RESULT.md`@estimator:31-32).
- **Dependencies:**
  - `scripts/experiments/tensor_mom_stage_a_run.py` and `er_stage_a.tensor_mom_estimate` exist @estimator only;
    `tests/test_er_stage_a.py` passes 10/10.
  - `er_stage_a.py` imports `cvf_v2` from `MAIN/.worktrees/cumulative-vote-fusion-v2/scripts/experiments` (l.24-31). Three tests
    fail "in a cloud checkout" without it (`docs/HANDOFF_COLLECTION_2026-09-29.md`@estimator:143-144).
  - Stage-A inputs: `results/expectation_realization_v1/run_20260927/INPUT_MANIFEST.json` (tracked @estimator and @ssl; all 8
    input paths exist locally).
  - CPU.
  - The estimator branch must be merged. The tensor-mom-v1 worktree's off-git files are on Drive since pass 2 (6 members).
- **Known risks:**
  - A merge that takes only `@ssl` loses Steps 463-464 and the estimator.
  - Step 463 collides: `HISTORY.md`@self-labels:18363 vs `HISTORY.md`@estimator:18832. Keep both blocks, tagged.

### 4. Position

- **Source:**
  - `Research_Directions.md`@ssl:3739-3749 (Claude's proposals 1-3) and 3722 ("Decided or closed in this line").
  - `HISTORY.md`@ssl:18824 (Step 462).
  - `docs/line_status/claude__ssl-pseudolabel-residual-v1.md`@ssl:44-48.
  - Controls: `docs/HANDOFF_LABEL_FREE_ALGORITHM_2026-09-30.md`@ssl:73-74 and `LESSONS.md`@ssl:346-350.
- **Current status:**
  - COMPLETED FINDING (Steps 460-462; `results/{position_channel_v1,position_prior_v1,per_dataset_fit_v1}/SUMMARY.md`@ssl):
    - The position channel's direction is identified without labels; its weight is not (`Research_Directions.md`@ssl:3730).
    - The prior's weight is 2-5x below the PRMBench optimum (3731).
    - Fitted per dataset, the filter keeps position on PRMBench (4/4) and drops it on ProcessBench (32/32) (3741; `HISTORY.md`@ssl:18828).
    - On ProcessBench, the prior latches onto a start-of-answer telemetry artefact (3732-3733). The count is 17/32 grouped fits,
      but the source words it two ways: "predict step 0 always" (3734) and "predicts step 0 in about 95% of answers"
      (`HISTORY.md`@ssl:18828).
  - PROPOSED EXPERIMENT (Claude; none started) (`Research_Directions.md`@ssl:3739-3749). Three distinct proposals:
    1. Use the filter's per-dataset position decision as the gate for the stronger prior.
    2. Set the position weight from the cross-fitted slope without re-weighting the content ("plain average + logit(pi_b)/a_cf").
    3. Remove the per-dataset position profile of the content score before any position term. The source calls this "a
       safeguard, not a method": -0.7 to +1.0 points vs argmax, better on 4/8.
- **Next step:**
  - The choice is made "after the branch merge, among all options on the table" (`Research_Directions.md`@ssl:3739;
    line_status@ssl:44). That means the whole list in `Research_Directions.md`@ssl:3739-3755, not just these three.
  - Proposal 1 "was suggested by Step 462's outcome, so it needs a fresh frozen protocol and, for a claim, data other than
    PRMBench/ProcessBench" (3742-3743).
  - Every position gain carries the step-index row, the same-length swap null and an equal-share comparison
    (handoff@ssl:73-74; `LESSONS.md`@ssl:349).
- **Dependencies:**
  - Code @ssl: `scripts/experiments/{per_dataset_fit_run, position_prior_ds, position_channel_run, position_prior_run}.py`, with
    `tests/test_position_prior_ds.py`.
  - Data: everything in entry 1, including the pool files. Also `results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz`
    (79,472,175 B, ignored; in the SSL Drive archive record).
  - CPU.
  - Blocked by Omri's choice.
- **Known risks:**
  - A position channel gains more under the swap null by construction, so the content share is descriptive only (handoff@ssl:73-74).
  - PRMBench rewards position; ProcessBench penalizes it ("costs ProcessBench in 21/32 cells", `HISTORY.md`@ssl:18828).
  - Proposal 1 is post hoc.
  - The runners import from the depth worktree and run `git rev-parse HEAD` inside it (`per_dataset_fit_run.py`@ssl:14,47).

### 5. Fitting access

- **Source:**
  - `Research_Directions.md`@ssl:3687-3691 ("Fitting contract (Omri, 2026-09-29)") and 3-6.
  - `Research_Directions.md`@ssl:3647, inside the Step 421-422 block dated 2026-09-18 (heading at 3611).
  - Bank11 contract: `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1.md`@ssl:32-41 and
    `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md`@ssl:16-21.
- **Current status:**
  - OMRI DECISION (2026-09-29): fit "per model per dataset, on that dataset's own unlabeled answers ... Nothing fitted is shared
    across datasets or benchmarks" (`Research_Directions.md`@ssl:3687-3689; also `HISTORY.md`@ssl:18826, `LESSONS.md`@ssl:365).
  - COMPLETED FINDING (Step 462): this contract costs nothing on PRMBench. The ProcessBench numbers of Steps 456-461 are
    superseded by Step 462 (`Research_Directions.md`@ssl:3690-3691).
  - COMPLETED FINDING: the frozen source-fitted bank11 L-SML (external lock) and the answer-local studies are separate contracts.
    The 2026-09-29 contract supersedes the "bank11 leads" statement for this line only (`Research_Directions.md`@ssl:4-5).
  - PROPOSED EXPERIMENT (Claude, 2026-09-18 block): the matched answer-local vs pooled comparison on one fixed bank, recorded as
    "Open, and never run despite being specified three times" (`Research_Directions.md`@ssl:3647).
- **Next step:** the review's wording, not Omri's text: "Keep frozen source-fitted bank11 and answer-local studies as separately
  labelled contracts. Do not mix their metrics or silently apply the new contract to historical results"
  (`CONSOLIDATION_REVIEW`@rescue:103).
- **Dependencies:**
  - Documentation in the consolidated `CLAUDE.md`, `Research_Directions.md` and `PROGRESS.md`.
  - The per-dataset runner is `per_dataset_fit_run.py`@ssl.
- **Known risks:**
  - The main checkout `CLAUDE.md` (on disk) still says "Answer-local fitting remains primary" (line 30) and "primary method still
    fits each answer alone" (line 106). It has no 2026-09-29 contract text.
  - Historical tables mix contracts unless labelled.

### 6. Digits and Joint

- **Source:**
  - `CLAUDE.md`@ssl:21-29 (2026-09-30, digits stay), 13-14 (Joint discontinued), 71-83 (the 2026-09-17 Joint continuation) and
    85-106 (the 2026-09-17 digit exclusion).
  - Commits f7ef911e8 (2026-09-30) and 680c537c6 (2026-10-01), both @ssl.
  - DECISIONS items 2 and 4 @rescue.
  - Historical Joint records:
    - `results/declared_joint_prmbench_v1/`@ssl
    - `docs/JOINT_LSML_METHOD_CARD_2026-09-17.md`@ct7-levers and @rescue
    - `HANDOFF_JOINT_LSML_REVIEW_TO_CODEX_2026_09_04.md`@`origin/codex/og-sml-agent-b-v1`
- **Current status:**
  - OMRI DECISION (2026-09-30): keep the three digit features (digit_alternative, digit_spread, digit_alternative_innovation) in
    the label-free fusion line, banks B16/B23/B35/B54. "It does not by itself reopen digit-specific gates, digit anchors or
    digit-oriented fusion signs" (`CLAUDE.md`@ssl:23-28).
  - OMRI DECISION (2026-10-01): "Joint L-SML is discontinued": do not add it as a new arm; "Its historical results stay as
    records" (`CLAUDE.md`@ssl:13-14). This is the one explicit direction stop.
- **Next step:**
  - No experiment.
  - Two consolidation tasks follow. Neither is in Omri's text.
    - Record both decisions above the historical instructions in the consolidated `CLAUDE.md` (written by the draft).
    - Preserve the Joint studies "and reviewer concerns" as records (`CONSOLIDATION_REVIEW`@rescue:104).
- **Dependencies:**
  - The merge must carry the `@ssl` `CLAUDE.md` blocks.
  - `results/digit_family_extension_v1/FEATURES.npz` is TRACKED @ssl (3,136,228 B), so it is not part of the Drive archive.
- **Known risks:**
  - The 2026-09-17 blocks are still the top instructions on `@ct7-levers`, `@readout` and `@whitebox`.
  - The other branches open with 2026-09-24, but still contain the 2026-09-17 exclusion further down.
  - The 2026-09-17 exclusion still governs gates, anchors and signs. Do not drop it wholesale.

### 7. Answer gate and decision rule

- **Source:**
  - `docs/line_status/claude__decision-rule-v1.md`@decision-rule:16-27.
  - `HISTORY.md`@decision-rule:18800 ("Step 459 [Claude, decision_rule_v1]") and 18808 ("Step 460 [Claude, answer_gate_v1]").
  - `results/decision_rule_v1/SUMMARY.md` and `results/answer_gate_v1/SUMMARY.md`@decision-rule.
- **Current status:**
  - COMPLETED FINDING (Step 459 [decision_rule_v1]):
    - A per-answer flag count from raw channel levels (rule R2, the source's name) raises PRMScore 0.6565 -> 0.6635, Holm interval
      [0.0012, 0.0132] (status@decision-rule:13; decision_rule `SUMMARY.md`:20).
    - A threshold from Dawid-Skene sensitivity/specificity loses "because the estimates are biased".
  - COMPLETED FINDING (Step 460 [answer_gate_v1]):
    - Answer-level detectors find erroneous ProcessBench answers (U-PCR 0.773 AUROC). The edge over mean entropy is length.
    - A gate lifts ProcessBench official F1 mostly generically: random gate with the same unflagged share 0.272, pure gate 0.362
      (answer_gate `SUMMARY.md`@decision-rule:17-24).
    - On PRMBench it lowers PRMScore.
  - PROPOSED EXPERIMENT (Claude, decision-rule line), paused for Omri's question 1 (status@decision-rule:21-23):
    - (a) no gate: keep the frozen answer-z q80 rule;
    - (b) the Step-459 count rule R2, calibrated per benchmark;
    - (c) a pure answer gate on mean token entropy: "leave the least suspicious answers unflagged, frozen rule inside", which
      "helps ProcessBench official F1 but lowers PRMScore".
- **Next step:**
  - Omri chooses (a), (b) or (c). "The choice must be frozen before any external result is seen"; "no new experiment is started
    until they are answered" (status@decision-rule:24, 27).
  - The review adds: "Separate wrong-answer detection, number of flags and localization" (`CONSOLIDATION_REVIEW`@rescue:105).
- **Dependencies:**
  - Code @decision-rule only: `scripts/experiments/{decision_rule_run, answer_gate_run, answer_features_extract}.py`.
  - Data:
    - `results/answer_gate_v1/ANSWER_FEATURES.npz`: 2.5 MB per status@decision-rule:37, 2,599,729 B on disk. Ignored. About
      7 min CPU to regenerate. On Drive since pass 2 (decision-rule off-git tar).
    - The frozen stage-B scores `.worktrees/ssl-pseudolabel-residual-v1/results/expectation_realization_v1/run_20260927_stage_b/STEP_SCORES.npz`
      (49,152,394 B, in the SSL Drive archive record).
    - `TOKEN_MATRICES.npz` ("not archived by this line", status:42) and `DERIVATIVE_CHANNELS.npz` ("upstream input; not
      produced on this line", status:43). Both are on Drive since pass 2 (token-probability off-git tar).
  - CPU.
  - Blocked by Omri's decision.
- **Known risks:**
  - Hard2Verify and Socratic were already inspected in Steps 433 [Codex, external L-SML], 444, 446 and 459
    [algorithm_external_v1] (`HISTORY.md`@ssl:18555, 18800). So no gate choice can make them untouched evidence. The
    decision-rule Step 459 evaluates only PRMBench/ProcessBench.
  - The PRMBench construction prior (about 2 errors per answer) inflates any count rule. The top-2 diagnostic (0.6808) is post hoc.
  - The branch is not in `@ssl`. Its runners read the SSL worktree by absolute path (`decision_rule_run.py`@decision-rule:20-21).

### 8. Whitebox answer gate

- **Source:**
  - `results/whitebox_layer_views_localization_v1/RESULTS_STAGE_1A.md`@whitebox: section 3.3 at l.84 (label-derived
    orientation), section 5 at l.123, section 6 ("Artefacts") at l.148.
  - `RESULTS_STAGE_1B.md`@whitebox, section 7 at l.124.
  - `docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md`@whitebox.
  - DECISIONS item 14 @rescue.
- **Current status:**
  - OUT OF SCOPE FOR NOW (Omri, decision 14). It is kept indexed with its evidence; it is not closed and not a next step.
  - COMPLETED FINDING, stage 1b: "The locator branch ends, as pre-registered" (1B:126).
  - COMPLETED FINDING, stage 1a: a tie against the production CT7 gate, 0.727 vs 0.710 mean ProcessBench AUROC with overlapping
    intervals (1A:116). The errors are complementary: failure phi 0.402. There are 630 answers that the production gate gets
    wrong and the geometry gets right, against 1,048 the other way (1A:118). The 0.727 depends on a sign flip chosen with labels;
    under the contract's own orientation the feature scores 0.273 (1A:87).
- **Next step:** none while out of scope. The evidence notes record what any later restart would need. First, a label-free
  orientation rule ("Blocking before any candidate", 1A:143). Then a gate fusion judged on exact localizations gained vs lost at
  a matched opened fraction, not AUROC (1A:137-138).
- **Dependencies:**
  - Code @whitebox: `scripts/run_whitebox_kill_test_1a{,_followup}.py`, `cluster/reduce_layer_views_answer_level.py`,
    `scripts/land_layer_views_answer_level.py`.
  - Data:
    - `ANSWER_LEVEL.npz` (305,410,017 B, ignored). On Drive since pass 2 (whitebox off-git tar, 37 members).
    - CT7's gate, read through the sibling path `ROOT.parent/token-probability-fusion-v1/.../CT7_DEV_SCORES.npz`
      (`run_whitebox_kill_test_1a.py`@whitebox:47).
    - Raw layer-view captures on Drive `cluster_results/{pb_layer_views_qwen3_4b, pb_layer_views_qwen3_8b,
      prmbench_layer_views_qwen3_8b}` (handoff@whitebox:30-38; recorded by the handoff only).
  - On AIRCC: "Cycle-2 is very likely being retired" (handoff@whitebox:44).
- **Known risks:**
  - The headline number is oracle-oriented (the `max(auc, 1-auc)` trap).
  - The branch is not in `@ssl`.

### 9. Length confounding

- **Source:**
  - `RESULTS_STAGE_1B.md`@whitebox, section 5, l.95-113.
  - DECISIONS item 14 @rescue.
- **Current status:**
  - OUT OF SCOPE FOR NOW (Omri, decision 14). It is kept indexed with its evidence; it is not closed and not a next step.
  - COMPLETED FINDING (diagnostic): after within-answer rank residualization against step length, ProcessBench scores fall
    (1B:101-103). CT7's locator keeps a margin of +3.5 over chance; the two single-statistic arms sit at chance (1B:107).
    - final_lens_H: 33.61 -> 16.79
    - depth_contract: 31.56 -> 16.99
    - CT7 locator: 39.89 -> 20.09
  - The source's request: "This is worth its own follow-up regardless of the white-box line, because it bears on every
    step-level readout in the project" (1B:112-113). No protocol exists.
- **Next step:** none while out of scope.
- **Dependencies:**
  - `scripts/run_whitebox_kill_test_1b.py`@whitebox.
  - `STEP_LEVEL.npz`: 486 MB per 1B:9, 509,199,000 B on disk. Ignored. On Drive since pass 2 (whitebox off-git tar).
- **Known risks:** "a crude control, not a regression residual" (1B:107-108). Only the ordering survives.

### 10. External reevaluation (Hard2Verify, Socratic-PRMBench)

- **Source:**
  - `Research_Directions.md`@ssl:3750-3751 (proposal 4).
  - line_status@ssl:47-48 (option 4).
  - `results/algorithm_external_v1/SUMMARY.md`@ssl (Step 459 [algorithm_external_v1]).
  - `LESSONS.md`@ssl:346-350.
- **Current status:**
  - COMPLETED FINDING (Step 459 [algorithm_external_v1], exploratory):
    - The 13+digits plain average beats CT7 on Socratic, +0.0446 / +0.0424 PRMScore (Qwen3-8B / QwQ-32B, Bonferroni;
      `SUMMARY.md`@ssl:30), "most of the advantage is step position" (30-31).
    - Step index alone beats every method: within-answer AUC 0.7307 on Socratic and 0.8649 on Hard2Verify (`SUMMARY.md`@ssl:18, 41).
  - PROPOSED EXPERIMENT (Claude; none started): re-check under the per-dataset contract, with the step-index row, swap nulls and
    equal-share comparisons (`Research_Directions.md`@ssl:3750-3751).
- **Next step:**
  - Ask whether the filter keeps the position channel on Hard2Verify and Socratic.
  - Compare the method with the step index alone.
  - Label it exploratory: "not an untouched confirmation" (`SUMMARY.md`@ssl:4-5); "Do not tune on these inspected external labels"
    (`Research_Directions.md`@ssl:3662).
- **Dependencies:**
  - Code @ssl only (absent at `@codex-external`): `scripts/{fit,score,evaluate}_algorithm_external_v1.py`,
    `scripts/extract_external_banks_v4.py`, `scripts/verify_external_banks_v4_source.py`, `spectral_utils/external_banks_v4.py`
    (`tests/test_external_banks_v4.py`).
  - Data:
    - `results/external_banks_v4/*/FEATURES.npz` and `reference_arrays/*.npy` (ignored; in the SSL Drive archive record).
    - `algorithm_external_v1/run_20260929/*/PREDICTIONS_UNSEALED.json`, 3 files. They are untracked and NOT ignored; they are in
      the SSL Drive archive record. `SEAL.json` is tracked.
    - The private inputs `scratch/external_generalization_private/{inputs,sources}` in the main checkout (ignored, `.gitignore:248`).
      Before pass 2, only 24 of 6,670 files were matched on Drive (`RECONCILIATION_2026-10-01.md`@rescue:32). Pass 2 uploaded and
      verified the 6,646 missing files (7.65 GB).
  - CPU.
  - Blocked by Omri's choice.
- **Known risks:**
  - The labels have been inspected, so this cannot become untouched confirmation.
  - `score_algorithm_external_v1.py`@ssl:20 and `evaluate_algorithm_external_v1.py`@ssl:105,164 read `MAIN/scratch/...` by
    absolute path. The fitter reads the depth and readout-quickest worktrees (`fit_algorithm_external_v1.py`@ssl:12,31).

### 11. Untouched confirmation

- **Source:**
  - `Research_Directions.md`@ssl:3754 (proposal 6).
  - `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md`@ssl:64.
  - `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1.md`@ssl:4 ("MedPRMBench is deferred."), 59-60 (overlap audit) and 63-74
    (execution gates).
  - `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1_RUNBOOK.md`@ssl.
  - DECISIONS item 17 @rescue.
- **Current status:**
  - OMRI DECISION (2026-09-24 execution update, CLAUDE.md): "MedPRMBench is deferred." On 2026-10-02 (decision 17): "final
    confirmation stays on unexposed data".
  - COMPLETED FINDING: PRMBench, ProcessBench, Hard2Verify and Socratic are all exposed (`Research_Directions.md`@ssl:3754), so no
    benchmark already examined can give untouched confirmation.
  - PROPOSED EXPERIMENT (Claude, proposal 6): run the untouched confirmation on MedPRMBench.
- **Next step:**
  1. Freeze the method and decision rule on source first. The source is "MedPRMBench remains a candidate after method/decision lock
     and the applicable AIRCC preflight/budget gate" (`CONSOLIDATION_REVIEW`@rescue:109). Also "frozen on source"
     (handoff@ssl:64).
  2. Then apply the execution gates (spec@ssl:65-74):
     - validate source fits, adapters and evaluator parity, and pin and tokenize inputs without inspecting external quality;
     - at least 10 GB local/TMPDIR and 50 GB shared free, with CPU smoke and preflight PASS in the submission session;
     - a timing job of at most 12 deterministic examples per dataset/backbone, across length quartiles and including the longest,
       at most 1 allocated GPU-hour per job;
     - measure GPU/CPU hours, memory, storage, wall time and retry costs;
     - Omri's full-run budget decision BEFORE any full inference;
     - run under the same locked recipe.

     The gates were written for Hard2Verify/Socratic. Applying them to MedPRMBench is an inference, supported only by the review's
     "applicable AIRCC preflight/budget gate".
- **Dependencies:**
  - Collector code, tracked @ssl and @codex-external: `cluster/run_external_telemetry.py`, `cluster/submit_external_telemetry.sbatch`,
    `scripts/prepare_external_sources.py`, `scripts/prepare_external_inputs.py`.
  - No MedPRMBench adapter or data contract exists on any ref. A grep for "medprm" across all refs finds only notes and false
    positives:
    - `papers/2604.17282v1.pdf`
    - a base64 match in `notebooks/Spectral_Analysis_Phase5.ipynb`
    - `docs/HANDOFF_COLLECTION_2026-09-29.md`@estimator:103
    - `results/algorithm_external_v1/PROTOCOL.json`
    - the consolidation review
  - AIRCC GPU.
  - Blocked by the method/decision lock and the budget decision.
- **Known risks:**
  - The spec requires auditing questions and source IDs against PRMBench and ProcessBench (spec@ssl:59-60). Its application to
    MedPRMBench has not been written.
  - The adapter does not exist.
  - Any post-hoc choice made on exposed benchmarks must be frozen before the test.

### 12. Source-side transfer proxy

- **Source:**
  - `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md`@ssl: section 3 heading at l.54 ("each needs Omri's go"), items 1-2 at l.56-62.
  - `Research_Directions.md`@ssl:3677 ("Open direction").
  - DECISIONS items 17 and 18 @rescue.
- **Current status:**
  - OMRI DECISION (2026-10-02, decision 17): APPROVED as ONE bounded exploratory test.
    - Its exact scope, protocol and success criterion are documented before it runs.
    - It is not a method sweep and does not block the SML/MoM direction (entry 1).
    - Hard2Verify and Socratic are exposed, so a match with their ranking is exploratory evidence, not proof of transfer.
    - It has not been started (decision 21).
  - COMPLETED FINDING (motivation): source five-fold PRMScore rank did not predict external rank in three locks, V1-V3
    (handoff@ssl:49; `Research_Directions.md`@ssl:3677).
  - DEFERRED, not closed (decision 18): "Why bank11 L-SML transfers" (handoff section 3 item 2, l.62). That covers partition and
    weight stability across source folds and external per-channel behaviour. Its evidence and questions are preserved.
- **Next step:**
  1. Write the scope, protocol and success criterion.
  2. Build one source split that mimics a domain shift. The options named in the source are leave-one-benchmark-out (fit on
     ProcessBench, evaluate PRMBench), leave-one-ProcessBench-cell-out, or a PRMBench category holdout (handoff@ssl:57-59).
  3. Check retrospectively whether it would have ranked bank11 L-SML above the family variants and CT7. Label it "proxy
     validation, not confirmation". "If no proxy tracks transfer, say so and prefer simpler, more constrained representations
     (bank11)" (handoff@ssl:61).
- **Dependencies:**
  - Code @ssl only: `scripts/experiments/{family_tail_calfix_run, family_tail_calfix_eval, calfix_common, calfix_evaluate,
    tail_calib_common}.py`, tests `scripts/experiments/test_{calfix_common,tail_calib_common}.py`.
  - Data:
    - The pool files (restore; see the cross-cutting note).
    - `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/` (handoff@ssl:79; on Drive since pass 2).
    - `results/lsml_external_generalization_v1/evaluation/source/{VALIDATION_FITS,BUNDLE}.json`, tracked @codex-external and
      @ssl. The draft added these; the handoff's section 4 does not name them.
  - CPU.
- **Known risks:**
  - Validating the proxy against the known external ranking uses exposed data.
  - The short merge-handoff outline omitted this item (`CONSOLIDATION_REVIEW`@rescue:110).
  - Every source runner hard-codes the temporary scratchpad (`family_tail_calfix_run.py`@ssl:33; handoff@ssl:78).

### 13. Family-tail audits and library contract

- **Source:**
  - `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md`@ssl: section 3 items 3 and 5 (l.63, 65), l.30, section 5 (l.85-87).
  - `results/family_tail_external_v1/AUDIT_DEFERRED.md` and `RESTORE.md`@ssl.
  - DECISIONS item 16 @rescue.
- **Current status:**
  - COMPLETED FINDING: V1 was checked by Codex's three component audits (metric recompute, coverage, null), which PASS. But "no
    combined `RED_TEAM.md` sign-off is claimed. The aggregate interpretation remains provisional" (`AUDIT_DEFERRED.md`@ssl). Also
    "The independent review remains deferred" (`RESTORE.md`@ssl).
  - COMPLETED FINDING: each V2/V3 run replays every earlier arm exactly, difference 0.0 (handoff@ssl:48;
    `results/family_tail_external_v3/V2_REPLAY.json`).
  - OMRI DECISION (2026-10-02, decision 16): the V2/V3 independent audits are APPROVED as verification of existing work. They have
    not been started (decision 21).
  - PROPOSED EXPERIMENT (Claude, handoff item 5; a maintenance task): guard the library contract. Callers of `lsml_continuous` pass
    z-scored inputs or use `loading_scale='complete'`. `tail_calib_common.lsml_fit_scaled` (`tail_calib_common.py`@ssl:37) enforces
    this for new code.
- **Next step:**
  - Run the three audit scripts on V2 and V3:
    - `results/family_tail_external_v1/independent_metrics/recompute.py`
    - `independent_coverage/audit_population.py`
    - `independent_null/audit_null_math.py`
  - Any hard guard in `spectral_utils/external_generalization` needs a new version; frozen modules are not edited in place
    (handoff@ssl:65).
- **Dependencies:**
  - The audit scripts above, tracked @ssl and @codex-external.
  - Evaluators `scripts/run_family_external_v2.py`, `scripts/run_family_external_v3.py`, `scripts/evaluate_family_external_v2.py`,
    `scripts/evaluate_family_external_v3.py` (handoff@ssl:80).
  - The per-answer records and bootstrap arrays of V1/V2/V3. They are ignored in the main checkout, "v2/v3 can be regenerated from
    the v1 records in minutes" (handoff@ssl:80). They are in `main_checkout_ignored_results` (18,626 `results/family_tail_external_v*`
    paths in `main_ignored.txt`), verified in pass 2.
  - The private inputs in `scratch/external_generalization_private/` are now fully on Drive (pass 2).
  - CPU.
- **Known risks:**
  - `spectral_utils/external_generalization/_bank11/__init__.py` and `spectral_utils/external_generalization/_bank11/chosen_token_calibration.py`
    are committed with CRLF under an `eol=lf` rule. They are hash-locked; do not renormalize them
    (`docs/HANDOFF_COLLECTION_2026-09-29.md`@estimator:149-157).
  - The default `loading_scale='unit'` K criterion is scale-sensitive: features times 0.4 also give K=2 (handoff@ssl:30).

### 14. External published comparators

- **Source:**
  - `docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md`@ssl: l.25, 27, 68, 163-167.
  - `Research_Directions.md`@ssl:18.
  - Spec `LSML_EXTERNAL_GENERALIZATION_V1.md`@ssl:18-23, 92-93.
  - `CONSOLIDATION_REVIEW`@rescue:112.
- **Current status:**
  - OMRI DECISION (2026-09-24 specification): "Teacher-forced telemetry and separate comparator inference are authorized"
    (spec@ssl:18). The comparators are:
    - Hard2Verify: the same Qwen3 critic, Qwen/Qwen2.5-Math-PRM-7B and universalprm/Universal-PRM;
    - Socratic: the QwQ critic and Qwen PRM7 (spec@ssl:22-23).

    This work is unfinished: "published critic/PRM reproduction remains outstanding" (results@ssl:25, 27).
  - COMPLETED FINDING (published context only, not reproduced; results@ssl:165-167):
    - Hard2Verify Balanced F1 53.51 / 42.37 / 60.27;
    - Socratic PRMScore 68.0 (Qwen2.5-Math-PRM-7B) / 73.8 (QwQ-32B critic).
- **Next step:**
  - Reproduce the comparators while keeping "explicit asset/access/inference requirements" (`CONSOLIDATION_REVIEW`@rescue:112).
  - Keep target-calibrated PRM protocols separate from source-only calibration (spec@ssl:92-93).
  - No SOTA claim.
- **Dependencies:**
  - AIRCC GPU with preflight, a timing job and a budget decision (the entry 11 gates).
  - No Hard2Verify/Socratic comparator runner exists. Two reusable pieces exist @ssl and @codex-external:
    - a Qwen2.5-Math-PRM-7B step scorer: `spectral_utils/prm_scorer.py` (`PRM_MODEL_ID` l.24), `cluster/run_processbench_prm.py`,
      `cluster/run_prmbench_prm.py`, `cluster/submit_pb_prm.sbatch.template` (ProcessBench/PRMBench only);
    - `scripts/prepare_external_sources.py`@ssl:37, which prefetches all four models.
  - External inputs `scratch/external_generalization_private/inputs` (12 MB). On Drive: `answers.json` before pass 2; the rest
    (`evaluator_only/*`, `INPUT_MANIFEST.json`, `OVERLAP.json`) since pass 2.
  - Blocked by Omri's budget decision.
- **Known risks:**
  - The published Hard2Verify PRM thresholds were tuned on 100 target responses (results@ssl:165), a different access condition.
  - No paired predictions exist (results@ssl:68).

### 15. New information from another model

- **Source:**
  - `docs/experiments/LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md`@ssl:19-25.
  - `docs/experiments/EXTERNAL_TEACHER_FORCING_REPEATABILITY_20260924.md`@ssl:1-39.
  - line_status@ssl:55-56.
  - `CONSOLIDATION_REVIEW`@rescue:113.
- **Current status:**
  - COMPLETED FINDING:
    - Identical teacher-forcing repeats gave exactly equal saved telemetry for 36/36 answer/backbone pairs (51,637 token
      observations), so they supply no diversity (repeatability@ssl:4-6).
    - Frozen L-SML decisions disagree between Qwen3 and QwQ on 1751/26055 Socratic steps (6.72%); mean within-answer Spearman
      0.9003 (results@ssl:21).
    - "No cross-backbone fusion was fitted or evaluated here."
  - PROPOSED EXPERIMENT (proposer: the 2026-09-24 external results document, Codex line; kept by the review): a bounded source-side
    cross-backbone or residual option. It is "a complementary-view hypothesis, not demonstrated fusion benefit".
- **Next step:**
  - Investigate any residual or cross-backbone extension "on source development data first, using the retained telemetry and CPU
    computation" (results@ssl:25).
  - Freeze before external evaluation.
  - Compare with matched fixed-weight controls.
  - Prompt variants are a separate, changed-conditioning option (repeatability@ssl:32-39).
- **Dependencies:**
  - Source side: ProcessBench telemetry from three scorers in `dataset_cache/repgrid/{pb_qwen3_4b, pb_qwen3_8b, pb_llama31_8b}`
    (LFS, main checkout).
    - The conditions differ: the Qwen3 manifests use `" /no_think"`, thinking off; Llama uses no suffix, thinking on.
    - The `pb_qwen3_*` pickles are COVERED on Drive (`results/consolidation_2026-10-01/bplf_coverage.json`@rescue).
    - Drive coverage of `pb_llama31_8b` is unverified (see gaps).
  - External: Hard2Verify and Socratic telemetry in `scratch/external_generalization_private/evaluation_archives`
    (9,505,309,514 B). Its three `.tar.gz` and 12 gate JSONs were on Drive before pass 2. The 6,193 extracted members were
    uploaded and verified in pass 2.
  - `results/lsml_external_generalization_v1/evaluation/BACKBONE_COMPLEMENTARITY.json` and `scripts/audit_external_backbone_diversity.py`.
  - CPU.
- **Known risks:**
  - The gold-assisted bounds in `BACKBONE_COMPLEMENTARITY.json` are not deployable.
  - The external labels are exposed.
  - "Temperature transforms or duplicated feature columns do not supply independent model evidence" (repeatability@ssl:32-39).

### 16. Self-generated labels

- **Source:**
  - `docs/line_status/claude__self-generated-step-labels-v1.md`@self-labels:17-31 (decision, options 1-3) and 38-50 (dependencies).
  - `docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md`@self-labels.
  - `HISTORY.md`@self-labels:18363 (Step 463 [self-generated step labels]).
- **Current status:**
  - COMPLETED FINDING (Step 463):
    - 890 own answers received final two-judge labels (status@self-labels:9).
    - B16 ranks the first error as well on its own text as on teacher-forced ProcessBench: pooled within-answer AUC 0.757 vs
      0.763, inside the equivalence margin (status:10).
    - Per cell the test is underpowered. GSM8K / Qwen3-4B is borderline: 0.712 vs 0.801 (status:20).
  - PROPOSED EXPERIMENT (Claude, self-labels line), paused for Omri's choice (status:25-31):
    1. Close with the underpowered-cell caveat.
    2. Collect sampled own answers (temperature 1, about 3 per question, same prompts, Qwen3-4B and Qwen3-8B, GSM8K and MATH) on
       AIRCC. Re-label them with the two-judge protocol and rerun the same B16 per-dataset check.
    3. A paired test on existing answers, scoring each with both models (a short AIRCC teacher-forced run).
- **Next step:**
  - Omri chooses 1, 2 or 3.
  - Option 2 needs preflight, a timing run of at most 12 examples per cell, Omri's budget decision and judge runs on the GPT side.
  - Option 3 needs "minutes of GPU".
- **Dependencies:**
  - Code @self-labels: `scripts/self_generated_labels/{extract_own_answers, b16_fit, b16_eval}.py`, `.claude/agents/step-judge.md`.
  - Data:
    - `dataset_cache/repgrid/evdrop_{gsm8k,math}_qwen3_{4b,8b}/raw_*.pkl` (LFS, main checkout). All four plus the pilots are
      matched by SHA-256 and size on Drive `cluster_results/evdrop_*` (`bplf_coverage.json`@rescue).
    - In `.worktrees/self-generated-step-labels-v1/` only (both ignored): `results/self_generated_step_labels_v1/private/own_answers.jsonl`
      (15,309,510 B) and `results/self_generated_step_labels_v1/b16/FEATURES.npz` (5,428,009 B; about 65 min CPU to
      regenerate). Both are on Drive since pass 2 (self-labels off-git tar, 39 members).
  - `b16_fit.py` reads:
    - the SSL worktree: per-dataset code, `INPUT_MANIFEST.json` (l.120), Step 462 `STEP_SCORES.npz` (l.162);
    - the readout-quickest worktree (`OOF_STEP_SCORES.npz`, l.122-124);
    - the token-probability and cumulative-vote worktrees (status:48-50);
    - `MAIN/dataset_cache/repgrid/pb_qwen3_8b/processbench_gsm8k.pkl` (l.137).
  - Blocked by Omri's choice, and a budget for option 2.
- **Known risks:**
  - The quoted final merge plan omitted option 3 (`CONSOLIDATION_REVIEW`@rescue:114).
  - Step 463 collides with the estimator Step 463.
  - The branch is not in `@ssl`.

### 17. Supervised PRM: (a) the measured error-type split, (b) combining it with the label-free method

- **Source:**
  - `docs/line_status/lsml-ct7-levers-run.md`@ct7-levers:13, 39.
  - `HISTORY.md`@ct7-levers:58063 (Step 437): error-type paragraph at 58098-58100, access note at 58122-58124.
  - `results/prm_vs_ct7_prmbench_v1/{MEASUREMENT,PRMSCORE,RUN_FREEZE}.json`@ct7-levers.
  - DECISIONS item 18 @rescue.
- **(a) Current status: COMPLETED FINDING (Step 437).**
  - The supervised Qwen2.5-Math-PRM-7B leads CT7 on within-answer AUROC by +2.878 pp [+2.142, +3.601] (`HISTORY.md`@ct7-levers:58095).
  - It also leads on PRMScore, 0.6804 vs 0.6457 (status@ct7-levers:13).
  - It is below CT7 on argmax hit, 57.78% vs 61.14%. Both hit on 39.02% (`HISTORY.md`@ct7-levers:58111-58112).
  - By PRMBench error category:
    - the PRM leads on confidence (.923 vs .786), counterfactual (.857 vs .713) and deception (.812 vs .643);
    - CT7 leads on redundancy (the source spells it "redundency", .850 vs .654), circular (.807 vs .763) and
      domain_inconsistency (.857 vs .828) (`HISTORY.md`@ct7-levers:58098-58100).
  - No general claim about "structural errors" is made beyond these measured categories (decision 18).
- **(b) Current status: DEFERRED, not closed (Omri, decision 18).**
  - A combination need not be an error-type router; score combination is another option.
  - The open question is whether a combination improves measured performance, and under which information budget.
  - The earlier Claude proposal, "a new line, not a continuation" (status@ct7-levers:39), is kept as evidence.
- **Next step:** none while deferred. The review's conditions stay attached to any restart: "matched access and error-type
  controls"; "Complementary hit sets do not establish a working label-free router" (`CONSOLIDATION_REVIEW`@rescue:115).
- **Dependencies:**
  - PRM rewards: `dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl` (2,053,470 B, ignored; RUN_FREEZE
    sha 95be425c...). It is covered by the 2026-08-20 Drive record `dataset_cache/DRIVE_BACKUP_2026_08_20.json` ("differences": 0).
    It was not re-verified in this pass, and `dataset_cache/` is not part of the pass-2 main-checkout tar.
  - `results/localization_full_benchmark_v3/evaluation/JOINED.{json,npz}` (5,042,974 / 10,626,856 B) and `FOLDS_V2.json`
    (2,040,608 B). All three are committed at full size @ct7-levers. In the main checkout they are untracked / ignored / untracked.
  - `results/cumulative_vote_fusion_v2/ct7_profiles_v1/profiles.npy` (8,153,560 B, a RUN_FREEZE input; status@ct7-levers:52-53).
    It is untracked in the main checkout and covered by the pass-1 verified loose-file upload (`loose_results.txt` line 75;
    `RECONCILIATION_2026-10-01.md`@rescue:12).
  - `CT7_DEV_SCORES.npz` (token-probability worktree; on Drive since pass 2).
  - `results/prm_vs_ct7_prmbench_v1/PRM_OOF_DECISIONS.npz` is in the `lsml-ct7-levers-run` off-git tar, which FAILED in pass 2.
  - `dataset_cache/four_localization/pb_prm_qwen25math7b_full` (4 pickles, in the 2026-08-20 record) is not a Step 437 input. The
    draft added it for a possible ProcessBench combination.
- **Known risks:**
  - The PRM is a supervised, high-access reference row, never a label-free arm (`HISTORY.md`@ct7-levers:58122-58124).
  - The original error-type breakdown page (https://claude.ai/artifact/EArjBd9q2aXnvwoAimYPWr) was not verified. Its builder scripts
    were left in a temporary scratchpad (status@ct7-levers:62). The numbers survive in HISTORY Step 437.

### 18. PRMScore reporting

- **Source:**
  - `docs/line_status/claude__decision-rule-v1.md`@decision-rule:25 (question 2).
  - `results/answer_gate_v1/SUMMARY.md`@decision-rule:36-38.
  - `CONSOLIDATION_REVIEW`@rescue:116.
  - The `CLAUDE.md` 2026-09-23 block ("Separate within-AUC, any-error hit and official PRMScore"). It is present @ssl (l.65),
    @decision-rule, @rescue, @self-labels, @estimator and @codex-external (l.37). It is ABSENT @whitebox, @ct7-levers and @readout.
- **Current status:**
  - COMPLETED FINDING (Step 460, post hoc): "PRMBench inserts about 2 errors per answer". Flagging the top 2 steps per answer, with
    no answer-level score, "gives 0.6808 (post hoc, diagnostic only; it does not transfer to ProcessBench)" (answer_gate
    `SUMMARY.md`@decision-rule:36-38).
  - PROPOSED EXPERIMENT (Claude, decision-rule line; a reporting rule, not an experiment), paused for Omri's question 2: "Should
    every PRMScore report from now on include the fixed-count baseline (top-k steps per answer) and the error-count-per-answer
    structure ...?" (status@decision-rule:25). DECISIONS does not answer it.
- **Next step:**
  - Omri answers question 2.
  - The review: "Retain fixed-count top-k baselines ... k and the top-2 finding were post hoc. Do not present top-k as requiring no
    ranking method. Carry within-AUC, official PRMScore, first-error and no-error metrics separately" (`CONSOLIDATION_REVIEW`@rescue:116).
- **Dependencies:**
  - `scripts/experiments/answer_gate_run.py`@decision-rule.
  - The reporting conventions belong in the consolidated `CLAUDE.md`, which does not exist yet.
  - No compute.
- **Known risks:**
  - Top-2 does not transfer to ProcessBench.
  - Treating 0.6808 as a method would be selection on outcomes.

### 19. Saturated implementations

- **Source:**
  - `docs/line_status/claude__ssl-pseudolabel-residual-v1.md`@ssl:4-7, 51-61.
  - `docs/line_status/lsml-ct7-levers-run.md`@ct7-levers:17-39.
  - `docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md`@ssl:66-71.
  - `results/tensor_mom_v1/NEGATIVE_RESULT.md`@estimator.
  - `results/lsml_group_confidence_v1/NEGATIVE_RESULT.md`@rescue: verdict l.46-53, reopening l.55-59.
  - Readout: `HISTORY.md`@readout:57729 (Step 429; decision paragraph at 57767) and `results/readout_family_v1/`@readout (27
    tracked files).
  - `CONSOLIDATION_REVIEW`@rescue:83, 117.
  - `docs/line_status/claude__decision-rule-v1.md`@decision-rule. It has no SATURATED item; its status is "PAUSED-FOR-DECISION" (l.3).
- **Current status:** each line below is a COMPLETED FINDING, negative for its implementation. The "SATURATED" labels were
  written by the owning sessions (status@ssl:9; status@ct7-levers:5). They are proposals, not closure decisions: "proposals are not
  established closure decisions" (`CONSOLIDATION_REVIEW`@rescue:83).
  - **CT7 weighting** (Steps 434, 436, 441-442, 447): "no learned weighting has beaten equal weighting on ProcessBench"
    (status@ct7-levers:32). It can be reopened by any of these (status@ct7-levers:35-37):
    - a new stream with independent information;
    - the no-error gate or the late-miss readout;
    - an untouched dataset where equal and learned disagree.
  - **Family/tail recipes** (Steps 441-446): no tail variant beats matched equal on source (handoff@ssl:32; status@ssl:19). Do not
    reopen these "without a new idea" (handoff@ssl:66-71):
    - tail-threshold tuning;
    - per-family thresholds;
    - "16 families or within-family SML";
    - token-level tail fusion;
    - FUSE-style objectives.
  - **L-SML on the step bank**: below the plain average on 8/8 banks. Status@ssl:5 cites Steps 438-446 and 456; the evidence at
    status@ssl:52-53 also cites 447, 452 and 457. "Reopen only with channels that are conditionally independent given the label
    (e.g. a second model or internal layer states)" (status@ssl:54-56). This sits beside two Omri decisions:
    - Joint L-SML partitions (Step 440) are discontinued for new arms (decision 2), an OMRI DECISION.
    - A combination using the label-free SML/MoM estimates is the next fusion direction (decision 1).

    So the negative result covers L-SML as the fusion rule on these banks. It does not cover the use of SML estimates inside a combination.
  - **First-error product readout** (Step 462): below argmax in 64/64 cells. Reopen only with "a per-step error rate estimated at the
    true scale (about 0.14 on PRMBench)" (status@ssl:59-61).
  - **Partition switch** (Step 458): 0/80 switches with the family held out. Reopen only with banks that do not share channels
    (status@ssl:57-58).
  - **Readout family** (Step 429): "the label-free consensus rule fails, the extended family is exhausted even when label-selected".
    The Step 429 decision paragraph leaves open "the decision among competing steps on long chains (position prior, non-constant
    hazard, gate) and genuinely new views; HANDOFF_TOKEN_PROBABILITIES 5.1 (fusion before the readout) is still untested"
    (`HISTORY.md`@readout:57767). An untracked proposal,
    `docs/consolidation/line_status_proposals/claude__readout-quickest-detection-v1.md` (rescue worktree, l.41-45), drafts three
    reopening conditions:
    - a new channel from outside the one-pass output distribution;
    - a different decision component;
    - drift-based late-bias corrections only with new evidence against Step 430.
  - **Group confidence** (Codex, `lsml_group_confidence_v1`): 0 wins / 9 losses against family15-equal (`NEGATIVE_RESULT.md`@rescue:23).
    Reopen with "A separately declared source experiment with defensible family-error structure and more faithful reliability
    estimation that beats both matched equal controls and its own ablation on PRMScore before external testing" (l.57-59).
  - **Tensor MoM**: see entry 3.
- **Next step:**
  - No experiment.
  - Carry each implementation's evidence file and reopening condition into the consolidated handoff.
  - A negative implementation does not close an estimator family (`CONSOLIDATION_REVIEW`@rescue:117).
- **Dependencies:**
  - Documentation only.
  - The group-confidence code exists only @rescue: `spectral_utils/lsml_group_confidence{,_experiment}.py`,
    `scripts/run_lsml_group_confidence.py`, `tests/test_lsml_group_confidence.py`.
  - Its three ignored result files (22,786,417 B) were omitted in pass 1. They are inside the verified pass-2
    `main_checkout_ignored_results` tar (`main_ignored.txt` l.22245-22247).
- **Known risks:**
  - The readout line's reopening conditions exist only in an untracked proposal file.
  - The `lsml-ct7-levers-run` off-git files (43, including CT7-line bootstrap and PRM decision arrays) FAILED to upload in pass 2;
    a retry is running.

### 20. Historical transfer and supporting backlog

- **Source:**
  - `CLAUDE.md`@ssl: the 2026-09-07 blocks. "historical24" is at l.152, 737, 779, 806, 833, 855; "historical 24 final-answer cells"
    is at l.1078. The LOW-priority sentence is at l.144; the BOCPD request is at l.76; Step 318 IMM is at l.741.
  - `docs/experiments/LOCALIZATION_RESEARCH_MANDATE_20260907.md`@rescue: the stage table rows D, D2, E2 and E are at l.402-405;
    row F (l.406) and row G (l.407, "Advisor report and historical transfer") are "Pending".
  - Rescued early records @rescue:
    - `docs/reviews/temporal_geometry_revisit_2026-09-07.md`
    - `docs/reviews/bocpd_boundary_audit_2026-09-07.md`
    - `docs/experiments/{FULL_LOCALIZATION_SAMPLING_V3, HISTORICAL_FUSION_REFIT_V3, HISTORICAL_JOINT_REFIT_V3}.md`
    - `papers/digests/{loca-local-conformal-autoencoder, task-based-graph-signal-compression}.md`
  - The eight uncovered historical refs, all of which resolve: `origin/codex/reconstruction-science-results-v1`,
    `origin/codex/reconstruction-benchmark-v1`, `claude/advisor-letter-aug27`, `origin/codex/graph-geometry-selection-v1`,
    `origin/codex/iu-graph-smoothing-ablation-v1`, `origin/codex/deem-b3-moe-gating-v1`, `selector/a4-antigravity-unsupervised`,
    `origin/codex/og-sml-agent-b-v1`.
- **Current status:** each item needs an explicit disposition (completed, superseded or deferred). The dispositions below are
  Claude's proposals unless tagged otherwise.
  - PROPOSED EXPERIMENT (mandate, stage G; still required by `CLAUDE.md`): the historical24 transfer. It has never been run; mandate
    row G is "Pending".
  - COMPLETED FINDING: the full token/window sampling is complete. `results/localization_full_sampling_v3/RUN_STATE.json` (main
    checkout) reads "COMPLETE_REVIEWED_FULL_SAMPLING", 13,769/13,769, report `evaluation/REPORT.html`.
    - Short-error *fitting support* IS measured: `evaluation/DIAGNOSTICS.json` has 64 `error_le32_tokens` rows, scope "sampling-eligible
      only; fitting support is not sparse scoring recall".
    - UNRESOLVED GAP: sparse-*scoring* recall is still unmeasured (spec `FULL_LOCALIZATION_SAMPLING_V3.md`@rescue:82).
  - COMPLETED FINDING: IMM was implemented ("Step318 completed: full-trajectory fusion and supporting IMM", `CLAUDE.md`@ssl:741;
    mandate row D2, l.403). No winner.
  - DEFERRED (LOW-priority supporting backlog, `CLAUDE.md`@ssl:144): LOCA, Diverging Flows, KalmanNet and Shlezinger-inspired
    extensions. Only precursors were tested; LOCA, Diverging Flows and KalmanNet were never implemented. BOCPD is separate: it was
    "explicitly requested ... as an additional supporting component" on 2026-09-17 (`CLAUDE.md`@ssl:76) and is not in the LOW sentence.
  - OMRI DECISION: the Joint continuation proposals are superseded by the 2026-10-01 stop (decision 2).
  - DEFERRED: the RAG / general-task lines, outside the active claim since 2026-09-07.
- **Next step:**
  - Write the disposition per item.
  - Run historical24 only after the localization candidate is locked, with the historical contract and comparators.
  - For historical branches, "an archive with an index is acceptable". The index must record what each line accomplished, its
    evidence, and unfinished or superseded next steps (OMRI DECISION, item 5).
- **Dependencies:**
  - historical24 needs the historical cell caches; their location is unverified (see gaps).
  - `results/localization_full_sampling_v3/` is untracked (not ignored) and on no ref. Pass 2 uploaded and verified it
    (28,008 files).
  - The index work is documentation.
- **Known risks:**
  - An unqualified "saturated" label could silently drop these lines.
  - Many runners on `@ct7-levers` and `@readout` point at absent worktrees or at macOS paths (see the appendix).

### 21. Technical debt and advisor material

- **Source:**
  - MATH grader: `docs/experiments/SELF_GENERATED_STEP_LABELS_V1.md`@self-labels:81-91 and line_status@self-labels:33-35.
  - Paths and LFS: `docs/HANDOFF_COLLECTION_2026-09-29.md`@estimator:23-27, 131-139.
  - Drive client ID: `docs/archives/DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.md`@ssl:25-26.
  - Advisor: `Research_Directions.md`@ssl:3755 and DECISIONS items 1 and 19 @rescue.
  - LFS publication: DECISIONS item 9.
- **Current status:**
  - DEFERRED: MATH regrading. It is recorded as deferred by Omri on 2026-09-29, in agent-written records; there is no verbatim Omri
    quote. `spectral_utils.data_loaders.is_correct_math` misgrades equivalent forms: 81 of 366 non-truncated grader-wrong MATH own
    answers are correct according to both judges (line_status@self-labels:33-35). Old self-generated MATH final-answer results
    remain unverified.
  - PROPOSED EXPERIMENT (maintenance; proposer: estimator collection handoff): portable data paths. Lineage A is "not reproducible
    from its own checkout" (HANDOFF_COLLECTION@estimator:23-27).
  - OMRI DECISION (item 9): publish without uploading LFS objects to GitHub once all required LFS objects are verified on Drive and
    the restore instructions are tested. As of pass 2, the two previously uncovered backup-pre-lfs-fix objects (the GPQA pickles,
    6.17 GB) are on Drive (`rclone check`, 0 differences). The restore test has not been done.
  - PROPOSED EXPERIMENT (maintenance; proposer: SSL archive record): create a personal rclone client_id. The shared one "will stop
    working during 2026". It already caused the pass-2 403 quota failures.
  - OMRI DECISION (2026-10-02, decision 19): Monday advisor material (research summary and presentation) is prepared in parallel,
    without waiting for the merge or the proxy. This supersedes the earlier pause (`Research_Directions.md`@ssl:3755).
    - A deck exists as a private claude.ai artifact (reported by the consolidation session; not opened here).
    - A separate Hebrew preparation set is at `docs/meetings/advisor_prep_2026-10-05/` on `codex/advisor-prep-2026-10-05` (096b17128).
    - Decision 1 allows reporting the current averaging result, with Omri's stated limitation that it is not the final fusion step.
- **Next step:**
  - Regrade offline with a real equivalence checker (CPU), count flips per cell, and rescore old MATH results only if material
    (`SELF_GENERATED_STEP_LABELS_V1.md`@self-labels:89-91).
  - Use repo-relative paths in new runners only. Frozen runners keep their recorded paths (HANDOFF_COLLECTION@estimator:131-139).
  - Test the restore from Drive.
  - Create the client_id.
- **Dependencies:**
  - `spectral_utils/data_loaders.py`. The affected caches are `math500` (via `cluster/run_inference.py`), the AMC/AIME wrappers and
    `evdrop_math_*`.
  - Pass-2 manifest `MANIFEST_v2_run2.tsv`. The pass-2 section of `RECONCILIATION_2026-10-01.md` is still being written in the
    rescue worktree (uncommitted).
- **Known risks:**
  - The Colab-era MATH-500 headline cells are not verified to use the same grader.
  - The 2026-09-17 local `master` consolidation was never pushed, for exactly the LFS reason ("about 17 GB",
    HANDOFF_COLLECTION@estimator:131-139).

---

## Absolute worktree paths in runners

`MAIN` is the hard-coded `C:\Users\omris\TAU\hallucination_detection`. 30 files under `scripts/experiments/`@ssl define it.
"Temporary scratchpad" means `C:\Users\DELL\AppData\Local\Temp\claude\c--Users-omris-TAU-hallucination-detection\2d14a8c9-8b3f-489b-b26f-812b4b84a8b3\scratchpad`;
it exists, but the pool files are missing from it. A short worktree name stands for `MAIN/.worktrees/<name>`.

Worktree status on disk:

- Present: depth-feature-fusion-v1, readout-quickest-detection-v1, token-probability-fusion-v1, cumulative-vote-fusion-v2,
  lsml-ct7-levers-run, ssl-pseudolabel-residual-v1, consolidation-fusion-2026-09-22.
- Absent: fusion-independence-atlas-v1, antigravity, reconstruction-science-run-v1, dufs-moment-selection-v1.

### @ssl, library modules

- `spectral_utils/ssl_eval.py`:10: `MAIN = Path(r'C:\Users\omris\...')`; `W = MAIN / '.worktrees/readout-quickest-detection-v1'`.
- `spectral_utils/external_banks_v4.py`:56-62: provenance strings pointing into other worktrees.

### @ssl, scripts/experiments/*.py

- `algorithm_decisions_run.py`:
  - depth-feature-fusion-v1 (sys.path import; hashes `spectral_utils/prmbench.py`)
  - token-probability-fusion-v1
  - readout-quickest step_evidence_v1
  - cumulative-vote-fusion-v2 `ct7_profiles_v1` and `cvf_v2/{em,core}.py` (hashed)
  - the temporary scratchpad `pool_z.npy` and `pool_names.json` (MISSING; `SCR` at l.115)
- `bank20_lsml_run.py`:
  - depth-feature-fusion-v1 (import, plus `results/step_level_bank_baseline_v1/STEP_SCORES.npz`)
  - readout-quickest step_evidence_v1
  - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`
  - `MAIN/results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz`
  - `MAIN/results/localization_source_group_audit_v1/FOLDS_V2.json`
- `bank20_lsml_report.py`, `declared_joint_report.py`, `ssl_s0_report.py`: worktree paths in report text only.
- `calfix_common.py`: readout-quickest step_evidence_v1.
- `calfix_evaluate.py`:22: the literal path `C:\Users\omris\TAU\hallucination_detection\.worktrees\depth-feature-fusion-v1`.
- `core_virtual_lsml_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; temporary scratchpad.
- `declared_joint_run.py`:
  - depth-feature-fusion-v1; readout-quickest step_evidence_v1
  - token-probability-fusion-v1 results
  - `MAIN/results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz`
  - temporary scratchpad
- `er_digit_share.py`: token-probability-fusion-v1; lsml-ct7-levers-run; readout-quickest step_evidence_v1 (l.16, 26).
- `er_generality_run.py`:
  - depth-feature-fusion-v1; token-probability-fusion-v1; readout-quickest step_evidence_v1
  - cumulative-vote-fusion-v2 `ct7_profiles_v1` and `cvf_v2/{em,core}.py`
  - its own SSL worktree's stage-B `STEP_SCORES.npz`, by absolute path
  - temporary scratchpad
- `er_prmscore_decomposition.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1.
- `er_stage_a.py`: cumulative-vote-fusion-v2/scripts/experiments (imports `cvf_v2`).
- `er_stage_b_run.py`, `er_stage_b2_run.py`: depth-feature-fusion-v1; token-probability-fusion-v1; readout-quickest
  step_evidence_v1; cumulative-vote-fusion-v2 `ct7_profiles_v1` and `cvf_v2/{em,core}.py`.
- `error_cluster_lsml_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; temporary scratchpad.
- `expectation_realization_run.py`:
  - depth-feature-fusion-v1 (plus a hash of `spectral_utils/prmbench.py`)
  - token-probability-fusion-v1 (plus `spectral_utils/derivative_step_channel_v1.py`)
  - readout-quickest step_evidence_v1
  - cumulative-vote-fusion-v2 `ct7_profiles_v1` and `cvf_v2/{em,core}.py`
  - lsml-ct7-levers-run `spectral_utils/chosen_token_calibration.py`
- `family_tail_calfix_run.py`: depth-feature-fusion-v1; temporary scratchpad (l.33);
  `MAIN/results/lsml_external_generalization_v1/evaluation/source/VALIDATION_FITS.json`.
- `family_tail_calfix_eval.py`: `MAIN/results/lsml_external_generalization_v1/evaluation/source`.
- `family_tail_transfer_lock.py`: temporary scratchpad (l.19); `.../evaluation/source/BUNDLE.json` (l.47).
- `family_tail_transfer_lock_v2.py`: temporary scratchpad.
- `indbank_lsml_run.py`:
  - depth-feature-fusion-v1; readout-quickest step_evidence_v1; token-probability-fusion-v1 results
  - `MAIN/results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz`
  - temporary scratchpad
- `lsml_merge_step_run.py`:
  - depth-feature-fusion-v1; token-probability-fusion-v1; readout-quickest step_evidence_v1
  - cumulative-vote-fusion-v2 `ct7_profiles_v1` and `cvf_v2/{em,core}.py`
  - temporary scratchpad
- `mtg_reproduction_extract.py`, `mtg_reproduction_score.py`: token-probability-fusion-v1 (sys.path import in the extractor).
- `named_group_fusion_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; temporary scratchpad.
- `partition_ceiling_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; token-probability-fusion-v1 results;
  `MAIN/results/localization_source_group_audit_v1/FOLDS_V2.json`.
- `partition_switch_run.py`, `per_dataset_fit_run.py`, `position_channel_run.py`, `position_prior_run.py`:
  - Direct reads:
    - depth-feature-fusion-v1 by sys.path import. The last three also run `git rev-parse HEAD` with cwd in that worktree
      (`per_dataset_fit_run.py`:47; `position_channel_run.py`:47; `position_prior_run.py`:48).
      `partition_switch_run.py`:42 runs it with cwd=ROOT.
    - readout-quickest step_evidence_v1.
  - Indirect reads, through `results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json`:
    - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz` and `results/localization_full_benchmark_v3/evaluation/JOINED.json`
    - cumulative-vote-fusion-v2 `ct7_profiles_v1/{profiles.npy, PROFILE_VALIDATION.json}`
    - readout-quickest `OOF_ANSWERS.csv` and `OOF_STEP_SCORES.npz`
    - SSL `POOL_STRUCTURE.csv`, `digit_family_extension_v1/FEATURES.npz`, and the `lsml_merge_step_v1` and `stage_b2` `STEP_SCORES.npz`
    - the temporary scratchpad `pool_z.npy` and `pool_names.json` (MISSING: the hash check hard-stops)
- `ssl_s0_audit.py`, `ssl_s0c_calibration.py`, `ssl_s0c_ct7_diagnostic.py`, `ssl_s1_run.py`: readout-quickest step_evidence_v1.
  `ssl_s0_audit.py` and `ssl_s1_run.py` also read `MAIN/results/localization_source_group_audit_v1/FOLDS_V2.json`.
- `ssl_s2_run.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`; readout-quickest step_evidence_v1 (through `R` from
  `spectral_utils.ssl_eval`); `MAIN/.../FOLDS_V2.json` (l.29).
- `ssl_s5_run.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz` (l.40); readout-quickest step_evidence_v1 (`profiles_full.npy`,
  `OOF_ANSWERS.csv`, `OOF_STEP_SCORES.npz` via `R`, l.43-45); `MAIN/.../FOLDS_V2.json` (l.47).
- `tail1_transfer_v3_run.py`, `tail_threshold_calibration_run.py`: temporary scratchpad.
- `tail_label_share_run.py`: readout-quickest step_evidence_v1; temporary scratchpad.
- `tail_lsml_banks_run.py`, `tail_weighted_fusion_run.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1; temporary scratchpad.
- `test_calfix_common.py`: depth-feature-fusion-v1 (sys.path).

### @ssl, other scripts

- `scripts/fit_algorithm_external_v1.py`: depth-feature-fusion-v1; readout-quickest step_evidence_v1 (l.12, 31).
- `scripts/score_algorithm_external_v1.py`: depth-feature-fusion-v1; `MAIN/results/lsml_external_generalization_v1/evaluation`;
  `MAIN/scratch/external_generalization_private/inputs` (l.20).
- `scripts/evaluate_algorithm_external_v1.py`: `MAIN/scratch/external_generalization_private/{inputs,sources}` (l.105, 164).
- `scripts/extract_external_banks_v4.py`: `MAIN/scratch/external_generalization_private`.
- `scripts/verify_external_banks_v4_source.py`:
  - readout-quickest `OOF_STEP_SCORES.npz`
  - cumulative-vote-fusion-v2 `ct7_profiles_v1`
  - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`
  - provenance strings pointing into scripts in those worktrees
- `scripts/fit_external_source_bundle.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`, `DERIVATIVE_CHANNELS.npz`,
  `chosen_token_calibration_v1/CT7_DEV_SCORES.npz`.
- `scripts/verify_external_source_strict.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz` and `DERIVATIVE_CHANNELS.npz`;
  depth-feature-fusion-v1 `results/step_level_bank_baseline_v1/RESULTS.json`.
- `scripts/audit_family_tail_handoff.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`; readout-quickest `OOF_STEP_SCORES.npz`
  and `profiles_full.npy`.
- `scripts/audit_prmbench_lsml_saved_scores.py`:
  - depth-feature-fusion-v1 results and source
  - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`
  - readout-quickest step_evidence_v1
  - consolidation-fusion-2026-09-22 `results/digitfree20_ladder_v1/RUN.json`
- `scripts/repair_runtime_fusion_reports.py`: the SSL worktree `results/ssl_pseudolabel_residual_v1/<stage>/run_20260923`.
- `scripts/review_digit_alternative_probability_v1.py`:16: lsml-ct7-levers-run `spectral_utils/fusion_signal_registry.py` (via `ROOT.parents[1]`).
- `scripts/run_digit_family_extension_v1.py`: token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`.
- `scripts/index_month_lsml_prm_results.py`: the whole `.worktrees` directory.
- `scripts/selector_deep_report.py`:46: `.worktrees/antigravity` (ABSENT).
- `scripts/localization/our_arm.py`: derives the main tree from the `.worktrees/<name>` layout.
- `scripts/window_answer_local_fusion_v2.py`: a usage example `--source-root .worktrees/lsml-ct7-levers-run`.
- `scripts/run_unified_causal_iu_v1.py`:11: a usage example `--data-root /Users/osegev/Desktop/hallucination_detection` (macOS).
- These read `scratch/external_generalization_private/` relative to their own checkout root, so that folder must be inside
  whichever checkout runs them:
  - `scripts/evaluate_family_external{,_v2,_v3}.py`
  - `scripts/run_family_external.py`
  - `scripts/run_external_local_cpu.py`
  - `scripts/verify_external_official_metrics.py`
  - `scripts/archive_external_evaluation.py`
  - `scripts/audit_external_backbone_diversity.py`

### @estimator (files that differ from @ssl)

- `scripts/experiments/er_stage_a.py`: cumulative-vote-fusion-v2/scripts/experiments.
- `scripts/experiments/tensor_mom_stage_a_run.py` (l.24, 33-35): depth-feature-fusion-v1; token-probability-fusion-v1
  `results/token_probability_fusion_v1`; cumulative-vote-fusion-v2 `ct7_profiles_v1`; readout-quickest step_evidence_v1.

### @decision-rule

- `scripts/experiments/answer_features_extract.py`: token-probability-fusion-v1 `TOKEN_MATRICES.npz`; readout-quickest step_evidence_v1.
- `scripts/experiments/answer_gate_run.py` (l.19-20):
  - depth-feature-fusion-v1
  - ssl-pseudolabel-residual-v1
  - readout-quickest step_evidence_v1
  - token-probability-fusion-v1 `DERIVATIVE_CHANNELS.npz`
  - cumulative-vote-fusion-v2 `ct7_profiles_v1/{profiles.npy, PROFILE_VALIDATION.json}`
- `scripts/experiments/decision_rule_run.py` (l.20-21): the same as `answer_gate_run.py`, plus hashes of cumulative-vote-fusion-v2
  `cvf_v2/{em,core}.py` (l.365-366).

### @self-labels

`MAIN` in `b16_fit.py`:28 and `b16_eval.py`:21 can be overridden through the environment variable `HD_MAIN_CHECKOUT`.

- `scripts/self_generated_labels/b16_fit.py`:
  - ssl-pseudolabel-residual-v1: sys.path `scripts/experiments` (l.31); `results/algorithm_decisions_v1/run_20260928/INPUT_MANIFEST.json`
    (l.120); `results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz` (l.162)
  - readout-quickest step_evidence_v1 (l.122)
  - `MAIN/dataset_cache/repgrid/...`: own-answer pickles (l.180) and `pb_qwen3_8b/processbench_gsm8k.pkl` (l.137)
- `scripts/self_generated_labels/b16_eval.py`: ssl-pseudolabel-residual-v1 (`per_dataset_fit_v1` `STEP_SCORES.npz`); readout-quickest
  `OOF_ANSWERS.csv`.
- `scripts/self_generated_labels/extract_own_answers.py`: `MAIN/dataset_cache/repgrid/<cell>/{manifest.json, raw_*.pkl}`.

### @ct7-levers (files that differ from or are absent at @ssl)

The agent check found 56 path-bearing scripts here that differ from @ssl or are absent there. Only the notable ones are listed.

- `scripts/experiments/ct7_token_tail_lsml_calfix_v1.py`:32: `C:\Users\omris\TAU\hallucination_detection\.worktrees\ssl-pseudolabel-residual-v1\scripts\experiments` (sys.path).
- `scripts/experiments/pb_tail_weights_v1.py` (l.38, 40, 42): the temporary scratchpad; SSL
  `results/tail_threshold_calibration_v1/TRANSFER_LOCK_V2.json` and `results/tail1_transfer_v3/run_20260924_2259`.
- `scripts/diagnostics/{evidence_domain_fusion_probe_v1, evidence_domain_independence_v1, position_length_vs_ct7_v1}.py`:
  `.worktrees\fusion-independence-atlas-v1` (ABSENT).
- 15 scripts carry `/Users/osegev/...` macOS paths (`git grep -l osegev lsml-ct7-levers-run -- scripts`):
  - `scripts/joint_lsml_localization/{evaluate_existing_v1, evaluate_existing_v1_r1, evaluate_existing_v1_r2, evaluate_processbench_amendment_v1, finalize_existing_v1, run_existing_v1}.py`
  - `scripts/og_sml_agent_b/run_t0.py`
  - `scripts/reasoning_localization/{integrate_h3_reliability_fusion, register_h3_historical_headtohead, register_h3_llama_transfer, register_h3_prmbench_diagnostic, register_step_cut_premise_audit, run_phase1_baseline}.py`
  - `scripts/run_fusion_independence_atlas_v1.py`
  - `scripts/run_unified_causal_iu_v1.py`

  Examples: `finalize_existing_v1.py`:22 (.cursor canvases); `og_sml_agent_b/run_t0.py`:23 (`local_cache/worktrees/structured_fusion_c_v2`);
  `register_h3_historical_headtohead.py` (`.worktrees/reconstruction-science-run-v1`, ABSENT here too).

### @whitebox

- `scripts/run_whitebox_kill_test_1a.py` (l.45-47), `scripts/run_whitebox_kill_test_1a_followup.py` (l.39-41),
  `scripts/run_whitebox_kill_test_1b.py` (l.86-89): `ROOT.parent/token-probability-fusion-v1/results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz`.
  This is a sibling-worktree path, so it only works when run from inside `.worktrees/`.

### @codex-external and @readout

- `@codex-external`: no path-bearing script blob differs from @ssl. The branch is an ancestor of @ssl.
- `@readout`: the agent check found 55 path-bearing scripts that differ from @ssl or are absent there. Notable:
  - the three atlas diagnostics listed under @ct7-levers;
  - `run_fusion_independence_atlas_v1.py`:37 and `run_claude_feature_bank_v1.py`, which also read the absent
    fusion-independence-atlas-v1;
  - `run_rbm_completion_queue.py`:120, which reads the absent `.worktrees/dufs-moment-selection-v1`.

---

## Unresolved verification gaps

1. **Three uploads failed** in pass 2 because of the Drive 403 per-minute quota on rclone's shared client_id. A retry is running.
   These are not on Drive; keep them locally: `fusion_multiwidth_dense_v1` (27,546 files), the off-git files of `er-generality-v1`
   (37) and of `lsml-ct7-levers-run` (43).
2. **Pool files** are still missing from the temporary scratchpad. Every runner that reads them hard-stops until they are restored
   from `inputs_backup/` or the Codex archive.
3. **Five orphan files** (pass-1 "Not yet uploaded", `RECONCILIATION_2026-10-01.md`@rescue:18; copies in
   `results/consolidation_review_20261001/preserved_orphans/`). Their Drive status is not in the pass-2 facts used here; not confirmed.
4. **`pb_llama31_8b`** pickles (LFS, main checkout) have no entry in `bplf_coverage.json` or `scratch_coverage.json`. Their Drive
   coverage is unverified.
5. **`dataset_cache/four_localization/*`** (including `prmbench_prm.pkl`) relies on the 2026-08-20 Drive record. It was not
   re-verified in this pass and is not in the pass-2 main-checkout tar.
6. **Restore test** for the LFS-free publication (decision 9) has not been done.
7. **historical24 caches**: location not verified.
8. **Entry 2**: no primary record of Omri's 2026-09-27 request was located. It is attributed only in the 2026-10-01 records.
9. **Entry 4**: the source words the 17/32 step-0 count two ways ("always" vs "about 95% of answers").
10. **Entry 5**: the "never run" status of the answer-local vs pooled comparison dates from the 2026-09-18 block. No later source
    re-confirms it.
11. **Entry 11**: the execution gates and the overlap audit were written for Hard2Verify/Socratic. Their application to MedPRMBench
    is inferred, and no MedPRMBench overlap plan exists.
12. **Entry 17**: the error-type breakdown artifact page was not verified, and its builder scripts are in a temporary scratchpad.
13. **Entry 19**: the readout line's reopening conditions exist only in an untracked proposal file in the rescue worktree.
14. **Entry 20**: sparse-scoring recall for short errors remains unmeasured (fitting support is measured).
15. **Entry 21**: there is no verbatim Omri quote for the MATH-regrade deferral (agent records only). The advisor deck artifact was not
    opened here.
16. **Appendix**: the counts of 56 (`@ct7-levers`) and 55 (`@readout`) path-bearing scripts come from one verification agent's
    git grep. They are not listed one by one here.
17. **`localization_full_sampling_v3` size**: the pass-1 log quoted 6.9 GB (`RECONCILIATION_2026-10-01.md`@rescue:14). The verified
    pass-2 tar is 20,165,529,600 bytes for 28,008 members. The difference was not investigated; the member count and MD5 match.
18. **The pass-2 section** of `RECONCILIATION_2026-10-01.md` is not yet committed. The backup facts here come from the manifests,
    not from a committed reconciliation.
19. **No consolidated `CLAUDE.md`** exists yet. The rules cited as belonging there (entries 5, 6, 18) have no home until the merge.
