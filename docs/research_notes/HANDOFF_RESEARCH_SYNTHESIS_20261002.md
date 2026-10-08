# Handoff: consolidate research understanding before designing the next algorithm

Prepared 2026-10-02 for Omri. This is a source-based planning handoff, not authorization to launch experiments. The existing [Hebrew scope](RESEARCH_SYNTHESIS_SCOPE_20261002_HE.md) supplies the scientific detail. No new agent was started by preparing this file.

## Objective and success criteria

Turn the evolving CT7 / bank11 / B13-B16 / family15 / estimator / clustering work into a clear account of complete algorithms, matched experiments, established findings and unresolved questions. Propose a small, justified next work package. Subsequently, evaluate selected complete recipes against matched averaging controls across feature banks, then analyze those same recipes against supervised PRM and Mind the Gap. Historical results must retain their original recipe identity.

Assumption: existing reports, code and frozen scores support the initial inventory without new inference, fits or scoring. Success: each proposed next action has an evidence gap, source, status, dependencies and an explicit comparison that resolves it. Failure: relabeling a feature bank as an algorithm, importing CT7 performance into a new recipe, losing a deferred option, or treating a proposal as Omri's decision.

## Two stages; who can begin when

**Begin now: read-only research inventory and documentation.** Read pinned sources across branches, complete recipe cards, reconcile decisions and draft the experiment matrix. Save only owned documentation. Claude continues preservation and Git consolidation. Do not switch the shared checkout or change another worktree. Limited disk does not require a second full checkout; use `git show <commit>:<path>` and small source snapshots. A separate doc ref can preserve work without changing the active branch.

**After a validated consolidation commit: research implementation.** Record that commit as `BASE_COMMIT`, verify required code/data dependencies, and start any authorized implementation from it. The branch name `master` is not proof of consolidation. Research can branch from a verified integration commit before a later promotion to master; any publication/merge-to-master permission remains as already agreed. Do not invent a completion time for an upload affected by Drive rate limits.

Before proceeding to implementation, require these concrete outputs from consolidation:

1. Final backup reconciliation with verified / failed / pending / previously covered entries, restore evidence and local changes since archived snapshots accounted for. Any remaining gap must be explicit. Retain originals under decisions 13 and 15.
2. A committed research handoff, substantive historical index and line statuses reconciled with decisions 14-21. Ownerless-line proposals remain proposals. Include the whole `codex/advisor-prep-2026-10-05` branch, not just its initial commit.
3. A completed cumulative merge from the current source tips, content-level preservation of HISTORY / PROGRESS / LESSONS and research directions, resolved conflicts, appropriate tests, and the planned headline replay for each line. History numbering alone is insufficient; Step 353 was reported never written, not necessarily lost.
4. An explicit merged commit and a dependency/readiness statement for the research package. Preserve frozen benchmark contracts and per-model/per-dataset fitting decisions. Do not silently execute scripts with stale absolute worktree paths.

These are merge/readiness checks, not a requirement to run the 21 research ideas. Retention/cleanup has its own stricter checks and is not necessary for research to begin.

## Latest observed operational state (dated, not a live dashboard)

Checked 2026-10-02 at 22:29 Israel time. [Evidence and source hashes](research_synthesis_20261002/HANDOFF_EVIDENCE_20261002_EVENING.json):

- Completed run-2 manifest: 25 rows, 22 `OK`, 3 `FAILED`; scratch-private archive now passes MD5 and 6,646/6,646 member count. The run correctly reports INCOMPLETE. These are script verification results, not an independent restore of every archive in this review.
- Failures: `fusion_multiwidth_dense_v1`, `worktree_offgit_er-generality-v1`, `worktree_offgit_lsml-ct7-levers-run`. Recent logs contain Drive 403 rate-limit failures. Retry started at 22:28:38, carrying successes and retrying failures sequentially with longer backoff. One rclone uploader was observed sending `fusion_multiwidth_dense_v1`. The new live manifest initially has only carried/finished rows; use the preserved run-2 manifest for the completed 22/25 count.
- The two previously missing `backup-pre-lfs-fix` LFS objects now have `rclone check` success in the manifest. The earlier 61 objects have separate existing-backup coverage evidence.
- No local `claude/consolidation-2026-10-01` ref exists. Local `master` is still `c3b018bd1124f8d59e9833a9b84652e6724be6e3` and is not the consolidated base.
- Rescue decision record is committed at `6173c7354209009b42492149a054312a49773c82`. Its backup reconciliation still says pass-2 results are to be filled from the manifest.
- The 21-entry scratch handoff was last modified at 00:16 and still contains decisions superseded later that day. No historical index is yet in rescue's `docs/consolidation/` directory. Do not equate scratch drafts or agent replies with committed final deliverables.

Recheck these observations at entry. Do not wait for uploads to finish before preparing the inventory or Monday material.

## Read in this order

1. The applicable `AGENTS.md`, all of `CLAUDE.md`, current `PROGRESS.md`, the agent operating contract and LESSONS standing rules. Inspect source refs and the actual checkout before assuming it contains the newest rules.
2. [Accepted decisions, snapshot from rescue at 6173c7354](research_synthesis_20261002/handoff_sources/DECISIONS_2026-10-01.md). Later explicit user decisions override older prose. This decision list and the 21-topic research checklist are different lists.
3. [Research synthesis scope in Hebrew](RESEARCH_SYNTHESIS_SCOPE_20261002_HE.md), its [12 pinned research sources](research_synthesis_20261002/SOURCES.json), and the [afternoon content-level consolidation review](../meetings/advisor_prep_2026-10-05/CONSOLIDATION_REVIEW_20261002_AFTERNOON.md).
4. [Snapshot of the 21-topic draft](research_synthesis_20261002/handoff_sources/HANDOFF_CHECKLIST_DRAFT_20261002.md). It is a source index, NOT an approved current plan. Verify its references and apply the overrides below. Read its underlying source files, not just headings.
5. Newest SSL `CLAUDE.md` / `Research_Directions.md` / `PROGRESS.md` at `553de79c61ab1f527534727a5fbe93512bde4191`, estimator branch `aa8bf95d5`, other source refs in the draft, and any newer committed consolidated handoff. Expand abbreviated refs to full hashes in your output.
6. [Monday preparation plan](../meetings/advisor_prep_2026-10-05/PLAN_HE.md), [narrative](../meetings/advisor_prep_2026-10-05/PREPARATION_HE.html) and [documentation integration instructions](../meetings/advisor_prep_2026-10-05/MERGE_HANDOFF.md).

## Required initial deliverables

Create a compact research package, with a Hebrew overview for Omri and machine-readable ledgers where useful:

- **Recipe registry.** For every relevant historical recipe record inputs/model/teacher-forcing vs own-generation, exact feature names, token/window/step unit, signs/normalization, filtering, grouping/clustering, estimator inside/between groups, fusion/readout order, position term, decision rule/gate, fit/calibration scope, folds/population/metrics, source commit, entry point/config and score hashes. CT7 is a recipe; bank11 and B13 are banks; family15 is an organization of features. Do not invent a separate Family13 definition.
- **Experiment and analysis ledger.** Map historical experiments and negative results to those recipe IDs, matched controls, datasets, metric, sample coverage, fit access, uncertainty, source and audit status. Identify superseded label/fold contracts and analyses that cannot be pooled. Reuse the existing eight-bank experiment; do not automatically repeat its 331 arms.
- **Question-to-evidence-to-gap table.** Cover learned weights vs equal weights; filtering vs weighting; clustering/group normalization; estimator accuracy vs ranking utility; conditional dependence given true labels; feature composition/size/digits; position/length/readout; decision calibration; teacher forcing/own-generation; PRM and Mind the Gap performance and algorithmic differences. A diagnosis is not automatically a new method.
- **Direction coverage ledger.** Map every item below and each substantive source follow-up to a disposition, decision provenance, dependencies, outstanding question and proposed resolution. Unverified content remains marked unverified. Count of headings is not an acceptance test.
- **Proposed experiment package.** Recommend a small set of complete candidates/controls and a bounded sequence justified by the gaps. Preserve the documented runner-up as the starting point for the SML/MoM direction; do not silently replace it with an unrelated method. Define what each comparison isolates, success criteria and allowed regressions before any execution. Report proposed selection, not a new performance claim.
- **Post-merge portability check.** Map old entry points/data paths to the verified base and restore manifests, using original hashes. Record blockers instead of copying whole worktrees or downloading large caches without inspection.
- **Monday bridge.** State what is already defensible for 5 October, what remains exploratory, and what the next protocol would establish. Presentation work continues independently of the merge/proxy. Do not require a newly invented algorithm to be finished before Monday.

## How this incorporates all 21 collected directions

The synthesis organizes these directions; it does not supersede them. Source numbers below refer to the 21-topic draft. Status overrides refer to the accepted decision record; unapproved options stay options.

| Topic | Role in the unified work | Status / next question to preserve |
|---|---|---|
| 1. Final fusion and clustering | Core algorithm construction | Current reported result is DS filter + equal average. Omri wants SML/MoM-based combination; documented starting point is filter + label-free partition/merge + HEM within / DS between. Separate grouping benefit from weighting benefit. |
| 2. Level-family removal | Core component question | Removing 2-3 channels remains open, not already tested; do not assume it must leave two level groups. Preserve alternative decompositions. |
| 3. Tensor MoM | Estimator question | Separate prevalence, sensitivities/specificities, filtering and weighting; better prevalence estimation is not proof of better detection. Preserve existing implementation evidence and remaining comparisons. |
| 4. Position | Component and dataset analysis | Preserve all source options and the pending choice, rather than narrowing to three. Distinguish channel, prior and slope; source/target and step-zero artefacts matter. |
| 5. Fitting access | Shared evaluation contract | Per-model/per-dataset current decisions and historical source-fitted or answer-local panels must be explicit. Do not mix their reported scores. |
| 6. Digits and Joint | Feature ablation plus historical boundary | Digits remain in the fusion line. Joint has no new arms; digit gates/anchors/sign schemes are not reopened. Bank size and bank composition are separate factors. |
| 7. Answer gate and decision rule | Output/metric layer | Preserve all three documented options and the official-F1 vs PRMScore tradeoff. Ranking, calibration and answer gate are separate comparisons, with method choice still explicit. |
| 8. Whitebox answer gate | Indexed evidence | Out of scope now, not closed and not a next experiment (decision 14). Preserve rescues and reverse failures. |
| 9. Whitebox length follow-up | Indexed evidence | The separate whitebox follow-up is out of scope now, not closed. Existing length analyses may inform interpretation; this is not permission to reopen the line. |
| 10. External reevaluation | Later evaluation layer | Hard2Verify/Socratic are exposed; any new use is exploratory and tied to a frozen recipe and stated fit/calibration contract. Old CT7/bank11 tables are not new-candidate results. |
| 11. Untouched confirmation | Future confirmation | MedPRMBench remains deferred under Omri's decision; retain as confirmation direction after method lock and access/budget decisions. Do not treat prior approval of a proxy as approval to use it now. |
| 12. Transfer proxy / bank11 diagnosis | Bounded validation plus deferred question | One exploratory proxy approved, protocol/success written first, no sweep and no block on SML/MoM. Why bank11 transfers is separately deferred, not closed (17-18). |
| 13. Family-tail audits / library contract | Verify existing claims | V2/V3 audits approved (16); not certified complete by this handoff. Preserve z-scored-input contract and frozen modules. Do not report unaudited variants as audited because V1 passed. |
| 14. Published comparators | Common-candidate comparison | Preserve published/comparable baselines and protocol gaps. Mind the Gap requires paper/code/local-implementation identity plus matched evaluation and algorithmic difference. Do not claim a PRMBench error-category analysis for it without the actual artifact. |
| 15. Another model | Diversity question | Preserve observed cross-backbone diversity and pending utility questions. Different model/telemetry budgets must be stated; no automatic new inference or second-model fusion. |
| 16. Self-generated labels | Deployment/teacher-forcing evidence | Preserve three options: close with caveats, collect more own answers, paired comparison on the same existing answers. No option selected here. A nonsignificant difference is not equivalence. |
| 17. Supervised PRM by error type | Finding and future combination, separately | Report measured categories and overlap without universal structural-error claims. Combination is deferred, not closed, and need not use an error-type router; score combination is also an idea (18). Oracle union is not a deployable method. |
| 18. PRMScore reporting | Matched reporting controls | Preserve top-k-steps-per-answer baseline; specify threshold/flag budget and compare the same candidates/population, alongside within-answer ranking metrics. |
| 19. Saturated implementations | Negative-result/reopening ledger | Record each failed implementation and the evidence or new idea that would reopen it. Do not close whole research families from one negative experiment or approve ownerless proposals by default. |
| 20. Historical transfer/supporting backlog | Substantive historical index | Preserve historical24, reconstruction/geometry/DEEM and BOCPD/HMM/IMM/LOCA/flows/Kalman topics where present. Separate already-used BOCPD/readout components from broader unimplemented proposals; map dependencies and unfinished steps individually. |
| 21. Technical debt / advisor material | Reproducibility and communication | MATH grader, cross-worktree absolute paths, LFS restore/budget, rclone client ID and library constraints stay tracked. Monday preparation is active in parallel (19), not paused until merge/proxy. |

For comparator analysis, apply the surviving *same complete recipe* to PRMScore, within-answer ranking, measured error categories, overlap and failure patterns. First establish algorithm identity and matched controls; then describe any contribution. Do not stitch the best CT7 category result, bank11 external result and modern estimator result into one fictitious algorithm.

## Entry prompt for the new agent

> Read this handoff and the sources in its read order. Your first assignment is the research inventory and proposed work package, using existing code/reports/results across pinned refs. Recheck consolidation readiness. Before the merged base is validated, work only on owned documentation and source-pinned analysis. Produce recipe, experiment, question/gap and direction ledgers, then recommend a bounded next protocol grounded in the existing runner-up and controls. Preserve all 21 directions and their decision status, including deferred alternatives. Do not start experiments, switch the shared checkout, interfere with Claude's consolidation, or turn an unverified claim into a result. If a merged base is ready, record its exact commit; existing approvals remain valid but execution must follow their stated sequence and scope. Keep Monday preparation progressing with defensible existing findings.
