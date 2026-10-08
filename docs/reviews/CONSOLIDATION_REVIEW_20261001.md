# Consolidation review for 1 October 2026

The proposed merge covers the recent main research lines, but it does not yet justify an assurance that all prior work and all next steps are preserved. Proceed with an additive integration only after incorporating the inventory and handoff requirements below. Do not remove worktrees or local data on the strength of the quoted readiness message.

This is a review, not merge approval or a new research selection. No branches were merged, pushed, reset, removed or rewritten; no experiments were launched. Existing research documents were not edited. Five orphan files were copied into the audit directory without changing their originals. LESSONS receives the required audit/session lessons.

## Scope and evidence

Reviewed the exact Claude consolidation session `ae2dd164-1ddb-4b21-9e3e-cb18f86482aa`, including the original user decisions, and the related SSL conversation `2d14a8c9-8b3f-489b-b26f-812b4b84a8b3`. Cross-checked canonical guidance, the four line-status files, collection and research handoffs, negative-result records, whitebox stage reports, archive records and the actual upload script. Refreshed origin references and inventoried all 156 local/origin refs, all 17 registered worktrees and three orphan folders. This review does not independently repeat the previous session's claimed inspection of every Claude/Codex conversation, and cannot certify unexported claude.ai/web conversations or the vanished `codex/combined-fusion-v1` worktree.

Machine-readable evidence and the repeatable inventory script are in [results/consolidation_review_20261001](../../results/consolidation_review_20261001/). `INVENTORY.json` pins all 13 proposed source tips and lists uncovered refs, omitted paths and ignored result files. `ORPHAN_HISTORY_CHECK.json` records preserved-file hashes and the additional history-wide check. Counts describe the observed checkout, not all possible copies on Drive or other machines.

## Findings requiring an amendment

### 1 Older research is absent from the proposed merge sources

141 of 156 refs are ancestors of at least one of the 13 proposed sources. The other 15 include duplicate local/remote refs, backups and the old main skeleton; they are not 15 independent missing research lines. However, eight historical research/report refs below contain **1,052 distinct paths absent from the union of all 13 source-tip trees**. Those files are still recoverable from GitHub; they would be omitted from the proposed consolidated checkout. Ancestry coverage of September branches did not establish coverage of all prior work.

| Source ref | Paths absent from all proposed source tips | Disposition needed |
|---|---:|---|
| `origin/codex/reconstruction-science-results-v1` | 345 | Preserve compact application-science releases and certified tables |
| `origin/codex/reconstruction-benchmark-v1` | 301 | Preserve benchmark evidence and remote-worktree handoff |
| `claude/advisor-letter-aug27` | 297 | Preserve advisor narrative and evidence references |
| `origin/codex/graph-geometry-selection-v1` | 265 | Preserve graph studies and benchmark/method registries |
| `origin/codex/iu-graph-smoothing-ablation-v1` | 117 | Preserve graph-order experiment, frozen scores and plots |
| `origin/codex/deem-b3-moe-gating-v1` | 41 | Preserve routing challengers, configurations and ceiling diagnosis |
| `selector/a4-antigravity-unsupervised` | 4 | Preserve the four subset-sweep manifests and review differing blobs |
| `origin/codex/og-sml-agent-b-v1` | 1 | Preserve `HANDOFF_JOINT_LSML_REVIEW_TO_CODEX_2026_09_04.md` as historical review |

Row counts overlap; do not add them. The evidence JSON also lists changed blob paths, not just missing filenames. Give every uncovered ref a pinned disposition: merge it, preserve its distinct artifacts with provenance, or explicitly archive it with a discoverable index and restore instructions. An archive disposition must not silently become an active research recommendation. Also classify the two local backup refs and `origin/codex/consolidate-research-2026-08-19-lfs-backup`; retaining their Git pointers alone does not verify their LFS data.

### 2 The October 1 upload misses ignored main-checkout results

The upload's loose-file list contains 3,831 paths, plus eight whole result folders streamed as tar archives. Comparing those lists with Git's ignored-file inventory finds **36,143 ignored files, 1,541,252,083 bytes**, outside both sets. This is an omission from this upload, not proof that every file lacks an older archive.

Concrete examples:

- `results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz`: **447,292,975 bytes**, absent from both upload lists. The CT7 status file explicitly says the matrix was not archived; only its manifest was in the older archive.
- `results/lsml_group_confidence_v1/`: three ignored files totaling 22,786,417 bytes are omitted despite the code/reports being rescued.
- Ignored results under `historical_joint_refit_v3`, `historical_fusion_refit_v3`, `fusion_gate_calibration_v1` and other early localization runs are omitted.
- Family-tail V1/V2/V3 files are also in this exclusion list. Some have earlier restore records; reconcile them against those records instead of assuming either loss or redundant backup.

Use [OMITTED_MAIN_IGNORED_FILES.txt](../../results/consolidation_review_20261001/OMITTED_MAIN_IGNORED_FILES.txt) as a reconciliation list, not an automatic upload instruction. Every retained artifact needs either verified archive coverage or an explicit retain-local decision. A file being regenerable is not equivalent to preserving the frozen file and its hash.

### 3 Other worktrees and orphan folders still need preservation

Only the SSL worktree has a dedicated `docs/archives` record in the inspected worktrees. Other recorded Drive archives may exist elsewhere; their exact coverage remains to be reconciled. The current October 1 script does not upload their ignored files. Examples include:

| Worktree | Ignored result bytes observed | Examples or implications |
|---|---:|---|
| readout-quickest-detection-v1 | 1,564,746,818 | Frozen readout/source arrays |
| whitebox-layer-views-v1 | 814,609,017 | Answer-level and step-level reductions |
| token-probability-fusion-v1 | 536,534,795 | Includes `TOKEN_MATRICES.npz` and `DERIVATIVE_CHANNELS.npz`, inputs to newer lines |
| cumulative-vote-fusion-v2 | 404,367,410 | Reused CT7/source artifacts |
| er-generality-v1 | 144,813,712 | Compare against SSL archive copies |
| lsml-ct7-levers-run | 117,464,979 | Tail-weight scores and bootstrap/decision outputs |
| self-generated-step-labels-v1 | 20,737,519 | Private own answers and B16 features |
| decision-rule-v1 | 6,494,715 | Answer features and frozen decisions |

These sizes can overlap equivalent artifacts elsewhere. They quantify the inspection target, not an upload total. Depth's two ignored score files were explicitly rescued; verify their blobs before treating that worktree as uncovered.

The orphan folders contain five files totaling 1,035,310 bytes: four literature-direction audit files and one RBM setup-recovery runner. None of their raw or CRLF-normalized Git blob hashes occurs anywhere in the history reachable from the 13 planned sources. This proves a gap in that history, not absence from every disk or archive. Copies and SHA256 hashes are preserved under `results/consolidation_review_20261001/preserved_orphans/`. Review and archive these exact originals; do not replace the maintained RBM runner with the old recovery version. The third orphan folder, `binary-moment-fusion-v1`, has no files.

The excluded `scratch/`, `cache/` and `dataset_cache/` must remain intact until their research dependencies are individually mapped. In particular, `scratch/external_generalization_private/` is not a disposable browser profile, and self-generated labels depend on the repgrid raw pickles. Existing external archive records may cover these, but that must be verified per path. The explicitly excluded 19.7 GB backup-pre-lfs-fix payload remains an unresolved retention/archive decision.

### 4 Backup completion and restore verification are different checks

At the read-only Drive listing at 14:24 Israel time on October 1:

- The Git bundle existed, size **7,428,475 bytes**, MD5 `b8a5ab81ee6a0b687022366505ed331c`, matching the uploader's local fingerprint.
- The upload log recorded `lfs objects CHECK OK` for its 12 selected objects.
- Exactly **one of eight** result tar archives had a completion/hash-success entry: `lsml_external_generalization_v1.tar`, size **110,981,120 bytes**, MD5 `f020ef59ee5cd230f061275cc93cc959`. The sampling tar was still being streamed; there was no completed loose-results check.
- The SSL archive record separately reports 367 matching files, 3,969,226,611 bytes, zero differences. This review read that manifest/report; it did not independently download and rehash all SSL data.

Two attempts to download the small bundle for `git bundle verify` failed with Drive HTTP 403 `rateLimitExceeded` at 14:30 and 14:32. Thus remote existence and MD5 were checked, but bundle restore validity was **not** established in this review. This was a Drive API quota error, not an approval rejection.

The upload script uses `set -u`, without `pipefail` or fail-fast handling. Its streamed bundle rewrite removed `git bundle verify`. Matching an uploaded stream's MD5 does not prove that the producer completed successfully or included every intended member. Before treating the archive as complete, require producer exit-status checks, a validated bundle with its prerequisite commits, archive member/count reconciliation, and one final manifest of successful objects. The bundle uses `--not --remotes`, so it depends on existing prerequisite Git history; record restore prerequisites. Its 12 LFS IDs were derived from readout-quickest alone: reconcile the required LFS OID union for every intended restored ref before claiming complete LFS coverage.

Skip-LFS publication can be a documented storage arrangement after verification. It does not itself preserve missing data, and it does not require erasing old history or converting old frozen objects. Keep source branches and worktrees until both restore and dependency checks pass.

### 5 Status and history checks need more than block counts

There are **four** dedicated line-status files across the 13 proposed source tips: SSL, decision-rule, self-generated labels and CT7 levers. Other lines have substantial reports and handoffs, but the statement that every line already has its promised status file is too strong. The merge session proposed statuses for ownerless branches; proposals are not established closure decisions.

Reconcile complete HISTORY blocks by source ref, tagged heading and normalized body hash; equal counts can conceal a missing block replaced by a duplicate. Preserve colliding Step numbers with tags. Do the same content-level accounting for PROGRESS, LESSONS, Research_Directions, handoffs and the paper index. Maintain one current decision summary above historical instructions, so old Joint, digit and estimator next-action text is not accidentally revived.

The quoted three code conflicts came from earlier pairwise trials. Repeat conflict assessment against pinned current tips and the cumulative merge result. Inspect `.gitattributes`, `.gitignore` and `AGENTS.md`/`CLAUDE.md` as functional inputs, not merely prose. Preserve the frozen external modules' byte identities; the two dirty `_bank11` files on decision-rule and self-generated-label worktrees currently have no non-EOL diff.

## What is already preserved correctly

All **268** current loose files checked under `spectral_utils/`, `scripts/`, `tests/`, `docs/` and `.claude/agents` match content in a proposed source tip, allowing only CRLF/LF normalization. The main checkout's HISTORY, PROGRESS and LESSONS matched the rescue branch before this review appended its required lesson. The bank11-origin rescue and the SSL archive were worthwhile, concrete preservation steps. The major recent branches, including the estimator run, are in the proposed source set. No evidence here establishes that the planned additive merge has already destroyed work; the principal issue is incomplete coverage and premature readiness wording.

## Research handoff checklist

Keep these entries explicit in the consolidated handoff. Statuses below distinguish user decisions, completed experiments and proposals. Preserve the original source links and select the next experiment only after Omri reviews the combined options.

| Direction | Current interpretation and next step | Source |
|---|---|---|
| Final fusion and clustering | Averaging remains a reported control/current result, not the intended final fusion. Develop SML/MoM-based combination from the documented runner-up: DS filter, L-SML partition plus absorption merge, **HEM within groups, DS between groups**. Its slightly higher mean does not erase losses on two banks or satisfy the frozen no-bank-regression rule. | SSL `CLAUDE.md` October 1 update; `Research_Directions.md` October 1 decisions; `results/algorithm_decisions_v1/SUMMARY.md` |
| Level-family removal | Keep the requested removal of 2–3 level channels as an unrun experiment. Recheck the partition and matched controls; do not assume reducing five channels to three produces two groups. | Same October 1 decision |
| Tensor MoM | Step 464 is complete and negative **for this implementation**, not a closure of all MoM/SML research. Reopening conditions: remove vote dependence or replace uniform per-answer quantile marking. The fixed 20% threshold's self-consistency idea was deprioritized, not silently substituted for those questions. | Estimator `results/tensor_mom_v1/NEGATIVE_RESULT.md`; latest PROGRESS |
| Position | Preserve all three distinct proposals: filter-controlled position prior; cross-fitted position weight without reweighting content; remove positional telemetry artifacts before fitting a prior. Include step-index, same-length swap and equal-share controls. | SSL current Research_Directions and Step 462 |
| Fitting access | The newest label-free line fits separately per model/dataset on that dataset's unlabeled answers. Keep frozen source-fitted bank11 and answer-local studies as separately labelled contracts. Do not mix their metrics or silently apply the new contract to historical results. | SSL September 30 Research_Directions; external handoff |
| Digits and Joint | The three digit channels stay in the specified label-free banks; this does not reopen all digit gates/anchors. Joint is discontinued for new arms. Preserve its historical studies and reviewer concerns as records. | September 30 and October 1 canonical decisions |
| Answer gate and decision rule | Retain the actual choice: frozen no-gate q80, raw-level count rule, or pure entropy answer gate. Separate wrong-answer detection, number of flags and localization. Freeze a chosen rule before any new evaluation; previously exposed Hard2/Socratic results cannot become untouched evidence. | Decision-rule status file; Steps 459 and 460 |
| Whitebox answer gate | Locator stage 1b is closed; gate stage remains open. Before a candidate, establish label-free orientation. Judge gate fusion by exact localizations gained/lost at matched opened fraction, not AUROC alone. | Whitebox `RESULTS_STAGE_1A.md` sections 5–6 and `RESULTS_STAGE_1B.md` section 7 |
| Length confounding | Preserve the whitebox stage-1b request for a separate step-length/readout follow-up. Do not interpret crude rank residualization as a causal estimate. | Whitebox `RESULTS_STAGE_1B.md` section 5 |
| External reevaluation | Explicitly retain Hard2Verify/Socratic reevaluation under the per-dataset contract with position controls. This is exploratory, distinct from untouched confirmation. | SSL next directions item 4 |
| Untouched confirmation | MedPRMBench remains a candidate after method/decision lock and the applicable AIRCC preflight/budget gate. No current benchmark already examined supplies untouched confirmation. | SSL next directions; external specification |
| Source-side transfer proxy | Missing from the short merge handoff outline. Preserve source splits mimicking transfer: leave-benchmark/cell/category out; retrospective validation against already known external ranking must be labelled exposed. Preserve the diagnostic of why bank11 transfers and partition/weight stability. | `HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md` section 3 items 1–2 |
| Family-tail audits and library contract | Independent V2/V3 audits remain separate from V1 audits and exact replay checks. Preserve z-score/loading-scale guards and version frozen modules rather than editing them in place. | Same handoff section 3 items 3 and 5 |
| External published comparators | Published critic/PRM reproduction remains unfinished. Retain explicit asset/access/inference requirements; no unsupported SOTA claim. | `LSML_EXTERNAL_GENERALIZATION_RESULTS_20260924.md`; Research_Directions September 24 |
| New information from another model | Existing Qwen3/QwQ telemetry provides a complementary-view hypothesis, not demonstrated fusion benefit. Preserve a bounded source-side cross-backbone/residual option; identical teacher-forcing reruns did not provide observed diversity. | External results and `EXTERNAL_TEACHER_FORCING_REPEATABILITY_20260924.md` |
| Self-generated labels | Preserve **all three** options: close with the underpowered-cell caveat; collect sampled own answers; or score the same existing answers using both models. The third option was omitted from the quoted final plan. | Self-generated-label status file, options 1–3 |
| Supervised PRM by error type | Keep as an open proposal with matched access and error-type controls. Complementary hit sets do not establish a working label-free router. | CT7 status file; Step 437 measurement/error-type analysis |
| PRMScore reporting | Retain fixed-count top-k baselines and the error-count construction analysis; the choice of k and the top-2 finding were post hoc. Do not present top-k as requiring no ranking method. Carry within-AUC, official PRMScore, first-error and no-error metrics separately. | Decision-rule status file and answer_gate_v1 |
| Saturated implementations | CT7 weighting, tested family/tail recipes, first-error product readout, partition switch, readout experiments and the tested group-confidence implementation need individual evidence and reopening conditions. A negative implementation does not close an entire estimator family. | Four status files, family-tail handoff, tensor negative result, group-confidence report |
| Historical transfer and supporting backlog | Explicitly disposition historical24 transfer, sparse/short-error sampling and named temporal/supporting methods (BOCPD/HMM/IMM, LOCA/Flows/KalmanNet, task-based sampling). Keep historical/RAG/general-task lines indexed as deferred where appropriate. Their old open text must be resolved as completed, superseded or deferred, not dropped by an unqualified saturated label. Joint continuation proposals are superseded by the new stop decision. | Canonical research mandate, rescued early localization records, historical refs listed above |
| Technical debt and advisor report | Preserve MATH regrading (old own-answer final-answer results remain unverified), portable data paths/restores, LFS storage and Drive client ID. Advisor report was paused until cross-conversation collection; the current averaging result may be reported with the user's stated limitation. | Self-generated-label status, collection handoff, SSL handoff and canonical decisions |

## Concrete acceptance criteria for the merge owner

1. Pin the source-tip ledger and record every excluded ref's disposition, including the historical sources above. Recheck tips after the committing freeze; seven idle sessions yesterday do not prove current inactivity.
2. Add the orphan copies and missing compact provenance to the rescue/integration work. Reconcile ignored files in every worktree, all main upload exclusions and required scratch/cache inputs to verified archives or explicit retained paths. Do not delete excluded material.
3. Use a separate integration checkout with LFS smudging disabled and sufficient space. Keep original worktrees and branches. Integrate current SSL and estimator tips before bringing in the remaining lines; do not merely choose one as if it contained the other's latest commits.
4. Verify all intended source tips are ancestors of the final branch; compare the file inventories and explicitly account for deletions and resolved changes. Reconcile complete documentation blocks and all research-direction rows above. Append current guidance rather than allowing historical instructions to take precedence.
5. Run both lineages' affected tests, broader required checks and named numeric replays with their original labels/folds/fitting/calibration contracts. One aggregate number alone is insufficient: also check full prediction arrays/hashes for unchanged frozen outputs, coverage, seals and source snapshots. This review did not run a merged-tree test suite because no merge exists yet.
6. Prove the consolidated runners can resolve their inputs without the old worktree paths. Provide new wrappers/configuration for portability while preserving frozen runners and manifests. A successful test that still imports another worktree does not authorize deleting it.
7. After archive producer checks, restore checks, all content reconciliation and the handoff pass, report an explicit completion result. Only then consider publication/cleanup under the user's authorization. Skip-LFS versus budget increase is a separate storage choice, not the only unresolved issue.

The current finding is **amend the plan before an unconditional go**. An isolated additive merge can proceed while uploads finish if all originals remain; a claim that everything is consolidated or safely removable cannot.
