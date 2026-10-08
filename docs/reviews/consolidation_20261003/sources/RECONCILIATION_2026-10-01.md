# Backup reconciliation, 2026-10-01 consolidation

Drive root for this pass: `gdrive:hallucination_detection/consolidated_results/local_backup_2026-10-01/`
(local rclone, see memory note on the WinGet path). Process completion is not counted as success.

## State after pass 1 (`run_upload.sh`, 12:47-21:38, log reconciled 2026-10-01 evening)

| Category | Item | Evidence |
|---|---|---|
| Completed and verified | `lfs_blocked_branches/lfs_blocked_branches_2026-10-01.bundle` (7,428,475 B) | Drive MD5 `b8a5ab81ee6a0b687022366505ed331c` = stream MD5; downloaded copy passes `git bundle verify`; 7 heads (readout-quickest, depth, conditional-iu, consolidation/fusion-2026-09-22, master, backup/cumulative-vote-fusion-v2-before-publication-20260922, backup-pre-lfs-fix). Prerequisite commits are on GitHub. |
| Completed and verified | `lfs_blocked_branches/lfs_objects/` (12 objects, ~930 MB) | `rclone check` OK; equal to the LFS OID union of all bundled refs except backup-pre-lfs-fix |
| Completed and verified | `main_checkout_results/` loose files (3,831 paths) | `rclone check --one-way` OK |
| Uploaded, MD5 matches; member count NOT yet checked | `main_checkout_results/{lsml_external_generalization_v1, localization_full_benchmark_v3, localization_full_shortlist_v3, fusion_shrinkage_iu_v1, fusion_multiwidth_iu_v1, fusion_onset_innovation_iu_v1}.tar` | Stream MD5 = Drive MD5. The pass-1 script did not check tar's exit status or the member count (review finding 4). |
| Failed | `localization_full_sampling_v3.tar` (6.9 GB) | DNS error at 16:31; nothing on Drive |
| Failed | `fusion_multiwidth_dense_v1.tar` (88 MB) | Drive 403 quota at 19:46; nothing on Drive |
| Not yet uploaded | 36,143 ignored main-checkout files (1,541,252,083 B), list `results/consolidation_review_20261001/OMITTED_MAIN_IGNORED_FILES.txt` | incl. `results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz` |
| Not yet uploaded | Ignored files of the other worktrees (~3.6 GB observed by the review) | |
| Not yet uploaded | Five orphan files (`.worktrees/literature-data-diagnostics-v1`, `rbm-hierarchical-time-v1-setup-recovery`) | copies + SHA256 in `results/consolidation_review_20261001/preserved_orphans/` |
| Covered by a previously verified backup | SSL worktree off-git results: 367 files, 3,969,226,611 B | `docs/archives/DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.md` on the SSL branch (rclone check 367/367 MD5). Not re-hashed in this pass. |
| Not checked | backup-pre-lfs-fix LFS payload (63 objects, ~19.7 GB: 60 `dataset_cache/repgrid`, 3 `dataset_cache/edis_aime24`) | vs `dataset_cache/DRIVE_BACKUP_2026_08_20.json` |
| Not checked | `scratch/external_generalization_private` (9.3 GB) | vs `gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/` |

## Coverage checks against existing Drive backups (2026-10-01 evening)

Method: one full listing of `gdrive:hallucination_detection` with rclone hashes (24,816 files, 137.1 GB;
each with MD5, SHA-1 and SHA-256), matched by content hash AND size, independent of path. The Drive
path that covers each file is recorded. Evidence: `results/consolidation_2026-10-01/`.

| Item | Covered by an existing Drive copy | Not on Drive | Evidence |
|---|---|---|---|
| backup-pre-lfs-fix LFS objects (63, git-lfs reports none on GitHub) | 61, SHA-256 (= LFS OID) and size equal; mostly `cluster_results/<job>/raw_*.pkl` and `cluster_results/repgrid/...` | 2 objects, 6.17 GB: `dataset_cache/repgrid/gpqa_r1distill8b/raw_gpqa_T1.0.pkl` (OID 49d0afb9...) and `dataset_cache/repgrid/trace_gpqa_r1qwen7b/raw_gpqa_T1.0.pkl` (OID f3672406...), both from commit 138a8b32c (2026-08-07) | `bplf_coverage.json` |
| `scratch/external_generalization_private` (6,670 files, 9.9 GB) | 24 files, 2.24 GB, MD5 and size equal; under `cluster_results/lsml_external_generalization_v1/{full_archives,evaluation_archives,inputs_20260924,...}` and the SSL archive | 6,646 files, 7.64 GB (6,193 are extracted members of `evaluation_archives/`; 363 `snapshot_956_complete`; sources, wheels, smoke records, inputs, code tarballs) | `scratch_coverage.json` |

The 20 August record `dataset_cache/DRIVE_BACKUP_2026_08_20.json` covers 23 `four_localization`-type
payloads (3.35 GB), not the repgrid pickles; it is not the source of the 61 matches above.
Extracted members are uploaded although their parent tarballs may be on Drive: member-level coverage
inside a tarball was not verified, so it is not claimed.

## Pass 2 (`run_upload_v2.sh`, started after pass 1 ended; single sequential process)

Items: retry of the two pass-1 failures; local re-stream verification (MD5 + member count) of the six
pass-1 tars; main-checkout ignored files (36,143); ignored files of 14 worktrees (SSL excluded,
covered); the 2 uncovered backup-pre-lfs-fix objects; the 6,646 uncovered scratch files.

Acceptance rule for every tar (script `results/consolidation_2026-10-01/run_upload_v2.sh`): all five
pipeline statuses zero (tar, tee, rclone rcat, MD5 reader, member-count reader), local stream MD5 equal
to Drive MD5, and tar member count equal to the expected file count. A single-run lock prevents two
uploads writing the same destination. Anything else is FAILED.

Runs:

- **Run 1** (2026-10-02 00:39:55 to 04:11:10): repeated DNS failures from 01:44 until the end of the run, with some successes in between, failed 21 of 25
  items; 4 small worktree tars passed. Manifest: `results/consolidation_2026-10-01/MANIFEST_v2_run1.tsv`.
- **Run 2** (2026-10-02 12:17:36 to 22:27:57; script revision D: carries over run-1 OK rows, waits for the
  network, verifies pass-1 tars by local re-stream plus Drive MD5): 22 of 25 OK, 3 FAILED, all with Drive
  `403 RATE_LIMIT_EXCEEDED` on the per-minute quota of rclone's shared client_id (project 202264815644).
  Manifest `MANIFEST_v2_run2.tsv`; condensed log `upload_v2_run2_log_condensed.txt`.
- **Retry** (started 22:28:38 automatically after run 2 released the lock; `run_upload_v2_retry.sh`:
  gentler rate, 5 attempts 10 minutes apart, only the 3 failed items). Attempt 1 for
  `fusion_multiwidth_dense_v1` failed again with the same 403. Attempt 2 also failed with the 403; its local
  stream MD5 differed from attempt 1, most likely because a stray `grep` from a documentation agent opened
  the attempt's FIFO and consumed part of the MD5 stream (the shell was found and killed at 23:06). The
  acceptance rule would have rejected any such attempt; nothing was written to Drive. Result to be
  appended below.
- Drive listing at 22:53: no duplicate file names under the destination, and no partial object for any
  of the 3 failed items.

## Final reconciliation (as of 2026-10-02 22:55)

| Category | Item | Evidence |
|---|---|---|
| Completed and verified | Bundle of the LFS-blocked branches + 12 LFS objects + 3,831 loose main-checkout files | pass 1 (table above); bundle restore tested |
| Completed and verified | 6 pass-1 result tars (`lsml_external_generalization_v1` 20,094 files; `localization_full_benchmark_v3` 27,764; `localization_full_shortlist_v3` 27,551; `fusion_shrinkage_iu_v1` 27,544; `fusion_multiwidth_iu_v1` 27,543; `fusion_onset_innovation_iu_v1` 27,542) | run 2: local re-stream MD5 = Drive MD5 and member count equal |
| Completed and verified | `localization_full_sampling_v3.tar` (28,008 files, 20.2 GB) | run 2, MD5 `fabc2437...` |
| Completed and verified | `main_checkout_ignored_results.tar` (36,143 files, incl. `CT7_TOKEN_MATRICES.npz`) | run 2, MD5 `72fc37c0...` |
| Completed and verified | Off-git files of 12 worktrees: readout-quickest-detection-v1 (2,175), whitebox-layer-views-v1 (37), token-probability-fusion-v1 (113), cumulative-vote-fusion-v2 (1,296), self-generated-step-labels-v1 (39), decision-rule-v1 (38), depth-feature-fusion-v1 (35), digit-alternative-probability-v1 (38), family15-tail20-transfer-v1 (34), consolidation-fusion-2026-09-22 (51), a6-s0b (14), tensor-mom-v1 (6) | runs 1-2, per-item MD5 in `MANIFEST_v2_run2.tsv` |
| Completed and verified | 2 uncovered backup-pre-lfs-fix LFS objects (the GPQA pickles, 6.17 GB) | run 2, `rclone check` 0 differences, 2 matching |
| Completed and verified | 6,646 uncovered `scratch/external_generalization_private` files (7.65 GB) | run 2, MD5 `e50ed657...`, 6,646 members |
| Completed and verified | 5 orphan files from deleted worktrees | committed with SHA-256 in `results/consolidation_review_20261001/preserved_orphans/` (git, pushed) |
| Failed (retry running) | `fusion_multiwidth_dense_v1.tar` (27,546 files, ~88 MB) | 403 quota; local tar stream complete (member count equal); nothing on Drive |
| Failed (retry running) | Off-git files of `er-generality-v1` (37 files) | same |
| Failed (retry running) | Off-git files of `lsml-ct7-levers-run` (43 files) | same |
| Not yet uploaded | none known beyond the 3 failed items | |
| Covered by a previously verified backup | SSL worktree off-git results (367 files, 3.97 GB) | `DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.md` (367/367 MD5) |
| Covered by a previously verified backup | 61 of 63 backup-pre-lfs-fix LFS objects; 24 of 6,670 scratch files | content hash + size match, `bplf_coverage.json`, `scratch_coverage.json` |

Consequences: the 3 failed items, and the worktrees/directory that hold them, must not be deleted until
they are verified on Drive. A personal rclone client_id (Omri, Google Cloud Console) removes the shared
quota and is needed anyway: the shared one is being retired during 2026. The merge itself is additive and
does not depend on these 3 items.
