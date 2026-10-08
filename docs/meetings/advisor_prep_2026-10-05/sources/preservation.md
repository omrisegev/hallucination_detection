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
Results: (to be filled in from `MANIFEST_v2.tsv`)
