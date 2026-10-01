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

## Pass 2

(to be filled in)
