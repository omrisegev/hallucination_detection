# Drive archive: off-git results of claude/ssl-pseudolabel-residual-v1 (2026-09-30 / 2026-10-01)

Omri asked (2026-09-30) to upload to Drive the large results of this branch that are not in git, so nothing is lost when the
worktree is removed in the merge, and to document the upload.

- **Where:** `gdrive:hallucination_detection/consolidated_results/local_backup_2026-09-30/ssl-pseudolabel-residual-v1/`,
  mirroring the worktree's relative paths (e.g. `.../ssl-pseudolabel-residual-v1/results/per_dataset_fit_v1/run_20260929/STEP_SCORES.npz`).
- **What:** every file under `results/` of the worktree `.worktrees/ssl-pseudolabel-residual-v1` that git ignores or does not
  track: 367 files, 3,969,226,611 bytes (3.70 GiB), in 23 result directories (largest: algorithm_decisions_v1 1.45 GB,
  expectation_realization_v1 0.72 GB, lsml_merge_step_v1 0.31 GB, ssl_pseudolabel_residual_v1 0.23 GB, external_banks_v4 0.20 GB).
  It includes every STEP_SCORES.npz / BOOTSTRAP_DELTAS.npz of Steps 438-462, the external
  `algorithm_external_v1/run_20260929/*/PREDICTIONS_UNSEALED.json` whose digests are in SEAL.json, the external_banks_v4 features
  and reference arrays, and the hash-verified pool input backup `algorithm_decisions_v1/inputs_backup/pool_z.npy`.
- **Manifest:** [DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.csv](DRIVE_ARCHIVE_ssl-pseudolabel-residual-v1_2026-09-30.csv)
  (path, bytes, md5 as reported by Drive).
- **How:** local rclone 1.75.0 (`gdrive:` remote), `rclone copy <worktree> <dest> --files-from <list> --transfers 4 --checkers 8
  --drive-chunk-size 64M`, started 2026-09-30 23:21, finished 2026-10-01 about 02:15 (network stalls, automatic retries,
  no errors).
- **Verification:** `rclone check <worktree> <dest> --files-from <list> --one-way` (size + MD5): **367 matching files, 0
  differences** (2026-10-01 02:16). No local file changed during the upload (sizes re-checked: 3.697 GiB).
- **Restore:** `rclone copy "gdrive:hallucination_detection/consolidated_results/local_backup_2026-09-30/ssl-pseudolabel-residual-v1/results/<dir>" "<worktree>/results/<dir>"`,
  then compare with the manifest's md5 (`rclone md5sum` or `certutil -hashfile <file> MD5`).
- **Not included:** the private external inputs in the main checkout (`scratch/external_generalization_private/inputs`, backed up
  separately per the 2026-09-24 external-telemetry record) and anything outside `results/`.
- **Caveat:** rclone warns that the shared Google Drive client_id it uses will stop working during 2026; creating a personal
  client_id (https://rclone.org/drive/#making-your-own-client-id) keeps restores possible.
