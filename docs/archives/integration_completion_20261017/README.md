# Integration preservation — 17 October 2026

[Open the verified Drive folder](https://drive.google.com/drive/folders/1JXHIBnL5Wtt5rFbkcjDIjotodgQbtHJt).

Cloud destination: `gdrive:hallucination_detection/consolidated_results/integration_completion_2026-10-17_143809Z`.

`BACKUP_MANIFEST.json` enumerates every archive/member with original source, member path, size and SHA256. `BACKUP_VERIFICATION.json` records the completed local member replay and remote archive checksum check. These completion records are published only after verification succeeds.

## Contents

- 100 NPZ files from clustered IU-PCR, F1/F2/F7, signal gallery and error-count studies.
- 155 off-Git advisor v11 input/result files and eight derived agent N16 files.
- All 7,964 inventoried nonignored untracked files from the historical worktrees.
- Exact copies of 21 locally modified tracked files and five patches; base commits are in `WORKTREE_DIRTY_INVENTORY.json`.

Archiving preserves these historical bytes; it does not activate old research directions or certify their results. No source worktree was changed or removed.

## Restore into a new directory

Use `rclone copyto` to fetch the manifest and the required named archives from the destination above. Verify each archive against its `sha256` in the manifest before extraction. For example, from the repository root:

```sh
mkdir /tmp/integration-restore-new
rclone copyto gdrive:hallucination_detection/consolidated_results/integration_completion_2026-10-17_143809Z/BACKUP_MANIFEST.json /tmp/integration-restore-new/BACKUP_MANIFEST.json
rclone copyto gdrive:hallucination_detection/consolidated_results/integration_completion_2026-10-17_143809Z/raw_signal_gallery_extensions_2026-10-12.tar.gz /tmp/integration-restore-new/raw_signal_gallery_extensions_2026-10-12.tar.gz
```

Each archive lists its `storage` files in order. If there is one file named exactly like the archive, download it directly. If there are `.partNNN` files, download every listed part, verify each part hash, concatenate them in the listed order, and verify the reconstructed archive against its top-level SHA256. Never extract an individual part.

After verifying the hash, extract the selected archive into a fresh restore directory. Recent payloads retain `results/<package>/...` member paths. Historical loose files use `worktrees/<original-name>/...`; modified tracked snapshots use `dirty_worktrees/<original-name>/...`, and the five diffs use `patches/`. Do not replace live HISTORY/PROGRESS files with old complete snapshots. Apply a patch only to a separate checkout at its recorded base, or inspect the snapshot directly.

Every member was hashed while packing and independently read back from the completed local tar. `rclone check` checks every uploaded archive against remote object checksums. The remote manifest is also read back byte-for-byte. This is not a claim that every historical algorithm was rerun.

The original `prm_boosting_v1.tar` and `thinkfix_rescore_v1.tar` inputs were not duplicated: their exact source copies remain under `gdrive:hallucination_detection/consolidated_results/local_backup_2026-10-12/results/*.tar.partx000`, as recorded in the original audit.

## Verified completion

21 archives preserve 8,253 source files (4,521,223,525 bytes; compressed archives 2,339,085,998 bytes). Storage uses 26 payload objects; the remote checksum check matched all 26 plus the manifest, with zero differences. The downloaded restore sample also passed all 11 member hashes.
