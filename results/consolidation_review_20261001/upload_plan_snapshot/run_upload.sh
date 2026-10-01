#!/bin/bash
# Drive archive of local-only material, 2026-10-01 (branch-merge consolidation).
set -u
RC="/c/Users/DELL/AppData/Local/Microsoft/WinGet/Packages/Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe/rclone-v1.75.0-windows-amd64/rclone"
REPO=/c/Users/omris/TAU/hallucination_detection
U=$(dirname "$0"); DEST=gdrive:hallucination_detection/consolidated_results/local_backup_2026-10-01
cd "$REPO"
echo "START $(date -Is)"
# 1) git bundle of branches blocked by the GitHub LFS budget (objects not on GitHub only), streamed
git bundle create - claude/readout-quickest-detection-v1 claude/depth-feature-fusion-v1 codex/conditional-iu-followups-v1 consolidation/fusion-2026-09-22 master backup/cumulative-vote-fusion-v2-before-publication-20260922 backup-pre-lfs-fix --not --remotes | tee >(md5sum | cut -d' ' -f1 > "$U/bundle.md5") | "$RC" rcat "$DEST/lfs_blocked_branches/lfs_blocked_branches_2026-10-01.bundle"
sleep 2; l=$(cat "$U/bundle.md5"); r=$("$RC" md5sum "$DEST/lfs_blocked_branches/lfs_blocked_branches_2026-10-01.bundle" | cut -d' ' -f1)
[ "$l" = "$r" ] && echo "bundle MD5 OK" || echo "bundle MD5 MISMATCH local=$l remote=$r"
# 2) the 12 LFS objects those branches reference that GitHub does not have (uploaded in place)
"$RC" copy .git/lfs/objects "$DEST/lfs_blocked_branches/lfs_objects/" --files-from "$U/lfs_paths.txt" --transfers 4 -q && echo "lfs objects uploaded"
"$RC" check .git/lfs/objects "$DEST/lfs_blocked_branches/lfs_objects/" --files-from "$U/lfs_paths.txt" --one-way -q && echo "lfs objects CHECK OK"
# 3) heavy per-answer result dirs, one tar each, streamed; md5 of the stream recorded
while read d; do
  tar -cf - "results/$d" | tee >(md5sum | cut -d' ' -f1 > "$U/$d.tar.md5") | "$RC" rcat "$DEST/main_checkout_results/$d.tar"
  sleep 2; l=$(cat "$U/$d.tar.md5"); r=$("$RC" md5sum "$DEST/main_checkout_results/$d.tar" | cut -d' ' -f1)
  [ "$l" = "$r" ] && echo "tar $d MD5 OK" || echo "tar $d MD5 MISMATCH local=$l remote=$r"
done < "$U/heavy_dirs.txt"
# 4) all other untracked result files, as files
"$RC" copy . "$DEST/main_checkout_results/" --files-from "$U/loose_results.txt" --transfers 8 --checkers 16 -q && echo "loose results uploaded"
"$RC" check . "$DEST/main_checkout_results/" --files-from "$U/loose_results.txt" --one-way -q && echo "loose results CHECK OK"
echo "END $(date -Is)"
