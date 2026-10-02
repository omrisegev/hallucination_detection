#!/bin/bash
# Drive archive pass 2 (2026-10-02 rev. D: rev. C + carry-over of run-1 OK rows, Drive-only re-check of verified tars, network wait, failed git scan = FAILED)
# found missing, plus verification of the pass-1 tars. Single sequential process (pass 1 ended 21:38;
# the first pass-2 attempt was stopped before writing anything to Drive).
# Rev. B fixes (Codex check, 2026-10-01 23:53): explicit failure propagation; MD5/member-count readers
# run on FIFOs and are waited for by PID (no sleep-based wait); final summary verifies that every
# expected item has an OK manifest row and exits non-zero otherwise ("END" is not success);
# worktree scan includes untracked (??) as well as ignored (!!) files.
set -u -o pipefail
RC="/c/Users/DELL/AppData/Local/Microsoft/WinGet/Packages/Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe/rclone-v1.75.0-windows-amd64/rclone"
RCO=(--tpslimit 2 --retries 8 --low-level-retries 30)   # retry: gentler on the shared client's per-minute quota
REPO=/c/Users/omris/TAU/hallucination_detection
U=$(cd "$(dirname "$0")" && pwd)
DEST=gdrive:hallucination_detection/consolidated_results/local_backup_2026-10-01
MAN="$U/MANIFEST_v2.tsv"
# Single-run lock (rev. C): a second run refuses to start while one is alive. On Windows/MSYS `ps` does
# not show script names, so the lock records the script PID for stopping it (stop the script PID first,
# then its children).
LOCK="$U/upload_v2.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  old=$(cat "$LOCK/pid" 2>/dev/null)
  if [ -n "$old" ] && kill -0 "$old" 2>/dev/null; then echo "REFUSED: run $old is alive"; exit 3; fi
  echo "stale lock (pid $old not alive); taking it"
fi
echo $$ > "$LOCK/pid"; trap 'rm -rf "$LOCK"' EXIT
printf 'item\tstatus\tmembers_expected\tmembers_in_tar\tbytes_on_drive\tmd5\tdest\n' > "$MAN"
EXPECTED=(); FAILED=()
row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@" >> "$MAN"; }
remote_md5()  { "$RC" md5sum "${RCO[@]}" "$1" 2>/dev/null | cut -d' ' -f1; }
remote_size() { "$RC" size --json "${RCO[@]}" "$1" 2>/dev/null | python -c "import sys,json;print(json.load(sys.stdin)['bytes'])" 2>/dev/null; }

# stream_check NAME PRODUCER... : runs PRODUCER, writes stream MD5 to $U/NAME.md5 and member count to
# $U/NAME.count via FIFO readers that are waited for; consumer command is in global CONSUMER (array).
stream_check() {
  local name="$1"; shift
  local f1="$U/$name.fifo1" f2="$U/$name.fifo2" p1 p2 rp rc1 rc2
  rm -f "$f1" "$f2" "$U/$name.md5" "$U/$name.count"; mkfifo "$f1" "$f2"
  ( md5sum < "$f1" | cut -d' ' -f1 > "$U/$name.md5" ) & p1=$!
  # tar -t must exit 0; grep -c exits 1 when it counts zero members (the count itself is checked by the
  # caller), so only grep status 2 is an error here.
  ( tar -tf - < "$f2" | grep -vc '/$' > "$U/$name.count"; s=("${PIPESTATUS[@]}")
    [ "${s[0]}" -eq 0 ] && [ "${s[1]}" -le 1 ] ) & p2=$!
  "$@" | tee "$f1" "$f2" | "${CONSUMER[@]}"
  rp=("${PIPESTATUS[@]}")
  wait "$p1"; rc1=$?; wait "$p2"; rc2=$?
  rm -f "$f1" "$f2"
  # producer, tee, consumer, md5 reader, count reader
  STATUS=("${rp[0]}" "${rp[1]}" "${rp[2]}" "$rc1" "$rc2")
}

# Rev. D (after run 1 lost DNS 01:44-~03:15): items with an OK row in the previous run's manifest are
# carried over (their row is copied, not redone); a verify item whose previous run had tar status 0 and
# a full member count only re-queries the Drive MD5 and compares it with the recorded local MD5.
PREV="${PREV:-$U/MANIFEST_v2_run1.tsv}"   # retry: PREV=MANIFEST_v2_run2.tsv (Drive 403 per-minute quota failures)
wait_net() {  # up to 60 min for Drive to answer (run 1 lost DNS for ~1.5 h and burned all attempts)
  local i; for i in $(seq 1 60); do "$RC" lsf --max-depth 1 "$DEST/" >/dev/null 2>&1 && return 0; sleep 60; done; return 1
}
carried() {  # carried NAME -> 0 if PREV has an OK row for NAME (row copied into MAN)
  local line; line=$(awk -F'\t' -v i="$1" '$1==i && $2=="OK"' "$PREV" 2>/dev/null)
  [ -n "$line" ] || return 1
  printf '%s\n' "$line" >> "$MAN"; echo "CARRIED OK $1 (from run 1)"; return 0
}

# send NAME BASEDIR LIST SUBDIR : tar the files in LIST (relative to BASEDIR), stream to Drive, verify
send() {
  local name="$1" base="$2" list="$3" dest="$DEST/$4/$1.tar" exp got l r a
  EXPECTED+=("$name"); carried "$name" && return 0; exp=$(grep -c . "$list")
  for a in 1 2 3 4 5; do
    wait_net || { echo "network down >60 min before $name attempt $a"; }
    CONSUMER=("$RC" rcat "${RCO[@]}" "$dest")
    stream_check "$name" tar -cf - -C "$base" -T "$list"
    l=$(cat "$U/$name.md5" 2>/dev/null); got=$(cat "$U/$name.count" 2>/dev/null); r=$(remote_md5 "$dest")
    if [ "${STATUS[*]}" = "0 0 0 0 0" ] && [ -n "$l" ] && [ "$l" = "$r" ] && [ "$got" = "$exp" ]; then
      row "$name" OK "$exp" "$got" "$(remote_size "$dest")" "$l" "$dest"; echo "OK $name ($got files)"; return 0
    fi
    echo "attempt $a FAILED $name status=[${STATUS[*]}] md5 local=$l remote=$r members $got/$exp"; sleep 600
  done
  row "$name" FAILED "$exp" "$got" "" "$l" "$dest"; FAILED+=("$name"); return 1
}

# verify_pass1 NAME : re-stream a pass-1 tar locally with the pass-1 command; compare MD5 with Drive and
# members with the files on disk. No upload.
verify_pass1() {
  local name="$1" item="verify_$1" dest="$DEST/main_checkout_results/$1.tar" exp got l r
  EXPECTED+=("$item"); carried "$item" && return 0; exp=$(cd "$REPO" && find "results/$name" -type f | wc -l)
  # run 1 already re-streamed locally with all statuses 0 and a full count: only the Drive side is missing
  local prev; prev=$(awk -F'\t' -v i="$item" '$1==i && $5=="" && $6 ~ /status=0 0 0 0 0/' "$PREV" 2>/dev/null)
  if [ -n "$prev" ] && [ "$(echo "$prev" | cut -f3)" = "$exp" ] && [ "$(echo "$prev" | cut -f4)" = "$exp" ]; then
    l=$(echo "$prev" | cut -f6 | sed -E 's/^local=([0-9a-f]{32}).*/\1/'); r=$(remote_md5 "$dest"); got=$exp
    if [ -n "$r" ] && [ "$l" = "$r" ]; then
      row "$item" OK "$exp" "$got" "$(remote_size "$dest")" "$l" "$dest"; echo "VERIFIED $name ($got files; local re-stream from run 1, Drive MD5 now)"; return 0
    fi
    row "$item" FAILED "$exp" "$got" "" "local=$l remote=$r (run-1 local)" "$dest"; FAILED+=("$item"); echo "VERIFY FAILED $name md5 $l/$r"; return 1
  fi
  CONSUMER=(cat); stream_check "$item" tar -cf - -C "$REPO" "results/$name" > /dev/null
  l=$(cat "$U/$item.md5" 2>/dev/null); got=$(cat "$U/$item.count" 2>/dev/null); r=$(remote_md5 "$dest")
  if [ "${STATUS[*]}" = "0 0 0 0 0" ] && [ -n "$l" ] && [ "$l" = "$r" ] && [ "$got" = "$exp" ]; then
    row "$item" OK "$exp" "$got" "$(remote_size "$dest")" "$l" "$dest"; echo "VERIFIED $name ($got files)"; return 0
  fi
  row "$item" FAILED "$exp" "$got" "" "local=$l remote=$r status=${STATUS[*]}" "$dest"; FAILED+=("$item")
  echo "VERIFY FAILED $name status=[${STATUS[*]}] md5 $l/$r members $got/$exp"; return 1
}

echo "START $(date -Is)"
cd "$REPO"

# 1) pass-1 failures, retried with member counts
for d in localization_full_sampling_v3 fusion_multiwidth_dense_v1; do
  find "results/$d" -type f > "$U/l_$d.txt"; send "$d" "$REPO" "$U/l_$d.txt" main_checkout_results || true
done

# 2) verification of the six pass-1 tars that had only an MD5 match
for d in lsml_external_generalization_v1 localization_full_benchmark_v3 localization_full_shortlist_v3 \
         fusion_shrinkage_iu_v1 fusion_multiwidth_iu_v1 fusion_onset_innovation_iu_v1; do
  verify_pass1 "$d" || true
done

# 3) main-checkout ignored files outside the pass-1 lists (review list, 36,143 files)
send main_checkout_ignored_results "$REPO" "$U/main_ignored.txt" main_checkout_results || true

# 4) ignored (!!) AND untracked (??) files of each worktree (SSL covered by its verified 2026-09-30 archive)
for w in readout-quickest-detection-v1 whitebox-layer-views-v1 token-probability-fusion-v1 cumulative-vote-fusion-v2 \
         er-generality-v1 lsml-ct7-levers-run self-generated-step-labels-v1 decision-rule-v1 depth-feature-fusion-v1 \
         digit-alternative-probability-v1 family15-tail20-transfer-v1 consolidation-fusion-2026-09-22 a6-s0b \
         tensor-mom-v1; do
  W="$REPO/.worktrees/$w"; [ -d "$W" ] || { echo "missing worktree $w"; FAILED+=("worktree_$w"); continue; }
  # git status must succeed; a failed scan is a FAILED item, never an EMPTY one
  if ! git -C "$W" status --porcelain --ignored -uall > "$U/wt_$w.status"; then
    EXPECTED+=("worktree_offgit_$w"); row "worktree_offgit_$w" FAILED "" "" "" "git status failed" ""
    FAILED+=("worktree_offgit_$w"); continue
  fi
  grep -E '^(!!|\?\?) ' "$U/wt_$w.status" | cut -c4- | sed 's/^"//; s/"$//' > "$U/wt_$w.txt"
  if [ -s "$U/wt_$w.txt" ]; then send "worktree_offgit_$w" "$W" "$U/wt_$w.txt" worktree_offgit || true
  else row "worktree_offgit_$w" EMPTY 0 0 "" "git status ok, no off-git files" ""; fi
done

# 5) the two backup-pre-lfs-fix LFS objects not found on Drive by SHA-256 (GPQA raw pickles)
EXPECTED+=(backup_pre_lfs_fix_missing_objects)
if carried backup_pre_lfs_fix_missing_objects; then :
elif "$RC" copy "${RCO[@]}" "$REPO/.git/lfs/objects" "$DEST/backup_pre_lfs_fix_lfs_objects/" --files-from "$U/bplf_missing_paths.txt" -q \
   && "$RC" check "${RCO[@]}" "$REPO/.git/lfs/objects" "$DEST/backup_pre_lfs_fix_lfs_objects/" --files-from "$U/bplf_missing_paths.txt" --one-way; then
  row backup_pre_lfs_fix_missing_objects OK "$(grep -c . "$U/bplf_missing_paths.txt")" "" "" "rclone check" "$DEST/backup_pre_lfs_fix_lfs_objects/"
else
  row backup_pre_lfs_fix_missing_objects FAILED "" "" "" "" "$DEST/backup_pre_lfs_fix_lfs_objects/"; FAILED+=(backup_pre_lfs_fix_missing_objects)
fi

# 6) scratch/external_generalization_private files not found on Drive by MD5+size
send scratch_external_generalization_private_missing "$REPO/scratch" "$U/scratch_private_missing.txt" scratch || true

# Final summary: every expected item must have an OK row
"$RC" copy "${RCO[@]}" "$MAN" "$DEST/" -q || FAILED+=(manifest_upload)
for it in "${EXPECTED[@]}"; do
  awk -F'\t' -v i="$it" '$1==i && $2=="OK"{f=1} END{exit !f}' "$MAN" || { [[ " ${FAILED[*]} " == *" $it "* ]] || FAILED+=("$it(no OK row)"); }
done
echo "END $(date -Is)"
if [ ${#FAILED[@]} -eq 0 ]; then echo "PASS2 RESULT: ALL ${#EXPECTED[@]} EXPECTED ITEMS VERIFIED"; exit 0
else echo "PASS2 RESULT: INCOMPLETE - failed: ${FAILED[*]}"; exit 1; fi
