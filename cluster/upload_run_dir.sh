#!/usr/bin/env bash
# Back a finished paper_exact run directory up to Google Drive, from the cluster.
#
# Usage:  bash cluster/upload_run_dir.sh <run_dir_name> [--force] [--status]
#   bash cluster/upload_run_dir.sh m2_deepconf_full
#   bash cluster/upload_run_dir.sh m2_deepconf_full --status
#
# Why this exists as a script rather than an ad-hoc ssh line:
#
#   1. The destination is `cluster_results/paper_exact/<run>`, NOT
#      `cluster_results/<run>`. Typing the shallow path by hand returns
#      "directory not found" from `rclone size`, which reads as MISSING and
#      invites a full re-upload into a second, divergent copy. Encoding the
#      path once removes that failure entirely.
#   2. Uploading a directory that a running job is still writing to produces a
#      backup that is torn across files. The freshness guard below refuses it.
#   3. `rclone copy` is idempotent and resumable, so re-running this after a run
#      produces more output is the correct way to top a partial backup up.
#
# The upload runs detached on the cluster, so it survives the ssh session.
set -euo pipefail

REMOTE=${REMOTE:-aircc}
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
RC="$SHARED/bin/rclone"
SRC_ROOT="$SHARED/results/paper_exact"
DST_ROOT="gdrive:hallucination_detection/cluster_results/paper_exact"
# A run is treated as still being written if any file changed this recently.
FRESH_MIN=${FRESH_MIN:-30}

RUN=${1:?usage: upload_run_dir.sh <run_dir_name> [--force] [--status]}
shift || true

FORCE=0
STATUS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --force)  FORCE=1 ;;
        --status) STATUS_ONLY=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

case "$RUN" in
    */*|..|.) echo "run must be a bare directory name, not a path: $RUN" >&2; exit 2 ;;
esac

# The login banner lands on stdout, so fence the payload and cut everything
# outside it. Without this the banner ends up parsed as command output.
rsh() {
    ssh "$REMOTE" "printf '__BEGIN__\n'; { $1; }; printf '\n__END__\n'" 2>/dev/null \
        | sed -n '/^__BEGIN__$/,/^__END__$/p' | sed '1d;$d'
}

SRC="$SRC_ROOT/$RUN"
DST="$DST_ROOT/$RUN"
LOG="$SHARED/upload_${RUN}.log"

# `pgrep -f` sees the command line of the shell that is running it, and that
# command line contains this very pattern -- so a naive `pgrep -f 'rclone copy'`
# always matches itself and reports an upload that does not exist. Writing the
# first character as a bracket expression breaks the self-match: the regex still
# matches a real `rclone copy ...` process, but the literal `[r]clone` sitting in
# our own command line does not match it.
if [ "$STATUS_ONLY" = 1 ]; then
    echo "=== $RUN — backup status ==="
    rsh "test -d '$SRC' && { printf 'local:  '; du -sh '$SRC' | cut -f1; printf 'files:  '; find '$SRC' -type f | wc -l; } || echo 'local:  MISSING'"
    rsh "$RC size '$DST' 2>&1 | sed 's/^/drive:  /'"
    rsh "test -f '$LOG' && tail -3 '$LOG' | sed 's/^/log:    /' || echo 'log:    (none)'"
    rsh "pgrep -f '[r]clone copy .*$RUN' >/dev/null && echo 'upload: RUNNING' || echo 'upload: not running'"
    exit 0
fi

if ! rsh "test -d '$SRC' && echo yes" | grep -q yes; then
    echo "no such run directory on the cluster: $SRC" >&2
    exit 1
fi

# Refuse a torn backup. A run whose jobs are still writing gets skipped, not
# half-copied, unless the caller states outright that they accept it.
FRESH=$(rsh "find '$SRC' -type f -mmin -$FRESH_MIN 2>/dev/null | wc -l")
if [ "${FRESH:-0}" -gt 0 ] && [ "$FORCE" = 0 ]; then
    cat >&2 <<EOF
$RUN has $FRESH file(s) modified in the last $FRESH_MIN minutes — a job is very
likely still writing to it. Backing it up now would capture a torn state.

Wait for the run to finish, or pass --force if a partial snapshot is what you want
(rclone copy is idempotent, so a later top-up run will complete it either way).
EOF
    exit 3
fi

if rsh "pgrep -f '[r]clone copy .*$RUN' >/dev/null && echo yes" | grep -q yes; then
    echo "an upload of $RUN is already running; leaving it alone."
    echo "check it with: bash cluster/upload_run_dir.sh $RUN --status"
    exit 0
fi

echo "uploading $SRC  ->  $DST"
rsh "cd '$SHARED' && nohup '$RC' copy '$SRC' '$DST' --transfers 4 --checkers 8 \
    --drive-chunk-size 64M --stats-one-line --stats 60s -v > '$LOG' 2>&1 < /dev/null & echo started"

sleep 15
echo "--- first lines of $LOG ---"
rsh "tail -5 '$LOG' 2>/dev/null || echo '(log not written yet)'"
echo
echo "check progress with: bash cluster/upload_run_dir.sh $RUN --status"
