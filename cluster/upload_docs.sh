#!/usr/bin/env bash
# Publish the small text record -- handoffs, HISTORY/PROGRESS/roadmap/glossary,
# the frozen experiment protocols and the fetched cluster summaries -- to Drive,
# so a reader with Drive access but no repo checkout can follow the work.
#
# Usage:  bash cluster/upload_docs.sh [--status]
#
# Notes that are easy to get wrong and are therefore encoded here:
#
#   1. The source is the cluster's synced tree, NOT the local machine. Results
#      reach Drive from the cluster (project rule), and the tree already lives
#      there after `bash cluster/sync_code.sh`. Run the sync first or this
#      publishes a stale copy -- so the script prints the synced commit and
#      refuses if the stamp is missing.
#   2. Filters are include-only, which means everything not named is excluded.
#      That is deliberate: this destination must never accumulate pickles or
#      model weights, and an allow-list fails closed while a deny-list fails
#      open.
#   3. A separate destination from cluster_results/. These are documents about
#      the runs, not the runs, and mixing them makes the acquisition mirror
#      stop being a faithful mirror.
set -euo pipefail

REMOTE=${REMOTE:-aircc}
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
RC="$SHARED/bin/rclone"
SRC="$SHARED/code"
DST="gdrive:hallucination_detection/repo_docs"
LOG="$SHARED/upload_repo_docs.log"

STATUS_ONLY=0
for arg in "$@"; do
    case "$arg" in
        --status) STATUS_ONLY=1 ;;
        *) echo "unknown argument: $arg" >&2; exit 2 ;;
    esac
done

# The login banner lands on stdout, so fence the payload and cut everything
# outside it, exactly as upload_run_dir.sh does.
rsh() {
    ssh "$REMOTE" "printf '__BEGIN__\n'; { $1; }; printf '\n__END__\n'" 2>/dev/null \
        | sed -n '/^__BEGIN__$/,/^__END__$/p' | sed '1d;$d'
}

# Written as [r]clone so the pattern does not match the command line of the
# shell running pgrep -- see the long note in upload_run_dir.sh.
if [ "$STATUS_ONLY" = 1 ]; then
    echo "=== repo_docs — publish status ==="
    rsh "cat '$SRC/SYNC_COMMIT.json' 2>/dev/null | sed 's/^/synced: /' || echo 'synced: (no stamp)'"
    rsh "$RC size '$DST' 2>&1 | sed 's/^/drive:  /'"
    rsh "test -f '$LOG' && tail -3 '$LOG' | sed 's/^/log:    /' || echo 'log:    (none)'"
    rsh "pgrep -f '[r]clone copy .*repo_docs' >/dev/null && echo 'upload: RUNNING' || echo 'upload: not running'"
    exit 0
fi

STAMP=$(rsh "cat '$SRC/SYNC_COMMIT.json' 2>/dev/null || true")
if [ -z "$STAMP" ]; then
    echo "no SYNC_COMMIT.json under $SRC — run 'bash cluster/sync_code.sh' first." >&2
    echo "Publishing without it would put an unidentifiable snapshot on Drive." >&2
    exit 1
fi
echo "publishing the synced tree: $STAMP"

echo "uploading $SRC  ->  $DST"
rsh "cd '$SHARED' && nohup '$RC' copy '$SRC' '$DST' \
    --include '*.md' \
    --include 'SYNC_COMMIT.json' \
    --include 'docs/**' \
    --include 'cluster/*.md' \
    --include 'results/paper_exact_summaries/**' \
    --include 'results/local_online_comprehensive_v1/**' \
    --max-size 16M \
    --transfers 4 --checkers 8 --stats-one-line --stats 30s -v \
    > '$LOG' 2>&1 < /dev/null & echo started"

sleep 20
echo "--- tail of $LOG ---"
rsh "tail -5 '$LOG' 2>/dev/null || echo '(log not written yet)'"
echo
echo "check progress with: bash cluster/upload_docs.sh --status"
