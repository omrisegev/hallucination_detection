#!/bin/bash
# Sync the local working tree to $SHARED/code on AIRCC via tar-over-ssh.
# Push-independent: works with uncommitted changes and without GitHub credentials.
# Run from anywhere (cds to repo root itself):   bash cluster/sync_code.sh [ssh_host]
set -euo pipefail

REMOTE=${1:-aircc}
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1

cd "$(dirname "$0")/.."
echo "syncing $(pwd) -> $REMOTE:$SHARED/code"

# Codex review addendum (handoff §8, "do not perturb active runs"): a sync while a job is
# running or queued can change the code a requeue resumes from, which invalidates resume
# provenance even when the driver itself did not change. Sync only at a clean run boundary.
# Override with ALLOW_SYNC_DURING_RUNS=1 when the change provably cannot reach an active
# acquisition (docs, or an offline analysis script no driver imports) — and say why.
if [ "${ALLOW_SYNC_DURING_RUNS:-0}" != "1" ]; then
    ACTIVE=$(ssh "$REMOTE" "squeue -u \$USER -h -t RUNNING,PENDING 2>/dev/null | wc -l" 2>/dev/null || echo 0)
    if [ "${ACTIVE:-0}" -gt 0 ]; then
        echo "REFUSING TO SYNC: $ACTIVE job(s) running or queued on $REMOTE." >&2
        echo "A requeue would resume from the new tree and its manifest would record a" >&2
        echo "different commit than the run was launched with." >&2
        echo "Wait for a clean boundary, or re-run with ALLOW_SYNC_DURING_RUNS=1 if the" >&2
        echo "change cannot reach an active acquisition." >&2
        exit 1
    fi
fi

# Stamp the commit being synced. `.git` is excluded below, so without this the cluster's
# RUN_MANIFEST.json could only record repo_commit="unknown" — and paper_exact's rule that a
# full run must be traceable to a commit hash would be unenforceable precisely where the
# numbers are produced. spectral_utils/paper_exact/manifest.py:git_info reads this file when
# there is no .git, and treats a missing stamp as a dirty tree so a full run's gate refuses.
COMMIT=$(git rev-parse HEAD 2>/dev/null || echo unknown)
DIRTY=$(test -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" && echo true || echo false)
cat > SYNC_COMMIT.json <<EOF
{"commit": "$COMMIT", "dirty": $DIRTY, "synced_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
echo "  stamped commit $COMMIT (dirty=$DIRTY)"

tar czf - \
  --exclude=.git \
  --exclude='*.pkl' \
  --exclude='*.pkl.part-*' \
  --exclude=dataset_cache \
  --exclude=.worktrees \
  --exclude='*.exe' \
  --exclude='*.pptx' \
  --exclude='*.ipynb' \
  --exclude='*.pdf' \
  --exclude='*.docx' \
  --exclude='*.html' \
  --exclude=cache \
  --exclude=results \
  --exclude=__pycache__ \
  --exclude=.claude \
  . | ssh "$REMOTE" "mkdir -p $SHARED/code && tar xzf - --overwrite -C $SHARED/code"

echo "sync done"
