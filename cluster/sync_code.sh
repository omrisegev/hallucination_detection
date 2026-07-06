#!/bin/bash
# Sync the local working tree to $SHARED/code on AIRCC via tar-over-ssh.
# Push-independent: works with uncommitted changes and without GitHub credentials.
# Run from anywhere (cds to repo root itself):   bash cluster/sync_code.sh [ssh_host]
set -euo pipefail

REMOTE=${1:-aircc}
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1

cd "$(dirname "$0")/.."
echo "syncing $(pwd) -> $REMOTE:$SHARED/code"

tar czf - \
  --exclude=.git \
  --exclude='*.pkl' \
  --exclude='*.ipynb' \
  --exclude='*.pdf' \
  --exclude='*.docx' \
  --exclude='*.html' \
  --exclude=cache \
  --exclude=results \
  --exclude=__pycache__ \
  --exclude=.claude \
  . | ssh "$REMOTE" "mkdir -p $SHARED/code && tar xzf - -C $SHARED/code"

echo "sync done"
