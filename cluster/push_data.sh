#!/bin/bash
# Push a local data directory to $SHARED/<remote_rel_dir> on AIRCC via tar-over-ssh,
# mirroring sync_code.sh (which deliberately excludes pkls — this script exists to
# move DATA: raw inference pkls for the view backfill, Colab-era caches, etc.).
#
# Usage:
#   bash cluster/push_data.sh <local_dir> <remote_rel_dir> [ssh_host]
#   bash cluster/push_data.sh --verify <local_dir> <remote_rel_dir> [ssh_host]
#
# Examples:
#   bash cluster/push_data.sh cache/repgrid/sciq_llama8b results/sciq_llama8b
#   bash cluster/push_data.sh cache/colab_src/math500__Qwen-Math-7B_T1.0 \
#        data/colab/math500__Qwen-Math-7B_T1.0
#
# --verify re-hashes every file on both ends (sha256) and diffs — run it after any
# push whose source is irreplaceable (Colab-era pkls are the only copy outside Drive).
set -euo pipefail

VERIFY=0
if [ "${1:-}" = "--verify" ]; then
  VERIFY=1
  shift
fi

LOCAL_DIR=${1:?usage: push_data.sh [--verify] <local_dir> <remote_rel_dir> [ssh_host]}
REMOTE_REL=${2:?usage: push_data.sh [--verify] <local_dir> <remote_rel_dir> [ssh_host]}
REMOTE=${3:-aircc}
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1

[ -d "$LOCAL_DIR" ] || { echo "error: $LOCAL_DIR is not a directory" >&2; exit 1; }

echo "pushing $LOCAL_DIR -> $REMOTE:$SHARED/$REMOTE_REL"
tar czf - -C "$LOCAL_DIR" --exclude='*.tmp' . \
  | ssh "$REMOTE" "mkdir -p $SHARED/$REMOTE_REL && tar xzf - -C $SHARED/$REMOTE_REL"
echo "push done"

if [ "$VERIFY" = 1 ]; then
  echo "verifying sha256 on both ends..."
  LOCAL_SUMS=$(cd "$LOCAL_DIR" && find . -type f ! -name '*.tmp' -print0 \
    | sort -z | xargs -0 sha256sum)
  REMOTE_SUMS=$(ssh "$REMOTE" "cd $SHARED/$REMOTE_REL && find . -type f ! -name '*.tmp' -print0 \
    | sort -z | xargs -0 sha256sum")
  if [ "$LOCAL_SUMS" = "$REMOTE_SUMS" ]; then
    echo "verify OK ($(echo "$LOCAL_SUMS" | wc -l) files match)"
  else
    echo "VERIFY FAILED — local vs remote diff:" >&2
    diff <(echo "$LOCAL_SUMS") <(echo "$REMOTE_SUMS") >&2 || true
    exit 1
  fi
fi
