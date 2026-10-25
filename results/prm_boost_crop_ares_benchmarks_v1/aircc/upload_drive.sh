#!/bin/bash
# rclone ON THE CLUSTER (CLAUDE.md): manifests, census, parity, banks, scores. The raw external_telemetry.pkl files
# (~1 GB each) are uploaded too when UPLOAD_RAW=1. Idempotent; re-run to top up.
set -euo pipefail; source "$(dirname "$0")/env.sh"
inc=(--include '*/manifest.json' --include 'census/**' --include 'parity*/**' --include 'PARITY_*' --include '*/bank_*/**'
     --include 'scores*/**' --include 'code/**')
[ "${UPLOAD_RAW:-0}" = 1 ] && inc+=(--include '*/external_telemetry.pkl')
$RC copy $O $GDRIVE "${inc[@]}" --transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v
$RC copy $DATA $GDRIVE/inputs --transfers 4 -v
$RC lsl $GDRIVE | sort -k4 | tail -40
