#!/bin/bash
# Upload the small downstream outputs to Google Drive with rclone ON THE CLUSTER (CLAUDE.md): bank_v2 (FEATURES,
# REWARDS, MANIFEST), telemetry manifests, census, SCORES. The ~1 GB external_telemetry.pkl files stay on AIRCC.
set -euo pipefail; source "$(dirname "$0")/env.sh"
$RC copy $O $GDRIVE --include 'full_*/bank_v2/**' --include 'full_*/manifest.json' --include 'timing_*/manifest.json' \
  --include 'census/**' --include 'scores/**' --include 'timing_*/bank_v2/**' --include 'scores_timing/**' \
  --transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v
$RC lsl $GDRIVE | sort -k4
