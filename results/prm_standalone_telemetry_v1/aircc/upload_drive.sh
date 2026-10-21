#!/bin/bash
# Outputs -> gdrive:hallucination_detection/cluster_results/prm_standalone_telemetry_v1/ (copy, idempotent). Run detached:
#   nohup bash upload_drive.sh > $W/upload.log 2>&1 &
set -euo pipefail
source "$(dirname "$0")/common.sh"
RC="$S/bin/rclone --config $S/.config/rclone/rclone.conf"
D=gdrive:hallucination_detection/cluster_results/prm_standalone_telemetry_v1
F="--transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v"
$RC copy $W/abl5 $D/abl5 $F
for p in $PRMS_SA; do
  $RC copy $W/full_$p $D/full_$p $F --exclude '*.pkl'          # manifests and identity only; raw pkls stay on AIRCC
  $RC copy $W/bank_$p $D/bank_$p $F
done
$RC copy $W $D/jobs $F --max-depth 1
$RC check $W/abl5 $D/abl5 --one-way && echo CHECK_ABL5_OK
echo UPLOAD_DONE
