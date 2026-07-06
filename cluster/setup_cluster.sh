#!/bin/bash
# One-time cluster environment setup. Run from the LOCAL machine:
#   ssh aircc 'bash -s' < cluster/setup_cluster.sh
set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1

echo "== creating directory layout under $SHARED =="
mkdir -p "$SHARED"/{code,hf_cache,results,logs,pip_cache}
ls -la "$SHARED"

echo
echo "== account / QoS info (use these to fill -p / --qos at submit time) =="
sdata 2>/dev/null || echo "(sdata not available — ask support for partition/QoS names)"

echo
echo "== storage =="
df -h /shared 2>/dev/null || true

echo
echo "setup done. Next: bash cluster/sync_code.sh  (from the local repo root)"
