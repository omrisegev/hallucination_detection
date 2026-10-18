#!/bin/bash
# Timing: 12 deterministic ProcessBench answers (3 per subset, seed 20261018) per PRM, 1 GPU x 1 h wall each
# (= at most 1 allocated GPU-hour per pair). Same driver, sbatch and code dir as the Socratic timing jobs.
# Needs a /preflight PASS in the same session. Prints one job id per adapter.
set -euo pipefail; source "$(dirname "$0")/env.sh"
for a in $ADAPTERS; do
  out=$O/timing_${a}
  if [ -e "$out" ]; then echo "REFUSING: $out exists" >&2; exit 1; fi
  j=$(sbatch --parsable -A $ACCT -p $PART --qos=$QOS --time=01:00:00 --job-name=pb_time_${a} \
      --export=ALL,CODE_DIR=$TEL_CODE $TEL_CODE/cluster/submit_prm_telemetry_offline.sbatch \
      --mode prm --adapter $a --answers $DATA/answers_timing12.json --out $out)
  echo "$a $j"
done
