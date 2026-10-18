#!/bin/bash
# Full run: one linear 2-link chain per PRM (link 2 = --dependency=afterany:<link 1>, same --out, resumes from the
# checkpoint; never two jobs on the same link, never a fan-out). Do NOT run before Omri's budget decision:
#   BUDGET_OK=1 WALL_7B=03:00:00 WALL_72B=08:00:00 bash submit_full_chains.sh
# Set the walls from the measured timing (see HANDOFF.md). Prints "<adapter> <link1> <link2>".
set -euo pipefail; source "$(dirname "$0")/env.sh"
[ "${BUDGET_OK:-0}" = 1 ] || { echo "REFUSING: set BUDGET_OK=1 only after Omri approves the full-run budget" >&2; exit 1; }
: "${WALL_7B:?set WALL_7B (per link, from timing)}"; : "${WALL_72B:?set WALL_72B (per link, from timing)}"
for a in $ADAPTERS; do
  out=$O/full_${a}
  if [ -e "$out/manifest.json" ]; then echo "REFUSING: $out already has a manifest" >&2; exit 1; fi
  w=$WALL_7B; [ "$a" = qwen_prm_72b ] && w=$WALL_72B
  args=(--mode prm --adapter $a --answers $DATA/answers.json --out $out)
  common=(-A $ACCT -p $PART --qos=$QOS --time=$w --export=ALL,CODE_DIR=$TEL_CODE)
  j1=$(sbatch --parsable "${common[@]}" --job-name=pb_full_${a}_1 $TEL_CODE/cluster/submit_prm_telemetry_offline.sbatch "${args[@]}")
  j2=$(sbatch --parsable "${common[@]}" --job-name=pb_full_${a}_2 --dependency=afterany:$j1 \
       $TEL_CODE/cluster/submit_prm_telemetry_offline.sbatch "${args[@]}")
  echo "$a $j1 $j2"
done
