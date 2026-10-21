#!/bin/bash
# GPU telemetry of the standalone models: one independent job per PRM (different outputs, so no chain is needed).
#   bash submit_gpu.sh timing [prm ...]    12 rows each, 1 h wall: the budget estimate (HANDOFF step 4)
#   bash submit_gpu.sh full   [prm ...]    all 6,969 PRMBench rows; resubmit with the same args to resume an exit-85 job
# Uses the existing cluster/submit_prm_telemetry_offline.sbatch from the frozen code dir. Prints one job id per PRM.
set -euo pipefail
source "$(dirname "$0")/common.sh"
phase=${1:?timing or full}; shift; list=${*:-$PRMS_SA}
[ -d "$CODE_DIR" ] || { echo "missing CODE_DIR $CODE_DIR (HANDOFF step 2)"; exit 2; }
mkdir -p $W $S/logs
for p in $list; do
  m=${SA_MODEL[$p]:?unknown prm $p}; snapshot "$m" >/dev/null; test -s "${PRM_PKL[$p]}"
  if [ "$phase" = timing ]; then out=$W/timing_$p; wall=01:00:00; extra="--n-samples 12"
  elif [ "$phase" = full ]; then out=$W/full_$p; wall=${SA_WALL[$p]}; extra=""
  else echo "phase must be timing or full"; exit 2; fi
  CODE_DIR=$CODE_DIR sbatch --parsable -A $ACCOUNT -p power-gpu --qos=$QOS --time=$wall \
    --job-name=sa_${phase}_$p --export=ALL,CODE_DIR=$CODE_DIR \
    $CODE_DIR/cluster/submit_prm_telemetry_offline.sbatch \
    --mode instruct --model "$m" --prompt-style ${SA_STYLE[$p]} --meta-pkl "${PRM_PKL[$p]}" \
    --max-batch-tokens ${SA_MAXTOK[$p]} --out $out $extra | sed "s/^/$p $phase job /"
done
