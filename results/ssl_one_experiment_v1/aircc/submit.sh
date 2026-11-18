#!/bin/bash
# Submit jobs of the one SSL experiment (RC_HANDOFF D). Run on the AIRCC login node:
#   bash $S/code_ssl_one_v1/results/ssl_one_experiment_v1/aircc/submit.sh [processbench_C prmbench_V2 ...]
# Default: the 6 jobs C/V1/V2 x 2 benchmarks. V3/V4 (dev only, make_variants_v34.py) only when named.
# Re-running it for a job that ended (exit 85 at the wall, or any stop) resumes it: every step is idempotent.
# Prints "<bench>_<V> <job id>".
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ -d "$HD_ROOT/.git" ] || { echo "missing frozen code dir $HD_ROOT" >&2; exit 2; }
snapshot $MODEL >/dev/null; test -s $PRMB_C; test -s $PB_C; test -s $PRMB_FILL; test -s $PB_PRM_RAW
list=${*:-"processbench_C processbench_V1 processbench_V2 prmbench_C prmbench_V1 prmbench_V2"}
for t in $list; do   # every input of every named job must exist before anything is submitted
  b=${t%_*}; v=${t##*_}; [ $v = C ] && continue
  for f in $(input_args $b $v | cut -d' ' -f2) $W/inputs/SIDECAR_${b}_$v.jsonl; do
    test -s $f || { echo "missing $f (make_variants.py / make_variants_v34.py)" >&2; exit 2; }
  done
  case $v in V3|V4) f=$(input_args $b preC_$v | cut -d' ' -f2); test -s $f || { echo "missing $f" >&2; exit 2; } ;; esac
done
mkdir -p $S/logs
for t in $list; do
  b=${t%_*}; v=${t##*_}; name=ssl1_$t
  if [ -n "$(squeue -h -u $USER -n $name 2>/dev/null)" ]; then echo "REFUSING: $name is already queued or running" >&2; continue; fi
  j=$(sbatch --parsable -A $ACCT -p $PART --qos=$QOS --job-name=$name --export=ALL,BENCH=$b,V=$v $X/aircc/job.sbatch)
  echo "$t $j"
done
