#!/bin/bash
# Submit the 6 independent jobs of the one SSL experiment (RC_HANDOFF D). Run on the AIRCC login node:
#   bash $S/code_ssl_one_v1/results/ssl_one_experiment_v1/aircc/submit.sh [processbench_C prmbench_V2 ...]
# Re-running it for a job that ended (exit 85 at the wall, or any stop) resumes it: every step is idempotent.
# Prints "<bench>_<V> <job id>".
set -euo pipefail
source "$(dirname "$0")/common.sh"
[ -d "$HD_ROOT/.git" ] || { echo "missing frozen code dir $HD_ROOT" >&2; exit 2; }
snapshot $MODEL >/dev/null; test -s $PRMB_C; test -s $PB_C; test -s $PRMB_FILL; test -s $PB_PRM_RAW
for f in pb_V1_answers.json pb_V2_answers.json prmbench_V1_meta.pkl prmbench_V2_meta.pkl \
         SIDECAR_processbench_V1.jsonl SIDECAR_processbench_V2.jsonl SIDECAR_prmbench_V1.jsonl SIDECAR_prmbench_V2.jsonl; do
  test -s $W/inputs/$f || { echo "missing $W/inputs/$f (make_variants.py)" >&2; exit 2; }
done
mkdir -p $S/logs
list=${*:-"processbench_C processbench_V1 processbench_V2 prmbench_C prmbench_V1 prmbench_V2"}
for t in $list; do
  b=${t%_*}; v=${t##*_}; name=ssl1_$t
  if [ -n "$(squeue -h -u $USER -n $name 2>/dev/null)" ]; then echo "REFUSING: $name is already queued or running" >&2; continue; fi
  j=$(sbatch --parsable -A $ACCT -p $PART --qos=$QOS --job-name=$name --export=ALL,BENCH=$b,V=$v $X/aircc/job.sbatch)
  echo "$t $j"
done
