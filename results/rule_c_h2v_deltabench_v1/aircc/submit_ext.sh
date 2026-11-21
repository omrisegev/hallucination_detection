#!/bin/bash
# Submit rule C transfer jobs on cycle4 (Omri 2026-11-21 22:10: "yes, set up cycle4 and run H2V + DeltaBench").
#   bash submit_ext.sh BENCH V [DEPENDENCY_JOBID]
# Needs a /preflight PASS in the same session. Refuses a duplicate job name.
set -euo pipefail
source "$(dirname "$0")/common_c4.sh"
BENCH=${1:?bench}; V=${2:?V}; DEP=${3:-}
name=rc_${BENCH:0:3}_$V
if [ -n "$(squeue -h -u $USER -n $name 2>/dev/null)" ]; then echo "REFUSING: $name is queued or running" >&2; exit 1; fi
mkdir -p $S/logs
sbatch --parsable -A $ACCT -p $PART --qos=$QOS --job-name=$name ${DEP:+--dependency=afterany:$DEP} \
  --export=ALL,BENCH=$BENCH,V=$V $Y/aircc/job_ext.sbatch
