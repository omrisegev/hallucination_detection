#!/bin/bash
# Submit pb_down.sh as one CPU job (8 cores, 64 GB, 6 h). Optional DEP="afterany:<id>:<id>..." waits for the chains;
# pb_down.sh itself refuses a PRM whose manifest is not COMPLETE / COMPLETE_WITH_FAILURES(context_overflow only).
# For the 12-row timing cells (pipeline smoke only): RUN=timing bash submit_down.sh
set -euo pipefail; source "$(dirname "$0")/env.sh"
dep=(); [ -n "${DEP:-}" ] && dep=(--dependency=$DEP)
sbatch --parsable -A $ACCT -p $PART --qos=$QOS --job-name=pb_down "${dep[@]}" --chdir=$TEL_CODE --container-workdir=$TEL_CODE \
  --export=ALL,WORKDIR=$TEL_CODE,RUN=${RUN:-full},ONLY="${ONLY:-}" "$O/code/submit_cpu.sbatch" bash $O/code/pb_down.sh
