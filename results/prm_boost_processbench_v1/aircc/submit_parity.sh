#!/bin/bash
# Submit bank_parity.sh as one CPU job (8 cores, 64 GB, 6 h). Optional DEP="afterany:<id>:<id>..." waits for the chains.
set -euo pipefail; source "$(dirname "$0")/env.sh"
dep=(); [ -n "${DEP:-}" ] && dep=(--dependency=$DEP)
sbatch --parsable -A $ACCT -p $PART --qos=$QOS --job-name=pb_parity "${dep[@]}" --chdir=$TEL_CODE --container-workdir=$TEL_CODE \
  --export=ALL,WORKDIR=$TEL_CODE,ONLY="${ONLY:-}" "$O/code/submit_cpu.sbatch" bash $O/code/bank_parity.sh
