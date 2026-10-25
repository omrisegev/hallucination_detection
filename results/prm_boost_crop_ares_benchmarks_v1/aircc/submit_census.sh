#!/bin/bash
# CPU: tokenizer-only context census of every new benchmark for the 5 PRM adapters (prm mode), before any GPU job.
# Reuses results/prm_boost_processbench_v1/aircc/length_census.py (copied to $O/code). Over-long answers are never
# truncated: the driver records them as failure='context_overflow'. Prints one job id per benchmark.
set -euo pipefail; source "$(dirname "$0")/env.sh"
mkdir -p $O/census
for b in $NEW_BENCHES; do
  sbatch --parsable -A $ACCT -p $PART --qos=$QOS --job-name=cab_census_$b --chdir=$TEL_CODE --container-workdir=$TEL_CODE \
    --export=ALL,WORKDIR=$TEL_CODE,PYTHONPATH=$TEL_CODE "$O/code/submit_cpu.sbatch" \
    python -B $O/code/length_census.py --answers ${ANSWERS[$b]} --out $O/census/$b | sed "s/^/$b /"
done
