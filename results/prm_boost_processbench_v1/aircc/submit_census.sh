#!/bin/bash
# CPU job: tokenizer-only context census for all 5 adapters (no GPU, no weights). Run on the AIRCC login node.
set -euo pipefail; source "$(dirname "$0")/env.sh"
mkdir -p $O/census
sbatch --parsable -A $ACCT -p $PART --qos=$QOS --job-name=pb_census --chdir=$TEL_CODE --container-workdir=$TEL_CODE \
  --export=ALL,WORKDIR=$TEL_CODE,PYTHONPATH=$TEL_CODE "$O/code/submit_cpu.sbatch" \
  python -B $O/code/length_census.py --answers $DATA/answers.json --out $O/census
