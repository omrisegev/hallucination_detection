#!/bin/bash
# Submit A -> B -> C linearly (afterok: a parity failure in A stops B and C).
set -euo pipefail
export SLURM_CONF_SERVER=controller-primary
cd /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_collection_v1
A=$(sbatch --parsable job_a_prmbench.sh); B=$(sbatch --parsable --dependency=afterok:$A job_b_signabs.sh)
Cj=$(sbatch --parsable --dependency=afterok:$B job_c_external.sh)
echo "JOBS A=$A B=$B C=$Cj" | tee -a JOB_IDS.txt
