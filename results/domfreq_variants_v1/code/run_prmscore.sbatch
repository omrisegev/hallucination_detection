#!/bin/bash
# domfreq_variants_v1 stage 2 (CPU): Step-0 alignment + metrics (a)-(e). Labels used for evaluation only.
#SBATCH --job-name=dfv_prmscore
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/results/domfreq_variants_v1/logs/%x_%j.out
set -uo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/pyenv_native_312_20261011/activate.sh
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
D=/shared/cycle2_tau_averbuch_prj/omrisegev1/results/domfreq_variants_v1
python3 -B $D/code/prmscore_dep.py $D || exit 3
echo ALL_DONE
