#!/bin/bash
# CPU-only analysis job for the 2026-10-10 think-fix re-scoring (no GPU requested).
# Pass --chdir=<immutable code snapshot>. First arg = repo-relative script, rest = its args.
#SBATCH --job-name=thinkfix_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
set -uo pipefail
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export HF_HOME=$SHARED/hf_cache HF_DATASETS_CACHE=$SHARED/hf_cache/datasets
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
mkdir -p "$PYTHONUSERBASE"
cd "$SLURM_SUBMIT_DIR"
echo "[sbatch] job $SLURM_JOB_ID on $(hostname) in $(pwd), args: $*"
source cluster/install_offline_tf_thinkfix.sh
exec python -B "$@"
