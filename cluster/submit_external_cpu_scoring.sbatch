#!/bin/bash
#SBATCH --job-name=external_cpu_score
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --no-requeue
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%A_%a.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
set -euo pipefail
export TMPDIR=/tmp/external_cpu_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR"
exec python -B cluster/run_external_cpu_scoring.py --shard "${SLURM_ARRAY_TASK_ID:-0}" "$@"
