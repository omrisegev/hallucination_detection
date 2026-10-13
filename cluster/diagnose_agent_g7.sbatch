#!/bin/bash
# One G7 diagnostic job (records, never raises). Submit with matching --chdir and --container-workdir:
#   sbatch --chdir=$SNAP --container-workdir=$SNAP cluster/diagnose_agent_g7.sbatch --inputs ... --model-path ... --out ...
#SBATCH --job-name=agent_g7_diag
#SBATCH --account=cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --partition=power-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
set -euo pipefail
TASK_SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
case "$PWD" in "$TASK_SHARED"/code_agent_g7_diag_*) ;; *) echo "Requires code_agent_g7_diag_<sha> workdir, got $PWD" >&2; exit 2;; esac
export HF_HOME="$TASK_SHARED/hf_cache"
export PYTHONUSERBASE="/tmp/agent_py_${SLURM_JOB_ID}"
export TMPDIR="/tmp/agent_tmp_${SLURM_JOB_ID}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$PYTHONUSERBASE" "$TMPDIR"
source cluster/install_offline_tf_thinkfix.sh
exec python -B cluster/diagnose_agent_g7.py "$@"
