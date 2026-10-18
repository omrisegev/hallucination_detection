#!/bin/bash
# CPU-only job in the NGC container with the offline wheelhouse (cluster/install_offline_tf_thinkfix.sh from --chdir).
# Arguments = the command to run. Pass --export=ALL,WORKDIR=<code dir with cluster/install_offline_tf_thinkfix.sh>
# (and the same dir as --chdir / --container-workdir).
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
set -uo pipefail
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export HF_HOME=$SHARED/hf_cache HF_DATASETS_CACHE=$SHARED/hf_cache/datasets
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
mkdir -p "$PYTHONUSERBASE"
cd "${WORKDIR:?pass --export=ALL,WORKDIR=<code dir>}"
echo "[sbatch] job $SLURM_JOB_ID on $(hostname) in $(pwd), args: $*"
source cluster/install_offline_tf_thinkfix.sh
exec "$@"
