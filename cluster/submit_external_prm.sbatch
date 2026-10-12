#!/bin/bash
# Qwen2.5-Math-PRM-7B on an external answers.json (cluster/run_external_prm.py).
# The PRM checkpoint is already in $SHARED/hf_cache, so the job runs offline (no HF token).
#
#   sbatch -A cycle3_tau_averbuch_prj -p power-gpu --qos=owner_940 cluster/submit_external_prm.sbatch \
#       --answers $SHARED/data/lsml_external_20260924/socratic/answers.json \
#       --out $SHARED/results/external_prm_v1/socratic
#
#SBATCH --job-name=external_prm
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code_external_prm_v1

set -uo pipefail
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export HF_HOME=$SHARED/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_CACHE_DIR=$SHARED/pip_cache
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
mkdir -p "$PYTHONUSERBASE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "[sbatch] job $SLURM_JOB_ID on $(hostname), args: $*"
pip install -q --user -r cluster/requirements.txt
exec python cluster/run_external_prm.py "$@"
