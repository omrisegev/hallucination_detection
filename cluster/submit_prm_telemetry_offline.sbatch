#!/bin/bash
# PRM rewards + backbone telemetry (cluster/run_prmbench_prm_telemetry.py), offline wheelhouse; --answers for external benchmarks.
# Both checkpoints and the PRMBench dataset are already in $SHARED/hf_cache, so the job runs offline.
#
#   sbatch -A cycle3_tau_averbuch_prj -p power-gpu --qos=owner_940 cluster/submit_prmbench_prm_telemetry.sbatch \
#       --mode prm --out $SHARED/results/prmbench_prm7b_telemetry
#
#SBATCH --job-name=prm_tel_offline
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1

set -uo pipefail
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export HF_HOME=$SHARED/hf_cache
export HF_DATASETS_CACHE=$SHARED/hf_cache/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_CACHE_DIR=$SHARED/pip_cache
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
mkdir -p "$PYTHONUSERBASE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${CODE_DIR:-$SHARED/code_prm_boost_v1}"
echo "[sbatch] job $SLURM_JOB_ID on $(hostname) in $(pwd), args: $*"
WHEELS=$SHARED/wheelhouse/tf_thinkfix_20261010
python -m pip install -q --user --no-deps "$WHEELS"/*.whl   # compute nodes are offline (2026-10-16)
exec python cluster/run_prmbench_prm_telemetry.py "$@"
