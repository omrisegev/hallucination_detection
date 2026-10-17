#!/bin/bash
# PRMBench step voters (cluster/run_prmbench_step_voters.py): uPRM LLM-as-a-Judge control
# (Qwen3-8B, Qwen2.5-14B-Instruct) and the ProcessBench critic (Qwen2.5-72B-Instruct) on PRMBench.
#
# Runs from an ISOLATED code dir (default $SHARED/code_prm_voters_v1), never $SHARED/code, so
# syncing this branch cannot overwrite code another session's jobs resume from. Sync with:
#   tar czf - --exclude=.git --exclude=results --exclude=cache --exclude=dataset_cache \
#       --exclude=local_cache --exclude=.worktrees --exclude='*.pkl' --exclude='*.pdf' . \
#     | ssh aircc "mkdir -p $SHARED/code_prm_voters_v1 && tar xzf - -C $SHARED/code_prm_voters_v1"
#
# No HF token: all three models are ungated, and the job runs offline from $SHARED/hf_cache
# (prefetch first if a model is missing; the offline flags make a miss fail loudly instead of
# downloading inside a GPU job).
#
#   sbatch -p power-gpu --qos=owner_880 cluster/submit_prmbench_step_voters.sbatch \
#       --kind uprm_judge --model Qwen/Qwen3-8B --n-samples 30 \
#       --out $SHARED/results/prmbench_voters_v1/uprm_qwen3_8b_pilot
#
# The 72B critic does not fit one wall at full N: shard it (--n-shards K --shard i, one job per
# shard) and chain each shard LINEARLY with --dependency=afterany (exit 85 is not auto-requeued).
#
#SBATCH --job-name=prmb_voters
#SBATCH --gpus=1
#SBATCH --time=08:00:00
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
cd "${CODE_DIR:-$SHARED/code_prm_voters_v1}"
echo "[sbatch] job $SLURM_JOB_ID on $(hostname) in $(pwd), args: $*"
pip install -q --user -r cluster/requirements.txt
exec python cluster/run_prmbench_step_voters.py "$@"
