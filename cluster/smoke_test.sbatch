#!/bin/bash
# Sandbox smoke test: docker + B200 + spectral_utils inside the NGC container.
# Submit:   sbatch cluster/smoke_test.sbatch
# Owner-queue variant (Stage 2): sbatch -p <partition> --qos=<qos> cluster/smoke_test.sbatch
#
#SBATCH --job-name=smoke_test
#SBATCH --partition=sandbox
#SBATCH --qos=sandbox_owner_880
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out

set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
CODE_DIR=$SHARED/code
IMAGE=nvcr.io/nvidia/pytorch:25.01-py3

echo "[smoke] job $SLURM_JOB_ID on $(hostname)"
nvidia-smi || echo "[smoke] nvidia-smi not available on host"

systemctl --user start docker || true
export DOCKER_HOST=unix:///run/user/$UID/docker.sock
docker info >/dev/null || { echo "[smoke] FATAL: docker daemon not reachable"; exit 1; }

docker run --rm --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$SHARED":"$SHARED" -w "$CODE_DIR" \
  -e HF_HOME="$SHARED/hf_cache" \
  -e PIP_CACHE_DIR="$SHARED/pip_cache" \
  -e PYTHONUNBUFFERED=1 \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" \
  -e SMOKE_OUT="$SHARED/results" \
  "$IMAGE" \
  bash -c "pip install -q -r cluster/requirements.txt && python cluster/smoke_test.py"
