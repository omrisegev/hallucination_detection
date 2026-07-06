#!/bin/bash
# One-time prefetch on a compute node (login node has no GPU/docker):
# pulls the ~10 GB NGC image into the node's docker cache and downloads the
# model into HF_HOME on /shared so real jobs start warm.
# Submit:   sbatch cluster/prefetch.sbatch [model_id]
#
#SBATCH --job-name=prefetch
#SBATCH --partition=sandbox
#SBATCH --qos=sandbox_owner_880
#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out

set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
MODEL_ID=${1:-Qwen/Qwen2.5-Math-1.5B-Instruct}

systemctl --user start docker || true
export DOCKER_HOST=unix:///run/user/$UID/docker.sock
docker info >/dev/null || { echo "[prefetch] FATAL: docker daemon not reachable"; exit 1; }

echo "[prefetch] pulling $IMAGE (node-local cache)"
docker pull "$IMAGE"

echo "[prefetch] downloading $MODEL_ID into $SHARED/hf_cache"
docker run --rm \
  -v "$SHARED":"$SHARED" \
  -e HF_HOME="$SHARED/hf_cache" \
  -e PIP_CACHE_DIR="$SHARED/pip_cache" \
  "$IMAGE" \
  bash -c "pip install -q huggingface_hub && python -c \"
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL_ID')
print('model at', p)
\""
echo "[prefetch] done"
