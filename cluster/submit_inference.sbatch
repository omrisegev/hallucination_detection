#!/bin/bash
# AIRCC batch inference job — wraps cluster/run_inference.py in the B200-compatible
# NGC container with full preemption handling.
#
# Submit (partition/QoS are NOT hardcoded — discover them with `sdata` and pass on
# the command line; trailing args go straight to run_inference.py):
#
#   sbatch -p <owner_partition> --qos=<owner_qos> cluster/submit_inference.sbatch \
#       --dataset aime24 --temps 0.2,0.6,1.0 --k 8 --n-samples 30 --max-new 1024 \
#       --out /shared/cycle2_tau_averbuch_prj/omrisegev1/results/edis_aime24
#
#SBATCH --job-name=spectral_infer
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out

set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
CODE_DIR=$SHARED/code
IMAGE=nvcr.io/nvidia/pytorch:25.01-py3
CONTAINER=job_${SLURM_JOB_ID}

echo "[sbatch] job $SLURM_JOB_ID on $(hostname), args: $*"

# Rootless docker daemon often does not auto-start on a fresh allocation.
systemctl --user start docker || true
export DOCKER_HOST=unix:///run/user/$UID/docker.sock
docker info >/dev/null || { echo "[sbatch] FATAL: docker daemon not reachable"; exit 1; }

docker rm -f "$CONTAINER" 2>/dev/null || true

# `exec python` makes python PID 1 inside the container so the TERM we forward on
# preemption reaches the driver's SIGTERM handler (bash would swallow it otherwise).
docker run -d --name "$CONTAINER" --gpus all \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$SHARED":"$SHARED" -w "$CODE_DIR" \
  -e HF_HOME="$SHARED/hf_cache" \
  -e HF_DATASETS_CACHE="$SHARED/hf_cache/datasets" \
  -e PIP_CACHE_DIR="$SHARED/pip_cache" \
  -e PYTHONUNBUFFERED=1 \
  -e SLURM_JOB_ID="$SLURM_JOB_ID" \
  "$IMAGE" \
  bash -c "pip install -q -r cluster/requirements.txt && exec python cluster/run_inference.py $*"

# Slurm sends TERM to this batch shell 900 s before SIGKILL (--signal=B:TERM@900).
# Forward it into the container; the driver checkpoints and exits 0.
forward_term() {
  echo "[sbatch] TERM received — forwarding to container for checkpoint"
  docker kill -s TERM "$CONTAINER" 2>/dev/null || true
  docker wait "$CONTAINER" >/dev/null 2>&1 || true
  docker rm -f "$CONTAINER" 2>/dev/null || true
  echo "[sbatch] checkpointed; Slurm will requeue"
  exit 0
}
trap forward_term TERM

# Stream container output into the Slurm log; wait in background so the bash
# builtin `wait` (interruptible by the trap, unlike a foreground docker wait)
# is what blocks.
docker logs -f "$CONTAINER" &
docker wait "$CONTAINER" > "$SHARED/logs/exit_${SLURM_JOB_ID}" &
WAITER=$!
wait $WAITER || true

EXIT_CODE=$(cat "$SHARED/logs/exit_${SLURM_JOB_ID}" 2>/dev/null || echo 1)
docker rm -f "$CONTAINER" 2>/dev/null || true
echo "[sbatch] container exit code: $EXIT_CODE"
exit "$EXIT_CODE"
