#!/bin/bash
# Timing-only default. No automatic requeue or full-run launch.
# Pass an immutable --chdir snapshot when submitting.
#SBATCH --job-name=external_telemetry
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1

set -euo pipefail
export HF_HOME=/shared/cycle2_tau_averbuch_prj/omrisegev1/hf_cache
export PIP_CACHE_DIR=/shared/cycle2_tau_averbuch_prj/omrisegev1/pip_cache
export PYTHONUSERBASE=/tmp/external_py_${SLURM_JOB_ID}
export TMPDIR=/tmp/external_tmp_${SLURM_JOB_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$PYTHONUSERBASE" "$TMPDIR"
# Offline wheels avoid compute-container DNS; retain NGC torch/numpy/scipy.
WHEELS=/shared/cycle2_tau_averbuch_prj/omrisegev1/wheelhouse/external_20260924
python -m pip install -q --user --no-deps --no-index --find-links="$WHEELS" \
  'transformers==4.51.3' 'accelerate==1.6.0' 'tokenizers==0.21.4' 'huggingface_hub==0.34.4'
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python -B cluster/run_external_telemetry.py "$@"
