#!/bin/bash
# One existing timing example, no feature fitting or benchmark quality.
#SBATCH --job-name=external_alignment
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:10:00
#SBATCH --no-requeue
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
set -euo pipefail
export HF_HOME=/shared/cycle2_tau_averbuch_prj/omrisegev1/hf_cache
export PYTHONUSERBASE=/tmp/external_diag_${SLURM_JOB_ID}
export TMPDIR=/tmp/external_diag_tmp_${SLURM_JOB_ID}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$PYTHONUSERBASE" "$TMPDIR"
python -m pip install -q --user --no-deps --no-index \
  --find-links=/shared/cycle2_tau_averbuch_prj/omrisegev1/wheelhouse/external_20260924 \
  'transformers==4.51.3' 'accelerate==1.6.0' 'tokenizers==0.21.4' 'huggingface_hub==0.34.4'
exec python -B cluster/diagnose_external_alignment.py
