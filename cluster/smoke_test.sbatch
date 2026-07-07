#!/bin/bash
# Sandbox smoke test: Pyxis container + B200 + spectral_utils inside the NGC image.
# Submit (sandbox):     sbatch cluster/smoke_test.sbatch
# Submit (owner queue): sbatch -p power-gpu --qos=owner_880 cluster/smoke_test.sbatch
#
#SBATCH --job-name=smoke_test
#SBATCH --partition=sandbox
#SBATCH --qos=sandbox_owner_880
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code
#SBATCH --container-name=ngc_pytorch_2501

set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1

echo "[smoke] job $SLURM_JOB_ID on $(hostname)"

pip install -q -r cluster/requirements.txt

SMOKE_OUT=$SHARED/results python cluster/smoke_test.py
