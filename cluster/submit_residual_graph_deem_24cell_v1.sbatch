#!/bin/bash
# Frozen AIRCC launcher. Submit partition/QoS on the sbatch command line.
# Long stages are submitted as afterany chains; cell/seed checkpoints make every
# repeated invocation idempotent.
#SBATCH --job-name=rgdeem24
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code

set -euo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export PIP_CACHE_DIR=$SHARED/pip_cache
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/matplotlib_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$PYTHONUSERBASE" "$MPLCONFIGDIR"

pip install -q --user -e ".[dependency-experiment]"

exec python cluster/run_residual_graph_deem_24cell_v1.py "$@"
