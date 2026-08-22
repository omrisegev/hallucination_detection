#!/bin/bash
#SBATCH --job-name=deemcmp24
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
export PYTHONUSERBASE=/tmp/deemcmp_pyuserbase_${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/deemcmp_matplotlib_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$PYTHONUSERBASE" "$MPLCONFIGDIR"

# Compute nodes have no DNS. These pinned pure-Python wheels are staged once on
# the login node; never let an offline pip failure surface halfway through B1/B2.
pip install -q --user --no-index --find-links="$SHARED/wheelhouse" "deem==0.2.0" "entmax==1.3"
pip install -q --user --no-deps -e .
python - <<'PY'
import importlib.metadata
import deem, entmax, spectral_utils
assert importlib.metadata.version("deem") == "0.2.0"
assert importlib.metadata.version("entmax") == "1.3"
print("pinned adapter dependencies verified")
PY
exec python cluster/run_deem_vs_iupcr_24cell_v1.py "$@"
