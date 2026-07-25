#!/bin/bash
# Generic CPU-side analysis job on AIRCC: runs any repo script inside the NGC container.
#
# Why this exists: the raw cells under $SHARED/results are multi-GB, and the analyses that
# read them (featcache build, cap audits, schema inspection) are CPU-only. Running them on
# the cluster and fetching only the small outputs keeps 20+ GB off the VPN. The login node
# is not an option — it has no numpy, and parsing GBs of pkl there is antisocial anyway.
#
# Submit (first arg = repo-relative script, rest = its args):
#   sbatch -p power-gpu --qos=owner_880 cluster/cpu_job.sbatch \
#       scripts/build_repgrid_featcache.py --cache-dir ... --out-dir ...
#   sbatch -p power-gpu --qos=owner_880 cluster/cpu_job.sbatch \
#       scripts/cap_pilot_audit.py /shared/.../results/pilot_cap/foo_pilot
#
# NOTE: --gpus=1 is requested even though nothing here needs a GPU. power-gpu is the only
# up partition (compute-gpu is down), and the Pyxis container path is proven under it.
#
#SBATCH --job-name=cpujob
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code
# No --container-name, for the same enroot-rootfs race reason as submit_inference.sbatch.

set -uo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
export PIP_CACHE_DIR=$SHARED/pip_cache
# Per-job node-local pip target — concurrent jobs must not share one (see
# submit_inference.sbatch for the corruption this prevents).
export PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID}
mkdir -p "$PYTHONUSERBASE"

echo "[sbatch] job $SLURM_JOB_ID on $(hostname), args: $*"

pip install -q --user -r cluster/requirements.txt

exec python "$@"
