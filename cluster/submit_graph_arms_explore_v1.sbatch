#!/bin/bash
#SBATCH --job-name=gexplore24
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code

# EXPLORATORY graph-arm run (G2/G3/G4 on the real 24 cells), commissioned by
# Omri after the registered deem_vs_iupcr_24cell_v1 decision.  Outside every
# frozen protocol; see the docstring of scripts/explore_graph_arms_24cell_v1.py.
# CPU-only float64 fits; the GPU request exists only because the allocation
# schedules GPU jobs.  Per-fit atomic saves + skip-complete make requeue after
# preemption a clean resume.

set -euo pipefail
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
RUN=$SHARED/results/deem_vs_iupcr_24cell_v1
OUT=$SHARED/results/graph_arms_exploratory_v1
export PIP_CACHE_DIR=$SHARED/pip_cache
export PYTHONUSERBASE=/tmp/gexplore_pyuserbase_${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/gexplore_matplotlib_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p "$PYTHONUSERBASE" "$MPLCONFIGDIR" "$OUT"

pip install -q --user --no-deps -e .

# Stage 1: label-free fits (no sidecar path is ever passed to this process).
python scripts/explore_graph_arms_24cell_v1.py fit \
    --registry configs/residual_graph_deem_24cell_v1_registry.json \
    --bundle-dir "$RUN/data/bundles" \
    --out-dir "$OUT" \
    --workers 24

# Stage 2: separate process opens the v1 label sidecars and scores everything
# against the frozen B0/B3 reference.
python scripts/explore_graph_arms_24cell_v1.py evaluate \
    --registry configs/residual_graph_deem_24cell_v1_registry.json \
    --bundle-dir "$RUN/data/bundles" \
    --sidecar-dir "$RUN/evaluation/label_sidecars" \
    --frozen-stage-a "$RUN/stage_a" \
    --out-dir "$OUT"

echo "gexplore24 complete"
