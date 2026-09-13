#!/bin/bash
# Isolated code; verified previous remote inputs are reused read-only.
#SBATCH --job-name=conditional_iu
#SBATCH --account=cycle3_tau_averbuch_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=owner_940
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1

set -euo pipefail
task_bundle=${1:?absolute isolated bundle root}
task_source=${2:?absolute existing frozen source root}
task_family=${3:?position or graph_local or graph_tv}
cd "$task_bundle/code"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
python -B scripts/verify_conditional_iu_aircc.py --bundle-root "$task_bundle" --source-root "$task_source" --family "$task_family"
python -B scripts/test_conditional_iu_fusion.py
python -B scripts/test_conditional_iu_graph.py
python -B scripts/test_conditional_iu_driver.py
python -B scripts/test_conditional_iu_supervisor.py
exec python -B scripts/complete_conditional_iu_fusion.py --source-root "$task_source" --family "$task_family"
