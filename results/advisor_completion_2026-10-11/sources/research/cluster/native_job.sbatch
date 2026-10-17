#!/bin/bash
# CPU-only job for thinkfix_rescore_v1 WITHOUT a container: the CPU nodes cannot start the NGC
# image (pyxis 'Failed to invoke spank plugin stack', 2026-10-10/11), so this uses system
# python3.12 plus the pinned wheels unpacked in pyenv_native_312_20261011 (same versions as the
# local frozen runs: numpy 2.2.4, scipy 1.17.1, pandas 3.0.3, scikit-learn 1.8.0, torch 2.6.0 cpu).
# Usage: sbatch -p cpu --chdir=<dir> native_job.sbatch <script.py> [args...]
#SBATCH --job-name=thinkfix_native
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
set -uo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/pyenv_native_312_20261011/activate.sh
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2} MPLBACKEND=Agg
cd "$SLURM_SUBMIT_DIR"
echo "[sbatch] job $SLURM_JOB_ID on $(hostname) in $(pwd), args: $*"
python3 -c "import numpy, scipy, pandas, sklearn, torch; print('[env]', numpy.__version__, scipy.__version__, pandas.__version__, sklearn.__version__, torch.__version__)"
exec python3 -B "$@"
