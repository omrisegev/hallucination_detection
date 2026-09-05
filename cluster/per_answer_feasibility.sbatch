#!/bin/bash
#SBATCH --job-name=window_feasibility
#SBATCH --partition=power-gpu
#SBATCH --account=cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --signal=B:TERM@60
#SBATCH --requeue

set -euo pipefail

# Three explicit task-specific paths; never reuse Claude's code/output tree.
TASK_ROOT=${1:?task directory required}
CAPSULE=${2:?capsule directory required}
CELLS=${3:?label-free cell bundle directory required}
ROWS=${4:-30}
OUTPUT_NAME=${5:-feasibility_pilot}
[[ "$ROWS" =~ ^[0-9]+$ ]] || { echo 'Rows must be nonnegative' >&2; exit 2; }
[[ "$OUTPUT_NAME" =~ ^[a-zA-Z0-9_]+$ ]] || { echo 'Invalid output name' >&2; exit 2; }
case "$TASK_ROOT" in
  /shared/cycle3_tau_averbuch_prj/omrisegev1/*) ;;
  *) echo 'Task root is outside the verified non-home project directory' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PIP_CACHE_DIR="$TASK_ROOT/pip_cache"

python3 -m venv "$TASK_ROOT/venv"
"$TASK_ROOT/venv/bin/python" -m pip install --disable-pip-version-check numpy==2.2.4 scipy==1.17.1
"$TASK_ROOT/venv/bin/python" -c 'import numpy,scipy; print("NumPy",numpy.__version__,"SciPy",scipy.__version__)'

"$TASK_ROOT/venv/bin/python" -u "$CAPSULE/scripts/per_answer_localization/feasibility.py" \
  --cells-dir "$CELLS" --schema "$CAPSULE/stream_schema.json" \
  --output "$TASK_ROOT/$OUTPUT_NAME" --rows-per-cell "$ROWS" \
  --workers "${SLURM_CPUS_PER_TASK:-8}" &
worker_pid=$!
trap 'kill -TERM "$worker_pid" 2>/dev/null || true; wait "$worker_pid" || true; exit 85' TERM INT
wait "$worker_pid"

# A handled signal/exit85 is a checkpoint, not automatic requeue. A longer
# campaign must submit a LINEAR afterany chain in advance if its wall is insufficient.
