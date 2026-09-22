#!/bin/bash
# One cell-complete Phase-2 shard per array task.  Submit exactly array 0-17:
# A cells are 0-8 and independently repeated B cells are 9-17.
# Partition/QoS and --array=0-17%<concurrency> are passed at submission time.
#SBATCH --job-name=tokb3fit
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%A_%a.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code

set -euo pipefail

: "${PHASE2_EXPECTED_COMMIT:?export the exact pushed commit before submission}"
: "${PHASE2_LOCALIZATION_RELEASE:?export the shared prepared release path}"
: "${PHASE2_OUTPUT_ROOT:?export a new shared result root}"

EXPECTED_INPUT_MANIFEST_SHA256=a0785ca055346c1148f9b5ba12a32803a86b761cfd0d8381893a4b71ddac0eac
INPUT_MANIFEST="$PHASE2_LOCALIZATION_RELEASE/build_A/localization/inputs/MANIFEST.json"
if [[ ! -f "$INPUT_MANIFEST" ]]; then
    echo "missing prepared localization manifest: $INPUT_MANIFEST" >&2
    exit 2
fi
printf '%s  %s\n' "$EXPECTED_INPUT_MANIFEST_SHA256" "$INPUT_MANIFEST" | sha256sum -c -

python - "$PHASE2_EXPECTED_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

expected = sys.argv[1]
stamp = json.loads(Path("SYNC_COMMIT.json").read_text())
if stamp.get("commit") != expected or stamp.get("dirty") is not False:
    raise SystemExit(f"cluster source stamp mismatch: {stamp!r}")
PY

CELLS=(
  processbench_gsm8k_qwen3_4b
  processbench_math_qwen3_4b
  processbench_olympiadbench_qwen3_4b
  processbench_omnimath_qwen3_4b
  processbench_gsm8k_qwen3_8b
  processbench_math_qwen3_8b
  processbench_olympiadbench_qwen3_8b
  processbench_omnimath_qwen3_8b
  prmbench_response_qwen3_8b
)

TASK_INDEX=${SLURM_ARRAY_TASK_ID:?submit as an array task}
if (( TASK_INDEX < 0 || TASK_INDEX >= 18 )); then
    echo "Phase-2 array index must be in 0..17, got $TASK_INDEX" >&2
    exit 2
fi
if (( TASK_INDEX < 9 )); then
    RUN_LABEL=A
    CELL_INDEX=$TASK_INDEX
else
    RUN_LABEL=B
    CELL_INDEX=$((TASK_INDEX - 9))
fi
CELL_ID=${CELLS[$CELL_INDEX]}
OUT_DIR="$PHASE2_OUTPUT_ROOT/fit_${RUN_LABEL}"
TARGET_DIR="$OUT_DIR/cells/$CELL_ID"
if [[ -e "$TARGET_DIR" ]]; then
    echo "refusing to overwrite existing cell shard: $TARGET_DIR" >&2
    exit 3
fi

export PYTHONUSERBASE=/tmp/tokb3fit_pyuserbase_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export MPLCONFIGDIR=/tmp/tokb3fit_matplotlib_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK}
mkdir -p "$PYTHONUSERBASE" "$MPLCONFIGDIR"
pip install -q --user --no-deps -e .

WORKERS=${PHASE2_FIT_WORKERS:-24}
echo "fit label=$RUN_LABEL cell=$CELL_ID workers=$WORKERS commit=$PHASE2_EXPECTED_COMMIT"
exec python scripts/reconstruction_benchmark/run_token_local_temporal_innovation_b3.py \
    --localization-release "$PHASE2_LOCALIZATION_RELEASE" \
    --out-dir "$OUT_DIR" \
    --cell "$CELL_ID" \
    --workers "$WORKERS"
