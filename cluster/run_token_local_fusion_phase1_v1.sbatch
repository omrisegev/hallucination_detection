#!/bin/bash
# Re-freeze the equal-global/LOCAL_IU29 Phase-1 baseline under the exact same
# source/environment generation used by the Phase-2 evaluator.
#SBATCH --job-name=tokp1freeze
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code

set -euo pipefail

: "${PHASE2_EXPECTED_COMMIT:?export the exact pushed commit before submission}"
: "${PHASE2_LOCALIZATION_RELEASE:?export the shared prepared release path}"
: "${PHASE2_OUTPUT_ROOT:?export the shared result root}"

EXPECTED_INPUT_MANIFEST_SHA256=a0785ca055346c1148f9b5ba12a32803a86b761cfd0d8381893a4b71ddac0eac
INPUT_MANIFEST="$PHASE2_LOCALIZATION_RELEASE/build_A/localization/inputs/MANIFEST.json"
if [[ ! -f "$INPUT_MANIFEST" ]]; then
    echo "missing prepared localization manifest: $INPUT_MANIFEST" >&2
    exit 2
fi
printf '%s  %s\n' "$EXPECTED_INPUT_MANIFEST_SHA256" "$INPUT_MANIFEST" | sha256sum -c -

export PYTHONUSERBASE=/tmp/tokp1_pyuserbase_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK}
mkdir -p "$PYTHONUSERBASE"

python - "$PHASE2_EXPECTED_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

expected = sys.argv[1]
stamp = json.loads(Path("SYNC_COMMIT.json").read_text())
if stamp.get("commit") != expected or stamp.get("dirty") is not False:
    raise SystemExit(f"cluster source stamp mismatch: {stamp!r}")
PY

pip install -q --user --no-deps -e .

for RUN_LABEL in A B; do
    OUT_DIR="$PHASE2_OUTPUT_ROOT/phase1_fit_${RUN_LABEL}"
    if [[ -e "$OUT_DIR" ]]; then
        echo "refusing to overwrite existing Phase-1 freeze: $OUT_DIR" >&2
        exit 3
    fi
    python scripts/reconstruction_benchmark/run_token_local_fusion_phase1.py \
        --localization-release "$PHASE2_LOCALIZATION_RELEASE" \
        --out-dir "$OUT_DIR"
done

diff -qr "$PHASE2_OUTPUT_ROOT/phase1_fit_A" "$PHASE2_OUTPUT_ROOT/phase1_fit_B"
sha256sum \
    "$PHASE2_OUTPUT_ROOT/phase1_fit_A/SCORE_FREEZE_MANIFEST.json" \
    "$PHASE2_OUTPUT_ROOT/phase1_fit_B/SCORE_FREEZE_MANIFEST.json"
echo "Phase-1 baseline A/B score trees are byte-identical"
