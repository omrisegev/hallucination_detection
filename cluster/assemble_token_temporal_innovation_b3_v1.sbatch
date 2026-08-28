#!/bin/bash
# Assemble and compare complete Phase-2 A/B cell shards.  Submit with an
# afterok dependency on the 18-task fit array.
#SBATCH --job-name=tokb3freeze
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code

set -euo pipefail

: "${PHASE2_EXPECTED_COMMIT:?export the exact pushed commit before submission}"
: "${PHASE2_LOCALIZATION_RELEASE:?export the shared prepared release path}"
: "${PHASE2_OUTPUT_ROOT:?export the shared result root used by the fit array}"

export PYTHONUSERBASE=/tmp/tokb3freeze_pyuserbase_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=32
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
    python scripts/reconstruction_benchmark/run_token_local_temporal_innovation_b3.py \
        --localization-release "$PHASE2_LOCALIZATION_RELEASE" \
        --out-dir "$PHASE2_OUTPUT_ROOT/fit_${RUN_LABEL}" \
        --assemble
done

diff -qr "$PHASE2_OUTPUT_ROOT/fit_A" "$PHASE2_OUTPUT_ROOT/fit_B"
sha256sum \
    "$PHASE2_OUTPUT_ROOT/fit_A/SCORE_FREEZE_MANIFEST.json" \
    "$PHASE2_OUTPUT_ROOT/fit_B/SCORE_FREEZE_MANIFEST.json"
echo "Phase-2 A/B score trees are byte-identical"
