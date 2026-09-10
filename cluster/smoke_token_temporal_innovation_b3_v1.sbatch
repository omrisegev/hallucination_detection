#!/bin/bash
# AIRCC mechanical gate for TOKEN_LOCAL_TEMPORAL_INNOVATION_B3_V1.
# Submit partition/QoS on the sbatch command line.
#SBATCH --job-name=tokb3smoke
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
# 96G, not 32G: pyxis unsquashfs of pytorch:25.01-py3 needs tens of GB of
# page cache inside the memory cgroup on a cold node; job 232712 died OOM
# during container extraction before any test ran.
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
#SBATCH --container-workdir=/shared/cycle2_tau_averbuch_prj/omrisegev1/code

set -euo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
: "${PHASE2_EXPECTED_COMMIT:?export the exact pushed commit before submission}"

export PYTHONUSERBASE=/tmp/tokb3smoke_pyuserbase_${SLURM_JOB_ID}
export MPLCONFIGDIR=/tmp/tokb3smoke_matplotlib_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=${SLURM_CPUS_PER_TASK}
mkdir -p "$PYTHONUSERBASE" "$MPLCONFIGDIR"

python - "$PHASE2_EXPECTED_COMMIT" <<'PY'
import json
from pathlib import Path
import sys

expected = sys.argv[1]
stamp = json.loads(Path("SYNC_COMMIT.json").read_text())
if stamp.get("commit") != expected or stamp.get("dirty") is not False:
    raise SystemExit(f"cluster source stamp mismatch: {stamp!r}")
print(f"clean source stamp verified: {expected}")
PY

pip install -q --user --no-deps -e .
python - <<'PY'
import numpy, scipy, sklearn, torch, spectral_utils
print(
    "runtime imports ok",
    numpy.__version__, scipy.__version__, sklearn.__version__, torch.__version__,
)
PY

python scripts/test_token_temporal_innovation_b3.py
python scripts/test_token_local_fusion.py
