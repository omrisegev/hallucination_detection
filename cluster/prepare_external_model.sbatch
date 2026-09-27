#!/bin/bash
# CPU-only preparation: no model scoring and no downloads on the login node.
#SBATCH --job-name=external_model_prepare
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
set -euo pipefail
S=/shared/cycle2_tau_averbuch_prj/omrisegev1
export HF_HOME=$S/hf_cache
export HF_HUB_DISABLE_XET=1
export TMPDIR=/tmp/external_prepare_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
python3 - <<'PY'
import os, socket, sys, zipfile
from pathlib import Path
# CPU hosts lack ensurepip. Extract the audited wheels into a job-local import
# path; no system package changes, pip network calls or privileged installation.
deps = Path(os.environ['TMPDIR']) / 'deps'
deps.mkdir()
for wheel in Path('/shared/cycle2_tau_averbuch_prj/omrisegev1/wheelhouse/external_20260924').glob('*.whl'):
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(deps)
sys.path.insert(0, str(deps))
from huggingface_hub import snapshot_download
print('public_host_resolves', bool(socket.getaddrinfo('huggingface.co', 443)), flush=True)
path = snapshot_download('Qwen/QwQ-32B', revision='976055f8c83f394f35dbd3ab09a285a984907bd0',
                         max_workers=4, allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
print('PREPARE_COMPLETE', path, flush=True)
PY
