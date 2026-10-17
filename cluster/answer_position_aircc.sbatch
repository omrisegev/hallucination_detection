#!/bin/bash
# Dedicated, portable copy; never overwrites shared code or other jobs.
# Pass one argument: absolute isolated bundle directory containing code/source.
# Live allocation verified 2026-09-13; override only after rediscovery.
#SBATCH --job-name=answer_position
#SBATCH --account=cycle3_tau_averbuch_prj
#SBATCH --partition=power-gpu
#SBATCH --qos=owner_940
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1

set -euo pipefail
task_bundle=${1:?absolute bundle root required}
cd "$task_bundle/code"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Verify the copied bytes before importing project code. No package upgrades.
python - "$task_bundle" <<'PY'
import hashlib, json, pathlib, platform, sys
root=pathlib.Path(sys.argv[1])
manifest=json.loads((root/'BUNDLE_MANIFEST.json').read_text())
for item in manifest['files']:
    path=root/item['path']
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsafe bundle member')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''): h.update(block)
    assert h.hexdigest()==item['sha256'], item['path']
import numpy, scipy, sklearn, torch
review=dict(status='PASS',files=len(manifest['files']),python=sys.version,
            platform=platform.platform(),numpy=numpy.__version__,scipy=scipy.__version__,
            sklearn=sklearn.__version__,torch=torch.__version__)
(root/'TRANSFER_REVIEW.json').write_text(json.dumps(review,indent=2))
print(review,flush=True)
PY
python -B scripts/test_conditional_iu_preparation.py
python -B scripts/test_aircc_supervisor.py
exec python -B scripts/complete_answer_position_fusion.py --source-root "$task_bundle/source"
