"""Entrypoint; all research helpers live in spectral_utils/monday_fusion*.py."""
from pathlib import Path
import os
import sys
import types

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key, '2')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
ROOT = Path(__file__).resolve().parents[2]
# Avoid the package's optional model/GPU imports for this CPU-only program.
pkg = types.ModuleType('spectral_utils')
pkg.__path__ = [str(ROOT / 'spectral_utils')]
sys.modules['spectral_utils'] = pkg
sys.path.insert(0, str(ROOT))
from spectral_utils.monday_fusion_gates import main

if __name__ == '__main__':
    main()
