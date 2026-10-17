"""Standalone, two-thread LR checkpoint helper; never starts the workflow."""
from pathlib import Path
import os
import sys
import types

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
CODE = Path(__file__).resolve().parents[2]
pkg = types.ModuleType('spectral_utils')
pkg.__path__ = [str(CODE / 'spectral_utils')]
sys.modules['spectral_utils'] = pkg
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / 'scripts/experiments'))
from spectral_utils.monday_lr_helper import main

if __name__ == '__main__':
    main()
