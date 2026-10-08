"""CPU-only CT7 fixed-profile comparison launcher."""
import os
import sys
from pathlib import Path

for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.experiments.cvf_v2.ct7 import main

if __name__ == '__main__':
    main()
