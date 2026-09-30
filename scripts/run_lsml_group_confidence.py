"""Run only the fixed full-source CPU experiment; no external quality access."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ[key] = '1'
import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.lsml_group_confidence_experiment import run_fit, evaluate
from spectral_utils.external_generalization.artifacts import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['fit', 'evaluate'])
    args = parser.parse_args()
    out = ROOT/'results/lsml_group_confidence_v1'
    if args.stage == 'fit':
        ns = runpy.run_path(str(ROOT/'tests/test_lsml_group_confidence.py'))
        tests = []
        for name, fn in ns.items():
            if name.startswith('test_'):
                fn()
                tests.append({'test': name, 'status': 'PASS'})
        print('mechanism tests PASS', len(tests), flush=True)
        run_fit(ROOT, out, 'python scripts/run_lsml_group_confidence.py fit', tests)
    else:
        evaluate(ROOT, out)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        atomic_json(ROOT/'results/lsml_group_confidence_v1/FAILURE.json',
                    {'type': type(exc).__name__, 'reason': str(exc), 'stage': sys.argv[1:]})
        raise
