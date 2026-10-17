"""Replay the frozen A–D specification into a new directory, CPU only."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'results/label_free_error_count_v1_2026-10-12'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_reference(root=ROOT, reference=REFERENCE):
    """Reject drift before any new destination exists."""
    expected = json.loads((reference / 'CODE_FREEZE.json').read_text())
    seal = json.loads((reference / 'PREDICTION_SEAL.json').read_text())
    if expected != seal['code_sha256']:
        raise ValueError('Reference code map disagrees with original prediction seal')
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError('Frozen reference source changed: ' + relative)
    for name in ['AMENDMENTS.json', 'INPUT_FREEZE.json']:
        if sha(reference / name) != seal['files'][name]['sha256']:
            raise ValueError('Frozen reference identity changed: ' + name)
    frozen = json.loads((reference / 'INPUT_FREEZE.json').read_text())
    current = {name: importlib.metadata.version(name) for name in frozen['versions']}
    if frozen['python'] != sys.version or frozen['executable'] != sys.executable or current != frozen['versions']:
        raise ValueError('Frozen reference runtime changed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--predictions-only', action='store_true')
    args = parser.parse_args()
    destination = args.destination.resolve()
    if not destination.is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Replay destination must be a new directory inside repository results/')
    if destination.exists():
        raise ValueError('Refusing to overwrite or resume an existing delivery; use a fresh destination')
    validate_reference()
    spec = importlib.util.spec_from_file_location('count_replay_driver', ROOT / 'scripts/run_label_free_error_count_v1.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    # The original source remains byte-identical to the archived CODE_FREEZE.
    driver.OUT = destination
    driver.prepare()
    shutil.copyfile(REFERENCE / 'AMENDMENTS.json', destination / 'AMENDMENTS.json')
    driver.dump(destination / 'REPLAY_OF.json', {
        'reference': str(REFERENCE), 'reference_seal_sha256': driver.sha(REFERENCE / 'PREDICTION_SEAL.json'),
        'wrapper_sha256': driver.sha(__file__), 'scope': 'Computational replay of the existing specification, not a new preregistration',
        'amendments': 'Copied final original amendment lineage; original failed attempt remains in reference package',
        'source_driver_sha256': driver.sha(ROOT / 'scripts/run_label_free_error_count_v1.py')})
    driver.historical_baselines()
    driver.predict()
    driver.seal()
    driver.verify()
    if not args.predictions_only:
        from spectral_utils.error_count_evaluation import evaluate_package, build_report
        evaluate_package(destination, driver.FROZEN, 20000, 20261012)
        build_report(destination)
    print('FRESH_REPLAY_COMPLETE', destination, flush=True)


if __name__ == '__main__':
    main()
