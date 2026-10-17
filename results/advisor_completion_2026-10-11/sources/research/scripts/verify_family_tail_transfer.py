"""Full-population replay of the portable F15 port; reads no external benchmark."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'LOKY_MAX_CPU_COUNT'):
    os.environ[key] = '1'
import argparse
import json
import runpy
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.family_tail_transfer import (load_lock, build_representations, fit_family_tail,
    answer_standardize, tail_marks, score_locked, LOCK_SHA256)
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-run', type=Path, required=True,
                    help='Original family_tail_calfix_v1/run_20260924_1542 directory containing SCORES.npz')
    ap.add_argument('--pool', type=Path, required=True)
    ap.add_argument('--names', type=Path, required=True)
    ap.add_argument('--answers', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('will not overwrite replay evidence')
    tests = runpy.run_path(str(ROOT/'tests/test_family_tail_transfer.py'))
    passed = []
    for name, fn in tests.items():
        if name.startswith('test_'):
            fn(); passed.append(name)
    snapshot = ROOT/'results/family_tail_transfer_v1/source_snapshot'
    manifest = json.loads((snapshot/'INPUT_MANIFEST.json').read_text())
    assert file_hash(args.pool) == manifest['pool_z']['sha256']
    assert file_hash(args.names) == manifest['pool_names']
    assert file_hash(args.answers) == manifest['oof_answers']['sha256']
    assert file_hash(args.source_run/'MODELS.jsonl') == file_hash(snapshot/'MODELS.jsonl')
    # Trusted local upstream artifact: answer_uid was saved as an object array.
    ref = np.load(args.source_run/'SCORES.npz', allow_pickle=True)
    off, folds = ref['offsets'], ref['answer_fold']
    ans = pd.read_csv(args.answers, usecols=['uid', 'fold'])
    np.testing.assert_array_equal(ref['answer_uid'], ans.uid)
    np.testing.assert_array_equal(folds, ans.fold)
    assert len(folds) == 13769 and off[-1] == 145597
    pool = np.load(args.pool); names = json.loads(args.names.read_text())
    v, columns = build_representations(pool, names, off)['F15']
    # Compare the port's markers byte-for-byte with the archived upstream code.
    upstream = runpy.run_path(str(snapshot/'calfix_common.py'))
    old_marks, _ = upstream['tail_marks'](v, off, .2, tie_aware=True)
    np.testing.assert_array_equal(tail_marks(v, off), old_marks)
    models = [json.loads(s) for s in (snapshot/'MODELS.jsonl').read_text().splitlines()]
    sf = np.repeat(folds, np.diff(off))
    checks = []
    original = {role: ref[f'{role}__F15_tailtie_lsml'] for role in ('cal', 'eval')}
    for k in range(5):
        mask = (folds != k) & (folds != (k+1)%5)
        fit = fit_family_tail(v, off, mask)
        model = next(m for m in models if m['method'] == 'F15_tailtie_lsml' and m['outer_fold'] == k)
        w = np.array(fit['weights'])
        weight_error = float(np.max(np.abs(w-[model['weights'][c] for c in columns])))
        assert weight_error < 1e-9  # Upstream log rounds weights to10 decimal places.
        assert fit['groups'] == [model['groups'][c] for c in columns]
        values = answer_standardize(v@w, off, final=True)
        errs = {role: float(np.max(np.abs(values[sf == f]-original[role][sf == f])))
                for role, f in [('eval', k), ('cal', (k+1)%5)]}
        assert max(errs.values()) < 1e-10
        checks.append({'fold': k, 'K': fit['K'], 'weights_max_error_rounded_reference': weight_error,
                       'scores_max_error': errs})
        print('PASS source fold', k, errs, flush=True)
    deployed = fit_family_tail(v, off, folds < 4)
    locked = load_lock()['deployment']['F15_tailtie_lsml']
    error = float(np.max(np.abs(np.array(deployed['weights'])-[locked['weights'][c] for c in columns])))
    assert error < 1e-12
    predictions = score_locked(pool, names, off)
    values = predictions['F15_tailtie_lsml']['scores']
    threshold = float(np.quantile(values[sf == 4], .8))
    assert abs(threshold-locked['q80_threshold_fold4']) < 1e-12
    api_error = float(np.max(np.abs(values-answer_standardize(v@np.array(deployed['weights']), off, final=True))))
    assert api_error < 1e-12
    atomic_json(args.output, {'status': 'PASS', 'answers': len(folds), 'steps': int(off[-1]),
       'checks': checks, 'unit_tests': passed, 'tail_marks_exact_replay': True,
       'deployment_weight_max_error': error, 'deployment_score_max_error': api_error,
       'deployment_threshold': threshold, 'threshold_max_error': abs(threshold-locked['q80_threshold_fold4']),
       'lock_sha256': LOCK_SHA256,
       'code_sha256': file_hash(ROOT/'spectral_utils/family_tail_transfer.py'),
       'inputs': {str(p): file_hash(p) for p in (args.pool, args.names, args.answers, args.source_run/'SCORES.npz')},
       'scope': 'source fusion/fit parity only; raw telemetry feature extraction parity remains pending'})
    print('FULL SOURCE REPLAY PASS', flush=True)


if __name__ == '__main__':
    main()
