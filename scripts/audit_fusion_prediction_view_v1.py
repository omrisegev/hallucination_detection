"""Source-frozen, target-free feasibility audit; does not score localizers."""
import os
for option in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[option] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/fusion_prediction_view_audit_v1'
PARENT = ROOT/'results/fusion_replication_v1'
sys.path.insert(0, str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.answer_localization_v2 import moment_plan, moment_matrix, PRIMITIVES, STREAM_NAMES
from spectral_utils.fusion_context_bank import context_matrix
from spectral_utils.fusion_prediction_view import prediction_views, residual_window_features, augment_bank, KINDS


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(v):
    if isinstance(v, dict): return {str(k): safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)): return [safe(x) for x in v]
    if isinstance(v, (bool, np.bool_)): return bool(v)
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, (float, np.floating)): return float(v) if np.isfinite(v) else None
    return v


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(7):
        try: tmp.replace(path); return
        except PermissionError:
            if attempt == 6: raise
            time.sleep(.025 * 2**attempt)


def tests():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    command = [sys.executable, str(ROOT/'tests/test_fusion_prediction_view.py')]
    result = subprocess.run(command, capture_output=True, timeout=180)
    path = OUT/'TESTS.txt'; path.write_bytes(result.stdout + result.stderr)
    save(OUT/'TEST_EXECUTION.json', {'command': command, 'exit_code': result.returncode,
         'seconds': time.monotonic()-started, 'captured_output_sha256': sha(path),
         'source_sha256': sha(ROOT/'tests/test_fusion_prediction_view.py')})
    print(path.read_text(encoding='utf-8'), flush=True)
    if result.returncode: raise RuntimeError('SCIENTIFIC_TEST_FAILURE')


def prepare():
    if (OUT/'MANIFEST.json').exists(): verify(); print('Manifest verified.'); return
    test = load(OUT/'TEST_EXECUTION.json'); assert test['exit_code'] == 0
    assert sha(OUT/'TESTS.txt') == test['captured_output_sha256']
    assert sha(ROOT/'tests/test_fusion_prediction_view.py') == test['source_sha256']
    m = load(PARENT/'MANIFEST.json'); selected = m['selected']
    frozen = load(PARENT/'SCORES_FROZEN.json')
    assert frozen['manifest_sha256'] == sha(PARENT/'MANIFEST.json')
    paths = [Path(__file__), ROOT/'docs/experiments/FUSION_PREDICTION_VIEW_AUDIT_V1.md',
             ROOT/'tests/test_fusion_prediction_view.py', OUT/'TESTS.txt', OUT/'TEST_EXECUTION.json',
             PARENT/'MANIFEST.json', PARENT/'SCORES_FROZEN.json']
    for rec in selected:
        input_path = PARENT/'inputs'/(rec['uid']+'.npz')
        assert sha(input_path) == rec['input_sha256']
        paths.append(input_path)
        for ending in ('.json', '.npz'):
            path = PARENT/'scores'/(rec['uid']+ending)
            assert sha(path) == frozen['files'][str(path)]
            paths.append(path)
    for name, mod in list(sys.modules.items()):
        if name.startswith('spectral_utils') and getattr(mod, '__file__', None):
            paths.append(Path(mod.__file__).resolve())
    paths += [ROOT/p for p in (
        'spectral_utils/temporal_models.py', 'spectral_utils/token_temporal_innovation_b3.py',
        'spectral_utils/local_online_comprehensive.py', 'spectral_utils/unified_causal_iu.py',
        'spectral_utils/ciw_cross_scale_localization.py',
        'docs/experiments/TOKEN_LOCAL_TEMPORAL_INNOVATION_B3_V1.md',
        'docs/experiments/CIW_CROSS_SCALE_LOCALIZATION_V1.md',
        'results/local_online_comprehensive_v1/REPORT.md',
        'results/ciw_cross_scale_localization_v1/REPORT.md')]
    save(OUT/'MANIFEST.json', {'status': 'TARGET_FREE_FEASIBILITY_ONLY',
        'release_id': m['release_id'], 'selected': selected, 'primitive_names': PRIMITIVES,
        'variants': KINDS, 'banks': ['moment', 'context'], 'execution_cap_seconds': 180,
        'hashes': {str(p): sha(p) for p in paths}, 'labels_decoded': False,
        'localization_heads_fitted': False, 'created_unix': time.time()})
    print('Frozen 110 inputs; one AR(1) supporting view and two prediction controls.', flush=True)


def verify():
    m = load(OUT/'MANIFEST.json')
    for p, h in m['hashes'].items(): assert sha(p) == h, p
    return m


def column_diagnostics(base, extra, indices, names):
    b, e = base[indices], extra[indices]
    def active(a):
        return np.ptp(a, axis=0) > 1e-10 * np.maximum(1., np.max(np.abs(a), axis=0))
    ba, ea = active(b), active(e)
    br = rankdata(b, axis=0); er = rankdata(e, axis=0)
    br -= br.mean(axis=0); er -= er.mean(axis=0)
    denominator = np.sqrt(np.sum(er**2, axis=0)[:, None] * np.sum(br**2, axis=0)[None, :])
    corr = np.full((e.shape[1], b.shape[1]), np.nan)
    np.divide(er.T @ br, denominator, out=corr, where=denominator > 0)
    entropy = names.index('entropy_series__level')
    best = []
    for row in corr:
        j = int(np.nanargmax(np.abs(row))) if np.isfinite(row).any() else None
        best.append({'original_feature': names[j] if j is not None else None,
                     'absolute_spearman': abs(float(row[j])) if j is not None else None})
    def rank(a):
        a = (a-a.mean(axis=0)) / a.std(axis=0)
        return int(np.linalg.matrix_rank(a)) if a.shape[1] else 0
    return {'active_original': int(ba.sum()), 'active_added': int(ea.sum()),
            'n_fit_windows': len(indices), 'original_rank': rank(b[:, ba]),
            'augmented_rank': rank(np.column_stack((b[:, ba], e[:, ea]))),
            'entropy_spearman': corr[:, entropy], 'closest_original': best}


def run():
    m = verify(); digest = sha(OUT/'MANIFEST.json'); started = time.monotonic()
    for i, rec in enumerate(m['selected']):
        uid = rec['uid']; mp = OUT/'answers'/(uid+'.json'); ap = mp.with_suffix('.npz')
        if mp.exists():
            old = load(mp); assert old['manifest_sha256'] == digest and old['arrays_sha256'] == sha(ap)
            continue
        if time.monotonic()-started > m['execution_cap_seconds']:
            print('Paused at cap; checkpoints retained.', flush=True); return
        with np.load(PARENT/'inputs'/(uid+'.npz'), allow_pickle=False) as data: raw = data['raw']
        plan = moment_plan(len(raw), 8)
        x = raw[:, [STREAM_NAMES.index(s) for s in PRIMITIVES]]
        pred = prediction_views(x); arrays = {'mask': pred['mask'], 'slope': pred['slope'],
            'drift': pred['drift'], 'fit_pair_counts': pred['fit_pair_counts'],
            'window_starts': plan.starts, 'window_ends': plan.ends, 'fit_indices': plan.fit_indices}
        details = {'banks': {}, 'prediction_mse': {}, 'labels_accessed': False}
        for kind in KINDS:
            arrays[kind+'__predictions'] = pred['predictions'][kind]
            values, counts = residual_window_features(pred['residuals'][kind], pred['mask'], plan.starts, plan.ends)
            arrays[kind+'__extra'] = values; arrays['residual_counts'] = counts
            details['prediction_mse'][kind] = np.mean(pred['residuals'][kind][17:]**2, axis=0)
        with np.load(PARENT/'scores'/(uid+'.npz'), allow_pickle=False) as source:
            for bank, builder in (('moment', moment_matrix), ('context', context_matrix)):
                base, names = builder(raw, plan)
                np.testing.assert_array_equal(base, source[bank+'__features'])
                for kind in KINDS:
                    extra = arrays[kind+'__extra']
                    augmented, augmented_names = augment_bank(base, names, extra, PRIMITIVES)
                    np.testing.assert_array_equal(augmented[:, :27], base)
                    arrays[bank+'__'+kind+'__features'] = augmented
                    details['banks'][bank+'__'+kind] = {
                        'names': augmented_names, **column_diagnostics(base, extra, plan.fit_indices, names)}
        ap.parent.mkdir(parents=True, exist_ok=True)
        with ap.with_suffix('.npz.tmp').open('wb') as f: np.savez_compressed(f, **arrays)
        ap.with_suffix('.npz.tmp').replace(ap)
        save(mp, {**rec, 'diagnostics': details, 'manifest_sha256': digest, 'arrays_sha256': sha(ap)})
        if (i+1) % 20 == 0: print('Audited', i+1, '/ 110', flush=True)
    verify()
    files = sorted((OUT/'answers').glob('*.json')) + sorted((OUT/'answers').glob('*.npz'))
    assert len(files) == 220
    save(OUT/'AUDIT_FROZEN.json', {'manifest_sha256': digest, 'files': {str(p): sha(p) for p in files},
        'status': 'COMPLETE', 'labels_decoded': False, 'localization_heads_fitted': False,
        'seconds_this_invocation': time.monotonic()-started})
    print('All 110 label-free feasibility records complete.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--phase', choices=('tests', 'prepare', 'run'), required=True)
    args = p.parse_args(); {'tests': tests, 'prepare': prepare, 'run': run}[args.phase]()
