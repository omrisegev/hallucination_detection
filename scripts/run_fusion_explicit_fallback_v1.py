"""Frozen composition/evaluation of explicit answer-local Joint fallbacks."""
import os
for option in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[option] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'results/fusion_context_bank_pilot_v1'
RELEASE = ROOT / 'results/answer_localization_representation_pilot_v1/RELEASE.json'
OUT = ROOT / 'results/fusion_explicit_fallback_pilot_v1'
sys.path.insert(0, str(ROOT / 'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT / 'spectral_utils'))
from spectral_utils.fusion_explicit_fallback import ARMS, PARENT_ARMS, CORES, compose_fallback
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def safe(x):
    if isinstance(x, dict): return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)): return [safe(v) for v in x]
    if isinstance(x, (bool, np.bool_)): return bool(x)
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    return x
def save(path, value):
    path = Path(path); path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def registered_pairs():
    pairs = []
    for core in CORES:
        dual, single = 'dual__' + core, 'single__' + core
        pairs.extend([(dual, single), (dual, 'moment__iu'), (dual, 'dual__equal'),
                      (dual, 'dual__iu'), (single, 'moment__iu'), (single, 'moment__equal')])
    for policy in ('single', 'dual'):
        pairs.extend([(policy + '__graph010', policy + '__joint0'),
                      (policy + '__graph010', policy + '__graph_perm')])
    pairs.extend([('dual__equal', 'moment__equal'), ('dual__iu', 'moment__iu'), ('dual__iu', 'dual__equal')])
    pairs.extend([('dual__' + c, 'context__' + c) for c in ('joint0', 'graph010')])
    pairs.extend([('single__' + c, 'moment__' + c) for c in CORES])
    assert len(pairs) == len(set(pairs)) == 30
    return pairs


def verify():
    manifest = load(OUT / 'MANIFEST.json')
    for path, expected in manifest['hashes'].items():
        assert sha(path) == expected, path
    return manifest


def prepare():
    if (OUT / 'MANIFEST.json').exists():
        verify(); print('Existing frozen fallback manifest verified.'); return
    parent = load(PARENT / 'MANIFEST.json'); freeze = load(PARENT / 'SCORES_FROZEN.json')
    review = load(PARENT / 'REVIEW.json')
    assert freeze['manifest_sha256'] == sha(PARENT / 'MANIFEST.json')
    assert review['status'] == 'PASS' and review['evaluation_sha256'] == sha(PARENT / 'EVALUATION.json')
    assert review['contrasts_sha256'] == sha(PARENT / 'CONTRASTS.json')
    hashes = {**parent['hashes'], **freeze['files']}
    for name in ('MANIFEST.json', 'SCORES_FROZEN.json', 'EVALUATION.json', 'REVIEW.json', 'CONTRASTS.json'):
        hashes[str(PARENT / name)] = sha(PARENT / name)
    for path in (Path(__file__), ROOT / 'spectral_utils/fusion_explicit_fallback.py',
                 ROOT / 'tests/test_fusion_explicit_fallback.py',
                 ROOT / 'docs/experiments/FUSION_EXPLICIT_FALLBACK_PILOT_V1.md', RELEASE):
        hashes[str(path)] = sha(path)
    for path, expected in hashes.items(): assert sha(path) == expected, path
    save(OUT / 'MANIFEST.json', {'release_id': parent['release_id'], 'selected': parent['selected'],
         'arms': ARMS, 'contrasts': registered_pairs(), 'hashes': hashes, 'created_unix': time.time(),
         'labels_decoded_for_composition': False, 'status': 'ADAPTIVE_DEVELOPMENT', 'worker_cap': 1,
         'policy': 'moment Joint -> eligible context Joint (dual only) -> moment IU; fit validity only',
         'gate_contract': 'inherit selected source native gate; common parent-IU gate diagnostic only'})
    print('Frozen 8 new composites, 17 unchanged anchors, 30 contrasts.', flush=True)


def scores():
    manifest = verify(); digest = sha(OUT / 'MANIFEST.json'); started = time.monotonic()
    if (OUT / 'SCORES_FROZEN.json').exists():
        frozen = load(OUT / 'SCORES_FROZEN.json'); assert frozen['manifest_sha256'] == digest
        for path, expected in frozen['files'].items(): assert sha(path) == expected, path
        print('Existing fallback scores verified.'); return
    selected = manifest['selected']
    save(OUT / 'RUN_STATE.json', {'state': 'RUNNING', 'pid': os.getpid(), 'completed': 0, 'total': len(selected)})
    for i, rec in enumerate(selected, 1):
        uid = rec['uid']; path = OUT / 'scores' / (uid + '.npz'); meta_path = path.with_suffix('.json')
        if meta_path.exists():
            old = load(meta_path)
            assert old['manifest_sha256'] == digest and old['array_sha256'] == sha(path)
        else:
            parent = load(PARENT / 'scores' / (uid + '.json'))
            with np.load(PARENT / 'scores' / (uid + '.npz'), allow_pickle=False) as source:
                arrays, methods, routing = compose_fallback(source, parent['methods'])
            path.parent.mkdir(exist_ok=True, parents=True)
            with path.with_suffix('.npz.tmp').open('wb') as stream: np.savez_compressed(stream, **arrays)
            path.with_suffix('.npz.tmp').replace(path)
            save(meta_path, {**rec, 'methods': methods, 'routing': routing, 'manifest_sha256': digest,
                            'array_sha256': sha(path), 'labels_decoded': False})
        save(OUT / 'RUN_STATE.json', {'state': 'RUNNING', 'pid': os.getpid(), 'completed': i,
                                     'total': len(selected), 'seconds': time.monotonic() - started})
    verify()
    files = sorted((OUT / 'scores').glob('*.npz')) + sorted((OUT / 'scores').glob('*.json'))
    assert len(files) == 2 * len(selected)
    elapsed = time.monotonic() - started
    save(OUT / 'SCORES_FROZEN.json', {'manifest_sha256': digest, 'files': {str(p): sha(p) for p in files},
          'labels_decoded': False, 'seconds': elapsed, 'timing_scope': 'cached composition, not end-to-end fitting'})
    save(OUT / 'RUN_STATE.json', {'state': 'COMPLETE', 'completed': len(selected), 'total': len(selected), 'seconds': elapsed})
    print('All 58 score bundles frozen before label decoding; seconds', elapsed, flush=True)


def metric_module():
    spec = importlib.util.spec_from_file_location('fallback_parent_metrics', ROOT / 'scripts/run_answer_localization_v2.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def fixed_gate(rows):
    return [{**r, 'decision_valid': r['fixed_iu_valid'], 'predictions': r['fixed_iu_predictions']} for r in rows]


def evaluate():
    manifest = verify(); frozen = load(OUT / 'SCORES_FROZEN.json')
    assert frozen['manifest_sha256'] == sha(OUT / 'MANIFEST.json') and frozen['labels_decoded'] is False
    for path, expected in frozen['files'].items(): assert sha(path) == expected, path
    release = load(RELEASE); rows = []
    for cell in sorted({r['cell'] for r in manifest['selected']}):
        info = release['cells'][cell]; assert sha(info['label_path']) == info['label_opaque_sha256']
        with np.load(info['label_path'], allow_pickle=False) as labels:
            positions = {str(r): i for i, r in enumerate(labels['row_ids'])}
            assert len(positions) == len(labels['row_ids'])
            for rec in (r for r in manifest['selected'] if r['cell'] == cell):
                i = positions[rec['row_id']]
                if cell.startswith('prm'):
                    a, b = labels['step_flag_offsets'][i:i+2]; target = labels['step_error_flags'][a:b].copy()
                else: target = int(labels['first_error'][i])
                meta = load(OUT / 'scores' / (rec['uid'] + '.json'))
                row = {**rec, 'target': target, 'scores': {}, 'valid': {}, 'decision_valid': {}, 'predictions': {},
                       'fixed_iu_valid': {}, 'fixed_iu_predictions': {}, 'peaks': {}, 'sources': {}, 'routing': meta['routing']}
                with np.load(OUT / 'scores' / (rec['uid'] + '.npz'), allow_pickle=False) as arrays:
                    for arm, detail in meta['methods'].items():
                        for field in ('valid', 'decision_valid', 'fixed_iu_valid'): row[field][arm] = detail[field]
                        row['predictions'][arm] = detail.get('prediction')
                        row['fixed_iu_predictions'][arm] = detail.get('fixed_iu_prediction')
                        row['peaks'][arm] = detail.get('peak'); row['sources'][arm] = detail['source_arm']
                        if detail['valid']:
                            score = arrays[arm + '__risk']
                            assert len(score) == rec['steps'] and np.isfinite(score).all()
                            if cell.startswith('prm'): assert len(score) == len(target)
                            row['scores'][arm] = score
                rows.append(row)
    metrics = metric_module(); fixed = fixed_gate(rows)
    bundles = {arm: {'prm': metrics.prm_metric(rows, arm), 'pb': metrics.pb_metric(rows, arm),
                     'pb_common_iu_gate': metrics.pb_metric(fixed, arm)} for arm in ARMS}
    prior = load(PARENT / 'EVALUATION.json')
    for arm in PARENT_ARMS: assert bundles[arm] == prior['metrics'][arm], arm
    save(OUT / 'EVALUATION.json', {'status': 'ADAPTIVE_DEVELOPMENT', 'scores_sha256': sha(OUT / 'SCORES_FROZEN.json'),
         'rows': rows, 'metrics': bundles, 'labels_decoded': True, 'parent_metric_bundle_replays': len(PARENT_ARMS)})
    print('All 17 parent metric bundles replay exactly.', flush=True)
    for arm in ARMS:
        if arm.startswith(('single', 'dual')):
            print(arm, bundles[arm]['prm'], 'PB', bundles[arm]['pb']['macro_f1'], flush=True)


def contrasts():
    started = time.monotonic(); manifest = verify(); e = load(OUT / 'EVALUATION.json'); digest = sha(OUT / 'EVALUATION.json')
    path = OUT / 'CONTRASTS.json'
    state = load(path) if path.exists() else {'evaluation_sha256': digest, 'pairs': {}}
    assert state['evaluation_sha256'] == digest
    metrics = metric_module(); rows = e['rows']; fixed = fixed_gate(rows)
    for left, right in manifest['contrasts']:
        key = left + ' minus ' + right
        if key in state['pairs']: continue
        common = [r for r in rows if r['valid'][left] and r['valid'][right]]
        state['pairs'][key] = {'left': left, 'right': right,
            'left_prm': metrics.prm_metric(common, left), 'right_prm': metrics.prm_metric(common, right),
            'left_pb': metrics.pb_metric(rows, left), 'right_pb': metrics.pb_metric(rows, right),
            'left_pb_common_iu_gate': metrics.pb_metric(fixed, left), 'right_pb_common_iu_gate': metrics.pb_metric(fixed, right),
            'uncertainty': paired_source_group_intervals(rows, left, right)}
        save(path, state)
    assert len(state['pairs']) == 30
    state.update(state='COMPLETE', seconds_this_invocation=time.monotonic() - started)
    save(path, state); print('All 30 paired contrasts complete in', state['seconds_this_invocation'], 'seconds.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', required=True, choices=('prepare', 'scores', 'evaluate', 'contrasts'))
    phase = parser.parse_args().phase
    {'prepare': prepare, 'scores': scores, 'evaluate': evaluate, 'contrasts': contrasts}[phase]()
