"""Frozen unlabeled structural audit of minimum-two Joint feature groups."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'results/fusion_replication_v1'
OUT = ROOT / 'results/joint_pair_identifiability_audit_v1'
sys.path.insert(0, str(ROOT / 'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT / 'spectral_utils'))
from spectral_utils.answer_localization_v2 import JOINT_SEED, json_safe, prepare_local
from spectral_utils.joint_lsml import covariance_matrix, discover_loao_consensus_groups, regularized_joint_map_weights
from spectral_utils.joint_pair_extension import fit_joint_pairs


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')
    for attempt in range(7):
        try:
            temporary.replace(path); return
        except PermissionError:
            if attempt == 6: raise
            time.sleep(.025 * 2**attempt)


def prepare():
    if (OUT / 'MANIFEST.json').exists():
        verify(); print('Existing audit manifest verified.'); return
    m, f = load(PARENT / 'MANIFEST.json'), load(PARENT / 'SCORES_FROZEN.json')
    assert f['manifest_sha256'] == sha(PARENT / 'MANIFEST.json') and not f['labels_decoded']
    for path, digest in f['files'].items(): assert sha(path) == digest, path
    # Do not read the parent EVALUATION or any benchmark label array.
    paths = [Path(__file__), ROOT / 'spectral_utils/joint_pair_extension.py',
             ROOT / 'tests/test_joint_pair_extension.py',
             ROOT / 'docs/experiments/JOINT_PAIR_IDENTIFIABILITY_AUDIT_V1.md',
             PARENT / 'MANIFEST.json', PARENT / 'SCORES_FROZEN.json', OUT / 'TESTS.txt']
    paths += [Path(module.__file__).resolve() for name, module in sys.modules.copy().items()
              if name.startswith('spectral_utils') and getattr(module, '__file__', None)]
    # The documentary claim is read-only evidence from Claude's worktree.
    paths += [Path(r'C:\Users\omris\TAU\hd_jlsml_v2_wt') / relative for relative in
              ('results/joint_lsml_optimization_v2/REPORT.md',
               'docs/experiments/JOINT_LSML_OPTIMIZATION_V2_AMENDMENT_R1.md')]
    hashes = {str(p): sha(p) for p in paths}; hashes.update(f['files'])
    save(OUT / 'MANIFEST.json', {'status': 'FROZEN_UNLABELED_STRUCTURAL_AUDIT',
        'selected': m['selected'], 'parent_manifest_sha256': sha(PARENT / 'MANIFEST.json'),
        'parent_scores_sha256': sha(PARENT / 'SCORES_FROZEN.json'), 'hashes': hashes,
        'k_range': [3, 4, 6, 8], 'minimum_group_size': 2, 'seed': JOINT_SEED,
        'condition_target': 1000., 'max_workers': 3, 'seconds_cap': 1200,
        'labels_decoded': False, 'created_unix': time.time()})
    print('Frozen pair-group audit: 110 answers, two banks, no label evaluation.')


def verify():
    m = load(OUT / 'MANIFEST.json')
    for path, digest in m['hashes'].items(): assert sha(path) == digest, path
    return m


def process_one(rec, digest):
    started = time.monotonic(); uid = rec['uid']; path = OUT / 'rows' / (uid + '.json')
    if path.exists():
        row = load(path); assert row['manifest_sha256'] == digest
        assert row['arrays_sha256'] == sha(path.with_suffix('.npz'))
        return row
    parent = load(PARENT / 'scores' / (uid + '.json'))
    arrays, banks = {}, {}
    with np.load(PARENT / 'scores' / (uid + '.npz'), allow_pickle=False) as source:
        indices = source['fit_indices']
        for bank in ('moment', 'context'):
            before = parent['diagnostics']['banks'][bank]
            result = {'old_valid': parent['methods'][bank+'__joint0']['valid'],
                      'old_grouping': before['shared'].get('grouping'),
                      'old_groups': before['shared'].get('joint', {}).get('groups'), 'valid': False}
            banks[bank] = result
            z, anchor, normal = prepare_local(source[bank+'__features'], before['names'], indices)
            fit = z[indices]; result['normalization'] = normal
            grouping = discover_loao_consensus_groups(fit, np.minimum(3, np.arange(len(fit))*4//len(fit)),
                k_range=(3, 4, 6, 8), seed=JOINT_SEED, minimum_group_size=2,
                minimum_held_admissible_fraction=.95, use_minimum_ari_tiebreak=True)
            result['grouping'] = {k: grouping.get(k) for k in ('status','K','group_sizes','median_ari','candidates','labels')}
            if grouping['status'] != 'SELECTED':
                result['status'] = 'BLOCKED_NO_ADMISSIBLE_PARTITION'; continue
            result['same_partition_as_parent'] = bool(result['old_groups'] is not None and
                np.array_equal(grouping['labels'], result['old_groups']))
            try:
                wrapper = fit_joint_pairs(covariance_matrix(fit), grouping['labels'], anchor_index=anchor)
                joint = wrapper.joint; jac = joint.jacobian_audit
                valid = bool(joint.converged and joint.multistart_audit['status'] == 'PASS' and jac['full_global_rank']
                             and np.isfinite(jac['condition_number']) and jac['condition_number'] <= 1e8)
                weight, inverse = regularized_joint_map_weights(fit, joint.model_covariance, joint.global_loading,
                    mode='liu', lam=0., target_condition=1000.)
                result.update(status='OK' if valid else 'FIT_DIAGNOSTIC_ONLY', valid=valid,
                    pairs=wrapper.pair_audit, native_map_audit=wrapper.native_map_audit,
                    converged=joint.converged, multistart=joint.multistart_audit, jacobian=jac,
                    diagonal=joint.diagonal_audit, objective=joint.objective,
                    relative_offdiag_misfit=joint.relative_offdiag_misfit, inverse=inverse)
                for name, value in [('covariance',joint.model_covariance),('v',joint.global_loading),
                                    ('u',joint.group_loading),('weight',weight),('fitted_offdiag',joint.fitted_offdiag)]:
                    arrays[bank+'__'+name] = value
            except ValueError as exc:
                # Expected mathematical infeasibility only. Unexpected errors abort.
                if not str(exc).startswith('PAIR_'): raise
                result.update(status=str(exc), failure_detail=getattr(exc, 'detail', {}))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix('.npz'), **arrays)
    row = {**rec, 'banks': banks, 'seconds': time.monotonic()-started, 'labels_decoded': False,
           'manifest_sha256': digest, 'arrays_sha256': sha(path.with_suffix('.npz'))}
    save(path, row); return row


def run():
    m = verify(); digest = sha(OUT / 'MANIFEST.json'); started = time.monotonic()
    done = 0; remaining = []
    for rec in m['selected']:
        path = OUT / 'rows' / (rec['uid'] + '.json')
        if path.exists():
            row = load(path); assert row['manifest_sha256'] == digest and row['arrays_sha256'] == sha(path.with_suffix('.npz')); done += 1
        else: remaining.append(rec)
    def state(status):
        save(OUT / 'RUN_STATE.json', {'state': status, 'pid': os.getpid(), 'completed': done,
            'total': len(m['selected']), 'seconds_this_invocation': time.monotonic()-started})
    state('RUNNING'); index = 0
    with ProcessPoolExecutor(max_workers=3) as pool:
        active = {}
        while active or index < len(remaining):
            while len(active) < 3 and index < len(remaining) and time.monotonic()-started < m['seconds_cap']:
                rec = remaining[index]; index += 1; active[pool.submit(process_one, rec, digest)] = rec['uid']
            if not active: break
            ready, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in ready:
                row = future.result(); del active[future]; done += 1
                if done % 10 == 0 or done == len(m['selected']):
                    print(done, '/', len(m['selected']), {bank: value['status'] for bank,value in row['banks'].items()}, flush=True)
            state('RUNNING')
    if done != len(m['selected']): state('PAUSED_AT_CAP'); return
    verify(); files = sorted((OUT / 'rows').glob('*')); assert len(files) == 2*done
    save(OUT / 'FROZEN.json', {'manifest_sha256': digest, 'labels_decoded': False,
        'files': {str(p): sha(p) for p in files}, 'seconds_this_invocation': time.monotonic()-started})
    state('COMPLETE'); print('Unlabeled pair audit complete.', flush=True)


def summarize():
    m = verify(); frozen = load(OUT / 'FROZEN.json')
    for path,digest in frozen['files'].items(): assert sha(path)==digest,path
    rows = [load(OUT / 'rows' / (r['uid']+'.json')) for r in m['selected']]
    banks = {}
    for bank in ('moment','context'):
        rr = [r['banks'][bank] for r in rows]
        banks[bank] = {'status_counts': dict(Counter(r['status'] for r in rr)),
            'old_valid': sum(r['old_valid'] for r in rr), 'new_valid': sum(r['valid'] for r in rr),
            'rescued': sum(r['valid'] and not r['old_valid'] for r in rr),
            'lost': sum(r['old_valid'] and not r['valid'] for r in rr),
            'selected_K': dict(Counter(r['grouping']['K'] for r in rr if r['grouping']['status']=='SELECTED')),
            'valid_K': dict(Counter(r['grouping']['K'] for r in rr if r['valid'])),
            'same_partition_as_parent': sum(r.get('same_partition_as_parent', False) for r in rr),
            'valid_pair_fits': sum(r['valid'] and r.get('pairs',{}).get('pair_count',0)>0 for r in rr)}
    save(OUT / 'SUMMARY.json', {'status':'COMPLETE_UNLABELED','banks':banks,'answers':len(rows),
        'native_joint_union_old':sum(any(r['banks'][b]['old_valid'] for b in banks) for r in rows),
        'native_joint_union_new':sum(any(r['banks'][b]['valid'] for b in banks) for r in rows),
        'labels_decoded':False,'frozen_sha256':sha(OUT / 'FROZEN.json')})
    print(json.dumps(banks,indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',choices=('prepare','run','summarize'),required=True)
    args=parser.parse_args(); {'prepare':prepare,'run':run,'summarize':summarize}[args.phase]()
