"""Frozen answer-only sampling replication with historical anchors."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_sampling_replication import NEW_ARMS, CORES, SELECTORS, ANCHORS, arm_name, score_sampling
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals

OUT = ROOT/'results/fusion_sampling_replication_v1'
ORIGINAL = ROOT/'results/fusion_replication_v1'
ANCHOR = ROOT/'results/fusion_graph_conditioning_v1'
PARENT = ROOT/'results/fusion_token_gap_v1'
LABELS = ROOT/'results/localization_prm_label_audit_v1'
EVALUATION = PARENT/'EVALUATION.json'


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path); obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj


io_module = module(ROOT/'scripts/audit_fusion_localization_forensics_v1.py', 'sampling_io')
load, sha = io_module.load, io_module.sha


def retry(operation):
    for attempt in range(8):
        try: return operation()
        except PermissionError:
            if attempt == 7: raise
            time.sleep(min(.1*2**attempt, 2.))


def save(path, value):
    path = Path(path).resolve(); assert path.is_relative_to(OUT.resolve()), path
    return retry(lambda: io_module.save(path, value))


def key(p): return p['left']+' minus '+p['right']+' ['+p['scope']+']'


def pairs():
    out = []
    def add(left, right, scope='all'):
        item = dict(left=left, right=right, scope=scope)
        if key(item) not in {key(v) for v in out}: out.append(item)
    for selector in SELECTORS[1:]:
        for core in CORES: add(arm_name(selector, core), arm_name('full', core))
        for a,b in [('iu','equal'),('graph010','joint0'),('graph010','graph_perm'),('graph010','iu'),('graph010','equal_graph010')]:
            add(arm_name(selector,a), arm_name(selector,b))
        for core in ('iu', 'graph010'):
            add(arm_name(selector,core), arm_name('full',core), 'eligible')
            add(arm_name(selector,core), 'dual__equal_graph_perm')
        add(arm_name(selector,'graph010'), arm_name('full','graph010'), 'eligible_both_native')
    for core in ('iu','graph010'):
        for other in ('uniform','risk_top','dufs_permuted'):
            add(arm_name('dufs_transposed',core), arm_name(other,core))
        add(arm_name('window_diffusion',core), arm_name('uniform',core))
    assert len(out) == 93
    return out


def tests():
    suite = unittest.defaultTestLoader.loadTestsFromModule(module(ROOT/'tests/test_fusion_sampling_replication.py', 'sampling_tests'))
    stream = io.StringIO(); result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    save(OUT/'TESTS.json', dict(passed=result.wasSuccessful(), tests=result.testsRun, output=stream.getvalue()))
    print(stream.getvalue()); assert result.wasSuccessful()


def prepare():
    assert not (OUT/'MANIFEST.json').exists(); assert load(OUT/'TESTS.json')['passed']
    original = load(ORIGINAL/'MANIFEST.json'); parent = load(PARENT/'MANIFEST.json'); release = load(LABELS/'RELEASE_V3.json')
    arms = parent['arms']; assert len(arms) == 107
    paths = [Path(__file__), ROOT/'spectral_utils/fusion_sampling_replication.py', ROOT/'tests/test_fusion_sampling_replication.py',
        ROOT/'docs/experiments/FUSION_SAMPLING_REPLICATION_V1.md', ROOT/'scripts/run_answer_localization_v2.py',
        ROOT/'scripts/audit_fusion_localization_forensics_v1.py', ORIGINAL/'MANIFEST.json', PARENT/'MANIFEST.json',
        EVALUATION, PARENT/'REVIEW.json', LABELS/'RELEASE_V3.json', LABELS/'REVIEW.json',
        ROOT/'results/localization_source_group_audit_v1/CANONICAL_GROUPS.json']
    for name, mod in list(sys.modules.items()):
        if name.startswith('spectral_utils') and getattr(mod, '__file__', None): paths.append(Path(mod.__file__).resolve())
    for rec in original['selected']:
        uid = rec['uid']; paths.append(ORIGINAL/'inputs'/(uid+'.npz'))
        for directory in (ORIGINAL, ANCHOR): paths.extend(directory/'scores'/(uid+ext) for ext in ('.json','.npz'))
    paths += [Path(info['label_path']) for cell,info in release['cells'].items() if cell in {r['cell'] for r in original['selected']}]
    paths += list(io_module.RAW_FILES.values())
    save(OUT/'MANIFEST.json', dict(status='DEVELOPMENT_FIXED_BANK_SAMPLING_REPLICATION', release_id=release['release_id'],
        scoring_namespace=original['scoring_namespace'], selected=original['selected'], arms=arms+list(NEW_ARMS),
        new_arms=list(NEW_ARMS), external_arms=arms, contrasts=pairs(), hashes={str(p):sha(p) for p in paths},
        created_unix=time.time(), labels_decoded_for_scoring=False, workers=3, score_seconds_cap=1200,
        gate_support='all_original_fit_rows', expected_eligible=72))
    print('Frozen149 entries (107 anchors +42 selector/core entries),93 contrasts.', flush=True)


def verify():
    m = load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items(): assert sha(p) == h, p
    return m


def one(rec, digest, namespace):
    start = time.monotonic(); uid = rec['uid']; jp = OUT/'scores'/(uid+'.json'); npz = OUT/'scores'/(uid+'.npz')
    if jp.exists():
        old = load(jp); assert old['manifest_sha256'] == digest and sha(npz) == old['array_sha256']; return uid
    with np.load(ORIGINAL/'inputs'/(uid+'.npz'), allow_pickle=False) as a: raw,ss,ee = a['raw'],a['step_starts'],a['step_ends']
    data = []
    for directory in (ORIGINAL, ANCHOR):
        with np.load(directory/'scores'/(uid+'.npz'), allow_pickle=False) as a: data.append({k:a[k] for k in a.files})
        data.append(load(directory/'scores'/(uid+'.json')))
    identity = namespace+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
    arrays, methods, diagnostics = score_sampling(raw,ss,ee,*data,identity)
    npz.parent.mkdir(parents=True, exist_ok=True); tmp = npz.with_suffix('.npz.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f, **arrays)
    retry(lambda: tmp.replace(npz))
    save(jp, {**rec, 'methods':methods, 'diagnostics':diagnostics, 'routing':data[1]['routing'],
        'manifest_sha256':digest, 'array_sha256':sha(npz), 'labels_used':False, 'seconds':time.monotonic()-start})
    return uid


def scores():
    m = verify(); started = time.monotonic(); digest = sha(OUT/'MANIFEST.json'); remaining = iter(m['selected']); completed = []
    with ProcessPoolExecutor(max_workers=m['workers']) as executor:
        pending = {}
        def submit():
            try: rec = next(remaining)
            except StopIteration: return
            pending[executor.submit(one, rec, digest, m['scoring_namespace'])] = rec['uid']
        for _ in range(m['workers']): submit()
        while pending:
            done,_ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                completed.append(future.result()); pending.pop(future)
                if time.monotonic()-started < m['score_seconds_cap']: submit()
            if len(completed)%10 == 0: print('Completed',len(completed),'/110',flush=True)
    assert len(completed) == 110, 'Submission cap reached; saved answer checkpoints retained.'
    paths = [OUT/'scores'/(r['uid']+ext) for r in m['selected'] for ext in ('.json','.npz')]
    save(OUT/'SCORES_FROZEN.json', dict(status='COMPLETE', files={str(p):sha(p) for p in paths},
        manifest_sha256=digest, labels_decoded=False, seconds=time.monotonic()-started))
    print('All42 outputs frozen for110 answers.',flush=True)


def fixed(rows): return [{**r,'predictions':r['fixed_iu_predictions'],'decision_valid':r['fixed_iu_valid']} for r in rows]
def metrics_module(): return module(ROOT/'scripts/run_answer_localization_v2.py', 'sampling_metric_reference')


def evaluate():
    m = verify(); freeze = load(OUT/'SCORES_FROZEN.json'); assert freeze['status']=='COMPLETE' and not freeze['labels_decoded']
    assert freeze['manifest_sha256'] == sha(OUT/'MANIFEST.json')
    for p,h in freeze['files'].items(): assert sha(p)==h, p
    previous = load(EVALUATION); rows = deepcopy(previous['rows']); release = load(LABELS/'RELEASE_V3.json')
    for cell in {r['cell'] for r in rows}:
        with np.load(release['cells'][cell]['label_path'], allow_pickle=False) as labels:
            index = {str(v):i for i,v in enumerate(labels['row_ids'])}
            for row in (r for r in rows if r['cell']==cell):
                uid = row['uid']; i=index[row['row_id']]
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2]; np.testing.assert_array_equal(row['target'],labels['step_error_flags'][a:b])
                else: assert row['target']==int(labels['first_error'][i])
                meta = load(OUT/'scores'/(uid+'.json')); assert meta['routing']==row['routing']
                dg=meta['diagnostics']; row['sampling']={k:dg[k] for k in ('bank','eligible','original_joint_valid','original_rows','budget')}
                row['sampling']['joint_valid']={s:d['joint_valid'] for s,d in dg['fits'].items()}
                with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as a:
                    for arm,detail in meta['methods'].items():
                        for k in ('valid','decision_valid','fixed_iu_valid'): row[k][arm]=detail[k]
                        for dst,src in [('predictions','prediction'),('fixed_iu_predictions','fixed_iu_prediction'),('peaks','peak')]: row[dst][arm]=detail.get(src)
                        row['sources'][arm]=detail.get('source_arm')
                        if detail['valid']: row['scores'][arm]=a[arm+'__risk'].tolist()
    mm=metrics_module(); fx=fixed(rows)
    def bundle(subset):
        fixed_subset=fixed(subset)
        return {arm:dict(prm=mm.prm_metric(subset,arm),pb=mm.pb_metric(subset,arm),pb_common_iu_gate=mm.pb_metric(fixed_subset,arm)) for arm in m['arms']}
    metrics=bundle(rows); eligible=[r for r in rows if r['sampling']['eligible']]; assert len(eligible)==m['expected_eligible']
    for arm in m['external_arms']: assert metrics[arm]==previous['metrics'][arm],arm
    save(OUT/'EVALUATION.json', dict(status='DEVELOPMENT_SAMPLING_QUALITY',release_id=m['release_id'],
        scores_sha256=sha(OUT/'SCORES_FROZEN.json'),rows=rows,metrics=metrics,eligible_metrics=bundle(eligible)))
    for selector in SELECTORS:
        for core in ('iu','graph010'):
            arm=arm_name(selector,core);print(arm,metrics[arm]['prm']['auroc'],metrics[arm]['pb']['macro_f1'],flush=True)


def select(rows,p):
    if p['scope']=='all': return rows
    chosen=[r for r in rows if r['sampling']['eligible']]
    if p['scope']=='eligible': return chosen
    assert p['scope']=='eligible_both_native'
    selector=p['left'].split('__')[0].removeprefix('sample_')
    return [r for r in chosen if r['sampling']['joint_valid'][selector] and r['sampling']['original_joint_valid']]


def contrasts():
    m=verify(); e=load(OUT/'EVALUATION.json'); path=OUT/'CONTRASTS.json'; digest=sha(OUT/'EVALUATION.json'); started=time.monotonic(); mm=metrics_module()
    state=load(path) if path.exists() else dict(evaluation_sha256=digest,pairs={}); assert state['evaluation_sha256']==digest
    for p in m['contrasts']:
        k=key(p)
        if k in state['pairs']: continue
        rows=select(e['rows'],p); common=[r for r in rows if r['valid'][p['left']] and r['valid'][p['right']]]
        state['pairs'][k]={**p,'selected_ids':[r['uid'] for r in rows],
            'left_prm':mm.prm_metric(common,p['left']),'right_prm':mm.prm_metric(common,p['right']),
            'left_pb':mm.pb_metric(rows,p['left']),'right_pb':mm.pb_metric(rows,p['right']),
            'uncertainty':paired_source_group_intervals(rows,p['left'],p['right'])}
        save(path,state)
    assert len(state['pairs'])==len(m['contrasts']); state.update(status='COMPLETE',seconds=time.monotonic()-started); save(path,state)
    print('All93 comparisons complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['tests','prepare','scores','evaluate','contrasts']);globals()[parser.parse_args().phase]()
