"""Full cached answer-only anchor pass; preserve old scores and every row.

The subsequent fixed shortlist and historical refit passes are explicitly
pending in METHOD_REGISTRY.json, not implied by this driver's completion.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/localization_full_benchmark_v3'
OLD = ROOT/'results/fusion_replication_v1'
RELEASE = ROOT/'results/localization_prm_label_audit_v1/RELEASE_V3.json'
sys.path.insert(0, str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_replication import ARMS, score_fixed_banks
from spectral_utils.answer_localization_v2 import json_safe

FIELDS = ('raw', 'token_offsets', 'step_row_offsets', 'row_ids', 'group_ids', 'step_starts', 'step_ends')
_INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def replace(tmp, path):
    for attempt in range(8):
        try: tmp.replace(path); return
        except PermissionError:
            if attempt == 7: raise
            time.sleep(min(.05*2**attempt, 2))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding='utf-8')
    replace(tmp, path)


def source_cell(cell):
    if cell not in _INPUTS:
        _INPUTS[cell] = {name: np.load(OUT/'inputs'/cell/(name+'.npy'), mmap_mode='r', allow_pickle=False) for name in FIELDS}
    return _INPUTS[cell]


def answer(rec):
    c = source_cell(rec['cell']); i = rec['row']
    assert str(c['row_ids'][i]) == rec['row_id']
    assert str(c['group_ids'][i]) == rec['legacy_group_id']
    lo, hi = map(int, c['token_offsets'][i:i+2]); a, b = map(int, c['step_row_offsets'][i:i+2])
    raw = c['raw'][lo:hi]; ss = np.asarray(c['step_starts'][a:b])-lo; ee = np.asarray(c['step_ends'][a:b])-lo
    assert hi-lo == rec['tokens'] and b-a == rec['steps']
    assert len(ss) and ss.shape == ee.shape and np.all(ss >= 0) and np.all(ee <= len(raw)) and np.all(ee > ss)
    return raw, ss, ee


def registry():
    rows = [dict(method=a, stage=1, status='IMPLEMENTED_PENDING_FULL_RUN', scope='current_answer_only',
                 implementation='spectral_utils.fusion_replication.score_fixed_banks') for a in ARMS]
    groups = [
        ('condition100 Joint lambda0 / graph / permuted graph', 2, 'same_answer_fixed_shortlist'),
        ('equal graph / permuted graph controls', 2, 'same_answer_fixed_shortlist'),
        ('risk sampling IU / Joint / equal / graph controls', 2, 'same_answer_fixed_shortlist'),
        ('IU+Joint static mean / GLS; IMM negative control', 2, 'same_answer_fixed_shortlist'),
        ('canonical IU/U-PCR; LIU/DUFS-LIU; CONT/L-SML', 3, 'corrected_group_fold_refit_required'),
        ('Claude model-inverse Joint lambda0 / LIU graph / permutation', 3, 'corrected_group_fold_refit_required'),
        ('family6 top5 / GL-LIU / token-IU29 / Unified28 / entropy top5', 3, 'exact_scope_and_threshold_adapter_required'),
        ('CIW-DEEM / DEEM-B3 plus token-IU29', 4, 'separate_pooled_or_transductive_adapter'),
        ('Mind-the-Gap adaptation / external PRM / 72B critic', 4, 'declared_access_panels'),
    ]
    rows += [dict(method=a, stage=s, status='PENDING_FULL_ADAPTER_OR_REFIT', scope=scope) for a,s,scope in groups]
    return dict(status='CONTINUING_BENCHMARK_NOT_YET_COMPLETE', methods=rows,
                historical24='Separate final-answer transfer after locking localization recipe')


def verify():
    m = load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items(): assert sha(p) == h, p
    return m


def prepare():
    if (OUT/'MANIFEST.json').exists(): verify(); print('Existing full benchmark manifest verified.'); return
    release = load(RELEASE); previous = load(OLD/'MANIFEST.json'); frozen = load(OLD/'SCORES_FROZEN.json')
    assert release['scoring_namespace'] == previous['scoring_namespace']
    assert frozen['manifest_sha256'] == sha(OLD/'MANIFEST.json')
    old = {(r['cell'],r['row_id']):r for r in previous['selected']}
    hashes = {str(RELEASE):sha(RELEASE)}; selected = []; counts = {}; expanded = 0
    for cell, info in release['cells'].items():
        for pathkey, hashkey in [('telemetry_path','telemetry_sha256'), ('label_path','label_opaque_sha256')]:
            assert sha(info[pathkey]) == info[hashkey], info[pathkey]
            hashes[info[pathkey]] = info[hashkey]
        with zipfile.ZipFile(info['telemetry_path']) as z:
            needed = sum(z.getinfo(name+'.npy').file_size for name in FIELDS)
            if shutil.disk_usage(ROOT).free < needed+(4 << 30): raise RuntimeError('Insufficient free disk for bounded input cache')
            for name in FIELDS:
                path = OUT/'inputs'/cell/(name+'.npy'); path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name+'.npy') as src, path.with_suffix('.npy.tmp').open('wb') as dst: shutil.copyfileobj(src,dst,1<<20)
                replace(path.with_suffix('.npy.tmp'),path); hashes[str(path)] = sha(path)
            expanded += needed
        src = source_cell(cell); assert len(src['row_ids']) == len(info['rows'])
        assert len(set(map(str,src['row_ids']))) == len(info['rows'])
        assert len(src['token_offsets']) == len(info['rows'])+1
        assert int(src['token_offsets'][-1]) == len(src['raw']) and src['raw'].shape[1] == 29
        assert len(src['step_row_offsets']) == len(info['rows'])+1
        assert int(src['step_row_offsets'][-1]) == len(src['step_starts']) == len(src['step_ends'])
        assert np.all(np.diff(src['token_offsets'])>0) and np.all(np.diff(src['step_row_offsets'])>0)
        for row in info['rows']:
            rec = dict(row,cell=cell)
            prior = old.get((cell,row['row_id']))
            rec['uid'] = prior['uid'] if prior else cell+'__'+hashlib.sha256((cell+'/'+row['row_id']).encode()).hexdigest()[:16]
            rec['reuse_original110'] = bool(prior)
            raw,ss,ee = answer(rec)
            if prior:
                for key in ('row','row_id','group_id','legacy_group_id','tokens','steps'): assert prior[key] == rec[key],key
                with np.load(OLD/'inputs'/(prior['uid']+'.npz'),allow_pickle=False) as a:
                    for key,value in [('raw',raw),('step_starts',ss),('step_ends',ee)]: np.testing.assert_array_equal(a[key],value)
                for suffix in ('.json','.npz'):
                    p=OLD/'scores'/(rec['uid']+suffix); assert sha(p)==frozen['files'][str(p)]; hashes[str(p)]=sha(p)
            selected.append(rec)
        counts[cell] = dict(rows=len(info['rows']),tokens=sum(r['tokens'] for r in info['rows']),
                            insufficient_window_support=sum(r['tokens']<64 for r in info['rows']),
                            beyond_previous_length_range=sum(r['tokens']>2048 for r in info['rows']))
        print('Prepared',cell,counts[cell]['rows'],'rows',flush=True)
    assert len(selected)==13769 and len({r['uid'] for r in selected})==13769
    assert sum(r['reuse_original110'] for r in selected)==110
    paths = [Path(__file__),ROOT/'docs/experiments/LOCALIZATION_FULL_BENCHMARK_V3.md',OLD/'MANIFEST.json',OLD/'SCORES_FROZEN.json',
             Path(release['folds_path']),Path(release['canonical_groups_path'])]
    paths += [Path(m.__file__).resolve() for n,m in list(sys.modules.items()) if n.startswith('spectral_utils') and getattr(m,'__file__',None)]
    hashes.update({str(p):sha(p) for p in paths})
    save(OUT/'METHOD_REGISTRY.json',registry())
    save(OUT/'MANIFEST.json',dict(status='FROZEN_FULL_CACHED_DEVELOPMENT',release_id=release['release_id'],
        scoring_namespace=release['scoring_namespace'],selected=selected,counts=counts,arms=list(ARMS),hashes=hashes,
        workers=3,submission_seconds_cap=8*3600,expanded_input_bytes=expanded,labels_used_for_scoring=False,
        python=sys.version,created_unix=time.time()))
    print('Frozen13769 rows,110 reusable,21 short rows retained.',flush=True)


def score_one(rec,digest,namespace):
    started=time.monotonic(); path=OUT/'scores'/(rec['uid']+'.npz'); jp=path.with_suffix('.json')
    if jp.exists():
        meta=load(jp); assert meta['manifest_sha256']==digest and meta['array_sha256']==sha(path)
        return rec['cell']
    raw,ss,ee=answer(rec)
    if rec['reuse_original110']:
        oldmeta=load(OLD/'scores'/jp.name)
        with np.load(OLD/'scores'/path.name,allow_pickle=False) as a: arrays={k:a[k] for k in a.files}
        methods=oldmeta['methods']; routing=oldmeta['routing']; diagnostics=oldmeta['diagnostics']
    elif len(raw)<64:
        arrays=dict(step_starts=ss,step_ends=ee);routing=None;diagnostics=dict(labels_accessed=False)
        methods={a:dict(valid=False,decision_valid=False,fixed_iu_valid=False,prediction=None,
                       fixed_iu_prediction=None,reason='TOO_FEW_FIT_WINDOWS',source_arm=a) for a in ARMS}
    else:
        arrays,methods,routing,diagnostics=score_fixed_banks(raw,ss,ee,namespace+'/'+rec['cell']+'/'+rec['row_id'])
    assert set(methods)==set(ARMS)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as f: np.savez_compressed(f,**arrays)
    replace(tmp,path)
    save(jp,dict(**rec,methods=methods,routing=routing,diagnostics=diagnostics,
                 manifest_sha256=digest,array_sha256=sha(path),labels_used=False,seconds=time.monotonic()-started))
    return rec['cell']


def replay():
    m=verify(); start=time.monotonic(); tested=[]
    # Cover all five cells, both bank routes, fallback, short/long support.
    original=[r for r in m['selected'] if r['reuse_original110']]
    chosen={}
    for rec in original:
        meta=load(OLD/'scores'/(rec['uid']+'.json'));key=(rec['cell'],meta['routing']['routes']['dual'])
        chosen.setdefault(key,rec)
    for rec in chosen.values():
        raw,ss,ee=answer(rec)
        arrays,methods,routing,_=score_fixed_banks(raw,ss,ee,m['scoring_namespace']+'/'+rec['cell']+'/'+rec['row_id'])
        old=load(OLD/'scores'/(rec['uid']+'.json')); assert routing==old['routing']
        with np.load(OLD/'scores'/(rec['uid']+'.npz'),allow_pickle=False) as saved:
            for arm in ARMS:
                for key in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction'):
                    assert methods[arm].get(key)==old['methods'][arm].get(key),(rec['uid'],arm,key)
                if methods[arm]['valid']:
                    for suffix in ('window','risk'): np.testing.assert_allclose(arrays[arm+'__'+suffix],saved[arm+'__'+suffix],atol=1e-10,rtol=1e-10)
        tested.append(rec['uid'])
    shorts=[r for r in m['selected'] if r['tokens']<64]
    assert len(shorts)==21
    for rec in (shorts[0],shorts[-1]):
        score_one(rec,sha(OUT/'MANIFEST.json'),m['scoring_namespace'])
        meta=load(OUT/'scores'/(rec['uid']+'.json'))
        assert not any(d['valid'] or d['decision_valid'] for d in meta['methods'].values())
    save(OUT/'PREFLIGHT_REVIEW.json',dict(status='PASS',all_source_joins=13769,exact_reuse_inputs=110,
        replay_answers=tested,replay_methods_per_answer=len(ARMS),short_population=21,short_execution_cases=2,
        seconds=time.monotonic()-start,manifest_sha256=sha(OUT/'MANIFEST.json'),
        scope='same-session existing kernels; not external scientific review'))
    print('PreflightPASS:',len(tested),'numerical answer replays; all13769 joins and110 reuse inputs.',flush=True)


def run():
    m=verify();digest=sha(OUT/'MANIFEST.json');review=load(OUT/'PREFLIGHT_REVIEW.json')
    assert review['status']=='PASS' and review['manifest_sha256']==digest
    started=time.monotonic();done=0;counts={c:0 for c in m['counts']};remaining=[]
    for rec in m['selected']:
        jp=OUT/'scores'/(rec['uid']+'.json')
        if jp.exists():
            meta=load(jp);assert meta['manifest_sha256']==digest and meta['array_sha256']==sha(jp.with_suffix('.npz'))
            done+=1;counts[rec['cell']]+=1
        else:remaining.append(rec)
    # Interleave nine cells; early progress is not a new evaluation cohort.
    remaining.sort(key=lambda r:hashlib.sha256(('full-execution-order/'+r['uid']).encode()).hexdigest())
    index=0
    def state(status):
        save(OUT/'RUN_STATE.json',dict(state=status,pid=os.getpid(),completed=done,total=len(m['selected']),
             cell_completed=counts,seconds_this_invocation=time.monotonic()-started,updated_unix=time.time(),
             pass_name='unchanged19_answer_only_anchors',historical_comparison_complete=False))
    state('RUNNING')
    with ProcessPoolExecutor(max_workers=m['workers']) as pool:
        pending={}
        while pending or index<len(remaining):
            while len(pending)<m['workers'] and index<len(remaining) and time.monotonic()-started<m['submission_seconds_cap']:
                rec=remaining[index];index+=1;pending[pool.submit(score_one,rec,digest,m['scoring_namespace'])]=rec['uid']
            if not pending:break
            ready,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
            for future in ready:
                try: cell=future.result()
                except Exception as exc:
                    save(OUT/'FAILURE.json',dict(uid=pending[future],reason=repr(exc),manifest_sha256=digest))
                    state('FAILED_CHECKPOINTS_PRESERVED')
                    raise
                del pending[future];done+=1;counts[cell]+=1
                if done%50==0:print('Anchors',done,'/',len(m['selected']),flush=True)
            state('RUNNING')
    if done<len(m['selected']):state('PAUSED_AT_SUBMISSION_CAP');return
    verify();files={}
    for rec in m['selected']:
        for ext in ('.json','.npz'):
            path=OUT/'scores'/(rec['uid']+ext);files[str(path)]=sha(path)
    save(OUT/'SCORES_FROZEN.json',dict(status='COMPLETE_ANCHOR_PASS',manifest_sha256=digest,files=files,
         rows=done,arms=list(ARMS),labels_used=False,seconds_this_invocation=time.monotonic()-started))
    reg=load(OUT/'METHOD_REGISTRY.json')
    for rec in reg['methods']:
        if rec['stage']==1:rec['status']='FULL_SCORES_FROZEN_PENDING_EVALUATION'
    save(OUT/'METHOD_REGISTRY.json',reg)
    state('ANCHOR_SCORES_COMPLETE_PENDING_EVALUATION')
    print('Full anchor scores frozen; evaluation and later comparator passes remain.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','replay','run'],required=True)
    args=parser.parse_args();dict(prepare=prepare,replay=replay,run=run)[args.phase]()
