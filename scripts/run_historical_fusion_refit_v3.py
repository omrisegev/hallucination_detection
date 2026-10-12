"""Run five historical comparators on corrected full-population group folds."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / 'hd_jlsml_v2_wt'
PARENT = ROOT / 'results/localization_full_benchmark_v3'
OUT = ROOT / 'results/historical_fusion_refit_v3'
FOLDS = ROOT / 'results/localization_source_group_audit_v1/FOLDS_V2.json'
PROTOCOL = ROOT / 'docs/experiments/HISTORICAL_FUSION_REFIT_V3.md'
CORE = ROOT / 'spectral_utils/historical_fusion_refit.py'
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location('historical_fusion_refit_core', CORE)
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1<<20), b''): h.update(block)
    return h.hexdigest()


def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def safe(value):
    if isinstance(value, dict): return {str(k): safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [safe(x) for x in value]
    if isinstance(value, np.ndarray): return safe(value.tolist())
    if isinstance(value, np.generic): return safe(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(safe(value), indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def state(phase, **extra):
    save(OUT/'RUN_STATE.json', dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra))


def cell_data(cell):
    return {p.stem: np.load(p, mmap_mode='r', allow_pickle=False)
            for p in (PARENT/'inputs'/cell).glob('*.npy')}


def preflight(reference):
    cell = cell_data('pb_gsm8k_q8')
    base = SOURCE/'results/joint_lsml_optimization_v2'
    old = load(base/'folds/folds.json')['processbench']['outer']
    train = np.array([old[str(g)] != 0 for g in cell['group_ids']])
    with np.load(base/'cells/pb_gsm8k_q8.npz', allow_pickle=False) as previous:
        for key in cell: np.testing.assert_array_equal(cell[key], previous[key])
    prep = core.prepare(cell, train, reference)
    weights, metadata, failures = core.fit(prep, reference)
    assert not failures, failures
    output = core.score(prep, cell, weights, np.arange(len(cell['row_ids'])))
    checks = []
    for filename, arms in [('scores_outer.npz',core.ARMS[:3]+core.ARMS[4:]),
                           ('scores_continuity.npz',core.ARMS[3:4])]:
        path = base/'structure/pb_gsm8k_q8/outer0'/filename
        with np.load(path,allow_pickle=False) as previous:
            for arm in arms:
                for suffix in ('w','top10','spanmax','detector'):
                    key = arm+'__'+suffix
                    np.testing.assert_allclose(output[key],previous[key],rtol=2e-7,atol=2e-7)
                    checks.append(dict(key=key,max_abs=float(np.max(np.abs(output[key]-previous[key])))))
    # Constructed group mask check: repeated source never crosses fit/evaluation.
    groups = ['a','a','b','c','d','e']
    om = dict(a=0,b=1,c=2,d=3,e=4); im = dict(b=0,c=1,d=2,e=3)
    for inner in (None,0,1,2,3): core.fold_masks(groups,om,im,0,inner)
    return dict(status='PASS',purpose='implementation_fidelity_only',old_cell='pb_gsm8k_q8',
                old_outer=0,arms=list(core.ARMS),checks=checks,labels_used=False)


def prepare_run():
    reference, files = core.reference_modules(SOURCE)
    if (OUT/'MANIFEST.json').exists():
        manifest = verify()
        assert load(OUT/'PREFLIGHT.json')['status']=='PASS'
        return manifest, reference
    state('PREFLIGHT')
    review = preflight(reference)
    _,files=core.reference_modules(SOURCE)  # Include any lazy reference imports used by the fits.
    parent = load(PARENT/'MANIFEST.json'); folds = load(FOLDS)
    for rec in parent['selected']: assert rec['group_id'] in folds['outer']
    cells = sorted({r['cell'] for r in parent['selected']})
    jobs = [dict(cell=cell,outer=k,inner=inner)
            for cell in cells for k in range(5)
            for inner in ([None]+list(range(5)) if cell.startswith('pb_') else [None])]
    assert len(jobs)==245
    bound = [CORE,ROOT/'spectral_utils/historical_fusion_evaluation.py',Path(__file__),PROTOCOL,FOLDS,PARENT/'MANIFEST.json',
             PARENT/'evaluation/JOINED.json',PARENT/'evaluation/JOINED.npz',
             ROOT/'scripts/evaluate_historical_fusion_refit_v3.py']
    bound += [Path(x) for x in files]
    bound += [p for cell in cells for p in (PARENT/'inputs'/cell).glob('*.npy')]
    # Labels are bound for the later evaluator, never passed to a fitting function.
    release_path=ROOT/'results/localization_prm_label_audit_v1/RELEASE_V3.json'
    bound += [release_path]+[Path(v['label_path']) for v in load(release_path)['cells'].values()]
    manifest = dict(status='FROZEN_FULL_DEVELOPMENT_HISTORICAL_FIRST_PANEL',
                    release_id=parent['release_id'],arms=list(core.ARMS),selected=parent['selected'],jobs=jobs,
                    reference_checkout=str(SOURCE),hashes={str(p):sha(p) for p in bound},
                    label_access=dict(fusion=False,pb_threshold='nested_outer_training_labels_only'),
                    versions=dict(python=sys.version,numpy=np.__version__,
                        scipy=__import__('scipy').__version__,torch=__import__('torch').__version__),
                    created_unix=time.time(),workers=1,invocation_seconds_cap=28800)
    save(OUT/'MANIFEST.json',manifest);save(OUT/'PREFLIGHT.json',review)
    return manifest, reference


def verify():
    manifest=load(OUT/'MANIFEST.json')
    for path,h in manifest['hashes'].items(): assert sha(path)==h,path
    return manifest


def job_name(job):
    return job['cell']+'/outer'+str(job['outer'])+'/'+('test' if job['inner'] is None else 'inner'+str(job['inner']))


def run():
    manifest, reference = prepare_run(); mh=sha(OUT/'MANIFEST.json'); folds=load(FOLDS)
    started=time.time();current_cell=None;done=0
    for job in manifest['jobs']:
        target=OUT/'fits'/job_name(job)
        if target.with_suffix('.json').exists():
            meta=load(target.with_suffix('.json'))
            assert meta['manifest_sha256']==mh and sha(target.with_suffix('.npz'))==meta['array_sha256']
            done+=1;continue
        state('FITTING',completed=done,total=len(manifest['jobs']),job=job)
        if current_cell != job['cell']:
            cell=cell_data(job['cell']);current_cell=job['cell']
            records=sorted([r for r in manifest['selected'] if r['cell']==current_cell],key=lambda x:x['row'])
            assert [r['row_id'] for r in records]==list(map(str,cell['row_ids']))
            groups=[r['group_id'] for r in records]
        train,evaluate=core.fold_masks(groups,folds['outer'],folds['inner'][str(job['outer'])],job['outer'],job['inner'])
        outer_groups={g for g,k in folds['outer'].items() if k==job['outer']}
        assert set(np.array(groups)[train]).isdisjoint(outer_groups)
        if job['inner'] is not None: assert set(np.array(groups)[evaluate]).isdisjoint(outer_groups)
        tick=time.time()
        try:
            prep=core.prepare(cell,train,reference)
            weights,details,failures=core.fit(prep,reference)
            arrays=core.score(prep,cell,weights,np.flatnonzero(evaluate))
            prep_meta=dict(prep.diagnostics)
            arrays.update(mean=prep.mean,std=prep.std,medians=prep.medians,fit_indices=prep.fit_indices)
            del prep
        except (ValueError,RuntimeError,FloatingPointError) as error:
            failures={arm:type(error).__name__+': '+str(error) for arm in core.ARMS}
            details={};prep_meta={}
            offsets=cell['step_row_offsets'];rows=np.flatnonzero(evaluate)
            arrays=dict(rows=rows,steps=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in rows]))
        arrays['train_rows']=np.flatnonzero(train)
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.npz.tmp')
        with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        tmp.replace(target.with_suffix('.npz'))
        save(target.with_suffix('.json'),dict(job=job,manifest_sha256=mh,array_sha256=sha(target.with_suffix('.npz')),
             preparation=prep_meta,methods=details,failures=failures,labels_accessed=False,
             train_groups=sorted(set(np.array(groups)[train])),evaluation_groups=sorted(set(np.array(groups)[evaluate])),
             seconds=time.time()-tick))
        done+=1;gc.collect();print('Historical refit',done,'/',len(manifest['jobs']),job_name(job),flush=True)
        if time.time()-started>manifest['invocation_seconds_cap']:
            state('CHECKPOINTED_INVOCATION_CAP',completed=done,total=len(manifest['jobs']));return
    state('SCORING_COMPLETE',completed=done,total=len(manifest['jobs']),seconds=time.time()-started)
    verify()
    spec=importlib.util.spec_from_file_location('historical_refit_evaluation',ROOT/'scripts/evaluate_historical_fusion_refit_v3.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main()
    state('COMPLETE_REVIEWED_FIRST_HISTORICAL_PANEL',completed=done,total=len(manifest['jobs']))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('preflight','run'),default='run')
    args=parser.parse_args()
    try:
        if args.phase=='preflight':
            reference,_=core.reference_modules(SOURCE)
            result=preflight(reference);save(OUT/'PREFLIGHT_DRAFT.json',result);print(json.dumps(result))
        else:run()
    except BaseException as error:
        state('FAILED',error=type(error).__name__+': '+str(error));raise
