"""Execute registered full pass2a, then evaluate and review automatically."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/localization_full_benchmark_v3'
OUT=ROOT/'results/localization_full_shortlist_v3'
PILOT=ROOT/'results/fusion_entropy_sampling_v1/EVALUATION.json'
PROTOCOL=ROOT/'docs/experiments/FULL_LOCALIZATION_SHORTLIST_V3.md'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.answer_localization_v2 import json_safe
from spectral_utils.fusion_full_shortlist import NEW_ARMS,CORE_ARMS,score_full_shortlist


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(json_safe(value),indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(8):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.05*2**attempt,2))


def state(phase,**extra):save(OUT/'RUN_STATE.json',dict(phase=phase,pid=os.getpid(),updated_unix=time.time(),**extra))


def source(rec):
    path=PARENT/'scores'/(rec['uid']+'.json');meta=load(path)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:a={k:z[k] for k in z.files}
    assert sha(path.with_suffix('.npz'))==meta['array_sha256']
    for key in ('uid','cell','row_id','group_id','tokens','steps'):assert meta[key]==rec[key]
    return a,meta


def compute(rec,manifest_hash):
    started=time.time();a,old=source(rec)
    identity='localization-cached-v1-20260907/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
    arrays,methods,diag=score_full_shortlist(a,old,identity)
    if rec.get('reuse_original110'):
        for folder,roster in [('fusion_graph_conditioning_v1',CORE_ARMS),('fusion_trajectory_imm_v1',tuple(x for x in NEW_ARMS if x not in CORE_ARMS))]:
            path=ROOT/'results'/folder/'scores'/(rec['uid']+'.json');reference=load(path)
            with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
                for arm in roster:
                    d=methods[arm];ref=reference['methods'][arm]
                    for k in ('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak'):
                        assert d.get(k)==ref.get(k),(rec['uid'],arm,k)
                    if d['valid']:
                        for suffix in ('window','risk'):
                            np.testing.assert_allclose(arrays[arm+'__'+suffix],saved[arm+'__'+suffix],atol=1e-10,rtol=1e-10)
        diag['pilot_new_maps_replayed']=len(NEW_ARMS)
    keep=('valid','decision_valid','fixed_iu_valid','prediction','fixed_iu_prediction','peak','source_arm','route','reason','status')
    legacy={arm:{k:d[k] for k in keep if k in d} for arm,d in old['methods'].items()}
    methods={**legacy,**methods}
    for arm,d in old['methods'].items():
        if d['valid']:arrays[arm+'__risk']=a[arm+'__risk'].copy()
    for arm in ('dual__iu','dual__equal'):
        if old['methods'][arm]['valid']:arrays[arm+'__window']=a[arm+'__window'].copy()
    path=OUT/'scores'/(rec['uid']+'.npz');path.parent.mkdir(parents=True,exist_ok=True)
    tmp=Path(str(path)+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(path)
    meta=dict(rec,methods=methods,routing=old['routing'],diagnostics=diag,array_sha256=sha(path),
        manifest_sha256=manifest_hash,labels_used=False,seconds=time.time()-started,
        parent_metadata_sha256=sha(PARENT/'scores'/(rec['uid']+'.json')),
        parent_array_sha256=old['array_sha256'])
    save(path.with_suffix('.json'),meta)
    return rec['uid']


def verify():
    m=load(OUT/'MANIFEST.json')
    for path,h in m['hashes'].items():assert sha(path)==h,path
    return m


def prepare():
    if (OUT/'MANIFEST.json').exists():
        verify()
        if (OUT/'PREFLIGHT.json').exists() and load(OUT/'PREFLIGHT.json')['status']=='PASS':return
    m=load(PARENT/'MANIFEST.json');assert load(PARENT/'evaluation/REVIEW_SUPPLEMENT.json')['status']=='PASS'
    imported={Path(mod.__file__).resolve() for name,mod in list(sys.modules.items()) if name.startswith('spectral_utils') and getattr(mod,'__file__',None)}
    files=imported|{Path(__file__),PROTOCOL,PARENT/'MANIFEST.json',PARENT/'SCORES_FROZEN.json',PILOT,
                   ROOT/'scripts/evaluate_full_localization_shortlist_v3.py',ROOT/'scripts/evaluate_localization_full_anchors_v3.py'}
    hashes=deepcopy(m['hashes'])
    for path in files:hashes[str(path)]=sha(path)
    manifest=dict(status='FROZEN_FULL_DEVELOPMENT_PASS2A',selected=m['selected'],arms=m['arms']+list(NEW_ARMS),
        external_arms=m['arms'],new_arms=list(NEW_ARMS),hashes=hashes,release_id=m['release_id'],
        workers=3,submission_seconds_cap=28800,labels_used_for_scoring=False,created_unix=time.time())
    if not (OUT/'MANIFEST.json').exists():save(OUT/'MANIFEST.json',manifest)
    mh=sha(OUT/'MANIFEST.json');chosen=[r for r in m['selected'] if r.get('reuse_original110')]
    assert len(chosen)==110
    chosen+=[next(r for r in m['selected'] if r['tokens']<64),next(r for r in m['selected'] if r['tokens']>2048)]
    started=time.time();state('PREFLIGHT',completed=0,total=len(chosen))
    for i,rec in enumerate(chosen):
        compute(rec,mh)
        if (i+1)%10==0:state('PREFLIGHT',completed=i+1,total=len(chosen));print('Preflight',i+1,'/',len(chosen),flush=True)
    save(OUT/'PREFLIGHT.json',dict(status='PASS',pilot_answers=110,new_maps_per_pilot=len(NEW_ARMS),
       short_and_long_cases=2,seconds=time.time()-started,manifest_sha256=mh))
    state('PREFLIGHT_PASS',completed=len(chosen),total=len(chosen))


def score():
    m=verify();assert load(OUT/'PREFLIGHT.json')['status']=='PASS';mh=sha(OUT/'MANIFEST.json')
    pending=[];done=[]
    for rec in m['selected']:
        path=OUT/'scores'/(rec['uid']+'.json')
        if path.exists():
            d=load(path);assert d['manifest_sha256']==mh
            assert sha(path.with_suffix('.npz'))==d['array_sha256'];done.append(rec['uid'])
        else:pending.append(rec)
    started=time.time();state('SCORING',completed=len(done),total=len(m['selected']),workers=m['workers'])
    if pending:
        with ProcessPoolExecutor(max_workers=m['workers']) as pool:
            futures={pool.submit(compute,rec,mh):rec for rec in pending}
            for future in as_completed(futures):
                try:done.append(future.result())
                except BaseException:
                    for f in futures:f.cancel()
                    raise
                if len(done)%100==0:
                    state('SCORING',completed=len(done),total=len(m['selected']),seconds=time.time()-started)
                    print('Full shortlist',len(done),'/',len(m['selected']),flush=True)
                if time.time()-started>m['submission_seconds_cap']:
                    for f in futures:f.cancel()
                    state('CHECKPOINTED_INVOCATION_CAP',completed=len(done),total=len(m['selected']));return False
    assert len(done)==len(m['selected'])
    files={str(p):sha(p) for rec in m['selected'] for p in [OUT/'scores'/(rec['uid']+'.json'),OUT/'scores'/(rec['uid']+'.npz')]}
    save(OUT/'SCORES_FROZEN.json',dict(status='COMPLETE_SHORTLIST_PASS2A',files=files,rows=len(done),arms=m['arms'],
        manifest_sha256=mh,seconds_this_invocation=time.time()-started,labels_used=False))
    state('SCORES_COMPLETE_STARTING_EVALUATION',completed=len(done),total=len(done));return True


def run():
    if score():
        path=ROOT/'scripts/evaluate_full_localization_shortlist_v3.py'
        spec=importlib.util.spec_from_file_location('full_shortlist_evaluation',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.main()
        state('COMPLETE_REVIEWED_PASS2A',completed=13769,total=13769,report='evaluation/REPORT.html',historical_comparison_complete=False)
        print('Full shortlist scoring/evaluation/review COMPLETE.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['prepare','run'],required=True);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    try:prepare() if args.phase=='prepare' else run()
    except Exception as exc:state('FAILED_CHECKPOINTS_PRESERVED',reason=repr(exc));raise
