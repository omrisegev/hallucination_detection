"""Freeze, fit and evaluate fixed fusion recipes on additional source groups."""
import os
for option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[option]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/fusion_replication_v1'
SOURCE=ROOT/'results/localization_source_group_audit_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_replication import score_fixed_banks,ARMS
from spectral_utils.fusion_replication_cohort import select_cohort
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def safe(v):
    if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [safe(x) for x in v]
    if isinstance(v,(bool,np.bool_)):return bool(v)
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    return v
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(value),indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(7):
        try:tmp.replace(path);return
        except PermissionError:
            if attempt==6:raise
            time.sleep(.025*2**attempt)
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def pairs():
    parent=module(ROOT/'scripts/run_fusion_explicit_fallback_v1.py','fallback_pair_registry')
    result=parent.registered_pairs()+[(x,'context__equal') for x in ('single__joint0','dual__joint0')]
    result += [('context__'+x,'moment__'+x) for x in ('equal','iu','joint0','graph010')]
    result += [('context__'+x,'context__equal') for x in ('joint0','graph010')]
    assert len(result)==len(set(result))==38
    assert all(x in ARMS and y in ARMS for x,y in result)
    return result


def verify():
    m=load(OUT/'MANIFEST.json')
    for path,h in m['hashes'].items():assert sha(path)==h,path
    for rec in m['selected']:assert sha(OUT/'inputs'/(rec['uid']+'.npz'))==rec['input_sha256']
    return m


def prepare():
    if (OUT/'MANIFEST.json').exists():verify();print('Existing replication manifest verified.');return
    release=load(SOURCE/'RELEASE_V2.json');audit=load(SOURCE/'AUDIT.json');review=load(SOURCE/'REVIEW.json')
    assert review['status']=='PASS'
    for path,h in review['hashes'].items():assert sha(path)==h,path
    namespace=release['release_id']+'/'+OUT.name
    selected,support=select_cohort(release,audit['excluded_canonical_groups'],namespace)
    assert not {r['group_id'] for r in selected}&set(audit['excluded_canonical_groups'])
    assert len(selected)==len({r['group_id'] for r in selected})
    roster=pairs();hashes={}
    for cell in dict.fromkeys(r['cell'] for r in selected):
        info=release['cells'][cell]
        assert sha(info['telemetry_path'])==info['telemetry_sha256']
        assert sha(info['label_path'])==info['label_opaque_sha256']
        hashes[info['telemetry_path']]=info['telemetry_sha256'];hashes[info['label_path']]=info['label_opaque_sha256']
        with np.load(info['telemetry_path'],allow_pickle=False) as source:
            raw=source['raw'];offsets=source['token_offsets'];step_offsets=source['step_row_offsets']
            ids=source['row_ids'].astype(str);groups=source['group_ids'].astype(str)
            ss,ee=source['step_starts'],source['step_ends']
            for rec in (r for r in selected if r['cell']==cell):
                i=rec['row'];assert ids[i]==rec['row_id'] and groups[i]==rec['legacy_group_id']
                lo,hi=map(int,offsets[i:i+2]);a,b=map(int,step_offsets[i:i+2])
                assert hi-lo==rec['tokens'] and b-a==rec['steps']
                path=OUT/'inputs'/(rec['uid']+'.npz');path.parent.mkdir(exist_ok=True,parents=True)
                np.savez_compressed(path,raw=raw[lo:hi],step_starts=ss[a:b]-lo,step_ends=ee[a:b]-lo)
                rec['input_sha256']=sha(path)
            del raw
        print('Prepared',cell,sum(r['cell']==cell for r in selected),'answers.',flush=True)
    paths=[Path(__file__),ROOT/'scripts/run_fusion_explicit_fallback_v1.py',ROOT/'scripts/run_answer_localization_v2.py',
           ROOT/'docs/experiments/FUSION_SOURCE_DISJOINT_REPLICATION_V1.md',
           ROOT/'tests/test_fusion_replication.py',ROOT/'tests/test_fusion_replication_cohort.py']
    paths += [SOURCE/n for n in ('RELEASE_V2.json','AUDIT.json','CANONICAL_GROUPS.json','REVIEW.json','EVALUATION_V2.json','CONTRASTS_V2.json')]
    for name,m in list(sys.modules.items()):
        if name.startswith('spectral_utils') and getattr(m,'__file__',None):paths.append(Path(m.__file__).resolve())
    hashes.update({str(p):sha(p) for p in paths})
    save(OUT/'MANIFEST.json',{'release_id':release['release_id'],'scoring_namespace':release['predecessor_release_id'],
        'selection_namespace':namespace,'selected':selected,'length_support':support,'arms':ARMS,'contrasts':roster,
        'hashes':hashes,'excluded_components':audit['excluded_canonical_groups'],'worker_cap':3,'score_seconds_cap':1200,
        'labels_decoded':False,'status':'SOURCE_DISJOINT_DEVELOPMENT_REPLICATION','created_unix':time.time()})
    print('Frozen',len(selected),'unique source groups, 19 recipes, 38 comparisons.',flush=True)


def process_one(rec,digest,namespace):
    started=time.monotonic();uid=rec['uid'];path=OUT/'scores'/(uid+'.npz');mp=path.with_suffix('.json')
    if mp.exists():
        m=load(mp);assert m['manifest_sha256']==digest and m['array_sha256']==sha(path)
        return {'uid':uid,'resumed':True,'seconds':m['seconds']}
    with np.load(OUT/'inputs'/(uid+'.npz'),allow_pickle=False) as z:
        arrays,methods,routing,diagnostics=score_fixed_banks(z['raw'],z['step_starts'],z['step_ends'],
            namespace+'/'+rec['cell']+'/'+rec['row_id'])
    path.parent.mkdir(exist_ok=True,parents=True)
    with path.with_suffix('.npz.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
    path.with_suffix('.npz.tmp').replace(path)
    save(mp,{**rec,'methods':methods,'routing':routing,'diagnostics':diagnostics,'manifest_sha256':digest,
        'array_sha256':sha(path),'labels_decoded':False,'seconds':time.monotonic()-started})
    return {'uid':uid,'valid':sum(x['valid'] for x in methods.values()),'seconds':time.monotonic()-started}


def scores(workers):
    m=verify();digest=sha(OUT/'MANIFEST.json');started=time.monotonic()
    if not 1<=workers<=m['worker_cap']:raise ValueError('WORKER_CAP')
    if (OUT/'SCORES_FROZEN.json').exists():
        f=load(OUT/'SCORES_FROZEN.json');assert f['manifest_sha256']==digest
        for path,h in f['files'].items():assert sha(path)==h,path
        print('Existing scores verified.');return
    remaining=[];done=0
    for rec in m['selected']:
        p=OUT/'scores'/(rec['uid']+'.json')
        if p.exists():
            meta=load(p);assert meta['manifest_sha256']==digest and meta['array_sha256']==sha(p.with_suffix('.npz'));done+=1
        else:remaining.append(rec)
    def state(status):save(OUT/'RUN_STATE.json',{'state':status,'pid':os.getpid(),'completed':done,'total':len(m['selected']),
        'seconds_this_invocation':time.monotonic()-started,'start_unix':time.time()-(time.monotonic()-started)})
    state('RUNNING');index=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        active={}
        while active or index<len(remaining):
            while len(active)<workers and index<len(remaining) and time.monotonic()-started<m['score_seconds_cap']:
                rec=remaining[index];index+=1;active[pool.submit(process_one,rec,digest,m['scoring_namespace'])]=rec['uid']
            if not active:break
            ready,_=wait(active,return_when=FIRST_COMPLETED)
            for future in ready:
                result=future.result();del active[future];done+=1
                if done%5==0 or done==len(m['selected']):print(done,'/',len(m['selected']),json.dumps(result),flush=True)
            state('RUNNING')
    if done!=len(m['selected']):state('PAUSED_AT_CAP');print('Paused at the registered cap; completed checkpoints retained.');return
    verify();files=sorted((OUT/'scores').glob('*.json'))+sorted((OUT/'scores').glob('*.npz'));assert len(files)==2*done
    save(OUT/'SCORES_FROZEN.json',{'manifest_sha256':digest,'files':{str(p):sha(p) for p in files},'labels_decoded':False,
        'seconds_this_invocation':time.monotonic()-started,'worker_cap':workers})
    state('COMPLETE');print('All predictions frozen before evaluation.',flush=True)


def metrics_module():return module(ROOT/'scripts/run_answer_localization_v2.py','replication_metric_reference')
def fixed_gate(rows):return [{**r,'decision_valid':r['fixed_iu_valid'],'predictions':r['fixed_iu_predictions']} for r in rows]


def evaluate():
    m=verify();f=load(OUT/'SCORES_FROZEN.json');assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and not f['labels_decoded']
    for p,h in f['files'].items():assert sha(p)==h,p
    release=load(SOURCE/'RELEASE_V2.json');rows=[]
    for cell in sorted({r['cell'] for r in m['selected']}):
        info=release['cells'][cell];assert sha(info['label_path'])==info['label_opaque_sha256']
        with np.load(info['label_path'],allow_pickle=False) as labels:
            positions={str(v):i for i,v in enumerate(labels['row_ids'])};assert len(positions)==len(labels['row_ids'])
            for rec in (r for r in m['selected'] if r['cell']==cell):
                i=positions[rec['row_id']]
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2];target=labels['step_error_flags'][a:b].copy()
                else:target=int(labels['first_error'][i])
                meta=load(OUT/'scores'/(rec['uid']+'.json'))
                row={**rec,'target':target,'scores':{},'valid':{},'decision_valid':{},'predictions':{},
                     'fixed_iu_valid':{},'fixed_iu_predictions':{},'peaks':{},'sources':{},'routing':meta['routing']}
                with np.load(OUT/'scores'/(rec['uid']+'.npz'),allow_pickle=False) as arrays:
                    for arm,d in meta['methods'].items():
                        for key in ('valid','decision_valid','fixed_iu_valid'):row[key][arm]=d[key]
                        row['predictions'][arm]=d.get('prediction');row['fixed_iu_predictions'][arm]=d.get('fixed_iu_prediction')
                        row['peaks'][arm]=d.get('peak');row['sources'][arm]=d['source_arm']
                        if d['valid']:
                            x=arrays[arm+'__risk'];assert len(x)==rec['steps'] and np.isfinite(x).all()
                            if cell.startswith('prm'):assert len(x)==len(target)
                            row['scores'][arm]=x
                rows.append(row)
    mm=metrics_module();fixed=fixed_gate(rows)
    metrics={arm:{'prm':mm.prm_metric(rows,arm),'pb':mm.pb_metric(rows,arm),'pb_common_iu_gate':mm.pb_metric(fixed,arm)} for arm in ARMS}
    previous=load(SOURCE/'EVALUATION_V2.json')
    save(OUT/'EVALUATION.json',{'release_id':m['release_id'],'status':'SOURCE_DISJOINT_DEVELOPMENT_REPLICATION',
        'scores_sha256':sha(OUT/'SCORES_FROZEN.json'),'rows':rows,'metrics':metrics,
        'previous_cohort_metrics':{arm:previous['metrics'][arm] for arm in ARMS},'labels_decoded':True})
    for arm in ARMS:print(arm,'PRM',metrics[arm]['prm']['auroc'],'PB',metrics[arm]['pb']['macro_f1'],flush=True)


def contrasts():
    m=verify();e=load(OUT/'EVALUATION.json');path=OUT/'CONTRASTS.json';digest=sha(OUT/'EVALUATION.json');started=time.monotonic()
    state=load(path) if path.exists() else {'evaluation_sha256':digest,'pairs':{}}
    assert state['evaluation_sha256']==digest;mm=metrics_module();fixed=fixed_gate(e['rows'])
    for left,right in m['contrasts']:
        key=left+' minus '+right
        if key in state['pairs']:continue
        common=[r for r in e['rows'] if r['valid'][left] and r['valid'][right]]
        state['pairs'][key]={'left':left,'right':right,'left_prm':mm.prm_metric(common,left),'right_prm':mm.prm_metric(common,right),
            'left_pb':mm.pb_metric(e['rows'],left),'right_pb':mm.pb_metric(e['rows'],right),
            'left_pb_common_iu_gate':mm.pb_metric(fixed,left),'right_pb_common_iu_gate':mm.pb_metric(fixed,right),
            'uncertainty':paired_source_group_intervals(e['rows'],left,right)}
        save(path,state)
    assert len(state['pairs'])==38;state.update(status='COMPLETE',seconds_this_invocation=time.monotonic()-started);save(path,state)
    print('All 38 paired comparisons complete.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=('prepare','scores','evaluate','contrasts'))
    p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    {'prepare':prepare,'scores':lambda:scores(args.workers),'evaluate':evaluate,'contrasts':contrasts}[args.phase]()
