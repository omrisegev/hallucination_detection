"""Freeze, score, evaluate and compare the checked-pair Joint extension."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/fusion_pair_quality_v1'
PARENT=ROOT/'results/fusion_replication_v1';AUDIT=ROOT/'results/joint_pair_identifiability_audit_v1';SOURCE=ROOT/'results/localization_source_group_audit_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_pair_quality import ARMS,PARENT_ARMS,CORES,score_pair_banks
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [safe(v) for v in x]
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(safe(x),indent=2,allow_nan=False),encoding='utf-8')
    for attempt in range(7):
        try:temp.replace(p);return
        except PermissionError:
            if attempt==6:raise
            time.sleep(.025*2**attempt)
def module(p,name):
    spec=importlib.util.spec_from_file_location(name,p);obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


def registered_pairs():
    pairs=[]
    for bank in ('moment','context'):
        for core in CORES:
            left='pair_'+bank+'__'+core
            pairs += [(left,bank+'__'+core),(left,bank+'__iu'),(left,bank+'__equal')]
        pairs += [('pair_'+bank+'__graph010','pair_'+bank+'__'+c) for c in ('joint0','graph_perm')]
    for policy in ('single','dual'):
        for core in CORES:
            left='pair_'+policy+'__'+core
            pairs += [(left,policy+'__'+core),(left,'moment__iu'),(left,'context__equal')]
        pairs += [('pair_'+policy+'__graph010','pair_'+policy+'__'+c) for c in ('joint0','graph_perm')]
    for core in CORES:
        left='pair_dual__'+core
        pairs += [(left,'pair_single__'+core),(left,'pair_dual__equal'),(left,'pair_dual__iu')]
    pairs += [('pair_dual__equal',x) for x in ('dual__equal','moment__equal','context__equal')]
    pairs += [('pair_dual__iu',x) for x in ('dual__iu','moment__iu','context__iu','pair_dual__equal')]
    pairs += [('pair_context__'+c,'pair_moment__'+c) for c in CORES]
    assert len(pairs)==len(set(pairs))==63
    assert all(a in ARMS and b in ARMS for a,b in pairs)
    return pairs


def prepare():
    if (OUT/'MANIFEST.json').exists():verify();print('Existing manifest verified.');return
    p=load(PARENT/'MANIFEST.json');score=load(PARENT/'SCORES_FROZEN.json');ar=load(AUDIT/'REVIEW.json')
    assert ar['status']=='PASS' and not ar['jacobian_amendment']['guard_changes']
    assert sha(ROOT/'scripts/review_joint_pairs_v1.py')==ar['review_script_sha256']
    for path,h in {**ar['hashes'],**ar['dependencies']}.items():assert sha(path)==h,path
    paths=[Path(__file__),ROOT/'spectral_utils/fusion_pair_quality.py',ROOT/'tests/test_fusion_pair_quality.py',OUT/'TESTS.txt',
        ROOT/'scripts/run_answer_localization_v2.py',ROOT/'docs/experiments/JOINT_PAIR_LOCALIZATION_QUALITY_V1.md',
        SOURCE/'RELEASE_V2.json',PARENT/'MANIFEST.json',PARENT/'SCORES_FROZEN.json',PARENT/'EVALUATION.json',
        PARENT/'REVIEW.json',AUDIT/'MANIFEST.json',AUDIT/'FROZEN.json',AUDIT/'REVIEW.json',AUDIT/'JACOBIAN_AMENDMENT.json']
    paths += [Path(obj.__file__).resolve() for name,obj in sys.modules.copy().items() if name.startswith('spectral_utils') and getattr(obj,'__file__',None)]
    hashes={str(path):sha(path) for path in paths}
    for path,h in {**score['files'],**load(AUDIT/'FROZEN.json')['files']}.items():assert sha(path)==h,path;hashes[path]=h
    for rec in p['selected']:
        path=PARENT/'inputs'/(rec['uid']+'.npz');assert sha(path)==rec['input_sha256'];hashes[str(path)]=sha(path)
    release=load(SOURCE/'RELEASE_V2.json')
    for cell in {r['cell'] for r in p['selected']}:
        info=release['cells'][cell];assert sha(info['label_path'])==info['label_opaque_sha256'];hashes[info['label_path']]=info['label_opaque_sha256']
    save(OUT/'MANIFEST.json',{'release_id':p['release_id'],'scoring_namespace':p['scoring_namespace'],
        'selected':p['selected'],'arms':ARMS,'contrasts':registered_pairs(),'hashes':hashes,
        'status':'FROZEN_DEVELOPMENT_QUALITY_COMPARISON','labels_decoded':False,'worker_cap':3,'seconds_cap':1200,'created_unix':time.time()})
    print('Frozen 110 answers, 33 arms, 63 comparisons.',flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for path,h in m['hashes'].items():assert sha(path)==h,path
    return m


def process_one(rec,digest,namespace):
    started=time.monotonic();uid=rec['uid'];path=OUT/'scores'/(uid+'.json')
    if path.exists():
        row=load(path);assert row['manifest_sha256']==digest and row['array_sha256']==sha(path.with_suffix('.npz'));return row
    parent=load(PARENT/'scores'/(uid+'.json'))
    with np.load(PARENT/'scores'/(uid+'.npz'),allow_pickle=False) as source:pa={k:source[k] for k in source.files}
    with np.load(PARENT/'inputs'/(uid+'.npz'),allow_pickle=False) as inp:
        arrays,methods,routing,diagnostics=score_pair_banks(inp['raw'],inp['step_starts'],inp['step_ends'],
            namespace+'/'+rec['cell']+'/'+rec['row_id'],pa,parent)
    audit=load(AUDIT/'rows'/(uid+'.json'))
    with np.load(AUDIT/'rows'/(uid+'.npz'),allow_pickle=False) as expected:
        for bank in ('moment','context'):
            assert methods['pair_'+bank+'__joint0']['valid']==audit['banks'][bank]['valid']
            if bank+'__covariance' in expected.files:
                np.testing.assert_allclose(arrays['pair_'+bank+'__covariance'],expected[bank+'__covariance'],atol=1e-10,rtol=1e-10)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path.with_suffix('.npz'),**arrays)
    row={**rec,'methods':methods,'routing':routing,'diagnostics':diagnostics,'labels_decoded':False,
         'manifest_sha256':digest,'array_sha256':sha(path.with_suffix('.npz')),'seconds':time.monotonic()-started}
    save(path,row);return row


def scores():
    m=verify();digest=sha(OUT/'MANIFEST.json');started=time.monotonic();done=0;remaining=[]
    if (OUT/'SCORES_FROZEN.json').exists():
        f=load(OUT/'SCORES_FROZEN.json');assert f['manifest_sha256']==digest
        for p,h in f['files'].items():assert sha(p)==h,p
        print('Existing scores verified.');return
    for rec in m['selected']:
        path=OUT/'scores'/(rec['uid']+'.json')
        if path.exists():
            row=load(path);assert row['manifest_sha256']==digest and row['array_sha256']==sha(path.with_suffix('.npz'));done+=1
        else:remaining.append(rec)
    def state(status):save(OUT/'RUN_STATE.json',{'state':status,'pid':os.getpid(),'completed':done,'total':len(m['selected']),
        'seconds_this_invocation':time.monotonic()-started})
    state('RUNNING');index=0
    with ProcessPoolExecutor(max_workers=3) as pool:
        active={}
        while active or index<len(remaining):
            while len(active)<3 and index<len(remaining) and time.monotonic()-started<m['seconds_cap']:
                rec=remaining[index];index+=1;active[pool.submit(process_one,rec,digest,m['scoring_namespace'])]=rec['uid']
            if not active:break
            ready,_=wait(active,return_when=FIRST_COMPLETED)
            for future in ready:
                row=future.result();del active[future];done+=1
                if done%10==0 or done==len(m['selected']):print(done,'/',len(m['selected']),'valid',sum(v['valid'] for v in row['methods'].values()),flush=True)
            state('RUNNING')
    if done!=len(m['selected']):state('PAUSED_AT_CAP');return
    verify();files=sorted((OUT/'scores').glob('*'));assert len(files)==2*done
    save(OUT/'SCORES_FROZEN.json',{'manifest_sha256':digest,'files':{str(p):sha(p) for p in files},
        'labels_decoded':False,'seconds_this_invocation':time.monotonic()-started,'workers':3})
    state('COMPLETE');print('All predictions frozen before evaluation.',flush=True)


def metric_module():return module(ROOT/'scripts/run_answer_localization_v2.py','pair_quality_metrics')
def fixed_rows(rows):return [{**r,'decision_valid':r['fixed_iu_valid'],'predictions':r['fixed_iu_predictions']} for r in rows]


def evaluate():
    m=verify();f=load(OUT/'SCORES_FROZEN.json');assert f['manifest_sha256']==sha(OUT/'MANIFEST.json') and not f['labels_decoded']
    for p,h in f['files'].items():assert sha(p)==h,p
    release=load(SOURCE/'RELEASE_V2.json');rows=[]
    for cell in sorted({r['cell'] for r in m['selected']}):
        info=release['cells'][cell]
        with np.load(info['label_path'],allow_pickle=False) as labels:
            positions={str(v):i for i,v in enumerate(labels['row_ids'])};assert len(positions)==len(labels['row_ids'])
            for rec in (r for r in m['selected'] if r['cell']==cell):
                i=positions[rec['row_id']]
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2];target=labels['step_error_flags'][a:b]
                else:target=int(labels['first_error'][i])
                meta=load(OUT/'scores'/(rec['uid']+'.json'));row={**rec,'target':target,'scores':{},'valid':{},'decision_valid':{},
                    'fixed_iu_valid':{},'predictions':{},'fixed_iu_predictions':{},'peaks':{},'sources':{},'routing':meta['routing']}
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
    mm=metric_module();fixed=fixed_rows(rows)
    metrics={arm:{'prm':mm.prm_metric(rows,arm),'pb':mm.pb_metric(rows,arm),'pb_common_iu_gate':mm.pb_metric(fixed,arm)} for arm in ARMS}
    previous=load(PARENT/'EVALUATION.json')
    for arm in PARENT_ARMS:assert metrics[arm]==previous['metrics'][arm],arm
    save(OUT/'EVALUATION.json',{'release_id':m['release_id'],'status':'DEVELOPMENT_QUALITY_COMPARISON','scores_sha256':sha(OUT/'SCORES_FROZEN.json'),
        'rows':rows,'metrics':metrics,'parent_metrics':previous['metrics'],'older_58_answer_metrics':previous['previous_cohort_metrics'],'labels_decoded':True})
    for arm in ARMS:print(arm,'PRM',metrics[arm]['prm']['auroc'],'PB',metrics[arm]['pb']['macro_f1'],flush=True)


def contrasts():
    m=verify();e=load(OUT/'EVALUATION.json');path=OUT/'CONTRASTS.json';digest=sha(OUT/'EVALUATION.json');started=time.monotonic()
    state=load(path) if path.exists() else {'evaluation_sha256':digest,'pairs':{}};assert state['evaluation_sha256']==digest
    mm=metric_module();fixed=fixed_rows(e['rows'])
    for left,right in m['contrasts']:
        key=left+' minus '+right
        if key in state['pairs']:continue
        common=[r for r in e['rows'] if r['valid'][left] and r['valid'][right]]
        state['pairs'][key]={'left':left,'right':right,'left_prm':mm.prm_metric(common,left),'right_prm':mm.prm_metric(common,right),
            'left_pb':mm.pb_metric(e['rows'],left),'right_pb':mm.pb_metric(e['rows'],right),
            'left_pb_common_iu_gate':mm.pb_metric(fixed,left),'right_pb_common_iu_gate':mm.pb_metric(fixed,right),
            'uncertainty':paired_source_group_intervals(e['rows'],left,right)}
        save(path,state)
    assert len(state['pairs'])==63;state.update(status='COMPLETE',seconds_this_invocation=time.monotonic()-started);save(path,state)
    print('All 63 paired comparisons complete.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=('prepare','scores','evaluate','contrasts'));args=p.parse_args()
    {'prepare':prepare,'scores':scores,'evaluate':evaluate,'contrasts':contrasts}[args.phase]()
