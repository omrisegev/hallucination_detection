"""Freeze, score and evaluate one context-bank / Joint-K factorial pilot."""
import os
for option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[option]='1'
import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'results/answer_localization_representation_pilot_v1'
PEAK=ROOT/'results/fused_trajectory_readout_pilot_v1'
OUT=ROOT/'results/fusion_context_bank_pilot_v1'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_context_bank import ARMS,CORE_NAMES,JOINT_NAMES,PARENT_METHODS,REP,score_context_bank
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def safe(x):
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [safe(v) for v in x]
    if isinstance(x,(bool,np.bool_)):return bool(x)
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    return x
def save(path,x):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(safe(x),indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def registered_pairs():
    pairs=[('context__'+c,'moment__'+c) for c in CORE_NAMES]
    for bank in ('moment','context'):pairs += [(bank+'_allk__'+c,bank+'__'+c) for c in JOINT_NAMES]
    pairs += [('context_allk__'+c,'moment_allk__'+c) for c in JOINT_NAMES]
    pairs += [('context__'+a,'context__'+b) for a,b in (
        ('iu','equal'),('joint0','iu'),('graph010','joint0'),('graph010','graph_perm'),('graph010','iu'))]
    for bank in ('moment','context'):
        pairs += [(bank+'_allk__joint0',bank+'__iu'),(bank+'_allk__graph010',bank+'_allk__joint0'),
                  (bank+'_allk__graph010',bank+'_allk__graph_perm'),(bank+'_allk__graph010',bank+'__iu')]
    pairs += [('moment__'+a,'moment__'+b) for a,b in (
        ('iu','equal'),('graph010','iu'),('graph010','joint0'),('graph010','graph_perm'))]
    assert len(pairs)==len(set(pairs))==31
    return pairs


def verify():
    manifest=load(OUT/'MANIFEST.json')
    for p,h in manifest['hashes'].items():assert sha(p)==h,p
    return manifest


def prepare():
    if (OUT/'MANIFEST.json').exists():verify();print('Existing frozen protocol verified.');return
    p=load(PARENT/'PREPARED.json');f=load(PARENT/'SCORES_FROZEN.json')
    assert f['prepared_sha256']==sha(PARENT/'PREPARED.json')
    hashes={**p['source_hashes'],**f['files']}
    for name in ('PREPARED.json','SCORES_FROZEN.json','RELEASE.json'):hashes[str(PARENT/name)]=sha(PARENT/name)
    for name in ('MANIFEST.json','SCORES_FROZEN.json','EVALUATION.json'):hashes[str(PEAK/name)]=sha(PEAK/name)
    for path in (Path(__file__),ROOT/'spectral_utils/fusion_context_bank.py',ROOT/'spectral_utils/fusion_benchmark_bootstrap.py',
        ROOT/'tests/test_fusion_context_bank.py',ROOT/'docs/experiments/FUSION_CONTEXT_BANK_PILOT_V1.md',
        ROOT/'spectral_utils/unified_causal_iu.py',ROOT/'spectral_utils/unified_causal_subset_search.py'):
        hashes[str(path)]=sha(path)
    for rec in p['selected']:hashes[str(PARENT/'inputs'/f"{rec['uid']}.npz")]=rec['input_sha256']
    for path,digest in hashes.items():assert sha(path)==digest,path
    save(OUT/'MANIFEST.json',{'release_id':p['release_id'],'selected':p['selected'],'arms':ARMS,
        'contrasts':registered_pairs(),'hashes':hashes,'created_unix':time.time(),'labels_decoded':False,
        'status':'ADAPTIVE_DEVELOPMENT','worker_cap':3,'grouping_factors':['legacy','all_admissible_k'],
        'reference_gate':'same parent IU binary decision for every core, same answer'})
    print('Frozen 17 arms / 31 contrasts on the existing 58-answer release.',flush=True)


def process_one(rec,digest,release_id):
    started=time.monotonic();uid=rec['uid'];p=OUT/'scores'/f'{uid}.npz';mp=p.with_suffix('.json')
    if mp.exists():
        m=load(mp);assert m['manifest_sha256']==digest and m['array_sha256']==sha(p)
        return {'uid':uid,'reused':True}
    parent=load(PARENT/'scores'/f'{uid}.json')
    with np.load(PARENT/'inputs'/f'{uid}.npz',allow_pickle=False) as raw:values=raw['raw'].copy()
    with np.load(PARENT/'scores'/f'{uid}.npz',allow_pickle=False) as old:
        arrays,methods,diagnostics=score_context_bank(values,old,parent,release_id+'/'+rec['cell']+'/'+rec['row_id'])
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
    p.with_suffix('.npz.tmp').replace(p)
    save(mp,{**rec,'methods':methods,'diagnostics':diagnostics,'manifest_sha256':digest,'array_sha256':sha(p),
        'labels_decoded':False,'seconds':time.monotonic()-started})
    return {'uid':uid,'valid':sum(d['valid'] for d in methods.values()),'seconds':time.monotonic()-started}


def scores(workers):
    manifest=verify();digest=sha(OUT/'MANIFEST.json');started=time.monotonic()
    if not 1<=workers<=3:raise ValueError('At most three workers')
    if (OUT/'SCORES_FROZEN.json').exists():
        frozen=load(OUT/'SCORES_FROZEN.json');assert frozen['manifest_sha256']==digest
        for p,h in frozen['files'].items():assert sha(p)==h,p
        print('Existing scores frozen and verified.');return
    selected=manifest['selected'];save(OUT/'RUN_STATE.json',{'state':'RUNNING','pid':os.getpid(),'completed':0,'total':len(selected)})
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(process_one,rec,digest,manifest['release_id']) for rec in selected]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            print(f'{i}/{len(futures)} {json.dumps(future.result())}',flush=True)
            save(OUT/'RUN_STATE.json',{'state':'RUNNING','pid':os.getpid(),'completed':i,'total':len(selected),'seconds':time.monotonic()-started})
    verify();paths=sorted((OUT/'scores').glob('*.npz'))+sorted((OUT/'scores').glob('*.json'));assert len(paths)==116
    save(OUT/'SCORES_FROZEN.json',{'manifest_sha256':digest,'files':{str(p):sha(p) for p in paths},
        'labels_decoded':False,'seconds':time.monotonic()-started})
    save(OUT/'RUN_STATE.json',{'state':'COMPLETE','completed':58,'total':58,'seconds':time.monotonic()-started})
    print('All scores and decisions frozen before evaluation.',flush=True)


def metric_module():
    spec=importlib.util.spec_from_file_location('parent_metrics',ROOT/'scripts/run_answer_localization_v2.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def with_fixed_gate(rows):
    return [{**r,'decision_valid':r['fixed_iu_valid'],'predictions':r['fixed_iu_predictions']} for r in rows]


def evaluate():
    manifest=verify();frozen=load(OUT/'SCORES_FROZEN.json');assert frozen['manifest_sha256']==sha(OUT/'MANIFEST.json')
    for p,h in frozen['files'].items():assert sha(p)==h,p
    release=load(PARENT/'RELEASE.json');rows=[]
    for cell in sorted({r['cell'] for r in manifest['selected']}):
        info=release['cells'][cell];assert sha(info['label_path'])==info['label_opaque_sha256']
        with np.load(info['label_path'],allow_pickle=False) as labels:
            positions={str(r):i for i,r in enumerate(labels['row_ids'])};assert len(positions)==len(labels['row_ids'])
            for rec in (r for r in manifest['selected'] if r['cell']==cell):
                i=positions[rec['row_id']]
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2];target=labels['step_error_flags'][a:b].copy()
                else:target=int(labels['first_error'][i])
                meta=load(OUT/'scores'/f"{rec['uid']}.json")
                row={**rec,'target':target,'scores':{},'valid':{},'decision_valid':{},'predictions':{},
                    'fixed_iu_valid':{},'fixed_iu_predictions':{},'peaks':{}}
                with np.load(OUT/'scores'/f"{rec['uid']}.npz",allow_pickle=False) as arrays:
                    for arm,d in meta['methods'].items():
                        for field in ('valid','decision_valid','fixed_iu_valid'):row[field][arm]=d[field]
                        row['predictions'][arm]=d.get('prediction');row['fixed_iu_predictions'][arm]=d.get('fixed_iu_prediction')
                        row['peaks'][arm]=d.get('peak')
                        if d['valid']:
                            x=arrays[arm+'__risk'];assert len(x)==rec['steps'] and np.isfinite(x).all();row['scores'][arm]=x
                rows.append(row)
    m=metric_module();fixed=with_fixed_gate(rows)
    metrics={arm:{'prm':m.prm_metric(rows,arm),'pb':m.pb_metric(rows,arm),'pb_common_iu_gate':m.pb_metric(fixed,arm)} for arm in ARMS}
    peak=load(PEAK/'EVALUATION.json');replays=0
    for arm,old in {**{'moment__'+c:REP+'__'+v+'@@parent_peak' for c,v in PARENT_METHODS.items()},'entropy_parent':'entropy_mean_w8@@parent_peak'}.items():
        for task in ('prm','pb'):assert metrics[arm][task]==peak['metrics'][old][task];replays+=1
    save(OUT/'EVALUATION.json',{'status':'ADAPTIVE_DEVELOPMENT','scores_sha256':sha(OUT/'SCORES_FROZEN.json'),
        'rows':rows,'metrics':metrics,'labels_decoded':True,'parent_endpoint_replays':replays})
    print('Evaluated; parent endpoints replayed:',replays,flush=True)
    for arm,me in metrics.items():print(arm,'PRM',me['prm'],'PB',me['pb']['macro_f1'],'fixed IU',me['pb_common_iu_gate']['macro_f1'],flush=True)


def contrasts():
    started=time.monotonic();manifest=verify();e=load(OUT/'EVALUATION.json');digest=sha(OUT/'EVALUATION.json')
    p=OUT/'CONTRASTS.json';state=load(p) if p.exists() else {'evaluation_sha256':digest,'pairs':{}}
    assert state['evaluation_sha256']==digest;m=metric_module();rows=e['rows'];fixed=with_fixed_gate(rows)
    for left,right in manifest['contrasts']:
        key=left+' minus '+right
        if key in state['pairs']:continue
        common=[r for r in rows if r['valid'][left] and r['valid'][right]]
        state['pairs'][key]={'left':left,'right':right,'left_prm':m.prm_metric(common,left),'right_prm':m.prm_metric(common,right),
            'left_pb':m.pb_metric(rows,left),'right_pb':m.pb_metric(rows,right),
            'left_pb_common_iu_gate':m.pb_metric(fixed,left),'right_pb_common_iu_gate':m.pb_metric(fixed,right),
            'uncertainty':paired_source_group_intervals(rows,left,right)}
        save(p,state);print('Contrasts',len(state['pairs']),'/',31,flush=True)
    assert len(state['pairs'])==31;state.update(state='COMPLETE',seconds_this_invocation=time.monotonic()-started);save(p,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','scores','evaluate','contrasts'),required=True)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    {'prepare':prepare,'scores':lambda:scores(args.workers),'evaluate':evaluate,'contrasts':contrasts}[args.phase]()
