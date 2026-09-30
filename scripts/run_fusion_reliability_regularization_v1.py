"""Freeze, score and evaluate the bounded fusion sensitivity regularization experiment."""
import os
for _option in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_option]='1'
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
OUT=ROOT/'results/fusion_reliability_regularization_v1'
PROTOCOL=ROOT/'docs/experiments/FUSION_RELIABILITY_REGULARIZATION_V1.md'
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.fusion_reliability_regularization import CORES,FAMILIES,ARMS,score_reliability


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def safe(value):
    if isinstance(value,dict): return {str(k):safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list,np.ndarray)): return [safe(v) for v in value]
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.bool_,)): return bool(value)
    if isinstance(value,(np.floating,float)): return float(value) if np.isfinite(value) else None
    return value
def save(path,value):
    path=Path(path); path.parent.mkdir(exist_ok=True,parents=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(safe(value),indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def verify():
    manifest=load(OUT/'MANIFEST.json')
    for filename,expected in manifest['hashes'].items():
        if sha(filename)!=expected: raise RuntimeError('SOURCE_DRIFT: '+filename)
    return manifest


def prepare():
    if (OUT/'MANIFEST.json').exists(): verify(); print('Existing manifest verified; unchanged.'); return
    parent=load(PARENT/'PREPARED.json'); frozen=load(PARENT/'SCORES_FROZEN.json')
    assert frozen['prepared_sha256']==sha(PARENT/'PREPARED.json')
    hashes={**parent['source_hashes'],**frozen['files']}
    for name in ('PREPARED.json','SCORES_FROZEN.json','RELEASE.json'):
        hashes[str(PARENT/name)]=sha(PARENT/name)
    for path in (Path(__file__),PROTOCOL,ROOT/'spectral_utils/fusion_reliability_regularization.py',
                 ROOT/'spectral_utils/fusion_window_sampling.py',ROOT/'tests/test_fusion_reliability_regularization.py'):
        hashes[str(path)]=sha(path)
    for r in parent['selected']:
        hashes[str(PARENT/'inputs'/f"{r['uid']}.npz")]=r['input_sha256']
    for filename,expected in hashes.items():
        if sha(filename)!=expected: raise RuntimeError('PARENT_DRIFT: '+filename)
    assert parent['release_manifest_sha256']==sha(PARENT/'RELEASE.json')
    save(OUT/'MANIFEST.json',{'release_id':parent['release_id'],'selected':parent['selected'],
         'cores':CORES,'families':FAMILIES,'arms':ARMS,'hashes':hashes,'new_version_labels_decoded':False,
         'exposure':'Previously evaluated parent development answers; adaptive follow-up',
         'created_unix':time.time(),'worker_cap':3})
    print('Frozen protocol/code/parent; selected answers:',len(parent['selected']),flush=True)


def process_one(record,manifest_hash):
    started=time.monotonic(); uid=record['uid']
    array_path,meta_path=OUT/'scores'/f'{uid}.npz',OUT/'scores'/f'{uid}.json'
    if meta_path.exists():
        meta=load(meta_path)
        if meta['manifest_sha256']!=manifest_hash or sha(array_path)!=meta['array_sha256']:
            raise RuntimeError('CHECKPOINT_DRIFT: '+uid)
        return {'uid':uid,'state':'REUSED'}
    parent=load(PARENT/'scores'/f'{uid}.json')
    with np.load(PARENT/'inputs'/f'{uid}.npz',allow_pickle=False) as z:
        raw=z['raw'].copy()
    identity=record['cell']+'/'+record['row_id']
    with np.load(PARENT/'scores'/f'{uid}.npz',allow_pickle=False) as z:
        arrays,details,regularization=score_reliability(z,parent,raw,identity)
    array_path.parent.mkdir(exist_ok=True,parents=True)
    with array_path.with_suffix('.npz.tmp').open('wb') as stream: np.savez_compressed(stream,**arrays)
    array_path.with_suffix('.npz.tmp').replace(array_path)
    save(meta_path,{**record,'methods':details,'manifest_sha256':manifest_hash,'array_sha256':sha(array_path),
                    'seconds':time.monotonic()-started,'labels_decoded':False,'regularization':regularization})
    return {'uid':uid,'seconds':time.monotonic()-started,'valid':sum(v['valid'] for v in details.values())}


def scores(workers):
    manifest=verify()
    if not 1<=workers<=3: raise ValueError('At most three workers')
    started=time.monotonic(); selected=manifest['selected']; digest=sha(OUT/'MANIFEST.json')
    if (OUT/'SCORES_FROZEN.json').exists():
        frozen=load(OUT/'SCORES_FROZEN.json')
        for p,h in frozen['files'].items(): assert sha(p)==h
        print('Already frozen and verified; no rescore.'); return
    save(OUT/'RUN_STATE.json',{'state':'RUNNING','pid':os.getpid(),'completed':0,'total':len(selected)})
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        tasks=[pool.submit(process_one,r,digest) for r in selected]
        for i,future in enumerate(concurrent.futures.as_completed(tasks),1):
            print(f'{i}/{len(tasks)} {json.dumps(future.result())}',flush=True)
            save(OUT/'RUN_STATE.json',{'state':'RUNNING','pid':os.getpid(),'completed':i,'total':len(tasks),
                                      'seconds':time.monotonic()-started})
    verify()
    paths=sorted((OUT/'scores').glob('*.npz'))+sorted((OUT/'scores').glob('*.json'))
    assert len(paths)==2*len(selected)
    save(OUT/'SCORES_FROZEN.json',{'manifest_sha256':digest,'files':{str(p):sha(p) for p in paths},
                                 'labels_decoded':False,'seconds':time.monotonic()-started})
    save(OUT/'RUN_STATE.json',{'state':'COMPLETE','completed':len(selected),'total':len(selected),
                              'seconds':time.monotonic()-started})
    print('All new scores frozen before evaluation.',flush=True)


def metric_module():
    spec=importlib.util.spec_from_file_location('parent_metric_source',ROOT/'scripts/run_answer_localization_v2.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def evaluate():
    manifest=verify(); frozen=load(OUT/'SCORES_FROZEN.json')
    assert frozen['manifest_sha256']==sha(OUT/'MANIFEST.json') and frozen['labels_decoded'] is False
    for p,h in frozen['files'].items(): assert sha(p)==h
    release=load(PARENT/'RELEASE.json'); rows=[]
    for cell in sorted({r['cell'] for r in manifest['selected']}):
        info=release['cells'][cell]
        assert sha(info['label_path'])==info['label_opaque_sha256']
        with np.load(info['label_path'],allow_pickle=False) as z:
            positions={str(r):i for i,r in enumerate(z['row_ids'])}
            assert len(positions)==len(z['row_ids'])
            for record in (r for r in manifest['selected'] if r['cell']==cell):
                i=positions[record['row_id']]
                if cell.startswith('prm'):
                    a,b=z['step_flag_offsets'][i:i+2]; target=z['step_error_flags'][a:b].copy()
                    assert len(target)==record['steps']
                else: target=int(z['first_error'][i])
                meta=load(OUT/'scores'/f"{record['uid']}.json")
                row={**record,'target':target,'scores':{},'valid':{},'decision_valid':{},'predictions':{},'fixed_parent_predictions':{}}
                with np.load(OUT/'scores'/f"{record['uid']}.npz",allow_pickle=False) as arrays:
                    for arm,detail in meta['methods'].items():
                        row['valid'][arm]=row['decision_valid'][arm]=detail['valid']
                        row['predictions'][arm]=detail.get('prediction')
                        row['fixed_parent_predictions'][arm]=detail.get('fixed_parent_gate_prediction')
                        if detail['valid']:
                            score=arrays[arm+'__risk']
                            assert len(score)==record['steps'] and np.isfinite(score).all()
                            row['scores'][arm]=score
                rows.append(row)
    metrics=metric_module()
    names=list(ARMS)
    summary={name:{'prm':metrics.prm_metric(rows,name),'pb':metrics.pb_metric(rows,name)} for name in names}
    save(OUT/'EVALUATION.json',{'status':'ADAPTIVE_DEVELOPMENT','scores_sha256':sha(OUT/'SCORES_FROZEN.json'),
                              'metrics':summary,'rows':rows,'labels_decoded':True})
    print('Point estimates saved; paired contrasts are a separate resumable phase.',flush=True)
    for core in CORES:
        print(core,{r:round(summary[core+'__'+r]['pb']['macro_f1'],5) for r in FAMILIES},flush=True)


def contrast_job(rows,pair):
    metrics=metric_module(); left,right=pair
    common=[r for r in rows if r['valid'][left] and r['valid'][right]]
    return {'left':left,'right':right,'left_prm':metrics.prm_metric(common,left),
            'right_prm':metrics.prm_metric(common,right),'left_pb':metrics.pb_metric(rows,left),
            'right_pb':metrics.pb_metric(rows,right),'uncertainty':metrics.paired_intervals(rows,left,right)}


def contrasts(workers):
    verify(); evaluation=load(OUT/'EVALUATION.json')
    pairs=[]
    for core in CORES:
        pairs.extend((core+'__'+f,core+'_parent') for f in FAMILIES)
        pairs.extend((core+'__block_diag',core+'__'+f) for f in ('isotropic','block_diag_permuted','block_full'))
    for family in FAMILIES:
        pairs.extend((left+'__'+family,right+'__'+family) for left,right in (('iu','equal'),('joint','iu')))
    pairs.extend((left,'joint_graph010_parent') for left in ('joint__dufs_graph','joint_graph_fixed1','joint_graph_fixed10'))
    pairs.append(('joint__block_full','joint__dufs_graph'))
    destination=OUT/'CONTRASTS.json'; digest=sha(OUT/'EVALUATION.json')
    state=load(destination) if destination.exists() else {'evaluation_sha256':digest,'pairs':{}}
    assert state['evaluation_sha256']==digest
    if not 1<=workers<=3: raise ValueError('At most three workers')
    remaining=[p for p in pairs if ' minus '.join(p) not in state['pairs']]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(contrast_job,evaluation['rows'],p):p for p in remaining}
        for future in concurrent.futures.as_completed(futures):
            pair=futures[future]; state['pairs'][' minus '.join(pair)]=future.result()
            save(destination,state)
            print(f"Paired contrasts {len(state['pairs'])}/{len(pairs)}",flush=True)
    state['state']='COMPLETE';save(destination,state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=('prepare','scores','evaluate','contrasts'),required=True)
    parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    {'prepare':prepare,'scores':lambda:scores(args.workers),'evaluate':evaluate,'contrasts':lambda:contrasts(args.workers)}[args.phase]()
