"""Frozen family-tail transfer on existing external telemetry; no label access."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
import time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.family_external_features import step_features
from spectral_utils.family_tail_transfer import score_locked,load_lock,LOCK_SHA256
from spectral_utils.external_generalization.artifacts import RecordStore,atomic_json,file_hash

OUT=ROOT/'results/family_tail_external_v1'
OLD=ROOT/'results/lsml_external_generalization_v1/evaluation'
PRIVATE=ROOT/'scratch/external_generalization_private/evaluation_archives'
CELLS=('hard2verify_qwen3_8b','socratic_qwen3_8b','socratic_qwq32b')
ANCHORS={'B11_lsml':'frozen_lsml','B11_equal':'frozen_equal','B11_partition_equal':'frozen_partition_equal'}


def one(job):
    cell,path=job;started=time.perf_counter();cpu=time.process_time()
    record=json.loads(path.read_text());row=record['payload']['telemetry']
    oldpath=OLD/cell/'shard_000'/path.name
    oldrecord=json.loads(oldpath.read_text());old=oldrecord['payload']
    telemetry_hash=file_hash(path)
    if oldrecord['uid']!=record['uid'] or old['telemetry_sha256']!=telemetry_hash:
        raise ValueError('old reference telemetry identity mismatch')
    spans=np.asarray(row['step_token_spans'],int);valid=spans[:,1]>spans[:,0]
    features,names=step_features(row,spans[valid])
    methods=tuple(m for m in load_lock()['rows'] if m!='ct7')
    arms=score_locked(features,names,np.array([0,len(features)]),methods=methods)
    scores={};predictions={};errors={}
    for name,item in arms.items():
        values=np.full(len(spans),np.nan);values[valid]=item['scores']
        pred=np.zeros(len(spans),int);pred[valid]=item['pred_valid']
        scores[name]=[float(v) if np.isfinite(v) else None for v in values]
        predictions[name]=pred.tolist()
        if name in ANCHORS:
            ref=ANCHORS[name];refscore=np.asarray([np.nan if v is None else v for v in old['scores'][ref]])
            errors[name]=float(np.max(np.abs(values[valid]-refscore[valid])))
            if errors[name]>1e-6 or predictions[name]!=old['predictions'][ref]:
                raise ValueError('historical bank11 prediction replay failed: '+name)
    scores['ct7']=old['scores']['ct7'];predictions['ct7']=old['predictions']['ct7']
    if valid.tolist()!=old['nonempty']:raise ValueError('empty-step mask changed')
    return cell,record['uid'],{'scores':scores,'predictions':predictions,'nonempty':valid.tolist(),
        'features':features.tolist(),'feature_names':names,'telemetry_sha256':telemetry_hash,
        'reference_record_sha256':file_hash(oldpath),'reference_score_errors':errors,
        'ct7_provenance':'reused sealed historical prediction on identical telemetry',
        'tokens':len(row['gen_token_ids']),'cpu_seconds':time.perf_counter()-started,
        'process_cpu_seconds':time.process_time()-cpu}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    gate=json.loads((OUT/'source_full_v1/GATE.json').read_text())
    if gate['status']!='PASS' or gate['n_checked']!=13769 or gate['scope']!='FULL':
        raise ValueError('full source parity PASS required before external scoring')
    freeze=json.loads((OUT/'IMPLEMENTATION_FREEZE.json').read_text())
    for rel,expected in freeze['files'].items():
        if file_hash(ROOT/rel)!=expected:raise ValueError('changed frozen code: '+rel)
    identity={'lock':LOCK_SHA256,'source_gate':file_hash(OUT/'source_full_v1/GATE.json'),
              'implementation_freeze':file_hash(OUT/'IMPLEMENTATION_FREEZE.json'),
              'code':freeze['files'],'arms':list(load_lock()['rows'])}
    stores={};jobs=[];started=time.perf_counter()
    try:
        for cell in CELLS:
            store=RecordStore(OUT/cell/'shard_000',identity);store.__enter__();stores[cell]=store
            paths=sorted((PRIVATE/cell/'records').glob('*.record.json'))
            if len(paths)!=(200 if cell.startswith('hard') else 2995):raise ValueError('incomplete raw population')
            for path in paths:
                if not (store.directory/path.name).exists():jobs.append((cell,path))
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            for n,(cell,uid,payload) in enumerate(executor.map(one,jobs,chunksize=1)):
                stores[cell].put(uid,payload)
                if n%50==0:print('external predictions',n+1,'/',len(jobs),cell,'seconds',round(time.perf_counter()-started),flush=True)
        atomic_json(OUT/'CPU_EXECUTION.json',{'identity':identity,'workers':args.workers,
                    'records_this_run':len(jobs),'elapsed_seconds':time.perf_counter()-started,'new_gpu_hours':0})
    finally:
        for store in stores.values():store.__exit__(None,None,None)


if __name__=='__main__':main()
