"""Create a compact, label-free memory-mapped bundle for context-model training."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import sqlite3
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_temporal_research_baseline as base
from spectral_utils.temporal_research_features import prefix_innovation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--baseline',type=Path,default=ROOT/'results/temporal_research_baseline_v1')
    p.add_argument('--out',type=Path,default=ROOT/'results/temporal_context_data_v1');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'MANIFEST.json').exists():
        saved=base.read_json(a.out/'MANIFEST.json')
        for name,digest in saved['files'].items():
            if base.common.sha256_file(a.out/name)!=digest:raise ValueError('existing context bundle changed: '+name)
        print('Existing label-free bundle verified');return
    if base.read_json(a.baseline/'BASELINE_REPLAY.json')['status']!='PASS':raise ValueError('baseline replay required')
    records,joined=base.load_contract(a.source_root)
    # Only unlabeled metadata is serialized below. Annotation arrays are never saved.
    folds=base.read_json(base.evaluator.old.FOLDS)['outer']
    total=sum(r['tokens'] for r in records)
    features=np.lib.format.open_memmap(a.out/'features.npy',mode='w+',dtype=np.float32,shape=(total,5))
    logits=np.lib.format.open_memmap(a.out/'logprobs15.npy',mode='w+',dtype=np.float32,shape=(total,15))
    prefix=np.lib.format.open_memmap(a.out/'h0_prefix_sum.npy',mode='w+',dtype=np.float64,shape=(total,))
    step_spans=np.zeros((int(joined['offsets'][-1]),2),np.int64)
    metadata=[];offset=0
    con=sqlite3.connect('file:'+str(a.baseline/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    try:
        for expected,(i,blob,info_json) in enumerate(con.execute('SELECT idx,payload,info FROM answers ORDER BY idx')):
            if i!=expected:raise ValueError('roster gap')
            info=json.loads(info_json);r=records[i]
            if info['uid']!=r['uid']:raise ValueError('UID mismatch')
            with np.load(io.BytesIO(blob),allow_pickle=False) as f:
                raw=f['features'];lp=f['logprobs15'];spans=f['spans']
            innovation,_=prefix_innovation(raw[:,0]);augmented=np.column_stack((raw,innovation))
            stop=offset+len(raw);features[offset:stop]=augmented;logits[offset:stop]=lp
            prefix[offset:stop]=np.r_[0.,np.cumsum(raw[:-1,0])]
            step_spans[joined['offsets'][i]:joined['offsets'][i+1]]=spans+offset
            metadata.append(dict(uid=r['uid'],cell=r['cell'],group_id=r['group_id'],fold=int(folds[r['group_id']]),
                offset=offset,tokens=len(raw),step_start=int(joined['offsets'][i]),step_stop=int(joined['offsets'][i+1]),
                mean=augmented.mean(axis=0).tolist(),scale=np.maximum(augmented.std(axis=0),1e-8).tolist(),signs=info['signs']))
            offset=stop
            if (i+1)%1000==0:print('[bundle]',i+1,len(records),flush=True)
        if len(metadata)!=13769 or offset!=total:raise ValueError('incomplete bundle')
    finally:con.close()
    features.flush();logits.flush();prefix.flush();del features,logits,prefix
    np.save(a.out/'step_spans.npy',step_spans)
    base.common.atomic_json(a.out/'METADATA.json',metadata)
    files=['features.npy','logprobs15.npy','h0_prefix_sum.npy','step_spans.npy','METADATA.json']
    manifest=dict(schema='label-free-temporal-context-data-v1',answers=len(records),tokens=total,steps=len(step_spans),
        features=['H0lim','VE0','VE075','VE1','H0lim_prefix_innovation'],
        banks={'original4':[0,1,2,3],'innovation5':[0,1,2,3,4]},
        files={n:base.common.sha256_file(a.out/n) for n in files},
        metadata_keys=list(metadata[0]),correctness_labels_in_bundle=False,
        access='whole-answer normalization and orientation; offline; source-group exclusion required for fitting',
        original_score_freeze=base.read_json(a.baseline/'SCORE_FREEZE.json'))
    base.common.atomic_json(a.out/'MANIFEST.json',manifest)
    print(base.common.dumps({k:manifest[k] for k in ('answers','tokens','steps','correctness_labels_in_bundle')}))


if __name__=='__main__':main()
