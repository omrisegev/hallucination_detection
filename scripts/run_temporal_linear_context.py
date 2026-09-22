"""Full-population source-disjoint linear context reference with nested calibration."""
from __future__ import annotations
import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_context_models import RidgePredictor,history_windows
from spectral_utils.context_readout import prediction_readouts
from spectral_utils.temporal_research_features import BASELINE
from scripts import run_temporal_research_baseline as base


def design(bundle, ids, positions):
    rows=[];targets=[]
    for i,p in zip(ids,positions):
        start=bundle.offset[i];n=bundle.length[i]
        local=p-np.arange(16,0,-1);mask=local>=0
        raw=bundle.features[start+np.maximum(local,0)][:,bundle.columns]
        z=(raw-bundle.mean[i])/bundle.scale[i];z[~mask]=0
        rows.append(np.concatenate((z.ravel(),mask,[(p+.5)/n])))
        targets.append((bundle.features[start+p,bundle.columns]-bundle.mean[i])/bundle.scale[i])
    return np.asarray(rows),np.asarray(targets)


def fit(bundle,excluded,path):
    train,validation,held=bundle.split(excluded)
    if path.exists():
        with np.load(path,allow_pickle=False) as f:return RidgePredictor(f['coefficient'],f['mean'],f['scale'])
    rng=np.random.default_rng(20260915+sum(11**f for f in excluded))
    ids,pos=bundle.sampler(train,rng,16384)
    x,y=design(bundle,ids,pos);model=RidgePredictor.fit(x,y,ridge=1.)
    vi,vp=bundle.sampler(validation,np.random.default_rng(991),4096);vx,vy=design(bundle,vi,vp)
    np.savez_compressed(path,coefficient=model.coefficient,mean=model.mean,scale=model.scale)
    base.common.atomic_json(path.with_suffix('.json'),dict(excluded_folds=excluded,
        training_groups=sorted({bundle.metadata[i]['group_id'] for i in train}),
        validation_groups=sorted({bundle.metadata[i]['group_id'] for i in validation}),
        fit_mse=float(np.square(model.predict(x)-y).mean()),validation_mse=float(np.square(model.predict(vx)-vy).mean()),
        held_groups=sorted({bundle.metadata[i]['group_id'] for i in held}),correctness_labels_used=False))
    return model


def answer_scores(bundle,i,model,frozen):
    start=bundle.offset[i];n=bundle.length[i];m=bundle.metadata[i]
    raw=np.asarray(bundle.features[start:start+n])[:,bundle.columns].astype(float)
    z=(raw-bundle.mean[i])/bundle.scale[i]
    windows,mask=history_windows(z,np.arange(n),history=16)
    position=(np.arange(n)+.5)/n
    spans=np.asarray(bundle.spans[m['step_start']:m['step_stop']])-start
    bankbase=frozen[m['step_start']:m['step_stop']]
    # Same whole-answer draw for every model/seed; complete vectors move together.
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(m['uid'].encode()).digest()[:4],'little'))
    shuffled=windows.copy()
    for t in range(n):
        observed=np.flatnonzero(mask[t]);shuffled[t,observed]=windows[t,rng.permutation(observed)]
    output={}
    for name,history in [('real',windows),('shuffled',shuffled),('zero',np.zeros_like(windows))]:
        x=np.column_stack((history.reshape(n,-1),mask,position))
        pred=model.predict(x)
        if not np.isfinite(pred).all():raise FloatingPointError('linear fit produced nonfinite predictions')
        output.update({name+'__'+k:v for k,v in prediction_readouts(raw,z,pred,spans,bankbase).items()})
    return output


def run(a):
    records,joined=base.load_contract(a.source_root)
    total=int(joined['offsets'][-1]);meta=FeatureBundle(a.data).metadata
    if [m['uid'] for m in meta]!=[r['uid'] for r in records]:raise ValueError('bundle/contract roster mismatch')
    manifest=dict(schema='temporal-linear-context-v1',answers=len(records),banks=['original4','innovation5'],
        samples=16384,ridge=1.,source_hash=base.common.sha256_file(a.data/'MANIFEST.json'),
        code_hash={str(p.relative_to(ROOT)):base.common.sha256_file(p) for p in
                   [Path(__file__),ROOT/'spectral_utils/context_readout.py',ROOT/'spectral_utils/context_training.py']})
    if (a.out/'MANIFEST.json').exists() and base.read_json(a.out/'MANIFEST.json')!=manifest:raise ValueError('immutable run manifest changed')
    base.common.atomic_json(a.out/'MANIFEST.json',manifest)
    with np.load(a.baseline/'SCORES_FROZEN.npz',allow_pickle=False) as f:
        gate=f['gate_percentile']>=.33
        frozen={'original4':f['steps__'+BASELINE],'innovation5':f['steps__append_innovation__H0lim']}
    scores={};thresholds={};began=time.perf_counter()
    for bank in manifest['banks']:
        bundle=FeatureBundle(a.data,bank)
        for excluded in [(h,) for h in range(5)]+list(combinations(range(5),2)):
            key=bank+'__exclude_'+'_'.join(map(str,excluded));path=a.out/(key+'.npz')
            if not path.exists():
                model=fit(bundle,excluded,a.out/(key+'_model.npz'));arrays={}
                ids=[i for i,m in enumerate(meta) if m['fold'] in excluded and (len(excluded)==1 or not m['cell'].startswith('pb_'))]
                for counter,i in enumerate(ids):
                    sl=slice(meta[i]['step_start'],meta[i]['step_stop'])
                    for name,values in answer_scores(bundle,i,model,frozen[bank]).items():
                        if name not in arrays:arrays[name]=np.full(total,np.nan)
                        arrays[name][sl]=values
                    if (counter+1)%1000==0:print('[linear-score]',key,counter+1,len(ids),flush=True)
                with path.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
                path.with_suffix('.tmp').replace(path)
            print('[linear-complete]',key,flush=True)
        outer={};nested={h:{} for h in range(5)}
        for excluded in [(h,) for h in range(5)]+list(combinations(range(5),2)):
            key=bank+'__exclude_'+'_'.join(map(str,excluded))
            with np.load(a.out/(key+'.npz'),allow_pickle=False) as f:
                for name in f.files:
                    values=f[name];finite=np.isfinite(values)
                    if len(excluded)==1:
                        if name not in outer:outer[name]=np.full(total,np.nan)
                        outer[name][finite]=values[finite]
                    else:
                        for held in excluded:
                            other=next(h for h in excluded if h!=held)
                            if name not in nested[held]:nested[held][name]=[]
                            take=np.concatenate([values[m['step_start']:m['step_stop']] for m in meta if m['fold']==other and not m['cell'].startswith('pb_')])
                            if not np.isfinite(take).all():raise ValueError('nested calibration gap')
                            nested[held][name].append(take)
        for name,values in outer.items():
            if not np.isfinite(values).all():raise ValueError('full quality population not covered')
            full=bank+'__'+name;scores[full]=values
            thresholds[full]={str(h):float(np.quantile(np.concatenate(nested[h][name]),.8)) for h in range(5)}
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,fold_auc=True,calibration_thresholds=thresholds,pb_gate_open=gate)
    reference={bank+'__base':values for bank,values in frozen.items()}
    rm,rp=base.evaluator.evaluate_arrays(records,joined,reference,fold_auc=True,pb_gate_open=gate)
    metrics.update(rm);per.update(rp);scores.update(reference)
    pairs=[(bank+'__real__squared_residual_0.25',bank+'__base') for bank in frozen]
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(pairs),primary_ci=.975)
    for x,y in pairs:contrasts[x+'_minus_'+y]['pb_delta']=metrics[x]['pb_all8']-metrics[y]['pb_all8']
    np.savez_compressed(a.out/'SCORES_FROZEN.npz',**{'steps__'+k:v for k,v in scores.items()})
    base.common.atomic_json(a.out/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,development_only=True))
    base.common.atomic_json(a.out/'RUN_STATE.json',dict(status='COMPLETE',answers=len(records),steps=total,seconds=time.perf_counter()-began))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True);p.add_argument('--data',type=Path,default=ROOT/'results/temporal_context_data_v1')
    p.add_argument('--baseline',type=Path,default=ROOT/'results/temporal_research_baseline_v1');p.add_argument('--out',type=Path,default=ROOT/'results/temporal_linear_context_v1')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    try:
        with threadpool_limits(limits=1):run(a)
    except BaseException as e:
        base.common.atomic_json(a.out/'RUN_STATE.json',dict(status='FAILED',error=f'{type(e).__name__}: {e}'));raise


if __name__=='__main__':main()
