"""Evaluate complete neural outer/nested predictions; reject partial rosters."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import run_temporal_research_baseline as base
from scripts.run_temporal_neural_job import key
from spectral_utils.temporal_research_features import BASELINE
from spectral_utils.context_training import FeatureBundle


def assemble(root,method,bank,seeds,metadata,total):
    ensemble={};thresholds={};all_nested={h:{} for h in range(5)}
    for seed in seeds:
        outer={};nested={h:{} for h in range(5)}
        for excluded in [(h,) for h in range(5)]+list(combinations(range(5),2)):
            path=root/key(dict(method=method,bank=bank,seed=seed,excluded=excluded))/'scoring'
            status=base.read_json(path/'RUN_STATE.json');manifest=base.read_json(path/'MANIFEST.json')
            if status['status']!='SCORED' or manifest['smoke'] or status['answers']!=status['expected']:raise ValueError('incomplete or smoke quality input')
            with np.load(path/'STEP_SCORES.npz',allow_pickle=False) as f:
                for name in f.files:
                    values=f[name]
                    if len(excluded)==1:
                        if name not in outer:outer[name]=np.full(total,np.nan)
                        for m in metadata:
                            if m['fold']==excluded[0]:outer[name][m['step_start']:m['step_stop']]=values[m['step_start']:m['step_stop']]
                    else:
                        for held in excluded:
                            other=next(h for h in excluded if h!=held)
                            if name not in nested[held]:nested[held][name]=np.full(total,np.nan)
                            for m in metadata:
                                if m['fold']==other and not m['cell'].startswith('pb_'):
                                    nested[held][name][m['step_start']:m['step_stop']]=values[m['step_start']:m['step_stop']]
        for name,values in outer.items():
            if not np.isfinite(values).all():raise ValueError('missing full-population outer predictions')
            if name not in ensemble:ensemble[name]=np.zeros(total)
            ensemble[name]+=values/len(seeds)
            for held in range(5):
                if name not in all_nested[held]:all_nested[held][name]=np.zeros(total)
                all_nested[held][name]+=nested[held][name]/len(seeds)
    for name in ensemble:
        thresholds[name]={}
        for held in range(5):
            values=np.concatenate([all_nested[held][name][m['step_start']:m['step_stop']] for m in metadata if m['fold']!=held and not m['cell'].startswith('pb_')])
            if not np.isfinite(values).all():raise ValueError('missing nested calibration predictions')
            thresholds[name][str(held)]=float(np.quantile(values,.8))
    return ensemble,thresholds


def run(a):
    records,joined=base.load_contract(a.source_root);bundle=FeatureBundle(a.data);metadata=bundle.metadata
    if [r['uid'] for r in records]!=[m['uid'] for m in metadata]:raise ValueError('evaluation roster drift')
    scores={};thresholds={};total=int(joined['offsets'][-1]);seeds=tuple(int(x) for x in a.seeds.split(','))
    for bank in ('original4','innovation5'):
        for method in ('fm','diflo','tcn'):
            values,calibration=assemble(a.models,method,bank,seeds,metadata,total)
            scores.update({method+'__'+bank+'__'+n:v for n,v in values.items()})
            thresholds.update({method+'__'+bank+'__'+n:v for n,v in calibration.items()})
    with np.load(a.baseline/'SCORES_FROZEN.npz',allow_pickle=False) as f:
        gate=f['gate_percentile']>=.33
        reference={'original4__base':f['steps__'+BASELINE],'innovation5__base':f['steps__append_innovation__H0lim']}
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,fold_auc=True,calibration_thresholds=thresholds,pb_gate_open=gate)
    rm,rp=base.evaluator.evaluate_arrays(records,joined,reference,fold_auc=True,pb_gate_open=gate)
    metrics.update(rm);per.update(rp);scores.update(reference)
    flowpairs=[('diflo__'+b+'__real__dot_residual_0.25','fm__'+b+'__real__dot_residual_0.25') for b in ('original4','innovation5')]
    tcnpairs=[('tcn__'+b+'__real__squared_residual_0.25',b+'__base') for b in ('original4','innovation5')]
    contrasts={}
    for family,pairs in [('flow',flowpairs),('tcn',tcnpairs)]:
        result=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(pairs),primary_ci=.975)
        for x,y in pairs:result[x+'_minus_'+y]['pb_delta']=metrics[x]['pb_all8']-metrics[y]['pb_all8']
        contrasts[family]=result
    a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'SCORES_FROZEN.npz',**{'steps__'+k:v for k,v in scores.items()})
    base.common.atomic_json(a.out/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,seeds=seeds,development_only=True,
        uncertainty='Within-family two-comparison correction; no claim of whole adaptive-program familywise coverage.'))
    base.common.atomic_json(a.out/'RUN_STATE.json',dict(status='COMPLETE',answers=len(records),steps=total,seeds=seeds))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--models',type=Path,default=ROOT/'results/temporal_context_models_v1');p.add_argument('--data',type=Path,default=ROOT/'results/temporal_context_data_v1')
    p.add_argument('--baseline',type=Path,default=ROOT/'results/temporal_research_baseline_v1');p.add_argument('--out',type=Path,default=ROOT/'results/temporal_context_evaluation_v1')
    p.add_argument('--seeds',default='0,1,2');a=p.parse_args()
    with threadpool_limits(limits=1):run(a)
