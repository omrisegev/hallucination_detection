"""Complete-population complementarity, position and DUFS stability diagnostics."""
from collections import Counter,defaultdict
from itertools import combinations
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import run_temporal_research_baseline as base
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_research_features import BASELINE,SUBSETS,FEATURES
from scripts.analyze_temporal_research_baseline import independent_pb
from spectral_utils.renyi_alpha_sweep import VIEW_NAMES


def run(source):
    records,joined=base.load_contract(source);root=ROOT/'results'
    bundle=FeatureBundle(root/'temporal_context_data_v1');metadata=bundle.metadata
    out=root/'temporal_feature_diagnostics_v1';out.mkdir(parents=True,exist_ok=True)
    with np.load(root/'temporal_research_baseline_v1/PREDICTIONS.npz',allow_pickle=False) as f:
        pred={n:f['prediction__'+n] for n in SUBSETS}
        peak={n:f['peak__'+n] for n in SUBSETS}
    target=np.asarray(joined['target']);cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_');error=pb&(target>=0)
    changes={}
    for n,indices in SUBSETS.items():
        for j in range(4):
            if j in indices:continue
            expanded=tuple(sorted((*indices,j)))
            other=next(k for k,v in SUBSETS.items() if tuple(v)==expanded)
            old=(pred[n]==target)&error;new=(pred[other]==target)&error
            changes[n+'__add_'+FEATURES[j]]=dict(gained=int((new&~old).sum()),lost=int((old&~new).sum()),shared=int((old&new).sum()))
    position_counts=np.zeros(16,int);first=np.zeros((16,5));second=np.zeros((16,5,5));top_positions=defaultdict(list)
    for i,m in enumerate(metadata):
        a=m['offset'];n=m['tokens'];x=np.asarray(bundle.features[a:a+n],float)
        bins=np.minimum(15,np.arange(n)*16//n)
        for k in range(16):
            rows=x[bins==k];position_counts[k]+=len(rows);first[k]+=rows.sum(0);second[k]+=rows.T@rows
        for lo,hi in np.asarray(bundle.spans[m['step_start']:m['step_stop']])-a:
            for j in range(5):
                # Same largest-k set as step_top_mean, index-stable handling of ties.
                ids=np.lexsort((np.arange(hi-lo),-x[lo:hi,j]))[:min(10,hi-lo)]+lo
                top_positions[str(j)+'__answer'].extend(((ids+.5)/n).tolist())
                top_positions[str(j)+'__step'].extend(((ids-lo+.5)/(hi-lo)).tolist())
        if (i+1)%2000==0:print('[feature-diagnostics]',i+1,len(metadata),flush=True)
    mean=first/position_counts[:,None]
    covariance=second/position_counts[:,None,None]-mean[:,:,None]*mean[:,None,:]
    std=np.sqrt(np.maximum(np.diagonal(covariance,axis1=1,axis2=2),0))
    correlation=np.divide(covariance,std[:,:,None]*std[:,None,:],out=np.zeros_like(covariance),where=(std[:,:,None]*std[:,None,:])>1e-12)
    strata={}
    for n in pred:
        rows={}
        for k in range(4):
            selection=error&np.array([min(3,int(4*(max(0,t)+.5)/(m['step_stop']-m['step_start'])))==k for t,m in zip(target,metadata)])
            rows[str(k)]=dict(answers=int(selection.sum()),hits=int(((pred[n]==target)&selection).sum()),
                              early=int(((peak[n]<target)&selection).sum()),late=int(((peak[n]>target)&selection).sum()))
        strata[n]=rows
    diagnostics=dict(answers=len(records),tokens=int(position_counts.sum()),features=list(FEATURES)+['H0lim_prefix_innovation'],
        add_feature_gains_losses=changes,first_error_position_quartiles=strata,
        negative_orientation_fraction=np.mean(np.array([m['signs'] for m in metadata])<0,axis=0),
        actual_token_position=dict(count=position_counts,mean=mean,std=std,correlation=correlation),
        top10_position_quantiles={k:np.quantile(v,[.1,.25,.5,.75,.9]) for k,v in top_positions.items()},
        caveats=['PB labels identify the first error only; later steps are not assigned error labels.',
                 'First-error strata describe answers; actual-token profiles describe feature distributions, not token correctness.',
                 'Pooled token covariances include answer-level offsets; centered within-answer correlations are reported separately.'])
    within_cov=np.zeros((5,5));answers=0
    for m in metadata:
        x=np.asarray(bundle.features[m['offset']:m['offset']+m['tokens']],float)
        centered=x-x.mean(0);scale=x.std(0);active=scale>1e-10
        z=np.divide(centered,scale,out=np.zeros_like(x),where=active)
        within_cov+=z.T@z/len(z);answers+=1
    diagnostics['mean_within_answer_standardized_second_moment']=within_cov/answers
    base.common.atomic_json(out/'FEATURE_DIAGNOSTICS.json',diagnostics)
    fits=base.read_json(root/'temporal_dufs31_v1/SELECTORS.json');stability={}
    for cell in sorted({f['cell'] for f in fits.values()}):
        outer=[f for f in fits.values() if f['cell']==cell and len(f['excluded_folds'])==1]
        stability[cell]={}
        for k in ('2','3','4'):
            sets=[set(f['selected'][k]) for f in outer];count=Counter(j for s in sets for j in s)
            stability[cell][k]=dict(folds=len(sets),pairwise_jaccard=[len(a&b)/len(a|b) for a,b in combinations(sets,2)],
                                   selected_frequency={VIEW_NAMES[j]:n/len(sets) for j,n in count.items()})
    base.common.atomic_json(out/'DUFS_STABILITY.json',stability)
    # Independent arithmetic on all newly selected-bank PB predictions.
    with np.load(root/'temporal_dufs31_v1/SCORES_FROZEN.npz',allow_pickle=False) as f:
        scores={k[7:]:f[k] for k in f.files if k.startswith('steps__dufs')}
    with np.load(root/'temporal_research_baseline_v1/SCORES_FROZEN.npz',allow_pickle=False) as f:opened=f['gate_percentile']>=.33
    metrics=base.read_json(root/'temporal_dufs31_v1/METRICS.json')['metrics'];audit={}
    for name,values in scores.items():
        peaks=np.array([int(np.argmax(values[m['step_start']:m['step_stop']])) for m in metadata])
        prediction=np.where(opened,peaks,-1);table,macro=independent_pb(target,cells,prediction,np.ones(len(records),bool))
        np.testing.assert_allclose(macro,metrics[name]['pb_all8'],rtol=0,atol=1e-14)
        audit[name]=dict(macro=macro,cells=table)
    base.common.atomic_json(out/'DUFS_INDEPENDENT_PB_AUDIT.json',dict(status='PASS',methods=audit))
    base.common.atomic_json(out/'RUN_STATE.json',dict(status='COMPLETE',answers=len(records)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args();run(a.source_root)
