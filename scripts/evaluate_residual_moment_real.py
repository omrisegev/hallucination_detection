"""Separated full-data evaluation and independent audit of frozen residual fusion."""
from pathlib import Path
import sys, json, time, html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.temporal_research_features import BASELINE
from spectral_utils.context_training import FeatureBundle
from spectral_utils.contextual_iu import DEFAULT_IU_FIT
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.residual_moment_fusion import residuals
from scripts import run_temporal_research_baseline as base
from scripts.run_residual_moment_real import OUT, EXCLUSIONS, key, sha, write, Models

def independent_quality(records,joined,scores,gate,metrics,per):
    cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_')
    target=joined['target'];offsets=joined['offsets'];labels=joined['labels'];audit={}
    for name,flat in scores.items():
        peaks=np.array([int(np.argmax(flat[a:b])) for a,b in zip(offsets[:-1],offsets[1:])])
        prediction=np.where(gate,peaks,-1);cell_values={}
        for cell in sorted(set(cells[pb])):
            clean=(cells==cell)&(target<0);error=(cells==cell)&(target>=0)
            ca=np.mean(prediction[clean]==-1);ea=np.mean(prediction[error]==target[error])
            cell_values[cell]=0. if ca+ea==0 else 2*ca*ea/(ca+ea)
        pb_value=np.mean(list(cell_values.values()));aucs=[]
        for i in np.flatnonzero(~pb):
            sl=slice(offsets[i],offsets[i+1]);y=labels[sl];s=flat[sl]
            positive=s[y==1];negative=s[y==0]
            if len(positive) and len(negative):
                # Independent pairwise AUC, no rank helper/evaluator reuse.
                d=positive[:,None]-negative[None,:]
                aucs.append(float(np.mean((d>0)+.5*(d==0))))
        within=np.mean(aucs)
        np.testing.assert_allclose([pb_value,within],[metrics[name]['pb_all8'],metrics[name]['prm_within']],atol=2e-14,rtol=0)
        np.testing.assert_array_equal(peaks,per[name]['peak'])
        suppressed=int(np.sum(pb&(target>=0)&(peaks==target)&~gate))
        assert suppressed==metrics[name]['pb_correct_peaks_suppressed']
        audit[name]=dict(pb_independent=float(pb_value),within_pairwise=float(within),within_n=len(aucs),
            correct_peaks_suppressed=suppressed,valid_answers=len(peaks))
    return audit

def numeric_audit(bundle):
    models=Models(bundle);checks=0;max_coefficient_delta=0.;max_readout_delta=0.
    for f in range(5):
        with np.load(OUT/(key((f,))+'_DIAGNOSTICS.npz')) as diag,np.load(OUT/(key((f,))+'.npz')) as score:
            for j in np.linspace(0,len(diag['ids'])-1,6,dtype=int):
                i=int(diag['ids'][j]);m=bundle.metadata[i];n=bundle.length[i];start=bundle.offset[i]
                L=np.asarray(bundle.features[start:start+n],float)[:,bundle.columns]
                R=residuals(bundle,np.full(n,i),np.arange(n),models.get((f,)))
                spans=np.asarray(bundle.spans[m['step_start']:m['step_stop']])-start
                summaries={rep:np.array([np.sort(X[a:b],axis=0)[-min(10,b-a):].mean(0) for a,b in spans]) for rep,X in [('L',L),('R',R)]}
                for k,rep in enumerate(('L','R')):
                    ref=upcr_fit_covariance(diag['covariance'][j,k],var_y=float(diag['var_y'][j]),**DEFAULT_IU_FIT)
                    raw=ref.w/diag['sd'][j];raw/=np.abs(raw).sum()
                    delta=float(np.max(np.abs(raw-diag['weights'][j,k,0])));max_coefficient_delta=max(max_coefficient_delta,delta)
                    np.testing.assert_allclose(raw,diag['weights'][j,k,0],atol=2e-9,rtol=1e-7)
                    for h,head in enumerate(('native','simplex')):
                        for ss in ('L','R'):
                            expected=summaries[ss]@diag['weights'][j,k,h]
                            actual=score[f'local__{head}__{rep}{ss}'][m['step_start']:m['step_stop']]
                            max_readout_delta=max(max_readout_delta,float(np.max(np.abs(expected-actual))))
                            np.testing.assert_allclose(expected,actual,atol=2e-12,rtol=0)
                    checks+=1
    return dict(canonical_fits=checks,max_raw_coefficient_delta=max_coefficient_delta,
        independently_sorted_step_readout_max_delta=max_readout_delta)

def run():
    start=time.perf_counter();state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status'] not in ('SCORING_COMPLETE_PENDING_EVALUATION','COMPLETE_REVIEWED'):raise ValueError('Scoring not complete')
    source=ROOT.parents[1];records,joined=base.load_contract(source)
    bundle=FeatureBundle(ROOT/'results/temporal_context_data_v1','innovation5');meta=bundle.metadata
    if [r['uid'] for r in records]!=[m['uid'] for m in meta]:raise ValueError('Roster mismatch')
    total=int(joined['offsets'][-1]);scores={};nested={f:{} for f in range(5)}
    for e in EXCLUSIONS:
        path=OUT/(key(e)+'.npz')
        if sha(path)!=json.loads((OUT/(key(e)+'_COMPLETE.json')).read_text())['scores_sha256']:raise ValueError('Changed prediction checkpoint')
        with np.load(path,allow_pickle=False) as archive:
            for name in archive.files:
                value=archive[name];finite=np.isfinite(value)
                if len(e)==1:
                    if name not in scores:scores[name]=np.full(total,np.nan)
                    if np.isfinite(scores[name][finite]).any():raise ValueError('Overlapping outer score assignment')
                    scores[name][finite]=value[finite]
                else:
                    for f in e:
                        other=next(k for k in e if k!=f)
                        take=np.concatenate([value[m['step_start']:m['step_stop']] for m in meta if m['fold']==other and not m['cell'].startswith('pb_')])
                        if not np.isfinite(take).all():raise ValueError('Nested calibration coverage gap')
                        nested[f].setdefault(name,[]).append(take)
    if any(not np.isfinite(v).all() for v in scores.values()):raise ValueError('Incomplete full population')
    thresholds={name:{str(f):float(np.quantile(np.concatenate(nested[f][name]),.8)) for f in range(5)} for name in scores}
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as archive:
        gate=archive['gate_percentile']>=.33
        references={'original4':archive['steps__'+BASELINE],'innovation5':archive['steps__append_innovation__H0lim']}
        for feature in ('H0lim','VE0','VE075','VE1'):references['single__'+feature]=archive['steps__mean__'+feature]
        references['entropy15']=archive['steps__entropy15']
    with np.load(ROOT/'results/temporal_research_mechanism_v1/SCORES_FROZEN.npz') as archive:references['RBM12']=archive['steps__RBM12_logit']
    delta=float(np.max(np.abs(scores['equal__L']-references['innovation5'])))
    np.testing.assert_allclose(scores['equal__L'],references['innovation5'],atol=1e-6,rtol=0)
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,fold_auc=True,calibration_thresholds=thresholds,pb_gate_open=gate)
    rm,rp=base.evaluator.evaluate_arrays(records,joined,references,fold_auc=True,pb_gate_open=gate)
    metrics.update(rm);per.update(rp);scores.update(references)
    for metric in ('pb_all8','prm_within','prmscore_q08'):
        np.testing.assert_allclose(metrics['equal__L'][metric],metrics['innovation5'][metric],atol=2e-14,rtol=0)
    np.testing.assert_allclose([metrics['original4']['pb_all8'],metrics['original4']['prm_within'],metrics['original4']['prmscore_q08']],
        [.37474898261944,.7534358509472404,.6344124357811041],atol=2e-14,rtol=0)
    # A historical secondary ridge readout is an explicitly secondary anchor.
    with np.load(ROOT/'results/temporal_linear_context_v1/SCORES_FROZEN.npz') as archive:
        historical={'ridge_signed025_secondary':archive['steps__innovation5__real__signed_residual_0.25']}
    lm=json.loads((ROOT/'results/temporal_linear_context_v1/METRICS.json').read_text())['metrics']['innovation5__real__signed_residual_0.25']
    hm,hp=base.evaluator.evaluate_arrays(records,joined,historical,fold_auc=True,pb_gate_open=gate,
        calibration_thresholds={'ridge_signed025_secondary':lm['prmscore_thresholds']})
    metrics.update(hm);per.update(hp);scores.update(historical)
    primary=[]
    for scope in ('pooled','local'):
        for head in ('native','simplex'):
            a=f'{scope}__{head}__RR'
            primary.extend([(a,f'{scope}__{head}__LR'),(a,'equal__R')])
    secondary=[(name,'innovation5') for name in state['methods'] if name!='equal__L']
    secondary.extend([('equal__R','equal__L')])
    pairs=list(dict.fromkeys(primary+secondary))
    print('[evaluate] full-population metrics and anchors computed; source bootstrap10000',flush=True)
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(primary),primary_ci=.996875)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    print('[audit] independent PB and pairwise within for',len(scores),'methods',flush=True)
    audit=independent_quality(records,joined,scores,gate,metrics,per)
    numeric=numeric_audit(bundle)
    # Same per-answer grouping, with error positions used only for evaluation.
    cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_');target=joined['target']
    position=(target+.5)/np.diff(joined['offsets']);strata={}
    for name in state['methods']:
        rows={}
        for group,mask in [('early',pb&(target>=0)&(position<1/3)),('middle',pb&(target>=0)&(position>=1/3)&(position<2/3)),('late',pb&(target>=0)&(position>=2/3))]:
            hit=per[name]['prediction']==target;anchor=per['innovation5']['prediction']==target
            rows[group]=dict(answers=int(mask.sum()),hits=int(np.sum(mask&hit)),gained=int(np.sum(mask&hit&~anchor)),lost=int(np.sum(mask&~hit&anchor)))
        strata[name]=rows
    diagnostics={};fall=np.zeros(2,int);ratio=[];cosines=[];weight_cos=[]
    for f in range(5):
        with np.load(OUT/(key((f,))+'_DIAGNOSTICS.npz')) as d:
            fall+=d['native_fallback'].sum(0);ratio.extend(d['residual_variance_ratio'])
            r=d['rho'];cosines.extend(np.sum(r[:,0]*r[:,1],1)/np.maximum(np.linalg.norm(r[:,0],axis=1)*np.linalg.norm(r[:,1],axis=1),1e-20))
            w=d['weights'];weight_cos.extend(np.sum(w[:,0]*w[:,1],axis=2)/np.maximum(np.linalg.norm(w[:,0],axis=2)*np.linalg.norm(w[:,1],axis=2),1e-20))
    diagnostics=dict(local_native_fallback_counts=fall,residual_variance_ratio_median=np.median(ratio,axis=0),
        rho_level_residual_cosine_median=float(np.median(cosines)),weight_level_residual_cosine_median=np.median(weight_cos,axis=0))
    pareto=[]
    for name,m in metrics.items():
        dominated=any(n['pb_all8']>=m['pb_all8'] and n['prm_within']>=m['prm_within'] and
            (n['pb_all8']>m['pb_all8'] or n['prm_within']>m['prm_within']) for n in metrics.values())
        if not dominated:pareto.append(name)
    np.savez_compressed(OUT/'SCORES_FROZEN.npz',**{'steps__'+k:v for k,v in scores.items()})
    write(OUT/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,pareto=pareto,development_only=True,
        primary_endpoints=16,primary_ci=.996875,early_middle_late=strata,diagnostics=diagnostics))
    write(OUT/'AUDIT.json',dict(status='PASS',methods=len(scores),answers=len(records),steps=total,tokens=int(bundle.length.sum()),
        baseline_float32_max_score_difference=delta,anchor_metrics_exact=True,independent_metrics=audit,numeric=numeric,
        source_excluded_models=25,correctness_labels_in_fit=False,
        evaluation_sources={str(p):sha(p) for p in [source/'results/localization_full_benchmark_v3/evaluation/JOINED.json',source/'results/localization_full_benchmark_v3/evaluation/JOINED.npz',base.evaluator.old.FOLDS]},
        scores_sha256=sha(OUT/'SCORES_FROZEN.npz'),evaluation_code_sha256=sha(Path(__file__))))
    state.pop('quality_labels_used',None)
    state.update(status='COMPLETE_REVIEWED',evaluation_seconds=time.perf_counter()-start,pareto=pareto,
        fitting_correctness_labels_used=False,evaluation_correctness_labels_used=True,
        source_excluded_ridge_fits=dict(reused=15,new_triples=10),development_only=True)
    write(OUT/'RUN_STATE.json',state)
    for name,m in metrics.items():print('[result]',name,round(100*m['pb_all8'],4),round(m['prm_within'],6),round(m['prmscore_q08'],6),flush=True)
    print('[pareto]',pareto,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
