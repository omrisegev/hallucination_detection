"""Evaluate frozen aligned predictions; this is the only correctness-label boundary."""
from pathlib import Path
import sys, json, time, html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.run_aligned_context_predictors import OUT, METHODS, sha, write
from scripts import run_temporal_research_baseline as base
from scripts.evaluate_residual_moment_real import independent_quality
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_research_features import BASELINE


def distribution(values):
    a=np.array([v for v in values if v is not None],float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
        quantiles=np.quantile(a,[.1,.5,.9]).tolist() if len(a) else None)


def mse_summary(meta,diags):
    result={};groups,inv=np.unique([m['group_id'] for m in meta],return_inverse=True)
    differences=[];valid=[]
    for name in METHODS:
        values=np.array([d['methods'][name]['mse'] for d in diags])
        late=np.array([d['methods'][name]['mse_after16'] if d['methods'][name]['mse_after16'] is not None else [np.nan]*5 for d in diags])
        result[name]=dict(per_feature_answer_mean=values.mean(0).tolist(),
            per_feature_token_mean=np.average(values,axis=0,weights=[m['tokens'] for m in meta]).tolist(),
            per_feature_after16_answer_mean=np.nanmean(late,axis=0).tolist(),
            scalar_mse_answer_mean=float(values.mean()),
            correlations={k:distribution([d['methods'][name][k] for d in diags]) for k in
                ('prediction_correlation_ridge','signed_residual_correlation_ridge',
                 'residual_correlation_after_removing_current','signed_residual_correlation_current')})
        reference=np.array([d['methods']['ridge']['mse'] for d in diags]).mean(1)
        delta=values.mean(1)-reference
        differences.append(np.bincount(inv,weights=delta,minlength=len(groups)))
    num=np.stack(differences,axis=1);den=np.bincount(inv,minlength=len(groups));draws=[]
    rng=np.random.default_rng(38720260915)
    for _ in range(100):
        W=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=100)
        draws.append((W@num)/(W@den)[:,None])
    draws=np.concatenate(draws)
    for j,name in enumerate(METHODS):
        result[name]['mse_minus_ridge_secondary95_ci']=np.quantile(draws[:,j],[.025,.975]).tolist()
    return result


def run():
    began=time.perf_counter();state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status'] not in ('SCORING_COMPLETE_PENDING_EVALUATION','COMPLETE_REVIEWED'):raise ValueError('scoring incomplete')
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for path,expected in manifest['source_hashes'].items():
        if sha(ROOT/path)!=expected:raise ValueError('Changed source '+path)
    bundle=FeatureBundle(ROOT/'results/temporal_context_data_v1','innovation5');meta=bundle.metadata
    for filename,expected in bundle.manifest['files'].items():
        if sha(ROOT/'results/temporal_context_data_v1'/filename)!=expected:raise ValueError('Bundle fingerprint mismatch')
    audit=json.loads((OUT/'SCORING_AUDIT.json').read_text())
    if sha(OUT/'SCORES_FROZEN.npz')!=audit['scores_sha256'] or sha(OUT/'DIAGNOSTICS.json')!=audit['diagnostics_sha256']:
        raise ValueError('Frozen output mismatch')
    records,joined=base.load_contract(ROOT.parents[1])
    if [r['uid'] for r in records]!=[m['uid'] for m in meta]:raise ValueError('Roster mismatch')
    with np.load(OUT/'SCORES_FROZEN.npz') as f:scores={k:f[k] for k in f.files}
    thresholds={}
    for name in METHODS:
        if name=='ridge':continue
        thresholds[name]={str(f):float(np.quantile(np.concatenate([scores[name][m['step_start']:m['step_stop']] for m in meta
            if m['fold']!=f and not m['cell'].startswith('pb_')]),.8)) for f in range(5)}
    linear=json.loads((ROOT/'results/temporal_linear_context_v1/METRICS.json').read_text())
    ridgekey='innovation5__real__signed_residual_0.25';thresholds['ridge']=linear['metrics'][ridgekey]['prmscore_thresholds']
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:
        gate=f['gate_percentile']>=.33
        refs={'original4':f['steps__'+BASELINE],'innovation5':f['steps__append_innovation__H0lim']}
        for feature in ('H0lim','VE0','VE075','VE1'):refs['single__'+feature]=f['steps__mean__'+feature]
        refs['entropy15']=f['steps__entropy15']
    with np.load(ROOT/'results/temporal_research_mechanism_v1/SCORES_FROZEN.npz') as f:refs['RBM12']=f['steps__RBM12_logit']
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True,pb_gate_open=gate)
    rm,rp=base.evaluator.evaluate_arrays(records,joined,refs,fold_auc=True,pb_gate_open=gate)
    scores.update(refs);metrics.update(rm);per.update(rp)
    for k in ('pb_all8','prm_within','prmscore_q08'):
        np.testing.assert_allclose(metrics['ridge'][k],linear['metrics'][ridgekey][k],atol=2e-14,rtol=0)
    previous=json.loads((ROOT/'results/context_weighted_levels_v1/METRICS.json').read_text())['metrics']
    for name in refs:
        for k in ('pb_all8','prm_within','prmscore_q08'):
            np.testing.assert_allclose(metrics[name][k],previous[name][k],atol=2e-14,rtol=0)
    primary=[('bocpd','ridge'),('bocpd','innovation5'),('bocpd','noreset'),('mean16','ridge'),('mean16','innovation5')]
    secondary=[(name,ref) for name in METHODS for ref in ('ridge','innovation5','zero') if name!=ref]
    pairs=list(dict.fromkeys(primary+secondary))
    print('[evaluate] full metrics; paired bootstrap10000',flush=True)
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(primary),primary_ci=.995)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    independent=independent_quality(records,joined,scores,gate,metrics,per)
    diags=json.loads((OUT/'DIAGNOSTICS.json').read_text())
    if [d['uid'] for d in diags]!=[m['uid'] for m in meta]:raise ValueError('Diagnostic roster mismatch')
    mse=mse_summary(meta,diags)
    pb=np.array([r['cell'].startswith('pb_') for r in records]);target=joined['target'];nsteps=np.diff(joined['offsets'])
    location=(target+.5)/nsteps;transitions={}
    for name in METHODS:
        transitions[name]={}
        for ref in ('ridge','innovation5'):
            hit=per[name]['prediction']==target;anchor=per[ref]['prediction']==target
            transitions[name][ref]={}
            for label,mask in [('all_errors',pb&(target>=0)),('early',pb&(target>=0)&(location<1/3)),
                ('middle',pb&(target>=0)&(location>=1/3)&(location<2/3)),('late',pb&(target>=0)&(location>=2/3))]:
                transitions[name][ref][label]=dict(answers=int(mask.sum()),both=int(np.sum(mask&hit&anchor)),
                    gained=int(np.sum(mask&hit&~anchor)),lost=int(np.sum(mask&~hit&anchor)),
                    neither=int(np.sum(mask&~hit&~anchor)))
    pareto=[name for name,m in metrics.items() if not any(n['pb_all8']>=m['pb_all8'] and n['prm_within']>=m['prm_within']
        and (n['pb_all8']>m['pb_all8'] or n['prm_within']>m['prm_within']) for n in metrics.values())]
    primary_positive=[a+'_minus_'+b for a,b in primary if contrasts[a+'_minus_'+b]['pb_ci'][0]>0 or contrasts[a+'_minus_'+b]['prm_within_ci'][0]>0]
    write(OUT/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,pareto=pareto,development_only=True,
        primary_endpoints=10,primary_ci=.995,primary_positive_endpoints_in=primary_positive,
        prediction=mse,peak_transitions=transitions))
    write(OUT/'AUDIT.json',dict(status='PASS',answers=len(records),steps=int(joined['offsets'][-1]),tokens=int(bundle.length.sum()),
        independent=independent,scoring=audit,all_source_and_bundle_hashes_match=True,all9_reference_headlines_exact=True,
        evaluation_sha256=sha(Path(__file__)),quality_labels_used_only_in_evaluation=True))
    lines=['method,PB_percent,within_AUC,PRMScore,prediction_MSE']
    for name,m in metrics.items():lines.append(f"{name},{100*m['pb_all8']:.6f},{m['prm_within']:.9f},{m['prmscore_q08']:.9f},{mse.get(name,{}).get('scalar_mse_answer_mean','')}")
    (OUT/'METRICS.csv').write_text('\n'.join(lines)+'\n',encoding='utf8')
    np.savez_compressed(OUT/'EVALUATED_SCORES.npz',**scores)
    state.update(status='COMPLETE_REVIEWED',evaluation_seconds=time.perf_counter()-began,pareto=pareto,
        primary_positive_endpoints_in=primary_positive,development_only=True,neural_queue_unchanged=True,fusion_fitted=False)
    write(OUT/'RUN_STATE.json',state)
    print('\n'.join(lines),flush=True)
    print('[primary]',json.dumps({a+'_minus_'+b:contrasts[a+'_minus_'+b] for a,b in primary}),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
