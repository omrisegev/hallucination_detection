"""Evaluate all32 frozen combinations; selection labels enter only here."""
from pathlib import Path
import sys,json,sqlite3,io,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.run_predictor_subset_study import OUT,DATA,JOBS,key,sha,write
from spectral_utils.predictor_subset_fusion import METHODS,SUBSETS,PREDICTORS,subset_name
from spectral_utils.context_training import FeatureBundle
from scripts import run_temporal_research_baseline as base
from scripts.evaluate_residual_moment_real import independent_quality

def assemble(meta,total):
    outer=np.full((total,32),np.nan);nested=[np.full((total,32),np.nan) for _ in range(5)];audits={}
    for excluded in JOBS:
        folder=OUT/key(excluded);state=json.loads((folder/'RUN_STATE.json').read_text())
        audit=json.loads((folder/'AUDIT.json').read_text())
        if state['status']!='SCORED' or state['answers']!=state['expected'] or audit['status']!='PASS':raise ValueError('Incomplete job')
        if sha(folder/'STEP_SCORES.npz')!=audit['scores_sha256'] or sha(folder/'ANSWERS.sqlite')!=audit['sqlite_sha256']:raise ValueError('Changed score checkpoint')
        expected=[i for i,m in enumerate(meta) if m['fold'] in excluded and (len(excluded)==1 or not m['cell'].startswith('pb_'))]
        if len(expected)!=state['answers']:raise ValueError('Job roster mismatch')
        with np.load(folder/'STEP_SCORES.npz') as f:arr=f['scores']
        for i in expected:
            m=meta[i];sl=slice(m['step_start'],m['step_stop'])
            if not np.isfinite(arr[sl]).all():raise ValueError('Missing held scores')
            if len(excluded)==1:outer[sl]=arr[sl]
            else:
                held=next(f for f in excluded if f!=m['fold']);nested[held][sl]=arr[sl]
        audits[key(excluded)]=audit
    if not np.isfinite(outer).all():raise ValueError('Incomplete full population')
    thresholds={n:{} for n in METHODS}
    for held in range(5):
        a=np.concatenate([nested[held][m['step_start']:m['step_stop']] for m in meta if m['fold']!=held and not m['cell'].startswith('pb_')])
        if not np.isfinite(a).all():raise ValueError('Calibration coverage missing')
        q=np.quantile(a,.8,axis=0)
        for j,n in enumerate(METHODS):thresholds[n][str(held)]=float(q[j])
    return {n:outer[:,j] for j,n in enumerate(METHODS)},thresholds,audits

def mechanism_audit(bundle,scores,bankbase):
    n=len(bundle.metadata);weights=np.full((n,16,5),np.nan);pair=np.full((n,16),np.nan)
    ceiling=np.zeros((n,16),bool);flip=np.zeros_like(ceiling);corr=np.full((n,5,5),np.nan)
    eig=np.full((n,16),np.nan);constant=np.zeros((n,32),bool)
    rcache=np.load(OUT/'RESIDUALS.npy',mmap_mode='r');seen=set();maxdelta=0.;maxcanonical=0.
    for fold in range(5):
        con=sqlite3.connect(OUT/key((fold,))/'ANSWERS.sqlite')
        for i,blob,diag in con.execute('SELECT idx,scores,diagnostic FROM answers ORDER BY idx'):
            if i in seen:raise ValueError('Duplicate outer answer')
            m=bundle.metadata[i];d=json.loads(diag)
            if d['uid']!=m['uid'] or m['fold']!=fold:raise ValueError('Diagnostic identity mismatch')
            for j,x in enumerate(d['subsets']):
                weights[i,j]=x['weights'];pair[i,j]=x['pair_residual']
                ceiling[i,j]=x['at_ceiling'];flip[i,j]=x['flipped'];eig[i,j]=x['second_eigenvalue']
            corr[i]=d['correlation'];constant[i]=d['constant_auxiliary'];seen.add(i)
            maxcanonical=max(maxcanonical,d['canonical_weight_delta'] or 0.)
            # Independent sum-of-contributions and sorted readout of ALL32 arms.
            a=m['offset'];r=np.asarray(rcache[a:a+m['tokens']]);Z=(r-r.mean(0))/r.std(0)
            signal=np.empty((len(r),32))
            for j,s in enumerate(SUBSETS):
                signal[:,2*j]=sum(Z[:,k]*weights[i,j,k] for k in s)
                signal[:,2*j+1]=sum(Z[:,k]/len(s) for k in s)
            sl=slice(m['step_start'],m['step_stop']);spans=np.asarray(bundle.spans[sl])-a
            aux=np.array([np.sort(signal[u:v],axis=0)[-min(10,v-u):].mean(0) for u,v in spans])
            scale=aux.std(0);expected=np.broadcast_to(bankbase[sl,None],aux.shape).copy();live=scale>1e-12
            expected[:,live]+=.25*bankbase[sl].std()*(aux[:,live]-aux[:,live].mean(0))/scale[live]
            actual=np.column_stack([scores[name][sl] for name in METHODS])
            np.testing.assert_allclose(expected,actual,atol=2e-10,rtol=0)
            maxdelta=max(maxdelta,float(np.max(np.abs(expected-actual))))
        con.close()
    if len(seen)!=n or not np.isfinite(weights).all():raise ValueError('Incomplete weights audit')
    summary={}
    for j,s in enumerate(SUBSETS):
        w=weights[:,j][:,s];l1=np.abs(w).sum(1)
        summary[subset_name(s)]=dict(predictors=[PREDICTORS[k] for k in s],mean_weights=w.mean(0).tolist(),
            median_weights=np.median(w,axis=0).tolist(),negative_weight_answer_fraction=float(np.any(w<0,axis=1).mean()),
            weight_l1_quantiles=np.quantile(l1,[.1,.5,.9,.99,1]).tolist(),
            pair_residual_quantiles=np.quantile(pair[:,j],[.1,.5,.9]).tolist(),
            second_eigenvalue_quantiles=np.quantile(eig[:,j],[.1,.5,.9]).tolist(),
            ceiling_fraction=float(ceiling[:,j].mean()),flip_fraction=float(flip[:,j].mean()),
            constant_iu_auxiliary=int(constant[:,2*j].sum()),constant_equal_auxiliary=int(constant[:,2*j+1].sum()))
    np.savez_compressed(OUT/'WEIGHTS.npz',weights=weights,pair_residual=pair,correlation=corr,
        ceiling=ceiling,flipped=flip,second_eigenvalue=eig,constant_auxiliary=constant)
    return dict(subsets=summary,median_residual_correlation=np.median(corr,axis=0).tolist(),
        max_independent_readout_delta=maxdelta,max_canonical_weight_delta=maxcanonical,answers=n)

def leaders(names,metrics):
    def order(name,endpoint):
        m=metrics[name];other='prm_within' if endpoint=='pb_all8' else 'pb_all8'
        return (-m[endpoint],-m[other],-m['prmscore_q08'],len(name.split('__')[1].split('+')),name)
    return {e:min(names,key=lambda n:order(n,e)) for e in ('pb_all8','prm_within')}

def run():
    began=time.perf_counter();state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status'] not in ('SCORED_PENDING_EVALUATION','COMPLETE_REVIEWED'):raise ValueError('Full scoring required')
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for name,value in manifest['source_hashes'].items():
        if sha(ROOT/name)!=value:raise ValueError('Changed source '+name)
    bundle=FeatureBundle(DATA,'innovation5');meta=bundle.metadata
    for name,value in bundle.manifest['files'].items():
        if sha(DATA/name)!=value:raise ValueError('Changed data '+name)
    if sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json')!=manifest['old_flow_state_sha256']:raise ValueError('Flow queue changed')
    records,joined=base.load_contract(ROOT.parents[1])
    if [r['uid'] for r in records]!=[m['uid'] for m in meta]:raise ValueError('Evaluation roster drift')
    scores,thresholds,audits=assemble(meta,int(joined['offsets'][-1]))
    previous=ROOT/'results/tcn_aligned_predictor_seed0_v1';old=json.loads((previous/'METRICS.json').read_text())
    with np.load(previous/'SCORES_FROZEN.npz') as f:
        for name in f.files:scores[name]=f[name];thresholds[name]=old['metrics'][name]['prmscore_thresholds']
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True,pb_gate_open=gate)
    for name,m in old['metrics'].items():
        for k in ('pb_all8','prm_within','prmscore_q08'):np.testing.assert_allclose(metrics[name][k],m[k],atol=2e-14,rtol=0)
    primary=[('iu__'+subset_name(s),'equal__'+subset_name(s)) for s in SUBSETS]
    secondary=[(n,ref) for n in METHODS for ref in ('ridge','tcn__real','bocpd')]
    pairs=primary+secondary
    print('[subset-evaluate]48 methods;10000 paired group draws',flush=True)
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(primary),primary_ci=1-.05/32)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    independent=independent_quality(records,joined,scores,gate,metrics,per)
    mechanism=mechanism_audit(bundle,scores,scores['innovation5'])
    choose=leaders(METHODS,metrics)
    by_size={str(k):leaders([n for n in METHODS if len(n.split('__')[1].split('+'))==k],metrics) for k in (3,4,5)}
    by_head={head:leaders([n for n in METHODS if n.startswith(head+'__')],metrics) for head in ('iu','equal')}
    def pareto(names):
        return [n for n in names if not any(metrics[o]['pb_all8']>=metrics[n]['pb_all8'] and metrics[o]['prm_within']>=metrics[n]['prm_within'] and
            (metrics[o]['pb_all8']>metrics[n]['pb_all8'] or metrics[o]['prm_within']>metrics[n]['prm_within']) for o in names)]
    write(OUT/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,leaders=choose,leaders_by_size=by_size,
        leaders_by_head=by_head,candidate_pareto=pareto(METHODS),all_method_pareto=pareto(list(metrics)),
        mechanism=mechanism,primary_ci=1-.05/32,selection_uses_development_labels=True,seed=0))
    np.savez_compressed(OUT/'SCORES_FROZEN.npz',**scores)
    write(OUT/'AUDIT.json',dict(status='PASS',answers=len(records),steps=int(joined['offsets'][-1]),tokens=int(bundle.length.sum()),
        methods=len(metrics),jobs=audits,independent_metrics=independent,all16_reference_headlines_exact=True,
        max_independent_readout_delta=mechanism['max_independent_readout_delta'],max_canonical_weight_delta=mechanism['max_canonical_weight_delta'],
        scores_sha256=sha(OUT/'SCORES_FROZEN.npz'),weights_sha256=sha(OUT/'WEIGHTS.npz'),evaluation_code_sha256=sha(Path(__file__))))
    lines=['method,PB_percent,within_AUC,PRMScore']+[f"{n},{100*m['pb_all8']:.6f},{m['prm_within']:.9f},{m['prmscore_q08']:.9f}" for n,m in metrics.items()]
    (OUT/'METRICS.csv').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    state.update(status='COMPLETE_REVIEWED',evaluation_seconds=time.perf_counter()-began,leaders=choose,
        development_only=True,fit_uses_correctness_labels=False,selection_uses_development_labels=True)
    write(OUT/'RUN_STATE.json',state)
    print('\n'.join(lines),flush=True);print('LEADERS',json.dumps(choose),flush=True)
    print('PRIMARY',json.dumps({a+'_minus_'+b:contrasts[a+'_minus_'+b] for a,b in primary}),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
