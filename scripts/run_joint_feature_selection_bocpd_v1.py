"""Full-population alternating Joint inclusion with cached BOCPD residuals."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_digitfree_broad50_v1 import load_data,sha,uncertainty
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from spectral_utils.digitfree_broad50 import NAMES,ANCHOR
from spectral_utils.lsml_gate_locator_research import answer_standardize,FusionRecipe,fit_fusion_weights
from spectral_utils.joint_block_balanced import discover_sourcefold_groups
from spectral_utils.joint_feature_selection import pruning_path

OUT=ROOT/'results/joint_feature_selection_bocpd_v1'
TEMPORAL=ROOT.parents[1]/'.worktrees/temporal-research-20260915'
SEED=399170
BANKS=('B50','B51_bocpd','B51_noreset')


def state(status,**kwargs):
    dump(OUT/'RUN_STATE.json',dict(status=status,time=time.strftime('%Y-%m-%dT%H:%M:%S'),**kwargs))
    print(status,kwargs,flush=True)


def inputs():
    OUT.mkdir(parents=True,exist_ok=True)
    predictor=TEMPORAL/'results/aligned_context_predictors_v1/SCORES_FROZEN.npz'
    review=json.loads((predictor.parent/'SCORING_AUDIT.json').read_text())
    assert sha(predictor)==review['scores_sha256']
    files=[Path(__file__),ROOT/'spectral_utils/joint_feature_selection.py',
        ROOT/'docs/experiments/JOINT_FEATURE_SELECTION_BOCPD_V1.md',predictor,
        TEMPORAL/'spectral_utils/aligned_context_predictors.py',ROOT/'results/digitfree_broad50_v1/MANIFEST.json']
    manifest=dict(schema='joint-feature-selection-bocpd-v1',seed=SEED,banks=BANKS,hashes={str(p):sha(p) for p in files})
    canonical=json.loads(json.dumps(manifest));path=OUT/'MANIFEST.json'
    if path.exists() and json.loads(path.read_text())!=canonical:raise ValueError('immutable manifest drift')
    dump(path,manifest)
    data=load_data()
    with np.load(predictor) as z:predictors={k:z[k] for k in ('bocpd','noreset')}
    with np.load(TEMPORAL/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as z:
        predictor_base=z['steps__append_innovation__H0lim'].copy()
        np.testing.assert_allclose(data['references']['innovation5'],predictor_base,atol=2e-12,rtol=0)
    dump(OUT/'BASELINE_ALIGNMENT.json',dict(status='PASS',
        max_absolute_difference=float(np.max(np.abs(data['references']['innovation5']-predictor_base))),
        subtraction='exact baseline from the predictor source archive'))
    # The cached scores are base + .25 * base SD * standardized auxiliary.
    # Answer standardization removes this positive scalar exactly.
    base=predictor_base
    for name,score in predictors.items():
        assert score.shape==base.shape and np.isfinite(score).all()
        data[name]=answer_standardize((score-base)[:,None],data['offsets'])[:,0]
        data['references']['historical_'+name]=score
    data['banks']={'B50':data['x'],**{f'B51_{k}':np.column_stack((data['x'],data[k])) for k in predictors}}
    np.savez_compressed(OUT/'INPUTS.npz',bocpd=data['bocpd'],noreset=data['noreset'])
    return data


def fit(data):
    rowfold=np.repeat(data['folds'],np.diff(data['offsets']));all_scores={};all_meta={}
    for bank in BANKS:
        x=data['banks'][bank];names=tuple(NAMES)+( () if bank=='B50' else (bank.removeprefix('B51_')+'_residual',))
        arms=['joint_full','joint_auto','equal_full','equal_selected','continuous']
        if bank=='B50':arms += [f'joint_keep{k}' for k in (40,30,20,12,8)]
        scores={a:np.full(len(x),np.nan) for a in arms};metas=[]
        for outer in range(5):
            target=OUT/f'{bank}_fold{outer}';npz=target.with_suffix('.npz');js=target.with_suffix('.json')
            held=rowfold==outer
            if npz.exists() and js.exists():
                with np.load(npz) as z:
                    for a in arms:scores[a][held]=z[a]
                metas.append(json.loads(js.read_text()));continue
            start=time.perf_counter();train=x[~held]
            state('FIT',bank=bank,outer=outer)
            if bank=='B50':
                old=json.loads((ROOT/f'results/digitfree_broad50_v1/fold_{outer}.json').read_text())
                discovery=old['discovery']
            else:
                discovery=discover_sourcefold_groups(train,rowfold[~held],seed=SEED+100+outer)
            meta=dict(outer=outer,names=names,discovery=discovery,weights={},valid={})
            weights={'equal_full':np.ones(x.shape[1])/x.shape[1]}
            try:
                weights['continuous'],cmeta=fit_fusion_weights(train,FusionRecipe(bank,names,'continuous',anchor=ANCHOR),seed=SEED+outer)
                meta['continuous']=cmeta;meta['valid']['continuous']=True
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                meta['valid']['continuous']=False;meta['continuous_failure']=str(exc)
            try:
                if discovery['status']!='SELECTED':raise ValueError('NO_ADMISSIBLE_GROUPS')
                def notify(step,kept,retention):
                    if step%5==0:state('PRUNING',bank=bank,outer=outer,removed=step,kept=kept,retention=retention)
                path=pruning_path(train,np.asarray(discovery['labels']),anchor_index=ANCHOR,seed=SEED+200+outer,
                                  notify=notify,stop_after_automatic=bank!='B50')
                weights['joint_full']=path['path'][0]['weights'];weights['joint_auto']=path['automatic']['weights']
                equal=np.zeros(x.shape[1]);equal[path['automatic']['active']]=1/len(path['automatic']['active'])
                weights['equal_selected']=equal
                if bank=='B50':
                    for k in (40,30,20,12,8):
                        candidates=[s for s in path['path'] if len(s['active'])>=k]
                        chosen=min(candidates,key=lambda s:len(s['active']))
                        weights[f'joint_keep{k}']=chosen['weights']
                meta['selection']=path
                meta['selected_names']=[names[i] for i in path['automatic']['active']]
                for a in weights:meta['valid'][a]=True
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                meta['selection_failure']=f'{type(exc).__name__}: {exc}'
            for a in arms:
                if a not in weights:
                    weights[a]=np.eye(x.shape[1])[ANCHOR];meta['valid'][a]=False
                scores[a][held]=x[held]@weights[a]
            meta['weights']=weights;meta['seconds']=time.perf_counter()-start
            dump(js,meta);np.savez_compressed(npz,**{a:scores[a][held] for a in arms})
            metas.append(meta);state('FOLD_COMPLETE',bank=bank,outer=outer,seconds=meta['seconds'])
        all_scores.update({bank+'__'+a:s for a,s in scores.items()});all_meta[bank]=metas
    return all_scores,all_meta


def bootstrap(data,metrics,pairs):
    cells=data['cells'].astype(str);target=data['target'];unique,g=np.unique(data['groups'],return_inverse=True)
    keys=sorted({a for pair in pairs for a in pair});ng=len(unique)
    cellnames=sorted(c for c in set(cells) if c.startswith('pb_'))
    statistics=np.zeros((ng,len(cellnames),2+2*len(keys)));within=np.zeros((ng,1+len(keys)))
    for j,c in enumerate(cellnames):
        clean=(cells==c)&(target==-1);error=(cells==c)&(target>=0);vectors=[clean,error]
        for k in keys:
            correct=metrics[k]['prediction']==target;vectors.extend([clean&correct,error&correct])
        for i,v in enumerate(vectors):statistics[:,j,i]=np.bincount(g,weights=v,minlength=ng)
    valid=np.isfinite(metrics[keys[0]]['within_values'])
    within[:,0]=np.bincount(g,weights=valid,minlength=ng)
    for j,k in enumerate(keys):
        v=metrics[k]['within_values'];assert np.array_equal(np.isfinite(v),valid)
        within[:,j+1]=np.bincount(g,weights=np.nan_to_num(v),minlength=ng)
    rng=np.random.default_rng(400001);draws={p:{'pb':[],'within':[]} for p in pairs}
    for start in range(0,10000,100):
        count=rng.multinomial(ng,np.full(ng,1/ng),size=100);s=np.einsum('bg,gck->bck',count,statistics)
        pbs=[]
        for j in range(len(keys)):
            ca=s[:,:,2+2*j]/s[:,:,0];ea=s[:,:,3+2*j]/s[:,:,1]
            pbs.append(np.divide(2*ca*ea,ca+ea,out=np.zeros_like(ca),where=(ca+ea)>0).mean(axis=1))
        w=count@within;aucs=w[:,1:]/w[:,:1]
        for a,b in pairs:
            ia,ib=keys.index(a),keys.index(b)
            draws[(a,b)]['pb'].extend(pbs[ia]-pbs[ib]);draws[(a,b)]['within'].extend(aucs[:,ia]-aucs[:,ib])
    result={};alpha=.05/(2*len(pairs))
    for a,b in pairs:
        result[a+' minus '+b]={e:dict(point=metrics[a][e]-metrics[b][e],
            low=float(np.quantile(draws[(a,b)][e],alpha/2)),high=float(np.quantile(draws[(a,b)][e],1-alpha/2)),
            confidence=1-alpha,draws=10000) for e in ('pb','within')}
    return result


def main():
    data=inputs();scores,meta=fit(data);scores.update(data['references']);state('EVALUATING')
    metrics={a:score_locator(s,data['gate'],data) for a,s in scores.items()}
    pairs=[('B50__joint_auto','B50__joint_full'),('B51_bocpd__joint_auto','B51_bocpd__joint_full'),
        ('B51_bocpd__joint_auto','B50__joint_auto'),('B51_bocpd__joint_auto','B51_noreset__joint_auto')]
    intervals=bootstrap(data,metrics,pairs);coverage={}
    for bank,folds in meta.items():
        for a in folds[0]['valid']:
            coverage[bank+'__'+a]=sum(int(np.sum(data['folds']==f['outer'])) for f in folds if f['valid'][a])
    summary=dict(status='COMPLETE',metrics={a:scalar_metrics(m) for a,m in metrics.items()},
        primary_contrasts=intervals,native_answers=coverage,
        selected_counts={b:[len(f.get('selected_names',[])) for f in fs] for b,fs in meta.items()})
    dump(OUT/'RESULTS.json',summary);np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    lines=['# Alternating Joint feature selection plus BOCPD','',
        'Full development population; 95% information retention rule; frozen non-digit PB gate.',
        '', '| Method | PB macro % | PRMB within AUC |','|---|---:|---:|']
    for a,m in summary['metrics'].items():lines.append(f"| {a} | {100*m['pb']:.4f} | {m['within']:.6f} |")
    lines+=['','Automatic retained counts: '+str(summary['selected_counts']),'',
        'Fixed-count paths are descriptive; none was chosen using outer labels as the automatic rule.',
        'BOCPD means the pure cached signed residual channel; historical_bocpd includes its old innovation5 correction.',
        '', 'Primary paired contrasts (8 endpoints corrected):','```json',json.dumps(intervals,indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    state('COMPLETE',selected_counts=summary['selected_counts'])
    print(json.dumps({a:{k:m[k] for k in ('pb','within')} for a,m in metrics.items()},indent=2),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('FAILED',error=str(exc));raise
