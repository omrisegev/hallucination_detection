"""Full quality and operational error complementarity; labels used here only."""
from pathlib import Path
import sys,json,csv,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.alternative_views_fusion import METHODS,VIEWS,BANKS
from spectral_utils.context_training import FeatureBundle
from scripts.run_alternative_views_fusion import OUT,DATA,REF,sha,write
from scripts import run_temporal_research_baseline as base
from scripts.evaluate_residual_moment_real import independent_quality

PRIMARY=[(b+'__'+a,b+'__'+r) for b in BANKS for a,r in [('iu','equal'),('diag','iu'),('block','iu'),('family_equal','equal')]]

def failure_table(hit):
    fail=1-hit.astype(float);p=fail.mean(0);joint=fail.T@fail/len(fail)
    den=np.sqrt((p*(1-p))[:,None]*(p*(1-p))[None,:])
    phi=np.divide(joint-p[:,None]*p[None,:],den,out=np.zeros_like(joint),where=den>1e-15)
    return dict(answers=len(hit),fail_rate=p.tolist(),both_fail_rate=joint.tolist(),phi=phi.tolist(),
        constant_failure_columns=np.flatnonzero(den.diagonal()<=1e-15).tolist(),
        row_hits_column_misses=(hit.astype(int).T@(1-hit.astype(int))).tolist())

def run():
    start=time.perf_counter();state=json.loads((OUT/'RUN_STATE.json').read_text())
    assert state['status']=='SCORED_PENDING_EVALUATION'
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for name,h in manifest['code_inputs'].items():assert sha(ROOT/name)==h,name
    audit0=json.loads((OUT/'SCORING_AUDIT.json').read_text());assert sha(OUT/'NEW_SCORES.npz')==audit0['scores_sha256']
    meta=FeatureBundle(DATA,'innovation5').metadata;records,joined=base.load_contract(ROOT.parents[1])
    assert [m['uid'] for m in meta]==[r['uid'] for r in records]
    with np.load(OUT/'NEW_SCORES.npz') as f:scores={n:f[n] for n in METHODS}
    thresholds={n:{} for n in METHODS}
    for fold in range(5):
        ix=np.concatenate([np.arange(m['step_start'],m['step_stop']) for m in meta if m['fold']!=fold and not m['cell'].startswith('pb_')])
        for n in METHODS:thresholds[n][str(fold)]=float(np.quantile(scores[n][ix],.8))
    refs=json.loads((REF/'METRICS.json').read_text())['metrics']
    with np.load(REF/'SCORES_FROZEN.npz') as f:
        for n in f.files:scores[n]=f[n];thresholds[n]=refs[n]['prmscore_thresholds']
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    print('[evaluate]',len(scores),'methods',flush=True)
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True,pb_gate_open=gate)
    for n,m in refs.items():
        for k in ('pb_all8','prm_within','prmscore_q08'):np.testing.assert_allclose(metrics[n][k],m[k],atol=2e-14,rtol=0)
    for k in ('pb_all8','prm_within','prmscore_q08'):np.testing.assert_allclose(metrics['add__digit'][k],metrics['digit025'][k],atol=2e-14,rtol=0)
    secondary=[('add__'+v,r) for v in VIEWS for r in ('innovation5','digit025') if not(v=='digit' and r=='digit025')]
    secondary +=[(b+'__'+h,'digit025') for b in BANKS for h in ('equal','family_equal','iu','diag','block')]
    secondary +=[('add__logtail15','add__tail15'),('add__logtail50','add__tail50')]
    pairs=PRIMARY+secondary
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(PRIMARY),primary_ci=1-.05/16)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    independent=independent_quality(records,joined,scores,gate,metrics,per)
    target=joined['target'];cells=np.array([r['cell'] for r in records]);pb=np.char.startswith(cells,'pb_');error=pb&(target>=0)
    diag=json.loads((OUT/'DIAGNOSTICS.json').read_text());mechanism={}
    for b in BANKS:
        mechanism[b]={}
        for h in ('equal','family_equal','iu','diag','block'):
            d=[r['banks'][b]['heads'][h] for r in diag];w=np.array([r['weights'] for r in d]);aa=[r['alpha'] for r in d if r['alpha'] is not None]
            mechanism[b][h]=dict(median_weights=np.median(w,axis=0).tolist(),negative_any_fraction=float((w<0).any(1).mean()),
                fallback_answers=sum(r['fallback'] is not None for r in d),alpha_quantiles=np.quantile(aa,[0,.25,.5,.75,1]).tolist() if aa else None)
    common=set()
    with (ROOT/'results/predictor_error_profiles_v1/COMMON_MISSES.csv').open(encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['no_archive_peak_correct_even_without_gate']=='True':common.add(r['uid'])
    cohort=np.array([r['uid'] in common for r in records]);assert cohort.sum()==885 and (cohort&gate).sum()==707
    errors={}
    for n in METHODS:
        raw=error&(per[n]['peak']==target);hit=error&(per[n]['prediction']==target)
        e=dict(raw_hits=int(raw.sum()),hits=int(hit.sum()),common885_raw=int((raw&cohort).sum()),common707_final=int((hit&cohort).sum()))
        for r in ('innovation5','digit025','tcn_digit_sum'):
            rh=error&(per[r]['prediction']==target);e[r]=dict(gained=int((hit&~rh).sum()),lost=int((rh&~hit).sum()))
        errors[n]=e
    names=['raw__'+v for v in VIEWS]+['innovation5','digit025','tcn__real']
    hits=np.column_stack([per[n]['peak']==target for n in names])
    stratified={}
    relative=np.array([target[i]/max(meta[i]['step_stop']-meta[i]['step_start']-1,1) for i in range(len(meta))])
    for label,mask in [(c,error&(cells==c)) for c in np.unique(cells[pb])]+[(f'first_error_{a}',error&(relative>=lo)&(relative<=hi if hi==1 else relative<hi)) for a,lo,hi in [('early',0,1/3),('middle',1/3,2/3),('late',2/3,1)]]:
        if mask.any():stratified[label]=failure_table(hits[mask])
    # Common positive-negative step pairs per PRMB answer, with half-errors on ties.
    means=[];products=[]
    labels=joined['labels'];offsets=joined['offsets']
    for i in np.flatnonzero(~pb):
        sl=slice(offsets[i],offsets[i+1]);y=labels[sl];pos=np.flatnonzero(y==1);neg=np.flatnonzero(y==0)
        if not len(pos) or not len(neg):continue
        x=np.column_stack([scores[n][sl] for n in names]);delta=x[pos,None,:]-x[None,neg,:]
        loss=((delta<0).astype(float)+.5*(delta==0)).reshape(-1,len(names))
        means.append(loss.mean(0));products.append(loss.T@loss/len(loss))
    mean=np.mean(means,axis=0);product=np.mean(products,axis=0)
    complementarity=dict(methods=names,pb_raw=failure_table(hits[error]),pb_strata=stratified,
        prm_answer_balanced=dict(answers=len(means),pairwise_ranking_loss=mean.tolist(),pair_error_products=product.tolist(),
            pair_error_covariance=(product-mean[:,None]*mean[None,:]).tolist()),
        scope='Operational localization/ranking errors on development labels; not latent residual independence. PB uses first-error labels only. PRMB ties count half an error.')
    write(OUT/'ERROR_COMPLEMENTARITY.json',complementarity)
    ledger=[]
    for i in np.flatnonzero(error):
        row=dict(uid=records[i]['uid'],cell=records[i]['cell'],group_id=records[i]['group_id'],gold_1based=int(target[i]+1),gate_open=bool(gate[i]),common885=bool(cohort[i]))
        for n in list(METHODS)+['innovation5','digit025','tcn_digit_sum']:row[n+'_peak']=int(per[n]['peak'][i]+1)
        ledger.append(row)
    with (OUT/'PB_ERROR_LEDGER.csv').open('w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ledger[0]));w.writeheader();w.writerows(ledger)
    write(OUT/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,mechanism=mechanism,error_analysis=errors,primary_ci=1-.05/16,development_only=True))
    np.savez_compressed(OUT/'SCORES_FROZEN.npz',**scores)
    audit=dict(status='PASS',answers=len(records),steps=int(offsets[-1]),methods=len(scores),independent_metrics=independent,
        scoring=audit0,reference_rows_exact=len(refs),digit_anchor_exact=True,manifest_sha256=sha(OUT/'MANIFEST.json'),scores_sha256=sha(OUT/'SCORES_FROZEN.npz'),
        flow_state_unchanged=sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json')==manifest['flow_state_sha256'])
    assert audit['flow_state_unchanged'];write(OUT/'AUDIT.json',audit)
    lines=['method,PB_percent,within_AUC,PRMScore']+[f"{n},{100*m['pb_all8']:.6f},{m['prm_within']:.9f},{m['prmscore_q08']:.9f}" for n,m in metrics.items()]
    (OUT/'METRICS.csv').write_text('\n'.join(lines)+'\n',encoding='utf8')
    state.update(status='COMPLETE_PENDING_REPORT',evaluation_seconds=time.perf_counter()-start);write(OUT/'RUN_STATE.json',state)
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
