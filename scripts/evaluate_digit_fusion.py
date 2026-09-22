"""Full frozen digit experiment: quality, nested calibration and error accounting."""
from pathlib import Path
import sys,json,csv,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.digit_fusion import METHODS
from spectral_utils.context_training import FeatureBundle
from scripts.run_digit_fusion import OUT,DATA,REF,JOBS,key,sha,write
from scripts import run_temporal_research_baseline as base
from scripts.evaluate_residual_moment_real import independent_quality

PRIMARY=[('digit025','innovation5'),('digit025','presence'),('digit025','permuted'),
         ('tcn_digit_matched','tcn__real'),('bank6_iu','bank6_equal'),('bank6_iu','bank5_iu')]


def participation(C):
    e=np.linalg.eigvalsh(C)
    return float(e.sum()**2/max(np.square(e).sum(),1e-12))


def run():
    began=time.perf_counter();state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status']!='SCORED_PENDING_EVALUATION':raise ValueError('Full scored roster required')
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for name,h in manifest['code_inputs'].items():
        if sha(ROOT/name)!=h:raise ValueError('Code/input drift '+name)
    bundle=FeatureBundle(DATA,'innovation5');meta=bundle.metadata
    records,joined=base.load_contract(ROOT.parents[1]);assert [m['uid'] for m in meta]==[r['uid'] for r in records]
    total=int(joined['offsets'][-1]);outer=np.full((total,len(METHODS)),np.nan);nested=[np.full_like(outer,np.nan) for _ in range(5)]
    for e in JOBS:
        p=OUT/(key(e)+'.npz');a=json.loads((OUT/(key(e)+'_AUDIT.json')).read_text());assert sha(p)==a['scores_sha256']
        with np.load(p) as f:s=f['scores'];ids=f['ids']
        expected=[i for i,m in enumerate(meta) if m['fold'] in e and (len(e)==1 or not m['cell'].startswith('pb_'))]
        np.testing.assert_array_equal(ids,expected)
        for i in ids:
            m=meta[i];sl=slice(m['step_start'],m['step_stop'])
            dest=outer if len(e)==1 else nested[next(f for f in e if f!=m['fold'])]
            assert not np.isfinite(dest[sl]).any() and np.isfinite(s[sl]).all()
            dest[sl]=s[sl]
    assert np.isfinite(outer).all()
    scores={n:outer[:,j] for j,n in enumerate(METHODS)};thresholds={n:{} for n in METHODS}
    for f in range(5):
        idx=[i for i,m in enumerate(meta) if m['fold']!=f and not m['cell'].startswith('pb_')]
        a=np.concatenate([nested[f][meta[i]['step_start']:meta[i]['step_stop']] for i in idx]);assert np.isfinite(a).all()
        for j,n in enumerate(METHODS):thresholds[n][str(f)]=float(np.quantile(a[:,j],.8))
    refs=json.loads((REF/'METRICS.json').read_text())['metrics']
    with np.load(REF/'SCORES_FROZEN.npz') as f:
        for n in f.files:scores[n]=f[n];thresholds[n]=refs[n]['prmscore_thresholds']
    previous=ROOT/'results/predictor_subset_iu_v1';pm=json.loads((previous/'METRICS.json').read_text())['metrics']
    with np.load(previous/'SCORES_FROZEN.npz') as f:
        for n in ('iu__ridge+tcn+noreset','equal__ridge+bocpd+noreset'):
            scores[n]=f[n];refs[n]=pm[n];thresholds[n]=pm[n]['prmscore_thresholds']
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    print('[digit-eval]30 methods, complete population',flush=True)
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True,pb_gate_open=gate)
    for n,m in refs.items():
        for k in ('pb_all8','prm_within','prmscore_q08'):np.testing.assert_allclose(metrics[n][k],m[k],atol=2e-14,rtol=0)
    claude=json.loads((ROOT/'results/claude_real_checks_v1/DIGIT_DISAGREE_EVAL.json').read_text())
    for n,k in [('digit025','innovation5_plus_digit_g0.25'),('digit1','innovation5_plus_digit_g1'),('digit_standalone','digit_aux_standalone')]:
        for key_,ck in [('pb_all8','pb'),('prm_within','within'),('prmscore_q08','prmscore_default')]:
            np.testing.assert_allclose(metrics[n][key_],claude['table'][k][ck],atol=2e-14,rtol=0)
    secondary=[('digit025','tcn__real'),('digit1','innovation5'),('rate','digit025'),('tcn_digit_sum','tcn__real'),
               ('tcn_digit_sum','tcn_digit_matched'),('tcn_digit_matched','digit025'),('bank6_equal','bank5_equal'),
               ('bank6_iu','digit025'),('bank6_equal','digit025')]
    pairs=PRIMARY+secondary
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(PRIMARY),primary_ci=1-.05/12)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    independent=independent_quality(records,joined,scores,gate,metrics,per)
    ds=json.loads((OUT/'DIAGNOSTICS.json').read_text());C=np.array([d['covariance6'] for d in ds])
    live=np.array([not d['digit_constant'] for d in ds])
    cells=np.array([r['cell'] for r in records]);target=joined['target'];pb=np.char.startswith(cells,'pb_');error=pb&(target>=0)
    mechanism={}
    for label,mask in [('all_answers',np.ones(len(ds),bool)),('variable_digit_answers',live),('pb_error_answers',error)]:
        c=C[mask].mean(0);mechanism[label]=dict(answers=int(mask.sum()),participation5=participation(c[:5,:5]),participation6=participation(c),mean_covariance=c.tolist())
    mechanism['constant_digit_answers']=int((~live).sum());mechanism['total_disagreements']=sum(d['disagreements'] for d in ds)
    for j in (0,1):
        dd=[d['banks'][j] for d in ds];W=np.array([d['weights'] for d in dd])
        mechanism['bank'+str(5+j)]=dict(native_answers=sum(d['native'] for d in dd),
            negative_weight_fraction=float(np.any(W<0,axis=1).mean()),median_weights=np.median(W,axis=0).tolist(),
            last_weight_median_if_digit_present=float(np.median(W[live,-1])) if j else None)
    common=set()
    with (ROOT/'results/predictor_error_profiles_v1/COMMON_MISSES.csv').open(encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['no_archive_peak_correct_even_without_gate']=='True':common.add(r['uid'])
    cohort=np.array([r['uid'] in common for r in records]);assert cohort.sum()==885 and (cohort&gate).sum()==707
    with np.load(OUT/'AUXILIARY.npz') as f:aux=f['aux'];counts=f['counts'];opp=f['opportunities']
    # Tie-aware diagnostic: distinguish a positive disagreement at gold from a
    # rank1 tie of an all-zero stream.
    gold_positive=np.zeros(len(records),bool)
    for i in np.flatnonzero(error):gold_positive[i]=aux[joined['offsets'][i]+target[i],0]>0
    error_report={}
    for n in METHODS:
        hit=error&(per[n]['prediction']==target);raw=error&(per[n]['peak']==target)
        e=dict(hits=int(hit.sum()),raw_hits=int(raw.sum()),common885_raw=int((raw&cohort).sum()),
               common707_final=int((hit&cohort).sum()),common707_positive_final=int((hit&cohort&gold_positive).sum()))
        for ref in ('innovation5','tcn__real'):
            rh=error&(per[ref]['prediction']==target);g=hit&~rh;l=~hit&rh
            e[ref]=dict(gained=int(g.sum()),lost=int(l.sum()),net=int(g.sum()-l.sum()))
        error_report[n]=e
    # Descriptive length/opportunity stratification; fixed geometry cutoffs.
    groups={'step_tokens_le16':np.zeros(len(records),bool),'step_tokens_17_64':np.zeros(len(records),bool),
            'step_tokens_gt64':np.zeros(len(records),bool),'gold_no_digits':np.zeros(len(records),bool),
            'gold_1_4_digits':np.zeros(len(records),bool),'gold_5plus_digits':np.zeros(len(records),bool)}
    for i in np.flatnonzero(error):
        k=joined['offsets'][i]+target[i];a,b=bundle.spans[k];length=b-a
        groups['step_tokens_le16' if length<=16 else 'step_tokens_17_64' if length<=64 else 'step_tokens_gt64'][i]=True
        groups['gold_no_digits' if opp[k]==0 else 'gold_1_4_digits' if opp[k]<=4 else 'gold_5plus_digits'][i]=True
    strata={}
    for label,mask in groups.items():
        rr=error&(per['innovation5']['prediction']==target);hh=error&(per['digit025']['prediction']==target)
        strata[label]=dict(answers=int(mask.sum()),gained=int((mask&hh&~rr).sum()),lost=int((mask&~hh&rr).sum()))
    ledger=[]
    for i in np.flatnonzero(error):
        r=records[i];k=joined['offsets'][i]+target[i]
        row=dict(uid=r['uid'],cell=r['cell'],source_group=r['group_id'],true_step_1based=int(target[i]+1),
                 gate_open=bool(gate[i]),common885=bool(cohort[i]),gold_disagreements=float(counts[k]),gold_digits=float(opp[k]))
        for n in list(METHODS)+['innovation5','tcn__real']:
            row[n+'_peak']=int(per[n]['peak'][i]+1);row[n+'_hit']=bool(gate[i] and per[n]['peak'][i]==target[i])
        ledger.append(row)
    with (OUT/'PB_ERROR_LEDGER.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    write(OUT/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,mechanism=mechanism,error_analysis=error_report,
        strata=strata,primary_ci=1-.05/12,development_only=True,screening_used_labels=True,
        original_claude_contrast=claude['contrasts']['innovation5_plus_digit_g0.25_minus_innovation5']))
    np.savez_compressed(OUT/'SCORES_FROZEN.npz',**scores)
    audit=dict(status='PASS',answers=len(records),steps=total,methods=len(metrics),independent_metrics=independent,
               original_reference_rows=18,claude_three_rows_exact=True,source_manifest_sha256=sha(OUT/'MANIFEST.json'),
               scoring=json.loads((OUT/'SCORING_AUDIT.json').read_text()),evaluation_source_sha256=sha(Path(__file__)),
               scores_sha256=sha(OUT/'SCORES_FROZEN.npz'),
               flow_state_unchanged=sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json')==manifest['flow_state_sha256'])
    assert audit['flow_state_unchanged'];write(OUT/'AUDIT.json',audit)
    lines=['method,PB_percent,within_AUC,PRMScore']+[f"{n},{100*m['pb_all8']:.6f},{m['prm_within']:.9f},{m['prmscore_q08']:.9f}" for n,m in metrics.items()]
    (OUT/'METRICS.csv').write_text('\n'.join(lines)+'\n',encoding='utf8')
    state.update(status='COMPLETE_REVIEWED',evaluation_seconds=time.perf_counter()-began,development_only=True);write(OUT/'RUN_STATE.json',state)
    print('\n'.join(lines),flush=True);print('PRIMARY',json.dumps({a+'_minus_'+b:contrasts[a+'_minus_'+b] for a,b in PRIMARY}),flush=True)
    print('ERRORS',json.dumps(error_report),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
