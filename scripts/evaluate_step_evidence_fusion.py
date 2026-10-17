"""Evaluate the frozen step-evidence experiment on the complete development set."""
from pathlib import Path
import sys,json,csv,time,html
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.step_evidence_fusion import METHODS
from spectral_utils.context_training import FeatureBundle
from scripts.run_step_evidence_fusion import OUT,DATA,REF,JOBS,key,sha,write
from scripts import run_temporal_research_baseline as base
from scripts.evaluate_residual_moment_real import independent_quality

PREFIX='evidence__'
PRIMARY=[('iu','equal'),('iu','context'),('equal','context'),
         ('iu','iu_shuffled'),('equal','equal_shuffled')]


def run():
    began=time.perf_counter()
    state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status'] not in ('SCORED_PENDING_EVALUATION','COMPLETE_REVIEWED'):raise ValueError('Scoring incomplete')
    source_audit=json.loads((OUT/'SCORING_AUDIT.json').read_text())
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for name,h in manifest['hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('Source drift: '+name)
    bundle=FeatureBundle(DATA,'innovation5');meta=bundle.metadata
    records,joined=base.load_contract(ROOT.parents[1])
    if [m['uid'] for m in meta]!=[r['uid'] for r in records]:raise ValueError('Roster drift')
    total=int(joined['offsets'][-1]);outer=np.full((total,len(METHODS)),np.nan)
    nested=[np.full_like(outer,np.nan) for _ in range(5)]
    diagnostics=[];job_counts={}
    for e in JOBS:
        path=OUT/(key(e)+'.npz');a=source_audit['jobs'][key(e)]
        if sha(path)!=a['scores_sha256']:raise ValueError('Scored artifact drift')
        with np.load(path,allow_pickle=False) as f:s=f['scores'];ids=f['ids']
        expected=[i for i,m in enumerate(meta) if m['fold'] in e and (len(e)==1 or not m['cell'].startswith('pb_'))]
        np.testing.assert_array_equal(ids,expected)
        for i in ids:
            m=meta[i];sl=slice(m['step_start'],m['step_stop'])
            if not np.isfinite(s[sl]).all():raise ValueError('Missing scores')
            dest=outer if len(e)==1 else nested[next(f for f in e if f!=m['fold'])]
            if np.isfinite(dest[sl]).any():raise ValueError('Overlapping scored answers')
            dest[sl]=s[sl]
        job_counts[key(e)]=len(ids)
        if len(e)==1:diagnostics+=json.loads((OUT/(key(e)+'_DIAGNOSTICS.json')).read_text())
    if not np.isfinite(outer).all():raise ValueError('Full population not scored')
    scores={PREFIX+n:outer[:,j] for j,n in enumerate(METHODS)}
    thresholds={n:{} for n in scores}
    for f in range(5):
        ids=[i for i,m in enumerate(meta) if m['fold']!=f and not m['cell'].startswith('pb_')]
        a=np.concatenate([nested[f][meta[i]['step_start']:meta[i]['step_stop']] for i in ids])
        if not np.isfinite(a).all():raise ValueError('Incomplete nested calibration')
        for j,n in enumerate(METHODS):thresholds[PREFIX+n][str(f)]=float(np.quantile(a[:,j],.8))
    old=json.loads((REF/'METRICS.json').read_text())['metrics'];references=dict(old)
    with np.load(REF/'SCORES_FROZEN.npz') as f:
        for n in f.files:scores[n]=f[n];thresholds[n]=old[n]['prmscore_thresholds']
    previous=ROOT/'results/predictor_subset_iu_v1'
    pm=json.loads((previous/'METRICS.json').read_text())['metrics']
    with np.load(previous/'SCORES_FROZEN.npz') as f:
        for n in ('iu__ridge+tcn+noreset','equal__ridge+bocpd+noreset'):
            scores[n]=f[n];references[n]=pm[n];thresholds[n]=pm[n]['prmscore_thresholds']
    np.testing.assert_allclose(scores[PREFIX+'context'],scores['tcn__real'],atol=2e-10,rtol=0)
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    print('[evaluate]27 methods, complete population',flush=True)
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True,pb_gate_open=gate)
    for n,m in references.items():
        for k in ('pb_all8','prm_within','prmscore_q08'):
            np.testing.assert_allclose(metrics[n][k],m[k],atol=2e-14,rtol=0)
    for k in ('pb_all8','prm_within','prmscore_q08'):
        np.testing.assert_allclose(metrics[PREFIX+'context'][k],metrics['tcn__real'][k],atol=2e-14,rtol=0)
    primary=[(PREFIX+a,PREFIX+b) for a,b in PRIMARY]
    secondary=[(PREFIX+n,r) for n in ('end','sustained','equal_context_end','equal_context_sustained','equal','iu') for r in ('tcn__real','innovation5')]
    pairs=primary+secondary
    print('[evaluate]10000 source-group bootstrap draws',flush=True)
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(primary),primary_ci=.995)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    independent=independent_quality(records,joined,scores,gate,metrics,per)
    mechanism={}
    for kind in ('real','shuffled'):
        native=[d[kind] for d in diagnostics if d[kind]['native']]
        ws=np.array([d['weights'] for d in native]);c=np.array([d['correlation'] for d in native])
        mechanism[kind]=dict(native_answers=len(native),fallback_answers=len(diagnostics)-len(native),
            fallback_reasons=dict(Counter(d[kind]['reason'] for d in diagnostics if not d[kind]['native'])),
            median_weights=np.median(ws,axis=0).tolist(),negative_weight_fraction=float(np.any(ws<0,axis=1).mean()),
            median_correlation=np.median(c,axis=0).tolist(),
            spearman_filter_would_reject_fraction=float(np.mean([d['max_abs_spearman']>=.75 for d in native])),
            weight_l1_quantiles=np.quantile(np.abs(ws).sum(1),[.1,.5,.9,.99,1]).tolist(),
            g2_ceiling_fraction=float(np.mean([d['at_ceiling'] for d in native])))
    target=joined['target'];pb=np.array([r['cell'].startswith('pb_') for r in records]);err=pb&(target>=0)
    rawmiss=set()
    with (ROOT/'results/predictor_error_profiles_v1/COMMON_MISSES.csv').open(encoding='utf8') as f:
        for r in csv.DictReader(f):
            if r['no_archive_peak_correct_even_without_gate']=='True':rawmiss.add(r['uid'])
    cohort=np.array([r['uid'] in rawmiss for r in records])
    assert int(cohort.sum())==885 and int((cohort&gate).sum())==707
    error_report={};ledger=[]
    for n in (PREFIX+m for m in METHODS):
        p=per[n];raw=err&(p['peak']==target);hit=raw&gate
        entry=dict(final_error_hits=int(hit.sum()),raw_error_hits=int(raw.sum()),
            recovered_common885_raw=int((raw&cohort).sum()),recovered_common707_final=int((hit&cohort).sum()))
        for ref in ('innovation5','tcn__real'):
            refhit=err&gate&(per[ref]['peak']==target)
            entry[ref]=dict(gained=int((hit&~refhit).sum()),lost=int((~hit&refhit).sum()),net=int(hit.sum()-refhit.sum()))
        error_report[n]=entry
    for i in np.flatnonzero(err):
        r=records[i];row=dict(uid=r['uid'],cell=r['cell'],source_group=r['group_id'],
            error_step_1based=int(target[i]+1),gate_open=bool(gate[i]),old_common885=bool(cohort[i]))
        for n in [PREFIX+m for m in METHODS]+['innovation5','tcn__real']:
            row[n+'_peak_1based']=int(per[n]['peak'][i]+1);row[n+'_hit']=bool(gate[i] and per[n]['peak'][i]==target[i])
        ledger.append(row)
    with (OUT/'PB_ERROR_LEDGER.csv').open('w',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(ledger[0]));writer.writeheader();writer.writerows(ledger)
    result=dict(metrics=metrics,contrasts=contrasts,mechanism=mechanism,error_analysis=error_report,
        primary_ci=.995,development_only=True,no_new_selection=True)
    write(OUT/'METRICS.json',result)
    np.savez_compressed(OUT/'SCORES_FROZEN.npz',**scores)
    audit=dict(status='PASS',answers=len(records),steps=total,methods=len(metrics),reference_headlines_replayed=len(references),
        independent_metrics=independent,jobs=job_counts,scoring=source_audit,
        scores_sha256=sha(OUT/'SCORES_FROZEN.npz'),evaluation_code_sha256=sha(Path(__file__)),
        source_manifest_sha256=sha(OUT/'MANIFEST.json'),cohorts=dict(common_raw=885,common_open=707),
        flow_state_unchanged=manifest['flow_state_sha256']==sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json'))
    if not audit['flow_state_unchanged']:raise ValueError('Unexpected old queue change')
    write(OUT/'AUDIT.json',audit)
    lines=['method,PB_percent,within_AUC,PRMScore,final_error_hits,raw_common885_recovered,final_common707_recovered']
    for n,m in metrics.items():
        e=error_report.get(n,{})
        lines.append(','.join(map(str,[n,100*m['pb_all8'],m['prm_within'],m['prmscore_q08'],
            e.get('final_error_hits',''),e.get('recovered_common885_raw',''),e.get('recovered_common707_final','')])))
    (OUT/'METRICS.csv').write_text('\n'.join(lines)+'\n',encoding='utf8')
    state.update(status='COMPLETE_REVIEWED',evaluation_seconds=time.perf_counter()-began,
        correctness_labels_in_fit=False,development_only=True)
    write(OUT/'RUN_STATE.json',state)
    print('\n'.join(lines),flush=True)
    print('PRIMARY',json.dumps({a+'_minus_'+b:contrasts[a+'_minus_'+b] for a,b in primary}),flush=True)
    print('MECHANISM',json.dumps(mechanism),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
