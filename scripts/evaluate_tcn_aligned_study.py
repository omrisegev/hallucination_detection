"""Only the frozen signed TCN correction is a candidate; full nested calibration."""
from pathlib import Path
import sys,json,time
from itertools import combinations
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.run_tcn_aligned_study import OUT,MODELS,DATA,sha,write
from scripts.evaluate_temporal_context_models import assemble
from scripts.evaluate_residual_moment_real import independent_quality
from scripts.evaluate_aligned_context_predictors import distribution
from scripts import run_temporal_research_baseline as base
from spectral_utils.context_training import FeatureBundle


def verify_inputs(bundle):
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    for name,expected in manifest['source_hashes'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Changed study source '+name)
    for name,expected in bundle.manifest['files'].items():
        if sha(DATA/name)!=expected:raise ValueError('Changed data '+name)
    reuse=json.loads((OUT/'REUSE.json').read_text())
    for name,expected in reuse['initial_artifacts'].items():
        if sha(ROOT/name)!=expected:raise ValueError('Reused fold0 mutated')
    fits={}
    for excluded in [(f,) for f in range(5)]+list(combinations(range(5),2)):
        key='tcn__innovation5__seed0__exclude'+'_'.join(map(str,excluded));path=MODELS/key
        m=json.loads((path/'MANIFEST.json').read_text());state=json.loads((path/'RUN_STATE.json').read_text())
        sm=json.loads((path/'scoring/MANIFEST.json').read_text())
        if m['method']!='tcn' or m['bank']!='innovation5' or m['seed']!=0 or m['smoke'] or m['excluded_folds']!=list(excluded):raise ValueError('Wrong training contract')
        if state['status']!='TRAINED' or m['max_updates']!=50000 or m['batch_size']!=256:raise ValueError('Wrong training status/budget')
        if m['data_manifest_sha256']!=manifest['data_manifest_sha256']:raise ValueError('Different training data')
        for name,expected in m['code_sha256'].items():
            if sha(ROOT/name)!=expected:raise ValueError('Changed training code')
        for name,ids in zip(('training','validation','held'),bundle.split(excluded)):
            if set(m[name+'_groups'])!={bundle.metadata[i]['group_id'] for i in ids}:raise ValueError('Group drift')
        if sm['checkpoint_sha256']!=sha(path/'BEST.pt') or sm['code_sha256']!=sha(ROOT/'scripts/score_temporal_context_model.py'):
            raise ValueError('Scoring checkpoint/code drift')
        fits[key]=dict(best_step=state['best_step'],steps=state['steps'],checkpoint_sha256=sha(path/'BEST.pt'),
            training_manifest_sha256=sha(path/'MANIFEST.json'),training_log_sha256=sha(path/'TRAINING.json'),
            scoring_manifest_sha256=sha(path/'scoring/MANIFEST.json'),score_sha256=sha(path/'scoring/STEP_SCORES.npz'))
    if sha(ROOT/'results/temporal_neural_queue_seed0_v1/RUN_STATE.json')!=manifest['old_flow_queue_state_sha256']:raise ValueError('Old flow queue state changed')
    return fits


def run():
    began=time.perf_counter();state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status'] not in ('SCORED_PENDING_EVALUATION','COMPLETE_REVIEWED'):raise ValueError('Incomplete TCN study')
    bundle=FeatureBundle(DATA,'innovation5');meta=bundle.metadata;fits=verify_inputs(bundle)
    records,joined=base.load_contract(ROOT.parents[1]);total=int(joined['offsets'][-1])
    if [r['uid'] for r in records]!=[m['uid'] for m in meta]:raise ValueError('Roster drift')
    all_scores,all_q=assemble(MODELS,'tcn','innovation5',(0,),meta,total)
    names={condition:'tcn__'+condition for condition in ('real','shuffled','zero')}
    scores={name:all_scores[condition+'__signed_residual_0.25'] for condition,name in names.items()}
    thresholds={name:all_q[condition+'__signed_residual_0.25'] for condition,name in names.items()}
    reference_root=ROOT/'results/aligned_context_predictors_v1';previous=json.loads((reference_root/'METRICS.json').read_text())
    with np.load(reference_root/'EVALUATED_SCORES.npz') as f:
        for name in f.files:scores[name]=f[name];thresholds[name]=previous['metrics'][name]['prmscore_thresholds']
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,calibration_thresholds=thresholds,fold_auc=True,pb_gate_open=gate)
    for name,old in previous['metrics'].items():
        for k in ('pb_all8','prm_within','prmscore_q08'):np.testing.assert_allclose(metrics[name][k],old[k],atol=2e-14,rtol=0)
    if (OUT/'RANKING_INTERIM.json').exists():
        interim=json.loads((OUT/'RANKING_INTERIM.json').read_text())
        for name,old in interim['metrics'].items():
            for k in ('pb_all8','prm_within'):np.testing.assert_allclose(metrics[name][k],old[k],atol=2e-14,rtol=0)
    primary=[('tcn__real',name) for name in ('ridge','bocpd','noreset','innovation5','tcn__shuffled','tcn__zero')]
    print('[tcn-evaluate] full metrics;10000 group draws',flush=True)
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=primary,primary_pairs=set(primary),primary_ci=1-.05/12)
    for a,b in primary:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    independent=independent_quality(records,joined,scores,gate,metrics,per)
    diagnostics=json.loads((OUT/'DIAGNOSTICS.json').read_text());prediction={}
    pa=json.loads((OUT/'PREDICTION_AUDIT.json').read_text())
    if sha(OUT/'DIAGNOSTICS.json')!=pa['diagnostics_sha256']:raise ValueError('Changed predictor diagnostics')
    if [d['uid'] for d in diagnostics]!=[m['uid'] for m in meta]:raise ValueError('Diagnostic roster mismatch')
    for condition,name in names.items():
        values=np.array([d['methods'][condition]['mse'] for d in diagnostics])
        late=np.array([d['methods'][condition]['mse_after16'] or [np.nan]*5 for d in diagnostics])
        prediction[name]=dict(scalar_mse_answer_mean=float(values.mean()),per_feature_answer_mean=values.mean(0).tolist(),
            per_feature_token_mean=np.average(values,axis=0,weights=bundle.length).tolist(),
            per_feature_after16_answer_mean=np.nanmean(late,axis=0).tolist(),
            gaussian_nll_answer_mean=float(np.mean([d['methods'][condition]['gaussian_nll'] for d in diagnostics])),
            correlations={k:distribution([d['methods'][condition][k] for d in diagnostics]) for k in
                ('prediction_correlation_ridge','signed_residual_correlation_ridge','residual_correlation_after_removing_current',
                 'signed_residual_correlation_current','prediction_real_correlation','signed_residual_real_correlation')})
    target=joined['target'];pb=np.array([r['cell'].startswith('pb_') for r in records]);location=(target+.5)/np.diff(joined['offsets']);transitions={}
    for ref in ('ridge','bocpd','noreset','innovation5'):
        hit=per['tcn__real']['prediction']==target;old=per[ref]['prediction']==target;transitions[ref]={}
        for label,mask in [('all_errors',pb&(target>=0)),('early',pb&(target>=0)&(location<1/3)),('middle',pb&(target>=0)&(location>=1/3)&(location<2/3)),('late',pb&(target>=0)&(location>=2/3))]:
            transitions[ref][label]=dict(answers=int(mask.sum()),both=int(np.sum(mask&hit&old)),gained=int(np.sum(mask&hit&~old)),lost=int(np.sum(mask&~hit&old)),neither=int(np.sum(mask&~hit&~old)))
    pareto=[name for name,m in metrics.items() if not any(n['pb_all8']>=m['pb_all8'] and n['prm_within']>=m['prm_within'] and
        (n['pb_all8']>m['pb_all8'] or n['prm_within']>m['prm_within']) for n in metrics.values())]
    write(OUT/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,prediction=prediction,reference_prediction=previous['prediction'],
        peak_transitions=transitions,pareto=pareto,seed=0,development_only=True,primary_endpoints=12,primary_ci=1-.05/12))
    np.savez_compressed(OUT/'SCORES_FROZEN.npz',**scores)
    write(OUT/'AUDIT.json',dict(status='PASS',answers=len(records),steps=total,tokens=int(bundle.length.sum()),fits=fits,
        independent_metrics=independent,prediction_audit=pa,all13_reference_headlines_exact=True,reused_fold0_unchanged=True,
        source_group_and_code_audit=True,evaluator_sha256=sha(Path(__file__)),scores_sha256=sha(OUT/'SCORES_FROZEN.npz')))
    lines=['method,PB_percent,within_AUC,PRMScore']+[f"{name},{100*m['pb_all8']:.6f},{m['prm_within']:.9f},{m['prmscore_q08']:.9f}" for name,m in metrics.items()]
    (OUT/'METRICS.csv').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    state.update(status='COMPLETE_REVIEWED',evaluation_seconds=time.perf_counter()-began,pareto=pareto,seed=0,
        development_only=True,correctness_labels_used=True,labels_scope='evaluation only; no fitting/scoring labels',fusion_fitted=False)
    write(OUT/'RUN_STATE.json',state)
    print('\n'.join(lines),flush=True);print(json.dumps(contrasts,indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
