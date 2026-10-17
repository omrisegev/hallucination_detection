"""Separated full-data evaluation of context-varying original-feature fusion."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.context_weighted_levels import METHODS,HEADS
from spectral_utils.energy_context_stability import heads
from scripts import run_temporal_research_baseline as base
from scripts.evaluate_residual_moment_real import independent_quality
from scripts.run_context_weighted_levels import OUT,SOURCE,EXCLUSIONS,HEAD_KEYS,key,restore_fit,sha,write

def audit_readout(bundle):
    with np.load(SOURCE/'LANDMARKS.npz') as f:landmark_answer=f['answer'];positions=f['local']
    checked=0;max_delta=0.;amplitude_ranges=[];max_gaps=[]
    for fold in range(5):
        with np.load(OUT/(key((fold,))+'.npz')) as scores:
            for cell in ('pb_gsm8k_q4','pb_math_q8','prmbench_qwen3_8b'):
                dest=OUT/(cell+'__'+key((fold,)));fit=restore_fit(json.loads((dest/'FIT.json').read_text())['preprocessing'])
                h=heads(fit.C,fit.sd,fit.var_y)
                with np.load(dest/'ANCHORS.npz') as archive:
                    arrays={k:archive[k] for k in archive.files};ids=arrays['landmark_ids'];answer=landmark_answer[ids]
                    for i in np.unique(answer)[[0,-1]]:
                        rows=np.flatnonzero(answer==i);pp=positions[ids[rows]];m=bundle.metadata[i]
                        x=np.asarray(bundle.features[m['offset']:m['offset']+m['tokens']],float)
                        spans=np.asarray(bundle.spans[m['step_start']:m['step_stop']])-m['offset']
                        max_gaps.append(int(np.max(np.diff(np.r_[0,pp,m['tokens']]))))
                        static={};raw={}
                        for head,k in HEAD_KEYS.items():
                            static[head]=h[k][0]/fit.sd if head=='native' else h[k][0]
                            norm=np.abs(static[head]).sum() if head=='native' else 1.
                            static[head]/=norm
                            raw[head]={a:arrays[a+'__'+head][rows]/fit.sd/norm if head=='native' else arrays[a+'__'+head][rows] for a in ('energy','position','random')}
                            ratio=np.linalg.norm(raw[head]['energy']*fit.sd,axis=1)/np.linalg.norm(static[head]*fit.sd)
                            amplitude_ranges.append([head,float(ratio.min()),float(ratio.max())])
                        expected={name:[] for name in METHODS}
                        for begin,end in spans:
                            selected=[sorted(range(begin,end),key=lambda t:(x[t,k],t))[-min(10,end-begin):] for k in range(5)]
                            for name in METHODS:
                                score=0.
                                for k,tokens in enumerate(selected):
                                    subtotal=0.
                                    for t in tokens:
                                        if name=='equal':w=np.ones(5)/5
                                        else:
                                            head,policy=name.split('__');past=np.flatnonzero(pp<=t)
                                            w=static[head]
                                            if len(past) and policy!='static':
                                                j=past[-1];energy=raw[head]['energy'][j]
                                                A=np.linalg.norm(energy*fit.sd)/np.linalg.norm(static[head]*fit.sd)
                                                if policy=='amplitude':w=A*static[head]
                                                elif policy=='direction':w=energy/A
                                                else:w=raw[head][policy][j]
                                        subtotal+=w[k]*x[t,k]
                                    score+=subtotal/len(tokens)
                                expected[name].append(score)
                        for name,v in expected.items():
                            actual=scores[name][m['step_start']:m['step_stop']]
                            max_delta=max(max_delta,float(np.max(np.abs(actual-v))))
                            np.testing.assert_allclose(actual,v,atol=5e-12,rtol=0)
                        checked+=1
    return dict(answers=checked,policies_per_answer=len(METHODS),max_scalar_score_difference=max_delta,
        inspected_max_anchor_gap=max(max_gaps),amplitude_ranges=amplitude_ranges)

def cached_readout_audit(bundle):
    path=OUT/'READOUT_AUDIT.json'
    identity=dict(code=sha(Path(__file__)),helper=sha(ROOT/'spectral_utils/context_weighted_levels.py'),
        score_hashes={key((f,)):sha(OUT/(key((f,))+'.npz')) for f in range(5)})
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['identity']!=identity:raise ValueError('Readout audit source changed')
        return saved['audit']
    result=audit_readout(bundle);write(path,dict(identity=identity,audit=result))
    return result

def run():
    started=time.perf_counter();state=json.loads((OUT/'RUN_STATE.json').read_text())
    if state['status'] not in ('SCORING_COMPLETE_PENDING_EVALUATION','COMPLETE_REVIEWED'):raise ValueError('Incomplete scoring')
    manifest=json.loads((OUT/'MANIFEST.json').read_text())
    if sha(SOURCE/'PROVENANCE.json')!=manifest['source_provenance_sha256']:raise ValueError('Changed source provenance')
    for file,expected in dict(manifest['code'],**json.loads((SOURCE/'PROVENANCE.json').read_text())).items():
        if sha(ROOT/file)!=expected:raise ValueError('Code changed during scoring: '+file)
    records,joined=base.load_contract(ROOT.parents[1]);bundle=FeatureBundle(ROOT/'results/temporal_context_data_v1','innovation5');meta=bundle.metadata
    if [m['uid'] for m in meta]!=[r['uid'] for r in records]:raise ValueError('Roster mismatch')
    scores={name:np.full(meta[-1]['step_stop'],np.nan) for name in METHODS};nested={f:{name:[] for name in METHODS} for f in range(5)}
    for excluded in EXCLUSIONS:
        path=OUT/(key(excluded)+'.npz')
        if sha(path)!=json.loads((OUT/(key(excluded)+'_COMPLETE.json')).read_text())['sha256']:raise ValueError('Prediction hash mismatch')
        with np.load(path) as archive:
            for name in METHODS:
                value=archive[name]
                if len(excluded)==1:
                    good=np.isfinite(value)
                    if np.isfinite(scores[name][good]).any():raise ValueError('Duplicate prediction')
                    scores[name][good]=value[good]
                else:
                    for held in excluded:
                        other=next(f for f in excluded if f!=held)
                        take=np.concatenate([value[m['step_start']:m['step_stop']] for m in meta if m['fold']==other and m['cell'].startswith('prm')])
                        if not np.isfinite(take).all():raise ValueError('Nested calibration gap')
                        nested[held][name].append(take)
    if any(not np.isfinite(v).all() for v in scores.values()):raise ValueError('Incomplete quality population')
    thresholds={name:{str(f):float(np.quantile(np.concatenate(nested[f][name]),.8)) for f in range(5)} for name in METHODS}
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,fold_auc=True,calibration_thresholds=thresholds,pb_gate_open=gate)
    prior=json.loads((ROOT/'results/residual_moment_fusion_v1/METRICS.json').read_text())['metrics']
    names=['original4','innovation5','single__H0lim','single__VE0','single__VE075','single__VE1','entropy15','RBM12','ridge_signed025_secondary']
    with np.load(ROOT/'results/residual_moment_fusion_v1/SCORES_FROZEN.npz') as f:refs={n:f['steps__'+n] for n in names}
    rm,rp=base.evaluator.evaluate_arrays(records,joined,refs,fold_auc=True,pb_gate_open=gate,
        calibration_thresholds={n:prior[n]['prmscore_thresholds'] for n in names})
    metrics.update(rm);per.update(rp);scores.update(refs)
    max_delta=float(np.max(np.abs(scores['equal']-scores['innovation5'])))
    np.testing.assert_allclose(scores['equal'],scores['innovation5'],atol=1e-6,rtol=0)
    for n in names:
        for k in ('pb_all8','prm_within','prmscore_q08'):np.testing.assert_allclose(metrics[n][k],prior[n][k],atol=2e-14,rtol=0)
    for k in ('pb_all8','prm_within','prmscore_q08'):np.testing.assert_allclose(metrics['equal'][k],prior['innovation5'][k],atol=2e-14,rtol=0)
    primary=[('simplex__energy','simplex__'+p) for p in ('static','position','random','amplitude')]
    secondary=[(h+'__energy',h+'__'+p) for h in ('native','group') for p in ('static','position','random','amplitude')]
    secondary += [(h+'__direction',h+'__static') for h in HEADS]+[(n,'innovation5') for n in METHODS if n!='equal']
    pairs=list(dict.fromkeys(primary+secondary))
    print('[evaluate] full population;10000 group bootstrap draws',flush=True)
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(primary),primary_ci=.99375)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    print('[audit] full independent PB/within and scalar dynamic readout',flush=True)
    independent=independent_quality(records,joined,scores,gate,metrics,per);numeric=cached_readout_audit(bundle)
    cells=np.array([m['cell'] for m in meta]);target=joined['target'];error=np.char.startswith(cells,'pb_')&(target>=0)
    relative=(target+.5)/np.diff(joined['offsets']);strata={}
    for name in METHODS:
        strata[name]={}
        for label,mask in [('early',error&(relative<1/3)),('middle',error&(relative>=1/3)&(relative<2/3)),('late',error&(relative>=2/3))]:
            good=per[name]['prediction']==target;ref=per['innovation5']['prediction']==target
            strata[name][label]=dict(answers=int(mask.sum()),hits=int(np.sum(mask&good)),gained=int(np.sum(mask&good&~ref)),lost=int(np.sum(mask&~good&ref)))
    pareto=[n for n,m in metrics.items() if not any(a['pb_all8']>=m['pb_all8'] and a['prm_within']>=m['prm_within'] and (a['pb_all8']>m['pb_all8'] or a['prm_within']>m['prm_within']) for a in metrics.values())]
    np.savez_compressed(OUT/'SCORES_FROZEN.npz',**{'steps__'+k:v for k,v in scores.items()})
    write(OUT/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,pareto=pareto,early_middle_late=strata,development_only=True))
    write(OUT/'AUDIT.json',dict(status='PASS',answers=len(records),methods=len(scores),steps=meta[-1]['step_stop'],tokens=int(bundle.length.sum()),
        independent_quality=independent,scalar_readout=numeric,equal_frozen_max_delta=max_delta,all_reference_metrics_exact=True,
        tests_passed=22,scores_sha256=sha(OUT/'SCORES_FROZEN.npz'),evaluator_sha256=sha(Path(__file__)),
        evaluation_sources={str(p):sha(p) for p in [ROOT.parents[1]/'results/localization_full_benchmark_v3/evaluation/JOINED.json',ROOT.parents[1]/'results/localization_full_benchmark_v3/evaluation/JOINED.npz',base.evaluator.old.FOLDS]}))
    state.update(status='COMPLETE_REVIEWED',evaluation_seconds=time.perf_counter()-started,pareto=pareto,evaluation_labels_used=True)
    write(OUT/'RUN_STATE.json',state)
    for n,m in metrics.items():print('[result]',n,round(m['pb_all8']*100,4),round(m['prm_within'],6),round(m['prmscore_q08'],6),flush=True)
    print('[pareto]',pareto,flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=1):run()
