"""Full frozen-benchmark position-profile diagnostic; no neural training."""
import argparse
import hashlib
import io
import itertools
import json
from pathlib import Path
import sqlite3
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from spectral_utils.context_training import FeatureBundle
from spectral_utils.temporal_position_profiles import METHODS,answer_moments,fit_profiles,transform,length_class
from spectral_utils.temporal_research_features import BASELINE,prefix_innovation
from spectral_utils.direct_probability_fusion import step_top_mean
from scripts import run_temporal_research_baseline as base
from scripts.analyze_temporal_research_baseline import independent_pb

OUT=ROOT/'results/temporal_position_control_v1'
DATA=ROOT/'results/temporal_context_data_v1'
BASE=ROOT/'results/temporal_research_baseline_v1'
INNOV='append_innovation__H0lim'
def save(path,x):base.common.atomic_json(path,x)
def sha(path):return base.common.sha256_file(path)

def prepare(bundle,reference):
    state=OUT/'PREPARED.json'
    if state.exists():
        old=base.read_json(state)
        for name in ('h0_float64.npy','answer_moments.npy'):
            if sha(OUT/name)!=old['hashes'][name]:raise ValueError('prepared data hash changed')
        return np.load(OUT/'h0_float64.npy',mmap_mode='r'),np.load(OUT/'answer_moments.npy')
    h0=np.lib.format.open_memmap(OUT/'h0_float64.npy',mode='w+',dtype=np.float64,shape=(int(bundle.manifest['tokens']),))
    stats=np.zeros((len(bundle.metadata),16,3));largest=0.
    con=sqlite3.connect('file:'+str(BASE/'CHECKPOINT.sqlite')+'?mode=ro',uri=True)
    count=0
    for i,blob in con.execute('SELECT idx,payload FROM answers ORDER BY idx'):
        if i!=count:raise ValueError('source roster gap')
        m=bundle.metadata[i];start=m['offset'];n=m['tokens'];sl=slice(m['step_start'],m['step_stop'])
        with np.load(io.BytesIO(blob),allow_pickle=False) as f:
            x=f['features'][:,0];spans=f['spans']
            np.testing.assert_array_equal(spans,bundle.spans[sl]-start)
        if len(x)!=n:raise ValueError('source token count mismatch')
        h0[start:start+n]=x;innovation,_=prefix_innovation(x);stats[i]=answer_moments(innovation)
        score=step_top_mean(innovation,spans[:,0],spans[:,1],10)
        delta=float(np.max(np.abs(score-reference[sl])));largest=max(largest,delta)
        if delta>1e-12:raise ValueError('original innovation replay mismatch')
        count+=1
        if count%1000==0:print('[prepare]',count,flush=True)
    con.close()
    if count!=13769:raise ValueError('full population required')
    h0.flush();np.save(OUT/'answer_moments.npy',stats)
    save(state,dict(answers=count,max_innovation_step_difference=largest,
        hashes={n:sha(OUT/n) for n in ('h0_float64.npy','answer_moments.npy')}))
    return h0,stats

def score_fit(bundle,h0,stats,baseline,excluded):
    name='exclude_'+'_'.join(map(str,excluded));dest=OUT/name;dest.mkdir(exist_ok=True)
    if (dest/'COMPLETE.json').exists():
        audit=base.read_json(dest/'COMPLETE.json')
        if audit['scores_sha256']!=sha(dest/'SCORES.npz'):raise ValueError('fit scores changed')
        return dest
    profiles=fit_profiles(bundle.metadata,stats,excluded);save(dest/'PROFILES.json',profiles)
    training_groups=set(profiles['training_groups'])
    scores={n:np.full(len(baseline),np.nan) for n in METHODS};count=0
    for i,m in enumerate(bundle.metadata):
        if m['fold'] not in excluded or (len(excluded)==2 and m['cell'].startswith('pb_')):continue
        if m['group_id'] in training_groups:raise ValueError('fit/score source overlap')
        start=m['offset'];innovation,_=prefix_innovation(h0[start:start+m['tokens']])
        sl=slice(m['step_start'],m['step_stop']);spans=bundle.spans[sl]-start
        p=profiles['profiles'][m['cell']+'|'+str(length_class(m['tokens']))]
        for key,values in transform(innovation,p).items():
            scores[key][sl]=(4*baseline[sl]+step_top_mean(values,spans[:,0],spans[:,1],10))/5
        count+=1
    np.savez_compressed(dest/'SCORES.npz',**scores)
    save(dest/'COMPLETE.json',dict(answers=count,excluded_folds=list(excluded),scores_sha256=sha(dest/'SCORES.npz'),
        profile_sha256=sha(dest/'PROFILES.json'),stratum_fallbacks=profiles['fallbacks']))
    print('[fit and score]',name,count,flush=True);return dest

def assemble(bundle,paths,total):
    scores={n:np.full(total,np.nan) for n in METHODS};thresholds={n:{} for n in METHODS}
    nested={h:{n:[] for n in METHODS} for h in range(5)}
    for excluded,path in paths.items():
        with np.load(path/'SCORES.npz') as archive:
            f={n:archive[n] for n in METHODS}
            for m in bundle.metadata:
                if m['fold'] not in excluded:continue
                sl=slice(m['step_start'],m['step_stop'])
                if len(excluded)==1:
                    for n in METHODS:scores[n][sl]=f[n][sl]
                elif not m['cell'].startswith('pb_'):
                    held=next(h for h in excluded if h!=m['fold'])
                    for n in METHODS:nested[held][n].append(f[n][sl])
    for n in METHODS:
        if not np.isfinite(scores[n]).all():raise ValueError('incomplete outer coverage')
        for h in range(5):
            values=np.concatenate(nested[h][n])
            if not np.isfinite(values).all():raise ValueError('incomplete nested calibration')
            thresholds[n][str(h)]=float(np.quantile(values,.8))
    return scores,thresholds

def audit_metrics(records,joined,scores,metrics,per):
    cells=np.array([r['cell'] for r in records]);audit={}
    for name,flat in scores.items():
        table,pb=independent_pb(joined['target'],cells,per[name]['prediction'],per[name]['decision_valid'])
        np.testing.assert_allclose(pb,metrics[name]['pb_all8'],rtol=0,atol=1e-14)
        for c,x in table.items():
            for field in ('clean','erroneous','f1'):
                np.testing.assert_allclose(x[field],metrics[name]['pb_cells'][c][field],rtol=0,atol=1e-14)
            np.testing.assert_allclose(x['clean_hits']/x['clean'],metrics[name]['pb_cells'][c]['clean_accuracy'],rtol=0,atol=1e-14)
            np.testing.assert_allclose(x['error_hits']/x['erroneous'],metrics[name]['pb_cells'][c]['error_exact_accuracy'],rtol=0,atol=1e-14)
        values=[]
        for i,r in enumerate(records):
            if r['cell'].startswith('pb_'):continue
            sl=slice(joined['offsets'][i],joined['offsets'][i+1]);y=joined['labels'][sl];s=flat[sl]
            pos=s[y==1];neg=s[(y>=0)&(y!=1)]
            if len(pos) and len(neg):
                auc=float(((pos[:,None]>neg).sum()+.5*(pos[:,None]==neg).sum())/(len(pos)*len(neg)))
                np.testing.assert_allclose(auc,per[name]['within'][i],rtol=0,atol=1e-14);values.append(auc)
        np.testing.assert_allclose(np.mean(values),metrics[name]['prm_within'],rtol=0,atol=1e-14)
        audit[name]=dict(pb=pb,pb_cells=table,within=float(np.mean(values)),within_answers=len(values))
    return audit

def run(source):
    began=time.time();OUT.mkdir(parents=True,exist_ok=True)
    bundle=FeatureBundle(DATA);metadata=bundle.metadata
    spec=dict(schema='temporal-position-control-v1',contract_sha256=sha(ROOT/'docs/experiments/TEMPORAL_REVIEW_FOLLOWUP_20260915.md'),
        data_manifest_sha256=sha(DATA/'MANIFEST.json'),baseline_scores_sha256=sha(BASE/'SCORES_FROZEN.npz'),
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'spectral_utils/temporal_position_profiles.py']},
        methods=list(METHODS),source_root=str(source),development_only=True,bootstrap_draws=10000)
    manifest=OUT/'MANIFEST.json'
    if manifest.exists() and base.read_json(manifest)!=spec:raise ValueError('immutable run manifest changed')
    save(manifest,spec)
    with np.load(BASE/'SCORES_FROZEN.npz') as f:
        baseline=f['steps__'+BASELINE];innovation_step=f['steps__innovation__H0lim'];gate=f['gate_percentile']>=.33
        references={BASELINE:baseline,INNOV:f['steps__'+INNOV]}
    h0,stats=prepare(bundle,innovation_step)
    exclusions=[(h,) for h in range(5)]+list(itertools.combinations(range(5),2))
    paths={}
    for ex in exclusions:
        paths[ex]=score_fit(bundle,h0,stats,baseline,ex)
        save(OUT/'RUN_STATE.json',dict(status='SCORING',fits=len(paths),expected_fits=15,seconds=time.time()-began))
    scores,thresholds=assemble(bundle,paths,len(baseline));save(OUT/'CALIBRATION.json',thresholds)
    # Correctness labels are first loaded here, after all fits and scores exist.
    records,joined=base.load_contract(source)
    if [r['uid'] for r in records]!=[m['uid'] for m in metadata]:raise ValueError('evaluation roster mismatch')
    folds=base.read_json(base.evaluator.old.FOLDS)['outer']
    for r,m in zip(records,metadata):
        if r['group_id']!=m['group_id'] or int(folds[r['group_id']])!=m['fold']:raise ValueError('fold/group drift')
    metrics,per=base.evaluator.evaluate_arrays(records,joined,scores,fold_auc=True,calibration_thresholds=thresholds,pb_gate_open=gate)
    rm,rp=base.evaluator.evaluate_arrays(records,joined,references,fold_auc=True,pb_gate_open=gate)
    metrics.update(rm);per.update(rp);scores.update(references)
    for n in references:
        expected=base.read_json(BASE/'METRICS.json')['metrics'][n]
        for k in ('pb_all8','prm_within','prmscore_q08'):np.testing.assert_allclose(metrics[n][k],expected[k],rtol=0,atol=1e-14)
    print('[evaluate] all answers',len(records),flush=True)
    save(OUT/'INDEPENDENT_AUDIT.json',dict(status='PASS',results=audit_metrics(records,joined,scores,metrics,per)))
    np.savez_compressed(OUT/'SCORES_FROZEN.npz',**{'steps__'+k:v for k,v in scores.items()})
    np.savez_compressed(OUT/'PREDICTIONS.npz',**{k+'__'+n:v for n,p in per.items() for k,v in p.items()})
    primary=[('mean_detrended','profile_only'),('location_scale_detrended','profile_only')]
    pairs=primary+[(n,ref) for n in ('mean_detrended','location_scale_detrended') for ref in (BASELINE,INNOV)]
    pairs += [(INNOV,BASELINE),(INNOV,'profile_only'),('profile_only','constant_profile')]
    print('[bootstrap]',len(pairs),'contrasts',flush=True)
    contrasts=base.evaluator.paired_bootstrap(records,joined,per,draws=10000,pairs=pairs,primary_pairs=set(primary),primary_ci=.9875)
    for a,b in pairs:contrasts[a+'_minus_'+b]['pb_delta']=metrics[a]['pb_all8']-metrics[b]['pb_all8']
    save(OUT/'METRICS.json',dict(metrics=metrics,contrasts=contrasts,development_only=True,
        uncertainty='98.75% for two primary pairs x two endpoints; descriptive95% elsewhere; no historical adaptive-selection correction.'))
    error=np.array([r['cell'].startswith('pb_') for r in records])&(joined['target']>=0)
    relative=joined['target']/np.maximum(np.array([r['steps'] for r in records])-1,1)
    strata={'early':error&(relative<1/3),'middle':error&(relative>=1/3)&(relative<2/3),'late':error&(relative>=2/3)}
    strata.update({'length_'+str(k):error&np.array([length_class(r['tokens'])==k for r in records]) for k in range(4)})
    save(OUT/'STRATA.json',{s:dict(answers=int(mask.sum()),methods={n:dict(exact=int((mask&p['decision_valid']&(p['prediction']==joined['target'])).sum()),
        gained=int((mask&p['decision_valid']&(p['prediction']==joined['target'])&~(per[INNOV]['decision_valid']&(per[INNOV]['prediction']==joined['target']))).sum()),
        lost=int((mask&per[INNOV]['decision_valid']&(per[INNOV]['prediction']==joined['target'])&~(p['decision_valid']&(p['prediction']==joined['target']))).sum())) for n,p in per.items()}) for s,mask in strata.items()})
    lines=['# Full-population position-profile controls','','Frozen post-review diagnostic; development-only.','',
        '| Method | PB % | Within AUC | PRMScore |','|---|---:|---:|---:|']
    for n,m in metrics.items():lines.append(f"| {n} | {100*m['pb_all8']:.4f} | {m['prm_within']:.6f} | {m['prmscore_q08']:.6f} |")
    historical=base.read_json(BASE/'METRICS.json')['metrics']['readout__earlier_VE0_VE075_peak']
    lines+=['',f"Historical earlier-VE0/VE075-peak PB only: {100*historical['pb_all8']:.4f}%. This is a step-choice rule, not a PRMB score curve.",
        '', 'See METRICS.json for all paired contrasts and primary98.75% intervals. Independent scalar PB and pairwise within-AUC audit: PASS.',
        'Fits use no labels; source-excluded cell/length profiles borrow other answers. First-token missing history remains zero.',
        'All references, Top10 order and gate are unchanged. Profile controls do not establish complete removal of every position interaction.',
        'Original innovation5 remains visible; no new candidate is promoted by this diagnostic.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save(OUT/'RUN_STATE.json',dict(status='COMPLETE_REVIEWED',answers=len(records),steps=len(baseline),tokens=len(h0),fits=15,
        seconds=time.time()-began,score_sha256=sha(OUT/'SCORES_FROZEN.npz'),full_program_complete=False))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-root',type=Path,required=True);a=p.parse_args()
    try:
        with threadpool_limits(limits=1):run(a.source_root)
    except BaseException as e:
        OUT.mkdir(parents=True,exist_ok=True);save(OUT/'RUN_STATE.json',dict(status='FAILED',error=f'{type(e).__name__}: {e}'));raise
