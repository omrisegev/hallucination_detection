"""Frozen noise-aware Joint on near-copies and structured nuisance families."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_fixed_readout_v1 import inputs
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from spectral_utils.joint_structured_stress import structured_augmentation
from spectral_utils.joint_noise_aware import fit_noise_aware_joint,score_noise_aware_joint
from spectral_utils.lsml_gate_locator_research import fit_fusion_weights,FusionRecipe
from spectral_utils.digitfree_broad50 import ANCHOR,NAMES
PREVIOUS=ROOT/'results/joint_sparse_refinement_v1';OUT=ROOT/'results/joint_structured_stress_v1'
KINDS=('near_copies','structured_noise')


def state(status,**kwargs):
    dump(OUT/'RUN_STATE.json',dict(status=status,time=time.strftime('%Y-%m-%dT%H:%M:%S'),**kwargs))
    print(status,kwargs,flush=True)


def training_inputs(data,x,outer):
    answers=data['folds']!=outer;rows=np.repeat(answers,np.diff(data['offsets']))
    offsets=np.r_[0,np.cumsum(np.diff(data['offsets'])[answers])]
    return rows,(x[rows],offsets,data['groups'][answers],data['folds'][answers])


def main():
    OUT.mkdir(exist_ok=True)
    modules=('joint_noise_aware','joint_structured_stress','joint_sparse_membership',
        'joint_sparse_refinement','joint_feature_selection','joint_block_balanced','joint_group_reliability')
    files=[Path(__file__),ROOT/'docs/experiments/JOINT_STRUCTURED_REDUNDANCY_STRESS_V1.md',
        *[ROOT/f'spectral_utils/{m}.py' for m in modules],PREVIOUS/'SCORES.npz',PREVIOUS/'AUDIT.json',
        PREVIOUS/'MANIFEST.json',*[PREVIOUS/f'base_fold{f}.json' for f in range(5)]]
    manifest=dict(schema='joint-structured-stress-v1',hashes={str(p):sha(p) for p in files})
    path=OUT/'MANIFEST.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('MANIFEST_DRIFT')
    dump(path,manifest);state('LOADING');data,base,uids=inputs()
    with np.load(PREVIOUS/'SCORES.npz') as z:
        scores={k:z[k] for k in ('base__refined','base__continuous','base__equal_full','base__reliability_auto','innovation5','historical_bocpd')}
        np.testing.assert_array_equal(z['gate'],data['gate'])
    replay=[]
    for outer in range(5):
        path=OUT/f'base_replay_fold{outer}.json';rows,args=training_inputs(data,base,outer)
        if path.exists():record=json.loads(path.read_text());model=record['model']
        else:
            def notify(stage,**detail):state(stage,bank='base_replay',outer=outer,**detail)
            started=time.perf_counter();model=fit_noise_aware_joint(*args,anchor_index=ANCHOR,seed=399170+outer,notify=notify)
            assert model['valid'],model.get('failure')
            frozen=json.loads((PREVIOUS/f'base_fold{outer}.json').read_text())
            np.testing.assert_array_equal(model['active'],frozen['active'])
            np.testing.assert_allclose(model['canonical_weights'],frozen['canonical_weights'],atol=1e-10,rtol=1e-10)
            record=dict(model=model,seconds=time.perf_counter()-started)
            dump(path,record)
        score=score_noise_aware_joint(model,base[~rows])
        error=float(np.max(np.abs(score-scores['base__refined'][~rows])))
        np.testing.assert_allclose(score,scores['base__refined'][~rows],atol=1e-11,rtol=1e-11)
        replay.append(error);state('BASE_REPLAY_PASS',outer=outer,max_error=error)
    native={};all_meta={}
    for kind in KINDS:
        x=structured_augmentation(base,data['offsets'],uids,kind)
        names=tuple(NAMES)+('bocpd_residual',)+tuple(f'{kind}_{i}' for i in range(15))
        for arm in ('joint','continuous','equal'):
            scores[kind+'__'+arm]=np.full(len(x),np.nan);native[kind+'__'+arm]=0
        metas=[]
        for outer in range(5):
            path=OUT/f'{kind}_fold{outer}.json';rows,args=training_inputs(data,x,outer)
            if path.exists():meta=json.loads(path.read_text());model=meta['model']
            else:
                started=time.perf_counter()
                def notify(stage,**detail):state(stage,bank=kind,outer=outer,**detail)
                model=fit_noise_aware_joint(*args,anchor_index=ANCHOR,seed=399170+outer,notify=notify)
                weights={'equal':np.ones(x.shape[1])/x.shape[1]};valid={'equal':True,'joint':model['valid']}
                meta=dict(kind=kind,outer=outer,model=model,valid=valid)
                try:
                    w,detail=fit_fusion_weights(args[0],FusionRecipe(kind,names,'continuous',anchor=ANCHOR),seed=399170+outer)
                    weights['continuous']=w;valid['continuous']=True;meta['continuous']=detail
                except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                    weights['continuous']=np.eye(x.shape[1])[ANCHOR];valid['continuous']=False;meta['continuous_failure']=str(exc)
                meta.update(weights=weights,seconds=time.perf_counter()-started)
                if model['valid']:
                    selected=model['active_original'];meta.update(selected_added=sum(i>=51 for i in selected),
                        selected_original=sum(i<51 for i in selected),bocpd_retained=50 in selected)
                dump(path,meta)
            scores[kind+'__joint'][~rows]=score_noise_aware_joint(model,x[~rows],allow_anchor_fallback=True)
            for arm in ('continuous','equal'):scores[kind+'__'+arm][~rows]=x[~rows]@np.asarray(meta['weights'][arm])
            for arm,valid in meta['valid'].items():
                if valid:native[kind+'__'+arm]+=int(np.sum(data['folds']==outer))
            metas.append(meta);state('FOLD_COMPLETE',bank=kind,outer=outer,valid=meta['valid'],
                selected_added=meta.get('selected_added'),seconds=meta['seconds'])
        all_meta[kind]=metas
    state('EVALUATING');metrics={k:score_locator(s,data['gate'],data) for k,s in scores.items()}
    ci=bootstrap(data,metrics,[(k+'__joint','base__refined') for k in KINDS])
    preservation={k:bool(native[k+'__joint']==13769 and ci[k+'__joint minus base__refined']['pb']['low']>-.01 and
        ci[k+'__joint minus base__refined']['within']['low']>-.002) for k in KINDS}
    result=dict(status='COMPLETE',metrics={k:scalar_metrics(m) for k,m in metrics.items()},primary_contrasts=ci,
        native_answers=native,practical_preservation=preservation,baseline_api_replay_max_error=max(replay),
        selected_original={k:[m.get('selected_original') for m in fs] for k,fs in all_meta.items()},
        selected_added={k:[m.get('selected_added') for m in fs] for k,fs in all_meta.items()},
        bocpd_retained={k:[m.get('bocpd_retained') for m in fs] for k,fs in all_meta.items()})
    dump(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    lines=['# Approximate-copy and structured nuisance verification','','| Method | PB % | Within AUC | Native |','|---|---:|---:|---:|']
    for k,m in result['metrics'].items():lines.append(f"| {k} | {100*m['pb']:.4f} | {m['within']:.6f} | {native.get(k,'frozen reference')} |")
    lines+=['','Practical preservation: '+str(preservation),'Added retained: '+str(result['selected_added']),
        'Original retained: '+str(result['selected_original']),'','```json',json.dumps(ci,indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');state('COMPLETE',practical_preservation=preservation)
    print('\n'.join(lines[:18]),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:state('FAILED',error=str(exc));raise
