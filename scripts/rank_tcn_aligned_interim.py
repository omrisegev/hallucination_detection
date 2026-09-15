"""Full-population PB/within only, while nested PRMScore fits finish.

No partial-answer ranking, no calibration or parameter selection. Final
evaluation must reproduce these points; this is not a completed study.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts.run_tcn_aligned_study import OUT,MODELS,DATA,write,sha
from scripts import run_temporal_research_baseline as base
from spectral_utils.context_training import FeatureBundle
from spectral_utils.pb_prediction_bundle import prediction_bundle


def run():
    bundle=FeatureBundle(DATA,'innovation5');records,joined=base.load_contract(ROOT.parents[1]);meta=bundle.metadata
    if [r['uid'] for r in records]!=[m['uid'] for m in meta]:raise ValueError('Roster drift')
    total=int(joined['offsets'][-1]);scores={'tcn__'+k:np.full(total,np.nan) for k in ('real','shuffled','zero')};hashes={}
    for fold in range(5):
        path=MODELS/f'tcn__innovation5__seed0__exclude{fold}'/'scoring'
        if json.loads((path/'RUN_STATE.json').read_text())['status']!='SCORED':raise ValueError('All five outer folds required')
        with np.load(path/'STEP_SCORES.npz') as f:
            arrays={k:f[k+'__signed_residual_0.25'] for k in ('real','shuffled','zero')}
            for m in meta:
                if m['fold']!=fold:continue
                sl=slice(m['step_start'],m['step_stop'])
                for k in ('real','shuffled','zero'):scores['tcn__'+k][sl]=arrays[k][sl]
        hashes[str(fold)]=sha(path/'STEP_SCORES.npz')
    previous=ROOT/'results/aligned_context_predictors_v1'
    with np.load(previous/'EVALUATED_SCORES.npz') as f:
        for k in ('ridge','bocpd','noreset','innovation5'):scores[k]=f[k]
    with np.load(ROOT/'results/temporal_research_baseline_v1/SCORES_FROZEN.npz') as f:gate=f['gate_percentile']>=.33
    cells=np.array([r['cell'] for r in records]);valid=np.ones(len(records),bool);metrics={}
    for name,flat in scores.items():
        if not np.isfinite(flat).all():raise ValueError('Incomplete full population')
        peaks=np.array([int(np.argmax(flat[a:b])) for a,b in zip(joined['offsets'][:-1],joined['offsets'][1:])])
        pb,_,_=prediction_bundle(joined['target'],cells,peaks,valid,gate);within=[]
        for i,r in enumerate(records):
            if r['cell'].startswith('pb_'):continue
            sl=slice(joined['offsets'][i],joined['offsets'][i+1]);y=joined['labels'][sl];s=flat[sl]
            pos=s[y==1];neg=s[y==0]
            if len(pos) and len(neg):
                d=pos[:,None]-neg[None,:];within.append(float(np.mean((d>0)+.5*(d==0))))
        metrics[name]=dict(pb_all8=pb['pb_all8'],prm_within=float(np.mean(within)),within_answers=len(within))
    old=json.loads((previous/'METRICS.json').read_text())['metrics']
    for name in ('ridge','bocpd','noreset','innovation5'):
        for k in ('pb_all8','prm_within'):np.testing.assert_allclose(metrics[name][k],old[name][k],atol=2e-14,rtol=0)
    write(OUT/'RANKING_INTERIM.json',dict(status='FULL_RANKING_ONLY_NESTED_CALIBRATION_PENDING',answers=len(records),
        metrics=metrics,outer_score_hashes=hashes,code_sha256=sha(Path(__file__)),development_only=True,parameter_selection=False))
    print(json.dumps(metrics,indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
