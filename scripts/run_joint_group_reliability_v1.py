"""Change only outer group fusion using saved frozen Joint models."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_fixed_readout_v1 import inputs,OUT as PREVIOUS,BANKS
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_digitfree_broad50_v1 import sha
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from spectral_utils.joint_redundancy_stress import augment
from spectral_utils.joint_group_reliability import group_reliability_weights
from spectral_utils.lsml_gate_locator_research import _orient
from spectral_utils.digitfree_broad50 import ANCHOR
OUT=ROOT/'results/joint_group_reliability_v1'


def state(status,**kwargs):
    dump(OUT/'RUN_STATE.json',dict(status=status,time=time.strftime('%Y-%m-%dT%H:%M:%S'),**kwargs))
    print(status,kwargs,flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    paths=[Path(__file__),ROOT/'spectral_utils/joint_group_reliability.py',
        ROOT/'docs/experiments/JOINT_GROUP_RELIABILITY_V1.md',PREVIOUS/'MANIFEST.json',
        PREVIOUS/'AUDIT.json',PREVIOUS/'SCORES.npz',
        *[PREVIOUS/f'{b}_fold{f}.json' for b in BANKS for f in range(5)]]
    manifest=dict(schema='joint-group-reliability-v1',hashes={str(p):sha(p) for p in paths})
    path=OUT/'MANIFEST.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('MANIFEST_DRIFT')
    dump(path,manifest);state('LOADING');data,base,uids=inputs()
    rowfold=np.repeat(data['folds'],np.diff(data['offsets']))
    with np.load(PREVIOUS/'SCORES.npz') as z:
        scores={k:z[k] for k in z.files if k!='gate'};np.testing.assert_array_equal(z['gate'],data['gate'])
    native={}
    for bank in BANKS:
        x=base if bank=='base' else augment(base,data['offsets'],uids,bank)
        for arm in ('full','auto'):
            scores[bank+'__reliability_'+arm]=np.full(len(x),np.nan);native[bank+'__reliability_'+arm]=0
        for outer in range(5):
            old=json.loads((PREVIOUS/f'{bank}_fold{outer}.json').read_text());held=rowfold==outer
            meta=dict(bank=bank,outer=outer,arms={})
            for arm,a in old['arms'].items():
                if a['valid']:
                    local,detail=group_reliability_weights(a['model_covariance'],a['global_loading'],a['labels'])
                    w=np.zeros(x.shape[1]);w[a['active']]=local;w,orientation=_orient(x[~held],w,ANCHOR)
                    m=dict(valid=True,weights=w,active=a['active'],readout=detail,orientation=orientation,
                        added_absolute_weight=float(np.abs(w[51:]).sum()))
                    native[bank+'__reliability_'+arm]+=int(np.sum(data['folds']==outer))
                else:
                    w=np.eye(x.shape[1])[ANCHOR];m=dict(valid=False,weights=w,failure=a['failure'])
                scores[bank+'__reliability_'+arm][held]=x[held]@w;meta['arms'][arm]=m
            dump(OUT/f'{bank}_fold{outer}.json',meta);state('FOLD_COMPLETE',bank=bank,outer=outer)
    state('EVALUATING');metrics={k:score_locator(s,data['gate'],data) for k,s in scores.items()}
    pairs=[(b+'__reliability_auto',b+'__joint_auto') for b in BANKS]
    pairs += [(b+'__reliability_auto','base__reliability_auto') for b in ('duplicates','noise')]
    ci=bootstrap(data,metrics,pairs)
    preservation={b:bool(native[b+'__reliability_auto']==13769 and
        ci[b+'__reliability_auto minus base__reliability_auto']['pb']['low']>-.01 and
        ci[b+'__reliability_auto minus base__reliability_auto']['within']['low']>-.002) for b in ('duplicates','noise')}
    result=dict(status='COMPLETE',metrics={k:scalar_metrics(m) for k,m in metrics.items()},
        primary_contrasts=ci,native_answers=native,practical_preservation=preservation)
    dump(OUT/'RESULTS.json',result);np.savez_compressed(OUT/'SCORES.npz',**scores,gate=data['gate'])
    lines=['# Model-derived outer group reliability','','| Method | PB % | Within AUC | Native |','|---|---:|---:|---:|']
    for k,m in result['metrics'].items():lines.append(f"| {k} | {100*m['pb']:.4f} | {m['within']:.6f} | {native.get(k,'frozen reference')} |")
    lines+=['','Noise candidates include2730 explicit H1 fallback answers.',
        'Only outer group weighting changed; all fits, selected supports and within-group v weights frozen.',
        'Practical preservation: '+str(preservation),'','```json',json.dumps(ci,indent=2),'```']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    state('COMPLETE',practical_preservation=preservation);print('\n'.join(lines[27:36]),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:state('FAILED',error=str(exc));raise
