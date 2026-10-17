"""Post-evaluation diagnostic: full versus refined readout, base/near copies.

No new fits or selected parameters. All13769 matched rows; cannot be called
an untouched candidate or a deployable parent-aware grouping rule.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_fixed_readout_v1 import inputs
from scripts.run_joint_structured_stress_v1 import training_inputs
from scripts.run_joint_feature_selection_bocpd_v1 import bootstrap
from scripts.run_lsml_gate_locator_research_v1 import dump,score_locator,scalar_metrics
from scripts.audit_joint_selection_results import metrics as independent_metrics
from scripts.run_digitfree_broad50_v1 import sha
from spectral_utils.joint_sparse_membership import alias_coordinates
from spectral_utils.joint_structured_stress import structured_augmentation
from spectral_utils.lsml_gate_locator_research import _orient
from spectral_utils.digitfree_broad50 import ANCHOR
PREV=ROOT/'results/joint_structured_stress_v1';OUT=ROOT/'results/joint_staged_membership_v1'


def main():
    OUT.mkdir(exist_ok=True)
    sources=[Path(__file__),PREV/'SCORES.npz',PREV/'AUDIT.json',
        *[PREV/f'{k}_fold{f}.json' for k in ('base_replay','near_copies') for f in range(5)]]
    dump(OUT/'NEAR_STAGE_MANIFEST.json',dict(scope='post-evaluation fixed-score diagnostic',hashes={str(p):sha(p) for p in sources}))
    data,base,uids=inputs();scores={};weights={}
    with np.load(PREV/'SCORES.npz') as z:
        scores['base_refined']=z['base__refined'];scores['near_refined']=z['near_copies__joint']
        np.testing.assert_array_equal(z['gate'],data['gate'])
    for kind in ('base','near'):
        x=base if kind=='base' else structured_augmentation(base,data['offsets'],uids,'near_copies')
        scores[kind+'_full']=np.full(len(base),np.nan)
        for outer in range(5):
            prefix='base_replay' if kind=='base' else 'near_copies'
            m=json.loads((PREV/f'{prefix}_fold{outer}.json').read_text())['model'];assert m['valid']
            rows,args=training_inputs(data,x,outer);z=alias_coordinates(x,m['aliases'])
            raw=np.asarray(m['membership']['unoriented_weights'])
            w,orient=_orient(np.column_stack((z[rows],x[rows,ANCHOR])),np.r_[raw,0.],z.shape[1]);w=w[:-1]
            scores[kind+'_full'][~rows]=z[~rows]@w
            weights[f'{kind}_{outer}']=dict(weights=w,orientation=orient,source_model=str(PREV/f'{prefix}_fold{outer}.json'))
    with np.load(ROOT/'results/joint_sparse_membership_v1/SCORES.npz') as z:
        np.testing.assert_allclose(scores['base_full'],z['base__sparse'],atol=1e-11,rtol=1e-11)
    methods={k:score_locator(s,data['gate'],data) for k,s in scores.items()}
    for k,s in scores.items():
        pb,auc,n=independent_metrics(s,data['gate'],data)
        np.testing.assert_allclose([pb,auc],[methods[k]['pb'],methods[k]['within']],atol=1e-12,rtol=0);assert n==6030
    pairs=[('near_full','base_full'),('near_refined','base_refined'),
        ('base_refined','base_full'),('near_refined','near_full')]
    intervals=bootstrap(data,methods,pairs)
    result=dict(status='COMPLETE',scope='post-evaluation fixed-model stage diagnostic; no new candidate selected',
        metrics={k:scalar_metrics(v) for k,v in methods.items()},primary_contrasts=intervals,
        weights=weights,answers=len(uids),native_answers=13769,
        independent_point_metric_audit='PASS',base_full_historical_replay='PASS')
    dump(OUT/'NEAR_STAGE_DIAGNOSTIC.json',result)
    np.savez_compressed(OUT/'NEAR_STAGE_SCORES.npz',**scores,gate=data['gate'])
    print(json.dumps({k:dict(pb=v['pb'],within=v['within']) for k,v in methods.items()},indent=2),flush=True)


if __name__=='__main__':main()
