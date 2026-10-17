"""Post-evaluation information diagnostic; no fits or held-label selection."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/joint_feasible_membership_v1'

def information(c,a):
    regular=(1-1e-4)*c+1e-4*np.diag(np.maximum(np.diag(c),1e-12))
    solved=np.linalg.solve(regular,a)
    return np.sum(a*solved,axis=0),solved,regular

def main():
    rows=[]
    for kind in ('base','near_copies'):
        for outer in range(5):
            m=json.loads((OUT/f'{kind}_fold{outer}.json').read_text())['model']
            selection=m['refinement']['selection'];state=selection['automatic']
            c=np.asarray(m['membership']['observed_covariance'])
            a=np.asarray(selection['initial_factor_matrix']);reference=np.asarray(selection['initial_factor_information'])
            assert c.shape==(len(a),len(a))
            ids=np.asarray(state['active']);labels=np.asarray(state['labels'])
            relevant=reference>1e-8
            current_info,solved,regular=information(c[np.ix_(ids,ids)],a[ids])
            inverse=np.linalg.solve(regular,np.eye(len(ids)))
            loss=solved**2/np.diag(inverse)[:,None]
            attempts=[]
            for i,feature in enumerate(ids):
                if np.sum(labels==labels[i])<=2:continue
                proposal=np.delete(ids,i)
                direct,_,_=information(c[np.ix_(proposal,proposal)],a[proposal])
                np.testing.assert_allclose(direct,current_info-loss[i],atol=1e-8,rtol=1e-8)
                retention=direct[relevant]/reference[relevant]
                attempts.append(dict(feature=int(feature),minimum_retention=float(retention.min()),feasible=bool(np.all(retention>=.95))))
            last=selection['deletion_audit'][-1]
            assert last['stopped']=='NO_FEASIBLE_VALID_DELETION_AMONG_THREE_BEST'
            rejected=last['rejected_candidates']
            assert len(rejected)==3 and all(t['reason']=='INFORMATION_BUDGET' for t in rejected)
            rows.append(dict(kind=kind,outer=outer,kept=len(ids),eligible=len(attempts),
                information_feasible=sum(t['feasible'] for t in attempts),
                current_top3=[t['feature'] for t in rejected],candidates=attempts))
    result=dict(status='PASS',scope='post-evaluation mechanism diagnostic; no regrouping, fitting or quality inference',
        evidence='saved training covariance and initial factor matrix; direct solves verified against deletion identity',
        states_with_feasible_alternatives=sum(r['information_feasible']>0 for r in rows),states=rows)
    (OUT/'STOP_FEASIBILITY_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='states'}))
    for r in rows:print({k:v for k,v in r.items() if k!='candidates'})

if __name__=='__main__':main()
