"""Capture numerical failure during an exact replay; no repaired scoring arm."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[k]='1'
import json,sys,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_joint_mass_membership_v1 import OUT,inputs,bank,training_inputs
from spectral_utils.joint_mass_membership import fit_mass_joint
from spectral_utils.digitfree_broad50 import ANCHOR
import spectral_utils.joint_mass_groups as group_module

def main():
    data,base,uids=inputs();x=bank(data,base,uids,'near_copies');records=[];matrices={}
    original_optimizer=group_module.minimize;original_mass=group_module.feature_mass
    for outer in range(5):
        path=OUT/f'near_copies_fold{outer}.json'
        if not path.exists():continue
        model=json.loads(path.read_text())['model']
        if model.get('failure')!='ValueError: FEATURE_MASS_KKT_FAILURE':continue
        _,args=training_inputs(data,x,outer);last={};failure={}
        def optimizer(*a,**kw):
            fit=original_optimizer(*a,**kw);last['fit']=fit;return fit
        def mass(cov):
            try:return original_mass(cov)
            except ValueError as exc:
                if str(exc)=='FEATURE_MASS_KKT_FAILURE':
                    c=np.asarray(cov);fit=last['fit'];raw=np.maximum(fit.x,0)
                    scale=np.sqrt(np.maximum(np.diag(c),1e-12));kernel=np.clip(c/scale[:,None]/scale[None,:],-1,1)**2
                    gradient=kernel@raw-1;projected=np.where(raw>0,gradient,np.minimum(gradient,0))
                    matrices[f'outer{outer}_covariance']=c;matrices[f'outer{outer}_raw_mass']=raw
                    failure.update(outer=outer,features=len(c),optimizer_success=bool(fit.success),
                        optimizer_message=str(fit.message),iterations=int(fit.nit),
                        projected_kkt=float(np.max(np.abs(projected))),required_kkt=1e-6,
                        objective=float(.5*raw@kernel@raw-raw.sum()))
                raise
        with patch.object(group_module,'minimize',side_effect=optimizer),patch.object(group_module,'feature_mass',side_effect=mass):
            replay=fit_mass_joint(*args,anchor_index=ANCHOR,seed=model['seed'])
        assert not replay['valid'] and replay['failure']==model['failure'] and failure
        failure['frozen_model_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();records.append(failure)
        print('NUMERICAL_FAILURE_REPLAY',failure,flush=True)
    np.savez_compressed(OUT/'MASS_SOLVER_FAILURE_INPUTS.npz',**matrices)
    result=dict(status='PASS',scope='exact failed-fit replay; no criterion relaxation, repaired model, held-score or quality comparison',failures=records)
    (OUT/'MASS_SOLVER_FAILURE_DIAGNOSTIC.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())

if __name__=='__main__':main()
