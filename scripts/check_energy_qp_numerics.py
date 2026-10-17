"""Independent numeric investigation of the saved stop; does not fit models."""
import inspect
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils import cca_iu_isolation as original
from scripts.run_energy_context_stability import save
OUT=ROOT/'results/energy_context_stability_v1'
source=inspect.getsource(original.simplex_qp)
scope=dict(original.__dict__)
exec(source.replace('best-1e-13','best').replace("if violation > 1e-7: raise ValueError(f'KKT residual {violation}')",''),scope)
corrected=scope['simplex_qp']
old_scope=dict(original.__dict__)
exec(source.replace("if violation > 1e-7: raise ValueError(f'KKT residual {violation}')",''),old_scope)
f=np.load(OUT/'NUMERICAL_FAILURE.npz');Q,r=f['Q'],f['r']
old=old_scope['simplex_qp'](Q,r);new=corrected(Q,r)
def kkt(w):
    g=np.einsum('nij,nj->ni',Q,w)-r;level=(g*w).sum(1)
    return np.max(np.where(w>1e-8,abs(g-level[:,None]),np.maximum(level[:,None]-g,0)),axis=1)
bad=int(kkt(old).argmax());q=Q[bad];rr=r[bad]
sol=minimize(lambda w:.5*w@q@w-w@rr,np.ones(5)/5,jac=lambda w:q@w-rr,
    constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(5))],bounds=[(0,1)]*5,
    method='SLSQP',options=dict(ftol=1e-14,maxiter=1000))
assert sol.success
record=dict(batch_size=len(Q),bad_row=bad,old_max_kkt=float(kkt(old).max()),new_max_kkt=float(kkt(new).max()),
    max_weight_delta=float(abs(new-old).max()),old_weight=old[bad].tolist(),new_weight=new[bad].tolist(),
    slsqp_weight=sol.x.tolist(),slsqp_max_weight_delta=float(abs(sol.x-new[bad]).max()),
    objective_gain=float((.5*old[bad]@q@old[bad]-old[bad]@rr)-(.5*new[bad]@q@new[bad]-new[bad]@rr)),
    eigenvalues=np.linalg.eigvalsh(q).tolist(),reason='Absolute 1e-13 objective improvement threshold retained a suboptimal boundary face.')
save(OUT/'NUMERICAL_ANALYSIS.json',record)
np.savez_compressed(OUT/'QP_REGRESSION.npz',Q=q,r=rr,expected=new[bad])
print(record)
