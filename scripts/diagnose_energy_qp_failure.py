"""Reproduce the numerical stop without changing the frozen fit or its solver."""
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import run_energy_context_stability as run
from spectral_utils import energy_context_stability as core
from spectral_utils.cca_iu_isolation import simplex_qp as original

run.OUT=ROOT/'results/energy_context_stability_v1'
def capture(Q,r):
    try:return original(Q,r)
    except ValueError as e:
        np.savez_compressed(run.OUT/'NUMERICAL_FAILURE.npz',Q=Q,r=r)
        run.save(run.OUT/'NUMERICAL_FAILURE.json',dict(error=str(e),cell='pb_omnimath_q4',fold=3,
            completed=33,action='Stopped; original outputs retained; no quality labels opened'))
        run.save(run.OUT/'RUN_STATE.json',dict(state='STOPPED_NUMERICAL_CHECK',completed=33,total=45,error=str(e)))
        raise

if __name__=='__main__':
    core.simplex_qp=capture
    with threadpool_limits(limits=1):
        metadata,data=run.prepare()
        run.fit_case('pb_omnimath_q4',3,metadata,data)
