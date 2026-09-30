"""Fidelity and cost check for historical Joint refits; never a performance pilot."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import importlib.util
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    output=importlib.util.module_from_spec(spec);spec.loader.exec_module(output)
    return output


base=module('historical_base_runner','scripts/run_historical_fusion_refit_v3.py')
joint=module('historical_joint_core','spectral_utils/historical_joint_refit.py')
OUT=ROOT/'results/historical_joint_refit_v3'


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True)
    base.save(OUT/'PREFLIGHT_STATE.json',dict(phase='RUNNING',pid=os.getpid(),started_unix=started))
    reference,_=base.core.reference_modules(base.SOURCE)
    cell=base.cell_data('pb_gsm8k_q8');old=base.SOURCE/'results/joint_lsml_optimization_v2'
    groups=base.load(old/'folds/folds.json')['processbench']['outer']
    mask=np.array([groups[str(g)]!=0 for g in cell['group_ids']])
    prep=base.core.prepare(cell,mask,reference)
    weights,metadata,failures,audit=joint.fit(prep,reference,cell='pb_gsm8k_q8',outer=0)
    arrays=base.core.score(prep,cell,weights,np.arange(len(cell['row_ids'])))
    checks=[];sources={}
    for filename,roster in [('scores_outer.npz',tuple(a for a in joint.ARMS if a!='internal_joint_modelinv_lam0')),
                           ('scores_amend_r3.npz',('internal_joint_modelinv_lam0',))]:
        path=old/'structure/pb_gsm8k_q8/outer0'/filename;sources[str(path)]=base.sha(path)
        with np.load(path,allow_pickle=False) as previous:
            for arm in roster:
                if arm in failures:
                    assert arm+'__w' not in previous,(arm,failures[arm]);continue
                for suffix in ('w','top10','spanmax','detector'):
                    key=arm+'__'+suffix
                    np.testing.assert_allclose(arrays[key],previous[key],atol=2e-7,rtol=2e-7)
                    checks.append(dict(key=key,max_abs=float(np.max(np.abs(arrays[key]-previous[key])))))
    _,files=base.core.reference_modules(base.SOURCE)
    sources.update({path:base.sha(path) for path in files})
    sources[str(ROOT/'spectral_utils/historical_joint_refit.py')]=base.sha(ROOT/'spectral_utils/historical_joint_refit.py')
    sources[str(Path(__file__))]=base.sha(Path(__file__))
    result=dict(status='PASS',purpose='implementation_fidelity_and_runtime_only',cell='pb_gsm8k_q8',outer=0,
        seconds=time.time()-started,arms=list(joint.ARMS),checks=checks,failures=failures,
        diagnostics=audit,source_hashes=sources,labels_used=False)
    base.save(OUT/'PREFLIGHT.json',result)
    base.save(OUT/'PREFLIGHT_STATE.json',dict(phase='PASS',pid=os.getpid(),seconds=result['seconds'],
        replayed_arrays=len(checks),fitting_failures=failures))
    print('Joint historical fidelity PASS:',len(checks),'arrays;',round(result['seconds'],2),'seconds',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        base.save(OUT/'PREFLIGHT_STATE.json',dict(phase='FAILED',pid=os.getpid(),error=type(error).__name__+': '+str(error)))
        raise
