"""Verify accelerated historical grouping against all saved historical outputs."""
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
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


base=module('historical_base_driver','scripts/run_historical_fusion_refit_v3.py')
joint=module('historical_joint_core','spectral_utils/historical_joint_refit.py')
fast=module('historical_joint_acceleration','spectral_utils/historical_joint_acceleration.py')
OUT=ROOT/'results/historical_joint_refit_v3'


def main():
    started=time.time();base.save(OUT/'PREFLIGHT_FAST_STATE.json',dict(phase='RUNNING',pid=os.getpid(),started_unix=started))
    arithmetic=fast.preflight();reference,_=base.core.reference_modules(base.SOURCE)
    cell=base.cell_data('pb_gsm8k_q8');old=base.SOURCE/'results/joint_lsml_optimization_v2'
    groups=base.load(old/'folds/folds.json')['processbench']['outer']
    train=np.array([groups[str(g)]!=0 for g in cell['group_ids']]);prep=base.core.prepare(cell,train,reference)
    with fast.compatible_ari(reference) as acceleration:
        weights,metadata,failures,audit=joint.fit(prep,reference,cell='pb_gsm8k_q8',outer=0)
    arrays=base.core.score(prep,cell,weights,np.arange(len(cell['row_ids'])))
    checks=[]
    for filename,roster in [('scores_outer.npz',tuple(a for a in joint.ARMS if a!='internal_joint_modelinv_lam0')),
                           ('scores_amend_r3.npz',('internal_joint_modelinv_lam0',))]:
        with np.load(old/'structure/pb_gsm8k_q8/outer0'/filename,allow_pickle=False) as previous:
            for arm in roster:
                assert arm not in failures,failures
                for suffix in ('w','top10','spanmax','detector'):
                    key=arm+'__'+suffix;np.testing.assert_array_equal(arrays[key],previous[key])
                    checks.append(dict(key=key,max_abs=0.0))
    original=base.load(OUT/'PREFLIGHT.json')
    assert audit==original['diagnostics']
    sources=dict(original['source_hashes'])
    for path in (Path(__file__),ROOT/'spectral_utils/historical_joint_acceleration.py',OUT/'PREFLIGHT.json'):
        sources[str(path)]=base.sha(path)
    result=dict(status='PASS',purpose='runtime optimization fidelity; no performance selection',seconds=time.time()-started,
        arithmetic=arithmetic,acceleration=acceleration,checks=checks,diagnostics=audit,source_hashes=sources,
        original_seconds=original['seconds'],labels_used=False)
    base.save(OUT/'PREFLIGHT_FAST.json',result)
    base.save(OUT/'PREFLIGHT_FAST_STATE.json',dict(phase='PASS',pid=os.getpid(),seconds=result['seconds'],ari_calls=acceleration['calls']))
    print('Fast historical Joint replay PASS',len(checks),'arrays;',round(result['seconds'],2),'seconds',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        base.save(OUT/'PREFLIGHT_FAST_STATE.json',dict(phase='FAILED',pid=os.getpid(),error=type(error).__name__+': '+str(error)));raise
