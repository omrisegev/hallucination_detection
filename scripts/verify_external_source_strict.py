"""Verify every source decision under observable numerical failures before freeze."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','LOKY_MAX_CPU_COUNT'):os.environ[k]='1'
import json,sys,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.external_generalization.fusion import local_scores,fit_weights,standardize
from spectral_utils.external_generalization.artifacts import atomic_json,file_hash


def one(args):
    i,t,s,old=args;new,d=local_scores(t,s)
    if d['native']!=old['diagnostics']['native']:raise AssertionError(('coverage changed',i,d))
    err=max(float(np.max(np.abs(v-np.array(old['scores'][k])))) for k,v in new.items())
    if err>1e-10:raise AssertionError(('strict replay changed scores',i,err))
    return i,err


def main():
    started=time.perf_counter();out=ROOT/'results/lsml_external_generalization_v1/evaluation/source'
    z=np.load(ROOT/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/TOKEN_MATRICES.npz')
    tokens=z['tokens'];toff=z['token_offsets'];sp=z['step_spans']
    off=np.load(ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.npz')['offsets']
    n=len(off)-1;maxerr=0
    def jobs():
        for i in range(n):
            a,b=off[i:i+2];ta,tb=toff[i:i+2]
            old=json.loads((out/'local_records'/(str(i)+'.json')).read_text())
            yield i,tokens[ta:tb],sp[a:b],old
    with ProcessPoolExecutor(max_workers=8) as pool:
        for j,(i,e) in enumerate(pool.map(one,jobs(),chunksize=8)):
            maxerr=max(maxerr,e)
            if j%1000==0:print('strict replay',j+1,'/',n,flush=True)
    records=json.loads((ROOT/'results/localization_full_benchmark_v3/evaluation/JOINED.json').read_text())['records']
    fm=json.loads((ROOT/'results/localization_source_group_audit_v1/FOLDS_V2.json').read_text())['outer']
    sf=np.repeat([fm[r['group_id']] for r in records],np.diff(off))
    level=np.load(ROOT/'.worktrees/token-probability-fusion-v1/results/token_probability_fusion_v1/DERIVATIVE_CHANNELS.npz')['level']
    for a,b in zip(off[:-1],off[1:]):level[a:b]=standardize(level[a:b])
    expected=json.loads((out/'VALIDATION_FITS.json').read_text())
    for fold in expected:
        fit=fit_weights(level[(sf!=fold['test'])&(sf!=fold['calibration'])])
        np.testing.assert_allclose(fit['weights'],fold['fit']['weights'],atol=1e-12,rtol=0)
    fit=fit_weights(level[sf<4]);bundle=json.loads((out/'BUNDLE.json').read_text())
    np.testing.assert_allclose(fit['weights'],bundle['fit']['weights'],atol=1e-12,rtol=0)
    historical=json.loads((ROOT/'.worktrees/depth-feature-fusion-v1/results/step_level_bank_baseline_v1/RESULTS.json').read_text())
    old=next(f for f in historical['fits']['continuous'] if f['fold']==4)
    historical_error=float(np.max(np.abs(np.array(old['weights'])-np.array(fit['weights']))))
    if historical_error>1e-10:raise AssertionError(('historical deployment weight replay',historical_error))
    atomic_json(out/'STRICT_REPLAY.json',{'answers':n,'max_score_error':maxerr,'frozen_fits':6,
       'historical_fold4_weight_error':historical_error,'elapsed':time.perf_counter()-started,
       'code':{str(p.relative_to(ROOT)):file_hash(p) for p in (ROOT/'spectral_utils/external_generalization').rglob('*.py')}})
    print('STRICT REPLAY PASS',flush=True)

if __name__=='__main__':main()
