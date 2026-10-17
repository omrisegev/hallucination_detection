"""Compare optimized complete-pair IU against the frozen v2 source file."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.upcr import upcr_fit_covariance
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS


def main():
    p=argparse.ArgumentParser();p.add_argument('--v2-root',type=Path,required=True);args=p.parse_args()
    spec=importlib.util.spec_from_file_location('spectral_utils._frozen_v2_upcr',args.v2_root/'spectral_utils/upcr.py')
    ref=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ref;spec.loader.exec_module(ref)
    rng=np.random.default_rng(927);worst=0.;told=tnew=0.;n=0
    with threadpool_limits(limits=1):
        for m in (3,17,34,64,136):
            for kind in ('full','short','correlated','probabilities'):
                for _ in range(3):
                    X=rng.normal(size=(180 if kind!='short' else 20,m))
                    if kind=='correlated':X=X*.03+rng.normal(size=(len(X),1))
                    if kind=='probabilities':
                        X=np.sort(np.exp(X),axis=1)[:,::-1];X/=X.sum(axis=1,keepdims=True)
                    X=(X-X.mean(axis=0))/X.std(axis=0);C=X.T@X/len(X)
                    t=time.perf_counter();a=ref.upcr_fit_covariance(C,**dict(IU_FIT_DEFAULTS));told+=time.perf_counter()-t
                    t=time.perf_counter();b=upcr_fit_covariance(C,**dict(IU_FIT_DEFAULTS));tnew+=time.perf_counter()-t
                    np.testing.assert_allclose(a.w,b.w,atol=1e-10,rtol=1e-9)
                    np.testing.assert_allclose(a.rho_hat,b.rho_hat,atol=1e-10,rtol=1e-9)
                    assert a.g2_hat==b.g2_hat
                    worst=max(worst,float(np.max(np.abs(a.w-b.w))));n+=1
        # Restricted pair systems must retain the original least-squares path.
        pairs=[(i,j) for i in range(64) for j in range(i+1,64) if (i+j)%3]
        X=rng.normal(size=(100,64));C=X.T@X/len(X)
        a=ref.upcr_fit_covariance(C,pairs=pairs,**dict(IU_FIT_DEFAULTS))
        b=upcr_fit_covariance(C,pairs=pairs,**dict(IU_FIT_DEFAULTS))
        np.testing.assert_array_equal(a.w,b.w)
    print(dict(covariance_cases=n,max_weight_error=worst,original_seconds=told,optimized_seconds=tnew,
        speedup_across_mixed_sizes=told/tnew,restricted_pairs_exact=True))


if __name__=='__main__':main()
