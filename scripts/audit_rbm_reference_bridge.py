"""Small numerical audit; does not fit a benchmark or modify frozen experiments.

RBMpaper/forward.m uses sigmoid(data*vishid+hidbias) for one layer.
Compare that shared formula, then independently integrate OUR Gaussian model.
This is not execution of MATLAB or reproduction of RBMpaper's CD training.
"""
from pathlib import Path
import sys,json
import numpy as np
from scipy.integrate import quad
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spectral_utils.moment_rbm_fusion import rbm_objective


def quadrature_objective(theta,X):
    p=X.shape[1];a,w,b=theta[:p],theta[p:2*p],theta[-1]
    # Integrate exp(-E) relative to the common (2*pi)^(P/2) constant.
    integrals=[quad(lambda x:np.exp(-.5*(x-aj)**2+wj*x)/np.sqrt(2*np.pi),
                    -np.inf,np.inf,epsabs=1e-11,epsrel=1e-11)[0] for aj,wj in zip(a,w)]
    logz=np.log1p(np.exp(b)*np.prod(integrals))
    return .5*np.mean(np.sum((X-a)**2,axis=1))-np.logaddexp(0,b+X@w).mean()+logz


def main():
    rng=np.random.default_rng(20260911)
    objective_error=0.;gradient_error=0.;posterior_error=0.;cases=0
    for p in (2,6,12):
        for _ in range(4):
            X=rng.normal(size=(31,p));theta=rng.normal(scale=.4,size=2*p+1)
            a,w,b=theta[:p],theta[p:2*p],theta[-1]
            # Direct transcription of ONE-LAYER forward.m probability formula.
            reference_forward=1./(1.+np.exp(-X@w-b))
            ours=expit(b+X@w)
            e0=.5*np.sum((X-a)**2,axis=1);e1=e0-b-X@w
            independent=np.exp(-e1-np.logaddexp(-e0,-e1))
            np.testing.assert_allclose(ours,reference_forward,atol=1e-14,rtol=0)
            np.testing.assert_allclose(ours,independent,atol=1e-14,rtol=0)
            posterior_error=max(posterior_error,float(np.max(np.abs(ours-independent))))
            loss,gradient=rbm_objective(theta,X);q=quadrature_objective(theta,X)
            objective_error=max(objective_error,abs(loss-q))
            np.testing.assert_allclose(loss,q,atol=1e-10,rtol=0)
            numeric=[]
            for j in range(len(theta)):
                delta=np.zeros_like(theta);delta[j]=1e-5
                numeric.append((quadrature_objective(theta+delta,X)-quadrature_objective(theta-delta,X))/2e-5)
            numeric=np.asarray(numeric)
            np.testing.assert_allclose(gradient,numeric,atol=1e-8,rtol=1e-6)
            gradient_error=max(gradient_error,float(np.max(np.abs(gradient-numeric))));cases+=1
    out=dict(status='PASS',parameter_cases=cases,dimensions=[2,6,12],
        max_objective_quadrature_error=objective_error,max_gradient_quadrature_error=gradient_error,
        max_posterior_energy_error=posterior_error,
        source='https://github.com/ushaham/RBMpaper/blob/master/forward.m',
        scope='One-layer forward formula transcribed, not MATLAB execution. Gaussian partition integrated numerically, gradient checked by finite differences. Does not validate CD training, Gaussian modeling assumptions, semantic class alignment or task performance.')
    directory=ROOT/'results/rbm_reference_bridge_v1';directory.mkdir(parents=True,exist_ok=True)
    (directory/'REVIEW.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print(json.dumps(out))


if __name__=='__main__':main()
