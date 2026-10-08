"""Unlabelled fusion conditional on position in the COMPLETE answer.

Step spans are a readout contract only. They never reset the position clock.
The Gaussian-factor and exact Gaussian/Bernoulli RBM arms constrain the 16 x P
loading map, not the number of hidden classes. IU uses the maintained estimator
on centered, standardized regional covariances; it has no loading-rank claim.
"""
import hashlib
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from . import two_axis_factor_fusion as factor
from .upcr import upcr_fit_covariance
from .laplacian_upcr import IU_FIT_DEFAULTS

BINS = 16
RIDGE = 1e-4
ANCHOR = 1  # Frozen varentropy15 risk convention, never inferred from labels.
METHODS = ('factor_stationary', 'factor_position_mean', 'factor_rank1', 'factor_rank2', 'factor_rank2_shuffled',
           'iu_stationary', 'iu_position_mean', 'iu_position', 'iu_position_shuffled',
           'rbm_stationary', 'rbm_rank1', 'rbm_rank2', 'rbm_rank2_shuffled')
CONTROLS = ('equal', 'position_early', 'position_late')
LABELS = {
    'top10': 'Original answer-local RBM12', 'equal': 'Equal feature weights',
    'position_early': 'Position only: early', 'position_late': 'Position only: late',
    'factor_stationary': 'Gaussian factor: fixed weights',
    'factor_position_mean': 'Gaussian factor: fixed weights, position mean',
    'factor_rank1': 'Gaussian factor: answer position, rank 1',
    'factor_rank2': 'Gaussian factor: answer position, rank 2',
    'factor_rank2_shuffled': 'Gaussian factor: shuffled positions, rank 2',
    'iu_stationary': 'IU-PCR: fixed weights', 'iu_position': 'IU-PCR: answer position',
    'iu_position_mean': 'IU-PCR: fixed weights, position mean',
    'iu_position_shuffled': 'IU-PCR: shuffled positions',
    'rbm_stationary': 'Gaussian RBM: fixed weights',
    'rbm_rank1': 'Gaussian RBM: answer position, rank 1',
    'rbm_rank2': 'Gaussian RBM: answer position, rank 2',
    'rbm_rank2_shuffled': 'Gaussian RBM: shuffled positions, rank 2',
}


def position_overlap(length, uid='', shuffled=False):
    """Assign token intervals to whole-answer regions, including short answers.

The shuffled arm permutes position assignments, NEVER the token feature rows.
Thus token/step/annotation identity and within-step content are preserved.
"""
    o = factor.overlap(length, BINS)
    if shuffled:
        seed = int.from_bytes(hashlib.sha256(('answer-position-v1:' + str(uid)).encode()).digest()[:8], 'little')
        o = o[np.random.default_rng(seed).permutation(length)]
    return o


def statistics(x, uid):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('invalid full-answer features')
    out = {}
    for name, shuffled in [('real', False), ('shuffle', True)]:
        o = position_overlap(len(x), uid, shuffled) * BINS / len(x)
        out[name] = np.einsum('tj,tk,tl->jkl', o, x, x, optimize=True)
        out[name + '_mean'] = o.T @ x
    return out


def centered_moments(second, mean):
    c = np.asarray(second) - np.einsum('ji,jk->jik', mean, mean)
    return .5 * (c + c.transpose(0, 2, 1))


def position_mean_control(arrays, mean, groups=None):
    """Fixed coefficients with EXACTLY the temporal arm's centering rule."""
    mu=np.asarray(mean)
    if groups is not None:
        alpha=groups/(groups+BINS)
        mu=alpha*mu+(1-alpha)*mu.mean(0)
    out={k:v.copy() for k,v in arrays.items()}
    out['intercept']=-np.sum(out['coefficients']*mu,axis=1)
    out['scoring_mean']=mu
    return out


def fit_iu(second, mean, *, stationary=False, groups=1):
    """Canonical two-component IU-PCR, with a declared regional adaptation.

Recenter/restandardize from TRAINING moments, then transform coefficients back
to the shared input coordinates. Position arms shrink moments toward pooled
moments by G/(G+16), where G counts source groups (an engineering rule).
"""
    second, mean = np.asarray(second), np.asarray(mean)
    pooled, middle = second.mean(0), mean.mean(0)
    alpha = 0. if stationary else groups / (groups + BINS)
    if stationary:
        seconds, means = pooled[None], middle[None]
    else:
        seconds = alpha * second + (1 - alpha) * pooled
        means = alpha * mean + (1 - alpha) * middle
    cov = centered_moments(seconds, means)
    coef, intercept, details = [], [], []
    for c, mu in zip(cov, means):
        scale = np.sqrt(np.maximum(np.diag(c), 0.))
        active = scale > 1e-10
        if active.sum() < 3:
            raise ValueError('IU needs three varying training views')
        corr = c[np.ix_(active, active)] / np.outer(scale[active], scale[active])
        f = upcr_fit_covariance(corr, **IU_FIT_DEFAULTS)
        if f.abstained or f.used_simple_average:
            raise ValueError('IU abstention or fallback')
        w = np.zeros(len(mu)); w[active] = f.w / scale[active]
        anchor_cov = float(w @ c[:, ANCHOR])
        sign = -1. if anchor_cov < 0 else 1.
        w *= sign
        coef.append(w); intercept.append(-mu @ w)
        details.append(dict(active_columns=np.flatnonzero(active).tolist(),
                            rho=f.rho_hat.tolist(), g2=f.g2_hat,
                            projection_residual=f.proj_residual, orientation=sign,
                            anchor_covariance=anchor_cov, condition=float(np.linalg.cond(corr)),
                            components=f.n_components_used))
    a = dict(coefficients=np.asarray(coef), intercept=np.asarray(intercept))
    if stationary:
        a = {k:np.repeat(v, BINS, axis=0) for k,v in a.items()}
    return a, dict(algorithm='canonical IU-PCR covariance API', alpha=alpha,
                   source_groups=groups, regions=details, converged=None,
                   preprocessing='train-region mean and scale; pooled moments shrinkage')


def fit_factor(second, mean, rank, stationary=False):
    """Centered factor likelihood; scoring includes the fitted regional means.

Using means explicitly repairs the previous zero-regional-mean restriction.
All time-aware arms of this family use the same training means.
"""
    if stationary:
        mu = np.repeat(mean.mean(0)[None], BINS, axis=0)
        c = np.repeat(centered_moments(second.mean(0)[None], mu[:1]), BINS, axis=0)
    else:
        mu = mean
        c = centered_moments(second, mean)
    a, d = factor.fit(c, rank, stationary=stationary)
    a['intercept'] = -np.sum(a['coefficients'] * mu, axis=1)
    a['mean'] = mu
    d['preprocessing'] = 'training mean; frozen answer-standardized coordinates'
    d['mean_energy_fraction'] = float(np.sum(mu*mu) / max(np.trace(second, axis1=1, axis2=2).sum(), 1e-30))
    return a, d


class RBMObjective:
    """Exact Gaussian-visible / one Bernoulli-hidden-unit likelihood.

E_j(x,h) = ||x-a||^2/2 - h*(b + x.w_j), W=UV'. Shared visible bias a,
shared hidden bias b, identity conditional variance. This extends the existing
moment_rbm_fusion.rbm_objective by POSITION, not by another latent unit.
Blocks contain every training token, weighted by group/answer and overlap.
"""
    def __init__(self, blocks, rank, stationary=False, ridge=RIDGE):
        self.blocks = blocks
        self.p = blocks[0][0].shape[1]
        self.j = 1 if stationary else BINS
        self.rank, self.ridge = rank, ridge
        self.mass = np.array([weight.sum() for _, weight in blocks])
        if not np.isclose(self.mass.sum(), 1., atol=1e-12):
            raise ValueError('RBM observation weights must sum to one')
        self.mean = sum(weight @ x for x, weight in blocks)
        self.square = sum(float(weight @ np.sum(x*x, axis=1)) for x, weight in blocks)

    def unpack(self, theta):
        p,j,r = self.p,self.j,self.rank
        return theta[:p], theta[p:p+j*r].reshape(j,r), theta[p+j*r:-1].reshape(p,r), theta[-1]

    def __call__(self, theta):
        a,u,v,b = self.unpack(theta); w=u@v.T
        ell_prior = b + w@a + .5*np.sum(w*w, axis=1)
        prior = expit(ell_prior)
        mass = np.array([1.]) if self.j==1 else self.mass
        loss = .5*(self.square - 2*self.mean@a + a@a) + mass@np.logaddexp(0.,ell_prior)
        ga = a-self.mean + (mass*prior)@w
        gw = (mass*prior)[:,None]*(a+w)
        gb = float(mass@prior)
        for j,(x,weight) in enumerate(self.blocks):
            k = 0 if self.j==1 else j
            ell = b+x@w[k];post=expit(ell);weighted=weight*post
            loss -= weight@np.logaddexp(0.,ell)
            gw[k] -= weighted@x;gb -= weighted.sum()
        loss += .5*self.ridge*np.mean(w*w)
        gw += self.ridge*w/w.size
        grad = np.r_[ga, (gw@v).ravel(), (gw.T@u).ravel(), gb]
        return float(loss), grad


def fit_rbm(blocks, rank, stationary=False, heartbeat=None, maxiter=1000):
    obj=RBMObjective(blocks,rank,stationary)
    p,j=obj.p,obj.j
    # Same equal positive initialization as the historical H1; second start is
    # deterministic spectral initialization. Neither inspects any label.
    c=sum((x.T*weight)@x for x,weight in blocks)
    ev,vec=np.linalg.eigh(c-np.outer(obj.mean,obj.mean))
    spectral=vec[:,-1]*np.sqrt(max(ev[-1]-1.,.01))
    if spectral[ANCHOR]<0:spectral=-spectral
    fits=[];health=[]
    for start,initial in enumerate((np.full(p,2./p),spectral)):
        w=np.tile(initial,(j,1))
        if rank==2:
            w += .05*np.outer(np.linspace(-1,1,j),vec[:,-2])
        u,v=factor.factorize(w,rank)
        theta=np.r_[np.zeros(p),u.ravel(),v.ravel(),0.]
        initial_loss,_=obj(theta);initial_penalty=.5*RIDGE*np.mean(w*w)
        started=time.perf_counter();iterations=[0]
        def callback(_):
            iterations[0]+=1
            if heartbeat and iterations[0]%10==0:
                heartbeat(start=start,iteration=iterations[0])
        try:
            opt=minimize(obj,theta,jac=True,method='L-BFGS-B',callback=callback,
                         options=dict(maxiter=maxiter,ftol=1e-10,gtol=1e-6,maxls=40))
            loss,g=obj(opt.x)
            if not np.isfinite(loss) or not np.isfinite(opt.x).all() or loss>initial_loss+1e-8:
                raise FloatingPointError('invalid RBM optimum')
            a,u,v,b=obj.unpack(opt.x);w=u@v.T
            penalty=.5*RIDGE*np.mean(w*w);nll=loss-penalty
            sign=np.where(w[:,ANCHOR]<0,-1.,1.)
            arrays=dict(coefficients=w*sign[:,None],intercept=sign*b,visible_bias=a,
                        hidden_bias=np.asarray(b),loadings=w,orientation=sign)
            if stationary:
                for key in ('coefficients','intercept','loadings','orientation'):
                    arrays[key]=np.repeat(arrays[key],BINS,axis=0)
            info=dict(start=start,success=True,converged=bool(opt.success),message=str(opt.message),
                      iterations=int(opt.nit),nll=nll,nll_initial=initial_loss-initial_penalty,
                      regularized_objective=loss,penalty=penalty,gradient_max=float(abs(g).max()),
                      seconds=time.perf_counter()-started,anchor_ties=int(np.sum(abs(w[:,ANCHOR])<=1e-12)))
            health.append(info);fits.append((nll,arrays,info))
        except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
            health.append(dict(start=start,success=False,failure=type(exc).__name__+': '+str(exc)))
    if not fits:raise FloatingPointError('both exact RBM starts failed')
    _,arrays,info=min(fits,key=lambda z:z[0])
    return arrays,dict(**info,starts=health,algorithm='exact Gaussian/Bernoulli RBM H1',
                       rank=rank,stationary=stationary,selected_by='pure_gaussian_rbm_nll',
                       conditional_variance='identity',token_observations=sum(len(x) for x,_ in blocks))


def token_scores(x, arrays, uid, shuffled=False):
    x=np.asarray(x);coef=arrays['coefficients'];bias=arrays['intercept']
    if coef.shape!=(BINS,x.shape[1]) or bias.shape!=(BINS,):
        raise ValueError('invalid coefficient map')
    o=position_overlap(len(x),uid,shuffled)
    return np.sum(x*(o@coef),axis=1)+o@bias


def step_scores(x, spans, arrays, uid, shuffled=False):
    token=token_scores(x,arrays,uid,shuffled)
    if not np.isfinite(token).all():return np.full(len(spans),np.nan)
    out=[]
    for start,end in spans:
        if not 0<=start<end<=len(token):raise ValueError('invalid step span')
        s=token[int(start):int(end)];n=min(10,len(s))
        out.append(np.partition(s,len(s)-n)[-n:].mean())
    return np.asarray(out)
