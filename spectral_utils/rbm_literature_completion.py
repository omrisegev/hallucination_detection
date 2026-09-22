"""Continuous, answer-local extensions of the moment RBM.

These are Gaussian adaptations, not a reproduction of the Bernoulli RBMpaper.
No function accepts correctness labels. The two mixture states are unnamed.
"""
from functools import lru_cache
import hashlib
import math
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def seed_for(uid, purpose):
    return int.from_bytes(hashlib.sha256((uid + ':' + purpose).encode()).digest()[:8], 'little')


@lru_cache(None)
def hidden_states(h):
    if not 1 <= h <= 12:
        raise ValueError('exact enumeration supports 1..12 binary units')
    return ((np.arange(2 ** h)[:, None] >> np.arange(h)) & 1).astype(float)


def unpack(theta, p, h):
    return theta[:p], theta[p:p+p*h].reshape(p, h), theta[p+p*h:]


def pack(a, w, b):
    return np.r_[a, np.asarray(w).ravel(), np.atleast_1d(b)]


def initial(p, h, seed=0):
    rng = np.random.default_rng(seed)
    w = np.full((p, h), 2. / (p * np.sqrt(h)))
    if h > 1 or seed:
        w += rng.normal(0, .02, w.shape)
    return pack(np.zeros(p), w, np.zeros(h))


class ExactRBM:
    """E(x,h)=||x-a||^2/2-b.h-x.W.h, unit conditional variance."""
    def __init__(self, x, h):
        self.x = np.asarray(x, float)
        self.n, self.p = self.x.shape
        self.h = h
        self.states = hidden_states(h)
        self.mean = self.x.mean(axis=0)
        self.square_mean = np.square(self.x).sum(axis=1).mean()

    def __call__(self, theta):
        if self.h == 1:
            # Preserve the exact historical floating-point evaluation as well
            # as its mathematical objective in the optimizer-only comparison.
            from .moment_rbm_fusion import rbm_objective
            return rbm_objective(theta, self.x)
        a, w, b = unpack(theta, self.p, self.h)
        wh = self.states @ w.T
        lp = self.states @ b + wh @ a + .5 * np.square(wh).sum(axis=1)
        logz = logsumexp(lp)
        pi = np.exp(lp - logz)
        ell = self.x @ w + b
        post = expit(ell)
        loss = (.5*(self.square_mean-2*self.mean@a+a@a)
                - np.logaddexp(0., ell).sum(axis=1).mean() + logz)
        ga = a - self.mean + pi @ wh
        gw = (a + wh).T @ (pi[:, None] * self.states) - self.x.T @ post / self.n
        gb = pi @ self.states - post.mean(axis=0)
        return float(loss), pack(ga, gw, gb)


def exact_fit(x, h, *, theta=None, seed=0, maxiter=100):
    obj = ExactRBM(x, h)
    t0 = initial(x.shape[1], h, seed) if theta is None else np.asarray(theta).copy()
    start = time.perf_counter()
    loss0, _ = obj(t0)
    opt = minimize(obj, t0, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6, maxls=40))
    loss, g = obj(opt.x)
    if not np.isfinite(opt.x).all() or not np.isfinite(loss) or loss > loss0 + 1e-8:
        raise ValueError('invalid/nondecreasing exact fit')
    return opt.x, dict(nll_initial=loss0, nll_final=loss, iterations=int(opt.nit),
                      converged=bool(opt.success), message=str(opt.message),
                      gradient_max=float(np.abs(g).max()), seconds=time.perf_counter()-start)


def cd_fit(x, h, *, theta=None, seed=0, epochs=100, batch_size=128, cd_steps=10):
    """Actual alternating Bernoulli/Gaussian Gibbs CD, not mean-field CD.

    Rao-Blackwellize positive/negative sufficient statistics; sample every Gibbs
    hidden state and visible reconstruction. Exact NLL is diagnostic only and
    never selects an epoch, learning rate, or model checkpoint.
    """
    rng = np.random.default_rng(seed)
    t0 = initial(x.shape[1], h, seed=0) if theta is None else np.asarray(theta).copy()
    a, w, b = (v.copy() for v in unpack(t0, x.shape[1], h))
    obj = ExactRBM(x, h)
    loss0, _ = obj(t0)
    start = time.perf_counter()
    clips = 0
    for epoch in range(epochs):
        rate = .005 / math.sqrt(1 + epoch / 20)
        order = rng.permutation(len(x))
        for begin in range(0, len(x), batch_size):
            pos = x[order[begin:begin+batch_size]]
            hp = expit(pos @ w + b)
            prob = hp
            for _ in range(cd_steps):
                hidden = rng.random(prob.shape) < prob
                neg = a + hidden @ w.T + rng.normal(size=pos.shape)
                prob = expit(neg @ w + b)
            ga = (pos-neg).mean(axis=0)
            gw = (pos.T@hp-neg.T@prob)/len(pos)
            gb = (hp-prob).mean(axis=0)
            norm = np.linalg.norm(pack(ga, gw, gb))
            factor = min(1., 5./max(norm, 1e-12))
            clips += int(factor < 1)
            a += rate*factor*ga
            w += rate*factor*gw
            b += rate*factor*gb
            if not all(np.isfinite(t).all() for t in (a, w, b)):
                raise ValueError('nonfinite CD parameters')
    theta = pack(a, w, b)
    loss, g = obj(theta)
    return theta, dict(epochs=epochs, cd_steps=cd_steps, batch_size=batch_size,
                      initial_rate=.005, gradient_clips=clips, nll_initial=loss0,
                      nll_final=loss, gradient_max=float(np.abs(g).max()),
                      seconds=time.perf_counter()-start, seed=int(seed))


def oriented_units(x, theta, h, anchor):
    _, w, b = unpack(theta, x.shape[1], h)
    ell = x@w+b
    post = expit(ell)
    signs = np.ones(h)
    for j in range(h):
        if np.std(post[:, j]) > 1e-12 and np.std(anchor) > 1e-12:
            signs[j] = -1 if np.corrcoef(post[:, j], anchor)[0, 1] < 0 else 1
    return ell*signs, signs


def mean_unit_scores(ell):
    # logit(mean sigmoid(ell)) without inverting rounded probabilities.
    log_positive = logsumexp(-np.logaddexp(0., -ell), axis=1)
    log_negative = logsumexp(-np.logaddexp(0., ell), axis=1)
    return log_positive-log_negative, expit(ell).mean(axis=1)


def shared_to_mixture(a, w, b):
    """Exact mapping from unit-variance, one-hidden-unit Gaussian RBM."""
    return np.stack((a, a+w)), float(b+a@w+.5*w@w)


class TwoStateVariance:
    """Two Gaussian components: shared or state-specific diagonal variance.

    Shared penalty is .1*sum(log D**2). Separate penalty has the SAME shared
    term on the mean log variance, plus .1*sum(log D1-log D0)**2. At equality
    the parameterization, likelihood and penalty agree exactly.
    """
    def __init__(self, x, separate):
        self.x = np.asarray(x, float)
        self.p = x.shape[1]
        self.separate = bool(separate)

    def decode(self, theta):
        mu = theta[:2*self.p].reshape(2, self.p)
        v = theta[2*self.p:-1].reshape(2 if self.separate else 1, self.p)
        return mu, np.repeat(v, 2, axis=0) if not self.separate else v, theta[-1]

    def component_logp(self, theta):
        mu, lv, eta = self.decode(theta)
        diff = self.x[:, None, :]-mu
        parts = -.5*np.sum(lv + diff*diff*np.exp(-lv), axis=2)
        parts[:, 0] -= np.logaddexp(0., eta)
        parts[:, 1] -= np.logaddexp(0., -eta)
        return parts, diff

    def __call__(self, theta):
        mu, lv, eta = self.decode(theta)
        parts, diff = self.component_logp(theta)
        den = logsumexp(parts, axis=1)
        resp = np.exp(parts-den[:, None])
        loss = -den.mean()
        gmu = -np.mean(resp[:, :, None]*diff*np.exp(-lv), axis=0)
        gv = .5*np.mean(resp[:, :, None]*(1-diff*diff*np.exp(-lv)), axis=0)
        middle = lv.mean(axis=0)
        if self.separate:
            delta = lv[1]-lv[0]
            loss += .1*(middle@middle+delta@delta)
            gv += .1*middle + np.stack((-.2*delta, .2*delta))
        else:
            loss += .1*(middle@middle)
            gv = gv.sum(axis=0, keepdims=True)+.2*middle
        geta = expit(eta)-resp[:, 1].mean()
        return float(loss), np.r_[gmu.ravel(), gv.ravel(), geta]


def variance_fit(x, a, w, b, separate, *, maxiter=100):
    obj = TwoStateVariance(x, separate)
    mu, eta = shared_to_mixture(a, w, b)
    t0 = np.r_[mu.ravel(), np.zeros(x.shape[1]*(2 if separate else 1)), eta]
    p = x.shape[1]
    bounds = [(None, None)]*(2*p)+[(math.log(.05), None)]*(len(t0)-2*p-1)+[(None, None)]
    start = time.perf_counter()
    loss0, _ = obj(t0)
    opt = minimize(obj, t0, jac=True, method='L-BFGS-B', bounds=bounds,
                   options=dict(maxiter=maxiter, ftol=1e-10, gtol=1e-6, maxls=40))
    loss, grad = obj(opt.x)
    if not np.isfinite(opt.x).all() or loss > loss0+1e-8:
        raise ValueError('invalid variance fit')
    parts, _ = obj.component_logp(opt.x)
    _, lv, _ = obj.decode(opt.x)
    return parts[:, 1]-parts[:, 0], opt.x, dict(nll_penalty_initial=loss0,
        nll_penalty_final=loss, converged=bool(opt.success), iterations=int(opt.nit),
        message=str(opt.message), gradient_max=float(np.max(np.abs(grad))),
        min_variance=float(np.exp(lv).min()), floor_hits=int(np.sum(np.exp(lv)<=.05+1e-8)),
        log_variance_state_distance=float(np.linalg.norm(lv[1]-lv[0])),
        seconds=time.perf_counter()-start)


def _softplus(x):
    return max(x, 0.)+math.log1p(math.exp(-abs(x)))


def _lae(a, b):
    return max(a, b)+math.log1p(math.exp(-abs(a-b)))


def markov_inference(likelihood_ratio, transition, prior_logit, starts):
    """Exact two-state forward/backward using log odds; no posterior clipping.

    Returns smoothed token logits and expected transition counts. Segment starts
    reset BOTH recursions, so transitions never cross answers or reset boundaries.
    """
    ll = np.asarray(likelihood_ratio, float)
    n = len(ll)
    starts = np.asarray(starts, bool)
    if n == 0 or starts.shape != (n,) or not starts[0]:
        raise ValueError('nonempty sequence and explicit initial boundary required')
    a00, a01, a10, a11 = np.log(transition).ravel()
    alpha = np.empty(n)
    beta = np.zeros(n)
    for t in range(n):
        if starts[t]:
            pred = prior_logit
        else:
            prev = alpha[t-1]
            pred = _lae(a01, prev+a11)-_lae(a00, prev+a10)
        alpha[t] = ll[t]+pred
    counts = np.zeros((2, 2))
    for t in range(n-2, -1, -1):
        if starts[t+1]:
            continue
        r = ll[t+1]+beta[t+1]
        beta[t] = _lae(a10, a11+r)-_lae(a00, a01+r)
        pa0, pa1 = -_softplus(alpha[t]), -_softplus(-alpha[t])
        c00, c01, c10, c11 = pa0+a00, pa0+a01+r, pa1+a10, pa1+a11+r
        normal = _lae(_lae(c00, c01), _lae(c10, c11))
        counts[0, 0] += math.exp(c00-normal)
        counts[0, 1] += math.exp(c01-normal)
        counts[1, 0] += math.exp(c10-normal)
        counts[1, 1] += math.exp(c11-normal)
    return alpha+beta, counts


def markov_fit(raw_logit, prior_logit, starts, *, maxiter=25):
    prior = expit(prior_logit)
    # Starting independence reproduces the original RBM. Epsilon only protects
    # an exactly rounded PRIOR in transition initialization, not token logits.
    p = min(max(float(prior), 1e-12), 1-1e-12)
    a = np.tile([1-p, p], (2, 1))
    ll = raw_logit-prior_logit
    start = time.perf_counter()
    for it in range(maxiter):
        _, counts = markov_inference(ll, a, prior_logit, starts)
        new = counts+1.  # Fixed Dirichlet(2,2) row prior: MAP transition update.
        new /= new.sum(axis=1, keepdims=True)
        change = float(np.max(np.abs(new-a)))
        a = new
        if change < 1e-6:
            break
    score, counts = markov_inference(ll, a, prior_logit, starts)
    return score, a, dict(iterations=it+1, converged=change<1e-6,
                         transition_change=change, expected_pairs=float(counts.sum()),
                         segments=int(np.sum(starts)), seconds=time.perf_counter()-start)
