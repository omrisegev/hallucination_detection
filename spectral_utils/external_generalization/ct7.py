"""Frozen CT7 step reference; equations ported from the recorded levers worktree.

No target labels. Preserve float32 storage round trips of the original pipeline.
This is the step recipe, not Top10 of seven token streams.
"""
import numpy as np
from .fusion import standardize, answer_z, top10
from ._bank11.chosen_token_calibration import token_calibration, step_sufficient_stats, step_z_readouts


def bocpd_mean(values, hazard=1/32):
    x = np.asarray(values, float)
    n, d = x.shape
    out = np.zeros_like(x)
    p = np.zeros((n, d)); mu = np.zeros_like(p); p[0] = 1
    v = 1/(np.arange(n, dtype=float)+1); variance = v+1
    lognorm = -.5*np.log(2*np.pi*variance); gain = v/variance
    for t in range(n):
        count = t+1
        if t:
            p[1:count] = (1-hazard)*p[:t]; p[0] = hazard
            mu[1:count] = mu[:t]; mu[0] = 0
        out[t] = np.sum(p[:count]*mu[:count], axis=0)
        loglike = lognorm[:count,None]-.5*(x[t]-mu[:count])**2/variance[:count,None]
        p[:count] *= np.exp(loglike-loglike.max(axis=0))
        p[:count] /= p[:count].sum(axis=0)
        mu[:count] += gain[:count,None]*(x[t]-mu[:count])
    return out


def step_scores(row, spans):
    lp = np.asarray(row['top_k_logprobs']['logprobs'], float)
    p = np.exp(lp[:,:15]); q = p/(p.sum(axis=1,keepdims=True)+1e-12)
    s = -np.log(q+1e-12)
    columns = [np.log(q+1e-12).mean(axis=1)]
    for alpha in (0., .75, 1.):
        w = np.full_like(q, 1/q.shape[1]) if alpha == 0 else q**alpha
        if alpha: w /= w.sum(axis=1,keepdims=True)
        m = (w*s).sum(axis=1)
        columns.append((w*s*s).sum(axis=1)-m*m)
    raw = np.column_stack(columns)
    anchor = raw[:,-1]-raw[:,-1].mean()
    bank = raw.copy()
    for j in (1,2,3):
        if np.dot(bank[:,j]-bank[:,j].mean(), anchor) < 0: bank[:,j] *= -1
    innovation = np.zeros(len(lp))
    innovation[1:] = bank[1:,0]-np.cumsum(bank[:,0])[:-1]/np.arange(1,len(lp))
    bank = np.column_stack((bank,innovation))
    per_step = top10(bank,spans)
    # Only the prefix-innovation stream excludes token zero.
    available = np.ones(per_step.shape, bool)
    for i,(a,b) in enumerate(spans):
        if a == 0:
            if b > 1: per_step[i,4] = top10(bank[:,4],[(1,b)])[0]
            else: per_step[i,4] = 0; available[i,4] = False
    per_step = per_step.astype(np.float32).astype(float)
    scaled = np.zeros_like(per_step)
    for j in range(5):
        keep = available[:,j]; v = per_step[keep,j]
        if len(v) and v.std() > 1e-12: scaled[keep,j] = (v-v.mean())/v.std()
    # Temporal recipe orients H0lim too, unlike the broad-bank five columns.
    temporal = raw.copy()
    for j in range(4):
        if np.dot(temporal[:,j]-temporal[:,j].mean(),anchor) < 0: temporal[:,j] *= -1
    innov = np.zeros(len(lp))
    innov[1:] = temporal[1:,0]-np.cumsum(temporal[:,0])[:-1]/np.arange(1,len(lp))
    augmented = np.column_stack((temporal,innov))
    z = (augmented.astype(np.float32).astype(float)-augmented.mean(0))/np.maximum(augmented.std(0),1e-8)
    residual = (z-bocpd_mean(z)).mean(1)
    residual_step = top10(residual,spans)
    residual_step = standardize(residual_step)
    x,c,d = token_calibration(lp,np.asarray(row['top_k_logprobs']['ids']),
                              np.asarray(row['gen_token_ids']),np.asarray(row['token_spilled_energies']))
    chosen = standardize(step_z_readouts(step_sufficient_stats(x,c,d,spans))[:,2])
    if len(chosen)>1: chosen[0] = chosen[1:].mean()
    chosen = standardize(chosen)
    return answer_z(np.column_stack((scaled,residual_step,chosen)).mean(1))
