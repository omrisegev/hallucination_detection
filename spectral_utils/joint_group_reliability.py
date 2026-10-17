"""Model-derived SML reliability on frozen Joint virtual group scores.

For t_g=x_g v_g, a_g=Cov(t_g,z)=||v_g||^2 and V_g=Var_model(t_g).
Standardizing t_g and multiplying by model reliability a_g/sqrt(V_g)
gives a_g/V_g times t_g. No group is promoted by unit normalization alone.
This estimates the shared factor, not proven correctness reliability.
"""
import numpy as np


def group_reliability_weights(covariance, global_loading, labels):
    c = np.asarray(covariance, float); v = np.asarray(global_loading, float)
    labels = np.asarray(labels)
    if c.shape != (len(v), len(v)) or labels.shape != v.shape:
        raise ValueError('COVARIANCE_LOADING_PARTITION_MISMATCH')
    if not np.isfinite(c).all() or not np.isfinite(v).all():
        raise ValueError('NONFINITE_FACTOR_MODEL')
    weight = np.zeros(len(v)); rows = []
    for group in np.unique(labels):
        ids = np.flatnonzero(labels == group); vg = v[ids]
        a = float(vg@vg); variance = float(vg@c[np.ix_(ids,ids)]@vg)
        scale = a/variance if variance > 1e-14 else 0.
        weight[ids] = scale*vg
        rows.append(dict(group=int(group), coefficient=a, model_variance=variance,
            modeled_correlation=a/np.sqrt(variance) if variance > 1e-14 else 0.,
            multiplier=scale))
    return weight, dict(groups=rows, within_weights='unchanged global_loading',
        outer_weights='modeled correlation of standardized virtual score with shared factor')
