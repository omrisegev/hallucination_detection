"""Non-negative SML weights for pre-oriented views (diagnostic, label-free).

The L-SML paper (Jaffe et al. 2016, Lemma 4) assumes every classifier beats random, so under the model all
entries of the rank-one vector v (R_off ~ v v^T) share one sign. fusion_utils.sml_fuse_signed instead lets
components be negative ("inversely informative") and fixes the global sign by majority. For risk-oriented
views that permissiveness is what inverted Math-Shepherd's CHAIN on Socratic.

Modes (replace fusion_utils.sml_fuse_signed; everything else in lsml_continuous unchanged):
  'signed' : the shipped function (control, must reproduce shipped scores)
  'abs'    : |v| of the same eigenvector
  'clip'   : majority-signed v, negative components set to 0
  'nnr1'   : non-negative rank-one fit of R_off: min sum_{i!=j} (r_ij - v_i v_j)^2, v >= 0
             (projected coordinate descent from |v| * sqrt(lambda))
Each non-signed mode also records whether the signed eigenvector had a negative component (a model-violation flag).
"""
import numpy as np
from scipy.linalg import eigh

FLAGS = []


def _nnr1(R_off, v0, iters=500, tol=1e-10):
    v = np.maximum(v0, 0.).astype(float)
    for _ in range(iters):
        old = v.copy()
        for i in range(len(v)):
            m = np.ones(len(v), bool); m[i] = False
            den = float(v[m] @ v[m])
            v[i] = max(0., float(R_off[i, m] @ v[m]) / den) if den > 1e-15 else 0.
        if np.max(np.abs(v - old)) < tol:
            break
    return v


def make(mode, orig):
    if mode == 'signed':
        return orig

    def fuse(*classifiers, gates=None, small_m_guard=False):
        if gates is not None or small_m_guard:
            raise NotImplementedError('diagnostic covers the ungated, unguarded path only')
        X = np.column_stack(classifiers); k = X.shape[1]
        R = np.atleast_2d(np.cov(X.T)); R_off = R - np.diag(np.diag(R))
        lam, vecs = eigh(R_off); v = vecs[:, -1]
        if np.sum(v > 0) < k / 2: v = -v
        FLAGS.append({'k': int(k), 'negative_component': bool(np.any(v < -1e-12)),
                      'min_v': float(v.min()), 'neg_offdiag_frac': float(np.mean(R_off[~np.eye(k, dtype=bool)] < 0))})
        if mode == 'signedcap':
            w = v.copy()
        elif mode == 'abs':
            w = np.abs(v)
        elif mode == 'clip':
            w = np.maximum(v, 0.)
        elif mode == 'nnr1':
            w = _nnr1(R_off, np.abs(v) * np.sqrt(max(lam[-1], 1e-12)))
        else:
            raise ValueError(mode)
        if not np.any(w > 0):
            w = np.ones(k) / k
        FLAGS[-1]['v'] = [float(x) for x in v]; FLAGS[-1]['w'] = [float(x) for x in w]
        return X @ w, w
    return fuse
