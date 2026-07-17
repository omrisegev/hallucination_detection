"""
Classical unsupervised (label-free) feature-selection methods — the
foundational "select the subset first, then fuse" line, and the ancestors of
the Lindenbaum Gated-Laplacian family (Step 186, the pre-fusion FS stage).

Each method scores/ranks features from the unlabeled feature matrix alone;
the SAME L-SML then fuses the chosen subset (via the bench). This is a
genuine feature-selection stage, NOT the fusion model judging subsets by its
own residual (that is the A1 track).

Three methods, sharing one sample graph per cell:

  * Laplacian Score — He, Cai, Niyogi, NIPS 2005. Rank each feature by how
    well it respects the sample-similarity graph: L_r = (f~ᵀ L f~)/(f~ᵀ D f~)
    with f~ the degree-centred feature; LOWER = better (varies along the
    manifold, not across it). Scale-invariant, so the z-scored views are fine.

  * SPEC (φ2) — Zhao & Liu, ICML 2007. Spectral-FS generalization: score a
    feature by its normalized-Laplacian Rayleigh quotient with the trivial
    (all-ones) eigen-direction removed; LOWER = better.

  * MCFS — Cai, Zhang, He, KDD 2010. Multi-Cluster FS: take the top-K
    spectral-embedding eigenvectors of the graph, L1-regress each on the
    features, score feature j = max_k |coef_k[j]|; HIGHER = better (feature
    reconstructs the cluster structure).

All three emit subsets at sizes {4,5,6} plus an adaptive size (largest gap in
the sorted score). Runtime: one graph + three rankings per cell, well under a
second at our scales (sample graph capped at 1200 nodes).

NOTE on the simple-stats floor: pure VARIANCE selection is degenerate here —
`UnlabeledCell.V` is z-scored per column (unit variance), so variance carries
no signal. The meaningful naive baselines (MAD, kurtosis, decorrelation,
random) live in simple_stats.py.
"""

import numpy as np
from scipy.sparse import csr_matrix, eye as speye
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import Lasso
from sklearn.neighbors import kneighbors_graph

from ..subset_sweep import GOOD_5
from . import register

SIZES = (4, 5, 6)
GRAPH_CAP = 1200        # subsample rows for the O(n²) neighbour graph
N_NEIGHBORS = 10
MCFS_K = 5              # spectral-embedding dimension (cluster count prior)
MCFS_ALPHA = 0.05


# ---------------------------------------------------------------------------
# shared sample graph
# ---------------------------------------------------------------------------

def _sample_graph(V, rng, n_neighbors=N_NEIGHBORS, cap=GRAPH_CAP):
    """Symmetric Gaussian kNN affinity on (subsampled) sample rows of V.
    Returns (W sparse, sub_idx). Bandwidth = median non-zero kNN distance."""
    n = V.shape[0]
    if n > cap:
        sub = np.sort(rng.choice(n, size=cap, replace=False))
        X = V[sub]
    else:
        sub = np.arange(n)
        X = V
    k = int(min(n_neighbors, X.shape[0] - 1))
    D = kneighbors_graph(X, n_neighbors=k, mode='distance', include_self=False)
    d = D.data
    sigma = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    D.data = np.exp(-(d ** 2) / (2.0 * sigma * sigma + 1e-12))
    W = D.maximum(D.T)          # symmetrize
    return W, sub


def _adaptive_size(scores_desc):
    """k = index of the largest gap in the top of the sorted (descending,
    best-first) score vector, clamped to [3, 8]."""
    s = np.asarray(scores_desc, dtype=float)
    top = s[:min(9, len(s))]
    if len(top) < 4:
        return max(3, len(s))
    gaps = -np.diff(top)        # drops between consecutive best features
    k = int(np.argmax(gaps[2:]) + 3)   # never below 3
    return int(np.clip(k, 3, 8))


def _emit(prefix, order_best_first, scores_best_first, extra=None):
    """Emit size-{4,5,6}+adaptive selections for a ranking (best first)."""
    out = []
    p = len(order_best_first)
    for s in SIZES:
        if s > p:
            continue
        out.append({'variant': f'{prefix}_s{s}',
                    'cols': np.sort(order_best_first[:s]),
                    'diag': dict(extra or {}, rule=prefix, size=s)})
    ka = _adaptive_size(scores_best_first)
    ka = min(ka, p)
    out.append({'variant': f'{prefix}_adapt',
                'cols': np.sort(order_best_first[:ka]),
                'diag': dict(extra or {}, rule=prefix, size=ka, adaptive=True)})
    return out


# ---------------------------------------------------------------------------
# the three rankings
# ---------------------------------------------------------------------------

def _laplacian_score(Vg, W):
    """He-Cai-Niyogi 2005. Returns per-feature score (LOWER better)."""
    deg = np.asarray(W.sum(axis=1)).ravel()
    D = deg
    one_D_one = float(deg.sum())
    scores = np.empty(Vg.shape[1])
    Wcsr = csr_matrix(W)
    for j in range(Vg.shape[1]):
        f = Vg[:, j]
        fbar = f - (f @ D) / (one_D_one + 1e-12)
        num = fbar @ (D * fbar) - fbar @ (Wcsr @ fbar)   # f~ᵀ L f~, L=D-W
        den = fbar @ (D * fbar)                          # f~ᵀ D f~
        scores[j] = num / (den + 1e-12)
    return scores


def _spec_phi2(Vg, W):
    """Zhao-Liu 2007 φ2 on the normalized Laplacian (LOWER better)."""
    deg = np.asarray(W.sum(axis=1)).ravel()
    dsqrt = np.sqrt(np.maximum(deg, 1e-12))
    xi0 = dsqrt / (np.linalg.norm(dsqrt) + 1e-12)        # trivial eigenvector
    Wcsr = csr_matrix(W)
    scores = np.empty(Vg.shape[1])
    for j in range(Vg.shape[1]):
        fh = dsqrt * Vg[:, j]
        nrm = np.linalg.norm(fh)
        if nrm < 1e-12:
            scores[j] = np.inf
            continue
        fh = fh / nrm
        # f_hatᵀ L_norm f_hat = f_hatᵀ f_hat - (D^-1/2 W D^-1/2 quadratic form)
        Df = fh / dsqrt
        rq = float(fh @ fh - Df @ (Wcsr @ Df))
        overlap = float(fh @ xi0) ** 2
        scores[j] = rq / (1.0 - overlap + 1e-12)         # remove trivial mass
    return scores


def _mcfs(Vg, W, rng, K=MCFS_K, alpha=MCFS_ALPHA):
    """Cai-Zhang-He 2010 (HIGHER better = max |Lasso coef| across embeddings)."""
    n, p = Vg.shape
    deg = np.asarray(W.sum(axis=1)).ravel()
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    Wcsr = csr_matrix(W)
    Lnorm = speye(n) - csr_matrix((Wcsr.multiply(dinv[:, None])).multiply(dinv[None, :]))
    k = int(min(K + 1, n - 2, p))
    if k < 2:
        return np.full(p, np.nan)
    try:
        vals, vecs = eigsh(Lnorm, k=k, sigma=0, which='LM')
    except Exception:
        vals, vecs = eigsh(Lnorm + 1e-6 * speye(n), k=k, which='SM')
    order = np.argsort(vals)
    Y = vecs[:, order[1:]]                                # drop trivial
    scores = np.zeros(p)
    for c in range(Y.shape[1]):
        las = Lasso(alpha=alpha, max_iter=2000)
        las.fit(Vg, Y[:, c])
        scores = np.maximum(scores, np.abs(las.coef_))
    return scores


@register('classical_fs')
def classical_fs(cell, rng, cache=None):
    V = cell.V
    W, sub = _sample_graph(V, rng)
    Vg = V[sub]
    out = []

    ls = _laplacian_score(Vg, W)
    order = np.argsort(ls)                    # ascending: lowest score first
    out += _emit('lapscore', order, -ls[order])

    sp = _spec_phi2(Vg, W)
    order = np.argsort(sp)
    out += _emit('spec', order, -sp[order])

    mc = _mcfs(Vg, W, rng)
    if np.all(np.isfinite(mc)) and np.any(mc > 0):
        order = np.argsort(mc)[::-1]          # descending: highest first
        out += _emit('mcfs', order, mc[order])
    else:
        out.append({'variant': 'mcfs_s5',
                    'cols': np.arange(min(5, cell.p)), 'fallback': True,
                    'diag': {'error': 'mcfs embedding failed'}})
    return out


# ---------------------------------------------------------------------------
# smoke() — planted-informative recovery on a known-answer world
# ---------------------------------------------------------------------------

def smoke():
    import sys
    sys.path.insert(0, __file__.rsplit('spectral_utils', 1)[0] + 'scripts')
    from smoke_selectors import _tiny_ctx
    from ..selector_bench import UnlabeledCell

    ctx = _tiny_ctx(informative=5)            # 5 y-correlated, 11 noise
    cell = UnlabeledCell.from_context(ctx)

    s1 = classical_fs(cell, np.random.default_rng([0, 7]))
    s2 = classical_fs(cell, np.random.default_rng([0, 7]))
    v1 = {d['variant']: list(d['cols']) for d in s1}
    v2 = {d['variant']: list(d['cols']) for d in s2}
    assert v1 == v2, "classical_fs not deterministic under equal-seeded rng"

    informative = set(range(5))
    # each ranking method's size-5 pick should recover a majority of the
    # planted informative features (the graph-respect signal coincides with
    # label-relevance in this constructed world)
    for method in ('lapscore', 'spec', 'mcfs'):
        key = f'{method}_s5'
        if key not in v1:
            continue
        hit = len(informative & set(v1[key]))
        assert hit >= 3, f"{key} recovered only {hit}/5 informative (cols {v1[key]})"
    names = {d['variant'] for d in s1}
    assert any(n.startswith('lapscore') for n in names)
    assert any(n.startswith('mcfs') for n in names)
    for d in s1:
        c = np.asarray(d['cols'])
        assert len(c) >= 3 and c.min() >= 0 and c.max() < cell.p
    print(f"    [note] classical_fs smoke: {len(s1)} variants "
          f"(lapscore/spec/mcfs × sizes)")
