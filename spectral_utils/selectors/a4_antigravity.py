"""
A4 — Antigravity unsupervised / self-supervised feature-subset selectors (Step 186+).
"""
import numpy as np
import scipy.stats
from . import register
from ..subset_sweep import GOOD_5
from ..rank_tests import ahn_horenstein_K, cov_eigvals, kritchman_nadler_K

# Pre-registered adaptive-size range and fixed sizes
ADAPT_SIZES = (3, 4, 5, 6, 7, 8)
FIXED_SIZES = (4, 5, 6)

def _reconstruction_error(V, cols):
    """
    Compute the linear projection/reconstruction error of all columns of V
    onto the subspace spanned by V[:, cols].
    """
    if len(cols) == 0:
        return float(np.sum(V ** 2))
    V_cols = V[:, cols]
    # Solve linear regression V_cols * W = V in a least-squares sense
    W, _, _, _ = np.linalg.lstsq(V_cols, V, rcond=None)
    preds = V_cols @ W
    error = float(np.sum((V - preds) ** 2))
    return error

def _greedy_reconstruction_path(V, max_size=8):
    """
    Greedy forward selection minimizing linear reconstruction error.
    """
    p = V.shape[1]
    cols = []
    path = {}
    while len(cols) < min(max_size, p):
        best_j, best_err = None, np.inf
        for j in range(p):
            if j in cols:
                continue
            cand_cols = sorted(cols + [j])
            err = _reconstruction_error(V, cand_cols)
            if err < best_err:
                best_err, best_j = err, j
        if best_j is None:
            break
        cols.append(best_j)
        path[len(cols)] = (sorted(cols), best_err)
    return path

@register('a4_antigravity')
def a4_antigravity(cell, rng, cache=None):
    V = cell.V
    p = V.shape[1]
    out = []

    # -----------------------------------------------------------------------
    # 1. Candidate 1: Self-Supervised Selection (Anchor-Targeted)
    # -----------------------------------------------------------------------
    corr_vals = []
    for j in range(p):
        r, _ = scipy.stats.spearmanr(V[:, j], cell.anchor)
        corr_vals.append(abs(r) if np.isfinite(r) else 0.0)

    corr_vals = np.array(corr_vals)
    sorted_indices = np.argsort(corr_vals)[::-1]  # descending order

    def get_anchor_variants():
        res = []
        for s in FIXED_SIZES:
            if s <= p:
                cols = np.sort(sorted_indices[:s])
                res.append({
                    'variant': f'a4.anchor_s{s}',
                    'cols': cols,
                    'diag': {
                        'mean_correlation': float(np.mean(corr_vals[cols]))
                    }
                })
        # Adaptive size based on drop-off elbow (above 60% of max correlation, capped [3, 8])
        max_corr = np.max(corr_vals)
        if max_corr > 1e-8:
            adapt_mask = corr_vals >= 0.6 * max_corr
            adapt_cols = np.flatnonzero(adapt_mask)
            if len(adapt_cols) < 3:
                adapt_cols = sorted_indices[:3]
            elif len(adapt_cols) > 8:
                adapt_cols = sorted_indices[:8]
        else:
            adapt_cols = sorted_indices[:5]
        
        adapt_cols = np.sort(adapt_cols)
        res.append({
            'variant': 'a4.anchor_adapt',
            'cols': adapt_cols,
            'diag': {
                'size': len(adapt_cols),
                'mean_correlation': float(np.mean(corr_vals[adapt_cols]))
            }
        })
        return res

    out.extend(get_anchor_variants())

    # -----------------------------------------------------------------------
    # 2. Candidate 4: Intrinsic-Dimensionality Adaptive Selection (LOCA/Eigen-Ratio)
    # -----------------------------------------------------------------------
    lam = cov_eigvals(V)
    d_int = ahn_horenstein_K(lam, max_K=8, n=cell.n)
    d_int = max(int(d_int), 1)  # Capped >= 1
    
    d_kn = kritchman_nadler_K(lam, n=cell.n, alpha=0.01, max_K=8)
    d_kn = max(int(d_kn), 1)

    k_dim = int(np.clip(d_int + 3, 3, min(8, p)))
    id_cols = np.sort(sorted_indices[:k_dim])

    out.append({
        'variant': 'a4.intrinsic_k_ah',
        'cols': id_cols,
        'K': d_int,
        'diag': {
            'intrinsic_dim_ah': d_int,
            'intrinsic_dim_kn': d_kn,
            'k_dim': k_dim
        }
    })

    # GOOD_5 with intrinsic K swap
    g5_cols = np.asarray([cell.pool.index(f) for f in GOOD_5 if f in cell.pool], dtype=np.int64)
    g5_fallback = len(g5_cols) < 3
    if g5_fallback:
        g5_cols = np.arange(min(5, p))

    out.append({
        'variant': 'a4.good5+K_intrinsic_ah',
        'cols': g5_cols,
        'K': d_int,
        'fallback': g5_fallback,
        'diag': {
            'intrinsic_dim_ah': d_int,
            'intrinsic_dim_kn': d_kn
        }
    })
    
    out.append({
        'variant': 'a4.good5+K_intrinsic_kn',
        'cols': g5_cols,
        'K': d_kn,
        'fallback': g5_fallback,
        'diag': {
            'intrinsic_dim_ah': d_int,
            'intrinsic_dim_kn': d_kn
        }
    })

    # -----------------------------------------------------------------------
    # 3. Candidate 3: Unsupervised Column Reconstruction Selection (Greedy CSSP)
    # -----------------------------------------------------------------------
    recon_path = _greedy_reconstruction_path(V, max_size=8)
    
    for s in FIXED_SIZES:
        if s in recon_path:
            cols, err = recon_path[s]
            out.append({
                'variant': f'a4.recon_s{s}',
                'cols': np.sort(cols),
                'diag': {
                    'reconstruction_error': float(err)
                }
            })
            
    # Adaptive size based on reconstruction error elbow (relative error reduction)
    # R(s) = err(s) / err(none). We want to find s where reduction slows down.
    err_none = _reconstruction_error(V, [])
    best_size = 5
    min_ratio_diff = 0.0
    for s in range(4, min(8, p)):
        if s in recon_path and (s-1) in recon_path and (s+1) in recon_path:
            # check the second difference (elbow metric)
            y_prev = recon_path[s-1][1] / err_none
            y_curr = recon_path[s][1] / err_none
            y_next = recon_path[s+1][1] / err_none
            diff = (y_prev - y_curr) - (y_curr - y_next)
            if diff > min_ratio_diff:
                min_ratio_diff = diff
                best_size = s

    best_size = int(np.clip(best_size, 3, min(8, p)))
    if best_size in recon_path:
        cols, err = recon_path[best_size]
        out.append({
            'variant': 'a4.recon_adapt',
            'cols': np.sort(cols),
            'diag': {
                'size': len(cols),
                'reconstruction_error': float(err)
            }
        })

    return out

# ---------------------------------------------------------------------------
# smoke() — known-answer unit-test gate
# ---------------------------------------------------------------------------
def smoke():
    from ..selector_bench import UnlabeledCell
    from ..fusion_utils import zscore
    from ..subset_sweep import CANONICAL_POOL
    
    # Generate synthetic data with 4 highly correlated views and 6 noise columns
    rng_np = np.random.default_rng(20260717)
    n, p = 200, 10
    y = rng_np.standard_normal(n)  # true latent consensus
    cols = []
    # 4 consensus-aligned features
    for _ in range(4):
        cols.append(zscore(y + 0.3 * rng_np.standard_normal(n)))
    # 6 pure-noise features
    for _ in range(6):
        cols.append(zscore(rng_np.standard_normal(n)))
        
    V = np.column_stack(cols)
    pool = list(CANONICAL_POOL[:p])
    pool_bits = np.arange(p, dtype=np.uint8)
    rho = np.abs(np.corrcoef(V.T))
    
    # We pass an anchor that is correlated with y (self-supervised proxy)
    anchor = zscore(y + 0.1 * rng_np.standard_normal(n))
    
    cell = UnlabeledCell(domain='smoke', cell_key='antigravity', pool=pool,
                         pool_bits=pool_bits, V=V, anchor=anchor,
                         anchor_name=pool[0], rho=rho)
                         
    sels1 = a4_antigravity(cell, np.random.default_rng(123))
    sels2 = a4_antigravity(cell, np.random.default_rng(123))
    
    # Ensure variants are deterministic and match expected set
    expected_variants = {
        'a4.anchor_s4', 'a4.anchor_s5', 'a4.anchor_s6', 'a4.anchor_adapt',
        'a4.intrinsic_k_ah', 'a4.good5+K_intrinsic_ah', 'a4.good5+K_intrinsic_kn',
        'a4.recon_s4', 'a4.recon_s5', 'a4.recon_s6', 'a4.recon_adapt'
    }
    
    variants1 = {s['variant'] for s in sels1}
    assert variants1 == expected_variants, f"variant set mismatch: {variants1} vs {expected_variants}"
    
    for s1, s2 in zip(sels1, sels2):
        assert s1['variant'] == s2['variant'], "order mismatch"
        assert list(s1['cols']) == list(s2['cols']), f"{s1['variant']} cols not deterministic"
        
    # Check that a4.anchor_s4 selected mostly the consensus features (indices 0..3)
    anchor_s4 = next(s for s in sels1 if s['variant'] == 'a4.anchor_s4')
    # The consensus features are the first 4 (0, 1, 2, 3), they must be preferred
    assert set(anchor_s4['cols']).issubset({0, 1, 2, 3}), f"anchor_s4 selected non-consensus features: {anchor_s4['cols']}"

    # Check that reconstruction selector successfully runs and has descending error with size
    recon_s4 = next(s for s in sels1 if s['variant'] == 'a4.recon_s4')
    recon_s6 = next(s for s in sels1 if s['variant'] == 'a4.recon_s6')
    assert recon_s6['diag']['reconstruction_error'] < recon_s4['diag']['reconstruction_error'], "recon error does not descend"
    
    print(f"    [note] a4 smoke: {len(sels1)} variants successfully benched & verified.")
