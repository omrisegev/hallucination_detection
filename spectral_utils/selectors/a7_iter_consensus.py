"""
a7_iter_consensus — Prior-Free Iterative Consensus Refinement Selector (Extension H / Phase 3).

MOTIVATION:
Step 199 prior-free decision. The one-shot full-pool L-SML consensus provides an initial
target y_0. Features are iteratively ranked by absolute correlation with the current
consensus target |corr(V_j, y_t)|, and L-SML fusion is refitted on the top-m subset.

CONTRACT:
Sees only an UnlabeledCell (cell.V, cell.feat_names) — zero labels, zero positive rate.
On any failure, degrades to full-pool fallback (never raises).
Exposes smoke() for auto-discovery.
"""

import numpy as np

from ..fusion_utils import lsml_continuous, zscore
from ..orientation import distributional_orient, z2_sign_recovery
from . import register
from .adaptive_k import predict_k

_EPS = 1e-12


def _corr_with(V, target):
    """Compute absolute sample correlation of each column of V with target vector."""
    V = np.asarray(V, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).ravel()
    n, p = V.shape
    if n < 3 or p == 0:
        return np.zeros(p)

    t_centered = target - np.mean(target)
    t_norm = np.linalg.norm(t_centered) + _EPS

    V_centered = V - np.mean(V, axis=0, keepdims=True)
    V_norms = np.linalg.norm(V_centered, axis=0) + _EPS

    corrs = np.dot(t_centered, V_centered) / (t_norm * V_norms)
    return np.abs(corrs)


@register('a7_iter_consensus')
def a7_iter_consensus(cell, rng, cache=None):
    """Iterative prior-free consensus selector.

    Refines the pseudo-label target iteratively starting from the full-pool
    fused score y0.
    """
    V = np.asarray(cell.V, dtype=np.float64)
    n, p = V.shape
    full_cols = list(range(p))

    # Fallback response (full pool)
    fallback_res = [{
        'variant': 'a7.iter_consensus',
        'cols': full_cols,
        'fallback': True,
        'diag': {'reason': 'initial_setup_fallback'}
    }]

    if p < 3 or n < 20:
        return fallback_res

    try:
        # Step 1: Initial full-pool L-SML continuous fusion with Z2 sign recovery
        z2_full = z2_sign_recovery(V)
        V_signed = V * z2_full
        y_curr, meta0 = lsml_continuous(*[V_signed[:, j] for j in range(p)])
        y_curr = np.asarray(y_curr, dtype=np.float64)

        # Step 2: Determine initial feature ranking by correlation with full consensus
        corrs = _corr_with(V_signed, y_curr)
        ranking = list(np.argsort(corrs)[::-1])

        # Step 3: Determine adaptive subset budget K* (label-free size rule)
        k_star = predict_k(V_signed, ranking, rule='eff_rank', k_min=3, k_max=min(15, p))

        prev_cols = set()
        selected_cols = ranking[:k_star]
        max_iter = 5

        # Step 4: Iterative target refinement
        for it in range(max_iter):
            curr_set = set(selected_cols)
            if curr_set == prev_cols or len(selected_cols) < 3:
                break
            prev_cols = curr_set

            # Refit L-SML fusion on selected top-k features with Z2 relative signs
            sub_V = V[:, selected_cols]
            z2_sub = z2_sign_recovery(sub_V)
            sub_V_signed = sub_V * z2_sub
            try:
                y_curr, meta = lsml_continuous(*[sub_V_signed[:, c] for c in range(len(selected_cols))])
                y_curr = np.asarray(y_curr, dtype=np.float64)
            except Exception:
                break

            # Re-rank full feature pool against updated target
            corrs = _corr_with(V_signed, y_curr)
            ranking = list(np.argsort(corrs)[::-1])
            selected_cols = ranking[:k_star]

        # Step 5: Prior-free global orientation
        oriented_score, flipped = distributional_orient(y_curr)

        return [{
            'variant': 'a7.iter_consensus',
            'cols': selected_cols,
            'fallback': False,
            'diag': {
                'k_star': int(k_star),
                'n_iter': int(it + 1),
                'flipped': bool(flipped),
                'final_residual': float(meta.get('residual', np.nan)) if 'meta' in locals() else np.nan
            }
        }]

    except Exception as exc:
        return [{
            'variant': 'a7.iter_consensus',
            'cols': full_cols,
            'fallback': True,
            'diag': {'error': str(exc)}
        }]


def smoke():
    """Known-answer sanity test for a7_iter_consensus."""
    class DummyCell:
        def __init__(self):
            rng = np.random.RandomState(42)
            n, p = 100, 10
            # 3 strong correlated signal columns + 7 noise columns
            signal = rng.randn(n, 1)
            noise = rng.randn(n, p - 3)
            self.V = np.hstack([signal + 0.1 * rng.randn(n, 3), noise])
            self.feat_names = [f"f{i}" for i in range(p)]
            self.cell_key = "dummy_cell"

    cell = DummyCell()
    rng = np.random.default_rng(42)
    res = a7_iter_consensus(cell, rng)
    assert isinstance(res, list) and len(res) == 1
    item = res[0]
    assert 'variant' in item and item['variant'] == 'a7.iter_consensus'
    assert 'cols' in item and len(item['cols']) >= 3
    print("a7_iter_consensus smoke test passed!")


if __name__ == '__main__':
    smoke()
