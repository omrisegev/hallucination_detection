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

        # NOTE (Step 201, defect 5): this used to call `distributional_orient`
        # here and keep only the `flipped` flag, so the prior-free orientation
        # never reached a scored number -- the bench re-fuses `cols` through its
        # own path anyway. Orientation is the BENCH's job (it scores an anchored
        # and a prior-free arm side by side); the selector's contract is to
        # return `cols`. The dead call is removed rather than papered over.
        return [{
            'variant': 'a7.iter_consensus',
            'cols': selected_cols,
            'fallback': False,
            'diag': {
                'k_star': int(k_star),
                'n_iter': int(it + 1),
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
    """Planted-signal known-answer test (auto-discovered by smoke_selectors.py).

    Rewritten for Step 201 (defect 6). The previous version could not fail: the
    fallback path returns ALL p columns, and the only assertion was
    `len(cols) >= 3`, so a selector that fell back on every cell still passed. It
    also used a local `DummyCell` carrying only `.V`/`.feat_names`, so the real
    `UnlabeledCell` contract (pool / anchor / rho / pool_bits) was never
    exercised. This version asserts the things that can actually break.
    """
    from ..selector_bench import UnlabeledCell
    from ..subset_sweep import CANONICAL_POOL
    from ..fusion_utils import zscore

    # Planted world: N_INFO informative columns driven by a latent consensus,
    # the rest pure noise. A working selector must PREFER the informative ones.
    rng_np = np.random.default_rng(20260725)
    n, p, N_INFO = 400, 14, 6
    y = rng_np.standard_normal(n)
    cols = [zscore(y + 0.55 * rng_np.standard_normal(n)) for _ in range(N_INFO)]
    cols += [zscore(rng_np.standard_normal(n)) for _ in range(p - N_INFO)]
    V = np.column_stack(cols)

    pool = list(CANONICAL_POOL[:p])
    assert len(pool) == p
    rho = np.abs(np.corrcoef(V.T))
    cell = UnlabeledCell(domain='smoke', cell_key='iter_consensus', pool=pool,
                         pool_bits=np.arange(p, dtype=np.uint8), V=V,
                         anchor=zscore(V[:, 0]), anchor_name=pool[0], rho=rho)

    res = a7_iter_consensus(cell, np.random.default_rng([0, 7]))
    assert isinstance(res, list) and len(res) == 1, f"expected 1 result, got {res}"
    item = res[0]
    assert item['variant'] == 'a7.iter_consensus', item['variant']

    # (a) THE assertion the old test was missing: it must not silently fall back.
    assert not item.get('fallback', False), \
        f"(a) selector fell back instead of selecting: {item['diag']}"

    sel = set(int(c) for c in item['cols'])
    assert 3 <= len(sel) < p, f"(b) degenerate selection size {len(sel)} (p={p})"

    # (c) it must actually prefer the planted signal over the planted noise
    info = sel & set(range(N_INFO))
    noise = sel - set(range(N_INFO))
    assert len(info) > len(noise), \
        f"(c) selection is noise-dominated: {len(info)} informative vs {len(noise)} noise"

    # (d) determinism under an equal-seeded rng
    res2 = a7_iter_consensus(cell, np.random.default_rng([0, 7]))
    assert list(res2[0]['cols']) == list(item['cols']), "(d) cols not deterministic"

    d = item['diag']
    print(f"    [note] a7 smoke: k_star={d.get('k_star')} n_iter={d.get('n_iter')} "
          f"selected={len(sel)} (informative {len(info)}/{N_INFO}, "
          f"noise {len(noise)}/{p - N_INFO})")


if __name__ == '__main__':
    smoke()
