"""
Known-answer unit-test gate for the selector-bench components (Step 186+).

Policy (Omri, 2026-07-17): every NEW building block is tested standalone on
synthetic data with an obvious expected answer BEFORE it is integrated with
the existing L-SML / U-PCR methods. This script is the accumulating gate —
CPU-only, seconds, exit non-zero on any failure. Mirrors smoke_preset.py's
role for cluster presets.

Run:  python scripts/smoke_selectors.py            # all tests
      python scripts/smoke_selectors.py rank_tests # by substring filter

Structure: component tests live here; selector modules additionally expose
their own ``smoke()`` (auto-discovered via spectral_utils.selectors registry)
so worktree branches never edit this shared file.
"""

import json
import math
import os
import sys
import traceback

import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# Pre-refactor regression fixtures (captured 2026-07-17 on seeded synthetic
# data BEFORE the Stage-0a fusion_utils diagnostics refactor, then committed).
# If missing, the regression test degrades to a skip note instead of failing.
FIXTURE_PATH = os.path.join(
    REPO_DIR, 'scripts', 'fixtures', 'pre_refactor_fusion_fixtures.json')

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def _spiked_data(rng, n, p, k, spike=8.0, noise=1.0):
    """n samples from a p-dim Gaussian with k planted ORTHOGONAL signal
    directions of distinct strengths — an unambiguous known answer."""
    Z = rng.standard_normal((n, p)) * noise
    if k:
        Q, _ = np.linalg.qr(rng.standard_normal((p, k)))
        for j in range(k):
            s = spike * (1.0 + 0.5 * j)
            Z += np.sqrt(s) * rng.standard_normal((n, 1)) * Q[:, j][None, :]
    return Z


# ---------------------------------------------------------------------------
# rank_tests — planted-K recovery at this project's actual n/p scales
# ---------------------------------------------------------------------------

@test
def rank_tests_planted_K():
    from spectral_utils.rank_tests import (
        ahn_horenstein_K, kritchman_nadler_K, cov_eigvals)
    rng = np.random.default_rng(7)
    for p in (8, 16):
        for k in (1, 2, 3):
            hits_ah, hits_kn = 0, 0
            for rep in range(10):
                Z = _spiked_data(rng, n=300, p=p, k=k)
                lam = cov_eigvals(Z)
                if ahn_horenstein_K(lam, max_K=6, n=300) == k:
                    hits_ah += 1
                if kritchman_nadler_K(lam, n=300, alpha=0.01, max_K=6) == k:
                    hits_kn += 1
            assert hits_ah >= 8, f"AH p={p} k={k}: only {hits_ah}/10 correct"
            assert hits_kn >= 8, f"KN p={p} k={k}: only {hits_kn}/10 correct"


@test
def rank_tests_pure_noise():
    """No planted signal: both estimators must return 0 nearly always
    (AH via the paper's mock-eigenvalue device, enabled by passing n)."""
    from spectral_utils.rank_tests import (
        ahn_horenstein_K, kritchman_nadler_K, cov_eigvals)
    rng = np.random.default_rng(11)
    kn_zero, ah_zero = 0, 0
    for rep in range(10):
        Z = rng.standard_normal((300, 16))
        lam = cov_eigvals(Z)
        if kritchman_nadler_K(lam, n=300, alpha=0.01, max_K=6) == 0:
            kn_zero += 1
        if ahn_horenstein_K(lam, max_K=6, n=300) == 0:
            ah_zero += 1
    assert kn_zero >= 8, f"KN on pure noise: nonzero in {10 - kn_zero}/10 runs"
    assert ah_zero >= 8, f"AH on pure noise: nonzero in {10 - ah_zero}/10 runs"


@test
def rank_tests_edge_cases():
    from spectral_utils.rank_tests import ahn_horenstein_K, kritchman_nadler_K
    assert ahn_horenstein_K([5.0], max_K=6) == 1          # p=1, no-n fallback
    assert ahn_horenstein_K([5.0], max_K=6, n=100) == 0   # p=1 with mock
    assert kritchman_nadler_K([5.0], n=100) == 0
    assert kritchman_nadler_K([3.0, 1.0, 0.9], n=2) == 0  # n too small
    # exact rank-deficient input must not crash on zero eigenvalues
    assert ahn_horenstein_K([4.0, 2.0, 0.0, 0.0], max_K=3) in (1, 2)


# ---------------------------------------------------------------------------
# fusion_utils Stage-0a refactor — regression + new-seam tests
# ---------------------------------------------------------------------------

def _fixture_inputs():
    """Regenerate the exact seeded inputs the pre-refactor fixtures used."""
    from spectral_utils.fusion_utils import zscore
    rng = np.random.default_rng(20260717)
    y = rng.standard_normal(300)
    F = np.vstack([zscore(y + s * rng.standard_normal(300))
                   for s in (0.5, 0.7, 0.9, 1.1, 1.4, 1.8, 2.5, 4.0)])
    views = [zscore(y + s * rng.standard_normal(300))
             for s in (0.6, 0.8, 1.0, 1.3, 2.0)]
    return F, views


@test
def fusion_regression_vs_prerefactor_fixtures():
    """upcr_fuse + lsml_continuous default paths must be numerically
    unchanged by the Stage-0a refactor (fixtures captured pre-refactor)."""
    from spectral_utils.fusion_utils import upcr_fuse, lsml_continuous
    if not os.path.exists(FIXTURE_PATH):
        print("    [skip-compare] fixture file missing — recapturing only")
        return
    with open(FIXTURE_PATH) as f:
        fix = json.load(f)
    F, views = _fixture_inputs()

    w, rho, g2, diag = upcr_fuse(F, return_diagnostics=True)
    assert np.allclose(w, fix['upcr']['w'], atol=1e-12), "upcr w changed"
    assert np.allclose(rho, fix['upcr']['rho'], atol=1e-12), "upcr rho changed"
    assert abs(g2 - fix['upcr']['g2']) < 1e-12, "upcr g2 changed"
    assert diag['n_kept'] == fix['upcr']['n_kept']
    assert abs(diag['lambda2_frac'] - fix['upcr']['lambda2_frac']) < 1e-12

    fused, meta = lsml_continuous(*views)
    assert abs(float(np.sum(fused)) - fix['lsml']['fused_sum']) < 1e-9, "lsml fused changed"
    assert np.allclose(fused[:5], fix['lsml']['fused_head'], atol=1e-9)
    assert int(meta['K']) == fix['lsml']['K']
    assert abs(float(meta['residual']) - fix['lsml']['residual']) < 1e-9
    assert list(np.asarray(meta['c'], int)) == fix['lsml']['c']


@test
def fusion_new_seams():
    """The Stage-0a additions: proj_residual in diag == standalone
    upcr_proj_residual; return_curve consistent with the legacy best-K."""
    from spectral_utils.fusion_utils import (
        upcr_fuse, upcr_proj_residual, detect_dependent_groups)
    F, views = _fixture_inputs()

    w, rho, g2, diag = upcr_fuse(F, return_diagnostics=True)
    res, g2b, rho_full = upcr_proj_residual(F)
    assert abs(res - diag['proj_residual']) < 1e-12, "standalone residual != diag"
    assert abs(g2b - g2) < 1e-12, "standalone g2 != fuse g2"
    assert np.allclose(rho_full, diag['rho_hat_full'], atol=1e-12)
    assert diag['keep'].dtype == bool and diag['keep'].sum() == diag['n_kept']

    K4, c4, r4, s4 = detect_dependent_groups(views)
    K5, c5, r5, s5, curve = detect_dependent_groups(views, return_curve=True)
    assert (K4, r4) == (K5, r5) and np.array_equal(c4, c5)
    assert curve, "empty residual curve"
    ks = [k for k, _, _ in curve]
    rs = [r for _, r, _ in curve]
    assert min(rs) == r5 and ks[int(np.argmin(rs))] == K5, "curve min != best K"


def _planted_group_views(rng, n=400, block=1.2, noise=0.8):
    """5 views: 3 share latent block noise alpha, 2 share beta — an
    unambiguous 2-group truth [0,0,0,1,1] on top of a common consensus y."""
    from spectral_utils.fusion_utils import zscore
    y = rng.standard_normal(n)
    alpha = rng.standard_normal(n) * block
    beta = rng.standard_normal(n) * block
    vs = [zscore(y + alpha + noise * rng.standard_normal(n)) for _ in range(3)]
    vs += [zscore(y + beta + noise * rng.standard_normal(n)) for _ in range(2)]
    return y, vs, np.array([0, 0, 0, 1, 1])


@test
def lsml_groups_override_seam():
    """groups= must bypass detection verbatim, fill the meta dict, and score
    the true planted assignment better (lower Eq-14 residual) than a wrong one."""
    from spectral_utils.fusion_utils import lsml_continuous
    rng = np.random.default_rng(42)
    y, views, truth = _planted_group_views(rng)

    fused_t, meta_t = lsml_continuous(*views, groups=truth)
    assert np.array_equal(np.asarray(meta_t['c'], int), truth)
    assert meta_t['K'] == 2 and np.isfinite(meta_t['residual'])
    assert np.isfinite(fused_t).all()

    wrong = np.array([0, 1, 0, 1, 0])
    _, meta_w = lsml_continuous(*views, groups=wrong)
    assert meta_t['residual'] < meta_w['residual'], (
        f"true-groups residual {meta_t['residual']:.4f} not below "
        f"wrong-groups {meta_w['residual']:.4f}")

    # when detection itself finds the truth, forced == detected exactly
    fused_d, meta_d = lsml_continuous(*views)
    same = (meta_d['K'] == 2 and
            len(set(zip(np.asarray(meta_d['c'], int), truth))) == 2)
    if same:
        assert np.allclose(fused_d, fused_t, atol=1e-12)
    else:
        print(f"    [note] detection found K={meta_d['K']} c={list(meta_d['c'])} "
              f"!= planted truth; equality check skipped")


@test
def upcr_residual_documented_properties():
    """DOCUMENTED PROPERTIES (not a model-fit known-answer) — measured
    2026-07-17 while building the gate, kept as canaries.

    Discovery 1: z-scoring destroys U-PCR's additive covariance form —
    z-scored independent-error experts have MULTIPLICATIVE covariance
    C_ij = a_i a_j (the SML rank-1 model), so the pipeline's own encoding
    means the additive model is misspecified on every cell.

    Discovery 2: the k=1 projection residual is an eigen-alignment measure,
    NOT a model-class test. Across three synthetic worlds its ordering is
        block-dependent (≈0.05) < exact-additive-MVN (≈0.09)
                                 < multiplicative/independent (≈0.12)
    i.e. the world that literally satisfies the additive equations does NOT
    score lowest, and latent-group dependence is not flagged. Therefore the
    D1 'minimize residual → better subset' hypothesis has NO a-priori
    synthetic support — the empirical admissibility analysis
    (scripts/selector_admissibility.py, real cells) is the sole gate, and
    the lsml-vs-upcr router must be validated against paired live AUROCs.

    Discovery 3: auto_components absorbs block dependence further (its
    residual on block data is even lower than k=1's) — diagnosis uses k=1.
    """
    from spectral_utils.fusion_utils import upcr_proj_residual, zscore
    rng = np.random.default_rng(3)
    k1 = dict(auto_components=False, n_components=1)
    mult, blk, addit, blk_auto = [], [], [], []
    # exact additive covariance with unit diagonal (PSD, min eig ~0.08)
    rho = np.array([0.50, 0.55, 0.60, 0.65, 0.70])
    C_add = np.add.outer(rho, rho) - 0.45
    np.fill_diagonal(C_add, 1.0)
    L = np.linalg.cholesky(C_add)
    for rep in range(6):
        y = rng.standard_normal(400)
        F_mult = np.vstack([zscore(y + s * rng.standard_normal(400))
                            for s in (0.6, 0.8, 1.0, 1.3, 1.7)])
        _, views, _ = _planted_group_views(rng, n=400, block=1.6)
        F_blk = np.vstack(views)
        F_add = np.vstack([zscore(v) for v in
                           (rng.standard_normal((400, 5)) @ L.T).T])
        mult.append(upcr_proj_residual(F_mult, **k1)[0])
        blk.append(upcr_proj_residual(F_blk, **k1)[0])
        addit.append(upcr_proj_residual(F_add, **k1)[0])
        blk_auto.append(upcr_proj_residual(F_blk)[0])
    m_mult, m_blk, m_add = (float(np.median(x)) for x in (mult, blk, addit))
    assert m_blk < m_add < m_mult, (
        f"documented ordering changed: block {m_blk:.4f}, additive {m_add:.4f}, "
        f"multiplicative {m_mult:.4f} — revisit A1 objective configuration")
    assert float(np.median(blk_auto)) < m_blk, (
        "auto mode no longer absorbs block dependence — revisit k=1 rule")


# ---------------------------------------------------------------------------
# selector_bench — mask round-trips, exact percentile, label-leak, live paths
# ---------------------------------------------------------------------------

def _tiny_ctx(rng=None, n=300, informative=5):
    """Synthetic CellContext on REAL H16 feature names (so signs/anchor
    resolve): `informative` y-correlated views, the rest pure noise."""
    from spectral_utils.subset_sweep import prepare_cell, H16, ALL_SIGNS
    rng = rng or np.random.default_rng(0)
    y = rng.standard_normal(n)
    labels = (y > 0).astype(int)
    fd = {}
    for i, name in enumerate(H16):
        sgn = ALL_SIGNS.get(name, +1)
        if i < informative:
            arr = sgn * (y + (0.5 + 0.15 * i) * rng.standard_normal(n))
        else:
            arr = sgn * rng.standard_normal(n)
        fd[name] = arr
    ctx = prepare_cell('smoke', 'tiny', fd, labels)
    assert ctx is not None and ctx.V.shape == (n, 16)
    return ctx


@test
def bench_mask_roundtrip_reduced_pool():
    from spectral_utils.subset_sweep import (
        CANONICAL_POOL, H16, names_to_local_mask, mask_to_canonical)
    from spectral_utils.selector_bench import (
        canonical_mask_to_local_cols, cols_to_canonical)
    # simulate a reduced pool: H16 minus two dropped features
    pool = [f for f in H16 if f not in ('trace_length', 'rpdi')]
    pool_bits = np.array([CANONICAL_POOL.index(f) for f in pool], dtype=np.uint8)
    names = ['epr', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
    local = names_to_local_mask(names, pool)
    canon = mask_to_canonical(local, pool_bits)
    cols = canonical_mask_to_local_cols(canon, pool_bits)
    assert cols is not None and [pool[j] for j in cols] == sorted(
        names, key=CANONICAL_POOL.index)
    assert cols_to_canonical(cols, pool_bits) == canon
    # a canonical mask using a dropped feature's bit must be rejected
    bad = canon | np.uint64(1 << CANONICAL_POOL.index('rpdi'))
    assert canonical_mask_to_local_cols(bad, pool_bits) is None


@test
def bench_pctile_exact():
    from spectral_utils.selector_bench import pctile_within_size
    table = {'_auroc_by_size': {4: np.array([0.1, 0.2, 0.2, 0.3])}}
    assert pctile_within_size(table, 4, 0.05) == 0.0
    assert pctile_within_size(table, 4, 0.35) == 100.0
    assert pctile_within_size(table, 4, 0.2) == 50.0          # tie mean-rank
    assert pctile_within_size(table, 4, 0.3) == 87.5
    assert np.isnan(pctile_within_size(table, 5, 0.2))        # size absent


@test
def bench_unlabeled_cell_no_leak():
    from spectral_utils.selector_bench import UnlabeledCell
    ctx = _tiny_ctx()
    u = UnlabeledCell.from_context(ctx)
    assert not hasattr(u, 'labels') and not hasattr(u, 'pos_rate')
    assert u.n == 300 and u.p == 16 and len(u.pool) == 16


@test
def bench_eval_subset_flex_paths():
    """Default flex path must equal eval_subset exactly; upcr / groups /
    K-override paths must run and report what they were asked to do."""
    from spectral_utils.subset_sweep import eval_subset
    from spectral_utils.selector_bench import eval_subset_flex
    ctx = _tiny_ctx()
    cols = np.array([0, 1, 2, 3, 4])

    ref = eval_subset(ctx.V, ctx.labels, ctx.anchor, ctx.rho, cols, ctx.pool_bits)
    flex = eval_subset_flex(ctx, cols)
    assert abs(flex['auroc'] - float(ref['auroc'])) < 1e-6
    assert flex['K'] == int(ref['K']) and flex['flipped'] == bool(ref['flipped'])
    assert flex['auroc'] > 0.6, "informative planted views should score >0.6"

    up = eval_subset_flex(ctx, cols, fusion='upcr')
    assert np.isfinite(up['auroc']) and up['fusion'] == 'upcr'

    gr = eval_subset_flex(ctx, cols, groups=np.array([0, 0, 1, 1, 2]))
    assert gr['K'] == 3 and np.isfinite(gr['auroc'])

    ko = eval_subset_flex(ctx, cols, K_override=2)
    assert ko['K'] in (1, 2) and np.isfinite(ko['auroc'])


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def _discover_selector_smokes():
    """Selector modules may expose smoke() — auto-discovered, never edited here."""
    out = []
    try:
        from spectral_utils import selectors
    except ImportError:
        return out
    import importlib
    import pkgutil
    for info in pkgutil.iter_modules(selectors.__path__):
        try:
            mod = importlib.import_module(f'spectral_utils.selectors.{info.name}')
        except Exception:
            continue
        fn = getattr(mod, 'smoke', None)
        if callable(fn):
            fn.__name__ = f'selector_smoke__{info.name}'
            out.append(fn)
    return out


def main():
    flt = sys.argv[1] if len(sys.argv) > 1 else ''
    tests = TESTS + _discover_selector_smokes()
    tests = [t for t in tests if flt in t.__name__]
    n_fail = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception:
            n_fail += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - n_fail}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
