"""Unit tests for spectral_utils.composite_reliability — run BEFORE exp15.

Every check here is one the Step-223 plan named in its verification section. The
duplicate-degeneracy test is the important one: it asserts that the degeneracy is
REAL under a plain factor fit and that the sparsity prior is what removes it. If that
test ever starts passing trivially, the objective has stopped doing what it claims.

    python scripts/test_composite_reliability.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral_utils.composite_reliability import (      # noqa: E402
    fit_sparse_factor, delta_structure, greedy_backward,
    omega_sparse, resid_dep, m_eff, cohesion_set, loading_sum, OBJECTIVES,
)

FAILED = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def corr_from_model(lam, delta, n=None, seed=0):
    """Population correlation matrix for C = lam lam' + Psi + Delta, optionally with
    sampling noise from n rows."""
    m = len(lam)
    C = np.outer(lam, lam) + delta
    np.fill_diagonal(C, 1.0)
    if n is None:
        return C
    ev, evec = np.linalg.eigh(C)
    ev = np.clip(ev, 1e-8, None)
    L = evec @ np.diag(np.sqrt(ev))
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, m)) @ L.T
    X = (X - X.mean(0)) / X.std(0)
    return (X.T @ X) / n


# ---------------------------------------------------------------------------
def test_plain_rank1_recovery():
    """With Delta = 0 the fit must recover the planted loadings."""
    print("\n1. plain rank-1 recovery (Delta = 0)")
    for m in (12, 21, 28):
        rng = np.random.default_rng(m)
        lam = rng.uniform(0.35, 0.8, m)
        C = corr_from_model(lam, np.zeros((m, m)))
        fit = fit_sparse_factor(C, sparsity=0.0)
        err = float(np.max(np.abs(fit.lam - lam)))
        check(f"m={m}: max |lambda_hat - lambda| < 0.02", err < 0.02, f"{err:.5f}")


def test_planted_delta_support():
    """The plan's requirement: recover a PLANTED sparse Delta's support, at our real
    m values and at n matched to the smallest in-scope cell (198)."""
    print("\n2. planted sparse-Delta support recovery")
    for m in (12, 21, 28):
        for n in (198, 2000, None):
            rng = np.random.default_rng(1000 + m)
            lam = rng.uniform(0.35, 0.75, m)
            n_pairs = max(1, int(round(0.06 * m * (m - 1) / 2)))
            iu = np.triu_indices(m, 1)
            pick = rng.choice(len(iu[0]), n_pairs, replace=False)
            D = np.zeros((m, m))
            for p in pick:
                i, j = iu[0][p], iu[1][p]
                D[i, j] = D[j, i] = rng.uniform(0.25, 0.4) * rng.choice([-1.0, 1.0])
            planted = {(int(iu[0][p]), int(iu[1][p])) for p in pick}
            C = corr_from_model(lam, D, n=n, seed=m + (n or 0))
            fit = fit_sparse_factor(C, sparsity=0.10)
            got = np.abs(fit.delta[iu])
            top = set()
            for p in np.argsort(got)[::-1][:n_pairs]:
                top.add((int(iu[0][p]), int(iu[1][p])))
            rec = len(planted & top) / len(planted)
            lab = f"m={m}, n={'inf' if n is None else n}"
            check(f"{lab}: planted-pair recall >= 0.6", rec >= 0.6, f"recall {rec:.2f}")


def _block_dup(lam, block):
    """m views of which `block` are EXACT copies of view 0."""
    m = len(lam)
    n_real = m - block + 1
    base = corr_from_model(lam[:n_real], np.zeros((n_real, n_real)))
    C = np.eye(m)
    C[:n_real, :n_real] = base
    for i in range(n_real, m):                       # copies of view 0
        for j in range(m):
            v = 1.0 if (j >= n_real or j == 0) else base[0, j]
            C[i, j] = C[j, i] = v
    np.fill_diagonal(C, 1.0)
    return C


def test_duplicate_degeneracy():
    """THE REGISTERED DEGENERACY, and the BOUNDARY of the fix.

    Plain omega is maximised by exact duplicates. The sparsity prior breaks the tie
    only while the duplicate block fits the budget: a block of b copies contributes
    C(b,2) dependent pairs against a budget of sparsity*C(m,2). Past that the fit
    credits the block to lambda and omega is fooled again. That boundary is a real
    limitation of M1 and is measured here rather than asserted away.
    """
    print("\n3. duplicate degeneracy, and the boundary of the sparsity fix")
    m = 12
    rng = np.random.default_rng(7)
    lam = rng.uniform(0.4, 0.7, m)
    C_indep = corr_from_model(lam, np.zeros((m, m)))
    fi = fit_sparse_factor(C_indep, sparsity=0.10)
    om_honest = omega_sparse(C_indep, fi)

    # the PURE degeneracy: every view identical -> omega is exactly 1
    C_all = np.ones((m, m))
    f_all = fit_sparse_factor(C_all, sparsity=0.0)
    om_all = omega_sparse(C_all, f_all)
    check("plain fit: m identical views give omega = 1.0 (the degeneracy is REAL)",
          abs(om_all - 1.0) < 1e-6, f"omega={om_all:.6f}")

    C3 = _block_dup(lam, 3)
    om3_plain = omega_sparse(C3, fit_sparse_factor(C3, sparsity=0.0))
    om3_sparse = omega_sparse(C3, fit_sparse_factor(C3, sparsity=0.10))
    check("sparse fit LOWERS omega on a small duplicate block",
          om3_sparse < om3_plain, f"{om3_plain:.4f} -> {om3_sparse:.4f}")
    check("a 3-copy block is penalised BELOW the honest set of the same size",
          om3_sparse < om_honest, f"dup {om3_sparse:.4f} vs honest {om_honest:.4f}")

    # where does the fix stop working? reported, not asserted — this is the number
    # the pre-registration quotes as M1's named failure mode.
    budget = int(np.floor(0.10 * m * (m - 1) / 2))
    print(f"       sparsity budget at m={m}, 10%: {budget} pairs")
    for b in (2, 3, 4, 5, 6, 7):
        Cb = _block_dup(lam, b)
        omb = omega_sparse(Cb, fit_sparse_factor(Cb, sparsity=0.10))
        verdict = "penalised" if omb < om_honest else "FOOLED"
        print(f"       block={b} ({b*(b-1)//2:2d} dependent pairs): "
              f"omega={omb:.4f} vs honest {om_honest:.4f}  -> {verdict}")


def test_omega_closed_form():
    """omega = (sum lambda)^2 / (1' C 1) must equal the direct arithmetic."""
    print("\n4. omega against its closed form")
    rng = np.random.default_rng(3)
    m = 10
    lam = rng.uniform(0.3, 0.8, m)
    C = corr_from_model(lam, np.zeros((m, m)))
    fit = fit_sparse_factor(C, sparsity=0.0)
    want = float(np.sum(fit.lam) ** 2 / np.sum(C))
    got = omega_sparse(C, fit)
    check("matches (sum lambda)^2 / sum(C)", abs(got - want) < 1e-12,
          f"{got:.10f}")
    # equal loadings + zero error correlation is exactly Cronbach's alpha
    lam_eq = np.full(m, 0.6)
    C_eq = corr_from_model(lam_eq, np.zeros((m, m)))
    rbar = float(np.mean(C_eq[np.triu_indices(m, 1)]))
    alpha = m * rbar / (1 + (m - 1) * rbar)
    fit_eq = fit_sparse_factor(C_eq, sparsity=0.0)
    om = omega_sparse(C_eq, fit_eq)
    check("equals Cronbach's alpha under equal loadings, zero Delta",
          abs(om - alpha) < 1e-6, f"omega={om:.6f} alpha={alpha:.6f}")


def test_delta_structure_discriminates():
    """The diagnostic must tell SPARSE from LOW-RANK — the prior rests on it."""
    print("\n5. delta_structure separates sparse from low-rank")
    m = 20
    rng = np.random.default_rng(11)
    lam = rng.uniform(0.4, 0.7, m)

    D_sparse = np.zeros((m, m))
    iu = np.triu_indices(m, 1)
    for p in rng.choice(len(iu[0]), 10, replace=False):
        i, j = iu[0][p], iu[1][p]
        D_sparse[i, j] = D_sparse[j, i] = 0.35
    f_s = fit_sparse_factor(corr_from_model(lam, D_sparse), sparsity=0.10)
    top_s, r1_s = delta_structure(f_s)

    u = rng.standard_normal(m) * 0.3          # a genuine SECOND factor
    D_lr = np.outer(u, u)
    np.fill_diagonal(D_lr, 0.0)
    f_l = fit_sparse_factor(corr_from_model(lam, D_lr), sparsity=0.50)
    top_l, r1_l = delta_structure(f_l)

    check("sparse Delta -> higher top-decile mass share", top_s > top_l,
          f"sparse {top_s:.3f} vs low-rank {top_l:.3f}")
    check("low-rank Delta -> higher leading-eigenvalue share", r1_l > r1_s,
          f"low-rank {r1_l:.3f} vs sparse {r1_s:.3f}")


def test_greedy_behaviour():
    """Free-size must drop a planted duplicate; fixed-k must land on exactly k; the
    search must not reach below MIN_SET."""
    print("\n6. greedy search behaviour")
    m = 10
    rng = np.random.default_rng(5)
    lam = rng.uniform(0.45, 0.7, m)
    C = corr_from_model(lam, np.zeros((m, m)))
    C = np.vstack([np.hstack([C, C[:, [0]]]), np.append(C[0, :], 1.0)[None, :]])
    dup_idx = m                                   # exact copy of view 0

    fn, self_sizing, needs = OBJECTIVES["omega_sparse"]
    sel, _, steps = greedy_backward(C, list(range(m + 1)), fn, target_k=None,
                                    needs_fit=needs)
    check("free-size drops the exact duplicate", dup_idx not in sel or 0 not in sel,
          f"kept {len(sel)} of {m+1}, steps={steps}")

    for k in (4, 6, 8):
        sel_k, _, _ = greedy_backward(C, list(range(m + 1)), fn, target_k=k,
                                      needs_fit=needs)
        check(f"fixed-k lands on exactly k={k}", len(sel_k) == k, f"got {len(sel_k)}")

    sel_lo, _, _ = greedy_backward(C, list(range(m + 1)), fn, target_k=1,
                                   needs_fit=needs)
    check("never goes below MIN_SET=3", len(sel_lo) >= 3, f"got {len(sel_lo)}")


def test_directions():
    """Every objective is MAXIMISED — the LOWER-is-better ones must be negated inside
    the module, not by the caller. Step 222's lesson about free directions."""
    print("\n7. all five objectives point the same way (higher = keep)")
    m = 12
    rng = np.random.default_rng(13)
    lam = rng.uniform(0.4, 0.7, m)
    D = np.zeros((m, m))
    D[0, 1] = D[1, 0] = 0.45                       # one strongly dependent pair
    C = corr_from_model(lam, D)
    fit = fit_sparse_factor(C, sparsity=0.10)
    C_clean = corr_from_model(lam, np.zeros((m, m)))
    fit_clean = fit_sparse_factor(C_clean, sparsity=0.10)
    for nm, fn in (("resid_dep", resid_dep), ("m_eff", m_eff)):
        a, b = fn(C, fit), fn(C_clean, fit_clean)
        check(f"{nm}: the contaminated set scores LOWER", b > a, f"{a:.4f} < {b:.4f}")
    hi = corr_from_model(np.full(m, 0.8), np.zeros((m, m)))
    lo = corr_from_model(np.full(m, 0.3), np.zeros((m, m)))
    f_hi = fit_sparse_factor(hi, sparsity=0.10)
    f_lo = fit_sparse_factor(lo, sparsity=0.10)
    check("loading_sum: stronger loadings score HIGHER",
          loading_sum(hi, f_hi) > loading_sum(lo, f_lo))
    check("cohesion_set: the more correlated set scores LOWER",
          cohesion_set(lo, None) > cohesion_set(hi, None))


def main():
    print("=" * 74)
    print("UNIT TESTS — spectral_utils/composite_reliability.py")
    print("=" * 74)
    test_plain_rank1_recovery()
    test_planted_delta_support()
    test_duplicate_degeneracy()
    test_omega_closed_form()
    test_delta_structure_discriminates()
    test_greedy_behaviour()
    test_directions()
    print("\n" + "=" * 74)
    if FAILED:
        print(f"{len(FAILED)} FAILED: " + ", ".join(FAILED))
        sys.exit(1)
    print("ALL PASSED")


if __name__ == "__main__":
    main()
