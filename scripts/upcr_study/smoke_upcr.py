"""
smoke_upcr.py — unit gates for the faithful U-PCR module (Phase A).

Follows the `scripts/smoke_selectors.py` convention: fast, no GPU, run before any
experiment. A failure here voids everything downstream.

  U0  the Eq.16 shift identity holds for BOTH losses -- this is what lets the g2
      grid be solved once instead of 300 times, so it must be exact, not close.
  U1  moment-matching to E[Y], Var(Y) == z-scoring with scale_ratio=1.
      Proves Assumption A1 is free for us: the accuracy rate p cancels.
  U2  exclusion=False really keeps every feature (the legacy min_frac=0 /
      exclude_frac=inf spelling does NOT -- it collapses to rho >= 0).
  U3  g2_projection_k=1 projects onto v1 only, even when 2 weight components fire.
  U4  legacy-flag settings reproduce fusion_utils.upcr_fuse exactly on real cells.
  U5  the project anti-regression anchor, GOOD_6 = 0.7594, still holds.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spectral_utils.upcr import (                                  # noqa: E402
    upcr_fit, additive_design, solve_additive, moment_match,
)
from spectral_utils.fusion_utils import upcr_fuse                  # noqa: E402

LEGACY = dict(var_y=0.25, loss="l2", g2_projection_k="match", exclusion=True,
              min_frac=0.05, exclude_frac=3.0, simple_avg_fallback=False,
              difficulty_gate=False, recompute_after_exclusion=False)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    return ok


def rand_F(m=12, n=200, seed=0, blocks=True):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)
    F = []
    for i in range(m):
        base = g if not blocks else g * (0.8 if i < m // 2 else 0.4)
        F.append(base + rng.standard_normal(n) * 0.6)
    F = np.asarray(F)
    return (F - F.mean(1, keepdims=True)) / F.std(1, keepdims=True)


def u0_shift_identity():
    print("\nU0 - Eq.16 shift identity rho(q) = rho(0) + q/2, for l2 AND l1")
    ok = True
    for seed in (0, 1, 2):
        F = rand_F(seed=seed)
        m, n = F.shape
        C = (F @ F.T) / n
        A, prs = additive_design(m)
        b = np.array([C[i, j] for i, j in prs], float)
        for loss in ("l2", "l1"):
            r0 = solve_additive(A, b, loss=loss)
            for q in (0.05, 0.37, 1.0):
                rq = solve_additive(A, b + q, loss=loss)
                dev = np.abs(rq - (r0 + 0.5 * q)).max()
                # l1 LPs have non-unique optima; compare the RESIDUAL, which is
                # what the identity actually claims is invariant
                res_dev = np.abs((A @ rq - (b + q)) - (A @ r0 - b)).max()
                ok &= (dev < 1e-6) or (res_dev < 1e-6)
    return check("U0 shift identity exact for l2 and l1", ok)


def u1_moment_match():
    print("\nU1 - moment-match(E[Y],Var(Y)) == z-score with scale_ratio=1")
    ok = True
    worst = 0.0
    for seed in (0, 1, 2):
        F = rand_F(seed=seed)
        for p in (0.3, 0.5, 0.72):
            a = upcr_fit(F, scale_ratio=1.0)
            # moment_match adds a constant offset; U-PCR mean-centres via C, so
            # feed the centred version -- the identity is about SCALE.
            Fm = moment_match(F, p)
            Fm = Fm - Fm.mean(1, keepdims=True)
            b = upcr_fit(Fm, var_y=p * (1 - p))
            # weights are scale-covariant; compare the induced SCORE ranking
            sa, sb = a.w @ F, b.w @ Fm
            if sa.std() > 1e-12 and sb.std() > 1e-12:
                corr = float(np.corrcoef(sa, sb)[0, 1])
                worst = max(worst, abs(1.0 - corr))
                ok &= corr > 1 - 1e-9
            ok &= abs(a.g2_frac_of_var_y - b.g2_frac_of_var_y) < 1e-9
            ok &= bool((a.keep == b.keep).all())
    return check("U1 A1 is free -- p cancels out entirely", ok,
                 f"worst |1-corr| = {worst:.2e}")


def u2_exclusion_off(cells):
    """Also pins the finding that the OBVIOUS way to disable exclusion does not
    work: `min_frac=0, exclude_frac=inf` collapses to `rho >= 0`, and on real
    cells 7-14 features per cell have negative estimated rho. An arm labelled
    'all features, no selection' spelled that way is silently selecting."""
    print("\nU2 - exclusion=False keeps every feature (legacy spelling does not)")
    ok = True
    silently_dropped = 0
    for ck, cd in cells.items():
        F = cd["V"].T
        m = F.shape[0]
        ok &= int(upcr_fit(F, exclusion=False).keep.sum()) == m
        _, rho_legacy, _ = upcr_fuse(F, var_y=0.25, min_frac=0.0,
                                     exclude_frac=np.inf)
        silently_dropped += m - int((rho_legacy != 0).sum())
    ok &= silently_dropped > 0        # the trap is real, not hypothetical
    return check("U2 exclusion=False keeps all m", ok,
                 f"legacy min_frac=0/exclude_frac=inf silently drops "
                 f"{silently_dropped} features across {len(cells)} cells")


def u3_projection_k():
    print("\nU3 - g2_projection_k=1 uses v1 only, even at 2 weight components")
    ok = True
    for seed in (0, 1, 2):
        F = rand_F(seed=seed)
        one = upcr_fit(F, g2_projection_k=1, auto_components=True)
        match = upcr_fit(F, g2_projection_k="match", auto_components=True)
        if one.n_components_used == 2:
            # with 2 weight components the two spellings must diverge; if they
            # never do, the flag is not wired through
            ok &= abs(one.g2_hat - match.g2_hat) > 0 or abs(
                one.proj_residual - match.proj_residual) > 0
    return check("U3 g2 projection independent of the weight count", ok)


def u4_legacy_reproduction(cells):
    print("\nU4 - LEGACY flags reproduce fusion_utils.upcr_fuse on real cells")
    worst_w = worst_g2 = 0.0
    for ck, cd in cells.items():
        F = cd["V"].T                       # (m, n)
        w_old, rho_old, g2_old = upcr_fuse(F, var_y=0.25)
        new = upcr_fit(F, **LEGACY)
        worst_w = max(worst_w, float(np.abs(new.w - w_old).max()))
        worst_g2 = max(worst_g2, abs(new.g2_hat - g2_old))
    return check("U4 legacy path is bit-compatible", worst_w < 1e-8 and worst_g2 < 1e-8,
                 f"max |dw| = {worst_w:.2e}, max |dg2| = {worst_g2:.2e}")


def u5_anchor(cells):
    print("\nU5 - project anti-regression anchor")
    from inscope_bench_common import assert_good6
    # assert_good6 returns the TUPLE (ok, macro) -- bool(tuple) is ALWAYS True,
    # so `bool(assert_good6(...))` was a gate that could never fail. Unpack it.
    ok, macro = assert_good6(cells, verbose=False)
    return check("U5 GOOD_6 == 0.7594", ok, f"macro = {macro:.4f}")


def main():
    print("=" * 70)
    print("smoke_upcr.py - Phase A unit gates")
    print("=" * 70)
    u0_shift_identity()
    u1_moment_match()
    u3_projection_k()

    from inscope_bench_common import load_cells
    cells = load_cells()
    u2_exclusion_off(cells)
    u4_legacy_reproduction(cells)
    u5_anchor(cells)

    print("\n" + "=" * 70)
    n_ok = sum(1 for _, ok in RESULTS if ok)
    for name, ok in RESULTS:
        if not ok:
            print(f"  FAILED: {name}")
    print(f"{n_ok}/{len(RESULTS)} gates passed")
    return 0 if n_ok == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
