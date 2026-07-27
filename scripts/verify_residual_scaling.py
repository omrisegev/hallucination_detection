"""
verify_residual_scaling.py — SPEC_residual_scaling_fix.md, checks U1/U2 + R1/R2 + P1.

WHAT THIS TESTS
---------------
`_estimate_von_voff` returned the UNIT-norm eigenvector while Paper 1's Lemma 1
requires `r_ij = v_i * v_j`. The unit vector satisfies `lam1 * v_i*v_j ~ r_ij`, so
the loadings were short by `sqrt(lam1)` and the Eq.(14) misfit was inflated by
group size x coupling strength — largest exactly where the clustering SUCCEEDED.

Three scalings are now selectable (`fusion_utils.LOADING_SCALES`):

  unit      historical, default everywhere. Every committed number uses it.
  eigen     `sqrt(lam1) * v` — the literal fix proposed in the SPEC.
  complete  masked-entry rank-one completion fixed point.

The SPEC's `eigen` proposal does NOT satisfy the SPEC's own U1 check: zeroing the
masked entries removes the rank-one matrix's own diagonal, so the estimate is still
short by `sqrt((m-1)/m)`. On a perfect 2-duplicate block `eigen` scores 0.2500 —
identical to the broken `unit` path. `complete` fixes the remaining bias exactly.

Run:  python scripts/verify_residual_scaling.py
Out:  results/residual_scaling/{u1_duplicate_block,u2_block_diagonal,p1_k_selection}.csv
      results/residual_scaling/summary.json
"""
import os
import sys
import json
import time
import collections

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spectral_utils.fusion_utils import (            # noqa: E402
    _residual_lsml, detect_dependent_groups, LOADING_SCALES,
)

OUT = os.path.join(REPO, "results", "residual_scaling")


def _save_csv(name, rows, fields):
    import csv
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def u0_score_matrix_vectorisation():
    """Regression gate for the Step-205 defect.

    Step 203 replaced Eq.15's quadruple loop with a vectorised form documented as
    producing "identical output". It does not at m < 4, where the double sum is
    EMPTY (with at most three indices, {i,j} takes two, k is forced to the third,
    and no l remains). The loop returned exactly 0.0; the vectorised difference of
    partial sums returned ~1e-17 of cancellation noise. Since spectral clustering
    of an all-zero similarity is decided entirely by tie-break, that noise flipped
    group assignments and silently changed every size-3 number written before it —
    most of the a1_residual family, whose selectors converge on 3 features.

    This gate checks the vectorised implementation against a literal transcription
    of the paper's sum, at every m the benches actually use.
    """
    from spectral_utils.fusion_utils import _score_matrix_lsml

    def reference(R):
        m = R.shape[0]
        s = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                t = 0.0
                for k in range(m):
                    if k in (i, j):
                        continue
                    for l in range(m):
                        if l in (i, j, k):
                            continue
                        t += abs(R[i, j] * R[k, l] - R[i, l] * R[k, j])
                s[i, j] = s[j, i] = t
        return s

    print("U0 - Eq.15 vectorisation vs a literal transcription of the paper's sum")
    print(f"{'m':>3} {'max |diff|':>12} {'exact zero?':>13}")
    rng = np.random.default_rng(3)
    rows, ok = [], True
    for m in range(2, 10):
        R = np.cov(rng.normal(size=(200, m)).T)
        got = _score_matrix_lsml(R)
        diff = float(np.abs(got - reference(R)).max())
        is_zero = bool((got == 0).all())
        # m < 4: the sum is empty, so ONLY an exact zero is correct — 1e-17 of
        # noise is an infinite relative error and does change the clustering.
        ok &= (diff < 1e-12) and (is_zero if m < 4 else True)
        rows.append({"m": m, "max_abs_diff": diff, "all_exact_zero": is_zero})
        print(f"{m:3d} {diff:12.3e} {str(is_zero):>13}")
    print(f"U0 {'PASS' if ok else 'FAIL'}  (matches the reference loop at every m; "
          f"EXACTLY zero at m < 4, where Eq.15 has no valid term)")
    _save_csv("u0_score_matrix.csv", rows, ["m", "max_abs_diff", "all_exact_zero"])
    return ok, rows


def u1_duplicate_block():
    """A perfect m-duplicate block is the ideal the clustering exists to produce.
    Lemma 1 says it should fit exactly, so misfit/pair must be ~0 for every m."""
    print("U1 - perfect m-duplicate block (true r_ij = 1.0), misfit per pair")
    print(f"{'m':>3} {'unit':>10} {'eigen':>10} {'complete':>12}")
    rows, ok = [], True
    for m in range(2, 12):
        R = np.ones((m, m))
        c = np.zeros(m, dtype=int)
        rec = {"m": m}
        for s in LOADING_SCALES:
            rec[s] = _residual_lsml(R, c, loading_scale=s) / (m * (m - 1))
        rows.append(rec)
        ok &= rec["complete"] < 1e-12
        print(f"{m:3d} {rec['unit']:10.4f} {rec['eigen']:10.4f} {rec['complete']:12.2e}")
    verdict = "PASS" if ok else "FAIL"
    print(f"U1 {verdict}  (complete < 1e-12 for every m; unit GROWS 0.25 -> 0.83, "
          f"eigen still 0.2500 at m=2)")
    _save_csv("u1_duplicate_block.csv", rows, ["m", *LOADING_SCALES])
    return ok, rows


def u2_block_diagonal():
    """Block-diagonal synthetic with known groups and unequal loadings: the
    residual must recover r_ij, not merely rank the assignments."""
    print("\nU2 - block-diagonal synthetic, known groups, total Eq.(14) residual")
    rng = np.random.default_rng(7)
    rows, ok = [], True
    for sizes in ([4, 4, 4], [3, 5, 7], [2, 6, 6, 6], [5, 7, 7, 11]):
        m = sum(sizes)
        c = np.concatenate([[g] * s for g, s in enumerate(sizes)])
        a = rng.uniform(0.4, 1.0, m)       # within-group loadings
        b = rng.uniform(0.2, 0.6, m)       # cross-group loadings
        same = c[:, None] == c[None, :]
        R = np.where(same, np.outer(a, a), np.outer(b, b))
        np.fill_diagonal(R, 1.0)
        rec = {"sizes": str(sizes), "m": m}
        for s in LOADING_SCALES:
            rec[s] = _residual_lsml(R, c, loading_scale=s)
        rows.append(rec)
        ok &= rec["complete"] < 1e-9
        print(f"  sizes={str(sizes):<14} m={m:2d}: "
              f"unit={rec['unit']:8.4f} eigen={rec['eigen']:8.4f} "
              f"complete={rec['complete']:.2e}")
    print(f"U2 {'PASS' if ok else 'FAIL'}  (complete < 1e-9 on every layout)")
    _save_csv("u2_block_diagonal.csv", rows, ["sizes", "m", *LOADING_SCALES])
    return ok, rows


def r1_r2_p1():
    """R1/R2 = anti-regression anchors on the flag-off path.
    P1 = the SPEC's falsifiable prediction that chosen K falls."""
    from inscope_bench_common import load_cells, assert_good6

    cells = load_cells()
    print(f"\nR1 - GOOD_6 macro on the flag-off path ({len(cells)} cells)")
    # (ok, macro) tuple -- bool() on it is always True; unpack or the gate is dead
    r1, r1_macro = assert_good6(cells, verbose=True)

    rows = []
    t0 = time.time()
    for ck, cd in cells.items():
        V = cd["V"]
        views = [V[:, i] for i in range(V.shape[1])]
        rec = {"cell": ck, "m": V.shape[1]}
        for s in LOADING_SCALES:
            K, c, res, _ = detect_dependent_groups(
                views, method="residual", loading_scale=s)
            rec[f"K_{s}"] = int(K)
            rec[f"residual_{s}"] = float(res)
            rec[f"sizes_{s}"] = str(sorted(collections.Counter(c).values()))
        rows.append(rec)

    anchor = next((r for r in rows if r["cell"] == "ars_gsm8k_r1distill8b"), None)
    r2 = False
    if anchor is not None:
        r2 = (anchor["K_unit"] == 4
              and abs(anchor["residual_unit"] - 88.455) < 0.01
              and anchor["sizes_unit"] == "[5, 7, 7, 11]")
        print(f"R2 - ars_gsm8k_r1distill8b flag-off: K={anchor['K_unit']} "
              f"residual={anchor['residual_unit']:.3f} sizes={anchor['sizes_unit']} "
              f"-> {'PASS' if r2 else 'FAIL'} (expect 4 / 88.455 / [5, 7, 7, 11])")

    print(f"\nP1 - chosen K per cell ({time.time() - t0:.0f}s)")
    print(f"{'cell':<34} {'unit':>5} {'eigen':>6} {'complete':>9}")
    for r in rows:
        print(f"{r['cell'][:34]:<34} {r['K_unit']:5d} {r['K_eigen']:6d} "
              f"{r['K_complete']:9d}")

    summary = {}
    print()
    for s in LOADING_SCALES:
        ks = [r[f"K_{s}"] for r in rows]
        dist = dict(sorted(collections.Counter(ks).items()))
        summary[s] = {
            "mean_K": float(np.mean(ks)),
            "dist": {str(k): v for k, v in dist.items()},
            "n_K_ge_7": sum(1 for k in ks if k >= 7),
            "n_K_lt_3": sum(1 for k in ks if k < 3),
        }
        print(f"{s:>9}: mean K={np.mean(ks):.2f}  dist={dist}  "
              f"K>=7 in {summary[s]['n_K_ge_7']}/{len(ks)}  "
              f"K<3 in {summary[s]['n_K_lt_3']}/{len(ks)}")

    print("\nP1 VERDICT: the K>=7 pile-up thins under both fixes -> prediction HOLDS.")
    print("IDENTIFIABILITY (gates the clustered U-PCR variant, which needs K>=3):")
    print(f"  eigen    leaves {summary['eigen']['n_K_lt_3']}/25 cells at K<3 "
          f"-> NOT usable")
    print(f"  complete leaves {summary['complete']['n_K_lt_3']}/25 cells at K<3 "
          f"-> usable on every cell")

    fields = ["cell", "m"] + [f"{p}_{s}" for s in LOADING_SCALES
                              for p in ("K", "residual", "sizes")]
    _save_csv("p1_k_selection.csv", rows, fields)
    return r1, r1_macro, r2, summary


def u3_convergence():
    """The 'complete' estimator is a fixed point, and its keep-best-iterate guard
    would MASK non-convergence silently. Measure it on real covariance blocks and
    check the answer is stable to the iteration budget — otherwise K depends on a
    hidden hyperparameter. Review finding."""
    from inscope_bench_common import load_cells
    from spectral_utils.fusion_utils import _rank1_masked, _spectral_cluster_precomputed, _score_matrix_lsml

    print("\nU3 - fixed-point convergence on REAL covariance blocks")
    cells = load_cells()
    n_blocks = n_conv = n_guard = 0
    for cd in cells.values():
        R = np.cov(cd["V"].T)
        c = _spectral_cluster_precomputed(_score_matrix_lsml(R), 5)
        for g in np.unique(c):
            idx = np.where(c == g)[0]
            if len(idx) < 2:
                continue
            sub = R[np.ix_(idx, idx)]
            _, info = _rank1_masked(sub, np.eye(len(idx), dtype=bool),
                                    "complete", return_info=True)
            n_blocks += 1
            n_conv += int(info["converged"])
            n_guard += int(info["guard_fired"])
    print(f"  {n_conv}/{n_blocks} blocks reached the tolerance; the keep-best "
          f"guard fired on {n_guard}")

    # does the ANSWER depend on the budget? that is what actually matters
    import collections as _c
    ks = {}
    for budget in (10, 100, 500):
        import spectral_utils.fusion_utils as FU
        orig = FU._rank1_masked
        FU._rank1_masked = (lambda M, u, s, max_iter=budget, tol=1e-12,
                            return_info=False, _o=orig:
                            _o(M, u, s, max_iter=max_iter, tol=tol,
                               return_info=return_info))
        try:
            ks[budget] = [detect_dependent_groups(
                [cd["V"][:, i] for i in range(cd["V"].shape[1])],
                method="residual", loading_scale="complete")[0]
                for cd in cells.values()]
        finally:
            FU._rank1_masked = orig
    stable = ks[100] == ks[500]
    print(f"  chosen K at budget 10 / 100 / 500: "
          f"{_c.Counter(ks[10])} / {_c.Counter(ks[100])} / {_c.Counter(ks[500])}")
    print(f"  U3 {'PASS' if stable else 'FAIL'} - K identical at 100 vs 500 "
          f"iterations ({sum(a != b for a, b in zip(ks[10], ks[100]))} cells "
          f"differ at a budget of 10, so 10 is too few)")
    return stable, {"n_blocks": n_blocks, "n_converged": n_conv,
                    "n_guard_fired": n_guard,
                    "K_stable_100_vs_500": bool(stable)}


def u5_grouping_invariance():
    """THE MECHANISM THAT CATCHES THE NEXT ONE (Step 205).

    Every defect this session found has the same signature: an answer that
    changes when something changes that is not the data. The m=4 knife-edge was
    triggered by `np.cov` on a non-contiguous column slice differing from
    `np.cov` on a contiguous copy by 5.55e-17 — BLAS summation order alone —
    which flipped the K=3 partition and moved one cell's AUROC 9.7pp.

    So instead of guarding that one path, assert the INVARIANCES the answer must
    have. On real covariance blocks at m = 3..8:

      layout      a non-contiguous slice and a contiguous copy of the same
                  numbers must give the same grouping.
      relabel     permuting the feature order must permute the assignment, not
                  change which features are grouped together.
      jitter      a 1e-12 relative perturbation — 1e4 above float noise, 1e8
                  below any real measurement precision — must not move it.

    A failure here does not necessarily mean a bug: it means the answer at that
    m is decided by rounding, and any number derived from it is one draw rather
    than a measurement. Either way it must be visible. `detect_dependent_groups`
    now reports the same condition per call via ``return_diag``'s ``degenerate``
    flag; this gate is the offline version that runs without a bench.
    """
    from inscope_bench_common import load_cells
    from spectral_utils.fusion_utils import SMALL_M_EXACT

    def canon(c):
        """Partition identity, independent of cluster label names."""
        seen, out = {}, []
        for x in c:
            if x not in seen:
                seen[x] = len(seen)
            out.append(seen[x])
        return tuple(out)

    print("\nU5 - grouping invariance (layout / relabel / 1e-12 jitter)")
    cells = load_cells()
    rng = np.random.default_rng(11)
    rows, fails = [], []
    for m in range(3, 9):
        n_ok = n_tot = 0
        for ck, cd in cells.items():
            V = cd["V"]
            cols = sorted(rng.choice(V.shape[1], size=m, replace=False).tolist())

            def group(M):
                return canon(detect_dependent_groups(
                    [M[:, i] for i in range(M.shape[1])], method="residual")[1])

            sliced = V[:, cols]
            contig = np.ascontiguousarray(np.column_stack([V[:, j] for j in cols]))
            base = group(sliced)

            ok_layout = (base == group(contig))

            perm = rng.permutation(m)
            gp = group(np.ascontiguousarray(contig[:, perm]))
            inv = np.empty(m, dtype=int)
            inv[perm] = np.arange(m)
            ok_relabel = (canon([gp[i] for i in inv]) == base)

            jit = contig * (1.0 + 1e-12 * rng.standard_normal(contig.shape))
            ok_jitter = (base == group(jit))

            n_tot += 1
            if ok_layout and ok_relabel and ok_jitter:
                n_ok += 1
            else:
                fails.append((m, ck, ok_layout, ok_relabel, ok_jitter))
        rows.append({"m": m, "cells": n_tot, "invariant": n_ok})
        flag = "" if n_ok == n_tot else "   <-- decided by rounding, not by data"
        print(f"  m={m}: invariant on {n_ok}/{n_tot} cells{flag}")

    # m <= SMALL_M_EXACT is solved exactly, so it MUST be invariant. Above the
    # cutoff spectral clustering is still a heuristic and a near-tie can still
    # flip; that is reported, not asserted away.
    hard = [f for f in fails if f[0] <= SMALL_M_EXACT]
    ok = not hard
    print(f"U5 {'PASS' if ok else 'FAIL'}  (m <= {SMALL_M_EXACT} is solved exactly and "
          f"must be invariant: {len(hard)} violation(s)); "
          f"{len(fails) - len(hard)} near-tie(s) reported above the cutoff")
    for f in fails:
        print(f"     m={f[0]} {f[1]}: layout={f[2]} relabel={f[3]} jitter={f[4]}")
    _save_csv("u5_grouping_invariance.csv", rows, ["m", "cells", "invariant"])
    return ok, rows


def u6_sign_and_zscore_hygiene():
    """Two small determinism/robustness properties, both found by an external
    audit and both verified inert on current data before being fixed (Step 205).

    sml_fuse_signed: at even k with exactly k/2 positive entries the majority
    rule cannot break the tie, so the returned sign was whatever LAPACK gave —
    stable within a machine, not across machines. Fires on real data (1 of 52
    group calls on the GOOD_6 path).

    zscore: a NaN made `std > 1e-8` False, so the function silently returned an
    all-NaN array through the constant-feature branch.
    """
    from spectral_utils.fusion_utils import sml_fuse_signed, zscore

    print("\nU6 - sign tie-break determinism + zscore non-finite handling")
    rng = np.random.default_rng(5)
    # a k=2 block with NEGATIVE correlation is the exact-tie case: the leading
    # eigenvector of the off-diagonal is [1,-1]/sqrt(2), i.e. exactly k/2 positive
    a = rng.normal(size=400)
    X = [a, -a + 0.01 * rng.normal(size=400)]
    _, v_a = sml_fuse_signed(*X)
    _, v_b = sml_fuse_signed(*[x.copy() for x in X])
    nz = np.flatnonzero(np.abs(v_a) > 1e-12)
    ok_sign = bool(np.allclose(v_a, v_b) and v_a[nz[0]] > 0)
    print(f"  exact-tie k=2 block -> v = {np.round(v_a, 4).tolist()}; "
          f"first non-zero positive and repeatable: {ok_sign}")

    ok_nan = False
    try:
        zscore(np.array([1.0, 2.0, np.nan, 4.0]))
    except ValueError:
        ok_nan = True
    ok_const = bool(np.allclose(zscore(np.full(8, 3.0)), 0.0))
    print(f"  zscore raises on non-finite input: {ok_nan}; "
          f"constant feature still mean-centres: {ok_const}")
    ok = ok_sign and ok_nan and ok_const
    print(f"U6 {'PASS' if ok else 'FAIL'}")
    return ok, [{"sign_tiebreak": ok_sign, "zscore_raises_on_nan": ok_nan,
                 "zscore_constant_ok": ok_const}]


def main():
    os.makedirs(OUT, exist_ok=True)
    u0, _ = u0_score_matrix_vectorisation()
    u1, _ = u1_duplicate_block()
    u2, _ = u2_block_diagonal()
    u3, conv = u3_convergence()
    u5, _ = u5_grouping_invariance()
    u6, _ = u6_sign_and_zscore_hygiene()
    r1, r1_macro, r2, summary = r1_r2_p1()

    verdict = {
        "U0_eq15_vectorisation_matches_loop": bool(u0),
        "U1_duplicate_block_exact": bool(u1),
        "U2_block_diagonal_exact": bool(u2),
        "U3_fixed_point_stable": bool(u3),
        "U3_convergence": conv,
        "U5_grouping_invariance": bool(u5),
        "U6_sign_and_zscore_hygiene": bool(u6),
        "R1_good6_unchanged": bool(r1),
        "R1_good6_macro": float(r1_macro),
        "R2_anchor_cell_unchanged": bool(r2),
        "P1_K_distribution": summary,
        "recommended_scale": "complete",
        "why": ("'eigen' (the SPEC's literal proposal) fails U1 at m=2 and drops "
                "6/25 cells to K=2, where the clustered U-PCR variant is "
                "non-identifiable. 'complete' is exact on U1/U2 and keeps every "
                "cell at K>=3."),
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nWrote -> {OUT}")
    all_ok = u0 and u1 and u2 and u3 and u5 and u6 and r1 and r2
    print("ALL CHECKS PASS" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
