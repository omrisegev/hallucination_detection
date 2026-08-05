"""Is a triplet-consistency score pointed at the good features, or away from them?

STATUS: EXPLORATORY. Not pre-registered, no multiplicity control, no floor. Its numbers are
quotable as orientation only, never in a results table. It is committed because it is the
starting point for the live line in `HANDOFF_FEATURE_SELECTION_AND_FUSE.md` section 4.2 —
extend it, do not rewrite it. Requires the per-cell feature pkls, which are NOT in the repo.

WHAT IT MEASURES IS THE NAIVE FORM ON PURPOSE. Binary pass/fail, unweighted, at m=3 where
only admissibility (not fit) is testable. Section 4.2 of the handoff lists six developments
of this statistic; the two with the least-explored ground are (1) quadruplets, where the
rank-1 model first has spare equations and therefore a real residual, and (2) the variance
of the implied v_i across a view's triplets, rather than its pass rate. Both are different
statistics from this one, not tunings of it.

Result when run 2026-08-05: sign test passes on 83.4% of triplets (range 0.66-0.98, and 0 of
119 splits have all triplets passing, so the test is not degenerate). As a per-feature
ranker: Spearman +0.0386, CI [-0.0073, +0.0856], 60+/59-. Adding the magnitude bound makes
it worse (-0.0084). Good-minus-non-good pass rate +0.0127 [+0.0021, +0.0235].

THE IDEA UNDER TEST (Omri, 2026-08-05). Under conditional independence given the latent,
the off-diagonal covariance of oriented views is rank-1: C_ij = v_i v_j. For a TRIPLET
(i,j,k) that is 3 equations in 3 unknowns and solves exactly:

    v_i^2 = C_ij * C_ik / C_jk

So there is NO residual at m=3 — every triplet fits by construction. But the SOLUTION
still has to be admissible, which gives two genuine per-triplet tests:

    SIGN      v_i^2 >= 0    <=>  C_ij * C_ik * C_jk > 0
    MAGNITUDE v_i^2 <= Var(view i) = 1  (views are z-scored)

Score each feature by the fraction of triplets containing it that pass. Rank by that.

WHAT THIS PROBE MEASURES: whether that per-feature score is positively or negatively
associated with membership in exp12's label-guided good sets. It does NOT run the arm
through the exp16 harness — it asks the cheaper prior question of whether the score even
points the right way, which is what Step 223's misfit diagnostic already put in doubt
(good subsets fit the rank-1 model WORSE than matched random ones, +0.100, 21W/3L).

Read-only. No files written.
"""
import io
import os
import sys
import zlib
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
REPO = r"c:/Users/omris/TAU/hallucination_detection"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "upcr_study"))

import common as S                                                    # noqa: E402
from exp10_channel_ceilings import FIT, N_SPLITS, derive_cell, _zsc   # noqa: E402
from exp13_incumbent_anchored_ranking import load_exp12_splits        # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                     # noqa: E402
from spectral_utils.upcr import upcr_fit                              # noqa: E402


def triplet_scores(C):
    """Per-feature pass rates for the sign and magnitude admissibility tests."""
    m = C.shape[0]
    n_tot = np.zeros(m)
    n_sign = np.zeros(m)
    n_both = np.zeros(m)
    n_sign_pass_global = 0
    n_trip = 0
    for i, j, k in combinations(range(m), 3):
        cij, cik, cjk = C[i, j], C[i, k], C[j, k]
        n_trip += 1
        sign_ok = (cij * cik * cjk) > 0
        n_sign_pass_global += int(sign_ok)
        for a, (p, q) in ((i, (cij, cik)), (j, (cij, cjk)), (k, (cik, cjk))):
            r = cjk if a == i else (cik if a == j else cij)
            n_tot[a] += 1
            if sign_ok:
                n_sign[a] += 1
                v2 = (p * q) / r if abs(r) > 1e-12 else np.inf
                if 0.0 <= v2 <= 1.0:
                    n_both[a] += 1
    return (n_sign / np.maximum(n_tot, 1), n_both / np.maximum(n_tot, 1),
            n_sign_pass_global / max(n_trip, 1))


def main():
    cells = S.load()
    prev12 = load_exp12_splits()
    rows_sign, rows_both, glob = [], [], []
    per_cell = []

    for ck, cell in cells.items():
        V0 = cell["V"]
        dcell = derive_cell(cell)
        y = dcell["labels"]
        n, m = V0.shape
        hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], float)
        pool = list(cell["pool"])
        rng = np.random.default_rng(zlib.crc32(ck.encode()) % (2 ** 32))
        for rep in range(N_SPLITS):
            idx = rng.permutation(n)
            a_idx, b_idx = idx[: n // 2], idx[n // 2:]
            if len(np.unique(y[a_idx])) < 2 or len(np.unique(y[b_idx])) < 2:
                continue
            raw_a = np.column_stack([_zsc(V0[a_idx, j]) for j in range(m)]) * hand
            try:
                pol = np.sign(upcr_fit(raw_a.T, **FIT).rho_hat_full)
            except Exception:
                continue
            pol[pol == 0] = 1.0
            Va = raw_a * pol                       # DERIVED polarity, as deployed
            C = np.corrcoef(Va.T)
            C = np.nan_to_num(np.atleast_2d(C), nan=0.0)
            ref12 = prev12.get((ck, rep))
            if ref12 is None:
                continue
            good = set(pool.index(f) for f in ref12["greedy_cols"].split("|"))
            s_sign, s_both, gfrac = triplet_scores(C)
            glob.append(gfrac)
            lab = np.array([1.0 if j in good else 0.0 for j in range(m)])
            if lab.std() > 0:
                if np.std(s_sign) > 1e-12:
                    rows_sign.append(spearmanr(s_sign, lab).statistic)
                if np.std(s_both) > 1e-12:
                    rows_both.append(spearmanr(s_both, lab).statistic)
                per_cell.append((ck, float(np.mean(s_sign)), float(np.mean(s_both)),
                                 float(np.mean(s_sign[list(good)])),
                                 float(np.mean(s_sign[[j for j in range(m)
                                                       if j not in good]]))))
            for _ in range(3):
                rng.choice(m, 5, replace=False)

    def rep(name, arr):
        a = np.array([x for x in arr if np.isfinite(x)])
        if len(a) == 0:
            print(f"  {name:34s} DEGENERATE (no variance in the score)")
            return
        g = np.random.default_rng(0)
        bs = a[g.integers(0, len(a), size=(20000, len(a)))].mean(axis=1)
        print(f"  {name:34s} mean Spearman {a.mean():+.4f} "
              f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] "
              f"{int((a > 0).sum())}+/{int((a < 0).sum())}-  n={len(a)}")

    print("\nHOW OFTEN DOES A TRIPLET PASS AT ALL?")
    g = np.array(glob)
    print(f"  sign test pass rate over all triplets : mean {g.mean():.4f} "
          f"min {g.min():.4f} max {g.max():.4f}")
    print(f"  splits where EVERY triplet passes     : {int((g >= 0.9999).sum())}/{len(g)}")

    print("\nDOES THE PER-FEATURE PASS RATE POINT AT THE GOOD FEATURES?")
    print("  (Spearman between the feature's triplet pass rate and its membership")
    print("   in exp12's label-guided good set, computed per split)")
    rep("sign test only", rows_sign)
    rep("sign + magnitude bound", rows_both)

    print("\nPASS RATE INSIDE vs OUTSIDE THE GOOD SET (sign test)")
    ins = np.array([r[3] for r in per_cell])
    outs = np.array([r[4] for r in per_cell])
    d = ins - outs
    d = d[np.isfinite(d)]
    if len(d):
        g2 = np.random.default_rng(0)
        bs = d[g2.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
        print(f"  good minus non-good: {d.mean():+.4f} "
              f"[{np.percentile(bs, 2.5):+.4f}, {np.percentile(bs, 97.5):+.4f}] "
              f"{int((d > 0).sum())}+/{int((d < 0).sum())}-  n={len(d)}")


if __name__ == "__main__":
    main()
