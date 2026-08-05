"""Quick check on Omri's idea: how badly is E[h_i h_j] = 0 violated, is the
violation STRUCTURED (sparse / low-rank), and do the good feature sets have a
smaller violation than matched random sets?

Diagnostic only. No claims, no adoption. Replays exp12's splits via exp13's
machinery so the good sets are the real ones.

STATUS: EXPLORATORY, committed as PROVENANCE. Not pre-registered, no multiplicity
control. This is the script that produced the numbers quoted in
`HANDOFF_FEATURE_SELECTION_AND_FUSE.md` section 3.3 and section 6 — the normalised
additive misfit of 0.464 on the full pool, the sparsity of the violation (top decile of
pairs carries 44% of the residual mass against 10% for uniform; leading eigenvalue share
only 0.33), and the finding that GOOD subsets fit the 1-factor model WORSE than matched
random ones (additive misfit +0.131, 23W/1L, p=6e-7).

That last sign is the one that constrains everything downstream: any criterion rewarding
better conditional-independence fit is aimed AWAY from the target. Re-check it before
building on it. Requires the per-cell feature pkls, which are NOT in the repo.
"""
import os
import sys
import zlib

import numpy as np

REPO = r"c:/Users/omris/TAU/hallucination_detection"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts", "upcr_study"))

import common as S                                                    # noqa: E402
from exp10_channel_ceilings import (                                  # noqa: E402
    FIT, N_SPLITS, derive_cell, fit_cols, _zsc,
)
from exp13_incumbent_anchored_ranking import (                        # noqa: E402
    N_RANDOM, N_OVERLAP_DRAWS, load_exp12_splits,
)
from spectral_utils.subset_sweep import ALL_SIGNS                     # noqa: E402
from spectral_utils.upcr import upcr_fit, additive_design, solve_additive  # noqa: E402

N_MATCHED = 50          # random size-k subsets from the keep set, per split


def additive_misfit(C, cols):
    """Fit C_ij = rho_i + rho_j - g2 on the SUBSET's own pair system and return
    (normalised misfit, raw resid norm, offdiag norm).

    Normalised because a raw residual is a SCALE measure: a subset of mutually
    uncorrelated features has residual ~0 and would look like a perfect fit.
    """
    cols = sorted(int(c) for c in cols)
    m = len(cols)
    if m < 3:
        return np.nan, np.nan, np.nan
    Cs = C[np.ix_(cols, cols)]
    A, prs = additive_design(m)
    b = np.array([Cs[i, j] for i, j in prs], dtype=float)
    # g2 shifts every fitted value by a constant, so absorb it as an intercept
    A1 = np.column_stack([A, -np.ones(len(prs))])
    coef, *_ = np.linalg.lstsq(A1, b, rcond=None)
    resid = b - A1 @ coef
    nb = float(np.linalg.norm(b))
    nr = float(np.linalg.norm(resid))
    return (nr / nb if nb > 1e-12 else np.nan), nr, nb


def delta_structure(C, cols):
    """Is the violation sparse or dense / low-rank or full-rank?

    Builds the residual matrix Delta on the subset, then reports
      top10_share  fraction of squared residual mass in the top 10% of pairs
      rank1_share  leading eigenvalue share of |Delta| spectrum
    """
    cols = sorted(int(c) for c in cols)
    m = len(cols)
    if m < 4:
        return np.nan, np.nan
    Cs = C[np.ix_(cols, cols)]
    A, prs = additive_design(m)
    b = np.array([Cs[i, j] for i, j in prs], dtype=float)
    A1 = np.column_stack([A, -np.ones(len(prs))])
    coef, *_ = np.linalg.lstsq(A1, b, rcond=None)
    resid = b - A1 @ coef
    D = np.zeros((m, m))
    for (i, j), r in zip(prs, resid):
        D[i, j] = D[j, i] = r
    sq = resid ** 2
    k = max(1, int(np.ceil(0.10 * len(sq))))
    top10 = float(np.sort(sq)[::-1][:k].sum() / (sq.sum() + 1e-30))
    ev = np.abs(np.linalg.eigvalsh(D))
    r1 = float(ev.max() / (ev.sum() + 1e-30))
    return top10, r1


def main():
    cells = S.load()
    S.validity_check(cells)
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    prev = load_exp12_splits()

    rows = []
    for ci, ck in enumerate(cells, 1):
        cell, dcell = cells[ck], dcells[ck]
        V0 = cell["V"]
        y = dcell["labels"]
        n, m = V0.shape
        hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
        pool = cell["pool"]

        rng = np.random.default_rng(zlib.crc32(ck.encode()) % (2 ** 32))
        trng = np.random.default_rng(0xDE17A + (zlib.crc32(ck.encode()) % 10 ** 6))

        per = []
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
            ca = {"V": raw_a * pol, "anchor": _zsc(dcell["anchor"][a_idx]),
                  "labels": y[a_idx]}

            _, res_a = fit_cols(ca, range(m))
            if res_a is None:
                continue
            start_a = [int(j) for j in np.where(res_a.keep)[0]]

            ref = prev.get((ck, rep))
            if ref is None:
                continue
            k = int(ref["k"])
            cols_good = [pool.index(f) for f in ref["greedy_cols"].split("|")]

            F = ca["V"].T
            C = (F @ F.T) / F.shape[1]

            mf_pool, _, _ = additive_misfit(C, range(m))
            mf_keep, _, _ = additive_misfit(C, start_a)
            mf_good, _, _ = additive_misfit(C, cols_good)
            top10, r1 = delta_structure(C, start_a)

            # matched random: same size k, drawn from the keep set (same
            # construction as the pruning floor)
            ins = np.array(sorted(set(start_a)), dtype=int)
            kk = min(k, len(ins))
            rnd = []
            for _ in range(N_MATCHED):
                pick = trng.choice(ins, kk, replace=False)
                rnd.append(additive_misfit(C, pick)[0])
            mf_rand = float(np.nanmean(rnd))

            per.append(dict(mf_pool=mf_pool, mf_keep=mf_keep, mf_good=mf_good,
                            mf_rand=mf_rand, top10=top10, rank1=r1,
                            k=k, m=m, n_keep=len(start_a)))

            # exp12's random consumption, replayed so the next split lines up
            for _ in range(2 * N_OVERLAP_DRAWS + N_RANDOM):
                rng.choice(m, k, replace=False)

        if not per:
            continue
        agg = {kk2: float(np.nanmean([p[kk2] for p in per])) for kk2 in per[0]}
        agg["cell"] = ck
        rows.append(agg)
        print(f"[{ci:2d}/{len(cells)}] {S.plain_cell(ck)[:30]:30s} "
              f"misfit pool {agg['mf_pool']:.3f} keep {agg['mf_keep']:.3f} | "
              f"good {agg['mf_good']:.3f} vs matched random {agg['mf_rand']:.3f} "
              f"({(agg['mf_good']-agg['mf_rand'])*100:+.2f}) | "
              f"top10% {agg['top10']:.2f} rank1 {agg['rank1']:.2f}", flush=True)

    print("\n" + "=" * 78)
    print("HOW BADLY IS E[h_i h_j] = 0 VIOLATED?  (normalised additive misfit)")
    for key, lab in (("mf_pool", "full pool"), ("mf_keep", "deployed keep set"),
                     ("mf_good", "the good sets")):
        v = np.array([r[key] for r in rows], float)
        print(f"  {lab:22s} {np.nanmean(v):.4f}   "
              f"[{np.nanmin(v):.3f}, {np.nanmax(v):.3f}]")

    print("\nIS THE VIOLATION STRUCTURED?  (on the deployed keep set)")
    t = np.array([r["top10"] for r in rows], float)
    r1 = np.array([r["rank1"] for r in rows], float)
    print(f"  share of residual mass in top 10% of pairs   {np.nanmean(t):.3f}"
          f"   (uniform would be 0.10)")
    print(f"  leading eigenvalue share of |Delta|          {np.nanmean(r1):.3f}")

    print("\nDOES IT HAVE SELECTION LEVERAGE?  good sets vs MATCHED random")
    d = np.array([r["mf_good"] - r["mf_rand"] for r in rows], float)
    d = d[np.isfinite(d)]
    from scipy.stats import wilcoxon
    bs = np.random.default_rng(0)
    boots = [float(np.mean(bs.choice(d, len(d), replace=True))) for _ in range(20000)]
    print(f"  mean difference {d.mean():+.4f}  "
          f"CI [{np.percentile(boots,2.5):+.4f}, {np.percentile(boots,97.5):+.4f}]  "
          f"{int((d<0).sum())}W/{int((d>0).sum())}L (W = good set fits BETTER)  "
          f"p={wilcoxon(d).pvalue:.4f}")
    print(f"  n_cells={len(d)}   mean k={np.mean([r['k'] for r in rows]):.1f}   "
          f"mean keep={np.mean([r['n_keep'] for r in rows]):.1f}   "
          f"mean m={np.mean([r['m'] for r in rows]):.1f}")

    m_ = np.mean([r["m"] for r in rows])
    k_ = np.mean([r["k"] for r in rows])
    for label, mm in (("full pool", m_), ("good set", k_)):
        eq = mm * (mm - 1) / 2
        print(f"\nDEGREES OF FREEDOM, {label} (m={mm:.1f}): "
              f"{eq:.0f} pair equations, {mm+1:.0f} unknowns (rho + g2), "
              f"{eq-mm-1:.0f} spare.  A FULL Delta adds {eq:.0f} unknowns -> saturated.")


if __name__ == "__main__":
    main()
