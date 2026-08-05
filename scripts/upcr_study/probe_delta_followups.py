"""Three follow-ups to check_delta.py.

(a) Is "good sets fit the model worse" just cohesion in disguise?
(b) Does the MULTIPLICATIVE rank-1 model (SML / tetrad / Gemini's arms) give the
    same direction as the ADDITIVE one (U-PCR's own)?
(c) THE ONE THAT MATTERS FOR OMRI'S IDEA: Step 220 priced the weighting channel by
    sweeping w(theta) inside span(v1, v2) of C. A Delta-aware model would give
    eigenvectors of a DIFFERENT matrix, which need not lie in that 2-D span. So how
    much room is there between the best vector IN the span and the best linear
    combination overall?  Selected on half A, scored on half B, both.

STATUS: EXPLORATORY, committed as PROVENANCE. Not pre-registered, no multiplicity
control. Follow-up (c) is the script that produced **the number the whole live line aims
at**: supervised linear minus best-in-span = **+1.24pp, CI [+0.17, +2.29], p = 0.016** —
the room OUTSIDE `span(v1, v2)` that Step 220's weight sweep could not reach. See
`HANDOFF_FEATURE_SELECTION_AND_FUSE.md` sections 4.3 and 6.

Before building on that number, re-run this and confirm it. It is an exploratory
measurement with a CI that clears zero but not by much, and it is the single load-bearing
result for choosing the weights channel over the (closed) selection channel.
Requires the per-cell feature pkls, which are NOT in the repo.
"""
import os
import sys
import zlib

import numpy as np
from scipy.linalg import eigh
from scipy.stats import wilcoxon, spearmanr
from sklearn.linear_model import LogisticRegression

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
from spectral_utils.upcr import upcr_fit, additive_design             # noqa: E402

N_MATCHED = 50
N_THETA = 721          # 0.25 deg steps over [0, pi)


def add_misfit(C, cols):
    cols = sorted(int(c) for c in cols)
    m = len(cols)
    if m < 3:
        return np.nan
    Cs = C[np.ix_(cols, cols)]
    A, prs = additive_design(m)
    b = np.array([Cs[i, j] for i, j in prs], float)
    A1 = np.column_stack([A, -np.ones(len(prs))])
    coef, *_ = np.linalg.lstsq(A1, b, rcond=None)
    r = b - A1 @ coef
    nb = np.linalg.norm(b)
    return float(np.linalg.norm(r) / nb) if nb > 1e-12 else np.nan


def mult_misfit(C, cols):
    """Multiplicative rank-1: C_off ~ a a^T. Normalised residual of the best
    rank-1 fit to the zero-diagonal matrix. This is Gemini's / SML's model."""
    cols = sorted(int(c) for c in cols)
    m = len(cols)
    if m < 3:
        return np.nan
    Cs = C[np.ix_(cols, cols)].copy()
    np.fill_diagonal(Cs, 0.0)
    nb = np.linalg.norm(Cs)
    if nb < 1e-12:
        return np.nan
    # best rank-1 approximation with zero diagonal: alternate a few times
    ev, evec = eigh(Cs)
    a = evec[:, -1] * np.sqrt(max(ev[-1], 1e-12))
    for _ in range(50):
        M = Cs.copy()
        np.fill_diagonal(M, np.outer(a, a).diagonal())
        ev, evec = eigh(M)
        a = evec[:, -1] * np.sqrt(max(ev[-1], 1e-12))
    R = Cs - np.outer(a, a)
    np.fill_diagonal(R, 0.0)
    return float(np.linalg.norm(R) / nb)


def cohesion(C, cols):
    cols = sorted(int(c) for c in cols)
    Cs = C[np.ix_(cols, cols)]
    iu = np.triu_indices(len(cols), 1)
    return float(np.mean(np.abs(Cs[iu])))


def auroc(cell, score):
    return S.auroc_from_score(cell, score)


def main():
    cells = S.load()
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    prev = load_exp12_splits()

    rows = []
    for ci, ck in enumerate(cells, 1):
        cell, dcell = cells[ck], dcells[ck]
        V0 = cell["V"]; y = dcell["labels"]; n, m = V0.shape
        hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], float)
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
            raw_b = np.column_stack([_zsc(V0[b_idx, j]) for j in range(m)]) * hand
            try:
                pol = np.sign(upcr_fit(raw_a.T, **FIT).rho_hat_full)
            except Exception:
                continue
            pol[pol == 0] = 1.0
            ca = {"V": raw_a * pol, "anchor": _zsc(dcell["anchor"][a_idx]), "labels": y[a_idx]}
            cb = {"V": raw_b * pol, "anchor": _zsc(dcell["anchor"][b_idx]), "labels": y[b_idx]}

            _, res_a = fit_cols(ca, range(m))
            if res_a is None:
                continue
            keep = [int(j) for j in np.where(res_a.keep)[0]]
            ref = prev.get((ck, rep))
            if ref is None:
                continue
            k = int(ref["k"])
            good = [pool.index(f) for f in ref["greedy_cols"].split("|")]

            F = ca["V"].T
            C = (F @ F.T) / F.shape[1]

            ins = np.array(sorted(set(keep)), int)
            kk = min(k, len(ins))
            rnd_sets = [trng.choice(ins, kk, replace=False) for _ in range(N_MATCHED)]

            row = dict(
                add_good=add_misfit(C, good),
                add_rand=float(np.nanmean([add_misfit(C, s) for s in rnd_sets])),
                mul_good=mult_misfit(C, good),
                mul_rand=float(np.nanmean([mult_misfit(C, s) for s in rnd_sets])),
                coh_good=cohesion(C, good),
                coh_rand=float(np.nanmean([cohesion(C, s) for s in rnd_sets])),
            )

            # ---- (c) the 2-D span vs the whole weight space, on the KEEP set ----
            Fa = ca["V"][:, keep].T
            Fb = cb["V"][:, keep].T
            Ck = (Fa @ Fa.T) / Fa.shape[1]
            nk = len(keep)
            if nk >= 3:
                ev, evec = eigh(Ck, subset_by_index=[nk - 2, nk - 1])
                v1, v2 = evec[:, ::-1].T
                th = np.linspace(0, np.pi, N_THETA, endpoint=False)
                best_a, best_w = -1.0, None
                for t in th:                      # SELECT on half A
                    w = np.cos(t) * v1 + np.sin(t) * v2
                    a_ = auroc(ca, w @ Fa)
                    if np.isfinite(a_) and a_ > best_a:
                        best_a, best_w = a_, w
                row["span2d_B"] = auroc(cb, best_w @ Fb) if best_w is not None else np.nan
                row["deployed_B"] = fit_cols(cb, keep)[0]
                try:                              # the whole weight space, on half A
                    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
                    lr.fit(ca["V"][:, keep], ca["labels"])
                    row["linear_B"] = auroc(cb, lr.decision_function(cb["V"][:, keep]))
                except Exception:
                    row["linear_B"] = np.nan
            per.append(row)
            for _ in range(2 * N_OVERLAP_DRAWS + N_RANDOM):
                rng.choice(m, k, replace=False)

        if not per:
            continue
        agg = {kk2: float(np.nanmean([p[kk2] for p in per if np.isfinite(p.get(kk2, np.nan))]))
               for kk2 in per[0]}
        agg["cell"] = ck
        rows.append(agg)
        print(f"[{ci:2d}/{len(cells)}] {S.plain_cell(ck)[:28]:28s} "
              f"add {agg['add_good']-agg['add_rand']:+.3f}  "
              f"mult {agg['mul_good']-agg['mul_rand']:+.3f}  "
              f"coh {agg['coh_good']-agg['coh_rand']:+.3f} | "
              f"span2d {agg['span2d_B']:.4f} linear {agg['linear_B']:.4f}", flush=True)

    def paired(key_a, key_b, lab, scale=1.0):
        d = np.array([r[key_b] - r[key_a] for r in rows], float) * scale
        d = d[np.isfinite(d)]
        g = np.random.default_rng(0)
        bs = [float(np.mean(g.choice(d, len(d), replace=True))) for _ in range(20000)]
        unit = "pp" if scale == 100 else ""
        print(f"  {lab:52s} {d.mean():+7.4f}{unit} "
              f"[{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}] "
              f"{int((d>0).sum())}W/{int((d<0).sum())}L p={wilcoxon(d).pvalue:.4g}")
        return d

    print("\n" + "=" * 82)
    print("(a)+(b) DO THE GOOD SETS FIT THE 1-FACTOR MODEL BETTER?  (+ = fits WORSE)")
    da = paired("add_rand", "add_good", "ADDITIVE misfit (U-PCR's own model)")
    dm = paired("mul_rand", "mul_good", "MULTIPLICATIVE rank-1 misfit (SML / tetrad / Gemini)")
    dc = paired("coh_rand", "coh_good", "cohesion, mean |corr| within the set")

    print("\n  IS IT JUST COHESION?")
    for nm, dd in (("additive", da), ("multiplicative", dm)):
        r, p = spearmanr(dd, dc)
        print(f"    Spearman(excess {nm} misfit, excess cohesion) = {r:+.3f}  p={p:.4f}")

    print("\n(c) THE CHANNEL OMRI'S IDEA OPENS — is there room OUTSIDE span(v1, v2)?")
    print("    all on the DEPLOYED keep set, selected on half A, scored on half B")
    paired("deployed_B", "span2d_B", "best in span(v1,v2)  minus deployed", 100)
    paired("span2d_B", "linear_B", "supervised linear  minus  best in span(v1,v2)", 100)
    paired("deployed_B", "linear_B", "supervised linear  minus  deployed", 100)


if __name__ == "__main__":
    main()
