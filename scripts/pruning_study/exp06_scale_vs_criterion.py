"""
exp06_scale_vs_criterion.py — SPEC_residual_scaling_fix prediction P2.

Step 203 measured Spearman(fit misfit, AUROC) = +0.223 within size, positive in
24/25 cells, and concluded every selector minimised a quantity that should have
been maximised (-> Extension I1, "sign-flip the selectors").

SPEC_residual_scaling_fix argues that correlation is an ARTIFACT of the loading
scale: `_estimate_von_voff` returned the unit-norm eigenvector, so misfit was
inflated by group size x coupling strength — largest exactly where the clustering
succeeded. P2 predicts the +0.223 weakens or flips once the scale is corrected.

This re-runs exp03_preflight's live sweep under all three loading scales
(`fusion_utils.LOADING_SCALES`) on identical subsets, so the correlation is
comparable across scales draw-for-draw.

If P2 holds -> the criterion never needed inverting, it needed scaling, and
Extension I1 is curing a symptom.
If P2 fails -> the scaling is a real bug AND the inversion has a second,
independent cause, which is the more interesting result.

Run:  python scripts/pruning_study/exp06_scale_vs_criterion.py
Out:  results/pruning_study/06_scale_vs_criterion/{draws.csv,per_cell.csv,summary.json}
"""
import os
import sys
import json
import time

import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study_common as S                                          # noqa: E402

from spectral_utils.fusion_utils import (                         # noqa: E402
    lsml_continuous, LOADING_SCALES,
)
from spectral_utils.streaming_utils import anchor_orient          # noqa: E402

SIZES = [3, 4, 6, 8, 11, 14, 17, 21, 25, 30]   # identical to exp03_preflight
N_DRAWS = 30
SEED = 0


def fuse_at_scale(cell, cols, loading_scale):
    """Mirrors study_common.fuse_meta exactly, plus the loading_scale knob.

    Deliberately NOT a call into fuse_meta: that helper pins the default scale,
    and silently diverging from the canonical path is how Step 201's benches
    ended up comparing against a mis-computed baseline.
    """
    cols = sorted(set(int(c) for c in cols))
    V = cell["V"]
    fused, meta = lsml_continuous(*[V[:, c] for c in cols],
                                  compute_score_matrix=False,
                                  loading_scale=loading_scale)
    fused = np.asarray(fused, dtype=float)
    if not np.isfinite(fused).all() or fused.std() < 1e-12:
        return float("nan"), meta
    score, _ = anchor_orient(fused, cell["anchor"])
    return float(roc_auc_score(cell["labels"], score)), meta


def main():
    out = S.outdir("06_scale_vs_criterion")
    cells = S.load()
    S.validity_check(cells)

    rng = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    for ck, cell in cells.items():
        p = len(cell["pool"])
        for size in SIZES:
            if size > p:
                continue
            draws = ([list(range(p))] if size >= p else
                     [sorted(rng.choice(p, size=size, replace=False))
                      for _ in range(N_DRAWS)])
            for cols in draws:
                rec = {"test_set_code": ck, "size": size,
                       "measurements_code": ",".join(cell["pool"][i] for i in cols)}
                for sc in LOADING_SCALES:
                    auc, meta = fuse_at_scale(cell, cols, sc)
                    rec[f"auroc_{sc}"] = auc
                    rec[f"misfit_{sc}"] = float(meta["residual"])
                    rec[f"K_{sc}"] = int(meta["K"])
                rows.append(rec)
        print(f"  {ck[:34]:34s} done  ({time.time() - t0:.0f}s)", flush=True)

    S.save_csv(os.path.join(out, "draws.csv"), rows)

    # ---- P2: within-size Spearman(misfit, AUROC), per cell, per scale ----
    by_cell = {}
    for r in rows:
        by_cell.setdefault(r["test_set_code"], []).append(r)

    per_cell, summary = [], {}
    for ck, rs in by_cell.items():
        rec = {"test_set_code": ck}
        for sc in LOADING_SCALES:
            sz = np.array([r["size"] for r in rs])
            au = np.array([r[f"auroc_{sc}"] for r in rs], float)
            mf = np.array([r[f"misfit_{sc}"] for r in rs], float)
            # pool the within-size correlations (identical to Step 203's recipe)
            cors = []
            for s in np.unique(sz):
                m = (sz == s) & np.isfinite(au) & np.isfinite(mf)
                if m.sum() >= 5 and np.ptp(mf[m]) > 0 and np.ptp(au[m]) > 0:
                    cors.append(spearmanr(mf[m], au[m]).correlation)
            rec[f"rho_{sc}"] = float(np.nanmean(cors)) if cors else float("nan")
            rec[f"macro_auroc_{sc}"] = float(np.nanmean(au))
        per_cell.append(rec)

    S.save_csv(os.path.join(out, "per_cell.csv"), per_cell)

    print("\nP2 - within-size Spearman(misfit, AUROC). Step 203 baseline: "
          "+0.223 mean, positive in 24/25")
    print(f"{'scale':>9} {'mean rho':>10} {'median':>9} {'n positive':>12} "
          f"{'macro AUROC':>12}")
    for sc in LOADING_SCALES:
        v = np.array([r[f"rho_{sc}"] for r in per_cell], float)
        v = v[np.isfinite(v)]
        mac = np.nanmean([r[f"macro_auroc_{sc}"] for r in per_cell])
        summary[sc] = {"mean_rho": float(np.mean(v)),
                       "median_rho": float(np.median(v)),
                       "n_positive": int((v > 0).sum()),
                       "n_cells": int(v.size),
                       "macro_auroc": float(mac)}
        print(f"{sc:>9} {np.mean(v):10.3f} {np.median(v):9.3f} "
              f"{int((v > 0).sum()):8d}/{v.size:<3d} {mac:12.4f}")

    # paired test: does the correlation move relative to the incumbent?
    for sc in ("eigen", "complete"):
        a = np.array([r["rho_unit"] for r in per_cell], float)
        b = np.array([r[f"rho_{sc}"] for r in per_cell], float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() >= 5:
            st = wilcoxon(a[m], b[m])
            summary[sc]["wilcoxon_vs_unit_p"] = float(st.pvalue)
            summary[sc]["mean_shift_vs_unit"] = float(np.mean(b[m] - a[m]))
            print(f"  {sc} vs unit: mean shift {np.mean(b[m] - a[m]):+.3f}, "
                  f"Wilcoxon p={st.pvalue:.4f}")

    base = summary["unit"]["mean_rho"]
    best = summary["complete"]["mean_rho"]
    verdict = ("P2 HOLDS - correlation weakened/flipped; Extension I1 (sign-flip "
               "the selectors) is curing a symptom"
               if abs(best) < abs(base) * 0.5 or np.sign(best) != np.sign(base)
               else "P2 FAILS - the inversion survives the scale fix, so it has a "
                    "second independent cause")
    print(f"\n{verdict}")
    summary["verdict"] = verdict
    S.save_json(os.path.join(out, "summary.json"), summary)
    print(f"Wrote -> {out}")


if __name__ == "__main__":
    main()
