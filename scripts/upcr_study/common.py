"""
common.py — shared harness for the U-PCR study.

Re-exports the pruning study's pure-Python SVG chart helpers rather than
duplicating them (no matplotlib anywhere in this pipeline), but writes to its own
OUT_ROOT so the two studies never collide.

Scoring goes through the SAME canonical path as everything else in the repo:
cells from `inscope_bench_common.load_cells` (z-scored over CANONICAL_POOL, each
cell's own resolved anchor), global sign by `anchor_orient`, raw AUROC — never
`max(a, 1-a)`. Step 201's benches diverged from this and spent a session comparing
against a mis-computed baseline.
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for _p in (REPO, os.path.join(REPO, "scripts"),
           os.path.join(REPO, "scripts", "pruning_study")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from study_common import (                                          # noqa: E402,F401
    line_chart, bar_chart, scatter_chart, html_table, write_page,
    save_csv, save_json, save_npz, PALETTE, plain, plain_cell,
)
from spectral_utils.streaming_utils import anchor_orient            # noqa: E402
from spectral_utils.upcr import upcr_fit                            # noqa: E402

OUT_ROOT = os.path.join(REPO, "results", "upcr_study")

# ---------------------------------------------------------------------------
# LOADING SCALE — reported three ways, never chosen
# ---------------------------------------------------------------------------
# `SPEC_residual_scaling_fix.md` pre-registered 'eigen'. Phase 0 found 'eigen'
# fails the spec's OWN unit check (a 2-feature duplicate block scores 0.2500,
# identical to the broken 'unit' path) and built 'complete', which is exact.
#
# Switching to an estimator chosen after seeing data is a researcher degree of
# freedom, and 'complete' also happens to be the only scale under which the
# clustered variant is runnable on every cell (K>=3) — i.e. the convenient
# answer. Rather than adjudicate that, EVERY scale-dependent number in this study
# is reported under all three so the reader sees the sensitivity directly.
#
# Only experiments that invoke L-SML clustering depend on this at all. exp01 and
# exp06 are pure U-PCR and have no loading scale; they say so rather than
# reporting a fake three-way split.
SCALES = ("unit", "eigen", "complete")
SPEC_PREREGISTERED_SCALE = "eigen"


def outdir(name):
    d = os.path.join(OUT_ROOT, name)
    os.makedirs(d, exist_ok=True)
    return d


def load():
    from inscope_bench_common import load_cells
    return load_cells()


def validity_check(cells, verbose=True):
    """Anti-regression anchor. RAISES on failure — a gate that only prints is not
    a gate, and this project already lost a session to a mis-computed baseline
    (Step 201). Note `assert_good6` returns the TUPLE (ok, macro), so
    `bool(assert_good6(...))` is always True; unpack it."""
    from inscope_bench_common import assert_good6, GOOD6_EXPECTED
    ok, macro = assert_good6(cells, verbose=verbose)
    if not ok:
        raise SystemExit(
            f"VALIDITY CHECK FAILED: GOOD_6 macro = {macro:.4f}, expected "
            f"{GOOD6_EXPECTED}. "
            "The loaded data is not the data our numbers came from — refusing to "
            "report anything downstream.")
    return ok, macro


def auroc_from_score(cell, score):
    """Canonical: resolve the global sign against the cell's anchor, then raw AUROC."""
    score = np.asarray(score, dtype=float)
    if not np.isfinite(score).all() or score.std() < 1e-12:
        return float("nan")
    oriented, _ = anchor_orient(score, cell["anchor"])
    return float(roc_auc_score(cell["labels"], oriented))


def upcr_auroc(cell, cols=None, **kwargs):
    """Fit U-PCR on a cell (default: the whole pool) and score it."""
    V = cell["V"]
    F = (V if cols is None else V[:, sorted(cols)]).T
    res = upcr_fit(F, **kwargs)
    if not np.isfinite(res.w).all():
        return float("nan"), res
    return auroc_from_score(cell, res.w @ F), res


def wl_wilcoxon(a, b, groups=None):
    """Paired effect size the way this project reports it (Omri, Step 203):
    mean delta with wins/losses and a Wilcoxon p — never an average alone.

    ``groups``: the INDEPENDENT unit each observation belongs to, normally the
    test-set id. Pass it whenever the same cell contributes more than one row —
    e.g. a factorial where each cell appears once per configuration-combination.
    Without it the deltas are averaged within each cell FIRST, which is the only
    legitimate pairing; pooling 25 cells x 32 combinations into one n=800 test is
    pseudo-replication and produced p-values as absurd as 1e-44. Found in review.

    Also reports the MEDIAN delta, because several of this study's mean effects
    are tail-driven with a median of exactly 0.00pp.
    """
    from scipy.stats import wilcoxon
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]

    if groups is not None:
        g = np.asarray(groups)[m]
        keys = list(dict.fromkeys(g.tolist()))
        a = np.array([np.mean(a[g == k]) for k in keys])
        b = np.array([np.mean(b[g == k]) for k in keys])

    d = b - a
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    p = float("nan")
    if d.size >= 5 and np.any(d != 0):
        try:
            p = float(wilcoxon(a, b).pvalue)
        except Exception:
            pass
    return {"mean_delta": float(np.mean(d)) if d.size else float("nan"),
            "median_delta": float(np.median(d)) if d.size else float("nan"),
            "wins": wins, "losses": losses, "n": int(d.size), "p": p,
            "unit": "test sets" if groups is not None else "observations"}


def fmt_wl(s):
    med = s.get("median_delta", float("nan"))
    return (f"{s['mean_delta']*100:+.2f}pp mean / {med*100:+.2f}pp median, "
            f"{s['wins']}W/{s['losses']}L over {s['n']} {s.get('unit','obs')}, "
            f"p={s['p']:.3f}")


# ---------------------------------------------------------------------------
# scale-consistent reference points
# ---------------------------------------------------------------------------
def score_cols_at_scale(cell, cols, scale):
    """Canonical L-SML score for a column subset AT A GIVEN LOADING SCALE.

    `inscope_bench_common.good6_score` pins the default ('unit'), so mixing it
    with an L-SML number computed at another scale puts two scales in one table.
    Use this for both sides instead.
    """
    from spectral_utils.fusion_utils import lsml_continuous
    V = cell["V"]
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return float("nan")
    fused, _ = lsml_continuous(*[V[:, c] for c in cols],
                               compute_score_matrix=False, loading_scale=scale)
    return auroc_from_score(cell, fused)


def reference_macros(cells, scale):
    """GOOD_6 and all-30 continuous L-SML, both at the SAME loading scale."""
    from inscope_bench_common import good6_cols
    g6, all30 = [], []
    for cell in cells.values():
        g6.append(score_cols_at_scale(cell, good6_cols(cell), scale))
        all30.append(score_cols_at_scale(cell, range(cell["V"].shape[1]), scale))
    return {"good6": float(np.nanmean(g6)), "lsml_all30": float(np.nanmean(all30)),
            "good6_per_cell": g6, "lsml_all30_per_cell": all30}
