"""
inscope_bench_common.py — one canonical loading + scoring path for in-scope benches.

WHY THIS EXISTS (Step 201, defects 1, 2, 8)
-------------------------------------------
Three Extension-H scripts each rolled their own cell loading and scoring, and each
got it wrong in a different way:
  * `prior_free_bench.py` fused RAW, un-z-scored columns in one arm and z-scored
    ones in another (column-std ratio 9.3e+08 on cell 1);
  * `test_user_pipeline.py` and `test_iterative_lsml_pruning.py` built the pool
    from `fd.keys()` instead of `CANONICAL_POOL` and used **column 0** as the
    orientation anchor;
  * consequently their GOOD_6 reference macro came out **0.7273** instead of the
    canonical **0.7594**, differing on 25/25 cells (max diff 0.1294) — so every
    comparison in those files was against a mis-computed baseline.

Everything here mirrors `spectral_utils/repgrid_scoring.score_subset`: cells come
from `prepare_cell` (z-scored V over `CANONICAL_POOL`, with the cell's own resolved
anchor), fusion is `lsml_continuous`, the global sign is resolved label-free by
`anchor_orient` against that anchor, and AUROC is raw — never `max(a, 1-a)`.

Use `assert_good6(cells)` as the validity anchor before reporting anything.
"""
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from spectral_utils.fusion_utils import lsml_continuous          # noqa: E402
from spectral_utils.streaming_utils import anchor_orient         # noqa: E402
from spectral_utils.subset_sweep import GOOD_6                   # noqa: E402

GOOD6_EXPECTED = 0.7594
GOOD6_TOL = 0.002


def load_cells():
    """All 25 in-scope cells through the canonical `prepare_cell` path.

    Returns {cell_key: {V, anchor, pool, labels, unlabeled}} where V is z-scored
    over CANONICAL_POOL and `anchor` is the cell's own resolved anchor view.
    """
    from compare_anchor_quality import load_all_inscope_cells
    out = {}
    for ck, cd in load_all_inscope_cells().items():
        u = cd["unlabeled"]
        out[ck] = {"V": np.asarray(u.V, dtype=np.float64),
                   "anchor": np.asarray(u.anchor, dtype=np.float64),
                   "pool": list(u.pool),
                   "labels": np.asarray(cd["labels"], dtype=int),
                   "unlabeled": u}
    return out


def score_cols(cell, cols):
    """Canonical score for a column subset: L-SML fuse -> label-free anchor
    orientation -> raw AUROC. Returns np.nan for subsets smaller than 3."""
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return float("nan")
    V = cell["V"]
    fused, _ = lsml_continuous(*[V[:, c] for c in cols])
    score, _ = anchor_orient(np.asarray(fused, dtype=float), cell["anchor"])
    return float(roc_auc_score(cell["labels"], score))


def good6_cols(cell):
    return [cell["pool"].index(f) for f in GOOD_6 if f in cell["pool"]]


def good6_score(cell):
    cols = good6_cols(cell)
    return score_cols(cell, cols) if len(cols) >= 3 else float("nan")


def assert_good6(cells, verbose=True):
    """Validity anchor (SPEC_gap_ladder.md §8). The GOOD_6 macro must reproduce
    0.7594; if it does not, the loaded data is not the data the canonical numbers
    came from and every downstream conclusion is void."""
    vals = [good6_score(c) for c in cells.values()]
    macro = float(np.nanmean(vals))
    ok = abs(macro - GOOD6_EXPECTED) <= GOOD6_TOL
    if verbose:
        print(f"VALIDITY: GOOD_6 macro = {macro:.4f} "
              f"(expected {GOOD6_EXPECTED} +/- {GOOD6_TOL}) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return ok, macro
