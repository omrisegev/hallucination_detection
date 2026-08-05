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

# STEP 216 — THIS CONSTANT CHANGED, AND IT WAS RE-DERIVED, NOT ADJUSTED TO FIT.
# It was 0.7594 on the 25-cell roster. Two intentional data changes moved it, and the
# decomposition is exact (each step measured, per cell, in that order):
#
#   0.759398  25 cells, pre-repair                      <- the old constant, reproduced
#   0.763232  24 cells: `inside_coqa_llama7b` REJECTED  (+0.38pp; its GOOD_6 was 0.6674,
#             for a generation defect, not a result      below macro, so dropping it lifts)
#   0.773344  + `seiclr_triviaqa_opt30b` answer-cropped (+1.01pp; 0.5884 -> 0.8311)
#
# After an intentional data change the macro is NOT the gate — it is expected to move.
# The real gate is per-cell equality on the untouched cells, which
# `scripts/answer_span_score_check.py` asserts (all 3 scored metrics bit-identical on
# 23/23). This constant remains the anti-regression anchor for everything AFTER that.
#
# CAVEAT carried by the 0.7733 figure: on the cropped cell GOOD_6 has only 4 of its 6
# views (`low_band_power` and `spectral_entropy` need >= 8 tokens; the median answer is
# 3). Per Step 205 L-SML is numerically undetermined at 4 views, so that cell's 0.8311
# is reported with an explicit caveat rather than quoted as comparable to the others.
GOOD6_EXPECTED = 0.7733
GOOD6_TOL = 0.002


def load_cells():
    """All in-scope cells (24 since Step 216) through the canonical `prepare_cell` path.

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
    GOOD6_EXPECTED; if it does not, the loaded data is not the data the canonical numbers
    came from and every downstream conclusion is void."""
    vals = [good6_score(c) for c in cells.values()]
    macro = float(np.nanmean(vals))
    ok = abs(macro - GOOD6_EXPECTED) <= GOOD6_TOL
    if verbose:
        print(f"VALIDITY: GOOD_6 macro = {macro:.4f} "
              f"(expected {GOOD6_EXPECTED} +/- {GOOD6_TOL}) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return ok, macro
