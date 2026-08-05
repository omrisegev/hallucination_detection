#!/usr/bin/env python
"""
answer_span_score_check.py — per-cell SCORED equality across the Step-216 repair.

WHY THIS EXISTS
---------------
`answer_span_repair.py` already asserts that every non-repaired cell is
bit-identical to the pre-repair cache at the FEATURE level. That is necessary but
not sufficient: the canonical scoring path re-derives the pool, the z-scoring and
the anchor per cell, so a feature-level match still leaves open whether the
SCORES moved. After an intentional data change the honest gate is per-cell
equality on the unaffected cells — never the macro, which is expected to move.

This loads both `local_cache/repgrid_cells.pkl` and
`local_cache/repgrid_cells.precrop.pkl` through the same `prepare_cell` path
`inscope_bench_common.load_cells` uses, and compares per cell:

    GOOD_6      L-SML over the GOOD_6 views  -> anchor_orient -> raw AUROC
    FULL        L-SML over the whole resolved pool, same orientation
    UPCR        U-PCR + sign(rho) polarity, same orientation (the deployed arm)

Every cell except those in `answer_span.RUNON_CELLS` must match to `--tol`
(default 0.0 — exact). The repaired cells are reported, never gated.

Usage:
    python scripts/answer_span_score_check.py
    python scripts/answer_span_score_check.py --tol 1e-12
"""
import argparse
import os
import pickle
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_cells import INSCOPE                                   # noqa: E402
from spectral_utils.answer_span import RUNON_CELLS                  # noqa: E402
from spectral_utils.fusion_utils import lsml_continuous             # noqa: E402
from spectral_utils.streaming_utils import anchor_orient            # noqa: E402
from spectral_utils.subset_sweep import (                           # noqa: E402
    ALL_SIGNS, CANONICAL_POOL, GOOD_6, prepare_cell)
from spectral_utils.upcr import upcr_fit                            # noqa: E402

CACHE = os.path.join(REPO, "local_cache", "repgrid_cells.pkl")
BACKUP = os.path.join(REPO, "local_cache", "repgrid_cells.precrop.pkl")
DERIVED = os.path.join(REPO, "local_cache", "derived_views.pkl")

FIT = dict(loss="l2", exclusion=True, difficulty_gate=False,
           simple_avg_fallback=True, recompute_after_exclusion=True,
           g2_projection_k=1, scale_ratio=0.25)


def load_side(path, roster):
    """{cell: prepared cell dict} from one repgrid cache, canonical path."""
    with open(path, "rb") as f:
        raw = pickle.load(f)
    derived = {}
    if os.path.exists(DERIVED):
        with open(DERIVED, "rb") as f:
            derived = pickle.load(f)

    out = {}
    for ck, payload in raw.items():
        if ck not in roster:
            continue
        fd = dict(payload["feats"])
        labels = np.asarray(payload["labels"], dtype=int)
        for name, arr in derived.get(("repgrid", ck), {}).items():
            if arr is not None and len(arr) == len(labels):
                fd[name] = arr
        c = prepare_cell("repgrid", ck, fd, labels, feature_pool=CANONICAL_POOL)
        if c is None:
            continue
        out[ck] = {"V": np.asarray(c.V, dtype=np.float64),
                   "anchor": np.asarray(c.anchor, dtype=np.float64),
                   "pool": list(c.pool),
                   "labels": np.asarray(c.labels, dtype=int)}
    return out


def _oriented_auc(cell, score):
    s, _ = anchor_orient(np.asarray(score, dtype=float), cell["anchor"])
    return float(roc_auc_score(cell["labels"], s))


def lsml_auc(cell, cols):
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return float("nan")
    fused, _ = lsml_continuous(*[cell["V"][:, c] for c in cols])
    return _oriented_auc(cell, fused)


def upcr_auc(cell):
    V, pool = cell["V"], cell["pool"]
    hand = np.array([ALL_SIGNS.get(f, +1) for f in pool], dtype=float)
    V_un = V * hand
    derived = np.sign(upcr_fit(V_un.T, **FIT).rho_hat_full)
    derived[derived == 0] = 1.0
    F = (V_un * derived).T
    res = upcr_fit(F, **FIT)
    return _oriented_auc(cell, res.w @ F), int(res.keep.sum())


def score_all(cells):
    out = {}
    for ck, cell in cells.items():
        g6 = [cell["pool"].index(f) for f in GOOD_6 if f in cell["pool"]]
        up, kept = upcr_auc(cell)
        out[ck] = {"n": len(cell["labels"]), "pool": len(cell["pool"]),
                   "g6_views": len(g6), "GOOD_6": lsml_auc(cell, g6),
                   "FULL": lsml_auc(cell, range(len(cell["pool"]))),
                   "UPCR": up, "kept": kept}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max allowed |delta| on an unaffected cell (default exact)")
    ap.add_argument("--roster", choices=("inscope", "all"), default="all",
                    help="'all' checks every cell present in both caches")
    args = ap.parse_args()

    if args.roster == "inscope":
        roster = set(INSCOPE)
    else:
        with open(BACKUP, "rb") as f:
            roster = set(pickle.load(f))

    print(f"loading pre-repair  <- {BACKUP}", flush=True)
    old = score_all(load_side(BACKUP, roster))
    print(f"loading post-repair <- {CACHE}", flush=True)
    new = score_all(load_side(CACHE, roster))

    shared = sorted(set(old) & set(new))
    bad, moved = [], []
    print(f"\n{'cell':<34} {'metric':<7} {'before':>8} {'after':>8} {'delta':>9}")
    print("-" * 70)
    for ck in shared:
        for m in ("GOOD_6", "FULL", "UPCR"):
            a, b = old[ck][m], new[ck][m]
            if np.isnan(a) and np.isnan(b):
                continue
            d = (b - a) if not (np.isnan(a) or np.isnan(b)) else float("nan")
            if ck in RUNON_CELLS:
                moved.append((ck, m, a, b, d))
                print(f"{ck:<34} {m:<7} {a:8.4f} {b:8.4f} {d:+9.4f}   REPAIRED")
                continue
            if np.isnan(d) or abs(d) > args.tol:
                bad.append((ck, m, a, b, d))
                print(f"{ck:<34} {m:<7} {a:8.4f} {b:8.4f} {d:+9.4f}   *** MOVED")

    only = (set(old) ^ set(new))
    if only:
        print(f"\nroster differs between caches: {sorted(only)}")

    for ck in RUNON_CELLS:
        if ck in old and ck in new:
            print(f"\n{ck}: n {old[ck]['n']} -> {new[ck]['n']}, "
                  f"pool {old[ck]['pool']} -> {new[ck]['pool']}, "
                  f"GOOD_6 views {old[ck]['g6_views']} -> {new[ck]['g6_views']}, "
                  f"U-PCR kept {old[ck]['kept']} -> {new[ck]['kept']}")

    n_rep = len(set(RUNON_CELLS) & set(shared))
    if bad:
        print(f"\nFAIL — {len(bad)} scored value(s) moved on cells the repair should "
              f"not have touched (tol={args.tol}).")
        sys.exit(1)
    print(f"\nOK — all 3 scored metrics reproduce exactly on {len(shared) - n_rep} "
          f"unaffected cells (tol={args.tol}); {n_rep} repaired cell(s) moved "
          f"as intended.")


if __name__ == "__main__":
    main()
