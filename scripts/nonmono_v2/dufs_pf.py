#!/usr/bin/env python
"""
dufs_pf.py — the `a2.dufs_pf` selector, standalone, so it can be re-run per config.

WHY THIS EXISTS
---------------
Step 217 held arm A's selected view set FROZEN and got exact zeros on 19 of 24
cells. That is not a null result about transforms; it is the pipeline reporting
that a column it never looks at was changed. The candidates are precisely the
views DUFS does not select (`mechanism.csv`: `dufs_selected == 0` on 34 of 38),
so with a frozen selection the transform has no channel at all.

Re-running the selector on the transformed matrix is the only way to ask the
question that matters: DOES REPAIRING A VIEW GET IT SELECTED? But `a2_groupfs`
costs ~17 s/cell because it also runs GroupFS plus 15 lambda-stability DUFS
trainings; `a2.dufs_pf` needs only the last five (`param_free=True`, DUFS Eq. 7 —
the paper's own answer to lambda tuning, so no stability search is required).

THE ONLY DELICATE PART is the RNG stream. `a2_groupfs` draws, in this order:
    rng.choice(n, R_MAX)          -> the row subsample
    rng.integers(2**31)           -> _choose_C / kmeans seed
    rng.integers(2**31)           -> the _init_magnitudes torch generator
    rng.integers(2**31)           -> the final GroupFS training seed
    5 x rng.integers(2**31)       -> the stability seeds, REUSED by dufs_pf
Three draws must therefore be consumed and discarded, or the five seeds differ
and the selection is a different selector wearing the same name. `--verify`
checks the extraction against the published `a2_groupfs__c46.csv` on all 24
cells; anything less than 24/24 means this file is wrong, not the bench.
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import DUFS_BENCH, INSCOPE, load_cells_cached          # noqa: E402
from spectral_utils.selectors.a2_groupfs import (                  # noqa: E402
    dufs_pf_cell_rng as cell_rng, dufs_pf_gates as _gates,
    DUFS_PF_RNG_DISCARD as N_RNG_DISCARD)


def bench_domains(path=DUFS_BENCH):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["cell"]: r["domain"] for r in csv.DictReader(fh)
                if r["variant"] == "a2.dufs_pf"}


def bench_chosen(path=DUFS_BENCH):
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["cell"]: [f for f in r["chosen"].split("|") if f]
                for r in csv.DictReader(fh) if r["variant"] == "a2.dufs_pf"}


def dufs_pf_gates(V, cell_key, domain, seed=0):
    """The seed-averaged gate vector `mu` this selector thresholds at zero.

    Thin wrapper over `spectral_utils.selectors.a2_groupfs.dufs_pf_gates`, which is
    where the RNG dance now lives so callers that cannot import this script (module
    name `common` collides with `scripts/upcr_study/common.py`) still get the SAME
    stream. `--verify` gates both.
    """
    return _gates(V, cell_rng(cell_key, domain, seed))


def dufs_pf_cols(V, cell_key, domain, seed=0):
    """Column indices `a2.dufs_pf` selects on this matrix.

    V is the matrix the selector sees; passing a TRANSFORMED V is the whole point.
    The row subsample depends only on n, so it is identical between base and
    config — the selection can only move because the columns moved."""
    p = np.asarray(V).shape[1]
    mu = dufs_pf_gates(V, cell_key, domain, seed)
    sel = sorted(int(i) for i in np.where(mu > 0.0)[0])
    if len(sel) < 3:                      # `_fallback` -> the full pool
        return list(range(p)), False
    return sel, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="reproduce a2_groupfs__c46.csv on all 24 cells")
    ap.add_argument("--cells", nargs="*", default=None)
    args = ap.parse_args()

    cells = load_cells_cached()
    dom, ref = bench_domains(), bench_chosen()
    keys = args.cells or list(INSCOPE)
    n_ok = 0
    print(f"{'cell':<32}{'ours':>6}{'bench':>7}{'jaccard':>9}  verdict")
    print("-" * 72)
    for ck in keys:
        cell = cells[ck]
        sel, _ = dufs_pf_cols(cell["V"], ck, dom[ck])
        ours = {cell["pool"][i] for i in sel}
        theirs = set(ref[ck])
        jac = len(ours & theirs) / max(len(ours | theirs), 1)
        ok = ours == theirs
        n_ok += ok
        print(f"{ck[:31]:<32}{len(ours):>6}{len(theirs):>7}{jac:>9.3f}  "
              f"{'EXACT' if ok else 'DIFFERS'}", flush=True)
    print(f"\nexact on {n_ok}/{len(keys)} cells")
    if args.verify and n_ok < len(keys):
        print("REPRODUCTION GATE FAILED — do not use this for arm A")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
