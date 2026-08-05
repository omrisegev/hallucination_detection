#!/usr/bin/env python
"""
answer_span_repair.py — rebuild the run-on cells over the answer span, in place.

Rebuilds every cell in `spectral_utils.answer_span.RUNON_CELLS` from its raw pkl with
`crop=True` and patches the result into `local_cache/repgrid_cells.pkl`, leaving every
other cell byte-identical. The pre-repair file is preserved as
`local_cache/repgrid_cells.precrop.pkl` so both sides of the comparison stay loadable.

WHY IN PLACE. The cropped features are the correct ones — they are computed over the
same span the GRADER reads (`first_answer_line`), which the un-cropped features were
not. Keeping the defective version as the canonical cache would mean every downstream
report keeps re-measuring a generation defect. The backup exists for the before/after
table, not as a fallback.

Usage:
    python scripts/answer_span_repair.py            # repair + verify
    python scripts/answer_span_repair.py --dry-run  # report, write nothing
"""
import argparse
import glob
import os
import pickle
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_repgrid_featcache import build_cell                     # noqa: E402
from spectral_utils.answer_span import RUNON_CELLS                 # noqa: E402
from spectral_utils.subset_sweep import CANONICAL_POOL             # noqa: E402

CACHE = os.path.join(REPO, "local_cache", "repgrid_cells.pkl")
BACKUP = os.path.join(REPO, "local_cache", "repgrid_cells.precrop.pkl")


def raw_pkl(cell):
    p = sorted(glob.glob(os.path.join(REPO, "cache", "repgrid", cell, "raw_*.pkl")))
    if not p:
        p = sorted(glob.glob(os.path.join(REPO, "cache", "repgrid", cell, "*.pkl")))
    return p[0] if p else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(CACHE, "rb") as f:
        cells = pickle.load(f)
    if not os.path.exists(BACKUP) and not args.dry_run:
        shutil.copy2(CACHE, BACKUP)
        print(f"backed up -> {BACKUP}")

    for ck in RUNON_CELLS:
        pk = raw_pkl(ck)
        if pk is None:
            raise SystemExit(f"{ck}: no raw pkl under cache/repgrid/")
        before = cells.get(ck)
        built = build_cell(pk, crop=True)
        if built is None:
            raise SystemExit(f"{ck}: cropped build produced <20 usable rows")
        fd, y, pid = built

        keep_b = sorted(set(before["feats"]) & set(CANONICAL_POOL)) if before else []
        keep_a = sorted(set(fd) & set(CANONICAL_POOL))
        print(f"\n=== {ck}")
        print(f"  rows      {len(before['labels']) if before else '-':>6} -> {len(y):>6}")
        print(f"  pos_rate  {float(np.mean(before['labels'])) if before else float('nan'):>6.4f}"
              f" -> {float(np.mean(y)):>6.4f}")
        print(f"  pool      {len(keep_b):>6} -> {len(keep_a):>6}")
        print(f"  dropped:  {sorted(set(keep_b) - set(keep_a))}")
        print(f"  kept:     {keep_a}")

        meta = dict(before or {})
        meta.update({"feats": fd, "labels": y, "problem_id": pid,
                     "acc": float(np.mean(y)), "answer_cropped": True})
        cells[ck] = meta

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    with open(CACHE, "wb") as f:
        pickle.dump(cells, f)
    print(f"\nwrote {CACHE}")

    # verification: every non-repaired cell must be identical to the backup
    with open(BACKUP, "rb") as f:
        old = pickle.load(f)
    with open(CACHE, "rb") as f:
        new = pickle.load(f)
    bad = []
    if set(old) != set(new):
        bad.append(f"cell roster changed: {set(old) ^ set(new)}")
    for ck in sorted(set(old) & set(new)):
        if ck in RUNON_CELLS:
            continue
        a, b = old[ck], new[ck]
        if set(a["feats"]) != set(b["feats"]) or not np.array_equal(a["labels"], b["labels"]):
            bad.append(f"{ck}: schema/labels changed")
            continue
        for k in a["feats"]:
            if not np.array_equal(a["feats"][k], b["feats"][k]):
                bad.append(f"{ck}/{k}: values changed")
                break
    if bad:
        print("\nVERIFY FAILED — the repair touched cells it should not have:")
        for m in bad:
            print("  " + m)
        sys.exit(1)
    print(f"VERIFY OK — {len(set(old)) - len(RUNON_CELLS)} untouched cells are "
          f"bit-identical to the pre-repair cache.")


if __name__ == "__main__":
    main()
