#!/usr/bin/env python
"""
c46_sweep_analysis.py — the honest (LOCO) read of the Step-194 sizes-3-5 sweep over
the 30-view pool, and the pre-registered stop rule for extending to sizes 3-6.

Everything here is computed from the enumeration npz files alone (no selector
re-runs). Two leave-one-cell-out tests, rotating over the 25 in-scope cells:

1. LOCO CONSENSUS SUBSET — for each fold, rank every enumerated subset by its
   mean AUROC over the 24 training cells (only masks scoreable on >= MIN_COVER
   training cells), take the winner, score it on the held-out cell. This is the
   honest version of "search the 30-view pool for a better fixed subset" — the
   Step-154 protocol, now on the enlarged pool. Compare vs GOOD_5 (size 5, in
   the enumeration) per cell.

2. LOCO PRUNE TEST — for each fold, derive the drop list from the 24 training
   cells only (absent from every training cell's top-100 AND max LOVO across
   training cells <= LOVO_TOL), then compare the held-out cell's best reachable
   AUROC inside the pruned pool vs the full pool. Both sides peek at the
   held-out labels equally (it is a ceiling-vs-ceiling comparison), so the
   delta isolates what pruning costs/gains.

STOP RULE (pre-registered in PROGRESS Step 194): if the LOCO consensus delta
vs GOOD_5 is <= +0.2pp, the sizes-3-6 extension (~4 days CPU) is NOT run and
the result is written up as negative.

Output: results/advisor_inscope/c46_loco_analysis.csv + console verdicts.
"""

import csv
import glob
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from spectral_utils.subset_sweep import CANONICAL_POOL, GOOD_5   # noqa: E402
from spectral_utils.feature_utils import FEAT_NAMES              # noqa: E402
from spectral_utils.subset_sweep import REPGRID_VIEWS            # noqa: E402
from inscope_cells import INSCOPE                                # noqa: E402

SWEEP = os.path.join(REPO, "results", "subset_sweep_c46")
AI = os.path.join(REPO, "results", "advisor_inscope")
POOL30 = list(FEAT_NAMES) + REPGRID_VIEWS
TOP_N = 100
LOVO_TOL = 0.0005      # 0.05pp — "banning this view costs nothing" on a training cell
MIN_COVER = 20         # a mask must be scoreable on >= this many training cells
STOP_RULE_PP = 0.2


def load_cells():
    cells = {}
    for c in INSCOPE:
        path = os.path.join(SWEEP, f"repgrid__{c}.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        mask = z["mask"].astype(np.uint64)
        auroc = z["auroc"].astype(float)
        ok = np.isfinite(auroc)
        cells[c] = dict(mask=mask[ok], auroc=auroc[ok],
                        lut=dict(zip(mask[ok].tolist(), auroc[ok].tolist())))
    return cells


def good5_mask():
    m = np.uint64(0)
    for f in GOOD_5:
        m |= np.uint64(1) << np.uint64(CANONICAL_POOL.index(f))
    return int(m)


def main():
    cells = load_cells()
    names = sorted(cells)
    print(f"loaded {len(names)}/25 in-scope enumerations "
          f"({len(next(iter(cells.values()))['mask'])} subsets typical)")

    g5m = good5_mask()
    g5 = {c: cells[c]["lut"].get(g5m) for c in names}
    n_g5 = sum(1 for v in g5.values() if v is not None)
    print(f"GOOD_5 mask found in {n_g5}/{len(names)} cells' enumerations")

    # ---- union mask table: mean AUROC per mask over scoreable cells ---------
    # (build once; per-fold means come from sums minus the held-out column)
    all_masks = {}
    for ci, c in enumerate(names):
        for m, a in cells[c]["lut"].items():
            rec = all_masks.setdefault(m, [0.0, 0, np.zeros(len(names))])
            rec[0] += a
            rec[1] += 1
            rec[2][ci] = a  # 0.0 elsewhere; coverage tracked separately
    masks = np.array(list(all_masks.keys()), dtype=np.uint64)
    sums = np.array([all_masks[int(m)][0] for m in masks])
    covers = np.array([all_masks[int(m)][1] for m in masks])
    per_cell_a = np.stack([all_masks[int(m)][2] for m in masks])   # (M, C)
    has = per_cell_a > 0.0   # enumeration AUROCs are anchor-oriented, > 0 in practice

    bit_of = {f: CANONICAL_POOL.index(f) for f in POOL30}

    # per-cell per-view stats for the prune test
    view_stats = {}
    for c in names:
        mask, auroc = cells[c]["mask"], cells[c]["auroc"]
        order = np.argsort(-auroc)
        topN = order[:TOP_N]
        best = float(auroc[order[0]])
        vs = {}
        for f in POOL30:
            b = np.uint64(1) << np.uint64(bit_of[f])
            hasv = (mask & b) != 0
            without = auroc[~hasv]
            vs[f] = dict(in_top=int(hasv[topN].sum()),
                         lovo=float(best - without.max()) if without.size else 0.0)
        view_stats[c] = dict(stats=vs, best=best)

    rows = []
    cons_deltas, prune_deltas, prune_sizes = [], [], []
    for hi, hold in enumerate(names):
        train_ix = [i for i in range(len(names)) if i != hi]

        # ---- 1. consensus subset from the 24 training cells -----------------
        cov_train = has[:, train_ix].sum(axis=1)
        sum_train = per_cell_a[:, train_ix].sum(axis=1)
        elig = cov_train >= MIN_COVER
        mean_train = np.where(elig, sum_train / np.maximum(cov_train, 1), -1.0)
        wi = int(np.argmax(mean_train))
        wmask = int(masks[wi])
        held_auc = cells[hold]["lut"].get(wmask)
        g5_auc = g5.get(hold)
        d_cons = (held_auc - g5_auc) if (held_auc is not None and g5_auc is not None) else None
        if d_cons is not None:
            cons_deltas.append(d_cons)

        # ---- 2. prune list from the 24 training cells -----------------------
        drop = []
        for f in POOL30:
            tin = [view_stats[names[i]]["stats"][f]["in_top"] for i in train_ix]
            tlo = [view_stats[names[i]]["stats"][f]["lovo"] for i in train_ix]
            if max(tin) == 0 and max(tlo) <= LOVO_TOL:
                drop.append(f)
        drop_mask = np.uint64(0)
        for f in drop:
            drop_mask |= np.uint64(1) << np.uint64(bit_of[f])
        hm, ha = cells[hold]["mask"], cells[hold]["auroc"]
        full_ceiling = float(ha.max())
        keep = (hm & drop_mask) == 0
        pruned_ceiling = float(ha[keep].max()) if keep.any() else float("nan")
        d_prune = pruned_ceiling - full_ceiling
        prune_deltas.append(d_prune)
        prune_sizes.append(len(drop))

        rows.append(dict(
            held_out=hold,
            consensus_subset="|".join(sorted(
                f for f in CANONICAL_POOL
                if wmask & (1 << CANONICAL_POOL.index(f)))),
            consensus_auroc_held=f"{held_auc:.4f}" if held_auc is not None else "",
            good5_auroc=f"{g5_auc:.4f}" if g5_auc is not None else "",
            delta_consensus_vs_good5=f"{d_cons:+.4f}" if d_cons is not None else "",
            n_pruned=len(drop),
            pruned_views="|".join(drop),
            ceiling_full=f"{full_ceiling:.4f}",
            ceiling_pruned=f"{pruned_ceiling:.4f}",
            delta_prune_ceiling=f"{d_prune:+.4f}",
        ))

    out_path = os.path.join(AI, "c46_loco_analysis.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    d = 100 * float(np.mean(cons_deltas))
    wn = sum(1 for x in cons_deltas if x > 1e-9)
    ln = sum(1 for x in cons_deltas if x < -1e-9)
    print(f"\nLOCO CONSENSUS (24-cell best subset -> held-out cell), {len(cons_deltas)} folds:")
    print(f"  mean delta vs GOOD_5 = {d:+.2f}pp  ({wn}W/{ln}L)")
    print(f"\nSTOP RULE (<= +{STOP_RULE_PP}pp => negative, do NOT extend to sizes 3-6): "
          f"{'EXTEND justified' if d > STOP_RULE_PP else 'NEGATIVE — do not extend'}")

    dp = 100 * float(np.mean(prune_deltas))
    print(f"\nLOCO PRUNE (drop list from 24 cells, ceiling on held-out): "
          f"mean drop-list size {np.mean(prune_sizes):.1f} views; "
          f"mean ceiling delta = {dp:+.3f}pp (0 = pruning costs nothing)")
    from collections import Counter
    dropped_all = Counter()
    for r in rows:
        for f in r["pruned_views"].split("|"):
            if f:
                dropped_all[f] += 1
    print("  views in the drop list in >= 20/25 folds (stable candidates):")
    for f, n in dropped_all.most_common():
        if n >= 20:
            print(f"    {f}: {n}/25 folds")
    print(f"\nwrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
