#!/usr/bin/env python
"""
nonmono_shape_test.py — is a view's label curve genuinely non-monotone, and can that
be detected WITHOUT labels? (Step 217, following Omri's objection to the C1 gate)

WHY THIS EXISTS
---------------
C1 (`nonmono_transform_bench.py --stage c1`) gated on the PER-FEATURE MEAN of
`nonmono_gain` over cells, and concluded nothing survived. Omri's objection: the
per-cell deep-dive pages (`results/action_items_jul2026/item1_failure_deepdive/`)
plainly show views whose two class-conditional densities are not one-shape-shifted,
and several show two humps. Averaging a cell-specific effect over 24 cells is the
wrong aggregation, and it hid this: the largest per-cell gains carry ZERO fold
inflation and sit on the biggest cells in the grid —

    se_squad_v2_llama8b   / pe_mean     +0.0983  (n=2933)
    semenergy_..._qwen3_8b/ epr_energy  +0.0936  (n=4392)
    truthfulqa_llama8b    / rpdi        +0.0715  (n=7633)
    se_nq_open_llama8b    / rpdi        +0.0582  (n=8460)

So the question is not "is feature F non-monotone everywhere" (it is not) but
"is this CELL x VIEW non-monotone, and can we tell without the answer key".

TWO INSTRUMENTS, AND WHY NEITHER IS `nonmono_gain`
---------------------------------------------------
`nonmono_gain` is not a fair test in either direction. Its non-monotone side is a
CROSS-FITTED 4-bin map; its monotone side is a SINGLE GLOBAL RAW AUROC that was
never cross-fitted. So it charges estimation noise to one side only, and 4 quantile
bins is too coarse to resolve a U-shape anyway. (It also folds `max(p, 1-p)` per
fold — `gap_ladder.py:64,220` — a separate one-sided noise floor.)

  TEST 1 — FAIR SHAPE TEST (label-derived, the ground truth here).
      Both sides are fitted on the same training folds and scored on the same held-out
      folds, so estimation noise is charged symmetrically:
        monotone side     isotonic regression of y on x (the BEST monotone function,
                          not merely the linear/raw one)
        unconstrained side K-bin quantile mean map, K=10 (fine enough to see a U)
      `shape_gain = AUROC(unconstrained) - AUROC(isotonic)`, raw AUROC on both, no
      folding anywhere. Calibrated against a LABEL-PERMUTATION null so "how big is
      big" is measured rather than asserted.

  TEST 2 — TWO-PEAK TEST (LABEL-FREE, the deployable one).
      Omri's actual proposal: the shape of the MARGINAL distribution is visible
      without any answer key, so if "two peaks" predicts where the label curve bends,
      the transform can be applied per cell x view label-free. Measured three ways
      because each fails differently on this data:
        n_modes    peaks in a Gaussian KDE, counted with a prominence floor -- the
                   literal "two peaks", non-parametric
        gmm_dbic   BIC(1-component Gaussian) - BIC(2-component); > 0 favours two modes
        bc         bimodality coefficient (skew^2 + 1) / (kurtosis + 3(n-1)^2 /
                   ((n-2)(n-3))); > 5/9 is the conventional flag
      `spike_frac` (largest share of mass at a single value) is reported alongside,
      because several views here are DISCRETE (`cusum_shift_idx` is an index,
      `trace_length` a count) and a degenerate spike is not a usable second mode.

THE BRIDGE IS THE POINT. A cell x view can only be transformed in deployment if the
decision is label-free, so the last section asks directly whether Test 2 predicts
Test 1. If it does not, the non-monotonicity is real but unreachable, and that is a
finding rather than a failure.

Usage:
    python scripts/nonmono_shape_test.py
    python scripts/nonmono_shape_test.py --n-perm 50 --seeds 5
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, kurtosis, skew, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from inscope_bench_common import load_cells, assert_good6            # noqa: E402
from inscope_cells import INSCOPE                                    # noqa: E402

OUT = os.path.join(REPO, "results", "nonmono_transform")
N_BINS = 10
MIN_N = 150


# ── Test 1: the fair shape test ───────────────────────────────────────────────
def _binmap(xtr, ytr, xte, k=N_BINS):
    """Unconstrained quantile-bin mean map, fitted on train, applied to test."""
    edges = np.unique(np.quantile(xtr, np.linspace(0, 1, k + 1)))
    if len(edges) < 3:
        return None
    base = float(ytr.mean())
    tb = np.digitize(xtr, edges[1:-1])
    rates = {b: (float(ytr[tb == b].mean()) if np.any(tb == b) else base)
             for b in range(len(edges) - 1)}
    return np.array([rates.get(b, base) for b in np.digitize(xte, edges[1:-1])])


def _isomap(xtr, ytr, xte):
    """Best MONOTONE function of x, direction chosen on train only."""
    inc = bool(np.corrcoef(xtr, ytr)[0, 1] >= 0)
    ir = IsotonicRegression(increasing=inc, out_of_bounds="clip")
    ir.fit(xtr, ytr)
    return ir.predict(xte)


def shape_gain(x, y, seeds=(0, 1, 2), k=N_BINS):
    """AUROC(unconstrained bin map) - AUROC(isotonic), both cross-fitted identically.

    Returns (mean gain, across-seed sd). NaN when the cell is too small or the view
    is degenerate. No `max(a, 1-a)` anywhere: the isotonic side resolves direction on
    TRAIN, and the bin map carries its own direction by construction.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    if len(y) < MIN_N or len(np.unique(y)) < 2 or np.std(x) < 1e-12:
        return float("nan"), float("nan")
    per_seed = []
    for s in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42 + s)
        gu, gi = [], []
        for tr, te in skf.split(x.reshape(-1, 1), y):
            if len(np.unique(y[te])) < 2:
                continue
            u = _binmap(x[tr], y[tr], x[te], k)
            if u is None:
                continue
            i = _isomap(x[tr], y[tr], x[te])
            gu.append(0.5 if np.std(u) < 1e-12 else roc_auc_score(y[te], u))
            gi.append(0.5 if np.std(i) < 1e-12 else roc_auc_score(y[te], i))
        if gu:
            per_seed.append(float(np.mean(gu) - np.mean(gi)))
    if not per_seed:
        return float("nan"), float("nan")
    return float(np.mean(per_seed)), float(np.std(per_seed))


def perm_null(x, y, n_perm, rng, k=N_BINS):
    """95th percentile of `shape_gain` under label permutation — the honest bar.

    Under a permuted label there is no shape at all, so anything the statistic
    reports is the flexibility of a 10-bin map over an isotonic fit. One seed per
    permutation keeps this affordable; the quantile is over permutations."""
    vals = []
    for _ in range(n_perm):
        g, _ = shape_gain(x, rng.permutation(y), seeds=(0,), k=k)
        if np.isfinite(g):
            vals.append(g)
    if not vals:
        return float("nan")
    return float(np.percentile(vals, 95))


# ── Test 2: label-free two-peak measures ──────────────────────────────────────
def kde_peaks(x, grid=512, min_prominence=0.05):
    """(number of modes, relative prominence of the 2nd mode) from a Gaussian KDE.

    This is the literal form of the observation that started this test — "some of
    them show 2 peaks" — and it is fully non-parametric and label-free. A peak counts
    only if its prominence is at least `min_prominence` of the peak density, so a
    ripple on a shoulder is not a mode.

    Chosen over Hartigan's dip because scipy has no dip test and a first attempt at
    implementing it here returned 0 on a textbook bimodal mixture (the GCM/LCM
    bisection was degenerate); over a Gaussian mixture BIC because that assumes the
    very shape it is asked to detect. `gmm_dbic` is still reported beside it as a
    parametric second opinion.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 50 or np.std(x) < 1e-12:
        return float("nan"), float("nan")
    try:
        kde = gaussian_kde(x)
    except Exception:
        return float("nan"), float("nan")
    g = np.linspace(x.min(), x.max(), grid)
    d = kde(g)
    if not np.isfinite(d).all() or d.max() <= 0:
        return float("nan"), float("nan")
    pk, props = find_peaks(d, prominence=min_prominence * d.max())
    if len(pk) == 0:
        return 0.0, 0.0
    order = np.argsort(-props["prominences"])
    second = (props["prominences"][order[1]] / d.max()) if len(pk) > 1 else 0.0
    return float(len(pk)), float(second)


def gmm_dbic(x, seed=0):
    """BIC(1 component) - BIC(2 components). Positive favours two modes."""
    x = np.asarray(x, float).reshape(-1, 1)
    if len(x) < 50 or np.std(x) < 1e-12:
        return float("nan")
    x = (x - x.mean()) / x.std()
    try:
        b1 = GaussianMixture(1, random_state=seed, n_init=2).fit(x).bic(x)
        b2 = GaussianMixture(2, random_state=seed, n_init=2).fit(x).bic(x)
    except Exception:
        return float("nan")
    return float(b1 - b2)


def bimodality_coefficient(x):
    x = np.asarray(x, float)
    n = len(x)
    if n < 8 or np.std(x) < 1e-12:
        return float("nan")
    g = skew(x)
    k = kurtosis(x, fisher=True)
    return float((g ** 2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def spike_frac(x):
    """Largest share of the mass sitting on one exact value — a discreteness flag,
    not a mode. `cusum_shift_idx` and `trace_length` are counts."""
    _, c = np.unique(np.asarray(x, float), return_counts=True)
    return float(c.max() / len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--screen", type=float, default=0.02,
                    help="only permutation-test pairs whose shape_gain exceeds this")
    args = ap.parse_args()

    cells = load_cells()
    ok, macro = assert_good6(cells, verbose=True)
    if not ok:
        raise SystemExit(f"validity gate failed: GOOD_6 macro {macro:.4f}")

    seeds = tuple(range(args.seeds))
    rng = np.random.default_rng(0)
    rows = []
    for ck in INSCOPE:
        cell = cells[ck]
        y = cell["labels"]
        for j, f in enumerate(cell["pool"]):
            x = cell["V"][:, j]
            g, sd = shape_gain(x, y, seeds=seeds)
            nmodes, prom2 = kde_peaks(x)
            r = dict(cell=ck, feature=f, n=len(y),
                     shape_gain=round(g, 4) if np.isfinite(g) else "",
                     shape_gain_sd=round(sd, 4) if np.isfinite(sd) else "",
                     n_modes=nmodes, mode2_prominence=round(prom2, 4),
                     gmm_dbic=round(gmm_dbic(x), 1),
                     bimodality_coef=round(bimodality_coefficient(x), 4),
                     spike_frac=round(spike_frac(x), 4),
                     perm_null_p95="", exceeds_null="")
            if np.isfinite(g) and g > args.screen:
                nul = perm_null(x, y, args.n_perm, rng)
                r["perm_null_p95"] = round(nul, 4) if np.isfinite(nul) else ""
                r["exceeds_null"] = int(np.isfinite(nul) and g > nul)
            rows.append(r)
        print(f"  {ck:<34} {len(cell['pool'])} views", flush=True)

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "shape_test.csv")
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsaved {p}  ({len(rows)} rows)")

    # ── what survives the permutation null ────────────────────────────────────
    surv = [r for r in rows if r["exceeds_null"] == 1]
    surv.sort(key=lambda r: -float(r["shape_gain"]))
    print(f"\n{'='*94}\nTEST 1 — cell x view pairs whose shape_gain BEATS its own "
          f"label-permutation null ({len(surv)} of {len(rows)})\n{'='*94}")
    print(f"{'cell':<32}{'feature':<22}{'gain':>8}{'sd':>7}{'null95':>9}"
          f"{'n':>7}{'spike':>7}")
    print("-" * 94)
    for r in surv[:30]:
        print(f"{r['cell']:<32}{r['feature']:<22}{float(r['shape_gain']):>+8.4f}"
              f"{float(r['shape_gain_sd']):>7.4f}{float(r['perm_null_p95']):>9.4f}"
              f"{r['n']:>7}{r['spike_frac']:>7.2f}")

    # ── does the LABEL-FREE shape predict the label-derived one? ──────────────
    ok_rows = [r for r in rows if r["shape_gain"] != ""
               and np.isfinite(r["n_modes"])]
    g = np.array([float(r["shape_gain"]) for r in ok_rows])
    print(f"\n{'='*94}\nTEST 2 — does a label-free two-peak measure PREDICT the "
          f"non-monotonicity? (n={len(ok_rows)} pairs)\n{'='*94}")
    for name in ("n_modes", "mode2_prominence", "gmm_dbic",
                 "bimodality_coef", "spike_frac"):
        v = np.array([float(r[name]) for r in ok_rows])
        m = np.isfinite(v) & np.isfinite(g)
        rho, pv = spearmanr(v[m], g[m])
        print(f"  Spearman(shape_gain, {name:<16}) = {rho:+.3f}  p={pv:.2e}  "
              f"n={int(m.sum())}")

    hit = set((r["cell"], r["feature"]) for r in surv)
    for name, thr, lab in (("n_modes", 1.5, "KDE shows >= 2 peaks"),
                           ("mode2_prominence", 0.15, "2nd peak prominence > 0.15"),
                           ("gmm_dbic", 0.0, "BIC favours 2 components"),
                           ("bimodality_coef", 5 / 9, "BC > 5/9")):
        flag = [(r["cell"], r["feature"]) for r in ok_rows
                if np.isfinite(float(r[name])) and float(r[name]) > thr]
        if not flag:
            continue
        tp = len(set(flag) & hit)
        prec = tp / len(flag)
        rec = tp / len(hit) if hit else float("nan")
        print(f"  {lab:<28} flags {len(flag):>4} pairs -> precision {prec:.3f}, "
              f"recall {rec:.3f}   (base rate {len(hit)/len(ok_rows):.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
