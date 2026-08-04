"""
exp12_what_separates_good_features.py — does a feature's TRUE correlation with
correctness pick the good feature sets?

WHY THIS EXISTS
---------------
Step 220 priced every channel in U-PCR by letting the labels do that step perfectly.
Three of four are worth zero. The fourth — which features get kept — is worth +1.48pp
held out, CI [+0.97, +2.03]. But the good sets overlap U-PCR's own |rho_hat| ranking at
0.340 against a random baseline of 0.360, i.e. AT CHANCE, and keeping fewer features by
that ranking loses at every size.

So the live question is: what separates the good features, if not their ESTIMATED
correlation with correctness? This script asks the one question that decides the branch:

    is the correlation the WRONG QUANTITY, or the RIGHT QUANTITY BADLY ESTIMATED?

Rank each test set's features by their TRUE correlation with correctness — computed on
the same selection half the greedy search gets, so the two are equally label-privileged —
take the good set's own size, and score held out.

  reproduces the good set  -> the quantity is right, the estimator is bad
                              -> build learned per-pair weights on U-PCR's system
  does not                 -> the quantity is wrong, no estimator of it can help
                              -> build DUFS-supplies-the-ranking

A CORRECTION TO THE HANDOFF THIS SCRIPT INHERITS
------------------------------------------------
`HANDOFF_upcr_selection.md` says to rank against the HELD-OUT good sets in
`r2_oracle_mask.csv`. Those are not on disk: `exp10_channel_ceilings.py:443` writes
`oracle_cols` from `cols_in`, the IN-SAMPLE greedy. The split-half greedy's own choices
(`cols_a`, exp10:405) are scored on half B and then discarded. So this script re-runs the
split-half selection rather than reading the CSV, and persists the held-out sets
(`splits.csv:greedy_cols`) so the next thing that needs them does not have to.

That is not just bookkeeping. Ranking by a correlation computed on ALL the rows against a
set chosen from HALF of them would give the ranking strictly more information than the
search it is being compared to.

THE SECOND READING OF A NULL, AND HOW IT IS SEPARATED
-----------------------------------------------------
The greedy chooses a SET, with interactions. Any per-feature ranking is marginal. So a
null has two readings: the correlation is the wrong quantity, or NO marginal ranking can
reproduce a set-level optimum. The second does not condemn DUFS (its gates are trained
jointly on the sample graph and can express redundancy) but it does condemn any better
estimate of a per-feature rho. `mean_abs_corr_greedy` vs `mean_abs_corr_topk` separates
them: if the greedy prefers individually WEAKER features, what it is exploiting is
interaction, not marginal strength.

PRE-REGISTERED READING (written before the run)
-----------------------------------------------
  right quantity : recovers a majority of the greedy's held-out gain AND overlaps the
                   good set clearly above the random-overlap baseline
  wrong quantity : lands near the random floor AND overlaps at chance
  anything else  : reported as in between, not rounded to a branch

GATES
-----
  GOOD_6 0.7733 and U-PCR + sign(rho) 0.7741 (S.validity_check, derived_arm_gate), and
  this script's OWN reproduction of the ceiling: the greedy arm must land inside
  [+0.97, +2.03] on its own splits. Splits are NOT bit-identical to exp10's (that rng is
  consumed interleaved with the permutation null), so the ceiling reproduction is what
  establishes the splits are equivalent.

Run:  python scripts/upcr_study/exp12_what_separates_good_features.py [--cells N]
Out:  results/upcr_study/12_what_separates_good_features/
"""
import argparse
import os
import sys
import zlib

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as S                                                  # noqa: E402

from exp10_channel_ceilings import (                                # noqa: E402
    FIT, N_SPLITS, derive_cell, derived_arm_gate, fit_cols, _greedy_mask,
    _zsc, _all_fields,
)
from spectral_utils.subset_sweep import ALL_SIGNS                   # noqa: E402
from spectral_utils.upcr import upcr_fit                            # noqa: E402

SIZES = (6, 8, 10, 12, 14, 16)   # so a null is not a size artefact
N_RANDOM = 25                    # random subsets per split for the floor
N_OVERLAP_DRAWS = 2000           # draws for the random-overlap baseline

CEILING_LO_PP, CEILING_HI_PP = 0.97, 2.03   # Step 220's held-out interval


# ---------------------------------------------------------------------------
def true_corr(V, y):
    """Each feature's TRUE correlation with correctness. Sign-free: the ranking is by
    magnitude, and U-PCR's polarity step owns the direction (which Step 220 priced at
    -0.06pp anyway)."""
    y = np.asarray(y, dtype=float)
    yc = y - y.mean()
    if yc.std() < 1e-12:
        return np.zeros(V.shape[1])
    out = np.zeros(V.shape[1])
    for j in range(V.shape[1]):
        v = V[:, j] - V[:, j].mean()
        d = v.std() * yc.std()
        out[j] = float(v @ yc) / (len(y) * d) if d > 1e-12 else 0.0
    return out


def topk(stat, k):
    return [int(j) for j in np.argsort(-np.abs(stat))[:k]]


def overlap_vs_random(want, got, m, k, rng):
    """Overlap fraction, and the random-overlap baseline for the SAME k and pool size.
    Step 220's lesson: a null must be checked against chance before clearing it means
    anything. |rho_hat|'s 0.340 looked like a weak signal until the baseline came in
    at 0.360."""
    if not k:
        return float("nan"), float("nan")
    ov = len(set(want) & set(got)) / k
    base = float(np.mean([len(set(want) & set(rng.choice(m, k, replace=False))) / k
                          for _ in range(N_OVERLAP_DRAWS)]))
    return ov, base


# ---------------------------------------------------------------------------
def run_cell(ck, cell, dcell, rng):
    """One test set. Every arm on the SAME five splits."""
    V0 = cell["V"]                                    # hand-oriented, full-cell z-scored
    y = dcell["labels"]
    n, m = V0.shape
    hand = np.array([ALL_SIGNS.get(f, +1) for f in cell["pool"]], dtype=float)
    pool = cell["pool"]

    split_rows = []
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
            continue                                  # no legitimate start; skip
        start_a = list(np.where(res_a.keep)[0])

        # --- the good set: greedy on half A's labels, scored on half B -------------
        cols_greedy, _, _, conv = _greedy_mask(
            lambda c: fit_cols(ca, c, exclusion=False)[0], m, start_a)
        k = len(cols_greedy)
        auc_greedy = fit_cols(cb, cols_greedy, exclusion=False)[0]
        auc_dep = fit_cols(cb, range(m))[0]

        # --- the rankings, all computed on half A only -----------------------------
        corr_a = true_corr(ca["V"], ca["labels"])
        rho_a = np.asarray(res_a.rho_hat_full, dtype=float)

        row = {"cell": ck, "rep": rep, "m": m, "k": k,
               "greedy_converged": bool(conv),
               "n_keep_deployed_halfA": len(start_a),
               "auroc_deployed": auc_dep, "auroc_greedy": auc_greedy,
               "greedy_cols": "|".join(pool[c] for c in cols_greedy)}

        for name, stat in (("truecorr", corr_a), ("rho", rho_a)):
            cols = topk(stat, k)
            row[f"auroc_{name}_ownk"] = fit_cols(cb, cols, exclusion=False)[0]
            ov, base = overlap_vs_random(cols_greedy, cols, m, k, rng)
            row[f"overlap_{name}"] = ov
            row[f"overlap_{name}_random_baseline"] = base
            row[f"{name}_cols_ownk"] = "|".join(pool[c] for c in cols)

        for kk in SIZES:
            if kk > m:
                continue
            row[f"auroc_truecorr_top{kk}"] = fit_cols(
                cb, topk(corr_a, kk), exclusion=False)[0]

        # --- the floor: random subsets at the good set's own size ------------------
        rnd = [fit_cols(cb, sorted(rng.choice(m, k, replace=False)),
                        exclusion=False)[0] for _ in range(N_RANDOM)]
        row["auroc_random_ownk"] = float(np.nanmean(rnd))

        # --- interaction diagnostic: is the greedy picking WEAKER features? --------
        a = np.abs(corr_a)
        row["mean_abs_corr_greedy"] = float(np.mean(a[cols_greedy]))
        row["mean_abs_corr_topk"] = float(np.mean(a[topk(corr_a, k)]))
        row["mean_abs_corr_pool"] = float(np.mean(a))
        split_rows.append(row)

    return split_rows


def collapse(split_rows):
    """Per test set: mean over its splits. The independent unit is the TEST SET, never
    the split — pooling splits is pseudo-replication (common.wl_wilcoxon's docstring)."""
    by_cell = {}
    for r in split_rows:
        by_cell.setdefault(r["cell"], []).append(r)
    out = []
    for ck, rs in by_cell.items():
        row = {"cell": ck, "n_splits": len(rs), "m": rs[0]["m"],
               "k_mean": float(np.mean([r["k"] for r in rs])),
               "n_keep_deployed_halfA": float(np.mean(
                   [r["n_keep_deployed_halfA"] for r in rs])),
               "n_splits_unconverged": int(sum(1 for r in rs
                                               if not r["greedy_converged"]))}
        keys = [k for k in rs[0] if k.startswith(("auroc_", "overlap_",
                                                  "mean_abs_corr_"))]
        for k in keys:
            vals = [r[k] for r in rs if k in r and np.isfinite(r.get(k, np.nan))]
            row[k] = float(np.mean(vals)) if vals else float("nan")
        out.append(row)
    return out


def paired(rows, a, b, name):
    x = np.array([r.get(a, np.nan) for r in rows], float)
    y = np.array([r.get(b, np.nan) for r in rows], float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    d = (y - x) * 100.0
    rng = np.random.default_rng(0)
    boot = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(20000)]
    p = float(wilcoxon(x, y).pvalue) if len(d) >= 5 and np.any(d != 0) else float("nan")
    out = {"mean_delta_pp": float(d.mean()), "median_delta_pp": float(np.median(d)),
           "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
           "n": int(len(d)), "p": p}
    print(f"  {name:52s} {out['mean_delta_pp']:+.2f}pp  CI "
          f"[{out['ci95'][0]:+.2f}, {out['ci95'][1]:+.2f}]  "
          f"{out['wins']}W/{out['losses']}L  p={out['p']:.4f}")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=0,
                    help="smoke test: run only the first N test sets (no gates)")
    args = ap.parse_args()

    out = S.outdir("12_what_separates_good_features")
    cells = S.load()
    S.validity_check(cells)
    dcells = {k: derive_cell(c) for k, c in cells.items()}
    derived_arm_gate(dcells)

    keys = list(cells)[: args.cells] if args.cells else list(cells)
    split_rows = []
    for i, ck in enumerate(keys, 1):
        # Per-cell deterministic seed: every arm inside a cell shares its splits, and
        # the run is reproducible without depending on iteration order (exp10's single
        # shared rng is consumed interleaved with its permutation null, so its splits
        # cannot be reconstructed here anyway).
        rng = np.random.default_rng(zlib.crc32(ck.encode()) % (2 ** 32))
        rs = run_cell(ck, cells[ck], dcells[ck], rng)
        split_rows += rs
        if rs:
            g = float(np.nanmean([r["auroc_greedy"] for r in rs]))
            d = float(np.nanmean([r["auroc_deployed"] for r in rs]))
            t = float(np.nanmean([r["auroc_truecorr_ownk"] for r in rs]))
            ov = float(np.nanmean([r["overlap_truecorr"] for r in rs]))
            bs = float(np.nanmean([r["overlap_truecorr_random_baseline"] for r in rs]))
            print(f"[{i:2d}/{len(keys)}] {S.plain_cell(ck)[:32]:32s} "
                  f"k={np.mean([r['k'] for r in rs]):4.1f}  deployed {d:.4f} | "
                  f"good set {g:.4f} ({(g-d)*100:+.2f}pp) | true corr {t:.4f} "
                  f"({(t-d)*100:+.2f}pp) | overlap {ov:.3f} (random {bs:.3f})",
                  flush=True)
        else:
            print(f"[{i:2d}/{len(keys)}] {S.plain_cell(ck)[:32]:32s}  NO USABLE SPLITS",
                  flush=True)

    S.save_csv(os.path.join(out, "splits.csv"), split_rows, _all_fields(split_rows))
    rows = collapse(split_rows)
    S.save_csv(os.path.join(out, "per_cell.csv"), rows, _all_fields(rows))

    print("\n=== held out on half B, paired over test sets, vs the deployed full pool ===")
    summary = {"n_cells": len(rows), "n_splits_per_cell": N_SPLITS,
               "arms": {}, "PROVENANCE": (
                   "Pre-registered in this file's docstring before the run. The good "
                   "set (greedy) and every ranking see the SAME half A labels.")}
    summary["arms"]["good_set_ceiling"] = paired(
        rows, "auroc_deployed", "auroc_greedy", "the good set (the ceiling)")
    summary["arms"]["true_correlation_own_size"] = paired(
        rows, "auroc_deployed", "auroc_truecorr_ownk", "true correlation, good set's size")
    summary["arms"]["estimated_correlation_own_size"] = paired(
        rows, "auroc_deployed", "auroc_rho_ownk", "estimated correlation, same size")
    summary["arms"]["random_own_size"] = paired(
        rows, "auroc_deployed", "auroc_random_ownk", "random subsets, same size (floor)")
    for kk in SIZES:
        key = f"auroc_truecorr_top{kk}"
        if any(np.isfinite(r.get(key, np.nan)) for r in rows):
            summary["arms"][f"true_correlation_top{kk}"] = paired(
                rows, "auroc_deployed", key, f"true correlation, top {kk}")

    print("\n=== does the true correlation reproduce the good set? ===")
    ov = np.array([r["overlap_truecorr"] for r in rows], float)
    bs = np.array([r["overlap_truecorr_random_baseline"] for r in rows], float)
    ov_r = np.array([r["overlap_rho"] for r in rows], float)
    bs_r = np.array([r["overlap_rho_random_baseline"] for r in rows], float)
    summary["overlap"] = {
        "true_correlation": float(np.nanmean(ov)),
        "true_correlation_random_baseline": float(np.nanmean(bs)),
        "estimated_correlation": float(np.nanmean(ov_r)),
        "estimated_correlation_random_baseline": float(np.nanmean(bs_r)),
        "n_cells_true_above_own_baseline": int(np.nansum(ov > bs)),
    }
    print(f"  true correlation      overlap {np.nanmean(ov):.3f}  "
          f"random {np.nanmean(bs):.3f}  ({int(np.nansum(ov > bs))}/{len(rows)} "
          f"test sets above their own baseline)")
    print(f"  estimated correlation overlap {np.nanmean(ov_r):.3f}  "
          f"random {np.nanmean(bs_r):.3f}")

    print("\n=== is the good set made of individually STRONG features? ===")
    g = np.array([r["mean_abs_corr_greedy"] for r in rows], float)
    t = np.array([r["mean_abs_corr_topk"] for r in rows], float)
    p = np.array([r["mean_abs_corr_pool"] for r in rows], float)
    summary["marginal_strength"] = {
        "mean_abs_corr_good_set": float(np.nanmean(g)),
        "mean_abs_corr_top_k": float(np.nanmean(t)),
        "mean_abs_corr_whole_pool": float(np.nanmean(p)),
        "n_cells_good_set_weaker_than_topk": int(np.nansum(g < t)),
    }
    print(f"  good set {np.nanmean(g):.4f} | top-k by true corr {np.nanmean(t):.4f} | "
          f"whole pool {np.nanmean(p):.4f}   "
          f"({int(np.nansum(g < t))}/{len(rows)} test sets: good set is WEAKER)")

    # --- the reproduction gate ------------------------------------------------
    ceil = summary["arms"]["good_set_ceiling"]["mean_delta_pp"]
    if not args.cells:
        ok = CEILING_LO_PP <= ceil <= CEILING_HI_PP
        summary["ceiling_reproduction_gate"] = {
            "ceiling_pp": ceil, "interval": [CEILING_LO_PP, CEILING_HI_PP],
            "pass": bool(ok)}
        print(f"\nCEILING REPRODUCTION GATE: {ceil:+.2f}pp against Step 220's "
              f"[{CEILING_LO_PP:+.2f}, {CEILING_HI_PP:+.2f}] -> "
              f"{'PASS' if ok else 'FAIL'}")
        if not ok:
            print("  These splits do not reproduce the measured ceiling. Every arm "
                  "below is scored against a target this run did not recover — do not "
                  "read the branch off it.")

    S.save_json(os.path.join(out, "summary.json"), summary)
    print(f"\nWrote -> {out}")


if __name__ == "__main__":
    main()
