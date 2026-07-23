#!/usr/bin/env python
"""
pool_variants_experiment.py — can we improve the algorithm just by offering it a better
pool / better fixed subsets? (Step 193h)

Two things at once, both CPU-local and both over the 25 in-scope cells:

  A. NAMED POOL VARIANTS — run the trace-based Gated-Laplacian selector (`a2.dufs`) on a
     handful of hand-motivated pools rather than the nested top-N of
     pool_size_experiment.py. Step 193f already showed pool SIZE barely matters
     (30/24/20/16 all within 0.11 pp); this asks whether pool COMPOSITION does.

  B. NEW FIXED SUBSETS — score subsets that have never been offered to the scorer at all.
     `topk_tail_mass` and `renyi_entropy_2` rank #1 and #5 of 30 by informativeness yet
     appear in NONE of GOOD_5/GOOD_6/top_macro_5/consensus_4/GOOD_5+{spilled,energy,logprob}.
     A cheap, never-run check.

Baselines to beat: GOOD_6 = 0.7594, a2.dufs on the full 30-view pool = 0.7502.

CAVEAT: pools built from labelled per-cell AUROC (informativeness, winning-subset union)
carry corpus-level label information — in-sample upper bounds, same asymmetry as GOOD_6.
Pools defined by feature FAMILY (logprob/energy/spectral) are structural and label-free.

Usage:
    python scripts/pool_variants_experiment.py [--only-subsets] [--only-pools]
Writes: results/advisor_inscope/pool_variants_{pools,subsets}.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.subset_sweep import (
    iter_cells, prepare_cell, CANONICAL_POOL, GOOD_5, GOOD_6,
)
from spectral_utils.selector_bench import eval_subset_flex, UnlabeledCell
from spectral_utils.selectors import get_selector
from inscope_cells import INSCOPE, GROUP
from pool_size_experiment import informativeness_ranking

AI = os.path.join(REPO, "results", "advisor_inscope")
DD = os.path.join(REPO, "local_cache")

# ── feature families (structural, label-free definitions) ─────────────────────────────
LOGPROB = ["mean_top1_logprob", "logprob_margin", "mean_logprob_entropy"]
LOGPROB_EXT = ["varentropy", "renyi_entropy_2", "topk_tail_mass"]
ENERGY = ["epr_energy", "min_energy", "sw_var_peak_energy", "cusum_max_energy"]
SPILLED = ["epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled", "min_spilled"]
DEAD = ["pe_mean", "stft_spectral_entropy"]          # only views within 0.08 of chance

# ── B: fixed subsets, including ones never offered before ─────────────────────────────
SUBSETS = {
    "GOOD_5": GOOD_5,
    "GOOD_6": GOOD_6,
    "GOOD_6+topk": GOOD_6 + ["topk_tail_mass"],
    "GOOD_6+renyi": GOOD_6 + ["renyi_entropy_2"],
    "GOOD_6+topk+renyi": GOOD_6 + ["topk_tail_mass", "renyi_entropy_2"],
    "GOOD_5+logprob_ext": GOOD_5 + LOGPROB_EXT,
    "GOOD_6+logprob": GOOD_6 + LOGPROB,
    "GOOD_6+logprob+ext": GOOD_6 + LOGPROB + ["topk_tail_mass", "renyi_entropy_2"],
    "GOOD_6+energy": GOOD_6 + ENERGY,
    "TOP8_info": None,        # filled from the informativeness ranking
    "TOP6_info": None,
}


def build_pools(ranked):
    top = {n: [f for f in CANONICAL_POOL if f in set(ranked[:n])] for n in (14, 20)}
    return {
        "full30": list(CANONICAL_POOL),
        "no_dead28": [f for f in CANONICAL_POOL if f not in DEAD],
        "top20_info": top[20],
        "top14_info": top[14],
        # structural, label-free: the curated core plus one augmentation family
        "core+logprob": GOOD_6 + LOGPROB + LOGPROB_EXT[1:],
        "core+energy": GOOD_6 + ENERGY,
        "core+all_new": GOOD_6 + LOGPROB + LOGPROB_EXT + ENERGY + SPILLED,
        "spectral_only": [f for f in CANONICAL_POOL
                          if f not in LOGPROB + LOGPROB_EXT + ENERGY + SPILLED],
    }


def load_cells():
    raw = []
    for domain, ck, fd, labels in iter_cells(
            DD, domains=None, cells=None,
            derived_views_pkl=os.path.join(DD, "derived_views.pkl"),
            trace_cells_pkl=os.path.join(DD, "trace_cells.pkl")):
        if ck in INSCOPE:
            raw.append((domain, ck, fd, labels))
    return raw


def run_subsets(raw, ranked):
    SUBSETS["TOP8_info"] = [f for f in CANONICAL_POOL if f in set(ranked[:8])]
    SUBSETS["TOP6_info"] = [f for f in CANONICAL_POOL if f in set(ranked[:6])]
    rows = []
    for name, feats in SUBSETS.items():
        per = []
        for domain, ck, fd, labels in raw:
            ctx = prepare_cell(domain, ck, fd, labels, feature_pool=CANONICAL_POOL)
            if ctx is None:
                continue
            cols = sorted(ctx.pool.index(f) for f in feats if f in ctx.pool)
            if len(cols) < 3:
                continue
            a = eval_subset_flex(ctx, cols)["auroc"]
            if np.isfinite(a):
                per.append((ck, a))
        if not per:
            continue
        qa = [a for c, a in per if GROUP[c] == "QA"]
        mt = [a for c, a in per if GROUP[c] == "math"]
        rows.append(dict(subset=name, n_feats=len(feats), n_cells=len(per),
                         macro=float(np.mean([a for _, a in per])),
                         qa=float(np.mean(qa)) if qa else None,
                         math=float(np.mean(mt)) if mt else None,
                         members="|".join(feats)))
    return rows


def run_pools(raw, pools):
    sel_fn = get_selector("a2_groupfs")
    rows = []
    for name, pool in pools.items():
        per, picked = [], []
        for domain, ck, fd, labels in raw:
            ctx = prepare_cell(domain, ck, fd, labels, feature_pool=pool)
            if ctx is None:
                continue
            try:
                out = sel_fn(UnlabeledCell.from_context(ctx), np.random.default_rng(0))
            except Exception as e:
                print(f"   [{name}/{ck}] {type(e).__name__}: {e}")
                continue
            d = next((x for x in out if x["variant"] == "a2.dufs"), None)
            if d is None or len(d["cols"]) < 3:
                continue
            a = eval_subset_flex(ctx, list(d["cols"]))["auroc"]
            if np.isfinite(a):
                per.append((ck, a)); picked.append(len(d["cols"]))
        if not per:
            continue
        qa = [a for c, a in per if GROUP[c] == "QA"]
        mt = [a for c, a in per if GROUP[c] == "math"]
        rows.append(dict(pool=name, n_views=len(pool), n_cells=len(per),
                         selector_macro=float(np.mean([a for _, a in per])),
                         qa=float(np.mean(qa)) if qa else None,
                         math=float(np.mean(mt)) if mt else None,
                         mean_picked=float(np.mean(picked)),
                         members="|".join(pool)))
        print(f"  pool {name:<16} ({len(pool):>2} views) -> {rows[-1]['selector_macro']:.4f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-subsets", action="store_true")
    ap.add_argument("--only-pools", action="store_true")
    args = ap.parse_args()

    ranked, _ = informativeness_ranking()
    raw = load_cells()
    print(f"cells: {len(raw)}\n")
    os.makedirs(AI, exist_ok=True)

    if not args.only_pools:
        print("=== B. fixed subsets (incl. never-offered topk / renyi variants) ===")
        rows = run_subsets(raw, ranked)
        rows.sort(key=lambda d: -d["macro"])
        with open(os.path.join(AI, "pool_variants_subsets.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        base = next((d["macro"] for d in rows if d["subset"] == "GOOD_6"), None)
        print(f"{'subset':<22}{'feats':>6}{'macro':>9}{'vs GOOD_6':>11}{'QA':>8}{'math':>8}")
        print("-" * 66)
        for d in rows:
            print(f"{d['subset']:<22}{d['n_feats']:>6}{d['macro']:>9.4f}"
                  f"{100*(d['macro']-base):>+11.2f}{d['qa']:>8.4f}{d['math']:>8.4f}")

    if not args.only_subsets:
        print("\n=== A. pool variants, selector a2.dufs ===")
        rows = run_pools(raw, build_pools(ranked))
        rows.sort(key=lambda d: -d["selector_macro"])
        with open(os.path.join(AI, "pool_variants_pools.csv"), "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        base = next((d["selector_macro"] for d in rows if d["pool"] == "full30"), None)
        print(f"\n{'pool':<18}{'views':>6}{'selector':>10}{'vs full30':>11}"
              f"{'QA':>8}{'math':>8}{'picked':>8}")
        print("-" * 70)
        for d in rows:
            print(f"{d['pool']:<18}{d['n_views']:>6}{d['selector_macro']:>10.4f}"
                  f"{100*(d['selector_macro']-base):>+11.2f}{d['qa']:>8.4f}"
                  f"{d['math']:>8.4f}{d['mean_picked']:>8.1f}")
    print(f"\nwrote -> {AI}")


if __name__ == "__main__":
    main()
