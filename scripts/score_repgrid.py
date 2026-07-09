#!/usr/bin/env python
"""
score_repgrid.py — run OUR methods (L-SML continuous + U-PCR) on the replication-grid
cells and place each AUROC next to the paper's PUBLISHED number.

This is offline, local-CPU scoring. It does NOT reproduce any competitor detector — the
competitor value Y is read from each cell's manifest.json (the paper's reported AUROC).
For every cell x subset x method it prints: our AUROC X (raw, 95% CI), the published Y,
Delta = X - Y, and the head_to_head tag (SAME-MODEL means X and Y share the exact model).

Subsets are the ones that ranked high in the Step-154 sweep (results/subset_sweep/top_subsets.csv),
plus augmented views that add the new spilled / raw-energy / logprob features.

Usage:
    python scripts/score_repgrid.py [--cache-dir cache/repgrid] [--out results/repgrid]
"""
import argparse
import csv
import glob
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.feature_utils import FEAT_NAMES
from spectral_utils.repgrid_scoring import (
    load_repgrid_cell, score_subset, ENERGY_FEATS, LOGPROB_FEATS,
)

H16 = list(FEAT_NAMES[:16])
SPILLED = ["epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled", "min_spilled"]
STABLE_H9 = ["epr", "low_band_power", "high_band_power", "hl_ratio",
             "spectral_centroid", "sw_var_peak", "rpdi", "pe_mean", "cusum_max"]

# High-ranked subsets from results/subset_sweep/top_subsets.csv (Step 154).
BASE_SUBSETS = {
    "consensus_4":  ["spectral_entropy", "sw_var_peak", "cusum_max", "cusum_shift_idx"],
    "GOOD_5":       ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"],
    "top_macro_5":  ["epr", "spectral_entropy", "hl_ratio", "sw_var_peak", "cusum_max"],
    "STABLE_H9":    STABLE_H9,
    "ALL_H16":      H16,
}
# Augmented views (added to GOOD_5 when the cell has those features).
AUGMENTS = {
    "GOOD_5+spilled": ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"] + SPILLED,
    "GOOD_5+energy":  ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"] + ENERGY_FEATS,
    "GOOD_5+logprob": ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"] + LOGPROB_FEATS,
}


def discover_cells(cache_dir, only=None):
    """Yield (preset_id, manifest, pkl_path) for every cell with a raw pkl.
    `only` = optional list of substrings; a cell is included if any matches its id."""
    for man_path in sorted(glob.glob(os.path.join(cache_dir, "*", "manifest.json"))):
        cell_dir = os.path.dirname(man_path)
        preset_id = os.path.basename(cell_dir)
        if only and not any(s in preset_id for s in only):
            continue
        with open(man_path) as f:
            man = json.load(f)
        pkls = [p for p in glob.glob(os.path.join(cell_dir, "raw_*.pkl"))]
        if pkls:
            yield preset_id, man, sorted(pkls)[0]


def published_value(man):
    pub = man.get("published") or {}
    v = pub.get("value")
    return (float(v) / 100.0 if v is not None and v > 1.5 else v), pub.get("method", "")


def score_cell_all(cell, subsets):
    """Score every subset (that has >=3 available features on this cell) x method."""
    avail = set(cell["available"])
    out = []
    for name, feats in subsets.items():
        present = [f for f in feats if f in avail]
        if len(present) < 3:
            continue
        for method in ("lsml", "upcr"):
            r = score_subset(cell, present, method=method)
            r["subset"] = name
            r["n_feats_used"] = len(present)
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache/repgrid")
    ap.add_argument("--out", default="results/repgrid")
    ap.add_argument("--cells", default=None, help="comma-sep substrings; score only matching cells")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    only = args.cells.split(",") if args.cells else None
    subsets = {**BASE_SUBSETS, **AUGMENTS}
    rows_out = []
    for preset_id, man, pkl in discover_cells(args.cache_dir, only=only):
        Y, Ymethod = published_value(man)
        h2h = man.get("head_to_head")
        cell = load_repgrid_cell(pkl)
        acc = float(cell["labels"].mean())
        print(f"\n== {preset_id} | {man.get('model')} | {man.get('dataset')} | "
              f"N={cell['n_problems']} acc={acc:.3f} | Y={Y} ({Ymethod}) h2h={h2h} ==")
        for r in score_cell_all(cell, subsets):
            delta = (r["auroc"] - Y) if (Y is not None and r["auroc"] == r["auroc"]) else None
            rows_out.append({
                "cell": preset_id, "model": man.get("model"), "dataset": man.get("dataset"),
                "n_problems": cell["n_problems"], "acc": round(acc, 4),
                "subset": r["subset"], "method": r["method"], "n_feats": r["n_feats_used"],
                "auroc_X": round(r["auroc"], 4) if r["auroc"] == r["auroc"] else None,
                "lo": round(r["lo"], 4) if r["lo"] == r["lo"] else None,
                "hi": round(r["hi"], 4) if r["hi"] == r["hi"] else None,
                "n_rows": r["n"], "valid_rate": round(r["valid_rate"], 3),
                "published_Y": Y, "Y_method": Ymethod, "delta_X_minus_Y": round(delta, 4) if delta is not None else None,
                "head_to_head": h2h, "flipped": r["flipped"],
            })
            if r["subset"] in ("GOOD_5", "GOOD_5+energy", "GOOD_5+logprob"):
                dtxt = f"{delta:+.3f}" if delta is not None else "  n/a"
                xtxt = f"{r['auroc']:.4f}" if r['auroc'] == r['auroc'] else "  nan"
                print(f"   {r['subset']:16s} {r['method']:5s} X={xtxt} "
                      f"vs Y={Y}  d={dtxt}  (n={r['n']}, valid={r['valid_rate']:.2f})")

    out_csv = os.path.join(args.out, "scores_lsml_upcr.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nwrote {len(rows_out)} rows -> {out_csv}")


if __name__ == "__main__":
    main()
