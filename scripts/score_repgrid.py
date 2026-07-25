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
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.feature_utils import FEAT_NAMES
from spectral_utils.repgrid_scoring import (
    load_repgrid_cell, score_subset, ENERGY_FEATS, LOGPROB_FEATS,
    _candidate_features, logprob_features_extended,
)

H16 = list(FEAT_NAMES[:16])
SPILLED = ["epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled", "min_spilled"]
STABLE_H9 = ["epr", "low_band_power", "high_band_power", "hl_ratio",
             "spectral_centroid", "sw_var_peak", "rpdi", "pe_mean", "cusum_max"]
GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]

# High-ranked subsets from results/subset_sweep/top_subsets.csv (Step 154).
BASE_SUBSETS = {
    "consensus_4":  ["spectral_entropy", "sw_var_peak", "cusum_max", "cusum_shift_idx"],
    "GOOD_5":       GOOD_5,
    # GOOD_6 = GOOD_5 + varentropy (Step 182 sweep: +1.12pp macro on the 19-cell replication
    # grid). Needs top_k_logprobs -> load_repgrid_cell_ext (below), not the base loader.
    "GOOD_6":       GOOD_5 + ["varentropy"],
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


def load_repgrid_cell_ext(pkl_path, label_key="label"):
    """Same as repgrid_scoring.load_repgrid_cell but also merges
    logprob_features_extended (varentropy/renyi_entropy_2/topk_tail_mass), needed for
    GOOD_6 = GOOD_5 + varentropy. Kept local (mirrors build_repgrid_featcache.py's
    candidate_feats() pattern) to avoid touching the shared repgrid_scoring module other
    scorers (score_edis_grid.py, score_ubaselines.py, inspect_cell.py) depend on."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    rows, labels, labels_lex, pid = [], [], [], []
    for idx in sorted(data.keys()):
        for c in data[idx]["candidates"]:
            feats = dict(_candidate_features(c))
            if c.get("top_k_logprobs") is not None:
                feats.update(logprob_features_extended(c["top_k_logprobs"]))
            rows.append(feats)
            labels.append(bool(c.get(label_key, c.get("label", False))))
            labels_lex.append(bool(c.get("label_lexical", c.get("label", False))))
            pid.append(int(idx))
    avail = sorted({k for r in rows for k in r})
    return {
        "rows": rows,
        "labels": np.asarray(labels, dtype=bool),
        "labels_lex": np.asarray(labels_lex, dtype=bool),
        "problem_id": np.asarray(pid, dtype=int),
        "n_problems": len(data),
        "available": avail,
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


def score_cell_all(cell, subsets, anchor="epr"):
    """Score every subset (that has >=3 available features on this cell) x method."""
    avail = set(cell["available"])
    out = []
    for name, feats in subsets.items():
        present = [f for f in feats if f in avail]
        if len(present) < 3:
            continue
        for method in ("lsml", "upcr"):
            r = score_subset(cell, present, method=method, anchor=anchor)
            r["subset"] = name
            r["n_feats_used"] = len(present)
            r["anchor"] = anchor
            out.append(r)
    return out


# Task B3: anchor-choice robustness. score_subset's global-sign resolution defaults to
# an `epr` anchor; re-score GOOD_5/GOOD_6 with an alternate anchor (cusum_max, the same
# swap already validated -- "changes nothing" -- on a single T-varied MATH-500 cache in
# scripts/temperature_followups.py, Step 182 Item B) to see if that holds on the wider
# 19-cell replication grid too. Extends, does not resolve, the open anchor-fragility
# thread the concurrent EDIS session flagged on a different domain (Step 183).
ANCHOR_ROBUSTNESS_SUBSETS = ("GOOD_5", "GOOD_6")
ALT_ANCHOR = "cusum_max"


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
        # Never let the concurrent EDIS-grid replication's pilot cells land in this
        # tracked CSV -- separate, still-in-progress effort with its own scorer/CSVs.
        if preset_id.startswith("edis_"):
            continue
        Y, Ymethod = published_value(man)
        h2h = man.get("head_to_head")
        cell = load_repgrid_cell_ext(pkl)
        acc = float(cell["labels"].mean())
        print(f"\n== {preset_id} | {man.get('model')} | {man.get('dataset')} | "
              f"N={cell['n_problems']} acc={acc:.3f} | Y={Y} ({Ymethod}) h2h={h2h} ==")

        def emit(r):
            delta = (r["auroc"] - Y) if (Y is not None and r["auroc"] == r["auroc"]) else None
            rows_out.append({
                "cell": preset_id, "model": man.get("model"), "dataset": man.get("dataset"),
                "n_problems": cell["n_problems"], "acc": round(acc, 4),
                "subset": r["subset"], "method": r["method"], "n_feats": r["n_feats_used"],
                "anchor": r["anchor"],
                "auroc_X": round(r["auroc"], 4) if r["auroc"] == r["auroc"] else None,
                "lo": round(r["lo"], 4) if r["lo"] == r["lo"] else None,
                "hi": round(r["hi"], 4) if r["hi"] == r["hi"] else None,
                "n_rows": r["n"], "valid_rate": round(r["valid_rate"], 3),
                "published_Y": Y, "Y_method": Ymethod, "delta_X_minus_Y": round(delta, 4) if delta is not None else None,
                "head_to_head": h2h, "flipped": r["flipped"],
            })
            return delta

        for r in score_cell_all(cell, subsets):
            delta = emit(r)
            if r["subset"] in ("GOOD_5", "GOOD_5+energy", "GOOD_5+logprob"):
                dtxt = f"{delta:+.3f}" if delta is not None else "  n/a"
                xtxt = f"{r['auroc']:.4f}" if r['auroc'] == r['auroc'] else "  nan"
                print(f"   {r['subset']:16s} {r['method']:5s} X={xtxt} "
                      f"vs Y={Y}  d={dtxt}  (n={r['n']}, valid={r['valid_rate']:.2f})")

        # Task B3: anchor-choice robustness -- re-score GOOD_5/GOOD_6 with an alternate
        # anchor and emit as extra rows (same subset name, anchor="cusum_max"), so the
        # closed_subset comparison can flag cells where the two anchors disagree.
        alt_subsets = {k: v for k, v in subsets.items() if k in ANCHOR_ROBUSTNESS_SUBSETS}
        for r in score_cell_all(cell, alt_subsets, anchor=ALT_ANCHOR):
            emit(r)

    # Merge-on-write: keep rows of cells NOT re-scored this run, so a --cells run never
    # drops the other cells' scores (a --cells overwrite silently lost the Step-163 rows).
    out_csv = os.path.join(args.out, "scores_lsml_upcr.csv")
    scored_cells = {r["cell"] for r in rows_out}
    kept = []
    if os.path.exists(out_csv):
        with open(out_csv, newline="") as f:
            kept = [r for r in csv.DictReader(f) if r.get("cell") not in scored_cells]
            for r in kept:
                r.setdefault("anchor", "epr")  # backward-compat: pre-Task-B3 rows were all epr-anchored
    fieldnames = list(rows_out[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(kept)
        w.writerows(rows_out)
    print(f"\nwrote {len(rows_out)} new + {len(kept)} kept rows -> {out_csv}")


if __name__ == "__main__":
    main()
