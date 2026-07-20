#!/usr/bin/env python
"""
build_repgrid_featcache.py — Phase-2 shared loader adapter (HANDOFF_punchlist_and_reruns.md).

The replication-grid cells (cache/repgrid/*/) use a per-candidate schema that neither the
feature-subset sweep (spectral_utils/subset_sweep.py) nor the LR oracle
(scripts/logistic_oracle.py) can read — both expect the legacy per-cell
{cell_key: (feats_dict, labels)} / {'feats','labels'} schema sourced from local_cache/.
Rather than teach each consumer the candidate schema, we adapt the DATA once here: extract
every feature per candidate, complete-case-filter on the spectral pool (so the sample set
exactly reproduces repgrid_scoring.score_subset's valid rows), and emit one legacy-schema pkl
that BOTH consumers read.

Outputs (local_cache/):
  repgrid_cells.pkl   {preset_id: {'feats': fd, 'labels': y, 'problem_id': pid, 'k', 'dataset',
                       'model', 'acc'}} — dict payload; subset_sweep + logistic_oracle both read it.
  repgrid_cont.pkl    [{'cell': 'repgrid/<id>', 'cont_5','cont_9','cont_16', 'n','prevalence'}]
                       — the unsupervised L-SML CONT column for the oracle, read from
                       results/repgrid/scores_lsml_upcr.csv (GOOD_5->5, STABLE_H9->9, ALL_H16->16).

Validation gate: GOOD_5 L-SML recomputed from the emitted feats_dict must match the CSV's
GOOD_5/lsml AUROC for that cell (same recipe, expect <= 0.005). A larger gap fails the build.

Feature pool emitted per cell: the 20 FEAT_NAMES (spectral+spilled) + 4 energy (ENERGY_FEATS,
when logsumexp captured) + 3 logprob (LOGPROB_FEATS) + 3 extended logprob (LOGPROB_FEATS_EXT),
each kept only when finite on ALL complete-case rows (a length-aligned column).

Usage:
    python scripts/build_repgrid_featcache.py            # all 19 analysis cells (skip reject/partial/pilot)
    python scripts/build_repgrid_featcache.py --cells lapeigvals_gsm8k_llama8b
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.feature_utils import FEAT_NAMES
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient
from spectral_utils.repgrid_scoring import (
    _candidate_features, logprob_features_extended,
    ENERGY_FEATS, LOGPROB_FEATS, LOGPROB_FEATS_EXT,
)

H16 = list(FEAT_NAMES[:16])
GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]
# skip archived / non-analysis cell dirs
SKIP_SUFFIXES = ("_reject", "_partial", "_pilot")
CSV = os.path.join(REPO, "results", "repgrid", "scores_lsml_upcr.csv")


def is_analysis_cell(preset_id):
    return not any(preset_id.endswith(s) for s in SKIP_SUFFIXES) and "_pilot" not in preset_id


def candidate_feats(c):
    """All computable features for one candidate incl. the extended logprob views."""
    feats = dict(_candidate_features(c))            # 20 spectral+spilled (+energy +logprob)
    if c.get("top_k_logprobs") is not None:
        feats.update(logprob_features_extended(c["top_k_logprobs"]))
    return feats


def build_cell(pkl_path):
    """Load a raw repgrid pkl -> (feats_dict, labels, problem_id) on the complete-case
    (spectral-scorable) rows. feats_dict keeps every feature finite on ALL kept rows."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    rows, labels, pid = [], [], []
    for idx in sorted(data.keys()):
        for c in data[idx]["candidates"]:
            rows.append(candidate_feats(c))
            labels.append(bool(c.get("label", False)))
            pid.append(int(idx))
    labels = np.asarray(labels, dtype=int)
    pid = np.asarray(pid, dtype=int)

    # complete-case = rows where every spectral feature is present (trace >= 8). Because
    # extract_all_features is all-or-none, this is exactly score_subset's GOOD_5 valid set.
    keep = np.array([all(np.isfinite(r.get(f, np.nan)) for f in H16) for r in rows], dtype=bool)
    if keep.sum() < 20:
        return None
    kept_rows = [r for r, k in zip(rows, keep) if k]

    pool = list(FEAT_NAMES) + ENERGY_FEATS + LOGPROB_FEATS + LOGPROB_FEATS_EXT
    fd = {}
    for f in pool:
        col = np.array([r.get(f, np.nan) for r in kept_rows], dtype=float)
        if np.isfinite(col).all() and col.std() > 1e-12:
            fd[f] = col
    return fd, labels[keep], pid[keep]


def good5_lsml_auroc(fd, labels):
    """score_subset (lsml branch) recipe: pipeline + label-free epr-anchor orientation."""
    present = [f for f in GOOD_5 if f in fd]
    if len(present) < 3:
        return None
    sub = {f: fd[f] for f in present}
    score, _ = lsml_continuous_pipeline(sub, present, FEATURE_SIGNS)
    anchor_feat = "epr" if "epr" in present else present[0]
    anchor = zscore(fd[anchor_feat] * FEATURE_SIGNS.get(anchor_feat, +1))
    score, _ = anchor_orient(np.asarray(score, dtype=float), anchor)
    auc, _, _ = boot_auc(labels, score, n=500)
    return float(auc)


# GOOD_6 = GOOD_5 + varentropy (Step 182 sweep finding, Task B). This featcache already
# computes varentropy in candidate_feats() via LOGPROB_FEATS_EXT -- reuse it as a free,
# independent cross-check that score_repgrid.py's new GOOD_6 rows are correct, via a
# completely separate code path (fd built here vs. score_repgrid's load_repgrid_cell_ext).
GOOD_6 = GOOD_5 + ["varentropy"]


def good6_lsml_auroc(fd, labels):
    """Same recipe as good5_lsml_auroc, over GOOD_6. FEATURE_SIGNS must know varentropy's
    sign (-1, higher varentropy -> more likely wrong) -- see LOGPROB_SIGNS_EXT in
    spectral_utils.repgrid_scoring; anchor_orient corrects the global sign regardless."""
    present = [f for f in GOOD_6 if f in fd]
    if len(present) < 3:
        return None
    sub = {f: fd[f] for f in present}
    signs = {**FEATURE_SIGNS, "varentropy": -1}
    score, _ = lsml_continuous_pipeline(sub, present, signs)
    anchor_feat = "epr" if "epr" in present else present[0]
    anchor = zscore(fd[anchor_feat] * signs.get(anchor_feat, +1))
    score, _ = anchor_orient(np.asarray(score, dtype=float), anchor)
    auc, _, _ = boot_auc(labels, score, n=500)
    return float(auc)


def load_csv_refs():
    """GOOD_5/GOOD_6 gate reference values from the canonical CSV.

    The featcache's good5/good6_lsml_auroc orient against the *epr* anchor, so the
    gate must compare against the epr-anchored CSV row. score_repgrid.py also emits
    a cusum_max-anchored robustness row per (cell, GOOD_5/GOOD_6, method); keying only
    on (cell, subset, method) let last-write-wins pick up that cusum_max row, which
    on a chance-level cell (where the two anchors resolve the arbitrary global sign
    oppositely, AUROCs reflected about 0.5) produced a spurious gate FAIL — e.g.
    gpqa_r1distill8b epr=0.4856 vs cusum_max=0.5144. Filter to the epr anchor so the
    gate compares like-with-like. Rows predating the anchor column default to epr.
    """
    refs = {}
    if not os.path.exists(CSV):
        return refs
    for r in csv.DictReader(open(CSV)):
        if (r.get("anchor") or "epr") != "epr":
            continue
        key = (r["cell"], r["subset"], r["method"])
        try:
            refs[key] = float(r["auroc_X"]) if r["auroc_X"] not in ("", None) else None
        except (TypeError, ValueError):
            refs[key] = None
    return refs


def discover(cache_dir, only=None):
    for man_path in sorted(glob.glob(os.path.join(cache_dir, "*", "manifest.json"))):
        d = os.path.dirname(man_path)
        pid = os.path.basename(d)
        if not is_analysis_cell(pid):
            continue
        if only and not any(s in pid for s in only):
            continue
        pkls = sorted(glob.glob(os.path.join(d, "raw_*.pkl")))
        if pkls:
            with open(man_path) as f:
                yield pid, json.load(f), pkls[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache/repgrid")
    ap.add_argument("--out-dir", default="local_cache")
    ap.add_argument("--cells", default=None, help="comma-sep substrings")
    ap.add_argument("--gate-tol", type=float, default=0.005)
    args = ap.parse_args()

    only = args.cells.split(",") if args.cells else None
    refs = load_csv_refs()
    csv_sub = {"cont_5": "GOOD_5", "cont_9": "STABLE_H9", "cont_16": "ALL_H16"}

    cells, cont_rows, gate_fail = {}, [], []
    for pid, man, pkl in discover(args.cache_dir, only=only):
        built = build_cell(pkl)
        if built is None:
            print(f"[skip] {pid}: <20 spectral-scorable candidates")
            continue
        fd, y, pid_arr = built
        acc = float(y.mean())
        cells[pid] = {"feats": fd, "labels": y, "problem_id": pid_arr,
                      "k": int(man.get("k", 1)), "dataset": man.get("dataset"),
                      "model": man.get("model"), "acc": acc}

        # validation gate vs the CSV GOOD_5/lsml value
        g5 = good5_lsml_auroc(fd, y)
        ref = refs.get((pid, "GOOD_5", "lsml"))
        status = "n/a"
        if g5 is not None and ref is not None:
            d = abs(g5 - ref)
            status = f"{'OK' if d <= args.gate_tol else 'FAIL'} (Δ={d:.4f})"
            if d > args.gate_tol:
                gate_fail.append(("GOOD_5", pid, g5, ref, d))
        print(f"[cell] {pid:<32} n={len(y):>5} acc={acc:.3f} feats={len(fd):>2} "
              f"k={man.get('k',1)} | GOOD_5 lsml={g5 if g5 is None else round(g5,4)} "
              f"vs CSV {ref} -> {status}")

        # same gate for GOOD_6 (Task B) -- only meaningful on cells with top_k_logprobs
        g6 = good6_lsml_auroc(fd, y)
        ref6 = refs.get((pid, "GOOD_6", "lsml"))
        status6 = "n/a"
        if g6 is not None and ref6 is not None:
            d6 = abs(g6 - ref6)
            status6 = f"{'OK' if d6 <= args.gate_tol else 'FAIL'} (Δ={d6:.4f})"
            if d6 > args.gate_tol:
                gate_fail.append(("GOOD_6", pid, g6, ref6, d6))
        print(f"         {'':<32} {'':<5} {'':<5} {'':<7} "
              f"  GOOD_6 lsml={g6 if g6 is None else round(g6,4)} "
              f"vs CSV {ref6} -> {status6}")

        # CONT row for the oracle (repgrid/<id>), from the CSV's lsml AUROCs
        cont = {"cell": f"repgrid/{pid}", "n": int(len(y)), "prevalence": acc}
        for k, sub in csv_sub.items():
            cont[k] = refs.get((pid, sub, "lsml"))
        cont_rows.append(cont)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "repgrid_cells.pkl"), "wb") as f:
        pickle.dump(cells, f)
    with open(os.path.join(args.out_dir, "repgrid_cont.pkl"), "wb") as f:
        pickle.dump(cont_rows, f)

    print(f"\nwrote {len(cells)} cells -> {args.out_dir}/repgrid_cells.pkl")
    print(f"wrote {len(cont_rows)} CONT rows -> {args.out_dir}/repgrid_cont.pkl")
    if gate_fail:
        print(f"\nGATE FAIL on {len(gate_fail)} (subset, cell) pairs (Δ>{args.gate_tol}):")
        for subset, pid, val, ref, d in gate_fail:
            print(f"  {subset}/{pid}: featcache {val:.4f} vs CSV {ref:.4f} (Δ={d:.4f})")
        sys.exit(1)
    print("\nvalidation gate PASS — featcache GOOD_5/GOOD_6 reproduce the canonical CSV.")


if __name__ == "__main__":
    main()
