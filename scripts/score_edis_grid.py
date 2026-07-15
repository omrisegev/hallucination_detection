#!/usr/bin/env python
"""
score_edis_grid.py — score the EDIS-grid replication (base Qwen2.5-Math-1.5B x
GSM8K/MATH500/AMC23/AIME24, arXiv 2602.01288 Section 5.3 protocol) locally.

Mirrors scripts/score_edis.py's per-cell scoring (EDIS AUROC, L-SML GOOD_5 AUROC,
rho(EDIS,L-SML)) but extends it three ways needed for this specific replication:

1. Per (dataset, T) cell also reports mean-entropy AUROC (the paper's own §5.3
   comparison arm) alongside EDIS and our L-SML GOOD_5.
2. Pools ALL 4 datasets x 3 temps into one (score, label) array -- replicating the
   paper's own §5.3 pooled comparison (EDIS AUC 0.804 vs mean-entropy AUC 0.673,
   N=26356) with L-SML GOOD_5 reported alongside on IDENTICAL traces.
3. Draws a paper-faithful subsample (fixed seed): 100 random GSM8K + 100 random
   MATH500 problems at K=8 (matching the paper's own N, vs. our over-collected
   N=500), and 8 of our K candidates per AMC23/AIME24 problem (matching the
   paper's K=8 there too, vs. our over-collected K=32/64) -- the over-collection
   changes CI width, never the faithful headline number.

Prefers the full-vocabulary entropy trace (token_entropies_full, present when a
preset set capture_full_entropy=True) for EDIS/mean-entropy scoring -- the paper's
own H_t definition (Eq. 1) -- and falls back to the top-K=15 token_entropies when
full-vocab wasn't captured (older cells), recording which source was used per cell
so a truncation-sensitivity ablation (EDIS-on-top15 vs EDIS-on-full) can be read
directly off the per-row source column.

Usage:
    PYTHONPATH=. python scripts/score_edis_grid.py --cache-dir cache/repgrid \
        --cell-dirs edis_gsm8k_qwenmath15b edis_math500_qwenmath15b \
                    edis_amc23_qwenmath15b edis_aime24_qwenmath15b
"""
import argparse
import csv
import glob
import os
import pickle
import re
import sys

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.feature_utils import compute_edis
from spectral_utils.fusion_utils import zscore, boot_auc, lsml_continuous_pipeline
from spectral_utils.repgrid_scoring import load_repgrid_cell, subset_matrix, ALL_SIGNS
from spectral_utils.streaming_utils import anchor_orient

GOOD_5 = ["epr", "low_band_power", "sw_var_peak", "cusum_max", "spectral_entropy"]
DATASETS = ["gsm8k", "math500", "amc23", "aime24"]
TEMPS = [0.2, 0.6, 1.0]
PAPER_POOLED = {"edis_auc": 0.804, "meanh_auc": 0.673, "n_responses": 26356}
# Paper's own N per dataset (§5.1/§5.3: GSM8K/MATH random-100, AMC23/AIME24 full sets, K=8).
PAPER_N_PROBLEMS = {"gsm8k": 100, "math500": 100, "amc23": 40, "aime24": 30}
PAPER_K = 8
SEED = 0


def _entropy_trace(cand):
    """Prefer full-vocab H(n) (paper's Eq.1 def); fall back to our top-K=15 trace."""
    full = cand.get("token_entropies_full")
    if full is not None and len(full) >= 2:
        return full, "full"
    top15 = cand.get("token_entropies")
    if top15 is not None and len(top15) >= 2:
        return top15, "topk15"
    return None, None


def per_candidate_scores(pkl_path):
    """Confidence scores in the SAME order load_repgrid_cell walks (sorted problem ids,
    candidates in list order), plus the entropy source used and each candidate's
    problem id (for paper-faithful subsampling)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    edis, mean_h, src, ok, pid = [], [], [], [], []
    for idx in sorted(data.keys()):
        for c in data[idx]["candidates"]:
            H, source = _entropy_trace(c)
            if H is None:
                edis.append(np.nan); mean_h.append(np.nan); src.append(None); ok.append(False)
            else:
                edis.append(-compute_edis(H))        # confidence: higher -> more likely correct
                mean_h.append(-float(np.mean(H)))     # confidence: higher -> more likely correct
                src.append(source); ok.append(True)
            pid.append(int(idx))
    return (np.asarray(edis, dtype=float), np.asarray(mean_h, dtype=float),
            src, np.asarray(ok, dtype=bool), np.asarray(pid, dtype=int))


def lsml_good5_score(cell):
    """L-SML continuous GOOD_5 confidence vector + valid mask (label-free anchor_orient).
    Mirrors scripts/score_edis.py's lsml_good5_score exactly (canonical-scorer-first)."""
    rows = cell["rows"]
    present = [f for f in GOOD_5 if f in set(cell["available"])]
    if len(present) < 3:
        return None, None, present
    X, valid = subset_matrix(rows, present)
    feats_dict = {f: X[valid, j] for j, f in enumerate(present)}
    score, _ = lsml_continuous_pipeline(feats_dict, present, ALL_SIGNS)
    anchor_feat = "epr" if "epr" in present else present[0]
    anchor_view = zscore(np.asarray(feats_dict[anchor_feat], dtype=float)
                         * ALL_SIGNS.get(anchor_feat, +1))
    score, _ = anchor_orient(np.asarray(score, dtype=float), anchor_view)
    return score, valid, present


def _auc_row(y, s, n_boot):
    if len(y) < 20 or not (0 < y.sum() < len(y)):
        return None, None, None
    auc, lo, hi = boot_auc(y, s, n=n_boot)
    if np.isnan(auc):
        return None, None, None
    return round(float(auc), 4), round(float(lo), 4), round(float(hi), 4)


def score_cell(pkl_path, cell_id, n_boot=1000):
    """Score one (dataset, T) pkl: EDIS / mean-entropy / L-SML GOOD_5 AUROC + correlations.
    Returns (result_row_dict, raw_arrays_dict) -- raw_arrays feed the pooled/subsample passes."""
    cell = load_repgrid_cell(pkl_path)
    labels = cell["labels"]
    edis, mean_h, src, ok, pid = per_candidate_scores(pkl_path)
    assert len(edis) == len(labels), f"row mismatch {len(edis)} vs {len(labels)} in {pkl_path}"

    sources_used = sorted({s for s in src if s})
    res = {"cell": cell_id, "n_rows": int(ok.sum()), "acc": round(float(labels.mean()), 4),
           "entropy_source": "+".join(sources_used) if sources_used else None,
           "edis_auroc": None, "edis_lo": None, "edis_hi": None,
           "meanh_auroc": None, "meanh_lo": None, "meanh_hi": None,
           "lsml_good5_auroc": None,
           "rho_edis_meanh": None, "rho_edis_lsml": None, "n_overlap": None}

    m = ok & np.isfinite(edis)
    y = labels[m].astype(int)
    if m.sum() >= 20:
        res["edis_auroc"], res["edis_lo"], res["edis_hi"] = _auc_row(y, edis[m], n_boot)
        res["meanh_auroc"], res["meanh_lo"], res["meanh_hi"] = _auc_row(y, mean_h[m], n_boot)
        if np.std(edis[m]) > 1e-8 and np.std(mean_h[m]) > 1e-8:
            rho, _ = spearmanr(edis[m], mean_h[m])
            res["rho_edis_meanh"] = round(float(rho), 4)

    lsml, lsml_valid, present = lsml_good5_score(cell)
    if lsml is not None:
        yv = labels[lsml_valid].astype(int)
        auc, _, _ = _auc_row(yv, lsml, n_boot)
        res["lsml_good5_auroc"] = auc
        edis_on_valid = edis[lsml_valid]
        both = np.isfinite(edis_on_valid) & np.isfinite(lsml)
        if both.sum() >= 20:
            rho, _ = spearmanr(edis_on_valid[both], np.asarray(lsml)[both])
            res["rho_edis_lsml"] = round(float(rho), 4)
            res["n_overlap"] = int(both.sum())

    print(f"== {cell_id}: EDIS={res['edis_auroc']} meanH={res['meanh_auroc']} "
          f"L-SML={res['lsml_good5_auroc']} rho(E,H)={res['rho_edis_meanh']} "
          f"rho(E,L)={res['rho_edis_lsml']} src={res['entropy_source']} "
          f"(n={res['n_rows']}, acc={res['acc']})", flush=True)
    return res, {"edis": edis, "mean_h": mean_h, "labels": labels.astype(int), "ok": ok,
                 "problem_id": pid, "lsml": lsml, "lsml_valid": lsml_valid}


def _pool(raw_list, keys=("edis", "mean_h", "labels", "ok")):
    """Concatenate the requested arrays across a list of raw_arrays dicts."""
    return {k: np.concatenate([r[k] for r in raw_list]) for k in keys}


def pooled_auc_report(raw_list, title, n_boot=1000):
    pooled = _pool(raw_list)
    m = pooled["ok"] & np.isfinite(pooled["edis"])
    y = pooled["labels"][m]
    n = int(m.sum())
    edis_auc, edis_lo, edis_hi = _auc_row(y, pooled["edis"][m], n_boot)
    meanh_auc, meanh_lo, meanh_hi = _auc_row(y, pooled["mean_h"][m], n_boot)

    lsml_parts = [r["lsml"] for r in raw_list if r["lsml"] is not None]
    lsml_auc = None
    if lsml_parts:
        lsml_labels = np.concatenate([r["labels"][r["lsml_valid"]] for r in raw_list if r["lsml"] is not None])
        lsml_scores = np.concatenate(lsml_parts)
        lsml_auc, lsml_lo, lsml_hi = _auc_row(lsml_labels, lsml_scores, n_boot)
    else:
        lsml_lo = lsml_hi = None

    print(f"\n=== {title} (n={n}) ===")
    print(f"  EDIS AUROC:       {edis_auc} [{edis_lo},{edis_hi}]  (paper: {PAPER_POOLED['edis_auc']})")
    print(f"  mean-H AUROC:     {meanh_auc} [{meanh_lo},{meanh_hi}]  (paper: {PAPER_POOLED['meanh_auc']})")
    print(f"  L-SML GOOD_5 AUROC: {lsml_auc} [{lsml_lo},{lsml_hi}]  (n={len(lsml_parts) and sum(len(p) for p in lsml_parts)})")
    return {"title": title, "n": n, "edis_auroc": edis_auc, "edis_lo": edis_lo, "edis_hi": edis_hi,
            "meanh_auroc": meanh_auc, "meanh_lo": meanh_lo, "meanh_hi": meanh_hi,
            "lsml_good5_auroc": lsml_auc, "lsml_lo": lsml_lo, "lsml_hi": lsml_hi}


def paper_faithful_mask(dataset, pid, k_per_problem, rng):
    """Boolean mask selecting a paper-faithful subsample of rows for one (dataset,T) cell.

    GSM8K/MATH500: subsample PAPER_N_PROBLEMS[dataset] of the collected problems (fixed
    seed), keep all K candidates (our k already == paper's K=8 for these two datasets).
    AMC23/AIME24: keep all problems (already the paper's full set), subsample PAPER_K of
    our over-collected K candidates per problem (fixed seed).
    """
    n_target = PAPER_N_PROBLEMS[dataset]
    uniq = np.unique(pid)
    if dataset in ("gsm8k", "math500"):
        chosen_problems = set(rng.choice(uniq, size=min(n_target, len(uniq)), replace=False).tolist())
        return np.isin(pid, list(chosen_problems))
    # amc23 / aime24: subsample PAPER_K candidates per problem
    mask = np.zeros(len(pid), dtype=bool)
    for p in uniq:
        idxs = np.where(pid == p)[0]
        keep = rng.choice(idxs, size=min(PAPER_K, len(idxs)), replace=False)
        mask[keep] = True
    return mask


def discover_cells(cache_dir, cell_dirs):
    """Map cell_dir_name -> {dataset: [(temp, pkl_path), ...]}."""
    out = {}
    for name in cell_dirs:
        d = os.path.join(cache_dir, name)
        pkls = sorted(glob.glob(os.path.join(d, "raw_*_T*.pkl")))
        if not pkls:
            print(f"WARNING: no raw_*_T*.pkl under {d} -- skipping", file=sys.stderr)
            continue
        per_ds = {}
        for p in pkls:
            m = re.match(r"raw_(.+)_T([\d.]+)\.pkl$", os.path.basename(p))
            if not m:
                continue
            ds, t = m.group(1), float(m.group(2))
            per_ds.setdefault(ds, []).append((t, p))
        out[name] = per_ds
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache/repgrid")
    ap.add_argument("--cell-dirs", nargs="+",
                    default=["edis_gsm8k_qwenmath15b", "edis_math500_qwenmath15b",
                             "edis_amc23_qwenmath15b", "edis_aime24_qwenmath15b"],
                    help="cell directory names under --cache-dir, one per dataset")
    ap.add_argument("--out", default="results/repgrid/edis_grid.csv")
    ap.add_argument("--pooled-out", default="results/repgrid/edis_grid_pooled.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    layout = discover_cells(args.cache_dir, args.cell_dirs)
    if not layout:
        sys.exit(f"no cells found under {args.cache_dir} for {args.cell_dirs}")

    per_cell_rows = []
    all_raw = []                 # every (dataset,T) cell's raw arrays -- full over-collected pool
    faithful_raw = []            # paper-faithful subsample only
    rng = np.random.default_rng(SEED)

    for cell_dir, per_ds in layout.items():
        for dataset, temp_pkls in per_ds.items():
            for temp, pkl_path in sorted(temp_pkls):
                cell_id = f"{dataset}_T{temp}"
                res, raw = score_cell(pkl_path, cell_id, n_boot=args.n_boot)
                res["dataset"] = dataset
                res["temp"] = temp
                per_cell_rows.append(res)
                all_raw.append(raw)

                if dataset in PAPER_N_PROBLEMS:
                    fmask = paper_faithful_mask(dataset, raw["problem_id"], PAPER_K, rng)
                    fraw = {"edis": raw["edis"][fmask], "mean_h": raw["mean_h"][fmask],
                            "labels": raw["labels"][fmask], "ok": raw["ok"][fmask],
                            "lsml": None, "lsml_valid": None}
                    # lsml rows are only defined on lsml_valid (a subset of all rows) --
                    # intersect the faithful row-mask with lsml_valid to keep alignment.
                    if raw["lsml"] is not None:
                        lv = raw["lsml_valid"]
                        # position within the lsml-valid subsequence for each faithful+valid row
                        keep_in_lsml = fmask[lv]
                        fraw["lsml"] = np.asarray(raw["lsml"])[keep_in_lsml]
                        fraw["lsml_valid"] = np.zeros(0, dtype=bool)  # unused downstream (labels precomputed below)
                        fraw["labels_lsml"] = raw["labels"][lv][keep_in_lsml]
                    faithful_raw.append(fraw)

    if not per_cell_rows:
        sys.exit("no cells scored")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fieldnames = ["cell", "dataset", "temp", "n_rows", "acc", "entropy_source",
                  "edis_auroc", "edis_lo", "edis_hi", "meanh_auroc", "meanh_lo", "meanh_hi",
                  "lsml_good5_auroc", "rho_edis_meanh", "rho_edis_lsml", "n_overlap"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in per_cell_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"\nwrote {len(per_cell_rows)} per-cell rows -> {args.out}")

    # Pooled §5.3 replication: every collected response, all 4 datasets x 3 temps.
    pooled_full = pooled_auc_report(all_raw, "POOLED (full over-collected grid)", args.n_boot)

    # Paper-faithful subsample: matches the paper's own N/K exactly.
    def _pool_faithful(keys):
        out = {k: np.concatenate([r[k] for r in faithful_raw if r[k] is not None and len(r[k])])
               for k in keys}
        return out

    pf = _pool_faithful(["edis", "mean_h", "labels", "ok"])
    m = pf["ok"] & np.isfinite(pf["edis"])
    y = pf["labels"][m]
    edis_auc, edis_lo, edis_hi = _auc_row(y, pf["edis"][m], args.n_boot)
    meanh_auc, meanh_lo, meanh_hi = _auc_row(y, pf["mean_h"][m], args.n_boot)
    lsml_labels_parts = [r["labels_lsml"] for r in faithful_raw if r.get("lsml") is not None and len(r["lsml"])]
    lsml_score_parts = [r["lsml"] for r in faithful_raw if r.get("lsml") is not None and len(r["lsml"])]
    lsml_auc = lsml_lo = lsml_hi = None
    if lsml_score_parts:
        lsml_auc, lsml_lo, lsml_hi = _auc_row(np.concatenate(lsml_labels_parts),
                                              np.concatenate(lsml_score_parts), args.n_boot)
    print(f"\n=== PAPER-FAITHFUL SUBSAMPLE (n={int(m.sum())}, target N={PAPER_POOLED['n_responses']} paper) ===")
    print(f"  EDIS AUROC:       {edis_auc} [{edis_lo},{edis_hi}]  (paper: {PAPER_POOLED['edis_auc']})")
    print(f"  mean-H AUROC:     {meanh_auc} [{meanh_lo},{meanh_hi}]  (paper: {PAPER_POOLED['meanh_auc']})")
    print(f"  L-SML GOOD_5 AUROC: {lsml_auc} [{lsml_lo},{lsml_hi}]")

    os.makedirs(os.path.dirname(args.pooled_out), exist_ok=True)
    with open(args.pooled_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant", "n", "edis_auroc", "edis_lo", "edis_hi",
                    "meanh_auroc", "meanh_lo", "meanh_hi",
                    "lsml_good5_auroc", "lsml_lo", "lsml_hi",
                    "paper_edis_auc", "paper_meanh_auc"])
        w.writerow(["pooled_full", pooled_full["n"], pooled_full["edis_auroc"], pooled_full["edis_lo"],
                    pooled_full["edis_hi"], pooled_full["meanh_auroc"], pooled_full["meanh_lo"],
                    pooled_full["meanh_hi"], pooled_full["lsml_good5_auroc"], pooled_full["lsml_lo"],
                    pooled_full["lsml_hi"], PAPER_POOLED["edis_auc"], PAPER_POOLED["meanh_auc"]])
        w.writerow(["paper_faithful", int(m.sum()), edis_auc, edis_lo, edis_hi, meanh_auc, meanh_lo,
                    meanh_hi, lsml_auc, lsml_lo, lsml_hi, PAPER_POOLED["edis_auc"], PAPER_POOLED["meanh_auc"]])
    print(f"\nwrote pooled summary -> {args.pooled_out}")


if __name__ == "__main__":
    main()
