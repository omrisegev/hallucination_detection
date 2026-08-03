#!/usr/bin/env python
"""
score_evdrop.py — answer-level scoring: reproduce "Mind the Gap" Tables 1 and 2, with our arm
as extra rows under the identical protocol.

Local CPU only; consumes a cluster raw pkl. Emits two CSVs shaped like the paper's tables plus
a diagnostics block that must be read BEFORE the numbers:

  * `accuracy` vs the paper's own pretrained accuracy for that (dataset, model). Selective
    accuracy and AURC are monotone in base error rate, so a mismatch here means the numbers are
    not comparable no matter how faithful the estimator is.
  * `n_cal_incorrect` — the paper's threshold is a quantile of the INCORRECT calibration
    samples only. When this is ~14, the alpha=0.05 "quantile" is the minimum order statistic.
  * `frac_pinned_in_negatives` — the fraction of the negative class that is cap-truncation
    rather than hallucination. Above 5%, report metrics with and without capped traces.

Usage:
    python scripts/localization/score_evdrop.py <cell_dir_or_pkl> [--out-dir results/evdrop]
    python scripts/localization/score_evdrop.py cache/repgrid/ars_gsm8k_qwen3_8b_reject
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from evidence_drop import METHODS as EVDROP_METHODS, candidate_risks
from our_arm import METHODS as OUR_METHODS, SUBSETS, fused_risk
from selective_metrics import aurc, repeated_split_eval
from spectral_utils.repgrid_scoring import load_repgrid_cell

ALPHAS = (0.05, 0.10, 0.50)


def resolve_pkl(path):
    if os.path.isfile(path):
        return path
    hits = sorted(glob.glob(os.path.join(path, "raw_*.pkl")))
    if not hits:
        raise SystemExit(f"no raw_*.pkl under {path}")
    if len(hits) > 1:
        print(f"[warn] {len(hits)} pkls found, scoring {os.path.basename(hits[0])}")
    return hits[0]


def load_diagnostics(pkl_path, manifest):
    """Trace-level facts that decide whether the metrics below mean anything."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    cap = (manifest or {}).get("max_new")
    labels, pinned = [], []
    for idx in sorted(data):
        for c in data[idx]["candidates"]:
            labels.append(bool(c.get("label", False)))
            n = len(c.get("token_entropies") or [])
            pinned.append(bool(cap) and n >= cap)
    labels, pinned = np.asarray(labels), np.asarray(pinned)
    neg = ~labels
    return {
        "n": int(labels.size),
        "accuracy": float(labels.mean()) if labels.size else float("nan"),
        "n_incorrect": int(neg.sum()),
        "max_new": cap,
        "frac_pinned": float(pinned.mean()) if pinned.size else float("nan"),
        "frac_pinned_in_negatives": float(pinned[neg].mean()) if neg.any() else float("nan"),
        "n_pinned_in_negatives": int(pinned[neg].sum()) if neg.any() else 0,
    }


def score(pkl_path, out_dir, n_splits=200, seed=0, delta=None):
    manifest_path = os.path.join(os.path.dirname(pkl_path), "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}
    diag = load_diagnostics(pkl_path, manifest)
    cell_name = os.path.basename(os.path.dirname(pkl_path)) or os.path.basename(pkl_path)

    print(f"=== {cell_name} ===")
    print(f"  model={manifest.get('model','?')} dataset={manifest.get('dataset','?')} "
          f"temps={manifest.get('temps','?')} max_new={diag['max_new']}")
    print(f"  n={diag['n']}  accuracy={diag['accuracy']:.4f}  n_incorrect={diag['n_incorrect']}")
    print(f"  frac_pinned={diag['frac_pinned']:.2%}  "
          f"WITHIN negatives={diag['frac_pinned_in_negatives']:.1%} "
          f"({diag['n_pinned_in_negatives']}/{diag['n_incorrect']})")
    paper_acc = (manifest.get("published") or {}).get("pretrained_accuracy")
    if paper_acc:
        gap = diag["accuracy"] * 100 - paper_acc
        print(f"  paper pretrained accuracy={paper_acc}  ->  gap {gap:+.2f}pp "
              f"{'[COMPARABLE]' if abs(gap) < 3 else '[NOT COMPARABLE — see module docstring]'}")
    print()

    # ── the six paper methods, on the raw candidate series ───────────────────
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    risks, labels = {m: [] for m in EVDROP_METHODS}, []
    for idx in sorted(data):
        for c in data[idx]["candidates"]:
            r = candidate_risks(c)
            for m in EVDROP_METHODS:
                risks[m].append(r[m])
            labels.append(int(bool(c.get("label", False))))
    labels = np.asarray(labels)

    # ── our arm, via the canonical feature path ──────────────────────────────
    cell = load_repgrid_cell(pkl_path)
    for sub_name, feats in SUBSETS.items():
        for meth in OUR_METHODS:
            fr, valid = fused_risk(cell, feats, method=meth)
            if fr is None:
                continue
            full = np.full(len(labels), np.nan)
            full[valid] = fr
            risks[f"ours_{meth}_{sub_name}"] = full

    rows = []
    for name, v in risks.items():
        v = np.asarray(v, dtype=float)
        ok = np.isfinite(v)
        if ok.sum() < 20 or len(set(labels[ok])) < 2:
            continue
        rec = {"cell": cell_name, "method": name,
               "n_scored": int(ok.sum()), "valid_rate": float(ok.mean()),
               "aurc_x1000": aurc(v[ok], labels[ok])}
        for a in ALPHAS:
            res = repeated_split_eval(v[ok], labels[ok], alpha=a,
                                      n_splits=n_splits, seed=seed, delta=delta)
            rec[f"sel_acc@{a}"] = res["selective_accuracy"] * 100
            rec[f"sel_acc_sd@{a}"] = res["selective_accuracy_sd"] * 100
            rec[f"coverage@{a}"] = res["coverage"]
            rec["n_cal_incorrect_mean"] = res["n_cal_incorrect_mean"]
            rec["n_cal_incorrect_min"] = res["n_cal_incorrect_min"]
        rows.append(rec)

    rows.sort(key=lambda r: r["aurc_x1000"])
    hdr = f"{'method':26s} {'AURC x1000':>10s} " + " ".join(f"{'acc@'+str(a):>12s}" for a in ALPHAS)
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        cells = " ".join(f"{r[f'sel_acc@{a}']:7.2f}+/-{r[f'sel_acc_sd@{a}']:4.2f}" for a in ALPHAS)
        print(f"{r['method']:26s} {r['aurc_x1000']:10.1f} {cells}")
    print(f"\n  n_cal_incorrect: mean {rows[0]['n_cal_incorrect_mean']:.1f}, "
          f"min {rows[0]['n_cal_incorrect_min']}"
          + ("   <-- TOO SMALL: the alpha-quantile is the minimum order statistic"
             if rows[0]["n_cal_incorrect_min"] < 20 else ""))

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{cell_name}__evdrop.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, f"{cell_name}__diagnostics.json"), "w") as f:
        json.dump({"cell": cell_name, "manifest": manifest, "diagnostics": diag}, f, indent=2)
    print(f"\n  -> {out}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cell", help="cell directory or a raw_*.pkl")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "evdrop"))
    ap.add_argument("--n-splits", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delta", type=float, default=None,
                    help="Eq. 44 finite-sample confidence level (omit for the plain Eq. 43 quantile)")
    a = ap.parse_args()
    score(resolve_pkl(a.cell), a.out_dir, a.n_splits, a.seed, a.delta)


if __name__ == "__main__":
    main()
