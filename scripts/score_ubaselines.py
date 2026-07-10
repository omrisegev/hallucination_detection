#!/usr/bin/env python
"""
score_ubaselines.py — score the standard unsupervised gray-box baselines from the
July-2026 SOTA survey on OUR replication-grid traces, so they sit next to our L-SML
numbers computed on the exact same rows.

Baselines (all computed from raw signals already saved per candidate — no model runs):
    Perplexity / mean logprob   confidence = -mean(token_spilled_energies)   [K=1 units]
    Sequence logprob            confidence = -sum(token_spilled_energies)    [K=1 units]
    Naive entropy               confidence = -mean(token_entropies)  (== EPR / our `epr`)
    LN-entropy (Malinin-Gales)  K>=2 cells only, QUESTION level: mean over the K samples
                                of per-sample length-normalized -logprob; majority-vote
                                correctness label per question (survey/NI convention)
    Predictive entropy          K>=2 cells only, question level, non-length-normalized

The script deliberately avoids extract_all_features (FFT et al.) so the 100 MB - 1 GB
cells stay cheap: one pickle load, one pass. Our L-SML GOOD_5 AUROC per cell is JOINED
from results/repgrid/scores_lsml_upcr.csv (already computed by score_repgrid.py), not
recomputed.

Dual-label reporting (Janiak et al. 2508.08285 caveat): whenever a cell carries both a
judge label and a lexical label that disagree, every candidate-level baseline is scored
under BOTH schemes and the agreement rate is reported.

Usage:
    PYTHONPATH=. python scripts/score_ubaselines.py                       # all cells
    PYTHONPATH=. python scripts/score_ubaselines.py --cells lapeigvals_gsm8k_llama8b
Cells >100 MB: run in the background with a generous timeout (CLAUDE.md rule).
"""
import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from spectral_utils.fusion_utils import boot_auc

LSML_CSV = "results/repgrid/scores_lsml_upcr.csv"


def _auc(y, score, n_boot):
    """AUROC + CI on finite rows with both classes; None-triple otherwise."""
    m = np.isfinite(score)
    y, score = y[m].astype(int), score[m]
    if len(y) < 20 or y.sum() in (0, len(y)):
        return None, None, None
    auc, lo, hi = boot_auc(y, score, n=n_boot)
    return round(float(auc), 4), round(float(lo), 4), round(float(hi), 4)


def load_cell_signals(pkl_path):
    """One pass over the raw pkl -> per-candidate baseline signals + labels.
    Candidate order matches load_repgrid_cell (sorted problem ids, list order)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    ppl, seqlp, nent, lab, lab_lex, pid = [], [], [], [], [], []
    for idx in sorted(data.keys()):
        for c in data[idx]["candidates"]:
            dE = c.get("token_spilled_energies")
            H = c.get("token_entropies")
            ppl.append(-float(np.mean(dE)) if dE is not None and len(dE) else np.nan)
            seqlp.append(-float(np.sum(dE)) if dE is not None and len(dE) else np.nan)
            nent.append(-float(np.mean(H)) if H is not None and len(H) else np.nan)
            label = bool(c.get("label", False))
            lab.append(label)
            lab_lex.append(bool(c.get("label_lexical", label)))
            pid.append(int(idx))
    return {
        "ppl": np.asarray(ppl), "seqlp": np.asarray(seqlp), "nent": np.asarray(nent),
        "labels": np.asarray(lab, dtype=bool), "labels_lex": np.asarray(lab_lex, dtype=bool),
        "problem_id": np.asarray(pid, dtype=int), "n_problems": len(data),
    }


def question_level(sig, n_boot):
    """LN-entropy + predictive entropy per question (K>=2), majority-vote label."""
    out = {"k": None, "lnpe_q_auroc": None, "pe_q_auroc": None, "n_questions": None}
    pids = np.unique(sig["problem_id"])
    counts = np.bincount(np.searchsorted(pids, sig["problem_id"]))
    k = int(np.median(counts))
    out["k"] = k
    if k < 2:
        return out
    ln_conf, pe_conf, maj = [], [], []
    for p in pids:
        m = sig["problem_id"] == p
        if not np.isfinite(sig["ppl"][m]).any():
            continue
        # ppl = -mean(dE) per candidate; LNPE uncertainty = mean_k mean_t dE = -mean(ppl)
        ln_conf.append(float(np.nanmean(sig["ppl"][m])))
        pe_conf.append(float(np.nanmean(sig["seqlp"][m])))
        maj.append(float(sig["labels"][m].mean()) > 0.5)
    y = np.asarray(maj, dtype=bool)
    out["n_questions"] = len(y)
    out["lnpe_q_auroc"] = _auc(y, np.asarray(ln_conf), n_boot)[0]
    out["pe_q_auroc"] = _auc(y, np.asarray(pe_conf), n_boot)[0]
    return out


def load_lsml_join(path=LSML_CSV):
    """cell -> GOOD_5 L-SML AUROC from the existing score_repgrid output."""
    join = {}
    if not os.path.exists(path):
        return join
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("subset") == "GOOD_5" and r.get("method", "lsml") == "lsml":
                try:
                    join[r["cell"]] = round(float(r["auroc_X"]), 4)
                except (KeyError, ValueError):
                    pass
    return join


def score_one(pkl_path, cell_id, lsml_join, n_boot):
    sig = load_cell_signals(pkl_path)
    y, y_lex = sig["labels"], sig["labels_lex"]
    agree = float((y == y_lex).mean())
    res = {"cell": cell_id, "n_cands": len(y), "n_problems": sig["n_problems"],
           "acc": round(float(y.mean()), 4), "acc_lex": round(float(y_lex.mean()), 4),
           "label_agreement": round(agree, 4)}
    for name, s in (("ppl", sig["ppl"]), ("seqlp", sig["seqlp"]), ("nent", sig["nent"])):
        auc, lo, hi = _auc(y, s, n_boot)
        res[f"{name}_auroc"], res[f"{name}_lo"], res[f"{name}_hi"] = auc, lo, hi
        # Janiak dual-label check: same baseline under the OTHER label scheme.
        res[f"{name}_auroc_lex"] = _auc(y_lex, s, n_boot)[0] if agree < 1.0 else auc
    res.update(question_level(sig, n_boot))
    res["lsml_good5_auroc"] = lsml_join.get(cell_id)
    print(f"== {cell_id}: ppl={res['ppl_auroc']} seqlp={res['seqlp_auroc']} "
          f"nent={res['nent_auroc']} | LNPE_q={res['lnpe_q_auroc']} (K={res['k']}) "
          f"| L-SML GOOD_5={res['lsml_good5_auroc']} | agree={res['label_agreement']} "
          f"(n={res['n_cands']}, acc={res['acc']})", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache/repgrid")
    ap.add_argument("--cells", nargs="*", default=None, help="cell dir names; default all")
    ap.add_argument("--out", default="results/repgrid/ubaseline_scores.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()

    jobs = []
    for man in sorted(glob.glob(os.path.join(args.cache_dir, "*", "manifest.json"))):
        cell_dir = os.path.dirname(man)
        cell_id = os.path.basename(cell_dir)
        if args.cells and cell_id not in args.cells:
            continue
        pkls = sorted(glob.glob(os.path.join(cell_dir, "raw_*.pkl")))
        if pkls:
            jobs.append((pkls[0], cell_id))

    lsml_join = load_lsml_join()
    rows = [score_one(p, c, lsml_join, args.n_boot) for p, c in jobs]
    if not rows:
        print("no cells matched")
        return
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
