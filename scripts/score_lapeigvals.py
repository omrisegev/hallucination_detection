#!/usr/bin/env python
"""
score_lapeigvals.py — reproduce LapEigvals' OWN detector number (arXiv 2502.17598)
from a fetched cell's captured attention-Laplacian eigenvalues.

This is the offline half of punch-list item 8: the on-GPU reducer
(spectral_utils.model_utils.attn_laplacian_capture) stores per-candidate
`attn_lap_eigvals` [L, H, top_k] + `attn_diag_logmean` [L, H]; here we build the
paper's supervised probe from them so a lapeigvals_* cell carries a SELF-reproduced Y
(instead of only the cited 72.0/87.2). Unlike our own L-SML, this reproduces a
COMPETITOR detector, so it is explicitly supervised (the paper's probe is supervised).

Supervised probe (paper Sec. 3): concat top-k largest eigvals over all (layer, head)
-> PCA-512 -> logistic regression (class_weight='balanced'). Evaluated with the
project's mandatory recipe (SUPERVISED_ORACLE_CORRECTION.md): 5-fold stratified CV,
per-fold AUROC AVERAGED (never concatenated OOF), balanced class weights. For K>1
cells the folds are grouped by problem id so a question's candidates never straddle
folds.

Unsupervised AttentionScore (secondary): a single per-candidate scalar from
attn_diag_logmean (mean log attention self-mass) — reported orientation-free (both
signs) since it is an unsupervised statistic, matching how the paper's unsupervised
baseline family is positioned.

Usage:
    python scripts/score_lapeigvals.py --self-test           # synthetic, no data needed
    python scripts/score_lapeigvals.py --cells lapeigvals_gsm8k_llama8b
    python scripts/score_lapeigvals.py                        # all cells with attn capture
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spectral_utils.fusion_utils import boot_auc

PCA_DIM = 512
PAPER_KS = [5, 10, 20, 50, 100]   # the paper's k sweep; best-k reported


def eigvals_feature_matrix(cands, k):
    """[n, L*H*k] from candidates' attn_lap_eigvals [L,H,top_k], top-k slice.
    NaN entries (trace shorter than k) are column-median imputed. Returns (X, mask)
    where mask drops candidates lacking the capture."""
    rows, keep = [], []
    for c in cands:
        e = c.get("attn_lap_eigvals")
        if e is None:
            keep.append(False)
            continue
        e = np.asarray(e, dtype=np.float32)          # [L, H, top_k]
        rows.append(e[:, :, :k].reshape(-1))
        keep.append(True)
    if not rows:
        return None, np.asarray(keep, dtype=bool)
    X = np.vstack(rows)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X)
    X[bad] = np.take(med, np.where(bad)[1])
    return X, np.asarray(keep, dtype=bool)


def supervised_auc(X, y, groups=None, n_boot=1000, seed=42):
    """PCA-512 + balanced LR, 5-fold CV, per-fold AUROC averaged + bootstrap CI.
    Grouped folds when `groups` is given (K>1 cells)."""
    if groups is not None and len(np.unique(groups)) >= 5:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y, groups))
    else:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        splits = list(splitter.split(X, y))
    # PCA can extract at most min(n_train, n_features) components; size it to the
    # SMALLEST actual training fold so no fold's fit exceeds its rank (with 500
    # candidates the paper's 512 caps to ~n_train-1 — a data limit, noted honestly).
    min_train = min(len(tr) for tr, _ in splits)
    n_comp = min(PCA_DIM, X.shape[1], min_train - 1)
    pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_comp, random_state=seed),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, solver="lbfgs"),
    )
    fold_y, fold_p = [], []
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        pipe.fit(X[tr], y[tr])
        fold_y.append(y[te])
        fold_p.append(pipe.predict_proba(X[te])[:, 1])
    if not fold_y:
        return None
    fold_aucs = [roc_auc_score(fy, fp) for fy, fp in zip(fold_y, fold_p)]
    base = float(np.mean(fold_aucs))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        vals = []
        for fy, fp in zip(fold_y, fold_p):
            idx = rng.integers(0, len(fy), len(fy))
            if len(np.unique(fy[idx])) < 2:
                continue
            vals.append(roc_auc_score(fy[idx], fp[idx]))
        if vals:
            boot.append(np.mean(vals))
    lo, hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if boot else (base, base)
    return {"auroc": base, "lo": lo, "hi": hi, "n_comp": n_comp, "n_folds": len(fold_aucs)}


def unsup_attention_score(cands, keep, y):
    """Orientation-free AUROC of the mean attn_diag_logmean scalar (unsup AttentionScore)."""
    s = []
    for c, k in zip(cands, keep):
        if not k:
            continue
        d = c.get("attn_diag_logmean")
        s.append(float(np.nanmean(np.asarray(d, dtype=np.float32))) if d is not None else np.nan)
    s = np.asarray(s)
    ok = np.isfinite(s)
    if ok.sum() < 20 or len(np.unique(y[ok])) < 2:
        return None
    a = roc_auc_score(y[ok], s[ok])
    return {"auroc_raw": float(a), "auroc_oriented": float(max(a, 1 - a)), "n": int(ok.sum())}


def load_candidates(pkl_path):
    import pickle
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    cands, labels, pid = [], [], []
    for idx in sorted(data.keys()):
        for c in data[idx]["candidates"]:
            cands.append(c)
            labels.append(bool(c.get("label", False)))
            pid.append(int(idx))
    return cands, np.asarray(labels, dtype=int), np.asarray(pid, dtype=int)


def score_cell(preset_id, man, pkl_path):
    cands, y, pid = load_candidates(pkl_path)
    has_attn = sum(1 for c in cands if c.get("attn_lap_eigvals") is not None)
    print(f"\n== {preset_id} | {man.get('model')} | N_cand={len(cands)} "
          f"acc={y.mean():.3f} | attn-captured={has_attn}/{len(cands)} ==")
    if has_attn < 20:
        print("  [skip] fewer than 20 candidates carry attn_lap_eigvals — "
              "cell not re-run with capture_attention yet")
        return []
    k_per_cell = man.get("k", 1)
    out = []
    best = None
    for k in PAPER_KS:
        X, keep = eigvals_feature_matrix(cands, k)
        yk = y[keep]
        groups = pid[keep] if k_per_cell > 1 else None
        res = supervised_auc(X, yk, groups=groups)
        if res is None:
            continue
        print(f"  k={k:>3}: supervised PCA{res['n_comp']}+LR = {res['auroc']:.4f} "
              f"[{res['lo']:.3f},{res['hi']:.3f}]  (folds={res['n_folds']})")
        row = {"cell": preset_id, "model": man.get("model"), "k": k,
               "sup_auroc": round(res["auroc"], 4), "lo": round(res["lo"], 4),
               "hi": round(res["hi"], 4), "n": int(keep.sum()),
               "grouped": groups is not None}
        out.append(row)
        if best is None or res["auroc"] > best["sup_auroc"]:
            best = row
    _, keep = eigvals_feature_matrix(cands, PAPER_KS[0])
    us = unsup_attention_score(cands, keep, y[keep])
    if us:
        print(f"  unsup AttentionScore (mean diag-logmean): raw {us['auroc_raw']:.4f} / "
              f"oriented {us['auroc_oriented']:.4f}")
    if best:
        pub = man.get("published", {})
        print(f"  -> best supervised k={best['k']}: {best['sup_auroc']:.4f}  "
              f"(paper sup {pub.get('supervised')} / unsup {pub.get('value')})")
    return out


def discover(cache_dir, only=None):
    for man_path in sorted(glob.glob(os.path.join(cache_dir, "*", "manifest.json"))):
        d = os.path.dirname(man_path)
        pid = os.path.basename(d)
        if only and not any(s in pid for s in only):
            continue
        pkls = sorted(glob.glob(os.path.join(d, "raw_*.pkl")))
        if pkls:
            with open(man_path) as f:
                yield pid, json.load(f), pkls[0]


def self_test():
    """Synthetic end-to-end: eigval tensors whose top-block correlates with the label;
    the supervised probe must clear ~0.5, NaN-imputation + grouped folds must run."""
    print("== self-test (synthetic) ==")
    rng = np.random.default_rng(0)
    L, H, K = 4, 8, 100
    n = 200
    y = (rng.random(n) > 0.5).astype(int)
    cands = []
    for i in range(n):
        e = rng.standard_normal((L, H, K)).astype(np.float32)
        e[0, 0, :10] += 1.5 * y[i]                    # a learnable signal block
        if i % 25 == 0:                                # a few short traces -> NaN tail
            e[:, :, 60:] = np.nan
        cands.append({"attn_lap_eigvals": e,
                      "attn_diag_logmean": rng.standard_normal((L, H)).astype(np.float32) + 0.4 * y[i],
                      "label": bool(y[i])})
    ok = True
    X, keep = eigvals_feature_matrix(cands, 100)
    ok &= (X.shape == (n, L * H * 100)) and np.isfinite(X).all()
    print(f"  [{'ok  ' if ok else 'FAIL'}] feature matrix {X.shape}, NaN-imputed finite")
    res = supervised_auc(X, y[keep], groups=None)
    good = res is not None and res["auroc"] > 0.6
    print(f"  [{'ok  ' if good else 'FAIL'}] supervised AUROC={res['auroc']:.3f} on planted signal (>0.6)")
    grp = np.repeat(np.arange(n // 2), 2)[:n]
    resg = supervised_auc(X, y[keep], groups=grp)
    okg = resg is not None
    print(f"  [{'ok  ' if okg else 'FAIL'}] grouped-CV path runs (AUROC={resg['auroc']:.3f})" if okg
          else "  [FAIL] grouped-CV path")
    us = unsup_attention_score(cands, keep, y[keep])
    oku = us is not None and us["auroc_oriented"] >= 0.5
    print(f"  [{'ok  ' if oku else 'FAIL'}] unsup AttentionScore oriented={us['auroc_oriented']:.3f}")
    passed = ok and good and okg and oku
    print(f"\n{'SELF-TEST PASS' if passed else 'SELF-TEST FAIL'}")
    sys.exit(0 if passed else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache/repgrid")
    ap.add_argument("--out", default="results/repgrid")
    ap.add_argument("--cells", default=None, help="comma-sep substrings")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()

    only = args.cells.split(",") if args.cells else None
    rows = []
    for pid, man, pkl in discover(args.cache_dir, only=only):
        rows.extend(score_cell(pid, man, pkl))
    if rows:
        os.makedirs(args.out, exist_ok=True)
        out_csv = os.path.join(args.out, "lapeigvals_selfrepro.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {out_csv}")
    else:
        print("\nNo cells with attention capture found "
              "(run a lapeigvals_* preset with capture={'attention': True} first).")


if __name__ == "__main__":
    main()
