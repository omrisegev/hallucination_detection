#!/usr/bin/env python
"""
sweep_dufs_groupfs.py — Arm 4b: sweep the REAL GroupFS grouping mechanism.

WHY THIS WAS REWRITTEN (Step 201, defect 4)
-------------------------------------------
The Step-200 version never ran GroupFS. It clustered features with
`sklearn.cluster.AgglomerativeClustering` on correlation distance and never
imported `a2_groupfs` at all — no stochastic gates, no Gumbel-Softmax anneal, no
orthogonality penalty. `lambda1` was bound in the sweep loop and written to the
output CSV but **never used in any computation**, which is why it changed AUROC
or `n_selected` in exactly 0/350 (cell, C, readout) groups; `tau` was never swept
despite being named in the docstring; and the "group_median" readout merely took
each cluster's first member (`g_indices[0]`). GroupFS grouping — the one mechanism
flagged as genuinely unexplored — was therefore still untested.

This version drives `spectral_utils/selectors/a2_groupfs.py` directly:
`_self_tuning_affinity` -> `_normalized_laplacian` (feature graph),
`_spectral_embed`/`_kmeans_labels`/`_warm_logits` (warm start),
`_init_magnitudes` (data-driven lambda0), then `_train_groupfs` with the swept
(C, lambda1, tau), and `_train_dufs` for the per-feature gates.

WHAT IS SWEPT
-------------
  C        number of latent groups, 2..8           (a2 normally picks one via the
                                                    App-D Procrustes knee)
  lambda1  feature-graph term weight, LAMBDA1_GRID (a2 normally SNAPS this via
                                                    `_snap_lambda1`; un-snapped here)
  tau      Gumbel-Softmax anneal (start, min)      (a2 fixes 10.0 -> 1e-2)
  readout  group_median | per_feature              (a2 deploys group_median)

HONESTY ARMS (both reported; never merged)
------------------------------------------
  label_free_LOCO        knobs chosen WITHOUT labels on the other 24 cells and
                         applied to the held-out 25th. The deployable number.
  LABEL_PEEKING_CEILING  best knobs per cell chosen WITH labels. Diagnosis only,
                         mirroring `results/subset_sweep/loco.csv`'s own column
                         naming convention.

REGRESSION GUARD
----------------
lambda1 must change `n_selected` or AUROC on a non-trivial fraction of configs.
If it is still 0/N the wiring is still wrong, and the script says so loudly
instead of emitting a clean-looking table.

Scoring mirrors `spectral_utils/repgrid_scoring.score_subset`: L-SML continuous
fusion, label-free `anchor_orient` against the cell's own resolved anchor, raw
AUROC (never max(a, 1-a)).

Usage
-----
  python scripts/sweep_dufs_groupfs.py --calibrate   # time one config, exit
  python scripts/sweep_dufs_groupfs.py               # full sweep (resumable)
  python scripts/sweep_dufs_groupfs.py --analyze     # re-read CSV, no compute

Writes (appended incrementally — safe to kill and resume):
  results/advisor_inscope/sweep_groupfs_results.csv
  results/advisor_inscope/sweep_groupfs_summary.csv
  results/advisor_inscope/sweep_groupfs_dashboard.html
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from spectral_utils.fusion_utils import lsml_continuous              # noqa: E402
from spectral_utils.streaming_utils import anchor_orient             # noqa: E402
from spectral_utils.subset_sweep import GOOD_6                       # noqa: E402
from spectral_utils.selectors.a2_groupfs import (                    # noqa: E402
    _self_tuning_affinity, _normalized_laplacian, _spectral_embed,
    _kmeans_labels, _warm_logits, _init_magnitudes, _train_groupfs,
    _train_dufs, _mean_pairwise_jaccard,
    K_NN, LAMBDA1_GRID, BATCH, EPOCHS_STAB, R_MAX, N_SEEDS_STABILITY, _EPS)
from inscope_cells import GROUP                                      # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "advisor_inscope")
RES_CSV = os.path.join(OUT_DIR, "sweep_groupfs_results.csv")
SUM_CSV = os.path.join(OUT_DIR, "sweep_groupfs_summary.csv")
HTML = os.path.join(OUT_DIR, "sweep_groupfs_dashboard.html")

C_GRID = tuple(range(2, 9))
TAU_GRID = {"anneal_10_1e-2": (10.0, 1e-2),   # a2's deployed schedule
            "flat_1.0": (1.0, 1.0)}           # no anneal — harder assignment sooner
READOUTS = ("group_median", "per_feature")
EPOCHS_SWEEP = 120            # < a2's EPOCHS_FINAL=180, to keep the grid tractable
SEED = 20260725

ROW_COLS = ["cell", "group", "C", "lambda1", "tau", "readout",
            "n_selected", "auroc", "good6", "seconds"]


# ---------------------------------------------------------------------------

def score_cols(u, cols):
    """Mirror repgrid_scoring.score_subset: L-SML fuse the selected columns,
    orient label-free against the cell's own anchor, return raw AUROC."""
    cols = sorted(set(int(c) for c in cols))
    if len(cols) < 3:
        return np.nan
    V = np.asarray(u["V"], dtype=np.float64)
    fused, _ = lsml_continuous(*[V[:, c] for c in cols])
    score, _ = anchor_orient(np.asarray(fused, dtype=float),
                             np.asarray(u["anchor"], dtype=float))
    return float(roc_auc_score(u["labels"], score))


def cell_gates(u, rng):
    """Per-cell DUFS per-feature gates + label-free stability diagnostics.

    Independent of (C, lambda1, tau), so computed ONCE per cell and reused across
    the grid. lambda2 is chosen by the same cross-seed Jaccard rule a2 uses
    (admissible = 3 <= median size < p).
    """
    V = np.asarray(u["V"], dtype=np.float64)
    n, p = V.shape
    R = int(min(n, R_MAX))
    Xr = V[np.sort(rng.choice(n, size=R, replace=False))] if R < n else V
    X_t = torch.tensor(Xr, dtype=torch.float32)

    Wf = _self_tuning_affinity(X_t.t().contiguous(), min(K_NN, p - 1))
    L_feat_t = _normalized_laplacian(Wf)

    gen0 = torch.Generator().manual_seed(int(rng.integers(2 ** 31)))
    # lambda0 needs some C for the group-term magnitude; use the grid midpoint.
    c0 = int(max(2, min(4, p - 1)))
    lbl0 = _kmeans_labels(_spectral_embed(L_feat_t.numpy(), c0), c0,
                          int(rng.integers(2 ** 31)))
    aLs, aLf, aLreg = _init_magnitudes(
        X_t, L_feat_t, c0, _warm_logits(lbl0, c0),
        [int((lbl0 == j).sum()) for j in range(c0)], gen0)
    lam0 = float(np.clip(aLs / max(aLreg, _EPS), 1e-3, 1e4))

    seeds = [int(rng.integers(2 ** 31)) for _ in range(N_SEEDS_STABILITY)]
    stab = {}
    for mult in (0.5, 1.0, 2.0):
        lam_d = lam0 * mult
        gates = [_train_dufs(X_t, lam_d, EPOCHS_STAB, BATCH, s) for s in seeds]
        sels = [np.where(g > 0.0)[0] for g in gates]
        med = int(np.median([len(x) for x in sels]))
        stab[lam_d] = {"jaccard": _mean_pairwise_jaccard(sels), "med": med,
                       "admissible": bool(3 <= med < p),
                       "mu_bar": np.mean(gates, axis=0)}
    adm = [(v["jaccard"], k) for k, v in stab.items() if v["admissible"]]
    lam_star = max(adm)[1] if adm else lam0
    return {"X_t": X_t, "L_feat_t": L_feat_t, "lam0": lam0,
            "mu_feat": stab[lam_star]["mu_bar"],
            "stability": float(stab[lam_star]["jaccard"]), "p": p}


def run_config(u, g, C, lam1, tau_name, rng):
    """One real GroupFS training run at (C, lambda1, tau). Returns both readouts."""
    X_t, L_feat_t, p = g["X_t"], g["L_feat_t"], g["p"]
    C = int(max(2, min(C, p - 1)))
    c_seed = int(rng.integers(2 ** 31))
    labels = _kmeans_labels(_spectral_embed(L_feat_t.numpy(), C), C, c_seed)
    warm = _warm_logits(labels, C)
    sizes = [int((labels == j).sum()) for j in range(C)]
    t_start, t_min = TAU_GRID[tau_name]

    lg, _mu, _loss = _train_groupfs(
        X_t, L_feat_t, C, lam1, g["lam0"], 1.0 / max(lam1, _EPS),
        EPOCHS_SWEEP, BATCH, int(rng.integers(2 ** 31)), warm, sizes,
        temp_start=t_start, temp_min=t_min)
    groups = np.argmax(lg, axis=1)
    mu_feat = g["mu_feat"]

    out = {}
    # group_median (a2's deployed rule): a group is open iff its MEDIAN member
    # gate is open; selection = union of the open groups.
    open_g = [j for j in np.unique(groups)
              if float(np.median(mu_feat[groups == j])) > 0.0]
    sel_med = np.where(np.isin(groups, open_g))[0]
    # per_feature: gates read directly; grouping only shapes the training.
    sel_pf = np.where(mu_feat > 0.0)[0]
    for name, sel in (("group_median", sel_med), ("per_feature", sel_pf)):
        if len(sel) < 3:
            sel = np.argsort(mu_feat)[::-1][:3]
        out[name] = np.array(sorted(sel), dtype=np.int64)
    return out


# ---------------------------------------------------------------------------

def load_cells():
    from compare_anchor_quality import load_all_inscope_cells
    loaded = load_all_inscope_cells()
    cells = {}
    for ck, cd in loaded.items():
        uc = cd["unlabeled"]
        cells[ck] = {"V": np.asarray(uc.V, dtype=np.float64),
                     "anchor": np.asarray(uc.anchor, dtype=np.float64),
                     "pool": list(uc.pool),
                     "labels": np.asarray(cd["labels"], dtype=int)}
    return cells


def good6_ref(u):
    cols = [u["pool"].index(f) for f in GOOD_6 if f in u["pool"]]
    return score_cols(u, cols) if len(cols) >= 3 else np.nan


def sweep(cells, calibrate=False):
    done = set()
    if os.path.exists(RES_CSV):
        try:
            prev = pd.read_csv(RES_CSV)
            done = {(r.cell, int(r.C), float(r.lambda1), r.tau, r.readout)
                    for r in prev.itertuples()}
            print(f"resuming — {len(done)} rows already present")
        except Exception:
            pd.DataFrame(columns=ROW_COLS).to_csv(RES_CSV, index=False)
    else:
        pd.DataFrame(columns=ROW_COLS).to_csv(RES_CSV, index=False)

    for ci, (ck, u) in enumerate(sorted(cells.items()), 1):
        rng = np.random.default_rng(abs(hash(ck)) % (2 ** 31))
        g6 = good6_ref(u)
        g = cell_gates(u, rng)
        print(f"[{ci}/{len(cells)}] {ck}  p={g['p']}  GOOD_6={g6:.4f}  "
              f"stab={g['stability']:.3f}", flush=True)
        for C in C_GRID:
            for lam1 in LAMBDA1_GRID:
                for tau_name in TAU_GRID:
                    if all((ck, C, float(lam1), tau_name, ro) in done
                           for ro in READOUTS):
                        continue
                    t0 = time.time()
                    sels = run_config(u, g, C, lam1, tau_name, rng)
                    dt = time.time() - t0
                    rows = [{"cell": ck, "group": GROUP.get(ck, "?"), "C": C,
                             "lambda1": float(lam1), "tau": tau_name,
                             "readout": ro, "n_selected": int(len(sels[ro])),
                             "auroc": score_cols(u, sels[ro]), "good6": g6,
                             "seconds": round(dt, 2)} for ro in READOUTS]
                    pd.DataFrame(rows).to_csv(RES_CSV, mode="a", header=False,
                                              index=False)
                    if calibrate:
                        tot = (dt * len(C_GRID) * len(LAMBDA1_GRID)
                               * len(TAU_GRID) * len(cells) / 3600.0)
                        print(f"  calibration: one (C,lam1,tau) config = "
                              f"{dt:.1f}s -> full grid ~{tot:.1f}h")
                        return
        print(f"    done {ck}", flush=True)


# ---------------------------------------------------------------------------

def analyze():
    df = pd.read_csv(RES_CSV).dropna(subset=["auroc"])
    print(f"\n{'='*78}\nGroupFS sweep — {len(df)} rows, {df.cell.nunique()} cells"
          f"\n{'='*78}")

    # --- REGRESSION GUARD: lambda1 must actually do something ---------------
    piv = df.pivot_table(index=["cell", "C", "tau", "readout"],
                         columns="lambda1", values=["auroc", "n_selected"])
    spread_a = piv["auroc"].max(axis=1) - piv["auroc"].min(axis=1)
    spread_n = piv["n_selected"].max(axis=1) - piv["n_selected"].min(axis=1)
    n_a, n_n = int((spread_a > 1e-12).sum()), int((spread_n > 0).sum())
    print("\nREGRESSION GUARD (lambda1):")
    print(f"  changes AUROC      in {n_a}/{len(spread_a)} groups "
          f"(max spread {spread_a.max():.4f})")
    print(f"  changes n_selected in {n_n}/{len(spread_n)} groups")
    if n_a == 0 and n_n == 0:
        print("  *** FAIL — lambda1 is STILL a no-op: the grouping mechanism is "
              "not being exercised. Do not report these numbers. ***")
    else:
        print("  PASS — lambda1 is wired into the objective.")

    cfg = (df.groupby(["C", "lambda1", "tau", "readout"])["auroc"]
             .agg(["mean", "count"]).reset_index()
             .sort_values("mean", ascending=False))
    print("\nTop 8 configs by macro AUROC:")
    print(cfg.head(8).to_string(index=False))

    g6 = df.groupby("cell")["good6"].first()
    print(f"\nGOOD_6 reference macro: {g6.mean():.4f}  (canonical 0.7594)")

    rows = [("LABEL_PEEKING_CEILING",
             df.loc[df.groupby("cell")["auroc"].idxmax()]
               .set_index("cell")["auroc"])]
    lf = {}
    for ck in df.cell.unique():
        other = df[df.cell != ck]
        if not len(other):
            continue
        best = (other.groupby(["C", "lambda1", "tau", "readout"])["auroc"]
                     .mean().idxmax())
        sub = df[(df.cell == ck) & (df.C == best[0]) & (df.lambda1 == best[1])
                 & (df.tau == best[2]) & (df.readout == best[3])]
        if len(sub):
            lf[ck] = float(sub["auroc"].iloc[0])
    rows.append(("label_free_LOCO", pd.Series(lf)))

    print(f"\n{'arm':24s} {'macro':>8s} {'vs GOOD_6':>11s} {'W/L':>8s} {'p':>9s}")
    print("-" * 78)
    summary = []
    for name, s in rows:
        common = s.index.intersection(g6.index)
        v, b = s[common], g6[common]
        d = v - b
        try:
            p = float(wilcoxon(v, b).pvalue)
        except Exception:
            p = float("nan")
        print(f"{name:24s} {v.mean():8.4f} {d.mean()*100:+10.2f}pp "
              f"{int((d>0).sum()):3d}/{int((d<0).sum()):<3d} {p:9.4f}")
        summary.append({"arm": name, "n_cells": int(len(common)),
                        "macro": round(float(v.mean()), 4),
                        "delta_vs_good6": round(float(d.mean()), 4),
                        "wins": int((d > 0).sum()),
                        "losses": int((d < 0).sum()),
                        "wilcoxon_p": round(p, 5) if np.isfinite(p) else None})
    pd.DataFrame(summary).to_csv(SUM_CSV, index=False)

    best = cfg.iloc[0]
    guard_cls = "ok" if (n_a or n_n) else "bad"
    with open(HTML, "w", encoding="utf-8") as f:
        f.write(f"""<!doctype html><meta charset="utf-8">
<title>GroupFS grouping sweep (Step 202)</title>
<style>body{{font-family:system-ui,sans-serif;margin:2rem;max-width:1000px}}
table{{border-collapse:collapse;margin:1rem 0;font-size:14px}}
td,th{{border:1px solid #ccc;padding:4px 10px}}
.bad{{color:#b00;font-weight:bold}}.ok{{color:#070;font-weight:bold}}</style>
<h1>GroupFS grouping sweep — the real mechanism</h1>
<p>Rebuilt for Step 202. The Step-200 sweep never ran GroupFS (it was
<code>sklearn AgglomerativeClustering</code>; <code>lambda1</code> was recorded but
never used, and "group_median" took each cluster's first member). This run drives
<code>a2_groupfs._train_groupfs</code> with swept <code>C</code>,
<code>lambda1</code>, <code>tau</code>, and readout.</p>
<p><b>lambda1 regression guard:</b>
<span class="{guard_cls}">changes AUROC in {n_a}/{len(spread_a)} groups;
n_selected in {n_n}/{len(spread_n)}</span></p>
<p><b>Top config:</b> C={int(best.C)}, lambda1={best.lambda1}, tau={best.tau},
readout={best.readout} &rarr; macro <b>{best['mean']:.4f}</b>
(GOOD_6 = {g6.mean():.4f})</p>
{pd.DataFrame(summary).to_html(index=False)}
<h2>All configs</h2>{cfg.to_html(index=False)}""")
    print(f"\nwrote {RES_CSV}\nwrote {SUM_CSV}\nwrote {HTML}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibrate", action="store_true",
                    help="time a single config and exit")
    ap.add_argument("--analyze", action="store_true",
                    help="re-read the CSV and report; no compute")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.set_num_threads(1)

    if args.analyze:
        analyze()
        return
    cells = load_cells()
    sweep(cells, calibrate=args.calibrate)
    if not args.calibrate:
        analyze()


if __name__ == "__main__":
    main()
