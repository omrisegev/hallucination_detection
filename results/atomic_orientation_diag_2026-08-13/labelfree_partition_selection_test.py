#!/usr/bin/env python3
"""Can label-free criteria identify the good random partitions?

Reproduces Codex's 50 seeded random partitions of the 17 frozen atoms
(size profile matched to the provenance families), computes label-free
quality criteria for each (originals only, no labels), and correlates them
with the already-published labeled quality (cross-domain mean delta from
results/atomic_nrm_retrospective_controls_v1/random_partition_summary.csv).

Pre-stated criteria (all label-free):
  C1_med  = median over 8 LOFO folds of |cos(fold direction, all-23 direction)|
  C1_min  = min over folds of the same
  C2_g3   = |cos(all-23 direction, pooled group-level gamma3 witness)|
  C3_gap  = unit-distance gap of the selected mode (2nd closest |l-1| - closest)
  C4_sign = |sum(direction)| / sqrt(G)   (sign-anchor margin)
  C_comb  = rank(C1_med) + rank(C2_g3)

Fidelity anchor: partition #36 scored through this pipeline on originals LOFO
must match the published +0.514pp (labels used ONLY for this check and for
the correlation targets, never inside a criterion).
"""

from __future__ import annotations

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).resolve().parent
MW = SCRATCH / "mw"
REAL = Path(r"c:/Users/omris/TAU/hallucination_detection")
sys.path.insert(0, str(MW))

from sklearn.metrics import roc_auc_score  # noqa: E402

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    load_contract,
    family as original_family,
)
from scripts.atomic_nrm_structural_audit import SOURCE_CELLS  # noqa: E402
from scripts.atomic_nrm_retrospective_controls import (  # noqa: E402
    FROZEN_FEATURES,
    RANDOM_PARTITIONS,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.specrage_views import FEATURE_TO_VIEW  # noqa: E402

OUT = SCRATCH / "labelfree_partition_selection"
OUT.mkdir(exist_ok=True)
BUNDLE = REAL / "results" / "dependency_fusion_raw" / "cells.npz"
SUMMARY = (MW / "results" / "atomic_nrm_retrospective_controls_v1"
           / "random_partition_summary.csv")
EPS = 1e-12


def log(msg=""):
    print(msg, flush=True)


def standardize(x):
    m, s = float(np.mean(x)), float(np.std(x))
    return (x - m) / (s if s > EPS else 1.0), s


def residual_column(h, b):
    hz, s = standardize(h)
    if s <= EPS:
        return None
    beta = float(hz @ b / max(b @ b, EPS))
    rz, s2 = standardize(hz - beta * b)
    return rz if s2 > EPS else None


def hermite3(col, b):
    phi = b ** 3 - 3.0 * b
    phi -= float(phi @ b / (b @ b)) * b
    phi -= float(np.mean(phi))
    return -float(col @ phi) / len(b)


def cosv(u, v):
    du, dv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (du * dv)) if du > 0 and dv > 0 else float("nan")


def random_mappings():
    """Exact reproduction of the controls script's partition generator."""
    fam = {name: FEATURE_TO_VIEW[name] for name in FROZEN_FEATURES}
    sizes = sorted((
        sum(value == group for value in fam.values())
        for group in set(fam.values())
    ), reverse=True)
    groups = tuple(f"random_{index}" for index in range(len(sizes)))
    mappings = []
    for seed in range(RANDOM_PARTITIONS):
        rng = np.random.default_rng(73000 + seed)
        shuffled = np.asarray(FROZEN_FEATURES)[
            rng.permutation(len(FROZEN_FEATURES))
        ]
        mapping, start = {}, 0
        for group, size in zip(groups, sizes):
            mapping.update({name: group for name in shuffled[start:start + size]})
            start += size
        mappings.append(mapping)
    return mappings, groups


def group_residuals(cell, mapping, group_order):
    cols = {}
    for g in group_order:
        idx = [i for i, nm in enumerate(cell["names"])
               if mapping.get(nm) == g]
        if not idx:
            continue
        col = residual_column(cell["w"][idx] @ cell["F"][idx], cell["b"])
        if col is not None:
            cols[g] = col
    return cols


def calibrate(cells, mapping, group_order):
    """Equal-cell pairwise covariance, argmin|l-1|, all-ones sign —
    exactly the rule that produced the published random-partition scores."""
    G = len(group_order)
    cov = np.zeros((G, G))
    cnt = np.zeros((G, G), dtype=int)
    cache = []
    for c in cells:
        cols = group_residuals(c, mapping, group_order)
        cache.append(cols)
        present = [g for g in group_order if g in cols]
        li = [group_order.index(g) for g in present]
        V = np.column_stack([cols[g] for g in present])
        cov[np.ix_(li, li)] += V.T @ V / len(V)
        cnt[np.ix_(li, li)] += 1
    cov = cov / np.maximum(cnt, 1)
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    dist = np.abs(vals - 1.0)
    j = int(np.argmin(dist))
    v = vecs[:, j].copy()
    if float(np.sum(v)) < 0:
        v *= -1
    rest = np.delete(dist, j)
    gap = float(np.min(rest) - dist[j]) if len(rest) else 0.0
    return {"direction": v, "eigenvalues": vals, "selected": j,
            "gap": gap, "cols_cache": cache}


def score_cell(cell, mapping, group_order, direction):
    cols = group_residuals(cell, mapping, group_order)
    present = [g for g in group_order if g in cols]
    if len(present) < 2:
        return 0.0
    li = [group_order.index(g) for g in present]
    d = np.asarray(direction)[li]
    R = np.column_stack([cols[g] for g in present])
    q = R @ d
    sd = float(np.std(q))
    iu = float(roc_auc_score(cell["y"], cell["b"]))
    if sd <= EPS:
        return 0.0
    s = cell["b"] + (q / sd) * (1.0 / len(present))
    return 100.0 * (float(roc_auc_score(cell["y"], s)) - iu)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t0 = time.time()
    log("loading originals...")
    cells = []
    with np.load(BUNDLE, allow_pickle=True) as data:
        for name in SOURCE_CELLS:
            F, names = load_contract(data, name, "mixed_v2")
            y = np.asarray(data[f"{name}__labels"], dtype=int)
            F = np.asarray(F, float)
            w = upcr_fit(F, **IU_FIT_DEFAULTS).w
            b, _ = standardize(w @ F)
            cells.append({"cell": name, "group": original_family(name),
                          "names": tuple(names), "F": F, "w": w, "b": b,
                          "y": y, "n": len(y)})
    groups_lofo = sorted({c["group"] for c in cells})

    # published labeled quality
    labeled = defaultdict(dict)
    for r in csv.DictReader(SUMMARY.open(encoding="utf-8")):
        labeled[int(r["random_partition"])][r["domain"]] = float(
            r["equal_group_delta_pp"]
        )
    doms = sorted(next(iter(labeled.values())))
    lab_mean = {p: float(np.mean([labeled[p][d] for d in doms]))
                for p in labeled}
    lab_orig = {p: labeled[p]["original_23"] for p in labeled}

    mappings, group_order = random_mappings()
    log(f"reproduced {len(mappings)} partitions, groups {group_order}")

    # ---- fidelity anchor: LOFO-score partition 36 and compare to +0.514 ----
    def lofo_delta(mapping):
        rows = []
        for c in cells:
            src = [cc for cc in cells if cc["group"] != c["group"]]
            cal = calibrate(src, mapping, list(group_order))
            rows.append((c["group"], score_cell(
                c, mapping, list(group_order), cal["direction"]
            )))
        per_group = [np.mean([d for g, d in rows if g == gg])
                     for gg in groups_lofo]
        return float(np.mean(per_group))

    check36 = lofo_delta(mappings[36])
    log(f"fidelity: partition 36 originals LOFO through this pipeline: "
        f"{check36:+.3f}pp (published +0.514)")

    # ---- label-free criteria for every partition ----
    results = []
    for k, mapping in enumerate(mappings):
        cal_all = calibrate(cells, mapping, list(group_order))
        d_all = cal_all["direction"]
        # C1: LOFO direction stability
        fold_cos = []
        for gg in groups_lofo:
            src = [c for c in cells if c["group"] != gg]
            cal_f = calibrate(src, mapping, list(group_order))
            fold_cos.append(abs(cosv(cal_f["direction"], d_all)))
        # C2: gamma3 witness alignment (n-weighted pooled group gamma3)
        g3 = np.zeros(len(group_order))
        wt = np.zeros(len(group_order))
        for c, cols in zip(cells, cal_all["cols_cache"]):
            for j, g in enumerate(group_order):
                if g in cols:
                    g3[j] += c["n"] * hermite3(cols[g], c["b"])
                    wt[j] += c["n"]
        g3 = g3 / np.maximum(wt, 1)
        results.append({
            "partition": k,
            "C1_med": float(np.median(fold_cos)),
            "C1_min": float(np.min(fold_cos)),
            "C2_g3": abs(cosv(d_all, g3)),
            "C3_gap": cal_all["gap"],
            "C4_sign": abs(float(np.sum(d_all))) / np.sqrt(len(d_all)),
            "labeled_mean": lab_mean[k],
            "labeled_orig": lab_orig[k],
        })
        if (k + 1) % 10 == 0:
            log(f"  {k + 1}/50 partitions done")

    crits = ["C1_med", "C1_min", "C2_g3", "C3_gap", "C4_sign"]
    arr = {c: np.asarray([r[c] for r in results]) for c in crits}
    lm = np.asarray([r["labeled_mean"] for r in results])
    lo = np.asarray([r["labeled_orig"] for r in results])
    # combined rank criterion
    comb = (np.argsort(np.argsort(arr["C1_med"]))
            + np.argsort(np.argsort(arr["C2_g3"]))).astype(float)
    arr["C_comb"] = comb
    crits.append("C_comb")

    log("")
    log("=== correlation of label-free criteria with labeled quality ===")
    log(f"{'criterion':>10} {'rho(mean4)':>10} {'rho(orig)':>10} "
        f"{'top3 by criterion':>22} {'their labeled means':>24}")
    for c in crits:
        order = np.argsort(-arr[c])
        top3 = [int(results[i]['partition']) for i in order[:3]]
        top3m = [round(results[i]['labeled_mean'], 2) for i in order[:3]]
        log(f"{c:>10} {spearman(arr[c], lm):>10.3f} "
            f"{spearman(arr[c], lo):>10.3f} {str(top3):>22} {str(top3m):>24}")

    log("")
    log("=== where do the labeled winners rank under each criterion? ===")
    for c in crits:
        order = list(np.argsort(-arr[c]))
        ranks = {p: order.index(p) + 1 for p in (1, 24, 36)}
        log(f"  {c:>10}: rank of #1={ranks[1]}, #24={ranks[24]}, "
            f"#36={ranks[36]}  (of 50)")

    log("")
    log("=== the label-free choice, and what it would have scored ===")
    for c in ("C1_med", "C2_g3", "C_comb"):
        pick = int(results[int(np.argmax(arr[c]))]["partition"])
        log(f"  argmax {c}: partition #{pick}  labeled mean "
            f"{lab_mean[pick]:+.3f}  per-domain "
            f"{ {d: round(labeled[pick][d], 2) for d in doms} }")
    log(f"  family NRM reference mean: +0.931")

    with (OUT / "RESULT.json").open("w", encoding="utf-8") as handle:
        json.dump({"results": results, "fidelity_check36": check36},
                  handle, indent=1, default=float)
    log("")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
