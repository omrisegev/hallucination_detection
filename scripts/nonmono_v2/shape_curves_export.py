#!/usr/bin/env python
"""
shape_curves_export.py — export P(correct)-by-decile curves for the flagged pairs.

WHY: Step 217 printed four of these curves as ASCII in a commit message and nothing
else. The shapes are the actual evidence -- "non-monotone" is a claim about a curve,
and a reader cannot check it from a scalar `shape_gain`. This dumps the curves (with
binomial error bars and the best monotone fit for contrast) so they can be plotted.

Reads the flag list from `results/nonmono_transform/shape_test.csv` (Step 217) and
re-derives every curve from the canonical cell loading path, so nothing here depends
on the Step-217 numbers being right -- only on which pairs it pointed at.

Output: results/nonmono_v2/shape_curves.json
"""
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (REPO, OUT, GROUP, load_cells_cached, iso_fit,      # noqa: E402
                    d1_shape_gain, kde_modes, spike_frac, is_discrete,
                    n_min, pct)

SHAPE_TEST = os.path.join(REPO, "results", "nonmono_transform", "shape_test.csv")
N_DEC = 10
BIG_N = 2000          # the stratification boundary used in the audit


def decile_curve(x, y, n_dec=N_DEC):
    """P(correct) per within-cell decile of x, with binomial SEs, plus the best
    monotone fit evaluated at the same decile midpoints (the contrast curve)."""
    x = np.asarray(x, float)
    y = np.asarray(y, int)
    u = pct(x)
    edges = np.linspace(0, 1, n_dec + 1)
    b = np.clip(np.digitize(u, edges[1:-1]), 0, n_dec - 1)
    ir = iso_fit(x, y)
    rows = []
    for d in range(n_dec):
        idx = np.where(b == d)[0]
        if len(idx) == 0:
            rows.append(None)
            continue
        p = float(y[idx].mean())
        rows.append(dict(
            decile=d + 1,
            n=int(len(idx)),
            p=round(p, 5),
            se=round(float(np.sqrt(max(p * (1 - p), 1e-9) / len(idx))), 5),
            x_mid=round(float(np.median(x[idx])), 5),
            iso=round(float(np.mean(ir.predict(x[idx]))), 5) if ir is not None else None,
        ))
    return rows


def classify(rows, base):
    """A coarse, deterministic shape label for the page. Deliberately simple: the
    plan's full taxonomy is a Stage-1 deliverable computed against bootstrap CIs;
    this is only for grouping the panels."""
    ok = [r for r in rows if r]
    if len(ok) < 4:
        return "sparse"
    p = np.array([r["p"] for r in ok])
    se = np.array([r["se"] for r in ok])
    rng_ = p.max() - p.min()
    if rng_ < 2 * se.max():
        return "flat"
    # significant turning points: sign changes in the smoothed first difference
    sm = np.convolve(p, np.ones(3) / 3, mode="valid")
    dsm = np.diff(sm)
    sgn = np.sign(dsm[np.abs(dsm) > 0.25 * np.std(dsm)])
    changes = int(np.sum(sgn[1:] != sgn[:-1])) if len(sgn) > 1 else 0
    amax, amin = int(np.argmax(p)), int(np.argmin(p))
    edge = lambda i: i <= 1 or i >= len(p) - 2      # noqa: E731
    if changes == 0:
        return "monotone"
    if changes >= 2:
        return "W / multi-bend"
    if not edge(amax) and edge(amin):
        return "inverted-U"
    if not edge(amin) and edge(amax):
        return "U / interior dip"
    return "edge-peak"


def main():
    with open(SHAPE_TEST, newline="", encoding="utf-8") as fh:
        st = list(csv.DictReader(fh))
    flagged = [(r["cell"], r["feature"]) for r in st if r["exceeds_null"] == "1"]
    st_by = {(r["cell"], r["feature"]): r for r in st}
    print(f"{len(flagged)} flagged pairs in shape_test.csv")

    cells = load_cells_cached()

    # Contrast panels: on each big flagged cell, the single most MONOTONE view
    # (lowest Step-217 shape_gain) -- so the page shows what "no defect" looks like
    # measured the same way, not just an assertion that most views are fine.
    big_cells = sorted({c for c, _ in flagged
                        if len(cells[c]["labels"]) >= BIG_N})
    contrast = []
    for ck in big_cells:
        cand = [(float(r["shape_gain"]), r["feature"]) for r in st
                if r["cell"] == ck and r["shape_gain"] not in ("", None)]
        if cand:
            contrast.append((ck, min(cand)[1]))

    out, seen = [], set()
    for ck, feat in flagged + contrast:
        if (ck, feat) in seen:
            continue
        seen.add((ck, feat))
        cell = cells[ck]
        if feat not in cell["pool"]:
            print(f"  SKIP {ck}/{feat}: not in live pool")
            continue
        j = cell["pool"].index(feat)
        x, y = cell["V"][:, j], cell["labels"]
        rows = decile_curve(x, y)
        disc = is_discrete(x)
        g, sd, kbest = d1_shape_gain(x, y, discrete=bool(disc))
        nm, prom2, mode_pct = kde_modes(x)
        base = float(np.mean(y))
        rec = dict(
            cell=ck, feature=feat, domain=GROUP.get(ck, "?"),
            n=int(len(y)), n_pos=int(y.sum()), n_min=n_min(y),
            base_rate=round(base, 4),
            big=bool(len(y) >= BIG_N),
            flagged=bool((ck, feat) in set(flagged)),
            v1_shape_gain=(float(st_by[(ck, feat)]["shape_gain"])
                           if st_by[(ck, feat)]["shape_gain"] not in ("", None) else None),
            v2_shape_gain=(round(g, 4) if np.isfinite(g) else None),
            v2_sd=(round(sd, 4) if np.isfinite(sd) else None),
            v2_best_k=int(kbest),
            discrete=int(disc), spike_frac=round(spike_frac(x), 3),
            kde_modes=(None if not np.isfinite(nm) else float(nm)),
            kde_mode_pct=(None if not np.isfinite(mode_pct) else round(float(mode_pct), 3)),
            curve=rows,
            shape=classify(rows, base),
        )
        out.append(rec)
        tag = "FLAG" if rec["flagged"] else "ctrl"
        print(f"  {tag} {ck:<32}{feat:<20} n={rec['n']:>5} "
              f"nmin={rec['n_min']:>4} v1={rec['v1_shape_gain']} "
              f"v2={rec['v2_shape_gain']} {rec['shape']}")

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "shape_curves.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(dict(
            n_pairs_total=len(st),
            n_flagged=len(flagged),
            big_n=BIG_N,
            panels=out,
        ), fh, indent=1)
    print(f"\nsaved {p} ({len(out)} panels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
