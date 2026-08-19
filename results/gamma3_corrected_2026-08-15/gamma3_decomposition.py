#!/usr/bin/env python3
"""Step A addendum: WHY does the corrected gamma3 lose the +0.76 alignment?

The primary recompute showed pooled cos(gamma3, g*) falling from +0.7617
({1,b}-only probe) to +0.3350 (Gram-Schmidt vs {1,b,phi2}, b winsorized at 1%)
and to -0.0806 with no winsorization.  Before reporting that as a stop, rule
out the two boring explanations:

  1. a pooling-SCALE artifact -- removing phi2 shrinks ||phi3|| by a
     cell-dependent factor, which reweights the n-weighted pooled sum;
  2. attributing to "the correction" what is really only winsorization.

So this script crosses the two ingredients (phi2-orthogonalization x
winsorization) and repeats each cell under three pooling conventions:
raw moment (the original), unit-RMS probe, and direction-only (each cell's
gamma3 normalized before pooling, n-weighted).

Diagnostic only.  Labels enter through the Fisher reference directions alone.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import load_contract  # noqa: E402
from scripts.atomic_nrm_structural_audit import SOURCE_CELLS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.atomic_neutral_residual import (  # noqa: E402
    atomic_contribution_space,
    fit_atomic_neutral_calibration,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    fit_contribution_transform,
)

BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
WINSOR_LEVELS = (0.0, 0.005, 0.01, 0.025, 0.05, 0.10)

LOG = []


def log(msg=""):
    print(msg, flush=True)
    LOG.append(str(msg))


def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    du, dv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (du * dv)) if du > 0 and dv > 0 else float("nan")


def _project_out(v, basis):
    v = np.asarray(v, float).copy()
    for u in basis:
        d = float(u @ u)
        if d > 0:
            v -= float(v @ u) / d * u
    return v


def winsorize(b, q):
    if q <= 0:
        return np.asarray(b, float).copy()
    lo, hi = np.quantile(b, [q, 1.0 - q])
    return np.clip(np.asarray(b, float), lo, hi)


def standardize(x):
    x = np.asarray(x, float) - float(np.mean(x))
    s = float(np.std(x))
    return x / s if s > 0 else x


def build_probe(b, winsor_q, orthogonalize_phi2):
    """Cross the two ingredients of the correction."""
    bw = standardize(winsorize(b, winsor_q))
    one = np.ones_like(bw)
    if orthogonalize_phi2:
        phi2 = _project_out(bw ** 2, [one, bw])
        return _project_out(bw ** 3, [one, bw, phi2])
    # the 2026-08-13 form, evaluated on the (possibly winsorized) b
    phi = bw ** 3 - 3.0 * bw
    phi = _project_out(phi, [bw])
    phi -= float(np.mean(phi))
    return phi


def main():
    t0 = time.time()
    log("=== load the 23 source cells ===")
    cells = []
    with np.load(BUNDLE, allow_pickle=True) as data:
        for name in SOURCE_CELLS:
            F, names = load_contract(data, name, "mixed_v2")
            y = np.asarray(data[f"{name}__labels"], dtype=int)
            F = np.asarray(F, float)
            w = upcr_fit(F, **IU_FIT_DEFAULTS).w
            aspace = atomic_contribution_space(F, names, w)
            tr = fit_contribution_transform(
                aspace, np.arange(len(aspace.baseline_score), dtype=int)
            )
            b, R = tr.apply(aspace.baseline_score, aspace.contributions)
            cells.append({"cell": name, "names": tuple(names), "y": y,
                          "aspace": aspace, "b": b, "R_full": R,
                          "n": int(len(y))})
    log(f"loaded {len(cells)} cells")

    cal = fit_atomic_neutral_calibration([c["aspace"] for c in cells])
    atoms = cal.feature_names
    for c in cells:
        lookup = {n: i for i, n in enumerate(c["names"])}
        c["R"] = c["R_full"][:, [lookup[a] for a in atoms]]
        c["fisher"] = (c["R"][c["y"] == 1].mean(axis=0)
                       - c["R"][c["y"] == 0].mean(axis=0))
    g_star = normalize(sum(c["n"] * c["fisher"] for c in cells))
    log(f"atoms: {len(atoms)}")

    results = {"atoms": list(atoms), "grid": {}}

    log("")
    log("=== crossed design: phi2-orthogonalization x winsorization ===")
    log("pooled cos(gamma3, g*) under three pooling conventions")
    log(f"{'phi2?':>6} {'winsor':>7} {'raw':>9} {'unitRMS':>9} "
        f"{'dir-only':>9} {'percell med':>12} {'signs':>7}")
    for orth in (False, True):
        for q in WINSOR_LEVELS:
            raws, units, dirs, percell = [], [], [], []
            for c in cells:
                phi = build_probe(c["b"], q, orth)
                n = len(phi)
                g_raw = -(c["R"].T @ phi) / n
                rms = float(np.sqrt(np.mean(phi ** 2)))
                g_unit = g_raw / rms if rms > 0 else g_raw
                raws.append(c["n"] * g_raw)
                units.append(c["n"] * g_unit)
                dirs.append(c["n"] * normalize(g_raw))
                percell.append(cos(g_raw, c["fisher"]))
            p_raw = normalize(np.sum(raws, axis=0))
            p_unit = normalize(np.sum(units, axis=0))
            p_dir = normalize(np.sum(dirs, axis=0))
            c_raw, c_unit, c_dir = (cos(p_raw, g_star), cos(p_unit, g_star),
                                    cos(p_dir, g_star))
            signs = int(np.sum(np.sign(p_raw) == np.sign(g_star)))
            log(f"{str(orth):>6} {q:>7.3f} {c_raw:>+9.4f} {c_unit:>+9.4f} "
                f"{c_dir:>+9.4f} {np.median(percell):>+12.4f} "
                f"{signs:>4}/{len(atoms)}")
            results["grid"][f"orth={orth},w={q:.3f}"] = {
                "pooled_cos_raw": c_raw,
                "pooled_cos_unitrms": c_unit,
                "pooled_cos_dironly": c_dir,
                "percell_median": float(np.median(percell)),
                "percell_frac_positive": float(np.mean(
                    np.asarray(percell) > 0
                )),
                "sign_agreement": signs,
            }

    # How different are the two probes, per cell?
    log("")
    log("=== how much does removing phi2 change the probe itself? ===")
    log(f"{'cell':>32} {'n':>6} {'cos(phi3_base, phi3_corr)':>26}")
    probe_cos = []
    for c in cells:
        pb = build_probe(c["b"], 0.0, False)
        pc = build_probe(c["b"], 0.01, True)
        v = cos(pb, pc)
        probe_cos.append(v)
        log(f"{c['cell']:>32} {c['n']:>6} {v:>+26.4f}")
    log(f"median cos between probes: {np.median(probe_cos):+.4f}")
    results["probe_similarity"] = {
        "per_cell": [float(v) for v in probe_cos],
        "median": float(np.median(probe_cos)),
    }

    log("")
    log(f"elapsed {time.time() - t0:.1f}s")
    (HERE / "DECOMPOSITION.json").write_text(json.dumps(results, indent=2))
    (HERE / "gamma3_decomposition.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
