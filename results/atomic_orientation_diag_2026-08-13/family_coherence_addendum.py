#!/usr/bin/env python3
"""Addendum: how coherent (transportable) is the target direction at the
family level vs the atomic level?  Labels used for diagnosis only."""

import sys
from pathlib import Path

import numpy as np

SCRATCH = Path(__file__).resolve().parent
MW = SCRATCH / "mw"
REAL = Path(r"c:/Users/omris/TAU/hallucination_detection")
sys.path.insert(0, str(MW))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    load_contract,
    family as original_family,
)
from scripts.atomic_nrm_structural_audit import SOURCE_CELLS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    iu_family_contributions,
    fit_contribution_transform,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402

BUNDLE = REAL / "results" / "dependency_fusion_raw" / "cells.npz"


def cos(u, v):
    du, dv = np.linalg.norm(u), np.linalg.norm(v)
    return float(u @ v / (du * dv)) if du > 0 and dv > 0 else float("nan")


cells = []
with np.load(BUNDLE, allow_pickle=True) as data:
    for name in SOURCE_CELLS:
        F, names = load_contract(data, name, "mixed_v2")
        y = np.asarray(data[f"{name}__labels"], dtype=int)
        w = upcr_fit(np.asarray(F, float), **IU_FIT_DEFAULTS).w
        space = iu_family_contributions(F, names, w)
        transform = fit_contribution_transform(
            space, np.arange(F.shape[1], dtype=int)
        )
        b, res = transform.apply(space.baseline_score, space.contributions)
        aligned = np.full((F.shape[1], len(VIEW_ORDER)), np.nan)
        for j, fam in enumerate(space.families):
            aligned[:, VIEW_ORDER.index(fam)] = res[:, j]
        fisher = np.nanmean(aligned[y == 1], axis=0) - np.nanmean(
            aligned[y == 0], axis=0
        )
        b3 = b ** 3
        phi = b3 - float(b3 @ b / (b @ b)) * b
        phi -= float(np.mean(phi))
        g3 = -np.nansum(aligned * phi[:, None], axis=0) / len(b)
        cells.append({
            "cell": name,
            "group": original_family(name),
            "n": len(y),
            "fisher": fisher,
            "g3": g3,
            "present": ~np.isnan(fisher),
        })

present_all = np.all([c["present"] for c in cells], axis=0)
print("families present in all 23 cells:",
      [v for v, keep in zip(VIEW_ORDER, present_all) if keep])

# Use the 5 always-present families for coherence comparison.
F5 = np.stack([c["fisher"][present_all] for c in cells])
G5 = np.stack([c["g3"][present_all] for c in cells])
ns = np.asarray([c["n"] for c in cells], float)
g_star = (ns[:, None] * F5).sum(axis=0)
g_star /= np.linalg.norm(g_star)
coh = [cos(F5[i], g_star) for i in range(len(cells))]
print("FAMILY per-cell cos(fisher_c, g*): "
      f"min {min(coh):+.3f} median {np.median(coh):+.3f} "
      f"max {max(coh):+.3f} frac>0 {np.mean(np.asarray(coh) > 0):.2f}")

g3_pooled = (ns[:, None] * G5).sum(axis=0)
g3_pooled /= np.linalg.norm(g3_pooled)
print(f"FAMILY pooled g3 cos with g*: {cos(g3_pooled, g_star):+.4f}")
sign_agree = int(np.sum(np.sign(g3_pooled) == np.sign(g_star)))
print(f"FAMILY pooled g3 sign agreement: {sign_agree}/{int(present_all.sum())}")
percell = [cos(G5[i], F5[i]) for i in range(len(cells))]
print("FAMILY per-cell cos(g3_c, fisher_c): "
      f"median {np.median(percell):+.3f} "
      f"frac>0 {np.mean(np.asarray(percell) > 0):.2f}")
print()
print("family g* (5 always-present families):",
      {v: round(float(x), 4) for v, x in zip(
          [v for v, k in zip(VIEW_ORDER, present_all) if k],
          g_star)})
print("family g3_pooled:",
      {v: round(float(x), 4) for v, x in zip(
          [v for v, k in zip(VIEW_ORDER, present_all) if k],
          g3_pooled)})
