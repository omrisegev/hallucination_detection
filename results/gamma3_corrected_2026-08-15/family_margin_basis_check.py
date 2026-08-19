#!/usr/bin/env python3
"""Reconcile the family sign-bit margin against the memo's quoted ~0.56.

The primary recompute reads the {1,b}-probe family margin as +0.4889, where
docs/research_notes/atomic_orientation_reply_2026-08-13.md §5 item 1 quotes
"margin ~= 0.56".  The suspected cause is a basis difference: the deployed
calibration (fit_neutral_residual_mode_calibration) spans all 6 provenance
families via pairwise-present covariance, while the memo's supporting
addendum (family_coherence_addendum.py) restricted to the 5 families present
in all 23 cells.  This script computes the margin under both bases, for the
baseline and the corrected probe, so the stop-report has no loose end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import load_contract  # noqa: E402
from scripts.atomic_nrm_structural_audit import SOURCE_CELLS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.contribution_subspace import (  # noqa: E402
    fit_contribution_transform,
    iu_family_contributions,
    fit_neutral_residual_mode_calibration,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402

BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
LOG = []


def log(m=""):
    print(m, flush=True)
    LOG.append(str(m))


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


def standardize(x):
    x = np.asarray(x, float) - float(np.mean(x))
    s = float(np.std(x))
    return x / s if s > 0 else x


def probe(b, winsor_q, orth):
    if winsor_q > 0:
        lo, hi = np.quantile(b, [winsor_q, 1.0 - winsor_q])
        b = np.clip(b, lo, hi)
    bw = standardize(b)
    one = np.ones_like(bw)
    if orth:
        phi2 = _project_out(bw ** 2, [one, bw])
        return _project_out(bw ** 3, [one, bw, phi2])
    phi = bw ** 3 - 3.0 * bw
    phi = _project_out(phi, [bw])
    return phi - float(np.mean(phi))


def main():
    cells = []
    with np.load(BUNDLE, allow_pickle=True) as data:
        for name in SOURCE_CELLS:
            F, names = load_contract(data, name, "mixed_v2")
            y = np.asarray(data[f"{name}__labels"], dtype=int)
            F = np.asarray(F, float)
            w = upcr_fit(F, **IU_FIT_DEFAULTS).w
            space = iu_family_contributions(F, names, w)
            tr = fit_contribution_transform(
                space, np.arange(len(space.baseline_score), dtype=int)
            )
            b, res = tr.apply(space.baseline_score, space.contributions)
            aligned = np.full((len(res), len(VIEW_ORDER)), np.nan)
            for j, fam in enumerate(space.families):
                aligned[:, VIEW_ORDER.index(fam)] = res[:, j]
            cells.append({"cell": name, "y": y, "space": space, "b": b,
                          "A": aligned, "n": int(len(y)),
                          "present": np.asarray([
                              fam in space.families for fam in VIEW_ORDER
                          ])})

    cal = fit_neutral_residual_mode_calibration([c["space"] for c in cells])
    v6 = np.asarray(cal.direction, float)
    present_all = np.all([c["present"] for c in cells], axis=0)
    fams5 = [f for f, k in zip(VIEW_ORDER, present_all) if k]
    log(f"all 6 families : {', '.join(VIEW_ORDER)}")
    log(f"present in all : {', '.join(fams5)}  ({int(present_all.sum())})")

    ns = np.asarray([c["n"] for c in cells], float)

    def pooled_gamma(winsor_q, orth):
        vecs = [
            -np.nansum(c["A"] * probe(c["b"], winsor_q, orth)[:, None], axis=0)
            / len(c["b"])
            for c in cells
        ]
        return np.nansum(ns[:, None] * np.stack(vecs), axis=0)

    # teacher, 6-family
    fisher = np.stack([
        np.nanmean(c["A"][c["y"] == 1], axis=0)
        - np.nanmean(c["A"][c["y"] == 0], axis=0)
        for c in cells
    ])
    g6 = normalize(np.nansum(ns[:, None] * fisher, axis=0))

    log("")
    log("margin = <normalize(pooled gamma3), v_neutral>")
    log(f"{'basis':>10} {'probe':>22} {'margin':>9} {'sign ok':>8}")

    rows = {}
    for basis, mask, vneu in (
        ("6-family", np.ones(len(VIEW_ORDER), bool), v6),
        ("5-family", present_all, normalize(v6[present_all])),
    ):
        teacher = cos(vneu, normalize(g6[mask]))
        ones = np.ones(int(mask.sum()))
        m_ones = float(normalize(ones) @ vneu)
        log(f"{basis:>10} {'all-ones (deployed)':>22} {m_ones:>+9.4f} "
            f"{str(np.sign(m_ones) == np.sign(teacher)):>8}")
        rows[f"{basis}/all_ones"] = {"margin": m_ones,
                                     "teacher_cos": teacher}
        for label, q, orth in (("baseline {1,b}", 0.0, False),
                               ("corrected w=0.010", 0.01, True),
                               ("corrected w=0.025", 0.025, True),
                               ("corrected w=0.000", 0.0, True)):
            g = normalize(pooled_gamma(q, orth)[mask])
            m = float(g @ vneu)
            log(f"{basis:>10} {label:>22} {m:>+9.4f} "
                f"{str(np.sign(m) == np.sign(teacher)):>8}")
            rows[f"{basis}/{label}"] = {"margin": m, "teacher_cos": teacher}

    (HERE / "FAMILY_BASIS_CHECK.json").write_text(json.dumps(rows, indent=2))
    (HERE / "family_margin_basis_check.log").write_text("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
