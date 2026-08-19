#!/usr/bin/env python3
"""Step A (Task 2): close the gamma3 technical debt.

The cos +0.76 / margin 0.56 numbers reported in
``docs/research_notes/atomic_orientation_reply_2026-08-13.md`` were produced
with the cubic probe orthogonalized against {1, b} only.  That memo's own §6
records the correction it demands of any future b-coupled estimator:

    "any future b-coupled estimator must Gram-Schmidt phi3 against {1, b, phi2}
     (the {1,b}-only version leaks a balance-dependent quadratic term that can
     flip its sign) and winsorize b."

This script recomputes gamma3 in that corrected form and reports whether the
two load-bearing numbers survive.  Nothing here reads a correctness label
except the supervised reference directions (Fisher class-mean difference) and
the AUROC-free cosine readouts built from them -- exactly the same diagnostic
posture as the original.

Reference values to reproduce as a fidelity control (RESULT.json of
results/atomic_orientation_diag_2026-08-13):
    pooled cos(gamma3, g*)          = +0.7617,  sign agreement 13/17
    per-cell median cos(g3, fisher) = +0.5129,  frac>0 = 0.870

Outputs RESULT.json + a log next to this file.
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

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    load_contract,
    family as original_family,
)
from scripts.atomic_nrm_structural_audit import SOURCE_CELLS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.atomic_neutral_residual import (  # noqa: E402
    atomic_contribution_space,
    fit_atomic_neutral_calibration,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    fit_contribution_transform,
    iu_family_contributions,
    fit_neutral_residual_mode_calibration,
)
from spectral_utils.specrage_views import VIEW_ORDER, FEATURE_TO_VIEW  # noqa: E402

BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
OUT = HERE

# Frozen reference numbers from the 2026-08-13 diagnosis ({1,b}-only probe).
REF_POOLED_COS_G3 = 0.7617044910062509
REF_PERCELL_MEDIAN = 0.5129
REF_SIGN_AGREEMENT = 13

# Registered winsorization grid.  The primary is the 1% symmetric level; the
# rest are reported so the verdict cannot be a knob artifact.
WINSOR_LEVELS = (0.0, 0.01, 0.025, 0.05)
PRIMARY_WINSOR = 0.01

LOG_LINES = []


def log(msg=""):
    print(msg, flush=True)
    LOG_LINES.append(str(msg))


def normalize(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cos(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    du, dv = np.linalg.norm(u), np.linalg.norm(v)
    if du <= 0 or dv <= 0:
        return float("nan")
    return float(u @ v / (du * dv))


# ---------- probe construction ----------

def _project_out(v, basis):
    """Classical Gram-Schmidt: remove the span of ``basis`` from ``v``."""
    v = np.asarray(v, float).copy()
    for u in basis:
        denom = float(u @ u)
        if denom > 0:
            v -= float(v @ u) / denom * u
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


def phi3_baseline(b):
    """The 2026-08-13 probe: b^3 - 3b, orthogonalized against b then centered.

    Reproduced verbatim from
    results/atomic_orientation_diag_2026-08-13/atomic_orientation_diag.py
    (``hermite3_moment``) so the fidelity control is exact.
    """
    phi = b ** 3 - 3.0 * b
    phi -= float(phi @ b / (b @ b)) * b
    phi -= float(np.mean(phi))
    return phi


def phi3_corrected(b, winsor_q):
    """Corrected probe: Gram-Schmidt of b^3 against {1, b, phi2}, winsorized b.

    phi2 is itself the Gram-Schmidt of b^2 against {1, b}, computed
    numerically rather than through the unit-variance closed form, so the
    orthogonality holds exactly in-sample after winsorization.
    """
    bw = standardize(winsorize(b, winsor_q))
    one = np.ones_like(bw)
    phi2 = _project_out(bw ** 2, [one, bw])
    phi3 = _project_out(bw ** 3, [one, bw, phi2])
    return phi3, phi2, bw


def gamma3_from_probe(R, phi):
    """-E[r * phi], the sign convention of the original estimator."""
    return -(R.T @ phi) / len(phi)


# ---------- cell construction ----------

def make_cell(name, F, names, y):
    y = np.asarray(y, dtype=int)
    F = np.asarray(F, float)
    w = upcr_fit(F, **IU_FIT_DEFAULTS).w
    aspace = atomic_contribution_space(F, names, w)
    transform = fit_contribution_transform(
        aspace, np.arange(len(aspace.baseline_score), dtype=int)
    )
    b, residuals = transform.apply(aspace.baseline_score, aspace.contributions)

    fspace = iu_family_contributions(F, names, w)
    ftransform = fit_contribution_transform(
        fspace, np.arange(len(fspace.baseline_score), dtype=int)
    )
    fb, fresiduals = ftransform.apply(
        fspace.baseline_score, fspace.contributions
    )
    aligned = np.full((len(fresiduals), len(VIEW_ORDER)), np.nan)
    for j, fam in enumerate(fspace.families):
        aligned[:, VIEW_ORDER.index(fam)] = fresiduals[:, j]

    return {
        "cell": name,
        "group": original_family(name),
        "names": tuple(names),
        "w": np.asarray(w, float),
        "y": y,
        "aspace": aspace,
        "fspace": fspace,
        "b": b,
        "R_full": residuals,
        "fb": fb,
        "F_aligned": aligned,
        "n": int(len(y)),
        "pi": float(np.mean(y)),
    }


def restrict(cellrec, atom_names):
    lookup = {n: i for i, n in enumerate(cellrec["names"])}
    return cellrec["R_full"][:, [lookup[n] for n in atom_names]]


def fisher_direction(R, y):
    return R[y == 1].mean(axis=0) - R[y == 0].mean(axis=0)


def nan_fisher(A, y):
    return np.nanmean(A[y == 1], axis=0) - np.nanmean(A[y == 0], axis=0)


# ---------- main ----------

def main():
    t0 = time.time()
    results = {"reference": {
        "pooled_cos_g3": REF_POOLED_COS_G3,
        "percell_median": REF_PERCELL_MEDIAN,
        "sign_agreement": REF_SIGN_AGREEMENT,
    }}

    log("=== load the 23 source cells ===")
    cells = []
    with np.load(BUNDLE, allow_pickle=True) as data:
        for name in SOURCE_CELLS:
            F, names = load_contract(data, name, "mixed_v2")
            y = np.asarray(data[f"{name}__labels"], dtype=int)
            cells.append(make_cell(name, F, names, y))
    log(f"loaded {len(cells)} cells ({sum(c['n'] for c in cells)} samples)")

    # ---- atomic calibration (frozen atom set) ----
    log("")
    log("=== atomic calibration (frozen 17-atom set) ===")
    cal = fit_atomic_neutral_calibration([c["aspace"] for c in cells])
    atoms = cal.feature_names
    p = len(atoms)
    log(f"eligible atoms ({p}): {', '.join(atoms)}")
    for c in cells:
        c["R"] = restrict(c, atoms)
        c["fisher"] = fisher_direction(c["R"], c["y"])
    g_star = normalize(sum(c["n"] * c["fisher"] for c in cells))

    atom_family = [FEATURE_TO_VIEW[a] for a in atoms]
    results["atoms"] = list(atoms)
    results["atom_family"] = atom_family

    # ---- fidelity control: reproduce the {1,b}-only probe ----
    log("")
    log("=== fidelity control: the {1,b}-only probe (2026-08-13) ===")
    for c in cells:
        c["g3_base"] = gamma3_from_probe(c["R"], phi3_baseline(c["b"]))
    pooled_base = normalize(sum(c["n"] * c["g3_base"] for c in cells))
    base_cos = cos(pooled_base, g_star)
    base_signs = int(np.sum(np.sign(pooled_base) == np.sign(g_star)))
    base_percell = np.asarray([cos(c["g3_base"], c["fisher"]) for c in cells])
    log(f"pooled cos(g3, g*)      = {base_cos:+.4f}   "
        f"(reference {REF_POOLED_COS_G3:+.4f})")
    log(f"sign agreement          = {base_signs}/{p}   "
        f"(reference {REF_SIGN_AGREEMENT}/{p})")
    log(f"per-cell median cos     = {np.median(base_percell):+.4f}   "
        f"frac>0 {np.mean(base_percell > 0):.3f}   "
        f"(reference {REF_PERCELL_MEDIAN:+.4f} / 0.870)")
    fidelity_ok = (
        abs(base_cos - REF_POOLED_COS_G3) < 5e-4
        and base_signs == REF_SIGN_AGREEMENT
    )
    log(f"FIDELITY CONTROL: {'PASS' if fidelity_ok else 'FAIL'}")
    results["fidelity_control"] = {
        "pooled_cos": base_cos,
        "sign_agreement": base_signs,
        "percell_median": float(np.median(base_percell)),
        "percell_frac_positive": float(np.mean(base_percell > 0)),
        "pass": bool(fidelity_ok),
    }

    # ---- corrected probe, across the winsorization grid ----
    log("")
    log("=== corrected probe: Gram-Schmidt vs {1, b, phi2} + winsorized b ===")
    log(f"{'winsor':>8} {'pooled cos':>11} {'signs':>7} "
        f"{'percell med':>12} {'frac>0':>7} {'within-fam':>11}")
    atomic_grid = {}
    for q in WINSOR_LEVELS:
        for c in cells:
            phi, _, _ = phi3_corrected(c["b"], q)
            c["g3_c"] = gamma3_from_probe(c["R"], phi)
        pooled_c = normalize(sum(c["n"] * c["g3_c"] for c in cells))
        cos_c = cos(pooled_c, g_star)
        signs_c = int(np.sum(np.sign(pooled_c) == np.sign(g_star)))
        percell_c = np.asarray([cos(c["g3_c"], c["fisher"]) for c in cells])
        # within-family sign agreement on the topk + dynamics atoms
        wf_idx = [
            i for i, fam in enumerate(atom_family)
            if fam in ("topk_distribution", "dynamics")
        ]
        wf = int(np.sum(
            np.sign(pooled_c[wf_idx]) == np.sign(g_star[wf_idx])
        ))
        log(f"{q:>8.3f} {cos_c:>+11.4f} {signs_c:>4}/{p} "
            f"{np.median(percell_c):>+12.4f} {np.mean(percell_c > 0):>7.3f} "
            f"{wf:>8}/{len(wf_idx)}")
        atomic_grid[f"{q:.3f}"] = {
            "pooled_cos": cos_c,
            "sign_agreement": signs_c,
            "percell_median": float(np.median(percell_c)),
            "percell_frac_positive": float(np.mean(percell_c > 0)),
            "within_family_signs": wf,
            "within_family_total": len(wf_idx),
            "pooled_vector": pooled_c.tolist(),
        }
    results["atomic_corrected_grid"] = atomic_grid
    results["g_star"] = g_star.tolist()

    # baseline within-family, for the same comparison
    wf_idx = [
        i for i, fam in enumerate(atom_family)
        if fam in ("topk_distribution", "dynamics")
    ]
    results["fidelity_control"]["within_family_signs"] = int(np.sum(
        np.sign(pooled_base[wf_idx]) == np.sign(g_star[wf_idx])
    ))
    results["fidelity_control"]["within_family_total"] = len(wf_idx)

    # ---- family level: the deployed NRM sign bit ----
    log("")
    log("=== family level: the deployed NRM sign bit ===")
    fam_cal = fit_neutral_residual_mode_calibration([c["fspace"] for c in cells])
    v_neutral = np.asarray(fam_cal.direction, float)
    G = len(VIEW_ORDER)
    log(f"families ({G}): {', '.join(VIEW_ORDER)}")
    log(f"selected eigenvalue: {fam_cal.eigenvalues[fam_cal.selected_index]:.4f}"
        f"  (index {fam_cal.selected_index})")

    ones_margin = float(np.ones(G) @ v_neutral / np.sqrt(G))
    log(f"all-ones sign-bit margin        = {ones_margin:+.4f}")

    # family-level supervised reference, for sign correctness
    ns = np.asarray([c["n"] for c in cells], float)
    fam_fisher = np.stack([nan_fisher(c["F_aligned"], c["y"]) for c in cells])
    fam_gstar = normalize(np.nansum(ns[:, None] * fam_fisher, axis=0))
    teacher_cos = cos(v_neutral, fam_gstar)
    log(f"cos(v_neutral, family g*)       = {teacher_cos:+.4f}"
        f"   -> deployed sign is "
        f"{'CORRECT' if teacher_cos > 0 else 'WRONG'}")

    fam_rows = {"all_ones": {
        "margin": ones_margin,
        "sign": int(np.sign(ones_margin)),
    }}

    log("")
    log(f"{'probe':>22} {'margin':>9} {'|margin|':>9} {'sign ok':>8}")
    log(f"{'all-ones (deployed)':>22} {ones_margin:>+9.4f} "
        f"{abs(ones_margin):>9.4f} "
        f"{str(np.sign(ones_margin) == np.sign(teacher_cos)):>8}")

    for label, q in [("baseline {1,b}", None)] + [
        (f"corrected w={q:.3f}", q) for q in WINSOR_LEVELS
    ]:
        vecs = []
        for c in cells:
            if q is None:
                phi = phi3_baseline(c["fb"])
            else:
                phi, _, _ = phi3_corrected(c["fb"], q)
            vecs.append(
                -np.nansum(c["F_aligned"] * phi[:, None], axis=0) / len(phi)
            )
        pooled_fam = normalize(np.nansum(
            ns[:, None] * np.stack(vecs), axis=0
        ))
        margin = float(pooled_fam @ v_neutral)
        sign_ok = bool(np.sign(margin) == np.sign(teacher_cos))
        log(f"{label:>22} {margin:>+9.4f} {abs(margin):>9.4f} "
            f"{str(sign_ok):>8}")
        fam_rows[label] = {
            "margin": margin,
            "abs_margin": abs(margin),
            "agrees_with_teacher": sign_ok,
            "pooled_family_vector": pooled_fam.tolist(),
        }

    results["family_sign_bit"] = {
        "families": list(VIEW_ORDER),
        "v_neutral": v_neutral.tolist(),
        "selected_eigenvalue": float(
            fam_cal.eigenvalues[fam_cal.selected_index]
        ),
        "family_g_star": fam_gstar.tolist(),
        "teacher_cos": teacher_cos,
        "probes": fam_rows,
    }

    # ---- verdict ----
    log("")
    log("=== VERDICT (Step A) ===")
    prim = atomic_grid[f"{PRIMARY_WINSOR:.3f}"]
    prim_fam = fam_rows[f"corrected w={PRIMARY_WINSOR:.3f}"]
    log(f"primary winsorization: {PRIMARY_WINSOR:.3f}")
    log(f"  atomic pooled cos : {prim['pooled_cos']:+.4f}  "
        f"(was {base_cos:+.4f})")
    log(f"  atomic signs      : {prim['sign_agreement']}/{p}  "
        f"(was {base_signs}/{p})")
    log(f"  within-family     : {prim['within_family_signs']}/"
        f"{prim['within_family_total']}  (was "
        f"{results['fidelity_control']['within_family_signs']}/"
        f"{results['fidelity_control']['within_family_total']})")
    log(f"  family |margin|   : {prim_fam['abs_margin']:.4f}  "
        f"(all-ones {abs(ones_margin):.4f})")
    results["verdict"] = {
        "primary_winsor": PRIMARY_WINSOR,
        "atomic_pooled_cos": prim["pooled_cos"],
        "atomic_sign_agreement": prim["sign_agreement"],
        "within_family_signs": prim["within_family_signs"],
        "family_abs_margin": prim_fam["abs_margin"],
        "family_sign_agrees_with_teacher": prim_fam["agrees_with_teacher"],
        "all_ones_abs_margin": abs(ones_margin),
    }

    log("")
    log(f"elapsed {time.time() - t0:.1f}s")
    (OUT / "RESULT.json").write_text(json.dumps(results, indent=2))
    (OUT / "gamma3_corrected.log").write_text("\n".join(LOG_LINES) + "\n")
    log(f"wrote {OUT / 'RESULT.json'}")


if __name__ == "__main__":
    main()
