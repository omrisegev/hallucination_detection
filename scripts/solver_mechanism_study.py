#!/usr/bin/env python3
"""SECONDARY diagnostic — Study A of ``SPEC_SOLVER_MECHANISM_STUDY.md``.

WHY THIS EXISTS
---------------
The registered H2 (`sdsf - su_pcr_reproduction = -5.65pp`) changes two things at once.  The
shipped fixed-rho 2x2 already split it into a solver leg (-3.74pp, observed covariance) and a
matrix leg (-0.37pp under the paper's own solver).  This script splits the *solver* leg, which
is where the loss lives, into the two things a ridge actually does:

    1. it rescales the top-two coefficients      rho_j / lambda_j  ->  rho_j / (lambda_j + gamma)
    2. it admits the low-eigenvalue tail         adds sum_{j>2} rho_j / (lambda_j + gamma) v_j

An earlier draft proposed `(PSD(C)+gI)^-1 P_2 rho` and `sum_{j<=2} rho_j/(lambda_j+g) v_j` as
two arms.  With P_2 built from the same eigenbasis those are the *same vector*; a reviewer
caught it.  They collapse into `h_ridge`, and the honest design is the 2x2 below:

                        tail absent                 + t_ridge
    PCR head scaling    h_PCR   (= su_pcr_repro)    h_PCR + t_ridge
    ridge head scaling  h_ridge                     h_ridge + t_ridge (= ridge_observed)

Two corners are committed arms, so the wiring gate is free.  A second gate asserts that the
full corner *is* the registered ridge solution rather than a lookalike.

Also here, per the same review: the held-out / sample-size test (does the ridge lose because of
estimation variance or because the model is wrong?), the leakage descriptives explicitly demoted
to supporting evidence, matrix inertia instead of a "signed condition number", and the corrected
three-way PSD attribution.

STATUS: secondary and diagnostic, per SPEC_DEPENDENCY_FUSION_EXPERIMENT.md §9.  Its own output
directory; it changes no registered arm and must never replace a registered row.

Label discipline: every weight vector and every orientation decision is computed from the
feature matrix alone.  Labels enter only in `evaluate_score`, after the score is frozen.

Usage:
    python scripts/solver_mechanism_study.py --data-dir local_cache
    python scripts/solver_mechanism_study.py --smoke        # wiring check, 2 cells, 3 repeats
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh, subspace_angles
from scipy.stats import pearsonr, spearmanr, wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import GROUP, INSCOPE                                   # noqa: E402
from inscope_bench_common import assert_good6                              # noqa: E402
# The private helpers are imported deliberately: they are the exact code paths that produced
# the committed numbers.  Re-implementing them here would break the project's rule to mirror
# the canonical scorer rather than hand-type a second copy of it.
from spectral_utils.dependency_fusion import (                             # noqa: E402
    _nearest_psd, _pcr_weights, _symmetrize, regularized_covariance_weights, sparse_upcr_fit,
)
from spectral_utils.streaming_utils import anchor_orient                   # noqa: E402
from spectral_utils.subset_sweep import ALL_SIGNS                          # noqa: E402
from run_dependency_fusion_experiment import (                             # noqa: E402
    SPARSE_FIT, dataset_family, derive_oriented_matrix, evaluate_score, load_cells,
    orient_score, stable_hash,
)

OUT = os.path.join(REPO, "results", "solver_mechanism")

# ---- registered constants (SPEC_SOLVER_MECHANISM_STUDY.md §3) -------------------------------
KAPPAS = (3.0, 10.0, 30.0, 100.0, 300.0)
REGISTERED_KAPPA = float(SPARSE_FIT["target_condition"])      # 100.0
FRACTIONS = (0.25, 0.50, 0.75)
N_REPEATS = 50
D_RES = 5
N_BOOT = 10000
FACTORIAL_ARMS = ("head_pcr", "head_ridge", "head_pcr_plus_tail", "head_ridge_plus_tail")
EXTRA_ARMS = ("rho_only", "pcr_structured", "pcr_structured_psd")


# ============================================================================================
# small numerics
# ============================================================================================

def eig_desc(A):
    """Eigenpairs sorted by descending algebraic eigenvalue."""
    vals, vecs = eigh(_symmetrize(A))
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def matrix_inertia(A, prefix):
    """Condition number reported WITH inertia.

    A condition number is not signed; the earlier `cond_raw_*` column was also a misnomer (it
    reported cond(PSD(C)), not cond(C)).  Report the singular-value condition number and the
    eigenvalue facts separately so an indefinite matrix cannot hide behind one number.
    """
    A = _symmetrize(A)
    evals = eigh(A, eigvals_only=True)
    svals = np.abs(np.linalg.svd(A, compute_uv=False))
    smin = float(svals.min())
    return {
        f"{prefix}_cond_singular": float(svals.max() / smin) if smin > 0 else float("inf"),
        f"{prefix}_eig_min": float(evals.min()),
        f"{prefix}_eig_max": float(evals.max()),
        f"{prefix}_n_negative_eig": int(np.sum(evals < -1e-12)),
    }


def factorial_weights(vals, vecs, rho, gamma):
    """Head/tail decomposition in the eigensystem of PSD(C).

    h_PCR   = sum_{j<=2} rho_j / lambda_j       v_j
    h_ridge = sum_{j<=2} rho_j / (lambda_j + g) v_j
    t_ridge = sum_{j>2}  rho_j / (lambda_j + g) v_j

    The `> 1e-12` guard on h_PCR mirrors `_pcr_weights` exactly, so the PCR corner reproduces
    the committed arm rather than merely resembling it.
    """
    coef = vecs.T @ rho
    h_pcr = np.zeros_like(rho)
    h_ridge = np.zeros_like(rho)
    t_ridge = np.zeros_like(rho)
    for j in range(len(vals)):
        contrib_ridge = coef[j] / (vals[j] + gamma) * vecs[:, j]
        if j < 2:
            if vals[j] > 1e-12:
                h_pcr += coef[j] / vals[j] * vecs[:, j]
            h_ridge += contrib_ridge
        else:
            t_ridge += contrib_ridge
    return h_pcr, h_ridge, t_ridge


def top_subspace(A, k):
    _, vecs = eig_desc(A)
    return vecs[:, :k]


def magnitude_subspace(A, k):
    """Top-k eigenvectors by |eigenvalue| — the right convention for an indefinite residual."""
    vals, vecs = eigh(_symmetrize(A))
    take = np.argsort(np.abs(vals))[::-1][:k]
    return vecs[:, take]


def mean_principal_angle_deg(A, B):
    try:
        return float(np.degrees(subspace_angles(A, B)).mean())
    except Exception:
        return float("nan")


def boot_seed(name):
    """Stable per-contrast bootstrap seed, as in the registered runner."""
    return int(stable_hash(name)[:8], 16)


def family_block_ci(deltas, families, name, n_boot=N_BOOT, alpha=0.05):
    """Equal-family bootstrap CI for the dataset-family macro.

    First average cells within each family, then resample those family means.  Concatenating
    every cell after resampling a family would still let gsm8k's 10 cells outweigh a singleton
    family, which is exactly the dependence problem this interval is intended to avoid.
    """
    deltas = np.asarray(deltas, dtype=float)
    families = np.asarray(families)
    fams = sorted(set(families.tolist()))
    family_delta = np.array([
        float(np.mean(deltas[np.flatnonzero(families == f)])) for f in fams
    ])
    rng = np.random.default_rng(boot_seed(name))
    stats = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(fams), size=len(fams))
        stats[b] = family_delta[pick].mean()
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def family_weights(families):
    """Weight each cell by 1/|its family| so the 8 families count equally."""
    families = np.asarray(families)
    counts = {f: int(np.sum(families == f)) for f in set(families.tolist())}
    return np.array([1.0 / counts[f] for f in families], dtype=float)


def contrast(rows, families, a, b, name):
    """Paired candidate-minus-reference in AUROC points, inferred by family."""
    d = np.array([100.0 * (r[b] - r[a]) for r in rows], dtype=float)
    lo, hi = family_block_ci(d, families, name)
    families_arr = np.asarray(families)
    family_delta = np.array([
        float(np.mean(d[families_arr == f])) for f in sorted(set(families))
    ])
    p = (float(wilcoxon(family_delta).pvalue)
         if len(family_delta) >= 5 and np.any(family_delta) else float("nan"))
    w = family_weights(families)
    return {
        "contrast": name, "reference": a, "candidate": b, "n_cells": len(d),
        "n_families": len(family_delta),
        "mean_delta_pp": float(d.mean()), "median_delta_pp": float(np.median(d)),
        "family_macro_delta_pp": float(np.average(d, weights=w)),
        "ci_lo_pp": lo, "ci_hi_pp": hi,
        "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
        "p_wilcoxon_family": p,
    }


# ============================================================================================
# committed anchors (the wiring gate)
# ============================================================================================

def committed_aucs():
    """Per-cell AUROCs the two factorial corners must reproduce.

    `su_pcr_reproduction` comes from the registered study; `ridge_observed` from the shipped
    fixed-rho 2x2 diagnostic.  Both are committed to the repository.
    """
    out = {"head_pcr": {}, "head_ridge_plus_tail": {}}
    reg = os.path.join(REPO, "results", "dependency_fusion_study", "per_cell.csv")
    with open(reg, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arena, _, arm = row["arm"].partition(".")
            if arena == "full" and arm == "su_pcr_reproduction":
                out["head_pcr"][row["cell"]] = float(row["auc"])
    mat = os.path.join(REPO, "results", "dependency_fusion_solver_matrix", "per_cell.csv")
    with open(mat, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            out["head_ridge_plus_tail"][row["cell"]] = float(row["auroc_ridge_observed"])
    return out


# ============================================================================================
# per-cell weights
# ============================================================================================

def cell_weight_sets(F):
    """Every arm's weights, for every kappa, plus the per-cell diagnostics.

    Returns (weights[kappa][arm], weights_extra[arm], diagnostics, fit).
    """
    fit = sparse_upcr_fit(F, **SPARSE_FIT)
    rho = fit.rho_hat
    C = fit.covariance
    C_str = fit.decomposition.structured_cov

    # The registered ridge solves (PSD(C) + gI) w = rho, so the factorial is built in PSD(C)'s
    # eigensystem.  On the observed covariance that is C itself (0 clipped eigenvalues,
    # ||PSD(C)-C||/||C|| <= 2.7e-15 on all 24 cells), so the split is exact, not approximate.
    psd_obs, _ = _nearest_psd(C)
    vals, vecs = eig_desc(psd_obs)
    P2 = vecs[:, :2] @ vecs[:, :2].T

    per_kappa, gammas, ridge_full_ref = {}, {}, {}
    for kappa in KAPPAS:
        w_full, d = regularized_covariance_weights(C, rho, target_condition=kappa)
        gamma = float(d["ridge"])
        h_pcr, h_ridge, t_ridge = factorial_weights(vals, vecs, rho, gamma)
        per_kappa[kappa] = {
            "head_pcr": h_pcr,
            "head_ridge": h_ridge,
            "head_pcr_plus_tail": h_pcr + t_ridge,
            "head_ridge_plus_tail": h_ridge + t_ridge,
            "_h_ridge": h_ridge, "_t_ridge": t_ridge,
        }
        gammas[kappa] = gamma
        ridge_full_ref[kappa] = w_full

    psd_str, _ = _nearest_psd(C_str)
    extra = {
        "rho_only": rho.copy(),
        "pcr_structured": _pcr_weights(C_str, rho, n_components=2)[0],
        "pcr_structured_psd": _pcr_weights(psd_str, rho, n_components=2)[0],
    }

    # ---- gates ------------------------------------------------------------------------
    gate = {}
    for kappa in KAPPAS:
        w = per_kappa[kappa]["head_ridge_plus_tail"]
        ref = ridge_full_ref[kappa]
        den = np.linalg.norm(ref) + 1e-30
        gate[f"identity_rel_error_k{kappa:g}"] = float(np.linalg.norm(w - ref) / den)
    w_pcr_canon, _ = _pcr_weights(C, rho, n_components=2)
    gate["pcr_consistency_rel_error"] = float(
        np.linalg.norm(per_kappa[REGISTERED_KAPPA]["head_pcr"] - w_pcr_canon)
        / (np.linalg.norm(w_pcr_canon) + 1e-30))

    # ---- diagnostics -------------------------------------------------------------------
    diag = {
        "m": int(F.shape[0]), "n": int(F.shape[1]),
        "nnz_pairs": int(fit.decomposition.meta["nnz_pairs"]),
        "sparse_fraction": float(fit.decomposition.sparse_fraction),
        "relative_residual_frobenius": float(fit.decomposition.relative_residual),
        "psd_distortion_observed": float(
            np.linalg.norm(psd_obs - C) / (np.linalg.norm(C) + 1e-12)),
        "psd_distortion_structured": float(
            np.linalg.norm(psd_str - C_str) / (np.linalg.norm(C_str) + 1e-12)),
        # SUPPORTING EVIDENCE ONLY — the factorial is the causal test (spec §3.4).
        "leakage_fraction_rho": float(
            np.linalg.norm(rho - P2 @ rho) / (np.linalg.norm(rho) + 1e-30)),
        **matrix_inertia(C, "observed"),
        **matrix_inertia(C_str, "structured_raw"),
        **matrix_inertia(psd_str, "structured_psd"),
        **gate,
    }
    for kappa in KAPPAS:
        tag = f"k{kappa:g}"
        h, t = per_kappa[kappa]["_h_ridge"], per_kappa[kappa]["_t_ridge"]
        s_h, s_t = h @ F, t @ F
        s_full = s_h + s_t
        diag[f"gamma_{tag}"] = gammas[kappa]
        diag[f"tail_to_head_norm_{tag}"] = float(
            np.linalg.norm(t) / (np.linalg.norm(h) + 1e-30))
        diag[f"tail_score_var_{tag}"] = float(np.var(s_t))
        diag[f"tail_score_var_share_{tag}"] = float(np.var(s_t) / (np.var(s_full) + 1e-30))
        diag[f"head_tail_pearson_{tag}"] = (
            float(pearsonr(s_h, s_t).statistic) if np.std(s_h) > 1e-12
            and np.std(s_t) > 1e-12 else float("nan"))
        diag[f"head_tail_spearman_{tag}"] = (
            float(spearmanr(s_h, s_t).statistic) if np.std(s_h) > 1e-12
            and np.std(s_t) > 1e-12 else float("nan"))
        diag.update(matrix_inertia(psd_obs + gammas[kappa] * np.eye(len(rho)),
                                   f"ridge_system_{tag}"))
    return per_kappa, extra, diag, fit


def score_arm(w, F, anchor, labels):
    score, _ = orient_score(w @ F, anchor)
    return evaluate_score(labels, score)


# ============================================================================================
# held-out / sample-size study (spec §3.3)
# ============================================================================================

def train_fitted_standardization(V, train_idx, test_idx):
    """Fit per-view centering/scaling on train and freeze it onto test.

    ``cell['V']`` was standardized when the complete cell was prepared.  Re-standardizing from
    the training rows removes that otherwise subtle test-distribution leakage.  Because the
    original transform is affine, doing this to ``V`` is equivalent to starting from the raw
    feature values for every nonconstant view.
    """
    V = np.asarray(V, dtype=float)
    mu = V[train_idx].mean(axis=0)
    sd = V[train_idx].std(axis=0)
    bad = np.flatnonzero(~np.isfinite(sd) | (sd < 1e-8))
    if len(bad):
        raise ValueError(f"{len(bad)} view(s) constant/non-finite on the training split")
    return (V[train_idx] - mu) / sd, (V[test_idx] - mu) / sd


def heldout_for_cell(cell_key, cell, verbose=True):
    """Repeated unlabeled train/test splits; weights AND orientation frozen on train."""
    V = np.asarray(cell["V"], dtype=float)                       # (n, m)
    labels = np.asarray(cell["labels"], dtype=int)
    anchor = np.asarray(cell["anchor"], dtype=float)
    hand = np.array([ALL_SIGNS.get(name, +1) for name in cell["pool"]], dtype=float)
    n = len(labels)

    rows, repetition_rows = [], []
    for frac in FRACTIONS:
        tails, top2s, resids, n_ok = [], [], [], 0
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(boot_seed(f"{cell_key}|{frac}|{rep}"))
            perm = rng.permutation(n)
            n_tr = int(round(frac * n))
            tr, te = perm[:n_tr], perm[n_tr:]
            rec = {
                "cell": cell_key, "domain": GROUP.get(cell_key),
                "family": dataset_family(cell_key), "fraction": frac, "rep": rep,
                "split_seed": boot_seed(f"{cell_key}|{frac}|{rep}"),
                "n_train": len(tr), "n_test": len(te),
            }
            try:
                V_tr, V_te = train_fitted_standardization(V, tr, te)
                anchor_mu, anchor_sd = float(anchor[tr].mean()), float(anchor[tr].std())
                if not np.isfinite(anchor_sd) or anchor_sd < 1e-8:
                    raise ValueError("anchor is constant/non-finite on the training split")
                anchor_tr = (anchor[tr] - anchor_mu) / anchor_sd
                sub = {"V": V_tr, "pool": cell["pool"], "anchor": anchor_tr,
                       "domain": cell.get("domain")}
                F_tr, polarity, _ = derive_oriented_matrix(sub)
                F_te = (V_te * hand * polarity).T
                per_kappa, _, _, fit = cell_weight_sets(F_tr)
            except Exception as exc:
                rec.update({"status": "fit_failed", "error_type": type(exc).__name__,
                            "error": str(exc)[:300], "test_auc_defined": 0})
                repetition_rows.append(rec)
                continue

            w_set = per_kappa[REGISTERED_KAPPA]
            rec.update({
                "status": "fit_ok",
                "test_auc_defined": int(len(np.unique(labels[te])) == 2),
                "test_n_positive": int(np.sum(labels[te] == 1)),
                "test_n_negative": int(np.sum(labels[te] == 0)),
            })
            for arm in FACTORIAL_ARMS:
                raw_tr = w_set[arm] @ F_tr
                if np.std(raw_tr) < 1e-12:
                    rec[f"heldout_auroc_{arm}"] = float("nan")
                    continue
                # orientation decided on TRAIN, applied frozen to TEST
                _, flipped = anchor_orient(raw_tr, anchor_tr)
                sign = -1.0 if flipped else 1.0
                rec[f"heldout_auroc_{arm}"] = (
                    evaluate_score(labels[te], sign * (w_set[arm] @ F_te))
                    if rec["test_auc_defined"] else float("nan")
                )
            for effect, candidate, reference in (
                ("ridge_minus_pcr_pp", "head_ridge_plus_tail", "head_pcr"),
                ("tail_effect_pp", "head_pcr_plus_tail", "head_pcr"),
                ("head_effect_pp", "head_ridge", "head_pcr"),
            ):
                a = rec.get(f"heldout_auroc_{candidate}", float("nan"))
                b = rec.get(f"heldout_auroc_{reference}", float("nan"))
                rec[f"heldout_{effect}"] = 100.0 * (a - b) if np.isfinite(a + b) else float("nan")
            repetition_rows.append(rec)
            tails.append(w_set["_t_ridge"])
            top2s.append(top_subspace(_nearest_psd(fit.covariance)[0], 2))
            resids.append(magnitude_subspace(fit.decomposition.residual, D_RES))
            n_ok += 1

        these = [r for r in repetition_rows if r["fraction"] == frac]
        row = {"cell": cell_key, "domain": GROUP.get(cell_key),
               "family": dataset_family(cell_key), "fraction": frac,
               "n": n, "m": int(V.shape[1]), "n_train": int(round(frac * n)),
               "n_train_over_m": float(round(frac * n) / V.shape[1]),
               "underdetermined_flag": int(round(frac * n) < 2 * V.shape[1]),
               "reps_fit_ok": n_ok,
               "reps_fit_failed": int(sum(r["status"] != "fit_ok" for r in these)),
               "reps_auc_defined": int(sum(r.get("test_auc_defined", 0) for r in these))}
        for arm in FACTORIAL_ARMS:
            vals = np.array([r.get(f"heldout_auroc_{arm}", float("nan")) for r in these],
                            dtype=float)
            finite = vals[np.isfinite(vals)]
            row[f"heldout_auroc_{arm}"] = float(finite.mean()) if finite.size else float("nan")
            row[f"heldout_auroc_sd_{arm}"] = float(finite.std()) if finite.size else float("nan")
        # Average the paired per-repetition effects; do not subtract arm means computed from
        # potentially different valid-repetition sets.
        for effect in ("ridge_minus_pcr_pp", "tail_effect_pp", "head_effect_pp"):
            vals = np.array([r.get(f"heldout_{effect}", float("nan")) for r in these], dtype=float)
            row[f"heldout_{effect}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

        if len(tails) >= 2:
            norms = np.array([np.linalg.norm(t) for t in tails])
            row["tail_norm_cv"] = float(norms.std() / (norms.mean() + 1e-30))
            cos, ang2, angr = [], [], []
            for i in range(len(tails)):
                for j in range(i + 1, len(tails)):
                    den = np.linalg.norm(tails[i]) * np.linalg.norm(tails[j])
                    if den > 1e-30:
                        cos.append(abs(float(tails[i] @ tails[j]) / den))
                    ang2.append(mean_principal_angle_deg(top2s[i], top2s[j]))
                    angr.append(mean_principal_angle_deg(resids[i], resids[j]))
            row["tail_cosine_median"] = float(np.median(cos)) if cos else float("nan")
            row["top2_subspace_angle_deg_mean"] = float(np.nanmean(ang2))
            row["residual_subspace_angle_deg_mean"] = float(np.nanmean(angr))
        else:
            row.update({"tail_norm_cv": float("nan"), "tail_cosine_median": float("nan"),
                        "top2_subspace_angle_deg_mean": float("nan"),
                        "residual_subspace_angle_deg_mean": float("nan")})
        rows.append(row)
        if verbose:
            print(f"    frac={frac:.2f}  ok={n_ok:3d}  ridge-PCR={row['heldout_ridge_minus_pcr_pp']:+6.2f}pp"
                  f"  tail={row['heldout_tail_effect_pp']:+6.2f}pp"
                  f"  head={row['heldout_head_effect_pp']:+6.2f}pp", flush=True)
    return rows, repetition_rows


# ============================================================================================
# main
# ============================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--out-dir", default=OUT)
    parser.add_argument("--skip-heldout", action="store_true",
                        help="factorial + descriptives only (the held-out study is the slow part)")
    parser.add_argument("--smoke", action="store_true",
                        help="wiring check on 2 cells with 3 repeats; writes to <out>/_smoke")
    args = parser.parse_args()

    started = time.time()
    cells = load_cells(os.path.abspath(args.data_dir))
    gate = {k: {"V": v["V"], "anchor": v["anchor"], "pool": v["pool"], "labels": v["labels"]}
            for k, v in cells.items()}
    ok, macro = assert_good6(gate, verbose=True)
    if not ok:
        raise SystemExit(f"validity gate failed: GOOD_6 {macro:.4f}")

    keys = [c for c in INSCOPE if c in cells]
    out_dir = args.out_dir
    if args.smoke:
        keys = keys[:2]
        out_dir = os.path.join(args.out_dir, "_smoke")
        globals()["N_REPEATS"] = 3
    os.makedirs(out_dir, exist_ok=True)

    committed = committed_aucs()
    rows, kappa_rows, drift, identity_fail = [], [], [], []

    for ck in keys:
        cell = cells[ck]
        F, _, _ = derive_oriented_matrix(cell)
        per_kappa, extra, diag, fit = cell_weight_sets(F)

        row = {"cell": ck, "domain": GROUP.get(ck), "family": dataset_family(ck), **diag}
        for arm, w in extra.items():
            row[f"auroc_{arm}"] = score_arm(w, F, cell["anchor"], cell["labels"])
        for kappa in KAPPAS:
            krow = {"cell": ck, "domain": GROUP.get(ck), "family": dataset_family(ck),
                    "kappa": kappa, "gamma": diag[f"gamma_k{kappa:g}"]}
            for arm in FACTORIAL_ARMS:
                auc = score_arm(per_kappa[kappa][arm], F, cell["anchor"], cell["labels"])
                krow[f"auroc_{arm}"] = auc
                if kappa == REGISTERED_KAPPA:
                    row[f"auroc_{arm}"] = auc
            krow["tail_effect_pp"] = 100.0 * (
                krow["auroc_head_pcr_plus_tail"] - krow["auroc_head_pcr"])
            krow["head_effect_pp"] = 100.0 * (
                krow["auroc_head_ridge"] - krow["auroc_head_pcr"])
            kappa_rows.append(krow)

            rel = diag[f"identity_rel_error_k{kappa:g}"]
            if rel > 1e-10:
                identity_fail.append(f"{ck} kappa={kappa:g}: rel error {rel:.3e}")

        for arm in ("head_pcr", "head_ridge_plus_tail"):
            want = committed[arm].get(ck)
            if want is not None and abs(row[f"auroc_{arm}"] - want) > 1e-9:
                drift.append(f"{ck}/{arm}: {row[f'auroc_{arm}']:.9f} vs committed {want:.9f}")
        rows.append(row)
        print(f"  {ck:34s} " + "  ".join(
            f"{a.replace('head_', '')[:12]}={row[f'auroc_{a}']:.4f}" for a in FACTORIAL_ARMS),
            flush=True)

    # ---- GATES, asserted before anything is interpreted --------------------------------
    if drift:
        raise SystemExit("WIRING GATE FAILED — the factorial corners do not reproduce the "
                         "committed arms, so this study is not measuring the same objects:\n  "
                         + "\n  ".join(drift))
    if identity_fail:
        raise SystemExit("ARITHMETIC GATE FAILED — h_ridge + t_ridge is not the registered "
                         "ridge solution:\n  " + "\n  ".join(identity_fail))
    print(f"\nwiring gate  : both factorial corners reproduce committed AUROC to 1e-9 "
          f"on all {len(rows)} cells")
    print(f"arithmetic gate: h_ridge + t_ridge == registered ridge to 1e-10 on all cells "
          f"and all {len(KAPPAS)} kappa")

    families = [r["family"] for r in rows]
    effects = [
        contrast(rows, families, "auroc_head_pcr", "auroc_head_ridge",
                 "head rescaling, tail absent"),
        contrast(rows, families, "auroc_head_pcr_plus_tail", "auroc_head_ridge_plus_tail",
                 "head rescaling, tail present"),
        contrast(rows, families, "auroc_head_pcr", "auroc_head_pcr_plus_tail",
                 "tail addition at PCR head"),
        contrast(rows, families, "auroc_head_ridge", "auroc_head_ridge_plus_tail",
                 "tail addition at ridge head"),
        contrast(rows, families, "auroc_head_pcr", "auroc_head_ridge_plus_tail",
                 "solver leg, both at once"),
        contrast(rows, families, "auroc_head_pcr", "auroc_pcr_structured",
                 "matrix leg, PCR solver (raw structured)"),
        contrast(rows, families, "auroc_head_pcr", "auroc_pcr_structured_psd",
                 "matrix leg, PCR solver (PSD structured)"),
        contrast(rows, families, "auroc_pcr_structured", "auroc_pcr_structured_psd",
                 "PSD repair on the structured matrix"),
        contrast(rows, families, "auroc_head_pcr", "auroc_rho_only",
                 "no inversion at all"),
    ]
    # interaction: difference of the two tail effects == difference of the two head effects
    d_tail_pcr = np.array([100 * (r["auroc_head_pcr_plus_tail"] - r["auroc_head_pcr"])
                           for r in rows])
    d_tail_ridge = np.array([100 * (r["auroc_head_ridge_plus_tail"] - r["auroc_head_ridge"])
                             for r in rows])
    inter = d_tail_ridge - d_tail_pcr
    lo, hi = family_block_ci(inter, families, "interaction")
    families_arr = np.asarray(families)
    inter_family = np.array([
        float(np.mean(inter[families_arr == f])) for f in sorted(set(families))
    ])
    effects.append({"contrast": "interaction (tail effect at ridge head - at PCR head)",
                    "reference": "-", "candidate": "-", "n_cells": len(inter),
                    "n_families": len(inter_family),
                    "mean_delta_pp": float(inter.mean()),
                    "median_delta_pp": float(np.median(inter)),
                    "family_macro_delta_pp": float(np.average(inter,
                                                              weights=family_weights(families))),
                    "ci_lo_pp": lo, "ci_hi_pp": hi,
                    "wins": int((inter > 0).sum()), "losses": int((inter < 0).sum()),
                    "p_wilcoxon_family": (float(wilcoxon(inter_family).pvalue)
                                            if np.any(inter_family) else float("nan"))})

    # ---- kappa trend: family-weighted slope vs log kappa (spec §3.2) --------------------
    by_cell = {}
    for kr in kappa_rows:
        by_cell.setdefault(kr["cell"], []).append(kr)
    cell_keys = sorted(by_cell)
    fam_of = {r["cell"]: r["family"] for r in rows}
    fams = np.array([fam_of[c] for c in cell_keys])
    w_cell = family_weights(fams)

    def weighted_slope(field, idx=None):
        x, y, w = [], [], []
        use = range(len(cell_keys)) if idx is None else idx
        for i in use:
            c = cell_keys[i]
            for kr in by_cell[c]:
                x.append(np.log(kr["kappa"])); y.append(kr[field]); w.append(w_cell[i])
        x, y, w = np.array(x), np.array(y), np.array(w)
        xm = np.average(x, weights=w); ym = np.average(y, weights=w)
        den = np.sum(w * (x - xm) ** 2)
        return float(np.sum(w * (x - xm) * (y - ym)) / den) if den > 1e-30 else float("nan")

    rng = np.random.default_rng(boot_seed("kappa_trend"))
    fam_list = sorted(set(fams.tolist()))
    fam_idx = {f: np.flatnonzero(fams == f) for f in fam_list}
    slopes = []
    for _ in range(2000):
        pick = rng.integers(0, len(fam_list), size=len(fam_list))
        idx = np.concatenate([fam_idx[fam_list[k]] for k in pick])
        slopes.append(weighted_slope("tail_effect_pp", idx))
    trend = {
        "statistic": "family-weighted OLS slope of the tail effect on log kappa",
        "prediction": "negative slope if low-eigenvalue amplification is causal",
        "slope_pp_per_log_kappa": weighted_slope("tail_effect_pp"),
        "ci_lo": float(np.nanpercentile(slopes, 2.5)),
        "ci_hi": float(np.nanpercentile(slopes, 97.5)),
        "head_effect_slope_pp_per_log_kappa": weighted_slope("head_effect_pp"),
    }

    # ---- PSD attribution, corrected three-way rule (spec §3.6) --------------------------
    clipped = [r["cell"] for r in rows if r["structured_raw_n_negative_eig"] > 0]
    sub = [r for r in rows if r["cell"] in clipped]
    subgroup_effects = []
    if sub:
        sub_families = [r["family"] for r in sub]
        subgroup_effects = [
            contrast(sub, sub_families, "auroc_head_pcr", "auroc_pcr_structured",
                     "clipped subgroup: observed to raw structured"),
            contrast(sub, sub_families, "auroc_head_pcr", "auroc_pcr_structured_psd",
                     "clipped subgroup: observed to PSD structured"),
            contrast(sub, sub_families, "auroc_pcr_structured", "auroc_pcr_structured_psd",
                     "clipped subgroup: PSD repair"),
        ]
        effects.extend(subgroup_effects)

    def equivalent(effect):
        return bool(abs(effect["family_macro_delta_pp"]) <= 0.25
                    and effect["ci_lo_pp"] >= -0.50 and effect["ci_hi_pp"] <= 0.50)

    structured_effects = {
        effect["contrast"]: {**effect, "equivalent": equivalent(effect)}
        for effect in effects if ("matrix leg" in effect["contrast"]
                                  or "PSD repair" in effect["contrast"]
                                  or "clipped subgroup" in effect["contrast"])
    }
    psd_attr = {
        "clipped_cells": clipped, "n_clipped_cells": len(clipped),
        "equivalence_criterion": "|macro delta| <= 0.25pp AND family-blocked 95% CI "
                                 "inside +/-0.50pp (an equivalence criterion, not TOST)",
        "macro_24": {
            "pcr_top2_observed": float(np.mean([r["auroc_head_pcr"] for r in rows])),
            "pcr_structured_raw": float(np.mean([r["auroc_pcr_structured"] for r in rows])),
            "pcr_structured_psd": float(np.mean([r["auroc_pcr_structured_psd"] for r in rows])),
        },
        "subgroup_clipped_only": {
            "n": len(sub),
            "pcr_top2_observed": float(np.mean([r["auroc_head_pcr"] for r in sub])) if sub else None,
            "pcr_structured_raw": float(np.mean([r["auroc_pcr_structured"] for r in sub])) if sub else None,
            "pcr_structured_psd": float(np.mean([r["auroc_pcr_structured_psd"] for r in sub])) if sub else None,
        },
        "equivalence_tests": structured_effects,
        "note": "raw and PSD structured weights are bit-identical on the non-clipped cells, so "
                "the 24-cell macro dilutes the effect by those zeros; the clipped subgroup is "
                "the informative read.",
    }

    # ---- held-out study ------------------------------------------------------------------
    heldout_rows, heldout_repetition_rows = [], []
    if not args.skip_heldout:
        print("\nheld-out / sample-size study "
              f"({len(FRACTIONS)} fractions x {N_REPEATS} repeats x {len(keys)} cells)")
        for ck in keys:
            print(f"  [{ck}]", flush=True)
            summaries, repetitions = heldout_for_cell(ck, cells[ck])
            heldout_rows.extend(summaries)
            heldout_repetition_rows.extend(repetitions)
            write_csv(os.path.join(out_dir, "heldout.csv"), heldout_rows)
            write_csv(os.path.join(out_dir, "heldout_repetitions.csv"),
                      heldout_repetition_rows)

    # ---- write ---------------------------------------------------------------------------
    write_csv(os.path.join(out_dir, "per_cell.csv"), rows)
    write_csv(os.path.join(out_dir, "kappa_path.csv"), kappa_rows)
    write_csv(os.path.join(out_dir, "factorial_effects.csv"), effects)

    print("\nfactorial at the registered kappa=100 (macro AUROC over cells):")
    macro = {a: float(np.mean([r[f"auroc_{a}"] for r in rows])) for a in FACTORIAL_ARMS}
    print(f"                     {'tail absent':>12s} {'+ t_ridge':>12s}")
    print(f"  PCR head scaling   {macro['head_pcr']:12.4f} {macro['head_pcr_plus_tail']:12.4f}")
    print(f"  ridge head scaling {macro['head_ridge']:12.4f} {macro['head_ridge_plus_tail']:12.4f}")
    print("\neffects (pp, positive = candidate better; CI is family-blocked):")
    for e in effects:
        print(f"  {e['contrast']:44s} {e['mean_delta_pp']:+7.2f}  "
              f"[{e['ci_lo_pp']:+6.2f},{e['ci_hi_pp']:+6.2f}]  "
              f"{e['wins']}W/{e['losses']}L  p_family={e['p_wilcoxon_family']:.3g}")
    print(f"\nkappa trend: slope {trend['slope_pp_per_log_kappa']:+.3f} pp per log-kappa "
          f"[{trend['ci_lo']:+.3f},{trend['ci_hi']:+.3f}]  "
          f"({'consistent with' if trend['ci_hi'] < 0 else 'does NOT support'} "
          f"tail amplification)")

    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "status": "SECONDARY diagnostic, not a registered arm",
            "spec": "SPEC_SOLVER_MECHANISM_STUDY.md",
            "rho": "held fixed at the SU (sparse-corrected) estimate for every arm",
            "registered_kappa": REGISTERED_KAPPA, "kappa_path": list(KAPPAS),
            "gates": {
                "wiring": "head_pcr and head_ridge_plus_tail reproduce committed per-cell "
                          "AUROC to 1e-9 on all cells",
                "arithmetic": "h_ridge + t_ridge == regularized_covariance_weights(C,rho) "
                              "to 1e-10 relative, all cells, all kappa",
                "validity": "assert_good6 passed before any arm ran",
            },
            "macro_auroc": macro, "factorial_effects": effects, "kappa_trend": trend,
            "psd_attribution": psd_attr,
            "heldout": {"fractions": list(FRACTIONS), "repeats": N_REPEATS,
                        "n_summary_rows": len(heldout_rows),
                        "n_repetition_rows": len(heldout_repetition_rows),
                        "preprocessing": "feature centering/scaling fit on train only; frozen "
                                         "onto test; train labels never inspected"},
            "runtime_seconds": time.time() - started,
        }, handle, indent=2, sort_keys=True, default=str)
    print(f"\nwrote {out_dir}  ({time.time() - started:.1f}s)")


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen and not k.startswith("_"):
                seen.add(k); fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
