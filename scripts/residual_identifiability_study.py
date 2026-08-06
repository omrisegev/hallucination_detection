#!/usr/bin/env python3
"""SECONDARY diagnostic — Study B of ``SPEC_SOLVER_MECHANISM_STUDY.md``.

WHY THIS EXISTS
---------------
Before anyone designs "SDSF v2", test whether there is structure left to model at all.  The
low-rank-plus-sparse decomposition leaves a residual R on the off-diagonal.  Is R distinguishable
from what the same pipeline produces on data whose errors really are independent given the latent
variable — or is it sampling noise plus the decomposition's own fitting bias?

This study has a preregistered ABANDONMENT condition, not only a success condition.  If the
residual is indistinguishable from the null, the answer is "stop building covariance
decompositions", and that is written down before the test runs.

TWO REVIEW CORRECTIONS ARE LOAD-BEARING HERE
--------------------------------------------
1.  The first draft's null permuted each feature's full sample axis independently.  That destroys
    the shared latent rank-two signal as well as the error dependence, so it tests "no cross-feature
    relationship at all" rather than "independent errors conditional on the latent variable" — and
    would have made residual structure look more significant than it is.  The nulls here preserve
    the fitted latent signal.

2.  Permuting residuals re-pairs them with the latent part and changes each feature's variance.
    Every null sample is therefore RE-STANDARDIZED with `prepare_cell`'s exact convention before
    the covariance is formed, with a hard `max|diag(C*) - 1| <= 1e-8` gate.  Without it the
    observation and the null would not have gone through identical preprocessing.

Inference is at the DATASET-FAMILY level (8 families), never by counting 24 correlated cells:
one global test, eight BH-corrected family tests, and leave-one-family-out on the global
statistic.  The 24 cell-level p-values are descriptive and carry no claim.

Usage:
    python scripts/residual_identifiability_study.py --data-dir local_cache
    python scripts/residual_identifiability_study.py --smoke      # 2 cells, B=20
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import eigh, subspace_angles
from scipy.stats import kurtosis

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import GROUP, INSCOPE                                   # noqa: E402
from inscope_bench_common import assert_good6                              # noqa: E402
from spectral_utils.dependency_fusion import (                             # noqa: E402
    _nearest_psd, _symmetrize, projected_sparse_decomposition, sparse_upcr_fit,
)
from spectral_utils.fusion_utils import zscore                             # noqa: E402
from run_dependency_fusion_experiment import (                             # noqa: E402
    SPARSE_FIT, dataset_family, derive_oriented_matrix, load_cells, stable_hash,
)

OUT = os.path.join(REPO, "results", "residual_identifiability")

# ---- registered constants (SPEC_SOLVER_MECHANISM_STUDY.md §4) -------------------------------
B_DRAWS = 1000
N_SPLITHALF = 50
D_RES = 5
FDR_Q = 0.10
ALPHA = 0.05
FAMILIES_REQUIRED = 5                 # of 8
ANGLE_MAX_DEG = 60.0
JACCARD_MIN = 0.50
SIGN_AGREEMENT_MIN = 0.80
SUPPORT_EPS_REL = 1e-8                # |S_ij| > eps * max|C_offdiag| counts as support
DIAG_TOL = 1e-8                       # the re-standardization gate

DECOMP_KWARGS = dict(
    rank=SPARSE_FIT["rank"],
    threshold_multiplier=SPARSE_FIT["threshold_multiplier"],
    max_iter=SPARSE_FIT["max_iter"],
    inner_completion_iter=SPARSE_FIT["inner_completion_iter"],
    tol=SPARSE_FIT["decomposition_tol"],
    max_sparse_fraction=SPARSE_FIT["max_sparse_fraction"],
)

FEATURE_FAMILY_RULES = (
    ("spilled", lambda f: f.endswith("_spilled") or "spilled" in f),
    ("energy", lambda f: f.endswith("_energy") or "energy" in f),
    ("logprob", lambda f: "logprob" in f or "top1" in f or "topk" in f),
)


# ============================================================================================
# helpers
# ============================================================================================

def seed_for(*parts):
    return int(stable_hash("|".join(str(p) for p in parts))[:8], 16)


def offdiag(A):
    A = _symmetrize(np.asarray(A, dtype=float)).copy()
    np.fill_diagonal(A, 0.0)
    return A


def restandardize(F):
    """Row-wise z-score with prepare_cell's exact convention (mean 0, std ddof=0).

    REVIEW ITEM: mandatory on every null sample.  Permuting residuals changes each feature's
    variance, so without this the null covariance and the observed covariance would not have
    gone through identical preprocessing and the comparison would be confounded by scale.
    """
    return np.vstack([zscore(row) for row in np.asarray(F, dtype=float)])


def covariance_of(F, *, gate=True):
    C = _symmetrize((F @ F.T) / F.shape[1])
    if gate:
        err = float(np.max(np.abs(np.diag(C) - 1.0)))
        if err > DIAG_TOL:
            raise ValueError(f"preprocessing gate failed: max|diag(C)-1| = {err:.3e} "
                             f"> {DIAG_TOL:.0e}; the sample is not comparable to the observation")
    return C


def residual_stats(residual, C):
    """Magnitude AND concentration.  A tiny residual can have a concentrated spectrum, so the
    top-5 share alone cannot answer the question."""
    R = offdiag(residual)
    Coff = offdiag(C)
    evals = np.sort(np.abs(eigh(R, eigvals_only=True)))[::-1]
    op_R = float(np.abs(eigh(R, eigvals_only=True)).max())
    op_C = float(np.abs(eigh(Coff, eigvals_only=True)).max())
    fro_R = float(np.linalg.norm(R, ord="fro"))
    fro_C = float(np.linalg.norm(Coff, ord="fro"))
    return {
        "op_norm_ratio": op_R / (op_C + 1e-30),
        "fro_norm_ratio": fro_R / (fro_C + 1e-30),
        "top5_share": float(evals[:5].sum() / evals.sum()) if evals.sum() > 1e-30 else float("nan"),
    }


def support_mask(sparse, C):
    """Numerically safe support test.  The code's exact-zero rule is not safe at this m — cf.
    the Step-205 finding that 1e-16 perturbations flip structure."""
    thresh = SUPPORT_EPS_REL * float(np.abs(offdiag(C)).max())
    m = sparse.shape[0]
    iu = np.triu_indices(m, 1)
    return np.abs(sparse[iu]) > thresh, sparse[iu], iu


def magnitude_subspace(A, k):
    vals, vecs = eigh(_symmetrize(A))
    take = np.argsort(np.abs(vals))[::-1][:k]
    return vecs[:, take]


def jaccard(a, b):
    inter = int(np.sum(a & b)); union = int(np.sum(a | b))
    return float(inter / union) if union else float("nan")


def sign_agreement(mask_a, val_a, mask_b, val_b):
    both = mask_a & mask_b
    if not np.any(both):
        return float("nan")
    return float(np.mean(np.sign(val_a[both]) == np.sign(val_b[both])))


def benjamini_hochberg(pvals, q):
    """Return the boolean reject vector under BH-FDR at level q."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    reject = np.zeros(n, dtype=bool)
    if np.any(passed):
        kmax = np.max(np.flatnonzero(passed))
        reject[order[:kmax + 1]] = True
    return reject


# ============================================================================================
# the two nulls
# ============================================================================================

def null_a_draw(F, V2, rng):
    """Fitted latent signal + independently permuted residuals.  PRIMARY.

    Preserves the rank-two latent signal exactly and every error's marginal distribution;
    destroys only the cross-view dependence of the errors.
    """
    F_lat = V2 @ (V2.T @ F)
    E = F - F_lat
    E_star = np.vstack([E[i, rng.permutation(E.shape[1])] for i in range(E.shape[0])])
    return restandardize(F_lat + E_star)


def null_b_draw(chol, n, rng):
    """Parametric bootstrap from the fitted independent-error latent model.  SECONDARY."""
    Z = rng.standard_normal((chol.shape[0], n))
    return restandardize(chol @ Z)


def null_b_factor(C, low_rank):
    """Cholesky-like factor of PSD(L) + diagonal error variance."""
    psd_L, _ = _nearest_psd(low_rank)
    eps = 1e-8 * float(np.mean(np.diag(C)))
    D = np.maximum(np.diag(C) - np.diag(psd_L), eps)
    C0 = _symmetrize(psd_L + np.diag(D))
    vals, vecs = eigh(C0)
    vals = np.maximum(vals, 0.0)
    return vecs * np.sqrt(vals)


def draw_statistics(F_star):
    """Run the COMPLETE decomposition pipeline on a null sample, as the reviewer required."""
    C_star = covariance_of(F_star)
    dec = projected_sparse_decomposition(C_star, **DECOMP_KWARGS)
    return residual_stats(dec.residual, C_star)


def fit_standardized_subset(F, indices):
    """Run the fitted method on a subset after repeating the original preprocessing."""
    F_sub = restandardize(np.asarray(F, dtype=float)[:, indices])
    covariance_of(F_sub)  # hard unit-diagonal gate before the fit
    return sparse_upcr_fit(F_sub, **SPARSE_FIT)


# ============================================================================================
# per-cell work
# ============================================================================================

def analyse_cell(cell_key, cell, n_draws, verbose=True, n_splithalf=None):
    n_splithalf = N_SPLITHALF if n_splithalf is None else int(n_splithalf)
    F, _, _ = derive_oriented_matrix(cell)
    fit = sparse_upcr_fit(F, **SPARSE_FIT)
    C = covariance_of(F)                       # gated: the observation must satisfy diag == 1 too
    obs = residual_stats(fit.decomposition.residual, C)

    V2 = magnitude_subspace(fit.decomposition.low_rank, 2)
    chol = null_b_factor(C, fit.decomposition.low_rank)

    nulls = {"a": [], "b": []}
    rng_a = np.random.default_rng(seed_for(cell_key, "null_a"))
    rng_b = np.random.default_rng(seed_for(cell_key, "null_b"))
    for b in range(n_draws):
        for tag, maker in (("a", lambda: null_a_draw(F, V2, rng_a)),
                           ("b", lambda: null_b_draw(chol, F.shape[1], rng_b))):
            try:
                nulls[tag].append(draw_statistics(maker()))
            except Exception as exc:
                # The preregistration requires exactly B comparable draws.  Continuing with a
                # shorter or ragged null changes the attainable p-values and can fail only after
                # hours of work, so any bad draw is a hard, contextualized error.
                raise RuntimeError(
                    f"{cell_key}: null-{tag} draw {b} failed; refusing fewer than "
                    f"B={n_draws} valid draws"
                ) from exc
        if verbose and (b + 1) % max(1, n_draws // 5) == 0:
            print(f"      draws {b + 1}/{n_draws}", flush=True)

    # ---- split-half stability, 50 deterministic repetitions (review item) ----------------
    angles, null_angles, jacc, signs = [], [], [], []
    rng_null_stability = np.random.default_rng(seed_for(cell_key, "null_stability"))
    for rep in range(n_splithalf):
        rng = np.random.default_rng(seed_for(cell_key, "splithalf", rep))
        perm = rng.permutation(F.shape[1])
        half = len(perm) // 2
        try:
            fa = fit_standardized_subset(F, perm[:half])
            fb = fit_standardized_subset(F, perm[half:2 * half])
            # The angle gate is relative to the same split-half procedure under the primary
            # independent-error null, not merely to an arbitrary 60-degree cutoff.
            F_null = null_a_draw(F, V2, rng_null_stability)
            fna = fit_standardized_subset(F_null, perm[:half])
            fnb = fit_standardized_subset(F_null, perm[half:2 * half])
        except Exception as exc:
            raise RuntimeError(
                f"{cell_key}: split-half repetition {rep} failed; refusing an incomplete "
                f"stability estimate ({n_splithalf} required)"
            ) from exc
        Sa = magnitude_subspace(fa.decomposition.residual, D_RES)
        Sb = magnitude_subspace(fb.decomposition.residual, D_RES)
        angles.append(float(np.degrees(subspace_angles(Sa, Sb)).mean()))
        Sna = magnitude_subspace(fna.decomposition.residual, D_RES)
        Snb = magnitude_subspace(fnb.decomposition.residual, D_RES)
        null_angles.append(float(np.degrees(subspace_angles(Sna, Snb)).mean()))
        ma, va, _ = support_mask(fa.decomposition.sparse, fa.covariance)
        mb, vb, _ = support_mask(fb.decomposition.sparse, fb.covariance)
        jacc.append(jaccard(ma, mb))
        signs.append(sign_agreement(ma, va, mb, vb))

    # ---- feature-family enrichment (SECONDARY, with the caveat recorded) -----------------
    pool = list(cell["pool"])
    fam_of = []
    for name in pool:
        tag = "base_spectral"
        for label, rule in FEATURE_FAMILY_RULES:
            if rule(name):
                tag = label
                break
        fam_of.append(tag)
    mask, _, iu = support_mask(fit.decomposition.sparse, C)
    within = sum(1 for k in range(len(mask))
                 if mask[k] and fam_of[iu[0][k]] == fam_of[iu[1][k]])
    total = int(mask.sum())
    same_family_pairs = sum(1 for a, b in zip(*iu) if fam_of[a] == fam_of[b])
    baseline = same_family_pairs / max(len(iu[0]), 1)

    row = {
        "cell": cell_key, "domain": GROUP.get(cell_key), "family": dataset_family(cell_key),
        "n": int(F.shape[1]), "m": int(F.shape[0]),
        "obs_op_norm_ratio": obs["op_norm_ratio"],
        "obs_fro_norm_ratio": obs["fro_norm_ratio"],
        "obs_top5_share": obs["top5_share"],
        "uniform_top5_share": 5.0 / F.shape[0],
        "null_draw_failures": 0,
        "null_draws_required": int(n_draws),
        "nulla_draws_completed": len(nulls["a"]),
        "nullb_draws_completed": len(nulls["b"]),
        "splithalf_angle_deg_median": float(np.median(angles)) if angles else float("nan"),
        "splithalf_angle_deg_p10": float(np.percentile(angles, 10)) if angles else float("nan"),
        "splithalf_angle_deg_p90": float(np.percentile(angles, 90)) if angles else float("nan"),
        "nulla_splithalf_angle_deg_median": (
            float(np.median(null_angles)) if null_angles else float("nan")),
        "angle_below_null_median": int(bool(angles and null_angles)
                                       and np.median(angles) < np.median(null_angles)),
        "splithalf_jaccard_median": float(np.nanmedian(jacc)) if jacc else float("nan"),
        "splithalf_jaccard_p10": float(np.nanpercentile(jacc, 10)) if jacc else float("nan"),
        "splithalf_jaccard_p90": float(np.nanpercentile(jacc, 90)) if jacc else float("nan"),
        "splithalf_sign_agreement_median": float(np.nanmedian(signs)) if signs else float("nan"),
        "splithalf_reps_ok": len(angles),
        "splithalf_reps_required": n_splithalf,
        "support_size": total,
        "support_within_family_frac": float(within / total) if total else float("nan"),
        "support_within_family_baseline": float(baseline),
        "median_excess_kurtosis": float(np.median(kurtosis(F, axis=1, fisher=True))),
    }
    for tag in ("a", "b"):
        for stat in ("op_norm_ratio", "fro_norm_ratio", "top5_share"):
            vals = np.array([d[stat] for d in nulls[tag]], dtype=float)
            row[f"null{tag}_{stat}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"null{tag}_{stat}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
            row[f"null{tag}_{stat}_p95"] = float(np.percentile(vals, 95)) if vals.size else float("nan")
            # descriptive per-cell percentile and p-value; NOT the inferential unit
            row[f"null{tag}_{stat}_obs_percentile"] = (
                float(100.0 * np.mean(vals <= obs[stat])) if vals.size else float("nan"))
            row[f"null{tag}_{stat}_p_descriptive"] = (
                float((1 + np.sum(vals >= obs[stat])) / (len(vals) + 1)) if vals.size
                else float("nan"))
    row["gaussian_dependent_null_b"] = int(row["median_excess_kurtosis"] > 2.0)
    return row, obs, nulls


# ============================================================================================
# aggregation: one global test, eight family tests, LOFO
# ============================================================================================

def standardized_matrix(obs_by_cell, nulls_by_cell, cells, stat="op_norm_ratio"):
    """z_obs per cell, and the B x n_cells matrix of leave-one-out standardized null draws."""
    z_obs, z_null = [], []
    lengths = {ck: len(nulls_by_cell[ck]) for ck in cells}
    if len(set(lengths.values())) != 1 or not lengths or min(lengths.values()) < 3:
        raise RuntimeError(f"null arrays are incomplete or ragged: {lengths}")
    for ck in cells:
        vals = np.array([d[stat] for d in nulls_by_cell[ck]], dtype=float)
        mu, sd = float(vals.mean()), float(vals.std(ddof=1))
        sd = sd if sd > 1e-30 else 1e-30
        z_obs.append((obs_by_cell[ck][stat] - mu) / sd)
        B = len(vals)
        tot = vals.sum(); tot2 = (vals ** 2).sum()
        loo_mu = (tot - vals) / (B - 1)
        loo_var = (tot2 - vals ** 2) / (B - 1) - loo_mu ** 2
        loo_sd = np.sqrt(np.maximum(loo_var, 1e-60)) * np.sqrt((B - 1) / max(B - 2, 1))
        z_null.append((vals - loo_mu) / np.maximum(loo_sd, 1e-30))
    return np.array(z_obs), np.array(z_null).T          # (n_cells,), (B, n_cells)


def aggregate(z, families, fam_list):
    """Family means, then the unweighted mean over families."""
    z = np.atleast_2d(z)
    out = np.zeros((z.shape[0], len(fam_list)))
    for i, f in enumerate(fam_list):
        idx = [k for k, x in enumerate(families) if x == f]
        out[:, i] = z[:, idx].mean(axis=1)
    return out.mean(axis=1), out


def empirical_p(obs, draws):
    return float((1 + np.sum(np.asarray(draws) >= obs)) / (len(draws) + 1))


# ============================================================================================
# main
# ============================================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--out-dir", default=OUT)
    parser.add_argument("--draws", type=int, default=B_DRAWS)
    parser.add_argument("--jobs", type=int, default=0,
                        help="parallel cell workers; 0 chooses automatically after timing")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.time()
    cells = load_cells(os.path.abspath(args.data_dir))
    gate = {k: {"V": v["V"], "anchor": v["anchor"], "pool": v["pool"], "labels": v["labels"]}
            for k, v in cells.items()}
    ok, macro = assert_good6(gate, verbose=True)
    if not ok:
        raise SystemExit(f"validity gate failed: GOOD_6 {macro:.4f}")

    keys = [c for c in INSCOPE if c in cells]
    out_dir, draws = args.out_dir, args.draws
    if args.smoke:
        keys, draws = keys[:2], 20
        out_dir = os.path.join(args.out_dir, "_smoke")
        globals()["N_SPLITHALF"] = 3
    os.makedirs(out_dir, exist_ok=True)

    # ---- timing probe (operational, label-free) -----------------------------------------
    probe_start = time.time()
    probe_draws = max(3, draws // 200)
    _ = analyse_cell(keys[0], cells[keys[0]], probe_draws, verbose=False, n_splithalf=0)
    per_draw = (time.time() - probe_start) / probe_draws
    projected_hours = per_draw * draws * len(keys) / 3600
    print(f"\ntiming probe: ~{per_draw:.2f}s per draw-pair -> projected "
          f"{projected_hours:.1f} h for {len(keys)} cells at B={draws}\n")

    jobs = int(args.jobs)
    if jobs < 0:
        raise SystemExit("--jobs must be >= 0")
    if jobs == 0:
        jobs = (min(4, len(keys), max(1, (os.cpu_count() or 2) // 2))
                if projected_hours > 8.0 else 1)
    print(f"execution workers: {jobs} ({'parallel' if jobs > 1 else 'sequential'})", flush=True)

    rows, obs_by_cell, nulls_a, nulls_b = [], {}, {}, {}
    completed = {}
    if jobs == 1:
        for ck in keys:
            print(f"  [{ck}]", flush=True)
            completed[ck] = analyse_cell(ck, cells[ck], draws)
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(analyse_cell, ck, cells[ck], draws, False, N_SPLITHALF): ck
                for ck in keys
            }
            for future in as_completed(futures):
                ck = futures[future]
                completed[ck] = future.result()
                print(f"  [{ck}] complete ({len(completed)}/{len(keys)})", flush=True)

    # Restore the canonical INSCOPE order regardless of parallel completion order.
    for ck in keys:
        row, obs, nulls = completed[ck]
        rows.append(row)
        obs_by_cell[ck] = obs
        nulls_a[ck], nulls_b[ck] = nulls["a"], nulls["b"]
        print(f"    op-norm ratio obs={obs['op_norm_ratio']:.4f} "
              f"null-a mean={row['nulla_op_norm_ratio_mean']:.4f} "
              f"pct={row['nulla_op_norm_ratio_obs_percentile']:.1f}", flush=True)
        write_csv(os.path.join(out_dir, "per_cell.csv"), rows)

    families = [r["family"] for r in rows]
    fam_list = sorted(set(families))
    if not args.smoke and len(fam_list) != 8:
        raise SystemExit(f"family gate failed: expected 8 dataset families, found {fam_list}")

    null_summary_rows = []
    for row in rows:
        for tag in ("a", "b"):
            for stat in ("op_norm_ratio", "fro_norm_ratio", "top5_share"):
                null_summary_rows.append({
                    "cell": row["cell"], "family": row["family"], "null": tag,
                    "statistic": stat, "draws": row[f"null{tag}_draws_completed"],
                    "mean": row[f"null{tag}_{stat}_mean"],
                    "sd": row[f"null{tag}_{stat}_sd"],
                    "p95": row[f"null{tag}_{stat}_p95"],
                    "observed_percentile": row[f"null{tag}_{stat}_obs_percentile"],
                    "p_descriptive": row[f"null{tag}_{stat}_p_descriptive"],
                })
    write_csv(os.path.join(out_dir, "null_draws_summary.csv"), null_summary_rows)

    # ---- GLOBAL primary endpoint (one test) ----------------------------------------------
    z_obs, z_null = standardized_matrix(obs_by_cell, nulls_a, keys)
    T_obs, fam_obs = aggregate(z_obs, families, fam_list)
    T_null, fam_null = aggregate(z_null, families, fam_list)
    T_obs = float(T_obs[0]) if np.ndim(T_obs) else float(T_obs)
    p_global = empirical_p(T_obs, T_null)

    # ---- EIGHT family tests, BH-FDR at q=0.10 (where the replication claim lives) ---------
    fam_rows = []
    fam_p = []
    for i, f in enumerate(fam_list):
        p = empirical_p(float(fam_obs[0, i]), fam_null[:, i])
        fam_p.append(p)
        members = [k for k, x in zip(keys, families) if x == f]
        med = lambda key: float(np.nanmedian([r[key] for r in rows if r["family"] == f]))
        stable = (med("splithalf_angle_deg_median") < ANGLE_MAX_DEG
                  and med("splithalf_angle_deg_median")
                      < med("nulla_splithalf_angle_deg_median")
                  and med("splithalf_jaccard_median") >= JACCARD_MIN
                  and med("splithalf_sign_agreement_median") >= SIGN_AGREEMENT_MIN)
        fam_rows.append({
            "family": f, "n_cells": len(members), "cells": "|".join(members),
            "family_z": float(fam_obs[0, i]), "p_empirical": p,
            "median_angle_deg": med("splithalf_angle_deg_median"),
            "null_median_angle_deg": med("nulla_splithalf_angle_deg_median"),
            "angle_below_null_median": int(
                med("splithalf_angle_deg_median")
                < med("nulla_splithalf_angle_deg_median")),
            "median_jaccard": med("splithalf_jaccard_median"),
            "median_sign_agreement": med("splithalf_sign_agreement_median"),
            "stability_pass": int(stable),
        })
    reject = benjamini_hochberg(fam_p, FDR_Q)
    for r, rj in zip(fam_rows, reject):
        r["bh_reject"] = int(rj)
        r["family_pass"] = int(bool(rj) and bool(r["stability_pass"]))
    n_pass = int(sum(r["family_pass"] for r in fam_rows))

    # ---- leave-one-family-out on the global statistic --------------------------------------
    lofo = []
    for f in fam_list:
        sub = [k for k, x in zip(keys, families) if x != f]
        sub_fams = [x for x in families if x != f]
        sub_list = sorted(set(sub_fams))
        idx = [keys.index(k) for k in sub]
        t_o, _ = aggregate(z_obs[idx], sub_fams, sub_list)
        t_n, _ = aggregate(z_null[:, idx], sub_fams, sub_list)
        lofo.append({"dropped_family": f, "T": float(np.atleast_1d(t_o)[0]),
                     "p_empirical": empirical_p(float(np.atleast_1d(t_o)[0]), t_n)})

    decision = {
        "global_p": p_global, "global_T": T_obs, "alpha": ALPHA,
        "families_passing": n_pass, "families_required": FAMILIES_REQUIRED,
        "lofo_max_p": float(max(x["p_empirical"] for x in lofo)),
        "success": bool(p_global < ALPHA and n_pass >= FAMILIES_REQUIRED
                        and max(x["p_empirical"] for x in lofo) < ALPHA),
    }
    decision["verdict"] = (
        "residual dependency is identifiable: build a block/group error model that corrects rho "
        "while keeping U-PCR's low-dimensional solver"
        if decision["success"] else
        "residual dependency is NOT identifiable at this sample size: stop building covariance "
        "decompositions")

    write_csv(os.path.join(out_dir, "family_tests.csv"), fam_rows)
    write_csv(os.path.join(out_dir, "lofo.csv"), lofo)
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "status": "SECONDARY diagnostic, not a registered arm",
            "spec": "SPEC_SOLVER_MECHANISM_STUDY.md §4",
            "primary_statistic": "||R||_2 / ||C_offdiag||_2, standardized against null (a), "
                                 "family-averaged, unweighted mean over 8 families",
            "primary_null": "fitted latent + independently permuted residuals, re-standardized",
            "secondary_null": "parametric bootstrap from the fitted independent-error latent model",
            "B": draws, "preprocessing_gate": f"max|diag(C*)-1| <= {DIAG_TOL:.0e} on every draw",
            "parallel_workers": jobs,
            "d_res": D_RES, "splithalf_repetitions": N_SPLITHALF, "fdr_q": FDR_Q,
            "thresholds": {"angle_deg_max": ANGLE_MAX_DEG, "jaccard_min": JACCARD_MIN,
                           "sign_agreement_min": SIGN_AGREEMENT_MIN,
                           "families_required": FAMILIES_REQUIRED},
            "cell_level_p_values": "DESCRIPTIVE ONLY — inference is at the family level",
            "family_tests": fam_rows, "lofo": lofo, "decision": decision,
            "runtime_seconds": time.time() - started,
        }, handle, indent=2, sort_keys=True, default=str)

    print(f"\nglobal primary endpoint T = {T_obs:+.3f}, p = {p_global:.4g} "
          f"(B={draws}, floor {1/(draws+1):.4g})")
    print(f"families passing BH q={FDR_Q} AND stability: {n_pass}/8 "
          f"(required {FAMILIES_REQUIRED})")
    for r in fam_rows:
        print(f"  {r['family']:12s} n={r['n_cells']:2d} z={r['family_z']:+6.2f} "
              f"p={r['p_empirical']:.4g} BH={'Y' if r['bh_reject'] else 'n'} "
              f"stable={'Y' if r['stability_pass'] else 'n'}")
    print(f"\nLOFO worst p = {decision['lofo_max_p']:.4g}")
    print(f"\nVERDICT: {decision['verdict']}")
    print(f"\nwrote {out_dir}  ({time.time() - started:.1f}s)")


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k); fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
