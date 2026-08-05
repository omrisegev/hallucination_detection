#!/usr/bin/env python3
"""SECONDARY diagnostic: separate the solver change from the covariance change.

WHY THIS EXISTS
---------------
`SPEC_DEPENDENCY_FUSION_EXPERIMENT.md` §3.1 presents its four arms as a 2x2 crossing
*reliability estimate* with *final weight rule*.  A review of commit 64f57cd pointed out
that the crossing is not clean, and the review is right.  Between `su_pcr_reproduction`
and `sdsf` **two** things change at once:

    su_pcr_reproduction :  rho = SU,  weights = 2-component PCR on the OBSERVED covariance
    sdsf                :  rho = SU,  weights = condition-100 ridge on the STRUCTURED covariance

and the registered ablation does not resolve it either, because

    iu_ridge            :  rho = IU,  weights = condition-100 ridge on the OBSERVED covariance

changes the reliability estimate as well as the matrix.  So the registered interaction term
cannot attribute the H2 loss to "sparse structure makes the ridge rule worse": it conflates
the covariance substitution with the solver substitution.

This script holds `rho` FIXED at the SU estimate and crosses the two remaining factors,
which is the clean 2x2:

                        observed C                 structured C
    2-component PCR     = su_pcr_reproduction      pcr_structured        (new)
    condition-100 ridge  ridge_observed  (new)     = sdsf

`su_pcr_reproduction` and `sdsf` must reproduce their committed AUROC exactly; that is the
wiring gate, asserted before anything else is read.  The two new cells are what separate
"the full-inverse spectral filter failed" from "the dependency-structured covariance failed".

STATUS: secondary and diagnostic, per §9 of the spec ("Any later sensitivity analysis gets a
new output directory and is explicitly secondary").  It writes to its own directory, it
changes no registered arm, and it must never replace a registered row.  It is an attribution
analysis of an already-published negative result, not a search for a configuration that wins.

Label discipline is unchanged: every weight vector is fitted from the feature matrix alone,
the global sign is resolved against the cell's existing anchor, and labels enter only in
`evaluate_score` after the score is frozen.

Usage:
    python scripts/dependency_fusion_solver_matrix_diagnostic.py --data-dir local_cache
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.linalg import eigh
from scipy.stats import spearmanr, wilcoxon

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import GROUP, INSCOPE                                  # noqa: E402
from inscope_bench_common import assert_good6                             # noqa: E402
# `_pcr_weights` is private, and imported deliberately: it is the exact code path that
# produced the committed `su_pcr_reproduction` numbers.  Re-implementing the two-component
# rule here would break the project's rule to mirror the canonical scorer rather than
# hand-type a second copy of it.
from spectral_utils.dependency_fusion import (                            # noqa: E402
    _pcr_weights, _nearest_psd, regularized_covariance_weights, sparse_upcr_fit,
)
from run_dependency_fusion_experiment import (                            # noqa: E402
    SPARSE_FIT, derive_oriented_matrix, evaluate_score, load_cells, orient_score,
)

OUT = os.path.join(REPO, "results", "dependency_fusion_solver_matrix")
ARMS = ("su_pcr_reproduction", "ridge_observed", "pcr_structured", "sdsf")
TARGET_CONDITION = SPARSE_FIT["target_condition"]


def committed_aucs():
    """The registered per-cell AUROCs the two anchor arms must reproduce."""
    path = os.path.join(REPO, "results", "dependency_fusion_study", "per_cell.csv")
    out = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arena, _, arm = row["arm"].partition(".")
            if arena == "full" and arm in ("su_pcr_reproduction", "sdsf"):
                out.setdefault(arm, {})[row["cell"]] = float(row["auc"])
    return out


def eigen_concentration(C, w, k=2):
    """Fraction of ||w||^2 lying in the top-k eigendirections of ``C``."""
    vals, vecs = eigh(C)
    order = np.argsort(vals)[::-1]
    coeff = vecs[:, order].T @ w
    total = float(np.sum(coeff ** 2))
    return float(np.sum(coeff[:k] ** 2) / total) if total > 1e-30 else float("nan")


def four_weights(F):
    """The clean 2x2 at fixed SU rho. Returns weights + per-cell diagnostics."""
    fit = sparse_upcr_fit(F, **SPARSE_FIT)
    rho = fit.rho_hat
    C_obs = fit.covariance
    C_str = fit.decomposition.structured_cov

    w_pcr_obs, _ = _pcr_weights(C_obs, rho, n_components=SPARSE_FIT["n_components"])
    w_pcr_str, _ = _pcr_weights(C_str, rho, n_components=SPARSE_FIT["n_components"])
    w_rdg_obs, d_obs = regularized_covariance_weights(
        C_obs, rho, target_condition=TARGET_CONDITION)
    w_rdg_str, d_str = regularized_covariance_weights(
        C_str, rho, target_condition=TARGET_CONDITION)

    psd_str, _ = _nearest_psd(C_str)
    psd_obs, _ = _nearest_psd(C_obs)
    resid_evals = np.sort(np.abs(eigh(fit.decomposition.residual, eigvals_only=True)))[::-1]

    w = {"su_pcr_reproduction": w_pcr_obs, "ridge_observed": w_rdg_obs,
         "pcr_structured": w_pcr_str, "sdsf": w_rdg_str}
    diag = {
        "m": int(F.shape[0]), "n": int(F.shape[1]),
        "nnz_pairs": int(fit.decomposition.meta["nnz_pairs"]),
        "sparse_fraction": float(fit.decomposition.sparse_fraction),
        "relative_residual_frobenius": float(fit.decomposition.relative_residual),
        # solver diagnostics, per matrix
        "ridge_gamma_observed": d_obs["ridge"],
        "ridge_gamma_structured": d_str["ridge"],
        "cond_raw_observed": float(np.linalg.cond(psd_obs)),
        "cond_raw_structured": float(np.linalg.cond(psd_str)),
        "cond_after_observed": d_obs["condition_after"],
        "cond_after_structured": d_str["condition_after"],
        "n_clipped_negative_observed": d_obs["n_negative_eigenvalues_clipped"],
        "n_clipped_negative_structured": d_str["n_negative_eigenvalues_clipped"],
        "psd_distortion_observed": float(
            np.linalg.norm(psd_obs - C_obs) / (np.linalg.norm(C_obs) + 1e-12)),
        "psd_distortion_structured": float(
            np.linalg.norm(psd_str - C_str) / (np.linalg.norm(C_str) + 1e-12)),
        "min_eigenvalue_structured": d_str["raw_min_eigenvalue"],
        "min_eigenvalue_observed": d_obs["raw_min_eigenvalue"],
        # how much of each weight vector lives in the leading two directions
        "top2_concentration_pcr_observed": eigen_concentration(C_obs, w_pcr_obs),
        "top2_concentration_ridge_observed": eigen_concentration(C_obs, w_rdg_obs),
        "top2_concentration_pcr_structured": eigen_concentration(C_str, w_pcr_str),
        "top2_concentration_ridge_structured": eigen_concentration(C_str, w_rdg_str),
        "residual_eig_top1": float(resid_evals[0]) if resid_evals.size else float("nan"),
        "residual_eig_top5_share": (
            float(resid_evals[:5].sum() / resid_evals.sum()) if resid_evals.sum() > 1e-30
            else float("nan")),
    }
    for name, vec in w.items():
        diag[f"weight_norm_{name}"] = float(np.linalg.norm(vec))
    return w, diag, fit


def split_half_stability(F):
    """Refit on even and odd samples; report weight agreement across the two halves."""
    out = {}
    try:
        wa, _, _ = four_weights(F[:, 0::2])
        wb, _, _ = four_weights(F[:, 1::2])
    except Exception:
        return {f"splithalf_weight_cosine_{a}": float("nan") for a in ARMS}
    for arm in ARMS:
        a, b = wa[arm], wb[arm]
        den = np.linalg.norm(a) * np.linalg.norm(b)
        out[f"splithalf_weight_cosine_{arm}"] = (
            float(abs(a @ b) / den) if den > 1e-30 else float("nan"))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--out-dir", default=OUT)
    args = parser.parse_args()

    cells = load_cells(os.path.abspath(args.data_dir))
    gate = {k: {"V": v["V"], "anchor": v["anchor"], "pool": v["pool"],
                "labels": v["labels"]} for k, v in cells.items()}
    ok, macro = assert_good6(gate, verbose=True)
    if not ok:
        raise SystemExit(f"validity gate failed: GOOD_6 {macro:.4f}")

    committed = committed_aucs()
    rows, drift = [], []
    for ck in [c for c in INSCOPE if c in cells]:
        cell = cells[ck]
        F, _, _ = derive_oriented_matrix(cell)
        w, diag, _ = four_weights(F)

        aucs, scores = {}, {}
        for arm in ARMS:
            score, _ = orient_score(w[arm] @ F, cell["anchor"])
            scores[arm] = score
            aucs[arm] = evaluate_score(cell["labels"], score)

        # WIRING GATE: the two anchor arms must land on the committed registered value.
        for arm in ("su_pcr_reproduction", "sdsf"):
            want = committed.get(arm, {}).get(ck)
            if want is not None and abs(aucs[arm] - want) > 1e-9:
                drift.append(f"{ck}/{arm}: {aucs[arm]:.6f} vs committed {want:.6f}")

        row = {"cell": ck, "domain": GROUP.get(ck), **diag}
        row.update({f"auroc_{arm}": aucs[arm] for arm in ARMS})
        for i, a in enumerate(ARMS):
            for b in ARMS[i + 1:]:
                den = np.linalg.norm(w[a]) * np.linalg.norm(w[b])
                row[f"weight_cosine_{a}__{b}"] = (
                    float(w[a] @ w[b] / den) if den > 1e-30 else float("nan"))
                row[f"score_spearman_{a}__{b}"] = float(
                    spearmanr(scores[a], scores[b]).statistic)
        row.update(split_half_stability(F))
        rows.append(row)
        print(f"  {ck:34s} " + "  ".join(f"{a[:14]}={aucs[a]:.4f}" for a in ARMS), flush=True)

    if drift:
        raise SystemExit("WIRING GATE FAILED — the anchor arms do not reproduce the "
                         "committed numbers, so this diagnostic is not measuring the same "
                         "objects:\n  " + "\n  ".join(drift))
    print(f"\nwiring gate: su_pcr_reproduction and sdsf reproduce all "
          f"{len(rows)} committed AUROCs to 1e-9")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "per_cell.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # ---- the two factors, isolated ----------------------------------------
    def delta(a, b):
        d = np.array([100 * (r[f"auroc_{b}"] - r[f"auroc_{a}"]) for r in rows])
        p = float(wilcoxon(d).pvalue) if len(d) >= 5 and np.any(d) else float("nan")
        return {"reference": a, "candidate": b, "n": len(d),
                "mean_delta_pp": float(d.mean()), "median_delta_pp": float(np.median(d)),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()),
                "p_wilcoxon": p}

    contrasts = [
        # solver effect, matrix held fixed
        delta("su_pcr_reproduction", "ridge_observed"),
        delta("pcr_structured", "sdsf"),
        # matrix effect, solver held fixed
        delta("su_pcr_reproduction", "pcr_structured"),
        delta("ridge_observed", "sdsf"),
        # the registered H2, decomposed by these two
        delta("su_pcr_reproduction", "sdsf"),
    ]
    labels = ["solver PCR->ridge, observed C", "solver PCR->ridge, structured C",
              "matrix observed->structured, PCR", "matrix observed->structured, ridge",
              "registered H2 (both at once)"]
    for row, label in zip(contrasts, labels):
        row["effect"] = label
    with open(os.path.join(args.out_dir, "factor_contrasts.csv"), "w", newline="",
              encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)

    macro = {arm: float(np.mean([r[f"auroc_{arm}"] for r in rows])) for arm in ARMS}
    print("\nmacro AUROC at fixed SU rho:")
    print(f"                      {'observed C':>12s} {'structured C':>13s}")
    print(f"  2-component PCR     {macro['su_pcr_reproduction']:12.4f} "
          f"{macro['pcr_structured']:13.4f}")
    print(f"  condition-100 ridge {macro['ridge_observed']:12.4f} {macro['sdsf']:13.4f}")
    print("\nfactor effects (pp, positive = candidate better):")
    for row in contrasts:
        print(f"  {row['effect']:34s} {row['mean_delta_pp']:+7.2f}  "
              f"median {row['median_delta_pp']:+6.2f}  "
              f"{row['wins']}W/{row['losses']}L  p={row['p_wilcoxon']:.3g}")

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({"status": "SECONDARY diagnostic, not a registered arm",
                   "target_condition": TARGET_CONDITION,
                   "rho": "held fixed at the SU (sparse-corrected) estimate for all four arms",
                   "macro_auroc": macro, "factor_contrasts": contrasts,
                   "wiring_gate": "su_pcr_reproduction and sdsf reproduce committed "
                                  "per-cell AUROC to 1e-9 on all cells"},
                  handle, indent=2, sort_keys=True)
    print(f"\nwrote {args.out_dir}")


if __name__ == "__main__":
    main()
