#!/usr/bin/env python3
"""Sidecar sweep: SU reliability, observed/clean PCR axes, and DUFS roughness.

The fit phase never reads correctness labels.  It freezes compact weight tensors
and diagnostics.  The report phase verifies those hashes before opening labels.
Nothing in this script changes the canonical mixed-v2 DUFS-LIU baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import eigh


REPO = Path(__file__).resolve().parents[1]
import sys

if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import family, load_contract  # noqa: E402
from scripts.inscope_cells import INSCOPE  # noqa: E402
from spectral_utils.dependency_fusion import sparse_upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_path,
)


VERSION = "su-clean-subspace-laplacian-sidecar-v1-2026-08-23"
DEFAULT_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_DIAGNOSTICS = REPO / "results" / "hard_filter_dufs_liu_24cell" / "diagnostics"
DEFAULT_OUT = REPO / "results" / "su_clean_subspace_laplacian_sidecar_v1"
LAMBDAS = np.asarray((0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0))
# All 24 mixed-v2 cells share 19 coordinates; larger K would silently change
# the evaluated roster because some cells have 27--30 coordinates and one has 19.
K_MAX = 19
DUFS_K = 7
CANONICAL_LAMBDA = 0.1
FULL_METHODS = ("uc_obs", "uclean_obs", "uclean_clean")
LITERAL_L_METHODS = ("ul_obs", "ul_clean", "ul_lraw")
ALL_METHODS = FULL_METHODS + LITERAL_L_METHODS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sym(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return 0.5 * (a + a.T)


def psd_projection(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals, vecs = eigh(sym(a))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    clipped = np.maximum(vals, 0.0)
    return sym((vecs * clipped) @ vecs.T), vals, vecs


def magnitude_basis(a: np.ndarray, rank: int = 2) -> tuple[np.ndarray, np.ndarray]:
    vals, vecs = eigh(sym(a))
    order = np.argsort(np.abs(vals))[::-1][:rank]
    return vecs[:, order], vals[order]


def solve_path(
    U: np.ndarray,
    covariance_term: np.ndarray,
    observed_covariance: np.ndarray,
    roughness: np.ndarray,
    rho: np.ndarray,
    max_k: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return weights[K, lambda, feature] with a dimensionless LIU scaling.

    Roughness is trace-matched to the covariance term.  For the intentionally
    non-convex raw-L control, the positive observed-C trace is used instead.
    """
    m = U.shape[0]
    max_k = min(int(max_k), U.shape[1])
    weights = np.zeros((max_k, len(LAMBDAS), m), dtype=np.float64)
    conditions = np.zeros((max_k, len(LAMBDAS)), dtype=np.float64)
    norms = np.zeros_like(conditions)
    effective_ranks = np.zeros_like(conditions, dtype=np.int16)
    roughness_scales = np.zeros(max_k, dtype=np.float64)
    for k in range(1, max_k + 1):
        Uk = U[:, :k]
        B = sym(Uk.T @ covariance_term @ Uk)
        Bobs = sym(Uk.T @ observed_covariance @ Uk)
        Rk = sym(Uk.T @ roughness @ Uk)
        trace_b = float(np.trace(B))
        if trace_b <= 1e-12:
            trace_b = float(np.trace(Bobs))
        trace_r = float(np.trace(Rk))
        scale = trace_b / trace_r if trace_r > 1e-12 else 0.0
        roughness_scales[k - 1] = scale
        rhs = Uk.T @ rho
        for li, lambda_ in enumerate(LAMBDAS):
            system = sym(B + float(lambda_) * scale * Rk)
            solution = np.linalg.pinv(system, rcond=1e-10) @ rhs
            w = Uk @ solution
            weights[k - 1, li] = w
            singular = np.linalg.svd(system, compute_uv=False)
            cutoff = max(float(singular[0]) * 1e-10, 1e-14) if singular.size else 1e-14
            effective_ranks[k - 1, li] = int(np.sum(singular > cutoff))
            conditions[k - 1, li] = float(np.linalg.cond(system))
            norms[k - 1, li] = float(np.linalg.norm(w))
    return weights, {
        "condition": conditions,
        "weight_norm": norms,
        "effective_rank": effective_ranks,
        "roughness_scale": roughness_scales,
    }


def fit_cell(bundle, cell: str, diagnostics_dir: Path) -> tuple[dict, dict]:
    F, names = load_contract(bundle, cell, "mixed_v2")
    m, n = F.shape
    if m < K_MAX:
        raise RuntimeError(f"{cell}: needs at least {K_MAX} mixed-v2 features, found {m}")
    prior = json.loads((diagnostics_dir / f"{cell}.json").read_text(encoding="utf-8"))
    gates = np.asarray(
        prior["contracts"]["mixed_v2"]["filters"]["full"]["dufs"]["raw_probabilities"],
        dtype=float,
    )
    graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
    canonical = laplacian_iu_path(F, (0.0, CANONICAL_LAMBDA), graph=graph)
    roughness = canonical[0.0].roughness
    su = sparse_upcr_fit(F)
    C = sym(su.covariance)
    S = sym(su.decomposition.sparse)
    L = sym(su.decomposition.low_rank)

    # The literal L is rank <=2 and generally indefinite.  C-S preserves the
    # observed marginal variances and is the coherent full-rank clean-covariance
    # extension needed for a K>2 sweep.
    clean_raw = sym(C - S)
    clean_psd, clean_raw_evals, Uclean = psd_projection(clean_raw)
    c_evals, Uc = eigh(C), None
    c_vals, c_vecs = c_evals
    c_order = np.argsort(c_vals)[::-1]
    c_vals, Uc = c_vals[c_order], c_vecs[:, c_order]
    Ul, l_vals = magnitude_basis(L, rank=2)

    output: dict[str, np.ndarray] = {
        "feature_names": np.asarray(names),
        "lambdas": LAMBDAS,
        "w_iu": np.asarray(canonical[0.0].w),
        "w_dufs_liu": np.asarray(canonical[CANONICAL_LAMBDA].w),
        "w_su_pcr": np.asarray(su.w_pcr),
        "c_eigenvalues": c_vals,
        "clean_raw_eigenvalues": clean_raw_evals,
        "l_magnitude_eigenvalues": l_vals,
    }
    method_specs = {
        "uc_obs": (Uc, C, K_MAX),
        "uclean_obs": (Uclean, C, K_MAX),
        "uclean_clean": (Uclean, clean_psd, K_MAX),
        "ul_obs": (Ul, C, 2),
        "ul_clean": (Ul, clean_psd, 2),
        "ul_lraw": (Ul, L, 2),
    }
    solve_diagnostics = {}
    for method, (basis, cov_term, max_k) in method_specs.items():
        weights, diagnostics = solve_path(basis, cov_term, C, roughness, su.rho_hat, max_k)
        output[f"weights_{method}"] = weights
        for key, values in diagnostics.items():
            output[f"diag_{method}_{key}"] = values
        solve_diagnostics[method] = {
            "k_max": int(max_k),
            "covariance_term": {
                "uc_obs": "observed_C",
                "uclean_obs": "observed_C",
                "uclean_clean": "PSD(C-S)",
                "ul_obs": "observed_C",
                "ul_clean": "PSD(C-S)",
                "ul_lraw": "literal_indefinite_L",
            }[method],
        }

    su_identity = float(np.max(np.abs(output["weights_uc_obs"][1, 0] - su.w_pcr)))
    if su_identity > 1e-9:
        raise RuntimeError(f"{cell}: U_C,K=2,lambda=0 does not reproduce SU-PCR ({su_identity})")
    overlap = np.asarray([
        np.linalg.norm(Uc[:, :k].T @ Uclean[:, :k], ord="fro") ** 2 / k
        for k in range(1, K_MAX + 1)
    ])
    diag = {
        "cell": cell,
        "n": int(n),
        "m": int(m),
        "labels_used": False,
        "su_identity_error": su_identity,
        "su": {
            "converged": bool(su.decomposition.converged),
            "iterations": int(su.decomposition.n_iter),
            "sparse_fraction": float(su.decomposition.sparse_fraction),
            "theorem_support_ok": bool(su.decomposition.theorem_support_ok),
            "relative_residual": float(su.decomposition.relative_residual),
            "rho_projection_residual": float(su.projection_residual),
            "g2_at_ceiling": bool(su.meta["g2_at_ceiling"]),
        },
        "clean_covariance": {
            "raw_min_eigenvalue": float(np.min(clean_raw_evals)),
            "n_negative_eigenvalues": int(np.sum(clean_raw_evals < -1e-10)),
            "psd_rank": int(np.sum(clean_raw_evals > 1e-10)),
            "sparse_frobenius_fraction": float(np.linalg.norm(S, "fro") / (np.linalg.norm(C, "fro") + 1e-12)),
        },
        "literal_L": {
            "numerical_rank": int(np.sum(np.abs(eigh(L, eigvals_only=True)) > 1e-10)),
            "selected_signed_eigenvalues": l_vals.tolist(),
            "indefinite": bool(np.min(eigh(L, eigvals_only=True)) < -1e-10),
            "k_above_2_identifiable": False,
        },
        "mean_subspace_overlap_by_k": overlap.tolist(),
        "methods": solve_diagnostics,
    }
    return output, diag


def fit_command(args) -> None:
    bundle_path = Path(args.bundle).resolve()
    diagnostics_dir = Path(args.diagnostics).resolve()
    out = Path(args.out).resolve()
    weights_dir = out / "weights"
    diag_out = out / "diagnostics"
    weights_dir.mkdir(parents=True, exist_ok=True)
    diag_out.mkdir(parents=True, exist_ok=True)
    definition = {
        "version": VERSION,
        "status": "retrospective_sidecar_not_canonical",
        "bundle": str(bundle_path),
        "bundle_sha256": sha256_file(bundle_path),
        "cells": list(INSCOPE),
        "contract": "mixed_v2",
        "lambdas": LAMBDAS.tolist(),
        "k_values_full": list(range(1, K_MAX + 1)),
        "k_values_literal_L": [1, 2],
        "dufs_k": DUFS_K,
        "canonical_lambda": CANONICAL_LAMBDA,
        "methods": {
            "uc_obs": "SU-rho; axes=U_C; covariance=C",
            "uclean_obs": "SU-rho; axes=U_PSD(C-S); covariance=C",
            "uclean_clean": "SU-rho; axes=U_PSD(C-S); covariance=PSD(C-S)",
            "ul_obs": "SU-rho; literal rank-2 L span; covariance=C",
            "ul_clean": "SU-rho; literal rank-2 L span; covariance=PSD(C-S)",
            "ul_lraw": "diagnostic only; literal rank-2 L span and indefinite L term",
        },
        "labels_used_during_fit": False,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(out / "RUN_DEFINITION.json", definition)
    bundle = np.load(bundle_path, allow_pickle=True)
    for index, cell in enumerate(INSCOPE, 1):
        print(f"[{index:02d}/{len(INSCOPE)}] {cell}", flush=True)
        payload, diag = fit_cell(bundle, cell, diagnostics_dir)
        np.savez_compressed(weights_dir / f"{cell}.npz", **payload)
        write_json(diag_out / f"{cell}.json", diag)
    manifest = {
        "version": VERSION,
        "weight_sha256": {cell: sha256_file(weights_dir / f"{cell}.npz") for cell in INSCOPE},
        "diagnostic_sha256": {cell: sha256_file(diag_out / f"{cell}.json") for cell in INSCOPE},
    }
    write_json(out / "WEIGHT_FREEZE_MANIFEST.json", manifest)
    write_json(out / "FIT_COMPLETE.json", {"version": VERSION, "cells": len(INSCOPE), "labels_used": False})


def family_means(rows: list[dict], value: str) -> dict[str, float]:
    families = sorted({row["family"] for row in rows})
    return {
        group: float(np.mean([row[value] for row in rows if row["family"] == group]))
        for group in families
    }


def bootstrap_family_ci(values: np.ndarray, seed_text: str, count: int = 50000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(seed_text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(count, len(values)))]
    return tuple(float(x) for x in np.quantile(sampled.mean(axis=1), (0.025, 0.975)))


def report_command(args) -> None:
    from sklearn.metrics import roc_auc_score
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bundle_path = Path(args.bundle).resolve()
    out = Path(args.out).resolve()
    manifest = json.loads((out / "WEIGHT_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    bundle = np.load(bundle_path, allow_pickle=True)
    rows: list[dict] = []
    diagnostics = []
    for cell in INSCOPE:
        weight_path = out / "weights" / f"{cell}.npz"
        if sha256_file(weight_path) != manifest["weight_sha256"][cell]:
            raise RuntimeError(f"weight hash mismatch: {cell}")
        payload = np.load(weight_path, allow_pickle=False)
        F, _ = load_contract(bundle, cell, "mixed_v2")
        y = np.asarray(bundle[f"{cell}__labels"], dtype=int)
        base_aucs = {
            "iu_pcr": float(roc_auc_score(y, payload["w_iu"] @ F)),
            "dufs_liu": float(roc_auc_score(y, payload["w_dufs_liu"] @ F)),
            "su_pcr": float(roc_auc_score(y, payload["w_su_pcr"] @ F)),
        }
        for name, auc in base_aucs.items():
            rows.append({
                "cell": cell, "family": family(cell), "kind": "baseline", "method": name,
                "k": "", "lambda": "", "auroc": auc,
            })
        for method in ALL_METHODS:
            weights = payload[f"weights_{method}"]
            for ki in range(weights.shape[0]):
                for li, lambda_ in enumerate(LAMBDAS):
                    rows.append({
                        "cell": cell, "family": family(cell), "kind": "sweep", "method": method,
                        "k": ki + 1, "lambda": float(lambda_),
                        "auroc": float(roc_auc_score(y, weights[ki, li] @ F)),
                    })
        diagnostics.append(json.loads((out / "diagnostics" / f"{cell}.json").read_text(encoding="utf-8")))
    write_csv(out / "CELL_METRICS.csv", rows)

    baseline_rows = [r for r in rows if r["kind"] == "baseline"]
    sweep_rows = [r for r in rows if r["kind"] == "sweep"]
    iu_by_cell = {r["cell"]: r["auroc"] for r in baseline_rows if r["method"] == "iu_pcr"}
    baseline_summary = []
    for method in ("iu_pcr", "dufs_liu", "su_pcr"):
        selected = [r for r in baseline_rows if r["method"] == method]
        fam = family_means(selected, "auroc")
        iu_fam = family_means([r for r in baseline_rows if r["method"] == "iu_pcr"], "auroc")
        baseline_summary.append({
            "method": method,
            "cell_macro_auroc": float(np.mean([r["auroc"] for r in selected])),
            "family_macro_auroc": float(np.mean(list(fam.values()))),
            "cell_delta_vs_iu_pp": float(100 * np.mean([r["auroc"] - iu_by_cell[r["cell"]] for r in selected])),
            "family_delta_vs_iu_pp": float(100 * np.mean([fam[g] - iu_fam[g] for g in fam])),
        })
    write_csv(out / "BASELINE_SUMMARY.csv", baseline_summary)

    surface = []
    for method in ALL_METHODS:
        method_rows = [r for r in sweep_rows if r["method"] == method]
        for k in sorted({int(r["k"]) for r in method_rows}):
            zero = {r["cell"]: r["auroc"] for r in method_rows if int(r["k"]) == k and float(r["lambda"]) == 0.0}
            for lambda_ in LAMBDAS:
                selected = [r for r in method_rows if int(r["k"]) == k and float(r["lambda"]) == float(lambda_)]
                fam = family_means(selected, "auroc")
                zero_selected = [r for r in method_rows if int(r["k"]) == k and float(r["lambda"]) == 0.0]
                zero_fam = family_means(zero_selected, "auroc")
                iu_family = {
                    g: float(np.mean([iu_by_cell[c] for c in iu_by_cell if family(c) == g])) for g in fam
                }
                surface.append({
                    "method": method, "k": k, "lambda": float(lambda_),
                    "cell_macro_auroc": float(np.mean([r["auroc"] for r in selected])),
                    "family_macro_auroc": float(np.mean(list(fam.values()))),
                    "cell_delta_vs_iu_pp": float(100 * np.mean([r["auroc"] - iu_by_cell[r["cell"]] for r in selected])),
                    "family_delta_vs_iu_pp": float(100 * np.mean([fam[g] - iu_family[g] for g in fam])),
                    "cell_laplacian_gain_pp": float(100 * np.mean([r["auroc"] - zero[r["cell"]] for r in selected])),
                    "family_laplacian_gain_pp": float(100 * np.mean([fam[g] - zero_fam[g] for g in fam])),
                })
    write_csv(out / "SURFACE_SUMMARY.csv", surface)

    # Leave-one-dataset-family-out selection of the complete (K, lambda) configuration.
    families = sorted({r["family"] for r in sweep_rows})
    lofo = {}
    for method in ALL_METHODS:
        method_k_max = K_MAX if method in FULL_METHODS else 2
        candidates = [(k, float(lam)) for k in range(1, method_k_max + 1) for lam in LAMBDAS]
        held = []
        choices = []
        method_rows = [r for r in sweep_rows if r["method"] == method]
        lookup = {(r["cell"], int(r["k"]), float(r["lambda"])): r["auroc"] for r in method_rows}
        for held_family in families:
            train_families = [g for g in families if g != held_family]
            candidate_score = {}
            for k, lam in candidates:
                per_family = []
                for group in train_families:
                    cells = [c for c in INSCOPE if family(c) == group]
                    per_family.append(np.mean([lookup[(c, k, lam)] - iu_by_cell[c] for c in cells]))
                candidate_score[(k, lam)] = float(np.mean(per_family))
            best = max(candidates, key=lambda item: (candidate_score[item], -item[0], -item[1]))
            held_cells = [c for c in INSCOPE if family(c) == held_family]
            held_delta = float(np.mean([lookup[(c, best[0], best[1])] - iu_by_cell[c] for c in held_cells]))
            choices.append({"held_family": held_family, "k": best[0], "lambda": best[1], "held_delta_pp": 100 * held_delta})
            held.append(100 * held_delta)
        ci = bootstrap_family_ci(np.asarray(held), f"lofo-{method}")
        lofo[method] = {
            "family_delta_vs_iu_pp": float(np.mean(held)),
            "ci95_pp": list(ci),
            "wins": int(np.sum(np.asarray(held) > 0)),
            "families": len(held),
            "choices": choices,
        }
    write_json(out / "LOFO_SELECTION.json", lofo)

    # Mechanism summaries by K: lambda=0 and best retrospective lambda.
    k_summary = []
    for method in FULL_METHODS:
        for k in range(1, K_MAX + 1):
            candidates = [r for r in surface if r["method"] == method and r["k"] == k]
            zero = next(r for r in candidates if r["lambda"] == 0.0)
            best = max(candidates, key=lambda r: r["family_delta_vs_iu_pp"])
            best_graph = max((r for r in candidates if r["lambda"] > 0), key=lambda r: r["family_delta_vs_iu_pp"])
            k_summary.append({
                "method": method, "k": k,
                "no_laplacian_family_delta_vs_iu_pp": zero["family_delta_vs_iu_pp"],
                "best_family_delta_vs_iu_pp": best["family_delta_vs_iu_pp"],
                "best_lambda": best["lambda"],
                "best_positive_lambda_family_delta_vs_iu_pp": best_graph["family_delta_vs_iu_pp"],
                "best_positive_lambda": best_graph["lambda"],
                "best_laplacian_gain_over_same_space_pp": best_graph["family_delta_vs_iu_pp"] - zero["family_delta_vs_iu_pp"],
            })
    write_csv(out / "K_MECHANISM_SUMMARY.csv", k_summary)

    # Static plots saved with the sidecar results.
    method_labels = {"uc_obs": r"$U_C$, term $C$", "uclean_obs": r"$U_{C-S}$, term $C$", "uclean_clean": r"$U_{C-S}$, term PSD$(C-S)$"}
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), constrained_layout=True)
    all_full = [r for r in surface if r["method"] in FULL_METHODS]
    vmax = max(abs(r["family_delta_vs_iu_pp"]) for r in all_full)
    for ax, method in zip(axes, FULL_METHODS):
        matrix = np.asarray([[next(r["family_delta_vs_iu_pp"] for r in surface if r["method"] == method and r["k"] == k and r["lambda"] == float(lam)) for lam in LAMBDAS] for k in range(1, K_MAX + 1)])
        im = ax.imshow(matrix, origin="lower", aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(method_labels[method])
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel("K")
        ax.set_xticks(range(len(LAMBDAS)), [f"{x:g}" for x in LAMBDAS], rotation=45)
        tick_positions = sorted(set([0, 1, 4, 9, 14, K_MAX - 1]))
        ax.set_yticks(tick_positions, [position + 1 for position in tick_positions])
    fig.colorbar(im, ax=axes, label="Family-macro AUROC delta vs IU-PCR (pp)", shrink=0.8)
    fig.savefig(out / "PLOT_SURFACES.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    palette = plt.get_cmap("tab10").colors
    for method_index, method in enumerate(FULL_METHODS):
        values = [r for r in k_summary if r["method"] == method]
        color = palette[method_index]
        axes[0].plot([r["k"] for r in values], [r["no_laplacian_family_delta_vs_iu_pp"] for r in values], marker="o", ms=3, color=color, label=method_labels[method])
        axes[0].plot([r["k"] for r in values], [r["best_positive_lambda_family_delta_vs_iu_pp"] for r in values], linestyle="--", color=color, alpha=0.8)
        axes[1].plot([r["k"] for r in values], [r["best_laplacian_gain_over_same_space_pp"] for r in values], marker="o", ms=3, color=color, label=method_labels[method])
    axes[0].axhline(0, color="black", lw=0.8)
    canonical_dufs_delta = next(r["family_delta_vs_iu_pp"] for r in baseline_summary if r["method"] == "dufs_liu")
    axes[0].axhline(canonical_dufs_delta, color="black", lw=0.8, linestyle=":")
    axes[0].annotate(f"canonical DUFS-LIU {canonical_dufs_delta:+.3f}pp", xy=(K_MAX, canonical_dufs_delta), xytext=(-4, 5), textcoords="offset points", ha="right", fontsize=8)
    axes[0].set(title="No-Laplacian (solid) vs best positive lambda (dashed)", xlabel="K", ylabel="Family-macro delta vs IU-PCR (pp)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(title="Increment supplied by the graph", xlabel="K", ylabel="Best Laplacian gain over matched lambda=0 (pp)")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.savefig(out / "PLOT_K_MECHANISM.png", dpi=180)
    plt.close(fig)

    # Compact report.
    best_rows = {}
    for method in ALL_METHODS:
        candidates = [r for r in surface if r["method"] == method]
        best_rows[method] = max(candidates, key=lambda r: r["family_delta_vs_iu_pp"])
    neg_clean = [d["clean_covariance"]["n_negative_eigenvalues"] for d in diagnostics]
    rank_clean = [d["clean_covariance"]["psd_rank"] for d in diagnostics]
    report = [
        "# SU clean-subspace + DUFS-Laplacian sidecar", "",
        "Retrospective mechanism study on the frozen 24-cell mixed-v2 roster. It does not change the canonical baseline.", "",
        "## Formulation boundary", "",
        "Literal `L` has rank at most two and need not be positive semidefinite, so it cannot define an identifiable K>2 PCR basis. The full-K clean-subspace arms therefore use `C-S` with observed diagonal preserved; the covariance-clean arm PSD-projects it. Literal-L K=1,2 arms are retained as diagnostics only.", "",
        "## Baselines", "",
        "| method | cell AUROC | family AUROC | family delta vs IU (pp) |", "|---|---:|---:|---:|",
    ]
    for row in baseline_summary:
        report.append(f"| {row['method']} | {row['cell_macro_auroc']:.6f} | {row['family_macro_auroc']:.6f} | {row['family_delta_vs_iu_pp']:+.4f} |")
    report += ["", "## Retrospective best surface points", "", "| method | K | lambda | family delta vs IU (pp) | graph gain vs matched lambda=0 (pp) |", "|---|---:|---:|---:|---:|"]
    for method in ALL_METHODS:
        row = best_rows[method]
        report.append(f"| {method} | {row['k']} | {row['lambda']:g} | {row['family_delta_vs_iu_pp']:+.4f} | {row['family_laplacian_gain_pp']:+.4f} |")
    report += ["", "## LOFO selection", "", "| method | family delta vs IU (pp) [95% CI] | wins |", "|---|---:|---:|"]
    for method in ALL_METHODS:
        item = lofo[method]
        report.append(f"| {method} | {item['family_delta_vs_iu_pp']:+.4f} [{item['ci95_pp'][0]:+.4f}, {item['ci95_pp'][1]:+.4f}] | {item['wins']}/{item['families']} |")
    report += [
        "", "## Diagnostics", "",
        f"- SU decompositions converged in {sum(d['su']['converged'] for d in diagnostics)}/{len(diagnostics)} cells.",
        f"- Literal L was indefinite in {sum(d['literal_L']['indefinite'] for d in diagnostics)}/{len(diagnostics)} cells and has K>2 identifiable = false by construction.",
        f"- Raw C-S had negative eigenvalues in {sum(x > 0 for x in neg_clean)}/{len(neg_clean)} cells; median PSD rank {float(np.median(rank_clean)):.1f}.",
        "- `uc_obs, K=2, lambda=0` reproduces SU-PCR to numerical precision in every cell.",
    ]
    (out / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    write_json(out / "REPORT_COMPLETE.json", {"version": VERSION, "labels_opened_after_weight_freeze": True, "rows": len(rows)})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("fit", "report"):
        q = sub.add_parser(name)
        q.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
        q.add_argument("--out", default=str(DEFAULT_OUT))
        if name == "fit":
            q.add_argument("--diagnostics", default=str(DEFAULT_DIAGNOSTICS))
    return p


def main() -> None:
    args = parser().parse_args()
    if args.command == "fit":
        fit_command(args)
    else:
        report_command(args)


if __name__ == "__main__":
    main()
