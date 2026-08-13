#!/usr/bin/env python3
"""Run the exploratory CM-LFF study on the frozen 24-cell mixed-v2 bundle.

``fit`` never reads labels.  It freezes and hashes every score first.
``report`` verifies those hashes before opening labels.  The 24 cells and the
mixed-v2 contract are retrospective development data, so the result is a
mechanism study rather than external confirmation.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import load_contract  # noqa: E402
from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402
from spectral_utils.coupled_moment_fusion import (  # noqa: E402
    fit_moment_factors,
    pca_deflation,
    permuted_cross_moment_values,
    select_rank_label_free,
    zscore_columns,
)
from spectral_utils.dependency_fusion import sparse_upcr_fit  # noqa: E402
from spectral_utils.laplacian_upcr import (  # noqa: E402
    IU_FIT_DEFAULTS,
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_path,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "cm-lff-mixed-v2-24cell-v1-2026-08-10"
DEFAULT_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_OUT = REPO / "results" / "coupled_moment_kfactor_24cell_v1"
RANKS = (1, 2, 3, 4, 5)  # nuisance k = rank - 1
AMBIENT_DIMENSION = 6
SPLIT_SEEDS = (101, 211, 307, 401)
CP_SELECTION_SEEDS = (11, 23, 37)
CP_FULL_SEEDS = (11, 23, 37)
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
LIU_LAMBDA = 0.1
PERMUTATION_SEED = 8_104_113

DEPLOYED_STYLE_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "min_frac": 0.05,
    "exclude_frac": 3.0,
}

SOURCE_FILES = (
    "scripts/coupled_moment_24cell_experiment.py",
    "scripts/test_coupled_moment_fusion.py",
    "scripts/inscope_cells.py",
    "scripts/hard_filter_dufs_liu_benchmark.py",
    "spectral_utils/coupled_moment_fusion.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/upcr.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/selectors/a2_groupfs.py",
)

HEADLINE = {
    "deployed_upcr_mixed_v2": "Deployed-style U-PCR (mixed-v2)",
    "iu_pcr": "IU-PCR",
    "su_pcr": "SU-PCR reproduction",
    "sdsf": "SDSF",
    "dufs_liu": "DUFS-LIU",
    "cm_direct_selected": "CM-LFF direct factor",
    "cm_iu_selected": "CM-deflated IU-PCR",
    "cm_dufs_selected": "CM-deflated DUFS-LIU",
    "pca_iu_selected": "Second-order deflation control",
    "permuted_cm_iu_selected": "Permuted-moment control",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_bundle_without_labels(data) -> None:
    required = {
        f"{cell}__{suffix}"
        for cell in INSCOPE
        for suffix in ("V", "pool", "hand_signs")
    }
    missing = required - set(data.files)
    if missing:
        raise RuntimeError("bundle is incomplete: " + ", ".join(sorted(missing)))


class LabelFreeBundleView:
    """Minimal bundle interface that makes label arrays inaccessible to fit code."""

    _ALLOWED_SUFFIXES = ("V", "pool", "hand_signs")

    def __init__(self, arrays: dict[str, np.ndarray]):
        self._arrays = arrays
        self.files = tuple(sorted(arrays))

    @classmethod
    def from_npz(cls, data) -> "LabelFreeBundleView":
        allowed = {
            f"{cell}__{suffix}"
            for cell in INSCOPE
            for suffix in cls._ALLOWED_SUFFIXES
        }
        return cls({key: np.asarray(data[key]) for key in sorted(allowed)})

    def __getitem__(self, key: str) -> np.ndarray:
        if "label" in str(key).lower() or key not in self._arrays:
            raise KeyError(f"fit-time bundle key is unavailable: {key}")
        return self._arrays[key]


def repetition_groups(cell: str, n_samples: int) -> tuple[np.ndarray, str]:
    """Keep known K=10 repeated generations together during rank selection."""
    if cell.endswith("_k10"):
        if n_samples % 10:
            raise RuntimeError(f"K=10 cell length is not divisible by ten: {cell}")
        return np.arange(n_samples) // 10, "contiguous_k10_question_blocks"
    return np.arange(n_samples), "independent_rows"


def rank_key(prefix: str, rank: int) -> str:
    return f"{prefix}_k{rank - 1}"


def _iu_fit(F):
    return upcr_fit(F, **IU_FIT_DEFAULTS)


def _apply_loading_deflation(values, loadings, target_index):
    values = np.asarray(values, dtype=float)
    loadings = np.asarray(loadings, dtype=float)
    scores = values @ loadings @ np.linalg.pinv(loadings.T @ loadings, rcond=1e-10)
    nuisance = [index for index in range(loadings.shape[1]) if index != target_index]
    if not nuisance:
        return values.copy()
    return zscore_columns(values - scores[:, nuisance] @ loadings[:, nuisance].T)


def _selection_diagnostics(selection) -> dict:
    return {
        "selected_rank": selection.selected_rank,
        "selected_nuisance_k": selection.selected_rank - 1,
        "proposed_rank": selection.proposed_rank,
        "fallback_reasons": selection.fallback_reasons,
        "third_moment_stability": selection.third_moment_stability,
        "validation_errors": selection.validation_errors,
        "validation_means": selection.validation_means,
        "validation_standard_errors": selection.validation_standard_errors,
        "target_stability": selection.target_stability,
        "target_margin": selection.target_margin,
        "near_best_frequency": selection.near_best_frequency,
        "convergence_frequency": selection.convergence_frequency,
        "rank_rejection_reasons": selection.rank_rejection_reasons,
    }


def fit_cell(data, cell: str) -> tuple[dict[str, np.ndarray], dict]:
    """Fit one cell without indexing its label array."""
    started = time.time()
    F, names = load_contract(data, cell, "mixed_v2")
    F = np.asarray(F, dtype=float)
    values = F.T
    if not np.isfinite(F).all():
        raise RuntimeError(f"non-finite mixed-v2 matrix: {cell}")

    baseline_gates, baseline_gate_diag = dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    baseline_graph = build_graph_from_features(F, gates=baseline_gates, k=DUFS_K)
    baseline_path = laplacian_iu_path(
        F, (0.0, LIU_LAMBDA), graph=baseline_graph
    )
    iu = baseline_path[0.0].baseline
    iu_score = np.asarray(iu.w @ F, dtype=float)
    dufs_score = np.asarray(baseline_path[LIU_LAMBDA].w @ F, dtype=float)
    if not np.array_equal(iu_score, baseline_path[0.0].w @ F):
        raise RuntimeError(f"baseline lambda=0 identity failed: {cell}")

    groups, grouping_rule = repetition_groups(cell, len(values))

    def split_iu_reliability(split_values: np.ndarray) -> np.ndarray:
        """Re-estimate the label-free target anchor on the training half only."""
        return np.asarray(_iu_fit(np.asarray(split_values).T).rho_hat, dtype=float)

    selection = select_rank_label_free(
        values,
        iu.rho_hat,
        max_rank=max(RANKS),
        ambient_dimension=AMBIENT_DIMENSION,
        groups=groups,
        split_seeds=SPLIT_SEEDS,
        cp_seeds=CP_SELECTION_SEEDS,
        max_nfev=240,
        reliability_estimator=split_iu_reliability,
    )

    full_fits = {}
    score_arrays = {
        "sample_index": np.arange(len(values), dtype=np.int64),
        "feature_names": np.asarray(names, dtype=str),
        "deployed_upcr_mixed_v2": np.asarray(
            upcr_fit(F, **DEPLOYED_STYLE_FIT).w @ F, dtype=float
        ),
        "iu_pcr": iu_score,
        "dufs_liu": dufs_score,
    }
    sparse = sparse_upcr_fit(F)
    score_arrays["su_pcr"] = np.asarray(sparse.w_pcr @ F, dtype=float)
    score_arrays["sdsf"] = np.asarray(sparse.w_structured @ F, dtype=float)

    for rank in RANKS:
        fitted = fit_moment_factors(
            values,
            rank,
            iu.rho_hat,
            standardize=False,
            ambient_dimension=AMBIENT_DIMENSION,
            seeds=CP_FULL_SEEDS,
        )
        full_fits[rank] = fitted
        if rank == 1:
            cm_iu = iu_score.copy()
        else:
            cm_fit = _iu_fit(fitted.deflated_values.T)
            cm_iu = np.asarray(cm_fit.w @ fitted.deflated_values.T, dtype=float)
        score_arrays[rank_key("cm_direct", rank)] = fitted.target_score
        score_arrays[rank_key("cm_iu", rank)] = cm_iu

    selected_rank = int(selection.selected_rank)
    selected_fit = full_fits[selected_rank]
    full_guard_failures = []
    if selected_rank > 1:
        if not selected_fit.cp.converged:
            full_guard_failures.append("cp_nonconvergence")
        if selected_fit.cp.seed_agreement < 0.80:
            full_guard_failures.append("cp_seed_disagreement")
        if selected_fit.target_alignment < 0.20:
            full_guard_failures.append("weak_iu_target_alignment")
        if selected_fit.target_margin < 0.05:
            full_guard_failures.append("full_fit_target_ambiguity")
        if selected_fit.loading_condition > 100.0:
            full_guard_failures.append("ill_conditioned_loadings")
    if full_guard_failures:
        selected_rank = 1
        selected_fit = full_fits[1]

    selected_values = selected_fit.deflated_values
    score_arrays["cm_direct_selected"] = np.asarray(
        selected_fit.target_score if selected_rank > 1 else iu_score,
        dtype=float,
    )
    score_arrays["cm_iu_selected"] = score_arrays[rank_key("cm_iu", selected_rank)].copy()

    if selected_rank == 1:
        score_arrays["cm_dufs_selected"] = dufs_score.copy()
        cm_gate_diag = baseline_gate_diag
        cm_lambda0_error = 0.0
    else:
        cm_F = selected_values.T
        cm_gates, cm_gate_diag = dufs_soft_gates(
            cm_F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
        )
        cm_graph = build_graph_from_features(cm_F, gates=cm_gates, k=DUFS_K)
        cm_path = laplacian_iu_path(cm_F, (0.0, LIU_LAMBDA), graph=cm_graph)
        cm_zero = np.asarray(cm_path[0.0].w @ cm_F, dtype=float)
        cm_lambda0_error = float(
            np.max(np.abs(cm_zero - score_arrays["cm_iu_selected"]))
        )
        if cm_lambda0_error > 1e-12:
            raise RuntimeError(f"CM lambda=0 identity failed: {cell}")
        score_arrays["cm_dufs_selected"] = np.asarray(
            cm_path[LIU_LAMBDA].w @ cm_F, dtype=float
        )

    pca_values = pca_deflation(values, selected_rank, iu.rho_hat)
    pca_fit = _iu_fit(pca_values.T)
    score_arrays["pca_iu_selected"] = np.asarray(pca_fit.w @ pca_values.T, dtype=float)

    if selected_rank == 1:
        score_arrays["permuted_cm_iu_selected"] = iu_score.copy()
        permuted_diag = {"identity_at_rank_one": True}
    else:
        cell_seed = PERMUTATION_SEED + int(
            hashlib.sha256(cell.encode("utf-8")).hexdigest()[:8], 16
        )
        permuted = permuted_cross_moment_values(values, cell_seed)
        permuted_fit = fit_moment_factors(
            permuted,
            selected_rank,
            iu.rho_hat,
            standardize=False,
            ambient_dimension=AMBIENT_DIMENSION,
            seeds=CP_FULL_SEEDS,
        )
        permuted_values = _apply_loading_deflation(
            values, permuted_fit.loadings, permuted_fit.target_index
        )
        permuted_iu = _iu_fit(permuted_values.T)
        score_arrays["permuted_cm_iu_selected"] = np.asarray(
            permuted_iu.w @ permuted_values.T, dtype=float
        )
        permuted_diag = {
            "identity_at_rank_one": False,
            "cp_relative_error": permuted_fit.cp.relative_error,
            "cp_seed_agreement": permuted_fit.cp.seed_agreement,
            "target_alignment": permuted_fit.target_alignment,
            "target_margin": permuted_fit.target_margin,
        }

    expected_length = len(values)
    for key, array in score_arrays.items():
        if key in {"sample_index", "feature_names"}:
            continue
        array = np.asarray(array, dtype=float)
        if array.shape != (expected_length,) or not np.isfinite(array).all():
            raise RuntimeError(f"invalid score {cell}/{key}: {array.shape}")

    rank_diagnostics = {}
    for rank, fitted in full_fits.items():
        rank_diagnostics[str(rank)] = {
            "nuisance_k": rank - 1,
            "cp_relative_error": fitted.cp.relative_error,
            "cp_converged": fitted.cp.converged,
            "cp_best_seed": fitted.cp.best_seed,
            "cp_seed_agreement": fitted.cp.seed_agreement,
            "target_index": fitted.target_index,
            "target_alignment": fitted.target_alignment,
            "target_margin": fitted.target_margin,
            "loading_condition": fitted.loading_condition,
            "covariance_eigenvalues": fitted.covariance_eigenvalues,
            "target_loading": fitted.loadings[:, fitted.target_index],
            "component_strengths": fitted.cp.strengths,
            "score_variance": float(np.var(fitted.target_score)),
        }

    diagnostics = {
        "cell": cell,
        "domain": GROUP[cell],
        "n_samples": len(values),
        "n_features": F.shape[0],
        "feature_names": list(names),
        "grouping_rule": grouping_rule,
        "rank_selection": _selection_diagnostics(selection),
        "final_selected_rank": selected_rank,
        "final_selected_nuisance_k": selected_rank - 1,
        "full_fit_guard_failures": full_guard_failures,
        "rank_fits": rank_diagnostics,
        "baseline": {
            "iu_projection_residual": iu.proj_residual,
            "iu_lambda2_fraction": iu.lambda2_frac,
            "dufs": baseline_gate_diag,
            "dufs_weight_cosine_vs_iu": baseline_path[LIU_LAMBDA].diagnostics[
                "weight_cosine_vs_iu"
            ],
        },
        "cm_dufs": {
            "gates": cm_gate_diag,
            "lambda0_identity_max_abs_error": cm_lambda0_error,
        },
        "su_pcr": {
            "sparse_fraction": sparse.decomposition.sparse_fraction,
            "relative_residual": sparse.decomposition.relative_residual,
            "converged": sparse.decomposition.converged,
            "projection_residual": sparse.projection_residual,
        },
        "permuted_control": permuted_diag,
        "runtime_seconds": time.time() - started,
    }
    return score_arrays, diagnostics


def run_definition(bundle: Path) -> dict:
    payload = {
        "version": VERSION,
        "scientific_status": "retrospective_mechanism_study",
        "cells": list(INSCOPE),
        "domains": {cell: GROUP[cell] for cell in INSCOPE},
        "bundle": str(bundle.relative_to(REPO)),
        "bundle_sha256": sha256_file(bundle),
        "feature_contract": "dufs-liu-mixed-v2-development-2026-08-07",
        "ranks": list(RANKS),
        "nuisance_k": [rank - 1 for rank in RANKS],
        "ambient_dimension": AMBIENT_DIMENSION,
        "split_seeds": list(SPLIT_SEEDS),
        "cp_selection_seeds": list(CP_SELECTION_SEEDS),
        "cp_full_seeds": list(CP_FULL_SEEDS),
        "rank_rule": (
            "smallest eligible rank within one SE of best held-out "
            "all-distinct-feature K3 loss"
        ),
        "rank_guards": {
            "minimum_K3_split_stability": 0.75,
            "minimum_target_loading_stability": 0.75,
            "minimum_target_margin": 0.05,
            "minimum_near_best_frequency": 0.70,
            "minimum_full_CP_seed_agreement": 0.80,
            "minimum_full_target_alignment": 0.20,
            "maximum_loading_condition": 100.0,
            "full_CP_must_converge": True,
        },
        "target_anchor": (
            "maximum absolute cosine with IU-PCR rho_hat; orient positive; "
            "rho_hat re-estimated on each rank-selection training half"
        ),
        "dufs": {
            "seeds": list(DUFS_SEEDS),
            "epochs": DUFS_EPOCHS,
            "graph_k": DUFS_K,
            "lambda": LIU_LAMBDA,
        },
        "labels_used_during_fit": False,
        "source_sha256": {
            path: sha256_file(REPO / path) for path in SOURCE_FILES
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def trusted_resume_entries(out: Path, definition: dict) -> dict[str, dict]:
    """Return only checkpoints protected by a complete prior score freeze.

    A partial run has no trusted manifest and is recomputed. This prevents a
    modified checkpoint from being silently blessed by ``--resume``.
    """
    definition_path = out / "RUN_DEFINITION.json"
    complete_path = out / "FIT_COMPLETE.json"
    freeze_path = out / "SCORE_FREEZE_MANIFEST.json"
    if not (definition_path.exists() and complete_path.exists() and freeze_path.exists()):
        return {}
    try:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if complete["version"] != VERSION or freeze["version"] != VERSION:
            return {}
        if complete["run_fingerprint"] != definition["run_fingerprint"]:
            return {}
        if freeze["run_fingerprint"] != definition["run_fingerprint"]:
            return {}
        if freeze["run_definition_sha256"] != sha256_file(definition_path):
            return {}
        if freeze["fit_complete_sha256"] != sha256_file(complete_path):
            return {}
        if complete["manifest"] != freeze["manifest"]:
            return {}
        if [item["cell"] for item in complete["manifest"]] != list(INSCOPE):
            return {}
        trusted = {}
        for item in complete["manifest"]:
            score_path = out / item["score_file"]
            diagnostic_path = out / item["diagnostic_file"]
            if sha256_file(score_path) != item["score_sha256"]:
                return {}
            if sha256_file(diagnostic_path) != item["diagnostic_sha256"]:
                return {}
            trusted[item["cell"]] = item
        return trusted
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return {}


def fit_command(args) -> None:
    bundle = Path(args.bundle).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    (out / "diagnostics").mkdir(exist_ok=True)
    definition = run_definition(bundle)
    definition_path = out / "RUN_DEFINITION.json"
    if definition_path.exists():
        previous = json.loads(definition_path.read_text(encoding="utf-8"))
        if previous != definition:
            raise RuntimeError("output directory has a different run definition")
    else:
        write_json(definition_path, definition)

    with np.load(bundle, allow_pickle=True) as raw_data:
        validate_bundle_without_labels(raw_data)
        data = LabelFreeBundleView.from_npz(raw_data)
    if any("label" in key.lower() for key in data.files):
        raise RuntimeError("label-like key leaked into the fit-time adapter")
    trusted = trusted_resume_entries(out, definition) if args.resume else {}
    manifest = []
    started = time.time()
    for index, cell in enumerate(INSCOPE, start=1):
        score_path = out / "scores" / f"{cell}.npz"
        diagnostic_path = out / "diagnostics" / f"{cell}.json"
        if cell in trusted:
            status = "reused"
        else:
            scores, diagnostics = fit_cell(data, cell)
            np.savez_compressed(score_path, **scores)
            write_json(diagnostic_path, diagnostics)
            status = "fit"
        with np.load(score_path, allow_pickle=False) as checkpoint:
            if any("label" in key.lower() for key in checkpoint.files):
                raise RuntimeError(f"label-like score key found: {cell}")
        manifest.append({
            "cell": cell,
            "score_file": str(score_path.relative_to(out)),
            "score_sha256": sha256_file(score_path),
            "diagnostic_file": str(diagnostic_path.relative_to(out)),
            "diagnostic_sha256": sha256_file(diagnostic_path),
        })
        print(json.dumps({
            "progress": f"{index}/24",
            "cell": cell,
            "status": status,
            "elapsed_seconds": round(time.time() - started, 2),
        }), flush=True)

    complete = {
        "version": VERSION,
        "run_fingerprint": definition["run_fingerprint"],
        "n_cells": 24,
        "labels_opened": False,
        "runtime_seconds": time.time() - started,
        "manifest": manifest,
    }
    write_json(out / "FIT_COMPLETE.json", complete)
    freeze = {
        "version": VERSION,
        "run_fingerprint": definition["run_fingerprint"],
        "run_definition_sha256": sha256_file(definition_path),
        "fit_complete_sha256": sha256_file(out / "FIT_COMPLETE.json"),
        "score_files_verified_before_labels": True,
        "manifest": manifest,
    }
    write_json(out / "SCORE_FREEZE_MANIFEST.json", freeze)
    print("Label-free fit complete and scores frozen.", flush=True)


def verify_freeze(out: Path, bundle: Path) -> tuple[dict, dict, dict]:
    definition = json.loads((out / "RUN_DEFINITION.json").read_text(encoding="utf-8"))
    complete = json.loads((out / "FIT_COMPLETE.json").read_text(encoding="utf-8"))
    freeze = json.loads((out / "SCORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    if definition["version"] != VERSION or complete["version"] != VERSION:
        raise RuntimeError("version mismatch")
    if definition["run_fingerprint"] != complete["run_fingerprint"]:
        raise RuntimeError("run fingerprint mismatch")
    if definition["bundle_sha256"] != sha256_file(bundle):
        raise RuntimeError("bundle changed after fitting")
    for relative, expected in definition["source_sha256"].items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError(f"source changed after fitting: {relative}")
    if freeze["fit_complete_sha256"] != sha256_file(out / "FIT_COMPLETE.json"):
        raise RuntimeError("fit completion file changed")
    scores = {}
    diagnostics = {}
    if [item["cell"] for item in complete["manifest"]] != list(INSCOPE):
        raise RuntimeError("cell roster mismatch")
    for item in complete["manifest"]:
        cell = item["cell"]
        score_path = out / item["score_file"]
        diagnostic_path = out / item["diagnostic_file"]
        if sha256_file(score_path) != item["score_sha256"]:
            raise RuntimeError(f"score hash changed: {cell}")
        if sha256_file(diagnostic_path) != item["diagnostic_sha256"]:
            raise RuntimeError(f"diagnostic hash changed: {cell}")
        checkpoint = np.load(score_path, allow_pickle=False)
        if any("label" in key.lower() for key in checkpoint.files):
            raise RuntimeError(f"label-like score key found: {cell}")
        scores[cell] = {key: np.asarray(checkpoint[key]) for key in checkpoint.files}
        diagnostics[cell] = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    return definition, scores, diagnostics


FAMILIES = (
    "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500",
)


def family(cell: str) -> str:
    return next((name for name in FAMILIES if name in cell), cell)


def bootstrap_ci(values, namespace: str, count: int = 20000):
    values = np.asarray(values, dtype=float)
    seed = int(hashlib.sha256(namespace.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(count, len(values)))]
    return tuple(float(value) for value in np.quantile(samples.mean(axis=1), (0.025, 0.975)))


def family_values(values_by_cell: dict[str, float]) -> np.ndarray:
    return np.asarray([
        np.mean([values_by_cell[cell] for cell in INSCOPE if family(cell) == item])
        for item in FAMILIES
    ], dtype=float)


def evaluate(scores, bundle):
    from sklearn.metrics import average_precision_score, roc_auc_score

    data = np.load(bundle, allow_pickle=True)
    rows = []
    method_keys = list(HEADLINE) + [
        rank_key(prefix, rank)
        for prefix in ("cm_direct", "cm_iu")
        for rank in RANKS
    ]
    for cell in INSCOPE:
        labels = np.asarray(data[f"{cell}__labels"], dtype=int)
        if not np.array_equal(scores[cell]["sample_index"], np.arange(len(labels))):
            raise RuntimeError(f"sample order mismatch: {cell}")
        for method in method_keys:
            values = np.asarray(scores[cell][method], dtype=float)
            rows.append({
                "cell": cell,
                "family": family(cell),
                "domain": GROUP[cell],
                "n": len(labels),
                "positive_rate": float(labels.mean()),
                "method_key": method,
                "method": HEADLINE.get(method, method),
                "auroc": float(roc_auc_score(labels, values)),
                "auprc": float(average_precision_score(labels, values)),
                "score_variance": float(np.var(values)),
            })
    return rows


def summarize(rows):
    lookup = {(row["cell"], row["method_key"]): row for row in rows}
    summary = []
    for method, name in HEADLINE.items():
        auc = {cell: lookup[cell, method]["auroc"] for cell in INSCOPE}
        pr = {cell: lookup[cell, method]["auprc"] for cell in INSCOPE}
        values = np.asarray(list(auc.values()))
        lo, hi = bootstrap_ci(values, f"headline-{method}")
        fam = family_values(auc)
        fam_lo, fam_hi = bootstrap_ci(fam, f"family-{method}")
        summary.append({
            "method_key": method,
            "method": name,
            "cell_macro_auroc": float(values.mean()),
            "cell_ci_low": lo,
            "cell_ci_high": hi,
            "family_macro_auroc": float(fam.mean()),
            "family_ci_low": fam_lo,
            "family_ci_high": fam_hi,
            "qa_macro_auroc": float(np.mean([
                auc[cell] for cell in INSCOPE if GROUP[cell] == "QA"
            ])),
            "math_macro_auroc": float(np.mean([
                auc[cell] for cell in INSCOPE if GROUP[cell] == "math"
            ])),
            "cell_macro_auprc": float(np.mean(list(pr.values()))),
        })
    return summary


def comparisons(rows):
    from scipy.stats import wilcoxon

    lookup = {(row["cell"], row["method_key"]): row["auroc"] for row in rows}
    output = []
    candidates = ("cm_direct_selected", "cm_iu_selected", "cm_dufs_selected")
    for candidate in candidates:
        for baseline in ("iu_pcr", "dufs_liu"):
            deltas = {
                cell: lookup[cell, candidate] - lookup[cell, baseline]
                for cell in INSCOPE
            }
            values = np.asarray(list(deltas.values()))
            fam = family_values(deltas)
            fam_lo, fam_hi = bootstrap_ci(fam, f"delta-{candidate}-{baseline}")
            try:
                pvalue = float(wilcoxon(values, zero_method="pratt").pvalue)
            except ValueError:
                pvalue = 1.0
            output.append({
                "candidate": candidate,
                "baseline": baseline,
                "cell_mean_delta_pp": float(100 * values.mean()),
                "family_mean_delta_pp": float(100 * fam.mean()),
                "family_ci_low_pp": float(100 * fam_lo),
                "family_ci_high_pp": float(100 * fam_hi),
                "wins": int(np.sum(values > 1e-12)),
                "ties": int(np.sum(np.abs(values) <= 1e-12)),
                "losses": int(np.sum(values < -1e-12)),
                "worst_delta_pp": float(100 * values.min()),
                "wilcoxon_p": pvalue,
            })
    return output


def rank_path(rows):
    lookup = {(row["cell"], row["method_key"]): row["auroc"] for row in rows}
    output = []
    for prefix in ("cm_direct", "cm_iu"):
        for rank in RANKS:
            values = np.asarray([
                lookup[cell, rank_key(prefix, rank)] for cell in INSCOPE
            ])
            lo, hi = bootstrap_ci(values, f"rank-path-{prefix}-{rank}")
            output.append({
                "method": prefix,
                "rank": rank,
                "nuisance_k": rank - 1,
                "mean_auroc": float(values.mean()),
                "ci_low": lo,
                "ci_high": hi,
            })
    return output


def _save_figure(path: Path):
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def make_plots(out, rows, summary, diagnostics):
    import matplotlib.pyplot as plt

    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    lookup = {(row["cell"], row["method_key"]): row["auroc"] for row in rows}

    shown = [
        "deployed_upcr_mixed_v2", "iu_pcr", "su_pcr", "dufs_liu",
        "cm_direct_selected", "cm_iu_selected", "cm_dufs_selected",
    ]
    summary_lookup = {row["method_key"]: row for row in summary}
    means = [summary_lookup[key]["cell_macro_auroc"] for key in shown]
    lows = [means[i] - summary_lookup[key]["cell_ci_low"] for i, key in enumerate(shown)]
    highs = [summary_lookup[key]["cell_ci_high"] - means[i] for i, key in enumerate(shown)]
    plt.figure(figsize=(10, 4.8))
    plt.errorbar(range(len(shown)), means, yerr=[lows, highs], fmt="o", capsize=4)
    plt.xticks(range(len(shown)), [HEADLINE[key] for key in shown], rotation=30, ha="right")
    plt.ylabel("Cell-macro AUROC")
    plt.title("24-cell comparison (95% cell-bootstrap intervals)")
    plt.grid(axis="y", alpha=0.25)
    _save_figure(figures / "headline_auroc.png")

    for prefix, title in (("cm_iu", "CM-deflated IU-PCR"), ("cm_direct", "Direct CM factor")):
        means, lower, upper = [], [], []
        for rank in RANKS:
            values = np.asarray([lookup[cell, rank_key(prefix, rank)] for cell in INSCOPE])
            lo, hi = bootstrap_ci(values, f"plot-{prefix}-{rank}")
            means.append(values.mean()); lower.append(values.mean() - lo); upper.append(hi - values.mean())
        plt.figure(figsize=(7, 4.2))
        plt.errorbar([rank - 1 for rank in RANKS], means, yerr=[lower, upper], marker="o", capsize=4)
        plt.axhline(
            summary_lookup["iu_pcr"]["cell_macro_auroc"], color="black", ls="--", label="IU-PCR"
        )
        plt.xlabel("Number of nuisance factors k")
        plt.ylabel("Cell-macro AUROC")
        plt.title(f"Fixed label-free rank path: {title}")
        plt.legend()
        plt.grid(alpha=0.25)
        _save_figure(figures / f"{prefix}_rank_path.png")

    delta = np.asarray([
        lookup[cell, "cm_iu_selected"] - lookup[cell, "iu_pcr"] for cell in INSCOPE
    ]) * 100
    order = np.argsort(delta)
    colors = ["#b91c1c" if value < 0 else "#047857" for value in delta[order]]
    plt.figure(figsize=(9, 7.2))
    plt.barh(range(24), delta[order], color=colors)
    plt.yticks(range(24), [INSCOPE[index] for index in order], fontsize=7)
    plt.axvline(0, color="black", lw=1)
    plt.xlabel("CM-deflated IU-PCR minus IU-PCR (AUROC points)")
    plt.title("Per-cell effect of the selected latent-factor correction")
    _save_figure(figures / "per_cell_delta.png")

    n = np.asarray([diagnostics[cell]["n_samples"] for cell in INSCOPE])
    stability = np.asarray([
        diagnostics[cell]["rank_selection"]["third_moment_stability"] for cell in INSCOPE
    ])
    selected = np.asarray([
        diagnostics[cell]["final_selected_nuisance_k"] for cell in INSCOPE
    ])
    plt.figure(figsize=(7, 4.6))
    scatter = plt.scatter(n, stability, c=selected, cmap="viridis", s=65, edgecolor="black")
    plt.xscale("log")
    plt.axhline(0.75, color="#b91c1c", ls="--", label="stability gate")
    plt.xlabel("Samples in cell (log scale)")
    plt.ylabel("Split-half all-distinct K3 correlation")
    plt.title("Third-moment reliability and selected nuisance rank")
    plt.colorbar(scatter, label="selected nuisance k")
    plt.legend()
    _save_figure(figures / "moment_stability.png")

    counts = [int(np.sum(selected == k)) for k in range(5)]
    plt.figure(figsize=(6.5, 4))
    plt.bar(range(5), counts, color="#4f46e5")
    plt.xticks(range(5))
    plt.xlabel("Selected nuisance factors k")
    plt.ylabel("Cells")
    plt.title("Label-free rank selection after stability guards")
    _save_figure(figures / "selected_rank_counts.png")


def figure_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def methods_text() -> str:
    return f"""# Coupled-Moment Latent-Factor Fusion: methods

## Claim boundary

This is an exploratory adaptation, not a method published in Ibrahim et al.
(2025), and not a theorem-backed extension of continuous U-PCR. The tensor
identifiability results in that survey concern categorical Dawid--Skene
confusion matrices. Our continuous feature model has weaker identifiability.

## Input

Every cell uses the frozen confidence-oriented mixed-v2 contract. The union has
30 features, but a cell keeps only the 19--30 features actually available. No
missing feature is imputed. Let X be samples by features.

## Model and fitting

Covariance selects its leading {AMBIENT_DIMENSION}-dimensional subspace Q. We
then fit a symmetric CP model using only third central moments whose three
original feature indexes are different:

    E[X_i X_j X_k] approximately equals
        sum_l kappa_l b_i,l b_j,l b_k,l, for i < j < k.

Repeated-index entries are excluded because feature-specific marginal skew can
create false shared components. The projected tensor is used only to initialize
the masked optimization.

For total rank r, nuisance k=r-1. The target candidate is the component whose
loading has the largest absolute cosine with IU-PCR's label-free rho estimate;
its sign is oriented toward rho. Other components are reconstructed and removed
from X, after which ordinary full-pool two-component IU-PCR is fitted again.
DUFS-LIU is optionally fitted on the same cleaned X with its frozen settings.

## Label-free rank choice and fallbacks

Ranks 1--5 are compared on four deterministic half splits. K=10 repeated
generations stay in the same half. Rank uses held-out third-moment reconstruction
and the smallest-rank one-standard-error rule. A rank above one must also pass:

- all-distinct K3 split stability >= 0.75;
- target-loading stability >= 0.75;
- target-alignment margin >= 0.05;
- within-5%-of-best frequency >= 0.70;
- full-fit seed agreement >= 0.80;
- full-fit target alignment >= 0.20;
- loading Gram condition number <= 100;
- convergence of every split fit and the selected full-data fit.

Failure returns the exact IU-PCR/DUFS-LIU input, not a tuned alternative.

## Comparators and controls

The same mixed-v2 rows are scored by deployed-style U-PCR, IU-PCR, the local
SU-PCR reproduction, SDSF, and frozen DUFS-LIU. A PCA nuisance-deflation arm
tests whether third moments matter beyond covariance. A feature-wise permutation
arm preserves each feature marginal but destroys covariance and higher-order
cross-feature dependence. It is a broad dependency-destruction control, not a
pure third-moment ablation.

Labels are structurally absent from fitting. Scores and diagnostics are hashed
before labels are loaded. Because mixed-v2 and these 24 cells were used during
development, results are retrospective evidence only.
"""


def render_report(out, rows, summary, comparison_rows, rank_rows, diagnostics):
    summary_lookup = {row["method_key"]: row for row in summary}
    selected_counts = {
        k: sum(diagnostics[cell]["final_selected_nuisance_k"] == k for cell in INSCOPE)
        for k in range(5)
    }
    fallbacks = sum(bool(diagnostics[cell]["full_fit_guard_failures"]) for cell in INSCOPE)
    cm = summary_lookup["cm_iu_selected"]
    iu = summary_lookup["iu_pcr"]
    dufs = summary_lookup["dufs_liu"]
    delta_iu = next(row for row in comparison_rows
                    if row["candidate"] == "cm_iu_selected" and row["baseline"] == "iu_pcr")
    delta_dufs = next(row for row in comparison_rows
                      if row["candidate"] == "cm_iu_selected" and row["baseline"] == "dufs_liu")
    promotion = {
        "gain_at_least_half_point_vs_iu": delta_iu["cell_mean_delta_pp"] >= 0.5,
        "gain_at_least_half_point_vs_dufs": delta_dufs["cell_mean_delta_pp"] >= 0.5,
        "family_interval_positive_vs_iu": delta_iu["family_ci_low_pp"] > 0,
        "qa_nonnegative": cm["qa_macro_auroc"] >= iu["qa_macro_auroc"],
        "math_nonnegative": cm["math_macro_auroc"] >= iu["math_macro_auroc"],
        "at_least_14_wins": delta_iu["wins"] >= 14,
        "worst_loss_no_more_than_2pp": delta_iu["worst_delta_pp"] >= -2.0,
    }
    promoted = all(promotion.values())

    lines = [
        "# Coupled-moment k-factor experiment",
        "",
        f"**Decision: {'PROMOTE FOR EXTERNAL CONFIRMATION' if promoted else 'DO NOT PROMOTE'}.**",
        "",
        "This experiment asked whether non-Gaussian latent nuisance factors can be identified from the original mixed-v2 features, removed, and followed by the same IU-PCR/DUFS-LIU solvers.",
        "",
        "## Main result",
        "",
        f"CM-deflated IU-PCR scored **{cm['cell_macro_auroc']:.4f}** cell-macro AUROC, compared with **{iu['cell_macro_auroc']:.4f}** for IU-PCR and **{dufs['cell_macro_auroc']:.4f}** for DUFS-LIU.",
        f"Its change versus IU-PCR was **{delta_iu['cell_mean_delta_pp']:+.3f} points** ({delta_iu['wins']} wins, {delta_iu['losses']} losses; worst {delta_iu['worst_delta_pp']:+.3f}). The equal-family interval was [{delta_iu['family_ci_low_pp']:+.3f}, {delta_iu['family_ci_high_pp']:+.3f}] points.",
        "",
        f"Selected nuisance-factor counts were {selected_counts}. {fallbacks} cells additionally failed a full-fit ambiguity guard.",
        "",
        "## Headline table",
        "",
        "| Method | Cell AUROC | Family AUROC | QA | Math | AUPRC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {row['cell_macro_auroc']:.4f} | "
            f"{row['family_macro_auroc']:.4f} | {row['qa_macro_auroc']:.4f} | "
            f"{row['math_macro_auroc']:.4f} | {row['cell_macro_auprc']:.4f} |"
        )
    lines += ["", "## Promotion gates", ""]
    for key, passed in promotion.items():
        lines.append(f"- {'PASS' if passed else 'FAIL'} — {key.replace('_', ' ')}")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "CM-LFF is not tensor-identifiable under the categorical theorem in the source survey. A stable component may still represent confidence, length, or difficulty rather than correctness. This report evaluates retrospective development cells and cannot establish an external claim.",
        "",
        "## Figures",
        "",
        "![Headline](figures/headline_auroc.png)",
        "![Rank path](figures/cm_iu_rank_path.png)",
        "![Per-cell delta](figures/per_cell_delta.png)",
        "![Moment stability](figures/moment_stability.png)",
        "![Selected ranks](figures/selected_rank_counts.png)",
    ]
    report_md = "\n".join(lines) + "\n"
    (out / "REPORT.md").write_text(report_md, encoding="utf-8")
    (out / "METHODS.md").write_text(methods_text(), encoding="utf-8")

    table_rows = "".join(
        "<tr>" + "".join([
            f"<td>{row['method']}</td>",
            f"<td>{row['cell_macro_auroc']:.4f}</td>",
            f"<td>{row['family_macro_auroc']:.4f}</td>",
            f"<td>{row['qa_macro_auroc']:.4f}</td>",
            f"<td>{row['math_macro_auroc']:.4f}</td>",
        ]) + "</tr>" for row in summary
    )
    gate_html = "".join(
        f"<li class='{'pass' if passed else 'fail'}'>{'PASS' if passed else 'FAIL'} — {key.replace('_', ' ')}</li>"
        for key, passed in promotion.items()
    )
    images = "".join(
        f"<section><h2>{title}</h2><img src='{figure_data_uri(out / 'figures' / filename)}'></section>"
        for title, filename in (
            ("Headline performance", "headline_auroc.png"),
            ("Fixed k path", "cm_iu_rank_path.png"),
            ("Per-cell changes", "per_cell_delta.png"),
            ("Moment reliability", "moment_stability.png"),
            ("Selected ranks", "selected_rank_counts.png"),
        )
    )
    html = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>CM-LFF 24-cell study</title><style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:32px auto;line-height:1.45;color:#172033}}
h1,h2{{color:#172554}} .card{{padding:18px;border:1px solid #dbeafe;border-radius:12px;background:#f8fafc}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}
img{{max-width:100%;border:1px solid #e5e7eb;border-radius:8px}}.pass{{color:#047857}}.fail{{color:#b91c1c}}
</style></head><body><h1>Coupled-Moment Latent-Factor Fusion</h1>
<div class='card'><h2>Decision</h2><p><strong>{'PROMOTE FOR EXTERNAL CONFIRMATION' if promoted else 'DO NOT PROMOTE'}</strong></p>
<p>CM-deflated IU-PCR: <strong>{cm['cell_macro_auroc']:.4f}</strong>; IU-PCR: {iu['cell_macro_auroc']:.4f}; DUFS-LIU: {dufs['cell_macro_auroc']:.4f}.</p>
<p>Difference from IU-PCR: {delta_iu['cell_mean_delta_pp']:+.3f} AUROC points; equal-family 95% interval [{delta_iu['family_ci_low_pp']:+.3f}, {delta_iu['family_ci_high_pp']:+.3f}].</p></div>
<h2>What changed</h2><p>Covariance defined a six-dimensional subspace. Third moments separated up to five latent components. The component closest to IU-PCR's label-free reliability vector was retained; other components were removed before rerunning the unchanged IU-PCR and DUFS-LIU solvers.</p>
<p>This is an exploratory continuous adaptation. It is not the categorical tensor algorithm or theorem from Ibrahim et al. (2025).</p>
<h2>Results</h2><table><thead><tr><th>Method</th><th>Cell AUROC</th><th>Family AUROC</th><th>QA</th><th>Math</th></tr></thead><tbody>{table_rows}</tbody></table>
<h2>Promotion gates</h2><ul>{gate_html}</ul>{images}
<h2>Audit boundary</h2><p>All scores were created and hashed before labels were opened. The feature contract and all 24 cells are retrospective development data.</p>
</body></html>"""
    (out / "REPORT.html").write_text(html, encoding="utf-8")
    return promotion, promoted


def report_command(args):
    out = Path(args.out).resolve()
    bundle = Path(args.bundle).resolve()
    definition, scores, diagnostics = verify_freeze(out, bundle)
    rows = evaluate(scores, bundle)
    summary = summarize(rows)
    comparison_rows = comparisons(rows)
    rank_rows = rank_path(rows)
    write_csv(out / "per_cell_metrics.csv", rows)
    write_csv(out / "headline_summary.csv", summary)
    write_csv(out / "paired_comparisons.csv", comparison_rows)
    write_csv(out / "rank_path.csv", rank_rows)
    selection_rows = []
    for cell in INSCOPE:
        item = diagnostics[cell]
        selection_rows.append({
            "cell": cell,
            "family": family(cell),
            "domain": GROUP[cell],
            "n": item["n_samples"],
            "m": item["n_features"],
            "third_moment_stability": item["rank_selection"]["third_moment_stability"],
            "proposed_rank": item["rank_selection"]["proposed_rank"],
            "final_rank": item["final_selected_rank"],
            "nuisance_k": item["final_selected_nuisance_k"],
            "fallback_reasons": ";".join(item["rank_selection"]["fallback_reasons"]),
            "full_guard_failures": ";".join(item["full_fit_guard_failures"]),
            "target_alignment": item["rank_fits"][str(item["final_selected_rank"])]["target_alignment"],
            "target_margin": item["rank_fits"][str(item["final_selected_rank"])]["target_margin"],
            "cp_seed_agreement": item["rank_fits"][str(item["final_selected_rank"])]["cp_seed_agreement"],
        })
    write_csv(out / "rank_selection.csv", selection_rows)
    make_plots(out, rows, summary, diagnostics)
    promotion, promoted = render_report(
        out, rows, summary, comparison_rows, rank_rows, diagnostics
    )
    write_json(out / "CONCLUSION.json", {
        "version": VERSION,
        "run_fingerprint": definition["run_fingerprint"],
        "promotion_gates": promotion,
        "promoted": promoted,
        "labels_opened_only_after_score_verification": True,
        "report_sha256": sha256_file(out / "REPORT.md"),
        "html_sha256": sha256_file(out / "REPORT.html"),
    })
    print(f"Report complete: {out / 'REPORT.html'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fit", "report", "all"))
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.command in ("fit", "all"):
        fit_command(args)
    if args.command in ("report", "all"):
        report_command(args)


if __name__ == "__main__":
    main()
