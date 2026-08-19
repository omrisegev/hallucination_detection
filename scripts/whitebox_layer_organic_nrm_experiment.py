#!/usr/bin/env python3
"""Retrospective layer-organic NRM addendum for white-box layer views.

This experiment keeps three residual-stream measurements as separate features
inside each transformer-layer group.  It reuses the immutable, label-free v2
``lens_grid_all`` matrices and fits every cross-cell calibration before an
evaluator opens correctness labels.  Exact layer identity is compared only
among the ten protocol-eligible 32-layer cells; 36/40-layer cells are excluded
rather than silently interpolated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.whitebox_layer_fusion_experiment import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    CELLS,
    MODEL,
    ORIGINAL_LLAMA_CELLS,
    PRIMARY_CELLS,
    SEED,
    TIE_TOLERANCE,
    group_bootstrap_indices,
    holm_adjust,
    load_feature_matrix,
)
from scripts.whitebox_layer_fusion_nrm_experiment import (  # noqa: E402
    _load_base_score_bundle,
    _metric_pair,
    _verify_base_prepared,
    _verify_base_scores,
    load_pickle,
    read_json,
    sha256_file,
    utcnow,
    write_csv,
    write_json,
)
from spectral_utils.whitebox_layer_fusion import (  # noqa: E402
    apply_neutral_residual_calibration,
    fit_group_contribution_space,
    fit_neutral_residual_calibration,
    load_evaluation_labels,
)
from spectral_utils.whitebox_layer_organic_nrm import (  # noqa: E402
    KL_SENSITIVITY_METRICS,
    LOCAL_RESID_METRICS,
    assert_layer_organic_contract,
    layer_organic_residual_matrix,
)


VERSION = "whitebox-layer-organic-nrm-v1-2026-08-13"
BASE_RESULTS = REPO / "results" / "whitebox_layer_fusion_v2"
DEFAULT_RESULTS = REPO / "results" / "whitebox_layer_organic_nrm_v1"
DEFAULT_CACHE = REPO / "dataset_cache" / "whitebox_layer_fusion_v1"
RUNNER_SOURCE = "scripts/whitebox_layer_organic_nrm_experiment.py"
REPORT_SOURCE = "scripts/whitebox_layer_organic_nrm_report.py"
FIT_SOURCE_FILES = (
    RUNNER_SOURCE,
    "spectral_utils/whitebox_layer_organic_nrm.py",
    "spectral_utils/whitebox_layer_fusion.py",
    "scripts/whitebox_layer_fusion_nrm_experiment.py",
    "scripts/whitebox_layer_fusion_experiment.py",
    "spectral_utils/paper_benchmark_suite.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
)

L32_CELLS = tuple(
    cell for cell in PRIMARY_CELLS if int(CELLS[cell].get("n_layers", 32)) == 32
)
LLAMA6_CELLS = tuple(cell for cell in ORIGINAL_LLAMA_CELLS if cell in L32_CELLS)
GSM8K_L32_CELLS = tuple(
    cell for cell in L32_CELLS if CELLS[cell]["dataset"] == "GSM8K"
)
NON_L32_CELLS = tuple(cell for cell in PRIMARY_CELLS if cell not in L32_CELLS)

BASE_KEYS = {
    "final_nll": "final_layer_nll__resid-core-L__all__flat",
    "iu_compressed": "iu_pcr__resid-core-L__all__flat",
}
METHODS = {
    "final_nll": "Final-layer NLL",
    "iu_compressed": "IU-PCR · one averaged expert/layer",
    "iu_layer_triad": "IU-PCR · layer-organic triad",
    "nrm_layer_lodo": "Layer-organic NRM · leave-dataset-out",
    "nrm_layer_lomo": "Layer-organic NRM · leave-model-out",
    "nrm_layer_loco": "Layer-organic NRM · leave-cell-out",
    "iu_layer_kl": "IU-PCR · layer-organic + KL sensitivity",
    "nrm_layer_kl_lodo": "Layer-organic NRM + KL · leave-dataset-out",
    "nrm_layer_llama_loco": "Layer-organic NRM · Llama-only leave-cell-out",
    "nrm_layer_gsm8k_loco": "Layer-organic NRM · GSM8K-32L leave-model-out",
}


def _model(cell: str) -> str:
    return str(CELLS[cell].get("model", MODEL))


def _source_cells(target: str, strategy: str) -> tuple[str, ...]:
    if target not in L32_CELLS:
        raise ValueError(f"{target} is not in the exact 32-layer cohort")
    if strategy == "loco":
        sources = tuple(cell for cell in L32_CELLS if cell != target)
    elif strategy == "lodo":
        sources = tuple(
            cell for cell in L32_CELLS
            if CELLS[cell]["dataset"] != CELLS[target]["dataset"]
        )
    elif strategy == "lomo":
        sources = tuple(
            cell for cell in L32_CELLS if _model(cell) != _model(target)
        )
    else:
        raise ValueError("strategy must be lodo, lomo, or loco")
    if len(sources) < 3 or target in sources:
        raise RuntimeError(f"{target}/{strategy}: invalid source roster {sources}")
    return sources


def _tailored_source_cells(target: str, cohort: Sequence[str]) -> tuple[str, ...]:
    roster = tuple(cohort)
    if target not in roster:
        raise ValueError(f"{target} is outside tailored cohort")
    sources = tuple(cell for cell in roster if cell != target)
    if len(sources) < 3:
        raise RuntimeError("tailored NRM needs at least three source cells")
    return sources


def _fit_score(space: Any, source_spaces: Sequence[Any], sources: Sequence[str]) -> tuple[np.ndarray, dict[str, Any]]:
    calibration = fit_neutral_residual_calibration(
        source_spaces, source_ids=tuple(sources)
    )
    fitted = apply_neutral_residual_calibration(space, calibration)
    return fitted.score, {
        "source_cells": list(sources),
        "source_count": len(sources),
        "calibration": {
            **dict(calibration.diagnostics),
            "direction": calibration.direction.tolist(),
            "eigenvalues": calibration.eigenvalues.tolist(),
            "residual_covariance": calibration.residual_covariance.tolist(),
        },
        "target_fit": dict(fitted.diagnostics),
        "iu_reconstruction_error": space.diagnostics["reconstruction_error"],
    }


def phase_fit(base_results: Path, results: Path) -> None:
    prepared = _verify_base_prepared(base_results)
    base_fit = _verify_base_scores(base_results)
    spaces: dict[str, dict[str, Any]] = defaultdict(dict)
    row_metadata: dict[str, dict[str, np.ndarray]] = {}

    for cell in L32_CELLS:
        print(f"[organic-fit] layer groups: {cell}", flush=True)
        with np.load(base_results / "prepared" / f"{cell}__rows.npz", allow_pickle=False) as bundle:
            row_metadata[cell] = {
                "row_ids": bundle["row_ids"].astype(str),
                "problem_ids": bundle["problem_ids"].astype(str),
            }
        grid = load_feature_matrix(
            base_results / "prepared" / f"{cell}__lens_grid_all.npz"
        )
        triad = layer_organic_residual_matrix(grid, metrics=LOCAL_RESID_METRICS)
        kl = layer_organic_residual_matrix(grid, metrics=KL_SENSITIVITY_METRICS)
        assert_layer_organic_contract(triad, n_layers=32)
        assert_layer_organic_contract(kl, n_layers=32)
        spaces["triad"][cell] = fit_group_contribution_space(triad)
        spaces["kl"][cell] = fit_group_contribution_space(kl)
        for contract in ("triad", "kl"):
            if spaces[contract][cell].diagnostics["reconstruction_error"] > 1e-10:
                raise RuntimeError(f"{cell}/{contract}: IU contribution reconstruction failed")

    score_dir = results / "scores"
    diagnostic_dir = results / "diagnostics"
    score_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    score_manifest = []
    calibration_rows = []
    for target in L32_CELLS:
        output = {
            "iu_layer_triad": spaces["triad"][target].baseline_score,
            "iu_layer_kl": spaces["kl"][target].baseline_score,
        }
        diagnostics: dict[str, Any] = {
            "target": target,
            "labels_seen_during_fit": False,
            "scores_fitted_before_outcomes_opened": True,
            "contracts": {
                "triad": dict(spaces["triad"][target].diagnostics),
                "kl_sensitivity": dict(spaces["kl"][target].diagnostics),
            },
            "fits": {},
        }
        for strategy in ("lodo", "lomo", "loco"):
            sources = _source_cells(target, strategy)
            score, fit = _fit_score(
                spaces["triad"][target],
                [spaces["triad"][cell] for cell in sources],
                sources,
            )
            method = f"nrm_layer_{strategy}"
            output[method] = score
            diagnostics["fits"][method] = fit
            calibration_rows.append(_calibration_row(target, method, "triad", fit))

        sources = _source_cells(target, "lodo")
        score, fit = _fit_score(
            spaces["kl"][target],
            [spaces["kl"][cell] for cell in sources],
            sources,
        )
        output["nrm_layer_kl_lodo"] = score
        diagnostics["fits"]["nrm_layer_kl_lodo"] = fit
        calibration_rows.append(_calibration_row(
            target, "nrm_layer_kl_lodo", "kl_sensitivity", fit
        ))

        if target in LLAMA6_CELLS:
            sources = _tailored_source_cells(target, LLAMA6_CELLS)
            score, fit = _fit_score(
                spaces["triad"][target],
                [spaces["triad"][cell] for cell in sources],
                sources,
            )
            output["nrm_layer_llama_loco"] = score
            diagnostics["fits"]["nrm_layer_llama_loco"] = fit
            calibration_rows.append(_calibration_row(
                target, "nrm_layer_llama_loco", "triad", fit
            ))
        if target in GSM8K_L32_CELLS:
            sources = _tailored_source_cells(target, GSM8K_L32_CELLS)
            score, fit = _fit_score(
                spaces["triad"][target],
                [spaces["triad"][cell] for cell in sources],
                sources,
            )
            output["nrm_layer_gsm8k_loco"] = score
            diagnostics["fits"]["nrm_layer_gsm8k_loco"] = fit
            calibration_rows.append(_calibration_row(
                target, "nrm_layer_gsm8k_loco", "triad", fit
            ))

        metadata = row_metadata[target]
        score_path = score_dir / f"{target}.npz"
        np.savez_compressed(
            score_path,
            row_ids=metadata["row_ids"],
            problem_ids=metadata["problem_ids"],
            **output,
        )
        diagnostic_path = diagnostic_dir / f"{target}.json"
        write_json(diagnostic_path, diagnostics)
        score_manifest.append({
            "cell": target,
            "score_file": str(score_path.relative_to(results)),
            "score_sha256": sha256_file(score_path),
            "diagnostic_file": str(diagnostic_path.relative_to(results)),
            "diagnostic_sha256": sha256_file(diagnostic_path),
            "n_rows": len(metadata["row_ids"]),
            "score_keys": sorted(output),
        })

    write_csv(results / "calibration_diagnostics.csv", calibration_rows)
    definition = {
        "version": VERSION,
        "written_utc": utcnow(),
        "status": "PRELIMINARY / VALIDATION BLOCKED",
        "analysis_role": "retrospective organic-group addendum; v2 and NRM-v1 remain unchanged",
        "base_result": str(base_results.relative_to(REPO)),
        "base_prepared_manifest_sha256": sha256_file(base_results / "PREPARED_FEATURE_MANIFEST.json"),
        "base_fit_complete_sha256": sha256_file(base_results / "FIT_COMPLETE.json"),
        "base_score_freeze_sha256": sha256_file(base_results / "SCORE_FREEZE_MANIFEST.json"),
        "base_prepared_n_files": prepared["n_files"],
        "base_score_n_cells": len(base_fit["score_manifest"]),
        "exact_layer_count": 32,
        "eligible_cells": list(L32_CELLS),
        "excluded_nonmatching_depth_cells": list(NON_L32_CELLS),
        "cohorts": {
            "all_32layer": list(L32_CELLS),
            "same_model_llama_six": list(LLAMA6_CELLS),
            "same_dataset_gsm8k_32layer": list(GSM8K_L32_CELLS),
        },
        "primary_contract": {
            "groups": "one residual transformer layer per group",
            "within_group_metrics": list(LOCAL_RESID_METRICS),
            "nominal_groups": 32,
            "nominal_features": 96,
        },
        "sensitivity_contract": {
            "within_group_metrics": list(KL_SENSITIVITY_METRICS),
            "nominal_features": 127,
            "reason_secondary": "KL-to-final couples a layer to the final layer and is not strictly local",
        },
        "nrm_rule": {
            "base": "anchor-oriented IU-PCR over atomic within-layer measurements",
            "cell_weighting": "equal covariance contribution per source cell",
            "mode": "eigenvector with eigenvalue closest to 1",
            "orientation": "equal-layer risk direction",
            "trust_ratio": "1 / 32",
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
            "unit": "problem group within cell",
            "tie_tolerance": TIE_TOLERANCE,
        },
        "labels_seen_during_fit": False,
        "source_sha256": {path: sha256_file(REPO / path) for path in FIT_SOURCE_FILES},
    }
    write_json(results / "RUN_DEFINITION.json", definition)
    write_json(results / "FIT_COMPLETE.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "scores_fitted_before_outcomes_opened": True,
        "score_manifest": score_manifest,
    })
    print("[organic-fit] scores frozen; correctness fields remain unopened", flush=True)


def _calibration_row(target: str, method: str, contract: str, fit: Mapping[str, Any]) -> dict[str, Any]:
    calibration = fit["calibration"]
    return {
        "target_cell": target,
        "method": method,
        "contract": contract,
        "source_count": fit["source_count"],
        "source_cells_json": json.dumps(fit["source_cells"]),
        "n_groups": len(calibration["direction"]),
        "selected_eigenvalue": calibration["selected_eigenvalue"],
        "distance_from_unit": calibration["distance_from_unit"],
        "unit_distance_gap": calibration["unit_distance_gap"],
        "direction_json": json.dumps(calibration["direction"]),
        "correction_scale": fit["target_fit"]["correction_scale"],
    }


def _verify_score_freeze(results: Path) -> dict[str, dict[str, np.ndarray]]:
    definition = read_json(results / "RUN_DEFINITION.json")
    for relative, expected in definition["source_sha256"].items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError(f"organic NRM source changed after fitting: {relative}")
    base_results = REPO / definition["base_result"]
    for name, key in (
        ("PREPARED_FEATURE_MANIFEST.json", "base_prepared_manifest_sha256"),
        ("FIT_COMPLETE.json", "base_fit_complete_sha256"),
        ("SCORE_FREEZE_MANIFEST.json", "base_score_freeze_sha256"),
    ):
        if sha256_file(base_results / name) != definition[key]:
            raise RuntimeError(f"base freeze changed after organic fit: {name}")
    fit = read_json(results / "FIT_COMPLETE.json")
    if fit.get("labels_seen_during_fit") is not False:
        raise RuntimeError("organic score freeze does not attest label-free fitting")
    observed: dict[str, dict[str, np.ndarray]] = {}
    for row in fit["score_manifest"]:
        score_path = results / row["score_file"]
        diagnostic_path = results / row["diagnostic_file"]
        if sha256_file(score_path) != row["score_sha256"]:
            raise RuntimeError(f"score hash mismatch: {score_path}")
        if sha256_file(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"diagnostic hash mismatch: {diagnostic_path}")
        diagnostic = read_json(diagnostic_path)
        if diagnostic.get("labels_seen_during_fit") is not False:
            raise RuntimeError(f"diagnostic leakage attestation failed: {row['cell']}")
        with np.load(score_path, allow_pickle=False) as bundle:
            if any("label" in key.lower() or key.lower() in {"y", "target"} for key in bundle.files):
                raise RuntimeError(f"label-like score array in {score_path}")
            observed[row["cell"]] = {key: np.asarray(bundle[key]) for key in bundle.files}
    freeze = {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True,
        "fit_complete_sha256": sha256_file(results / "FIT_COMPLETE.json"),
        "run_definition_sha256": sha256_file(results / "RUN_DEFINITION.json"),
        "score_manifest": fit["score_manifest"],
    }
    path = results / "SCORE_FREEZE_MANIFEST.json"
    if path.exists():
        previous = read_json(path)
        a, b = dict(previous), dict(freeze)
        a.pop("written_utc", None)
        b.pop("written_utc", None)
        if a != b:
            raise RuntimeError("immutable organic score freeze disagrees with current artifacts")
    else:
        write_json(path, freeze)
    return observed


def _available_methods(cell: str, bundle: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    base = _load_base_score_bundle(BASE_RESULTS, cell)
    output = {method: np.asarray(base[key], dtype=float) for method, key in BASE_KEYS.items()}
    for method in METHODS:
        if method in bundle:
            output[method] = np.asarray(bundle[method], dtype=float)
    return output


def phase_evaluate(base_results: Path, cache_root: Path, results: Path) -> None:
    global BASE_RESULTS
    BASE_RESULTS = base_results
    bundles = _verify_score_freeze(results)
    _verify_base_scores(base_results)
    metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    boots: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    per_cell: list[dict[str, Any]] = []
    draw_manifest = {}

    for cell in L32_CELLS:
        print(f"[organic-evaluate] {cell}", flush=True)
        bundle = bundles[cell]
        base = _load_base_score_bundle(base_results, cell)
        if not np.array_equal(bundle["row_ids"].astype(str), base["row_ids"].astype(str)):
            raise RuntimeError(f"{cell}: organic/base row order mismatch")
        row_ids = bundle["row_ids"].astype(str)
        groups = bundle["problem_ids"].astype(str)
        outcomes = load_evaluation_labels(
            load_pickle(cache_root / CELLS[cell]["raw"]), row_ids
        )
        scores = _available_methods(cell, bundle)
        cell_seed = SEED + int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16)
        indices, draw_hash = group_bootstrap_indices(
            groups, draws=BOOTSTRAP_DRAWS, seed=cell_seed
        )
        draw_manifest[cell] = {"seed": cell_seed, "draw_hash": draw_hash}
        for method, score in scores.items():
            auroc, auprc = _metric_pair(outcomes, score)
            metrics[method][cell] = {"auroc": auroc, "auprc": auprc}
            per_cell.append({
                "cell": cell,
                "dataset": CELLS[cell]["dataset"],
                "model": _model(cell),
                "method": method,
                "display_method": METHODS[method],
                "auroc": auroc,
                "auprc": auprc,
                "prevalence": float(np.mean(outcomes)),
                "n_samples": len(outcomes),
                "n_groups": len(np.unique(groups)),
                "n_layers": 32,
                "labels_seen_during_fit": False,
            })
            auc_draws, ap_draws = [], []
            for index in indices:
                auc, ap = _metric_pair(outcomes[index], score[index])
                auc_draws.append(auc)
                ap_draws.append(ap)
            boots[method][cell] = {
                "auroc": np.asarray(auc_draws),
                "auprc": np.asarray(ap_draws),
            }

    cohorts = {
        "all_32layer": L32_CELLS,
        "same_model_llama_six": LLAMA6_CELLS,
        "same_dataset_gsm8k_32layer": GSM8K_L32_CELLS,
    }
    cohort_rows = []
    for cohort, cells in cohorts.items():
        for method in METHODS:
            if not all(cell in metrics[method] for cell in cells):
                continue
            row = {
                "cohort": cohort,
                "n_cells": len(cells),
                "method": method,
                "display_method": METHODS[method],
            }
            for metric in ("auroc", "auprc"):
                point = float(np.mean([metrics[method][cell][metric] for cell in cells]))
                draws = np.mean(np.vstack([boots[method][cell][metric] for cell in cells]), axis=0)
                low, high = np.quantile(draws, (0.025, 0.975))
                row[f"macro_{metric}"] = point
                row[f"macro_{metric}_ci_low"] = float(low)
                row[f"macro_{metric}_ci_high"] = float(high)
            cohort_rows.append(row)

    comparisons = (
        ("organic_lodo_minus_atomic_iu", "nrm_layer_lodo", "iu_layer_triad", "all_32layer", True),
        ("organic_lomo_minus_atomic_iu", "nrm_layer_lomo", "iu_layer_triad", "all_32layer", True),
        ("organic_loco_minus_atomic_iu", "nrm_layer_loco", "iu_layer_triad", "all_32layer", False),
        ("atomic_iu_minus_compressed_iu", "iu_layer_triad", "iu_compressed", "all_32layer", False),
        ("organic_lodo_minus_final_nll", "nrm_layer_lodo", "final_nll", "all_32layer", False),
        ("kl_sensitivity_nrm_minus_atomic_iu", "nrm_layer_kl_lodo", "iu_layer_kl", "all_32layer", False),
        ("llama_only_organic_minus_atomic_iu", "nrm_layer_llama_loco", "iu_layer_triad", "same_model_llama_six", True),
        ("gsm8k_only_organic_minus_atomic_iu", "nrm_layer_gsm8k_loco", "iu_layer_triad", "same_dataset_gsm8k_32layer", True),
    )
    paired = []
    for name, lhs, rhs, cohort, focal in comparisons:
        cells = cohorts[cohort]
        for metric in ("auroc", "auprc"):
            cell_delta = np.asarray([
                metrics[lhs][cell][metric] - metrics[rhs][cell][metric] for cell in cells
            ])
            draw_delta = np.mean(np.vstack([
                boots[lhs][cell][metric] - boots[rhs][cell][metric] for cell in cells
            ]), axis=0)
            low, high = np.quantile(draw_delta, (0.025, 0.975))
            try:
                p_raw = float(wilcoxon(cell_delta, zero_method="pratt").pvalue)
            except ValueError:
                p_raw = 1.0
            paired.append({
                "contrast": name,
                "cohort": cohort,
                "n_cells": len(cells),
                "lhs": lhs,
                "rhs": rhs,
                "metric": metric,
                "delta": float(np.mean(cell_delta)),
                "ci_low": float(low),
                "ci_high": float(high),
                "wins": int(np.sum(cell_delta > TIE_TOLERANCE)),
                "ties": int(np.sum(np.abs(cell_delta) <= TIE_TOLERANCE)),
                "losses": int(np.sum(cell_delta < -TIE_TOLERANCE)),
                "worst_cell_delta": float(np.min(cell_delta)),
                "p_raw": p_raw,
                "p_holm": "",
                "focal": bool(focal and metric == "auroc"),
                "analysis_role": "retrospective_post_v2",
                "per_cell_deltas_json": json.dumps(dict(zip(cells, cell_delta.tolist())), sort_keys=True),
            })
    adjusted = holm_adjust([float(row["p_raw"]) for row in paired])
    for row, adjusted_p in zip(paired, adjusted):
        row["p_holm"] = adjusted_p

    write_csv(results / "per_cell_metrics.csv", per_cell)
    write_csv(results / "cohort_summary.csv", cohort_rows)
    write_csv(results / "paired_comparisons.csv", paired)
    write_json(results / "bootstrap_draw_manifest.json", {
        "draws": BOOTSTRAP_DRAWS,
        "root_seed": SEED,
        "unit": "problem_group_within_cell",
        "identical_draws_reused_across_methods_within_cell": True,
        "cells": draw_manifest,
    })
    base_validation = read_json(base_results / "validation_status.json")
    write_json(results / "validation_status.json", {
        "status": "PRELIMINARY / VALIDATION BLOCKED",
        "base_validation_status": base_validation.get("status"),
        "corrected_live_gate_b_all_pass": bool(base_validation.get("corrected_live_gate_b_all_pass", False)),
        "architecture_pilot_pass": bool(base_validation.get("architecture_pilot_pass", False)),
        "retrospective_post_v2": True,
        "promotion_allowed": False,
        "reason": "capture validation is open and this organic grouping was proposed after v2 evaluation",
    })
    print("[organic-evaluate] metrics complete; score hashes unchanged", flush=True)


def phase_report(results: Path) -> None:
    subprocess.run(
        [sys.executable, str(REPO / REPORT_SOURCE), "--results-dir", str(results)],
        cwd=REPO,
        check=True,
    )


def phase_all(base_results: Path, cache_root: Path, results: Path) -> None:
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--base-results", str(base_results),
        "--cache-root", str(cache_root),
        "--results-dir", str(results),
    ]
    for phase in ("fit", "evaluate", "report"):
        subprocess.run(base + ["--phase", phase], cwd=REPO, check=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fit", "evaluate", "report", "all"), default="all")
    parser.add_argument("--base-results", type=Path, default=BASE_RESULTS)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    base_results = args.base_results.resolve()
    cache_root = args.cache_root.resolve()
    results = args.results_dir.resolve()
    results.mkdir(parents=True, exist_ok=True)
    if args.phase == "fit":
        phase_fit(base_results, results)
    elif args.phase == "evaluate":
        phase_evaluate(base_results, cache_root, results)
    elif args.phase == "report":
        phase_report(results)
    else:
        phase_all(base_results, cache_root, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
