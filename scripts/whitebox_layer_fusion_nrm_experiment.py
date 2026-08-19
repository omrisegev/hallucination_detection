#!/usr/bin/env python3
"""Retrospective NRM-CS-IU addendum for the frozen white-box benchmark.

The addendum deliberately reuses the label-free prepared feature bundles from
``whitebox_layer_fusion_v2`` and never mutates that registered result.  ``fit``
constructs cross-cell Neutral Residual Mode directions and freezes scores.
Only ``evaluate`` opens raw correctness fields.  All comparisons remain
PRELIMINARY / VALIDATION BLOCKED because the capture-fidelity gates are still
open and because the NRM hypothesis was proposed after the v2 labels/results
were historically visible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.whitebox_layer_fusion_experiment import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    CELLS,
    GSM8K_ARCHITECTURE_CELLS,
    MODEL,
    ORIGINAL_LLAMA_CELLS,
    PRIMARY_CELLS,
    SEED,
    TIE_TOLERANCE,
    group_bootstrap_indices,
    holm_adjust,
    load_feature_matrix,
)
from spectral_utils.whitebox_layer_fusion import (  # noqa: E402
    FeatureMatrix,
    apply_neutral_residual_calibration,
    fit_group_contribution_space,
    fit_neutral_residual_calibration,
    load_evaluation_labels,
)


VERSION = "whitebox-layer-fusion-nrm-v1-2026-08-13"
BASE_RESULTS = REPO / "results" / "whitebox_layer_fusion_v2"
DEFAULT_RESULTS = REPO / "results" / "whitebox_layer_fusion_nrm_v1"
DEFAULT_CACHE = REPO / "dataset_cache" / "whitebox_layer_fusion_v1"
CORE_SOURCE = "spectral_utils/whitebox_layer_fusion.py"
RUNNER_SOURCE = "scripts/whitebox_layer_fusion_nrm_experiment.py"
REPORT_SOURCE = "scripts/whitebox_layer_fusion_nrm_report.py"
FIT_SOURCE_FILES = (
    CORE_SOURCE,
    RUNNER_SOURCE,
    "scripts/whitebox_layer_fusion_experiment.py",
    "spectral_utils/paper_benchmark_suite.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
)

METHODS = {
    "final_nll": {
        "display": "Final-layer NLL",
        "base_key": "final_layer_nll__resid-core-L__all__flat",
        "contract": "resid-core-L",
    },
    "iu_resid": {
        "display": "IU-PCR · residual core",
        "base_key": "iu_pcr__resid-core-L__all__flat",
        "contract": "resid-core-L",
    },
    "dufs_resid": {
        "display": "DUFS-LIU · residual core",
        "base_key": "dufs_liu_pcr__resid-core-L__all__flat",
        "contract": "resid-core-L",
    },
    "iu_lens96": {
        "display": "IU-PCR · lens-96",
        "base_key": "iu_pcr__lens-96__spaced8__flat",
        "contract": "lens-96",
    },
    "dufs_lens96": {
        "display": "DUFS-LIU · lens-96",
        "base_key": "dufs_liu_pcr__lens-96__spaced8__flat",
        "contract": "lens-96",
    },
    "nrm_depth_lodo": {
        "display": "Depth-NRM-CS-IU · leave-dataset-out",
        "score_key": "nrm_depth_lodo",
        "contract": "resid-core-L",
    },
    "nrm_depth_lomo": {
        "display": "Depth-NRM-CS-IU · leave-model-out",
        "score_key": "nrm_depth_lomo",
        "contract": "resid-core-L",
    },
    "nrm_depth_loco": {
        "display": "Depth-NRM-CS-IU · LOCO sensitivity",
        "score_key": "nrm_depth_loco",
        "contract": "resid-core-L",
    },
    "nrm_lens_lodo": {
        "display": "Lens-NRM-CS-IU · leave-dataset-out",
        "score_key": "nrm_lens_lodo",
        "contract": "lens-96",
    },
    "nrm_lens_lomo": {
        "display": "Lens-NRM-CS-IU · leave-model-out",
        "score_key": "nrm_lens_lomo",
        "contract": "lens-96",
    },
    "nrm_lens_loco": {
        "display": "Lens-NRM-CS-IU · LOCO sensitivity",
        "score_key": "nrm_lens_loco",
        "contract": "lens-96",
    },
}

CONTRACTS = {
    "depth": {"prepared": "resid_core_all", "family_basis": "relative-depth-quartiles"},
    "lens": {"prepared": "lens96", "family_basis": "module-by-metric"},
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _verify_base_prepared(base_results: Path) -> dict[str, Any]:
    manifest = read_json(base_results / "PREPARED_FEATURE_MANIFEST.json")
    files = manifest.get("files", [])
    if manifest.get("labels_present") is not False or manifest.get("n_files") != len(files):
        raise RuntimeError("base prepared manifest is not a complete label-free freeze")
    for row in files:
        path = base_results / row["file"]
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise RuntimeError(f"base prepared artifact missing or resized: {path}")
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"base prepared artifact hash mismatch: {path}")
        forbidden = [
            field for field in row.get("fields", [])
            if "label" in str(field).lower() or str(field).lower() in {"y", "target"}
        ]
        if forbidden:
            raise RuntimeError(f"label-like fields in base prepared artifact: {path}")
    return manifest


def _verify_base_scores(base_results: Path) -> dict[str, Any]:
    fit = read_json(base_results / "FIT_COMPLETE.json")
    if fit.get("labels_seen_during_fit") is not False:
        raise RuntimeError("base score freeze does not attest label-free fitting")
    if [row.get("cell") for row in fit.get("score_manifest", [])] != list(CELLS):
        raise RuntimeError("base score roster does not match the 14-cell benchmark")
    for row in fit["score_manifest"]:
        score_path = base_results / row["score_file"]
        diagnostic_path = base_results / row["diagnostic_file"]
        if sha256_file(score_path) != row["score_sha256"]:
            raise RuntimeError(f"base score hash mismatch: {score_path}")
        if sha256_file(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"base diagnostic hash mismatch: {diagnostic_path}")
    return fit


def _relative_depth_families(matrix: FeatureMatrix) -> tuple[str, ...]:
    order = list(dict.fromkeys(matrix.groups))
    if len(order) != 4:
        raise ValueError(f"depth NRM requires four ordered bands, got {order}")
    aliases = {name: f"depth_band_{index}" for index, name in enumerate(order)}
    return tuple(aliases[name] for name in matrix.groups)


def _source_cells(target: str, strategy: str) -> tuple[str, ...]:
    if strategy == "loco":
        return tuple(cell for cell in PRIMARY_CELLS if cell != target)
    target_spec = CELLS[target]
    if strategy == "lodo":
        sources = tuple(
            cell for cell in PRIMARY_CELLS
            if CELLS[cell]["dataset"] != target_spec["dataset"]
        )
    elif strategy == "lomo":
        sources = tuple(
            cell for cell in PRIMARY_CELLS
            if CELLS[cell].get("model", MODEL) != target_spec.get("model", MODEL)
        )
    else:
        raise ValueError("strategy must be lodo, lomo, or loco")
    if len(sources) < 3:
        raise RuntimeError(f"{target}/{strategy}: fewer than three NRM source cells")
    return sources


def _load_base_score_bundle(base_results: Path, cell: str) -> dict[str, np.ndarray]:
    with np.load(base_results / "scores" / f"{cell}.npz", allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def phase_fit(base_results: Path, results: Path) -> None:
    prepared_manifest = _verify_base_prepared(base_results)
    base_fit = _verify_base_scores(base_results)
    spaces: dict[str, dict[str, Any]] = defaultdict(dict)
    matrices: dict[str, dict[str, FeatureMatrix]] = defaultdict(dict)
    row_metadata: dict[str, dict[str, np.ndarray]] = {}

    for cell in CELLS:
        print(f"[nrm-fit] contribution spaces: {cell}", flush=True)
        row_path = base_results / "prepared" / f"{cell}__rows.npz"
        with np.load(row_path, allow_pickle=False) as bundle:
            row_metadata[cell] = {
                "row_ids": bundle["row_ids"].astype(str),
                "problem_ids": bundle["problem_ids"].astype(str),
            }
        for contract, spec in CONTRACTS.items():
            matrix = load_feature_matrix(
                base_results / "prepared" / f"{cell}__{spec['prepared']}.npz"
            )
            family_names = (
                _relative_depth_families(matrix) if contract == "depth" else matrix.groups
            )
            matrices[contract][cell] = matrix
            spaces[contract][cell] = fit_group_contribution_space(
                matrix, family_names=family_names
            )

        base = _load_base_score_bundle(base_results, cell)
        expected = {
            "depth": base[METHODS["iu_resid"]["base_key"]],
            "lens": base[METHODS["iu_lens96"]["base_key"]],
        }
        for contract in CONTRACTS:
            error = float(np.max(np.abs(
                spaces[contract][cell].baseline_score - expected[contract]
            )))
            if error > 1e-10:
                raise RuntimeError(
                    f"{cell}/{contract}: contribution IU disagrees with frozen v2 by {error:.3e}"
                )

    score_dir = results / "scores"
    diagnostic_dir = results / "diagnostics"
    score_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    calibration_rows = []
    for target in CELLS:
        output: dict[str, np.ndarray] = {}
        target_diagnostics: dict[str, Any] = {
            "labels_seen_during_fit": False,
            "scores_fitted_before_outcomes_opened": True,
            "target": target,
            "fits": {},
        }
        for contract in CONTRACTS:
            for strategy in ("lodo", "lomo", "loco"):
                sources = _source_cells(target, strategy)
                calibration = fit_neutral_residual_calibration(
                    [spaces[contract][cell] for cell in sources],
                    source_ids=sources,
                )
                fitted = apply_neutral_residual_calibration(
                    spaces[contract][target], calibration
                )
                method = f"nrm_{contract}_{strategy}"
                output[method] = fitted.score
                fit_key = f"{contract}_{strategy}"
                target_diagnostics["fits"][fit_key] = {
                    "family_basis": CONTRACTS[contract]["family_basis"],
                    "families": list(calibration.families),
                    "source_cells": list(sources),
                    "source_count": len(sources),
                    "calibration": {
                        **dict(calibration.diagnostics),
                        "direction": calibration.direction.tolist(),
                        "eigenvalues": calibration.eigenvalues.tolist(),
                        "residual_covariance": calibration.residual_covariance.tolist(),
                    },
                    "target_fit": dict(fitted.diagnostics),
                    "iu_reconstruction_error": spaces[contract][target].diagnostics[
                        "reconstruction_error"
                    ],
                }
                calibration_rows.append({
                    "target_cell": target,
                    "contract": contract,
                    "strategy": strategy,
                    "family_basis": CONTRACTS[contract]["family_basis"],
                    "source_count": len(sources),
                    "source_cells_json": json.dumps(sources),
                    "selected_eigenvalue": calibration.diagnostics["selected_eigenvalue"],
                    "distance_from_unit": calibration.diagnostics["distance_from_unit"],
                    "unit_distance_gap": calibration.diagnostics["unit_distance_gap"],
                    "direction_json": json.dumps(calibration.direction.tolist()),
                    "correction_scale": fitted.diagnostics["correction_scale"],
                    "baseline_correction_covariance": fitted.diagnostics[
                        "baseline_correction_covariance"
                    ],
                })

        metadata = row_metadata[target]
        score_path = score_dir / f"{target}.npz"
        np.savez_compressed(
            score_path,
            row_ids=metadata["row_ids"],
            problem_ids=metadata["problem_ids"],
            **output,
        )
        diagnostic_path = diagnostic_dir / f"{target}.json"
        write_json(diagnostic_path, target_diagnostics)
        manifest.append({
            "cell": target,
            "score_file": str(score_path.relative_to(results)),
            "score_sha256": sha256_file(score_path),
            "diagnostic_file": str(diagnostic_path.relative_to(results)),
            "diagnostic_sha256": sha256_file(diagnostic_path),
            "n_rows": len(metadata["row_ids"]),
            "n_methods": len(output),
        })

    write_csv(results / "calibration_diagnostics.csv", calibration_rows)
    source_hashes = {path: sha256_file(REPO / path) for path in FIT_SOURCE_FILES}
    run_definition = {
        "version": VERSION,
        "written_utc": utcnow(),
        "status": "PRELIMINARY / VALIDATION BLOCKED",
        "analysis_role": "retrospective post-v2 addendum; no replacement of registered v2 primary",
        "base_result": str(base_results.relative_to(REPO)),
        "base_prepared_manifest_sha256": sha256_file(
            base_results / "PREPARED_FEATURE_MANIFEST.json"
        ),
        "base_fit_complete_sha256": sha256_file(base_results / "FIT_COMPLETE.json"),
        "base_score_freeze_sha256": sha256_file(
            base_results / "SCORE_FREEZE_MANIFEST.json"
        ),
        "base_prepared_n_files": prepared_manifest["n_files"],
        "base_score_n_cells": len(base_fit["score_manifest"]),
        "eligible_cells": list(PRIMARY_CELLS),
        "appendix_rejected_cells": [
            cell for cell in CELLS if cell not in PRIMARY_CELLS
        ],
        "contracts": CONTRACTS,
        "strategies": {
            "lodo": "exclude every source sharing the target dataset",
            "lomo": "exclude every source sharing the target exact model",
            "loco": "exclude target cell only; sensitivity analysis",
        },
        "nrm_rule": {
            "base": "anchor-oriented IU-PCR",
            "cell_weighting": "equal covariance contribution per source cell",
            "mode": "eigenvector with eigenvalue closest to 1",
            "orientation": "equal-family risk direction",
            "trust_ratio": "1 / number_of_families",
            "target_transform": "unlabeled whole-cell standardization and IU residualization",
        },
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
            "unit": "problem group within cell",
            "tie_tolerance": TIE_TOLERANCE,
        },
        "labels_seen_during_fit": False,
        "source_sha256": source_hashes,
    }
    write_json(results / "RUN_DEFINITION.json", run_definition)
    write_json(results / "FIT_COMPLETE.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "scores_fitted_before_outcomes_opened": True,
        "score_manifest": manifest,
    })
    print("[nrm-fit] scores frozen; correctness fields remain unopened", flush=True)


def _verify_nrm_freeze(results: Path) -> dict[str, dict[str, np.ndarray]]:
    definition = read_json(results / "RUN_DEFINITION.json")
    for relative, expected in definition["source_sha256"].items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError(f"NRM source changed after fitting: {relative}")
    base_results = REPO / definition["base_result"]
    expected_base = {
        "PREPARED_FEATURE_MANIFEST.json": definition["base_prepared_manifest_sha256"],
        "FIT_COMPLETE.json": definition["base_fit_complete_sha256"],
        "SCORE_FREEZE_MANIFEST.json": definition["base_score_freeze_sha256"],
    }
    for name, expected in expected_base.items():
        if sha256_file(base_results / name) != expected:
            raise RuntimeError(f"base freeze changed after NRM fit: {name}")
    fit = read_json(results / "FIT_COMPLETE.json")
    if fit.get("labels_seen_during_fit") is not False:
        raise RuntimeError("NRM fit leakage attestation failed")
    observed: dict[str, dict[str, np.ndarray]] = {}
    for row in fit["score_manifest"]:
        score_path = results / row["score_file"]
        diagnostic_path = results / row["diagnostic_file"]
        if sha256_file(score_path) != row["score_sha256"]:
            raise RuntimeError(f"NRM score hash mismatch: {score_path}")
        if sha256_file(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"NRM diagnostic hash mismatch: {diagnostic_path}")
        diagnostic = read_json(diagnostic_path)
        if diagnostic.get("labels_seen_during_fit") is not False:
            raise RuntimeError(f"NRM diagnostic leakage attestation failed: {row['cell']}")
        with np.load(score_path, allow_pickle=False) as bundle:
            forbidden = [
                key for key in bundle.files
                if "label" in key.lower() or key.lower() in {"y", "target", "targets"}
            ]
            if forbidden:
                raise RuntimeError(f"label-like NRM score arrays: {forbidden}")
            observed[row["cell"]] = {
                key: np.asarray(bundle[key]) for key in bundle.files
            }
    freeze = {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True,
        "fit_complete_sha256": sha256_file(results / "FIT_COMPLETE.json"),
        "run_definition_sha256": sha256_file(results / "RUN_DEFINITION.json"),
        "score_manifest": fit["score_manifest"],
    }
    freeze_path = results / "SCORE_FREEZE_MANIFEST.json"
    if freeze_path.exists():
        previous = read_json(freeze_path)
        comparable_previous = dict(previous)
        comparable_current = dict(freeze)
        comparable_previous.pop("written_utc", None)
        comparable_current.pop("written_utc", None)
        if comparable_previous != comparable_current:
            raise RuntimeError("immutable NRM score freeze disagrees with current artifacts")
    else:
        write_json(freeze_path, freeze)
    return observed


def _metric_pair(outcomes: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(outcomes)) != 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(outcomes, score)),
        float(average_precision_score(outcomes, score)),
    )


def phase_evaluate(
    base_results: Path, cache_root: Path, results: Path
) -> None:
    nrm_bundles = _verify_nrm_freeze(results)
    _verify_base_scores(base_results)
    metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    boot: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    per_cell = []
    draw_manifest = {}

    for cell, spec in CELLS.items():
        print(f"[nrm-evaluate] {cell}", flush=True)
        nrm = nrm_bundles[cell]
        base = _load_base_score_bundle(base_results, cell)
        if not np.array_equal(nrm["row_ids"].astype(str), base["row_ids"].astype(str)):
            raise RuntimeError(f"{cell}: NRM/base row order mismatch")
        row_ids = nrm["row_ids"].astype(str)
        groups = nrm["problem_ids"].astype(str)
        raw = load_pickle(cache_root / spec["raw"])
        outcomes = load_evaluation_labels(raw, row_ids)
        prevalence = float(np.mean(outcomes))
        scores: dict[str, np.ndarray] = {}
        for method, meta in METHODS.items():
            key = meta.get("score_key")
            scores[method] = np.asarray(
                nrm[key] if key is not None else base[meta["base_key"]], dtype=float
            )
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
                "dataset": spec["dataset"],
                "model": spec.get("model", MODEL),
                "method": method,
                "display_method": METHODS[method]["display"],
                "feature_contract": METHODS[method]["contract"],
                "auroc": auroc,
                "auprc": auprc,
                "prevalence": prevalence,
                "n_samples": len(outcomes),
                "n_groups": len(np.unique(groups)),
                "status": (
                    "eligible_retrospective" if cell in PRIMARY_CELLS
                    else "appendix_protocol_rejected"
                ),
                "labels_seen_during_fit": False,
            })
            auc_draws, ap_draws = [], []
            for index in indices:
                auc, ap = _metric_pair(outcomes[index], score[index])
                auc_draws.append(auc)
                ap_draws.append(ap)
            boot[method][cell] = {
                "auroc": np.asarray(auc_draws),
                "auprc": np.asarray(ap_draws),
            }

    headline = []
    for method in METHODS:
        row = {
            "method": method,
            "display_method": METHODS[method]["display"],
            "feature_contract": METHODS[method]["contract"],
            "n_cells": len(PRIMARY_CELLS),
        }
        for metric in ("auroc", "auprc"):
            point = float(np.mean([
                metrics[method][cell][metric] for cell in PRIMARY_CELLS
            ]))
            draws = np.mean(np.vstack([
                boot[method][cell][metric] for cell in PRIMARY_CELLS
            ]), axis=0)
            low, high = np.quantile(draws, (0.025, 0.975))
            row[f"macro_{metric}"] = point
            row[f"macro_{metric}_ci_low"] = float(low)
            row[f"macro_{metric}_ci_high"] = float(high)
        headline.append(row)

    contrasts = [
        ("depth_lodo_minus_iu", "nrm_depth_lodo", "iu_resid", True),
        ("depth_lodo_minus_dufs", "nrm_depth_lodo", "dufs_resid", False),
        ("depth_lodo_minus_final_nll", "nrm_depth_lodo", "final_nll", False),
        ("depth_lomo_minus_iu", "nrm_depth_lomo", "iu_resid", False),
        ("depth_loco_minus_iu", "nrm_depth_loco", "iu_resid", False),
        ("lens_lodo_minus_iu", "nrm_lens_lodo", "iu_lens96", True),
        ("lens_lodo_minus_dufs", "nrm_lens_lodo", "dufs_lens96", False),
        ("lens_lodo_minus_final_nll", "nrm_lens_lodo", "final_nll", False),
        ("lens_lomo_minus_iu", "nrm_lens_lomo", "iu_lens96", False),
        ("lens_loco_minus_iu", "nrm_lens_loco", "iu_lens96", False),
    ]
    paired = []
    for name, lhs, rhs, focal in contrasts:
        for metric in ("auroc", "auprc"):
            cell_delta = np.asarray([
                metrics[lhs][cell][metric] - metrics[rhs][cell][metric]
                for cell in PRIMARY_CELLS
            ])
            draw_delta = np.mean(np.vstack([
                boot[lhs][cell][metric] - boot[rhs][cell][metric]
                for cell in PRIMARY_CELLS
            ]), axis=0)
            low, high = np.quantile(draw_delta, (0.025, 0.975))
            try:
                p_raw = float(wilcoxon(cell_delta, zero_method="pratt").pvalue)
            except ValueError:
                p_raw = 1.0
            paired.append({
                "contrast": name,
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
                "focal_addendum_contrast": bool(focal and metric == "auroc"),
                "analysis_role": "retrospective_post_v2",
                "per_cell_deltas_json": json.dumps(
                    dict(zip(PRIMARY_CELLS, cell_delta.tolist())), sort_keys=True
                ),
            })
    adjusted = holm_adjust([float(row["p_raw"]) for row in paired])
    for row, value in zip(paired, adjusted):
        row["p_holm"] = value

    cohorts = {
        "primary_13_cells": PRIMARY_CELLS,
        "original_llama_six": ORIGINAL_LLAMA_CELLS,
        "gsm8k_cross_architecture_seven": GSM8K_ARCHITECTURE_CELLS,
    }
    cohort_rows = []
    for cohort, cells in cohorts.items():
        for method in METHODS:
            cohort_rows.append({
                "cohort": cohort,
                "n_cells": len(cells),
                "method": method,
                "display_method": METHODS[method]["display"],
                "macro_auroc": float(np.mean([
                    metrics[method][cell]["auroc"] for cell in cells
                ])),
                "macro_auprc": float(np.mean([
                    metrics[method][cell]["auprc"] for cell in cells
                ])),
            })

    write_csv(results / "per_cell_metrics.csv", per_cell)
    write_csv(results / "headline_summary.csv", headline)
    write_csv(results / "paired_comparisons.csv", paired)
    write_csv(results / "cohort_summary.csv", cohort_rows)
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
        "corrected_live_gate_b_all_pass": bool(
            base_validation.get("corrected_live_gate_b_all_pass", False)
        ),
        "architecture_pilot_pass": bool(
            base_validation.get("architecture_pilot_pass", False)
        ),
        "retrospective_post_v2": True,
        "promotion_allowed": False,
        "reason": (
            "capture validation is blocked and NRM was proposed after v2 outcomes "
            "were historically available"
        ),
    })
    print("[nrm-evaluate] metrics complete; frozen score hashes unchanged", flush=True)


def phase_report(results: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(REPO / REPORT_SOURCE),
            "--results-dir",
            str(results),
        ],
        cwd=REPO,
        check=True,
    )


def phase_all(base_results: Path, cache_root: Path, results: Path) -> None:
    base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--base-results",
        str(base_results),
        "--cache-root",
        str(cache_root),
        "--results-dir",
        str(results),
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
