#!/usr/bin/env python3
"""Retrospective search for depth-distributed metrics suited to label-free fusion.

The pipeline is deliberately phase separated:

``prepare``
    Reopens raw/sidecar caches only to validate their join and writes four
    frozen, label-free metric matrices.
``fit``
    Can open only the prepared matrices.  It fits equal-mean, IU-PCR,
    DUFS-LIU-PCR, and a layer-organic hierarchical IU-PCR arm.
``evaluate``
    Verifies score hashes before opening correctness labels.  It adds
    evaluation-only best-layer and supervised-probe ceilings.

All findings are retrospective because the v2 outcomes were historically
available before this metric registry was proposed.  A win here is a discovery
candidate, not independent confirmation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.whitebox_layer_fusion_experiment import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    CELLS,
    DEFAULT_CACHE,
    MODEL,
    PRIMARY_CELLS,
    SEED,
    TIE_TOLERANCE,
    group_bootstrap_indices,
    holm_adjust,
    jsonable,
    load_evaluation_labels,
    load_feature_matrix,
    save_feature_matrix,
    sha256_file,
    validate_and_join,
)
from spectral_utils.paper_benchmark_suite import standardize  # noqa: E402
from spectral_utils.whitebox_depth_metrics import (  # noqa: E402
    DEPTH_METRIC_REGISTRY,
    EXTRACTORS,
    registry_hash,
)
from spectral_utils.whitebox_layer_fusion import (  # noqa: E402
    FeatureMatrix,
    fit_controls,
    fit_core_spectral,
    fit_hierarchical,
)


VERSION = "whitebox-depth-metric-search-v1-2026-08-13"
DEFAULT_BASE_RESULTS = REPO / "results" / "whitebox_layer_fusion_v2"
DEFAULT_RESULTS = REPO / "results" / "whitebox_depth_metric_search_v1"
SOURCE_FILES = (
    "scripts/whitebox_depth_metric_search.py",
    "spectral_utils/whitebox_depth_metrics.py",
    "spectral_utils/whitebox_layer_fusion.py",
    "spectral_utils/paper_benchmark_suite.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
)
BASELINE_KEYS = {
    "final_nll": "final_layer_nll__resid-core-L__all__flat",
    "generation_entropy": "generation_entropy_mean__raw-output__full-answer__flat",
    "trilens_dufs": "dufs_liu_pcr__trilens-entropy-3L__all__flat",
    "lens96_dufs": "dufs_liu_pcr__lens-96__spaced8__flat",
}


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(value), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def _string_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray(tuple(str(value) for value in values), dtype="U")


def _rows_path(results: Path, cell: str) -> Path:
    return results / "prepared" / f"{cell}__rows.npz"


def _matrix_path(results: Path, cell: str, contract: str) -> Path:
    return results / "prepared" / f"{cell}__{contract}.npz"


def _score_path(results: Path, cell: str) -> Path:
    return results / "scores" / f"{cell}.npz"


def _matrix_hash(matrix: FeatureMatrix) -> str:
    X, keep, mean, scale = standardize(matrix.values)
    digest = hashlib.sha256()
    for value in (X, keep, mean, scale, matrix.risk_anchor):
        digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def _structural_diagnostics(cell: str, contract: str, matrix: FeatureMatrix) -> dict[str, Any]:
    X, keep, _mean, _scale = standardize(matrix.values)
    correlation = np.corrcoef(X, rowvar=False)
    upper = np.abs(correlation[np.triu_indices(correlation.shape[0], 1)])
    covariance_eigenvalues = np.linalg.eigvalsh((X.T @ X) / max(len(X), 1))
    covariance_eigenvalues = np.maximum(covariance_eigenvalues, 0.0)
    probability = covariance_eigenvalues / max(float(np.sum(covariance_eigenvalues)), 1e-12)
    entropy = float(-np.sum(probability * np.log(probability + 1e-12)))
    kept_groups = [matrix.groups[int(index)] for index in keep]
    group_sizes = [kept_groups.count(group) for group in dict.fromkeys(kept_groups)]
    return {
        "cell": cell,
        "contract": contract,
        "n_samples": matrix.n_samples,
        "n_features_raw": matrix.n_features,
        "n_features_kept": len(keep),
        "n_layer_groups": len(set(kept_groups)),
        "median_features_per_layer_group": float(np.median(group_sizes)),
        "median_abs_feature_correlation": float(np.median(upper)) if len(upper) else 0.0,
        "p90_abs_feature_correlation": float(np.quantile(upper, 0.9)) if len(upper) else 0.0,
        "effective_rank": float(np.exp(entropy)),
        "effective_rank_fraction": float(np.exp(entropy) / len(keep)),
        "standardized_matrix_sha256": _matrix_hash(matrix),
        "outcomes_seen": False,
    }


def _base_source_index(base_results: Path) -> dict[str, Mapping[str, Any]]:
    manifest = read_json(base_results / "SOURCE_FREEZE_MANIFEST.json")
    return {str(row["local_path"]): row for row in manifest["sources"]}


def phase_prepare(cache_root: Path, base_results: Path, results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    prepared = results / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    source_index = _base_source_index(base_results)
    source_rows = []
    structural_rows = []
    prepared_manifest = []

    registration = {
        "version": VERSION,
        "written_utc": utcnow(),
        "analysis_role": "retrospective_discovery_screen",
        "metric_registry": DEPTH_METRIC_REGISTRY,
        "metric_registry_sha256": registry_hash(),
        "fusion_methods": ["equal_mean", "iu_pcr", "dufs_liu_pcr", "hierarchical_iu_pcr"],
        "primary_fusion_method": "iu_pcr",
        "secondary_fusion_method": "dufs_liu_pcr",
        "layer_organic_rule": "features sharing the same transformer layer form one fixed group",
        "success_rule": (
            "a discovery candidate must beat its evaluation-only per-cell best-layer oracle "
            "and the existing TriLens grouped-L2-probe macro AUROC; independent confirmation "
            "is still mandatory"
        ),
        "orientation": "fixed risk signs, globally oriented only by final-layer target-token NLL",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": SEED,
        "validation_status": "PRELIMINARY / VALIDATION BLOCKED",
    }
    write_json(results / "PREREGISTRATION.json", registration)

    for cell, spec in CELLS.items():
        print(f"[depth-prepare] {cell}", flush=True)
        raw_path = cache_root / spec["raw"]
        sidecar_path = cache_root / spec["sidecar"]
        for path in (raw_path, sidecar_path):
            frozen = source_index.get(str(path.resolve()))
            if frozen is None:
                raise RuntimeError(f"{path}: absent from the audited v2 source freeze")
            if path.stat().st_size != int(frozen["local_size"]):
                raise RuntimeError(f"{path}: size changed from the v2 source freeze")
            source_rows.append({
                "cell": cell,
                "local_path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": frozen["local_sha256"],
                "verified_by": "whitebox_layer_fusion_v2/SOURCE_FREEZE_MANIFEST.json",
            })
        raw = load_pickle(raw_path)
        sidecar = load_pickle(sidecar_path)
        joined, audit = validate_and_join(
            raw,
            sidecar,
            cell_id=cell,
            expected_model=spec.get("model", MODEL),
            expected_n_layers=int(spec.get("n_layers", 32)),
            expected_hidden_size=int(spec.get("hidden_size", 4096)),
            exclude_invalid=True,
            require_geometry_finite=False,
        )
        if audit["nonfinite_geometry_counts"]["hid_proj"] != 0:
            raise RuntimeError(f"{cell}: hidden projection is not finite")
        matrices = {name: extractor(joined) for name, extractor in EXTRACTORS.items()}
        for contract, matrix in matrices.items():
            path = _matrix_path(results, cell, contract)
            save_feature_matrix(path, matrix)
            structural_rows.append(_structural_diagnostics(cell, contract, matrix))
        row_path = _rows_path(results, cell)
        np.savez_compressed(
            row_path,
            row_ids=_string_array(joined.row_ids),
            problem_ids=_string_array(joined.problem_ids),
            n_gen_tokens=np.asarray(joined.n_gen_tokens, dtype=np.int64),
            protocol_signature=np.asarray(joined.protocol_signature),
        )
        del raw, sidecar, joined, matrices

    for path in sorted(prepared.glob("*.npz")):
        with np.load(path, allow_pickle=False) as bundle:
            fields = list(bundle.files)
            forbidden = [field for field in fields if "label" in field.lower() or field.lower() == "y"]
            if forbidden:
                raise RuntimeError(f"label-like fields in {path}: {forbidden}")
        prepared_manifest.append({
            "file": str(path.relative_to(results)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "fields": fields,
        })
    write_json(results / "SOURCE_FREEZE_MANIFEST.json", {
        "version": VERSION,
        "parent_manifest": str((base_results / "SOURCE_FREEZE_MANIFEST.json").resolve()),
        "parent_manifest_sha256": sha256_file(base_results / "SOURCE_FREEZE_MANIFEST.json"),
        "sources": source_rows,
    })
    write_json(results / "PREPARED_FEATURE_MANIFEST.json", {
        "version": VERSION,
        "labels_present": False,
        "n_files": len(prepared_manifest),
        "files": prepared_manifest,
    })
    write_json(results / "RUN_DEFINITION.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "cells": list(CELLS),
        "primary_cells": list(PRIMARY_CELLS),
        "contracts": list(EXTRACTORS),
        "metric_registry_sha256": registry_hash(),
        "source_sha256": {path: sha256_file(REPO / path) for path in SOURCE_FILES},
        "base_results": str(base_results.resolve()),
        "validation_status": "PRELIMINARY / VALIDATION BLOCKED",
    })
    write_csv(results / "structural_diagnostics.csv", structural_rows)
    print("[depth-prepare] label-free matrices frozen", flush=True)


def verify_prepared(results: Path) -> None:
    definition = read_json(results / "RUN_DEFINITION.json")
    if definition.get("metric_registry_sha256") != registry_hash():
        raise RuntimeError("metric registry changed after preparation")
    for relative, expected in definition["source_sha256"].items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError(f"registered source changed: {relative}")
    manifest = read_json(results / "PREPARED_FEATURE_MANIFEST.json")
    if manifest.get("labels_present") is not False:
        raise RuntimeError("prepared manifest does not attest labels_present=false")
    observed = [str(path.relative_to(results)) for path in sorted((results / "prepared").glob("*.npz"))]
    if observed != [row["file"] for row in manifest["files"]]:
        raise RuntimeError("prepared roster changed")
    for row in manifest["files"]:
        path = results / row["file"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"prepared file changed: {path}")


def _fit_cell(results: Path, cell: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    scores: dict[str, np.ndarray] = {}
    diagnostics: dict[str, Any] = {
        "cell": cell,
        "labels_seen_during_fit": False,
        "scores_fitted_before_outcomes_opened": True,
        "contracts": {},
    }
    for contract in EXTRACTORS:
        matrix = load_feature_matrix(_matrix_path(results, cell, contract))
        controls, control_diag = fit_controls(matrix)
        core, core_diag = fit_core_spectral(
            matrix, methods=("iu_pcr", "dufs_liu_pcr")
        )
        contract_scores = {
            "equal_mean": controls["equal_mean"],
            "iu_pcr": core["iu_pcr"],
            "dufs_liu_pcr": core["dufs_liu_pcr"],
        }
        hierarchical_diag = None
        if len(set(matrix.groups)) < matrix.n_features:
            hierarchical, hierarchical_diag = fit_hierarchical(matrix, "iu_pcr")
            contract_scores["hierarchical_iu_pcr"] = hierarchical
        for method, score in contract_scores.items():
            scores[f"{contract}__{method}"] = np.asarray(score, dtype=float)
        kept_control = control_diag["kept_column_indices"]
        kept_core = core_diag.get("keep", core_diag.get("kept_column_indices"))
        # Canonical fit diagnostics call this field ``keep``; our wrapper may
        # expose the more explicit alias.  Both must describe the same matrix.
        if kept_core is not None and list(kept_control) != list(kept_core):
            raise RuntimeError(f"{cell}/{contract}: methods saw different standardized columns")
        if hierarchical_diag is not None and list(kept_control) != list(hierarchical_diag["kept_column_indices"]):
            raise RuntimeError(f"{cell}/{contract}: hierarchy saw a different standardized matrix")
        diagnostics["contracts"][contract] = {
            "standardized_matrix_sha256": _matrix_hash(matrix),
            "protocol_signature": matrix.protocol_signature,
            "n_features": matrix.n_features,
            "n_groups": len(set(matrix.groups)),
            "controls": control_diag,
            "core": core_diag,
            "hierarchical_iu_pcr": hierarchical_diag,
        }
    return scores, diagnostics


def phase_fit(results: Path) -> None:
    verify_prepared(results)
    score_dir = results / "scores"
    diagnostic_dir = results / "diagnostics"
    score_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for cell in CELLS:
        print(f"[depth-fit] {cell}", flush=True)
        with np.load(_rows_path(results, cell), allow_pickle=False) as bundle:
            metadata = {key: np.asarray(bundle[key]) for key in bundle.files}
        scores, diagnostics = _fit_cell(results, cell)
        score_path = _score_path(results, cell)
        np.savez_compressed(
            score_path,
            row_ids=metadata["row_ids"],
            problem_ids=metadata["problem_ids"],
            n_gen_tokens=metadata["n_gen_tokens"],
            protocol_signature=metadata["protocol_signature"],
            **scores,
        )
        diagnostic_path = diagnostic_dir / f"{cell}.json"
        write_json(diagnostic_path, diagnostics)
        manifest.append({
            "cell": cell,
            "score_file": str(score_path.relative_to(results)),
            "score_sha256": sha256_file(score_path),
            "diagnostic_file": str(diagnostic_path.relative_to(results)),
            "diagnostic_sha256": sha256_file(diagnostic_path),
            "n_rows": len(metadata["row_ids"]),
            "n_methods": len(scores),
        })
    write_json(results / "FIT_COMPLETE.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True,
        "score_manifest": manifest,
    })
    print("[depth-fit] score hashes frozen; outcomes remain unopened", flush=True)


def verify_scores(results: Path) -> dict[str, dict[str, np.ndarray]]:
    verify_prepared(results)
    fit = read_json(results / "FIT_COMPLETE.json")
    if fit.get("labels_seen_during_fit") is not False or fit.get("scores_frozen_before_labels") is not True:
        raise RuntimeError("fit leakage attestation failed")
    if [row["cell"] for row in fit["score_manifest"]] != list(CELLS):
        raise RuntimeError("score roster differs from registered cells")
    bundles = {}
    for row in fit["score_manifest"]:
        score_path = results / row["score_file"]
        diagnostic_path = results / row["diagnostic_file"]
        if sha256_file(score_path) != row["score_sha256"] or sha256_file(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"frozen score artifact changed: {row['cell']}")
        diagnostic = read_json(diagnostic_path)
        if diagnostic.get("labels_seen_during_fit") is not False:
            raise RuntimeError(f"fit diagnostic leakage attestation failed: {row['cell']}")
        with np.load(score_path, allow_pickle=False) as bundle:
            forbidden = [key for key in bundle.files if "label" in key.lower() or key.lower() == "y"]
            if forbidden:
                raise RuntimeError(f"label-like arrays in score bundle: {forbidden}")
            bundles[row["cell"]] = {key: np.asarray(bundle[key]) for key in bundle.files}
    freeze = {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True,
        "fit_complete_sha256": sha256_file(results / "FIT_COMPLETE.json"),
        "run_definition_sha256": sha256_file(results / "RUN_DEFINITION.json"),
        "source_freeze_sha256": sha256_file(results / "SOURCE_FREEZE_MANIFEST.json"),
        "prepared_feature_manifest_sha256": sha256_file(results / "PREPARED_FEATURE_MANIFEST.json"),
        "score_manifest": fit["score_manifest"],
    }
    path = results / "SCORE_FREEZE_MANIFEST.json"
    if path.exists():
        old = read_json(path)
        left, right = dict(old), dict(freeze)
        left.pop("written_utc", None)
        right.pop("written_utc", None)
        if left != right:
            raise RuntimeError("score freeze manifest changed")
    else:
        write_json(path, freeze)
    return bundles


def _base_scores(base_results: Path, cell: str) -> dict[str, np.ndarray]:
    fit = read_json(base_results / "FIT_COMPLETE.json")
    row = next(item for item in fit["score_manifest"] if item["cell"] == cell)
    path = base_results / row["score_file"]
    if sha256_file(path) != row["score_sha256"]:
        raise RuntimeError(f"base score hash mismatch: {cell}")
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def _metric_pair(outcomes: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(outcomes)) != 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(outcomes, score)), float(average_precision_score(outcomes, score))


def _layer_oracle(matrix: FeatureMatrix, outcomes: np.ndarray) -> tuple[np.ndarray, str, float, float]:
    X, keep, _mean, _scale = standardize(matrix.values)
    groups = [matrix.groups[int(index)] for index in keep]
    names = list(dict.fromkeys(groups))
    candidates = []
    for group in names:
        indices = np.flatnonzero(np.asarray(groups, dtype=object) == group)
        score = np.mean(X[:, indices], axis=1)
        if np.corrcoef(score, matrix.risk_anchor)[0, 1] < 0:
            score = -score
        auc, ap = _metric_pair(outcomes, score)
        candidates.append((auc, ap, group, score))
    auc, ap, group, score = max(candidates, key=lambda value: value[0])
    return np.asarray(score), str(group), float(auc), float(ap)


def _grouped_lr(matrix: FeatureMatrix, outcomes: np.ndarray, groups: np.ndarray) -> tuple[float, float, list[dict[str, Any]]]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for fold, (train, test) in enumerate(splitter.split(matrix.values, outcomes, groups)):
        if set(groups[train]) & set(groups[test]):
            raise RuntimeError("problem overlap in grouped supervised diagnostic")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED + fold),
        )
        model.fit(matrix.values[train], outcomes[train])
        probability = model.predict_proba(matrix.values[test])[:, 1]
        auc, ap = _metric_pair(outcomes[test], probability)
        rows.append({"fold": fold, "auroc": auc, "auprc": ap, "problem_overlap": 0})
    return float(np.mean([row["auroc"] for row in rows])), float(np.mean([row["auprc"] for row in rows])), rows


def _trilens_probe_per_cell(base_results: Path) -> dict[str, dict[str, float]]:
    diagnostics = read_json(base_results / "supervised_grouped_cv_diagnostics.json")
    output = {}
    for cell in PRIMARY_CELLS:
        rows = diagnostics[cell]["trilens_supervised_lr"]
        output[cell] = {
            "auroc": float(np.mean([row["auroc"] for row in rows])),
            "auprc": float(np.mean([row["auprc"] for row in rows])),
        }
    return output


def phase_evaluate(cache_root: Path, base_results: Path, results: Path) -> None:
    bundles = verify_scores(results)
    # No correctness labels are opened above this line.
    metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    boot: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    per_cell = []
    oracle_choices = []
    lr_diagnostics = {}
    draw_manifest = {}

    for cell, spec in CELLS.items():
        print(f"[depth-evaluate] {cell}", flush=True)
        bundle = bundles[cell]
        base = _base_scores(base_results, cell)
        row_ids = bundle["row_ids"].astype(str)
        problem_ids = bundle["problem_ids"].astype(str)
        if not np.array_equal(row_ids, base["row_ids"].astype(str)):
            raise RuntimeError(f"{cell}: base/new row order mismatch")
        raw = load_pickle(cache_root / spec["raw"])
        outcomes = load_evaluation_labels(raw, row_ids)
        prevalence = float(np.mean(outcomes))
        scores: dict[str, np.ndarray] = {
            f"baseline__{name}": np.asarray(base[key], dtype=float)
            for name, key in BASELINE_KEYS.items()
        }
        scores.update({
            key: np.asarray(value, dtype=float)
            for key, value in bundle.items()
            if key not in {"row_ids", "problem_ids", "n_gen_tokens", "protocol_signature"}
        })
        lr_diagnostics[cell] = {}
        for contract in EXTRACTORS:
            matrix = load_feature_matrix(_matrix_path(results, cell, contract))
            oracle_score, selected, auc, ap = _layer_oracle(matrix, outcomes)
            oracle_key = f"{contract}__best_layer_oracle"
            scores[oracle_key] = oracle_score
            oracle_choices.append({
                "cell": cell,
                "contract": contract,
                "selected_layer_group": selected,
                "auroc": auc,
                "auprc": ap,
                "selection_uses_evaluation_outcomes": True,
            })
            if cell in PRIMARY_CELLS:
                lr_auc, lr_ap, fold_rows = _grouped_lr(matrix, outcomes, problem_ids)
                metrics[f"{contract}__supervised_lr"][cell] = {"auroc": lr_auc, "auprc": lr_ap}
                lr_diagnostics[cell][contract] = fold_rows

        cell_seed = SEED + int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16)
        indices, draw_hash = group_bootstrap_indices(problem_ids, draws=BOOTSTRAP_DRAWS, seed=cell_seed)
        draw_manifest[cell] = {"seed": cell_seed, "draw_hash": draw_hash}
        for method, score in scores.items():
            auc, ap = _metric_pair(outcomes, score)
            metrics[method][cell] = {"auroc": auc, "auprc": ap}
            role = (
                "appendix_protocol_rejected" if cell not in PRIMARY_CELLS
                else "evaluation_only_oracle" if method.endswith("best_layer_oracle")
                else "existing_baseline" if method.startswith("baseline__")
                else "retrospective_discovery"
            )
            per_cell.append({
                "cell": cell,
                "dataset": spec["dataset"],
                "model": spec.get("model", MODEL),
                "method": method,
                "auroc": auc,
                "auprc": ap,
                "prevalence": prevalence,
                "n_samples": len(outcomes),
                "n_groups": len(np.unique(problem_ids)),
                "analysis_role": role,
            })
            auc_draws, ap_draws = [], []
            for index in indices:
                draw_auc, draw_ap = _metric_pair(outcomes[index], score[index])
                auc_draws.append(draw_auc)
                ap_draws.append(draw_ap)
            boot[method][cell] = {"auroc": np.asarray(auc_draws), "auprc": np.asarray(ap_draws)}
        del raw

    headline = []
    row_level_methods = [
        method for method in metrics
        if all(cell in metrics[method] for cell in PRIMARY_CELLS)
        and not method.endswith("supervised_lr")
    ]
    for method in row_level_methods:
        row = {"method": method, "n_cells": len(PRIMARY_CELLS)}
        for metric in ("auroc", "auprc"):
            point = float(np.mean([metrics[method][cell][metric] for cell in PRIMARY_CELLS]))
            draws = np.mean(np.vstack([boot[method][cell][metric] for cell in PRIMARY_CELLS]), axis=0)
            low, high = np.quantile(draws, (0.025, 0.975))
            row[f"macro_{metric}"] = point
            row[f"macro_{metric}_ci_low"] = float(low)
            row[f"macro_{metric}_ci_high"] = float(high)
        row["analysis_role"] = (
            "evaluation_only_oracle" if method.endswith("best_layer_oracle")
            else "existing_baseline" if method.startswith("baseline__")
            else "retrospective_discovery"
        )
        headline.append(row)

    trilens = _trilens_probe_per_cell(base_results)
    for method in [f"{contract}__supervised_lr" for contract in EXTRACTORS]:
        headline.append({
            "method": method,
            "n_cells": len(PRIMARY_CELLS),
            "macro_auroc": float(np.mean([metrics[method][cell]["auroc"] for cell in PRIMARY_CELLS])),
            "macro_auroc_ci_low": "",
            "macro_auroc_ci_high": "",
            "macro_auprc": float(np.mean([metrics[method][cell]["auprc"] for cell in PRIMARY_CELLS])),
            "macro_auprc_ci_low": "",
            "macro_auprc_ci_high": "",
            "analysis_role": "supervised_diagnostic_ceiling",
        })
    headline.append({
        "method": "original_proxy__trilens_grouped_l2_probe",
        "n_cells": len(PRIMARY_CELLS),
        "macro_auroc": float(np.mean([trilens[cell]["auroc"] for cell in PRIMARY_CELLS])),
        "macro_auroc_ci_low": "",
        "macro_auroc_ci_high": "",
        "macro_auprc": float(np.mean([trilens[cell]["auprc"] for cell in PRIMARY_CELLS])),
        "macro_auprc_ci_low": "",
        "macro_auprc_ci_high": "",
        "analysis_role": "supervised_original_method_approximation",
    })
    headline.sort(key=lambda row: float(row["macro_auroc"]), reverse=True)

    contrasts = []
    for contract in EXTRACTORS:
        fusion_methods = ["iu_pcr", "dufs_liu_pcr"]
        if any(row["method"] == f"{contract}__hierarchical_iu_pcr" for row in headline):
            fusion_methods.append("hierarchical_iu_pcr")
        for fusion in fusion_methods:
            lhs = f"{contract}__{fusion}"
            for rhs in (f"{contract}__best_layer_oracle", f"{contract}__equal_mean"):
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
                    contrasts.append({
                        "contrast": f"{lhs}_minus_{rhs}",
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
                    })
    adjusted = holm_adjust([row["p_raw"] for row in contrasts])
    for row, adjusted_value in zip(contrasts, adjusted):
        row["p_holm"] = adjusted_value

    trilens_auc = next(row["macro_auroc"] for row in headline if row["method"] == "original_proxy__trilens_grouped_l2_probe")
    success_rows = []
    headline_index = {row["method"]: row for row in headline}
    for contract in EXTRACTORS:
        oracle_auc = float(headline_index[f"{contract}__best_layer_oracle"]["macro_auroc"])
        for fusion in ("iu_pcr", "dufs_liu_pcr", "hierarchical_iu_pcr"):
            key = f"{contract}__{fusion}"
            if key not in headline_index:
                continue
            auc = float(headline_index[key]["macro_auroc"])
            contrast = next(
                row for row in contrasts
                if row["lhs"] == key and row["rhs"] == f"{contract}__best_layer_oracle" and row["metric"] == "auroc"
            )
            success_rows.append({
                "contract": contract,
                "fusion": fusion,
                "macro_auroc": auc,
                "best_layer_macro_auroc": oracle_auc,
                "trilens_probe_macro_auroc": trilens_auc,
                "delta_vs_best_layer": auc - oracle_auc,
                "delta_vs_trilens_probe": auc - float(trilens_auc),
                "paired_ci_low_vs_best_layer": contrast["ci_low"],
                "paired_ci_high_vs_best_layer": contrast["ci_high"],
                "discovery_success": bool(
                    auc > oracle_auc
                    and auc > float(trilens_auc)
                    and float(contrast["ci_low"]) > 0.0
                ),
                "independent_confirmation_complete": False,
            })

    write_csv(results / "per_cell_metrics.csv", per_cell)
    write_csv(results / "headline_summary.csv", headline)
    write_csv(results / "paired_comparisons.csv", contrasts)
    write_csv(results / "best_layer_choices.csv", oracle_choices)
    write_csv(results / "success_audit.csv", success_rows)
    write_json(results / "supervised_grouped_cv_diagnostics.json", lr_diagnostics)
    write_json(results / "bootstrap_draw_manifest.json", {
        "draws": BOOTSTRAP_DRAWS,
        "root_seed": SEED,
        "unit": "problem_group_within_cell",
        "identical_draws_reused_across_methods_within_cell": True,
        "cells": draw_manifest,
    })
    write_json(results / "validation_status.json", {
        "status": "PRELIMINARY / VALIDATION BLOCKED",
        "corrected_live_gate_b_all_pass": False,
        "architecture_pilot_pass": False,
        "retrospective_discovery": True,
        "independent_confirmation_complete": False,
        "promotion_allowed": False,
    })
    print("[depth-evaluate] evaluation complete", flush=True)


def _display(method: str) -> str:
    aliases = {
        "baseline__final_nll": "Final-layer NLL",
        "baseline__generation_entropy": "Mean generation entropy",
        "baseline__trilens_dufs": "TriLens features + DUFS-LIU-PCR",
        "baseline__lens96_dufs": "Expanded lens + DUFS-LIU-PCR",
        "original_proxy__trilens_grouped_l2_probe": "TriLens L2 probe (supervised approximation)",
    }
    if method in aliases:
        return aliases[method]
    contract, _, fusion = method.partition("__")
    contract_name = DEPTH_METRIC_REGISTRY.get(contract, {}).get("display", contract.replace("_", " "))
    fusion_name = {
        "equal_mean": "equal mean",
        "iu_pcr": "IU-PCR",
        "dufs_liu_pcr": "DUFS-LIU-PCR",
        "hierarchical_iu_pcr": "layer-hierarchical IU-PCR",
        "best_layer_oracle": "best single layer (evaluation oracle)",
        "supervised_lr": "grouped logistic probe (supervised)",
    }.get(fusion, fusion.replace("_", " "))
    return f"{contract_name} · {fusion_name}"


def phase_report(results: Path) -> None:
    with (results / "headline_summary.csv").open(newline="", encoding="utf-8") as handle:
        headline = list(csv.DictReader(handle))
    with (results / "success_audit.csv").open(newline="", encoding="utf-8") as handle:
        success = list(csv.DictReader(handle))
    with (results / "structural_diagnostics.csv").open(newline="", encoding="utf-8") as handle:
        structural = list(csv.DictReader(handle))
    wins = [row for row in success if row["discovery_success"].lower() == "true"]
    width, height, left, top = 980, max(420, 40 + 28 * len(headline)), 320, 28
    aurocs = [float(row["macro_auroc"]) for row in headline]
    lo, hi = min(0.45, min(aurocs) - 0.02), max(0.82, max(aurocs) + 0.02)
    def xpos(value: float) -> float:
        return left + (value - lo) / (hi - lo) * (width - left - 28)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Methods sorted by macro AUROC">']
    svg.append('<style>text{font-family:system-ui,sans-serif;fill:#172033}.axis{stroke:#8a93a3}.bar{fill:#4f6bed}.oracle{fill:#b26a00}.sup{fill:#8b5cf6}</style>')
    for tick in np.linspace(lo, hi, 8):
        x = xpos(float(tick))
        svg.append(f'<line class="axis" x1="{x:.1f}" x2="{x:.1f}" y1="12" y2="{height-18}" opacity=".22"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-3}" text-anchor="middle" font-size="11">{tick:.2f}</text>')
    for index, row in enumerate(headline):
        y = top + index * 28
        label = html.escape(_display(row["method"]))
        value = float(row["macro_auroc"])
        role = row["analysis_role"]
        klass = "sup" if "supervised" in role else "oracle" if "oracle" in role else "bar"
        svg.append(f'<text x="8" y="{y+4}" font-size="11">{label}</text>')
        svg.append(f'<rect class="{klass}" x="{xpos(lo):.1f}" y="{y-9}" width="{max(1,xpos(value)-xpos(lo)):.1f}" height="14" rx="3"/>')
        svg.append(f'<text x="{xpos(value)+5:.1f}" y="{y+3}" font-size="11">{value:.4f}</text>')
    svg.append('</svg>')
    figure = "\n".join(svg)
    (results / "figures").mkdir(exist_ok=True)
    (results / "figures" / "macro_auroc_sorted.svg").write_text(figure, encoding="utf-8")

    def table(rows: Sequence[Mapping[str, Any]], fields: Sequence[tuple[str, str]]) -> str:
        head = "".join(f"<th>{html.escape(label)}</th>" for _, label in fields)
        body = []
        for row in rows:
            body.append("<tr>" + "".join(
                f"<td>{html.escape(str(row.get(key, '')))}</td>" for key, _ in fields
            ) + "</tr>")
        return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"

    report = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>White-box depth-metric search</title><style>
:root{{--bg:#f6f8fb;--card:#fff;--text:#172033;--muted:#586174;--line:#d9deea;--accent:#4f6bed;--bad:#9a3412}}
@media(prefers-color-scheme:dark){{:root{{--bg:#10131a;--card:#181d27;--text:#eef2ff;--muted:#aab2c5;--line:#343c4d;--accent:#8ca0ff;--bad:#fdba74}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,sans-serif}}main{{max-width:1220px;margin:auto;padding:28px}}section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin:16px 0;overflow:auto}}h1,h2{{line-height:1.2}}.badge{{display:inline-block;padding:6px 10px;border-radius:999px;background:#ffedd5;color:#9a3412;font-weight:750}}.lead{{font-size:18px;max-width:900px}}.muted{{color:var(--muted)}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{position:sticky;top:0;background:var(--card)}}figure{{margin:0;min-width:760px}}svg{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}@media(max-width:700px){{main{{padding:14px}}section{{padding:14px}}}}
</style></head><body><main>
<span class="badge">PRELIMINARY / VALIDATION BLOCKED</span>
<h1>White-box depth-metric fusion search</h1>
<p class="lead">Four literature-grounded, depth-distributed signals were frozen before this evaluation phase. This is a retrospective discovery screen. It cannot be promoted without corrected live Gate B, the architecture pilot, and an independent confirmation dataset.</p>
<section><h2>Outcome</h2><p>{'A discovery candidate met the numeric screen, but still needs independent confirmation.' if wins else 'No candidate met the strict discovery rule: beat both its per-cell best-layer oracle and the local TriLens supervised-probe approximation with a positive paired interval.'}</p>
{table(success, [('contract','Metric'),('fusion','Fusion'),('macro_auroc','AUROC'),('best_layer_macro_auroc','Best layer'),('trilens_probe_macro_auroc','TriLens probe'),('delta_vs_best_layer','Δ vs layer'),('delta_vs_trilens_probe','Δ vs TriLens'),('paired_ci_low_vs_best_layer','CI low'),('paired_ci_high_vs_best_layer','CI high'),('discovery_success','Pass')])}</section>
<section><h2>Methods sorted by macro AUROC</h2><p class="muted">Purple and ochre rows use labels and are ceilings, not label-free competitors.</p><figure>{figure}</figure></section>
<section><h2>Exact headline values</h2>{table(headline,[('method','Method'),('macro_auroc','AUROC'),('macro_auroc_ci_low','AUROC low'),('macro_auroc_ci_high','AUROC high'),('macro_auprc','AUPRC'),('analysis_role','Role')])}</section>
<section><h2>Frozen metric registry</h2>{table([{'metric':k,**v} for k,v in DEPTH_METRIC_REGISTRY.items()],[('metric','Metric'),('formula','Formula'),('layer_policy','Layers'),('risk_direction','Risk direction'),('readout','Readout')])}</section>
<section><h2>Structural depth coverage</h2>{table(structural,[('cell','Cell'),('contract','Metric'),('n_features_kept','Features'),('n_layer_groups','Layer groups'),('median_abs_feature_correlation','Median |corr|'),('effective_rank','Effective rank'),('effective_rank_fraction','Rank fraction')])}</section>
<section><h2>Claim boundary</h2><ul><li>Equal-cell macro across 13 protocol-eligible cells; no pooled 47,238-row headline.</li><li>Problem-group bootstrap, 2,000 deterministic draws, shared across methods within each cell.</li><li>Best-layer rows select the layer using evaluation outcomes and are deliberately optimistic ceilings.</li><li>TriLens is a grouped-CV L2 logistic-probe approximation on the saved 3L entropy features.</li><li>GHOST preserves the paper's angular equations but uses the cached mean-token 256-D projection, not every token's full hidden state.</li></ul></section>
</main></body></html>"""
    (results / "REPORT.html").write_text(report, encoding="utf-8")
    artifacts = []
    for path in sorted(results.rglob("*")):
        if path.is_file() and path.name != "REPORT_MANIFEST.json":
            artifacts.append({"file": str(path.relative_to(results)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(results / "REPORT_MANIFEST.json", {"version": VERSION, "written_utc": utcnow(), "artifacts": artifacts})
    print(f"[depth-report] wrote {results / 'REPORT.html'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit", "evaluate", "report", "all"))
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--base-results", type=Path, default=DEFAULT_BASE_RESULTS)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args()
    cache_root = args.cache_root.resolve()
    base_results = args.base_results.resolve()
    results = args.results_dir.resolve()
    if args.phase == "prepare":
        phase_prepare(cache_root, base_results, results)
    elif args.phase == "fit":
        phase_fit(results)
    elif args.phase == "evaluate":
        phase_evaluate(cache_root, base_results, results)
    elif args.phase == "report":
        phase_report(results)
    else:
        for phase in ("prepare", "fit", "evaluate", "report"):
            subprocess.run([
                sys.executable,
                str(Path(__file__).resolve()),
                phase,
                "--cache-root", str(cache_root),
                "--base-results", str(base_results),
                "--results-dir", str(results),
            ], cwd=REPO, check=True)


if __name__ == "__main__":
    main()
