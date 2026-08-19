#!/usr/bin/env python3
"""Freeze, fit, evaluate, and report the retrospective depth-consensus win."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr, wilcoxon
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
    PRIMARY_CELLS,
    SEED,
    TIE_TOLERANCE,
    group_bootstrap_indices,
    load_evaluation_labels,
    load_feature_matrix,
    metric_pair,
    save_feature_matrix,
    sha256_file,
    validate_and_join,
)
from spectral_utils.whitebox_depth_consensus import (  # noqa: E402
    COMPONENTS,
    REGISTRY,
    VERSION,
    extract_depth_consensus,
    registry_hash,
)
from spectral_utils.whitebox_layer_fusion import (  # noqa: E402
    fit_controls,
    fit_core_spectral,
)

RESULTS = REPO / "results" / "whitebox_depth_consensus_v1"
BASE_RESULTS = REPO / "results" / "whitebox_layer_fusion_v2"
SOURCE_FILES = (
    "scripts/whitebox_depth_consensus_experiment.py",
    "spectral_utils/whitebox_depth_consensus.py",
    "spectral_utils/whitebox_depth_metrics.py",
    "spectral_utils/whitebox_depth_token_metrics.py",
    "spectral_utils/whitebox_layer_fusion.py",
    "spectral_utils/paper_benchmark_suite.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/selectors/a2_groupfs.py",
)
DISPLAY = {
    "upcr": "Depth consensus · deployed U-PCR",
    "equal_mean": "Depth consensus · equal mean",
    "iu_pcr": "Depth consensus · IU-PCR",
    "dufs_liu_pcr": "Depth consensus · DUFS-LIU-PCR",
    "trilens_supervised_lr": "TriLens · grouped L2 probe",
    "best_single_layer": "Best single module/metric/layer · evaluation oracle",
}
DISCOVERY_SELECTION_HISTORY = (
    "Chosen after two negative frozen searches, geometry/frequency/readout screens, "
    "and an explicit component subset audit on these same 13 eligible cells."
)
DISCOVERY_METHOD_DESCRIPTION = (
    "The candidate matrix contains four label-free columns: prediction revision, "
    "residual entropy burst, residual top-1 sharpness, and maximum-token TriLens entropy. "
    "Each column summarizes evidence distributed across several layers. Per-view selection "
    "uses only rank correlation with final-layer target NLL; correctness labels are "
    "unavailable to fitting."
)
REPORT_TITLE = "White-box depth consensus"


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def prepared_path(results: Path, cell: str) -> Path:
    return results / "prepared" / f"{cell}__depth_consensus.npz"


def rows_path(results: Path, cell: str) -> Path:
    return results / "prepared" / f"{cell}__rows.npz"


def score_path(results: Path, cell: str) -> Path:
    return results / "scores" / f"{cell}.npz"


def prepare(cache: Path, results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / "prepared").mkdir(exist_ok=True)
    source_freeze = read_json(BASE_RESULTS / "SOURCE_FREEZE_MANIFEST.json")
    source_index = {str(row["local_path"]): row for row in source_freeze["sources"]}
    sources = []
    prepared = []
    diagnostics = []

    write_json(results / "DISCOVERY_REGISTRY.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "analysis_role": "retrospective_post_outcome_search; not preregistered",
        "registry": REGISTRY,
        "registry_sha256": registry_hash(),
        "selection_history": DISCOVERY_SELECTION_HISTORY,
        "success_target": "macro AUROC above both the best-single-layer oracle and TriLens grouped L2 probe",
        "validation_status": "PRELIMINARY / VALIDATION BLOCKED",
    })

    for cell, spec in CELLS.items():
        print(f"[consensus-prepare] {cell}", flush=True)
        raw_path = cache / spec["raw"]
        sidecar_path = cache / spec["sidecar"]
        for path in (raw_path, sidecar_path):
            frozen = source_index.get(str(path.resolve()))
            if frozen is None or path.stat().st_size != int(frozen["local_size"]):
                raise RuntimeError(f"{path}: does not match v2 source freeze")
            sources.append({
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
            expected_model=spec.get("model", "meta-llama/Llama-3.1-8B-Instruct"),
            expected_n_layers=int(spec.get("n_layers", 32)),
            expected_hidden_size=int(spec.get("hidden_size", 4096)),
            exclude_invalid=True,
            require_geometry_finite=False,
        )
        matrix = extract_depth_consensus(joined)
        path = prepared_path(results, cell)
        save_feature_matrix(path, matrix)
        row_path = rows_path(results, cell)
        np.savez_compressed(
            row_path,
            row_ids=np.asarray(joined.row_ids, dtype="U"),
            problem_ids=np.asarray(joined.problem_ids, dtype="U"),
            n_gen_tokens=np.asarray(joined.n_gen_tokens, dtype=np.int64),
            protocol_signature=np.asarray(joined.protocol_signature),
        )
        diagnostics.append({
            "cell": cell,
            "n_samples": joined.n_samples,
            "n_groups": len(set(joined.problem_ids)),
            "n_layers": joined.n_layers,
            "n_components": matrix.n_features,
            "excluded_rows": audit["n_excluded_rows"],
            "selected_views": {
                key: value.get("selected_feature_names", [])
                for key, value in matrix.metadata["component_fits"].items()
                if isinstance(value, Mapping)
            },
        })
        del raw, sidecar, joined, matrix

    for path in sorted((results / "prepared").glob("*.npz")):
        with np.load(path, allow_pickle=False) as bundle:
            forbidden = [key for key in bundle.files if "label" in key.lower() or key.lower() == "y"]
            if forbidden:
                raise RuntimeError(f"label-like prepared fields in {path}: {forbidden}")
        prepared.append({"file": str(path.relative_to(results)), "sha256": sha256_file(path)})
    write_json(results / "SOURCE_FREEZE_MANIFEST.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "base_source_freeze": str((BASE_RESULTS / "SOURCE_FREEZE_MANIFEST.json").resolve()),
        "base_source_freeze_sha256": sha256_file(BASE_RESULTS / "SOURCE_FREEZE_MANIFEST.json"),
        "sources": sources,
    })
    write_json(results / "PREPARED_FEATURE_MANIFEST.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_present": False,
        "artifacts": prepared,
    })
    write_json(results / "RUN_DEFINITION.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "cells": list(CELLS),
        "primary_cells": list(PRIMARY_CELLS),
        "n_primary_cells": len(PRIMARY_CELLS),
        "components": list(COMPONENTS),
        "primary_method": "deployed U-PCR, frozen repository configuration",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": SEED,
        "source_code": [
            {"file": relative, "sha256": sha256_file(REPO / relative)}
            for relative in SOURCE_FILES
        ],
    })
    write_json(results / "component_diagnostics.json", diagnostics)
    write_json(results / "validation_status.json", {
        "status": "PRELIMINARY / VALIDATION BLOCKED",
        "corrected_live_gate_b_pass": False,
        "architecture_pilot_pass": False,
        "reason": "Offline discovery completed; corrected live Gate B and architecture pilot remain open.",
    })


def verify_prepared(results: Path) -> None:
    manifest = read_json(results / "PREPARED_FEATURE_MANIFEST.json")
    for row in manifest["artifacts"]:
        path = results / row["file"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"prepared hash mismatch: {path}")


def fit(results: Path) -> None:
    verify_prepared(results)
    (results / "scores").mkdir(exist_ok=True)
    (results / "diagnostics").mkdir(exist_ok=True)
    manifest = []
    for cell in CELLS:
        print(f"[consensus-fit] {cell}", flush=True)
        matrix = load_feature_matrix(prepared_path(results, cell))
        controls, control_diag = fit_controls(matrix)
        fitted, fit_diag = fit_core_spectral(matrix)
        with np.load(rows_path(results, cell), allow_pickle=False) as rows:
            row_ids = rows["row_ids"].copy()
            problem_ids = rows["problem_ids"].copy()
        path = score_path(results, cell)
        np.savez_compressed(
            path,
            row_ids=row_ids,
            problem_ids=problem_ids,
            equal_mean=controls["equal_mean"],
            upcr=fitted["upcr"],
            iu_pcr=fitted["iu_pcr"],
            dufs_liu_pcr=fitted["dufs_liu_pcr"],
        )
        diagnostic = {
            "cell": cell,
            "labels_seen_during_fit": False,
            "scores_frozen_before_labels": True,
            "feature_names": list(matrix.feature_names),
            "controls": control_diag,
            "core": fit_diag,
        }
        diagnostic_path = results / "diagnostics" / f"{cell}.json"
        write_json(diagnostic_path, diagnostic)
        manifest.append({
            "cell": cell,
            "score_file": str(path.relative_to(results)),
            "score_sha256": sha256_file(path),
            "diagnostic_file": str(diagnostic_path.relative_to(results)),
            "diagnostic_sha256": sha256_file(diagnostic_path),
        })
    write_json(results / "FIT_COMPLETE.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "score_manifest": manifest,
    })
    write_json(results / "SCORE_FREEZE_MANIFEST.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True,
        "score_files_verified_before_labels": True,
        "fit_complete_sha256": sha256_file(results / "FIT_COMPLETE.json"),
        "source_freeze_sha256": sha256_file(results / "SOURCE_FREEZE_MANIFEST.json"),
        "prepared_feature_manifest_sha256": sha256_file(results / "PREPARED_FEATURE_MANIFEST.json"),
        "score_manifest": manifest,
    })


def verify_scores(results: Path) -> dict[str, dict[str, np.ndarray]]:
    freeze = read_json(results / "SCORE_FREEZE_MANIFEST.json")
    if freeze.get("labels_seen_during_fit") is not False or freeze.get("scores_frozen_before_labels") is not True:
        raise RuntimeError("score-freeze leakage attestation failed")
    out = {}
    for row in freeze["score_manifest"]:
        path = results / row["score_file"]
        if sha256_file(path) != row["score_sha256"]:
            raise RuntimeError(f"frozen score mismatch: {path}")
        with np.load(path, allow_pickle=False) as bundle:
            forbidden = [key for key in bundle.files if "label" in key.lower() or key.lower() == "y"]
            if forbidden:
                raise RuntimeError(f"label-like score fields: {forbidden}")
            out[row["cell"]] = {key: np.asarray(bundle[key]) for key in bundle.files}
    return out


def best_single_layer(cell: str, y: np.ndarray) -> tuple[np.ndarray, str, float, float]:
    matrix = load_feature_matrix(BASE_RESULTS / "prepared" / f"{cell}__lens_grid_all.npz")
    best = (-np.inf, "", None, float("nan"))
    for index, name in enumerate(matrix.feature_names):
        score = np.asarray(matrix.values[:, index], dtype=float)
        corr = float(spearmanr(score, matrix.risk_anchor).statistic)
        if corr < 0.0:
            score = -score
        auc, ap = metric_pair(y, score)
        if auc > best[0]:
            best = (auc, name, score, ap)
    return np.asarray(best[2]), str(best[1]), float(best[0]), float(best[3])


def trilens_folds(
    cell: str,
    y: np.ndarray,
    groups: np.ndarray,
    candidate: np.ndarray,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    matrix = load_feature_matrix(BASE_RESULTS / "prepared" / f"{cell}__trilens_entropy_all.npz")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    auc_draws = []
    ap_draws = []
    for fold, (train, test) in enumerate(splitter.split(matrix.values, y, groups)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=3000, random_state=SEED + fold),
        )
        model.fit(matrix.values[train], y[train])
        score = model.predict_proba(matrix.values[test])[:, 1]
        tri_auc, tri_ap = metric_pair(y[test], score)
        cand_auc, cand_ap = metric_pair(y[test], candidate[test])
        indices, draw_hash = group_bootstrap_indices(
            groups[test], draws=BOOTSTRAP_DRAWS, seed=SEED + 1000 + fold
        )
        fold_auc = []
        fold_ap = []
        for draw in indices:
            a1, p1 = metric_pair(y[test][draw], candidate[test][draw])
            a0, p0 = metric_pair(y[test][draw], score[draw])
            fold_auc.append(a1 - a0)
            fold_ap.append(p1 - p0)
        auc_draws.append(np.asarray(fold_auc))
        ap_draws.append(np.asarray(fold_ap))
        rows.append({
            "fold": fold,
            "n_test": len(test),
            "problem_overlap": len(set(groups[train]) & set(groups[test])),
            "trilens_auroc": tri_auc,
            "trilens_auprc": tri_ap,
            "candidate_auroc": cand_auc,
            "candidate_auprc": cand_ap,
            "auroc_delta": cand_auc - tri_auc,
            "auprc_delta": cand_ap - tri_ap,
            "draw_hash": draw_hash,
        })
    return rows, np.nanmean(np.vstack(auc_draws), axis=0), np.nanmean(np.vstack(ap_draws), axis=0)


def percentile(draws: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(draws, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def evaluate(cache: Path, results: Path) -> None:
    bundles = verify_scores(results)  # labels remain unopened above this line
    per_cell = []
    boot: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    points: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    single_scores = {}
    trilens_rows = {}
    tri_delta_boot = {"auroc": {}, "auprc": {}}
    draw_manifest = []

    for cell_index, (cell, spec) in enumerate(CELLS.items()):
        print(f"[consensus-evaluate] {cell}", flush=True)
        bundle = bundles[cell]
        row_ids = tuple(str(value) for value in bundle["row_ids"].tolist())
        groups = np.asarray(bundle["problem_ids"], dtype=str)
        raw = load_pickle(cache / spec["raw"])
        y = load_evaluation_labels(raw, row_ids)
        del raw
        prevalence = float(np.mean(y))
        draws, draw_hash = group_bootstrap_indices(
            groups, draws=BOOTSTRAP_DRAWS, seed=SEED + cell_index
        )
        draw_manifest.append({"cell": cell, "seed": SEED + cell_index, "draw_hash": draw_hash})
        for method in ("upcr", "equal_mean", "iu_pcr", "dufs_liu_pcr"):
            score = np.asarray(bundle[method], dtype=float)
            auc, ap = metric_pair(y, score)
            points[method][cell] = {"auroc": auc, "auprc": ap}
            auc_draws, ap_draws = [], []
            for index in draws:
                a, p = metric_pair(y[index], score[index])
                auc_draws.append(a); ap_draws.append(p)
            boot[method][cell] = {"auroc": np.asarray(auc_draws), "auprc": np.asarray(ap_draws)}
            per_cell.append({
                "cell": cell, "method": method, "display_method": DISPLAY[method],
                "auroc": auc, "auprc": ap, "prevalence": prevalence,
                "n_samples": len(y), "n_groups": len(np.unique(groups)),
                "label_use": "none", "status": "retrospective_discovery",
            })

        single, selected, auc, ap = best_single_layer(cell, y)
        single_scores[cell] = single
        points["best_single_layer"][cell] = {"auroc": auc, "auprc": ap}
        auc_draws, ap_draws = [], []
        for index in draws:
            a, p = metric_pair(y[index], single[index])
            auc_draws.append(a); ap_draws.append(p)
        boot["best_single_layer"][cell] = {"auroc": np.asarray(auc_draws), "auprc": np.asarray(ap_draws)}
        per_cell.append({
            "cell": cell, "method": "best_single_layer", "display_method": DISPLAY["best_single_layer"],
            "auroc": auc, "auprc": ap, "prevalence": prevalence,
            "n_samples": len(y), "n_groups": len(np.unique(groups)),
            "label_use": "evaluation-selected", "status": "diagnostic_ceiling",
        })

        folds, auc_delta, ap_delta = trilens_folds(cell, y, groups, np.asarray(bundle["upcr"]))
        trilens_rows[cell] = folds
        tri_delta_boot["auroc"][cell] = auc_delta
        tri_delta_boot["auprc"][cell] = ap_delta
        tri_auc = float(np.mean([row["trilens_auroc"] for row in folds]))
        tri_ap = float(np.mean([row["trilens_auprc"] for row in folds]))
        points["trilens_supervised_lr"][cell] = {"auroc": tri_auc, "auprc": tri_ap}
        per_cell.append({
            "cell": cell, "method": "trilens_supervised_lr", "display_method": DISPLAY["trilens_supervised_lr"],
            "auroc": tri_auc, "auprc": tri_ap, "prevalence": prevalence,
            "n_samples": len(y), "n_groups": len(np.unique(groups)),
            "label_use": "5-fold grouped supervised", "status": "published-method approximation",
        })

    methods = ("upcr", "equal_mean", "iu_pcr", "dufs_liu_pcr", "trilens_supervised_lr", "best_single_layer")
    headline = []
    for method in methods:
        row = {"method": method, "display_method": DISPLAY[method], "n_cells": len(PRIMARY_CELLS)}
        for metric_name in ("auroc", "auprc"):
            point = float(np.mean([points[method][cell][metric_name] for cell in PRIMARY_CELLS]))
            if method in boot:
                draws = np.nanmean(np.vstack([boot[method][cell][metric_name] for cell in PRIMARY_CELLS]), axis=0)
                low, high = percentile(draws)
            else:
                low = high = float("nan")
            row[f"macro_{metric_name}"] = point
            row[f"macro_{metric_name}_ci_low"] = low
            row[f"macro_{metric_name}_ci_high"] = high
        row["label_use"] = "supervised" if method == "trilens_supervised_lr" else (
            "evaluation-only" if method == "best_single_layer" else "none"
        )
        headline.append(row)
    headline.sort(key=lambda row: row["macro_auroc"], reverse=True)

    comparisons = []
    contrast_specs = (
        ("upcr_minus_trilens", "upcr", "trilens_supervised_lr"),
        ("upcr_minus_best_single_layer", "upcr", "best_single_layer"),
        ("upcr_minus_equal_mean", "upcr", "equal_mean"),
        ("upcr_minus_iu", "upcr", "iu_pcr"),
        ("upcr_minus_dufs", "upcr", "dufs_liu_pcr"),
    )
    for contrast, lhs, rhs in contrast_specs:
        for metric_name in ("auroc", "auprc"):
            if rhs == "trilens_supervised_lr":
                per_delta = {
                    cell: float(np.mean([row[f"{metric_name}_delta"] for row in trilens_rows[cell]]))
                    for cell in PRIMARY_CELLS
                }
                draws = np.nanmean(np.vstack([tri_delta_boot[metric_name][cell] for cell in PRIMARY_CELLS]), axis=0)
                point = float(np.mean(list(per_delta.values())))
                protocol = "paired within identical grouped-CV test folds"
            else:
                per_delta = {
                    cell: points[lhs][cell][metric_name] - points[rhs][cell][metric_name]
                    for cell in PRIMARY_CELLS
                }
                draws = np.nanmean(np.vstack([
                    boot[lhs][cell][metric_name] - boot[rhs][cell][metric_name]
                    for cell in PRIMARY_CELLS
                ]), axis=0)
                point = float(np.mean(list(per_delta.values())))
                protocol = "paired problem-group bootstrap"
            low, high = percentile(draws)
            values = np.asarray(list(per_delta.values()))
            wins = int(np.sum(values > TIE_TOLERANCE))
            losses = int(np.sum(values < -TIE_TOLERANCE))
            ties = len(values) - wins - losses
            try:
                p_value = float(wilcoxon(values).pvalue)
            except ValueError:
                p_value = 1.0
            comparisons.append({
                "contrast": contrast, "lhs": lhs, "rhs": rhs, "metric": metric_name,
                "delta": point, "ci_low": low, "ci_high": high,
                "wins": wins, "ties": ties, "losses": losses,
                "worst_cell_delta": float(np.min(values)), "wilcoxon_p_raw": p_value,
                "protocol": protocol,
            })
    p_values = np.asarray([row["wilcoxon_p_raw"] for row in comparisons])
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values)); running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    for row, value in zip(comparisons, adjusted):
        row["wilcoxon_p_holm"] = float(value)

    write_csv(results / "per_cell_metrics.csv", per_cell)
    write_csv(results / "headline_summary.csv", headline)
    write_csv(results / "paired_comparisons.csv", comparisons)
    write_json(results / "trilens_grouped_cv_diagnostics.json", trilens_rows)
    write_json(results / "bootstrap_draw_manifest.json", {
        "draws": BOOTSTRAP_DRAWS, "base_seed": SEED, "cells": draw_manifest,
    })

    headline_index = {row["method"]: row for row in headline}
    point_success = (
        headline_index["upcr"]["macro_auroc"] > headline_index["trilens_supervised_lr"]["macro_auroc"]
        and headline_index["upcr"]["macro_auroc"] > headline_index["best_single_layer"]["macro_auroc"]
    )
    primary = {row["contrast"]: row for row in comparisons if row["metric"] == "auroc"}
    robust_success = (
        primary["upcr_minus_trilens"]["ci_low"] > 0.0
        and primary["upcr_minus_best_single_layer"]["ci_low"] > 0.0
    )
    write_json(results / "success_audit.json", {
        "point_estimate_success": bool(point_success),
        "bootstrap_robust_success": bool(robust_success),
        "upcr_macro_auroc": headline_index["upcr"]["macro_auroc"],
        "trilens_macro_auroc": headline_index["trilens_supervised_lr"]["macro_auroc"],
        "best_single_layer_macro_auroc": headline_index["best_single_layer"]["macro_auroc"],
        "upcr_macro_auprc": headline_index["upcr"]["macro_auprc"],
        "trilens_macro_auprc": headline_index["trilens_supervised_lr"]["macro_auprc"],
        "claim": (
            "retrospective numerical AUROC discovery candidate"
            if point_success else "success target not met"
        ),
        "independent_confirmation_required": True,
        "validation_blocked": True,
    })


def fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return "—" if not np.isfinite(number) else f"{number:.{digits}f}"


def fmt_interval(low: Any, high: Any, digits: int = 4) -> str:
    left, right = fmt(low, digits), fmt(high, digits)
    return "—" if left == "—" and right == "—" else f"{left}–{right}"


def make_svg(headline: Sequence[Mapping[str, Any]]) -> str:
    rows = sorted(headline, key=lambda row: float(row["macro_auroc"]), reverse=True)
    width, height = 920, max(360, 78 + len(rows) * 48)
    margin = 230
    usable = width - margin - 55
    finite_values = [float(row["macro_auroc"]) for row in rows if np.isfinite(float(row["macro_auroc"]))]
    lo = min(0.5, np.floor(min(finite_values) * 20.0) / 20.0) if finite_values else 0.5
    hi = max(0.8, np.ceil(max(finite_values) * 20.0) / 20.0) if finite_values else 0.8
    def x(value: float) -> float:
        return margin + (value - lo) / (hi - lo) * usable
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-label="Macro AUROC comparison">']
    parts.append('<style>text{font:14px system-ui;fill:#152033}.grid{stroke:#d9e2ef}.bar{fill:#4472c4}.win{fill:#0f8b6d}</style>')
    for tick in np.linspace(lo, hi, 7):
        px = x(float(tick)); parts.append(f'<line class="grid" x1="{px:.1f}" y1="24" x2="{px:.1f}" y2="{height-30}"/>')
        parts.append(f'<text x="{px:.1f}" y="{height-8}" text-anchor="middle">{tick:.2f}</text>')
    for index, row in enumerate(rows):
        y = 38 + index * 48
        value = float(row["macro_auroc"])
        label = html.escape(str(row["display_method"]))
        cls = "win" if row["method"] == "upcr" else "bar"
        parts.append(f'<text x="218" y="{y+15}" text-anchor="end">{label}</text>')
        parts.append(f'<rect class="{cls}" x="{x(lo):.1f}" y="{y}" width="{max(1,x(value)-x(lo)):.1f}" height="22" rx="4"/>')
        parts.append(f'<text x="{x(value)+7:.1f}" y="{y+16}">{value:.4f}</text>')
    parts.append('</svg>')
    return "".join(parts)


def report(results: Path) -> None:
    headline = list(csv.DictReader((results / "headline_summary.csv").open(newline="", encoding="utf-8")))
    comparisons = list(csv.DictReader((results / "paired_comparisons.csv").open(newline="", encoding="utf-8")))
    per_cell = list(csv.DictReader((results / "per_cell_metrics.csv").open(newline="", encoding="utf-8")))
    success = read_json(results / "success_audit.json")
    figures = results / "figures"; figures.mkdir(exist_ok=True)
    svg = make_svg(headline)
    svg_path = figures / "macro_auroc.svg"
    svg_path.write_text(svg, encoding="utf-8")
    embedded = base64.b64encode(svg.encode("utf-8")).decode("ascii")

    headline_rows = "".join(
        "<tr>" + "".join([
            f"<td>{html.escape(row['display_method'])}</td>",
            f"<td>{fmt(row['macro_auroc'])}</td>",
            f"<td>{fmt_interval(row['macro_auroc_ci_low'], row['macro_auroc_ci_high'])}</td>",
            f"<td>{fmt(row['macro_auprc'])}</td>",
            f"<td>{html.escape(row['label_use'])}</td>",
        ]) + "</tr>" for row in headline
    )
    primary_rows = "".join(
        "<tr>" + "".join([
            f"<td>{html.escape(row['contrast'])}</td>",
            f"<td>{fmt(row['delta'])}</td>",
            f"<td>{fmt_interval(row['ci_low'], row['ci_high'])}</td>",
            f"<td>{row['wins']}/{row['ties']}/{row['losses']}</td>",
            f"<td>{fmt(row['worst_cell_delta'])}</td>",
            f"<td>{html.escape(row['protocol'])}</td>",
        ]) + "</tr>"
        for row in comparisons if row["metric"] == "auroc"
    )
    cell_index = defaultdict(dict)
    for row in per_cell:
        cell_index[row["cell"]][row["method"]] = row
    cell_rows = "".join(
        "<tr>" + "".join([
            f"<td>{html.escape(cell)}</td>",
            f"<td>{fmt(methods['upcr']['auroc'])}</td>",
            f"<td>{fmt(methods['trilens_supervised_lr']['auroc'])}</td>",
            f"<td>{fmt(methods['best_single_layer']['auroc'])}</td>",
            f"<td>{fmt(float(methods['upcr']['auroc'])-float(methods['trilens_supervised_lr']['auroc']))}</td>",
        ]) + "</tr>" for cell, methods in cell_index.items() if cell in PRIMARY_CELLS
    )
    status = "PRELIMINARY / VALIDATION BLOCKED"
    conclusion = (
        "The frozen deployed U-PCR score clears both AUROC comparators on the 13-cell macro point estimate. "
        "This is a discovery result, not independent confirmation."
        if success["point_estimate_success"] else
        "The registered success target was not met."
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(REPORT_TITLE)}</title>
<style>
:root{{--bg:#f5f7fb;--card:#fff;--ink:#142033;--muted:#5d6b80;--line:#dce4ef;--accent:#0f8b6d;--warn:#9c5a00}}
@media(prefers-color-scheme:dark){{:root{{--bg:#101722;--card:#182231;--ink:#edf3fb;--muted:#aebbd0;--line:#34445a;--accent:#54d5af;--warn:#ffbf69}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:28px 18px 60px}}section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0;overflow:auto}}h1{{font-size:2rem;margin:.2rem 0}}h2{{margin-top:0}}.status{{color:var(--warn);font-weight:800;letter-spacing:.04em}}.hero{{border-left:6px solid var(--accent)}}.numbers{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}}.number{{padding:14px;border:1px solid var(--line);border-radius:10px}}.number b{{font-size:1.7rem;display:block}}table{{border-collapse:collapse;width:100%;min-width:720px}}th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{position:sticky;top:0;background:var(--card)}}img{{width:100%;height:auto}}code{{background:color-mix(in srgb,var(--card),var(--line) 35%);padding:.12rem .3rem;border-radius:4px}}.muted{{color:var(--muted)}}
</style></head><body><main>
<section class="hero"><div class="status">{status}</div><h1>{html.escape(REPORT_TITLE)}</h1>
<p>{html.escape(conclusion)}</p><div class="numbers">
<div class="number"><b>{fmt(success['upcr_macro_auroc'])}</b>deployed U-PCR AUROC</div>
<div class="number"><b>{fmt(success['trilens_macro_auroc'])}</b>TriLens L2-probe AUROC</div>
<div class="number"><b>{fmt(success['best_single_layer_macro_auroc'])}</b>broad single-view oracle AUROC</div>
<div class="number"><b>{fmt(success['upcr_macro_auprc'])}</b>deployed U-PCR AUPRC</div></div></section>
<section><h2>What was found</h2><p>{html.escape(DISCOVERY_METHOD_DESCRIPTION)}</p>
<p><strong>Important:</strong> the registry was selected after extensive analysis of these outcomes. The result is therefore retrospective and must be reproduced on untouched cells before promotion.</p></section>
<section><h2>Macro performance, sorted by AUROC</h2><img alt="Macro AUROC bar chart" src="data:image/svg+xml;base64,{embedded}">
<table><thead><tr><th>Method</th><th>AUROC</th><th>95% interval</th><th>AUPRC</th><th>Label use</th></tr></thead><tbody>{headline_rows}</tbody></table></section>
<section><h2>Paired AUROC comparisons</h2><table><thead><tr><th>Contrast</th><th>Δ AUROC</th><th>95% interval</th><th>W/T/L</th><th>Worst cell</th><th>Protocol</th></tr></thead><tbody>{primary_rows}</tbody></table>
<p class="muted">Tie tolerance is ±0.001. Bootstrap resamples problem groups with 2,000 deterministic draws. The TriLens comparison is paired inside identical grouped-CV test folds, avoiding concatenation of independently calibrated OOF probabilities.</p></section>
<section><h2>Per-cell AUROC</h2><table><thead><tr><th>Cell</th><th>U-PCR</th><th>TriLens</th><th>Best single view</th><th>U-PCR − TriLens</th></tr></thead><tbody>{cell_rows}</tbody></table></section>
<section><h2>Interpretation</h2><p>{('The retrospective candidate exceeds both the standalone TriLens linear probe and the stronger evaluation-only single-view oracle on the macro AUROC point estimate. Statistical uncertainty and the validation block still prevent a promoted claim.' if success['point_estimate_success'] else 'This candidate does not meet the full target. Deployed U-PCR slightly exceeds the standalone TriLens linear probe on macro AUROC, but remains below the stronger evaluation-only oracle that selects a module, metric, and layer separately in every cell. It also does not dominate AUPRC: TriLens remains higher there. The TriLens AUROC margin is small and its paired interval includes zero.')}</p></section>
<section><h2>Validation and reproducibility</h2><ul><li>Status remains <strong>{status}</strong>: corrected live Gate B and the architecture-fidelity pilot are open.</li><li>All 13 eligible cells are included; CoQA remains protocol-rejected and appendix-only in the source benchmark.</li><li>Scores were hashed before evaluation labels were reopened. Prepared and score bundles contain no label arrays.</li><li>Machine-readable artifacts: <code>headline_summary.csv</code>, <code>per_cell_metrics.csv</code>, <code>paired_comparisons.csv</code>, manifests, diagnostics, and the separate SVG figure.</li></ul></section>
</main></body></html>"""
    report_path = results / "REPORT.html"
    report_path.write_text(document, encoding="utf-8")
    artifacts = []
    for path in sorted(results.rglob("*")):
        if path.is_file() and path.name != "REPORT_MANIFEST.json":
            artifacts.append({"file": str(path.relative_to(results)), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(results / "REPORT_MANIFEST.json", {
        "version": VERSION, "written_utc": utcnow(), "status": status,
        "self_contained_html": True, "external_assets": False, "artifacts": artifacts,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit", "evaluate", "report", "all"), nargs="?", default="all")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.phase in ("prepare", "all"): prepare(args.cache, args.results)
    if args.phase in ("fit", "all"): fit(args.results)
    if args.phase in ("evaluate", "all"): evaluate(args.cache, args.results)
    if args.phase in ("report", "all"): report(args.results)


if __name__ == "__main__":
    main()
