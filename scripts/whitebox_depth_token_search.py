#!/usr/bin/env python3
"""Retrospective depth-fusion screen for layer-local token dynamics."""

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
from sklearn.metrics import average_precision_score, roc_auc_score


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
    save_feature_matrix,
    sha256_file,
    validate_and_join,
)
from spectral_utils.paper_benchmark_suite import standardize  # noqa: E402
from spectral_utils.whitebox_depth_token_metrics import (  # noqa: E402
    EXTRACTORS,
    TOKEN_METRIC_REGISTRY,
    registry_hash,
)
from spectral_utils.whitebox_layer_fusion import FeatureMatrix, fit_controls, fit_core_spectral  # noqa: E402


VERSION = "whitebox-depth-token-search-v1-2026-08-13"
DEFAULT_BASE_RESULTS = REPO / "results" / "whitebox_layer_fusion_v2"
DEFAULT_RESULTS = REPO / "results" / "whitebox_depth_token_search_v1"
SOURCE_FILES = (
    "scripts/whitebox_depth_token_search.py",
    "spectral_utils/whitebox_depth_token_metrics.py",
    "spectral_utils/whitebox_layer_fusion.py",
    "spectral_utils/paper_benchmark_suite.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
)
BASELINE_KEYS = {
    "final_nll": "final_layer_nll__resid-core-L__all__flat",
    "generation_entropy": "generation_entropy_mean__raw-output__full-answer__flat",
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


def _rows_path(results: Path, cell: str) -> Path:
    return results / "prepared" / f"{cell}__rows.npz"


def _matrix_path(results: Path, cell: str, contract: str) -> Path:
    return results / "prepared" / f"{cell}__{contract}.npz"


def _score_path(results: Path, cell: str) -> Path:
    return results / "scores" / f"{cell}.npz"


def load_matrix(path: Path) -> FeatureMatrix:
    with np.load(path, allow_pickle=False) as bundle:
        forbidden = [key for key in bundle.files if "label" in key.lower() or key.lower() == "y"]
        if forbidden:
            raise RuntimeError(f"label-like arrays in {path}: {forbidden}")
        return FeatureMatrix(
            values=bundle["values"],
            feature_names=tuple(bundle["feature_names"].astype(str)),
            risk_anchor=bundle["risk_anchor"],
            groups=tuple(bundle["groups"].astype(str)),
            protocol_signature=str(bundle["protocol_signature"].item()),
            metadata=json.loads(str(bundle["metadata_json"].item())),
        )


def _matrix_hash(matrix: FeatureMatrix) -> str:
    X, keep, mean, scale = standardize(matrix.values)
    digest = hashlib.sha256()
    for array in (X, keep, mean, scale, matrix.risk_anchor):
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _source_index(base_results: Path) -> dict[str, Mapping[str, Any]]:
    manifest = read_json(base_results / "SOURCE_FREEZE_MANIFEST.json")
    return {str(row["local_path"]): row for row in manifest["sources"]}


def phase_prepare(cache_root: Path, base_results: Path, results: Path) -> None:
    prepared = results / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    write_json(results / "PREREGISTRATION.json", {
        "version": VERSION,
        "written_utc": utcnow(),
        "analysis_role": "retrospective_discovery_iteration_2",
        "iteration_1_result": "no fusion beat its best-layer oracle",
        "metric_registry": TOKEN_METRIC_REGISTRY,
        "metric_registry_sha256": registry_hash(),
        "methods": ["equal_mean", "iu_pcr", "dufs_liu_pcr"],
        "primary_method": "iu_pcr",
        "success_rule": (
            "fusion must beat its own per-cell best-layer oracle with positive paired macro-AUROC CI "
            "and exceed the local TriLens grouped-L2-probe macro AUROC; independent confirmation remains required"
        ),
        "validation_status": "PRELIMINARY / VALIDATION BLOCKED",
    })
    source_index = _source_index(base_results)
    sources, structural, manifest_rows = [], [], []
    for cell, spec in CELLS.items():
        print(f"[token-prepare] {cell}", flush=True)
        raw_path, sidecar_path = cache_root / spec["raw"], cache_root / spec["sidecar"]
        for path in (raw_path, sidecar_path):
            frozen = source_index.get(str(path.resolve()))
            if frozen is None or path.stat().st_size != int(frozen["local_size"]):
                raise RuntimeError(f"{path}: differs from v2 source freeze")
            sources.append({"cell": cell, "path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": frozen["local_sha256"]})
        raw, sidecar = load_pickle(raw_path), load_pickle(sidecar_path)
        joined, audit = validate_and_join(
            raw, sidecar, cell_id=cell,
            expected_model=spec.get("model", MODEL),
            expected_n_layers=int(spec.get("n_layers", 32)),
            expected_hidden_size=int(spec.get("hidden_size", 4096)),
            exclude_invalid=True, require_geometry_finite=False,
        )
        if audit["n_rows"] != len(joined.row_ids):
            raise RuntimeError(f"{cell}: join audit mismatch")
        for contract, extractor in EXTRACTORS.items():
            matrix = extractor(joined)
            save_feature_matrix(_matrix_path(results, cell, contract), matrix)
            X, keep, _mean, _scale = standardize(matrix.values)
            corr = np.corrcoef(X, rowvar=False)
            upper = np.abs(corr[np.triu_indices(corr.shape[0], 1)])
            eigen = np.maximum(np.linalg.eigvalsh((X.T @ X) / len(X)), 0.0)
            p = eigen / max(float(np.sum(eigen)), 1e-12)
            erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
            structural.append({
                "cell": cell, "contract": contract, "n_samples": matrix.n_samples,
                "n_layers": matrix.n_features, "median_abs_layer_correlation": float(np.median(upper)),
                "effective_rank": erank, "effective_rank_fraction": erank / len(keep),
                "standardized_matrix_sha256": _matrix_hash(matrix), "outcomes_seen": False,
            })
        np.savez_compressed(
            _rows_path(results, cell),
            row_ids=np.asarray(joined.row_ids, dtype="U"),
            problem_ids=np.asarray(joined.problem_ids, dtype="U"),
            n_gen_tokens=np.asarray(joined.n_gen_tokens, dtype=np.int64),
            protocol_signature=np.asarray(joined.protocol_signature),
        )
        del raw, sidecar, joined
    for path in sorted(prepared.glob("*.npz")):
        with np.load(path, allow_pickle=False) as bundle:
            fields = list(bundle.files)
            if any("label" in field.lower() or field.lower() == "y" for field in fields):
                raise RuntimeError(f"label-like prepared field in {path}")
        manifest_rows.append({"file": str(path.relative_to(results)), "bytes": path.stat().st_size, "sha256": sha256_file(path), "fields": fields})
    write_json(results / "SOURCE_FREEZE_MANIFEST.json", {"version": VERSION, "sources": sources})
    write_json(results / "PREPARED_FEATURE_MANIFEST.json", {"version": VERSION, "labels_present": False, "n_files": len(manifest_rows), "files": manifest_rows})
    write_json(results / "RUN_DEFINITION.json", {
        "version": VERSION, "cells": list(CELLS), "primary_cells": list(PRIMARY_CELLS),
        "contracts": list(EXTRACTORS), "metric_registry_sha256": registry_hash(),
        "source_sha256": {path: sha256_file(REPO / path) for path in SOURCE_FILES},
        "base_results": str(base_results.resolve()),
    })
    write_csv(results / "structural_diagnostics.csv", structural)
    print("[token-prepare] label-free token-dynamic matrices frozen", flush=True)


def verify_prepared(results: Path) -> None:
    definition = read_json(results / "RUN_DEFINITION.json")
    if definition["metric_registry_sha256"] != registry_hash():
        raise RuntimeError("token metric registry changed")
    for relative, expected in definition["source_sha256"].items():
        if sha256_file(REPO / relative) != expected:
            raise RuntimeError(f"registered source changed: {relative}")
    manifest = read_json(results / "PREPARED_FEATURE_MANIFEST.json")
    observed = [str(path.relative_to(results)) for path in sorted((results / "prepared").glob("*.npz"))]
    if manifest["labels_present"] is not False or observed != [row["file"] for row in manifest["files"]]:
        raise RuntimeError("prepared freeze roster changed")
    for row in manifest["files"]:
        path = results / row["file"]
        if path.stat().st_size != row["bytes"] or sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"prepared feature changed: {path}")


def phase_fit(results: Path) -> None:
    verify_prepared(results)
    (results / "scores").mkdir(exist_ok=True)
    (results / "diagnostics").mkdir(exist_ok=True)
    manifest = []
    for cell in CELLS:
        print(f"[token-fit] {cell}", flush=True)
        with np.load(_rows_path(results, cell), allow_pickle=False) as bundle:
            metadata = {key: np.asarray(bundle[key]) for key in bundle.files}
        scores, diagnostics = {}, {"cell": cell, "labels_seen_during_fit": False, "contracts": {}}
        for contract in EXTRACTORS:
            matrix = load_matrix(_matrix_path(results, cell, contract))
            controls, control_diag = fit_controls(matrix)
            core, core_diag = fit_core_spectral(matrix, methods=("iu_pcr", "dufs_liu_pcr"))
            scores[f"{contract}__equal_mean"] = controls["equal_mean"]
            scores[f"{contract}__iu_pcr"] = core["iu_pcr"]
            scores[f"{contract}__dufs_liu_pcr"] = core["dufs_liu_pcr"]
            diagnostics["contracts"][contract] = {
                "standardized_matrix_sha256": _matrix_hash(matrix),
                "protocol_signature": matrix.protocol_signature,
                "n_layers": matrix.n_features,
                "controls": control_diag,
                "core": core_diag,
            }
        score_path = _score_path(results, cell)
        np.savez_compressed(score_path, **metadata, **scores)
        diagnostic_path = results / "diagnostics" / f"{cell}.json"
        write_json(diagnostic_path, diagnostics)
        manifest.append({
            "cell": cell, "score_file": str(score_path.relative_to(results)), "score_sha256": sha256_file(score_path),
            "diagnostic_file": str(diagnostic_path.relative_to(results)), "diagnostic_sha256": sha256_file(diagnostic_path),
            "n_rows": len(metadata["row_ids"]), "n_methods": len(scores),
        })
    write_json(results / "FIT_COMPLETE.json", {
        "version": VERSION, "written_utc": utcnow(), "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True, "score_manifest": manifest,
    })
    print("[token-fit] score hashes frozen; outcomes remain unopened", flush=True)


def verify_scores(results: Path) -> dict[str, dict[str, np.ndarray]]:
    verify_prepared(results)
    fit = read_json(results / "FIT_COMPLETE.json")
    if fit["labels_seen_during_fit"] is not False or fit["scores_frozen_before_labels"] is not True:
        raise RuntimeError("fit leakage attestation failed")
    bundles = {}
    for row in fit["score_manifest"]:
        score_path, diagnostic_path = results / row["score_file"], results / row["diagnostic_file"]
        if sha256_file(score_path) != row["score_sha256"] or sha256_file(diagnostic_path) != row["diagnostic_sha256"]:
            raise RuntimeError(f"score freeze changed: {row['cell']}")
        if read_json(diagnostic_path)["labels_seen_during_fit"] is not False:
            raise RuntimeError("fit diagnostic leakage attestation failed")
        with np.load(score_path, allow_pickle=False) as bundle:
            if any("label" in key.lower() or key.lower() == "y" for key in bundle.files):
                raise RuntimeError("label-like score field")
            bundles[row["cell"]] = {key: np.asarray(bundle[key]) for key in bundle.files}
    freeze = {
        "version": VERSION, "written_utc": utcnow(), "labels_seen_during_fit": False,
        "scores_frozen_before_labels": True, "fit_complete_sha256": sha256_file(results / "FIT_COMPLETE.json"),
        "run_definition_sha256": sha256_file(results / "RUN_DEFINITION.json"),
        "source_freeze_sha256": sha256_file(results / "SOURCE_FREEZE_MANIFEST.json"),
        "prepared_feature_manifest_sha256": sha256_file(results / "PREPARED_FEATURE_MANIFEST.json"),
        "score_manifest": fit["score_manifest"],
    }
    write_json(results / "SCORE_FREEZE_MANIFEST.json", freeze)
    return bundles


def _base_scores(base_results: Path, cell: str) -> dict[str, np.ndarray]:
    fit = read_json(base_results / "FIT_COMPLETE.json")
    row = next(row for row in fit["score_manifest"] if row["cell"] == cell)
    path = base_results / row["score_file"]
    if sha256_file(path) != row["score_sha256"]:
        raise RuntimeError(f"base score changed: {cell}")
    with np.load(path, allow_pickle=False) as bundle:
        return {key: np.asarray(bundle[key]) for key in bundle.files}


def metric_pair(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) != 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, score)), float(average_precision_score(y, score))


def _best_layer(matrix: FeatureMatrix, y: np.ndarray) -> tuple[np.ndarray, str, float, float]:
    X, keep, _mean, _scale = standardize(matrix.values)
    candidates = []
    for position, original in enumerate(keep):
        score = X[:, position]
        # Feature columns were already sign-oriented against the unlabeled
        # anchor.  Evaluation outcomes choose only the strongest layer.
        auc, ap = metric_pair(y, score)
        candidates.append((auc, ap, matrix.feature_names[int(original)], score))
    auc, ap, name, score = max(candidates, key=lambda row: row[0])
    return np.asarray(score), str(name), float(auc), float(ap)


def _trilens_probe(base_results: Path) -> dict[str, dict[str, float]]:
    diagnostics = read_json(base_results / "supervised_grouped_cv_diagnostics.json")
    return {
        cell: {
            "auroc": float(np.mean([row["auroc"] for row in diagnostics[cell]["trilens_supervised_lr"]])),
            "auprc": float(np.mean([row["auprc"] for row in diagnostics[cell]["trilens_supervised_lr"]])),
        }
        for cell in PRIMARY_CELLS
    }


def phase_evaluate(cache_root: Path, base_results: Path, results: Path) -> None:
    bundles = verify_scores(results)
    # No raw outcome cache is opened above this line.
    metrics: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    boot: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    per_cell, oracle_rows, draw_manifest = [], [], {}
    for cell, spec in CELLS.items():
        print(f"[token-evaluate] {cell}", flush=True)
        bundle, base = bundles[cell], _base_scores(base_results, cell)
        row_ids, groups = bundle["row_ids"].astype(str), bundle["problem_ids"].astype(str)
        if not np.array_equal(row_ids, base["row_ids"].astype(str)):
            raise RuntimeError(f"{cell}: row order mismatch")
        y = load_evaluation_labels(load_pickle(cache_root / spec["raw"]), row_ids)
        prevalence = float(np.mean(y))
        scores = {f"baseline__{name}": np.asarray(base[key], dtype=float) for name, key in BASELINE_KEYS.items()}
        scores.update({key: np.asarray(value, dtype=float) for key, value in bundle.items() if key not in {"row_ids", "problem_ids", "n_gen_tokens", "protocol_signature"}})
        for contract in EXTRACTORS:
            oracle_score, selected, auc, ap = _best_layer(load_matrix(_matrix_path(results, cell, contract)), y)
            scores[f"{contract}__best_layer_oracle"] = oracle_score
            oracle_rows.append({"cell": cell, "contract": contract, "selected_feature": selected, "auroc": auc, "auprc": ap, "selection_uses_evaluation_outcomes": True})
        cell_seed = SEED + int(hashlib.sha256(cell.encode()).hexdigest()[:8], 16)
        indices, draw_hash = group_bootstrap_indices(groups, draws=BOOTSTRAP_DRAWS, seed=cell_seed)
        draw_manifest[cell] = {"seed": cell_seed, "draw_hash": draw_hash}
        for method, score in scores.items():
            auc, ap = metric_pair(y, score)
            metrics[method][cell] = {"auroc": auc, "auprc": ap}
            per_cell.append({
                "cell": cell, "dataset": spec["dataset"], "model": spec.get("model", MODEL), "method": method,
                "auroc": auc, "auprc": ap, "prevalence": prevalence, "n_samples": len(y),
                "n_groups": len(np.unique(groups)), "analysis_role": (
                    "appendix_protocol_rejected" if cell not in PRIMARY_CELLS else
                    "evaluation_only_oracle" if method.endswith("best_layer_oracle") else
                    "existing_baseline" if method.startswith("baseline__") else "retrospective_discovery_iteration_2"
                ),
            })
            auc_draws, ap_draws = [], []
            for index in indices:
                draw_auc, draw_ap = metric_pair(y[index], score[index])
                auc_draws.append(draw_auc); ap_draws.append(draw_ap)
            boot[method][cell] = {"auroc": np.asarray(auc_draws), "auprc": np.asarray(ap_draws)}
    headline = []
    eligible = [method for method in metrics if all(cell in metrics[method] for cell in PRIMARY_CELLS)]
    for method in eligible:
        row = {"method": method, "n_cells": len(PRIMARY_CELLS)}
        for metric in ("auroc", "auprc"):
            point = float(np.mean([metrics[method][cell][metric] for cell in PRIMARY_CELLS]))
            draws = np.mean(np.vstack([boot[method][cell][metric] for cell in PRIMARY_CELLS]), axis=0)
            low, high = np.quantile(draws, (0.025, 0.975))
            row[f"macro_{metric}"] = point; row[f"macro_{metric}_ci_low"] = float(low); row[f"macro_{metric}_ci_high"] = float(high)
        row["analysis_role"] = "evaluation_only_oracle" if method.endswith("best_layer_oracle") else "existing_baseline" if method.startswith("baseline__") else "retrospective_discovery_iteration_2"
        headline.append(row)
    trilens = _trilens_probe(base_results)
    headline.append({
        "method": "original_proxy__trilens_grouped_l2_probe", "n_cells": len(PRIMARY_CELLS),
        "macro_auroc": float(np.mean([trilens[cell]["auroc"] for cell in PRIMARY_CELLS])), "macro_auroc_ci_low": "", "macro_auroc_ci_high": "",
        "macro_auprc": float(np.mean([trilens[cell]["auprc"] for cell in PRIMARY_CELLS])), "macro_auprc_ci_low": "", "macro_auprc_ci_high": "",
        "analysis_role": "supervised_original_method_approximation",
    })
    headline.sort(key=lambda row: float(row["macro_auroc"]), reverse=True)
    paired = []
    for contract in EXTRACTORS:
        for fusion in ("iu_pcr", "dufs_liu_pcr"):
            lhs = f"{contract}__{fusion}"
            for rhs in (f"{contract}__best_layer_oracle", f"{contract}__equal_mean"):
                for metric in ("auroc", "auprc"):
                    cell_delta = np.asarray([metrics[lhs][cell][metric] - metrics[rhs][cell][metric] for cell in PRIMARY_CELLS])
                    draw_delta = np.mean(np.vstack([boot[lhs][cell][metric] - boot[rhs][cell][metric] for cell in PRIMARY_CELLS]), axis=0)
                    low, high = np.quantile(draw_delta, (0.025, 0.975))
                    try: p_raw = float(wilcoxon(cell_delta, zero_method="pratt").pvalue)
                    except ValueError: p_raw = 1.0
                    paired.append({
                        "contrast": f"{lhs}_minus_{rhs}", "lhs": lhs, "rhs": rhs, "metric": metric,
                        "delta": float(np.mean(cell_delta)), "ci_low": float(low), "ci_high": float(high),
                        "wins": int(np.sum(cell_delta > TIE_TOLERANCE)), "ties": int(np.sum(np.abs(cell_delta) <= TIE_TOLERANCE)),
                        "losses": int(np.sum(cell_delta < -TIE_TOLERANCE)), "worst_cell_delta": float(np.min(cell_delta)), "p_raw": p_raw, "p_holm": "",
                    })
    for row, value in zip(paired, holm_adjust([row["p_raw"] for row in paired])):
        row["p_holm"] = value
    index = {row["method"]: row for row in headline}
    trilens_auc = float(index["original_proxy__trilens_grouped_l2_probe"]["macro_auroc"])
    success = []
    for contract in EXTRACTORS:
        oracle_auc = float(index[f"{contract}__best_layer_oracle"]["macro_auroc"])
        for fusion in ("iu_pcr", "dufs_liu_pcr"):
            key = f"{contract}__{fusion}"; auc = float(index[key]["macro_auroc"])
            contrast = next(row for row in paired if row["lhs"] == key and row["rhs"] == f"{contract}__best_layer_oracle" and row["metric"] == "auroc")
            success.append({
                "contract": contract, "fusion": fusion, "macro_auroc": auc, "best_layer_macro_auroc": oracle_auc,
                "trilens_probe_macro_auroc": trilens_auc, "delta_vs_best_layer": auc-oracle_auc, "delta_vs_trilens_probe": auc-trilens_auc,
                "paired_ci_low_vs_best_layer": contrast["ci_low"], "paired_ci_high_vs_best_layer": contrast["ci_high"],
                "discovery_success": bool(auc > oracle_auc and auc > trilens_auc and float(contrast["ci_low"]) > 0.0),
                "independent_confirmation_complete": False,
            })
    write_csv(results / "per_cell_metrics.csv", per_cell); write_csv(results / "headline_summary.csv", headline)
    write_csv(results / "paired_comparisons.csv", paired); write_csv(results / "best_layer_choices.csv", oracle_rows); write_csv(results / "success_audit.csv", success)
    write_json(results / "bootstrap_draw_manifest.json", {"draws": BOOTSTRAP_DRAWS, "root_seed": SEED, "unit": "problem_group_within_cell", "identical_draws_reused_across_methods_within_cell": True, "cells": draw_manifest})
    write_json(results / "validation_status.json", {"status": "PRELIMINARY / VALIDATION BLOCKED", "retrospective_discovery": True, "independent_confirmation_complete": False, "promotion_allowed": False})
    print("[token-evaluate] metrics complete", flush=True)


def _display(method: str) -> str:
    aliases = {"baseline__final_nll": "Final-layer NLL", "baseline__generation_entropy": "Mean generation entropy", "baseline__lens96_dufs": "Expanded lens + DUFS-LIU-PCR", "original_proxy__trilens_grouped_l2_probe": "TriLens L2 probe (supervised approximation)"}
    if method in aliases: return aliases[method]
    contract, _, arm = method.partition("__")
    return f"{TOKEN_METRIC_REGISTRY.get(contract, {}).get('display', contract)} · " + {"equal_mean":"equal mean","iu_pcr":"IU-PCR","dufs_liu_pcr":"DUFS-LIU-PCR","best_layer_oracle":"best single layer (oracle)"}.get(arm, arm)


def phase_report(results: Path) -> None:
    def rows(name: str) -> list[dict[str, str]]:
        with (results / name).open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))
    headline, success, structural = rows("headline_summary.csv"), rows("success_audit.csv"), rows("structural_diagnostics.csv")
    def table(data: Sequence[Mapping[str, Any]], fields: Sequence[tuple[str,str]]) -> str:
        return '<table><thead><tr>'+''.join(f'<th>{html.escape(label)}</th>' for _,label in fields)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(row.get(key,"")))}</td>' for key,_ in fields)+'</tr>' for row in data)+'</tbody></table>'
    width, left, height = 980, 330, max(420, 45 + 27*len(headline)); values=[float(row['macro_auroc']) for row in headline]; lo=min(.45,min(values)-.02); hi=max(.82,max(values)+.02)
    xpos=lambda v:left+(v-lo)/(hi-lo)*(width-left-25)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img"><style>text{{font-family:system-ui;fill:#172033}}.b{{fill:#4f6bed}}.o{{fill:#b26a00}}.s{{fill:#8b5cf6}}</style>']
    for i,row in enumerate(headline):
        y=25+27*i; v=float(row['macro_auroc']); cls='s' if 'supervised' in row['analysis_role'] else 'o' if 'oracle' in row['analysis_role'] else 'b'; svg += [f'<text x="6" y="{y+3}" font-size="10">{html.escape(_display(row["method"]))}</text>',f'<rect class="{cls}" x="{xpos(lo):.1f}" y="{y-9}" width="{max(1,xpos(v)-xpos(lo)):.1f}" height="13" rx="3"/>',f'<text x="{xpos(v)+4:.1f}" y="{y+2}" font-size="10">{v:.4f}</text>']
    svg.append('</svg>'); figure='\n'.join(svg); (results/'figures').mkdir(exist_ok=True); (results/'figures'/'macro_auroc_sorted.svg').write_text(figure,encoding='utf-8')
    passed=any(row['discovery_success'].lower()=='true' for row in success)
    report=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Depth token-dynamic search</title><style>:root{{--bg:#f6f8fb;--card:#fff;--text:#172033;--line:#d9deea}}@media(prefers-color-scheme:dark){{:root{{--bg:#10131a;--card:#181d27;--text:#eef2ff;--line:#343c4d}}}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui}}main{{max-width:1220px;margin:auto;padding:28px}}section{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin:16px 0;overflow:auto}}.badge{{display:inline-block;padding:6px 10px;border-radius:99px;background:#ffedd5;color:#9a3412;font-weight:750}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}figure{{min-width:760px}}svg{{width:100%;height:auto}}</style></head><body><main><span class="badge">PRELIMINARY / VALIDATION BLOCKED</span><h1>Depth-distributed token-dynamic metric search</h1><p>Retrospective discovery iteration 2. Layer-local token dynamics were frozen and fitted without labels after iteration 1 showed that token means were not broadly distributed sensors.</p><section><h2>Outcome</h2><p>{'At least one candidate passed the retrospective numeric screen, but independent confirmation remains mandatory.' if passed else 'No candidate passed the strict screen.'}</p>{table(success,[('contract','Metric'),('fusion','Fusion'),('macro_auroc','AUROC'),('best_layer_macro_auroc','Best layer'),('trilens_probe_macro_auroc','TriLens probe'),('delta_vs_best_layer','Δ layer'),('delta_vs_trilens_probe','Δ TriLens'),('paired_ci_low_vs_best_layer','CI low'),('paired_ci_high_vs_best_layer','CI high'),('discovery_success','Pass')])}</section><section><h2>Methods sorted by macro AUROC</h2><figure>{figure}</figure></section><section><h2>Exact headline values</h2>{table(headline,[('method','Method'),('macro_auroc','AUROC'),('macro_auroc_ci_low','low'),('macro_auroc_ci_high','high'),('macro_auprc','AUPRC'),('analysis_role','Role')])}</section><section><h2>Frozen metrics</h2>{table([{'metric':k,**v} for k,v in TOKEN_METRIC_REGISTRY.items()],[('metric','Metric'),('formula','Formula'),('risk_direction','Direction'),('readout','Readout')])}</section><section><h2>Structural depth coverage</h2>{table(structural,[('cell','Cell'),('contract','Metric'),('n_layers','Layers'),('median_abs_layer_correlation','Median |corr|'),('effective_rank','Effective rank'),('effective_rank_fraction','Rank fraction')])}</section><section><h2>Claim boundary</h2><ul><li>13-cell equal-cell macro, not pooled candidates.</li><li>2,000 problem-group bootstrap draws shared across methods per cell.</li><li>Best-layer selection is evaluation-only and deliberately optimistic.</li><li>TriLens comparison is the local grouped-CV L2-probe approximation.</li><li>Outcomes were historically available before this second iteration; confirmation on new data is required.</li></ul></section></main></body></html>'''
    (results/'REPORT.html').write_text(report,encoding='utf-8')
    artifacts=[]
    for path in sorted(results.rglob('*')):
        if path.is_file() and path.name!='REPORT_MANIFEST.json': artifacts.append({'file':str(path.relative_to(results)),'bytes':path.stat().st_size,'sha256':sha256_file(path)})
    write_json(results/'REPORT_MANIFEST.json',{'version':VERSION,'written_utc':utcnow(),'artifacts':artifacts})
    print(f"[token-report] wrote {results/'REPORT.html'}",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=('prepare','fit','evaluate','report','all')); parser.add_argument('--cache-root',type=Path,default=DEFAULT_CACHE); parser.add_argument('--base-results',type=Path,default=DEFAULT_BASE_RESULTS); parser.add_argument('--results-dir',type=Path,default=DEFAULT_RESULTS); args=parser.parse_args()
    cache_root,base_results,results=args.cache_root.resolve(),args.base_results.resolve(),args.results_dir.resolve()
    if args.phase=='prepare': phase_prepare(cache_root,base_results,results)
    elif args.phase=='fit': phase_fit(results)
    elif args.phase=='evaluate': phase_evaluate(cache_root,base_results,results)
    elif args.phase=='report': phase_report(results)
    else:
        for phase in ('prepare','fit','evaluate','report'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),phase,'--cache-root',str(cache_root),'--base-results',str(base_results),'--results-dir',str(results)],cwd=REPO,check=True)


if __name__=='__main__': main()
