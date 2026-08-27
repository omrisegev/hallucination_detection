#!/usr/bin/env python3
"""Frozen retrospective transfer panels for Pooled Graph-Roughness V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.harp_global_contribution_teacher import (  # noqa: E402
    PROCESS_MODELS,
    PROCESS_SUBSETS,
    SEMGRAD_DATASETS,
    process_items,
    telemetry_only,
)
from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
    resolve_data_path,
)
from scripts.neutral_residual_mode_hle_confirmation import (  # noqa: E402
    DEFAULT_JUDGE_MANIFEST,
    DEFAULT_LABELS,
    DEFAULT_RAW as DEFAULT_HLE_RAW,
    read_jsonl,
    telemetry_payload as hle_telemetry,
)
from scripts.neutral_residual_mode_prmbench_confirmation import (  # noqa: E402
    DEFAULT_RAW as DEFAULT_PRM_RAW,
    ordered_eligible_rows,
    telemetry_payload as prm_telemetry,
)
from scripts.pooled_graph_roughness_controls import (  # noqa: E402
    DEFAULT_OUT as CONTROLS_OUT,
    read_selection,
    verify_controls,
)
from scripts.pooled_graph_roughness_fit import (  # noqa: E402
    DEFAULT_BUNDLE,
    DEFAULT_OUT as DEVELOPMENT_OUT,
    VERSION as DEVELOPMENT_VERSION,
    canonical_hash,
    sha256_file,
    write_json,
)
from scripts.pooled_graph_roughness_report import verify_fit  # noqa: E402
from spectral_utils.family_residual_graph import (  # noqa: E402
    fit_family_residual_state,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


VERSION = "pooled-graph-roughness-external-v1-2026-08-23"
DEFAULT_ROOT = REPO / "results" / "pooled_graph_roughness_external_v1"
DEFAULT_PRM_NRM = (
    REPO / "results" / "neutral_residual_mode_prmbench_v1"
    / "FROZEN_SCORES.npz"
)
DEFAULT_HLE_NRM = (
    REPO / "results" / "neutral_residual_mode_hle_v1"
    / "FROZEN_SCORES.npz"
)
BOOTSTRAP_SEED = 20260823
NODE_BOOTSTRAPS = 2000


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def control_args(args):
    class Values:
        pass
    values = Values()
    values.out = args.controls
    values.development = args.development
    values.bundle = args.bundle
    return values


def direction_registry(args):
    development_complete = verify_fit(args.development, args.bundle)
    controls_manifest = verify_controls(control_args(args))
    selection, _, selection_path, _ = read_selection(args.development)
    output = {
        "primary_one_se": {
            "direction": selection["direction"],
            "trust_factor": selection["selected_config"]["trust_factor"],
            "lambda": selection["selected_config"]["lambda"],
        },
        "max_mean_sensitivity": {
            "direction": selection["max_mean_sensitivity_direction"],
            "trust_factor": selection["max_mean_sensitivity_config"][
                "trust_factor"
            ],
            "lambda": selection["max_mean_sensitivity_config"]["lambda"],
        },
    }
    matched_lambda = float(selection["selected_config"]["lambda"])
    matched_trust = float(selection["selected_config"]["trust_factor"])
    for method in (
        "dufs_graph", "contribution_graph", "equal_cell_pooling", "cross_only"
    ):
        payload = json.loads(
            (args.controls / "calibrations" / f"{method}.json").read_text()
        )
        row = payload[f"exclude=none__lambda={matched_lambda:g}"]
        output[method] = {
            "direction": row["direction"],
            "trust_factor": matched_trust,
            "lambda": matched_lambda,
        }
    node = json.loads((
        args.controls / "calibrations" / "matched_node_permutations.json"
    ).read_text())
    for replicate in range(20):
        row = node[f"node_permuted_{replicate:02d}__exclude=none"]
        output[f"node_permuted_{replicate:02d}"] = {
            "direction": row["direction"],
            "trust_factor": row["trust_factor"],
            "lambda": row["lambda"],
        }
    for name, row in output.items():
        direction = np.asarray(row["direction"], dtype=float)
        if direction.shape != (len(VIEW_ORDER),) or not np.isfinite(direction).all():
            raise RuntimeError(f"invalid frozen direction: {name}")
    provenance = {
        "development_fit_manifest_hash": development_complete["manifest_hash"],
        "development_selection_hash": selection["selection_hash"],
        "controls_manifest_hash": controls_manifest["manifest_hash"],
        "selection_path": str(selection_path.resolve()),
    }
    return output, provenance


def score_state(state, directions):
    families = tuple(state.contribution_space.families)
    indices = np.asarray([VIEW_ORDER.index(name) for name in families], dtype=int)
    output = {"iu": np.asarray(state.baseline, dtype=float)}
    for name, row in directions.items():
        direction = np.asarray(row["direction"], dtype=float)[indices]
        raw = np.asarray(state.residuals, dtype=float) @ direction
        scale = float(np.std(raw, ddof=0))
        if scale <= 1e-12:
            correction = np.zeros_like(state.baseline)
        else:
            correction = (
                float(row["trust_factor"]) / len(families)
            ) * raw / scale
        output[name] = np.asarray(state.baseline + correction, dtype=float)
    return output, families


def fit_matrix(telemetry, directions):
    F, names, availability, contract = mixed_v2_matrix(telemetry)
    state = fit_family_residual_state(F, names)
    scores, families = score_state(state, directions)
    return scores, tuple(names), families, availability, contract


def process_semgrad_inputs():
    rows = []
    for model in PROCESS_MODELS:
        for subset in PROCESS_SUBSETS:
            path = (
                REPO / "dataset_cache" / "repgrid" / f"pb_{model}"
                / f"processbench_{subset}.pkl"
            )
            items = process_items(path)
            resolved_path = resolve_data_path(path)
            rows.append({
                "domain": "processbench_qwen", "group": subset,
                "cell": f"{model}__{subset}", "path": resolved_path,
                "row_ids": np.asarray([key for key, _ in items]),
                "telemetry": [telemetry_only(row) for _, row in items],
            })
    root = REPO / "dataset_cache" / "repgrid" / "pb_llama31_8b"
    for subset in PROCESS_SUBSETS:
        path = root / f"processbench_{subset}.pkl"
        items = process_items(path)
        resolved_path = resolve_data_path(path)
        rows.append({
            "domain": "processbench_llama", "group": subset,
            "cell": f"llama31_8b__{subset}", "path": resolved_path,
            "row_ids": np.asarray([key for key, _ in items]),
            "telemetry": [telemetry_only(row) for _, row in items],
        })
    root = REPO / "local_cache" / "semgrad_bem_regraded"
    for dataset in SEMGRAD_DATASETS:
        path = root / f"raw_semgrad_{dataset}_T0.0_bem.pkl"
        cache = load_pickle(path)
        keys, telemetry = [], []
        for key in sorted(cache):
            candidates_ = cache[key].get("candidates")
            if not candidates_:
                continue
            keys.append(str(key))
            telemetry.append(telemetry_only(candidates_[0]))
        rows.append({
            "domain": "semgrad", "group": dataset,
            "cell": f"semgrad__{dataset}", "path": path,
            "row_ids": np.asarray(keys), "telemetry": telemetry,
        })
    return rows


def source_hashes(args):
    return {
        "script": sha256_file(Path(__file__)),
        "development_fit": sha256_file(args.development / "FIT_COMPLETE.json"),
        "development_selection": sha256_file(
            args.development / "FROZEN_SELECTION.json"
        ),
        "controls_fit": sha256_file(args.controls / "FIT_MANIFEST.json"),
        "core": sha256_file(REPO / "spectral_utils" / "pooled_graph_roughness.py"),
        "family_graph": sha256_file(REPO / "spectral_utils" / "family_residual_graph.py"),
        "contribution_module": sha256_file(
            REPO / "spectral_utils" / "contribution_subspace.py"
        ),
        "upcr_module": sha256_file(REPO / "spectral_utils" / "upcr.py"),
        "fusion_utils_module": sha256_file(
            REPO / "spectral_utils" / "fusion_utils.py"
        ),
        "dufs_liu_feature_contract": sha256_file(
            REPO / "spectral_utils" / "dufs_liu_feature_contract.py"
        ),
        "base_feature_contract": sha256_file(
            REPO / "spectral_utils" / "feature_contract.py"
        ),
        "feature_utils": sha256_file(
            REPO / "spectral_utils" / "feature_utils.py"
        ),
        "repgrid_scoring": sha256_file(
            REPO / "spectral_utils" / "repgrid_scoring.py"
        ),
        "mixed_contract": sha256_file(
            REPO / "scripts" / "leverage_balanced_processbench_transfer.py"
        ),
        "process_semgrad_loader": sha256_file(
            REPO / "scripts" / "harp_global_contribution_teacher.py"
        ),
        "prmbench_loader": sha256_file(
            REPO / "scripts" / "neutral_residual_mode_prmbench_confirmation.py"
        ),
        "hle_loader": sha256_file(
            REPO / "scripts" / "neutral_residual_mode_hle_confirmation.py"
        ),
        "family_registry": sha256_file(
            REPO / "spectral_utils" / "specrage_views.py"
        ),
    }


def fit(args):
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    (args.out / "scores").mkdir()
    directions, provenance = direction_registry(args)
    write_json(args.out / "FROZEN_DIRECTIONS.json", directions)
    score_hashes, input_hashes, roster = {}, {}, []
    if args.panel == "process_semgrad":
        inputs = process_semgrad_inputs()
        for index, row in enumerate(inputs, start=1):
            print(f"[{index}/{len(inputs)}] score {row['cell']}", flush=True)
            scores, names, families, availability, contract = fit_matrix(
                row["telemetry"], directions
            )
            path = args.out / "scores" / f"{row['cell']}.npz"
            np.savez_compressed(
                path, row_ids=row["row_ids"], feature_names=np.asarray(names),
                families=np.asarray(families), **scores,
            )
            score_hashes[row["cell"]] = sha256_file(path)
            input_hashes[str(row["path"].resolve())] = sha256_file(row["path"])
            roster.append({
                "domain": row["domain"], "group": row["group"],
                "cell": row["cell"], "n": len(row["row_ids"]),
                "path": str(row["path"].resolve()),
                "availability": availability, "contract": contract,
            })
        reference = (
            REPO / "results" / "neutral_residual_mode_cs_iu_v1"
            / "cell_results.csv"
        )
        input_hashes[str(reference.resolve())] = sha256_file(reference)
    elif args.panel == "prmbench":
        cache = load_pickle(args.raw)
        selected = ordered_eligible_rows(cache)
        telemetry = [prm_telemetry(row) for _, _, _, row in selected]
        scores, names, families, availability, contract = fit_matrix(
            telemetry, directions
        )
        path = args.out / "scores" / "prmbench.npz"
        np.savez_compressed(
            path,
            row_keys=np.asarray([row[0] for row in selected]),
            row_ids=np.asarray([row[1] for row in selected]),
            source_ids=np.asarray([row[2] for row in selected]),
            feature_names=np.asarray(names), families=np.asarray(families),
            **scores,
        )
        score_hashes["prmbench"] = sha256_file(path)
        input_hashes[str(args.raw.resolve())] = sha256_file(args.raw)
        input_hashes[str(args.nrm.resolve())] = sha256_file(args.nrm)
        roster.append({
            "domain": "prmbench", "group": "prmbench", "cell": "prmbench",
            "n": len(selected), "path": str(args.raw.resolve()),
            "availability": availability, "contract": contract,
        })
    elif args.panel == "hle":
        cache = load_pickle(args.raw)
        row_keys = sorted(cache, key=lambda value: int(value))
        telemetry = []
        for key in row_keys:
            candidate_rows = cache[key].get("candidates")
            if not candidate_rows:
                raise RuntimeError(f"missing HLE candidate: {key}")
            telemetry.append(hle_telemetry(candidate_rows[0]))
        scores, names, families, availability, contract = fit_matrix(
            telemetry, directions
        )
        path = args.out / "scores" / "hle.npz"
        np.savez_compressed(
            path, row_keys=np.asarray([int(key) for key in row_keys]),
            feature_names=np.asarray(names), families=np.asarray(families),
            **scores,
        )
        score_hashes["hle"] = sha256_file(path)
        for source in (args.raw, args.nrm):
            input_hashes[str(source.resolve())] = sha256_file(source)
        roster.append({
            "domain": "hle", "group": "hle", "cell": "hle",
            "n": len(row_keys), "path": str(args.raw.resolve()),
            "availability": availability, "contract": contract,
        })
    else:
        raise ValueError(args.panel)
    manifest = {
        "version": VERSION,
        "panel": args.panel,
        "phase": "target_telemetry_only_scores_frozen",
        "scope": "retrospective_known_outcome_stress_test",
        **provenance,
        "directions_sha256": sha256_file(args.out / "FROZEN_DIRECTIONS.json"),
        "roster": roster,
        "score_hashes": score_hashes,
        "input_hashes": input_hashes,
        "source_hashes": source_hashes(args),
        "raw_target_fields_indexed_by_fit": [],
        "uses_target_labels_for_scoring": False,
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    write_json(args.out / "FIT_MANIFEST.json", manifest)
    print(json.dumps({
        "status": manifest["phase"], "panel": args.panel,
        "cells": len(roster), "manifest_hash": manifest["manifest_hash"],
    }, indent=2))


def verify(args):
    manifest = json.loads((args.out / "FIT_MANIFEST.json").read_text())
    payload = dict(manifest)
    recorded = payload.pop("manifest_hash")
    if canonical_hash(payload) != recorded:
        raise RuntimeError("external manifest is not self-consistent")
    if manifest["version"] != VERSION or manifest["panel"] != args.panel:
        raise RuntimeError("external version/panel mismatch")
    if manifest.get("phase") != "target_telemetry_only_scores_frozen":
        raise RuntimeError("external phase changed")
    if manifest.get("uses_target_labels_for_scoring") is not False:
        raise RuntimeError("external scoring used target labels")
    directions, provenance = direction_registry(args)
    if manifest["development_fit_manifest_hash"] != provenance[
        "development_fit_manifest_hash"
    ] or manifest["development_selection_hash"] != provenance[
        "development_selection_hash"
    ] or manifest["controls_manifest_hash"] != provenance[
        "controls_manifest_hash"
    ]:
        raise RuntimeError("external/development provenance mismatch")
    if sha256_file(args.out / "FROZEN_DIRECTIONS.json") != manifest[
        "directions_sha256"
    ]:
        raise RuntimeError("external direction hash changed")
    if json.loads((args.out / "FROZEN_DIRECTIONS.json").read_text()) != directions:
        raise RuntimeError("external directions changed")
    if manifest["source_hashes"] != source_hashes(args):
        raise RuntimeError("external source hash changed")
    for cell, expected in manifest["score_hashes"].items():
        if sha256_file(args.out / "scores" / f"{cell}.npz") != expected:
            raise RuntimeError(f"external score hash changed: {cell}")
    for path, expected in manifest["input_hashes"].items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"external input hash changed: {path}")
    if manifest["raw_target_fields_indexed_by_fit"] != []:
        raise RuntimeError("external fit indexed target fields")
    return manifest, directions


def metric_rows(y, scores):
    return {
        name: {
            "auroc": float(roc_auc_score(y, values)),
            "auprc": float(average_precision_score(y, values)),
        }
        for name, values in scores.items()
    }


def clustered_bootstrap(y, scores, cluster, draws=5000):
    unique, inverse = np.unique(cluster, return_inverse=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = {name: np.empty(draws) for name in scores if name != "iu"}
    for draw in range(draws):
        counts = rng.multinomial(len(unique), np.full(len(unique), 1 / len(unique)))
        weights = counts[inverse]
        base = roc_auc_score(y, scores["iu"], sample_weight=weights)
        for name in output:
            output[name][draw] = (
                roc_auc_score(y, scores[name], sample_weight=weights) - base
            )
    return output


def stratified_bootstrap(y, scores, draws=10000):
    positive, negative = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    output = {name: np.empty(draws) for name in scores if name != "iu"}
    for draw in range(draws):
        index = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        base = roc_auc_score(y[index], scores["iu"][index])
        for name in output:
            output[name][draw] = (
                roc_auc_score(y[index], scores[name][index]) - base
            )
    return output


def interval(values):
    return [
        100 * float(np.quantile(values, .025)),
        100 * float(np.quantile(values, .975)),
    ]


def node_null_summary(y, scores, *, cluster=None):
    node_names = [f"node_permuted_{replicate:02d}" for replicate in range(20)]
    base = roc_auc_score(y, scores["iu"])
    primary_delta = roc_auc_score(y, scores["primary_one_se"]) - base
    node_delta = np.asarray([
        roc_auc_score(y, scores[name]) - base for name in node_names
    ])
    subset = {
        "iu": scores["iu"], "primary_one_se": scores["primary_one_se"],
        **{name: scores[name] for name in node_names},
    }
    if cluster is None:
        draws = stratified_bootstrap(y, subset, draws=NODE_BOOTSTRAPS)
    else:
        draws = clustered_bootstrap(
            y, subset, cluster, draws=NODE_BOOTSTRAPS
        )
    mean_node_draw = np.mean([draws[name] for name in node_names], axis=0)
    difference = draws["primary_one_se"] - mean_node_draw
    return {
        "n_permutations": len(node_names),
        "bootstrap_draws": NODE_BOOTSTRAPS,
        "mean_node_delta_pp": 100 * float(np.mean(node_delta)),
        "min_node_delta_pp": 100 * float(np.min(node_delta)),
        "max_node_delta_pp": 100 * float(np.max(node_delta)),
        "primary_minus_mean_node_pp": 100 * float(
            primary_delta - np.mean(node_delta)
        ),
        "primary_minus_mean_node_ci_pp": interval(difference),
        "randomization_p_greater_or_equal": float(
            (1 + np.sum(node_delta >= primary_delta)) / (1 + len(node_delta))
        ),
    }


def nrm_cell_reference():
    path = (
        REPO / "results" / "neutral_residual_mode_cs_iu_v1"
        / "cell_results.csv"
    )
    output = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["regime"] == "source23_transfer":
                output[row["cell"]] = row
    return output


def process_semgrad_labels(cell, frozen_path):
    frozen_path = Path(frozen_path)
    if cell.startswith("semgrad__"):
        cache = load_pickle(frozen_path)
        ids, labels = [], []
        for key in sorted(cache):
            candidates_ = cache[key].get("candidates")
            if not candidates_:
                continue
            ids.append(str(key))
            labels.append(int(candidates_[0]["bem_correct"]))
        return np.asarray(ids), np.asarray(labels, dtype=int)
    items = process_items(frozen_path)
    return (
        np.asarray([key for key, _ in items]),
        np.asarray([int(row["label"] == -1) for _, row in items], dtype=int),
    )


def report_process(args, manifest):
    references = nrm_cell_reference()
    rows = []
    for entry in manifest["roster"]:
        cell = entry["cell"]
        with np.load(args.out / "scores" / f"{cell}.npz") as stored:
            ids = stored["row_ids"].astype(str)
            scores = {
                name: stored[name].astype(float)
                for name in stored.files
                if name not in ("row_ids", "feature_names", "families")
            }
        label_ids, y = process_semgrad_labels(cell, entry["path"])
        if not np.array_equal(ids, label_ids.astype(str)):
            raise RuntimeError(f"external row mismatch: {cell}")
        metrics = metric_rows(y, scores)
        reference = references[cell]
        if len(y) != int(reference["n"]) or int(np.sum(y)) != int(
            reference["n_correct"]
        ):
            raise RuntimeError(f"external NRM roster/prevalence drift: {cell}")
        if entry["domain"] != reference["domain"] or entry["group"] != reference[
            "group"
        ]:
            raise RuntimeError(f"external NRM group/domain drift: {cell}")
        if abs(metrics["iu"]["auroc"] - float(reference["iu_auroc"])) > 1e-12:
            raise RuntimeError(f"external IU/NRM drift: {cell}")
        for name, metric in metrics.items():
            rows.append({
                "domain": entry["domain"], "group": entry["group"],
                "cell": cell, "method": name,
                "auroc": metric["auroc"], "auprc": metric["auprc"],
                "delta_vs_iu_pp": 100 * (
                    metric["auroc"] - metrics["iu"]["auroc"]
                ),
                "nrm_delta_pp": float(reference["nrm_delta_pp"]),
            })
    summaries = {}
    methods = sorted({row["method"] for row in rows if row["method"] != "iu"})
    for domain in sorted({row["domain"] for row in rows}):
        selected = [row for row in rows if row["domain"] == domain]
        groups = sorted({row["group"] for row in selected})
        summaries[domain] = {}
        for method in methods:
            values = np.asarray([
                np.mean([
                    row["delta_vs_iu_pp"] for row in selected
                    if row["group"] == group and row["method"] == method
                ]) for group in groups
            ])
            nrm = np.asarray([
                np.mean([
                    row["nrm_delta_pp"] for row in selected
                    if row["group"] == group and row["method"] == method
                ]) for group in groups
            ])
            summaries[domain][method] = {
                "equal_group_delta_pp": float(np.mean(values)),
                "positive_groups": int(np.sum(values > 0)),
                "worst_group_pp": float(np.min(values)),
                "nrm_delta_pp": float(np.mean(nrm)),
                "nrm_recovery_fraction": float(np.mean(values) / np.mean(nrm)),
            }
    return rows, summaries


def report_prm(args):
    with np.load(args.out / "scores" / "prmbench.npz") as stored:
        row_keys = stored["row_keys"].astype(int)
        row_ids = stored["row_ids"].astype(str)
        source_ids = stored["source_ids"].astype(str)
        names = stored["feature_names"].astype(str)
        families = stored["families"].astype(str)
        scores = {
            name: stored[name].astype(float) for name in stored.files
            if name not in (
                "row_keys", "row_ids", "source_ids", "feature_names", "families"
            )
        }
    cache = load_pickle(args.raw)
    selected = ordered_eligible_rows(cache)
    if row_keys.tolist() != [row[0] for row in selected]:
        raise RuntimeError("PRMBench row-key mismatch")
    if row_ids.tolist() != [row[1] for row in selected]:
        raise RuntimeError("PRMBench row-id mismatch")
    if source_ids.tolist() != [row[2] for row in selected]:
        raise RuntimeError("PRMBench source-id mismatch")
    y = np.asarray([
        str(row[3]["classification"]) == "correct" for row in selected
    ], dtype=int)
    with np.load(args.nrm) as stored:
        if not np.array_equal(stored["row_keys"].astype(int), row_keys):
            raise RuntimeError("PRMBench NRM row-key mismatch")
        if not np.array_equal(stored["row_ids"].astype(str), row_ids):
            raise RuntimeError("PRMBench NRM row-id mismatch")
        if not np.array_equal(stored["source_ids"].astype(str), source_ids):
            raise RuntimeError("PRMBench NRM source-id mismatch")
        if not np.array_equal(stored["feature_names"].astype(str), names):
            raise RuntimeError("PRMBench NRM feature mismatch")
        if not np.array_equal(stored["families"].astype(str), families):
            raise RuntimeError("PRMBench NRM family mismatch")
        if not np.allclose(
            stored["iu_correctness_score"], scores["iu"], atol=1e-12, rtol=0
        ):
            raise RuntimeError("PRMBench NRM IU baseline drift")
        scores["family_nrm"] = stored["nrm_correctness_score"].astype(float)
    metrics = metric_rows(y, scores)
    bootstrap_scores = {
        name: scores[name] for name in (
            "iu", "primary_one_se", "max_mean_sensitivity", "family_nrm",
            "dufs_graph", "contribution_graph", "equal_cell_pooling", "cross_only",
        )
    }
    draws = clustered_bootstrap(y, bootstrap_scores, source_ids)
    node_summary = node_null_summary(y, scores, cluster=source_ids)
    return (
        metrics, {name: interval(value) for name, value in draws.items()},
        draws, node_summary,
    )


def report_hle(args):
    with np.load(args.out / "scores" / "hle.npz") as stored:
        row_keys = stored["row_keys"].astype(int)
        names = stored["feature_names"].astype(str)
        families = stored["families"].astype(str)
        scores = {
            name: stored[name].astype(float) for name in stored.files
            if name not in ("row_keys", "feature_names", "families")
        }
    labels = read_jsonl(args.labels)
    judge = json.loads(args.judge_manifest.read_text())
    if judge["hashes"]["output_judgments_sha256"] != sha256_file(args.labels):
        raise RuntimeError("HLE judge manifest does not authenticate labels")
    if row_keys.tolist() != [int(row["row_key"]) for row in labels]:
        raise RuntimeError("HLE row-key mismatch")
    y = np.asarray([row["correct"] == "yes" for row in labels], dtype=int)
    with np.load(args.nrm) as stored:
        if not np.array_equal(stored["row_keys"].astype(int), row_keys):
            raise RuntimeError("HLE NRM row-key mismatch")
        if not np.array_equal(stored["feature_names"].astype(str), names):
            raise RuntimeError("HLE NRM feature mismatch")
        if not np.array_equal(stored["families"].astype(str), families):
            raise RuntimeError("HLE NRM family mismatch")
        if not np.allclose(
            stored["iu_correctness_score"], scores["iu"], atol=1e-12, rtol=0
        ):
            raise RuntimeError("HLE NRM IU baseline drift")
        scores["family_nrm"] = stored["nrm_correctness_score"].astype(float)
    metrics = metric_rows(y, scores)
    bootstrap_scores = {
        name: scores[name] for name in (
            "iu", "primary_one_se", "max_mean_sensitivity", "family_nrm",
            "dufs_graph", "contribution_graph", "equal_cell_pooling", "cross_only",
        )
    }
    draws = stratified_bootstrap(y, bootstrap_scores)
    node_summary = node_null_summary(y, scores)
    return (
        metrics, {name: interval(value) for name, value in draws.items()},
        draws, node_summary,
    )


def report(args):
    manifest, _ = verify(args)
    if args.panel == "process_semgrad":
        rows, summaries = report_process(args, manifest)
        result = {
            "version": VERSION, "panel": args.panel,
            "status": "RETROSPECTIVE_STRESS_TEST",
            "scope": manifest["scope"], "summaries": summaries,
        }
        fields = list(rows[0])
        with (args.out / "cell_results.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        if args.panel == "prmbench":
            metrics, intervals, draws, node_summary = report_prm(args)
        else:
            metrics, intervals, draws, node_summary = report_hle(args)
        base = metrics["iu"]["auroc"]
        nrm_delta = metrics["family_nrm"]["auroc"] - base
        methods = {}
        for name, row in metrics.items():
            delta = row["auroc"] - base
            methods[name] = {
                **row, "delta_vs_iu_pp": 100 * delta,
                "ci_delta_vs_iu_pp": intervals.get(name),
                "nrm_recovery_fraction": (
                    float(delta / nrm_delta) if name != "iu" else 0.0
                ),
            }
        primary_draw = draws["primary_one_se"]
        nrm_draw = draws["family_nrm"]
        observed_primary = (
            metrics["primary_one_se"]["auroc"] - base
        )
        result = {
            "version": VERSION, "panel": args.panel,
            "status": "RETROSPECTIVE_STRESS_TEST",
            "scope": manifest["scope"], "n": manifest["roster"][0]["n"],
            "methods": methods,
            "node_permutation_null": node_summary,
            "primary_d30_pp": 100 * float(observed_primary - .3 * nrm_delta),
            "primary_d30_ci_pp": interval(primary_draw - .3 * nrm_draw),
        }
        if args.panel == "hle":
            result["post_freeze_label_hashes"] = {
                "labels": sha256_file(args.labels),
                "judge_manifest": sha256_file(args.judge_manifest),
            }
    write_json(args.out / "RESULT.json", result)
    lines = [
        f"# Pooled Graph-Roughness external — {args.panel}", "",
        "**Retrospective known-outcome stress test; not independent confirmation.**",
        "",
    ]
    if args.panel == "process_semgrad":
        lines += [
            "| domain | one-SE Δ (pp) | max-mean Δ (pp) | Family-NRM Δ (pp) |",
            "|---|---:|---:|---:|",
        ]
        for domain, values in result["summaries"].items():
            primary = values["primary_one_se"]
            maximum = values["max_mean_sensitivity"]
            lines.append(
                f"| `{domain}` | {primary['equal_group_delta_pp']:+.3f} | "
                f"{maximum['equal_group_delta_pp']:+.3f} | "
                f"{primary['nrm_delta_pp']:+.3f} |"
            )
    else:
        lines += [
            "| method | AUROC | Δ vs IU (pp) | 95% CI (pp) | NRM recovery |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in (
            "iu", "primary_one_se", "max_mean_sensitivity", "family_nrm",
            "dufs_graph", "contribution_graph", "cross_only"
        ):
            row = result["methods"][name]
            ci = row["ci_delta_vs_iu_pp"]
            ci_text = "—" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
            lines.append(
                f"| `{name}` | {row['auroc']:.6f} | "
                f"{row['delta_vs_iu_pp']:+.3f} | {ci_text} | "
                f"{100*row['nrm_recovery_fraction']:.1f}% |"
            )
    lines += ["", "All target scores and hashes were frozen before this report "
              "indexed target labels. Target transforms are transductive but label-free.", ""]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2)[:12000])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("fit", "report"))
    parser.add_argument(
        "panel", choices=("process_semgrad", "prmbench", "hle")
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT_OUT)
    parser.add_argument("--controls", type=Path, default=CONTROLS_OUT)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--judge-manifest", type=Path, default=DEFAULT_JUDGE_MANIFEST)
    parser.add_argument("--nrm", type=Path)
    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_ROOT / args.panel
    if args.raw is None:
        args.raw = DEFAULT_PRM_RAW if args.panel == "prmbench" else DEFAULT_HLE_RAW
    if args.nrm is None:
        args.nrm = DEFAULT_PRM_NRM if args.panel == "prmbench" else DEFAULT_HLE_NRM
    if args.phase == "fit":
        fit(args)
    else:
        report(args)


if __name__ == "__main__":
    main()
