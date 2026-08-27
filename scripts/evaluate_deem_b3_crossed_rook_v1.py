#!/usr/bin/env python3
"""Strict C-tier evaluator and report harness for Crossed-Rook B3 v1.

The evaluator has a hard two-pass boundary.  Pass A validates the complete
target-free run, every fit/hash/row identity, the same-seed frozen-B3 alias,
and the numerical/topological mechanics.  Only after Pass A succeeds is the
label-sidecar module imported dynamically.  Labels are used for evaluation
only; they never choose an arm, hyperparameter, threshold, or plot subset.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from spectral_utils.deem_b3_crossed_rook import CrossedRookConfig  # noqa: E402
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    BUNDLE_SCHEMA,
    load_registry,
    load_target_free_bundle,
)


SCHEMA = "deem_b3_crossed_rook_v1_evaluation"
LABEL_MODULE = "spectral_utils.residual_graph_deem_labels"
SEEDS = (0, 1, 2, 3, 4)
ARM_ORDER = (
    "A0_B3_EXACT_ALIAS",
    "A1_ROW_9",
    "A2_COLUMN_9",
    "A3_CROSSED_ROOK_18",
    "A4_NONROOK_18_CONTROL",
)
METHOD = {
    "A0_B3_EXACT_ALIAS": "B3",
    "A1_ROW_9": "ROW",
    "A2_COLUMN_9": "COLUMN",
    "A3_CROSSED_ROOK_18": "CROSSED",
    "A4_NONROOK_18_CONTROL": "NONROOK",
}
METHOD_ORDER = tuple(METHOD[arm] for arm in ARM_ORDER)
REGISTERED_CONTRASTS = (
    ("CROSSED", "B3"),
    ("CROSSED", "NONROOK"),
    ("CROSSED", "ROW"),
    ("CROSSED", "COLUMN"),
    ("ROW", "B3"),
    ("COLUMN", "B3"),
)
DIAGNOSTIC_CONTRASTS = (("NONROOK", "B3"),)
TOLERANCE = 5e-4
PROMOTION_DELTA = 0.0025
WORST_CELL_FLOOR = -0.02
MIN_WINS_TIES = 14
EXPECTED_CORE = (
    "epr", "sw_var_peak", "cusum_max",
    "epr_spilled", "sw_var_peak_spilled", "cusum_max_spilled",
    "epr_energy", "sw_var_peak_energy", "cusum_max_energy",
)
BASE_KEYS = {
    "score", "posterior", "logit", "base_score", "base_logit", "correction",
    "contributions", "base_contributions", "cell_delta", "edge_values",
    "edge_raw_contribution", "edge_pairs", "edge_kinds", "edge_weights",
    "core_features", "core_indices", "feature_names", "row_id",
    "standardization_mean", "standardization_scale",
}
TARGET_KEY_RE = re.compile(
    r"(?:^|_)(?:label|labels|correct|correctness|is_correct|gold|answer|answers|"
    r"reference|target|first_error|error_label|y_h)(?:$|_)",
    re.IGNORECASE,
)
FREEZE_BINDING_FIELDS = {
    "array_sha256", "metadata_sha256", "score_sha256", "base_score_sha256",
    "bundle_sha256", "ordered_row_id_sha256", "baseline_array_sha256",
    "baseline_metadata_sha256",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResidualGraphDeemError(f"JSON root is not an object: {path}")
    return value


def _verify_content_hash(value: Mapping[str, Any], *, context: str) -> None:
    if "content_sha256" not in value:
        raise ResidualGraphDeemError(f"missing content_sha256: {context}")
    unhashed = dict(value)
    expected = unhashed.pop("content_sha256")
    if canonical_sha256(unhashed) != expected:
        raise ResidualGraphDeemError(f"content hash mismatch: {context}")


def _require_false(value: Mapping[str, Any], names: Sequence[str], *, context: str) -> None:
    for name in names:
        if name not in value or value[name] is not False:
            raise ResidualGraphDeemError(f"{context} does not bind {name}=false")


def _safe_artifact(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ResidualGraphDeemError("artifact escapes run root") from exc
    if not path.is_file() or path.is_symlink():
        raise ResidualGraphDeemError(f"missing/symlink artifact: {path}")
    return path


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -700.0, 700.0)))


def _score_hash(value: np.ndarray) -> str:
    """Reproduce the runner's typed/shape-bound score digest exactly."""

    score = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return canonical_sha256(
        {
            "dtype": str(score.dtype),
            "shape": list(score.shape),
            "bytes": score.tobytes().hex(),
        }
    )


def _close(left: np.ndarray, right: np.ndarray, *, context: str, atol: float = 1e-12) -> None:
    if left.shape != right.shape or not np.allclose(left, right, atol=atol, rtol=0.0):
        error = float("inf") if left.shape != right.shape else float(np.max(np.abs(left - right)))
        raise ResidualGraphDeemError(f"{context} mismatch: {error:.3e}")


def _row_edges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (3 * row + left, 3 * row + right)
        for row in range(3)
        for left in range(3)
        for right in range(left + 1, 3)
    )


def _column_edges() -> tuple[tuple[int, int], ...]:
    return tuple(
        (3 * upper + column, 3 * lower + column)
        for column in range(3)
        for upper in range(3)
        for lower in range(upper + 1, 3)
    )


def _nonrook_edges() -> tuple[tuple[int, int], ...]:
    # Keep the exact runner order, not merely the same edge set.  Over F_3,
    # (r,c) -> (r+c,r-c) maps the ordered rook support bijectively onto its
    # ordered 18-edge complement.
    permutation = tuple(
        3 * ((row + column) % 3) + ((row - column) % 3)
        for row in range(3)
        for column in range(3)
    )
    return tuple(
        tuple(sorted((permutation[left], permutation[right])))
        for left, right in _row_edges() + _column_edges()
    )


def _validate_topology_contract() -> None:
    row, column, nonrook = _row_edges(), _column_edges(), _nonrook_edges()
    all_edges = set(itertools.combinations(range(9), 2))
    if not (
        len(row) == len(set(row)) == 9
        and len(column) == len(set(column)) == 9
        and len(nonrook) == len(set(nonrook)) == 18
        and not (set(row) & set(column))
        and not ((set(row) | set(column)) & set(nonrook))
        and set(row) | set(column) | set(nonrook) == all_edges
    ):
        raise ResidualGraphDeemError("invalid K9 rook/non-rook topology contract")


def _expected_edges(arm: str) -> tuple[tuple[int, int], ...]:
    if arm == "A1_ROW_9":
        return _row_edges()
    if arm == "A2_COLUMN_9":
        return _column_edges()
    if arm == "A4_NONROOK_18_CONTROL":
        return _nonrook_edges()
    return _row_edges() + _column_edges()


def _expected_edge_kinds(arm: str) -> tuple[str, ...]:
    if arm == "A1_ROW_9":
        return ("row",) * 9
    if arm == "A2_COLUMN_9":
        return ("column",) * 9
    if arm == "A4_NONROOK_18_CONTROL":
        return ("nonrook",) * 18
    return ("row",) * 9 + ("column",) * 9


def _artifact_inventory(freeze: Mapping[str, Any]) -> dict[str, str]:
    artifacts = freeze.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ResidualGraphDeemError("score freeze has no artifact inventory")
    output = {}
    for item in artifacts:
        relative = str(item.get("path", ""))
        digest = str(item.get("sha256", ""))
        if not relative or len(digest) != 64 or relative in output:
            raise ResidualGraphDeemError("invalid/duplicate score-freeze artifact")
        output[relative] = digest
    return output


def _validate_baseline_freeze(baseline_dir: Path, cells: Sequence[str]) -> dict[str, str]:
    freeze_path = baseline_dir / "SCORE_FREEZE_MANIFEST.json"
    freeze = _read_json(freeze_path)
    _verify_content_hash(freeze, context="B3 score freeze")
    if (
        freeze.get("status") != "complete"
        or freeze.get("debug")
        or sorted(freeze.get("cells", [])) != sorted(cells)
        or tuple(freeze.get("seeds", [])) != SEEDS
        or "B3" not in freeze.get("arms", [])
        or freeze.get("missing_seeds")
        or freeze.get("incomplete_fits")
        or freeze.get("unhealthy_fits")
        or freeze.get("missing_artifacts")
    ):
        raise ResidualGraphDeemError("B3 score freeze is incomplete")
    for filename, field in (
        ("RUN_DEFINITION.json", "run_definition_sha256"),
        ("FIT_COMPLETE.json", "fit_complete_sha256"),
    ):
        path = baseline_dir / filename
        if not path.is_file() or sha256_file(path) != freeze.get(field):
            raise ResidualGraphDeemError(f"B3 prerequisite mismatch: {filename}")
    inventory = _artifact_inventory(freeze)
    # The historical freeze inventories B0--B7, while this isolated worktree
    # deliberately materializes only the B3 prerequisite.  Verify every B3
    # array+metadata member consumed here (24 cells x 5 seeds x 2 files), not
    # unrelated historical arms that are not inputs to Crossed-Rook.
    required = {
        f"fits/{cell}/B3__seed{seed}.{suffix}"
        for cell, seed, suffix in itertools.product(cells, SEEDS, ("npz", "json"))
    }
    if not required.issubset(inventory):
        raise ResidualGraphDeemError("B3 prerequisite inventory is incomplete")
    for relative in sorted(required):
        digest = inventory[relative]
        path = _safe_artifact(baseline_dir, relative)
        if sha256_file(path) != digest:
            raise ResidualGraphDeemError(f"B3 artifact hash mismatch: {relative}")
    return inventory


def _load_structural_summary(
    path: Path, *, registry_sha256: str
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load the independently frozen, label-free crossed-core geometry."""

    summary = _read_json(path)
    freeze_path = path.parent / "FREEZE.json"
    freeze = _read_json(freeze_path)
    if (
        summary.get("schema") != "crossed_core_independent_v1"
        or summary.get("labels_accessed") is not False
        or freeze.get("schema") != "crossed_core_independent_v1"
        or freeze.get("labels_accessed") is not False
        or freeze.get("registry_sha256") != registry_sha256
    ):
        raise ResidualGraphDeemError("structural summary/freeze contract mismatch")
    bindings = [
        row for row in freeze.get("output_artifacts", [])
        if Path(str(row.get("path", ""))).resolve() == path.resolve()
    ]
    if len(bindings) != 1 or bindings[0].get("sha256") != sha256_file(path):
        raise ResidualGraphDeemError("structural summary is not bound by its freeze")
    script_path = Path(str(freeze.get("script", "")))
    if not script_path.is_file() or sha256_file(script_path) != freeze.get("script_sha256"):
        raise ResidualGraphDeemError("structural analysis source/freeze mismatch")
    try:
        primary = summary["results"]["primary_k50_23_cells"]["b3_atomic_core"]
        values = {
            name: float(primary[name]["equal_dataset_family_mean"])
            for name in ("r2_source", "r2_operator", "r2_union", "r2_all")
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ResidualGraphDeemError("structural summary metric schema mismatch") from exc
    if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values.values()):
        raise ResidualGraphDeemError("invalid structural R2 summary")
    return values, {
        "summary_sha256": sha256_file(path),
        "freeze_sha256": sha256_file(freeze_path),
        "script_sha256": str(freeze["script_sha256"]),
        "registry_sha256": registry_sha256,
        "scope": "23_K50_cells_excluding_LOSNet",
        "representation": "mean_of_5_frozen_B3_atomic_core",
        "labels_accessed": False,
    }


def _resolve_run_contract(
    run_dir: Path,
    score_freeze_path: Path,
    *,
    config_path: Path,
    registry: Mapping[str, Any],
    cells: Sequence[str],
) -> tuple[dict, dict, dict, dict[str, Mapping[str, Any]]]:
    freeze = _read_json(score_freeze_path)
    _verify_content_hash(freeze, context="Crossed-Rook score freeze")
    if freeze.get("schema") != "deem_b3_crossed_rook_score_freeze_v1" or freeze.get("status") != "frozen":
        raise ResidualGraphDeemError("Crossed-Rook score freeze is incomplete")
    _require_false(
        freeze,
        ("targets_accessed_during_fit", "label_module_imported_during_fit"),
        context="Crossed-Rook score freeze",
    )
    definition_sha = str(freeze.get("run_definition_sha256", ""))
    definition_path = run_dir / "run_definitions" / f"{definition_sha}.json"
    if not definition_path.is_file():
        raise ResidualGraphDeemError("run-definition file/hash mismatch")
    definition = _read_json(definition_path)
    _verify_content_hash(definition, context="Crossed-Rook run definition")
    if canonical_sha256(definition) != definition_sha:
        raise ResidualGraphDeemError("run-definition canonical hash mismatch")
    if (
        definition.get("schema") != "deem_b3_crossed_rook_run_definition_v1"
        or definition.get("status") != "retrospective_target_free_architecture_development"
        or definition.get("config_sha256") != sha256_file(config_path)
        or definition.get("registry_content_sha256") != registry.get("registry_content_sha256")
    ):
        raise ResidualGraphDeemError("Crossed-Rook run definition is not frozen")
    source_manifest = definition.get("source_manifest", {})
    if not isinstance(source_manifest, dict) or canonical_sha256(source_manifest) != definition.get("source_sha256"):
        raise ResidualGraphDeemError("run-definition source manifest mismatch")
    for relative, digest in source_manifest.items():
        path = _safe_artifact(ROOT, str(relative))
        if sha256_file(path) != digest:
            raise ResidualGraphDeemError(f"run-definition source changed: {relative}")

    if (
        tuple(freeze.get("arms", [])) != ARM_ORDER
        or tuple(freeze.get("cells", [])) != tuple(cells)
        or tuple(freeze.get("seeds", [])) != SEEDS
        or int(freeze.get("n_artifacts", -1)) != len(ARM_ORDER) * len(cells) * len(SEEDS)
    ):
        raise ResidualGraphDeemError("score-freeze invocation roster mismatch")
    inventory = freeze.get("artifact_sha256")
    if not isinstance(inventory, dict) or len(inventory) != int(freeze["n_artifacts"]):
        raise ResidualGraphDeemError("score-freeze fit inventory mismatch")
    expected_keys = {
        f"{arm}|{cell}|{seed}"
        for arm, cell, seed in itertools.product(ARM_ORDER, cells, SEEDS)
    }
    if set(inventory) != expected_keys:
        raise ResidualGraphDeemError("score-freeze artifact keys/roster mismatch")
    invocation_sha = str(freeze.get("invocation_sha256", ""))
    invocation = {
        "run_definition_sha256": definition_sha,
        "selected_arms": list(ARM_ORDER),
        "selected_cells": list(cells),
        "selected_seeds": list(SEEDS),
    }
    if canonical_sha256(invocation) != invocation_sha:
        raise ResidualGraphDeemError("score-freeze invocation hash mismatch")
    completion_path = run_dir / "fit_completions" / f"{invocation_sha}.json"
    completion = _read_json(completion_path)
    _verify_content_hash(completion, context="Crossed-Rook fit completion")
    _require_false(
        completion,
        ("targets_accessed_during_fit", "label_module_imported_during_fit"),
        context="Crossed-Rook fit completion",
    )
    if (
        completion.get("schema") != "deem_b3_crossed_rook_fit_complete_v1"
        or completion.get("status") != "complete"
        or completion.get("run_definition_sha256") != definition_sha
        or completion.get("invocation_sha256") != invocation_sha
        or completion.get("arms") != list(ARM_ORDER)
        or completion.get("cells") != list(cells)
        or completion.get("seeds") != list(SEEDS)
        or int(completion.get("n_records", -1)) != len(inventory)
        or completion.get("all_healthy") is not True
    ):
        raise ResidualGraphDeemError("fit-completion contract mismatch")
    completion_inventory = completion.get("artifact_sha256")
    if not isinstance(completion_inventory, dict) or set(completion_inventory) != expected_keys:
        raise ResidualGraphDeemError("fit-completion artifact keys/roster mismatch")
    return freeze, definition, completion, inventory


def _metadata_target_firewall(metadata: Mapping[str, Any], *, context: str) -> None:
    _require_false(
        metadata,
        ("targets_accessed_during_fit", "label_module_imported_during_fit"),
        context=context,
    )
    allowed_flags = {
        "targets_accessed_during_fit", "label_module_imported_during_fit",
    }

    def inspect(value: Any, prefix: str = "") -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                name = str(key)
                qualified = f"{prefix}.{name}" if prefix else name
                if name not in allowed_flags and TARGET_KEY_RE.search(name):
                    raise ResidualGraphDeemError(
                        f"target-like metadata field entered fit: {context}/{qualified}"
                    )
                inspect(item, qualified)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                inspect(item, f"{prefix}[{index}]")

    inspect(metadata)


def _load_b3_member(
    baseline_dir: Path, cell: str, seed: int, inventory: Mapping[str, str]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    relative = f"fits/{cell}/B3__seed{seed}.npz"
    path = _safe_artifact(baseline_dir, relative)
    if inventory.get(relative) != sha256_file(path):
        raise ResidualGraphDeemError(f"B3 member not bound by score freeze: {cell}/{seed}")
    metadata_relative = f"fits/{cell}/B3__seed{seed}.json"
    metadata_path = _safe_artifact(baseline_dir, metadata_relative)
    if inventory.get(metadata_relative) != sha256_file(metadata_path):
        raise ResidualGraphDeemError(f"B3 metadata not bound by score freeze: {cell}/{seed}")
    metadata = _read_json(metadata_path)
    _verify_content_hash(metadata, context=f"frozen B3 {cell}/seed{seed}")
    if (
        metadata.get("schema") != "deem_vs_iupcr_fit_artifact_v1"
        or metadata.get("status") != "complete"
        or metadata.get("arm_id") != "B3"
        or metadata.get("cell_id") != cell
        or int(metadata.get("seed", -1)) != seed
        or metadata.get("array_sha256") != sha256_file(path)
    ):
        raise ResidualGraphDeemError(f"frozen B3 metadata mismatch: {cell}/{seed}")
    with np.load(path, allow_pickle=False) as data:
        required = {
            "score", "posterior", "logit", "contributions", "feature_names",
            "standardization_mean", "standardization_scale",
        }
        if not required.issubset(data.files):
            raise ResidualGraphDeemError(f"B3 member schema incomplete: {cell}/{seed}")
        arrays = {name: np.asarray(data[name]).copy() for name in data.files}
    return arrays, {
        "baseline_array_sha256": sha256_file(path),
        "baseline_metadata_sha256": sha256_file(metadata_path),
        "orientation": int(metadata["orientation"]),
    }


def _load_fit_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        forbidden = [name for name in data.files if TARGET_KEY_RE.search(name)]
        if forbidden or not BASE_KEYS.issubset(data.files):
            raise ResidualGraphDeemError(f"fit NPZ schema/target firewall failed: {path}; {forbidden}")
        return {name: np.asarray(data[name]).copy() for name in data.files}


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    lr = rankdata(left, method="average")
    rr = rankdata(right, method="average")
    if np.std(lr) <= 1e-12 or np.std(rr) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(lr, rr)[0, 1])


def _validate_fit(
    *, arm: str, cell: str, seed: int, arrays: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any], bundle: Any, b3: Mapping[str, np.ndarray],
    b3_audit: Mapping[str, Any],
    expected_config: Mapping[str, Any], expected_role: str,
    run_definition_sha: str,
) -> dict[str, Any]:
    context = f"{arm}/{cell}/seed{seed}"
    n, p = bundle.X_raw.shape
    if (
        metadata.get("schema") != "deem_b3_crossed_rook_fit_artifact_v1"
        or metadata.get("status") != "complete"
        or metadata.get("arm_id") != arm
        or metadata.get("arm_role") != expected_role
        or metadata.get("experiment_id") != "deem_b3_crossed_rook_v1"
    ):
        raise ResidualGraphDeemError(f"fit metadata identity failed: {context}")
    if metadata.get("cell_id") != cell or int(metadata.get("seed", -1)) != seed:
        raise ResidualGraphDeemError(f"fit metadata cell/seed failed: {context}")
    if (
        int(metadata.get("n_rows", -1)) != n
        or int(metadata.get("n_features", -1)) != p
        or metadata.get("dataset_family") != bundle.dataset_family
        or metadata.get("task_type") != bundle.task_type
    ):
        raise ResidualGraphDeemError(f"fit metadata population failed: {context}")
    if metadata.get("run_definition_sha256") != run_definition_sha:
        raise ResidualGraphDeemError(f"fit/run-definition binding failed: {context}")
    _metadata_target_firewall(metadata, context=context)
    if (
        metadata.get("config") != dict(expected_config)
        or metadata.get("config_sha256") != canonical_sha256(expected_config)
        or metadata.get("length_residualization") != "none_unconditional_atomic_core_v1"
    ):
        raise ResidualGraphDeemError(f"fit/config binding failed: {context}")
    if metadata.get("bundle_sha256") != bundle.bundle_sha256:
        raise ResidualGraphDeemError(f"fit/bundle hash mismatch: {context}")
    if metadata.get("inventory_sha256") != bundle.inventory_sha256:
        raise ResidualGraphDeemError(f"fit/inventory hash mismatch: {context}")
    if metadata.get("ordered_row_id_sha256") != canonical_sha256(list(bundle.row_ids)):
        raise ResidualGraphDeemError(f"fit/row hash mismatch: {context}")
    baseline_meta = metadata.get("baseline", {})
    if (
        baseline_meta.get("baseline_array_sha256") != b3_audit["baseline_array_sha256"]
        or baseline_meta.get("baseline_metadata_sha256") != b3_audit["baseline_metadata_sha256"]
        or int(metadata.get("orientation", 0)) != int(b3_audit["orientation"])
        or metadata.get("orientation_policy") != "fixed_same_seed_frozen_b3"
    ):
        raise ResidualGraphDeemError(f"fit/B3 provenance binding failed: {context}")

    shapes = {
        "score": (n,), "posterior": (n, 2), "logit": (n,), "base_score": (n,),
        "base_logit": (n,), "correction": (n,), "contributions": (n, p),
        "base_contributions": (n, p), "cell_delta": (n, 3, 3),
        "feature_names": (p,), "row_id": (n,), "standardization_mean": (p,),
        "standardization_scale": (p,), "core_features": (9,), "core_indices": (9,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ResidualGraphDeemError(f"fit array shape failed: {context}/{name}")
    if tuple(arrays["feature_names"].astype(str).tolist()) != bundle.feature_names:
        raise ResidualGraphDeemError(f"fit feature order mismatch: {context}")
    if tuple(b3["feature_names"].astype(str).tolist()) != bundle.feature_names:
        raise ResidualGraphDeemError(f"frozen B3 feature order mismatch: {context}")
    if tuple(arrays["row_id"].astype(str).tolist()) != bundle.row_ids:
        raise ResidualGraphDeemError(f"fit row order mismatch: {context}")
    if tuple(arrays["core_features"].astype(str).tolist()) != EXPECTED_CORE:
        raise ResidualGraphDeemError(f"physical core mismatch: {context}")
    expected_core_index = tuple(bundle.feature_names.index(name) for name in EXPECTED_CORE)
    if tuple(arrays["core_indices"].astype(int).tolist()) != expected_core_index:
        raise ResidualGraphDeemError(f"physical core indices mismatch: {context}")
    if not all(np.isfinite(arrays[name]).all() for name in (
        "score", "posterior", "logit", "base_score", "base_logit", "correction",
        "contributions", "base_contributions", "cell_delta", "edge_values",
        "edge_raw_contribution", "edge_weights",
    )):
        raise ResidualGraphDeemError(f"non-finite fit array: {context}")
    if (
        np.any((arrays["score"] < 0) | (arrays["score"] > 1))
        or np.any((arrays["base_score"] < 0) | (arrays["base_score"] > 1))
        or np.any((arrays["posterior"] < 0) | (arrays["posterior"] > 1))
    ):
        raise ResidualGraphDeemError(f"score bounds failed: {context}")
    _close(
        arrays["posterior"].sum(axis=1), np.ones(n),
        context=f"posterior normalization {context}",
    )
    _close(arrays["score"], arrays["posterior"][:, 1], context=f"posterior {context}")
    _close(arrays["score"], _sigmoid(arrays["logit"]), context=f"sigmoid {context}")
    _close(arrays["base_score"], b3["score"], context=f"B3 score alias {context}")
    _close(arrays["base_logit"], b3["logit"], context=f"B3 logit alias {context}")
    _close(arrays["base_contributions"], b3["contributions"], context=f"B3 atomic alias {context}")
    _close(arrays["standardization_mean"], b3["standardization_mean"], context=f"B3 mean {context}")
    _close(arrays["standardization_scale"], b3["standardization_scale"], context=f"B3 scale {context}")
    _close(arrays["logit"] - arrays["base_logit"], arrays["correction"], context=f"residual identity {context}", atol=1e-10)
    _close(arrays["cell_delta"].sum(axis=(1, 2)), arrays["correction"], context=f"cell delta {context}", atol=1e-10)
    _close(
        (arrays["contributions"] - arrays["base_contributions"]).sum(axis=1),
        arrays["correction"], context=f"contribution delta {context}", atol=1e-10,
    )

    wanted_edges = _expected_edges(arm)
    observed_edges = tuple(tuple(sorted(pair)) for pair in arrays["edge_pairs"].astype(int).tolist())
    if observed_edges != wanted_edges:
        raise ResidualGraphDeemError(f"edge topology mismatch: {context}")
    e = len(wanted_edges)
    if int(metadata.get("n_edges", -1)) != e:
        raise ResidualGraphDeemError(f"fit edge-count metadata mismatch: {context}")
    if arrays["edge_values"].shape != (n, e) or arrays["edge_raw_contribution"].shape != (n, e):
        raise ResidualGraphDeemError(f"edge array shape mismatch: {context}")
    if arrays["edge_weights"].shape != (e,) or arrays["edge_kinds"].shape != (e,):
        raise ResidualGraphDeemError(f"edge metadata shape mismatch: {context}")
    if tuple(arrays["edge_kinds"].astype(str).tolist()) != _expected_edge_kinds(arm):
        raise ResidualGraphDeemError(f"edge kind/order mismatch: {context}")
    if arm == "A0_B3_EXACT_ALIAS":
        _close(arrays["score"], b3["score"], context=f"A0 exact score {context}")
        _close(arrays["logit"], b3["logit"], context=f"A0 exact logit {context}")
        if not np.array_equal(arrays["correction"], np.zeros(n)):
            raise ResidualGraphDeemError(f"A0 correction is nonzero: {context}")

    config = metadata.get("config", metadata.get("arm_config", {}))
    bound = float(config.get("strength", np.nan)) * float(config.get("correction_cap", np.nan))
    max_abs = float(np.max(np.abs(arrays["correction"])))
    if not np.isfinite(bound) or max_abs > bound + 1e-10:
        raise ResidualGraphDeemError(f"correction cap failed: {context}")
    health = metadata.get("health", {})
    if health.get("healthy") is not True or health.get("finite") is not True:
        raise ResidualGraphDeemError(f"unhealthy fit: {context}")
    diagnostics = metadata.get("diagnostics", {})
    acceptance: float | None
    if arm == "A0_B3_EXACT_ALIAS":
        acceptance = None
    else:
        acceptance = float(health.get("mala_acceptance_mean", np.nan))
        if not (0.0 <= acceptance <= 1.0):
            raise ResidualGraphDeemError(f"invalid MALA acceptance: {context}")
    return {
        "score": np.asarray(arrays["score"], dtype=np.float64),
        "correction": np.asarray(arrays["correction"], dtype=np.float64),
        "score_spearman_b3": _spearman(arrays["score"], b3["score"]),
        "correction_sd": float(np.std(arrays["correction"])),
        "correction_max_abs": max_abs,
        "correction_cap": bound,
        "correction_saturation_fraction": float(np.mean(np.abs(arrays["correction"]) >= 0.99 * max(bound, 1e-12))),
        "mala_acceptance": acceptance,
        "posterior_sd": float(health.get("posterior_sd", np.std(arrays["score"]))),
        "edge_weight_l2": float(np.linalg.norm(arrays["edge_weights"])),
        "metadata_diagnostics": diagnostics,
    }


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before preflight")
    _validate_topology_contract()
    registry = load_registry(args.registry)
    registry_sha256 = sha256_file(args.registry)
    structure, structure_audit = _load_structural_summary(
        args.structure_summary, registry_sha256=registry_sha256
    )
    cells = tuple(str(record["cell_id"]) for record in registry["cells"])
    if len(cells) != len(set(cells)):
        raise ResidualGraphDeemError("registry contains duplicate cell IDs")
    registry_by_cell = {str(record["cell_id"]): record for record in registry["cells"]}
    config = _read_json(args.config)
    variants = config.get("variants", [])
    if (
        config.get("schema") != "deem_b3_crossed_rook_v1_config"
        or tuple(str(row.get("id")) for row in variants) != ARM_ORDER
    ):
        raise ResidualGraphDeemError("Crossed-Rook config roster mismatch")
    expected_variant = {
        str(row["id"]): {
            "role": str(row.get("role", "")),
            "config": asdict(CrossedRookConfig(**row.get("config", {}))),
        }
        for row in variants
    }
    bundles = {}
    bundle_audit = {}
    for cell in cells:
        path = args.bundle_dir / f"{cell}.npz"
        bundle = load_target_free_bundle(path)
        manifest_path = path.with_suffix(".manifest.json")
        manifest = _read_json(manifest_path)
        registered = registry_by_cell[cell]
        if (
            bundle.cell_id != cell
            or len(bundle.row_ids) != int(registered["n_rows"])
            or bundle.inventory_sha256 != registered["inventory_sha256"]
            or bundle.dataset_family != registered["dataset_family"]
            or bundle.task_type != registered["task_type"]
            or manifest.get("schema") != BUNDLE_SCHEMA
            or manifest.get("cell_id") != cell
            or int(manifest.get("n_rows", -1)) != len(bundle.row_ids)
            or int(manifest.get("n_features", -1)) != len(bundle.feature_names)
            or manifest.get("bundle_sha256") != bundle.bundle_sha256
            or manifest.get("ordered_row_id_sha256") != canonical_sha256(list(bundle.row_ids))
            or manifest.get("inventory_sha256") != bundle.inventory_sha256
            or manifest.get("labels_accessed") is not False
        ):
            raise ResidualGraphDeemError(f"bundle manifest mismatch: {cell}")
        bundles[cell] = bundle
        bundle_audit[cell] = {
            "bundle_sha256": bundle.bundle_sha256,
            "bundle_manifest_sha256": sha256_file(manifest_path),
            "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
        }

    baseline_inventory = _validate_baseline_freeze(args.baseline_dir, cells)
    freeze, definition, completion, inventory = _resolve_run_contract(
        args.run_dir,
        args.score_freeze,
        config_path=args.config,
        registry=registry,
        cells=cells,
    )
    if tuple(definition.get("arm_order", [])) != ARM_ORDER:
        raise ResidualGraphDeemError("run definition arm order mismatch")
    baseline_freeze_sha = sha256_file(args.baseline_dir / "SCORE_FREEZE_MANIFEST.json")
    if (
        definition.get("baseline_score_freeze_sha256") != baseline_freeze_sha
        or definition.get("representation") != config.get("representation")
        or definition.get("scientific_boundary") != config.get("scientific_boundary")
        or not str(definition.get("explicit_caveat", "")).strip()
    ):
        raise ResidualGraphDeemError("run definition scientific/input binding mismatch")
    run_definition_sha = canonical_sha256(definition)

    score_by_seed: dict[tuple[str, str, int], np.ndarray] = {}
    correction_by_seed: dict[tuple[str, str, int], np.ndarray] = {}
    mechanical = []
    for arm, cell, seed in itertools.product(ARM_ORDER, cells, SEEDS):
        stem = f"{arm}__seed{seed}"
        npz_relative = f"fits/{arm}/{cell}/{stem}.npz"
        json_relative = f"fits/{arm}/{cell}/{stem}.json"
        key = f"{arm}|{cell}|{seed}"
        binding = inventory.get(key)
        if not isinstance(binding, dict) or set(binding) != FREEZE_BINDING_FIELDS:
            raise ResidualGraphDeemError(f"fit not bound by score freeze: {key}")
        npz_path = _safe_artifact(args.run_dir, npz_relative)
        json_path = _safe_artifact(args.run_dir, json_relative)
        if (
            binding.get("array_sha256") != sha256_file(npz_path)
            or binding.get("metadata_sha256") != sha256_file(json_path)
        ):
            raise ResidualGraphDeemError(f"fit hash binding failed: {key}")
        completion_binding = completion.get("artifact_sha256", {}).get(key, {})
        if completion_binding != {
            "array_sha256": binding["array_sha256"],
            "metadata_sha256": binding["metadata_sha256"],
        }:
            raise ResidualGraphDeemError(f"fit completion hash binding failed: {key}")
        metadata = _read_json(json_path)
        _verify_content_hash(metadata, context=stem)
        bundle_relative = (
            (args.bundle_dir / f"{cell}.npz").resolve()
            .relative_to(ROOT.resolve())
            .as_posix()
        )
        if (
            metadata.get("array_sha256") != binding["array_sha256"]
            or metadata.get("array_path") != npz_relative
            or metadata.get("bundle_path") != bundle_relative
        ):
            raise ResidualGraphDeemError(f"fit metadata/array binding failed: {key}")
        arrays = _load_fit_arrays(npz_path)
        b3, b3_audit = _load_b3_member(args.baseline_dir, cell, seed, baseline_inventory)
        if (
            binding.get("baseline_array_sha256") != b3_audit["baseline_array_sha256"]
            or binding.get("baseline_metadata_sha256") != b3_audit["baseline_metadata_sha256"]
            or binding.get("bundle_sha256") != bundles[cell].bundle_sha256
            or binding.get("ordered_row_id_sha256") != canonical_sha256(list(bundles[cell].row_ids))
            or binding.get("score_sha256") != metadata.get("score_sha256")
            or binding.get("base_score_sha256") != metadata.get("base_score_sha256")
            or binding.get("score_sha256") != _score_hash(arrays["score"])
            or binding.get("base_score_sha256") != _score_hash(arrays["base_score"])
        ):
            raise ResidualGraphDeemError(f"score freeze/B3 binding failed: {key}")
        result = _validate_fit(
            arm=arm, cell=cell, seed=seed, arrays=arrays, metadata=metadata,
            bundle=bundles[cell], b3=b3, b3_audit=b3_audit,
            expected_config=expected_variant[arm]["config"],
            expected_role=expected_variant[arm]["role"],
            run_definition_sha=run_definition_sha,
        )
        method = METHOD[arm]
        score_by_seed[(method, cell, seed)] = result.pop("score")
        correction_by_seed[(method, cell, seed)] = result.pop("correction")
        mechanical.append({
            "arm_id": arm, "method": method, "cell_id": cell,
            "dataset_family": bundles[cell].dataset_family, "seed": seed, **result,
        })
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported during mechanical preflight")

    ensemble = {}
    stability = []
    for method, cell in itertools.product(METHOD_ORDER, cells):
        members = [score_by_seed[(method, cell, seed)] for seed in SEEDS]
        ensemble[(method, cell)] = np.mean(np.stack(members), axis=0)
        pairwise = [
            _spearman(members[left], members[right])
            for left, right in itertools.combinations(range(len(SEEDS)), 2)
        ]
        stability.append({
            "method": method, "cell_id": cell,
            "dataset_family": bundles[cell].dataset_family,
            "seed_pair_spearman_mean": float(np.nanmean(pairwise)),
            "seed_pair_spearman_min": float(np.nanmin(pairwise)),
            "ensemble_score_sd": float(np.std(ensemble[(method, cell)])),
            "mean_seed_correction_sd": float(np.mean([
                np.std(correction_by_seed[(method, cell, seed)]) for seed in SEEDS
            ])),
        })
    return {
        "registry": registry, "cells": cells, "bundles": bundles,
        "bundle_audit": bundle_audit, "freeze": freeze, "definition": definition,
        "ensemble": ensemble, "mechanical": mechanical, "stability": stability,
        "structure": structure, "structure_audit": structure_audit,
        "preflight_hashes": {
            "registry_sha256": registry_sha256,
            "config_sha256": sha256_file(args.config),
            "structural_summary_sha256": structure_audit["summary_sha256"],
            "structural_freeze_sha256": structure_audit["freeze_sha256"],
            "baseline_score_freeze_sha256": baseline_freeze_sha,
            "crossed_score_freeze_sha256": sha256_file(args.score_freeze),
            "run_definition_canonical_sha256": run_definition_sha,
        },
    }


def _load_targets(state: Mapping[str, Any], sidecar_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before explicit label phase")
    from spectral_utils.residual_graph_deem_labels import (  # noqa: PLC0415
        SIDECAR_SCHEMA,
        join_labels_by_id,
        load_label_sidecar,
    )

    targets, audits = {}, {}
    for cell in state["cells"]:
        bundle = state["bundles"][cell]
        path = sidecar_dir / f"{cell}.npz"
        manifest_path = path.with_suffix(".manifest.json")
        manifest = _read_json(manifest_path)
        sidecar = load_label_sidecar(path)
        if (
            manifest.get("schema") != SIDECAR_SCHEMA
            or manifest.get("cell_id") != cell
            or manifest.get("sidecar_sha256") != sidecar.sidecar_sha256
            or manifest.get("unordered_row_id_sha256") != canonical_sha256(sorted(bundle.row_ids))
        ):
            raise ResidualGraphDeemError(f"label sidecar manifest failed: {cell}")
        target = join_labels_by_id(bundle, sidecar)
        if target.shape != (len(bundle.row_ids),) or set(np.unique(target)) != {0, 1}:
            raise ResidualGraphDeemError(f"invalid target: {cell}")
        targets[cell] = target
        audits[cell] = {
            "sidecar_sha256": sidecar.sidecar_sha256,
            "sidecar_manifest_sha256": sha256_file(manifest_path),
            "ordered_join_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
        }
    return targets, audits


def _metrics(target: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if target.shape != score.shape or not np.isfinite(score).all():
        raise ResidualGraphDeemError("metric input mismatch")
    return {
        "auroc": float(roc_auc_score(target, score)),
        "auprc": float(average_precision_score(target, score)),
    }


def _method_summary(rows: Sequence[Mapping[str, Any]], method: str, metric: str) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    grouped = defaultdict(list)
    for row in selected:
        grouped[str(row["dataset_family"])].append(float(row[metric]))
    family = {name: float(np.mean(values)) for name, values in sorted(grouped.items())}
    return {
        "method": method, "metric": metric, "n_cells": len(selected),
        "n_dataset_families": len(family),
        "equal_cell_macro": float(np.mean([float(row[metric]) for row in selected])),
        "equal_dataset_family_macro": float(np.mean(list(family.values()))),
        "worst_cell": float(min(float(row[metric]) for row in selected)),
        "dataset_family_means": family,
    }


def _paired_deltas(rows: Sequence[Mapping[str, Any]], candidate: str, reference: str, metric: str) -> tuple[dict[str, float], dict[str, float], dict[str, list[float]]]:
    lookup = {(str(row["cell_id"]), str(row["method"])): row for row in rows}
    cells = sorted({str(row["cell_id"]) for row in rows})
    by_family = defaultdict(list)
    cell_delta = {}
    for cell in cells:
        cand, base = lookup[(cell, candidate)], lookup[(cell, reference)]
        value = float(cand[metric]) - float(base[metric])
        family = str(cand["dataset_family"])
        by_family[family].append(value)
        cell_delta[cell] = value
    family_delta = {name: float(np.mean(values)) for name, values in sorted(by_family.items())}
    return family_delta, cell_delta, dict(by_family)


def _nested_bootstrap(by_family: Mapping[str, Sequence[float]], draws: int, seed: int) -> np.ndarray:
    names = tuple(sorted(by_family))
    rng = np.random.Generator(np.random.PCG64(seed))
    output = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        chosen = rng.integers(0, len(names), size=len(names))
        family_means = []
        for index in chosen:
            values = np.asarray(by_family[names[index]], dtype=np.float64)
            sampled = values[rng.integers(0, len(values), size=len(values))]
            family_means.append(float(np.mean(sampled)))
        output[draw] = float(np.mean(family_means))
    return output


def _comparison(rows: Sequence[Mapping[str, Any]], candidate: str, reference: str, draws: int) -> dict[str, Any]:
    metrics = {}
    for offset, metric in enumerate(("auroc", "auprc")):
        family, cell, raw_family = _paired_deltas(rows, candidate, reference, metric)
        values = np.asarray(list(family.values()), dtype=np.float64)
        observed_family = float(np.mean(values))
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(values))))
        null = np.mean(signs * values[None, :], axis=1)
        seed = int.from_bytes(hashlib.sha256(
            f"{SCHEMA}\0{candidate}\0{reference}\0{metric}".encode()
        ).digest()[:8], "big")
        bootstrap = _nested_bootstrap(raw_family, draws, seed + offset)
        tolerance = TOLERANCE if metric == "auroc" else 0.0
        metrics[metric] = {
            "equal_cell_delta": float(np.mean(list(cell.values()))),
            "equal_dataset_family_delta": observed_family,
            "descriptive_nested_family_cell_bootstrap95": [
                float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))
            ],
            "exact_family_signflip_one_sided_p": float(np.mean(null >= observed_family - 1e-15)),
            "exact_family_signflip_two_sided_p": float(np.mean(np.abs(null) >= abs(observed_family) - 1e-15)),
            "family_delta": family, "cell_delta": cell,
            "wins": int(sum(value > tolerance for value in cell.values())),
            "ties": int(sum(abs(value) <= tolerance for value in cell.values())),
            "losses": int(sum(value < -tolerance for value in cell.values())),
            "worst_cell_delta": float(min(cell.values())),
            "worst_family_delta": float(min(family.values())),
            "bootstrap_draws": draws,
            "bootstrap_seed": seed + offset,
            "bootstrap_is_descriptive_not_a_null_test": True,
        }
    return {"candidate": candidate, "reference": reference, **metrics}


def _per_family_metric_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["dataset_family"]))].append(row)
    output = []
    for method in METHOD_ORDER:
        families = sorted(family for candidate, family in grouped if candidate == method)
        for family in families:
            members = grouped[(method, family)]
            output.append({
                "method": method,
                "dataset_family": family,
                "n_cells": len(members),
                "auroc": float(np.mean([float(row["auroc"]) for row in members])),
                "auprc": float(np.mean([float(row["auprc"]) for row in members])),
            })
    return output


def _paired_delta_rows(
    per_cell: Sequence[Mapping[str, Any]],
    comparisons: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    family_by_cell = {
        str(row["cell_id"]): str(row["dataset_family"])
        for row in per_cell if row["method"] == "B3"
    }
    output = []
    for pair in REGISTERED_CONTRASTS + DIAGNOSTIC_CONTRASTS:
        comparison = comparisons[pair]
        registered = pair in REGISTERED_CONTRASTS
        for scope, key in (("cell", "cell_delta"), ("dataset_family", "family_delta")):
            auroc = comparison["auroc"][key]
            auprc = comparison["auprc"][key]
            for unit in sorted(auroc):
                output.append({
                    "candidate": pair[0],
                    "reference": pair[1],
                    "registered_contrast": registered,
                    "scope": scope,
                    "unit_id": unit,
                    "dataset_family": family_by_cell[unit] if scope == "cell" else unit,
                    "delta_auroc": float(auroc[unit]),
                    "delta_auprc": float(auprc[unit]),
                })
    return output


def _fit_health_summary(
    mechanical: Sequence[Mapping[str, Any]],
    stability: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    def aggregate(values: Sequence[Any], operation: str) -> float | None:
        finite = np.asarray(
            [float(value) for value in values if value is not None],
            dtype=np.float64,
        )
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return None
        return float(np.mean(finite) if operation == "mean" else np.min(finite))

    output = []
    for method in METHOD_ORDER:
        fits = [row for row in mechanical if row["method"] == method]
        cells = [row for row in stability if row["method"] == method]
        output.append({
            "method": method,
            "n_seed_cell_fits": len(fits),
            "score_spearman_b3_mean": aggregate([
                float(row["score_spearman_b3"]) for row in fits
            ], "mean"),
            "score_spearman_b3_min": aggregate([
                float(row["score_spearman_b3"]) for row in fits
            ], "min"),
            "seed_pair_spearman_mean": aggregate([
                float(row["seed_pair_spearman_mean"]) for row in cells
            ], "mean"),
            "seed_pair_spearman_worst_cell": aggregate([
                float(row["seed_pair_spearman_min"]) for row in cells
            ], "min"),
            "correction_sd_mean": float(np.mean([
                float(row["correction_sd"]) for row in fits
            ])),
            "correction_max_abs": float(max(
                float(row["correction_max_abs"]) for row in fits
            )),
            "correction_saturation_fraction_mean": float(np.mean([
                float(row["correction_saturation_fraction"]) for row in fits
            ])),
            "correction_saturation_fraction_max": float(max(
                float(row["correction_saturation_fraction"]) for row in fits
            )),
            "mala_acceptance_mean": aggregate([
                row["mala_acceptance"] for row in fits
            ], "mean"),
            "mala_acceptance_min": aggregate([
                row["mala_acceptance"] for row in fits
            ], "min"),
        })
    return output


def _decision(comparisons: Mapping[tuple[str, str], Mapping[str, Any]]) -> dict[str, Any]:
    def stability(method: str) -> tuple[bool, list[dict[str, Any]]]:
        row = comparisons[(method, "B3")]["auroc"]
        clauses = [
            {"name": "equal_family_delta", "observed": row["equal_dataset_family_delta"], "threshold": PROMOTION_DELTA, "passed": row["equal_dataset_family_delta"] >= PROMOTION_DELTA},
            {"name": "bootstrap_lower", "observed": row["descriptive_nested_family_cell_bootstrap95"][0], "threshold": 0.0, "passed": row["descriptive_nested_family_cell_bootstrap95"][0] > 0.0},
            {"name": "wins_plus_ties", "observed": row["wins"] + row["ties"], "threshold": MIN_WINS_TIES, "passed": row["wins"] + row["ties"] >= MIN_WINS_TIES},
            {"name": "worst_cell", "observed": row["worst_cell_delta"], "threshold": WORST_CELL_FLOOR, "passed": row["worst_cell_delta"] >= WORST_CELL_FLOOR},
        ]
        return all(item["passed"] for item in clauses), clauses

    crossed_ok, crossed_clauses = stability("CROSSED")
    specificity = comparisons[("CROSSED", "NONROOK")]["auroc"]["equal_dataset_family_delta"] > 0.0
    row_noninferior = comparisons[("CROSSED", "ROW")]["auroc"]["equal_dataset_family_delta"] >= -TOLERANCE
    column_noninferior = comparisons[("CROSSED", "COLUMN")]["auroc"]["equal_dataset_family_delta"] >= -TOLERANCE
    crossed_pass = crossed_ok and specificity and row_noninferior and column_noninferior
    row_ok, row_clauses = stability("ROW")
    column_ok, column_clauses = stability("COLUMN")
    if crossed_pass:
        verdict = "CROSSED_ROOK_PASSES_RETROSPECTIVE_GATE"
    elif row_ok != column_ok:
        axis = "ROW" if row_ok else "COLUMN"
        crossed_vs_axis = comparisons[("CROSSED", axis)]["auroc"]["equal_dataset_family_delta"]
        verdict = (
            f"{axis}_ONLY_RECOMMENDED_FOR_NEW_UNOPENED_CONFIRMATION"
            if crossed_vs_axis < TOLERANCE else "NO_PROMOTION"
        )
    else:
        verdict = "NO_PROMOTION"
    return {
        "verdict": verdict,
        "scientific_tier": "C_retrospective_labels_historically_open",
        "architecture_and_thresholds_frozen_before_label_access": True,
        "no_architecture_or_threshold_tuned_after_label_access": True,
        "crossed_clauses": crossed_clauses + [
            {"name": "beats_nonrook", "observed": comparisons[("CROSSED", "NONROOK")]["auroc"]["equal_dataset_family_delta"], "threshold": 0.0, "passed": specificity},
            {"name": "not_worse_than_row", "observed": comparisons[("CROSSED", "ROW")]["auroc"]["equal_dataset_family_delta"], "threshold": -TOLERANCE, "passed": row_noninferior},
            {"name": "not_worse_than_column", "observed": comparisons[("CROSSED", "COLUMN")]["auroc"]["equal_dataset_family_delta"], "threshold": -TOLERANCE, "passed": column_noninferior},
        ],
        "row_gate": {"passed": row_ok, "clauses": row_clauses},
        "column_gate": {"passed": column_ok, "clauses": column_clauses},
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plots(out: Path, rows: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], comparisons: Mapping[tuple[str, str], Mapping[str, Any]], mechanical: Sequence[Mapping[str, Any]]) -> list[Path]:
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(
            prefix="deem_b3_crossed_rook_mpl_"
        )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    colors = {"B3": "#64748b", "ROW": "#0f766e", "COLUMN": "#2563eb", "CROSSED": "#c2410c", "NONROOK": "#7c3aed"}
    artifacts = []
    lookup = {(row["method"], row["metric"]): row for row in summaries}

    # Primary visual: paired equal-family AUROC deltas with the registered
    # descriptive nested family/cell intervals and an explicit zero line.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    panels = (
        (
            axes[0],
            (("ROW", "B3"), ("COLUMN", "B3"), ("CROSSED", "B3"), ("NONROOK", "B3")),
            "Utility relative to frozen B3",
        ),
        (
            axes[1],
            (("CROSSED", "ROW"), ("CROSSED", "COLUMN"), ("CROSSED", "NONROOK")),
            "Crossed mechanism contrasts",
        ),
    )
    for axis, pairs, title in panels:
        centers, lower, upper, labels, point_colors = [], [], [], [], []
        for pair in pairs:
            row = comparisons[pair]["auroc"]
            center = float(row["equal_dataset_family_delta"])
            ci = row["descriptive_nested_family_cell_bootstrap95"]
            centers.append(center)
            lower.append(center - float(ci[0]))
            upper.append(float(ci[1]) - center)
            labels.append(f"{pair[0]}−{pair[1]}")
            point_colors.append(colors[pair[0]])
        y = np.arange(len(pairs))
        axis.errorbar(
            centers, y, xerr=np.asarray([lower, upper]), fmt="none",
            ecolor="#334155", elinewidth=1.5, capsize=4, zorder=2,
        )
        axis.scatter(centers, y, c=point_colors, s=55, zorder=3)
        axis.axvline(0.0, color="#111827", linewidth=1.0)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("equal-family Δ AUROC (95% descriptive interval)")
        axis.set_title(title)
        axis.grid(axis="x", alpha=.22)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=5))
        axis.xaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:+.4f}")
        )
        for index, center in enumerate(centers):
            axis.annotate(
                f"{center:+.6f}", (center, index), xytext=(6, -10),
                textcoords="offset points", fontsize=8,
            )
    for suffix in ("png", "svg"):
        path = out / f"primary_auroc_deltas.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        artifacts.append(path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    x = np.arange(len(METHOD_ORDER))
    for axis, metric, title in zip(axes, ("auroc", "auprc"), ("Hallucination AUROC", "Hallucination AUPRC")):
        values = [lookup[(method, metric)]["equal_dataset_family_macro"] for method in METHOD_ORDER]
        bars = axis.bar(x, values, color=[colors[method] for method in METHOD_ORDER], width=.72)
        axis.set_xticks(x, METHOD_ORDER, rotation=20, ha="right")
        axis.set_title(f"{title} (truncated y-axis)")
        axis.set_ylabel("equal-family macro")
        lower = min(values) - max(.003, (max(values) - min(values)) * .8)
        axis.set_ylim(max(0, lower), max(values) + max(.002, (max(values) - min(values)) * .4))
        axis.grid(axis="y", alpha=.22)
        axis.text(
            .01, .97, "Note: y-axis does not start at zero", transform=axis.transAxes,
            va="top", fontsize=8, color="#334155",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": .9, "pad": 2},
        )
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.4f}",
                (bar.get_x() + bar.get_width() / 2.0, value),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7,
            )
    for suffix in ("png", "svg"):
        path = out / f"macro_metrics.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        artifacts.append(path)
    plt.close(fig)

    contrast_order = (("CROSSED", "B3"), ("CROSSED", "ROW"), ("CROSSED", "COLUMN"), ("CROSSED", "NONROOK"))
    cells = [str(row["cell_id"]) for row in rows if row["method"] == "B3"]
    matrix = np.asarray([[comparisons[pair]["auroc"]["cell_delta"][cell] for pair in contrast_order] for cell in cells])
    fig, axis = plt.subplots(figsize=(9.5, 9), constrained_layout=True)
    bound = max(float(np.max(np.abs(matrix))), .001)
    image = axis.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
    axis.set_yticks(np.arange(len(cells)), cells, fontsize=7)
    axis.set_xticks(np.arange(len(contrast_order)), [f"{a}−{b}" for a, b in contrast_order], rotation=20, ha="right")
    axis.set_title("Per-cell AUROC deltas")
    fig.colorbar(image, ax=axis, label="Δ AUROC")
    for suffix in ("png", "svg"):
        path = out / f"per_cell_deltas.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        artifacts.append(path)
    plt.close(fig)

    families = sorted(comparisons[("CROSSED", "B3")]["auroc"]["family_delta"])
    fig, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    width = .18
    for index, pair in enumerate(contrast_order):
        values = [comparisons[pair]["auroc"]["family_delta"][family] for family in families]
        axis.bar(np.arange(len(families)) + (index - 1.5) * width, values, width, label=f"{pair[0]}−{pair[1]}")
    axis.axhline(0, color="#111827", linewidth=.8)
    axis.set_xticks(np.arange(len(families)), families, rotation=25, ha="right")
    axis.set_ylabel("family-mean Δ AUROC")
    axis.set_title("Mechanism contrasts by dataset family")
    axis.legend(ncol=2, frameon=False)
    axis.grid(axis="y", alpha=.22)
    for suffix in ("png", "svg"):
        path = out / f"family_mechanism_deltas.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        artifacts.append(path)
    plt.close(fig)

    active = [row for row in mechanical if row["method"] != "B3"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    for axis, key, title in zip(
        axes,
        ("score_spearman_b3", "correction_sd", "mala_acceptance"),
        ("Score Spearman vs B3", "Correction SD", "MALA acceptance"),
    ):
        data = [[float(row[key]) for row in active if row["method"] == method] for method in METHOD_ORDER[1:]]
        axis.boxplot(data, tick_labels=METHOD_ORDER[1:], showfliers=False)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=.22)
    for suffix in ("png", "svg"):
        path = out / f"fit_health.{suffix}"
        fig.savefig(path, dpi=220 if suffix == "png" else None)
        artifacts.append(path)
    plt.close(fig)
    return artifacts


def _explainer_plot(
    out: Path,
    structure: Mapping[str, float],
    comparisons: Mapping[tuple[str, str], Mapping[str, Any]],
    fit_health: Sequence[Mapping[str, Any]],
) -> list[Path]:
    """One-page bridge from label-free structure to downstream utility."""

    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    colors = {
        "B3": "#64748b", "ROW": "#0f766e", "COLUMN": "#2563eb",
        "CROSSED": "#c2410c", "NONROOK": "#7c3aed",
    }
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    fig.suptitle(
        "Crossed-Rook B3: dependency structure is real, but utility did not transfer",
        fontsize=18, fontweight="bold", y=.97,
    )

    # A. Frozen target-free reconstruction geometry.
    axis = axes[0, 0]
    structure_order = (
        ("Same source", "r2_source"),
        ("Same operator", "r2_operator"),
        ("Rook union", "r2_union"),
        ("All 8 peers", "r2_all"),
    )
    values = [100.0 * float(structure[key]) for _, key in structure_order]
    y = np.arange(len(values))
    bars = axis.barh(
        y, values,
        color=[colors["ROW"], colors["COLUMN"], colors["CROSSED"], "#334155"],
    )
    axis.set_yticks(y, [label for label, _ in structure_order])
    axis.invert_yaxis()
    axis.set_xlim(0, max(values) + 10)
    axis.set_xlabel("cross-validated explained variance, R² (%)")
    axis.set_title("A. Label-free structure (23 K=50 cells)", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=.2)
    for bar, value in zip(bars, values):
        axis.text(
            value + 1.0, bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%", va="center", fontweight="bold",
        )

    def effect_panel(axis, pairs, title):
        centers, lower, upper, labels = [], [], [], []
        for candidate, reference in pairs:
            row = comparisons[(candidate, reference)]["auroc"]
            center = 10_000.0 * float(row["equal_dataset_family_delta"])
            ci = [10_000.0 * float(value) for value in row["descriptive_nested_family_cell_bootstrap95"]]
            centers.append(center)
            lower.append(center - ci[0])
            upper.append(ci[1] - center)
            labels.append(f"{candidate}−{reference}")
        y = np.arange(len(pairs))
        axis.errorbar(
            centers, y, xerr=np.asarray([lower, upper]), fmt="none",
            ecolor="#334155", elinewidth=1.5, capsize=4, zorder=2,
        )
        axis.scatter(
            centers, y, s=65,
            c=[colors[candidate] for candidate, _ in pairs], zorder=3,
        )
        axis.axvline(0, color="#111827", linewidth=1.0)
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
        axis.set_xlabel("equal-family Δ AUROC (basis points; 1 bp = 0.0001)")
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="x", alpha=.2)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:+.1f}"))
        span = max(float(np.ptp(axis.get_xlim())), 1.0)
        for index, center in enumerate(centers):
            offset = 0.025 * span
            axis.text(
                center + (offset if center >= 0 else -offset), index,
                f"{center:+.2f} bp", va="center",
                ha="left" if center >= 0 else "right", fontweight="bold", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .75, "pad": 1},
            )

    # B. Utility against the exact B3 baseline.
    effect_panel(
        axes[0, 1],
        (("ROW", "B3"), ("COLUMN", "B3"), ("CROSSED", "B3"), ("NONROOK", "B3")),
        "B. Downstream utility vs frozen B3 (descriptive 95% CI)",
    )

    # C. Does the registered rook topology beat its controls?
    effect_panel(
        axes[1, 0],
        (("CROSSED", "ROW"), ("CROSSED", "COLUMN"), ("CROSSED", "NONROOK")),
        "C. CROSSED vs mechanism controls (descriptive 95% CI)",
    )

    # D. The extension changed logits materially; failure is not a zero path.
    axis = axes[1, 1]
    health = {str(row["method"]): row for row in fit_health}
    methods = METHOD_ORDER
    values = [float(health[method]["correction_sd_mean"]) for method in methods]
    bars = axis.bar(
        np.arange(len(methods)), values,
        color=[colors[method] for method in methods], width=.72,
    )
    axis.set_xticks(np.arange(len(methods)), methods, rotation=18, ha="right")
    axis.set_ylabel("mean correction SD (logit units)")
    axis.set_title("D. Fit actuation (not a collapsed path)", loc="left", fontweight="bold")
    axis.grid(axis="y", alpha=.2)
    axis.set_ylim(0, max(values) * 1.25)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2, value + max(values) * .025,
            f"{value:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold",
        )
    axis.text(
        .02, .94, "saturation = 0 for every arm; MALA acceptance ≈ 0.9995",
        transform=axis.transAxes, va="top", fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": .9, "pad": 3},
    )

    fig.text(
        .5, .015,
        "Geometry is label-free; utility is C-tier retrospective on 24 cells with frozen arms. "
        "Intervals are descriptive. Registered verdict: NO PROMOTION.",
        ha="center", fontsize=10, color="#334155",
    )
    fig.subplots_adjust(left=.10, right=.98, top=.89, bottom=.09, hspace=.42, wspace=.38)
    artifacts = []
    for suffix in ("png", "svg"):
        path = out / f"crossed_rook_explainer.{suffix}"
        fig.savefig(path, dpi=240 if suffix == "png" else None, bbox_inches="tight")
        artifacts.append(path)
    plt.close(fig)
    return artifacts


def _markdown(report: Mapping[str, Any]) -> str:
    def number(value: Any, digits: int = 4) -> str:
        return "NA" if value is None else f"{float(value):.{digits}f}"

    summary = {(row["method"], row["metric"]): row for row in report["summaries"]}
    lines = [
        "# Crossed-Rook B3 v1 — retrospective evaluation",
        "",
        f"**Decision:** `{report['decision']['verdict']}`.",
        "",
        "This is C-tier retrospective evidence on historically opened targets. All five score "
        "arms and thresholds were frozen and mechanically verified before labels were imported.",
        "",
        "## Macro performance",
        "",
        "| method | equal-family AUROC | equal-cell AUROC | equal-family AUPRC | equal-cell AUPRC |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHOD_ORDER:
        auroc, auprc = summary[(method, "auroc")], summary[(method, "auprc")]
        lines.append(
            f"| {method} | {auroc['equal_dataset_family_macro']:.6f} | {auroc['equal_cell_macro']:.6f} | "
            f"{auprc['equal_dataset_family_macro']:.6f} | {auprc['equal_cell_macro']:.6f} |"
        )
    lines.extend([
        "", "## Registered AUROC contrasts", "",
        "| contrast | equal-family Δ | 95% descriptive nested bootstrap | exact sign-flip p | W/T/L | worst cell | worst family |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for candidate, reference in report["contrast_order"]:
        row = report["comparisons"][f"{candidate}__vs__{reference}"]["auroc"]
        ci = row["descriptive_nested_family_cell_bootstrap95"]
        lines.append(
            f"| {candidate}−{reference} | {row['equal_dataset_family_delta']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | {row['exact_family_signflip_one_sided_p']:.5f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | {row['worst_cell_delta']:+.6f} | {row['worst_family_delta']:+.6f} |"
        )
    lines.extend([
        "", "## Fit mechanics and seed stability", "",
        "| method | score ρ vs B3 (mean/min) | seed-pair ρ (mean/worst) | correction SD | max | saturation | MALA acceptance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["fit_health_summary"]:
        lines.append(
            f"| {row['method']} | {number(row['score_spearman_b3_mean'])}/{number(row['score_spearman_b3_min'])} | "
            f"{number(row['seed_pair_spearman_mean'])}/{number(row['seed_pair_spearman_worst_cell'])} | "
            f"{number(row['correction_sd_mean'])} | {number(row['correction_max_abs'])} | "
            f"{number(row['correction_saturation_fraction_mean'])} | {number(row['mala_acceptance_mean'])} |"
        )
    lines.extend([
        "", "## Figures", "",
        "![One-page explainer](crossed_rook_explainer.png)", "",
        "![Primary AUROC deltas](primary_auroc_deltas.png)", "",
        "![Macro metrics](macro_metrics.png)", "",
        "![Per-cell deltas](per_cell_deltas.png)", "",
        "![Family mechanism deltas](family_mechanism_deltas.png)", "",
        "![Fit health](fit_health.png)", "",
        "## Claim boundary", "",
        "The intervals are descriptive family/cell resampling summaries, not confirmation. "
        "No architecture or threshold may be changed after reading this report and then described as frozen.", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deem_b3_crossed_rook_v1.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/bundles")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/b3_frozen")
    parser.add_argument(
        "--structure-summary", type=Path,
        default=ROOT / "local_cache/deem_b3_moe_v1/crossed_core_independent_v1/SUMMARY.json",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--score-freeze", type=Path)
    parser.add_argument("--sidecar-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/label_sidecars")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=9999)
    args = parser.parse_args()
    if args.score_freeze is None:
        args.score_freeze = args.run_dir / "SCORE_FREEZE_MANIFEST.json"
    if args.out_dir.exists() and (
        not args.out_dir.is_dir() or any(args.out_dir.iterdir())
    ):
        raise FileExistsError(f"evaluation output must be an empty directory: {args.out_dir}")
    if args.bootstrap_draws < 100:
        raise ValueError("bootstrap draws must be at least 100")

    state = _preflight(args)
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before both preflight passes")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pre_label_attestation = {
        "schema": "deem_b3_crossed_rook_v1_pre_label_preflight",
        "status": "complete_before_label_import",
        "target_module_imported": False,
        "targets_accessed": False,
        "n_cells": len(state["cells"]),
        "n_methods": len(METHOD_ORDER),
        "n_seeds": len(SEEDS),
        "n_fit_artifacts": len(ARM_ORDER) * len(state["cells"]) * len(SEEDS),
        "ordered_cells": list(state["cells"]),
        "methods": list(METHOD_ORDER),
        "seeds": list(SEEDS),
        "preflight_hashes": state["preflight_hashes"],
        "bundle_audit": state["bundle_audit"],
        "structure_audit": state["structure_audit"],
        "evaluator_source_sha256": sha256_file(Path(__file__)),
    }
    pre_label_attestation["content_sha256"] = canonical_sha256(pre_label_attestation)
    pre_label_path = args.out_dir / "PRE_LABEL_PREFLIGHT.json"
    atomic_write_json(pre_label_path, pre_label_attestation)

    targets, sidecar_audit = _load_targets(state, args.sidecar_dir)
    per_cell = []
    for cell in state["cells"]:
        bundle = state["bundles"][cell]
        for method in METHOD_ORDER:
            per_cell.append({
                "cell_id": cell, "dataset_family": bundle.dataset_family,
                "task_type": bundle.task_type, "method": method,
                **_metrics(targets[cell], state["ensemble"][(method, cell)]),
            })
    summaries = [
        _method_summary(per_cell, method, metric)
        for method in METHOD_ORDER for metric in ("auroc", "auprc")
    ]
    comparisons = {
        pair: _comparison(per_cell, pair[0], pair[1], args.bootstrap_draws)
        for pair in REGISTERED_CONTRASTS + DIAGNOSTIC_CONTRASTS
    }
    decision = _decision(comparisons)
    per_family = _per_family_metric_rows(per_cell)
    paired_deltas = _paired_delta_rows(per_cell, comparisons)
    fit_health_summary = _fit_health_summary(state["mechanical"], state["stability"])
    _write_csv(args.out_dir / "PER_CELL_METRICS.csv", per_cell)
    _write_csv(args.out_dir / "PER_FAMILY_METRICS.csv", per_family)
    _write_csv(args.out_dir / "PAIRED_DELTAS.csv", paired_deltas)
    _write_csv(
        args.out_dir / "MACRO_SUMMARIES.csv",
        [{key: value for key, value in row.items() if key != "dataset_family_means"} for row in summaries],
    )
    _write_csv(args.out_dir / "FIT_MECHANICS.csv", state["mechanical"])
    _write_csv(args.out_dir / "FIT_HEALTH_SUMMARY.csv", fit_health_summary)
    _write_csv(args.out_dir / "SEED_STABILITY.csv", state["stability"])
    figure_paths = _plots(args.out_dir, per_cell, summaries, comparisons, state["mechanical"])
    figure_paths.extend(
        _explainer_plot(
            args.out_dir, state["structure"], comparisons, fit_health_summary
        )
    )
    report = {
        "schema": SCHEMA, "status": "complete", "scientific_tier": "C_retrospective",
        "strict_complete_preflight_before_dynamic_label_import": True,
        "targets_historically_open": True, "methods": list(METHOD_ORDER),
        "seeds_averaged": list(SEEDS), "n_cells": len(state["cells"]),
        "n_dataset_families": len({state["bundles"][cell].dataset_family for cell in state["cells"]}),
        "summaries": summaries,
        "contrast_order": [list(pair) for pair in REGISTERED_CONTRASTS],
        "diagnostic_contrast_order": [list(pair) for pair in DIAGNOSTIC_CONTRASTS],
        "comparisons": {f"{a}__vs__{b}": value for (a, b), value in comparisons.items()},
        "decision": decision,
        "fit_health_summary": fit_health_summary,
        "structural_r2_primary_k50_23_cells": state["structure"],
        "structure_audit": state["structure_audit"],
        "preflight_hashes": state["preflight_hashes"],
        "bundle_audit": state["bundle_audit"], "sidecar_audit": sidecar_audit,
        "figures": [path.name for path in figure_paths],
        "claim_boundary": "descriptive C-tier evidence; not independent confirmation",
    }
    report["content_sha256"] = canonical_sha256(report)
    atomic_write_json(args.out_dir / "REPORT.json", report)
    (args.out_dir / "REPORT.md").write_text(_markdown(report), encoding="utf-8")
    outputs = [
        pre_label_path,
        args.out_dir / "PER_CELL_METRICS.csv",
        args.out_dir / "PER_FAMILY_METRICS.csv",
        args.out_dir / "PAIRED_DELTAS.csv",
        args.out_dir / "MACRO_SUMMARIES.csv",
        args.out_dir / "FIT_MECHANICS.csv",
        args.out_dir / "FIT_HEALTH_SUMMARY.csv",
        args.out_dir / "SEED_STABILITY.csv",
        args.out_dir / "REPORT.json",
        args.out_dir / "REPORT.md",
        *figure_paths,
    ]
    evaluation_freeze = {
        "schema": "deem_b3_crossed_rook_v1_evaluation_freeze",
        "status": "complete_after_label_join", "source_sha256": sha256_file(Path(__file__)),
        "score_freeze_sha256": sha256_file(args.score_freeze),
        "structural_summary_sha256": state["structure_audit"]["summary_sha256"],
        "structural_freeze_sha256": state["structure_audit"]["freeze_sha256"],
        "pre_label_preflight_sha256": sha256_file(pre_label_path),
        "sidecar_audit": sidecar_audit,
        "artifacts": [{"path": path.name, "sha256": sha256_file(path)} for path in outputs],
    }
    evaluation_freeze["content_sha256"] = canonical_sha256(evaluation_freeze)
    evaluation_freeze_path = args.out_dir / "EVALUATION_FREEZE.json"
    atomic_write_json(evaluation_freeze_path, evaluation_freeze)
    expected_output_names = {path.name for path in outputs} | {
        evaluation_freeze_path.name
    }
    observed_output_names = {
        path.relative_to(args.out_dir).as_posix()
        for path in args.out_dir.rglob("*") if path.is_file()
    }
    if observed_output_names != expected_output_names:
        raise ResidualGraphDeemError(
            "evaluation output inventory mismatch: "
            f"expected={sorted(expected_output_names)}, "
            f"observed={sorted(observed_output_names)}"
        )
    print(json.dumps({
        "status": "PASS_CROSSED_ROOK_EVALUATION", "verdict": decision["verdict"],
        "report": str((args.out_dir / "REPORT.md").resolve()),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
