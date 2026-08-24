"""Audited graph-assumption diagnostics for frozen-24 reconstruction v1.

This module is deliberately downstream of the score freeze and evaluator.  It
never opens the legacy label bundle.  Correctness is read only from the hashed
``PREDICTION_SNAPSHOT.npz`` published by the evaluator after the independent
A/B score gate passed.

The diagnostics are explanatory, not another method-selection stage.  They
therefore copy (rather than recompute) any AUROC delta from ``EVALUATION.json``
and bind every diagnostic to the exact prepared matrix, fitted artifact,
graph/operator, score freeze, A/B certificate, and evaluation manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr, spearmanr

from .contracts import prepared_matrix_sha256
from .io import (
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)


DIAGNOSTIC_VERSION = "frozen24-graph-assumption-diagnostics-v2"
DIAGNOSTICS_SCHEMA_VERSION = "reconstruction-graph-assumption-diagnostics-v2"
MANIFEST_SCHEMA_VERSION = "reconstruction-graph-diagnostics-manifest-v2"
PLOT_DATA_SCHEMA_VERSION = "reconstruction-graph-diagnostic-plot-data-v2"
EXAMPLE_DATA_SCHEMA_VERSION = "reconstruction-example-graph-data-v2"
NODE_PERMUTATION_COUNT = 32
GRAPH_BOOTSTRAP_COUNT = 32
SU_SUPPORT_BOOTSTRAP_COUNT = 32
NODE_PERMUTATION_NULL_ID = "node_permutation_fixed_signal_v1"
CA_CONTROL_NULL_ID = "ca_alpha_control_v1"
CA_CONTROL_SERIES = (
    "learned",
    "equal_view",
    "provenance_prior",
    "global_mean_alpha",
    "permuted",
)
GRAPH_BOOTSTRAP_NULL_ID = "fixed_fitted_graph_source_group_bootstrap_v1"
RANDOM_FAMILY_NULL_ID = "random_provenance_family_subspace_v1"
LENGTH_ONLY_CONTROL_ID = "trace_length_only_self_safe_knn_k7_v1"
SU_SUPPORT_BOOTSTRAP_ID = "su_pcr_source_group_bootstrap_support_v1"
EXAMPLE_RULE_ID = "nuisance_available_then_connected_then_no_isolates_then_max_gap_then_min_degree_cv_then_cell_id_v2"

GRAPH_METHOD_IDS = ("dufs_liu", "ca_specrage_atomic", "pgrd_a")
NONGRAPH_METHOD_IDS = ("continuous_lsml", "family_nrm_a", "su_pcr")
DIAGNOSTIC_METHOD_IDS = GRAPH_METHOD_IDS + NONGRAPH_METHOD_IDS
EXPECTED_METHOD_COUNT = 13
EXPECTED_CELL_COUNT = 24
EXPECTED_PAIR_COUNT = EXPECTED_METHOD_COUNT * EXPECTED_CELL_COUNT
_SUCCESS = frozenset({"OK", "OK_FALLBACK"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_EPS = 1e-12

_COMMON_GRAPH_PANELS = (
    "graph_health",
    "target_vs_nuisance_roughness",
    "node_permutation_null",
    "roughness_null_summary",
    "alignment_vs_improvement",
    "fixed_graph_group_bootstrap_stability",
    "length_only_graph_control",
    "random_family_graph_control",
)
REQUIRED_PANELS_BY_METHOD = MappingProxyType({
    "dufs_liu": _COMMON_GRAPH_PANELS + (
        "dufs_gate_weights",
        "dufs_gate_weights_per_seed",
        "dufs_gate_stability",
        "dufs_seed_graph_stability",
    ),
    "ca_specrage_atomic": _COMMON_GRAPH_PANELS + (
        "ca_view_weights",
        "ca_alpha_stability",
        "ca_seed_graph_stability",
        "ca_alpha_controls",
    ),
    "pgrd_a": _COMMON_GRAPH_PANELS + (
        "pgrd_seed_graph_stability",
        "pgrd_cross_gradient",
        "pgrd_cross_gradient_null",
    ),
    "continuous_lsml": (
        "continuous_lsml_cluster_boundaries",
        "continuous_lsml_correlation_clusters",
    ),
    "family_nrm_a": (
        "family_nrm_residual_covariance",
        "family_nrm_residual_eigenspectrum",
        "family_nrm_family_contributions",
    ),
    "su_pcr": (
        "su_pcr_decomposition",
        "su_pcr_low_rank_eigenspectrum",
        "su_pcr_sparse_support",
        "su_pcr_sparse_support_stability",
    ),
})
SOURCE_DEPENDENCY_PATHS = (
    "spectral_utils/reconstruction_benchmark/graph_diagnostics.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/contracts.py",
    "spectral_utils/reconstruction_benchmark/preparation.py",
    "spectral_utils/reconstruction_benchmark/serialization.py",
    "spectral_utils/specrage_laplacian.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/graph_topology.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/fusion_aware_views.py",
)


class GraphDiagnosticContractError(RuntimeError):
    """A provenance, artifact, or numerical diagnostic contract failed."""


def capture_source_environment_snapshot(
    repo: Path,
    *,
    extra_source_paths: Sequence[str] = (),
    source_paths: Sequence[str] = SOURCE_DEPENDENCY_PATHS,
) -> dict[str, Any]:
    """Capture and hash a clean-git producer/numerical-environment snapshot."""

    root = Path(repo).resolve()
    _require((root / ".git").exists(), f"not a git worktree: {root}")

    def git(*arguments: str) -> str:
        result = subprocess.run(
            ("git", "-C", str(root), *arguments),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        _require(
            result.returncode == 0,
            f"git {' '.join(arguments)} failed: {result.stderr.strip()}",
        )
        return result.stdout.rstrip("\n")

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    _require(status == "", "graph diagnostics require a clean git worktree before launch")
    head = git("rev-parse", "HEAD")
    _require(_HEX64.fullmatch(head) is not None or re.fullmatch(r"[0-9a-f]{40}", head) is not None, "invalid git HEAD")
    paths = tuple(dict.fromkeys(tuple(source_paths) + tuple(extra_source_paths)))
    files: list[dict[str, str]] = []
    for relative_value in sorted(paths, key=lambda value: str(value).encode("utf-8")):
        relative = Path(str(relative_value))
        _require(not relative.is_absolute() and ".." not in relative.parts, "source snapshot path escapes repository")
        path = (root / relative).resolve()
        _require(root in path.parents and path.is_file(), f"source snapshot file missing: {relative}")
        files.append({
            "path": relative.as_posix(),
            "sha256": sha256_file(path),
        })
    packages: dict[str, str] = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "NOT_INSTALLED"
    snapshot: dict[str, Any] = {
        "schema_version": "graph-diagnostics-source-environment-snapshot-v1",
        "git_head": head,
        "git_status_porcelain": status,
        "source_files": files,
        "environment": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_executable": str(Path(sys.executable).resolve()),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": packages,
        },
    }
    snapshot["snapshot_sha256"] = sha256_bytes(canonical_json_bytes(snapshot))
    return snapshot


def assert_source_environment_snapshot_unchanged(
    repo: Path,
    before: Mapping[str, Any],
    *,
    extra_source_paths: Sequence[str] = (),
    source_paths: Sequence[str] = SOURCE_DEPENDENCY_PATHS,
) -> dict[str, Any]:
    after = capture_source_environment_snapshot(
        repo,
        extra_source_paths=extra_source_paths,
        source_paths=source_paths,
    )
    _require(
        canonical_json_bytes(dict(before)) == canonical_json_bytes(after),
        "source or numerical environment changed during graph-diagnostic production",
    )
    return after


@dataclass(frozen=True)
class VerifiedArtifact:
    method_id: str
    method_version_id: str
    config: Mapping[str, Any]
    status: str
    fallback_reason: str | None
    record: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]
    index: Mapping[str, Any]
    score: np.ndarray
    record_path: str
    record_sha256: str
    score_path: str
    score_sha256: str
    artifact_path: str | None
    artifact_sha256: str | None
    artifact_index_path: str
    artifact_index_sha256: str


@dataclass(frozen=True)
class VerifiedDiagnosticCell:
    cell_id: str
    domain: str
    row_ids: tuple[str, ...]
    group_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    X_confidence: np.ndarray
    y_error: np.ndarray
    trace_length_coordinate: np.ndarray | None
    feature_matrix_sha256: str
    prepared_matrix_sha256: str
    prepared_path: str
    prepared_sha256: str
    artifacts: Mapping[str, VerifiedArtifact]


@dataclass(frozen=True)
class VerifiedDiagnosticRelease:
    release_root: Path
    release_id: str
    cells: Mapping[str, VerifiedDiagnosticCell]
    method_ids: tuple[str, ...]
    method_versions: Mapping[str, str]
    evaluation: Mapping[str, Any]
    auroc_delta_vs_iu: Mapping[tuple[str, str], float]
    provenance: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphDiagnosticContractError(message)


def _strict_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and reject duplicate keys."""

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise GraphDiagnosticContractError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            output[key] = value
        return output

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
    except GraphDiagnosticContractError:
        raise
    except Exception as exc:
        raise GraphDiagnosticContractError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _verify_payload_hash(
    payload: Mapping[str, Any], field: str, *, context: str
) -> str:
    declared = payload.get(field)
    _require(
        isinstance(declared, str) and _HEX64.fullmatch(declared) is not None,
        f"{context}: missing or invalid {field}",
    )
    body = dict(payload)
    body.pop(field, None)
    observed = sha256_bytes(canonical_json_bytes(body))
    _require(observed == declared, f"{context}: {field} mismatch")
    return observed


def _verify_file(path: Path, declared: Any, *, context: str) -> str:
    _require(path.is_file(), f"{context}: missing file {path}")
    _require(
        isinstance(declared, str) and _HEX64.fullmatch(declared) is not None,
        f"{context}: invalid declared SHA-256",
    )
    observed = sha256_file(path)
    _require(observed == declared, f"{context}: file SHA-256 mismatch")
    return observed


def _safe_relative(root: Path, value: Any, *, context: str) -> tuple[Path, str]:
    _require(isinstance(value, str) and value.strip(), f"{context}: invalid path")
    relative = Path(value)
    _require(not relative.is_absolute(), f"{context}: expected relative path")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    _require(resolved_root in resolved.parents, f"{context}: path escapes release root")
    _require(resolved.is_file(), f"{context}: file is missing")
    return resolved, relative.as_posix()


def _feature_matrix_hash(matrix: np.ndarray, names: Sequence[str]) -> str:
    value = np.ascontiguousarray(np.asarray(matrix, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(value.tobytes(order="C"))
    digest.update(b"\0")
    digest.update(canonical_json_bytes(tuple(str(name) for name in names)))
    return digest.hexdigest()


def _optional_trace_length_coordinate(
    matrix: np.ndarray,
    names: Sequence[str],
) -> np.ndarray | None:
    names = tuple(str(value) for value in names)
    X = np.asarray(matrix, dtype=np.float64)
    _require(X.ndim == 2 and X.shape[1] == len(names), "trace coordinate matrix/name mismatch")
    if "trace_length" not in names:
        return None
    coordinate = np.asarray(X[:, names.index("trace_length")], dtype=np.float64)
    _require(np.isfinite(coordinate).all(), "trace_length coordinate is non-finite")
    return coordinate


def _canonical_sparse(matrix: sparse.spmatrix) -> sparse.csr_matrix:
    value = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    value.sum_duplicates()
    value.sort_indices()
    value.eliminate_zeros()
    return value


def sparse_sha256(matrix: sparse.spmatrix) -> str:
    """Hash one CSR matrix independent of platform-native integer widths."""

    value = _canonical_sparse(matrix)
    header = {
        "format": "csr",
        "shape": list(value.shape),
        "data_dtype": "float64-le",
        "index_dtype": "int64-le",
        "nnz": int(value.nnz),
    }
    digest = hashlib.sha256(canonical_json_bytes(header))
    digest.update(b"\0")
    digest.update(np.ascontiguousarray(value.data, dtype="<f8").tobytes())
    digest.update(np.ascontiguousarray(value.indices, dtype="<i8").tobytes())
    digest.update(np.ascontiguousarray(value.indptr, dtype="<i8").tobytes())
    return digest.hexdigest()


def _csr_from_flat(arrays: Mapping[str, np.ndarray], prefix: str) -> sparse.csr_matrix:
    keys = {
        suffix: f"{prefix}__{suffix}"
        for suffix in ("data", "indices", "indptr", "shape")
    }
    missing = sorted(name for name in keys.values() if name not in arrays)
    _require(not missing, f"artifact CSR {prefix!r} is incomplete: {missing}")
    shape_array = np.asarray(arrays[keys["shape"]], dtype=np.int64)
    _require(shape_array.shape == (2,), f"artifact CSR {prefix!r} has bad shape")
    shape = tuple(int(value) for value in shape_array)
    _require(all(value >= 0 for value in shape), f"artifact CSR {prefix!r} has negative shape")
    data = np.asarray(arrays[keys["data"]], dtype=np.float64)
    indices = np.asarray(arrays[keys["indices"]], dtype=np.int64)
    indptr = np.asarray(arrays[keys["indptr"]], dtype=np.int64)
    _require(np.isfinite(data).all(), f"artifact CSR {prefix!r} is non-finite")
    try:
        matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)
        matrix.check_format(full_check=True)
    except Exception as exc:
        raise GraphDiagnosticContractError(
            f"artifact CSR {prefix!r} is invalid: {exc}"
        ) from exc
    return _canonical_sparse(matrix)


def _has_complete_csr(arrays: Mapping[str, np.ndarray], prefix: str) -> bool:
    members = {
        f"{prefix}__{suffix}" for suffix in ("data", "indices", "indptr", "shape")
    }
    present = members.intersection(arrays)
    _require(
        not present or present == members,
        f"artifact CSR {prefix!r} is partially present: {sorted(present)}",
    )
    return present == members


def _validate_artifact_index(
    arrays: Mapping[str, np.ndarray], index: Mapping[str, Any], *, context: str
) -> None:
    expected_arrays: set[str] = set()
    for key, metadata in index.items():
        _require(isinstance(key, str) and key, f"{context}: invalid index key")
        if not isinstance(metadata, dict) or "type" not in metadata:
            continue
        kind = metadata.get("type")
        if kind == "ndarray":
            expected_arrays.add(key)
            _require(key in arrays, f"{context}: indexed ndarray {key!r} is absent")
            value = np.asarray(arrays[key])
            _require(
                list(value.shape) == metadata.get("shape"),
                f"{context}: ndarray shape mismatch for {key!r}",
            )
            _require(
                str(value.dtype) == metadata.get("dtype"),
                f"{context}: ndarray dtype mismatch for {key!r}",
            )
        elif kind == "csr_matrix":
            expected_arrays.update(
                f"{key}__{suffix}" for suffix in ("data", "indices", "indptr", "shape")
            )
            matrix = _csr_from_flat(arrays, key)
            _require(
                list(matrix.shape) == metadata.get("shape"),
                f"{context}: CSR shape mismatch for {key!r}",
            )
    _require(
        set(arrays) == expected_arrays,
        f"{context}: artifact archive/index member mismatch; "
        f"extra={sorted(set(arrays)-expected_arrays)}, "
        f"missing={sorted(expected_arrays-set(arrays))}",
    )


def symmetric_normalized_laplacian(graph: sparse.spmatrix) -> sparse.csr_matrix:
    graph = _validate_graph(graph)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    inverse = np.zeros_like(degree)
    positive = degree > _EPS
    inverse[positive] = 1.0 / np.sqrt(degree[positive])
    D = sparse.diags(inverse)
    return _canonical_sparse(sparse.eye(graph.shape[0], format="csr") - D @ graph @ D)


def _validate_graph(graph: sparse.spmatrix) -> sparse.csr_matrix:
    value = _canonical_sparse(graph)
    _require(
        value.ndim == 2 and value.shape[0] == value.shape[1] and value.shape[0] >= 3,
        "graph must be square with at least three nodes",
    )
    _require(np.isfinite(value.data).all(), "graph contains non-finite weights")
    _require(not value.nnz or float(np.min(value.data)) >= -1e-12, "graph has negative weights")
    diagonal = value.diagonal()
    _require(float(np.max(np.abs(diagonal))) <= 1e-10, "graph has a nonzero diagonal")
    difference = _canonical_sparse(value - value.T)
    symmetry_error = float(np.max(np.abs(difference.data))) if difference.nnz else 0.0
    _require(symmetry_error <= 1e-10, f"graph is asymmetric ({symmetry_error})")
    return value


def _sparse_max_abs(matrix: sparse.spmatrix) -> float:
    value = _canonical_sparse(matrix)
    return float(np.max(np.abs(value.data))) if value.nnz else 0.0


def _spectral_gap(laplacian: sparse.spmatrix, n_components: int) -> float:
    L = _canonical_sparse(laplacian)
    if n_components != 1:
        return 0.0
    n = L.shape[0]
    v0 = np.linspace(1.0, 2.0, n, dtype=np.float64)
    v0 /= np.linalg.norm(v0)
    failures: list[str] = []
    for kwargs in (
        {"which": "SM", "tol": 1e-9, "maxiter": max(5000, 20 * n)},
        {"which": "LM", "sigma": 1e-9, "tol": 1e-9, "maxiter": max(5000, 20 * n)},
    ):
        try:
            values = eigsh(
                L,
                k=2,
                return_eigenvectors=False,
                v0=v0,
                **kwargs,
            )
            values = np.sort(np.asarray(values, dtype=np.float64))
            gap = float(max(values[1], 0.0))
            _require(math.isfinite(gap), "normalized spectral gap is non-finite")
            return gap
        except Exception as exc:  # pragma: no cover - second solver is platform fallback
            failures.append(f"{type(exc).__name__}: {exc}")
    raise GraphDiagnosticContractError(
        "normalized spectral-gap solve failed: " + " | ".join(failures)
    )


def graph_health(graph: sparse.spmatrix, laplacian: sparse.spmatrix | None = None) -> dict[str, Any]:
    """Return deterministic target-free health of one undirected graph."""

    W = _validate_graph(graph)
    L_expected = symmetric_normalized_laplacian(W)
    if laplacian is not None:
        L = _canonical_sparse(laplacian)
        _require(L.shape == W.shape, "graph and Laplacian shapes disagree")
        _require(
            _sparse_max_abs(L - L_expected) <= 1e-9,
            "stored Laplacian does not match the graph",
        )
    else:
        L = L_expected
    degree = np.asarray(W.sum(axis=1)).ravel()
    n_components, _ = connected_components(W, directed=False)
    isolated = int(np.sum(degree <= _EPS))
    mean = float(np.mean(degree))
    return {
        "n_nodes": int(W.shape[0]),
        "n_edges": int(W.nnz // 2),
        "degree_min": float(np.min(degree)),
        "degree_mean": mean,
        "degree_max": float(np.max(degree)),
        "degree_cv": float(np.std(degree) / mean) if mean > _EPS else 0.0,
        "n_components": int(n_components),
        "isolated_nodes": isolated,
        "normalized_spectral_gap": _spectral_gap(L, int(n_components)),
        "graph_sha256": sparse_sha256(W),
        "operator_sha256": sparse_sha256(L),
    }


def normalized_roughness(values: np.ndarray, laplacian: sparse.spmatrix) -> float:
    """Centered normalized-L Rayleigh quotient ``z.T L z / z.T z``."""

    vector = np.asarray(values, dtype=np.float64)
    _require(vector.ndim == 1 and np.isfinite(vector).all(), "roughness vector is invalid")
    L = _canonical_sparse(laplacian)
    _require(L.shape == (len(vector), len(vector)), "roughness vector/operator mismatch")
    centered = vector - float(np.mean(vector))
    denominator = float(np.dot(centered, centered))
    _require(denominator > _EPS, "roughness is undefined for a constant vector")
    value = float(centered @ (L @ centered) / denominator)
    _require(math.isfinite(value), "roughness is non-finite")
    return value


def deterministic_diagnostic_seed(
    null_id: str, cell_id: str, method_id: str, draw_index: int
) -> int:
    payload = (
        f"{DIAGNOSTIC_VERSION}|{null_id}|"
        f"{cell_id}|{method_id}|{int(draw_index)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def deterministic_null_seed(cell_id: str, method_id: str, draw_index: int) -> int:
    return deterministic_diagnostic_seed(
        NODE_PERMUTATION_NULL_ID, cell_id, method_id, draw_index
    )


def node_permutation_nulls(
    *,
    graph: sparse.spmatrix,
    laplacian: sparse.spmatrix,
    target: np.ndarray,
    nuisance: np.ndarray | None,
    cell_id: str,
    method_id: str,
    count: int = NODE_PERMUTATION_COUNT,
) -> list[dict[str, Any]]:
    """Relabel nodes while keeping target/nuisance vectors fixed."""

    _require(count >= 20, "at least 20 node-permutation nulls are required")
    W = _validate_graph(graph)
    L = _canonical_sparse(laplacian)
    output = []
    for draw_index in range(int(count)):
        seed = deterministic_null_seed(cell_id, method_id, draw_index)
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(W.shape[0])
        perm_graph = _canonical_sparse(W[permutation][:, permutation])
        perm_laplacian = _canonical_sparse(L[permutation][:, permutation])
        output.append({
            "draw_index": draw_index,
            "seed": seed,
            "target_roughness": normalized_roughness(target, perm_laplacian),
            "nuisance_roughness": (
                normalized_roughness(nuisance, perm_laplacian)
                if nuisance is not None else None
            ),
            "graph_sha256": sparse_sha256(perm_graph),
            "operator_sha256": sparse_sha256(perm_laplacian),
        })
    return output


def _row_id_tie_ranks(row_ids: Sequence[str]) -> np.ndarray:
    values = tuple(str(value) for value in row_ids)
    _require(len(set(values)) == len(values), "row IDs must be unique for graph tie-breaking")
    order = sorted(range(len(values)), key=lambda index: values[index].encode("utf-8"))
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[np.asarray(order, dtype=np.int64)] = np.arange(len(values), dtype=np.float64)
    return ranks


def _group_bootstrap_row_multiplicity(
    group_ids: Sequence[str],
    *,
    null_id: str,
    cell_id: str,
    method_id: str,
    draw_index: int,
) -> tuple[np.ndarray, int]:
    groups = tuple(str(value) for value in group_ids)
    unique = tuple(sorted(set(groups), key=lambda value: value.encode("utf-8")))
    _require(len(unique) >= 2, "group bootstrap needs at least two source groups")
    lookup = {value: index for index, value in enumerate(unique)}
    row_group = np.asarray([lookup[value] for value in groups], dtype=np.int64)
    seed = deterministic_diagnostic_seed(null_id, cell_id, method_id, draw_index)
    sampled = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(unique), size=len(unique), endpoint=False
    )
    group_counts = np.bincount(sampled, minlength=len(unique)).astype(np.float64)
    return group_counts[row_group], seed


def fixed_graph_group_bootstrap(
    *,
    graph: sparse.spmatrix,
    group_ids: Sequence[str],
    cell_id: str,
    method_id: str,
    count: int = GRAPH_BOOTSTRAP_COUNT,
) -> list[dict[str, Any]]:
    """Probe one already fitted graph under grouped source multiplicities.

    This is deliberately not a graph refit.  For a source-group bootstrap with
    row multiplicities ``m``, it forms ``diag(sqrt(m)) W diag(sqrt(m))`` and
    compares that weighted induced graph with the frozen fitted graph.
    """

    _require(count >= 20, "at least 20 graph-bootstrap draws are required")
    W = _validate_graph(graph)
    L = symmetric_normalized_laplacian(W)
    output: list[dict[str, Any]] = []
    for draw_index in range(int(count)):
        multiplicity, seed = _group_bootstrap_row_multiplicity(
            group_ids,
            null_id=GRAPH_BOOTSTRAP_NULL_ID,
            cell_id=cell_id,
            method_id=method_id,
            draw_index=draw_index,
        )
        D = sparse.diags(np.sqrt(multiplicity))
        boot_graph = _canonical_sparse(D @ W @ D)
        boot_graph.setdiag(0.0)
        boot_graph.eliminate_zeros()
        boot_laplacian = symmetric_normalized_laplacian(boot_graph)
        similarity = graph_operator_similarity(W, L, boot_graph, boot_laplacian)
        output.append({
            "draw_index": int(draw_index),
            "seed": int(seed),
            "retained_node_fraction": float(np.mean(multiplicity > 0)),
            "effective_row_mass": float(np.sum(multiplicity)),
            "graph_sha256": sparse_sha256(boot_graph),
            "operator_sha256": sparse_sha256(boot_laplacian),
            **similarity,
        })
    return output


def _provenance_family_graphs(cell: VerifiedDiagnosticCell) -> dict[str, sparse.csr_matrix]:
    from ..graph_topology import self_safe_knn_graph
    from ..specrage_views import FEATURE_TO_VIEW, VIEW_ORDER

    names = cell.feature_names
    unknown = sorted(set(names) - set(FEATURE_TO_VIEW))
    _require(not unknown, f"{cell.cell_id}: unknown provenance-family features {unknown}")
    tie_keys = _row_id_tie_ranks(cell.row_ids)
    output: dict[str, sparse.csr_matrix] = {}
    for family in VIEW_ORDER:
        indices = [index for index, name in enumerate(names) if FEATURE_TO_VIEW[name] == family]
        if indices:
            output[family] = _validate_graph(self_safe_knn_graph(
                cell.X_confidence[:, indices], k=7, tie_keys=tie_keys
            ))
    _require(bool(output), f"{cell.cell_id}: no registered provenance families")
    return output


def select_example_cell(
    health_by_cell: Mapping[str, Mapping[str, Any]],
    nuisance_available_by_cell: Mapping[str, bool] | None = None,
) -> str:
    """Choose an example with a frozen, entirely label-free health rule."""

    _require(bool(health_by_cell), "no healthy graph is available for example selection")
    availability = nuisance_available_by_cell or {
        cell_id: True for cell_id in health_by_cell
    }
    _require(set(health_by_cell).issubset(availability), "nuisance-availability roster is incomplete")

    def key(item: tuple[str, Mapping[str, Any]]) -> tuple[Any, ...]:
        cell_id, health = item
        return (
            int(not bool(availability[cell_id])),
            int(health["n_components"] != 1),
            int(health["isolated_nodes"] != 0),
            -float(health["normalized_spectral_gap"]),
            float(health["degree_cv"]),
            cell_id.encode("utf-8"),
        )

    return min(health_by_cell.items(), key=key)[0]


def _pairwise_stability(values: np.ndarray, labels: Sequence[int]) -> list[dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float64)
    _require(matrix.ndim == 2 and matrix.shape[0] == len(labels), "stability matrix shape mismatch")
    output = []
    for left, right in itertools.combinations(range(matrix.shape[0]), 2):
        correlation = float(spearmanr(matrix[left], matrix[right]).statistic)
        cosine = float(
            np.dot(matrix[left], matrix[right])
            / (np.linalg.norm(matrix[left]) * np.linalg.norm(matrix[right]) + _EPS)
        )
        _require(math.isfinite(cosine), "seed cosine stability is non-finite")
        output.append({
            "left_seed": int(labels[left]),
            "right_seed": int(labels[right]),
            "spearman": correlation if math.isfinite(correlation) else None,
            "cosine": cosine,
        })
    return output


def _graph_pair_stability(
    graphs: Sequence[sparse.spmatrix], labels: Sequence[int]
) -> list[dict[str, Any]]:
    _require(len(graphs) == len(labels), "seed graph labels disagree")
    canonical = [_validate_graph(graph) for graph in graphs]
    output = []
    for left, right in itertools.combinations(range(len(canonical)), 2):
        A, B = canonical[left], canonical[right]
        _require(A.shape == B.shape, "seed graph shapes disagree")
        support_a = A.copy()
        support_b = B.copy()
        support_a.data[:] = 1.0
        support_b.data[:] = 1.0
        intersection = float(support_a.multiply(support_b).sum())
        union = float((support_a + support_b).astype(bool).sum())
        product = float(A.multiply(B).sum())
        cosine = product / (
            math.sqrt(float(A.multiply(A).sum()) * float(B.multiply(B).sum())) + _EPS
        )
        output.append({
            "left_seed": int(labels[left]),
            "right_seed": int(labels[right]),
            "edge_jaccard": intersection / (union + _EPS),
            "weighted_frobenius_cosine": cosine,
        })
    return output


def graph_operator_similarity(
    left_graph: sparse.spmatrix,
    left_operator: sparse.spmatrix,
    right_graph: sparse.spmatrix,
    right_operator: sparse.spmatrix,
) -> dict[str, float]:
    """Compare two fitted graphs/operators on the exact same rows."""

    A, B = _validate_graph(left_graph), _validate_graph(right_graph)
    LA, LB = _canonical_sparse(left_operator), _canonical_sparse(right_operator)
    _require(A.shape == B.shape == LA.shape == LB.shape, "graph/operator comparison shape mismatch")
    support_a, support_b = A.copy(), B.copy()
    support_a.data[:] = 1.0
    support_b.data[:] = 1.0
    intersection = float(support_a.multiply(support_b).sum())
    union = float((support_a + support_b).astype(bool).sum())

    def cosine(left: sparse.csr_matrix, right: sparse.csr_matrix) -> float:
        numerator = float(left.multiply(right).sum())
        denominator = math.sqrt(
            float(left.multiply(left).sum()) * float(right.multiply(right).sum())
        ) + _EPS
        return numerator / denominator

    operator_scale = 0.5 * (
        math.sqrt(float(LA.multiply(LA).sum()))
        + math.sqrt(float(LB.multiply(LB).sum()))
    )
    return {
        "edge_support_jaccard": intersection / (union + _EPS),
        "weighted_graph_frobenius_cosine": cosine(A, B),
        "normalized_laplacian_frobenius_cosine": cosine(LA, LB),
        "normalized_laplacian_relative_difference": (
            math.sqrt(float((LA - LB).multiply(LA - LB).sum()))
            / (operator_scale + _EPS)
        ),
    }


def _safe_artifact_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def _verify_artifact(
    *,
    release_root: Path,
    fit_root: Path,
    cell_id: str,
    method_id: str,
    frozen: Mapping[str, Any],
    attested: Mapping[str, Any],
    prepared_matrix_hash: str,
    row_ids: tuple[str, ...],
    method_config: Mapping[str, Any],
) -> VerifiedArtifact:
    method_dir = fit_root / "cells" / cell_id / method_id
    record_path = method_dir / "RECORD.json"
    record = _strict_json(record_path)
    record_sha = _verify_file(
        record_path, frozen.get("record_sha256"), context=f"{cell_id}/{method_id} record"
    )
    _require(record_sha == attested.get("record_sha256"), f"{cell_id}/{method_id}: A/B record binding drift")
    for key in ("cell_id", "method_id", "method_version_id", "config_sha256", "prepared_matrix_sha256"):
        expected = (
            cell_id if key == "cell_id" else
            method_id if key == "method_id" else
            prepared_matrix_hash if key == "prepared_matrix_sha256" else
            frozen.get(key)
        )
        _require(record.get(key) == expected, f"{cell_id}/{method_id}: {key} mismatch")
    _require(record.get("status") in _SUCCESS, f"{cell_id}/{method_id}: fit did not succeed")
    _require(record.get("score_path") == "score.npz", f"{cell_id}/{method_id}: noncanonical score path")
    _require(record.get("score_semantics") == "higher_is_incorrect", f"{cell_id}/{method_id}: score semantics drift")
    _require(record.get("positive_class") == "incorrect", f"{cell_id}/{method_id}: positive class drift")

    score_path = method_dir / "score.npz"
    score_sha = _verify_file(
        score_path, record.get("score_sha256"), context=f"{cell_id}/{method_id} score"
    )
    _require(score_sha == frozen.get("score_sha256") == attested.get("score_sha256"), f"{cell_id}/{method_id}: score hash binding drift")
    score_arrays = load_npz_no_pickle(score_path)
    _require(set(score_arrays) == {"row_ids", "score"}, f"{cell_id}/{method_id}: score members drifted")
    score_rows = tuple(str(value) for value in score_arrays["row_ids"].tolist())
    score = np.asarray(score_arrays["score"], dtype=np.float64)
    _require(score_rows == row_ids and score.shape == (len(row_ids),), f"{cell_id}/{method_id}: score row mismatch")
    _require(np.isfinite(score).all(), f"{cell_id}/{method_id}: score is non-finite")
    _require(record.get("score_n") == len(row_ids), f"{cell_id}/{method_id}: score_n drift")

    index_path = method_dir / "ARTIFACT_INDEX.json"
    index_sha = _verify_file(
        index_path,
        record.get("artifact_index_sha256"),
        context=f"{cell_id}/{method_id} artifact index",
    )
    _require(index_sha == frozen.get("artifact_index_sha256"), f"{cell_id}/{method_id}: artifact-index freeze drift")
    index = _strict_json(index_path)

    artifact_relative: str | None = None
    artifact_sha: str | None = None
    arrays: dict[str, np.ndarray] = {}
    if record.get("artifacts_path") is not None:
        _require(record.get("artifacts_path") == "artifacts.npz", f"{cell_id}/{method_id}: noncanonical artifact path")
        artifact_path = method_dir / "artifacts.npz"
        artifact_sha = _verify_file(
            artifact_path,
            record.get("artifacts_sha256"),
            context=f"{cell_id}/{method_id} artifacts",
        )
        _require(
            artifact_sha == frozen.get("artifacts_sha256") == attested.get("artifacts_sha256"),
            f"{cell_id}/{method_id}: artifact hash binding drift",
        )
        arrays = load_npz_no_pickle(artifact_path)
        _validate_artifact_index(arrays, index, context=f"{cell_id}/{method_id}")
        artifact_relative = artifact_path.relative_to(release_root).as_posix()
    else:
        _require(
            record.get("artifacts_sha256") is None
            and frozen.get("artifacts_sha256") is None
            and attested.get("artifacts_sha256") is None,
            f"{cell_id}/{method_id}: null artifact binding drift",
        )
        _require(not index, f"{cell_id}/{method_id}: artifact index is nonempty without archive")

    return VerifiedArtifact(
        method_id=method_id,
        method_version_id=str(record["method_version_id"]),
        config=MappingProxyType(dict(method_config)),
        status=str(record["status"]),
        fallback_reason=record.get("fallback_reason"),
        record=MappingProxyType(record),
        arrays=MappingProxyType(arrays),
        index=MappingProxyType(index),
        score=score,
        record_path=record_path.relative_to(release_root).as_posix(),
        record_sha256=record_sha,
        score_path=score_path.relative_to(release_root).as_posix(),
        score_sha256=score_sha,
        artifact_path=artifact_relative,
        artifact_sha256=artifact_sha,
        artifact_index_path=index_path.relative_to(release_root).as_posix(),
        artifact_index_sha256=index_sha,
    )


def verify_diagnostic_release(release_root: str | Path) -> VerifiedDiagnosticRelease:
    """Independently verify all frozen inputs used by the diagnostics.

    The legacy ``cells.npz`` label bundle is neither accepted nor opened.
    """

    release = Path(release_root).resolve()
    _require(release.is_dir(), f"release root is missing: {release}")
    release_id = release.name

    ab_path = release / "SCORE_AB_VERIFICATION.json"
    ab = _strict_json(ab_path)
    _verify_payload_hash(ab, "payload_sha256", context="A/B certificate")
    ab_sha = sha256_file(ab_path)
    _require(ab.get("schema_version") == "reconstruction-score-ab-verification-v1", "A/B schema drift")
    _require(ab.get("pass") is True, "A/B certificate is not PASS")
    _require(ab.get("n_cells") == EXPECTED_CELL_COUNT, "A/B certificate cell count drift")
    _require(ab.get("n_methods") == EXPECTED_METHOD_COUNT, "A/B certificate method count drift")
    _require(ab.get("n_pairs") == EXPECTED_PAIR_COUNT, "A/B certificate pair count drift")
    cell_ids = tuple(str(value) for value in ab.get("cell_ids", ()))
    method_ids = tuple(str(value) for value in ab.get("method_ids", ()))
    _require(len(cell_ids) == EXPECTED_CELL_COUNT and len(set(cell_ids)) == len(cell_ids), "A/B cell roster invalid")
    _require(len(method_ids) == EXPECTED_METHOD_COUNT and len(set(method_ids)) == len(method_ids), "A/B method roster invalid")
    _require(set(DIAGNOSTIC_METHOD_IDS).issubset(method_ids), "diagnostic method is absent from A/B roster")
    pair_rows = ab.get("pairs")
    _require(isinstance(pair_rows, list) and len(pair_rows) == EXPECTED_PAIR_COUNT, "A/B pair ledger invalid")
    attested: dict[tuple[str, str], dict[str, Any]] = {}
    for row in pair_rows:
        _require(isinstance(row, dict), "A/B pair ledger contains a non-object")
        key = (str(row.get("cell_id", "")), str(row.get("method_id", "")))
        _require(key not in attested, f"duplicate A/B pair {key}")
        _require(key[0] in cell_ids and key[1] in method_ids, f"unknown A/B pair {key}")
        _require(row.get("byte_identical") is True, f"A/B pair is not byte-identical: {key}")
        attested[key] = row
    _require(len(attested) == EXPECTED_PAIR_COUNT, "A/B pair Cartesian set incomplete")

    # The certificate is not treated as a free-standing declaration: verify
    # the four immutable manifests it names before using its pair ledger.
    _verify_file(
        release / "build_B" / "inputs" / "MANIFEST.json",
        ab.get("input_manifest_B_sha256"),
        context="A/B certificate Build-B input reference",
    )
    _verify_file(
        release / "build_B" / "fit" / "SCORE_FREEZE_MANIFEST.json",
        ab.get("freeze_B_sha256"),
        context="A/B certificate Build-B freeze reference",
    )

    input_root = release / "build_A" / "inputs"
    input_manifest_path = input_root / "MANIFEST.json"
    input_manifest = _strict_json(input_manifest_path)
    _verify_payload_hash(input_manifest, "manifest_payload_sha256", context="Build-A input manifest")
    input_manifest_sha = _verify_file(
        input_manifest_path,
        ab.get("input_manifest_A_sha256"),
        context="Build-A input manifest",
    )
    _require(input_manifest.get("schema_version") == "reconstruction-target-free-input-v1", "prepared manifest schema drift")
    _require(input_manifest.get("build_id") == "A", "prepared manifest is not Build A")
    _require(input_manifest.get("scientific_run") is True, "prepared manifest is not scientific")
    _require(input_manifest.get("label_arrays_accessed") is False, "prepared manifest opened labels")
    _require(input_manifest.get("matrix_semantics") == "higher_is_confidence", "prepared matrix semantics drift")
    input_rows = input_manifest.get("cells")
    _require(isinstance(input_rows, list) and len(input_rows) == EXPECTED_CELL_COUNT, "prepared manifest must contain 24 cells")
    _require(tuple(str(row.get("cell_id", "")) for row in input_rows) == cell_ids, "prepared cell order differs from A/B certificate")

    fit_root = release / "build_A" / "fit"
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = _strict_json(freeze_path)
    _verify_payload_hash(freeze, "payload_sha256", context="Build-A score freeze")
    freeze_sha = _verify_file(freeze_path, ab.get("freeze_A_sha256"), context="Build-A score freeze")
    _require(freeze.get("schema_version") == "reconstruction-score-freeze-v1", "score-freeze schema drift")
    _require(freeze.get("build_id") == "A", "score freeze is not Build A")
    _require(freeze.get("labels_opened_by_fit") is False and freeze.get("runtime_labels_used") is False, "score freeze used labels")
    _require(freeze.get("all_headline_scores_present") is True, "score freeze is incomplete")
    _require(freeze.get("input_manifest_file_sha256") == input_manifest_sha, "score freeze input-manifest binding drift")
    _require(tuple(freeze.get("cell_ids", ())) == cell_ids, "score-freeze cell roster drift")
    _require(tuple(freeze.get("method_ids", ())) == method_ids, "score-freeze method roster drift")
    frozen_rows = freeze.get("records")
    _require(isinstance(frozen_rows, list) and len(frozen_rows) == EXPECTED_PAIR_COUNT, "score-freeze pair ledger invalid")
    frozen = {(str(row["cell_id"]), str(row["method_id"])): row for row in frozen_rows}
    _require(len(frozen) == EXPECTED_PAIR_COUNT and set(frozen) == set(attested), "score-freeze Cartesian set drift")
    method_specs = freeze.get("method_specs")
    _require(isinstance(method_specs, dict) and set(method_specs) == set(method_ids), "score-freeze method specs drift")

    # Verify every Build-A score and artifact hash, not only the five methods
    # analyzed below.  This makes the diagnostics consume the same exact score
    # freeze that the evaluator certified.
    prepared: dict[str, dict[str, Any]] = {}
    diagnostic_artifacts: dict[tuple[str, str], VerifiedArtifact] = {}
    all_scores: dict[tuple[str, str], np.ndarray] = {}
    for cell_row in input_rows:
        cell_id = str(cell_row["cell_id"])
        prepared_path, prepared_relative = _safe_relative(
            input_root, cell_row.get("artifact_path"), context=f"{cell_id} prepared artifact"
        )
        prepared_sha = _verify_file(
            prepared_path,
            cell_row.get("artifact_sha256"),
            context=f"{cell_id} prepared artifact",
        )
        arrays = load_npz_no_pickle(prepared_path)
        _require(
            set(arrays) == {"X_confidence", "feature_names", "family_ids", "row_ids", "row_index"},
            f"{cell_id}: prepared members drifted",
        )
        X = np.asarray(arrays["X_confidence"], dtype=np.float64)
        names = tuple(str(value) for value in arrays["feature_names"].tolist())
        rows = tuple(str(value) for value in arrays["row_ids"].tolist())
        _require(X.shape == (len(rows), len(names)), f"{cell_id}: prepared shape drift")
        _require(np.isfinite(X).all(), f"{cell_id}: prepared matrix is non-finite")
        matrix_hash = _feature_matrix_hash(X, names)
        _require(matrix_hash == cell_row.get("feature_matrix_sha256"), f"{cell_id}: feature-matrix hash drift")
        strong_hash = prepared_matrix_sha256(X, names, rows)
        prepared[cell_id] = {
            "domain": str(cell_row["domain"]),
            "X": X,
            "names": names,
            "rows": rows,
            "matrix_hash": matrix_hash,
            "strong_hash": strong_hash,
            "path": (Path("build_A/inputs") / prepared_relative).as_posix(),
            "sha": prepared_sha,
        }
        for method_id in method_ids:
            spec = method_specs[method_id]
            _require(isinstance(spec, dict) and isinstance(spec.get("config"), dict), f"{method_id}: frozen method spec invalid")
            artifact = _verify_artifact(
                release_root=release,
                fit_root=fit_root,
                cell_id=cell_id,
                method_id=method_id,
                frozen=frozen[(cell_id, method_id)],
                attested=attested[(cell_id, method_id)],
                prepared_matrix_hash=strong_hash,
                row_ids=rows,
                method_config=spec["config"],
            )
            all_scores[(cell_id, method_id)] = artifact.score
            if method_id in DIAGNOSTIC_METHOD_IDS:
                diagnostic_artifacts[(cell_id, method_id)] = artifact

    evaluation_dir = release / "evaluation"
    evaluation_manifest_path = evaluation_dir / "EVALUATION_MANIFEST.json"
    evaluation_manifest = _strict_json(evaluation_manifest_path)
    _verify_payload_hash(evaluation_manifest, "payload_sha256", context="evaluation manifest")
    evaluation_manifest_sha = sha256_file(evaluation_manifest_path)
    _require(evaluation_manifest.get("schema_version") == "reconstruction-evaluation-manifest-v1", "evaluation manifest schema drift")
    _require(evaluation_manifest.get("n_cells") == EXPECTED_CELL_COUNT and evaluation_manifest.get("n_methods") == EXPECTED_METHOD_COUNT, "evaluation coverage drift")
    _require(evaluation_manifest.get("bootstrap_draws") == 20_000, "evaluation is not the canonical 20,000-draw run")
    _require(evaluation_manifest.get("status") == "OK", "evaluation is not publishable")
    evaluation_path, evaluation_relative = _safe_relative(
        evaluation_dir,
        evaluation_manifest.get("evaluation_path"),
        context="evaluation payload",
    )
    evaluation_sha = _verify_file(
        evaluation_path,
        evaluation_manifest.get("evaluation_sha256"),
        context="evaluation payload",
    )
    evaluation = _strict_json(evaluation_path)
    _verify_payload_hash(evaluation, "payload_sha256", context="evaluation payload")
    _require(evaluation.get("schema_version") == "reconstruction-24cell-evaluation-v1", "evaluation schema drift")
    _require(evaluation.get("status") == "OK", "evaluation headline is blocked")
    _require(evaluation.get("population_id") == "frozen24_response_v1", "evaluation population drift")
    _require(evaluation.get("positive_class") == "incorrect", "evaluation positive class drift")
    _require(evaluation.get("score_semantics") == "higher_is_incorrect", "evaluation score semantics drift")
    _require(tuple(evaluation.get("method_ids", ())) == method_ids, "evaluation method roster drift")
    _require(evaluation.get("n_cells") == EXPECTED_CELL_COUNT, "evaluation cell count drift")
    evaluation_provenance = evaluation.get("provenance")
    _require(isinstance(evaluation_provenance, dict), "evaluation provenance is absent")
    _require(evaluation_provenance.get("score_ab_verification_sha256") == ab_sha, "evaluation is bound to another A/B certificate")
    _require(evaluation_provenance.get("freeze_A_sha256") == freeze_sha, "evaluation is bound to another Build-A freeze")
    _require(evaluation_provenance.get("input_manifest_A_sha256") == input_manifest_sha, "evaluation is bound to another Build-A input")
    _require(evaluation_manifest.get("input_provenance") == evaluation_provenance, "evaluation manifest provenance drift")

    snapshot_path, snapshot_relative = _safe_relative(
        evaluation_dir,
        evaluation_manifest.get("prediction_snapshot_path"),
        context="prediction snapshot",
    )
    snapshot_sha = _verify_file(
        snapshot_path,
        evaluation_manifest.get("prediction_snapshot_sha256"),
        context="prediction snapshot",
    )
    _require(
        evaluation_manifest.get("prediction_snapshot_schema")
        == "reconstruction-prediction-snapshot-v1",
        "prediction snapshot schema drift",
    )
    snapshot = load_npz_no_pickle(snapshot_path)
    expected_members: set[str] = set()
    for cell_id in cell_ids:
        expected_members.update({
            f"{cell_id}__row_ids",
            f"{cell_id}__group_ids",
            f"{cell_id}__y_error",
        })
        expected_members.update(
            f"{cell_id}__{method_id}__score" for method_id in method_ids
        )
    _require(set(snapshot) == expected_members, "prediction snapshot member roster drift")

    cells: dict[str, VerifiedDiagnosticCell] = {}
    for cell_id in cell_ids:
        source = prepared[cell_id]
        snapshot_rows = tuple(str(value) for value in snapshot[f"{cell_id}__row_ids"].tolist())
        _require(snapshot_rows == source["rows"], f"{cell_id}: prediction row order drift")
        groups = np.asarray(snapshot[f"{cell_id}__group_ids"])
        _require(groups.shape == (len(snapshot_rows),), f"{cell_id}: prediction group length drift")
        group_ids = tuple(str(value) for value in groups.tolist())
        _require(all(value.strip() for value in group_ids), f"{cell_id}: prediction group ID is blank")
        y_error = np.asarray(snapshot[f"{cell_id}__y_error"], dtype=np.int8)
        _require(y_error.shape == (len(snapshot_rows),), f"{cell_id}: prediction target length drift")
        _require(np.isin(y_error, (0, 1)).all(), f"{cell_id}: prediction target is non-binary")
        for method_id in method_ids:
            score = np.asarray(snapshot[f"{cell_id}__{method_id}__score"], dtype=np.float64)
            _require(
                np.array_equal(score, all_scores[(cell_id, method_id)]),
                f"{cell_id}/{method_id}: prediction snapshot score differs from Build A",
            )
        trace_coordinate = _optional_trace_length_coordinate(
            source["X"], source["names"]
        )
        cells[cell_id] = VerifiedDiagnosticCell(
            cell_id=cell_id,
            domain=source["domain"],
            row_ids=source["rows"],
            group_ids=group_ids,
            feature_names=source["names"],
            X_confidence=source["X"],
            y_error=y_error,
            trace_length_coordinate=trace_coordinate,
            feature_matrix_sha256=source["matrix_hash"],
            prepared_matrix_sha256=source["strong_hash"],
            prepared_path=source["path"],
            prepared_sha256=source["sha"],
            artifacts=MappingProxyType({
                method_id: diagnostic_artifacts[(cell_id, method_id)]
                for method_id in DIAGNOSTIC_METHOD_IDS
            }),
        )

    deltas: dict[tuple[str, str], float] = {}
    contrast_rows = evaluation.get("paired_contrasts_vs_iu_pcr")
    _require(isinstance(contrast_rows, list), "evaluation contrast ledger is absent")
    for row in contrast_rows:
        if (
            isinstance(row, dict)
            and row.get("scope_type") == "cell"
            and row.get("metric") == "auroc"
            and row.get("candidate_method_id") in GRAPH_METHOD_IDS
            and row.get("status") == "OK"
        ):
            cell_id = str(row.get("scope_value", ""))
            method_id = str(row["candidate_method_id"])
            _require(row.get("cell_ids") == [cell_id], f"{cell_id}/{method_id}: cell contrast identity drift")
            value = row.get("delta")
            _require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{cell_id}/{method_id}: AUROC delta invalid")
            key = (cell_id, method_id)
            _require(key not in deltas, f"duplicate evaluation contrast {key}")
            deltas[key] = float(value)
    _require(len(deltas) == EXPECTED_CELL_COUNT * len(GRAPH_METHOD_IDS), "evaluation lacks exact graph-method per-cell AUROC deltas")

    method_versions = {
        method_id: str(method_specs[method_id]["method_version_id"])
        for method_id in method_ids
    }
    provenance = {
        "score_ab_verification_path": "SCORE_AB_VERIFICATION.json",
        "score_ab_verification_sha256": ab_sha,
        "score_freeze_A_path": "build_A/fit/SCORE_FREEZE_MANIFEST.json",
        "score_freeze_A_sha256": freeze_sha,
        "input_manifest_A_path": "build_A/inputs/MANIFEST.json",
        "input_manifest_A_sha256": input_manifest_sha,
        "evaluation_manifest_path": "evaluation/EVALUATION_MANIFEST.json",
        "evaluation_manifest_sha256": evaluation_manifest_sha,
        "evaluation_path": (Path("evaluation") / evaluation_relative).as_posix(),
        "evaluation_sha256": evaluation_sha,
        "prediction_snapshot_path": (Path("evaluation") / snapshot_relative).as_posix(),
        "prediction_snapshot_sha256": snapshot_sha,
        "raw_label_bundle_opened": False,
        "targets_source": "hashed evaluator prediction snapshot only",
    }
    return VerifiedDiagnosticRelease(
        release_root=release,
        release_id=release_id,
        cells=MappingProxyType(cells),
        method_ids=method_ids,
        method_versions=MappingProxyType(method_versions),
        evaluation=MappingProxyType(evaluation),
        auroc_delta_vs_iu=MappingProxyType(deltas),
        provenance=MappingProxyType(provenance),
    )


class _RecordBuilder:
    """Build canonical scalar records and their plot-table projection."""

    def __init__(
        self,
        verified: VerifiedDiagnosticRelease,
        producer_snapshot_sha256: str | None = None,
    ):
        self.verified = verified
        self.producer_snapshot_sha256 = producer_snapshot_sha256
        self.records: list[dict[str, Any]] = []
        self.plot_rows: list[dict[str, Any]] = []
        self.bindings: dict[str, dict[str, Any]] = {}

    def binding(self, cell: VerifiedDiagnosticCell, artifact: VerifiedArtifact) -> str:
        payload = {
            "binding_type": "single_method_artifact",
            "cell_id": cell.cell_id,
            "method_id": artifact.method_id,
            "method_version_id": artifact.method_version_id,
            "feature_matrix_sha256": cell.feature_matrix_sha256,
            "prepared_matrix_sha256": cell.prepared_matrix_sha256,
            "prepared_artifact_path": cell.prepared_path,
            "prepared_artifact_sha256": cell.prepared_sha256,
            "score_record_path": artifact.record_path,
            "score_record_sha256": artifact.record_sha256,
            "score_path": artifact.score_path,
            "score_sha256": artifact.score_sha256,
            "method_artifact_path": artifact.artifact_path,
            "method_artifact_sha256": artifact.artifact_sha256,
            "artifact_index_path": artifact.artifact_index_path,
            "artifact_index_sha256": artifact.artifact_index_sha256,
            "score_freeze_A_path": self.verified.provenance["score_freeze_A_path"],
            "score_freeze_A_sha256": self.verified.provenance["score_freeze_A_sha256"],
            "score_ab_verification_path": self.verified.provenance["score_ab_verification_path"],
            "score_ab_verification_sha256": self.verified.provenance["score_ab_verification_sha256"],
            "evaluation_manifest_path": self.verified.provenance["evaluation_manifest_path"],
            "evaluation_manifest_sha256": self.verified.provenance["evaluation_manifest_sha256"],
            "prediction_snapshot_path": self.verified.provenance["prediction_snapshot_path"],
            "prediction_snapshot_sha256": self.verified.provenance["prediction_snapshot_sha256"],
            "producer_snapshot_sha256": self.producer_snapshot_sha256,
        }
        binding_id = "binding_" + sha256_bytes(canonical_json_bytes(payload))[:20]
        existing = self.bindings.get(binding_id)
        _require(existing is None or existing == payload, "source-binding hash collision")
        self.bindings[binding_id] = payload
        return binding_id

    def pair_binding(
        self,
        cell: VerifiedDiagnosticCell,
        left: VerifiedArtifact,
        right: VerifiedArtifact,
    ) -> str:
        left_id = self.binding(cell, left)
        right_id = self.binding(cell, right)
        payload = {
            "binding_type": "paired_method_artifacts",
            "cell_id": cell.cell_id,
            "left_source_binding_id": left_id,
            "right_source_binding_id": right_id,
        }
        binding_id = "binding_" + sha256_bytes(canonical_json_bytes(payload))[:20]
        existing = self.bindings.get(binding_id)
        _require(existing is None or existing == payload, "paired source-binding hash collision")
        self.bindings[binding_id] = payload
        return binding_id

    def release_binding(
        self,
        method_id: str,
        cell_ids: Sequence[str],
    ) -> str:
        ordered_cells = tuple(sorted((str(value) for value in cell_ids), key=lambda value: value.encode("utf-8")))
        _require(bool(ordered_cells), f"{method_id}: empty multi-cell binding")
        _require(len(set(ordered_cells)) == len(ordered_cells), f"{method_id}: duplicate cell in multi-cell binding")
        child_bindings = []
        for cell_id in ordered_cells:
            _require(cell_id in self.verified.cells, f"{method_id}: unknown cell in multi-cell binding")
            cell = self.verified.cells[cell_id]
            child_bindings.append({
                "cell_id": cell_id,
                "source_binding_id": self.binding(cell, cell.artifacts[method_id]),
            })
        payload = {
            "binding_type": "multi_cell_method_artifacts",
            "release_id": self.verified.release_id,
            "method_id": method_id,
            "method_version_id": self.verified.method_versions[method_id],
            "cell_source_bindings": child_bindings,
            "evaluation_manifest_sha256": self.verified.provenance["evaluation_manifest_sha256"],
            "prediction_snapshot_sha256": self.verified.provenance["prediction_snapshot_sha256"],
            "producer_snapshot_sha256": self.producer_snapshot_sha256,
        }
        binding_id = "binding_" + sha256_bytes(canonical_json_bytes(payload))[:20]
        existing = self.bindings.get(binding_id)
        _require(existing is None or existing == payload, "multi-cell source-binding hash collision")
        self.bindings[binding_id] = payload
        return binding_id

    def add(
        self,
        *,
        cell: VerifiedDiagnosticCell,
        artifact: VerifiedArtifact,
        stage: str,
        panel_id: str,
        metric_id: str,
        value: float | int | None,
        unit: str,
        graph_sha256: str | None = None,
        operator_sha256: str | None = None,
        compared_artifact: VerifiedArtifact | None = None,
        compared_graph_sha256: str | None = None,
        compared_operator_sha256: str | None = None,
        null_id: str | None = None,
        seed: int | None = None,
        draw_index: int | None = None,
        series_id: str = "observed",
        x_index: int = 0,
        x_value: float = 0.0,
        status: str = "OK",
        note: str | None = None,
        scope_type: str = "cell",
        scope_value: str | None = None,
        cell_id_override: str | None = None,
        source_binding_id_override: str | None = None,
        feature_matrix_sha256_override: str | None = None,
    ) -> str:
        _require(stage in {"target_free", "post_freeze"}, "invalid diagnostic stage")
        _require(scope_type in {"cell", "release"}, "invalid diagnostic scope")
        effective_cell_id = cell.cell_id if cell_id_override is None else str(cell_id_override)
        effective_scope_value = cell.cell_id if scope_value is None else str(scope_value)
        if scope_type == "cell":
            _require(effective_cell_id == cell.cell_id, "cell-scoped diagnostic cannot override cell ID")
            _require(effective_scope_value == cell.cell_id, "cell-scoped diagnostic scope drift")
        else:
            _require(effective_cell_id == "__release__", "release-scoped diagnostic must use __release__ cell ID")
            _require(effective_scope_value == self.verified.release_id, "release-scoped diagnostic release drift")
        if status == "OK":
            _require(isinstance(value, (int, float, np.integer, np.floating)), "OK diagnostic lacks numeric value")
            value = float(value)
            _require(math.isfinite(value), "OK diagnostic value is non-finite")
        else:
            _require(value is None, "unavailable diagnostic must not carry a numeric value")
        binding_id = source_binding_id_override or (
            self.pair_binding(cell, artifact, compared_artifact)
            if compared_artifact is not None
            else self.binding(cell, artifact)
        )
        _require(binding_id in self.bindings, "diagnostic references an unknown source binding")
        feature_hash = feature_matrix_sha256_override or cell.feature_matrix_sha256
        identity = {
            "diagnostic_version": DIAGNOSTIC_VERSION,
            "scope_type": scope_type,
            "scope_value": effective_scope_value,
            "cell_id": effective_cell_id,
            "method_id": artifact.method_id,
            "method_version_id": artifact.method_version_id,
            "compared_method_id": (
                compared_artifact.method_id if compared_artifact is not None else None
            ),
            "compared_method_version_id": (
                compared_artifact.method_version_id if compared_artifact is not None else None
            ),
            "stage": stage,
            "panel_id": panel_id,
            "metric_id": metric_id,
            "series_id": series_id,
            "x_index": int(x_index),
            "x_value": float(x_value),
            "null_id": null_id,
            "seed": int(seed) if seed is not None else None,
            "draw_index": int(draw_index) if draw_index is not None else None,
            "feature_matrix_sha256": feature_hash,
            "graph_sha256": graph_sha256,
            "operator_sha256": operator_sha256,
            "compared_graph_sha256": compared_graph_sha256,
            "compared_operator_sha256": compared_operator_sha256,
            "source_binding_id": binding_id,
        }
        diagnostic_id = "diag_" + sha256_bytes(canonical_json_bytes(identity))[:24]
        record = {
            "diagnostic_id": diagnostic_id,
            **identity,
            "status": status,
            "value": value,
            "unit": unit,
            "note": note,
        }
        self.records.append(record)
        if status == "OK":
            self.plot_rows.append({
                "diagnostic_id": diagnostic_id,
                "scope_type": scope_type,
                "scope_value": effective_scope_value,
                "cell_id": effective_cell_id,
                "method_id": artifact.method_id,
                "method_version_id": artifact.method_version_id,
                "compared_method_id": (
                    compared_artifact.method_id if compared_artifact is not None else "not_applicable"
                ),
                "compared_method_version_id": (
                    compared_artifact.method_version_id if compared_artifact is not None else "not_applicable"
                ),
                "stage": stage,
                "panel_id": panel_id,
                "metric_id": metric_id,
                "series_id": series_id,
                "x_index": int(x_index),
                "x_value": float(x_value),
                "y_value": float(value),
                "null_id": null_id or "observed",
                "seed": int(seed) if seed is not None else -1,
                "draw_index": int(draw_index) if draw_index is not None else -1,
                "feature_matrix_sha256": feature_hash,
                "graph_sha256": graph_sha256 or "not_applicable",
                "operator_sha256": operator_sha256 or "not_applicable",
                "compared_graph_sha256": compared_graph_sha256 or "not_applicable",
                "compared_operator_sha256": compared_operator_sha256 or "not_applicable",
                "source_binding_id": binding_id,
            })
        return diagnostic_id


def _graph_from_artifact(artifact: VerifiedArtifact) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    _require(artifact.artifact_path is not None, f"{artifact.method_id}: method artifact is absent")
    graph = _csr_from_flat(artifact.arrays, "graph")
    laplacian = _csr_from_flat(artifact.arrays, "laplacian")
    graph = _validate_graph(graph)
    expected = symmetric_normalized_laplacian(graph)
    _require(
        _sparse_max_abs(laplacian - expected) <= 1e-9,
        f"{artifact.method_id}: stored Laplacian does not match the graph",
    )
    return graph, laplacian


def _health_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
    health: Mapping[str, Any],
) -> None:
    graph_hash = str(health["graph_sha256"])
    operator_hash = str(health["operator_sha256"])
    units = {
        "n_nodes": "nodes",
        "n_edges": "undirected_edges",
        "degree_min": "weighted_degree",
        "degree_mean": "weighted_degree",
        "degree_max": "weighted_degree",
        "degree_cv": "ratio",
        "n_components": "components",
        "isolated_nodes": "nodes",
        "normalized_spectral_gap": "normalized_laplacian_eigenvalue",
    }
    for index, metric in enumerate(units):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="graph_health",
            metric_id=metric,
            value=health[metric],
            unit=units[metric],
            graph_sha256=graph_hash,
            operator_sha256=operator_hash,
            x_index=index,
            x_value=float(index),
        )


def _fixed_graph_bootstrap_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
    graph: sparse.csr_matrix,
    laplacian: sparse.csr_matrix,
) -> None:
    graph_hash = sparse_sha256(graph)
    operator_hash = sparse_sha256(laplacian)
    try:
        rows = fixed_graph_group_bootstrap(
            graph=graph,
            group_ids=cell.group_ids,
            cell_id=cell.cell_id,
            method_id=artifact.method_id,
        )
    except GraphDiagnosticContractError as exc:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="fixed_graph_group_bootstrap_stability",
            metric_id="edge_support_jaccard",
            value=None,
            unit="similarity",
            graph_sha256=graph_hash,
            operator_sha256=operator_hash,
            status="NOT_AVAILABLE_GROUP_STRUCTURE",
            note=f"fixed fitted-graph bootstrap unavailable: {exc}",
        )
        return
    metric_units = {
        "retained_node_fraction": "fraction",
        "effective_row_mass": "weighted_rows",
        "edge_support_jaccard": "similarity",
        "weighted_graph_frobenius_cosine": "similarity",
        "normalized_laplacian_frobenius_cosine": "similarity",
        "normalized_laplacian_relative_difference": "relative_norm",
    }
    for row in rows:
        for metric_index, (metric, unit) in enumerate(metric_units.items()):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="fixed_graph_group_bootstrap_stability",
                metric_id=metric,
                value=row[metric],
                unit=unit,
                graph_sha256=row["graph_sha256"],
                operator_sha256=row["operator_sha256"],
                null_id=GRAPH_BOOTSTRAP_NULL_ID,
                seed=row["seed"],
                draw_index=row["draw_index"],
                series_id="fixed_fitted_graph_weight_sensitivity",
                x_index=int(row["draw_index"]) * len(metric_units) + metric_index,
                x_value=float(row["draw_index"]),
                note="source groups are resampled; the fitted graph is not refit and is weighted by sqrt row multiplicity",
            )


def _length_only_control_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
    graph: sparse.csr_matrix,
    laplacian: sparse.csr_matrix,
) -> None:
    from ..graph_topology import self_safe_knn_graph

    if cell.trace_length_coordinate is None:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="post_freeze",
            panel_id="length_only_graph_control",
            metric_id="error_label_roughness",
            value=None,
            unit="centered_normalized_laplacian_rayleigh_quotient",
            graph_sha256=sparse_sha256(graph),
            operator_sha256=sparse_sha256(laplacian),
            null_id=LENGTH_ONLY_CONTROL_ID,
            status="NOT_AVAILABLE_FEATURE_MISSING",
            note="trace_length is absent from the frozen feature contract; no length proxy was substituted",
        )
        return
    control_graph = _validate_graph(self_safe_knn_graph(
        cell.trace_length_coordinate[:, None],
        k=7,
        tie_keys=_row_id_tie_ranks(cell.row_ids),
    ))
    control_laplacian = symmetric_normalized_laplacian(control_graph)
    control_graph_hash = sparse_sha256(control_graph)
    control_operator_hash = sparse_sha256(control_laplacian)
    similarity = graph_operator_similarity(
        graph, laplacian, control_graph, control_laplacian
    )
    values = (
        ("error_label_roughness", normalized_roughness(cell.y_error, control_laplacian), "post_freeze", "rayleigh_quotient"),
        ("trace_length_coordinate_roughness", normalized_roughness(cell.trace_length_coordinate, control_laplacian), "post_freeze", "rayleigh_quotient"),
        ("edge_support_jaccard_vs_fitted", similarity["edge_support_jaccard"], "target_free", "similarity"),
        ("operator_cosine_vs_fitted", similarity["normalized_laplacian_frobenius_cosine"], "target_free", "similarity"),
    )
    for index, (metric, value, stage, unit) in enumerate(values):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage=stage,
            panel_id="length_only_graph_control",
            metric_id=metric,
            value=value,
            unit=unit,
            graph_sha256=control_graph_hash,
            operator_sha256=control_operator_hash,
            null_id=LENGTH_ONLY_CONTROL_ID,
            series_id="trace_length_only",
            x_index=index,
            x_value=float(index),
            note="graph built only from the frozen transformed confidence-oriented trace_length coordinate",
        )


def _random_family_control_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
    graph: sparse.csr_matrix,
    laplacian: sparse.csr_matrix,
    family_graphs: Mapping[str, sparse.csr_matrix],
) -> None:
    families = tuple(sorted(family_graphs, key=lambda value: value.encode("utf-8")))
    _require(bool(families), f"{cell.cell_id}: random-family roster is empty")
    for draw_index in range(GRAPH_BOOTSTRAP_COUNT):
        seed = deterministic_diagnostic_seed(
            RANDOM_FAMILY_NULL_ID, cell.cell_id, artifact.method_id, draw_index
        )
        family_index = int(np.random.Generator(np.random.PCG64(seed)).integers(0, len(families)))
        family = families[family_index]
        control_graph = family_graphs[family]
        control_laplacian = symmetric_normalized_laplacian(control_graph)
        similarity = graph_operator_similarity(
            graph, laplacian, control_graph, control_laplacian
        )
        values: list[tuple[str, float, str]] = [
            ("error_label_roughness", normalized_roughness(cell.y_error, control_laplacian), "post_freeze"),
            ("edge_support_jaccard_vs_fitted", similarity["edge_support_jaccard"], "target_free"),
            ("operator_cosine_vs_fitted", similarity["normalized_laplacian_frobenius_cosine"], "target_free"),
        ]
        if cell.trace_length_coordinate is not None:
            values.insert(1, (
                "trace_length_coordinate_roughness",
                normalized_roughness(cell.trace_length_coordinate, control_laplacian),
                "post_freeze",
            ))
        for metric_index, (metric, value, stage) in enumerate(values):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage=stage,
                panel_id="random_family_graph_control",
                metric_id=metric,
                value=value,
                unit="similarity" if "fitted" in metric else "centered_normalized_laplacian_rayleigh_quotient",
                graph_sha256=sparse_sha256(control_graph),
                operator_sha256=sparse_sha256(control_laplacian),
                null_id=RANDOM_FAMILY_NULL_ID,
                seed=seed,
                draw_index=draw_index,
                series_id=family,
                x_index=draw_index * len(values) + metric_index,
                x_value=float(draw_index),
                note=f"target-blind draw selected provenance family {family}",
            )
    if cell.trace_length_coordinate is None:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="post_freeze",
            panel_id="random_family_graph_control",
            metric_id="trace_length_coordinate_roughness",
            value=None,
            unit="centered_normalized_laplacian_rayleigh_quotient",
            graph_sha256=sparse_sha256(graph),
            operator_sha256=sparse_sha256(laplacian),
            null_id=RANDOM_FAMILY_NULL_ID,
            series_id="trace_length_coordinate",
            status="NOT_AVAILABLE_FEATURE_MISSING",
            note="trace_length is absent; no nuisance random-family distribution was computed and no proxy was substituted",
        )


def _roughness_and_null_records(
    builder: _RecordBuilder,
    verified: VerifiedDiagnosticRelease,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
    graph: sparse.csr_matrix,
    laplacian: sparse.csr_matrix,
) -> float:
    graph_hash = sparse_sha256(graph)
    operator_hash = sparse_sha256(laplacian)
    target = normalized_roughness(cell.y_error, laplacian)
    nuisance = (
        normalized_roughness(cell.trace_length_coordinate, laplacian)
        if cell.trace_length_coordinate is not None else None
    )
    trace_note = (
        "trace_length is the frozen mixed-v2 transformed, confidence-oriented "
        "and population-standardized coordinate; it is a nuisance proxy, not raw tokens"
    )
    for index, (metric, value, note, status) in enumerate((
        ("error_label_roughness", target, "lower means graph neighbours have more similar correctness labels", "OK"),
        (
            "trace_length_coordinate_roughness",
            nuisance,
            trace_note if nuisance is not None else "trace_length is absent from this cell's frozen feature contract; no proxy was substituted",
            "OK" if nuisance is not None else "NOT_AVAILABLE_FEATURE_MISSING",
        ),
    )):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="post_freeze",
            panel_id="target_vs_nuisance_roughness",
            metric_id=metric,
            value=value,
            unit="centered_normalized_laplacian_rayleigh_quotient",
            graph_sha256=graph_hash,
            operator_sha256=operator_hash,
            x_index=index,
            x_value=float(index),
            status=status,
            note=note,
        )

    nulls = node_permutation_nulls(
        graph=graph,
        laplacian=laplacian,
        target=cell.y_error,
        nuisance=cell.trace_length_coordinate,
        cell_id=cell.cell_id,
        method_id=artifact.method_id,
    )
    target_null = np.asarray([row["target_roughness"] for row in nulls], dtype=np.float64)
    nuisance_null = (
        np.asarray([row["nuisance_roughness"] for row in nulls], dtype=np.float64)
        if nuisance is not None else None
    )
    for row in nulls:
        null_series = [
            ("error_label", "error_label_roughness", row["target_roughness"]),
        ]
        if nuisance is not None:
            null_series.append((
                "trace_length_coordinate",
                "trace_length_coordinate_roughness",
                row["nuisance_roughness"],
            ))
        for series, metric, value in null_series:
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="post_freeze",
                panel_id="node_permutation_null",
                metric_id=metric,
                value=value,
                unit="centered_normalized_laplacian_rayleigh_quotient",
                graph_sha256=row["graph_sha256"],
                operator_sha256=row["operator_sha256"],
                null_id=NODE_PERMUTATION_NULL_ID,
                seed=row["seed"],
                draw_index=row["draw_index"],
                series_id=series,
                x_index=row["draw_index"],
                x_value=float(row["draw_index"]),
            )
    if nuisance is None:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="post_freeze",
            panel_id="node_permutation_null",
            metric_id="trace_length_coordinate_roughness",
            value=None,
            unit="centered_normalized_laplacian_rayleigh_quotient",
            graph_sha256=graph_hash,
            operator_sha256=operator_hash,
            null_id=NODE_PERMUTATION_NULL_ID,
            series_id="trace_length_coordinate",
            status="NOT_AVAILABLE_FEATURE_MISSING",
            note="trace_length is absent; no nuisance permutation distribution was computed and no proxy was substituted",
        )
    target_median = float(np.median(target_null))
    nuisance_median = float(np.median(nuisance_null)) if nuisance_null is not None else None
    summaries = (
        ("error_alignment_null_minus_real", target_median - target, "error_label", "OK"),
        (
            "error_roughness_ratio_to_null_median",
            target / target_median if target_median > _EPS else None,
            "error_label",
            "OK" if target_median > _EPS else "METRIC_UNDEFINED_ZERO_NULL_MEDIAN",
        ),
        (
            "nuisance_alignment_null_minus_real",
            nuisance_median - nuisance if nuisance_median is not None and nuisance is not None else None,
            "trace_length_coordinate",
            "OK" if nuisance_median is not None and nuisance is not None else "NOT_AVAILABLE_FEATURE_MISSING",
        ),
        (
            "nuisance_roughness_ratio_to_null_median",
            nuisance / nuisance_median if nuisance is not None and nuisance_median is not None and nuisance_median > _EPS else None,
            "trace_length_coordinate",
            (
                "NOT_AVAILABLE_FEATURE_MISSING" if nuisance_median is None
                else "OK" if nuisance_median > _EPS
                else "METRIC_UNDEFINED_ZERO_NULL_MEDIAN"
            ),
        ),
    )
    for index, (metric, value, series, status) in enumerate(summaries):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="post_freeze",
            panel_id="roughness_null_summary",
            metric_id=metric,
            value=value,
            unit="roughness_difference" if "minus" in metric else "ratio",
            graph_sha256=graph_hash,
            operator_sha256=operator_hash,
            null_id=NODE_PERMUTATION_NULL_ID,
            series_id=series,
            x_index=index,
            x_value=float(index),
            status=status,
            note="positive null-minus-real means more graph alignment than the node-relabeling null median",
        )

    delta = verified.auroc_delta_vs_iu[(cell.cell_id, artifact.method_id)]
    builder.add(
        cell=cell,
        artifact=artifact,
        stage="post_freeze",
        panel_id="alignment_vs_improvement",
        metric_id="published_cell_auroc_delta_vs_iu_pcr",
        value=delta,
        unit="AUROC_delta_copied_from_evaluation",
        graph_sha256=graph_hash,
        operator_sha256=operator_hash,
        series_id="cell_relation",
        x_index=0,
        x_value=target_median - target,
        note="copied from the frozen evaluator; no performance metric is recomputed here",
    )
    return target_median - target


def _dufs_stability_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
    graph_hash: str,
    operator_hash: str,
) -> None:
    from ..graph_topology import self_safe_knn_graph

    key = "gate_probabilities_per_seed"
    if key not in artifact.arrays or "gates" not in artifact.arrays:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="dufs_gate_stability",
            metric_id="gate_seed_spearman",
            value=None,
            unit="correlation",
            graph_sha256=graph_hash,
            operator_sha256=operator_hash,
            status="NOT_AVAILABLE_ARTIFACT_MISSING",
            note="gate or per-seed gate artifact is missing; no substitute stability estimate was generated",
        )
        return
    per_seed = np.asarray(artifact.arrays[key], dtype=np.float64)
    seeds = tuple(int(value) for value in artifact.config.get("seeds", ()))
    _require(per_seed.shape[0] == len(seeds), f"{cell.cell_id}/dufs_liu: seed count drift")
    gates = np.asarray(artifact.arrays.get("gates"), dtype=np.float64)
    _require(gates.shape == (len(cell.feature_names),), f"{cell.cell_id}/dufs_liu: gate shape drift")
    for feature_index, feature_name in enumerate(cell.feature_names):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="dufs_gate_weights",
            metric_id="rms_normalized_gate_weight",
            value=gates[feature_index],
            unit="relative_gate_weight",
            graph_sha256=graph_hash,
            operator_sha256=operator_hash,
            series_id=feature_name,
            x_index=feature_index,
            x_value=float(feature_index),
        )
        for seed_index, seed in enumerate(seeds):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="dufs_gate_weights_per_seed",
                metric_id="gate_survival_probability",
                value=per_seed[seed_index, feature_index],
                unit="probability",
                graph_sha256=graph_hash,
                operator_sha256=operator_hash,
                seed=seed,
                series_id=feature_name,
                x_index=feature_index,
                x_value=float(feature_index),
            )
    for index, row in enumerate(_pairwise_stability(per_seed, seeds)):
        for metric in ("spearman", "cosine"):
            available = row[metric] is not None
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="dufs_gate_stability",
                metric_id=f"gate_seed_{metric}",
                value=row[metric] if available else None,
                unit="correlation",
                graph_sha256=graph_hash,
                operator_sha256=operator_hash,
                series_id=f"{row['left_seed']}_vs_{row['right_seed']}",
                x_index=index,
                x_value=float(index),
                status="OK" if available else "METRIC_UNDEFINED_CONSTANT_VECTOR",
                note=(
                    None if available else
                    "rank correlation is undefined because at least one seed gate vector is constant"
                ),
            )
    seed_graphs = []
    for seed_index in range(len(seeds)):
        seed_gate = per_seed[seed_index]
        rms = float(np.sqrt(np.mean(seed_gate ** 2)))
        _require(rms > _EPS, f"{cell.cell_id}/dufs_liu: seed gate RMS is zero")
        seed_graphs.append(self_safe_knn_graph(
            cell.X_confidence * (seed_gate / rms)[None, :],
            k=int(artifact.config.get("graph_k", 7)),
            tie_keys=_row_id_tie_ranks(cell.row_ids),
        ))
    for index, row in enumerate(_graph_pair_stability(seed_graphs, seeds)):
        for metric in ("edge_jaccard", "weighted_frobenius_cosine"):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="dufs_seed_graph_stability",
                metric_id=metric,
                value=row[metric],
                unit="similarity",
                graph_sha256=graph_hash,
                operator_sha256=operator_hash,
                series_id=f"{row['left_seed']}_vs_{row['right_seed']}",
                x_index=index,
                x_value=float(index),
                note="per-seed graphs reconstructed from the frozen per-seed DUFS gate probabilities",
            )


def _ca_control_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
    graph: sparse.csr_matrix,
) -> None:
    from ..specrage_laplacian import weighted_multiview_graph

    arrays = artifact.arrays
    required = {"alpha", "alpha_per_seed", "view_names", "view_prior"}
    missing = sorted(required - set(arrays))
    if missing:
        for panel_id, metric_id in (
            ("ca_view_weights", "mean_learned_alpha"),
            ("ca_alpha_stability", "alpha_seed_spearman"),
            ("ca_seed_graph_stability", "edge_jaccard"),
            ("ca_alpha_controls", "error_label_roughness"),
        ):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="post_freeze" if panel_id == "ca_alpha_controls" else "target_free",
                panel_id=panel_id,
                metric_id=metric_id,
                value=None,
                unit="unavailable",
                graph_sha256=sparse_sha256(graph),
                operator_sha256=sparse_sha256(symmetric_normalized_laplacian(graph)),
                status="NOT_AVAILABLE_ARTIFACT_MISSING",
                note=f"missing registered CA artifacts {missing}; no control was fabricated",
            )
        return
    alpha = np.asarray(arrays["alpha"], dtype=np.float64)
    alpha_per_seed = np.asarray(arrays["alpha_per_seed"], dtype=np.float64)
    names = tuple(str(value) for value in arrays["view_names"].tolist())
    prior = np.asarray(arrays["view_prior"], dtype=np.float64)
    _require(alpha.shape == (len(cell.row_ids), len(names)), f"{cell.cell_id}/CA: alpha shape drift")
    _require(alpha_per_seed.shape[1:] == alpha.shape, f"{cell.cell_id}/CA: seed-alpha shape drift")
    _require(prior.shape == (len(names),) and np.all(prior > 0), f"{cell.cell_id}/CA: prior shape drift")
    fitted_graph_hash = sparse_sha256(graph)
    fitted_operator_hash = sparse_sha256(symmetric_normalized_laplacian(graph))
    for view_index, view_name in enumerate(names):
        for metric, value in (
            ("mean_learned_alpha", float(np.mean(alpha[:, view_index]))),
            ("frozen_view_prior", float(prior[view_index] / prior.sum())),
        ):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="ca_view_weights",
                metric_id=metric,
                value=value,
                unit="simplex_mass",
                graph_sha256=fitted_graph_hash,
                operator_sha256=fitted_operator_hash,
                series_id=view_name,
                x_index=view_index,
                x_value=float(view_index),
            )
    base_graphs = tuple(
        _csr_from_flat(arrays, f"base_graphs__{_safe_artifact_token(name)}")
        for name in names
    )
    mass_normalize = bool(artifact.config.get("view_mass_normalization"))
    rebuilt = weighted_multiview_graph(
        base_graphs,
        alpha,
        view_prior=prior,
        mass_normalize=mass_normalize,
    )
    _require(
        _sparse_max_abs(_canonical_sparse(rebuilt) - graph) <= 1e-9,
        f"{cell.cell_id}/CA: base graphs and alpha do not rebuild the fitted graph",
    )
    seed_ids = tuple(int(value) for value in artifact.config.get("model_seeds", ()))
    _require(len(seed_ids) == alpha_per_seed.shape[0], f"{cell.cell_id}/CA: model-seed count drift")
    for index, row in enumerate(_pairwise_stability(alpha_per_seed.reshape(len(seed_ids), -1), seed_ids)):
        for metric in ("spearman", "cosine"):
            available = row[metric] is not None
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="ca_alpha_stability",
                metric_id=f"alpha_seed_{metric}",
                value=row[metric] if available else None,
                unit="correlation",
                graph_sha256=fitted_graph_hash,
                operator_sha256=fitted_operator_hash,
                series_id=f"{row['left_seed']}_vs_{row['right_seed']}",
                x_index=index,
                x_value=float(index),
                status="OK" if available else "METRIC_UNDEFINED_CONSTANT_VECTOR",
                note=(
                    None if available else
                    "rank correlation is undefined because at least one seed alpha vector is constant"
                ),
            )
    seed_graphs = tuple(
        _csr_from_flat(arrays, f"seed_graphs__{seed}") for seed in seed_ids
    )
    for index, row in enumerate(_graph_pair_stability(seed_graphs, seed_ids)):
        for metric in ("edge_jaccard", "weighted_frobenius_cosine"):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="ca_seed_graph_stability",
                metric_id=metric,
                value=row[metric],
                unit="similarity",
                graph_sha256=fitted_graph_hash,
                operator_sha256=fitted_operator_hash,
                series_id=f"{row['left_seed']}_vs_{row['right_seed']}",
                x_index=index,
                x_value=float(index),
            )

    n, views = alpha.shape
    uniform_alpha = np.full((n, views), 1.0 / views, dtype=np.float64)
    prior_alpha = np.repeat((prior / prior.sum())[None, :], n, axis=0)
    perm_seed = deterministic_diagnostic_seed(
        CA_CONTROL_NULL_ID, cell.cell_id, artifact.method_id, 0
    )
    permutation = np.random.Generator(np.random.PCG64(perm_seed)).permutation(n)
    global_mean = np.mean(alpha, axis=0)
    global_mean /= global_mean.sum()
    global_mean_alpha = np.repeat(global_mean[None, :], n, axis=0)
    control_alphas = {
        "learned": (alpha, None),
        "equal_view": (uniform_alpha, None),
        "provenance_prior": (prior_alpha, None),
        "global_mean_alpha": (global_mean_alpha, None),
        "permuted": (alpha[permutation], perm_seed),
    }
    _require(tuple(control_alphas) == CA_CONTROL_SERIES, "CA control roster drift")
    for index, (control, (control_alpha, seed)) in enumerate(control_alphas.items()):
        control_graph = _canonical_sparse(weighted_multiview_graph(
            base_graphs,
            control_alpha,
            view_prior=prior,
            mass_normalize=mass_normalize,
        ))
        control_laplacian = symmetric_normalized_laplacian(control_graph)
        control_graph = _validate_graph(control_graph)
        control_graph_hash = sparse_sha256(control_graph)
        control_operator_hash = sparse_sha256(control_laplacian)
        control_signals: list[tuple[str, np.ndarray]] = [("error_label", cell.y_error)]
        if cell.trace_length_coordinate is not None:
            control_signals.append(("trace_length_coordinate", cell.trace_length_coordinate))
        for series, values in control_signals:
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="post_freeze",
                panel_id="ca_alpha_controls",
                metric_id=f"{series}_roughness",
                value=normalized_roughness(values, control_laplacian),
                unit="centered_normalized_laplacian_rayleigh_quotient",
                graph_sha256=control_graph_hash,
                operator_sha256=control_operator_hash,
                null_id=CA_CONTROL_NULL_ID if control != "learned" else None,
                seed=seed,
                series_id=control,
                x_index=index,
                x_value=float(index),
            )
    if cell.trace_length_coordinate is None:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="post_freeze",
            panel_id="ca_alpha_controls",
            metric_id="trace_length_coordinate_roughness",
            value=None,
            unit="centered_normalized_laplacian_rayleigh_quotient",
            graph_sha256=fitted_graph_hash,
            operator_sha256=fitted_operator_hash,
            null_id=CA_CONTROL_NULL_ID,
            series_id="trace_length_coordinate",
            status="NOT_AVAILABLE_FEATURE_MISSING",
            note="trace_length is absent; no nuisance alpha-control values were computed and no proxy was substituted",
        )


def _pgrd_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
    graph: sparse.csr_matrix,
    laplacian: sparse.csr_matrix,
) -> None:
    arrays = artifact.arrays
    builder.add(
        cell=cell,
        artifact=artifact,
        stage="target_free",
        panel_id="pgrd_seed_graph_stability",
        metric_id="edge_jaccard",
        value=None,
        unit="similarity",
        graph_sha256=sparse_sha256(graph),
        operator_sha256=sparse_sha256(laplacian),
        status="NOT_AVAILABLE_NO_SEED_GRAPH_ARTIFACTS",
        note="PGRD-A has one deterministic graph and emitted no per-seed graphs; no pseudo-seeds were invented",
    )
    required = {"residuals", "baseline_standardized", "A0", "c0"}
    missing = sorted(required - set(arrays))
    if missing:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="pgrd_cross_gradient",
            metric_id="cross_gradient_norm",
            value=None,
            unit="l2_norm",
            graph_sha256=sparse_sha256(graph),
            operator_sha256=sparse_sha256(laplacian),
            status="NOT_AVAILABLE_FIT_FALLBACK",
            note=f"missing registered PGRD artifacts: {missing}; no substitute was computed",
        )
        return
    R = np.asarray(arrays["residuals"], dtype=np.float64)
    baseline = np.asarray(arrays["baseline_standardized"], dtype=np.float64)
    A0_stored = np.asarray(arrays["A0"], dtype=np.float64)
    c0_stored = np.asarray(arrays["c0"], dtype=np.float64)
    _require(R.ndim == 2 and R.shape[0] == len(cell.row_ids), f"{cell.cell_id}/PGRD: residual shape drift")
    _require(baseline.shape == (len(cell.row_ids),), f"{cell.cell_id}/PGRD: baseline shape drift")
    n = len(baseline)
    A0 = np.asarray(R.T @ (laplacian @ R) / n, dtype=np.float64)
    A0 = 0.5 * (A0 + A0.T)
    c0 = np.asarray(R.T @ (laplacian @ baseline) / n, dtype=np.float64)
    _require(np.allclose(A0, A0_stored, atol=1e-10, rtol=1e-9), f"{cell.cell_id}/PGRD: A0 artifact drift")
    _require(np.allclose(c0, c0_stored, atol=1e-10, rtol=1e-9), f"{cell.cell_id}/PGRD: c0 artifact drift")
    trace = float(np.trace(A0))
    _require(math.isfinite(trace) and trace > _EPS, f"{cell.cell_id}/PGRD: nonpositive A0 trace")
    trace_scale = float(R.shape[1] / trace)
    c = trace_scale * c0
    direction = -c
    if "trace_scale" in arrays:
        _require(
            np.allclose(float(np.asarray(arrays["trace_scale"])), trace_scale, atol=1e-10, rtol=1e-9),
            f"{cell.cell_id}/PGRD: trace-scale artifact drift",
        )
    if "c" in arrays:
        _require(np.allclose(np.asarray(arrays["c"], dtype=float), c, atol=1e-10, rtol=1e-9), f"{cell.cell_id}/PGRD: scaled cross-gradient artifact drift")
    if "direction" in arrays:
        _require(np.allclose(np.asarray(arrays["direction"], dtype=float), direction, atol=1e-10, rtol=1e-9), f"{cell.cell_id}/PGRD: direction artifact drift")
    derivative = float(2.0 * np.dot(c0, direction))
    baseline_energy = float(baseline @ (laplacian @ baseline) / n)
    cross_term = float(2.0 * np.dot(c0, direction))
    quadratic_term = float(direction @ A0 @ direction)
    metrics = {
        "full_quadratic_trace": trace,
        "full_quadratic_frobenius_norm": float(np.linalg.norm(A0)),
        "cross_gradient_norm": float(np.linalg.norm(c0)),
        "trace_scaled_cross_gradient_norm": float(np.linalg.norm(c)),
        "directional_derivative_at_zero": derivative,
        "baseline_graph_energy": baseline_energy,
        "cross_term_at_registered_direction": cross_term,
        "quadratic_term_at_registered_direction": quadratic_term,
        "predicted_energy_change_at_unit_step": cross_term + quadratic_term,
    }
    for index, (metric, value) in enumerate(metrics.items()):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="pgrd_cross_gradient",
            metric_id=metric,
            value=value,
            unit="quadratic_energy" if "energy" in metric or "term" in metric or "derivative" in metric else "matrix_norm",
            graph_sha256=sparse_sha256(graph),
            operator_sha256=sparse_sha256(laplacian),
            x_index=index,
            x_value=float(index),
            note="cross-only and full-quadratic terms are shown separately; the fitted rule uses only the cross gradient",
        )

    # PGRD-specific node-relabeling cross-gradient control.  This complements
    # the generic target/nuisance roughness null and never refits a score.
    for draw_index in range(NODE_PERMUTATION_COUNT):
        seed = deterministic_null_seed(cell.cell_id, artifact.method_id, draw_index)
        permutation = np.random.Generator(np.random.PCG64(seed)).permutation(n)
        perm_graph = _canonical_sparse(graph[permutation][:, permutation])
        perm_laplacian = _canonical_sparse(laplacian[permutation][:, permutation])
        c0_null = np.asarray(R.T @ (perm_laplacian @ baseline) / n, dtype=np.float64)
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="pgrd_cross_gradient_null",
            metric_id="cross_gradient_norm",
            value=float(np.linalg.norm(c0_null)),
            unit="l2_norm",
            graph_sha256=sparse_sha256(perm_graph),
            operator_sha256=sparse_sha256(perm_laplacian),
            null_id=NODE_PERMUTATION_NULL_ID,
            seed=seed,
            draw_index=draw_index,
            series_id="node_permuted_graph",
            x_index=draw_index,
            x_value=float(draw_index),
        )


def _family_nrm_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
) -> None:
    arrays = artifact.arrays
    required = {"residual_covariance", "residuals"}
    missing = sorted(required - set(arrays))
    if missing:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="family_nrm_residual_structure",
            metric_id="residual_eigenvalue",
            value=None,
            unit="covariance_eigenvalue",
            status="NOT_AVAILABLE_FIT_FALLBACK",
            note=f"missing registered Family-NRM artifacts: {missing}; no substitute was computed",
        )
        return
    covariance = np.asarray(arrays["residual_covariance"], dtype=np.float64)
    residuals = np.asarray(arrays["residuals"], dtype=np.float64)
    _require(covariance.ndim == 2 and covariance.shape[0] == covariance.shape[1], f"{cell.cell_id}/Family-NRM: covariance shape drift")
    _require(residuals.shape == (len(cell.row_ids), covariance.shape[0]), f"{cell.cell_id}/Family-NRM: residual shape drift")
    observed = residuals.T @ residuals / len(residuals)
    observed = 0.5 * (observed + observed.T)
    _require(np.allclose(observed, covariance, atol=1e-10, rtol=1e-9), f"{cell.cell_id}/Family-NRM: covariance artifact drift")
    eigenvalues = np.linalg.eigvalsh(covariance)
    family_names = tuple(str(value) for value in artifact.record.get("diagnostics", {}).get("present_families", ()))
    if len(family_names) != residuals.shape[1]:
        family_names = tuple(f"family_{index}" for index in range(residuals.shape[1]))
    for index, value in enumerate(eigenvalues):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="family_nrm_residual_eigenspectrum",
            metric_id="residual_eigenvalue",
            value=value,
            unit="covariance_eigenvalue",
            series_id="eigenspectrum",
            x_index=index,
            x_value=float(index),
        )
    for row_index, row_family in enumerate(family_names):
        for column_index, column_family in enumerate(family_names):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="family_nrm_residual_covariance",
                metric_id="residual_covariance",
                value=covariance[row_index, column_index],
                unit="covariance",
                series_id=row_family,
                x_index=column_index,
                x_value=float(column_index),
                note=f"column_family={column_family}",
            )
    direction = np.asarray(arrays.get("residual_direction", np.zeros(residuals.shape[1])), dtype=np.float64)
    if "residual_direction" not in arrays:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="family_nrm_family_contributions",
            metric_id="direction_coefficient",
            value=None,
            unit="coefficient",
            status="NOT_AVAILABLE_FIT_FALLBACK",
            note="fitted residual direction is absent; no direction was reconstructed",
        )
        return
    _require(direction.shape == (residuals.shape[1],), f"{cell.cell_id}/Family-NRM: direction shape drift")
    contribution_variance = np.var(residuals, axis=0)
    absolute_share = np.abs(direction) / (np.sum(np.abs(direction)) + _EPS)
    family_contributions = arrays.get("family_contributions")
    usable_indices = arrays.get("usable_family_indices")
    usable_contributions: np.ndarray | None = None
    if family_contributions is not None and usable_indices is not None:
        all_contributions = np.asarray(family_contributions, dtype=np.float64)
        indices = np.asarray(usable_indices, dtype=np.int64)
        _require(all_contributions.ndim == 2 and all_contributions.shape[0] == len(cell.row_ids), f"{cell.cell_id}/Family-NRM: family-contribution shape drift")
        _require(indices.shape == (residuals.shape[1],), f"{cell.cell_id}/Family-NRM: usable-family index drift")
        _require(np.all((indices >= 0) & (indices < all_contributions.shape[1])), f"{cell.cell_id}/Family-NRM: usable-family index out of range")
        usable_contributions = all_contributions[:, indices]
    for index, family in enumerate(family_names):
        metrics = [
            ("direction_coefficient", direction[index], "coefficient"),
            ("absolute_direction_share", absolute_share[index], "fraction"),
            ("residual_variance", contribution_variance[index], "variance"),
        ]
        if usable_contributions is not None:
            metrics.extend([
                ("iu_family_contribution_variance", float(np.var(usable_contributions[:, index])), "variance"),
                ("iu_family_contribution_mean_absolute", float(np.mean(np.abs(usable_contributions[:, index]))), "absolute_standardized_contribution"),
            ])
        for metric, value, unit in metrics:
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="family_nrm_family_contributions",
                metric_id=metric,
                value=value,
                unit=unit,
                series_id=family,
                x_index=index,
                x_value=float(index),
            )


def _continuous_lsml_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
) -> None:
    """Expose the frozen L-SML feature correlation and cluster boundaries."""

    diagnostics = artifact.record.get("diagnostics", {})
    lsml = diagnostics.get("lsml") if isinstance(diagnostics, dict) else None
    if not isinstance(lsml, dict) or "c" not in lsml:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="continuous_lsml_correlation_clusters",
            metric_id="feature_correlation",
            value=None,
            unit="pearson_correlation",
            status="NOT_AVAILABLE_ARTIFACT_MISSING",
            note="the frozen score record lacks the L-SML cluster assignment; no clustering was rerun",
        )
        return
    selected = tuple(str(value) for value in artifact.record.get("selected_features", ()))
    assignment = np.asarray(lsml["c"], dtype=np.int64)
    _require(selected == cell.feature_names, f"{cell.cell_id}/Continuous-LSML: expected full-pool feature order")
    _require(assignment.shape == (len(selected),), f"{cell.cell_id}/Continuous-LSML: cluster assignment shape drift")
    _require(np.all(assignment >= 0), f"{cell.cell_id}/Continuous-LSML: negative cluster ID")
    correlation = np.corrcoef(cell.X_confidence, rowvar=False)
    _require(correlation.shape == (len(selected), len(selected)), f"{cell.cell_id}/Continuous-LSML: correlation shape drift")
    _require(np.isfinite(correlation).all(), f"{cell.cell_id}/Continuous-LSML: correlation is non-finite")
    order = np.lexsort((np.arange(len(selected)), assignment))
    ordered_position = np.empty(len(selected), dtype=np.int64)
    ordered_position[order] = np.arange(len(selected), dtype=np.int64)
    for feature_index, feature_name in enumerate(selected):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="continuous_lsml_cluster_boundaries",
            metric_id="cluster_id",
            value=int(assignment[feature_index]),
            unit="cluster_index",
            series_id=feature_name,
            x_index=int(ordered_position[feature_index]),
            x_value=float(ordered_position[feature_index]),
            note="x position is ordered by frozen cluster ID then canonical feature order",
        )
    for row_feature_index in order:
        row_feature = selected[int(row_feature_index)]
        for column_feature_index in order:
            column_feature = selected[int(column_feature_index)]
            same_cluster = assignment[row_feature_index] == assignment[column_feature_index]
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id="continuous_lsml_correlation_clusters",
                metric_id="feature_correlation",
                value=correlation[row_feature_index, column_feature_index],
                unit="pearson_correlation",
                series_id=row_feature,
                x_index=int(ordered_position[column_feature_index]),
                x_value=float(ordered_position[column_feature_index]),
                note=(
                    f"column_feature={column_feature};same_cluster={str(bool(same_cluster)).lower()};"
                    f"row_cluster={int(assignment[row_feature_index])};"
                    f"column_cluster={int(assignment[column_feature_index])}"
                ),
            )


def _su_upper_offdiag_relative_residual(
    covariance: np.ndarray,
    low_rank: np.ndarray,
    sparse_part: np.ndarray,
) -> float:
    covariance = np.asarray(covariance, dtype=np.float64)
    low_rank = np.asarray(low_rank, dtype=np.float64)
    sparse_part = np.asarray(sparse_part, dtype=np.float64)
    _require(
        covariance.ndim == 2
        and covariance.shape[0] == covariance.shape[1]
        and low_rank.shape == sparse_part.shape == covariance.shape,
        "SU residual matrices disagree",
    )
    p = covariance.shape[0]
    upper = np.triu_indices(p, 1)
    observed_off_diagonal = covariance.copy()
    np.fill_diagonal(observed_off_diagonal, 0.0)
    residual_off_diagonal = observed_off_diagonal - low_rank - sparse_part
    np.fill_diagonal(residual_off_diagonal, 0.0)
    return float(
        np.linalg.norm(residual_off_diagonal[upper])
        / (np.linalg.norm(observed_off_diagonal[upper]) + _EPS)
    )


def _su_pcr_records(
    builder: _RecordBuilder,
    cell: VerifiedDiagnosticCell,
    artifact: VerifiedArtifact,
) -> None:
    arrays = artifact.arrays
    required = {"low_rank", "sparse", "sparse_support"}
    missing = sorted(required - set(arrays))
    if missing:
        for panel_id, metric_id in (
            ("su_pcr_decomposition", "decomposition_relative_residual_upper_off_diagonal_recomputed"),
            ("su_pcr_low_rank_eigenspectrum", "low_rank_eigenvalue"),
            ("su_pcr_sparse_support", "supported_sparse_coefficient"),
            ("su_pcr_sparse_support_stability", "support_jaccard_vs_frozen"),
        ):
            builder.add(
                cell=cell,
                artifact=artifact,
                stage="target_free",
                panel_id=panel_id,
                metric_id=metric_id,
                value=None,
                unit="unavailable",
                status="NOT_AVAILABLE_ARTIFACT_MISSING",
                note=f"missing registered SU-PCR artifacts {missing}; no decomposition was rerun as a substitute",
            )
        return
    low_rank = np.asarray(arrays["low_rank"], dtype=np.float64)
    sparse_part = np.asarray(arrays["sparse"], dtype=np.float64)
    support = np.asarray(arrays["sparse_support"], dtype=bool)
    p = cell.X_confidence.shape[1]
    _require(low_rank.shape == sparse_part.shape == support.shape == (p, p), f"{cell.cell_id}/SU-PCR: decomposition shape drift")
    covariance = cell.X_confidence.T @ cell.X_confidence / len(cell.row_ids)
    upper = np.triu_indices(p, 1)
    relative_residual = _su_upper_offdiag_relative_residual(
        covariance, low_rank, sparse_part
    )
    recorded = artifact.record.get("diagnostics", {}).get(
        "decomposition_relative_residual"
    )
    _require(
        isinstance(recorded, (int, float))
        and math.isfinite(float(recorded))
        and np.isclose(relative_residual, float(recorded), atol=1e-12, rtol=1e-9),
        f"{cell.cell_id}/SU-PCR: recomputed upper-off-diagonal residual "
        "does not match the frozen fit diagnostic",
    )
    off_diagonal = ~np.eye(p, dtype=bool)
    metrics = {
        "decomposition_relative_residual_upper_off_diagonal_recomputed": relative_residual,
        "decomposition_relative_residual_recorded": float(recorded),
        "sparse_support_fraction_all": float(np.mean(support)),
        "sparse_support_fraction_off_diagonal": float(np.mean(support[off_diagonal])),
        "low_rank_symmetry_error": float(np.max(np.abs(low_rank - low_rank.T))),
        "sparse_symmetry_error": float(np.max(np.abs(sparse_part - sparse_part.T))),
        "low_rank_effective_rank": float(
            np.sum(np.linalg.svd(low_rank, compute_uv=False) > 1e-10)
        ),
    }
    for index, (metric, value) in enumerate(metrics.items()):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="su_pcr_decomposition",
            metric_id=metric,
            value=value,
            unit="ratio" if "fraction" in metric or "residual" in metric else "value",
            x_index=index,
            x_value=float(index),
        )
    eigenvalues = np.linalg.eigvalsh(0.5 * (low_rank + low_rank.T))
    for index, value in enumerate(eigenvalues):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="su_pcr_low_rank_eigenspectrum",
            metric_id="low_rank_eigenvalue",
            value=value,
            unit="covariance_eigenvalue",
            series_id="eigenspectrum",
            x_index=index,
            x_value=float(index),
        )
    support_rows, support_columns = np.nonzero(support)
    for index, (row_index, column_index) in enumerate(zip(support_rows, support_columns)):
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="su_pcr_sparse_support",
            metric_id="supported_sparse_coefficient",
            value=sparse_part[int(row_index), int(column_index)],
            unit="covariance",
            series_id=f"row_{int(row_index):02d}",
            x_index=int(column_index),
            x_value=float(column_index),
            note=f"support_entry_index={index}",
        )

    from ..dependency_fusion import projected_sparse_decomposition

    full_support = np.asarray(support[upper], dtype=bool)
    try:
        # Probe the sparse-support mechanism under the evaluator's source-group
        # resampling unit.  This is an explanatory refit of the decomposition
        # only; it never changes or republishes the frozen SU-PCR score.
        for draw_index in range(SU_SUPPORT_BOOTSTRAP_COUNT):
            multiplicity, seed = _group_bootstrap_row_multiplicity(
                cell.group_ids,
                null_id=SU_SUPPORT_BOOTSTRAP_ID,
                cell_id=cell.cell_id,
                method_id=artifact.method_id,
                draw_index=draw_index,
            )
            total_mass = float(np.sum(multiplicity))
            _require(total_mass > _EPS, f"{cell.cell_id}/SU-PCR: zero bootstrap row mass")
            weighted_covariance = (
                cell.X_confidence.T
                @ (cell.X_confidence * multiplicity[:, None])
                / total_mass
            )
            weighted_covariance = 0.5 * (weighted_covariance + weighted_covariance.T)
            decomp = projected_sparse_decomposition(
                weighted_covariance,
                rank=int(artifact.config.get("rank", 2)),
                threshold_multiplier=float(artifact.config.get("threshold_multiplier", 1.0)),
                max_iter=int(artifact.config.get("max_iter", 100)),
                inner_completion_iter=int(artifact.config.get("inner_completion_iter", 40)),
                tol=float(artifact.config.get("decomposition_tol", 1e-8)),
                max_sparse_fraction=artifact.config.get("max_sparse_fraction"),
            )
            bootstrap_support = np.asarray(decomp.support[upper], dtype=bool)
            intersection = int(np.sum(full_support & bootstrap_support))
            union = int(np.sum(full_support | bootstrap_support))
            jaccard = 1.0 if union == 0 else float(intersection / union)
            values = (
                ("support_jaccard_vs_frozen", jaccard, "similarity"),
                ("bootstrap_sparse_support_fraction_upper_off_diagonal", float(np.mean(bootstrap_support)), "fraction"),
                ("bootstrap_relative_residual_upper_off_diagonal", float(decomp.relative_residual), "ratio"),
                ("bootstrap_converged", float(bool(decomp.converged)), "boolean"),
            )
            for metric_index, (metric, value, unit) in enumerate(values):
                builder.add(
                    cell=cell,
                    artifact=artifact,
                    stage="target_free",
                    panel_id="su_pcr_sparse_support_stability",
                    metric_id=metric,
                    value=value,
                    unit=unit,
                    null_id=SU_SUPPORT_BOOTSTRAP_ID,
                    seed=seed,
                    draw_index=draw_index,
                    series_id="source_group_bootstrap",
                    x_index=draw_index * len(values) + metric_index,
                    x_value=float(draw_index),
                    note="SU decomposition refit on source-group bootstrap weights; frozen SU-PCR scores are unchanged",
                )
    except GraphDiagnosticContractError as exc:
        builder.add(
            cell=cell,
            artifact=artifact,
            stage="target_free",
            panel_id="su_pcr_sparse_support_stability",
            metric_id="support_jaccard_vs_frozen",
            value=None,
            unit="similarity",
            null_id=SU_SUPPORT_BOOTSTRAP_ID,
            status="NOT_AVAILABLE_GROUP_STRUCTURE",
            note=f"SU support bootstrap unavailable: {exc}",
        )


def _deterministic_diffusion_embedding(
    laplacian: sparse.spmatrix,
    row_ids: Sequence[str],
    *,
    diffusion_steps: int = 8,
) -> np.ndarray:
    """A display-only embedding without an eigenbasis ambiguity.

    Repeated Laplacian eigenvalues make eigenvectors non-unique across BLAS
    implementations.  Two fixed SHA-derived probes are instead diffused by
    ``I-L`` and deterministically Gram--Schmidt orthogonalized.  This keeps the
    example payload byte-stable without claiming that the axes are eigenvectors.
    """

    L = _canonical_sparse(laplacian)
    n = L.shape[0]
    _require(n >= 3 and len(row_ids) == n, "display embedding row mismatch")
    operator = _canonical_sparse(sparse.eye(n, format="csr") - L)
    probes = np.empty((n, 2), dtype=np.float64)
    scale = float(2**64 - 1)
    for row_index, row_id in enumerate(row_ids):
        for axis in range(2):
            digest = hashlib.sha256(
                f"{DIAGNOSTIC_VERSION}|display_probe|{axis}|{row_id}".encode("utf-8")
            ).digest()
            probes[row_index, axis] = 2.0 * (
                int.from_bytes(digest[:8], "big", signed=False) / scale
            ) - 1.0
    coordinates = probes
    for _ in range(int(diffusion_steps)):
        coordinates = np.asarray(operator @ coordinates, dtype=np.float64)
        coordinates -= np.mean(coordinates, axis=0, keepdims=True)
    basis = np.empty_like(coordinates)
    for column in range(2):
        vector = coordinates[:, column].copy()
        for previous in range(column):
            vector -= float(np.dot(basis[:, previous], vector)) * basis[:, previous]
        norm = float(np.linalg.norm(vector))
        if norm <= _EPS:
            fallback = np.arange(n, dtype=np.float64) ** (column + 1)
            fallback -= np.mean(fallback)
            for previous in range(column):
                fallback -= float(np.dot(basis[:, previous], fallback)) * basis[:, previous]
            vector = fallback
            norm = float(np.linalg.norm(vector))
        _require(norm > _EPS, "display embedding probe collapsed")
        basis[:, column] = vector / norm
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1.0
    _require(np.isfinite(basis).all(), "display diffusion embedding is non-finite")
    return basis


def _text_array(values: Sequence[str]) -> np.ndarray:
    values = [str(value) for value in values]
    width = max([1] + [len(value) for value in values])
    return np.asarray(values, dtype=f"<U{width}")


def _plot_arrays(rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    ordered = sorted(rows, key=lambda row: str(row["diagnostic_id"]).encode("utf-8"))
    text_fields = (
        "diagnostic_id", "scope_type", "scope_value", "cell_id",
        "method_id", "method_version_id", "stage",
        "compared_method_id", "compared_method_version_id",
        "panel_id", "metric_id", "series_id", "null_id",
        "feature_matrix_sha256", "graph_sha256", "operator_sha256",
        "compared_graph_sha256", "compared_operator_sha256", "source_binding_id",
    )
    arrays: dict[str, np.ndarray] = {
        field: _text_array([str(row[field]) for row in ordered])
        for field in text_fields
    }
    for field in ("x_index", "seed", "draw_index"):
        arrays[field] = np.asarray([int(row[field]) for row in ordered], dtype="<i8")
    for field in ("x_value", "y_value"):
        arrays[field] = np.asarray([float(row[field]) for row in ordered], dtype="<f8")
    arrays["schema_version"] = _text_array(
        [PLOT_DATA_SCHEMA_VERSION] * len(ordered)
    )
    arrays["diagnostic_version"] = _text_array(
        [DIAGNOSTIC_VERSION] * len(ordered)
    )
    return arrays


def _complete_required_panel_coverage(
    builder: _RecordBuilder,
    verified: VerifiedDiagnosticRelease,
) -> dict[str, Any]:
    """Guarantee one explicit record for every preregistered panel/cell/method."""

    present = {
        (str(row["cell_id"]), str(row["method_id"]), str(row["panel_id"]))
        for row in builder.records
        if row.get("scope_type") == "cell"
    }
    added = 0
    for cell_id, cell in verified.cells.items():
        for method_id, panels in REQUIRED_PANELS_BY_METHOD.items():
            artifact = cell.artifacts[method_id]
            for panel_id in panels:
                key = (cell_id, method_id, panel_id)
                if key in present:
                    continue
                stage = "post_freeze" if panel_id in {
                    "target_vs_nuisance_roughness",
                    "node_permutation_null",
                    "roughness_null_summary",
                    "alignment_vs_improvement",
                    "length_only_graph_control",
                    "random_family_graph_control",
                } else "target_free"
                builder.add(
                    cell=cell,
                    artifact=artifact,
                    stage=stage,
                    panel_id=panel_id,
                    metric_id="diagnostic_available",
                    value=None,
                    unit="boolean",
                    status="NOT_AVAILABLE_REQUESTED_ARTIFACT_MISSING",
                    note=(
                        artifact.fallback_reason
                        or "the frozen fitted artifact does not contain the members required for this preregistered panel; no substitute was reconstructed"
                    ),
                )
                present.add(key)
                added += 1
    expected = sum(
        len(panels) * len(verified.cells)
        for panels in REQUIRED_PANELS_BY_METHOD.values()
    )
    observed = sum(
        1
        for cell_id in verified.cells
        for method_id, panels in REQUIRED_PANELS_BY_METHOD.items()
        for panel_id in panels
        if (cell_id, method_id, panel_id) in present
    )
    _require(observed == expected, "required diagnostic panel coverage is incomplete")
    return {
        "coverage_axis": "cell_x_method_x_preregistered_panel",
        "expected_panel_slots": int(expected),
        "observed_panel_slots": int(observed),
        "explicit_unavailable_slots_added": int(added),
        "complete": True,
        "required_panels_by_method": {
            method_id: list(panels)
            for method_id, panels in REQUIRED_PANELS_BY_METHOD.items()
        },
    }


def build_graph_diagnostics(
    verified: VerifiedDiagnosticRelease,
    *,
    producer_snapshot: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build all diagnostics without writing files."""

    snapshot_payload = dict(producer_snapshot) if producer_snapshot is not None else None
    snapshot_sha = None
    if snapshot_payload is not None:
        snapshot_sha = snapshot_payload.get("snapshot_sha256")
        _require(
            isinstance(snapshot_sha, str) and _HEX64.fullmatch(snapshot_sha) is not None,
            "producer snapshot lacks a valid hash",
        )
        body = dict(snapshot_payload)
        body.pop("snapshot_sha256", None)
        _require(
            sha256_bytes(canonical_json_bytes(body)) == snapshot_sha,
            "producer snapshot payload hash mismatch",
        )
    builder = _RecordBuilder(verified, snapshot_sha)
    graph_health_by_method: dict[str, dict[str, dict[str, Any]]] = {
        method_id: {} for method_id in GRAPH_METHOD_IDS
    }
    graph_material: dict[tuple[str, str], tuple[sparse.csr_matrix, sparse.csr_matrix]] = {}
    family_graphs_by_cell = {
        cell_id: _provenance_family_graphs(cell)
        for cell_id, cell in verified.cells.items()
    }
    alignment: dict[str, list[tuple[str, float, float]]] = {
        method_id: [] for method_id in GRAPH_METHOD_IDS
    }

    # First pass is target-free and determines every example cell before any
    # correctness-colored example payload is assembled.
    for cell_id, cell in verified.cells.items():
        for method_id in GRAPH_METHOD_IDS:
            artifact = cell.artifacts[method_id]
            graph_available = (
                artifact.artifact_path is not None
                and _has_complete_csr(artifact.arrays, "graph")
                and _has_complete_csr(artifact.arrays, "laplacian")
            )
            if not graph_available:
                builder.add(
                    cell=cell,
                    artifact=artifact,
                    stage="target_free",
                    panel_id="graph_health",
                    metric_id="graph_available",
                    value=None,
                    unit="boolean",
                    status="NOT_AVAILABLE_FIT_FALLBACK",
                    note=(
                        artifact.fallback_reason
                        or "required graph/laplacian members are absent; no substitute graph was built"
                    ),
                )
                continue
            try:
                graph, laplacian = _graph_from_artifact(artifact)
            except GraphDiagnosticContractError:
                # An inconsistent emitted graph is a provenance failure, not a
                # per-cell missing diagnostic.  Propagate and block publication.
                raise
            health = graph_health(graph, laplacian)
            _health_records(builder, cell, artifact, health)
            graph_health_by_method[method_id][cell_id] = health
            graph_material[(cell_id, method_id)] = (graph, laplacian)

    selected_examples = {
        method_id: (
            select_example_cell(
                graph_health_by_method[method_id],
                {
                    cell_id: verified.cells[cell_id].trace_length_coordinate is not None
                    for cell_id in graph_health_by_method[method_id]
                },
            )
            if graph_health_by_method[method_id] else None
        )
        for method_id in GRAPH_METHOD_IDS
    }

    # Pairwise operator checks answer whether the three graph methods are
    # actually inducing distinct neighbourhoods or nearly the same smoother.
    for cell_id, cell in verified.cells.items():
        for left_index, left_method in enumerate(GRAPH_METHOD_IDS):
            for right_method in GRAPH_METHOD_IDS[left_index + 1:]:
                left_material = graph_material.get((cell_id, left_method))
                right_material = graph_material.get((cell_id, right_method))
                if left_material is None or right_material is None:
                    builder.add(
                        cell=cell,
                        artifact=cell.artifacts[left_method],
                        compared_artifact=cell.artifacts[right_method],
                        stage="target_free",
                        panel_id="graph_operator_similarity",
                        metric_id="edge_support_jaccard",
                        value=None,
                        unit="similarity",
                        status="NOT_AVAILABLE_GRAPH_MISSING",
                        series_id=f"{left_method}_vs_{right_method}",
                        note="one or both fitted graph/laplacian artifacts are absent; no substitute graph was compared",
                    )
                    continue
                left_graph, left_operator = left_material
                right_graph, right_operator = right_material
                similarities = graph_operator_similarity(
                    left_graph, left_operator, right_graph, right_operator
                )
                left_artifact = cell.artifacts[left_method]
                right_artifact = cell.artifacts[right_method]
                for index, (metric, value) in enumerate(similarities.items()):
                    builder.add(
                        cell=cell,
                        artifact=left_artifact,
                        compared_artifact=right_artifact,
                        stage="target_free",
                        panel_id="graph_operator_similarity",
                        metric_id=metric,
                        value=value,
                        unit="similarity" if "difference" not in metric else "relative_norm",
                        graph_sha256=sparse_sha256(left_graph),
                        operator_sha256=sparse_sha256(left_operator),
                        compared_graph_sha256=sparse_sha256(right_graph),
                        compared_operator_sha256=sparse_sha256(right_operator),
                        series_id=f"{left_method}_vs_{right_method}",
                        x_index=index,
                        x_value=float(index),
                    )

    # Second pass may use the already published y_error snapshot, but never raw
    # labels.  It does not influence the example-cell selection above.
    for cell_id, cell in verified.cells.items():
        for method_id in GRAPH_METHOD_IDS:
            artifact = cell.artifacts[method_id]
            material = graph_material.get((cell_id, method_id))
            if material is None:
                continue
            graph, laplacian = material
            effect = _roughness_and_null_records(
                builder, verified, cell, artifact, graph, laplacian
            )
            alignment[method_id].append(
                (cell_id, effect, verified.auroc_delta_vs_iu[(cell_id, method_id)])
            )
            _fixed_graph_bootstrap_records(builder, cell, artifact, graph, laplacian)
            _length_only_control_records(builder, cell, artifact, graph, laplacian)
            _random_family_control_records(
                builder,
                cell,
                artifact,
                graph,
                laplacian,
                family_graphs_by_cell[cell_id],
            )
            if method_id == "dufs_liu":
                _dufs_stability_records(
                    builder,
                    cell,
                    artifact,
                    sparse_sha256(graph),
                    sparse_sha256(laplacian),
                )
            elif method_id == "ca_specrage_atomic":
                _ca_control_records(builder, cell, artifact, graph)
            elif method_id == "pgrd_a":
                _pgrd_records(builder, cell, artifact, graph, laplacian)

        _continuous_lsml_records(builder, cell, cell.artifacts["continuous_lsml"])
        _family_nrm_records(builder, cell, cell.artifacts["family_nrm_a"])
        _su_pcr_records(builder, cell, cell.artifacts["su_pcr"])

    # Across-cell relation is descriptive only.  Each improvement value is
    # copied from the frozen evaluator; this module never computes AUROC.
    for method_id, rows in alignment.items():
        rows = sorted(rows, key=lambda row: row[0].encode("utf-8"))
        _require(len(rows) >= 3, f"{method_id}: too few cells for alignment relation")
        effects = np.asarray([row[1] for row in rows], dtype=np.float64)
        deltas = np.asarray([row[2] for row in rows], dtype=np.float64)
        ordered_cell_ids = tuple(row[0] for row in rows)
        anchor_cell = verified.cells[ordered_cell_ids[0]]
        artifact = anchor_cell.artifacts[method_id]
        release_binding_id = builder.release_binding(method_id, ordered_cell_ids)
        aggregate_feature_hash = sha256_bytes(canonical_json_bytes([
            {
                "cell_id": cell_id,
                "feature_matrix_sha256": verified.cells[cell_id].feature_matrix_sha256,
            }
            for cell_id in ordered_cell_ids
        ]))
        aggregate_graph_hash = sha256_bytes(canonical_json_bytes([
            {
                "cell_id": cell_id,
                "graph_sha256": graph_health_by_method[method_id][cell_id]["graph_sha256"],
            }
            for cell_id, _, _ in rows
        ]))
        aggregate_operator_hash = sha256_bytes(canonical_json_bytes([
            {
                "cell_id": cell_id,
                "operator_sha256": graph_health_by_method[method_id][cell_id]["operator_sha256"],
            }
            for cell_id, _, _ in rows
        ]))
        relation_values: tuple[tuple[str, float | None], ...]
        if np.std(effects) <= _EPS or np.std(deltas) <= _EPS:
            relation_values = (
                ("spearman_error_alignment_vs_auroc_delta", None),
                ("pearson_error_alignment_vs_auroc_delta", None),
            )
        else:
            relation_values = (
                ("spearman_error_alignment_vs_auroc_delta", float(spearmanr(effects, deltas).statistic)),
                ("pearson_error_alignment_vs_auroc_delta", float(pearsonr(effects, deltas).statistic)),
            )
        for index, (metric, value) in enumerate(relation_values):
            status = "OK" if value is not None and math.isfinite(value) else "METRIC_UNDEFINED_CONSTANT_VECTOR"
            builder.add(
                cell=anchor_cell,
                artifact=artifact,
                stage="post_freeze",
                panel_id="alignment_vs_improvement_summary",
                metric_id=metric,
                value=value if status == "OK" else None,
                unit="correlation_across_explicitly_bound_graph_cells",
                graph_sha256=aggregate_graph_hash,
                operator_sha256=aggregate_operator_hash,
                series_id="descriptive_relation",
                x_index=index,
                x_value=float(index),
                status=status,
                note=f"descriptive across-cell association over {len(rows)} explicitly bound cells; not a causal or inferential claim",
                scope_type="release",
                scope_value=verified.release_id,
                cell_id_override="__release__",
                source_binding_id_override=release_binding_id,
                feature_matrix_sha256_override=aggregate_feature_hash,
            )

    example_arrays: dict[str, np.ndarray] = {
        "schema_version": _text_array([EXAMPLE_DATA_SCHEMA_VERSION]),
        "diagnostic_version": _text_array([DIAGNOSTIC_VERSION]),
        "selection_rule_id": _text_array([EXAMPLE_RULE_ID]),
    }
    for method_id, cell_id in selected_examples.items():
        if cell_id is None:
            continue
        cell = verified.cells[cell_id]
        graph, laplacian = graph_material[(cell_id, method_id)]
        coordinates = _deterministic_diffusion_embedding(laplacian, cell.row_ids)
        upper = sparse.triu(graph, k=1, format="coo")
        prefix = _safe_artifact_token(method_id)
        example_arrays[f"{prefix}__cell_id"] = _text_array([cell_id])
        example_arrays[f"{prefix}__row_ids"] = _text_array(cell.row_ids)
        example_arrays[f"{prefix}__embedding_x"] = np.asarray(coordinates[:, 0], dtype="<f8")
        example_arrays[f"{prefix}__embedding_y"] = np.asarray(coordinates[:, 1], dtype="<f8")
        example_arrays[f"{prefix}__y_error"] = np.asarray(cell.y_error, dtype="<i1")
        example_arrays[f"{prefix}__trace_length_available"] = np.asarray(
            [cell.trace_length_coordinate is not None], dtype=bool
        )
        if cell.trace_length_coordinate is not None:
            example_arrays[f"{prefix}__trace_length_coordinate"] = np.asarray(
                cell.trace_length_coordinate, dtype="<f8"
            )
        example_arrays[f"{prefix}__edge_source"] = np.asarray(upper.row, dtype="<i8")
        example_arrays[f"{prefix}__edge_target"] = np.asarray(upper.col, dtype="<i8")
        example_arrays[f"{prefix}__edge_weight"] = np.asarray(upper.data, dtype="<f8")
        example_arrays[f"{prefix}__feature_matrix_sha256"] = _text_array([cell.feature_matrix_sha256])
        example_arrays[f"{prefix}__graph_sha256"] = _text_array([sparse_sha256(graph)])
        example_arrays[f"{prefix}__operator_sha256"] = _text_array([sparse_sha256(laplacian)])
        if method_id == "pgrd_a" and "residuals" in cell.artifacts[method_id].arrays:
            residuals = np.asarray(cell.artifacts[method_id].arrays["residuals"], dtype="<f8")
            example_arrays[f"{prefix}__residual_coordinates"] = residuals

    coverage = _complete_required_panel_coverage(builder, verified)
    records = sorted(builder.records, key=lambda row: row["diagnostic_id"].encode("utf-8"))
    _require(len({row["diagnostic_id"] for row in records}) == len(records), "duplicate diagnostic ID")
    payload = {
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "release_id": verified.release_id,
        "status": "OK",
        "scope": {
            "population_id": "frozen24_response_v1",
            "n_cells": len(verified.cells),
            "graph_methods": list(GRAPH_METHOD_IDS),
            "non_graph_methods": list(NONGRAPH_METHOD_IDS),
            "performance_metrics_recomputed": False,
            "raw_label_bundle_opened": False,
        },
        "definitions": {
            "graph_health": "weighted degree, connected components, isolated nodes, and second eigenvalue of the symmetric normalized Laplacian",
            "roughness": "centered Rayleigh quotient z.T L z / z.T z; lower means the graph joins more similar values",
            "trace_length_coordinate": "frozen mixed-v2 transformed, confidence-oriented and population-standardized trace_length feature; not raw token count",
            "node_permutation_null": "the fitted graph/operator is node-relabelled while the target or nuisance vector stays fixed",
            "alignment_effect": "median node-permutation roughness minus observed roughness; positive means more alignment than the null median",
            "fixed_graph_group_bootstrap": "source groups are resampled and the already fitted graph is weighted by sqrt row multiplicity; this is sensitivity analysis, not graph refitting",
            "display_embedding": "two SHA-derived probes diffused by I-L and deterministically orthogonalized; display-only and not a Laplacian eigenbasis",
        },
        "null_registry": {
            "node_permutation": {
                "null_id": NODE_PERMUTATION_NULL_ID,
                "draws_per_cell_method": NODE_PERMUTATION_COUNT,
                "rng": "numpy.PCG64",
                "seed_rule": "first 64 bits of SHA256(diagnostic_version|null_id|cell_id|method_id|draw_index)",
            },
            "ca_alpha_controls": {
                "null_id": CA_CONTROL_NULL_ID,
                "controls": list(CA_CONTROL_SERIES),
            },
            "fixed_graph_group_bootstrap": {
                "null_id": GRAPH_BOOTSTRAP_NULL_ID,
                "draws_per_cell_method": GRAPH_BOOTSTRAP_COUNT,
                "rng": "numpy.PCG64",
            },
            "random_family_graph": {
                "null_id": RANDOM_FAMILY_NULL_ID,
                "draws_per_cell_method": GRAPH_BOOTSTRAP_COUNT,
                "selection": "uniform over provenance families present in the frozen cell",
            },
            "length_only_graph": {
                "null_id": LENGTH_ONLY_CONTROL_ID,
                "feature": "frozen transformed confidence-oriented trace_length only",
            },
            "su_sparse_support_bootstrap": {
                "null_id": SU_SUPPORT_BOOTSTRAP_ID,
                "draws_per_cell": SU_SUPPORT_BOOTSTRAP_COUNT,
                "resampling_unit": "source group from evaluator PREDICTION_SNAPSHOT",
            },
        },
        "example_selection": {
            "rule_id": EXAMPLE_RULE_ID,
            "labels_used": False,
            "selected_cell_by_method": selected_examples,
            "rule": "available trace_length nuisance first, then connected, no isolated nodes, largest normalized spectral gap, smallest degree CV, UTF-8 cell ID tie-break",
        },
        "coverage": coverage,
        "provenance": dict(verified.provenance),
        "producer_source_environment_snapshot": snapshot_payload,
        "source_bindings": [
            {"source_binding_id": binding_id, **builder.bindings[binding_id]}
            for binding_id in sorted(builder.bindings, key=lambda value: value.encode("utf-8"))
        ],
        "records": records,
    }
    payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
    return payload, _plot_arrays(builder.plot_rows), example_arrays


__all__ = [
    "CA_CONTROL_NULL_ID",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "DIAGNOSTIC_METHOD_IDS",
    "DIAGNOSTIC_VERSION",
    "EXAMPLE_RULE_ID",
    "GRAPH_METHOD_IDS",
    "GraphDiagnosticContractError",
    "MANIFEST_SCHEMA_VERSION",
    "NODE_PERMUTATION_COUNT",
    "NODE_PERMUTATION_NULL_ID",
    "NONGRAPH_METHOD_IDS",
    "VerifiedDiagnosticRelease",
    "build_graph_diagnostics",
    "deterministic_null_seed",
    "graph_operator_similarity",
    "graph_health",
    "node_permutation_nulls",
    "normalized_roughness",
    "select_example_cell",
    "sparse_sha256",
    "symmetric_normalized_laplacian",
    "verify_diagnostic_release",
]
