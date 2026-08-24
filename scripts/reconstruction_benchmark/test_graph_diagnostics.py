#!/usr/bin/env python3
"""Focused tests for frozen-24 graph diagnostic contracts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import MappingProxyType
import unittest

import numpy as np
from scipy import sparse


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.graph_diagnostics import (  # noqa: E402
    GraphDiagnosticContractError,
    CA_CONTROL_SERIES,
    GRAPH_BOOTSTRAP_COUNT,
    NODE_PERMUTATION_COUNT,
    REQUIRED_PANELS_BY_METHOD,
    VerifiedArtifact,
    VerifiedDiagnosticCell,
    VerifiedDiagnosticRelease,
    _RecordBuilder,
    _complete_required_panel_coverage,
    _deterministic_diffusion_embedding,
    _csr_from_flat,
    _has_complete_csr,
    _optional_trace_length_coordinate,
    _plot_arrays,
    _strict_json,
    _su_upper_offdiag_relative_residual,
    _validate_artifact_index,
    _verify_file,
    _verify_payload_hash,
    assert_source_environment_snapshot_unchanged,
    capture_source_environment_snapshot,
    deterministic_null_seed,
    fixed_graph_group_bootstrap,
    graph_operator_similarity,
    graph_health,
    node_permutation_nulls,
    normalized_roughness,
    select_example_cell,
    sparse_sha256,
    symmetric_normalized_laplacian,
)
from spectral_utils.reconstruction_benchmark import PreparedCell, run_method  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.serialization import write_score_result  # noqa: E402


def chain_graph(n: int) -> sparse.csr_matrix:
    row = np.concatenate([np.arange(n - 1), np.arange(1, n)])
    col = np.concatenate([np.arange(1, n), np.arange(n - 1)])
    return sparse.csr_matrix((np.ones(2 * (n - 1)), (row, col)), shape=(n, n))


def flatten_csr(prefix: str, matrix: sparse.spmatrix) -> dict[str, np.ndarray]:
    value = sparse.csr_matrix(matrix)
    return {
        f"{prefix}__data": np.asarray(value.data),
        f"{prefix}__indices": np.asarray(value.indices, dtype=np.int64),
        f"{prefix}__indptr": np.asarray(value.indptr, dtype=np.int64),
        f"{prefix}__shape": np.asarray(value.shape, dtype=np.int64),
    }


def synthetic_verified_release(cell_ids=("cell-a",)) -> VerifiedDiagnosticRelease:
    method_ids = tuple(REQUIRED_PANELS_BY_METHOD)
    versions = {method_id: f"{method_id}-v1" for method_id in method_ids}
    cells = {}
    provenance = {
        "score_freeze_A_path": "build_A/fit/SCORE_FREEZE_MANIFEST.json",
        "score_freeze_A_sha256": "1" * 64,
        "score_ab_verification_path": "SCORE_AB_VERIFICATION.json",
        "score_ab_verification_sha256": "2" * 64,
        "evaluation_manifest_path": "evaluation/EVALUATION_MANIFEST.json",
        "evaluation_manifest_sha256": "3" * 64,
        "prediction_snapshot_path": "evaluation/PREDICTION_SNAPSHOT.npz",
        "prediction_snapshot_sha256": "4" * 64,
    }
    for cell_id in cell_ids:
        artifacts = {}
        for method_id in method_ids:
            artifacts[method_id] = VerifiedArtifact(
                method_id=method_id,
                method_version_id=versions[method_id],
                config=MappingProxyType({}),
                status="OK_FALLBACK",
                fallback_reason="synthetic artifact intentionally unavailable",
                record=MappingProxyType({}),
                arrays=MappingProxyType({}),
                index=MappingProxyType({}),
                score=np.zeros(6),
                record_path=f"{cell_id}/{method_id}/RECORD.json",
                record_sha256="5" * 64,
                score_path=f"{cell_id}/{method_id}/score.npz",
                score_sha256="6" * 64,
                artifact_path=None,
                artifact_sha256=None,
                artifact_index_path=f"{cell_id}/{method_id}/ARTIFACT_INDEX.json",
                artifact_index_sha256="7" * 64,
            )
        cells[cell_id] = VerifiedDiagnosticCell(
            cell_id=cell_id,
            domain="synthetic",
            row_ids=tuple(f"row-{index}" for index in range(6)),
            group_ids=tuple(f"group-{index}" for index in range(6)),
            feature_names=("epr", "trace_length", "spectral_entropy"),
            X_confidence=np.zeros((6, 3)),
            y_error=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int8),
            trace_length_coordinate=None,
            feature_matrix_sha256="8" * 64,
            prepared_matrix_sha256="9" * 64,
            prepared_path=f"build_A/inputs/{cell_id}.npz",
            prepared_sha256="a" * 64,
            artifacts=MappingProxyType(artifacts),
        )
    return VerifiedDiagnosticRelease(
        release_root=Path("/synthetic"),
        release_id="synthetic-release",
        cells=MappingProxyType(cells),
        method_ids=method_ids,
        method_versions=MappingProxyType(versions),
        evaluation=MappingProxyType({}),
        auroc_delta_vs_iu=MappingProxyType({}),
        provenance=MappingProxyType(provenance),
    )


class GraphHealthTests(unittest.TestCase):
    def test_connected_chain_health(self) -> None:
        graph = chain_graph(6)
        laplacian = symmetric_normalized_laplacian(graph)
        health = graph_health(graph, laplacian)
        self.assertEqual(health["n_nodes"], 6)
        self.assertEqual(health["n_edges"], 5)
        self.assertEqual(health["n_components"], 1)
        self.assertEqual(health["isolated_nodes"], 0)
        self.assertGreater(health["normalized_spectral_gap"], 0.0)
        self.assertEqual(health["graph_sha256"], sparse_sha256(graph))

    def test_disconnected_and_isolated_health(self) -> None:
        graph = sparse.block_diag((chain_graph(3), sparse.csr_matrix((1, 1))), format="csr")
        health = graph_health(graph)
        self.assertEqual(health["n_components"], 2)
        self.assertEqual(health["isolated_nodes"], 1)
        self.assertEqual(health["normalized_spectral_gap"], 0.0)

    def test_tampered_laplacian_fails_closed(self) -> None:
        graph = chain_graph(5)
        laplacian = symmetric_normalized_laplacian(graph).tolil()
        laplacian[0, 0] += 0.1
        with self.assertRaises(GraphDiagnosticContractError):
            graph_health(graph, laplacian.tocsr())

    def test_sparse_hash_canonicalizes_index_order(self) -> None:
        graph = chain_graph(5)
        scrambled = graph.copy()
        for row in range(scrambled.shape[0]):
            start, end = scrambled.indptr[row:row + 2]
            scrambled.indices[start:end] = scrambled.indices[start:end][::-1]
            scrambled.data[start:end] = scrambled.data[start:end][::-1]
        self.assertEqual(sparse_sha256(graph), sparse_sha256(scrambled))

    def test_graph_operator_identity_is_one(self) -> None:
        graph = chain_graph(7)
        laplacian = symmetric_normalized_laplacian(graph)
        similarity = graph_operator_similarity(graph, laplacian, graph, laplacian)
        self.assertAlmostEqual(similarity["edge_support_jaccard"], 1.0)
        self.assertAlmostEqual(similarity["weighted_graph_frobenius_cosine"], 1.0)
        self.assertAlmostEqual(similarity["normalized_laplacian_frobenius_cosine"], 1.0)
        self.assertAlmostEqual(similarity["normalized_laplacian_relative_difference"], 0.0)


class RoughnessTests(unittest.TestCase):
    def test_smooth_signal_has_lower_roughness(self) -> None:
        graph = chain_graph(12)
        laplacian = symmetric_normalized_laplacian(graph)
        smooth = np.linspace(-1.0, 1.0, 12)
        alternating = (-1.0) ** np.arange(12)
        self.assertLess(
            normalized_roughness(smooth, laplacian),
            normalized_roughness(alternating, laplacian),
        )

    def test_constant_signal_is_undefined(self) -> None:
        with self.assertRaises(GraphDiagnosticContractError):
            normalized_roughness(np.ones(5), symmetric_normalized_laplacian(chain_graph(5)))

    def test_node_null_is_deterministic_and_has_registered_size(self) -> None:
        graph = chain_graph(10)
        laplacian = symmetric_normalized_laplacian(graph)
        target = np.asarray([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=float)
        nuisance = np.linspace(-2, 2, 10)
        first = node_permutation_nulls(
            graph=graph,
            laplacian=laplacian,
            target=target,
            nuisance=nuisance,
            cell_id="cell",
            method_id="method",
        )
        second = node_permutation_nulls(
            graph=graph,
            laplacian=laplacian,
            target=target,
            nuisance=nuisance,
            cell_id="cell",
            method_id="method",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), NODE_PERMUTATION_COUNT)
        self.assertGreaterEqual(len(first), 20)
        self.assertEqual(len({row["seed"] for row in first}), NODE_PERMUTATION_COUNT)
        self.assertEqual(first[0]["seed"], deterministic_null_seed("cell", "method", 0))

    def test_node_null_allows_missing_trace_without_substitution(self) -> None:
        graph = chain_graph(8)
        rows = node_permutation_nulls(
            graph=graph,
            laplacian=symmetric_normalized_laplacian(graph),
            target=np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=float),
            nuisance=None,
            cell_id="missing-trace",
            method_id="dufs_liu",
        )
        self.assertTrue(all(row["nuisance_roughness"] is None for row in rows))

    def test_group_bootstrap_is_deterministic(self) -> None:
        graph = chain_graph(10)
        groups = tuple(f"g-{index // 2}" for index in range(10))
        first = fixed_graph_group_bootstrap(
            graph=graph,
            group_ids=groups,
            cell_id="cell",
            method_id="pgrd_a",
        )
        second = fixed_graph_group_bootstrap(
            graph=graph,
            group_ids=groups,
            cell_id="cell",
            method_id="pgrd_a",
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), GRAPH_BOOTSTRAP_COUNT)


class SelectionTests(unittest.TestCase):
    def test_selection_uses_only_frozen_health_order(self) -> None:
        health = {
            "z_cell": {
                "n_components": 1,
                "isolated_nodes": 0,
                "normalized_spectral_gap": 0.2,
                "degree_cv": 0.1,
            },
            "a_cell": {
                "n_components": 1,
                "isolated_nodes": 0,
                "normalized_spectral_gap": 0.2,
                "degree_cv": 0.1,
            },
            "bad_cell": {
                "n_components": 2,
                "isolated_nodes": 0,
                "normalized_spectral_gap": 1.0,
                "degree_cv": 0.0,
            },
        }
        self.assertEqual(select_example_cell(health), "a_cell")
        # There is no label argument, so changing any external label array
        # cannot alter this result.
        fake_labels = np.asarray([1, 0, 1])
        fake_labels[:] = 1 - fake_labels
        self.assertEqual(select_example_cell(health), "a_cell")

    def test_selection_prefers_available_nuisance(self) -> None:
        health = {
            "beautiful_but_missing": {
                "n_components": 1,
                "isolated_nodes": 0,
                "normalized_spectral_gap": 0.9,
                "degree_cv": 0.01,
            },
            "available": {
                "n_components": 1,
                "isolated_nodes": 0,
                "normalized_spectral_gap": 0.1,
                "degree_cv": 0.2,
            },
        }
        self.assertEqual(
            select_example_cell(
                health,
                {"beautiful_but_missing": False, "available": True},
            ),
            "available",
        )


class DeterminismTests(unittest.TestCase):
    def test_plot_projection_preserves_full_unsigned_seed(self) -> None:
        seed = 2**64 - 2
        row = {
            "diagnostic_id": "diag-max-seed",
            "scope_type": "cell",
            "scope_value": "cell-a",
            "cell_id": "cell-a",
            "method_id": "dufs_liu",
            "method_version_id": "dufs-v1",
            "stage": "target_free",
            "compared_method_id": "not_applicable",
            "compared_method_version_id": "not_applicable",
            "panel_id": "node_permutation_null",
            "metric_id": "target_roughness",
            "series_id": "draw_0",
            "null_id": "node_permutation_fixed_signal_v1",
            "feature_matrix_sha256": "1" * 64,
            "graph_sha256": "2" * 64,
            "operator_sha256": "3" * 64,
            "compared_graph_sha256": "not_applicable",
            "compared_operator_sha256": "not_applicable",
            "source_binding_id": "binding-a",
            "x_index": 0,
            "seed": seed,
            "draw_index": 0,
            "x_value": 0.0,
            "y_value": 0.25,
        }
        arrays = _plot_arrays([row])
        self.assertEqual(arrays["seed"].dtype.kind, "U")
        self.assertEqual(arrays["seed"].tolist(), [str(seed)])

    def test_exact_ca_control_roster(self) -> None:
        self.assertEqual(
            CA_CONTROL_SERIES,
            (
                "learned",
                "equal_view",
                "provenance_prior",
                "global_mean_alpha",
                "permuted",
            ),
        )

    def test_display_embedding_is_byte_stable_on_repeated_eigenspaces(self) -> None:
        # A cycle has repeated Laplacian eigenvalues; the display coordinates
        # must not inherit an arbitrary eigenbasis rotation.
        n = 8
        rows = np.repeat(np.arange(n), 2)
        columns = np.column_stack([
            (np.arange(n) - 1) % n,
            (np.arange(n) + 1) % n,
        ]).reshape(-1)
        graph = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(n, n))
        laplacian = symmetric_normalized_laplacian(graph)
        row_ids = tuple(f"row-{index}" for index in range(n))
        first = _deterministic_diffusion_embedding(laplacian, row_ids)
        second = _deterministic_diffusion_embedding(laplacian, row_ids)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_su_residual_uses_strict_upper_off_diagonal(self) -> None:
        covariance = np.asarray([
            [100.0, 1.0, 2.0],
            [1.0, 200.0, 3.0],
            [2.0, 3.0, 300.0],
        ])
        low = np.asarray([
            [999.0, 0.5, 1.0],
            [0.5, 999.0, 1.5],
            [1.0, 1.5, 999.0],
        ])
        sparse_part = np.zeros((3, 3))
        observed = np.asarray([1.0, 2.0, 3.0])
        residual = np.asarray([0.5, 1.0, 1.5])
        expected = float(np.linalg.norm(residual) / np.linalg.norm(observed))
        self.assertAlmostEqual(
            _su_upper_offdiag_relative_residual(covariance, low, sparse_part),
            expected,
        )

    def test_graph_members_must_be_complete(self) -> None:
        arrays = flatten_csr("graph", chain_graph(5))
        self.assertTrue(_has_complete_csr(arrays, "graph"))
        arrays.pop("graph__shape")
        with self.assertRaises(GraphDiagnosticContractError):
            _has_complete_csr(arrays, "graph")


class CoverageAndScopeTests(unittest.TestCase):
    def test_requested_panel_coverage_is_exact_and_explicit(self) -> None:
        release = synthetic_verified_release()
        builder = _RecordBuilder(release)
        summary = _complete_required_panel_coverage(builder, release)
        expected = sum(len(panels) for panels in REQUIRED_PANELS_BY_METHOD.values())
        self.assertEqual(summary["expected_panel_slots"], expected)
        self.assertEqual(summary["observed_panel_slots"], expected)
        self.assertEqual(len(builder.records), expected)
        self.assertTrue(all(row["status"].startswith("NOT_AVAILABLE") for row in builder.records))
        keys = {
            (row["cell_id"], row["method_id"], row["panel_id"])
            for row in builder.records
        }
        self.assertEqual(len(keys), expected)

    def test_release_scope_uses_multi_cell_binding(self) -> None:
        release = synthetic_verified_release(("cell-b", "cell-a"))
        builder = _RecordBuilder(release, "b" * 64)
        binding_id = builder.release_binding("dufs_liu", ("cell-b", "cell-a"))
        binding = builder.bindings[binding_id]
        self.assertEqual(binding["binding_type"], "multi_cell_method_artifacts")
        self.assertEqual(
            [row["cell_id"] for row in binding["cell_source_bindings"]],
            ["cell-a", "cell-b"],
        )
        anchor = release.cells["cell-a"]
        builder.add(
            cell=anchor,
            artifact=anchor.artifacts["dufs_liu"],
            stage="post_freeze",
            panel_id="alignment_vs_improvement_summary",
            metric_id="spearman_error_alignment_vs_auroc_delta",
            value=0.25,
            unit="correlation_across_cells",
            scope_type="release",
            scope_value=release.release_id,
            cell_id_override="__release__",
            source_binding_id_override=binding_id,
            feature_matrix_sha256_override="c" * 64,
        )
        record = builder.records[-1]
        self.assertEqual(record["scope_type"], "release")
        self.assertEqual(record["scope_value"], release.release_id)
        self.assertEqual(record["cell_id"], "__release__")
        self.assertEqual(record["source_binding_id"], binding_id)


class ProvenanceTamperTests(unittest.TestCase):
    def test_verifier_trace_coordinate_is_optional_and_never_substituted(self) -> None:
        matrix = np.arange(12, dtype=float).reshape(4, 3)
        self.assertIsNone(
            _optional_trace_length_coordinate(
                matrix,
                ("epr", "spectral_entropy", "epr_energy"),
            )
        )
        present = _optional_trace_length_coordinate(
            matrix,
            ("epr", "trace_length", "epr_energy"),
        )
        np.testing.assert_array_equal(present, matrix[:, 1])

    def test_payload_tamper_is_rejected(self) -> None:
        payload = {"schema_version": "x", "value": 1}
        payload["payload_sha256"] = sha256_bytes(canonical_json_bytes(payload))
        _verify_payload_hash(payload, "payload_sha256", context="fixture")
        payload["value"] = 2
        with self.assertRaises(GraphDiagnosticContractError):
            _verify_payload_hash(payload, "payload_sha256", context="fixture")

    def test_file_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"original")
            digest = sha256_file(path)
            _verify_file(path, digest, context="fixture")
            path.write_bytes(b"tampered")
            with self.assertRaises(GraphDiagnosticContractError):
                _verify_file(path, digest, context="fixture")

    def test_duplicate_json_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaises(GraphDiagnosticContractError):
                _strict_json(path)

    def test_artifact_index_tamper_is_rejected(self) -> None:
        graph = chain_graph(4)
        arrays = flatten_csr("graph", graph)
        index = {"graph": {"type": "csr_matrix", "shape": [4, 4]}}
        _validate_artifact_index(arrays, index, context="fixture")
        tampered = dict(arrays)
        tampered.pop("graph__indptr")
        with self.assertRaises(GraphDiagnosticContractError):
            _validate_artifact_index(tampered, index, context="fixture")

    def test_invalid_csr_is_rejected(self) -> None:
        arrays = flatten_csr("graph", chain_graph(4))
        arrays["graph__indices"] = np.asarray([99] * len(arrays["graph__indices"]))
        with self.assertRaises(GraphDiagnosticContractError):
            _csr_from_flat(arrays, "graph")

    def test_real_pgrd_serialization_matches_diagnostic_loader(self) -> None:
        names = (
            "epr",
            "trace_length",
            "spectral_entropy",
            "epr_spilled",
            "epr_energy",
            "mean_top1_logprob",
        )
        rng = np.random.default_rng(91)
        matrix = rng.normal(size=(48, len(names)))
        matrix = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0)
        cell = PreparedCell(
            population_id="synthetic",
            cell_id="synthetic",
            domain="synthetic",
            matrix=matrix,
            feature_names=names,
            row_ids=tuple(f"row-{index:03d}" for index in range(len(matrix))),
        )
        result = run_method("pgrd_a", cell)
        self.assertEqual(result.status.value, "OK")
        with tempfile.TemporaryDirectory() as directory:
            write_score_result(result, cell.row_ids, directory)
            arrays = dict(np.load(Path(directory) / "artifacts.npz", allow_pickle=False))
            index = json.loads((Path(directory) / "ARTIFACT_INDEX.json").read_text())
            _validate_artifact_index(arrays, index, context="serialized PGRD")
            graph = _csr_from_flat(arrays, "graph")
            laplacian = _csr_from_flat(arrays, "laplacian")
            self.assertEqual(graph_health(graph, laplacian)["n_nodes"], len(matrix))

    def test_clean_source_snapshot_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(("git", "init", "-q", str(root)), check=True)
            subprocess.run(("git", "-C", str(root), "config", "user.email", "test@example.com"), check=True)
            subprocess.run(("git", "-C", str(root), "config", "user.name", "Test"), check=True)
            source = root / "producer.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(("git", "-C", str(root), "add", "producer.py"), check=True)
            subprocess.run(("git", "-C", str(root), "commit", "-qm", "fixture"), check=True)
            snapshot = capture_source_environment_snapshot(
                root,
                source_paths=("producer.py",),
            )
            self.assertEqual(
                assert_source_environment_snapshot_unchanged(
                    root,
                    snapshot,
                    source_paths=("producer.py",),
                )["snapshot_sha256"],
                snapshot["snapshot_sha256"],
            )
            source.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaises(GraphDiagnosticContractError):
                assert_source_environment_snapshot_unchanged(
                    root,
                    snapshot,
                    source_paths=("producer.py",),
                )


if __name__ == "__main__":
    unittest.main()
