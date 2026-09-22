#!/usr/bin/env python3
"""Focused target-free tests for the 24-cell group-sidecar builder."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spectral_utils.a5_target_free_data import CORE_FEATURES  # noqa: E402
from spectral_utils.dufs_liu_feature_contract import (  # noqa: E402
    dufs_liu_mixed_v2_from_bundle,
)
from spectral_utils.feature_contract import LEGACY_FEATURE_SIGNS  # noqa: E402
from spectral_utils.fair_comparisons.twentyfour import TwentyFourError  # noqa: E402
from spectral_utils.reconstruction_benchmark.group_sidecars import (  # noqa: E402
    GroupSidecarError,
    audit_feature_group_collisions,
    historical_featcache_order,
    load_source_registry,
    load_target_free_bundle_cell,
    prove_prepared_row_alignment,
    prove_positional_feature_alignment,
    repeated_group_ids,
    row_group_binding_sha256,
    singleton_group_ids,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    canonical_json_bytes,
    sha256_file,
)


@dataclass(frozen=True)
class _Identity:
    item_group_id: str
    candidate_ordinal: int
    group_id: str
    core_features: np.ndarray


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    scale = values.std(ddof=0)
    return values - values.mean() if scale <= 1e-8 else (
        values - values.mean()
    ) / scale


class SourceRegistryTests(unittest.TestCase):
    def test_registry_content_hash_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            registry = json.loads(
                (REPO_ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
                .read_text(encoding="utf-8")
            )
            path.write_text(json.dumps(registry), encoding="utf-8")
            self.assertEqual(len(load_source_registry(path)["cells"]), 24)

            registry["cells"][0]["source"]["environment_id"] = "tampered"
            path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(GroupSidecarError, "content hash mismatch"):
                load_source_registry(path)

            body = dict(registry)
            body.pop("registry_content_sha256")
            registry["registry_content_sha256"] = hashlib.sha256(
                canonical_json_bytes(body)
            ).hexdigest()
            path.write_text(json.dumps(registry), encoding="utf-8")
            with self.assertRaisesRegex(GroupSidecarError, "not the frozen v1"):
                load_source_registry(path)

    def test_bundle_reader_indexes_only_the_three_whitelisted_arrays(self) -> None:
        class GuardedBundle:
            files = (
                "cell__V",
                "cell__pool",
                "cell__hand_signs",
                "cell__labels",
            )

            def __init__(self) -> None:
                self.accessed: list[str] = []
                self.values = {
                    "cell__V": np.ones((2, 2)),
                    "cell__pool": np.asarray(["epr", "rpdi"]),
                    "cell__hand_signs": np.asarray([1, -1]),
                }

            def __getitem__(self, key: str) -> np.ndarray:
                self.accessed.append(key)
                if "label" in key:
                    raise AssertionError("target member was indexed")
                return self.values[key]

        bundle = GuardedBundle()
        matrix, pool, signs = load_target_free_bundle_cell(bundle, "cell")
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(pool, ("epr", "rpdi"))
        self.assertTrue(np.array_equal(signs, np.asarray([1.0, -1.0])))
        self.assertEqual(
            set(bundle.accessed),
            {"cell__V", "cell__pool", "cell__hand_signs"},
        )


class SingletonProofTests(unittest.TestCase):
    def test_k1_manifest_proves_unique_non_row_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = {
                "dataset": "demo",
                "split": "test",
                "k": 1,
                "cells": [{"k": 1}, {"k": 1}, {"k": 1}],
            }
            path.write_text(json.dumps(manifest), encoding="utf-8")
            row_ids = ("prepared::0", "prepared::1", "prepared::2")
            spec = {
                "dataset": "demo",
                "split": "test",
                "manifest_sha256": sha256_file(path),
                "expected_admitted_count": 3,
            }
            groups, proof = singleton_group_ids(
                cell_id="cell",
                row_ids=row_ids,
                manifest_path=path,
                source_spec=spec,
            )
            self.assertEqual(len(set(groups)), 3)
            self.assertNotEqual(groups, row_ids)
            self.assertFalse(proof["raw_source_read"])
            self.assertFalse(proof["source_identity_recovered"])

    def test_k1_manifest_tamper_and_repetition_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = {
                "dataset": "demo",
                "split": "test",
                "k": 1,
                "cells": [{"k": 1}],
            }
            path.write_text(json.dumps(manifest), encoding="utf-8")
            spec = {
                "dataset": "demo",
                "split": "test",
                "manifest_sha256": sha256_file(path),
                "expected_admitted_count": 1,
            }
            manifest["cells"][0]["k"] = 2
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(GroupSidecarError, "hash mismatch"):
                singleton_group_ids(
                    cell_id="cell",
                    row_ids=("prepared::0",),
                    manifest_path=path,
                    source_spec=spec,
                )

            spec["manifest_sha256"] = sha256_file(path)
            with self.assertRaisesRegex(GroupSidecarError, "requires k=1"):
                singleton_group_ids(
                    cell_id="cell",
                    row_ids=("prepared::0",),
                    manifest_path=path,
                    source_spec=spec,
                )


class RepeatedIdentityProofTests(unittest.TestCase):
    def test_collision_audit_allows_siblings_but_blocks_cross_group_rows(self) -> None:
        matrix = np.asarray([[1.0, 2.0], [1.0, 2.0], [3.0, 4.0]])
        proof = audit_feature_group_collisions(matrix, ("g0", "g0", "g1"))
        self.assertEqual(proof["cross_group_collisions"], 0)
        self.assertEqual(
            proof["regimes"]["exact_float64"]["duplicated_classes"], 1
        )
        with self.assertRaisesRegex(GroupSidecarError, "different source groups"):
            audit_feature_group_collisions(matrix, ("g0", "g1", "g2"))

    def test_repeated_cell_has_no_missing_raw_iid_fallback(self) -> None:
        registry = json.loads(
            (REPO_ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
            .read_text(encoding="utf-8")
        )
        source_spec = next(
            row["source"]
            for row in registry["cells"]
            if row["cell_id"] == "se_nq_open_llama8b"
        )
        with tempfile.TemporaryDirectory() as raw_root:
            with self.assertRaisesRegex(TwentyFourError, "missing materialized raw"):
                repeated_group_ids(
                    repo_root=REPO_ROOT,
                    raw_root=Path(raw_root),
                    cell_id="se_nq_open_llama8b",
                    row_ids=("row",),
                    source_spec=source_spec,
                    bundle_matrix=np.empty((0, 0)),
                    bundle_pool=(),
                    bundle_hand_signs=np.empty(0),
                    prepared_matrix=np.empty((0, 0)),
                    prepared_names=(),
                )

    def test_historical_order_uses_native_keys_not_lexical_keys(self) -> None:
        empty = np.zeros(len(CORE_FEATURES), dtype=float)
        source = {
            10: {"candidates": [{}, {}]},
            2: {"candidates": [{}]},
            1: {"candidates": [{}]},
        }
        lexical = (
            _Identity("1", 0, "g1", empty),
            _Identity("10", 0, "g10", empty),
            _Identity("10", 1, "g10", empty),
            _Identity("2", 0, "g2", empty),
        )
        ordered = historical_featcache_order(source, lexical)
        observed = tuple(
            (row.item_group_id, row.candidate_ordinal) for row in ordered
        )
        self.assertEqual(observed, (("1", 0), ("2", 0), ("10", 0), ("10", 1)))

    def test_full_column_feature_fingerprint_accepts_exact_order(self) -> None:
        n_rows = 11
        row_axis = np.arange(n_rows, dtype=float)[:, None]
        feature_axis = np.arange(len(CORE_FEATURES), dtype=float)[None, :]
        core = (
            np.sin((row_axis + 1.0) * (feature_axis + 1.0) / 7.0)
            + row_axis / (feature_axis + 2.0)
        )
        signs = np.where(np.arange(len(CORE_FEATURES)) % 2, -1.0, 1.0)
        matrix = np.column_stack([
            _standardize(core[:, index] * signs[index])
            for index in range(len(CORE_FEATURES))
        ])
        identities = tuple(
            _Identity(str(index), 0, f"g{index}", core[index])
            for index in range(n_rows)
        )
        proof = prove_positional_feature_alignment(
            ordered_identities=identities,
            bundle_matrix=matrix,
            pool=CORE_FEATURES,
            hand_signs=signs,
        )
        self.assertEqual(proof["n_exact_features"], len(CORE_FEATURES))
        self.assertLessEqual(proof["max_abs_error"], 1e-12)

        shuffled = matrix.copy()
        shuffled[[0, 1]] = shuffled[[1, 0]]
        with self.assertRaisesRegex(GroupSidecarError, "positional feature proof"):
            prove_positional_feature_alignment(
                ordered_identities=identities,
                bundle_matrix=shuffled,
                pool=CORE_FEATURES,
                hand_signs=signs,
            )

    def test_sparse_roster_cannot_hide_collision_in_unobserved_feature(self) -> None:
        n_rows = 10
        core = np.arange(
            n_rows * len(CORE_FEATURES), dtype=float
        ).reshape(n_rows, len(CORE_FEATURES))
        core[1, :8] = core[0, :8]
        self.assertNotEqual(core[1, 8], core[0, 8])
        names = CORE_FEATURES[:8]
        signs = np.ones(len(names), dtype=float)
        matrix = np.column_stack([
            _standardize(core[:, index]) for index in range(len(names))
        ])
        identities = tuple(
            _Identity(str(index), 0, f"g{index}", core[index])
            for index in range(n_rows)
        )
        with self.assertRaisesRegex(GroupSidecarError, "different source groups"):
            prove_positional_feature_alignment(
                ordered_identities=identities,
                bundle_matrix=matrix,
                pool=names,
                hand_signs=signs,
            )

    def test_one_tampered_common_column_blocks_the_proof(self) -> None:
        n_rows = 9
        core = np.arange(
            n_rows * len(CORE_FEATURES), dtype=float
        ).reshape(n_rows, len(CORE_FEATURES))
        signs = np.ones(len(CORE_FEATURES), dtype=float)
        matrix = np.column_stack([
            _standardize(core[:, index])
            for index in range(len(CORE_FEATURES))
        ])
        matrix[0, -1] += 1e-5
        identities = tuple(
            _Identity(str(index), 0, f"g{index}", core[index])
            for index in range(n_rows)
        )
        with self.assertRaisesRegex(GroupSidecarError, "drifting common columns"):
            prove_positional_feature_alignment(
                ordered_identities=identities,
                bundle_matrix=matrix,
                pool=CORE_FEATURES,
                hand_signs=signs,
            )

    def test_prepared_matrix_must_be_exact_mixed_v2_rebuild(self) -> None:
        generator = np.random.default_rng(8127)
        stored = generator.normal(size=(53, len(CORE_FEATURES)))
        signs = np.asarray(
            [LEGACY_FEATURE_SIGNS[name] for name in CORE_FEATURES], dtype=float
        )
        prepared, names, _ = dufs_liu_mixed_v2_from_bundle(
            stored, CORE_FEATURES, signs
        )
        proof = prove_prepared_row_alignment(
            source_matrix=stored,
            source_names=CORE_FEATURES,
            source_hand_signs=signs,
            prepared_matrix=prepared,
            prepared_names=names,
        )
        self.assertTrue(proof["exact_array_equality"])

        tampered = np.asarray(prepared).copy()
        tampered[0, 0] += np.finfo(float).eps
        with self.assertRaisesRegex(GroupSidecarError, "do not exactly reproduce"):
            prove_prepared_row_alignment(
                source_matrix=stored,
                source_names=CORE_FEATURES,
                source_hand_signs=signs,
                prepared_matrix=tampered,
                prepared_names=names,
            )

    def test_binding_hash_changes_if_one_group_is_tampered(self) -> None:
        rows = ("r0", "r1", "r2")
        original = row_group_binding_sha256(rows, ("g0", "g0", "g1"))
        tampered = row_group_binding_sha256(rows, ("g0", "g1", "g1"))
        self.assertNotEqual(original, tampered)


if __name__ == "__main__":
    unittest.main()
