#!/usr/bin/env python3
"""Freeze the label-free source, inventory, arm, and Phase-0 registry.

The historical NPZ is used only as a hash-verified catalogue of per-cell feature
names.  This script opens only ``__pool.npy`` members; it never deserializes a
label, feature matrix, score, or polarity member.  Scientific bundles are rebuilt
from the registered raw telemetry sources by ``build_residual_graph_deem_data_v1``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.a5_target_free_data import FROZEN_A0_SOURCE_SPECS  # noqa: E402
from spectral_utils.feature_contract import (  # noqa: E402
    CONFIDENCE_FEATURE_SIGNS_V1,
    FIXED_STABLE_EXCLUDED_V1,
)
from spectral_utils.specrage_views import FEATURE_TO_VIEW, VIEW_ORDER  # noqa: E402


OUT = ROOT / "configs" / "residual_graph_deem_24cell_v1_registry.json"
BUNDLE = ROOT / "results" / "dependency_fusion_raw" / "cells.npz"
MANIFEST = ROOT / "results" / "dependency_fusion_raw" / "cells_manifest.csv"
EXPECTED_BUNDLE_SHA256 = (
    "693a5b634f975ea32c7f840f3ab8366dd8ad638fe41cc76a60e24b1ac5a013e1"
)
PRE_AMENDMENT_PROTOCOL_SHA256 = (
    "ffb708a1e527b45caed245b783defa1b2cfe2f83f6a1e990ad00efb552c31e7d"
)
SPILLED_SOURCE = {
    "environment_id": "spilled_triviaqa_llama8b",
    "dataset": "trivia_qa",
    "split": "validation",
    "dataset_family": "triviaqa",
    "expected_admitted_count": 256,
    "admission_mode": "complete_h16",
    "raw_relative_path": (
        "dataset_cache/repgrid/spilled_triviaqa_llama8b/"
        "raw_trivia_qa_T1.0.pkl"
    ),
    "source_sha256": (
        "cf01350f5bc141908e3f0563c1bc3037148fbad3a30c4eb05c63cd3c13a51e65"
    ),
    "source_size": 7_808_360,
    "manifest_sha256": (
        "ca767e773b8ee5accb54b9f8c1a8ecf441bb8144e956323a1a4fe6ed0091f36f"
    ),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return sha256_bytes(payload)


def pool_catalogue() -> dict[str, list[str]]:
    if sha256_file(BUNDLE) != EXPECTED_BUNDLE_SHA256:
        raise RuntimeError("historical inventory bundle SHA-256 mismatch")
    output: dict[str, list[str]] = {}
    with zipfile.ZipFile(BUNDLE) as archive:
        members = sorted(
            name for name in archive.namelist() if name.endswith("__pool.npy")
        )
        for member in members:
            # The trusted, whole-file hash is checked above.  Only this explicitly
            # selected object-string catalogue is allowed to use pickle.
            names = np.load(BytesIO(archive.read(member)), allow_pickle=True)
            cell = member.removesuffix("__pool.npy")
            output[cell] = [str(name) for name in names.tolist()]
    return output


def source_dict(spec) -> dict:
    return {
        "environment_id": spec.environment_id,
        "dataset": spec.dataset,
        "split": spec.split,
        "dataset_family": spec.dataset_family,
        "expected_admitted_count": spec.expected_admitted_count,
        "admission_mode": spec.admission_mode,
        "raw_relative_path": spec.raw_relative_path,
        "source_sha256": spec.source_sha256,
        "source_size": spec.source_size,
        "manifest_sha256": spec.manifest_sha256,
    }


def arm_registry() -> list[dict]:
    names = {
        "B0": "iu_pcr_inventory",
        "B1": "deem_inventory_hard_adapter020",
        "B2": "deem_inventory_soft_rank_adapter020_repaired",
        "B3": "deem_inventory_continuous_additive",
        "G0": "deem_inventory_raw_graph_uniform_target",
        "G1": "deem_inventory_raw_graph_dufs_target",
        "G2": "deem_inventory_residual_graph_uniform_target",
        "G3": "deem_inventory_residual_graph_dufs_target",
        "G4": "deem_inventory_residual_graph_dufs_nuisance",
        "G5": "deem_inventory_present_family_laplacian",
    }
    return [
        {
            "id": arm_id,
            "name": name,
            "primary": arm_id in {"B3", "G3", "G4"},
            "graph": arm_id.startswith("G"),
            "lambda_zero_alias": "B3" if arm_id.startswith("G") else None,
        }
        for arm_id, name in names.items()
    ]


def synthetic_worlds() -> list[dict]:
    return [
        {"id": "shared_binary_signal", "expected": ["B3", "G0", "G2", "G3"]},
        {"id": "length_target", "expected": [], "negative_control": True},
        {"id": "smooth_nuisance", "expected": ["G4"]},
        {"id": "target_plus_orthogonal_nuisance", "expected": ["G4"]},
        {"id": "linear_target", "expected": ["B3"], "graph_advantage": False},
        {"id": "nonlinear_local_target", "expected": ["G2", "G3"]},
        {"id": "duplicates", "expected": [], "mechanical_only": True},
        {"id": "imbalance_near_constant", "expected": [], "mechanical_only": True},
        {"id": "class_permutation", "expected": [], "mechanical_only": True},
        {"id": "pure_noise", "expected": [], "negative_control": True},
    ]


def main() -> None:
    pools = pool_catalogue()
    source_specs = {spec.environment_id: source_dict(spec) for spec in FROZEN_A0_SOURCE_SPECS}
    if SPILLED_SOURCE["environment_id"] in source_specs:
        raise RuntimeError("spilled source unexpectedly already registered")
    source_specs[SPILLED_SOURCE["environment_id"]] = dict(SPILLED_SOURCE)
    if set(pools) != set(source_specs):
        raise RuntimeError("source and inventory rosters disagree")

    canonical = list(CONFIDENCE_FEATURE_SIGNS_V1)
    missing_by_feature: Counter[str] = Counter()
    missing_by_family: Counter[str] = Counter()
    schemas: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    cells = []
    for cell, names in sorted(pools.items()):
        if names != [name for name in canonical if name in set(names)]:
            raise RuntimeError(f"noncanonical feature order: {cell}")
        missing = [name for name in canonical if name not in names]
        for name in missing:
            missing_by_feature[name] += 1
            missing_by_family[FEATURE_TO_VIEW[name]] += 1
        families = [view for view in VIEW_ORDER if any(FEATURE_TO_VIEW[n] == view for n in names)]
        signs = [int(CONFIDENCE_FEATURE_SIGNS_V1[name]) for name in names]
        source = source_specs[cell]
        task_type = "QA" if source["dataset_family"] not in {"gsm8k", "math500"} else "math"
        inventory_payload = {"feature_names": names, "confidence_signs": signs}
        source["admission_contract_sha256"] = canonical_sha256(
            {
                "mode": source["admission_mode"],
                "expected_admitted_count": source["expected_admitted_count"],
                "row_id": "<cell>::<raw_problem_key>::candidate<ordinal>",
                "group_id": "<cell>::<raw_problem_key>",
            }
        )
        cells.append(
            {
                "cell_id": cell,
                "task_type": task_type,
                "dataset_family": source["dataset_family"],
                "n_rows": source["expected_admitted_count"],
                "n_features": len(names),
                "feature_names": names,
                "confidence_signs": signs,
                "present_families": families,
                "missing_features": missing,
                "stable_inventory_minus4": [
                    name for name in names if name not in FIXED_STABLE_EXCLUDED_V1
                ],
                "inventory_sha256": canonical_sha256(inventory_payload),
                "source": source,
            }
        )
        schemas[tuple(names)].append(cell)

    if sum(item["n_rows"] for item in cells) != 48_607:
        raise RuntimeError("registered row count is not 48,607")
    if sum(len(item["missing_features"]) for item in cells) != 38:
        raise RuntimeError("registered feature-by-cell missing count is not 38")
    if len(schemas) != 7:
        raise RuntimeError("registered inventory does not contain seven schemas")

    registry = {
        "schema": "residual_graph_deem_24cell_v1_registry",
        "frozen_utc": "2026-08-21T00:00:00Z",
        "base_commit": "0a631b28c61496cffb06b32972506cbadfc2cec1",
        "protocol_pre_amendment_sha256": PRE_AMENDMENT_PROTOCOL_SHA256,
        "historical_inventory_reference": {
            "path": "results/dependency_fusion_raw/cells.npz",
            "sha256": EXPECTED_BUNDLE_SHA256,
            "manifest_path": "results/dependency_fusion_raw/cells_manifest.csv",
            "manifest_sha256": sha256_file(MANIFEST),
            "fit_input": False,
            "opened_members": "only __pool.npy during registry construction",
        },
        "population": {
            "n_cells": 24,
            "n_rows": 48_607,
            "n_schemas": 7,
            "feature_count_range": [19, 30],
            "full_30_cells": sum(item["n_features"] == 30 for item in cells),
            "missing_feature_cell_pairs": 38,
            "dataset_families": [
                "triviaqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
                "truthfulqa", "gsm8k", "math500",
            ],
        },
        "canonical_feature_order": canonical,
        "feature_to_family": dict(FEATURE_TO_VIEW),
        "family_order": list(VIEW_ORDER),
        "stable_excluded": sorted(FIXED_STABLE_EXCLUDED_V1),
        "missing_by_feature": dict(sorted(missing_by_feature.items())),
        "missing_by_family": dict(sorted(missing_by_family.items())),
        "schemas": [
            {
                "schema_id": f"inventory_{index:02d}",
                "n_features": len(names),
                "feature_names": list(names),
                "cells": sorted(schema_cells),
            }
            for index, (names, schema_cells) in enumerate(
                sorted(schemas.items(), key=lambda pair: (len(pair[0]), pair[0])), start=1
            )
        ],
        "cells": cells,
        "arms": arm_registry(),
        "solver": {
            "seeds": [0, 1, 2, 3, 4],
            "continuous": {
                "dtype": "float64",
                "device": "cpu",
                "family_width": 8,
                "epochs": 100,
                "optimizer": "SGD",
                "lr": 0.001,
                "momentum": 0.0,
                "mala_delta": 0.10,
                "mala_steps": 5,
                "replay_refresh": 0.05,
                "full_batch": True,
            },
            "adapter_020": {
                "version": "0.2.0",
                "upstream_commit": "7740f606b8fb5506065a8c710da5a00c1425f9b7",
                "preprocessing_layers": 1,
                "preprocessing_activation": "Sparsemax",
                "preprocessing_initialization": "identity",
                "hidden_dim": 1,
                "sampler_steps": 5,
                "batch_size": 1024,
                "momentum": 0.9,
                "weighted_majority_vote": True,
                "initialization": "mv_rand",
                "hard_lr": 0.001,
                "soft_lr": 0.0001,
                "epochs": 100,
            },
        },
        "crossfit": {
            "folds": 5,
            "grouped_by": "group_id",
            "length_bins": 10,
            "residualizer": {
                "predictors": ["ell0", "log1p_raw_trace_length"],
                "polynomial_degree": 3,
                "include_bias": False,
                "ridge_alpha": 1.0,
                "fit_intercept": True,
                "donor_only": True,
            },
        },
        "graph": {
            "storage": "scipy.sparse.csr_matrix",
            "primary_k": 7,
            "sensitivity_k": [5, 10, 15],
            "lambdas": [0.0, 0.01, 0.03, 0.1, 0.3, 1.0],
            "laplacian": "symmetric_normalized",
            "largest_component_min": 0.90,
            "isolated_fraction_max": 0.05,
            "claim_min_healthy_cells": 22,
            "fold_artifact_permutations": 999,
        },
        "dufs": {
            "cross_view": True,
            "gate_sigma": 0.5,
            "mu0": 0.5,
            "optimizer": "Adam",
            "lr": 0.02,
            "epochs": 120,
            "seeds": [0, 1, 2, 3, 4],
            "median_cosine_min": 0.80,
        },
        "synthetic": {
            "base_seed": 20260821,
            "bit_generator": "PCG64",
            "n_rows": 1024,
            "n_groups": 256,
            "rows_per_group": 4,
            "noise_sd": 1.0,
            "signal_loading": 2.0,
            "nuisance_loading": 2.0,
            "imbalance": 0.05,
            "duplicate_fraction": 0.20,
            "worlds": synthetic_worlds(),
        },
        "evaluation": {
            "bootstrap_draws": 10_000,
            "bootstrap_seed": 20260821,
            "whole_search_B": 199,
            "promotion_B": 999,
            "no_pooled_row_auc": True,
        },
    }
    registry["registry_content_sha256"] = canonical_sha256(registry)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"registry_content_sha256={registry['registry_content_sha256']}")


if __name__ == "__main__":
    main()
