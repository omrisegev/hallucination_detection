#!/usr/bin/env python3
"""Fit target-free B3 on Feature Contract V2 and freeze all scores."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_feature_contract_v2 import (  # noqa: E402
    BLOCK_ORDER,
    fit_v2_b3,
    prepare_v2_risk,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import load_registry, load_target_free_bundle  # noqa: E402


SCHEMA = "deem_b3_feature_contract_v2_run_2026_08_25"


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "deem_b3_feature_contract_v2_config":
        raise ValueError("config schema mismatch")
    if value.get("seeds") != [0, 1, 2, 3, 4] or value.get("blocks") != list(BLOCK_ORDER):
        raise ValueError("config roster mismatch")
    return value


def source_hashes(config_path: Path) -> dict[str, str]:
    paths = {
        "runner": Path(__file__),
        "core": ROOT / "spectral_utils" / "deem_b3_feature_contract_v2.py",
        "frozen_b3_core": ROOT / "spectral_utils" / "residual_graph_deem.py",
        "config": config_path,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def load_v2_bundle(path: Path) -> dict[str, Any]:
    digest = sha256_file(path)
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"].item()) != "feature_contract_v2_2026_08_25":
            raise ValueError("Feature Contract V2 bundle schema mismatch")
        value = {
            "cell_id": str(data["cell_id"].item()),
            "X": np.asarray(data["X_contract_raw"], dtype=np.float64),
            "names": tuple(str(x) for x in data["feature_names"].tolist()),
            "row_ids": tuple(str(x) for x in data["row_id"].tolist()),
            "group_ids": tuple(str(x) for x in data["group_id"].tolist()),
            "raw_trace_length": np.asarray(data["raw_trace_length"], dtype=np.int64),
            "dataset_family": str(data["dataset_family"].item()),
            "source_bundle_sha256": str(data["source_bundle_sha256"].item()),
            "ordered_row_id_sha256": str(data["ordered_row_id_sha256"].item()),
            "bundle_sha256": digest,
        }
    if value["X"].shape != (len(value["row_ids"]), len(value["names"])):
        raise ValueError("V2 bundle alignment mismatch")
    return value


def write_fit(
    out_dir: Path,
    bundle: dict[str, Any],
    seed: int,
    result,
    transform,
    definition_hash: str,
) -> dict[str, Any]:
    cell_id = bundle["cell_id"]
    fit_dir = out_dir / "fits" / cell_id
    fit_dir.mkdir(parents=True, exist_ok=True)
    stem = f"B3_V2__seed{seed}"
    family_order = tuple(result.family_indices)
    arrays = {
        "schema": np.asarray("deem_b3_feature_contract_v2_fit"),
        "cell_id": np.asarray(cell_id),
        "seed": np.asarray(seed, dtype=np.int64),
        "score": np.asarray(result.score, dtype=np.float64),
        "posterior": np.asarray(result.posterior, dtype=np.float64),
        "logit": np.asarray(result.logit, dtype=np.float64),
        "contributions": np.asarray(result.contributions, dtype=np.float64),
        "family_contributions": np.column_stack([
            result.family_contributions[name] for name in family_order
        ]),
        "feature_names": np.asarray(result.feature_names, dtype=str),
        "family_names": np.asarray(family_order, dtype=str),
        "family_index_json": np.asarray(json.dumps({k: list(v) for k, v in result.family_indices.items()}, sort_keys=True)),
        "aligned_bias": np.asarray(result.aligned_bias, dtype=np.float64),
        "orientation": np.asarray(result.orientation, dtype=np.int8),
        "risk_anchor_difference": np.asarray(result.risk_anchor_difference, dtype=np.float64),
        "transform_mean": np.asarray(transform.mean, dtype=np.float64),
        "transform_scale": np.asarray(transform.scale, dtype=np.float64),
        "transform_constant": np.asarray(transform.constant_mask, dtype=np.int8),
        "row_id": np.asarray(bundle["row_ids"], dtype=str),
        "source_contract_bundle_sha256": np.asarray(bundle["bundle_sha256"]),
        "source_bundle_sha256": np.asarray(bundle["source_bundle_sha256"]),
        "ordered_row_id_sha256": np.asarray(bundle["ordered_row_id_sha256"]),
        "run_definition_sha256": np.asarray(definition_hash),
        "history_loss": np.asarray([row["loss"] for row in result.objective_history], dtype=np.float64),
        "history_mala_acceptance": np.asarray([row["mala_acceptance"] for row in result.objective_history], dtype=np.float64),
    }
    arrays.update({f"state__{name}": np.asarray(value) for name, value in result.state.items()})
    npz_path = fit_dir / f"{stem}.npz"
    npz_hash = atomic_save_npz(npz_path, **arrays)
    metadata = {
        "schema": "deem_b3_feature_contract_v2_fit_metadata",
        "cell_id": cell_id,
        "dataset_family": bundle["dataset_family"],
        "seed": int(seed),
        "arm_id": "B3_V2",
        "npz_sha256": npz_hash,
        "run_definition_sha256": definition_hash,
        "source_contract_bundle_sha256": bundle["bundle_sha256"],
        "source_bundle_sha256": bundle["source_bundle_sha256"],
        "ordered_row_id_sha256": bundle["ordered_row_id_sha256"],
        "health": result.health,
        "orientation": int(result.orientation),
        "risk_anchor_difference": float(result.risk_anchor_difference),
        "labels_accessed_during_fit": False,
        "target_module_imported_during_fit": False,
    }
    metadata["content_sha256"] = canonical_sha256(metadata)
    json_path = fit_dir / f"{stem}.json"
    atomic_write_json(json_path, metadata)
    return {
        "cell_id": cell_id,
        "seed": int(seed),
        "npz": npz_path.relative_to(out_dir).as_posix(),
        "npz_sha256": npz_hash,
        "json": json_path.relative_to(out_dir).as_posix(),
        "json_sha256": sha256_file(json_path),
        "healthy": bool(result.health["healthy"]),
    }


def fit_cell(payload: tuple[str, str, str, dict[str, Any], str]) -> list[dict[str, Any]]:
    cell_id, contract_dir_raw, out_dir_raw, config, definition_hash = payload
    contract_dir, out_dir = Path(contract_dir_raw), Path(out_dir_raw)
    bundle = load_v2_bundle(contract_dir / "bundles" / f"{cell_id}.npz")
    if bundle["cell_id"] != cell_id:
        raise ValueError("cell/V2 bundle mismatch")
    X_risk, transform = prepare_v2_risk(bundle["X"], bundle["names"])
    continuous = ContinuousDeemConfig(**config["continuous_deem"])
    records = []
    for seed in config["seeds"]:
        result = fit_v2_b3(X_risk, bundle["names"], seed=int(seed), config=continuous)
        if not result.health["healthy"]:
            raise ValueError(f"unhealthy V2 B3 fit: {cell_id} seed {seed}")
        records.append(write_fit(out_dir, bundle, int(seed), result, transform, definition_hash))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/deem_b3_feature_contract_v2.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    parser.add_argument("--bundle-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/bundles")
    parser.add_argument("--contract-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/feature_contract_v2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "local_cache/deem_b3_moe_v1/b3_feature_contract_v2")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cells", default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    registry = load_registry(args.registry)
    contract_freeze = json.loads((args.contract_dir / "FREEZE.json").read_text(encoding="utf-8"))
    for relative, digest in contract_freeze["inventory"].items():
        if sha256_file(args.contract_dir / relative) != digest:
            raise ValueError(f"Feature Contract V2 freeze drift: {relative}")
    cells = [str(row["cell_id"]) for row in registry["cells"]]
    if args.cells != "all":
        requested = set(args.cells.split(","))
        cells = [cell for cell in cells if cell in requested]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    definition = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "arm_id": "B3_V2",
        "cells": cells,
        "seeds": config["seeds"],
        "source_hashes": source_hashes(args.config),
        "config_sha256": sha256_file(args.config),
        "registry_content_sha256": registry["registry_content_sha256"],
        "feature_contract_freeze_sha256": sha256_file(args.contract_dir / "FREEZE.json"),
        "feature_contract_inventory_sha256": contract_freeze["inventory_content_sha256"],
        "labels_accessed_during_fit": False,
        "target_module_imported_during_fit": False,
    }
    definition["content_sha256"] = canonical_sha256(definition)
    atomic_write_json(args.out_dir / "RUN_DEFINITION.json", definition)
    payloads = [
        (cell, str(args.contract_dir), str(args.out_dir), config, definition["content_sha256"])
        for cell in cells
    ]
    records = []
    if args.workers == 1:
        for payload in payloads:
            current = fit_cell(payload)
            records.extend(current)
            print(f"completed {payload[0]}: {len(current)} fits", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(fit_cell, payload): payload[0] for payload in payloads}
            for future in as_completed(futures):
                cell = futures[future]
                current = future.result()
                records.extend(current)
                print(f"completed {cell}: {len(current)} fits", flush=True)
    expected = len(cells) * len(config["seeds"])
    if len(records) != expected or not all(record["healthy"] for record in records):
        raise ValueError("fit roster/health incomplete")
    records.sort(key=lambda row: (row["cell_id"], row["seed"]))
    freeze = {
        "schema": SCHEMA + "_score_freeze",
        "run_definition_sha256": definition["content_sha256"],
        "expected_fit_artifacts": expected,
        "records": records,
        "labels_accessed_during_fit": False,
        "target_module_imported_during_fit": False,
    }
    freeze["content_sha256"] = canonical_sha256(freeze)
    atomic_write_json(args.out_dir / "SCORE_FREEZE.json", freeze)
    completion = {
        "schema": SCHEMA + "_fit_complete",
        "run_definition_sha256": definition["content_sha256"],
        "score_freeze_sha256": sha256_file(args.out_dir / "SCORE_FREEZE.json"),
        "fit_count": len(records),
        "cell_count": len(cells),
        "all_healthy": True,
        "labels_accessed_during_fit": False,
        "target_module_imported_during_fit": False,
    }
    completion["content_sha256"] = canonical_sha256(completion)
    atomic_write_json(args.out_dir / "FIT_COMPLETE.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
