#!/usr/bin/env python3
"""Fit CIW-DEEM on every runnable external completed-response cell.

This is a target-free extension of reconstruction benchmark v1.  It rebuilds
the CIW D1 input directly from raw telemetry, uses the benchmark's opaque row
and source-group identities, and freezes all scores before any label artifact
is opened by the separate evaluator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.ciw_deem import METHOD_ID, fit_ciw_deem
from spectral_utils.deem_b3_contract_ablation import prepare_arm
from spectral_utils.reconstruction_benchmark.external_final_answer import (
    apply_external_id_contract,
    canonicalize_external_identity_order,
    external_id_contract_binding,
    load_external_registry,
    load_identity_key,
    load_raw_feature_cell,
    resolve_sources,
    sealed_group_roster_commitment,
    verify_sources,
)
from spectral_utils.reconstruction_benchmark.external_fit_contract import (
    build_fit_row_identity_contract,
)
from spectral_utils.reconstruction_benchmark.io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    sha256_file,
)
from spectral_utils.residual_graph_deem import ContinuousDeemConfig


SCHEMA = "ciw-deem-external-score-freeze-v1"
SEEDS = (0, 1, 2, 3, 4)
SOURCE_FILES = (
    "configs/ciw_deem_v1.json",
    "configs/reconstruction_benchmark_v1/external_final_answer.json",
    "configs/reconstruction_benchmark_v1/populations.json",
    "spectral_utils/ciw_deem.py",
    "spectral_utils/deem_b3_contract_ablation.py",
    "spectral_utils/deem_b3_unsupervised_input_gate.py",
    "spectral_utils/reconstruction_benchmark/external_final_answer.py",
    "scripts/reconstruction_benchmark/run_ciw_external.py",
)


def _payload_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _fit_one_impl(args: tuple[str, str, str, str, str]) -> dict[str, Any]:
    repo_s, source_root_s, key_path_s, cell_id, out_root_s = args
    repo = Path(repo_s)
    source_root = Path(source_root_s)
    out_root = Path(out_root_s)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    registry = load_external_registry(
        repo=repo,
        registry_path="configs/reconstruction_benchmark_v1/external_final_answer.json",
        population_registry_path="configs/reconstruction_benchmark_v1/populations.json",
    )
    spec = registry.by_cell[cell_id]
    sources = resolve_sources(registry, spec, repo=source_root)
    verified = verify_sources(sources, include_labels=False)
    raw = load_raw_feature_cell(spec, sources)
    identity_key = load_identity_key(key_path_s)
    identity = apply_external_id_contract(
        registry, spec, raw.row_ids, raw.group_ids, identity_key=identity_key
    )
    ordered_matrix, identity = canonicalize_external_identity_order(
        raw.raw_matrix, identity
    )
    prepared = prepare_arm(ordered_matrix, raw.feature_names, "D1_TRANSFORM_ONLY")
    length_index = raw.feature_names.index("trace_length")
    raw_lengths = np.asarray(ordered_matrix[:, length_index], dtype=np.float64)

    scores: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    health: list[dict[str, Any]] = []
    reliability = None
    for seed in SEEDS:
        result, gate_map = fit_ciw_deem(
            prepared,
            identity.group_ids,
            raw_lengths,
            seed=seed,
            config=ContinuousDeemConfig(),
        )
        valid = bool(result.health.get("healthy", False))
        valid = valid and np.isfinite(result.score).all()
        valid = valid and float(result.health.get("posterior_sd", 0.0)) >= 1e-3
        valid = valid and float(result.health.get("reconstruction", np.inf)) <= 1e-8
        if not valid:
            raise RuntimeError(f"{cell_id}: unhealthy CIW seed {seed}: {result.health}")
        scores.append(np.asarray(result.score, dtype=np.float64))
        logits.append(np.asarray(result.logit, dtype=np.float64))
        health.append({"seed": seed, **dict(result.health)})
        current = np.asarray(gate_map.reliability, dtype=np.float64)
        if reliability is None:
            reliability = current
        elif not np.array_equal(reliability, current):
            raise RuntimeError(f"{cell_id}: target-free reliability changed across seeds")

    per_seed = np.stack(scores, axis=1)
    score = np.mean(per_seed, axis=1)
    rank_corrs = []
    from scipy.stats import spearmanr
    for i in range(len(SEEDS)):
        for j in range(i + 1, len(SEEDS)):
            rank_corrs.append(float(spearmanr(per_seed[:, i], per_seed[:, j]).statistic))
    if not np.isfinite(score).all() or min(rank_corrs) < 0.90:
        raise RuntimeError(f"{cell_id}: CIW ensemble stability failed")

    cell_dir = out_root / "cells" / cell_id
    cell_dir.mkdir(parents=True, exist_ok=False)
    score_path = cell_dir / "score.npz"
    score_sha = atomic_write_npz(score_path, {
        "row_ids": np.asarray(identity.row_ids, dtype="<U80"),
        "score": score.astype("<f8"),
        "per_seed_score": per_seed.astype("<f8"),
        "per_seed_logit": np.stack(logits, axis=1).astype("<f8"),
        "reliability": np.asarray(reliability, dtype="<f8"),
        "feature_names": np.asarray(prepared.feature_names, dtype="<U64"),
    })
    record = {
        "schema_version": "ciw-deem-external-cell-v1",
        "method_id": METHOD_ID,
        "cell_id": cell_id,
        "population_id": spec.population_id,
        "dataset_id": spec.dataset_id,
        "model_id": spec.model_id,
        "slice_id": spec.slice_id,
        "comparison_group_id": spec.comparison_group_id,
        "n_rows": len(score),
        "n_features": len(prepared.feature_names),
        "seeds": list(SEEDS),
        "score_semantics": "higher_is_incorrect",
        "input_contract": "CIW_D1_TRANSFORM_ONLY_from_raw_telemetry",
        "labels_opened_during_fit": False,
        "targets_accessed_during_fit": False,
        "source_files": verified,
        "identity_contract": dict(identity.contract_binding),
        "row_namespace_sha256": identity.row_namespace_sha256,
        "group_namespace_sha256": identity.group_namespace_sha256,
        "sealed_group_roster_commitment_sha256": sealed_group_roster_commitment(identity),
        "median_seed_spearman": float(np.median(rank_corrs)),
        "health": health,
        "score_path": "score.npz",
        "score_sha256": score_sha,
    }
    record["payload_sha256"] = _payload_sha(record)
    record_path = cell_dir / "RECORD.json"
    atomic_write_json(record_path, record)
    return {
        "status": "COMPLETE",
        "cell_id": cell_id,
        "record_path": record_path.relative_to(out_root).as_posix(),
        "record_sha256": sha256_file(record_path),
        "score_sha256": score_sha,
        "n_rows": len(score),
    }


def _fit_one(args: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """Process-safe wrapper: never try to pickle a custom benchmark exception."""
    try:
        return _fit_one_impl(args)
    except Exception as exc:
        return {
            "status": "BLOCKED",
            "cell_id": args[3],
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def _load_completed(out: Path, cell_id: str) -> dict[str, Any] | None:
    cell_dir = out / "cells" / cell_id
    record_path = cell_dir / "RECORD.json"
    score_path = cell_dir / "score.npz"
    if not record_path.exists() and not score_path.exists():
        return None
    if not record_path.is_file() or not score_path.is_file():
        raise RuntimeError(f"{cell_id}: incomplete existing artifact pair")
    record = json.loads(record_path.read_text())
    claimed_payload = record.get("payload_sha256")
    payload = dict(record)
    payload.pop("payload_sha256", None)
    if claimed_payload != _payload_sha(payload):
        raise RuntimeError(f"{cell_id}: existing RECORD payload hash mismatch")
    if record.get("method_id") != METHOD_ID or record.get("cell_id") != cell_id:
        raise RuntimeError(f"{cell_id}: existing RECORD identity mismatch")
    score_sha = sha256_file(score_path)
    if score_sha != record.get("score_sha256"):
        raise RuntimeError(f"{cell_id}: existing score hash mismatch")
    with np.load(score_path, allow_pickle=False) as data:
        score = np.asarray(data["score"], dtype=np.float64)
        row_ids = np.asarray(data["row_ids"]).astype(str)
    if len(score) != int(record.get("n_rows", -1)) or len(row_ids) != len(score):
        raise RuntimeError(f"{cell_id}: existing score shape mismatch")
    if not np.isfinite(score).all() or len(set(row_ids.tolist())) != len(row_ids):
        raise RuntimeError(f"{cell_id}: invalid existing scores/row IDs")
    return {
        "status": "COMPLETE",
        "cell_id": cell_id,
        "record_path": record_path.relative_to(out).as_posix(),
        "record_sha256": sha256_file(record_path),
        "score_sha256": score_sha,
        "n_rows": len(score),
        "resumed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--identity-key", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cells", default="all")
    parser.add_argument("--eligibility-manifest")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    repo = ROOT.resolve()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    registry = load_external_registry(
        repo=repo,
        registry_path="configs/reconstruction_benchmark_v1/external_final_answer.json",
        population_registry_path="configs/reconstruction_benchmark_v1/populations.json",
    )
    runnable = [cell.cell_id for cell in registry.cells if cell.fit_policy == "run_if_compatible"]
    eligibility_binding = None
    if args.eligibility_manifest:
        eligibility_path = Path(args.eligibility_manifest).resolve()
        eligibility = json.loads(eligibility_path.read_text())
        if eligibility.get("external_registry_sha256") != registry.sha256:
            raise RuntimeError("eligibility manifest registry hash mismatch")
        eligible = {
            row["cell_id"] for row in eligibility.get("cells", [])
            if row.get("status") == "ELIGIBLE"
        }
        if int(eligibility.get("n_prepared_cells", -1)) != len(eligible):
            raise RuntimeError("eligibility manifest count mismatch")
        runnable = [cell_id for cell_id in runnable if cell_id in eligible]
        eligibility_binding = {
            "path": str(eligibility_path),
            "sha256": sha256_file(eligibility_path),
            "n_eligible": len(eligible),
        }
    if args.cells != "all":
        requested = [item.strip() for item in args.cells.split(",") if item.strip()]
        unknown = sorted(set(requested) - set(runnable))
        if unknown:
            raise KeyError(f"unknown/non-runnable cells: {unknown}")
        runnable = requested
    key = load_identity_key(args.identity_key)
    fit_identity = build_fit_row_identity_contract(
        external_id_contract_binding(registry, identity_key=key), identity_key=key
    )
    rows: list[dict[str, Any]] = []
    pending: list[str] = []
    for cell in runnable:
        completed = _load_completed(out, cell) if args.resume else None
        if completed is None:
            pending.append(cell)
        else:
            rows.append(completed)
            print(f"resumed {cell} ({len(rows)}/{len(runnable)})", flush=True)
    tasks = [
        (str(repo), str(Path(args.source_root).resolve()), str(Path(args.identity_key).resolve()), cell, str(out))
        for cell in pending
    ]
    blocked: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(_fit_one, task): task[3] for task in tasks}
        for future in as_completed(futures):
            row = future.result()
            if row.get("status") == "COMPLETE":
                rows.append(row)
                print(f"completed {row['cell_id']} ({len(rows)}/{len(runnable)})", flush=True)
            else:
                blocked.append(row)
                print(f"blocked {row['cell_id']}: {row['error_type']}: {row['message']}", flush=True)
    if blocked:
        atomic_write_json(out / "BLOCKED.json", {"cells": blocked})
        raise RuntimeError(f"{len(blocked)} eligible CIW cells were blocked")
    rows.sort(key=lambda item: item["cell_id"])
    snapshot = [
        {"path": path, "sha256": sha256_file(repo / path)} for path in SOURCE_FILES
    ]
    freeze = {
        "schema_version": SCHEMA,
        "method_id": METHOD_ID,
        "scientific_run": args.cells == "all",
        "all_expected_scores_present": len(rows) == len(runnable),
        "labels_opened_by_fit": False,
        "runtime_labels_used": False,
        "targets_accessed_during_fit": False,
        "external_registry_sha256": registry.sha256,
        "population_registry_sha256": registry.population_registry_sha256,
        "id_contract_version": fit_identity["version"],
        "identity_contract": fit_identity,
        "seeds": list(SEEDS),
        "n_cells": len(rows),
        "cells": rows,
        "eligibility_manifest": eligibility_binding,
        "source_snapshot": snapshot,
    }
    freeze["payload_sha256"] = _payload_sha(freeze)
    atomic_write_json(out / "SCORE_FREEZE_MANIFEST.json", freeze)
    print(json.dumps({"status": "PASS", "cells": len(rows), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
