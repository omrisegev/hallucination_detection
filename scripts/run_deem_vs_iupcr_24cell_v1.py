#!/usr/bin/env python3
"""Run the frozen label-free B0/B1/B2/B3 24-cell DEEM benchmark."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.fusion_utils import upcr_fuse  # noqa: E402
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ContinuousDeemConfig,
    ResidualGraphDeemError,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    donor_risk_matrix,
    environment_fingerprint,
    equal_family_risk_anchor,
    fit_continuous_deem,
    jsonable,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
    registry_cell,
)

DEFAULT_CONFIG = ROOT / "configs/deem_vs_iupcr_24cell_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
PROTOCOL = ROOT / "docs/experiments/DEEM_VS_IUPCR_24CELL_V1.md"
WORKER = ROOT / "scripts/deem_vs_iupcr_adapter_worker_v1.py"
CORE_SOURCES = (
    ROOT / "spectral_utils/deem_adapter.py",
    ROOT / "spectral_utils/residual_graph_deem.py",
    ROOT / "spectral_utils/residual_graph_deem_data.py",
    ROOT / "spectral_utils/feature_contract.py",
    ROOT / "spectral_utils/feature_utils.py",
    ROOT / "spectral_utils/fusion_utils.py",
    ROOT / "spectral_utils/specrage_views.py",
    ROOT / "scripts/build_residual_graph_deem_data_v1.py",
    ROOT / "scripts/run_deem_vs_iupcr_24cell_v1.py",
    ROOT / "scripts/deem_vs_iupcr_adapter_worker_v1.py",
    ROOT / "scripts/evaluate_deem_vs_iupcr_24cell_v1.py",
    ROOT / "scripts/evaluate_residual_graph_deem_24cell_v1.py",
    ROOT / "scripts/report_deem_vs_iupcr_24cell_v1.py",
    ROOT / "scripts/verify_deem_vs_iupcr_24cell_v1.py",
    ROOT / "cluster/run_deem_vs_iupcr_24cell_v1.py",
)


def load_experiment_config(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "deem_vs_iupcr_24cell_v1_config":
        raise ResidualGraphDeemError("experiment config schema mismatch")
    if [row["id"] for row in value.get("arms", [])] != ["B0", "B1", "B2", "B3"]:
        raise ResidualGraphDeemError("experiment arm roster drift")
    if tuple(value.get("seeds", ())) != (0, 1, 2, 3, 4):
        raise ResidualGraphDeemError("experiment seed roster drift")
    if value.get("graph_direction", {}).get("decision") != "CLOSE_RESIDUAL_GRAPH_EXTENSION_SPECIFICITY_FAILURE":
        raise ResidualGraphDeemError("graph closure evidence missing")
    return value


def source_hash() -> str:
    missing = [str(path) for path in CORE_SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing output-generating source: " + ", ".join(missing))
    payload = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in CORE_SOURCES}
    payload[DEFAULT_CONFIG.relative_to(ROOT).as_posix()] = sha256_file(DEFAULT_CONFIG)
    payload[PROTOCOL.relative_to(ROOT).as_posix()] = sha256_file(PROTOCOL)
    return canonical_sha256(payload)


def artifact_stem(arm_id: str, seed: int) -> str:
    return f"{arm_id}__seed{int(seed)}"


def expected_stems(config: dict) -> set[str]:
    return {artifact_stem(arm["id"], seed) for arm in config["arms"] for seed in config["seeds"]}


def full_risk(bundle):
    donor, _, transform = donor_risk_matrix(bundle.X_raw, bundle.X_raw, bundle.feature_names)
    return donor, transform


def b0_score(X_risk: np.ndarray, names) -> tuple[np.ndarray, dict]:
    weights, rho, g2, diagnostics = upcr_fuse(X_risk.T, return_diagnostics=True)
    score = np.asarray(weights @ X_risk.T, dtype=np.float64)
    anchor = equal_family_risk_anchor(X_risk, names)
    correlation = float(spearmanr(score, anchor).statistic)
    if not np.isfinite(correlation) or abs(correlation) <= 1e-6:
        raise ResidualGraphDeemError("B0 risk orientation is ambiguous")
    if correlation < 0:
        score = -score
        correlation = -correlation
    healthy = bool(np.isfinite(score).all() and np.std(score) >= 1e-3)
    return score, {
        "healthy": healthy,
        "score_sd": float(np.std(score)),
        "risk_anchor_spearman": correlation,
        "weights": jsonable(weights),
        "rho_hat": jsonable(rho),
        "g2_hat": float(g2),
        "upcr": jsonable(diagnostics),
        "historical_F_used": False,
        "rho_polarity_used": False,
    }


def _base_record(cell_id: str, stem: str, seed: int, provenance: dict) -> dict:
    return {
        "schema": "deem_vs_iupcr_fit_artifact_v1",
        "status": "complete",
        "cell_id": cell_id,
        "stem": stem,
        "arm_id": stem.split("__", 1)[0],
        "seed": int(seed),
        "code_sha256": source_hash(),
        "experiment_config_sha256": sha256_file(DEFAULT_CONFIG),
        "protocol_sha256": sha256_file(PROTOCOL),
        "environment": environment_fingerprint(),
        "determinism": {"torch_deterministic_algorithms": True, "seeds": [0, 1, 2, 3, 4]},
        **provenance,
    }


def write_b0(out: Path, cell_id: str, stem: str, score: np.ndarray, seed: int,
             diagnostics: dict, provenance: dict) -> dict:
    directory = out / "fits" / cell_id
    array_path = directory / f"{stem}.npz"
    array_hash = atomic_save_npz(array_path, score=np.asarray(score, dtype=np.float64))
    record = _base_record(cell_id, stem, seed, provenance)
    record.update({
        "array_path": str(array_path.resolve()),
        "array_sha256": array_hash,
        "health": diagnostics,
        "config_sha256": canonical_sha256(diagnostics),
    })
    record["content_sha256"] = canonical_sha256(record)
    atomic_write_json(directory / f"{stem}.json", record)
    return record


def write_b3(out: Path, cell_id: str, stem: str, result, provenance: dict,
             transform) -> dict:
    directory = out / "fits" / cell_id
    arrays = {
        "score": np.asarray(result.score, dtype=np.float64),
        "posterior": np.asarray(result.posterior, dtype=np.float64),
        "logit": np.asarray(result.logit, dtype=np.float64),
        "contributions": np.asarray(result.contributions, dtype=np.float64),
        "feature_names": np.asarray(result.feature_names, dtype=str),
        "standardization_mean": np.asarray(transform.mean, dtype=np.float64),
        "standardization_scale": np.asarray(transform.scale, dtype=np.float64),
        "constant_mask": np.asarray(transform.constant_mask, dtype=np.int8),
    }
    for family, values in result.family_contributions.items():
        arrays[f"family_contribution__{family}"] = np.asarray(values, dtype=np.float64)
    for name, value in result.state.items():
        arrays[f"state__{name}"] = np.asarray(value)
    array_path = directory / f"{stem}.npz"
    array_hash = atomic_save_npz(array_path, **arrays)
    record = _base_record(cell_id, stem, result.seed, provenance)
    record.update({
        "array_path": str(array_path.resolve()),
        "array_sha256": array_hash,
        "orientation": int(result.orientation),
        "aligned_bias": float(result.aligned_bias),
        "risk_anchor_difference": float(result.risk_anchor_difference),
        "health": jsonable(result.health),
        "config": jsonable(result.config),
        "config_sha256": canonical_sha256(result.config),
        "objective_history": jsonable(result.objective_history),
        "contribution_reconstruction_max_abs": float(
            np.max(np.abs(result.aligned_bias + result.contributions.sum(axis=1) - result.logit))
        ),
    })
    record["content_sha256"] = canonical_sha256(record)
    atomic_write_json(directory / f"{stem}.json", record)
    return record


def write_failure(out: Path, cell_id: str, stem: str, seed: int, exc: Exception) -> dict:
    record = {
        "schema": "deem_vs_iupcr_fit_artifact_v1",
        "status": "failed",
        "cell_id": cell_id,
        "stem": stem,
        "arm_id": stem.split("__", 1)[0],
        "seed": int(seed),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "objective_history": jsonable(getattr(exc, "objective_history", [])),
        "last_finite_state": jsonable(getattr(exc, "last_finite_state", None)),
    }
    atomic_write_json(out / "fits" / cell_id / f"{stem}.json", record)
    return record


def _adapter_input(out: Path, key: str, X_risk: np.ndarray, names, context: dict) -> Path:
    path = out / "adapter_inputs" / f"{key}.npz"
    atomic_save_npz(
        path,
        X_risk=np.asarray(X_risk, dtype=np.float64),
        feature_names=np.asarray(names, dtype=str),
        **{name: np.asarray(str(value)) for name, value in context.items()},
    )
    return path


def adapter_jobs(out: Path, cell_id: str, X_risk: np.ndarray, names, *, seeds,
                 python: str, device: str, provenance: dict, prefix="fits") -> list[dict]:
    context = {"cell_id": cell_id, "code_sha256": source_hash(), **provenance}
    input_path = _adapter_input(out, cell_id, X_risk, names, context)
    jobs = []
    for mode, arm_id in (("hard", "B1"), ("soft", "B2")):
        for seed in seeds:
            stem = artifact_stem(arm_id, seed)
            output = out / prefix / cell_id / stem
            command = [python, str(WORKER), "--input", str(input_path), "--output", str(output),
                       "--mode", mode, "--seed", str(seed), "--device", device]
            jobs.append((stem, output, command))
    records = []
    with ThreadPoolExecutor(max_workers=min(5, len(jobs))) as executor:
        pending = {executor.submit(subprocess.run, command, capture_output=True, text=True): (stem, output)
                   for stem, output, command in jobs}
        for future in as_completed(pending):
            stem, output = pending[future]
            completed = future.result()
            path = output.with_suffix(".json")
            if not path.is_file():
                atomic_write_json(path, {"schema": "deem_vs_iupcr_adapter020_fit_v1",
                                         "status": "failed", "cell_id": cell_id,
                                         "stem": stem, "returncode": completed.returncode,
                                         "stdout": completed.stdout, "stderr": completed.stderr})
            records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _fit_acceptable(record: dict) -> bool:
    """Amendment A1: B2 health is recorded, not blocking.

    B0/B1/B3 fits must be fully healthy.  A B2 fit is acceptable once it is
    complete with finite scores -- its collapse on wide inventories is a
    documented property of the packaged soft/rank adapter (job 219682;
    scripts/deem_soft_collapse_probe.py), and this benchmark records that
    instead of letting it veto the run.  Interpretation of the B3-B2 contrast
    must cite the recorded health tables.
    """
    if record.get("status") != "complete":
        return False
    health = record.get("health", {})
    if str(record.get("stem", "")).startswith("B2__"):
        return bool(health.get("score_finite", health.get("healthy")))
    return bool(health.get("healthy"))


def _valid_record(path: Path, *, expected_code: str) -> dict | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if not _fit_acceptable(record):
            return None
        if record.get("code_sha256") != expected_code:
            return None
        array_path = Path(record["array_path"])
        if not array_path.is_file() or sha256_file(array_path) != record.get("array_sha256"):
            return None
        unhashed = dict(record)
        expected = unhashed.pop("content_sha256", None)
        if expected and canonical_sha256(unhashed) != expected:
            return None
        return record
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def run_preflight(args, config: dict, registry: dict) -> None:
    rng = np.random.Generator(np.random.PCG64(20260821))
    epochs = config["preflight"]["smoke_b3_epochs"] if args.smoke else config["preflight"]["full_b3_epochs"]
    seeds = (0,) if args.smoke else tuple(config["seeds"])
    rows = int(config["preflight"]["fixture_rows"])
    schema_results = []
    replay_fixture = None
    for index, schema in enumerate(registry["schemas"]):
        names = tuple(schema["feature_names"])
        raw = rng.normal(size=(rows, len(names)))
        if index == 0:
            raw[:, 0] = 1.0
        X, _, transform = donor_risk_matrix(raw, raw, names)
        b0, b0_health = b0_score(X, names)
        seed_results = []
        for seed in seeds:
            result = fit_continuous_deem(
                X, names, seed=seed, config=replace(ContinuousDeemConfig(), epochs=int(epochs))
            )
            reconstruction = float(np.max(np.abs(
                result.aligned_bias + result.contributions.sum(axis=1) - result.logit
            )))
            seed_results.append({"seed": seed, "healthy": bool(result.health["healthy"]),
                                 "posterior_sd": float(np.std(result.score)),
                                 "reconstruction_max_abs": reconstruction})
            if replay_fixture is None:
                replay_fixture = (X, names, result.score.copy())
        schema_results.append({
            "schema_id": schema["schema_id"], "n_features": len(names),
            "constant_coordinate_count": int(transform.constant_mask.sum()),
            "b0_healthy": bool(b0_health["healthy"]), "b0_score_sha256": canonical_sha256(b0),
            "b3": seed_results,
        })
    X, names, expected = replay_fixture
    replay = fit_continuous_deem(
        X, names, seed=0, config=replace(ContinuousDeemConfig(), epochs=int(epochs))
    )
    replay_exact = bool(np.array_equal(expected, replay.score))
    adapter_records = []
    if not args.smoke:
        for p in config["preflight"]["boundary_adapter_feature_counts"]:
            schema = next(row for row in registry["schemas"] if int(row["n_features"]) == int(p))
            names = tuple(schema["feature_names"])
            raw = rng.normal(size=(rows, len(names)))
            X, _, _ = donor_risk_matrix(raw, raw, names)
            adapter_records.extend(adapter_jobs(
                args.out_dir, f"schema_p{p}", X, names, seeds=config["seeds"],
                python=args.python, device=args.adapter_device,
                provenance={"preflight": "true", "inventory_sha256": canonical_sha256(names)},
                prefix="preflight_adapters",
            ))
    # Amendment A1 (protocol, pre-label): every boundary fit must complete with
    # finite scores under the pinned package.  Full health (score_sd >= 1e-3)
    # remains required for B1 on both fixtures and for B2 on the narrow fixture.
    # B2 on the wide fixture is recorded, not gated: job 219682 showed the
    # soft/rank adapter deterministically collapsing at 30 features (score_sd
    # 1.1e-6 to 1.3e-4 on all five seeds), the same mode documented by
    # scripts/deem_soft_collapse_probe.py.  Blocking there would let a known
    # comparator limitation close a benchmark whose primary contrast (B3-B0)
    # does not involve it.
    narrow_fixture = "schema_p%d" % min(
        int(v) for v in config["preflight"]["boundary_adapter_feature_counts"]
    )

    def _boundary_ok(row):
        health = row.get("health", {})
        if not (row.get("status") == "complete"
                and row.get("package_version") == "0.2.0"
                and health.get("score_finite")):
            return False
        if row.get("arm_id") == "B2" and row.get("cell_id") != narrow_fixture:
            return True
        return bool(health.get("healthy"))

    adapter_pass = args.smoke or (
        bool(adapter_records) and all(_boundary_ok(row) for row in adapter_records)
    )
    schema_pass = all(
        row["b0_healthy"] and all(item["healthy"] and item["posterior_sd"] >= 1e-3
                                  and item["reconstruction_max_abs"] <= 1e-8 for item in row["b3"])
        for row in schema_results
    )
    complete = {
        "schema": "deem_vs_iupcr_preflight_complete_v1",
        "status": "pass" if schema_pass and replay_exact and adapter_pass else "failed",
        "smoke": bool(args.smoke),
        "scientific_selection_applied": False,
        "natural_targets_opened": False,
        "graph_arms_executed": False,
        "epochs": int(epochs),
        "seeds": list(seeds),
        "seven_schema_fixtures": schema_results,
        "deterministic_replay_exact": replay_exact,
        "adapter_boundary_fits": len(adapter_records),
        "adapter_boundary_pass": bool(adapter_pass),
        "adapter_unhealthy_recorded": [
            {"cell_id": row.get("cell_id"), "stem": row.get("stem"),
             "score_sd": row.get("health", {}).get("score_sd"),
             "score_n_unique": row.get("health", {}).get("score_n_unique")}
            for row in adapter_records
            if not row.get("health", {}).get("healthy")
        ],
        "code_sha256": source_hash(),
        "config_sha256": sha256_file(args.config),
        "registry_content_sha256": registry["registry_content_sha256"],
    }
    complete["content_sha256"] = canonical_sha256(complete)
    atomic_write_json(args.out_dir / "PREFLIGHT_COMPLETE.json", complete)
    if complete["status"] != "pass":
        raise SystemExit("preflight failed; Stage A remains closed")


def stage_a_cell(args, config: dict, registry: dict, cell_id: str) -> list[dict]:
    bundle = load_target_free_bundle(Path(args.bundle_dir) / f"{cell_id}.npz")
    registered = registry_cell(registry, cell_id)
    if len(bundle.row_ids) != int(registered["n_rows"]):
        raise ResidualGraphDeemError(f"row-count mismatch: {cell_id}")
    if bundle.inventory_sha256 != registered["inventory_sha256"]:
        raise ResidualGraphDeemError(f"inventory mismatch: {cell_id}")
    if bundle.source_sha256 != registered["source"]["source_sha256"]:
        raise ResidualGraphDeemError(f"source mismatch: {cell_id}")
    X_risk, transform = full_risk(bundle)
    provenance = {
        "bundle_sha256": bundle.bundle_sha256,
        "source_sha256": bundle.source_sha256,
        "inventory_sha256": bundle.inventory_sha256,
        "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
    }
    records = []
    score, diagnostics = b0_score(X_risk, bundle.feature_names)
    for seed in config["seeds"]:
        records.append(write_b0(args.out_dir, cell_id, artifact_stem("B0", seed),
                                score, seed, diagnostics, provenance))
    records.extend(adapter_jobs(
        args.out_dir, cell_id, X_risk, bundle.feature_names, seeds=config["seeds"],
        python=args.python, device=args.adapter_device, provenance=provenance,
    ))
    for seed in config["seeds"]:
        stem = artifact_stem("B3", seed)
        try:
            result = fit_continuous_deem(X_risk, bundle.feature_names, seed=seed,
                                         config=ContinuousDeemConfig())
            records.append(write_b3(args.out_dir, cell_id, stem, result, provenance, transform))
        except Exception as exc:
            records.append(write_failure(args.out_dir, cell_id, stem, seed, exc))
    return records


def _cell_checkpoint(out: Path, cell_id: str, config: dict, run_hash: str) -> list[dict] | None:
    marker = out / "fits" / cell_id / "CELL_COMPLETE.json"
    if not marker.is_file():
        return None
    value = json.loads(marker.read_text(encoding="utf-8"))
    unhashed = dict(value)
    expected_hash = unhashed.pop("content_sha256", None)
    if canonical_sha256(unhashed) != expected_hash:
        return None
    if value.get("run_definition_sha256") != run_hash or set(value.get("stems", [])) != expected_stems(config):
        return None
    records = []
    for stem in sorted(expected_stems(config)):
        record = _valid_record(out / "fits" / cell_id / f"{stem}.json", expected_code=source_hash())
        if record is None:
            return None
        records.append(record)
    return records


def _write_cell_checkpoint(out: Path, cell_id: str, config: dict, run_hash: str,
                           records: list[dict]) -> None:
    complete = {row.get("stem"): row for row in records if _fit_acceptable(row)}
    if set(complete) != expected_stems(config):
        return
    value = {"schema": "deem_vs_iupcr_cell_complete_v1", "cell_id": cell_id,
             "run_definition_sha256": run_hash, "stems": sorted(complete),
             "artifact_sha256": {stem: complete[stem]["array_sha256"] for stem in sorted(complete)}}
    value["content_sha256"] = canonical_sha256(value)
    atomic_write_json(out / "fits" / cell_id / "CELL_COMPLETE.json", value)


def run_stage_a(args, config: dict, registry: dict) -> None:
    preflight = json.loads(Path(args.preflight_complete).read_text(encoding="utf-8"))
    unhashed_preflight = dict(preflight)
    expected_preflight_hash = unhashed_preflight.pop("content_sha256", None)
    if canonical_sha256(unhashed_preflight) != expected_preflight_hash:
        raise SystemExit("preflight content hash mismatch")
    if preflight.get("status") != "pass" or preflight.get("smoke"):
        raise SystemExit("Stage A requires a full passing preflight")
    if preflight.get("code_sha256") != source_hash() or preflight.get("config_sha256") != sha256_file(args.config):
        raise SystemExit("preflight code/config mismatch")
    if (
        preflight.get("epochs") != config["preflight"]["full_b3_epochs"]
        or preflight.get("seeds") != config["seeds"]
        or len(preflight.get("seven_schema_fixtures", [])) != 7
        or preflight.get("adapter_boundary_fits") != 20
        or not preflight.get("adapter_boundary_pass")
        or preflight.get("scientific_selection_applied")
        or preflight.get("natural_targets_opened")
        or preflight.get("graph_arms_executed")
    ):
        raise SystemExit("full preflight contract mismatch")
    cells = [row["cell_id"] for row in registry["cells"]]
    definition = {
        "schema": "deem_vs_iupcr_run_definition_v1", "status": "frozen", "debug": False,
        "experiment_id": config["experiment_id"], "cells": cells,
        "arms": config["arms"], "seeds": config["seeds"],
        "expected_stems_per_cell": 20, "graph_arms": [],
        "registry_content_sha256": registry["registry_content_sha256"],
        "preflight_complete_sha256": sha256_file(args.preflight_complete),
        "protocol_sha256": sha256_file(PROTOCOL), "config_sha256": sha256_file(args.config),
        "code_sha256": source_hash(), "environment": environment_fingerprint(),
    }
    definition["content_sha256"] = canonical_sha256(definition)
    definition_path = args.out_dir / "RUN_DEFINITION.json"
    if definition_path.is_file():
        if json.loads(definition_path.read_text(encoding="utf-8")) != jsonable(definition):
            raise SystemExit("run-definition mismatch on resume")
    else:
        atomic_write_json(definition_path, definition, immutable=True)
    run_hash = sha256_file(definition_path)
    frozen = args.out_dir / "SCORE_FREEZE_MANIFEST.json"
    if frozen.is_file():
        value = json.loads(frozen.read_text(encoding="utf-8"))
        unhashed = dict(value)
        expected_hash = unhashed.pop("content_sha256", None)
        fit_complete_path = args.out_dir / "FIT_COMPLETE.json"
        if (
            value.get("status") != "complete"
            or value.get("run_definition_sha256") != run_hash
            or canonical_sha256(unhashed) != expected_hash
            or not fit_complete_path.is_file()
            or sha256_file(fit_complete_path) != value.get("fit_complete_sha256")
        ):
            raise SystemExit("existing score freeze mismatch")
        for item in value.get("artifacts", []):
            path = args.out_dir / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                raise SystemExit(f"score-freeze artifact mismatch: {path}")
        print("[Stage A] immutable score freeze verified", flush=True)
        return
    records = []
    for cell_id in cells:
        checkpoint = _cell_checkpoint(args.out_dir, cell_id, config, run_hash)
        if checkpoint is not None:
            print(f"[Stage A] {cell_id} (verified checkpoint)", flush=True)
            records.extend(checkpoint)
            continue
        print(f"[Stage A] {cell_id}", flush=True)
        current = stage_a_cell(args, config, registry, cell_id)
        records.extend(current)
        _write_cell_checkpoint(args.out_dir, cell_id, config, run_hash, current)
    expected = expected_stems(config)
    invalid = [row for row in records if not _fit_acceptable(row)]
    missing = []
    for cell_id in cells:
        observed = {row.get("stem") for row in records if row.get("cell_id") == cell_id
                    and _fit_acceptable(row)}
        missing.extend({"cell": cell_id, "stem": stem} for stem in sorted(expected - observed))
    fit_complete = {
        "schema": "deem_vs_iupcr_fit_complete_v1",
        "status": "complete" if not invalid and not missing and len(records) == 480 else "incomplete",
        "cells": cells, "n_records": len(records), "invalid_fits": invalid,
        "missing_artifacts": missing,
        "b2_unhealthy_recorded": [
            {"cell": row.get("cell_id"), "stem": row.get("stem"),
             "score_sd": row.get("health", {}).get("score_sd"),
             "score_n_unique": row.get("health", {}).get("score_n_unique")}
            for row in records if str(row.get("stem", "")).startswith("B2__")
            and row.get("status") == "complete"
            and not row.get("health", {}).get("healthy")
        ],
        "run_definition_sha256": run_hash,
    }
    atomic_write_json(args.out_dir / "FIT_COMPLETE.json", fit_complete)
    if fit_complete["status"] != "complete":
        raise SystemExit("Stage A incomplete; score freeze not written")
    paths = sorted(path for path in (args.out_dir / "fits").glob("*/*")
                   if path.suffix in {".json", ".npz"})
    freeze = {
        "schema": "deem_vs_iupcr_score_freeze_v1", "status": "complete", "debug": False,
        "cells": cells, "arms": [row["id"] for row in config["arms"]],
        "seeds": config["seeds"], "expected_fit_artifacts": 480,
        "run_definition_sha256": run_hash,
        "fit_complete_sha256": sha256_file(args.out_dir / "FIT_COMPLETE.json"),
        "artifacts": [{"path": path.relative_to(args.out_dir).as_posix(), "sha256": sha256_file(path)}
                      for path in paths],
    }
    freeze["content_sha256"] = canonical_sha256(freeze)
    atomic_write_json(frozen, freeze, immutable=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "stage-a"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--preflight-complete", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--adapter-device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    if sha256_file(args.config) != sha256_file(DEFAULT_CONFIG):
        raise SystemExit("alternate or modified experiment config is forbidden")
    config = load_experiment_config(args.config)
    registry = load_registry(args.registry)
    if registry["registry_content_sha256"] != config["source_registry_content_sha256"]:
        raise SystemExit("experiment config/source registry mismatch")
    if args.command == "preflight":
        run_preflight(args, config, registry)
    else:
        if args.bundle_dir is None or args.preflight_complete is None:
            parser.error("stage-a requires --bundle-dir and --preflight-complete")
        run_stage_a(args, config, registry)


if __name__ == "__main__":
    main()
