#!/usr/bin/env python3
"""Run frozen Phase 0 or label-free Stage A for Residual-Graph DEEM v1.

This module intentionally has no evaluation-sidecar argument and never imports
``residual_graph_deem_labels``.  Natural target access belongs exclusively to the
separate evaluator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.fusion_utils import upcr_fuse  # noqa: E402
from spectral_utils.feature_contract import confidence_sign_vector  # noqa: E402
from spectral_utils.graph_topology import self_safe_knn_graph  # noqa: E402
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ARM_SPECS,
    LAMBDA_GRID,
    SEEDS,
    ContinuousDeemConfig,
    DufsConfig,
    GraphDeemConfig,
    ResidualGraphDeemError,
    atomic_save_npz,
    atomic_write_json,
    build_inventory_graph,
    canonical_sha256,
    cross_view_dufs,
    crossfit_continuous_deem,
    donor_risk_matrix,
    equal_family_risk_anchor,
    environment_fingerprint,
    family_index_map,
    fit_continuous_deem,
    fold_artifact_diagnostics,
    graph_health,
    jsonable,
    permute_graph_nodes,
    present_family_laplacian,
    random_gate_control,
    row_id_tie_keys,
    sha256_file,
    symmetric_normalized_laplacian,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
    registry_cell,
)


DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
WORKER = ROOT / "scripts/residual_graph_deem_adapter_worker_v1.py"
CORE_SOURCES = (
    ROOT / "spectral_utils/residual_graph_deem.py",
    ROOT / "spectral_utils/residual_graph_deem_data.py",
    ROOT / "spectral_utils/residual_graph_deem_labels.py",
    ROOT / "spectral_utils/deem_adapter.py",
    ROOT / "scripts/run_residual_graph_deem_24cell_v1.py",
    ROOT / "scripts/residual_graph_deem_adapter_worker_v1.py",
    ROOT / "scripts/build_residual_graph_deem_data_v1.py",
    ROOT / "scripts/evaluate_residual_graph_deem_24cell_v1.py",
    ROOT / "scripts/report_residual_graph_deem_24cell_v1.py",
    ROOT / "scripts/plot_residual_graph_deem_24cell_v1.py",
    ROOT / "scripts/verify_residual_graph_deem_24cell_v1.py",
    ROOT / "setup.py",
)


def source_hash() -> str:
    missing = [str(path) for path in CORE_SOURCES if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing output-generating source: " + ", ".join(missing))
    return canonical_sha256({str(path.relative_to(ROOT)): sha256_file(path) for path in CORE_SOURCES})


def arm_name(arm_id: str) -> str:
    return next(spec.name for spec in ARM_SPECS if spec.arm_id == arm_id)


def artifact_stem(arm_id: str, seed: int, lambda_: float | None = None, control: str | None = None) -> str:
    parts = [arm_id]
    if lambda_ is not None:
        parts.append("lambda" + str(lambda_).replace(".", "p"))
    if control:
        parts.append(control)
    parts.append(f"seed{int(seed)}")
    return "__".join(parts)


def csr_arrays(graph) -> dict[str, np.ndarray]:
    W = sparse.csr_matrix(graph)
    return {
        "graph_data": W.data,
        "graph_indices": W.indices.astype(np.int64),
        "graph_indptr": W.indptr.astype(np.int64),
        "graph_shape": np.asarray(W.shape, dtype=np.int64),
    }


def write_fit_result(
    root: Path,
    cell_id: str,
    stem: str,
    result,
    *,
    extras: dict | None = None,
    graph=None,
    provenance: dict | None = None,
) -> dict:
    directory = root / "fits" / cell_id
    arrays = {
        "score": np.asarray(result.score, dtype=np.float64),
        "posterior": np.asarray(result.posterior, dtype=np.float64),
        "logit": np.asarray(result.logit, dtype=np.float64),
        "contributions": np.asarray(result.contributions, dtype=np.float64),
        "feature_names": np.asarray(result.feature_names, dtype=str),
    }
    for family, values in result.family_contributions.items():
        arrays[f"family_contribution__{family}"] = np.asarray(values, dtype=np.float64)
    for name, value in result.state.items():
        arrays[f"state__{name}"] = np.asarray(value)
    if graph is not None:
        arrays.update(csr_arrays(graph))
    array_path = directory / f"{stem}.npz"
    array_hash = atomic_save_npz(array_path, **arrays)
    metadata = {
        "schema": "residual_graph_deem_fit_artifact_v1",
        "status": "complete",
        "cell_id": cell_id,
        "stem": stem,
        "arm_id": stem.split("__", 1)[0],
        "seed": int(result.seed),
        "array_path": str(array_path),
        "array_sha256": array_hash,
        "orientation": int(result.orientation),
        "aligned_bias": float(result.aligned_bias),
        "risk_anchor_difference": float(result.risk_anchor_difference),
        "health": result.health,
        "config": result.config,
        "objective_history": result.objective_history,
        "alias_of": result.alias_of,
        "extras": extras or {},
        "code_sha256": source_hash(),
        "config_sha256": canonical_sha256(result.config),
        "environment": environment_fingerprint(),
        "determinism": {"torch_deterministic_algorithms": True, "seeds": list(SEEDS)},
        "graph_sha256": canonical_sha256(csr_arrays(graph)) if graph is not None else None,
        **(provenance or {}),
    }
    metadata["content_sha256"] = canonical_sha256(metadata)
    atomic_write_json(directory / f"{stem}.json", metadata)
    return metadata


def write_score_only(
    root: Path,
    cell_id: str,
    stem: str,
    score: np.ndarray,
    *,
    seed: int,
    diagnostics: dict,
    graph=None,
    provenance: dict | None = None,
) -> dict:
    directory = root / "fits" / cell_id
    arrays = {"score": np.asarray(score, dtype=np.float64)}
    if graph is not None:
        arrays.update(csr_arrays(graph))
    array_path = directory / f"{stem}.npz"
    array_hash = atomic_save_npz(array_path, **arrays)
    metadata = {
        "schema": "residual_graph_deem_score_artifact_v1",
        "status": "complete",
        "cell_id": cell_id,
        "stem": stem,
        "arm_id": stem.split("__", 1)[0],
        "seed": int(seed),
        "array_path": str(array_path),
        "array_sha256": array_hash,
        "health": diagnostics,
        "code_sha256": source_hash(),
        "config_sha256": canonical_sha256(diagnostics),
        "environment": environment_fingerprint(),
        "determinism": {"seeds": list(SEEDS)},
        "graph_sha256": canonical_sha256(csr_arrays(graph)) if graph is not None else None,
        **(provenance or {}),
    }
    metadata["content_sha256"] = canonical_sha256(metadata)
    atomic_write_json(directory / f"{stem}.json", metadata)
    return metadata


def write_failure(root: Path, cell_id: str, stem: str, exc: Exception) -> dict:
    record = {
        "schema": "residual_graph_deem_fit_artifact_v1",
        "status": "failed",
        "cell_id": cell_id,
        "stem": stem,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "objective_history": jsonable(getattr(exc, "objective_history", [])),
        "last_finite_state": jsonable(getattr(exc, "last_finite_state", None)),
    }
    atomic_write_json(root / "fits" / cell_id / f"{stem}.json", record)
    return record


def expected_stems(phase0: dict) -> set[str]:
    stems = {
        artifact_stem(arm, seed)
        for arm in ("B0", "B1", "B2", "B3")
        for seed in SEEDS
    }
    for arm in ("G0", "G1", "G2", "G3", "G4", "G5"):
        for seed in SEEDS:
            for lambda_ in LAMBDA_GRID:
                stems.add(artifact_stem(arm, seed, lambda_))
    headline = float(phase0["nominated_lambdas"]["target"])
    for seed in SEEDS:
        for control in (
            "length_only", "node_permuted", "random_gate",
            "family_permuted", "posterior_permuted",
        ):
            stems.add(artifact_stem("G3", seed, headline, control))
        for k in (5, 10, 15):
            stems.add(f"SENSITIVITY__k{k}__G3__lambda{str(headline).replace('.', 'p')}__seed{seed}")
        for arm in ("B0", "B3", "G3"):
            stems.add(f"SENSITIVITY__stable_inventory_minus4__{arm}__seed{seed}")
    return stems


def _valid_complete_metadata(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            return None
        array_path = Path(value["array_path"])
        if not array_path.is_file() or sha256_file(array_path) != value.get("array_sha256"):
            return None
        expected = value.get("content_sha256")
        if expected is not None:
            unhashed = dict(value)
            unhashed.pop("content_sha256", None)
            if canonical_sha256(unhashed) != expected:
                return None
        return value
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def load_cell_checkpoint(out: Path, cell_id: str, phase0: dict, run_hash: str) -> list[dict] | None:
    marker = out / "fits" / cell_id / "CELL_COMPLETE.json"
    if not marker.is_file():
        return None
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if value.get("run_definition_sha256") != run_hash:
        return None
    required = expected_stems(phase0)
    if set(value.get("stems", [])) != required:
        return None
    records = []
    for stem in sorted(required):
        record = _valid_complete_metadata(out / "fits" / cell_id / f"{stem}.json")
        if record is None:
            return None
        records.append(record)
    return records


def write_cell_checkpoint(
    out: Path, cell_id: str, phase0: dict, run_hash: str, records: list[dict]
) -> None:
    required = expected_stems(phase0)
    complete = {
        str(record.get("stem")): record
        for record in records
        if record.get("status") == "complete" and record.get("stem")
    }
    missing = sorted(required - set(complete))
    failures = [record for record in records if record.get("status") != "complete"]
    if missing or failures:
        return
    marker = {
        "schema": "residual_graph_deem_cell_complete_v1",
        "cell_id": cell_id,
        "run_definition_sha256": run_hash,
        "stems": sorted(required),
        "artifact_sha256": {
            stem: complete[stem]["array_sha256"] for stem in sorted(required)
        },
    }
    marker["content_sha256"] = canonical_sha256(marker)
    atomic_write_json(out / "fits" / cell_id / "CELL_COMPLETE.json", marker)


def b0_score(X_risk: np.ndarray, names) -> tuple[np.ndarray, dict]:
    weights, rho, g2, diagnostics = upcr_fuse(X_risk.T, return_diagnostics=True)
    score = np.asarray(weights @ X_risk.T, dtype=float)
    anchor = equal_family_risk_anchor(X_risk, names)
    correlation = float(spearmanr(score, anchor).statistic)
    if not np.isfinite(correlation) or abs(correlation) <= 1e-6:
        raise ResidualGraphDeemError("B0 risk orientation is ambiguous")
    if correlation < 0:
        score = -score
        correlation = -correlation
    return score, {
        "healthy": bool(np.isfinite(score).all() and np.std(score) >= 1e-3),
        "score_sd": float(np.std(score)),
        "risk_anchor_spearman": correlation,
        "weights": weights,
        "rho_hat": rho,
        "g2_hat": float(g2),
        "upcr": diagnostics,
        "historical_F_used": False,
        "rho_polarity_used": False,
    }


def adapter_jobs(
    out: Path,
    cell_id: str,
    X_risk: np.ndarray,
    names,
    *,
    python: str,
    device: str,
    provenance: dict,
) -> list[dict]:
    input_path = out / "adapter_inputs" / f"{cell_id}.npz"
    input_hash = canonical_sha256({"cell": cell_id, "X": X_risk, "names": names})
    atomic_save_npz(
        input_path,
        X_risk=np.asarray(X_risk, dtype=np.float64),
        cell_id=np.asarray(cell_id),
        feature_names=np.asarray(names, dtype=str),
        input_sha256=np.asarray(input_hash),
        bundle_sha256=np.asarray(provenance["bundle_sha256"]),
        source_sha256=np.asarray(provenance["source_sha256"]),
        inventory_sha256=np.asarray(provenance["inventory_sha256"]),
        code_sha256=np.asarray(source_hash()),
    )
    jobs = []
    for mode, arm_id in (("hard", "B1"), ("soft", "B2")):
        for seed in SEEDS:
            stem = artifact_stem(arm_id, seed)
            prefix = out / "fits" / cell_id / stem
            command = [
                python, str(WORKER), "--input", str(input_path), "--output", str(prefix),
                "--mode", mode, "--seed", str(seed), "--device", device,
            ]
            jobs.append((arm_id, seed, prefix, command))
    records = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(subprocess.run, command, capture_output=True, text=True): item
                      for item, command in [((arm_id, seed, prefix), command) for arm_id, seed, prefix, command in jobs]}
        for future in as_completed(future_map):
            arm_id, seed, prefix = future_map[future]
            completed = future.result()
            path = prefix.with_suffix(".json")
            if not path.is_file():
                atomic_write_json(path, {
                    "schema": "residual_graph_deem_adapter020_fit_v1",
                    "status": "failed", "cell_id": cell_id, "arm_id": arm_id,
                    "seed": seed, "returncode": completed.returncode,
                    "stdout": completed.stdout, "stderr": completed.stderr,
                })
            record = json.loads(path.read_text(encoding="utf-8"))
            records.append(record)
    return records


def full_risk(bundle):
    donor, _, transform = donor_risk_matrix(
        bundle.X_raw, bundle.X_raw, bundle.feature_names
    )
    return donor, transform


def stage_a_cell(args, registry, phase0, cell_id: str) -> list[dict]:
    bundle_path = Path(args.bundle_dir) / f"{cell_id}.npz"
    bundle = load_target_free_bundle(bundle_path)
    registered = registry_cell(registry, cell_id)
    if (
        len(bundle.row_ids) != int(registered["n_rows"])
        or bundle.inventory_sha256 != registered["inventory_sha256"]
        or bundle.source_sha256 != registered["source"]["source_sha256"]
    ):
        raise ResidualGraphDeemError(f"bundle/registry mismatch: {cell_id}")
    X_risk, transform = full_risk(bundle)
    provenance = {
        "bundle_sha256": bundle.bundle_sha256,
        "source_sha256": bundle.source_sha256,
        "inventory_sha256": bundle.inventory_sha256,
        "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
    }
    records = []
    score_b0, b0_diag = b0_score(X_risk, bundle.feature_names)
    for seed in SEEDS:
        records.append(write_score_only(
            args.out_dir, cell_id, artifact_stem("B0", seed), score_b0,
            seed=seed, diagnostics=b0_diag, provenance=provenance,
        ))
    one_dimensional = self_safe_knn_graph(
        score_b0[:, None], k=7, tie_keys=row_id_tie_keys(bundle.row_ids)
    )
    atomic_save_npz(
        args.out_dir / "graphs" / cell_id / "b0_one_dimensional_k7.npz",
        **csr_arrays(one_dimensional), score=score_b0,
    )
    records.extend(adapter_jobs(
        args.out_dir, cell_id, X_risk, bundle.feature_names,
        python=args.python, device=args.adapter_device, provenance=provenance,
    ))

    continuous = ContinuousDeemConfig()
    dufs_config = DufsConfig()
    nominations = phase0["nominated_lambdas"]
    stable_names = tuple(registered["stable_inventory_minus4"])
    stable_indices = [bundle.feature_names.index(name) for name in stable_names]
    stable_raw = bundle.X_raw[:, stable_indices]
    stable_risk, _, stable_transform = donor_risk_matrix(stable_raw, stable_raw, stable_names)
    stable_b0, stable_b0_diag = b0_score(stable_risk, stable_names)
    for seed in SEEDS:
        records.append(write_score_only(
            args.out_dir, cell_id,
            f"SENSITIVITY__stable_inventory_minus4__B0__seed{seed}",
            stable_b0, seed=seed,
            diagnostics={**stable_b0_diag, "sensitivity": "stable_inventory_minus4"},
            provenance=provenance,
        ))
    for seed in SEEDS:
        b3_stem = artifact_stem("B3", seed)
        try:
            b3 = fit_continuous_deem(X_risk, bundle.feature_names, seed=seed, config=continuous)
            records.append(write_fit_result(
                args.out_dir, cell_id, b3_stem, b3,
                extras={"full_standardization": jsonable(transform)},
                provenance=provenance,
            ))
            crossfit = crossfit_continuous_deem(
                bundle.X_raw, bundle.feature_names, bundle.confidence_signs,
                bundle.group_ids, bundle.raw_trace_length, seed=seed, config=continuous,
            )
            folds_path = args.out_dir / "crossfit" / cell_id / f"seed{seed}.npz"
            atomic_save_npz(
                folds_path,
                logit=crossfit.logit,
                posterior=crossfit.posterior,
                contributions=crossfit.contributions,
                residuals=crossfit.residuals,
                folds=crossfit.folds,
                feature_names=np.asarray(bundle.feature_names, dtype=str),
                row_id=np.asarray(bundle.row_ids, dtype=str),
            )
            atomic_write_json(
                folds_path.with_suffix(".json"),
                {"fold_manifests": crossfit.fold_manifests,
                 "residualizers": crossfit.residualizer_records},
            )

            raw_uniform = build_inventory_graph(
                X_risk, bundle.feature_names, bundle.row_ids, k=7
            )
            residual_uniform = build_inventory_graph(
                crossfit.residuals, bundle.feature_names, bundle.row_ids, k=7
            )
            raw_gates, raw_gate_diag = cross_view_dufs(
                X_risk, bundle.feature_names, crossfit.folds, bundle.row_ids,
                config=dufs_config,
            )
            residual_gates, residual_gate_diag = cross_view_dufs(
                crossfit.residuals, bundle.feature_names, crossfit.folds, bundle.row_ids,
                config=dufs_config,
            )
            raw_dufs = build_inventory_graph(
                X_risk, bundle.feature_names, bundle.row_ids, k=7, gates=raw_gates
            )
            residual_dufs = build_inventory_graph(
                crossfit.residuals, bundle.feature_names, bundle.row_ids, k=7,
                gates=residual_gates,
            )
            headline = float(nominations["target"])
            for k in (5, 10, 15):
                graph_k = build_inventory_graph(
                    crossfit.residuals, bundle.feature_names, bundle.row_ids,
                    k=k, gates=residual_gates,
                )
                result_k = fit_continuous_deem(
                    X_risk, bundle.feature_names, seed=seed, config=continuous,
                    graph_config=GraphDeemConfig(k=k, lambda_=headline, mechanism="target"),
                    laplacian=symmetric_normalized_laplacian(graph_k),
                )
                records.append(write_fit_result(
                    args.out_dir, cell_id,
                    f"SENSITIVITY__k{k}__G3__lambda{str(headline).replace('.', 'p')}__seed{seed}",
                    result_k,
                    extras={"sensitivity": "k", "k": k, "graph_health": graph_health(graph_k)},
                    graph=graph_k, provenance=provenance,
                ))

            stable_b3 = fit_continuous_deem(
                stable_risk, stable_names, seed=seed, config=continuous
            )
            records.append(write_fit_result(
                args.out_dir, cell_id,
                f"SENSITIVITY__stable_inventory_minus4__B3__seed{seed}",
                stable_b3,
                extras={"sensitivity": "stable_inventory_minus4",
                        "full_standardization": jsonable(stable_transform)},
                provenance=provenance,
            ))
            stable_crossfit = crossfit_continuous_deem(
                stable_raw, stable_names, bundle.confidence_signs[stable_indices],
                bundle.group_ids, bundle.raw_trace_length, seed=seed, config=continuous,
            )
            stable_gates, stable_gate_diag = cross_view_dufs(
                stable_crossfit.residuals, stable_names, stable_crossfit.folds,
                bundle.row_ids, config=dufs_config,
            )
            stable_graph = build_inventory_graph(
                stable_crossfit.residuals, stable_names, bundle.row_ids,
                k=7, gates=stable_gates,
            )
            stable_g3 = fit_continuous_deem(
                stable_risk, stable_names, seed=seed, config=continuous,
                graph_config=GraphDeemConfig(lambda_=headline, mechanism="target"),
                laplacian=symmetric_normalized_laplacian(stable_graph),
            )
            records.append(write_fit_result(
                args.out_dir, cell_id,
                f"SENSITIVITY__stable_inventory_minus4__G3__seed{seed}",
                stable_g3,
                extras={"sensitivity": "stable_inventory_minus4",
                        "graph_health": graph_health(stable_graph),
                        "gate_diagnostics": stable_gate_diag},
                graph=stable_graph, provenance=provenance,
            ))
            family_L, family_order, family_affinity = present_family_laplacian(
                crossfit.residuals, bundle.feature_names
            )
            graphs = {
                "G0": (raw_uniform, "target", None),
                "G1": (raw_dufs, "target", None),
                "G2": (residual_uniform, "target", None),
                "G3": (residual_dufs, "target", None),
                "G4": (residual_dufs, "nuisance", None),
                "G5": (None, "family", family_L),
            }
            graph_diagnostics = {}
            for arm_id, (graph, mechanism, family_graph) in graphs.items():
                if graph is not None:
                    health = graph_health(graph)
                    artifact = fold_artifact_diagnostics(
                        crossfit.residuals if arm_id in {"G2", "G3", "G4"} else X_risk,
                        crossfit.folds, bundle.group_ids, graph, permutations=999,
                        seed=20260821 + seed,
                    )
                else:
                    health = {"healthy": True, "family_order": family_order,
                              "family_affinity": family_affinity}
                    artifact = {"healthy": True, "not_applicable": True}
                graph_diagnostics[arm_id] = {"graph_health": health, "fold_artifact": artifact}
                for lambda_ in LAMBDA_GRID:
                    stem = artifact_stem(arm_id, seed, lambda_)
                    result = fit_continuous_deem(
                        X_risk, bundle.feature_names, seed=seed, config=continuous,
                        graph_config=GraphDeemConfig(lambda_=lambda_, mechanism=mechanism),
                        laplacian=(symmetric_normalized_laplacian(graph) if graph is not None and lambda_ else None),
                        family_laplacian=(family_graph if family_graph is not None and lambda_ else None),
                        baseline_result=b3,
                    )
                    extras = dict(graph_diagnostics[arm_id])
                    extras.update({
                        "headline_lambda": float(nominations[mechanism]),
                        "is_headline": bool(float(lambda_) == float(nominations[mechanism])),
                        "raw_gate_diagnostics": raw_gate_diag if arm_id == "G1" else None,
                        "residual_gate_diagnostics": residual_gate_diag if arm_id in {"G3", "G4"} else None,
                        "gate_weights": (
                            raw_gates if arm_id == "G1" else
                            residual_gates if arm_id in {"G3", "G4"} else None
                        ),
                        "gate_sha256": (
                            canonical_sha256(raw_gates) if arm_id == "G1" else
                            canonical_sha256(residual_gates) if arm_id in {"G3", "G4"} else None
                        ),
                    })
                    records.append(write_fit_result(
                        args.out_dir, cell_id, stem, result, extras=extras,
                        graph=None if lambda_ == 0 else graph,
                        provenance=provenance,
                    ))

            # Registered negative/nuisance controls at the nominated target lambda.
            length_graph = self_safe_knn_graph(
                np.log1p(bundle.raw_trace_length)[:, None], k=7,
                tie_keys=row_id_tie_keys(bundle.row_ids),
            )
            permutation = np.random.Generator(np.random.PCG64(20260821 + seed)).permutation(len(X_risk))
            node_graph = permute_graph_nodes(residual_dufs, permutation)
            random_gates = random_gate_control(residual_gates, bundle.feature_names, seed=20260821 + seed)
            random_graph = build_inventory_graph(
                crossfit.residuals, bundle.feature_names, bundle.row_ids, k=7, gates=random_gates
            )
            family_permuted = crossfit.residuals.copy()
            generator = np.random.Generator(np.random.PCG64(30260821 + seed))
            for indices in {key: tuple(value) for key, value in b3.family_indices.items()}.values():
                family_permuted[:, list(indices)] = family_permuted[generator.permutation(len(X_risk))][:, list(indices)]
            family_permuted_graph = build_inventory_graph(
                family_permuted, bundle.feature_names, bundle.row_ids, k=7, gates=residual_gates
            )
            for control, graph in {
                "length_only": length_graph,
                "node_permuted": node_graph,
                "random_gate": random_graph,
                "family_permuted": family_permuted_graph,
            }.items():
                result = fit_continuous_deem(
                    X_risk, bundle.feature_names, seed=seed, config=continuous,
                    graph_config=GraphDeemConfig(lambda_=headline, mechanism="target"),
                    laplacian=symmetric_normalized_laplacian(graph),
                )
                records.append(write_fit_result(
                    args.out_dir, cell_id, artifact_stem("G3", seed, headline, control),
                    result, extras={"control": control, "graph_health": graph_health(graph)},
                    graph=graph, provenance=provenance,
                ))
            posterior_permuted = b3.score[permutation]
            records.append(write_score_only(
                args.out_dir, cell_id,
                artifact_stem("G3", seed, headline, "posterior_permuted"),
                posterior_permuted, seed=seed,
                diagnostics={"healthy": True, "control": "posterior_permuted"},
                graph=residual_dufs,
                provenance=provenance,
            ))
        except Exception as exc:
            # Keep an already frozen B3 fit intact when a downstream graph,
            # DUFS, or control step fails.  This explicit pipeline failure still
            # prevents the score-freeze manifest from being emitted.
            records.append(write_failure(
                args.out_dir, cell_id, artifact_stem("PIPELINE", seed), exc
            ))
    return records


def generate_synthetic_worlds(registry, *, smoke: bool = False) -> dict:
    config = registry["synthetic"]
    n = 160 if smoke else int(config["n_rows"])
    rows_per_group = 4
    p = 30
    generator = np.random.Generator(np.random.PCG64(int(config["base_seed"])))
    loading_signal = generator.normal(size=p)
    loading_signal /= np.linalg.norm(loading_signal)
    loading_signal *= float(config["signal_loading"])
    loading_nuisance = generator.normal(size=p)
    loading_nuisance -= loading_signal * np.dot(loading_nuisance, loading_signal) / np.dot(loading_signal, loading_signal)
    loading_nuisance /= np.linalg.norm(loading_nuisance)
    loading_nuisance *= float(config["nuisance_loading"])
    groups = np.asarray([f"group{i // rows_per_group:04d}" for i in range(n)], dtype=str)
    worlds = {}
    for index, world in enumerate(config["worlds"]):
        rng = np.random.Generator(np.random.PCG64(int(config["base_seed"]) + index + 1))
        signal = rng.normal(size=n)
        nuisance = rng.normal(size=n)
        noise = rng.normal(scale=float(config["noise_sd"]), size=(n, p))
        y = (signal > 0).astype(int)
        X = noise.copy()
        world_id = world["id"]
        if world_id == "shared_binary_signal":
            X += (2 * y - 1)[:, None] * loading_signal
        elif world_id == "length_target":
            length_latent = rng.normal(size=n)
            y = (length_latent > np.quantile(length_latent, 0.95)).astype(int)
            X += length_latent[:, None] * loading_nuisance
            signal = length_latent
        elif world_id == "smooth_nuisance":
            y = (rng.normal(size=n) > 0).astype(int)
            X += nuisance[:, None] * loading_nuisance
        elif world_id == "target_plus_orthogonal_nuisance":
            X += (2 * y - 1)[:, None] * loading_signal + nuisance[:, None] * loading_nuisance
        elif world_id == "linear_target":
            X += signal[:, None] * loading_signal
        elif world_id == "nonlinear_local_target":
            angle = rng.uniform(-np.pi, np.pi, n)
            y = (np.sin(2 * angle) > 0).astype(int)
            X[:, 0] += np.cos(angle) * 2
            X[:, 1] += np.sin(angle) * 2
        elif world_id == "duplicates":
            duplicate_n = int(round(float(config["duplicate_fraction"]) * n))
            X[:duplicate_n] = X[duplicate_n:2 * duplicate_n]
        elif world_id == "imbalance_near_constant":
            y = np.zeros(n, dtype=int)
            y[rng.choice(n, size=max(2, int(0.05 * n)), replace=False)] = 1
            X[:, 0] = 1.0 + rng.normal(scale=1e-10, size=n)
        elif world_id == "class_permutation":
            X += (2 * y - 1)[:, None] * loading_signal
            y = 1 - y
        elif world_id == "pure_noise":
            y = (rng.normal(size=n) > 0).astype(int)
        lengths = np.maximum(1, np.rint(np.exp(signal - signal.min() + 1))).astype(int)
        worlds[world_id] = {"X_risk": X, "synthetic_target": y, "groups": groups, "lengths": lengths}
    return {
        "loading_signal": loading_signal,
        "loading_nuisance": loading_nuisance,
        "worlds": worlds,
        "n": n,
    }


def run_phase0(args, registry) -> None:
    generated = generate_synthetic_worlds(registry, smoke=args.smoke)
    names = tuple(registry["canonical_feature_order"])
    signs = np.asarray(
        next(cell for cell in registry["cells"] if cell["n_features"] == 30)["confidence_signs"],
        dtype=float,
    )
    freeze = {
        "schema": "residual_graph_deem_phase0_inputs_v1",
        "base_seed": registry["synthetic"]["base_seed"],
        "n_rows": generated["n"],
        "loading_signal": generated["loading_signal"],
        "loading_nuisance": generated["loading_nuisance"],
        "expected_winner_matrix": registry["synthetic"]["worlds"],
        "seven_inventory_schema_hashes": [canonical_sha256(schema["feature_names"]) for schema in registry["schemas"]],
    }
    freeze["content_sha256"] = canonical_sha256(freeze)
    freeze_path = args.out_dir / "SYNTHETIC_WORLD_REGISTRY.json"
    if not freeze_path.exists():
        atomic_write_json(freeze_path, freeze, immutable=True)
    schema_fixture_results = []
    for schema_position, schema in enumerate(registry["schemas"]):
        schema_names = tuple(schema["feature_names"])
        schema_signs = confidence_sign_vector(schema_names)
        rng = np.random.Generator(np.random.PCG64(20260821 + 10_000 + schema_position))
        fixture_risk = rng.normal(size=(64, len(schema_names)))
        fixture_risk[1] = fixture_risk[0]
        fixture_raw = -fixture_risk * schema_signs[None, :]
        fixture_groups = np.asarray([f"schema{schema_position}::g{i // 4}" for i in range(64)])
        fixture_lengths = np.arange(64) % 13 + 1
        fixture_config = ContinuousDeemConfig(
            epochs=2 if args.smoke else 5, anchor_tolerance=1e-12,
            posterior_sd_min=0.0,
        )
        fixture_b3 = fit_continuous_deem(
            fixture_risk, schema_names, seed=0, config=fixture_config
        )
        fixture_alias = fit_continuous_deem(
            fixture_risk, schema_names, seed=0, config=fixture_config,
            graph_config=GraphDeemConfig(lambda_=0.0), baseline_result=fixture_b3,
        )
        fixture_crossfit = crossfit_continuous_deem(
            fixture_raw, schema_names, schema_signs, fixture_groups,
            fixture_lengths, seed=0, config=fixture_config,
        )
        fixture_graph = build_inventory_graph(
            fixture_crossfit.residuals, schema_names,
            [f"schema{schema_position}::row{i}" for i in range(64)], k=7,
        )
        symmetry = fixture_graph - fixture_graph.T
        schema_fixture_results.append({
            "schema_id": schema["schema_id"], "n_features": len(schema_names),
            "cells": schema["cells"],
            "lambda_zero_max_abs": float(np.max(np.abs(fixture_alias.score - fixture_b3.score))),
            "contribution_reconstruction_max_abs": fixture_b3.health["contribution_reconstruction_max_abs"],
            "graph_symmetry_max_abs": float(np.max(np.abs(symmetry.data))) if symmetry.nnz else 0.0,
            "graph_health": graph_health(fixture_graph),
            "duplicate_rows_exercised": True,
            "donor_only_crossfit": len(fixture_crossfit.fold_manifests) == 5,
        })
    atomic_write_json(args.out_dir / "PHASE0_SCHEMA_FIXTURES.json", schema_fixture_results)
    results = []
    epochs = 3 if args.smoke else 100
    dufs_epochs = 3 if args.smoke else 120
    phase_seeds = (0,) if args.smoke else SEEDS
    phase_lambdas = (0.0, 0.1) if args.smoke else LAMBDA_GRID
    for world_id, world in generated["worlds"].items():
        X_risk = np.asarray(world["X_risk"], dtype=float)
        X_raw = -X_risk * signs[None, :]
        y = world["synthetic_target"]
        for seed in phase_seeds:
            config = ContinuousDeemConfig(epochs=epochs, anchor_tolerance=1e-12,
                                          posterior_sd_min=0.0 if args.smoke else 1e-3)
            b3 = fit_continuous_deem(X_risk, names, seed=seed, config=config)
            baseline_auc = float(roc_auc_score(y, b3.score))
            crossfit = crossfit_continuous_deem(
                X_raw, names, signs, world["groups"], world["lengths"], seed=seed,
                config=config,
            )
            residual_gates, residual_gate_diag = cross_view_dufs(
                crossfit.residuals, names, crossfit.folds,
                [f"{world_id}::row{i}" for i in range(len(X_risk))],
                config=DufsConfig(epochs=dufs_epochs, seeds=(0, 1) if args.smoke else SEEDS,
                                  median_cosine_min=-1.0 if args.smoke else 0.80),
            )
            raw_gates, raw_gate_diag = cross_view_dufs(
                X_risk, names, crossfit.folds,
                [f"{world_id}::row{i}" for i in range(len(X_risk))],
                config=DufsConfig(epochs=dufs_epochs, seeds=(0, 1) if args.smoke else SEEDS,
                                  median_cosine_min=-1.0 if args.smoke else 0.80),
            )
            row_ids = [f"{world_id}::row{i}" for i in range(len(X_risk))]
            residual_dufs = build_inventory_graph(
                crossfit.residuals, names, [f"{world_id}::row{i}" for i in range(len(X_risk))],
                gates=residual_gates,
            )
            raw_uniform = build_inventory_graph(X_risk, names, row_ids)
            raw_dufs = build_inventory_graph(X_risk, names, row_ids, gates=raw_gates)
            residual_uniform = build_inventory_graph(crossfit.residuals, names, row_ids)
            family_L, _, _ = present_family_laplacian(crossfit.residuals, names)
            arm_graphs = {
                "G0": ("target", raw_uniform, None),
                "G1": ("target", raw_dufs, None),
                "G2": ("target", residual_uniform, None),
                "G3": ("target", residual_dufs, None),
                "G4": ("nuisance", residual_dufs, None),
                "G5": ("family", None, family_L),
            }
            length_graph = self_safe_knn_graph(
                np.log1p(world["lengths"])[:, None], k=7,
                tie_keys=row_id_tie_keys(row_ids),
            )
            permutation = np.random.Generator(
                np.random.PCG64(20260821 + seed)
            ).permutation(len(X_risk))
            node_graph = permute_graph_nodes(residual_dufs, permutation)
            random_gates = random_gate_control(
                residual_gates, names, seed=30260821 + seed
            )
            random_graph = build_inventory_graph(
                crossfit.residuals, names, row_ids, gates=random_gates
            )
            family_permuted = crossfit.residuals.copy()
            family_rng = np.random.Generator(np.random.PCG64(40260821 + seed))
            for indices in family_index_map(names).values():
                family_permuted[:, list(indices)] = family_permuted[
                    family_rng.permutation(len(X_risk))
                ][:, list(indices)]
            family_permuted_graph = build_inventory_graph(
                family_permuted, names, row_ids, gates=residual_gates
            )
            controls = {
                "length_only": length_graph,
                "node_permuted": node_graph,
                "random_gate": random_graph,
                "family_permuted": family_permuted_graph,
            }
            for arm, (mechanism, graph, family_graph) in arm_graphs.items():
                for lambda_ in phase_lambdas:
                    candidate = fit_continuous_deem(
                        X_risk, names, seed=seed, config=config,
                        graph_config=GraphDeemConfig(lambda_=lambda_, mechanism=mechanism),
                        laplacian=(symmetric_normalized_laplacian(graph) if graph is not None and lambda_ else None),
                        family_laplacian=(family_graph if family_graph is not None and lambda_ else None),
                        baseline_result=b3,
                    )
                    results.append({
                        "world": world_id, "seed": seed, "arm": arm,
                        "mechanism": mechanism, "control": None,
                        "lambda": lambda_, "baseline_auc": baseline_auc,
                        "candidate_auc": float(roc_auc_score(y, candidate.score)),
                        "delta": float(roc_auc_score(y, candidate.score) - baseline_auc),
                        "healthy": bool(candidate.health["healthy"]),
                        "graph_healthy": bool(graph_health(graph)["healthy"]) if graph is not None else True,
                        "gate_effective_count": (
                            raw_gate_diag["effective_feature_count"] if arm == "G1" else
                            residual_gate_diag["effective_feature_count"] if arm in {"G3", "G4"} else None
                        ),
                    })
            for control, control_graph in controls.items():
                for lambda_ in phase_lambdas:
                    candidate = fit_continuous_deem(
                        X_risk, names, seed=seed, config=config,
                        graph_config=GraphDeemConfig(lambda_=lambda_, mechanism="target"),
                        laplacian=symmetric_normalized_laplacian(control_graph) if lambda_ else None,
                        baseline_result=b3,
                    )
                    auc = float(roc_auc_score(y, candidate.score))
                    results.append({
                        "world": world_id, "seed": seed, "arm": "G3_CONTROL",
                        "mechanism": "target", "control": control,
                        "lambda": lambda_, "baseline_auc": baseline_auc,
                        "candidate_auc": auc, "delta": float(auc - baseline_auc),
                        "healthy": bool(candidate.health["healthy"]),
                        "graph_healthy": bool(graph_health(control_graph)["healthy"]),
                        "gate_effective_count": residual_gate_diag["effective_feature_count"],
                    })
                if control == "family_permuted":
                    permuted_auc = float(roc_auc_score(y, b3.score[permutation]))
                    results.append({
                        "world": world_id, "seed": seed, "arm": "G3_CONTROL",
                        "mechanism": "target", "control": "posterior_permuted",
                        "lambda": None, "baseline_auc": baseline_auc,
                        "candidate_auc": permuted_auc, "delta": float(permuted_auc - baseline_auc),
                        "healthy": True, "graph_healthy": True,
                        "gate_effective_count": residual_gate_diag["effective_feature_count"],
                    })
    atomic_write_json(args.out_dir / "PHASE0_RESULTS.json", results)
    nominations = {}
    mechanism_arm = {"target": "G3", "nuisance": "G4", "family": "G5"}
    positive_worlds = {
        "target": {"shared_binary_signal", "nonlinear_local_target"},
        "nuisance": {"smooth_nuisance", "target_plus_orthogonal_nuisance"},
        "family": {
            "shared_binary_signal", "smooth_nuisance",
            "target_plus_orthogonal_nuisance", "linear_target",
            "nonlinear_local_target", "duplicates", "imbalance_near_constant",
        },
    }
    required_negative = {
        "target": {"length_target", "smooth_nuisance", "target_plus_orthogonal_nuisance",
                   "linear_target", "duplicates", "imbalance_near_constant",
                   "class_permutation", "pure_noise"},
        "nuisance": {"length_target", "class_permutation", "pure_noise"},
        "family": {"length_target", "class_permutation", "pure_noise"},
    }
    for mechanism in ("target", "nuisance", "family"):
        candidates = []
        for lambda_ in phase_lambdas[1:]:
            selected = [
                row for row in results
                if row["arm"] == mechanism_arm[mechanism] and row["lambda"] == lambda_
            ]
            positive = [row["delta"] for row in selected if row["world"] in positive_worlds[mechanism]]
            bad = [
                float(np.mean([row["delta"] for row in selected if row["world"] == world]))
                for world in sorted(required_negative[mechanism])
            ]
            mean = float(np.mean(positive))
            se = float(np.std(positive, ddof=1) / np.sqrt(max(len(positive), 1)))
            control_bad = []
            for world in sorted(required_negative[mechanism]):
                for control in sorted(controls):
                    values = [
                        row["delta"] for row in results
                        if row["arm"] == "G3_CONTROL" and row["lambda"] == lambda_
                        and row["world"] == world and row["control"] == control
                    ]
                    if values:
                        control_bad.append(float(np.mean(values)))
            promotes_negative = any(value > max(se, 1e-3) for value in bad + control_bad)
            candidates.append((lambda_, mean, se, promotes_negative))
        survivors = [row for row in candidates if not row[3]]
        if not survivors:
            nominations[mechanism] = None
            continue
        best = max(survivors, key=lambda row: row[1])
        threshold = best[1] - best[2]
        eligible = [row for row in survivors if row[1] >= threshold]
        nominations[mechanism] = float(min(row[0] for row in eligible))
    schema_pass = all(
        row["lambda_zero_max_abs"] <= 1e-10
        and row["contribution_reconstruction_max_abs"] <= 1e-8
        and row["graph_symmetry_max_abs"] <= 1e-10
        and row["donor_only_crossfit"]
        for row in schema_fixture_results
    )
    if args.smoke:
        # Smoke validates mechanics only; it is permanently barred from Stage A
        # and therefore must not pretend that n=160/one-seed outcomes selected
        # scientific hyperparameters.
        nominations = {
            mechanism: (value if value is not None else 0.1)
            for mechanism, value in nominations.items()
        }
    world_gates = []
    expected_by_world = {row["id"]: tuple(row.get("expected", ())) for row in registry["synthetic"]["worlds"]}
    for world_id, expected in expected_by_world.items():
        arm_auc = {}
        baseline = [
            row["baseline_auc"] for row in results
            if row["world"] == world_id and row["arm"] == "G3" and row["lambda"] == 0.0
        ]
        arm_auc["B3"] = float(np.mean(baseline))
        for arm in ("G0", "G1", "G2", "G3", "G4", "G5"):
            mechanism = "nuisance" if arm == "G4" else "family" if arm == "G5" else "target"
            selected = [
                row["candidate_auc"] for row in results
                if row["world"] == world_id and row["arm"] == arm
                and row["lambda"] == nominations[mechanism]
            ]
            arm_auc[arm] = float(np.mean(selected)) if selected else float("nan")
        finite = {arm: value for arm, value in arm_auc.items() if np.isfinite(value)}
        winner = max(finite, key=finite.get)
        if expected:
            available_expected = [arm for arm in expected if arm in finite]
            if available_expected:
                expected_best = max(finite[arm] for arm in available_expected)
                if winner == "B3":
                    spread = np.asarray(baseline, dtype=float)
                else:
                    winner_mechanism = "nuisance" if winner == "G4" else "family" if winner == "G5" else "target"
                    spread = np.asarray([
                        row["candidate_auc"] for row in results
                        if row["world"] == world_id and row["arm"] == winner
                        and row["lambda"] == nominations[winner_mechanism]
                    ], dtype=float)
                se = float(np.std(spread, ddof=1) / np.sqrt(len(spread))) if len(spread) > 1 else 0.0
                gate = bool(expected_best >= finite[winner] - max(se, 1e-3))
            else:
                gate = False
        else:
            graph_gain = max(finite[arm] - finite["B3"] for arm in finite if arm != "B3")
            gate = bool(graph_gain <= 1e-3)
        world_gates.append({
            "world": world_id, "expected": expected, "winner": winner,
            "arm_auc": arm_auc, "pass": gate,
        })
    expected_winners_pass = all(row["pass"] for row in world_gates)
    passed = (
        all(value is not None for value in nominations.values())
        and schema_pass
        and (expected_winners_pass or args.smoke)
    )
    complete = {
        "schema": "residual_graph_deem_phase0_complete_v1",
        "status": "pass" if passed else "stop_before_natural_targets",
        "smoke": bool(args.smoke),
        "nominated_lambdas": nominations,
        "n_results": len(results),
        "seven_schema_fixtures_pass": schema_pass,
        "scientific_selection_applied": not args.smoke,
        "expected_winner_matrix_pass": expected_winners_pass,
        "synthetic_world_gates": world_gates,
        "synthetic_registry_sha256": sha256_file(freeze_path),
        "results_sha256": sha256_file(args.out_dir / "PHASE0_RESULTS.json"),
        "schema_fixtures_sha256": sha256_file(args.out_dir / "PHASE0_SCHEMA_FIXTURES.json"),
    }
    atomic_write_json(args.out_dir / "PHASE0_COMPLETE.json", complete)
    if not passed:
        raise SystemExit("Phase 0 has no survivor; natural-target evaluation remains closed")


def run_stage_a(args, registry) -> None:
    phase0 = json.loads(Path(args.phase0_complete).read_text(encoding="utf-8"))
    if phase0.get("status") != "pass" or phase0.get("smoke"):
        raise SystemExit("Stage A requires a full passing Phase-0 freeze")
    cells = [cell["cell_id"] for cell in registry["cells"]]
    if args.cells:
        requested = [value.strip() for value in args.cells.split(",") if value.strip()]
        if set(requested) - set(cells):
            raise SystemExit("unknown Stage-A cell")
        cells = requested
    definition = {
        "schema": "residual_graph_deem_run_definition_v1",
        "status": "frozen",
        "debug": False,
        "cells": cells,
        "seeds": list(SEEDS),
        "lambdas": list(LAMBDA_GRID),
        "arms": ARM_SPECS,
        "registry_content_sha256": registry["registry_content_sha256"],
        "phase0_complete_sha256": sha256_file(args.phase0_complete),
        "protocol_sha256": sha256_file(ROOT / "docs/experiments/RESIDUAL_GRAPH_DEEM_24CELL_V1.md"),
        "registry_file_sha256": sha256_file(args.registry),
        "environment": environment_fingerprint(),
        "code_sha256": source_hash(),
        "python": args.python,
    }
    definition["config_sha256"] = canonical_sha256(definition)
    definition_path = args.out_dir / "RUN_DEFINITION.json"
    if definition_path.exists():
        existing = json.loads(definition_path.read_text(encoding="utf-8"))
        if existing != jsonable(definition):
            raise SystemExit("run-definition mismatch on resume")
    else:
        atomic_write_json(definition_path, definition, immutable=True)
    run_hash = sha256_file(definition_path)
    frozen_path = args.out_dir / "SCORE_FREEZE_MANIFEST.json"
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        unhashed = dict(frozen)
        expected = unhashed.pop("content_sha256", None)
        if (
            frozen.get("status") != "complete"
            or frozen.get("run_definition_sha256") != run_hash
            or canonical_sha256(unhashed) != expected
        ):
            raise SystemExit("existing score-freeze manifest is invalid or mismatched")
        for artifact in frozen.get("artifacts", []):
            path = args.out_dir / artifact["path"]
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                raise SystemExit(f"existing score-freeze artifact mismatch: {path}")
        print("[Stage A] verified immutable complete freeze; nothing to resume", flush=True)
        return
    records = []
    for cell_id in cells:
        checkpoint = load_cell_checkpoint(args.out_dir, cell_id, phase0, run_hash)
        if checkpoint is not None:
            print(f"[Stage A] {cell_id} (verified checkpoint)", flush=True)
            records.extend(checkpoint)
            continue
        print(f"[Stage A] {cell_id}", flush=True)
        current = stage_a_cell(args, registry, phase0, cell_id)
        records.extend(current)
        write_cell_checkpoint(args.out_dir, cell_id, phase0, run_hash, current)
    failures = [record for record in records if record.get("status") != "complete"]
    missing_seeds = []
    required = {"B0", "B1", "B2", "B3"}
    for cell_id in cells:
        for arm_id in required:
            found = sorted(
                int(record.get("seed")) for record in records
                if record.get("cell_id") == cell_id and record.get("arm_id", record.get("stem", "").split("__")[0]) == arm_id
                and record.get("status") == "complete"
            )
            if found != list(SEEDS):
                missing_seeds.append({"cell": cell_id, "arm": arm_id, "found": found})
    missing_artifacts = []
    required_stems = expected_stems(phase0)
    for cell_id in cells:
        observed = {
            str(record.get("stem")) for record in records
            if record.get("cell_id") == cell_id and record.get("status") == "complete"
        }
        for stem in sorted(required_stems - observed):
            missing_artifacts.append({"cell": cell_id, "stem": stem})
    fit_complete = {
        "schema": "residual_graph_deem_fit_complete_v1",
        "status": "complete" if not failures and not missing_seeds and not missing_artifacts and len(cells) == 24 else "incomplete",
        "cells": cells,
        "n_records": len(records),
        "incomplete_fits": failures,
        "missing_seeds": missing_seeds,
        "missing_artifacts": missing_artifacts,
        "run_definition_sha256": run_hash,
    }
    atomic_write_json(args.out_dir / "FIT_COMPLETE.json", fit_complete)
    if fit_complete["status"] != "complete":
        raise SystemExit("Stage A incomplete; score freeze not written")
    score_files = sorted((args.out_dir / "fits").glob("*/*.npz"))
    freeze = {
        "schema": "residual_graph_deem_score_freeze_v1",
        "status": "complete",
        "debug": False,
        "cells": cells,
        "missing_seeds": [],
        "incomplete_fits": [],
        "missing_artifacts": [],
        "run_definition_sha256": sha256_file(definition_path),
        "fit_complete_sha256": sha256_file(args.out_dir / "FIT_COMPLETE.json"),
        "artifacts": [{"path": str(path.relative_to(args.out_dir)), "sha256": sha256_file(path)} for path in score_files],
    }
    freeze["content_sha256"] = canonical_sha256(freeze)
    atomic_write_json(args.out_dir / "SCORE_FREEZE_MANIFEST.json", freeze, immutable=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("phase0", "stage-a"))
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--phase0-complete", type=Path)
    parser.add_argument("--cells")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--adapter-device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.out_dir = args.out_dir.resolve()
    registry = load_registry(args.registry)
    if args.command == "phase0":
        run_phase0(args, registry)
    else:
        if args.bundle_dir is None or args.phase0_complete is None:
            parser.error("stage-a requires --bundle-dir and --phase0-complete")
        run_stage_a(args, registry)


if __name__ == "__main__":
    main()
