#!/usr/bin/env python3
"""Freeze and evaluate the registered C3--C8 Phase-2 atomic candidates.

The runner is deliberately variant-parametric so the six contracts are frozen
against one code hash before any of their ProcessBench labels are opened.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    empirical_midrank, load_prepared_localization_cell, validate_fit_manifest,
)
from spectral_utils.fixed_application_pipelines import SHARED_TOKEN_VIEWS  # noqa: E402
from spectral_utils.token_local_fusion import (  # noqa: E402
    IU_CONFIG, fit_local_iu29, prepare_localization_cell,
)
from spectral_utils.upcr import upcr_fit  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402


REFERENCE = c1.REFERENCE
VARIANTS = (
    "C3_ENT_CCUSUM", "C4_ENT_SAMPLED", "C5_ENT_ENERGY",
    "C6_DSP12", "C7_EDIS_ONSET", "C8_SELF_INNOV",
)
ATOMIC_ROOT = p1.PROGRAM_ROOT / "phase_2/atomic"
PRIMARY_COMPARISON_FAMILY = 16
TOPK = 10
EWMA_ALPHA = 2.0 / 17.0
CUSUM_KAPPA = 0.0
EDIS_TAU_B = 1.36
EDIS_TAU_R = 1.33
INNOVATION_RIDGE = 1.0
EXTRA_PARENT = {"C8_SELF_INNOV": "C8_IU29_TOP10_PARENT"}


class AtomicRemainingError(RuntimeError):
    """Fail-closed contract error for the C3--C8 batch."""


def output_root(variant: str) -> Path:
    return ATOMIC_ROOT / variant.lower()


def registry_path(variant: str) -> Path:
    return ATOMIC_ROOT / f"{variant}_EXECUTION_REGISTRY.json"


def response_map(values: Sequence[float], offsets: Sequence[int], transform: Any) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    bounds = np.asarray(offsets, dtype=np.int64)
    if bounds.ndim != 1 or bounds[0] != 0 or bounds[-1] != len(x) or np.any(np.diff(bounds) <= 0):
        raise ValueError("response offsets do not partition the token curve")
    output = np.empty_like(x)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        output[int(lo):int(hi)] = transform(x[int(lo):int(hi)])
    if not np.isfinite(output).all():
        raise AtomicRemainingError("causal transform produced non-finite values")
    return output


def causal_cusum(trace: Sequence[float]) -> np.ndarray:
    """Two-sided absolute reset CUSUM around the frozen standardized zero."""

    x = np.asarray(trace, dtype=np.float64)
    positive = negative = 0.0
    output = np.empty_like(x)
    for index, value in enumerate(x):
        positive = max(0.0, positive + float(value) - CUSUM_KAPPA)
        negative = max(0.0, negative - float(value) - CUSUM_KAPPA)
        output[index] = max(positive, negative)
    return output


def ewma16(trace: Sequence[float]) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)
    output = np.empty_like(x)
    state = float(x[0])
    for index, value in enumerate(x):
        state = float(value) if index == 0 else EWMA_ALPHA * float(value) + (1.0 - EWMA_ALPHA) * state
        output[index] = state
    return output


def positive_area(trace: Sequence[float]) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)
    return np.cumsum(np.maximum(x, 0.0), dtype=np.float64) / np.arange(1, len(x) + 1)


def persistence(trace: Sequence[float]) -> np.ndarray:
    x = np.asarray(trace, dtype=np.float64)
    return np.cumsum(x > 0.0, dtype=np.float64) / np.arange(1, len(x) + 1)


def edis_onset(trace: Sequence[float]) -> np.ndarray:
    """Standardized local adaptation of EDIS burst/rebound morphology.

    The sealed input exposes a standardized affine entropy coordinate, not raw
    entropy in nats. Thresholds therefore mean standardized units here and the
    run is explicitly a repository adaptation, not a paper-exact replay.
    """

    x = np.asarray(trace, dtype=np.float64)
    burst = np.zeros_like(x)
    if len(x) > 1:
        burst[1:] = np.maximum(np.diff(x) - EDIS_TAU_B, 0.0)
    running_min = np.minimum.accumulate(x)
    rebound_excess = np.maximum(x - running_min - EDIS_TAU_R, 0.0)
    rebound_onset = np.maximum(np.diff(np.concatenate(([0.0], rebound_excess))), 0.0)
    return np.maximum(burst, rebound_onset)


def prefix_replay_error(trace: Sequence[float], transform: Any) -> float:
    x = np.asarray(trace, dtype=np.float64)
    full = transform(x)
    cuts = sorted({1, min(3, len(x)), min(16, len(x)), min(64, len(x)),
                   max(1, len(x) // 2), max(1, len(x) - 1), len(x)})
    return max(float(np.max(np.abs(full[:cut] - transform(x[:cut])))) for cut in cuts)


def fuse_channels(channels: Sequence[Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in channels]
    if not arrays or any(value.shape != arrays[0].shape for value in arrays):
        raise ValueError("step channels must be nonempty and aligned")
    return np.mean(np.vstack([empirical_midrank(value) for value in arrays]), axis=0)


def reduce_curve(curve: Sequence[float], cell: Any) -> np.ndarray:
    return p1.topk_step_mean(curve, cell.segment_starts, cell.segment_ends, k=TOPK)


def primitive_risks(cell: Any) -> dict[str, np.ndarray]:
    lookup = {name: index for index, name in enumerate(SHARED_TOKEN_VIEWS)}
    return {
        "entropy": -np.asarray(cell.token_confidence[:, lookup["entropy_series"]], dtype=np.float64),
        "sampled": -np.asarray(cell.token_confidence[:, lookup["spilled_series"]], dtype=np.float64),
        "energy": -np.asarray(cell.token_confidence[:, lookup["energy_series"]], dtype=np.float64),
    }


def _positions_predecessors(offsets: Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bounds = np.asarray(offsets, dtype=np.int64)
    n = int(bounds[-1])
    owners = np.searchsorted(bounds[1:], np.arange(n), side="right").astype(np.int64)
    positions = np.arange(n, dtype=np.int64) - bounds[owners]
    predecessor = np.arange(n, dtype=np.int64) - 1
    predecessor[positions == 0] = -1
    return positions, owners, predecessor


def _question_weights(owners: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(owners, dtype=np.int64))
    weights = 1.0 / counts[np.asarray(owners, dtype=np.int64)]
    return weights / float(np.mean(weights))


def _weighted_ridge(design: np.ndarray, target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    penalty = np.eye(design.shape[1], dtype=np.float64) * INNOVATION_RIDGE
    penalty[0, 0] = 0.0
    gram = design.T @ (design * weights[:, None]) + penalty
    rhs = design.T @ (weights * target)
    try:
        return np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram, rcond=1e-12) @ rhs


def fit_self_innovation(cell: Any) -> tuple[np.ndarray, np.ndarray, Mapping[str, Any]]:
    """Fit the smallest label-free self-lag residual block and augmented IU."""

    preparation = prepare_localization_cell(cell)
    parent = fit_local_iu29(preparation)
    standardized = preparation.standardized_slice(0, len(preparation.values))
    positions, owners, predecessor = _positions_predecessors(cell.token_offsets)
    fit = np.asarray([index for index in preparation.fit_indices if predecessor[int(index)] >= 0], dtype=np.int64)
    if len(fit) < 100:
        raise AtomicRemainingError("C8 has too few lag pairs")
    residual = np.zeros_like(standardized)
    coefficients = np.zeros((standardized.shape[1], 3), dtype=np.float64)
    scales = np.zeros(standardized.shape[1], dtype=np.float64)
    prior_fit = predecessor[fit]
    weights = _question_weights(owners[fit])
    for feature in range(standardized.shape[1]):
        design = np.column_stack([
            np.ones(len(fit), dtype=np.float64),
            np.log1p(positions[fit].astype(np.float64)),
            standardized[prior_fit, feature],
        ])
        beta = _weighted_ridge(design, standardized[fit, feature], weights)
        train_residual = standardized[fit, feature] - design @ beta
        scale = float(np.sqrt(np.average(np.square(train_residual), weights=weights)))
        if not np.isfinite(scale) or scale <= 1e-8:
            raise AtomicRemainingError(f"C8 degenerate residual scale for feature {feature}")
        valid = np.flatnonzero(predecessor >= 0)
        prediction = (beta[0] + beta[1] * np.log1p(positions[valid].astype(np.float64))
                      + beta[2] * standardized[predecessor[valid], feature])
        residual[valid, feature] = (standardized[valid, feature] - prediction) / scale
        coefficients[feature] = beta
        scales[feature] = scale

    innovation_confidence = -np.abs(residual)
    augmented = np.column_stack([np.asarray(cell.token_confidence, dtype=np.float64), innovation_confidence])
    fit_values = augmented[preparation.fit_indices]
    median = np.median(fit_values, axis=0)
    clean_fit = np.where(np.isfinite(fit_values), fit_values, median[None, :])
    std = clean_fit.std(axis=0)
    keep = np.isfinite(median) & np.isfinite(std) & (std > 1e-8)
    mean = clean_fit[:, keep].mean(axis=0)
    scale = clean_fit[:, keep].std(axis=0)
    standardized_fit = (clean_fit[:, keep] - mean[None, :]) / scale[None, :]
    fitted = upcr_fit(standardized_fit.T, **dict(IU_CONFIG))
    weights_iu = np.asarray(fitted.w, dtype=np.float64).copy()
    parent_fit_risk = parent.token_risk[preparation.fit_indices]
    raw_fit_confidence = standardized_fit @ weights_iu
    correlation = float(np.corrcoef(raw_fit_confidence, -parent_fit_risk)[0, 1])
    if np.isfinite(correlation) and correlation < 0.0:
        weights_iu *= -1.0
    candidate_risk = np.empty(len(augmented), dtype=np.float64)
    for lo in range(0, len(augmented), 100_000):
        hi = min(lo + 100_000, len(augmented))
        block = np.where(np.isfinite(augmented[lo:hi, keep]), augmented[lo:hi, keep], mean[None, :])
        candidate_risk[lo:hi] = -(((block - mean[None, :]) / scale[None, :]) @ weights_iu)
    diagnostics = {
        "ridge": INNOVATION_RIDGE,
        "predictors": ["intercept", "log1p(token_position)", "self_lag"],
        "residual_orientation": "negative absolute standardized residual as confidence",
        "original_streams": int(standardized.shape[1]),
        "augmented_kept_streams": int(keep.sum()),
        "fit_lag_pairs": int(len(fit)),
        "iu_components": 2,
        "iu_scale_ratio": 0.25,
        "orientation_correlation_with_frozen_iu29": correlation,
        "coefficient_sha256": c1.payload_sha({"beta": coefficients.tolist(), "scale": scales.tolist()}),
    }
    return np.asarray(parent.token_risk), candidate_risk, diagnostics


def score_candidate(variant: str, cell: Any) -> tuple[np.ndarray, dict[str, np.ndarray], Mapping[str, Any], float]:
    risks = primitive_risks(cell)
    entropy_step = reduce_curve(risks["entropy"], cell)
    extra: dict[str, np.ndarray] = {}
    suffix_error = 0.0
    if variant == "C3_ENT_CCUSUM":
        swvar = c1.response_reset_swvar(risks["entropy"], cell.token_offsets)
        cusum = response_map(risks["entropy"], cell.token_offsets, causal_cusum)
        channels = [entropy_step, reduce_curve(swvar, cell), reduce_curve(cusum, cell)]
        for lo, hi in zip(cell.token_offsets[:-1], cell.token_offsets[1:]):
            suffix_error = max(suffix_error, prefix_replay_error(risks["entropy"][int(lo):int(hi)], causal_cusum))
        return fuse_channels(channels), extra, {"channels": ["entropy", "swvar16", "absolute_reset_cusum"], "cusum_kappa": CUSUM_KAPPA}, suffix_error
    if variant == "C4_ENT_SAMPLED":
        return fuse_channels([entropy_step, reduce_curve(risks["sampled"], cell)]), extra, {"channels": ["entropy", "sampled_token_surprisal"]}, 0.0
    if variant == "C5_ENT_ENERGY":
        return fuse_channels([entropy_step, reduce_curve(risks["energy"], cell)]), extra, {"channels": ["entropy", "partition_energy"]}, 0.0
    if variant == "C6_DSP12":
        transforms = (("level", lambda x: np.asarray(x, dtype=np.float64)),
                      ("ewma16", ewma16), ("positive_area", positive_area), ("persistence", persistence))
        channels = []
        for source in ("entropy", "sampled", "energy"):
            for name, transform in transforms:
                curve = response_map(risks[source], cell.token_offsets, transform)
                channels.append(reduce_curve(curve, cell))
                for lo, hi in zip(cell.token_offsets[:-1], cell.token_offsets[1:]):
                    suffix_error = max(suffix_error, prefix_replay_error(risks[source][int(lo):int(hi)], transform))
        return fuse_channels(channels), extra, {"sources": ["entropy", "sampled", "energy"], "transforms": [name for name, _ in transforms], "n_channels": 12}, suffix_error
    if variant == "C7_EDIS_ONSET":
        onset = response_map(risks["entropy"], cell.token_offsets, edis_onset)
        for lo, hi in zip(cell.token_offsets[:-1], cell.token_offsets[1:]):
            suffix_error = max(suffix_error, prefix_replay_error(risks["entropy"][int(lo):int(hi)], edis_onset))
        return reduce_curve(onset, cell), extra, {"tau_b_standardized": EDIS_TAU_B, "tau_r_standardized": EDIS_TAU_R, "fidelity": "standardized repository adaptation; not paper-exact raw-nat EDIS"}, suffix_error
    if variant == "C8_SELF_INNOV":
        parent_token, candidate_token, diagnostics = fit_self_innovation(cell)
        parent_step = reduce_curve(parent_token, cell)
        extra[EXTRA_PARENT[variant]] = parent_step
        extra["__C8_IU29_STEPMAX_AUDIT"] = np.asarray([
            np.max(parent_token[int(lo):int(hi)])
            for lo, hi in zip(cell.segment_starts, cell.segment_ends)
        ], dtype=np.float64)
        return reduce_curve(candidate_token, cell), extra, diagnostics, 0.0
    raise KeyError(variant)


def require_sources(registry: Mapping[str, Any]) -> None:
    for source in registry["frozen_sources"]:
        path = Path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise AtomicRemainingError(f"frozen source changed or missing: {source['role']}")


def load_registry(path: Path, release: Path, variant: str) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema": "reasoning-localization-phase2-atomic-remaining-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN", "candidate": variant,
        "atomic_reference": REFERENCE, "processbench_cells": list(p2r.PB_CELLS),
        "topk": TOPK, "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "primary_comparison_family_size": PRIMARY_COMPARISON_FAMILY,
    }
    for key, value in required.items():
        if registry.get(key) != value:
            raise AtomicRemainingError(f"execution registry mismatch for {key}")
    if Path(registry["release_root"]).resolve() != release.resolve():
        raise AtomicRemainingError("release root differs from registry")
    if registry["runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise AtomicRemainingError("runner changed after freeze")
    require_sources(registry)
    return registry


def freeze_scores(variant: str, release: Path, output: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {variant} output: {output}")
    score_root = output / "score_freeze"
    score_root.mkdir(parents=True, exist_ok=False)
    input_root = release / "build_A/localization/inputs"
    manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_manifest(manifest_path, input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    records = []
    alias_local = alias_combined = suffix_error = parent_max_alias = 0.0
    started = time.perf_counter()
    for position, cell_id in enumerate(p2r.PB_CELLS, start=1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        entropy = primitive_risks(cell)["entropy"]
        reference_local = reduce_curve(entropy, cell)
        candidate_local, extra_parents, diagnostics, cell_suffix = score_candidate(variant, cell)
        suffix_error = max(suffix_error, cell_suffix)
        reference_combined = p1.combine_with_common_detector(cell, reference_local)
        candidate_combined = p1.combine_with_common_detector(cell, candidate_local)
        prior = load_npz_no_pickle(c1.P2R_TOP10_ROOT / "score_freeze/cells" / cell_id / "scores.npz")
        alias_local = max(alias_local, float(np.max(np.abs(reference_local - prior["local_step_scores"]))))
        alias_combined = max(alias_combined, float(np.max(np.abs(reference_combined - prior["combined_step_scores"]))))
        arrays: dict[str, np.ndarray] = {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "reference_local_step_scores": np.asarray(reference_local, dtype="<f8"),
            "reference_combined_step_scores": np.asarray(reference_combined, dtype="<f8"),
            "candidate_local_step_scores": np.asarray(candidate_local, dtype="<f8"),
            "candidate_combined_step_scores": np.asarray(candidate_combined, dtype="<f8"),
        }
        for parent_id, parent_local in extra_parents.items():
            if parent_id == "__C8_IU29_STEPMAX_AUDIT":
                strict_local, _strict_combined, _record = p1._strict_r3_scores(release, cell)
                parent_max_alias = max(
                    parent_max_alias,
                    float(np.max(np.abs(np.asarray(parent_local) - strict_local))),
                )
                continue
            parent_combined = p1.combine_with_common_detector(cell, parent_local)
            arrays[f"parent__{parent_id}__local_step_scores"] = np.asarray(parent_local, dtype="<f8")
            arrays[f"parent__{parent_id}__combined_step_scores"] = np.asarray(parent_combined, dtype="<f8")
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True, exist_ok=False)
        score_path = target / "scores.npz"
        score_sha = atomic_write_npz(score_path, arrays)
        record = {
            "schema": "reasoning-localization-phase2-atomic-remaining-cell-v1",
            "variant_id": variant, "cell_id": cell_id, "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id), "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids), "n_steps": len(candidate_local),
            "prepared_input": str(input_path), "prepared_input_sha256": sha256_file(input_path),
            "score_file": "scores.npz", "score_sha256": score_sha,
            "labels_seen_during_fit": False, "targets_accessed_during_fit": False,
            "diagnostics": dict(diagnostics),
        }
        record["payload_sha256"] = c1.payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({"cell_id": cell_id, "record_path": f"cells/{cell_id}/RECORD.json",
                        "record_sha256": sha256_file(target / "RECORD.json"), "score_sha256": score_sha})
        print(f"score-freeze {variant}: {cell_id} ({position}/8)", flush=True)
    require_sources(registry)
    freeze = {
        "schema": "reasoning-localization-phase2-atomic-remaining-score-freeze-v1",
        "status": "COMPLETE", "candidate": variant, "atomic_reference": REFERENCE,
        "cells": list(p2r.PB_CELLS), "labels_seen_during_fit": False,
        "targets_accessed_during_fit": False,
        "reference_local_alias_max_abs_error": alias_local,
        "reference_combined_alias_max_abs_error": alias_combined,
        "suffix_invariance_max_abs_error": suffix_error,
        "c8_iu29_stepmax_parent_alias_max_abs_error": parent_max_alias if variant == "C8_SELF_INNOV" else None,
        "input_manifest_sha256": sha256_file(manifest_path),
        "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "records": records,
    }
    freeze["payload_sha256"] = c1.payload_sha(freeze)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    return freeze


def rows_by_key(verified: Mapping[str, Mapping[str, Any]], labels: Mapping[str, Mapping[str, tuple[str, int]]], key: str) -> dict[str, list[dict[str, Any]]]:
    result = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
        record, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        row_ids = tuple(arrays["row_ids"].astype(str))
        if set(row_ids) != set(labels[cell_id]):
            raise AtomicRemainingError(f"{cell_id}: score/label population mismatch")
        offsets = np.asarray(arrays["segment_offsets"], dtype=np.int64)
        lengths = np.asarray(arrays["segment_lengths"], dtype=np.int64)
        scores = np.asarray(arrays[key], dtype=np.float64)
        for index, row_id in enumerate(row_ids):
            lo, hi = map(int, offsets[index:index + 2])
            group_id, first_error = labels[cell_id][row_id]
            result[record["model_id"]].append({"row_id": row_id, "group_id": group_id,
                "slice_id": record["slice_id"], "cell_id": cell_id, "model_id": record["model_id"],
                "first_error": first_error, "step_scores": scores[lo:hi].tolist(),
                "step_lengths": lengths[lo:hi].tolist()})
    return result


def _contrast(left: Mapping[str, Any], right: Mapping[str, Any], variant: str, comparator: str, metric: str, *, primary: bool) -> dict[str, Any]:
    right_cells = {str(row["cell_id"]): row for row in right["by_cell"]}
    left_point = float(np.mean([float(row[metric]) for row in left["by_cell"]]))
    right_point = float(np.mean([float(right_cells[row["cell_id"]][metric]) for row in left["by_cell"]]))
    draws = np.asarray(left["samples"][metric]) - np.asarray(right["samples"][metric])
    q = 0.025 / PRIMARY_COMPARISON_FAMILY if primary and metric == "official_macro_f1" else 0.025
    cell = {str(row["cell_id"]): float(row[metric]) - float(right_cells[str(row["cell_id"])][metric]) for row in left["by_cell"]}
    family = {name: float(np.mean([value for cell_id, value in cell.items() if right_cells[cell_id]["slice_id"] == name])) for name in p1.FAMILIES}
    eps = 1e-12
    return {"contrast_id": f"pb::{variant}::{comparator}::{metric}", "left_variant_id": variant,
        "right_variant_id": comparator, "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
        "source_metric_id": metric, "delta": left_point - right_point,
        "ci_low": float(np.quantile(draws, q)), "ci_high": float(np.quantile(draws, 1.0 - q)),
        "wins": sum(value > eps for value in cell.values()), "ties": sum(abs(value) <= eps for value in cell.values()),
        "losses": sum(value < -eps for value in cell.values()), "worst_unit_delta": min(cell.values()),
        "worst_unit_id": min(cell, key=cell.get), "family_wins": sum(value > eps for value in family.values()),
        "family_ties": sum(abs(value) <= eps for value in family.values()), "family_losses": sum(value < -eps for value in family.values()),
        "worst_family_delta": min(family.values()), "worst_family_id": min(family, key=family.get),
        "multiplicity_family_size": PRIMARY_COMPARISON_FAMILY if primary and metric == "official_macro_f1" else 1,
        "inference": ("Bonferroni simultaneous percentile interval across 16 opened C1--C8 primary contrasts"
                      if primary and metric == "official_macro_f1" else "unadjusted paired diagnostic percentile interval")}


def evaluate_scores(variant: str, release: Path, output: Path, registry: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    require_sources(registry)
    verified = c1.verified_scores(output, freeze)
    labels = p1._load_pb_labels(release)  # first target-bearing operation
    evaluation = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    arms = {
        REFERENCE: c1.evaluate_arm(REFERENCE, rows_by_key(verified, labels, "reference_combined_step_scores"), evaluation),
        variant: c1.evaluate_arm(variant, rows_by_key(verified, labels, "candidate_combined_step_scores"), evaluation),
    }
    for parent_id in EXTRA_PARENT.values() if variant == "C8_SELF_INNOV" else ():
        arms[parent_id] = c1.evaluate_arm(parent_id, rows_by_key(verified, labels, f"parent__{parent_id}__combined_step_scores"), evaluation)
    comparators = {REFERENCE: arms[REFERENCE], "R1_ENTROPY_TOP5": c1.comparator_top5()}
    contrasts = [_contrast(arms[variant], comparator, variant, comparator_id, metric, primary=True)
                 for comparator_id, comparator in comparators.items() for metric in p1.PB_METRICS]
    for parent_id in EXTRA_PARENT.values() if variant == "C8_SELF_INNOV" else ():
        contrasts.extend(_contrast(arms[variant], arms[parent_id], variant, parent_id, metric, primary=False) for metric in p1.PB_METRICS)
    primary = {row["right_variant_id"]: row for row in contrasts if row["metric_id"] == "macro_f1" and row["right_variant_id"] in comparators}
    by_metric = {(row["right_variant_id"], row["metric_id"]): row for row in contrasts}
    technical_failure = (freeze["reference_local_alias_max_abs_error"] > 1e-12
                         or freeze["reference_combined_alias_max_abs_error"] > 1e-12
                         or freeze["suffix_invariance_max_abs_error"] > 1e-12
                         or (variant == "C8_SELF_INNOV" and float(freeze["c8_iu29_stepmax_parent_alias_max_abs_error"]) > 1e-12))
    robustness_failure = min(float(row["worst_unit_delta"]) for row in primary.values()) < c1.HARD_WORST_CELL_BOUND
    eligible = bool(registry["promotion_eligible"])
    promotion = eligible and not technical_failure and not robustness_failure and all(
        float(row["delta"]) >= c1.BENEFIT and float(row["ci_low"]) > c1.BENEFIT
        and int(row["wins"]) + int(row["ties"]) >= 6
        and float(row["worst_unit_delta"]) >= c1.PROMOTION_WORST_CELL_BOUND
        and float(by_metric[(comparator, "first_error_exact")]["delta"]) >= c1.COMPONENT_BOUND
        and float(by_metric[(comparator, "clean_abstention_accuracy")]["delta"]) >= c1.COMPONENT_BOUND
        for comparator, row in primary.items())
    hard_failure = technical_failure or robustness_failure
    gates = [
        {"gate_id":"P2A_SCORE_FREEZE_COMPLETE","status":"PASS","observed":len(verified),"required":"8 cells","detail":"reference and candidate scores froze before labels"},
        {"gate_id":"P2A_LABEL_FIREWALL","status":"PASS","observed":"labels opened after score freeze","required":"no fit-side labels or targets","detail":"all transforms and fusion are target-free"},
        {"gate_id":"P2A_TOP10_LOCAL_ALIAS","status":"PASS" if freeze["reference_local_alias_max_abs_error"] <= 1e-12 else "HARD_FAIL","observed":freeze["reference_local_alias_max_abs_error"],"required":"<=1e-12","detail":"atomic top-ten local alias"},
        {"gate_id":"P2A_TOP10_COMBINED_ALIAS","status":"PASS" if freeze["reference_combined_alias_max_abs_error"] <= 1e-12 else "HARD_FAIL","observed":freeze["reference_combined_alias_max_abs_error"],"required":"<=1e-12","detail":"atomic top-ten combined alias"},
        {"gate_id":f"{variant}_SUFFIX_INVARIANCE","status":"PASS" if freeze["suffix_invariance_max_abs_error"] <= 1e-12 else "HARD_FAIL","observed":freeze["suffix_invariance_max_abs_error"],"required":"<=1e-12","detail":"fixed-map prefix replay"},
        {"gate_id":f"{variant}_WORST_CELL_HARD_BOUND","status":"HARD_FAIL" if robustness_failure else "PASS","observed":min(float(row["worst_unit_delta"]) for row in primary.values()),"required":f">={c1.HARD_WORST_CELL_BOUND}","detail":"minimum across both required comparators"},
        {"gate_id":f"{variant}_PROMOTION_ELIGIBILITY","status":"PASS" if eligible else "NOT_ELIGIBLE","observed":str(eligible).lower(),"required":"registered promotion-eligible branch","detail":registry["eligibility_reason"]},
        {"gate_id":f"{variant}_PREMISE_PROMOTION","status":"PASS" if promotion else "FAIL","observed":str(promotion).lower(),"required":"eligible and all promotion gates pass","detail":"atomic survivor gate"},
    ]
    if variant == "C8_SELF_INNOV":
        error = float(freeze["c8_iu29_stepmax_parent_alias_max_abs_error"])
        gates.append({"gate_id":"C8_IU29_STEPMAX_PARENT_ALIAS","status":"PASS" if error <= 1e-12 else "HARD_FAIL","observed":error,"required":"<=1e-12","detail":"original-only IU29 reconstruction aliases frozen R3 step-max"})
    flips, flip_summary = c1.prediction_flips(arms[variant]["decisions"], arms[REFERENCE]["decisions"])
    eval_root = output / "evaluation"
    eval_root.mkdir(parents=True, exist_ok=False)
    arm_order = (REFERENCE, variant) + tuple(parent for parent in EXTRA_PARENT.values() if parent in arms)
    c1.write_csv(eval_root / "PROCESSBENCH_DECISIONS.csv", [row for arm in arm_order for row in arms[arm]["decisions"]])
    c1.write_csv(eval_root / "PROCESSBENCH_BY_CELL.csv", [row for arm in arm_order for row in arms[arm]["by_cell"]])
    c1.write_csv(eval_root / "PROCESSBENCH_PANELS.csv", [row for arm in arm_order for row in arms[arm]["panels"]])
    atomic_write_npz(eval_root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz", {f"{arm}__{metric}": values for arm in arm_order for metric, values in arms[arm]["samples"].items()})
    atomic_write_json(eval_root / "CALIBRATION_LEDGERS.json", {"schema":"reasoning-localization-phase2-atomic-remaining-calibration-v1","arms":{arm:arms[arm]["ledgers"] for arm in arm_order}})
    c1.write_csv(eval_root / "PAIRWISE_CONTRASTS.csv", contrasts)
    c1.write_csv(eval_root / "STEP_LENGTH_STRATA.csv", p2r._length_strata(arms[variant]["decisions"], arms[variant]["by_cell"]))
    c1.write_csv(eval_root / "SELECTED_STEP_LENGTH.csv", p2r._selected_length_distribution(arms[variant]["decisions"]))
    c1.write_csv(eval_root / "PREDICTION_FLIPS.csv", flips)
    c1.write_csv(eval_root / "PREDICTION_FLIP_SUMMARY.csv", flip_summary)
    c1.write_csv(eval_root / "GATES.csv", gates)
    candidate_panel = next(row for row in arms[variant]["panels"] if row["metric_id"] == "official_macro_f1")
    reference_panel = next(row for row in arms[REFERENCE]["panels"] if row["metric_id"] == "official_macro_f1")
    status = "HARD_FAIL" if hard_failure else "COMPLETE"
    summary = {"schema":"reasoning-localization-phase2-atomic-remaining-evaluation-v1","variant_id":variant,
        "status":status,"promotion_eligible":eligible,"premise_gate_passed":promotion,
        "candidate_macro_f1":candidate_panel["value"],"candidate_macro_f1_ci":[candidate_panel["ci_low"],candidate_panel["ci_high"]],
        "atomic_reference_macro_f1":reference_panel["value"],"primary_contrasts":primary,
        "prediction_flips_vs_atomic_reference":sum(row["changed"] == "true" for row in flips),
        "reference_local_alias_max_abs_error":freeze["reference_local_alias_max_abs_error"],
        "reference_combined_alias_max_abs_error":freeze["reference_combined_alias_max_abs_error"],
        "suffix_invariance_max_abs_error":freeze["suffix_invariance_max_abs_error"],
        "bootstrap_draws":p1.BOOTSTRAP_DRAWS,"bootstrap_seed":p1.BOOTSTRAP_SEED,
        "peak_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    summary["payload_sha256"] = c1.payload_sha(summary)
    atomic_write_json(eval_root / "SUMMARY.json", summary)
    outputs = [path.name for path in sorted(eval_root.iterdir())]
    manifest = {"schema":"reasoning-localization-phase2-atomic-remaining-evaluation-manifest-v1","variant_id":variant,"status":status,
        "score_freeze_sha256":sha256_file(output / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
        "execution_registry_sha256":sha256_file(Path(registry["registry_path"])),
        "outputs":[{"path":name,"sha256":sha256_file(eval_root/name),"bytes":(eval_root/name).stat().st_size} for name in outputs]}
    manifest["payload_sha256"] = c1.payload_sha(manifest)
    atomic_write_json(eval_root / "EVALUATION_MANIFEST.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--release", type=Path, default=p1.DEFAULT_RELEASE)
    args = parser.parse_args()
    variant = args.variant
    release = args.release.resolve()
    reg_path = registry_path(variant).resolve()
    output = output_root(variant).resolve()
    registry = load_registry(reg_path, release, variant)
    registry["registry_path"] = str(reg_path)
    started = time.perf_counter()
    freeze = freeze_scores(variant, release, output, registry)
    summary = evaluate_scores(variant, release, output, registry, freeze)
    run = {"schema":"reasoning-localization-phase2-atomic-remaining-run-v1","variant_id":variant,
        "status":summary["status"],"execution_registry_sha256":sha256_file(reg_path),
        "runner_sha256":sha256_file(Path(__file__).resolve()),
        "score_freeze_manifest_sha256":sha256_file(output/"score_freeze/SCORE_FREEZE_MANIFEST.json"),
        "evaluation_manifest_sha256":sha256_file(output/"evaluation/EVALUATION_MANIFEST.json"),
        "elapsed_seconds":time.perf_counter()-started,"summary":summary}
    run["payload_sha256"] = c1.payload_sha(run)
    atomic_write_json(output/"RUN_MANIFEST.json", run)
    print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
