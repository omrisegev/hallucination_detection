#!/usr/bin/env python3
"""Run the frozen Phase-3 compact-view deployed U-PCR ladder.

All four score families are frozen before ProcessBench labels are imported.
P3D3 is a predeclared distributional control: its reported point and paired
bootstrap draws are the arithmetic mean over 20 frozen random-mask arms.
"""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    atomic_write_npz,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_cell,
    validate_fit_manifest,
)
from spectral_utils.token_local_fusion import (  # noqa: E402
    IU_CONFIG,
    fit_local_equal_family,
    prepare_localization_cell,
)
from spectral_utils.upcr import UPCRResult, upcr_fit  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_remaining as atomic  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization import run_phase3_compact_fusion as p3  # noqa: E402
from scripts.reasoning_localization.register_phase3_deployed_upcr_prune_refit import (  # noqa: E402
    EXPERIMENT_ID,
    VARIANT_IDS,
)

P3D0, P3D1, P3D2, P3D3 = VARIANT_IDS
H2_PARENT = "P3A_H2_EQUAL_OUTER_REFERENCE"
H0_REFERENCE = "P3_H0_REFERENCE"
ROOT = p1.PROGRAM_ROOT / "phase_3/deployed_upcr_prune_refit"
OUTPUT = ROOT / "p3d_compact_view_ladder_v1"
REGISTRY = ROOT / "P3D_COMPACT_VIEW_EXECUTION_REGISTRY.json"
SOURCE_H2 = (
    p1.PROGRAM_ROOT
    / "phase_2/diagnostic/h3_reliability_fusion_v1/score_freeze/cells"
)

DEPLOYED_CONFIG = {
    "loss": "l2",
    "scale_ratio": 0.25,
    "n_components": 1,
    "auto_components": True,
    "lambda2_threshold": 0.1,
    "g2_projection_k": 1,
    "exclusion": True,
    "min_frac": 0.05,
    "exclude_frac": 3.0,
    "simple_avg_fallback": True,
    "min_experts_for_eq21": 5,
    "difficulty_gate": False,
    "on_abstain": "flag",
    "recompute_after_exclusion": True,
    "g2_grid": 300,
}
FULLPOOL_CONFIG = dict(IU_CONFIG)
NO_PRUNE_ALIAS_CONFIG = dict(IU_CONFIG)
N_FOLDS = 5
RANDOM_MASK_SEEDS = tuple(range(2026083101, 2026083121))
PRIMARY_CONTRASTS = (
    (P3D0, H2_PARENT),
    (P3D1, P3D0),
    (P3D1, H2_PARENT),
    (P3D1, P3D2),
    (P3D2, P3D3),
    (P3D1, P3D3),
)
MULTIPLICITY_FAMILY_SIZE = len(PRIMARY_CONTRASTS)
PRACTICAL_BENEFIT = 0.003
PRACTICAL_HARM = -0.003
EXACT_FLOOR = -0.010
CLEAN_FLOOR = -0.010
WORST_CELL_FLOOR = -0.020
MASK_MEAN_JACCARD_FLOOR = 0.60


class Phase3DeployedError(RuntimeError):
    """Fail-closed contract error for P3D."""


def _member_matrix(cell: Any) -> tuple[Any, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Return confidence-oriented raw H2 member views, including raw C7."""

    prep = prepare_localization_cell(cell)
    kept = np.asarray(prep.values[:, prep.keep], dtype=np.float64)
    names = list(prep.kept_stream_names)
    families = list(prep.kept_family_names)

    selected: list[int] = []
    selected_names: list[str] = []
    selected_families: list[str] = []
    for index, (name, family) in enumerate(zip(names, families)):
        eligible = family in {
            "entropy_level",
            "entropy_dynamics",
            "partition_energy",
            "topk_distribution",
        }
        if family == "partition_energy" and name == "energy_series":
            eligible = False
        if eligible:
            selected.append(index)
            selected_names.append(name)
            selected_families.append(family)

    entropy_risk = atomic.primitive_risks(cell)["entropy"]
    c7_risk = atomic.response_map(entropy_risk, cell.token_offsets, atomic.edis_onset)
    dynamics_end = max(
        index for index, family in enumerate(selected_families)
        if family == "entropy_dynamics"
    ) + 1
    raw = np.column_stack(
        [kept[:, selected[:dynamics_end]], -c7_risk, kept[:, selected[dynamics_end:]]]
    )
    selected_names.insert(dynamics_end, "C7_EDIS_ONSET")
    selected_families.insert(dynamics_end, "entropy_dynamics")
    if raw.shape[1] < 5 or not np.isfinite(c7_risk).all():
        raise Phase3DeployedError("compact member matrix is malformed")
    return prep, raw, tuple(selected_names), tuple(selected_families)


def _fold_standardize(
    raw: np.ndarray,
    donor_indices: np.ndarray,
    held_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    donor = np.asarray(raw[donor_indices], dtype=np.float64)
    medians = np.nanmedian(donor, axis=0)
    donor_clean = np.where(np.isfinite(donor), donor, medians[None, :])
    mean = donor_clean.mean(axis=0)
    std = donor_clean.std(axis=0)
    if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 1e-8):
        raise Phase3DeployedError("donor fold has a degenerate member view")
    held = np.asarray(raw[held_indices], dtype=np.float64)
    held_clean = np.where(np.isfinite(held), held, mean[None, :])
    return (
        (donor_clean - mean[None, :]) / std[None, :],
        (held_clean - mean[None, :]) / std[None, :],
        {
            "n_donor_tokens": int(len(donor_indices)),
            "n_held_tokens": int(len(held_indices)),
            "donor_mean": mean.tolist(),
            "donor_std": std.tolist(),
        },
    )


def _oriented_weights(
    model: UPCRResult,
    donor_standardized: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    weights = np.asarray(model.w, dtype=np.float64).copy()
    anchor = donor_standardized.mean(axis=1)
    score = donor_standardized @ weights
    corr = float(np.corrcoef(score, anchor)[0, 1])
    flipped = bool(np.isfinite(corr) and corr < 0.0)
    if flipped:
        weights *= -1.0
    return weights, corr, flipped


def _model_diagnostics(
    model: UPCRResult,
    weights: np.ndarray,
    corr: float,
    flipped: bool,
) -> dict[str, Any]:
    return {
        "weights": weights.tolist(),
        "rho_hat_full": np.asarray(model.rho_hat_full).tolist(),
        "rho_hat_survivor_refit": np.asarray(model.rho_hat).tolist(),
        "keep_mask": np.asarray(model.keep, dtype=int).tolist(),
        "n_kept": int(np.asarray(model.keep).sum()),
        "used_simple_average": bool(model.used_simple_average),
        "n_components_used": int(model.n_components_used),
        "abstained": bool(model.abstained),
        "g2_hat": float(model.g2_hat),
        "g2_at_ceiling": bool(model.g2_at_ceiling),
        "g2_frac_of_var_y": float(model.g2_frac_of_var_y),
        "projection_residual": float(model.proj_residual),
        "lambda2_frac": float(model.lambda2_frac),
        "confidence_anchor_correlation": corr,
        "orientation_flipped": flipped,
    }


def _random_seed(base_seed: int, cell_id: str, fold: int) -> int:
    payload = f"{base_seed}|{cell_id}|{fold}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def _mask_stability(masks: Sequence[np.ndarray], names: Sequence[str]) -> dict[str, Any]:
    stacked = np.stack(masks).astype(bool)
    jaccards = []
    for left, right in combinations(stacked, 2):
        union = np.logical_or(left, right).sum()
        jaccards.append(float(np.logical_and(left, right).sum() / union) if union else 1.0)
    return {
        "selection_frequency": {
            name: float(value) for name, value in zip(names, stacked.mean(axis=0))
        },
        "mean_pairwise_jaccard": float(np.mean(jaccards)),
        "min_pairwise_jaccard": float(np.min(jaccards)),
        "fold_keep_counts": stacked.sum(axis=1).astype(int).tolist(),
    }


def _load_registry(release: Path) -> dict[str, Any]:
    row = json.loads(REGISTRY.read_text(encoding="utf-8"))
    required = {
        "schema": "reasoning-localization-p3d-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": EXPERIMENT_ID,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "variant_order": list(VARIANT_IDS),
        "random_mask_seeds": list(RANDOM_MASK_SEEDS),
        "primary_contrasts": [list(pair) for pair in PRIMARY_CONTRASTS],
        "multiplicity_family_size": MULTIPLICITY_FAMILY_SIZE,
    }
    for key, value in required.items():
        if row.get(key) != value:
            raise Phase3DeployedError(f"execution registry mismatch: {key}")
    if Path(row["release_root"]).resolve() != release.resolve():
        raise Phase3DeployedError("release mismatch")
    return row


def freeze(release: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    score_root = OUTPUT / "score_freeze"
    score_root.mkdir(parents=True)
    input_root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    records: list[dict[str, Any]] = []
    max_alias_error = 0.0
    max_h2_alias_error = 0.0

    for position, cell_id in enumerate(p2r.PB_CELLS, start=1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        prep, raw, member_names, member_families = _member_matrix(cell)
        if list(member_names) != registry["member_names"]:
            raise Phase3DeployedError(f"member roster drift in {cell_id}")
        if list(member_families) != registry["member_families"]:
            raise Phase3DeployedError(f"member family drift in {cell_id}")

        token_owner = np.repeat(
            np.arange(len(cell.row_ids), dtype=np.int64),
            np.diff(np.asarray(cell.token_offsets, dtype=np.int64)),
        )
        d0_token = np.full(len(raw), np.nan, dtype=np.float64)
        d1_token = np.full(len(raw), np.nan, dtype=np.float64)
        d2_token = np.full(len(raw), np.nan, dtype=np.float64)
        random_token = np.full((len(RANDOM_MASK_SEEDS), len(raw)), np.nan, dtype=np.float64)
        fold_diagnostics: list[dict[str, Any]] = []
        deployed_masks: list[np.ndarray] = []

        for fold in range(N_FOLDS):
            held_rows = np.flatnonzero(np.asarray(prep.row_folds) == fold)
            held_indices = np.flatnonzero(np.isin(token_owner, held_rows))
            fit_folds = np.asarray(prep.row_folds)[np.asarray(prep.fit_row_indices)]
            donor_indices = np.asarray(prep.fit_indices)[fit_folds != fold]
            donor, held, scale_diag = _fold_standardize(raw, donor_indices, held_indices)

            d0_model = upcr_fit(donor.T, **FULLPOOL_CONFIG)
            d0_weights, d0_corr, d0_flipped = _oriented_weights(d0_model, donor)
            d0_token[held_indices] = -(held @ d0_weights)

            alias_model = upcr_fit(donor.T, **NO_PRUNE_ALIAS_CONFIG)
            alias_weights, _, _ = _oriented_weights(alias_model, donor)
            alias_error = float(np.max(np.abs((held @ d0_weights) - (held @ alias_weights))))
            max_alias_error = max(max_alias_error, alias_error)

            d1_model = upcr_fit(donor.T, **DEPLOYED_CONFIG)
            d1_weights, d1_corr, d1_flipped = _oriented_weights(d1_model, donor)
            d1_token[held_indices] = -(held @ d1_weights)
            keep = np.asarray(d1_model.keep, dtype=bool)
            deployed_masks.append(keep.copy())
            d2_token[held_indices] = -held[:, keep].mean(axis=1)

            random_masks = []
            for seed_index, base_seed in enumerate(RANDOM_MASK_SEEDS):
                rng = np.random.default_rng(_random_seed(base_seed, cell_id, fold))
                mask = np.zeros(raw.shape[1], dtype=bool)
                mask[rng.choice(raw.shape[1], size=int(keep.sum()), replace=False)] = True
                random_masks.append(mask.astype(int).tolist())
                random_token[seed_index, held_indices] = -held[:, mask].mean(axis=1)

            fold_diagnostics.append({
                "fold": fold,
                **scale_diag,
                "fullpool": _model_diagnostics(
                    d0_model, d0_weights, d0_corr, d0_flipped
                ),
                "deployed": _model_diagnostics(
                    d1_model, d1_weights, d1_corr, d1_flipped
                ),
                "no_prune_alias_max_abs_error": alias_error,
                "random_masks": random_masks,
            })

        for array in (d0_token, d1_token, d2_token, random_token):
            if not np.isfinite(array).all():
                raise Phase3DeployedError(f"incomplete cross-fit score in {cell_id}")

        h0_token = np.asarray(fit_local_equal_family(prep).token_risk, dtype=np.float64)
        h0_local = p1.topk_step_mean(
            h0_token, cell.segment_starts, cell.segment_ends, k=10
        )
        h2_local = load_npz_no_pickle(SOURCE_H2 / cell_id / "scores.npz")["h2_local"]
        h2_rebuilt = p1.topk_step_mean(
            p3._h2_family_matrix(cell)[1], cell.segment_starts, cell.segment_ends, k=10
        )
        max_h2_alias_error = max(
            max_h2_alias_error, float(np.max(np.abs(h2_local - h2_rebuilt)))
        )
        arrays: dict[str, np.ndarray] = {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(
                cell.segment_ends - cell.segment_starts, dtype="<i8"
            ),
            "h0_combined": p1.combine_with_common_detector(cell, h0_local),
            "h2_local": np.asarray(h2_local, dtype=np.float64),
            "p3d0_local": p1.topk_step_mean(
                d0_token, cell.segment_starts, cell.segment_ends, k=10
            ),
            "p3d1_local": p1.topk_step_mean(
                d1_token, cell.segment_starts, cell.segment_ends, k=10
            ),
            "p3d2_local": p1.topk_step_mean(
                d2_token, cell.segment_starts, cell.segment_ends, k=10
            ),
        }
        for seed_index in range(len(RANDOM_MASK_SEEDS)):
            arrays[f"p3d3_random_{seed_index:02d}_local"] = p1.topk_step_mean(
                random_token[seed_index], cell.segment_starts, cell.segment_ends, k=10
            )

        target = score_root / "cells" / cell_id
        target.mkdir(parents=True)
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        stability = _mask_stability(deployed_masks, member_names)
        record = {
            "schema": "reasoning-localization-p3d-cell-v1",
            "experiment_id": EXPERIMENT_ID,
            "variant_ids": list(VARIANT_IDS),
            "cell_id": cell_id,
            "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id),
            "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids),
            "n_member_views": len(member_names),
            "member_names": list(member_names),
            "member_families": list(member_families),
            "fit_mode": "five-fold grouped cross-fit; held responses projection-only",
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
            "fold_diagnostics": fold_diagnostics,
            "mask_stability": stability,
            "score_sha256": score_sha,
            "prepared_input": str(input_path),
            "prepared_input_sha256": sha256_file(input_path),
        }
        record["payload_sha256"] = c1.payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({
            "cell_id": cell_id,
            "record_path": f"cells/{cell_id}/RECORD.json",
            "record_sha256": sha256_file(target / "RECORD.json"),
            "score_sha256": score_sha,
        })
        print(f"score-freeze P3D0-P3D3: {cell_id} ({position}/8)", flush=True)

    if max_alias_error > 1e-12:
        raise Phase3DeployedError(f"no-pruning alias failed: {max_alias_error}")
    if max_h2_alias_error > 1e-12:
        raise Phase3DeployedError(f"H2 source alias failed: {max_h2_alias_error}")
    result = {
        "schema": "reasoning-localization-p3d-score-freeze-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "variant_ids": list(VARIANT_IDS),
        "cells": list(p2r.PB_CELLS),
        "records": records,
        "no_pruning_alias_max_abs_error": max_alias_error,
        "h2_parent_alias_max_abs_error": max_h2_alias_error,
        "labels_seen_during_fit": False,
        "execution_registry_sha256": sha256_file(REGISTRY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["payload_sha256"] = c1.payload_sha(result)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", result)
    return result


def _verified(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for item in manifest["records"]:
        record_path = OUTPUT / "score_freeze" / item["record_path"]
        score_path = record_path.parent / "scores.npz"
        if sha256_file(record_path) != item["record_sha256"]:
            raise Phase3DeployedError("record hash mismatch")
        if sha256_file(score_path) != item["score_sha256"]:
            raise Phase3DeployedError("score hash mismatch")
        result[item["cell_id"]] = {
            "record": json.loads(record_path.read_text(encoding="utf-8")),
            "arrays": load_npz_no_pickle(score_path),
        }
    return result


def _rows(
    verified: Mapping[str, Any],
    labels: Mapping[str, Any],
    key: str,
) -> dict[str, list[dict[str, Any]]]:
    result = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
        record = verified[cell_id]["record"]
        arrays = verified[cell_id]["arrays"]
        offsets = arrays["segment_offsets"]
        lengths = arrays["segment_lengths"]
        for index, row_id in enumerate(arrays["row_ids"].astype(str)):
            lo, hi = map(int, offsets[index:index + 2])
            group_id, first_error = labels[cell_id][row_id]
            result[record["model_id"]].append({
                "row_id": row_id,
                "group_id": group_id,
                "slice_id": record["slice_id"],
                "cell_id": cell_id,
                "model_id": record["model_id"],
                "first_error": first_error,
                "step_scores": arrays[key][lo:hi].tolist(),
                "step_lengths": lengths[lo:hi].tolist(),
            })
    return result


def _aggregate_random(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    panels = []
    for metric in p1.PB_METRICS:
        rows = [{row["metric_id"]: row for row in arm["panels"]}[metric] for arm in arms]
        samples = np.stack([arm["samples"][metric] for arm in arms]).mean(axis=0)
        panels.append({
            "arm_id": P3D3,
            "population_id": "current_common_eight_qwen",
            "metric_id": metric,
            "value": float(np.mean([row["value"] for row in rows])),
            "ci_low": float(np.quantile(samples, 0.025)),
            "ci_high": float(np.quantile(samples, 0.975)),
            "n_rows": 6800,
            "n_groups": 3400,
        })
    by_cell = []
    for cell_id in p2r.PB_CELLS:
        rows = [
            next(row for row in arm["by_cell"] if row["cell_id"] == cell_id)
            for arm in arms
        ]
        base = {key: value for key, value in rows[0].items() if key not in p1.PB_METRICS}
        base["arm_id"] = P3D3
        for metric in p1.PB_METRICS:
            base[metric] = float(np.mean([row[metric] for row in rows]))
        by_cell.append(base)
    return {
        "decisions": None,
        "by_cell": by_cell,
        "samples": {
            metric: np.stack([arm["samples"][metric] for arm in arms]).mean(axis=0)
            for metric in p1.PB_METRICS
        },
        "panels": panels,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _statistical_status(delta: float, ci_low: float, ci_high: float) -> str:
    if ci_low > PRACTICAL_BENEFIT:
        return "SUPPORTED_IMPROVEMENT"
    if ci_high < PRACTICAL_HARM:
        return "SUPPORTED_HARM"
    if delta > 0.0 and ci_low <= 0.0:
        return "PROMISING_UNCONFIRMED"
    if ci_low > PRACTICAL_HARM:
        return "NONINFERIOR_ONLY"
    return "INCONCLUSIVE"


def _contrast(
    left: str,
    right: str,
    metric: str,
    arms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    left_panel = {row["metric_id"]: row for row in arms[left]["panels"]}[metric]
    right_panel = {row["metric_id"]: row for row in arms[right]["panels"]}[metric]
    draws = np.asarray(arms[left]["samples"][metric]) - np.asarray(arms[right]["samples"][metric])
    q = 0.025 / MULTIPLICITY_FAMILY_SIZE if metric == "official_macro_f1" else 0.025
    left_cells = {row["cell_id"]: row for row in arms[left]["by_cell"]}
    right_cells = {row["cell_id"]: row for row in arms[right]["by_cell"]}
    cells = {
        cell_id: float(left_cells[cell_id][metric]) - float(right_cells[cell_id][metric])
        for cell_id in p2r.PB_CELLS
    }
    delta = float(left_panel["value"] - right_panel["value"])
    ci_low = float(np.quantile(draws, q))
    ci_high = float(np.quantile(draws, 1.0 - q))
    return {
        "contrast_id": f"pb::{left}::{right}::{metric}",
        "left_variant_id": left,
        "right_variant_id": right,
        "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "statistical_status": _statistical_status(delta, ci_low, ci_high),
        "wins": sum(value > 1e-12 for value in cells.values()),
        "ties": sum(abs(value) <= 1e-12 for value in cells.values()),
        "losses": sum(value < -1e-12 for value in cells.values()),
        "worst_unit_delta": min(cells.values()),
        "worst_unit_id": min(cells, key=cells.get),
        "multiplicity_family_size": MULTIPLICITY_FAMILY_SIZE if metric == "official_macro_f1" else 1,
        "practical_benefit_bound": PRACTICAL_BENEFIT,
        "practical_harm_bound": PRACTICAL_HARM,
    }


def _prediction_flips(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, int]:
    right_by_id = {
        (row["cell_id"], row["row_id"]): row for row in right["decisions"]
    }
    counts = {"total": 0, "left_correct_right_wrong": 0, "left_wrong_right_correct": 0}
    for row in left["decisions"]:
        other = right_by_id[(row["cell_id"], row["row_id"])]
        if int(row["prediction_step"]) == int(other["prediction_step"]):
            continue
        counts["total"] += 1
        target = int(row["true_first_error"])
        if int(row["prediction_step"]) == target and int(other["prediction_step"]) != target:
            counts["left_correct_right_wrong"] += 1
        if int(row["prediction_step"]) != target and int(other["prediction_step"]) == target:
            counts["left_wrong_right_correct"] += 1
    return counts


def _write_plot(path: Path, panels: Sequence[Mapping[str, Any]], contrasts: Sequence[Mapping[str, Any]]) -> None:
    order = [H2_PARENT, P3D0, P3D1, P3D2, P3D3]
    values = {
        row["arm_id"]: float(row["value"])
        for row in panels if row["metric_id"] == "official_macro_f1" and row["arm_id"] in order
    }
    primary = [row for row in contrasts if row["metric_id"] == "macro_f1"]
    width, height = 1040, 590
    x0, x1 = 260, 980
    lo, hi = min(values.values()) - 0.005, max(values.values()) + 0.005
    scale = lambda value: x0 + (value - lo) / (hi - lo) * (x1 - x0)
    colors = {H2_PARENT: "#64748b", P3D0: "#2563eb", P3D1: "#7c3aed", P3D2: "#0f766e", P3D3: "#d97706"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:system-ui,sans-serif;fill:#172033}.t{font-size:22px;font-weight:700}.s{font-size:13px}.l{font-size:14px;font-weight:600}</style>',
        '<text x="28" y="36" class="t">P3D compact-view prune/refit ladder</text>',
        '<text x="28" y="60" class="s">ProcessBench macro F1; H0 abstention decision and top-10 reducer frozen</text>',
    ]
    for tick in np.linspace(lo, hi, 6):
        x = scale(float(tick))
        lines.append(f'<line x1="{x:.1f}" y1="85" x2="{x:.1f}" y2="330" stroke="#e2e8f0"/>')
        lines.append(f'<text x="{x:.1f}" y="350" text-anchor="middle" class="s">{tick:.3f}</text>')
    for index, variant in enumerate(order):
        y = 112 + index * 47
        x = scale(values[variant])
        lines.append(f'<text x="28" y="{y+5}" class="l">{variant}</text>')
        lines.append(f'<line x1="{x0}" y1="{y}" x2="{x:.1f}" y2="{y}" stroke="{colors[variant]}" stroke-width="8"/>')
        lines.append(f'<circle cx="{x:.1f}" cy="{y}" r="7" fill="{colors[variant]}"/>')
        lines.append(f'<text x="{min(x+12,995):.1f}" y="{y+5}" class="l">{values[variant]:.6f}</text>')
    lines.append('<text x="28" y="395" class="t">Registered paired deltas</text>')
    for index, row in enumerate(primary):
        y = 425 + index * 25
        label = f'{row["left_variant_id"]} - {row["right_variant_id"]}'
        value = f'{row["delta"]:+.5f} [{row["ci_low"]:+.5f}, {row["ci_high"]:+.5f}]'
        lines.append(f'<text x="28" y="{y}" class="s">{label}</text>')
        lines.append(f'<text x="680" y="{y}" class="s">{value}  {row["statistical_status"]}</text>')
    lines.append('</svg>')
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = _verified(manifest)
    labels = p1._load_pb_labels(release)  # Explicitly after every P3D score hash is frozen.
    evaluator = importlib.import_module(
        "spectral_utils.reconstruction_benchmark.localization_evaluation"
    )
    h0 = c1.evaluate_arm(H0_REFERENCE, _rows(verified, labels, "h0_combined"), evaluator)
    arms: dict[str, Mapping[str, Any]] = {H0_REFERENCE: h0}
    arm_keys = {
        H2_PARENT: "h2_local",
        P3D0: "p3d0_local",
        P3D1: "p3d1_local",
        P3D2: "p3d2_local",
    }
    for arm, key in arm_keys.items():
        arms[arm] = p3._rerank(arm, h0, _rows(verified, labels, key), evaluator)

    random_arms = []
    for seed_index in range(len(RANDOM_MASK_SEEDS)):
        arm = f"{P3D3}__SEED_{seed_index:02d}"
        random_arms.append(
            p3._rerank(
                arm,
                h0,
                _rows(verified, labels, f"p3d3_random_{seed_index:02d}_local"),
                evaluator,
            )
        )
    arms[P3D3] = _aggregate_random(random_arms)

    h0_abstain = {
        (row["cell_id"], row["row_id"]): int(row["prediction_step"]) == -1
        for row in h0["decisions"]
    }
    abstention_mismatches = {}
    for arm in (H2_PARENT, P3D0, P3D1, P3D2):
        abstention_mismatches[arm] = sum(
            (int(row["prediction_step"]) == -1)
            != h0_abstain[(row["cell_id"], row["row_id"])]
            for row in arms[arm]["decisions"]
        )
    for index, random_arm in enumerate(random_arms):
        abstention_mismatches[f"random_{index:02d}"] = sum(
            (int(row["prediction_step"]) == -1)
            != h0_abstain[(row["cell_id"], row["row_id"])]
            for row in random_arm["decisions"]
        )
    if any(abstention_mismatches.values()):
        raise Phase3DeployedError(f"H0 abstention alias failed: {abstention_mismatches}")

    contrasts = [
        _contrast(left, right, metric, arms)
        for left, right in PRIMARY_CONTRASTS
        for metric in p1.PB_METRICS
    ]
    mask_rows = []
    cell_jaccards = []
    for cell_id in p2r.PB_CELLS:
        stability = verified[cell_id]["record"]["mask_stability"]
        cell_jaccards.append(float(stability["mean_pairwise_jaccard"]))
        for name, frequency in stability["selection_frequency"].items():
            mask_rows.append({
                "cell_id": cell_id,
                "member_name": name,
                "selection_frequency": frequency,
                "mean_pairwise_jaccard": stability["mean_pairwise_jaccard"],
                "min_pairwise_jaccard": stability["min_pairwise_jaccard"],
            })

    by_cell_rows = [row for arm in arms.values() for row in arm["by_cell"]]
    panel_rows = [row for arm in arms.values() for row in arm["panels"]]
    evaluation_root = OUTPUT / "evaluation"
    evaluation_root.mkdir()
    _write_csv(evaluation_root / "PROCESSBENCH_BY_CELL.csv", by_cell_rows)
    _write_csv(evaluation_root / "PROCESSBENCH_PANELS.csv", panel_rows)
    _write_csv(evaluation_root / "PAIRWISE_CONTRASTS.csv", contrasts)
    _write_csv(evaluation_root / "MASK_STABILITY.csv", mask_rows)

    primary = next(
        row for row in contrasts
        if row["left_variant_id"] == P3D1
        and row["right_variant_id"] == P3D0
        and row["metric_id"] == "macro_f1"
    )
    vs_h2 = next(
        row for row in contrasts
        if row["left_variant_id"] == P3D1
        and row["right_variant_id"] == H2_PARENT
        and row["metric_id"] == "macro_f1"
    )
    exact = next(
        row for row in contrasts
        if row["left_variant_id"] == P3D1
        and row["right_variant_id"] == P3D0
        and row["metric_id"] == "first_error_exact"
    )
    clean = next(
        row for row in contrasts
        if row["left_variant_id"] == P3D1
        and row["right_variant_id"] == P3D0
        and row["metric_id"] == "clean_abstention_accuracy"
    )
    refit_control = next(
        row for row in contrasts
        if row["left_variant_id"] == P3D1
        and row["right_variant_id"] == P3D2
        and row["metric_id"] == "macro_f1"
    )
    mask_control = next(
        row for row in contrasts
        if row["left_variant_id"] == P3D2
        and row["right_variant_id"] == P3D3
        and row["metric_id"] == "macro_f1"
    )
    gates = {
        "primary_practical_improvement": primary["ci_low"] > PRACTICAL_BENEFIT,
        "strongest_compact_parent_improvement": vs_h2["ci_low"] > PRACTICAL_BENEFIT,
        "exact_error_floor": exact["delta"] >= EXACT_FLOOR,
        "clean_abstention_floor": clean["delta"] >= CLEAN_FLOOR,
        "worst_cell_floor": primary["worst_unit_delta"] >= WORST_CELL_FLOOR,
        "mask_mean_jaccard_floor": min(cell_jaccards) >= MASK_MEAN_JACCARD_FLOOR,
        "refit_beats_equal_mask": refit_control["ci_low"] > 0.0,
        "rho_mask_beats_random_mask": mask_control["ci_low"] > 0.0,
        "h0_abstention_alias": not any(abstention_mismatches.values()),
    }
    summary = {
        "schema": "reasoning-localization-p3d-evaluation-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "variant_ids": list(VARIANT_IDS),
        "primary_contrast": primary,
        "candidate_vs_h2": vs_h2,
        "mask_mean_jaccard_by_cell": dict(zip(p2r.PB_CELLS, cell_jaccards)),
        "abstention_mismatches": abstention_mismatches,
        "prediction_flips": {
            "P3D1_vs_P3D0": _prediction_flips(arms[P3D1], arms[P3D0]),
            "P3D1_vs_H2": _prediction_flips(arms[P3D1], arms[H2_PARENT]),
        },
        "gates": gates,
        "promotion_passed": all(gates.values()),
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "random_mask_control": {
            "n_masks": len(RANDOM_MASK_SEEDS),
            "aggregation": "arithmetic mean of per-mask metric and paired bootstrap draw",
        },
    }
    summary["payload_sha256"] = c1.payload_sha(summary)
    atomic_write_json(evaluation_root / "SUMMARY.json", summary)
    _write_plot(evaluation_root / "P3D_RESULTS.svg", panel_rows, contrasts)
    return summary


def main() -> None:
    started = time.perf_counter()
    release = p1.DEFAULT_RELEASE.resolve()
    registry = _load_registry(release)
    frozen = freeze(release, registry)
    summary = evaluate(release, frozen)
    atomic_write_json(OUTPUT / "RUN_COMPLETE.json", {
        "schema": "reasoning-localization-p3d-run-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT_ID,
        "variant_ids": list(VARIANT_IDS),
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
