#!/usr/bin/env python3
"""Retrospective 8/16 evaluation of a B3 confidence-envelope residual.

The target-free inputs are the frozen all-24 IU-PGRD states.  Hyperparameters
are selected from labels in the eight official screen cells only.  ``select``
writes a content-bound selection without loading any held-cell target.  A
separate ``evaluate`` invocation verifies that selection and opens the other
sixteen cells exactly once as a locked batch.

For each cell, all inference is label-free:

    z_alt = zscore(z_IU + (t / G) zscore(R d_PGRD))
    w_i   = clip((rank_B3_i - lo) / (hi - lo), 0, 1)
    s_i   = z_B3_i + alpha * w_i * (z_alt_i - z_B3_i)

The IU arm fixes ``t=0``.  The IU+PGRD arm searches the same alpha/ramp grid
with ``t in {0.5, 1, 2}``.  Labels choose only these global hyperparameters;
they never enter a cell's score construction.

This is procedural leakage control, not fresh confirmation.  Every natural
label in the 24-cell panel was historically opened before this diagnostic.
The resulting evidence is therefore exploratory / retrospective C-tier.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE_DIR = (
    ROOT / "local_cache/deem_b3_moe_v1/iupgrd_boost_all24_v1"
)
DEFAULT_LABEL_DIR = ROOT / "local_cache/deem_b3_moe_v1/label_sidecars"
DEFAULT_CONFIG = ROOT / "configs/deem_b3_iupgrd_boost_v1.json"
DEFAULT_OUT_DIR = (
    ROOT / "local_cache/deem_b3_moe_v1/confidence_envelope_8dev_16held_v1"
)

SCHEMA_SELECTION = "b3_confidence_envelope_v1_selection"
SCHEMA_EVALUATION = "b3_confidence_envelope_v1_evaluation"
ALPHAS = (0.15, 0.30, 0.50)
PGRD_TRUSTS = (0.50, 1.00, 2.00)
RAMPS = (
    ("broad", 0.00, 1.00),
    ("upper_half", 0.50, 0.75),
    ("upper_quartile", 0.75, 0.90),
)
GLOBAL_FAMILIES = (
    "entropy_level",
    "entropy_dynamics",
    "sampled_token_energy",
    "partition_energy",
    "topk_distribution",
    "structural",
)
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260825


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_content_json(path: Path, payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    if "content_sha256" in value:
        raise ValueError("content_sha256 is reserved")
    content_sha256 = _canonical_sha256(value)
    value["content_sha256"] = content_sha256
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return content_sha256


def _read_content_json(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("content_sha256")
    unhashed = dict(value)
    unhashed.pop("content_sha256", None)
    if value.get("schema") != schema or expected != _canonical_sha256(unhashed):
        raise ValueError(f"content-bound JSON failed verification: {path}")
    return value


def _zscore(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    scale = float(np.std(x))
    if x.ndim != 1 or not np.isfinite(x).all() or scale <= 1e-12:
        raise ValueError("cannot standardize a non-finite or constant vector")
    return np.asarray((x - float(np.mean(x))) / scale, dtype=np.float64)


def _fractional_rank(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("rank input must be a finite vector of length >=3")
    # Average ties are deterministic and depend on target-free B3 scores only.
    return np.asarray((rankdata(x, method="average") - 1.0) / (len(x) - 1.0))


def _rank_ramp(rank: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if not (0.0 <= float(lo) < float(hi) <= 1.0):
        raise ValueError("rank ramp requires 0 <= lo < hi <= 1")
    return np.clip((np.asarray(rank) - float(lo)) / (float(hi) - float(lo)), 0.0, 1.0)


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    target = np.asarray(y, dtype=np.int8)
    values = np.asarray(score, dtype=np.float64)
    if target.shape != values.shape or set(np.unique(target).tolist()) != {0, 1}:
        raise ValueError("metric arrays are misaligned or target is single-class")
    return {
        "auroc": float(roc_auc_score(target, values)),
        "auprc": float(average_precision_score(target, values)),
    }


def _load_label(label_dir: Path, cell_id: str, expected_rows: np.ndarray) -> tuple[np.ndarray, dict]:
    array_path = label_dir / f"{cell_id}.npz"
    manifest_path = array_path.with_suffix(".manifest.json")
    with np.load(array_path, allow_pickle=False) as data:
        row_id = np.asarray(data["row_id"], dtype=str)
        y = np.asarray(data["y_H"], dtype=np.int8)
        schema = str(data["schema"].item())
        stored_cell = str(data["cell_id"].item())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        schema != "residual_graph_deem_label_sidecar_v1"
        or stored_cell != cell_id
        or manifest.get("schema") != schema
        or manifest.get("cell_id") != cell_id
        or not np.array_equal(row_id, expected_rows)
        or y.shape != expected_rows.shape
        or set(np.unique(y).tolist()) != {0, 1}
    ):
        raise ValueError(f"label/state binding failed: {cell_id}")
    return y, {
        "array_path": str(array_path.resolve()),
        "array_sha256": _sha256_file(array_path),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
    }


def _load_cell(state_dir: Path, cell_id: str) -> dict[str, Any]:
    path = state_dir / "states" / f"{cell_id}.npz"
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    if (
        meta.get("schema") != "deem_b3_iupgrd_boost_v1_state"
        or meta.get("status") != "complete"
        or meta.get("cell_id") != cell_id
        or meta.get("uses_labels") is not False
        or meta.get("array_sha256") != _sha256_file(path)
    ):
        raise ValueError(f"invalid all24 IU-PGRD state: {cell_id}")
    required = {
        "row_id",
        "family_order",
        "global_family_order",
        "baseline_score",
        "baseline_z",
        "iu_score_aligned",
        "iu_family_residuals",
    }
    if not required.issubset(arrays):
        raise ValueError(f"state lacks required arrays: {cell_id}")
    row_id = np.asarray(arrays["row_id"], dtype=str)
    baseline_score = np.asarray(arrays["baseline_score"], dtype=np.float64)
    baseline_z = np.asarray(arrays["baseline_z"], dtype=np.float64)
    iu_z = np.asarray(arrays["iu_score_aligned"], dtype=np.float64)
    residuals = np.asarray(arrays["iu_family_residuals"], dtype=np.float64)
    families = tuple(str(value) for value in arrays["family_order"].tolist())
    global_families = tuple(str(value) for value in arrays["global_family_order"].tolist())
    n = len(row_id)
    if (
        global_families != GLOBAL_FAMILIES
        or baseline_score.shape != (n,)
        or baseline_z.shape != (n,)
        or iu_z.shape != (n,)
        or residuals.shape != (n, len(families))
        or not all(np.isfinite(value).all() for value in (baseline_score, baseline_z, iu_z, residuals))
        or abs(float(np.mean(baseline_z))) > 1e-10
        or abs(float(np.std(baseline_z)) - 1.0) > 1e-10
        or abs(float(np.mean(iu_z))) > 1e-10
        or abs(float(np.std(iu_z)) - 1.0) > 1e-10
    ):
        raise ValueError(f"all24 state invariant failed: {cell_id}")
    dataset_family = str(meta.get("dataset_family"))
    calibration_path = state_dir / "calibrations" / f"held_{dataset_family}.npz"
    calibration_meta_path = calibration_path.with_suffix(".json")
    calibration_meta = json.loads(calibration_meta_path.read_text(encoding="utf-8"))
    with np.load(calibration_path, allow_pickle=False) as calibration:
        direction = np.asarray(calibration["direction"], dtype=np.float64)
    if (
        direction.shape != (len(GLOBAL_FAMILIES),)
        or calibration_meta.get("schema") != "deem_b3_iupgrd_boost_v1_calibration"
        or calibration_meta.get("held_dataset_family") != dataset_family
        or calibration_meta.get("whole_held_dataset_family_excluded") is not True
        or calibration_meta.get("uses_labels") is not False
        or calibration_meta.get("array_sha256") != _sha256_file(calibration_path)
    ):
        raise ValueError(f"invalid target-family-excluded calibration: {cell_id}")
    local_indices = np.asarray([GLOBAL_FAMILIES.index(name) for name in families], dtype=int)
    raw_pgrd = np.asarray(residuals @ direction[local_indices], dtype=np.float64)
    pgrd_z = _zscore(raw_pgrd)
    return {
        "cell_id": cell_id,
        "dataset_family": dataset_family,
        "row_id": row_id,
        "baseline_score": baseline_score,
        "baseline_z": baseline_z,
        "baseline_rank": _fractional_rank(baseline_score),
        "iu_z": iu_z,
        "pgrd_z": pgrd_z,
        "n_families": len(families),
        "state_path": str(path.resolve()),
        "state_sha256": _sha256_file(path),
        "state_metadata_path": str(meta_path.resolve()),
        "state_metadata_sha256": _sha256_file(meta_path),
        "calibration_path": str(calibration_path.resolve()),
        "calibration_sha256": _sha256_file(calibration_path),
        "calibration_metadata_path": str(calibration_meta_path.resolve()),
        "calibration_metadata_sha256": _sha256_file(calibration_meta_path),
    }


def _candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for alpha, (ramp_id, lo, hi) in itertools.product(ALPHAS, RAMPS):
        candidates.append(
            {
                "alt": "IU",
                "pgrd_trust": 0.0,
                "alpha": float(alpha),
                "ramp_id": ramp_id,
                "rank_lo": float(lo),
                "rank_hi": float(hi),
            }
        )
    for trust, alpha, (ramp_id, lo, hi) in itertools.product(PGRD_TRUSTS, ALPHAS, RAMPS):
        candidates.append(
            {
                "alt": "IU_PLUS_PGRD",
                "pgrd_trust": float(trust),
                "alpha": float(alpha),
                "ramp_id": ramp_id,
                "rank_lo": float(lo),
                "rank_hi": float(hi),
            }
        )
    for index, candidate in enumerate(candidates):
        candidate["candidate_id"] = f"C{index:02d}"
    return candidates


def _score_candidate(cell: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple[np.ndarray, dict]:
    trust = float(candidate["pgrd_trust"])
    iu_z = np.asarray(cell["iu_z"], dtype=np.float64)
    if candidate["alt"] == "IU":
        if trust != 0.0:
            raise ValueError("IU candidate must have zero PGRD trust")
        alt_z = iu_z
    elif candidate["alt"] == "IU_PLUS_PGRD":
        correction = trust / int(cell["n_families"]) * np.asarray(cell["pgrd_z"])
        alt_z = _zscore(iu_z + correction)
    else:
        raise ValueError(f"unknown alternative: {candidate['alt']}")
    ramp = _rank_ramp(
        np.asarray(cell["baseline_rank"]),
        float(candidate["rank_lo"]),
        float(candidate["rank_hi"]),
    )
    residual = alt_z - np.asarray(cell["baseline_z"])
    applied = float(candidate["alpha"]) * ramp * residual
    score = np.asarray(cell["baseline_z"] + applied, dtype=np.float64)
    return score, {
        "ramp_mean": float(np.mean(ramp)),
        "ramp_active_fraction": float(np.mean(ramp > 0.0)),
        "ramp_full_fraction": float(np.mean(ramp >= 1.0)),
        "alternative_b3_correlation": float(np.corrcoef(alt_z, cell["baseline_z"])[0, 1]),
        "residual_sd": float(np.std(residual)),
        "applied_correction_sd": float(np.std(applied)),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    deltas = np.asarray([float(row[f"delta_{metric}"]) for row in rows])
    baseline = np.asarray([float(row[f"baseline_{metric}"]) for row in rows])
    candidate = np.asarray([float(row[f"candidate_{metric}"]) for row in rows])
    families = sorted({str(row["dataset_family"]) for row in rows})
    family_rows = {}
    for family in families:
        subset = [row for row in rows if str(row["dataset_family"]) == family]
        family_rows[family] = {
            "n_cells": len(subset),
            "baseline": float(np.mean([row[f"baseline_{metric}"] for row in subset])),
            "candidate": float(np.mean([row[f"candidate_{metric}"] for row in subset])),
            "delta": float(np.mean([row[f"delta_{metric}"] for row in subset])),
        }
    return {
        "equal_cell_baseline": float(np.mean(baseline)),
        "equal_cell_candidate": float(np.mean(candidate)),
        "equal_cell_delta": float(np.mean(deltas)),
        "equal_family_baseline": float(np.mean([row["baseline"] for row in family_rows.values()])),
        "equal_family_candidate": float(np.mean([row["candidate"] for row in family_rows.values()])),
        "equal_family_delta": float(np.mean([row["delta"] for row in family_rows.values()])),
        "cell_wins": int(np.sum(deltas > 0.0)),
        "cell_ties": int(np.sum(deltas == 0.0)),
        "cell_losses": int(np.sum(deltas < 0.0)),
        "family_wins": int(np.sum([row["delta"] > 0.0 for row in family_rows.values()])),
        "family_ties": int(np.sum([row["delta"] == 0.0 for row in family_rows.values()])),
        "family_losses": int(np.sum([row["delta"] < 0.0 for row in family_rows.values()])),
        "family_rows": family_rows,
    }


def _selection_key(summary: Mapping[str, Any], candidate: Mapping[str, Any]) -> tuple:
    # Max AUROC, then AUPRC.  Remaining terms are deterministic simplicity
    # preferences and are consulted only after exact metric ties.
    return (
        float(summary["auroc"]["equal_cell_delta"]),
        float(summary["auprc"]["equal_cell_delta"]),
        -float(candidate["alpha"]),
        -float(candidate["pgrd_trust"]),
        -float(candidate["rank_lo"]),
        -float(candidate["rank_hi"]),
        str(candidate["candidate_id"]),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _signflip_p(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x):
        return float("nan")
    observed = float(np.mean(x))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(x))))
    null = np.mean(signs * x[None, :], axis=1)
    return float(np.mean(null >= observed - 1e-15))


def _bootstrap_ci(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, list[float]]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if metric == "auroc" else 1))
    cell_values = np.asarray([float(row[f"delta_{metric}"]) for row in rows])
    cell_draws = np.mean(
        cell_values[rng.integers(0, len(cell_values), size=(BOOTSTRAP_DRAWS, len(cell_values)))],
        axis=1,
    )
    families = sorted({str(row["dataset_family"]) for row in rows})
    by_family = {
        family: np.asarray(
            [float(row[f"delta_{metric}"]) for row in rows if row["dataset_family"] == family]
        )
        for family in families
    }
    family_draws = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled_families = rng.integers(0, len(families), size=len(families))
        draw_values = []
        for family_index in sampled_families:
            values = by_family[families[int(family_index)]]
            sampled_cells = values[rng.integers(0, len(values), size=len(values))]
            draw_values.append(float(np.mean(sampled_cells)))
        family_draws[draw] = float(np.mean(draw_values))
    return {
        "equal_cell": [float(value) for value in np.quantile(cell_draws, [0.025, 0.975])],
        "equal_family_hierarchical": [
            float(value) for value in np.quantile(family_draws, [0.025, 0.975])
        ],
    }


def _screen_select(args: argparse.Namespace) -> None:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run_definition = _read_content_json(
        args.state_dir / "RUN_DEFINITION.json",
        "deem_b3_iupgrd_boost_v1_run_definition",
    )
    all_cells = tuple(str(value) for value in run_definition["cells"])
    screen_cells = tuple(str(value) for value in config["screen_cells"])
    held_cells = tuple(value for value in all_cells if value not in set(screen_cells))
    if len(all_cells) != 24 or len(screen_cells) != 8 or len(held_cells) != 16:
        raise ValueError("official all24 8/16 partition failed")

    # Only screen states and screen labels are loaded in this process.
    cells = {cell_id: _load_cell(args.state_dir, cell_id) for cell_id in screen_cells}
    labels = {}
    label_audits = {}
    for cell_id in screen_cells:
        labels[cell_id], label_audits[cell_id] = _load_label(
            args.label_dir, cell_id, cells[cell_id]["row_id"]
        )

    candidates = _candidate_grid()
    candidate_rows = []
    candidate_summaries = {}
    candidate_cell_rows = {}
    for candidate in candidates:
        rows = []
        for cell_id in screen_cells:
            cell = cells[cell_id]
            score, diagnostics = _score_candidate(cell, candidate)
            baseline_metrics = _metrics(labels[cell_id], cell["baseline_z"])
            candidate_metrics = _metrics(labels[cell_id], score)
            rows.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "alt": candidate["alt"],
                    "cell_id": cell_id,
                    "dataset_family": cell["dataset_family"],
                    "n_rows": len(labels[cell_id]),
                    **candidate,
                    **diagnostics,
                    **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                    **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                    **{
                        f"delta_{key}": candidate_metrics[key] - baseline_metrics[key]
                        for key in baseline_metrics
                    },
                }
            )
        summary = {
            metric: _aggregate(rows, metric) for metric in ("auroc", "auprc")
        }
        candidate_summaries[candidate["candidate_id"]] = summary
        candidate_cell_rows[candidate["candidate_id"]] = rows
        candidate_rows.append(
            {
                **candidate,
                "screen_equal_cell_delta_auroc": summary["auroc"]["equal_cell_delta"],
                "screen_equal_cell_delta_auprc": summary["auprc"]["equal_cell_delta"],
                "screen_cell_wins_auroc": summary["auroc"]["cell_wins"],
                "screen_cell_losses_auroc": summary["auroc"]["cell_losses"],
                "screen_worst_cell_delta_auroc": min(row["delta_auroc"] for row in rows),
            }
        )

    selected_by_alt = {}
    for alt in ("IU", "IU_PLUS_PGRD"):
        eligible = [candidate for candidate in candidates if candidate["alt"] == alt]
        selected = max(
            eligible,
            key=lambda candidate: _selection_key(
                candidate_summaries[candidate["candidate_id"]], candidate
            ),
        )
        selected_by_alt[alt] = dict(selected)
    overall = max(
        selected_by_alt.values(),
        key=lambda candidate: _selection_key(
            candidate_summaries[candidate["candidate_id"]], candidate
        ),
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_path = args.out_dir / "SCREEN_GRID.csv"
    cell_path = args.out_dir / "SCREEN_SELECTED_CELL_METRICS.csv"
    _write_csv(grid_path, candidate_rows)
    selected_cell_rows = []
    for alt, candidate in selected_by_alt.items():
        selected_cell_rows.extend(candidate_cell_rows[candidate["candidate_id"]])
    _write_csv(cell_path, selected_cell_rows)

    state_audits = {
        cell_id: {
            key: cells[cell_id][key]
            for key in (
                "dataset_family",
                "state_path",
                "state_sha256",
                "state_metadata_path",
                "state_metadata_sha256",
                "calibration_path",
                "calibration_sha256",
                "calibration_metadata_path",
                "calibration_metadata_sha256",
            )
        }
        for cell_id in screen_cells
    }
    selection_payload = {
        "schema": SCHEMA_SELECTION,
        "status": "frozen_after_screen_before_held_evaluation",
        "scientific_tier": "exploratory_retrospective_C_tier",
        "historically_open_natural_labels": True,
        "procedural_boundary": {
            "per_cell_inference_is_label_free": True,
            "screen_labels_choose_only_global_hyperparameters": True,
            "held_labels_loaded_during_selection": False,
            "fresh_confirmation": False,
        },
        "formula": "z_B3 + alpha * ramp(rank_B3; lo, hi) * (z_alt - z_B3)",
        "alternative_definition": {
            "IU": "z_alt = aligned standardized IU",
            "IU_PLUS_PGRD": (
                "z_alt = zscore(z_IU + (t/G_present) * "
                "zscore(R_IU_residual d_LODFO_cross_only))"
            ),
        },
        "grid": {
            "alphas": list(ALPHAS),
            "pgrd_trusts": list(PGRD_TRUSTS),
            "ramps": [
                {"id": ramp_id, "lo": lo, "hi": hi}
                for ramp_id, lo, hi in RAMPS
            ],
            "candidate_count": len(candidates),
        },
        "selection_policy": (
            "separately within each alt maximize screen equal-cell AUROC delta; "
            "then AUPRC; exact ties prefer smaller alpha, smaller t, broader ramp; "
            "overall winner uses the same ordering"
        ),
        "all_cells": list(all_cells),
        "screen_cells": list(screen_cells),
        "held_cells": list(held_cells),
        "selected_by_alt": selected_by_alt,
        "overall_selected": overall,
        "selected_screen_summaries": {
            alt: candidate_summaries[candidate["candidate_id"]]
            for alt, candidate in selected_by_alt.items()
        },
        "state_run_definition_path": str((args.state_dir / "RUN_DEFINITION.json").resolve()),
        "state_run_definition_file_sha256": _sha256_file(args.state_dir / "RUN_DEFINITION.json"),
        "state_run_definition_content_sha256": run_definition["content_sha256"],
        "state_audits_screen_only": state_audits,
        "label_audits_screen_only": label_audits,
        "screen_grid_path": str(grid_path.resolve()),
        "screen_grid_sha256": _sha256_file(grid_path),
        "screen_selected_cell_metrics_path": str(cell_path.resolve()),
        "screen_selected_cell_metrics_sha256": _sha256_file(cell_path),
    }
    selection_path = args.out_dir / "FROZEN_SELECTION.json"
    content_sha = _write_content_json(selection_path, selection_payload)
    print(
        _canonical_json(
            {
                "status": "selection_frozen",
                "selection_path": str(selection_path.resolve()),
                "selection_content_sha256": content_sha,
                "selected_by_alt": selected_by_alt,
                "overall_selected": overall,
            }
        )
    )


def _held_evaluate(args: argparse.Namespace) -> None:
    selection_path = args.out_dir / "FROZEN_SELECTION.json"
    selection = _read_content_json(selection_path, SCHEMA_SELECTION)
    if selection.get("status") != "frozen_after_screen_before_held_evaluation":
        raise ValueError("selection is not at the frozen pre-held boundary")
    held_cells = tuple(str(value) for value in selection["held_cells"])
    screen_cells = set(str(value) for value in selection["screen_cells"])
    if len(held_cells) != 16 or screen_cells.intersection(held_cells):
        raise ValueError("held partition is invalid")

    # Held state and label loading occurs only after verifying the frozen file.
    cells = {cell_id: _load_cell(args.state_dir, cell_id) for cell_id in held_cells}
    labels = {}
    label_audits = {}
    for cell_id in held_cells:
        labels[cell_id], label_audits[cell_id] = _load_label(
            args.label_dir, cell_id, cells[cell_id]["row_id"]
        )

    selected_by_alt = {
        str(key): dict(value) for key, value in selection["selected_by_alt"].items()
    }
    all_rows = []
    summaries = {}
    for alt in ("IU", "IU_PLUS_PGRD"):
        candidate = selected_by_alt[alt]
        rows = []
        for cell_id in held_cells:
            cell = cells[cell_id]
            score, diagnostics = _score_candidate(cell, candidate)
            baseline_metrics = _metrics(labels[cell_id], cell["baseline_z"])
            candidate_metrics = _metrics(labels[cell_id], score)
            row = {
                "alt": alt,
                "candidate_id": candidate["candidate_id"],
                "cell_id": cell_id,
                "dataset_family": cell["dataset_family"],
                "n_rows": len(labels[cell_id]),
                **candidate,
                **diagnostics,
                **{f"baseline_{key}": value for key, value in baseline_metrics.items()},
                **{f"candidate_{key}": value for key, value in candidate_metrics.items()},
                **{
                    f"delta_{key}": candidate_metrics[key] - baseline_metrics[key]
                    for key in baseline_metrics
                },
            }
            rows.append(row)
            all_rows.append(row)
        summary = {metric: _aggregate(rows, metric) for metric in ("auroc", "auprc")}
        for metric in ("auroc", "auprc"):
            summary[metric]["bootstrap_95"] = _bootstrap_ci(rows, metric)
            family_values = [
                value["delta"] for value in summary[metric]["family_rows"].values()
            ]
            summary[metric]["family_signflip_one_sided_p"] = _signflip_p(family_values)
            summary[metric]["cell_signflip_one_sided_p"] = _signflip_p(
                [row[f"delta_{metric}"] for row in rows]
            )
        summary["gain_at_least_0_0025_equal_cell_auroc"] = bool(
            summary["auroc"]["equal_cell_delta"] >= 0.0025
        )
        summary["gain_at_least_0_0025_equal_family_auroc"] = bool(
            summary["auroc"]["equal_family_delta"] >= 0.0025
        )
        summaries[alt] = summary

    metrics_path = args.out_dir / "HELD16_CELL_METRICS.csv"
    _write_csv(metrics_path, all_rows)
    state_audits = {
        cell_id: {
            key: cells[cell_id][key]
            for key in (
                "dataset_family",
                "state_path",
                "state_sha256",
                "state_metadata_path",
                "state_metadata_sha256",
                "calibration_path",
                "calibration_sha256",
                "calibration_metadata_path",
                "calibration_metadata_sha256",
            )
        }
        for cell_id in held_cells
    }
    evaluation_payload = {
        "schema": SCHEMA_EVALUATION,
        "status": "held16_evaluated_once_as_locked_batch",
        "scientific_tier": "exploratory_retrospective_C_tier",
        "historically_open_natural_labels": True,
        "fresh_confirmation": False,
        "selection_path": str(selection_path.resolve()),
        "selection_file_sha256": _sha256_file(selection_path),
        "selection_content_sha256": selection["content_sha256"],
        "held_cells": list(held_cells),
        "selected_by_alt": selected_by_alt,
        "overall_selected": selection["overall_selected"],
        "summaries": summaries,
        "bootstrap": {
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "equal_cell": "paired resampling of 16 cell deltas",
            "equal_family": (
                "hierarchical paired resampling of 3 held dataset families, "
                "then cells within selected family"
            ),
            "warning": "only three held dataset families; family uncertainty is coarse",
        },
        "state_audits_held_only": state_audits,
        "label_audits_held_only": label_audits,
        "held_cell_metrics_path": str(metrics_path.resolve()),
        "held_cell_metrics_sha256": _sha256_file(metrics_path),
    }
    evaluation_path = args.out_dir / "HELD16_EVALUATION.json"
    _write_content_json(evaluation_path, evaluation_payload)
    _write_report(args.out_dir / "REPORT.md", selection, evaluation_payload)
    print(
        _canonical_json(
            {
                "status": evaluation_payload["status"],
                "evaluation_path": str(evaluation_path.resolve()),
                "summaries": summaries,
            }
        )
    )


def _fmt(value: float) -> str:
    return f"{float(value):+.6f}"


def _write_report(path: Path, selection: Mapping[str, Any], evaluation: Mapping[str, Any]) -> None:
    lines = [
        "# B3 confidence-envelope residual — retrospective 8/16 diagnostic",
        "",
        "**Evidence tier: exploratory / retrospective C-tier.** All 24 natural-label cells",
        "were historically open before this procedure. The 8/16 split enforces a procedural",
        "development boundary; it is not fresh confirmation.",
        "",
        "Formula: `z_B3 + alpha * ramp(rank_B3; lo, hi) * (z_alt - z_B3)`.",
        "Per-cell B3 rank, IU, IU-PGRD residuals, and all standardizations are target-free.",
        "Only the global `(alt, t, alpha, lo, hi)` choice used the eight screen labels.",
        "",
        "## Frozen screen choices",
        "",
    ]
    for alt in ("IU", "IU_PLUS_PGRD"):
        candidate = selection["selected_by_alt"][alt]
        screen = selection["selected_screen_summaries"][alt]
        lines.extend(
            [
                f"- **{alt}**: `{candidate['candidate_id']}`; t={candidate['pgrd_trust']}, "
                f"alpha={candidate['alpha']}, ramp={candidate['ramp_id']} "
                f"({candidate['rank_lo']}, {candidate['rank_hi']}); screen "
                f"delta AUROC={_fmt(screen['auroc']['equal_cell_delta'])}, "
                f"AUPRC={_fmt(screen['auprc']['equal_cell_delta'])}.",
            ]
        )
    lines.extend(["", "## Locked held-16 result", ""])
    for alt in ("IU", "IU_PLUS_PGRD"):
        result = evaluation["summaries"][alt]
        auc = result["auroc"]
        pr = result["auprc"]
        lines.extend(
            [
                f"### {alt}",
                "",
                f"- Equal-cell AUROC: {auc['equal_cell_candidate']:.6f} vs B3 "
                f"{auc['equal_cell_baseline']:.6f}; delta {_fmt(auc['equal_cell_delta'])}, "
                f"95% cell bootstrap [{_fmt(auc['bootstrap_95']['equal_cell'][0])}, "
                f"{_fmt(auc['bootstrap_95']['equal_cell'][1])}], "
                f"W/T/L={auc['cell_wins']}/{auc['cell_ties']}/{auc['cell_losses']}.",
                f"- Equal-family AUROC: {auc['equal_family_candidate']:.6f} vs B3 "
                f"{auc['equal_family_baseline']:.6f}; delta {_fmt(auc['equal_family_delta'])}, "
                f"95% hierarchical family bootstrap "
                f"[{_fmt(auc['bootstrap_95']['equal_family_hierarchical'][0])}, "
                f"{_fmt(auc['bootstrap_95']['equal_family_hierarchical'][1])}], "
                f"family W/T/L={auc['family_wins']}/{auc['family_ties']}/{auc['family_losses']}, "
                f"one-sided exact family sign-flip p={auc['family_signflip_one_sided_p']:.6f}.",
                f"- Equal-cell AUPRC: {pr['equal_cell_candidate']:.6f} vs B3 "
                f"{pr['equal_cell_baseline']:.6f}; delta {_fmt(pr['equal_cell_delta'])}, "
                f"95% cell bootstrap [{_fmt(pr['bootstrap_95']['equal_cell'][0])}, "
                f"{_fmt(pr['bootstrap_95']['equal_cell'][1])}].",
                f"- Equal-family AUPRC: {pr['equal_family_candidate']:.6f} vs B3 "
                f"{pr['equal_family_baseline']:.6f}; delta {_fmt(pr['equal_family_delta'])}.",
                f"- AUROC gain >= 0.0025: equal-cell "
                f"**{'YES' if result['gain_at_least_0_0025_equal_cell_auroc'] else 'NO'}**; "
                f"equal-family **{'YES' if result['gain_at_least_0_0025_equal_family_auroc'] else 'NO'}**.",
                "",
            ]
        )
    lines.extend(
        [
            "Only three dataset families occur in held-16 (GSM8K, MATH-500, TriviaQA),",
            "so the family bootstrap and exact sign-flip test are necessarily coarse.",
            "No held result should be described as independent confirmation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("select", "evaluate"))
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if args.phase == "select":
        if (args.out_dir / "HELD16_EVALUATION.json").exists():
            raise FileExistsError("held evaluation already exists; refusing reselection")
        _screen_select(args)
    else:
        if (args.out_dir / "HELD16_EVALUATION.json").exists():
            raise FileExistsError("held-16 was already evaluated; refusing a second opening")
        _held_evaluate(args)


if __name__ == "__main__":
    main()
