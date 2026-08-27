#!/usr/bin/env python3
"""Retrospective donor-jackknife consensus test for B3 and IU-PGRD.

This is deliberately a sidecar experiment.  It does not modify or refit the
frozen B3, IU, graph, or pooled-direction implementations.  It consumes the
already frozen all-24-cell IU-PGRD states, constructs leave-one-donor-family-
out directions without targets, freezes every candidate score, and only then
imports the label-sidecar module for evaluation.

The roster is intentionally small and fixed in source:

* B3;
* a half-strength (0.5/G) B3-orthogonal PGRD correction, plain/gated/permuted;
* a 0.30 standardized-score blend from B3 toward historical IU+PGRD,
  plain/gated/permuted.

The consensus gate is the exact formula used by the companion diagnostic::

    agreement = abs(mean_j(sign(delta_j)))
    dispersion = sd_j(delta_j) / (mean_j(abs(delta_j)) + eps)
    reliability = agreement / (1 + dispersion)

Here ``delta_j`` is the unit-SD B3-orthogonal correction from the donor pool
with donor family ``j`` omitted.  A deterministic row-permutation of the
same reliability vector is the sample-specificity control.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_deem_b3_iupgrd_boost_v1 import (  # noqa: E402
    LABEL_MODULE,
    _comparison,
    _hash_contract_pass,
    _load_targets_after_preflight,
    _mechanical_artifact_pass,
    _metrics,
    _stable_seed,
    _summary,
)
from spectral_utils.deem_b3_iupgrd_boost import (  # noqa: E402
    deterministic_row_permutation,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ResidualGraphDeemError,
    atomic_save_npz,
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from spectral_utils.specrage_views import VIEW_ORDER  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/deem_b3_iupgrd_boost_v1.json"
DEFAULT_REGISTRY = ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"

EPS = 1e-12
DIRECT_TRUST = 0.5
BLEND_ALPHA = 0.30
PROMOTION_DELTA = 0.0025
PERMUTATION_SALT = "deem_b3_jackknife_consensus_v1_gate_row_control"

METHODS = (
    "B3",
    "J1_DIRECT_HALF",
    "J2_DIRECT_HALF_CONSENSUS",
    "J3_DIRECT_HALF_PERMUTED_CONSENSUS",
    "J4_BLEND_030_IUPGRD",
    "J5_BLEND_030_IUPGRD_CONSENSUS",
    "J6_BLEND_030_IUPGRD_PERMUTED_CONSENSUS",
)


def _write_hashed_json(path: Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    body["content_sha256"] = canonical_sha256(body)
    atomic_write_json(path, body)
    return str(body["content_sha256"])


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _project_and_standardize(raw: np.ndarray, baseline_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raw, dtype=np.float64)
    base = np.asarray(baseline_z, dtype=np.float64)
    if values.shape != base.shape or values.ndim != 1:
        raise ValueError("correction/B3 shape mismatch")
    design = np.column_stack([np.ones(len(base), dtype=np.float64), base])
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    projected = values - design @ coefficients
    scale = float(np.std(projected))
    if not np.isfinite(scale) or scale <= EPS:
        raise ResidualGraphDeemError("jackknife correction is constant")
    standardized = projected / scale
    if max(abs(float(np.mean(standardized))), abs(float(np.mean(standardized * base)))) > 1e-9:
        raise ResidualGraphDeemError("jackknife B3-orthogonality invariant failed")
    return np.asarray(standardized, dtype=np.float64), np.asarray(coefficients, dtype=np.float64)


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    centered = array - float(np.mean(array))
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale <= EPS:
        raise ResidualGraphDeemError("standardized score is constant")
    return centered / scale


def _load_state(run_dir: Path, cell: str) -> dict[str, np.ndarray]:
    path = run_dir / "states" / f"{cell}.npz"
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _donor_family_cross_vectors(
    *,
    run_cells: Sequence[str],
    family_by_cell: Mapping[str, str],
    states: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Mean trace-normalized cross moment per dataset family."""

    result: dict[str, np.ndarray] = {}
    for family in sorted(set(family_by_cell.values())):
        members = [cell for cell in run_cells if family_by_cell[cell] == family]
        if not members:
            raise AssertionError("empty donor family")
        result[family] = np.mean(
            [np.asarray(states[cell]["moment_c"], dtype=np.float64) for cell in members],
            axis=0,
        )
    return result


def _directions_for_held_family(
    held_family: str,
    family_cross: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    donors = tuple(sorted(family for family in family_cross if family != held_family))
    if len(donors) < 3:
        raise ResidualGraphDeemError("jackknife needs at least three donor families")
    stacked = np.stack([family_cross[family] for family in donors])
    full = -np.mean(stacked, axis=0)
    jackknife = np.stack(
        [-np.mean(np.delete(stacked, index, axis=0), axis=0) for index in range(len(donors))]
    )
    # Equal-family leave-one-out estimates average back to the full estimate.
    if not np.allclose(np.mean(jackknife, axis=0), full, rtol=0.0, atol=1e-12):
        raise ResidualGraphDeemError("jackknife/full direction identity failed")
    return np.asarray(full), donors, np.asarray(jackknife)


def _score_cell(
    *,
    cell: str,
    state: Mapping[str, np.ndarray],
    full_direction: np.ndarray,
    jackknife_directions: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    row_ids = np.asarray(state["row_id"], dtype=str)
    families = tuple(str(value) for value in state["family_order"].tolist())
    local = np.asarray([VIEW_ORDER.index(family) for family in families], dtype=np.int64)
    residuals = np.asarray(state["iu_family_residuals"], dtype=np.float64)
    baseline_score = np.asarray(state["baseline_score"], dtype=np.float64)
    baseline_logit = np.asarray(state["baseline_logit"], dtype=np.float64)
    baseline_z = np.asarray(state["baseline_z"], dtype=np.float64)
    baseline_mean = float(np.asarray(state["baseline_logit_mean"]).item())
    baseline_scale = float(np.asarray(state["baseline_logit_scale"]).item())
    iu = np.asarray(state["iu_score_aligned"], dtype=np.float64)
    if residuals.shape != (len(row_ids), len(families)):
        raise ResidualGraphDeemError(f"state residual shape mismatch: {cell}")

    full_raw = residuals @ np.asarray(full_direction, dtype=np.float64)[local]
    full_orth_z, full_coefficients = _project_and_standardize(full_raw, baseline_z)
    jackknife_z = []
    for direction in np.asarray(jackknife_directions, dtype=np.float64):
        raw = residuals @ direction[local]
        standardized, _ = _project_and_standardize(raw, baseline_z)
        jackknife_z.append(standardized)
    jackknife_z = np.column_stack(jackknife_z)

    agreement = np.abs(np.mean(np.sign(jackknife_z), axis=1))
    relative_dispersion = np.std(jackknife_z, axis=1) / (
        np.mean(np.abs(jackknife_z), axis=1) + EPS
    )
    reliability = agreement / (1.0 + relative_dispersion)
    if (
        not np.isfinite(reliability).all()
        or np.any(reliability < -1e-12)
        or np.any(reliability > 1.0 + 1e-12)
    ):
        raise ResidualGraphDeemError(f"invalid consensus reliability: {cell}")
    reliability = np.clip(reliability, 0.0, 1.0)
    permutation = deterministic_row_permutation(row_ids, salt=PERMUTATION_SALT)
    permuted_reliability = reliability[permutation]

    g = len(families)
    direct_half_z = DIRECT_TRUST * full_orth_z / g

    # Historical IU-PGRD works in the IU coordinate system: its residuals are
    # already centered, scaled, and orthogonal to IU.  It therefore uses the
    # unprojected PGRD correction, unlike the direct B3 residual arm above.
    raw_scale = float(np.std(full_raw))
    if not np.isfinite(raw_scale) or raw_scale <= EPS:
        raise ResidualGraphDeemError(f"IU-PGRD correction is constant: {cell}")
    iupgrd_z = _standardize(iu + full_raw / (g * raw_scale))
    blend_delta_z = BLEND_ALPHA * (iupgrd_z - baseline_z)

    standardized_scores = {
        "J1_DIRECT_HALF": baseline_z + direct_half_z,
        "J2_DIRECT_HALF_CONSENSUS": baseline_z + reliability * direct_half_z,
        "J3_DIRECT_HALF_PERMUTED_CONSENSUS": baseline_z
        + permuted_reliability * direct_half_z,
        "J4_BLEND_030_IUPGRD": baseline_z + blend_delta_z,
        "J5_BLEND_030_IUPGRD_CONSENSUS": baseline_z + reliability * blend_delta_z,
        "J6_BLEND_030_IUPGRD_PERMUTED_CONSENSUS": baseline_z
        + permuted_reliability * blend_delta_z,
    }
    scores = {"B3": baseline_score}
    for method, z_score in standardized_scores.items():
        scores[method] = expit(baseline_mean + baseline_scale * z_score)
    if not all(np.isfinite(score).all() for score in scores.values()):
        raise ResidualGraphDeemError(f"non-finite candidate score: {cell}")

    diagnostics = {
        "cell_id": cell,
        "n_rows": int(len(row_ids)),
        "n_present_families": int(g),
        "n_jackknife_directions": int(jackknife_z.shape[1]),
        "direct_trust": DIRECT_TRUST,
        "blend_alpha": BLEND_ALPHA,
        "reliability_mean": float(np.mean(reliability)),
        "reliability_sd": float(np.std(reliability)),
        "reliability_q10": float(np.quantile(reliability, 0.10)),
        "reliability_q50": float(np.quantile(reliability, 0.50)),
        "reliability_q90": float(np.quantile(reliability, 0.90)),
        "sign_agreement_mean": float(np.mean(agreement)),
        "relative_dispersion_mean": float(np.mean(relative_dispersion)),
        "full_orthogonal_correction_sd": float(np.std(full_orth_z)),
        "direct_half_correction_sd": float(np.std(direct_half_z)),
        "iupgrd_vs_b3_pearson": float(np.corrcoef(iupgrd_z, baseline_z)[0, 1]),
        "blend_delta_sd": float(np.std(blend_delta_z)),
        "gate_correction_abs_spearman_proxy": float(
            np.corrcoef(
                np.argsort(np.argsort(reliability, kind="mergesort"), kind="mergesort"),
                np.argsort(np.argsort(np.abs(direct_half_z), kind="mergesort"), kind="mergesort"),
            )[0, 1]
        ),
        "projection_intercept": float(full_coefficients[0]),
        "projection_b3": float(full_coefficients[1]),
        "uses_labels": False,
    }
    aux = {
        "reliability": reliability,
        "permuted_reliability": permuted_reliability,
        "sign_agreement": agreement,
        "relative_dispersion": relative_dispersion,
        "full_orthogonal_correction_z": full_orth_z,
        "jackknife_correction_z": jackknife_z,
        "iupgrd_z": iupgrd_z,
        "permutation": permutation,
    }
    return {**scores, **aux}, diagnostics


def _panel(
    *,
    name: str,
    cells: Sequence[str],
    per_cell: Sequence[Mapping[str, Any]],
    bootstrap_draws: int,
) -> dict[str, Any]:
    rows = [row for row in per_cell if row["cell_id"] in set(cells)]
    summaries = [
        _summary(rows, method, metric)
        for method in METHODS
        for metric in ("auroc", "auprc")
    ]
    comparisons_vs_b3 = []
    for method in METHODS[1:]:
        comparisons_vs_b3.append(
            _comparison(
                rows,
                method,
                "B3",
                draws=bootstrap_draws,
                seed=_stable_seed(
                    "deem_b3_jackknife_consensus_v1",
                    name,
                    method,
                    "B3",
                ),
            )
        )
    controls = []
    for candidate, reference in (
        ("J2_DIRECT_HALF_CONSENSUS", "J1_DIRECT_HALF"),
        ("J2_DIRECT_HALF_CONSENSUS", "J3_DIRECT_HALF_PERMUTED_CONSENSUS"),
        ("J5_BLEND_030_IUPGRD_CONSENSUS", "J4_BLEND_030_IUPGRD"),
        ("J5_BLEND_030_IUPGRD_CONSENSUS", "J6_BLEND_030_IUPGRD_PERMUTED_CONSENSUS"),
    ):
        controls.append(
            _comparison(
                rows,
                candidate,
                reference,
                draws=bootstrap_draws,
                seed=_stable_seed(
                    "deem_b3_jackknife_consensus_v1",
                    name,
                    candidate,
                    reference,
                ),
            )
        )
    by_candidate = {row["candidate"]: row for row in comparisons_vs_b3}
    direct_real_vs_permuted = next(
        row
        for row in controls
        if row["candidate"] == "J2_DIRECT_HALF_CONSENSUS"
        and row["reference"] == "J3_DIRECT_HALF_PERMUTED_CONSENSUS"
    )
    blend_real_vs_permuted = next(
        row
        for row in controls
        if row["candidate"] == "J5_BLEND_030_IUPGRD_CONSENSUS"
        and row["reference"] == "J6_BLEND_030_IUPGRD_PERMUTED_CONSENSUS"
    )
    return {
        "panel": name,
        "cells": list(cells),
        "n_cells": len(cells),
        "n_dataset_families": len(
            {str(row["dataset_family"]) for row in rows}
        ),
        "summaries": summaries,
        "comparisons_vs_b3": comparisons_vs_b3,
        "mechanism_controls": controls,
        "threshold_checks": {
            "threshold_equal_family_auroc_delta": PROMOTION_DELTA,
            "direct_consensus_reaches_threshold": by_candidate[
                "J2_DIRECT_HALF_CONSENSUS"
            ]["equal_dataset_family_auroc_delta"]
            >= PROMOTION_DELTA,
            "blend_consensus_reaches_threshold": by_candidate[
                "J5_BLEND_030_IUPGRD_CONSENSUS"
            ]["equal_dataset_family_auroc_delta"]
            >= PROMOTION_DELTA,
            "direct_real_beats_permuted": direct_real_vs_permuted[
                "equal_dataset_family_auroc_delta"
            ]
            > 0.0,
            "blend_real_beats_permuted": blend_real_vs_permuted[
                "equal_dataset_family_auroc_delta"
            ]
            > 0.0,
        },
    }


def _comparison_lookup(panel: Mapping[str, Any], candidate: str, reference: str) -> Mapping[str, Any]:
    source = (
        panel["comparisons_vs_b3"]
        if reference == "B3"
        else panel["mechanism_controls"]
    )
    return next(
        row
        for row in source
        if row["candidate"] == candidate and row["reference"] == reference
    )


def _report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Donor-family jackknife consensus for B3 / IU-PGRD",
        "",
        "This is retrospective exploratory evidence on previously opened natural labels. "
        "All candidate scores were frozen before the label module was imported.",
        "",
        "The official eight-cell screen and the remaining held-16 stress panel are reported "
        "separately; no method or hyperparameter changed after the screen.",
        "",
        "| panel | candidate | Δ AUROC vs B3 | 95% family bootstrap | Δ AUPRC | Δ AUROC vs permuted gate | >= 0.0025? |",
        "|---|---|---:|---:|---:|---:|:---:|",
    ]
    for panel_name in ("screen8", "held16"):
        panel = report["panels"][panel_name]
        for candidate, permuted in (
            ("J2_DIRECT_HALF_CONSENSUS", "J3_DIRECT_HALF_PERMUTED_CONSENSUS"),
            ("J5_BLEND_030_IUPGRD_CONSENSUS", "J6_BLEND_030_IUPGRD_PERMUTED_CONSENSUS"),
        ):
            versus = _comparison_lookup(panel, candidate, "B3")
            control = _comparison_lookup(panel, candidate, permuted)
            lines.append(
                "| {panel} | `{candidate}` | {delta:+.6f} | [{lower:+.6f}, {upper:+.6f}] | "
                "{auprc:+.6f} | {control:+.6f} | {passed} |".format(
                    panel=panel_name,
                    candidate=candidate,
                    delta=versus["equal_dataset_family_auroc_delta"],
                    lower=versus["descriptive_family_bootstrap_lower"],
                    upper=versus["descriptive_family_bootstrap_upper"],
                    auprc=versus["equal_dataset_family_auprc_delta"],
                    control=control["equal_dataset_family_auroc_delta"],
                    passed="yes"
                    if versus["equal_dataset_family_auroc_delta"] >= PROMOTION_DELTA
                    else "no",
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            report["interpretation"],
            "",
            "The row-permuted control preserves the reliability distribution and changes only "
            "its sample alignment. A positive real-minus-permuted point estimate is necessary "
            "mechanism evidence, but is not sufficient for promotion.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=9999)
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {args.out_dir}")
    if int(args.bootstrap_draws) < 100:
        raise ValueError("bootstrap draws must be at least 100")

    # Strictly validate the complete upstream target-free run before deriving
    # anything from it.  Selecting "all" makes the mechanical pass retain all
    # 24 frozen B3 and E0--E5 scores in memory.
    hash_state = _hash_contract_pass(
        config_path=args.config,
        registry_path=args.registry,
        bundle_dir=args.bundle_dir,
        baseline_dir=args.baseline_dir,
        run_dir=args.run_dir,
        cells_raw="all",
    )
    mechanics = _mechanical_artifact_pass(hash_state)
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before consensus fit")

    run_cells = tuple(hash_state["run_cells"])
    family_by_cell = dict(hash_state["family_by_cell"])
    states = {cell: _load_state(args.run_dir, cell) for cell in run_cells}
    family_cross = _donor_family_cross_vectors(
        run_cells=run_cells,
        family_by_cell=family_by_cell,
        states=states,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    score_dir = args.out_dir / "scores"
    artifacts = []
    score_cache: dict[tuple[str, str], np.ndarray] = {}
    diagnostics = []
    calibration_diagnostics = []
    for held_family in sorted(set(family_by_cell.values())):
        full_direction, donor_families, jackknife_directions = _directions_for_held_family(
            held_family, family_cross
        )
        calibration_path = args.run_dir / "calibrations" / f"held_{held_family}.npz"
        with np.load(calibration_path, allow_pickle=False) as data:
            upstream_direction = np.asarray(data["direction"], dtype=np.float64)
        max_error = float(np.max(np.abs(full_direction - upstream_direction)))
        if max_error > 1e-12:
            raise ResidualGraphDeemError(
                f"full pooled direction does not reproduce upstream: {held_family}"
            )
        calibration_diagnostics.append(
            {
                "held_dataset_family": held_family,
                "donor_dataset_families": list(donor_families),
                "n_jackknife_directions": len(donor_families),
                "full_direction_norm": float(np.linalg.norm(full_direction)),
                "jackknife_direction_norm_mean": float(
                    np.mean(np.linalg.norm(jackknife_directions, axis=1))
                ),
                "upstream_direction_reproduction_max_abs": max_error,
                "uses_labels": False,
            }
        )
        for cell in [item for item in run_cells if family_by_cell[item] == held_family]:
            values, diag = _score_cell(
                cell=cell,
                state=states[cell],
                full_direction=full_direction,
                jackknife_directions=jackknife_directions,
            )
            # The ungated direct arm must be exactly the already frozen E2 path.
            upstream_half = mechanics["scores"][("E2_B3_ORTH_IUPGRD_HALF", cell)]
            half_error = float(np.max(np.abs(values["J1_DIRECT_HALF"] - upstream_half)))
            if half_error > 1e-10:
                raise ResidualGraphDeemError(f"E2 alias failed: {cell}; {half_error}")
            diag["upstream_E2_score_reproduction_max_abs"] = half_error
            diag["dataset_family"] = held_family
            diagnostics.append(diag)
            for method in METHODS:
                score_cache[(method, cell)] = np.asarray(values[method], dtype=np.float64)

            path = score_dir / f"{cell}.npz"
            array_sha = atomic_save_npz(
                path,
                row_id=np.asarray(states[cell]["row_id"], dtype=str),
                method_order=np.asarray(METHODS, dtype=str),
                scores=np.column_stack([values[method] for method in METHODS]),
                reliability=np.asarray(values["reliability"], dtype=np.float64),
                permuted_reliability=np.asarray(
                    values["permuted_reliability"], dtype=np.float64
                ),
                sign_agreement=np.asarray(values["sign_agreement"], dtype=np.float64),
                relative_dispersion=np.asarray(
                    values["relative_dispersion"], dtype=np.float64
                ),
                full_orthogonal_correction_z=np.asarray(
                    values["full_orthogonal_correction_z"], dtype=np.float64
                ),
                jackknife_correction_z=np.asarray(
                    values["jackknife_correction_z"], dtype=np.float64
                ),
                iupgrd_z=np.asarray(values["iupgrd_z"], dtype=np.float64),
                reliability_row_permutation=np.asarray(
                    values["permutation"], dtype=np.int64
                ),
                donor_dataset_families=np.asarray(donor_families, dtype=str),
                full_direction=np.asarray(full_direction, dtype=np.float64),
                jackknife_directions=np.asarray(jackknife_directions, dtype=np.float64),
            )
            artifacts.append(
                {
                    "cell_id": cell,
                    "dataset_family": held_family,
                    "path": path.relative_to(args.out_dir).as_posix(),
                    "sha256": array_sha,
                }
            )

    source_sha = sha256_file(Path(__file__))
    freeze_payload = {
        "schema": "deem_b3_jackknife_consensus_v1_score_freeze",
        "status": "complete_before_label_import",
        "upstream_run_dir": str(args.run_dir.resolve()),
        "upstream_run_definition_sha256": sha256_file(
            args.run_dir / "RUN_DEFINITION.json"
        ),
        "upstream_fit_complete_sha256": sha256_file(args.run_dir / "FIT_COMPLETE.json"),
        "source_path": str(Path(__file__).resolve()),
        "source_sha256": source_sha,
        "methods": list(METHODS),
        "direct_trust": DIRECT_TRUST,
        "blend_alpha": BLEND_ALPHA,
        "promotion_delta": PROMOTION_DELTA,
        "consensus_formula": (
            "abs(mean(sign(delta_j))) / "
            "(1 + sd(delta_j)/(mean(abs(delta_j))+1e-12))"
        ),
        "jackknife_unit": "leave_one_donor_dataset_family_out",
        "artifacts": sorted(artifacts, key=lambda row: row["cell_id"]),
        "calibration_diagnostics": calibration_diagnostics,
        "cell_diagnostics": diagnostics,
        "labels_accessed": False,
        "label_module_imported": False,
        "natural_targets_previously_opened": True,
    }
    freeze_path = args.out_dir / "TARGET_FREE_SCORE_FREEZE.json"
    freeze_content_sha = _write_hashed_json(freeze_path, freeze_payload)
    if LABEL_MODULE in sys.modules:
        raise ResidualGraphDeemError("label module imported before score freeze")

    # Labels become importable only after the complete candidate roster exists
    # on disk and its manifest has been content-hashed.
    targets, sidecar_audit = _load_targets_after_preflight(
        hash_state, args.sidecar_dir
    )
    per_cell = []
    for cell in run_cells:
        bundle = hash_state["bundles"][cell]
        for method in METHODS:
            per_cell.append(
                {
                    "cell_id": cell,
                    "dataset_family": bundle.dataset_family,
                    "task_type": bundle.task_type,
                    "method": method,
                    **_metrics(targets[cell], score_cache[(method, cell)]),
                }
            )

    screen_cells = tuple(str(cell) for cell in hash_state["config"]["screen_cells"])
    held_cells = tuple(cell for cell in run_cells if cell not in set(screen_cells))
    if len(screen_cells) != 8 or len(held_cells) != 16:
        raise ResidualGraphDeemError("official screen/held16 split changed")
    panels = {
        "screen8": _panel(
            name="screen8",
            cells=screen_cells,
            per_cell=per_cell,
            bootstrap_draws=int(args.bootstrap_draws),
        ),
        "held16": _panel(
            name="held16",
            cells=held_cells,
            per_cell=per_cell,
            bootstrap_draws=int(args.bootstrap_draws),
        ),
        "all24_descriptive": _panel(
            name="all24_descriptive",
            cells=run_cells,
            per_cell=per_cell,
            bootstrap_draws=int(args.bootstrap_draws),
        ),
    }
    screen_blend = _comparison_lookup(
        panels["screen8"], "J5_BLEND_030_IUPGRD_CONSENSUS", "B3"
    )
    held_blend = _comparison_lookup(
        panels["held16"], "J5_BLEND_030_IUPGRD_CONSENSUS", "B3"
    )
    screen_control = _comparison_lookup(
        panels["screen8"],
        "J5_BLEND_030_IUPGRD_CONSENSUS",
        "J6_BLEND_030_IUPGRD_PERMUTED_CONSENSUS",
    )
    held_control = _comparison_lookup(
        panels["held16"],
        "J5_BLEND_030_IUPGRD_CONSENSUS",
        "J6_BLEND_030_IUPGRD_PERMUTED_CONSENSUS",
    )
    clear_win = (
        screen_blend["equal_dataset_family_auroc_delta"] >= PROMOTION_DELTA
        and held_blend["equal_dataset_family_auroc_delta"] >= PROMOTION_DELTA
        and screen_control["equal_dataset_family_auroc_delta"] > 0.0
        and held_control["equal_dataset_family_auroc_delta"] > 0.0
    )
    interpretation = (
        "The consensus IU+PGRD blend is a clear positive only if it reaches the "
        "+0.0025 equal-family AUROC bar and beats its row-permuted gate on both "
        "the official screen and held16. "
        + (
            "Those point-estimate conditions are met, but the result remains retrospective."
            if clear_win
            else "Those conditions are not jointly met, so the mechanism is not promoted."
        )
    )
    report = {
        "schema": "deem_b3_jackknife_consensus_v1_report",
        "status": "complete",
        "scientific_tier": "retrospective_exploratory_not_confirmation",
        "score_freeze_path": str(freeze_path.resolve()),
        "score_freeze_file_sha256": sha256_file(freeze_path),
        "score_freeze_content_sha256": freeze_content_sha,
        "strict_label_boundary": True,
        "screen_and_held16_reported_separately": True,
        "no_post_screen_roster_or_hyperparameter_change": True,
        "methods": list(METHODS),
        "panels": panels,
        "clear_positive_joint_gate": clear_win,
        "interpretation": interpretation,
        "sidecar_audit": sidecar_audit,
        "claim_boundary": (
            "All 24 natural target sets were previously opened. Held16 is a "
            "separated retrospective stress panel, not untouched confirmation."
        ),
    }
    _write_csv(args.out_dir / "PER_CELL_METRICS.csv", per_cell)
    _write_csv(args.out_dir / "CONSENSUS_DIAGNOSTICS.csv", diagnostics)
    _write_hashed_json(args.out_dir / "REPORT.json", report)
    (args.out_dir / "REPORT.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
