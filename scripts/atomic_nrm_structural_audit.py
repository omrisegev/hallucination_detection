#!/usr/bin/env python3
"""Freeze the label-free structural calibration for Atomic NRM candidate v1.

This script intentionally has no label loader and no metric import.  It reads
only the mixed-v2 feature matrices in the frozen 23-cell source roster.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import load_contract  # noqa: E402
from spectral_utils.atomic_neutral_residual import (  # noqa: E402
    atomic_contribution_space,
    atomic_neutral_score,
    fit_atomic_neutral_calibration,
)
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


VERSION = "atomic-neutral-residual-projector-cs-iu-candidate-v1-2026-08-13"
DEFAULT_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_OUT = REPO / "results" / "atomic_nrm_structural_audit_v1"
NULL_DRAWS = 1000
STABILITY_NULL_DRAWS = 200
SOURCE_CELLS = (
    "ars_gsm8k_r1distill8b",
    "internalstates_gsm8k_qwen25_7b",
    "lapeigvals_gsm8k_llama3b",
    "lapeigvals_gsm8k_llama8b",
    "lapeigvals_gsm8k_mistral24b",
    "lapeigvals_gsm8k_nemo",
    "lapeigvals_gsm8k_phi35",
    "noise_gsm8k_mistral7b",
    "noise_gsm8k_phi3mini",
    "trace_gsm8k_llama8b_k10",
    "losnet_hotpotqa_mistral7b",
    "math500_dsmath7b",
    "math500_qwenmath7b",
    "math500_r1distill8b",
    "math500_r1distill8b_mn4096",
    "trace_math500_qwenmath15b_k10",
    "se_nq_open_llama8b",
    "sciq_llama8b",
    "se_squad_v2_llama8b",
    "epr_triviaqa_mistral24b",
    "seiclr_triviaqa_opt30b",
    "semenergy_triviaqa_qwen3_8b",
    "truthfulqa_llama8b",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()


def write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calibration_payload(calibration):
    return {
        "feature_names": list(calibration.feature_names),
        "direction": calibration.direction.tolist(),
        "anchor": calibration.anchor.tolist(),
        "residual_covariance": calibration.residual_covariance.tolist(),
        "pair_counts": calibration.pair_counts.tolist(),
        "eigenvalues": calibration.eigenvalues.tolist(),
        "neutral_mask": calibration.neutral_mask.tolist(),
        "null_lower": calibration.null_lower,
        "null_upper": calibration.null_upper,
        "null_draws": calibration.null_draws,
        "null_alpha": calibration.null_alpha,
        "null_seed": calibration.null_seed,
        "diagnostics": calibration.diagnostics,
        "uses_labels": False,
    }


def load_spaces(bundle_path):
    bundle = np.load(bundle_path, allow_pickle=True)
    rows, spaces, weights = [], [], []
    for cell in SOURCE_CELLS:
        F, names = load_contract(bundle, cell, "mixed_v2")
        fitted = upcr_fit(F, **IU_FIT_DEFAULTS)
        space = atomic_contribution_space(F, names, fitted.w)
        absolute = np.abs(fitted.w)
        relative = absolute / max(float(np.max(absolute)), 1e-30)
        rows.append({
            "cell": cell,
            "n_samples": int(F.shape[1]),
            "n_features": int(F.shape[0]),
            "minimum_abs_iu_weight": float(np.min(absolute)),
            "maximum_abs_iu_weight": float(np.max(absolute)),
            "minimum_relative_iu_weight": float(np.min(relative)),
            "n_relative_weight_below_1e_6": int(np.sum(relative < 1e-6)),
            "atomic_reconstruction_error": space.diagnostics[
                "reconstruction_error"
            ],
        })
        spaces.append(space)
        weights.append(np.asarray(fitted.w, dtype=float))
    return rows, spaces, weights


def stability_rows(spaces, calibration):
    rows = []
    reference = calibration.direction
    frozen_names = calibration.feature_names
    for held_out in range(len(spaces)):
        kept = [space for index, space in enumerate(spaces) if index != held_out]
        candidate = fit_atomic_neutral_calibration(
            kept,
            feature_names=frozen_names,
            minimum_cell_fraction=1.0,
            null_draws=STABILITY_NULL_DRAWS,
            null_seed=calibration.null_seed,
        )
        rows.append({
            "held_out_cell": SOURCE_CELLS[held_out],
            "direction_absolute_cosine": float(abs(
                reference @ candidate.direction
            )),
            "neutral_dimension": int(np.sum(candidate.neutral_mask)),
            "null_lower": candidate.null_lower,
            "null_upper": candidate.null_upper,
        })
    return rows


def report_text(payload, feature_rows, stability):
    calibration = payload["calibration"]
    selected = [
        value for value, keep in zip(
            calibration["eigenvalues"], calibration["neutral_mask"]
        ) if keep
    ]
    cosines = np.asarray([
        row["direction_absolute_cosine"] for row in stability
    ])
    return f"""# Atomic NRM candidate v1 — label-free structural audit

- Version: `{VERSION}`
- Source telemetry: frozen 23-cell original roster; no correctness field was loaded.
- Atoms seen / eligible in every cell: {calibration['diagnostics']['n_seen_features']} / {len(calibration['feature_names'])}
- Eligible atoms: `{', '.join(calibration['feature_names'])}`
- Excluded for incomplete source coverage: `{', '.join(calibration['diagnostics']['excluded_features'])}`
- Permutation-null simultaneous interval: [{calibration['null_lower']:.6f}, {calibration['null_upper']:.6f}]
- Neutral dimension: {calibration['diagnostics']['neutral_dimension']}
- Retained eigenvalues: `{', '.join(f'{value:.6f}' for value in selected)}`
- Symmetric-anchor retained norm: {calibration['diagnostics']['anchor_retained_norm']:.6f}
- Leave-one-cell direction |cosine|: min {cosines.min():.6f}, median {np.median(cosines):.6f}, max {cosines.max():.6f}
- Minimum relative IU weight across source cells: {min(row['minimum_relative_iu_weight'] for row in feature_rows):.6g}; no atom was numerically inactive.
- Fixed correction scale: `1/sqrt(p)` standard deviations, with p={len(calibration['feature_names'])}.
- Direction SHA-256: `{payload['hashes']['direction_sha256']}`
- Covariance SHA-256: `{payload['hashes']['covariance_sha256']}`

This audit establishes only null geometry and affine/invariance properties.  It
does not identify a hallucination-target direction and reports no AUROC.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feature_rows, spaces, weights = load_spaces(Path(args.bundle))
    calibration = fit_atomic_neutral_calibration(
        spaces,
        minimum_cell_fraction=1.0,
        null_draws=NULL_DRAWS,
    )
    for row, space, cell_weights in zip(feature_rows, spaces, weights):
        scored = atomic_neutral_score(space, cell_weights, calibration)
        row.update({
            "correction_scale": scored.diagnostics["correction_scale"],
            "baseline_correction_covariance": scored.diagnostics[
                "baseline_correction_covariance"
            ],
            "weight_reconstruction_error": scored.diagnostics[
                "weight_reconstruction_error"
            ],
        })
    stability = stability_rows(spaces, calibration)
    payload = {
        "version": VERSION,
        "source_cells": list(SOURCE_CELLS),
        "source_contract": "mixed_v2",
        "iu_fit_defaults": dict(IU_FIT_DEFAULTS),
        "calibration": calibration_payload(calibration),
        "hashes": {
            "bundle_sha256": sha256_file(args.bundle),
            "direction_sha256": sha256_array(calibration.direction),
            "covariance_sha256": sha256_array(
                calibration.residual_covariance
            ),
        },
        "label_access": False,
    }
    np.savez_compressed(
        out / "calibration.npz",
        feature_names=np.asarray(calibration.feature_names),
        direction=calibration.direction,
        anchor=calibration.anchor,
        residual_covariance=calibration.residual_covariance,
        pair_counts=calibration.pair_counts,
        eigenvalues=calibration.eigenvalues,
        neutral_mask=calibration.neutral_mask,
        null_lower=np.asarray(calibration.null_lower),
        null_upper=np.asarray(calibration.null_upper),
    )
    write_csv(out / "feature_diagnostics.csv", feature_rows)
    write_csv(out / "leave_one_cell_stability.csv", stability)
    write_json(out / "structural_audit.json", payload)
    (out / "report.md").write_text(
        report_text(payload, feature_rows, stability), encoding="utf-8"
    )
    artifact_hashes = {
        path.name: sha256_file(path) for path in sorted(out.iterdir())
        if path.is_file()
    }
    write_json(out / "artifact_hashes.json", artifact_hashes)
    print(json.dumps({
        "out": str(out),
        "eligible_features": len(calibration.feature_names),
        "neutral_dimension": int(np.sum(calibration.neutral_mask)),
        "direction_sha256": payload["hashes"]["direction_sha256"],
        "minimum_leave_one_cell_cosine": min(
            row["direction_absolute_cosine"] for row in stability
        ),
        "label_access": False,
    }, indent=2))


if __name__ == "__main__":
    main()
