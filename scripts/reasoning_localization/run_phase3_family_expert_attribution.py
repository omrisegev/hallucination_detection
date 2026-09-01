#!/usr/bin/env python3
"""Freeze and evaluate the one-family-at-a-time H2 IU attribution ladder."""

from __future__ import annotations

import csv
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

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
from spectral_utils.token_local_fusion import IU_CONFIG, fit_local_equal_family  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization import run_phase3_compact_fusion as p3  # noqa: E402
from scripts.reasoning_localization import run_phase3_deployed_upcr_prune_refit as p3d  # noqa: E402
from scripts.reasoning_localization.register_phase3_family_expert_attribution import (  # noqa: E402
    EXPERIMENT,
    VARIANTS,
)

E0, E1, E2, E3, E4 = VARIANTS
H0 = "P3_H0_REFERENCE"
H2 = "P3A_H2_EQUAL_OUTER_REFERENCE"
ROOT = p1.PROGRAM_ROOT / "phase_3/family_expert_attribution"
OUTPUT = ROOT / "p3e_family_expert_attribution_v1"
REGISTRY = ROOT / "P3E_EXECUTION_REGISTRY.json"
SOURCE_H2 = p1.PROGRAM_ROOT / "phase_2/diagnostic/h3_reliability_fusion_v1/score_freeze/cells"
PRIMARY = ((E1, E0), (E2, E0), (E3, E0), (E4, E0))
FAMILY_SIZE = len(PRIMARY)
BENEFIT = 0.003
HARM = -0.003


class FamilyExpertError(RuntimeError):
    pass


def _fit_iu(donor: np.ndarray, held: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    model = upcr_fit(donor.T, **dict(IU_CONFIG))
    weights, corr, flipped = p3d._oriented_weights(model, donor)
    risk = -(held @ weights)
    return risk, {
        "weights": weights.tolist(),
        "anchor_correlation": corr,
        "orientation_flipped": flipped,
        "g2_hat": float(model.g2_hat),
        "projection_residual": float(model.proj_residual),
        "n_components": int(model.n_components_used),
    }


def _load_registry(release: Path) -> dict[str, Any]:
    row = json.loads(REGISTRY.read_text(encoding="utf-8"))
    required = {
        "schema": "reasoning-localization-p3e-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": EXPERIMENT,
        "variant_order": list(VARIANTS),
        "primary_contrasts": [list(pair) for pair in PRIMARY],
        "multiplicity_family_size": FAMILY_SIZE,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    for key, value in required.items():
        if row.get(key) != value:
            raise FamilyExpertError(f"execution registry mismatch: {key}")
    if Path(row["release_root"]).resolve() != release.resolve():
        raise FamilyExpertError("release mismatch")
    return row


def freeze(release: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    score_root = OUTPUT / "score_freeze"
    score_root.mkdir(parents=True)
    input_root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    records = []
    for position, cell_id in enumerate(p2r.PB_CELLS, start=1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        prep, raw, names, families = p3d._member_matrix(cell)
        if list(names) != registry["member_names"] or list(families) != registry["member_families"]:
            raise FamilyExpertError(f"roster drift in {cell_id}")
        indices = {
            family: np.asarray([i for i, value in enumerate(families) if value == family], dtype=np.int64)
            for family in ("entropy_level", "entropy_dynamics", "partition_energy", "topk_distribution")
        }
        if len(indices["entropy_level"]) != 1 or any(len(indices[name]) < 3 for name in indices if name != "entropy_level"):
            raise FamilyExpertError("family dimension contract failed")

        owner = np.repeat(np.arange(len(cell.row_ids)), np.diff(np.asarray(cell.token_offsets)))
        token_scores = {variant: np.full(len(raw), np.nan) for variant in VARIANTS}
        fold_diagnostics = []
        for fold in range(5):
            held_rows = np.flatnonzero(np.asarray(prep.row_folds) == fold)
            held_indices = np.flatnonzero(np.isin(owner, held_rows))
            fit_folds = np.asarray(prep.row_folds)[np.asarray(prep.fit_row_indices)]
            donor_indices = np.asarray(prep.fit_indices)[fit_folds != fold]
            donor, held, scale = p3d._fold_standardize(raw, donor_indices, held_indices)
            equal = {family: -held[:, idx].mean(axis=1) for family, idx in indices.items()}
            iu = {}
            diagnostics = {}
            for family in ("entropy_dynamics", "partition_energy", "topk_distribution"):
                idx = indices[family]
                iu[family], diagnostics[family] = _fit_iu(donor[:, idx], held[:, idx])
            level = equal["entropy_level"]
            token_scores[E0][held_indices] = np.mean([level, equal["entropy_dynamics"], equal["partition_energy"], equal["topk_distribution"]], axis=0)
            token_scores[E1][held_indices] = np.mean([level, iu["entropy_dynamics"], equal["partition_energy"], equal["topk_distribution"]], axis=0)
            token_scores[E2][held_indices] = np.mean([level, equal["entropy_dynamics"], iu["partition_energy"], equal["topk_distribution"]], axis=0)
            token_scores[E3][held_indices] = np.mean([level, equal["entropy_dynamics"], equal["partition_energy"], iu["topk_distribution"]], axis=0)
            token_scores[E4][held_indices] = np.mean([level, iu["entropy_dynamics"], iu["partition_energy"], iu["topk_distribution"]], axis=0)
            fold_diagnostics.append({"fold": fold, "scale": scale, "family_iu": diagnostics})
        if any(not np.isfinite(score).all() for score in token_scores.values()):
            raise FamilyExpertError(f"incomplete cross-fit score in {cell_id}")

        h0_token = np.asarray(fit_local_equal_family(prep).token_risk)
        h0_local = p1.topk_step_mean(h0_token, cell.segment_starts, cell.segment_ends, k=10)
        arrays = {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "h0_combined": p1.combine_with_common_detector(cell, h0_local),
            "h2_local": load_npz_no_pickle(SOURCE_H2 / cell_id / "scores.npz")["h2_local"],
        }
        for variant, score in token_scores.items():
            arrays[f"{variant.lower()}_local"] = p1.topk_step_mean(score, cell.segment_starts, cell.segment_ends, k=10)
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True)
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        record = {
            "schema": "reasoning-localization-p3e-cell-v1",
            "experiment_id": EXPERIMENT,
            "cell_id": cell_id,
            "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id),
            "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids),
            "member_names": list(names),
            "member_families": list(families),
            "family_counts": {key: len(value) for key, value in indices.items()},
            "fold_diagnostics": fold_diagnostics,
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
            "score_sha256": score_sha,
            "prepared_input_sha256": sha256_file(input_path),
        }
        record["payload_sha256"] = c1.payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({"cell_id": cell_id, "record_path": f"cells/{cell_id}/RECORD.json", "record_sha256": sha256_file(target / "RECORD.json"), "score_sha256": score_sha})
        print(f"score-freeze P3E0-P3E4: {cell_id} ({position}/8)", flush=True)
    result = {
        "schema": "reasoning-localization-p3e-score-freeze-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT,
        "variant_ids": list(VARIANTS),
        "records": records,
        "labels_seen_during_fit": False,
        "execution_registry_sha256": sha256_file(REGISTRY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["payload_sha256"] = c1.payload_sha(result)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", result)
    return result


def _verified(manifest: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for item in manifest["records"]:
        record_path = OUTPUT / "score_freeze" / item["record_path"]
        score_path = record_path.parent / "scores.npz"
        if sha256_file(record_path) != item["record_sha256"] or sha256_file(score_path) != item["score_sha256"]:
            raise FamilyExpertError("score-freeze hash mismatch")
        out[item["cell_id"]] = {"record": json.loads(record_path.read_text()), "arrays": load_npz_no_pickle(score_path)}
    return out


def _rows(verified: Mapping[str, Any], labels: Mapping[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    out = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
        record, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        for index, row_id in enumerate(arrays["row_ids"].astype(str)):
            lo, hi = map(int, arrays["segment_offsets"][index:index + 2])
            group_id, first_error = labels[cell_id][row_id]
            out[record["model_id"]].append({"row_id": row_id, "group_id": group_id, "slice_id": record["slice_id"], "cell_id": cell_id, "model_id": record["model_id"], "first_error": first_error, "step_scores": arrays[key][lo:hi].tolist(), "step_lengths": arrays["segment_lengths"][lo:hi].tolist()})
    return out


def _status(delta: float, lo: float, hi: float) -> str:
    if lo > BENEFIT:
        return "SUPPORTED_IMPROVEMENT"
    if hi < HARM:
        return "SUPPORTED_HARM"
    if delta > 0 and lo <= 0:
        return "PROMISING_UNCONFIRMED"
    return "INCONCLUSIVE"


def _contrast(left: str, right: str, metric: str, arms: Mapping[str, Any], simultaneous: bool) -> dict[str, Any]:
    lp = {row["metric_id"]: row for row in arms[left]["panels"]}[metric]
    rp = {row["metric_id"]: row for row in arms[right]["panels"]}[metric]
    draws = np.asarray(arms[left]["samples"][metric]) - np.asarray(arms[right]["samples"][metric])
    q = 0.025 / FAMILY_SIZE if simultaneous and metric == "official_macro_f1" else 0.025
    left_cells = {row["cell_id"]: row for row in arms[left]["by_cell"]}
    right_cells = {row["cell_id"]: row for row in arms[right]["by_cell"]}
    cells = {cell: float(left_cells[cell][metric]) - float(right_cells[cell][metric]) for cell in left_cells}
    delta = float(lp["value"] - rp["value"])
    lo, hi = float(np.quantile(draws, q)), float(np.quantile(draws, 1 - q))
    return {"contrast_id": f"pb::{left}::{right}::{metric}", "left_variant_id": left, "right_variant_id": right, "metric_id": "macro_f1" if metric == "official_macro_f1" else metric, "delta": delta, "ci_low": lo, "ci_high": hi, "statistical_status": _status(delta, lo, hi), "wins": sum(v > 1e-12 for v in cells.values()), "ties": sum(abs(v) <= 1e-12 for v in cells.values()), "losses": sum(v < -1e-12 for v in cells.values()), "worst_unit_delta": min(cells.values()), "worst_unit_id": min(cells, key=cells.get), "multiplicity_family_size": FAMILY_SIZE if simultaneous and metric == "official_macro_f1" else 1}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, panels: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> None:
    ids = [H2, E0, E1, E2, E3, E4]
    values = {row["arm_id"]: float(row["value"]) for row in panels if row["metric_id"] == "official_macro_f1" and row["arm_id"] in ids}
    lo, hi = min(values.values()) - .004, max(values.values()) + .004
    x = lambda value: 285 + (value - lo) / (hi - lo) * 680
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="560" viewBox="0 0 1040 560">', '<rect width="100%" height="100%" fill="white"/>', '<style>text{font-family:system-ui;fill:#172033}.t{font-size:22px;font-weight:700}.l{font-size:13px}.b{font-size:13px;font-weight:600}</style>', '<text x="25" y="34" class="t">One-family-at-a-time IU attribution</text>', '<text x="25" y="57" class="l">Cross-fitted H2 family experts; ProcessBench macro F1</text>']
    for i, variant in enumerate(ids):
        y = 95 + 42 * i
        parts += [f'<text x="25" y="{y+5}" class="b">{variant}</text>', f'<line x1="285" y1="{y}" x2="{x(values[variant]):.1f}" y2="{y}" stroke="#2563eb" stroke-width="7"/>', f'<circle cx="{x(values[variant]):.1f}" cy="{y}" r="6" fill="#7c3aed"/>', f'<text x="{x(values[variant])+10:.1f}" y="{y+5}" class="b">{values[variant]:.6f}</text>']
    parts.append('<text x="25" y="375" class="t">Candidate minus matched equal parent</text>')
    macro = [row for row in contrasts if row["metric_id"] == "macro_f1" and row["right_variant_id"] == E0]
    for i, row in enumerate(macro):
        y = 408 + 29 * i
        parts += [f'<text x="25" y="{y}" class="l">{row["left_variant_id"]}</text>', f'<text x="600" y="{y}" class="l">{row["delta"]:+.5f} [{row["ci_low"]:+.5f}, {row["ci_high"]:+.5f}] {row["statistical_status"]}</text>']
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n")


def evaluate(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = _verified(manifest)
    labels = p1._load_pb_labels(release)
    evaluator = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    h0 = c1.evaluate_arm(H0, _rows(verified, labels, "h0_combined"), evaluator)
    arms = {H0: h0}
    arms[H2] = p3._rerank(H2, h0, _rows(verified, labels, "h2_local"), evaluator)
    for variant in VARIANTS:
        arms[variant] = p3._rerank(variant, h0, _rows(verified, labels, f"{variant.lower()}_local"), evaluator)
    abstain = {(row["cell_id"], row["row_id"]): int(row["prediction_step"]) == -1 for row in h0["decisions"]}
    mismatches = {arm: sum((int(row["prediction_step"]) == -1) != abstain[(row["cell_id"], row["row_id"])] for row in arms[arm]["decisions"]) for arm in [H2, *VARIANTS]}
    if any(mismatches.values()):
        raise FamilyExpertError(f"H0 abstention alias failed: {mismatches}")
    pairs = [*PRIMARY, (E0, H2)]
    contrasts = [_contrast(left, right, metric, arms, (left, right) in PRIMARY) for left, right in pairs for metric in p1.PB_METRICS]
    evaluation_root = OUTPUT / "evaluation"
    evaluation_root.mkdir()
    panels = [row for arm in arms.values() for row in arm["panels"]]
    _write_csv(evaluation_root / "PROCESSBENCH_PANELS.csv", panels)
    _write_csv(evaluation_root / "PROCESSBENCH_BY_CELL.csv", [row for arm in arms.values() for row in arm["by_cell"]])
    _write_csv(evaluation_root / "PAIRWISE_CONTRASTS.csv", contrasts)
    primary = [row for row in contrasts if row["metric_id"] == "macro_f1" and (row["left_variant_id"], row["right_variant_id"]) in PRIMARY]
    eligible = [row["left_variant_id"] for row in primary if row["delta"] > 0 and row["ci_high"] >= HARM and row["worst_unit_delta"] >= -.020]
    summary = {"schema": "reasoning-localization-p3e-evaluation-v1", "status": "COMPLETE", "experiment_id": EXPERIMENT, "primary_contrasts": primary, "development_eligible_families": eligible, "abstention_mismatches": mismatches, "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED}
    summary["payload_sha256"] = c1.payload_sha(summary)
    atomic_write_json(evaluation_root / "SUMMARY.json", summary)
    _plot(evaluation_root / "P3E_RESULTS.svg", panels, contrasts)
    return summary


def main() -> None:
    started = time.perf_counter()
    release = p1.DEFAULT_RELEASE.resolve()
    registry = _load_registry(release)
    frozen = freeze(release, registry)
    summary = evaluate(release, frozen)
    atomic_write_json(OUTPUT / "RUN_COMPLETE.json", {"status": "COMPLETE", "elapsed_seconds": time.perf_counter() - started, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

