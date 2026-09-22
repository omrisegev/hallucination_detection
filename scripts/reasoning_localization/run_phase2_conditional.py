#!/usr/bin/env python3
"""Freeze and evaluate one member of the registered Phase-2C conditional roster."""

from __future__ import annotations

import argparse
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

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import load_prepared_localization_cell, validate_fit_manifest  # noqa: E402
from spectral_utils.token_local_fusion import fit_local_equal_family, prepare_localization_cell  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_remaining as atomic  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization.register_phase2_conditional_contract import PARENT, VARIANTS  # noqa: E402

ROOT = p1.PROGRAM_ROOT / "phase_2/conditional"
FAMILIES = ("entropy_level", "entropy_dynamics", "sampled_token_energy", "partition_energy", "topk_distribution")
FAMILY_REMOVE = {
    "P2C_F6_MINUS_ENTROPY_LEVEL": "entropy_level",
    "P2C_F6_MINUS_ENTROPY_DYNAMICS": "entropy_dynamics",
    "P2C_F6_MINUS_SAMPLED_ENERGY": "sampled_token_energy",
    "P2C_F6_MINUS_PARTITION_ENERGY": "partition_energy",
    "P2C_F6_MINUS_TOPK_DISTRIBUTION": "topk_distribution",
}
VIEW_REMOVE = {
    "P2C_F6_MINUS_ENTROPY_SWVAR16_VIEW": "entropy_sw_var_series",
    "P2C_F6_MINUS_ENTROPY_CUSUM_VIEW": "entropy_cusum_abs_series",
    "P2C_F6_MINUS_SAMPLED_LEVEL_VIEW": "spilled_series",
    "P2C_F6_MINUS_PARTITION_LEVEL_VIEW": "energy_series",
}
PRIMARY_FAMILY_SIZE = 13


class ConditionalError(RuntimeError):
    pass


def _weights(prep: Any, families: tuple[str, ...], removed_view: str | None = None) -> np.ndarray:
    weights = np.zeros(prep.n_features, dtype=np.float64)
    for family in families:
        members = [i for i, (name, fam) in enumerate(zip(prep.kept_stream_names, prep.kept_family_names))
                   if fam == family and name != removed_view]
        if not members:
            raise ConditionalError(f"family {family} has no members after removal")
        weights[members] = 1.0 / (len(families) * len(members))
    return weights


def _standardized_risk(curve: np.ndarray, fit_indices: np.ndarray) -> np.ndarray:
    fit = np.asarray(curve, dtype=np.float64)[np.asarray(fit_indices, dtype=np.int64)]
    mean, std = float(np.mean(fit)), float(np.std(fit))
    if not np.isfinite(std) or std <= 1e-8:
        raise ConditionalError("inserted curve has degenerate donor scale")
    return (np.asarray(curve, dtype=np.float64) - mean) / std


def _family_risk(prep: Any, family: str, removed_view: str | None = None) -> np.ndarray:
    members = [i for i, (name, fam) in enumerate(zip(prep.kept_stream_names, prep.kept_family_names))
               if fam == family and name != removed_view]
    if not members:
        raise ConditionalError(f"family {family} has no members")
    weights = np.zeros(prep.n_features, dtype=np.float64)
    weights[members] = 1.0 / len(members)
    return prep.token_risk(weights)


def _fit_rank(values: np.ndarray, fit_indices: np.ndarray) -> np.ndarray:
    donor = np.sort(np.asarray(values, dtype=np.float64)[np.asarray(fit_indices, dtype=np.int64)])
    x = np.asarray(values, dtype=np.float64)
    lo = np.searchsorted(donor, x, side="left")
    hi = np.searchsorted(donor, x, side="right")
    return (lo + hi + 1.0) / (2.0 * (len(donor) + 1.0))


def token_curve(variant: str, cell: Any) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    prep = prepare_localization_cell(cell)
    fitted = fit_local_equal_family(prep)
    parent = np.asarray(fitted.token_risk, dtype=np.float64)
    if variant == PARENT:
        return parent, parent, {"operation": "exact current five-family parent", "present_families": list(FAMILIES)}
    if variant in FAMILY_REMOVE:
        kept = tuple(f for f in FAMILIES if f != FAMILY_REMOVE[variant])
        return parent, prep.token_risk(_weights(prep, kept)), {"operation": "family leave-one-out", "removed": FAMILY_REMOVE[variant]}
    if variant in VIEW_REMOVE:
        removed = VIEW_REMOVE[variant]
        return parent, prep.token_risk(_weights(prep, FAMILIES, removed)), {"operation": "view leave-one-out", "removed": removed}
    if variant == "P2C_F6_PLUS_STRUCTURAL_CONTROL":
        return parent, prep.token_risk(_weights(prep, (*FAMILIES, "structural"))), {"operation": "structural insertion control"}
    if variant == "P2C_F6_SWAP_C1_SWVAR16":
        old = _family_risk(prep, "entropy_dynamics", "entropy_sw_var_series")
        n_old = sum(f == "entropy_dynamics" and n != "entropy_sw_var_series" for n, f in zip(prep.kept_stream_names, prep.kept_family_names))
        entropy = atomic.primitive_risks(cell)["entropy"]
        inserted = _standardized_risk(c1.response_reset_swvar(entropy, cell.token_offsets), prep.fit_indices)
        dynamics = (n_old * old + inserted) / (n_old + 1)
        others = [_family_risk(prep, f) for f in FAMILIES if f != "entropy_dynamics"]
        return parent, np.mean([dynamics, *others], axis=0), {"operation": "SWVar member swap", "n_surviving_dynamics": n_old}
    if variant == "P2C_F6_PLUS_C7_EDIS_VIEW":
        old = _family_risk(prep, "entropy_dynamics")
        n_old = sum(f == "entropy_dynamics" for f in prep.kept_family_names)
        entropy = atomic.primitive_risks(cell)["entropy"]
        onset = atomic.response_map(entropy, cell.token_offsets, atomic.edis_onset)
        inserted = _standardized_risk(onset, prep.fit_indices)
        dynamics = (n_old * old + inserted) / (n_old + 1)
        others = [_family_risk(prep, f) for f in FAMILIES if f != "entropy_dynamics"]
        return parent, np.mean([dynamics, *others], axis=0), {"operation": "C7 inserted inside entropy_dynamics", "n_original_dynamics": n_old}
    if variant == "P2C_F6_PLUS_C8_OUTER_EXPERT":
        _iu_parent, c8, diagnostics = atomic.fit_self_innovation(cell)
        candidate = 0.5 * (_fit_rank(parent, prep.fit_indices) + _fit_rank(c8, prep.fit_indices))
        return parent, candidate, {"operation": "equal donor-rank outer experts", "c8": dict(diagnostics)}
    raise KeyError(variant)


def output_root(variant: str) -> Path:
    return ROOT / variant.lower()


def registry_path(variant: str) -> Path:
    if variant == PARENT:
        return ROOT / f"{variant}_EXECUTION_REGISTRY.json"
    amendment = ROOT / f"{variant}_EXECUTION_REGISTRY_AMENDMENT_V2.json"
    return amendment if amendment.exists() else ROOT / f"{variant}_EXECUTION_REGISTRY.json"


def load_registry(path: Path, variant: str, release: Path) -> dict[str, Any]:
    row = json.loads(path.read_text())
    required = {"schema": "reasoning-localization-p2c-execution-v1", "status": "FROZEN_BEFORE_RUN",
                "variant_id": variant, "primary_family_size": PRIMARY_FAMILY_SIZE,
                "cells": list(p2r.PB_CELLS), "runner_sha256": sha256_file(Path(__file__).resolve())}
    for key, value in required.items():
        if row.get(key) != value:
            raise ConditionalError(f"registry mismatch: {key}")
    if Path(row["release_root"]).resolve() != release.resolve():
        raise ConditionalError("release mismatch")
    return row


def freeze(variant: str, release: Path, output: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    score_root = output / "score_freeze"
    score_root.mkdir(parents=True)
    input_root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    records, alias = [], 0.0
    for pos, cell_id in enumerate(p2r.PB_CELLS, 1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        parent_token, candidate_token, diagnostics = token_curve(variant, cell)
        parent_local = p1.topk_step_mean(parent_token, cell.segment_starts, cell.segment_ends, k=10)
        candidate_local = p1.topk_step_mean(candidate_token, cell.segment_starts, cell.segment_ends, k=10)
        if variant == PARENT:
            alias = max(alias, float(np.max(np.abs(parent_local - candidate_local))))
        arrays = {"row_ids": np.asarray(cell.row_ids, dtype="<U80"), "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
                  "segment_lengths": np.asarray(cell.segment_ends-cell.segment_starts, dtype="<i8"),
                  "parent_combined": p1.combine_with_common_detector(cell, parent_local),
                  "candidate_combined": p1.combine_with_common_detector(cell, candidate_local)}
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True)
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        record = {"schema": "reasoning-localization-p2c-cell-v1", "variant_id": variant, "cell_id": cell_id,
                  "model_id": str(cell.model_id), "slice_id": str(cell.slice_id), "population_id": str(cell.population_id),
                  "n_rows": len(cell.row_ids), "n_steps": len(candidate_local), "score_sha256": score_sha,
                  "prepared_input": str(input_path), "prepared_input_sha256": sha256_file(input_path),
                  "labels_seen_during_fit": False, "targets_accessed_during_fit": False, "diagnostics": diagnostics}
        record["payload_sha256"] = c1.payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({"cell_id": cell_id, "record_path": f"cells/{cell_id}/RECORD.json", "record_sha256": sha256_file(target/"RECORD.json"), "score_sha256": score_sha})
        print(f"score-freeze {variant}: {cell_id} ({pos}/8)", flush=True)
    result = {"schema": "reasoning-localization-p2c-score-freeze-v1", "status": "COMPLETE", "variant_id": variant,
              "cells": list(p2r.PB_CELLS), "parent_alias_max_abs_error": alias, "records": records,
              "execution_registry_sha256": sha256_file(registry_path(variant)), "runner_sha256": sha256_file(Path(__file__).resolve())}
    result["payload_sha256"] = c1.payload_sha(result)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", result)
    return result


def _verified(output: Path, freeze_manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for item in freeze_manifest["records"]:
        rp = output / "score_freeze" / item["record_path"]
        if sha256_file(rp) != item["record_sha256"]:
            raise ConditionalError("record hash mismatch")
        rec = json.loads(rp.read_text())
        sp = rp.parent / "scores.npz"
        if sha256_file(sp) != item["score_sha256"]:
            raise ConditionalError("score hash mismatch")
        result[item["cell_id"]] = {"record": rec, "arrays": load_npz_no_pickle(sp)}
    return result


def _rows(verified: Mapping[str, Any], labels: Mapping[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    result = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
        rec, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        offsets, lengths = arrays["segment_offsets"], arrays["segment_lengths"]
        for i, row_id in enumerate(arrays["row_ids"].astype(str)):
            lo, hi = map(int, offsets[i:i+2]); group_id, first_error = labels[cell_id][row_id]
            result[rec["model_id"]].append({"row_id": row_id, "group_id": group_id, "slice_id": rec["slice_id"], "cell_id": cell_id,
                "model_id": rec["model_id"], "first_error": first_error, "step_scores": arrays[key][lo:hi].tolist(), "step_lengths": lengths[lo:hi].tolist()})
    return result


def evaluate(variant: str, release: Path, output: Path, freeze_manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = _verified(output, freeze_manifest)
    labels = p1._load_pb_labels(release)
    evaluator = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    parent = c1.evaluate_arm(PARENT, _rows(verified, labels, "parent_combined"), evaluator)
    candidate = c1.evaluate_arm(variant, _rows(verified, labels, "candidate_combined"), evaluator)
    parent_panels = {row["metric_id"]: row for row in parent["panels"]}
    candidate_panels = {row["metric_id"]: row for row in candidate["panels"]}
    by_parent = {row["cell_id"]: row for row in parent["by_cell"]}
    contrasts = []
    for metric in p1.PB_METRICS:
        draws = np.asarray(candidate["samples"][metric]) - np.asarray(parent["samples"][metric])
        q = .025 / PRIMARY_FAMILY_SIZE if metric == "official_macro_f1" else .025
        cells = {row["cell_id"]: float(row[metric])-float(by_parent[row["cell_id"]][metric]) for row in candidate["by_cell"]}
        contrasts.append({"contrast_id": f"pb::{variant}::{PARENT}::{metric}", "left_variant_id": variant, "right_variant_id": PARENT,
            "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
            "candidate_minus_parent_delta": float(candidate_panels[metric]["value"]-parent_panels[metric]["value"]),
            "ci_low": float(np.quantile(draws, q)), "ci_high": float(np.quantile(draws, 1-q)),
            "wins": sum(v>1e-12 for v in cells.values()), "ties": sum(abs(v)<=1e-12 for v in cells.values()), "losses": sum(v<-1e-12 for v in cells.values()),
            "worst_unit_delta": min(cells.values()), "worst_unit_id": min(cells, key=cells.get),
            "multiplicity_family_size": PRIMARY_FAMILY_SIZE if metric == "official_macro_f1" else 1})
    contrast = next(row for row in contrasts if row["metric_id"] == "macro_f1")
    is_loo = variant in FAMILY_REMOVE or variant in VIEW_REMOVE
    contrast["conditional_contribution"] = -float(contrast["candidate_minus_parent_delta"]) if is_loo else float(contrast["candidate_minus_parent_delta"])
    contrast["contribution_sign"] = "parent-minus-ablated" if is_loo else "candidate-minus-parent"
    eval_root = output / "evaluation"; eval_root.mkdir()
    c1.write_csv(eval_root/"PROCESSBENCH_BY_CELL.csv", [*parent["by_cell"], *candidate["by_cell"]])
    c1.write_csv(eval_root/"PROCESSBENCH_PANELS.csv", [*parent["panels"], *candidate["panels"]])
    c1.write_csv(eval_root/"PAIRWISE_CONTRASTS.csv", contrasts)
    summary = {"schema": "reasoning-localization-p2c-evaluation-v1", "status": "COMPLETE", "variant_id": variant,
               "candidate_macro_f1": candidate_panels["official_macro_f1"]["value"], "parent_macro_f1": parent_panels["official_macro_f1"]["value"],
               "primary_contrast": contrast,
               "exact_error_delta": next(row for row in contrasts if row["metric_id"] == "first_error_exact")["candidate_minus_parent_delta"],
               "clean_abstention_delta": next(row for row in contrasts if row["metric_id"] == "clean_abstention_accuracy")["candidate_minus_parent_delta"],
               "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED}
    summary["payload_sha256"] = c1.payload_sha(summary); atomic_write_json(eval_root/"SUMMARY.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--variant", choices=VARIANTS, required=True); parser.add_argument("--release", type=Path, default=p1.DEFAULT_RELEASE)
    args = parser.parse_args(); release = args.release.resolve(); variant = args.variant; output = output_root(variant)
    registry = load_registry(registry_path(variant), variant, release)
    started = time.perf_counter(); frozen = freeze(variant, release, output, registry); summary = evaluate(variant, release, output, frozen)
    atomic_write_json(output/"RUN_COMPLETE.json", {"schema": "reasoning-localization-p2c-run-v1", "variant_id": variant, "status": "COMPLETE", "elapsed_seconds": time.perf_counter()-started, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
