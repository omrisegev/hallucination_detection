#!/usr/bin/env python3
"""Freeze and evaluate entropy plus causal adaptive trailing SWVar."""

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
    load_prepared_localization_cell, validate_fit_manifest,
)
from spectral_utils.fixed_application_pipelines import SHARED_TOKEN_VIEWS  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402


REFERENCE = c1.REFERENCE
CANDIDATE = "C2_ENT_SWADAPT"
ARMS = (REFERENCE, CANDIDATE)
ATOMIC_ROOT = p1.PROGRAM_ROOT / "phase_2/atomic"
OUTPUT_ROOT = ATOMIC_ROOT / CANDIDATE.lower()
REGISTRY_PATH = ATOMIC_ROOT / f"{CANDIDATE}_EXECUTION_REGISTRY.json"
PRIMARY_COMPARISON_FAMILY = 4
FRACTION = 0.10
MIN_WINDOW = 3
MAX_WINDOW = 32


class AtomicC2Error(RuntimeError):
    pass


def adaptive_window(prefix_length: int) -> int:
    if int(prefix_length) != prefix_length or prefix_length < 1:
        raise ValueError("prefix length must be a positive integer")
    return max(MIN_WINDOW, min(MAX_WINDOW, int(int(prefix_length) * FRACTION)))


def adaptive_trailing_population_variance(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("adaptive SWVar input must be a nonempty finite vector")
    prefix = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    prefix_sq = np.concatenate(([0.0], np.cumsum(x * x, dtype=np.float64)))
    output = np.empty_like(x)
    for index in range(len(x)):
        end = index + 1
        width = min(end, adaptive_window(end))
        start = end - width
        total = prefix[end] - prefix[start]
        total_sq = prefix_sq[end] - prefix_sq[start]
        output[index] = max(0.0, total_sq / width - (total / width) ** 2)
    output[0] = 0.0
    return output


def response_reset_adaptive_swvar(values: Sequence[float], offsets: Sequence[int]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    bounds = np.asarray(offsets, dtype=np.int64)
    if bounds.ndim != 1 or bounds[0] != 0 or bounds[-1] != len(x) or np.any(np.diff(bounds) <= 0):
        raise ValueError("response offsets do not partition the token curve")
    output = np.empty_like(x)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        output[int(lo):int(hi)] = adaptive_trailing_population_variance(x[int(lo):int(hi)])
    return output


def adaptive_suffix_invariance_audit(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    full = adaptive_trailing_population_variance(x)
    cuts = sorted({1, min(3, len(x)), min(40, len(x)), min(320, len(x)),
                   max(1, len(x) // 2), max(1, len(x) - 1), len(x)})
    return max(float(np.max(np.abs(full[:cut] - adaptive_trailing_population_variance(x[:cut])))) for cut in cuts)


def require_sources(registry: Mapping[str, Any]) -> None:
    for source in registry["frozen_sources"]:
        path = Path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise AtomicC2Error(f"frozen source changed or missing: {source['role']}")


def load_registry(path: Path, release: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": "reasoning-localization-phase2-atomic-c2-execution-registry-v1",
        "status": "FROZEN_BEFORE_RUN", "candidate": CANDIDATE,
        "atomic_reference": REFERENCE, "processbench_cells": list(p2r.PB_CELLS),
        "fraction": FRACTION, "min_window": MIN_WINDOW, "max_window": MAX_WINDOW,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "primary_comparison_family_size": PRIMARY_COMPARISON_FAMILY,
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise AtomicC2Error(f"execution registry mismatch for {key}")
    if Path(registry["release_root"]).resolve() != release.resolve():
        raise AtomicC2Error("release root differs from registry")
    if registry["runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise AtomicC2Error("runner changed after freeze")
    require_sources(registry)
    return registry


def freeze_scores(release: Path, output: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite C2 output: {output}")
    score_root = output / "score_freeze"
    score_root.mkdir(parents=True, exist_ok=False)
    input_root = release / "build_A/localization/inputs"
    manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_manifest(manifest_path, input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    entropy_index = SHARED_TOKEN_VIEWS.index("entropy_series")
    records = []
    alias_local = alias_combined = suffix_error = 0.0
    started = time.perf_counter()
    for position, cell_id in enumerate(p2r.PB_CELLS, start=1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        entropy_risk = -np.asarray(cell.token_confidence[:, entropy_index], dtype=np.float64)
        adaptive_risk = response_reset_adaptive_swvar(entropy_risk, cell.token_offsets)
        for lo, hi in zip(cell.token_offsets[:-1], cell.token_offsets[1:]):
            suffix_error = max(suffix_error, adaptive_suffix_invariance_audit(entropy_risk[int(lo):int(hi)]))
        entropy_step = p1.topk_step_mean(entropy_risk, cell.segment_starts, cell.segment_ends, k=10)
        adaptive_step = p1.topk_step_mean(adaptive_risk, cell.segment_starts, cell.segment_ends, k=10)
        candidate_local = c1.fuse_step_channels(entropy_step, adaptive_step)
        reference_combined = p1.combine_with_common_detector(cell, entropy_step)
        candidate_combined = p1.combine_with_common_detector(cell, candidate_local)
        prior = load_npz_no_pickle(c1.P2R_TOP10_ROOT / "score_freeze/cells" / cell_id / "scores.npz")
        alias_local = max(alias_local, float(np.max(np.abs(entropy_step - prior["local_step_scores"]))))
        alias_combined = max(alias_combined, float(np.max(np.abs(reference_combined - prior["combined_step_scores"]))))
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True, exist_ok=False)
        score_path = target / "scores.npz"
        score_sha = atomic_write_npz(score_path, {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "reference_local_step_scores": np.asarray(entropy_step, dtype="<f8"),
            "reference_combined_step_scores": np.asarray(reference_combined, dtype="<f8"),
            "candidate_local_step_scores": np.asarray(candidate_local, dtype="<f8"),
            "candidate_combined_step_scores": np.asarray(candidate_combined, dtype="<f8"),
            "adaptive_swvar_step_scores": np.asarray(adaptive_step, dtype="<f8"),
        })
        record = {
            "schema": "reasoning-localization-phase2-atomic-c2-cell-v1", "cell_id": cell_id,
            "model_id": str(cell.model_id), "slice_id": str(cell.slice_id),
            "population_id": str(cell.population_id), "n_rows": len(cell.row_ids),
            "n_steps": len(entropy_step), "prepared_input": str(input_path),
            "prepared_input_sha256": sha256_file(input_path), "score_file": "scores.npz",
            "score_sha256": score_sha, "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
        }
        record["payload_sha256"] = c1.payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({"cell_id": cell_id, "record_path": f"cells/{cell_id}/RECORD.json",
                        "record_sha256": sha256_file(target / "RECORD.json"), "score_sha256": score_sha})
        print(f"score-freeze {CANDIDATE}: {cell_id} ({position}/8)", flush=True)
    require_sources(registry)
    freeze = {
        "schema": "reasoning-localization-phase2-atomic-c2-score-freeze-v1", "status": "COMPLETE",
        "candidate": CANDIDATE, "atomic_reference": REFERENCE, "cells": list(p2r.PB_CELLS),
        "labels_seen_during_fit": False, "targets_accessed_during_fit": False,
        "reference_local_alias_max_abs_error": alias_local,
        "reference_combined_alias_max_abs_error": alias_combined,
        "suffix_invariance_max_abs_error": suffix_error,
        "input_manifest_sha256": sha256_file(manifest_path),
        "execution_registry_sha256": sha256_file(Path(registry["registry_path"])),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "elapsed_seconds": time.perf_counter() - started, "records": records,
    }
    freeze["payload_sha256"] = c1.payload_sha(freeze)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", freeze)
    return freeze


def build_contrasts(candidate: Mapping[str, Any], reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    comparators = {REFERENCE: reference, "R1_ENTROPY_TOP5": c1.comparator_top5()}
    output = []
    for comparator_id, comparator in comparators.items():
        right_cells = {str(row["cell_id"]): row for row in comparator["by_cell"]}
        for metric in p1.PB_METRICS:
            left_point = float(np.mean([float(row[metric]) for row in candidate["by_cell"]]))
            right_point = float(np.mean([float(right_cells[row["cell_id"]][metric]) for row in candidate["by_cell"]]))
            draws = np.asarray(candidate["samples"][metric]) - np.asarray(comparator["samples"][metric])
            q = 0.025 / PRIMARY_COMPARISON_FAMILY if metric == "official_macro_f1" else 0.025
            cell_deltas = {str(row["cell_id"]): float(row[metric]) - float(right_cells[str(row["cell_id"])][metric]) for row in candidate["by_cell"]}
            family_deltas = {family: float(np.mean([value for cell_id, value in cell_deltas.items() if str(right_cells[cell_id]["slice_id"]) == family])) for family in p1.FAMILIES}
            eps = 1e-12
            output.append({
                "contrast_id": f"pb::{CANDIDATE}::{comparator_id}::{metric}",
                "left_variant_id": CANDIDATE, "right_variant_id": comparator_id,
                "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                "source_metric_id": metric, "delta": left_point - right_point,
                "ci_low": float(np.quantile(draws, q)), "ci_high": float(np.quantile(draws, 1-q)),
                "wins": sum(v > eps for v in cell_deltas.values()), "ties": sum(abs(v) <= eps for v in cell_deltas.values()),
                "losses": sum(v < -eps for v in cell_deltas.values()), "worst_unit_delta": min(cell_deltas.values()),
                "worst_unit_id": min(cell_deltas, key=cell_deltas.get),
                "family_wins": sum(v > eps for v in family_deltas.values()), "family_ties": sum(abs(v) <= eps for v in family_deltas.values()),
                "family_losses": sum(v < -eps for v in family_deltas.values()), "worst_family_delta": min(family_deltas.values()),
                "worst_family_id": min(family_deltas, key=family_deltas.get),
                "multiplicity_family_size": PRIMARY_COMPARISON_FAMILY if metric == "official_macro_f1" else 1,
                "inference": "Bonferroni simultaneous percentile interval across four opened atomic primary contrasts" if metric == "official_macro_f1" else "unadjusted paired diagnostic percentile interval",
            })
    return output


def evaluate_scores(release: Path, output: Path, registry: Mapping[str, Any], freeze: Mapping[str, Any]) -> dict[str, Any]:
    require_sources(registry)
    verified = c1.verified_scores(output, freeze)
    labels = p1._load_pb_labels(release)
    evaluation = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    arms = {arm: c1.evaluate_arm(arm, c1.rows_by_model(verified, labels, arm), evaluation) for arm in ARMS}
    contrasts = build_contrasts(arms[CANDIDATE], arms[REFERENCE])
    primary = {row["right_variant_id"]: row for row in contrasts if row["metric_id"] == "macro_f1"}
    by_metric = {(row["right_variant_id"], row["metric_id"]): row for row in contrasts}
    hard = (freeze["reference_local_alias_max_abs_error"] > 1e-12 or freeze["reference_combined_alias_max_abs_error"] > 1e-12
            or freeze["suffix_invariance_max_abs_error"] > 1e-12 or min(float(row["worst_unit_delta"]) for row in primary.values()) < c1.HARD_WORST_CELL_BOUND)
    promotion = all(float(row["delta"]) >= c1.BENEFIT and float(row["ci_low"]) > c1.BENEFIT
                    and int(row["wins"]) + int(row["ties"]) >= 6
                    and float(row["worst_unit_delta"]) >= c1.PROMOTION_WORST_CELL_BOUND
                    and float(by_metric[(comp, "first_error_exact")]["delta"]) >= c1.COMPONENT_BOUND
                    and float(by_metric[(comp, "clean_abstention_accuracy")]["delta"]) >= c1.COMPONENT_BOUND
                    for comp, row in primary.items()) and not hard
    gates = [
        {"gate_id":"P2A_SCORE_FREEZE_COMPLETE","status":"PASS","observed":len(verified),"required":"8 cells","detail":"reference and C2 scores froze before labels"},
        {"gate_id":"P2A_LABEL_FIREWALL","status":"PASS","observed":"labels opened after score freeze","required":"no fit-side labels or targets","detail":"adaptive transform and fusion are label-free"},
        {"gate_id":"P2A_TOP10_LOCAL_ALIAS","status":"PASS" if freeze["reference_local_alias_max_abs_error"] <= 1e-12 else "HARD_FAIL","observed":freeze["reference_local_alias_max_abs_error"],"required":"<=1e-12","detail":"atomic top-ten local alias"},
        {"gate_id":"P2A_TOP10_COMBINED_ALIAS","status":"PASS" if freeze["reference_combined_alias_max_abs_error"] <= 1e-12 else "HARD_FAIL","observed":freeze["reference_combined_alias_max_abs_error"],"required":"<=1e-12","detail":"atomic top-ten combined alias"},
        {"gate_id":"C2_SUFFIX_INVARIANCE","status":"PASS" if freeze["suffix_invariance_max_abs_error"] <= 1e-12 else "HARD_FAIL","observed":freeze["suffix_invariance_max_abs_error"],"required":"<=1e-12","detail":"deterministic adaptive prefix replay"},
        {"gate_id":"C2_WORST_CELL_HARD_BOUND","status":"HARD_FAIL" if hard else "PASS","observed":min(float(row["worst_unit_delta"]) for row in primary.values()),"required":f">={c1.HARD_WORST_CELL_BOUND}","detail":"minimum across both required comparators"},
        {"gate_id":"C2_PREMISE_PROMOTION","status":"PASS" if promotion else "FAIL","observed":str(promotion).lower(),"required":"all promotion gates versus top-ten and top-five","detail":"adaptive SWVar premise"},
    ]
    flips, flip_summary = c1.prediction_flips(arms[CANDIDATE]["decisions"], arms[REFERENCE]["decisions"])
    eval_root = output / "evaluation"; eval_root.mkdir(parents=True, exist_ok=False)
    c1.write_csv(eval_root / "PROCESSBENCH_DECISIONS.csv", [row for arm in ARMS for row in arms[arm]["decisions"]])
    c1.write_csv(eval_root / "PROCESSBENCH_BY_CELL.csv", [row for arm in ARMS for row in arms[arm]["by_cell"]])
    c1.write_csv(eval_root / "PROCESSBENCH_PANELS.csv", [row for arm in ARMS for row in arms[arm]["panels"]])
    atomic_write_npz(eval_root / "PROCESSBENCH_BOOTSTRAP_SAMPLES.npz", {f"{arm}__{metric}": values for arm in ARMS for metric, values in arms[arm]["samples"].items()})
    atomic_write_json(eval_root / "CALIBRATION_LEDGERS.json", {"schema":"reasoning-localization-phase2-atomic-c2-calibration-v1","arms":{arm:arms[arm]["ledgers"] for arm in ARMS}})
    c1.write_csv(eval_root / "PAIRWISE_CONTRASTS.csv", contrasts)
    c1.write_csv(eval_root / "STEP_LENGTH_STRATA.csv", p2r._length_strata(arms[CANDIDATE]["decisions"], arms[CANDIDATE]["by_cell"]))
    c1.write_csv(eval_root / "SELECTED_STEP_LENGTH.csv", p2r._selected_length_distribution(arms[CANDIDATE]["decisions"]))
    c1.write_csv(eval_root / "PREDICTION_FLIPS.csv", flips); c1.write_csv(eval_root / "PREDICTION_FLIP_SUMMARY.csv", flip_summary)
    c1.write_csv(eval_root / "GATES.csv", gates)
    candidate_panel = next(row for row in arms[CANDIDATE]["panels"] if row["metric_id"] == "official_macro_f1")
    reference_panel = next(row for row in arms[REFERENCE]["panels"] if row["metric_id"] == "official_macro_f1")
    status = "HARD_FAIL" if hard else "COMPLETE"
    summary = {"schema":"reasoning-localization-phase2-atomic-c2-evaluation-v1","variant_id":CANDIDATE,"status":status,
               "premise_gate_passed":promotion,"candidate_macro_f1":candidate_panel["value"],
               "candidate_macro_f1_ci":[candidate_panel["ci_low"],candidate_panel["ci_high"]],
               "atomic_reference_macro_f1":reference_panel["value"],"primary_contrasts":primary,
               "prediction_flips_vs_atomic_reference":sum(row["changed"] == "true" for row in flips),
               "reference_local_alias_max_abs_error":freeze["reference_local_alias_max_abs_error"],
               "reference_combined_alias_max_abs_error":freeze["reference_combined_alias_max_abs_error"],
               "suffix_invariance_max_abs_error":freeze["suffix_invariance_max_abs_error"],
               "bootstrap_draws":p1.BOOTSTRAP_DRAWS,"bootstrap_seed":p1.BOOTSTRAP_SEED,
               "peak_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    summary["payload_sha256"] = c1.payload_sha(summary); atomic_write_json(eval_root / "SUMMARY.json", summary)
    outputs = [p.name for p in sorted(eval_root.iterdir())]
    manifest = {"schema":"reasoning-localization-phase2-atomic-c2-evaluation-manifest-v1","variant_id":CANDIDATE,"status":status,
                "score_freeze_sha256":sha256_file(output / "score_freeze/SCORE_FREEZE_MANIFEST.json"),
                "execution_registry_sha256":sha256_file(Path(registry["registry_path"])),
                "outputs":[{"path":name,"sha256":sha256_file(eval_root/name),"bytes":(eval_root/name).stat().st_size} for name in outputs]}
    manifest["payload_sha256"] = c1.payload_sha(manifest); atomic_write_json(eval_root / "EVALUATION_MANIFEST.json", manifest)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--release",type=Path,default=p1.DEFAULT_RELEASE)
    parser.add_argument("--registry",type=Path,default=REGISTRY_PATH); parser.add_argument("--output",type=Path,default=OUTPUT_ROOT)
    args=parser.parse_args(); release=args.release.resolve(); registry_path=args.registry.resolve(); output=args.output.resolve()
    registry=load_registry(registry_path,release); registry["registry_path"]=str(registry_path); started=time.perf_counter()
    freeze=freeze_scores(release,output,registry); summary=evaluate_scores(release,output,registry,freeze)
    run={"schema":"reasoning-localization-phase2-atomic-c2-run-v1","variant_id":CANDIDATE,"status":summary["status"],
         "execution_registry_sha256":sha256_file(registry_path),"runner_sha256":sha256_file(Path(__file__).resolve()),
         "score_freeze_manifest_sha256":sha256_file(output/"score_freeze/SCORE_FREEZE_MANIFEST.json"),
         "evaluation_manifest_sha256":sha256_file(output/"evaluation/EVALUATION_MANIFEST.json"),
         "elapsed_seconds":time.perf_counter()-started,"summary":summary}
    run["payload_sha256"]=c1.payload_sha(run); atomic_write_json(output/"RUN_MANIFEST.json",run); print(json.dumps(run,indent=2))


if __name__ == "__main__": main()
