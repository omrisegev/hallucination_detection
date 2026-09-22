#!/usr/bin/env python3
"""Freeze and evaluate the bounded ASTGI-Q1 point-query ProcessBench rung."""
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
from spectral_utils.token_local_fusion import fit_local_equal_family  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization import run_phase3_compact_fusion as p3  # noqa: E402
from scripts.reasoning_localization import run_phase3_deployed_upcr_prune_refit as p3d  # noqa: E402


EXPERIMENT = "P3_ASTGI_QUERY_HEADS"
VARIANT = "P3T_Q1_POINT_QUERY"
PARENT = "P3A_H2_EQUAL_OUTER_REFERENCE"
H0 = "P3_H0_REFERENCE"
PERMUTED = "P3T_Q1_QUERY_PERMUTED"
NO_BOUNDARY = "P3T_Q1_NO_BOUNDARY"
ROOT = p1.PROGRAM_ROOT / "phase_3/astgi_query_heads"
OUTPUT = ROOT / "p3t_q1_point_query_v1"
REGISTRY = ROOT / "P3T_Q1_EXECUTION_REGISTRY_AMENDMENT_V2.json"
SOURCE_H2 = p1.PROGRAM_ROOT / "phase_2/diagnostic/h3_reliability_fusion_v1/score_freeze/cells"
CELLS = tuple(p2r.PB_CELLS)
Q_ONSET = np.asarray((0.2, 0.4, 0.2, 0.2), dtype=np.float64)
TEMPERATURE = 1.0
BOUNDARY_GAMMA = 0.05
BENEFIT = 0.003
HARM = -0.005
FAMILY_SIZE = 3


class Q1Error(RuntimeError):
    pass


def _payload_sha(value: Any) -> str:
    return c1.payload_sha(value)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise Q1Error(f"refusing to write empty table: {path}")
    fields = list(dict.fromkeys(field for row in rows for field in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _load_registry(release: Path) -> dict[str, Any]:
    row = json.loads(REGISTRY.read_text(encoding="utf-8"))
    expected = {
        "schema": "reasoning-localization-p3tq1-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": EXPERIMENT,
        "variant_id": VARIANT,
        "parent_variant_id": PARENT,
        "processbench_cells": list(CELLS),
        "population_id": "current_common_eight_qwen",
        "metric": "official_macro_f1",
        "bootstrap_draws": 20000,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "multiplicity_family_size": FAMILY_SIZE,
        "benefit_delta": BENEFIT,
        "harm_delta": HARM,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise Q1Error(f"execution registry mismatch: {key}")
    if Path(row["release_root"]).resolve() != release.resolve():
        raise Q1Error("release mismatch")
    if row.get("runner_sha256") in (None, "", "PENDING_RUNNER_HASH"):
        raise Q1Error("runner hash was not frozen")
    if row["runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise Q1Error("runner hash mismatch")
    query = row.get("query", {})
    if query.get("q_onset") != Q_ONSET.tolist() or float(query.get("temperature")) != TEMPERATURE or float(query.get("boundary_gamma")) != BOUNDARY_GAMMA:
        raise Q1Error("query constants drifted")
    return row


def _relative_positions(cell: Any, n_tokens: int) -> np.ndarray:
    # The prepared contract may retain response-prefix/suffix tokens that are
    # intentionally outside scored step spans.  Give those masked tokens a
    # neutral position; the reducer never consumes them.
    position = np.ones(n_tokens, dtype=np.float64)
    for start, end in zip(cell.segment_starts, cell.segment_ends):
        lo, hi = int(start), int(end)
        if hi <= lo:
            raise Q1Error(f"{cell.cell_id}: malformed or overlapping step boundaries")
        position[lo:hi] = np.linspace(0.0, 1.0, hi - lo, dtype=np.float64)
    return position


def _query(risks: np.ndarray, position: np.ndarray, query: np.ndarray, gamma: float) -> np.ndarray:
    logits = np.asarray(risks, dtype=np.float64) / TEMPERATURE + query[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    attention = np.exp(logits)
    attention /= attention.sum(axis=1, keepdims=True)
    output = np.sum(attention * risks, axis=1) + float(gamma) * (1.0 - position)
    if not np.isfinite(output).all():
        raise Q1Error("query pooling produced non-finite scores")
    return output


def freeze(release: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {OUTPUT}")
    score_root = OUTPUT / "score_freeze"
    score_root.mkdir(parents=True)
    input_root = release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    records: list[dict[str, Any]] = []
    alias = 0.0
    for index, cell_id in enumerate(CELLS, start=1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        prep = p3.prepare_localization_cell(cell)
        risks, parent_token, diagnostics = p3._h2_family_matrix(cell)
        position = _relative_positions(cell, len(risks))
        candidate = _query(risks, position, Q_ONSET, BOUNDARY_GAMMA)
        permuted = _query(risks, position, Q_ONSET[::-1], BOUNDARY_GAMMA)
        no_boundary = _query(risks, position, Q_ONSET, 0.0)
        parent_local = p1.topk_step_mean(parent_token, cell.segment_starts, cell.segment_ends, k=10)
        frozen_h2 = load_npz_no_pickle(SOURCE_H2 / cell_id / "scores.npz")["h2_local"]
        alias = max(alias, float(np.max(np.abs(parent_local - frozen_h2))))
        arrays = {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "h0_combined": p1.combine_with_common_detector(cell, p1.topk_step_mean(np.asarray(fit_local_equal_family(prep).token_risk), cell.segment_starts, cell.segment_ends, k=10)),
            "parent_local": parent_local,
            "query_local": p1.topk_step_mean(candidate, cell.segment_starts, cell.segment_ends, k=10),
            "permuted_local": p1.topk_step_mean(permuted, cell.segment_starts, cell.segment_ends, k=10),
            "no_boundary_local": p1.topk_step_mean(no_boundary, cell.segment_starts, cell.segment_ends, k=10),
        }
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True)
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        record = {
            "schema": "reasoning-localization-p3tq1-cell-v1",
            "experiment_id": EXPERIMENT,
            "variant_id": VARIANT,
            "cell_id": cell_id,
            "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id),
            "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids),
            "n_tokens": len(risks),
            "score_sha256": score_sha,
            "prepared_input_sha256": sha256_file(input_path),
            "parent_alias_max_abs_error_cell": float(np.max(np.abs(parent_local - frozen_h2))),
            "query": {
                "family_order": diagnostics["family_names"],
                "q_onset": Q_ONSET.tolist(),
                "temperature": TEMPERATURE,
                "boundary_gamma": BOUNDARY_GAMMA,
                "formula": "a=softmax(z+q); r=a dot z+gamma*(1-u)",
            },
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
            "fit_contract": "H2 compact donor-fit family scores; query constants fixed before labels",
        }
        record["payload_sha256"] = _payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({
            "cell_id": cell_id,
            "record_path": f"cells/{cell_id}/RECORD.json",
            "record_sha256": sha256_file(target / "RECORD.json"),
            "score_sha256": score_sha,
        })
        print(f"score-freeze {VARIANT}: {cell_id} ({index}/{len(CELLS)})", flush=True)
    if alias > 1e-12:
        raise Q1Error(f"H2 parent alias failed: {alias}")
    result = {
        "schema": "reasoning-localization-p3tq1-score-freeze-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT,
        "variant_id": VARIANT,
        "records": records,
        "parent_alias_max_abs_error": alias,
        "execution_registry_sha256": sha256_file(REGISTRY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "labels_seen_during_fit": False,
        "detector_contract": "H0 threshold and abstention decision copied exactly; query arms rerank H0 non-abstentions only",
    }
    result["payload_sha256"] = _payload_sha(result)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", result)
    return result


def _verified(manifest: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for item in manifest["records"]:
        record_path = OUTPUT / "score_freeze" / item["record_path"]
        score_path = record_path.parent / "scores.npz"
        if sha256_file(record_path) != item["record_sha256"] or sha256_file(score_path) != item["score_sha256"]:
            raise Q1Error("score-freeze hash mismatch")
        output[item["cell_id"]] = {
            "record": json.loads(record_path.read_text(encoding="utf-8")),
            "arrays": load_npz_no_pickle(score_path),
        }
    if tuple(output) != CELLS:
        raise Q1Error("score-freeze cell order drifted")
    return output


def _rows(verified: Mapping[str, Any], labels: Mapping[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    output = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in CELLS:
        record, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        for index, row_id in enumerate(arrays["row_ids"].astype(str)):
            lo, hi = map(int, arrays["segment_offsets"][index:index + 2])
            group_id, first_error = labels[cell_id][row_id]
            output[record["model_id"]].append({
                "row_id": row_id,
                "group_id": group_id,
                "slice_id": record["slice_id"],
                "cell_id": cell_id,
                "model_id": record["model_id"],
                "first_error": first_error,
                "step_scores": arrays[key][lo:hi].tolist(),
                "step_lengths": arrays["segment_lengths"][lo:hi].tolist(),
            })
    return output


def _rerank(arm: str, h0: Mapping[str, Any], rows: Mapping[str, list[dict[str, Any]]], evaluator: Any) -> dict[str, Any]:
    score_rows = {(row["cell_id"], row["row_id"]): row for model_rows in rows.values() for row in model_rows}
    decisions = []
    for parent in h0["decisions"]:
        source = score_rows[(parent["cell_id"], parent["row_id"])]
        prediction = -1 if int(parent["prediction_step"]) == -1 else int(np.argmax(source["step_scores"]))
        decisions.append({**parent, "arm_id": arm, "prediction_step": prediction})
    by_cell = []
    for model in p1.QWEN_MODELS:
        for family in p1.FAMILIES:
            selected = [row for row in decisions if row["model_id"] == model and row["slice_id"] == family]
            metrics = evaluator.processbench_trace_metrics(
                [row["true_first_error"] for row in selected],
                [row["prediction_step"] for row in selected],
            )
            by_cell.append({
                "arm_id": arm,
                "model_id": model,
                "slice_id": family,
                "cell_id": f"processbench_{family}_{model}",
                **{metric: metrics[metric] for metric in p1.PB_METRICS},
                "n_examples": metrics["n_examples"],
                "n_error": metrics["n_error"],
                "n_clean": metrics["n_clean"],
            })
    samples = p1._bootstrap_pb_panel(decisions, p1.QWEN_MODELS)
    panels = []
    for metric in p1.PB_METRICS:
        values = np.asarray(samples[metric], dtype=np.float64)
        panels.append({
            "arm_id": arm,
            "population_id": "current_common_eight_qwen",
            "metric_id": metric,
            "value": float(np.mean([float(row[metric]) for row in by_cell])),
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "n_rows": sum(int(row["n_examples"]) for row in by_cell),
            "n_groups": 3400,
        })
    return {"decisions": decisions, "by_cell": by_cell, "samples": samples, "panels": panels}


def _status(delta: float, lo: float, hi: float) -> str:
    if lo > BENEFIT:
        return "SUPPORTED_IMPROVEMENT"
    if hi < HARM:
        return "SUPPORTED_HARM"
    if delta > 0 and lo <= 0:
        return "PROMISING_UNCONFIRMED"
    return "INCONCLUSIVE"


def _contrasts(arms: Mapping[str, Any]) -> list[dict[str, Any]]:
    pairs = ((VARIANT, PARENT), (VARIANT, PERMUTED), (VARIANT, NO_BOUNDARY))
    output = []
    for left, right in pairs:
        lp = {row["metric_id"]: row for row in arms[left]["panels"]}
        rp = {row["metric_id"]: row for row in arms[right]["panels"]}
        right_cells = {row["cell_id"]: row for row in arms[right]["by_cell"]}
        for metric in p1.PB_METRICS:
            draws = np.asarray(arms[left]["samples"][metric]) - np.asarray(arms[right]["samples"][metric])
            q = 0.025 / FAMILY_SIZE if metric == "official_macro_f1" else 0.025
            cell_deltas = {
                row["cell_id"]: float(row[metric]) - float(right_cells[row["cell_id"]][metric])
                for row in arms[left]["by_cell"]
            }
            delta = float(lp[metric]["value"] - rp[metric]["value"])
            lo, hi = float(np.quantile(draws, q)), float(np.quantile(draws, 1.0 - q))
            output.append({
                "contrast_id": f"pb::{left}::{right}::{metric}",
                "left_variant_id": left,
                "right_variant_id": right,
                "metric_id": "macro_f1" if metric == "official_macro_f1" else metric,
                "source_metric_id": metric,
                "delta": delta,
                "ci_low": lo,
                "ci_high": hi,
                "statistical_status": _status(delta, lo, hi),
                "wins": sum(value > 1e-12 for value in cell_deltas.values()),
                "ties": sum(abs(value) <= 1e-12 for value in cell_deltas.values()),
                "losses": sum(value < -1e-12 for value in cell_deltas.values()),
                "worst_unit_delta": min(cell_deltas.values()),
                "worst_unit_id": min(cell_deltas, key=cell_deltas.get),
                "multiplicity_family_size": FAMILY_SIZE if metric == "official_macro_f1" else 1,
                "inference": "Bonferroni simultaneous paired whole-question bootstrap" if metric == "official_macro_f1" else "paired whole-question bootstrap",
            })
    return output


def _plot(path: Path, panels: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> None:
    arms = [PARENT, VARIANT, PERMUTED, NO_BOUNDARY]
    values = {row["arm_id"]: float(row["value"]) for row in panels if row["metric_id"] == "official_macro_f1"}
    rows = [row for row in contrasts if row["metric_id"] == "macro_f1"]
    lo = min([*values.values(), 0.0]) - 0.006
    hi = max([*values.values(), 0.0]) + 0.006
    scale = lambda x: 325 + (x - lo) / (hi - lo) * 650
    pieces = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1040" height="560" viewBox="0 0 1040 560">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui;fill:#172033}.t{font-size:22px;font-weight:700}.s{font-size:13px}.b{font-size:13px;font-weight:600}</style>',
        '<text x="25" y="34" class="t">ASTGI-Q1 fixed point-query pooling</text>',
        '<text x="25" y="57" class="s">ProcessBench macro F1; same H2 parent, top-ten reducer and H0 detector</text>',
    ]
    for i, arm in enumerate(arms):
        y = 95 + 37 * i
        pieces += [f'<text x="25" y="{y+5}" class="b">{arm}</text>', f'<line x1="325" y1="{y}" x2="{scale(values[arm]):.1f}" y2="{y}" stroke="#2563eb" stroke-width="7"/>', f'<circle cx="{scale(values[arm]):.1f}" cy="{y}" r="6" fill="#7c3aed"/>', f'<text x="{scale(values[arm])+10:.1f}" y="{y+5}" class="b">{values[arm]:.6f}</text>']
    pieces += ['<text x="25" y="285" class="t">Query minus comparator (Bonferroni 95% CI)</text>']
    for i, row in enumerate(rows):
        y = 320 + 52 * i
        pieces += [f'<text x="25" y="{y}" class="s">{row["left_variant_id"]} − {row["right_variant_id"]}</text>', f'<line x1="{scale(row["ci_low"]):.1f}" y1="{y-5}" x2="{scale(row["ci_high"]):.1f}" y2="{y-5}" stroke="#dc2626" stroke-width="3"/><circle cx="{scale(row["delta"]):.1f}" cy="{y-5}" r="5" fill="#111827"/><text x="25" y="{y+17}" class="s">Δ {row["delta"]:+.6f} [{row["ci_low"]:+.6f}, {row["ci_high"]:+.6f}] — {row["statistical_status"]}</text>']
    pieces.append('</svg>')
    path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def evaluate(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = _verified(manifest)
    labels = p1._load_pb_labels(release)
    evaluator = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    arms = {
        H0: c1.evaluate_arm(H0, _rows(verified, labels, "h0_combined"), evaluator),
    }
    for arm, key in ((PARENT, "parent_local"), (VARIANT, "query_local"), (PERMUTED, "permuted_local"), (NO_BOUNDARY, "no_boundary_local")):
        arms[arm] = _rerank(arm, arms[H0], _rows(verified, labels, key), evaluator)
    h0_abstain = {(row["cell_id"], row["row_id"]): int(row["prediction_step"]) == -1 for row in arms[H0]["decisions"]}
    mismatches = {arm: sum((int(row["prediction_step"]) == -1) != h0_abstain[(row["cell_id"], row["row_id"])] for row in arms[arm]["decisions"]) for arm in (PARENT, VARIANT, PERMUTED, NO_BOUNDARY)}
    if any(mismatches.values()):
        raise Q1Error(f"H0 abstention alias failed: {mismatches}")
    contrasts = _contrasts(arms)
    evaluation_root = OUTPUT / "evaluation"
    evaluation_root.mkdir()
    _write_csv(evaluation_root / "PROCESSBENCH_BY_CELL.csv", [row for arm in arms.values() for row in arm["by_cell"]])
    _write_csv(evaluation_root / "PROCESSBENCH_PANELS.csv", [row for arm in arms.values() for row in arm["panels"]])
    _write_csv(evaluation_root / "PAIRWISE_CONTRASTS.csv", contrasts)
    parent_flips, flip_summary = c1.prediction_flips(arms[VARIANT]["decisions"], arms[PARENT]["decisions"])
    _write_csv(evaluation_root / "PREDICTION_FLIPS.csv", parent_flips)
    _write_csv(evaluation_root / "PREDICTION_FLIP_SUMMARY.csv", flip_summary)
    _plot(evaluation_root / "P3T_Q1_RESULTS.svg", [row for arm in arms.values() for row in arm["panels"]], contrasts)
    primary = next(row for row in contrasts if row["left_variant_id"] == VARIANT and row["right_variant_id"] == PARENT and row["metric_id"] == "macro_f1")
    summary = {
        "schema": "reasoning-localization-p3tq1-evaluation-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT,
        "variant_id": VARIANT,
        "primary_contrast": primary,
        "contrasts": contrasts,
        "abstention_mismatches": mismatches,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
        "control_ids": [PARENT, PERMUTED, NO_BOUNDARY],
        "plot": "evaluation/P3T_Q1_RESULTS.svg",
    }
    summary["payload_sha256"] = _payload_sha(summary)
    atomic_write_json(evaluation_root / "SUMMARY.json", summary)
    return summary


def main() -> None:
    started = time.perf_counter()
    release = p1.DEFAULT_RELEASE.resolve()
    registry = _load_registry(release)
    frozen = freeze(release, registry)
    summary = evaluate(release, frozen)
    atomic_write_json(OUTPUT / "RUN_COMPLETE.json", {
        "schema": "reasoning-localization-p3tq1-run-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT,
        "variant_id": VARIANT,
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
