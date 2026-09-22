#!/usr/bin/env python3
"""Freeze and evaluate the single top-k family-local DUFS control."""

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

from spectral_utils.adapted_dufs import adapted_dufs_soft_gates  # noqa: E402
from spectral_utils.laplacian_upcr import build_graph_from_features  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_cell, validate_fit_manifest,
)
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization import run_phase3_compact_fusion as p3  # noqa: E402
from scripts.reasoning_localization import run_phase3_context_dufs_family as p3f  # noqa: E402
from scripts.reasoning_localization import run_phase3_deployed_upcr_prune_refit as p3d  # noqa: E402
from scripts.reasoning_localization import run_phase3_family_expert_attribution as p3e  # noqa: E402
from scripts.reasoning_localization.register_phase3_topk_dufs_control import EXPERIMENT, VARIANTS  # noqa: E402

K0, K1 = VARIANTS
H0 = "P3_H0_REFERENCE"
ROOT = p1.PROGRAM_ROOT / "phase_3/topk_dufs_control"
OUTPUT = ROOT / "p3k_topk_local_dufs_v1"
REGISTRY = ROOT / "P3K_EXECUTION_REGISTRY.json"
SOURCE_P3E = p3e.OUTPUT / "score_freeze/cells"
PRIMARY = (K1, K0)
BENEFIT = 0.003
HARM = -0.003
ALIAS_TOLERANCE = 1e-12


class TopkDUFSError(RuntimeError):
    pass


def _load_registry(release: Path) -> dict[str, Any]:
    row = json.loads(REGISTRY.read_text())
    required = {
        "schema": "reasoning-localization-p3k-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": EXPERIMENT,
        "variant_order": list(VARIANTS),
        "primary_contrast": list(PRIMARY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dufs_seeds": list(p3f.DUFS_SEEDS),
        "dufs_epochs": p3f.DUFS_EPOCHS,
        "graph_k": p3f.GRAPH_K,
        "liu_lambda": p3f.LIU_LAMBDA,
    }
    for key, value in required.items():
        if row.get(key) != value:
            raise TopkDUFSError(f"execution registry mismatch: {key}")
    if Path(row["release_root"]).resolve() != release.resolve():
        raise TopkDUFSError("release mismatch")
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
    aliases = {"p3e_parent": 0.0, "lambda_zero": 0.0}
    for position, cell_id in enumerate(p2r.PB_CELLS, start=1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        prep, raw, names, families = p3d._member_matrix(cell)
        if list(names) != registry["member_names"] or list(families) != registry["member_families"]:
            raise TopkDUFSError(f"member roster drift in {cell_id}")
        indices = {
            family: np.asarray([i for i, value in enumerate(families) if value == family], dtype=np.int64)
            for family in ("entropy_level", "entropy_dynamics", "partition_energy", "topk_distribution")
        }
        topk = indices["topk_distribution"]
        if len(topk) != 6:
            raise TopkDUFSError("top-k family width drift")
        owner = np.repeat(np.arange(len(cell.row_ids)), np.diff(np.asarray(cell.token_offsets)))
        token_scores = {variant: np.full(len(raw), np.nan) for variant in VARIANTS}
        fold_diagnostics = []
        for fold in range(5):
            held_rows = np.flatnonzero(np.asarray(prep.row_folds) == fold)
            held_indices = np.flatnonzero(np.isin(owner, held_rows))
            fit_folds = np.asarray(prep.row_folds)[np.asarray(prep.fit_row_indices)]
            donor_indices = np.asarray(prep.fit_indices)[fit_folds != fold]
            donor, held, scale = p3d._fold_standardize(raw, donor_indices, held_indices)
            donor_topk, held_topk = donor[:, topk], held[:, topk]
            parent_topk, parent_diag = p3e._fit_iu(donor_topk, held_topk)
            gates, gate_diag = adapted_dufs_soft_gates(
                donor_topk.T, seeds=p3f.DUFS_SEEDS, epochs=p3f.DUFS_EPOCHS
            )
            graph = build_graph_from_features(donor_topk.T, gates=gates, k=p3f.GRAPH_K)
            zero, candidate_topk, path_diag = p3f._fit_dufs_path(donor_topk, held_topk, graph)
            alias = float(np.max(np.abs(zero - parent_topk)))
            aliases["lambda_zero"] = max(aliases["lambda_zero"], alias)
            if alias > ALIAS_TOLERANCE:
                raise TopkDUFSError(f"lambda-zero alias failed in {cell_id} fold {fold}: {alias}")
            equal = {family: -held[:, idx].mean(axis=1) for family, idx in indices.items()}
            token_scores[K0][held_indices] = np.mean([
                equal["entropy_level"], equal["entropy_dynamics"], equal["partition_energy"], parent_topk
            ], axis=0)
            token_scores[K1][held_indices] = np.mean([
                equal["entropy_level"], equal["entropy_dynamics"], equal["partition_energy"], candidate_topk
            ], axis=0)
            fold_diagnostics.append({
                "fold": fold, "scale": scale, "parent": parent_diag,
                "alias_error": alias, "gates": gates.tolist(),
                "gate_diagnostics": p3f._jsonable(gate_diag), "path": path_diag,
            })
        if any(not np.isfinite(score).all() for score in token_scores.values()):
            raise TopkDUFSError(f"incomplete score in {cell_id}")
        source_arrays = load_npz_no_pickle(SOURCE_P3E / cell_id / "scores.npz")
        arrays = {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "h0_combined": source_arrays["h0_combined"],
        }
        for variant, score in token_scores.items():
            arrays[f"{variant.lower()}_local"] = p1.topk_step_mean(score, cell.segment_starts, cell.segment_ends, k=10)
        parent_error = float(np.max(np.abs(
            arrays[f"{K0.lower()}_local"] - source_arrays["p3e3_topk_iu_only_local"]
        )))
        aliases["p3e_parent"] = max(aliases["p3e_parent"], parent_error)
        if parent_error > ALIAS_TOLERANCE:
            raise TopkDUFSError(f"P3E3 parent alias failed in {cell_id}: {parent_error}")
        target = score_root / "cells" / cell_id
        target.mkdir(parents=True)
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        record = {
            "schema": "reasoning-localization-p3k-cell-v1", "experiment_id": EXPERIMENT,
            "cell_id": cell_id, "model_id": str(cell.model_id), "slice_id": str(cell.slice_id),
            "population_id": str(cell.population_id), "n_rows": len(cell.row_ids),
            "member_names": list(names), "member_families": list(families),
            "p3e_parent_alias_max_error": parent_error, "fold_diagnostics": fold_diagnostics,
            "labels_seen_during_fit": False, "targets_accessed_during_fit": False,
            "score_sha256": score_sha, "prepared_input_sha256": sha256_file(input_path),
        }
        record["payload_sha256"] = c1.payload_sha(record)
        atomic_write_json(target / "RECORD.json", record)
        records.append({"cell_id": cell_id, "record_path": f"cells/{cell_id}/RECORD.json", "record_sha256": sha256_file(target / "RECORD.json"), "score_sha256": score_sha})
        print(f"score-freeze P3K0-P3K1: {cell_id} ({position}/8)", flush=True)
    result = {
        "schema": "reasoning-localization-p3k-score-freeze-v1", "status": "COMPLETE",
        "experiment_id": EXPERIMENT, "variant_ids": list(VARIANTS), "records": records,
        "alias_max_errors": aliases, "labels_seen_during_fit": False,
        "execution_registry_sha256": sha256_file(REGISTRY), "runner_sha256": sha256_file(Path(__file__).resolve()),
    }
    result["payload_sha256"] = c1.payload_sha(result)
    atomic_write_json(score_root / "SCORE_FREEZE_MANIFEST.json", result)
    return result


def _verified(manifest: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for item in manifest["records"]:
        record_path = OUTPUT / "score_freeze" / item["record_path"]
        score_path = record_path.parent / "scores.npz"
        if sha256_file(record_path) != item["record_sha256"] or sha256_file(score_path) != item["score_sha256"]:
            raise TopkDUFSError("score-freeze hash mismatch")
        output[item["cell_id"]] = {"record": json.loads(record_path.read_text()), "arrays": load_npz_no_pickle(score_path)}
    return output


def _rows(verified: Mapping[str, Any], labels: Mapping[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    output = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
        record, arrays = verified[cell_id]["record"], verified[cell_id]["arrays"]
        for index, row_id in enumerate(arrays["row_ids"].astype(str)):
            lo, hi = map(int, arrays["segment_offsets"][index:index + 2])
            group_id, first_error = labels[cell_id][row_id]
            output[record["model_id"]].append({
                "row_id": row_id, "group_id": group_id, "slice_id": record["slice_id"],
                "cell_id": cell_id, "model_id": record["model_id"], "first_error": first_error,
                "step_scores": arrays[key][lo:hi].tolist(), "step_lengths": arrays["segment_lengths"][lo:hi].tolist(),
            })
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def evaluate(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = _verified(manifest)
    labels = p1._load_pb_labels(release)
    evaluator = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    h0 = c1.evaluate_arm(H0, _rows(verified, labels, "h0_combined"), evaluator)
    arms = {H0: h0}
    for variant in VARIANTS:
        arms[variant] = p3._rerank(variant, h0, _rows(verified, labels, f"{variant.lower()}_local"), evaluator)
    abstain = {(row["cell_id"], row["row_id"]): int(row["prediction_step"]) == -1 for row in h0["decisions"]}
    mismatches = {arm: sum((int(row["prediction_step"]) == -1) != abstain[(row["cell_id"], row["row_id"])] for row in arms[arm]["decisions"]) for arm in VARIANTS}
    if any(mismatches.values()):
        raise TopkDUFSError(f"H0 abstention alias failed: {mismatches}")
    contrasts = [p3e._contrast(K1, K0, metric, arms, False) for metric in p1.PB_METRICS]
    root = OUTPUT / "evaluation"; root.mkdir()
    panels = [row for arm in arms.values() for row in arm["panels"]]
    _write_csv(root / "PROCESSBENCH_PANELS.csv", panels)
    _write_csv(root / "PROCESSBENCH_BY_CELL.csv", [row for arm in arms.values() for row in arm["by_cell"]])
    _write_csv(root / "PAIRWISE_CONTRASTS.csv", contrasts)
    parent = {(row["cell_id"], row["row_id"]): row for row in arms[K0]["decisions"]}
    flips = [{
        "variant_id": K1, "cell_id": row["cell_id"], "row_id": row["row_id"],
        "parent_prediction_step": parent[(row["cell_id"], row["row_id"])]["prediction_step"],
        "candidate_prediction_step": row["prediction_step"], "first_error": row["true_first_error"],
    } for row in arms[K1]["decisions"] if int(row["prediction_step"]) != int(parent[(row["cell_id"], row["row_id"])]["prediction_step"])]
    if flips: _write_csv(root / "PREDICTION_FLIPS.csv", flips)
    primary = next(row for row in contrasts if row["metric_id"] == "macro_f1")
    summary = {
        "schema": "reasoning-localization-p3k-evaluation-v1", "status": "COMPLETE",
        "experiment_id": EXPERIMENT, "primary_contrast": primary,
        "alias_max_errors": manifest["alias_max_errors"], "abstention_mismatches": mismatches,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS, "bootstrap_seed": p1.BOOTSTRAP_SEED,
    }
    summary["payload_sha256"] = c1.payload_sha(summary)
    atomic_write_json(root / "SUMMARY.json", summary)
    values = {row["arm_id"]: float(row["value"]) for row in panels if row["metric_id"] == "official_macro_f1" and row["arm_id"] in VARIANTS}
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="980" height="300" viewBox="0 0 980 300">
<rect width="100%" height="100%" fill="white"/><style>text{{font-family:system-ui;fill:#172033}}.t{{font-size:22px;font-weight:700}}.b{{font-size:15px;font-weight:600}}.l{{font-size:13px}}</style>
<text x="25" y="38" class="t">Top-k family-local DUFS control</text><text x="25" y="67" class="l">Six top-k views; all other H2 components fixed</text>
<text x="25" y="115" class="b">IU parent</text><text x="330" y="115" class="b">{values[K0]:.6f}</text>
<text x="25" y="155" class="b">Local DUFS-LIU</text><text x="330" y="155" class="b">{values[K1]:.6f}</text>
<text x="25" y="215" class="b">Paired delta</text><text x="330" y="215" class="b">{primary['delta']:+.6f} [{primary['ci_low']:+.6f}, {primary['ci_high']:+.6f}]</text>
<text x="25" y="255" class="l">{primary['statistical_status']} · W/T/L {primary['wins']}/{primary['ties']}/{primary['losses']} · worst {primary['worst_unit_delta']:+.6f}</text></svg>\n'''
    (root / "P3K_RESULTS.svg").write_text(svg)
    return summary


def main() -> None:
    started = time.perf_counter(); release = p1.DEFAULT_RELEASE.resolve()
    registry = _load_registry(release); frozen = freeze(release, registry); summary = evaluate(release, frozen)
    atomic_write_json(OUTPUT / "RUN_COMPLETE.json", {"status": "COMPLETE", "elapsed_seconds": time.perf_counter() - started, "summary": summary})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
