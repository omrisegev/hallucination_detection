#!/usr/bin/env python3
"""Freeze and evaluate dynamics-local and all-H2-context DUFS family experts."""

from __future__ import annotations

import csv
import hashlib
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
from spectral_utils.laplacian_upcr import (  # noqa: E402
    build_graph_from_features,
    laplacian_iu_path,
)
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
from spectral_utils.token_local_fusion import IU_CONFIG  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402
from scripts.reasoning_localization import run_phase3_compact_fusion as p3  # noqa: E402
from scripts.reasoning_localization import run_phase3_deployed_upcr_prune_refit as p3d  # noqa: E402
from scripts.reasoning_localization import run_phase3_family_expert_attribution as p3e  # noqa: E402
from scripts.reasoning_localization.register_phase3_context_dufs_family import (  # noqa: E402
    EXPERIMENT,
    VARIANTS,
)

F0, F1, F2, F3 = VARIANTS
H0 = "P3_H0_REFERENCE"
ROOT = p1.PROGRAM_ROOT / "phase_3/context_dufs_family"
OUTPUT = ROOT / "p3f_context_dufs_family_v1"
REGISTRY = ROOT / "P3F_EXECUTION_REGISTRY.json"
SOURCE_P3E = p3e.OUTPUT / "score_freeze/cells"
PRIMARY = ((F1, F0), (F2, F0), (F2, F1), (F2, F3))
FAMILY_SIZE = len(PRIMARY)
BENEFIT = 0.003
HARM = -0.003
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
GRAPH_K = 7
LIU_LAMBDA = 0.1
PERMUTATION_SEED = 2026083101
ALIAS_TOLERANCE = 1e-12


class ContextDUFSError(RuntimeError):
    pass


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _orient(weights: np.ndarray, donor: np.ndarray) -> tuple[np.ndarray, float, bool]:
    output = np.asarray(weights, dtype=float).copy()
    anchor = donor.mean(axis=1)
    score = donor @ output
    corr = float(np.corrcoef(score, anchor)[0, 1])
    flipped = bool(np.isfinite(corr) and corr < 0.0)
    if flipped:
        output *= -1.0
    return output, corr, flipped


def _fit_dufs_path(
    donor_dyn: np.ndarray,
    held_dyn: np.ndarray,
    graph: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = laplacian_iu_path(
        donor_dyn.T,
        (0.0, LIU_LAMBDA),
        graph=graph,
        baseline_kwargs=dict(IU_CONFIG),
    )
    scores: dict[float, np.ndarray] = {}
    orientation: dict[str, Any] = {}
    for lambda_, result in path.items():
        weights, corr, flipped = _orient(result.w, donor_dyn)
        scores[lambda_] = -(held_dyn @ weights)
        orientation[str(lambda_)] = {
            "weights": weights.tolist(),
            "anchor_correlation": corr,
            "orientation_flipped": flipped,
        }
    diagnostics = {
        "lambda_zero": _jsonable(path[0.0].diagnostics),
        "lambda_headline": _jsonable(path[LIU_LAMBDA].diagnostics),
        "orientation": orientation,
    }
    return scores[0.0], scores[LIU_LAMBDA], diagnostics


def _offset(cell_id: str, fold: int, row: int, length: int) -> int:
    if length < 2:
        return 0
    payload = f"{PERMUTATION_SEED}|{cell_id}|{fold}|{row}".encode("utf-8")
    return 1 + int(hashlib.sha256(payload).hexdigest()[:16], 16) % (length - 1)


def _shift_outside_context(
    donor: np.ndarray,
    donor_indices: np.ndarray,
    owner: np.ndarray,
    dynamics_indices: np.ndarray,
    cell_id: str,
    fold: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    shifted = donor.copy()
    outside = np.ones(donor.shape[1], dtype=bool)
    outside[dynamics_indices] = False
    donor_owners = owner[donor_indices]
    changed = 0
    singleton = 0
    offsets = []
    for row in np.unique(donor_owners):
        positions = np.flatnonzero(donor_owners == row)
        amount = _offset(cell_id, fold, int(row), len(positions))
        if amount == 0:
            singleton += 1
            continue
        shifted[np.ix_(positions, np.flatnonzero(outside))] = np.roll(
            donor[np.ix_(positions, np.flatnonzero(outside))], amount, axis=0
        )
        offsets.append(amount)
        changed += len(positions)
    return shifted, {
        "n_donor_tokens": int(len(donor)),
        "n_shifted_tokens": int(changed),
        "n_singleton_responses": int(singleton),
        "min_offset": int(min(offsets)) if offsets else 0,
        "max_offset": int(max(offsets)) if offsets else 0,
        "seed": PERMUTATION_SEED,
    }


def _load_registry(release: Path) -> dict[str, Any]:
    row = json.loads(REGISTRY.read_text(encoding="utf-8"))
    required = {
        "schema": "reasoning-localization-p3f-execution-v1",
        "status": "FROZEN_BEFORE_RUN",
        "experiment_id": EXPERIMENT,
        "variant_order": list(VARIANTS),
        "primary_contrasts": [list(pair) for pair in PRIMARY],
        "multiplicity_family_size": FAMILY_SIZE,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "dufs_seeds": list(DUFS_SEEDS),
        "dufs_epochs": DUFS_EPOCHS,
        "graph_k": GRAPH_K,
        "liu_lambda": LIU_LAMBDA,
        "permutation_seed": PERMUTATION_SEED,
    }
    for key, value in required.items():
        if row.get(key) != value:
            raise ContextDUFSError(f"execution registry mismatch: {key}")
    if Path(row["release_root"]).resolve() != release.resolve():
        raise ContextDUFSError("release mismatch")
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
    overall_alias = {"p3e_parent": 0.0, "local_lambda_zero": 0.0, "context_lambda_zero": 0.0, "permuted_lambda_zero": 0.0}

    for position, cell_id in enumerate(p2r.PB_CELLS, start=1):
        source = by_cell[cell_id]
        input_path = input_root / source["artifact_path"]
        cell = load_prepared_localization_cell(input_path, source)
        prep, raw, names, families = p3d._member_matrix(cell)
        if list(names) != registry["member_names"] or list(families) != registry["member_families"]:
            raise ContextDUFSError(f"member roster drift in {cell_id}")
        indices = {
            family: np.asarray([i for i, value in enumerate(families) if value == family], dtype=np.int64)
            for family in ("entropy_level", "entropy_dynamics", "partition_energy", "topk_distribution")
        }
        dynamics = indices["entropy_dynamics"]
        owner = np.repeat(np.arange(len(cell.row_ids)), np.diff(np.asarray(cell.token_offsets)))
        token_scores = {variant: np.full(len(raw), np.nan) for variant in VARIANTS}
        fold_diagnostics = []
        for fold in range(5):
            held_rows = np.flatnonzero(np.asarray(prep.row_folds) == fold)
            held_indices = np.flatnonzero(np.isin(owner, held_rows))
            fit_folds = np.asarray(prep.row_folds)[np.asarray(prep.fit_row_indices)]
            donor_indices = np.asarray(prep.fit_indices)[fit_folds != fold]
            donor, held, scale = p3d._fold_standardize(raw, donor_indices, held_indices)
            donor_dyn, held_dyn = donor[:, dynamics], held[:, dynamics]
            parent_dyn, parent_diag = p3e._fit_iu(donor_dyn, held_dyn)

            local_gates, local_gate_diag = adapted_dufs_soft_gates(
                donor_dyn.T, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )
            local_graph = build_graph_from_features(donor_dyn.T, gates=local_gates, k=GRAPH_K)
            local_zero, local_dyn, local_path_diag = _fit_dufs_path(donor_dyn, held_dyn, local_graph)

            context_gates, context_gate_diag = adapted_dufs_soft_gates(
                donor.T, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )
            context_graph = build_graph_from_features(donor.T, gates=context_gates, k=GRAPH_K)
            context_zero, context_dyn, context_path_diag = _fit_dufs_path(donor_dyn, held_dyn, context_graph)

            shifted, shift_diag = _shift_outside_context(
                donor, donor_indices, owner, dynamics, cell_id, fold
            )
            perm_gates, perm_gate_diag = adapted_dufs_soft_gates(
                shifted.T, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
            )
            perm_graph = build_graph_from_features(shifted.T, gates=perm_gates, k=GRAPH_K)
            perm_zero, perm_dyn, perm_path_diag = _fit_dufs_path(donor_dyn, held_dyn, perm_graph)

            alias_errors = {
                "local_lambda_zero": float(np.max(np.abs(local_zero - parent_dyn))),
                "context_lambda_zero": float(np.max(np.abs(context_zero - parent_dyn))),
                "permuted_lambda_zero": float(np.max(np.abs(perm_zero - parent_dyn))),
            }
            if max(alias_errors.values()) > ALIAS_TOLERANCE:
                raise ContextDUFSError(f"lambda-zero alias failed in {cell_id} fold {fold}: {alias_errors}")
            for key, value in alias_errors.items():
                overall_alias[key] = max(overall_alias[key], value)

            equal = {family: -held[:, idx].mean(axis=1) for family, idx in indices.items()}
            shared = [equal["entropy_level"], equal["partition_energy"], equal["topk_distribution"]]
            token_scores[F0][held_indices] = np.mean([shared[0], parent_dyn, shared[1], shared[2]], axis=0)
            token_scores[F1][held_indices] = np.mean([shared[0], local_dyn, shared[1], shared[2]], axis=0)
            token_scores[F2][held_indices] = np.mean([shared[0], context_dyn, shared[1], shared[2]], axis=0)
            token_scores[F3][held_indices] = np.mean([shared[0], perm_dyn, shared[1], shared[2]], axis=0)
            fold_diagnostics.append({
                "fold": fold,
                "scale": scale,
                "parent": parent_diag,
                "alias_errors": alias_errors,
                "local": {"gates": local_gates.tolist(), "gate_diagnostics": _jsonable(local_gate_diag), "path": local_path_diag},
                "context": {"gates": context_gates.tolist(), "gate_diagnostics": _jsonable(context_gate_diag), "path": context_path_diag},
                "permuted_context": {"gates": perm_gates.tolist(), "gate_diagnostics": _jsonable(perm_gate_diag), "path": perm_path_diag, "shift": shift_diag},
            })
        if any(not np.isfinite(score).all() for score in token_scores.values()):
            raise ContextDUFSError(f"incomplete cross-fit score in {cell_id}")

        source_arrays = load_npz_no_pickle(SOURCE_P3E / cell_id / "scores.npz")
        arrays = {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "segment_lengths": np.asarray(cell.segment_ends - cell.segment_starts, dtype="<i8"),
            "h0_combined": source_arrays["h0_combined"],
        }
        for variant, score in token_scores.items():
            arrays[f"{variant.lower()}_local"] = p1.topk_step_mean(
                score, cell.segment_starts, cell.segment_ends, k=10
            )
        parent_error = float(np.max(np.abs(
            arrays[f"{F0.lower()}_local"] - source_arrays["p3e1_dynamics_iu_only_local"]
        )))
        overall_alias["p3e_parent"] = max(overall_alias["p3e_parent"], parent_error)
        if parent_error > ALIAS_TOLERANCE:
            raise ContextDUFSError(f"P3E parent alias failed in {cell_id}: {parent_error}")

        target = score_root / "cells" / cell_id
        target.mkdir(parents=True)
        score_sha = atomic_write_npz(target / "scores.npz", arrays)
        record = {
            "schema": "reasoning-localization-p3f-cell-v1",
            "experiment_id": EXPERIMENT,
            "cell_id": cell_id,
            "model_id": str(cell.model_id),
            "slice_id": str(cell.slice_id),
            "population_id": str(cell.population_id),
            "n_rows": len(cell.row_ids),
            "member_names": list(names),
            "member_families": list(families),
            "family_counts": {key: len(value) for key, value in indices.items()},
            "p3e_parent_alias_max_error": parent_error,
            "fold_diagnostics": fold_diagnostics,
            "labels_seen_during_fit": False,
            "targets_accessed_during_fit": False,
            "score_sha256": score_sha,
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
        print(f"score-freeze P3F0-P3F3: {cell_id} ({position}/8)", flush=True)

    result = {
        "schema": "reasoning-localization-p3f-score-freeze-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT,
        "variant_ids": list(VARIANTS),
        "records": records,
        "alias_max_errors": overall_alias,
        "labels_seen_during_fit": False,
        "execution_registry_sha256": sha256_file(REGISTRY),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
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
            raise ContextDUFSError("score-freeze hash mismatch")
        output[item["cell_id"]] = {
            "record": json.loads(record_path.read_text()),
            "arrays": load_npz_no_pickle(score_path),
        }
    return output


def _rows(verified: Mapping[str, Any], labels: Mapping[str, Any], key: str) -> dict[str, list[dict[str, Any]]]:
    output = {model: [] for model in p1.QWEN_MODELS}
    for cell_id in p2r.PB_CELLS:
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _prediction_flips(arms: Mapping[str, Any]) -> list[dict[str, Any]]:
    parent = {(row["cell_id"], row["row_id"]): row for row in arms[F0]["decisions"]}
    rows = []
    for variant in (F1, F2, F3):
        for row in arms[variant]["decisions"]:
            key = (row["cell_id"], row["row_id"])
            base = parent[key]
            if int(row["prediction_step"]) != int(base["prediction_step"]):
                rows.append({
                    "variant_id": variant,
                    "cell_id": row["cell_id"],
                    "row_id": row["row_id"],
                    "parent_prediction_step": base["prediction_step"],
                    "candidate_prediction_step": row["prediction_step"],
                    "first_error": row["first_error"],
                })
    return rows


def _plot(path: Path, panels: list[dict[str, Any]], contrasts: list[dict[str, Any]]) -> None:
    values = {row["arm_id"]: float(row["value"]) for row in panels if row["metric_id"] == "official_macro_f1" and row["arm_id"] in VARIANTS}
    low, high = min(values.values()) - .003, max(values.values()) + .003
    x = lambda value: 320 + (value - low) / (high - low) * 650
    colors = {F0: "#64748b", F1: "#2563eb", F2: "#7c3aed", F3: "#d97706"}
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="535" viewBox="0 0 1080 535">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:system-ui;fill:#172033}.t{font-size:22px;font-weight:700}.l{font-size:13px}.b{font-size:13px;font-weight:600}</style>',
        '<text x="25" y="34" class="t">Dynamics family expert: local versus contextual DUFS</text>',
        '<text x="25" y="57" class="l">All H2 views may define geometry; only dynamics receives LIU weights</text>',
    ]
    for i, variant in enumerate(VARIANTS):
        y = 100 + 48 * i
        parts += [
            f'<text x="25" y="{y+5}" class="b">{variant}</text>',
            f'<line x1="320" y1="{y}" x2="{x(values[variant]):.1f}" y2="{y}" stroke="{colors[variant]}" stroke-width="8"/>',
            f'<circle cx="{x(values[variant]):.1f}" cy="{y}" r="6" fill="{colors[variant]}"/>',
            f'<text x="{x(values[variant])+10:.1f}" y="{y+5}" class="b">{values[variant]:.6f}</text>',
        ]
    parts += ['<text x="25" y="325" class="t">Frozen primary paired contrasts</text>']
    primary = [row for row in contrasts if (row["left_variant_id"], row["right_variant_id"]) in PRIMARY and row["metric_id"] == "macro_f1"]
    for i, row in enumerate(primary):
        y = 365 + 34 * i
        parts += [
            f'<text x="25" y="{y}" class="l">{row["left_variant_id"]} − {row["right_variant_id"]}</text>',
            f'<text x="590" y="{y}" class="b">{row["delta"]:+.5f} [{row["ci_low"]:+.5f}, {row["ci_high"]:+.5f}]</text>',
            f'<text x="880" y="{y}" class="l">{row["statistical_status"]}</text>',
        ]
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n")


def evaluate(release: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    verified = _verified(manifest)
    labels = p1._load_pb_labels(release)
    evaluator = importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    h0 = c1.evaluate_arm(H0, _rows(verified, labels, "h0_combined"), evaluator)
    arms = {H0: h0}
    for variant in VARIANTS:
        arms[variant] = p3._rerank(
            variant, h0, _rows(verified, labels, f"{variant.lower()}_local"), evaluator
        )
    abstain = {(row["cell_id"], row["row_id"]): int(row["prediction_step"]) == -1 for row in h0["decisions"]}
    mismatches = {
        arm: sum((int(row["prediction_step"]) == -1) != abstain[(row["cell_id"], row["row_id"])] for row in arms[arm]["decisions"])
        for arm in VARIANTS
    }
    if any(mismatches.values()):
        raise ContextDUFSError(f"H0 abstention alias failed: {mismatches}")

    pairs = [*PRIMARY, (F3, F0)]
    contrasts = [
        p3e._contrast(left, right, metric, arms, (left, right) in PRIMARY)
        for left, right in pairs for metric in p1.PB_METRICS
    ]
    evaluation_root = OUTPUT / "evaluation"
    evaluation_root.mkdir()
    panels = [row for arm in arms.values() for row in arm["panels"]]
    _write_csv(evaluation_root / "PROCESSBENCH_PANELS.csv", panels)
    _write_csv(evaluation_root / "PROCESSBENCH_BY_CELL.csv", [row for arm in arms.values() for row in arm["by_cell"]])
    _write_csv(evaluation_root / "PAIRWISE_CONTRASTS.csv", contrasts)
    flips = _prediction_flips(arms)
    if flips:
        _write_csv(evaluation_root / "PREDICTION_FLIPS.csv", flips)

    primary = [row for row in contrasts if row["metric_id"] == "macro_f1" and (row["left_variant_id"], row["right_variant_id"]) in PRIMARY]
    primary_map = {(row["left_variant_id"], row["right_variant_id"]): row for row in primary}
    hard_valid = max(manifest["alias_max_errors"].values()) <= ALIAS_TOLERANCE and not any(mismatches.values())
    topk_eligible = hard_valid and any(
        primary_map[(variant, F0)]["delta"] > 0
        and primary_map[(variant, F0)]["ci_high"] >= HARM
        and primary_map[(variant, F0)]["worst_unit_delta"] >= -.020
        for variant in (F1, F2)
    )
    context_supported = (
        primary_map[(F2, F1)]["ci_low"] > 0
        and primary_map[(F2, F3)]["ci_low"] > 0
    )
    summary = {
        "schema": "reasoning-localization-p3f-evaluation-v1",
        "status": "COMPLETE",
        "experiment_id": EXPERIMENT,
        "primary_contrasts": primary,
        "alias_max_errors": manifest["alias_max_errors"],
        "abstention_mismatches": mismatches,
        "context_mechanism_supported": context_supported,
        "topk_secondary_control_eligible": topk_eligible,
        "bootstrap_draws": p1.BOOTSTRAP_DRAWS,
        "bootstrap_seed": p1.BOOTSTRAP_SEED,
    }
    summary["payload_sha256"] = c1.payload_sha(summary)
    atomic_write_json(evaluation_root / "SUMMARY.json", summary)
    _plot(evaluation_root / "P3F_RESULTS.svg", panels, contrasts)
    return summary


def main() -> None:
    started = time.perf_counter()
    release = p1.DEFAULT_RELEASE.resolve()
    registry = _load_registry(release)
    frozen = freeze(release, registry)
    summary = evaluate(release, frozen)
    atomic_write_json(OUTPUT / "RUN_COMPLETE.json", {
        "status": "COMPLETE",
        "elapsed_seconds": time.perf_counter() - started,
        "summary": summary,
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
