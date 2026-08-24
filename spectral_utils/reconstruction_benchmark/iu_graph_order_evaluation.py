"""Strict post-freeze evaluation for the IU graph-order ablation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .evaluation import (
    _interval,
    _weighted_binary_metric_draws,
    grouped_bootstrap_multiplicity_chunks,
)
from .io import canonical_json_bytes, load_npz_no_pickle, sha256_bytes, sha256_file
from .iu_graph_order_ablation import expected_arm_ids, validate_config


AB_SCHEMA_VERSION = "iu-graph-order-score-ab-verification-v1"
EVALUATION_SCHEMA_VERSION = "iu-graph-order-evaluation-v1"
REFERENCE_METHODS = ("deem_b3", "family_nrm_a", "pgrd_a")
CONTRAST_REFERENCES = ("iu_pcr", "equal_family_mean", "signed_deem_b3")


@dataclass(frozen=True)
class VerifiedAblation:
    output_release: Path
    source_release: Path
    cell_ids: tuple[str, ...]
    arm_ids: tuple[str, ...]
    row_ids_by_cell: Mapping[str, tuple[str, ...]]
    score_by_cell: Mapping[str, Mapping[str, np.ndarray]]
    ab_record: Mapping[str, object]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _require_payload(value: Mapping[str, object], field: str = "payload_sha256") -> None:
    payload = dict(value)
    expected = payload.pop(field, None)
    observed = sha256_bytes(canonical_json_bytes(payload))
    _require(expected == observed, f"invalid {field}")


def _load_fit_build(path: Path, config: Mapping[str, object]) -> tuple[dict, dict[str, dict]]:
    manifest_path = path / "SCORE_FREEZE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_payload(manifest)
    _require(manifest.get("schema_version") == "iu-graph-order-score-freeze-v1", "bad fit schema")
    _require(manifest.get("runtime_labels_used") is False, "fit manifest claims target access")
    lambdas = validate_config(config)
    arm_ids = expected_arm_ids(lambdas)
    _require(tuple(manifest.get("arm_ids", ())) == arm_ids, "fit arm roster drifted")
    _require(int(manifest.get("n_cells", -1)) == 24, "fit does not contain 24 cells")
    records = list(manifest.get("cells", ()))
    _require(len(records) == 24, "fit cell record count drifted")
    output: dict[str, dict] = {}
    for record in records:
        cell_id = str(record["cell_id"])
        _require(cell_id not in output, f"duplicate fit cell: {cell_id}")
        score_path = path / str(record["score_path"])
        diagnostic_path = path / str(record["diagnostics_path"])
        _require(sha256_file(score_path) == record["score_sha256"], f"score hash drift: {cell_id}")
        _require(
            sha256_file(diagnostic_path) == record["diagnostics_sha256"],
            f"diagnostic hash drift: {cell_id}",
        )
        arrays = load_npz_no_pickle(score_path)
        _require(set(arrays) == {"row_ids", *arm_ids}, f"score member drift: {cell_id}")
        rows = tuple(str(value) for value in arrays["row_ids"].tolist())
        _require(len(rows) == int(record["n_rows"]), f"score row count drift: {cell_id}")
        scores = {}
        for arm in arm_ids:
            values = np.asarray(arrays[arm], dtype=np.float64)
            _require(values.shape == (len(rows),), f"score shape drift: {cell_id}/{arm}")
            _require(np.isfinite(values).all(), f"non-finite score: {cell_id}/{arm}")
            values.setflags(write=False)
            scores[arm] = values
        output[cell_id] = {
            "record": record,
            "row_ids": rows,
            "scores": MappingProxyType(scores),
        }
    return manifest, output


def verify_ab(
    *,
    output_release: Path,
    source_release: Path,
    config_path: Path,
) -> VerifiedAblation:
    """Verify exact A/B scores before any label-bearing snapshot is opened."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    manifests = {}
    builds = {}
    for build_id in ("A", "B"):
        manifest, cells = _load_fit_build(output_release / f"build_{build_id}", config)
        _require(manifest.get("build_id") == build_id, "fit build ID drifted")
        _require(
            manifest.get("source_prepared_ab_sha256")
            == sha256_file(source_release / "PREPARED_AB_VERIFICATION.json"),
            "fit is not bound to the current prepared A/B certificate",
        )
        manifests[build_id] = manifest
        builds[build_id] = cells

    ids_a = tuple(builds["A"])
    ids_b = tuple(builds["B"])
    _require(ids_a == ids_b and len(ids_a) == 24, "A/B cell roster mismatch")
    arm_ids = tuple(manifests["A"]["arm_ids"])
    _require(tuple(manifests["B"]["arm_ids"]) == arm_ids, "A/B arm roster mismatch")
    score_hashes = []
    row_ids: dict[str, tuple[str, ...]] = {}
    score_by_cell: dict[str, Mapping[str, np.ndarray]] = {}
    for cell_id in ids_a:
        left = builds["A"][cell_id]
        right = builds["B"][cell_id]
        _require(left["row_ids"] == right["row_ids"], f"A/B row mismatch: {cell_id}")
        _require(
            left["record"]["prepared_matrix_sha256"]
            == right["record"]["prepared_matrix_sha256"],
            f"A/B prepared matrix mismatch: {cell_id}",
        )
        _require(
            left["record"]["score_sha256"] == right["record"]["score_sha256"],
            f"A/B score bytes differ: {cell_id}",
        )
        _require(
            left["record"]["diagnostics_sha256"] == right["record"]["diagnostics_sha256"],
            f"A/B diagnostics differ: {cell_id}",
        )
        for arm in arm_ids:
            _require(
                np.array_equal(left["scores"][arm], right["scores"][arm]),
                f"A/B logical score mismatch: {cell_id}/{arm}",
            )
        score_hashes.append({
            "cell_id": cell_id,
            "score_sha256": left["record"]["score_sha256"],
            "diagnostics_sha256": left["record"]["diagnostics_sha256"],
        })
        row_ids[cell_id] = left["row_ids"]
        score_by_cell[cell_id] = left["scores"]
    record = {
        "schema_version": AB_SCHEMA_VERSION,
        "pass": True,
        "n_cells": 24,
        "n_arms": len(arm_ids),
        "cell_ids": list(ids_a),
        "arm_ids": list(arm_ids),
        "source_release": str(source_release),
        "prepared_ab_sha256": sha256_file(source_release / "PREPARED_AB_VERIFICATION.json"),
        "freeze_A_sha256": sha256_file(output_release / "build_A/SCORE_FREEZE_MANIFEST.json"),
        "freeze_B_sha256": sha256_file(output_release / "build_B/SCORE_FREEZE_MANIFEST.json"),
        "score_hashes": score_hashes,
    }
    record["payload_sha256"] = sha256_bytes(canonical_json_bytes(record))
    return VerifiedAblation(
        output_release=output_release,
        source_release=source_release,
        cell_ids=ids_a,
        arm_ids=arm_ids,
        row_ids_by_cell=MappingProxyType(row_ids),
        score_by_cell=MappingProxyType(score_by_cell),
        ab_record=MappingProxyType(record),
    )


def _open_signed_snapshot(
    verified: VerifiedAblation,
) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    manifest_path = verified.source_release / "evaluation/EVALUATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_payload(manifest)
    _require(manifest.get("status") == "OK", "source evaluation is not OK")
    snapshot_path = verified.source_release / "evaluation" / str(manifest["prediction_snapshot_path"])
    _require(
        sha256_file(snapshot_path) == manifest["prediction_snapshot_sha256"],
        "signed prediction snapshot drifted",
    )
    bootstrap_path = verified.source_release / "evaluation" / str(manifest["bootstrap_path"])
    _require(
        sha256_file(bootstrap_path) == manifest["bootstrap_sha256"],
        "signed bootstrap archive drifted",
    )
    return (
        manifest,
        load_npz_no_pickle(snapshot_path),
        load_npz_no_pickle(bootstrap_path),
    )


def _point_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, score)),
        "auprc": float(average_precision_score(y, score)),
    }


def _metric_row(
    *,
    scope: str,
    cell_id: str | None,
    arm_id: str,
    metric: str,
    value: float,
    draws: np.ndarray,
    n_rows: int | None,
    n_groups: int | None,
) -> dict:
    return {
        "scope": scope,
        "cell_id": cell_id,
        "arm_id": arm_id,
        "metric": metric,
        "value": float(value),
        **_interval(draws),
        "n_rows": n_rows,
        "n_groups": n_groups,
    }


def _contrast_row(
    *,
    scope: str,
    cell_id: str | None,
    arm_id: str,
    reference_id: str,
    metric: str,
    value: float,
    reference_value: float,
    draws: np.ndarray,
    reference_draws: np.ndarray,
) -> dict:
    delta = np.asarray(draws) - np.asarray(reference_draws)
    return {
        "scope": scope,
        "cell_id": cell_id,
        "arm_id": arm_id,
        "reference_id": reference_id,
        "metric": metric,
        "value": float(value - reference_value),
        **_interval(delta),
    }


def evaluate(
    verified: VerifiedAblation,
    *,
    draws: int = 20000,
    chunk_size: int = 1000,
) -> tuple[dict, list[dict], list[dict]]:
    """Open the signed labels only after A/B PASS and evaluate all arms."""

    _require(verified.ab_record.get("pass") is True, "label gate requires A/B PASS")
    _require(len(verified.cell_ids) == 24, "label gate requires 24 cells")
    source_manifest, snapshot, source_bootstrap = _open_signed_snapshot(verified)
    signed_ids = tuple(f"signed_{method}" for method in REFERENCE_METHODS)
    evaluation_ids = tuple(verified.arm_ids) + signed_ids
    macro_draws = {
        arm: {metric: np.zeros(draws, dtype=np.float64) for metric in ("auroc", "auprc")}
        for arm in evaluation_ids
    }
    metric_rows: list[dict] = []
    contrast_rows: list[dict] = []
    point_by_cell: dict[str, dict[str, dict[str, float]]] = {}

    for cell_id in verified.cell_ids:
        rows = tuple(str(value) for value in snapshot[f"{cell_id}__row_ids"].tolist())
        _require(rows == verified.row_ids_by_cell[cell_id], f"signed row join failed: {cell_id}")
        groups = tuple(str(value) for value in snapshot[f"{cell_id}__group_ids"].tolist())
        y = np.asarray(snapshot[f"{cell_id}__y_error"], dtype=np.int8)
        _require(y.shape == (len(rows),) and np.isin(y, (0, 1)).all(), f"bad labels: {cell_id}")
        scores = dict(verified.score_by_cell[cell_id])
        for method, signed_id in zip(REFERENCE_METHODS, signed_ids):
            score = np.asarray(snapshot[f"{cell_id}__{method}__score"], dtype=np.float64)
            _require(score.shape == y.shape and np.isfinite(score).all(), f"bad reference: {cell_id}/{method}")
            scores[signed_id] = score
        # The two recomputed anchors must be rank-identical to the signed v2
        # release before any new comparison is accepted.
        for arm, source_method in (("iu_pcr", "iu_pcr"), ("equal_family_mean", "equal_family_mean")):
            source_score = np.asarray(snapshot[f"{cell_id}__{source_method}__score"], dtype=np.float64)
            corr = float(np.corrcoef(scores[arm], source_score)[0, 1])
            _require(np.isfinite(corr) and corr > 1.0 - 1e-10, f"anchor reproduction failed: {cell_id}/{arm}")

        points = {arm: _point_metrics(y, score) for arm, score in scores.items()}
        point_by_cell[cell_id] = points
        cell_draws = {
            arm: {metric: np.empty(draws, dtype=np.float64) for metric in ("auroc", "auprc")}
            for arm in evaluation_ids
        }
        # The signed reconstruction evaluator used this exact cell-keyed draw
        # schedule.  Reuse its already verified anchor/reference metric draws;
        # new arms are evaluated against the same multiplicities below.
        source_draw_methods = {
            "iu_pcr": "iu_pcr",
            "equal_family_mean": "equal_family_mean",
            "signed_deem_b3": "deem_b3",
            "signed_family_nrm_a": "family_nrm_a",
            "signed_pgrd_a": "pgrd_a",
        }
        for arm, method in source_draw_methods.items():
            for metric in ("auroc", "auprc"):
                key = f"cell__{cell_id}__{method}__{metric}"
                values = np.asarray(source_bootstrap[key], dtype=np.float64)
                _require(values.shape == (draws,), f"signed draw shape drift: {key}")
                cell_draws[arm][metric][:] = values
        ordered_groups, group_columns, _, iterator = grouped_bootstrap_multiplicity_chunks(
            groups,
            cell_id=cell_id,
            draws=draws,
            chunk_size=chunk_size,
        )
        computed_arms = tuple(
            arm for arm in evaluation_ids if arm not in source_draw_methods
        )
        for offset, multiplicities, _ in iterator:
            count = len(multiplicities)
            for arm in computed_arms:
                auc, ap = _weighted_binary_metric_draws(
                    y,
                    scores[arm],
                    group_columns,
                    multiplicities,
                )
                cell_draws[arm]["auroc"][offset:offset + count] = auc
                cell_draws[arm]["auprc"][offset:offset + count] = ap

        print(f"EVAL {len(point_by_cell):02d}/24 {cell_id}", flush=True)

        for arm in evaluation_ids:
            for metric in ("auroc", "auprc"):
                values = cell_draws[arm][metric]
                _require(np.isfinite(values).mean() >= 0.95, f"too few valid draws: {cell_id}/{arm}/{metric}")
                macro_draws[arm][metric] += values / len(verified.cell_ids)
                metric_rows.append(_metric_row(
                    scope="cell",
                    cell_id=cell_id,
                    arm_id=arm,
                    metric=metric,
                    value=points[arm][metric],
                    draws=values,
                    n_rows=len(rows),
                    n_groups=len(ordered_groups),
                ))
        for arm in verified.arm_ids:
            for reference in CONTRAST_REFERENCES:
                for metric in ("auroc", "auprc"):
                    contrast_rows.append(_contrast_row(
                        scope="cell",
                        cell_id=cell_id,
                        arm_id=arm,
                        reference_id=reference,
                        metric=metric,
                        value=points[arm][metric],
                        reference_value=points[reference][metric],
                        draws=cell_draws[arm][metric],
                        reference_draws=cell_draws[reference][metric],
                    ))

    macro_points = {
        arm: {
            metric: float(np.mean([
                point_by_cell[cell][arm][metric] for cell in verified.cell_ids
            ]))
            for metric in ("auroc", "auprc")
        }
        for arm in evaluation_ids
    }
    for arm in evaluation_ids:
        for metric in ("auroc", "auprc"):
            metric_rows.append(_metric_row(
                scope="macro24",
                cell_id=None,
                arm_id=arm,
                metric=metric,
                value=macro_points[arm][metric],
                draws=macro_draws[arm][metric],
                n_rows=None,
                n_groups=None,
            ))
    for arm in verified.arm_ids:
        for reference in CONTRAST_REFERENCES:
            for metric in ("auroc", "auprc"):
                contrast_rows.append(_contrast_row(
                    scope="macro24",
                    cell_id=None,
                    arm_id=arm,
                    reference_id=reference,
                    metric=metric,
                    value=macro_points[arm][metric],
                    reference_value=macro_points[reference][metric],
                    draws=macro_draws[arm][metric],
                    reference_draws=macro_draws[reference][metric],
                ))

    result = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "OK",
        "headline_status": "D0_MECHANISM_ABLATION_NOT_INDEPENDENT_VALIDATION",
        "n_cells": 24,
        "n_new_arms": len(verified.arm_ids),
        "signed_context_methods": list(signed_ids),
        "bootstrap_draws": draws,
        "bootstrap_unit": "source_group_within_cell",
        "aggregation": "equal_cell_macro24",
        "ab_verification": dict(verified.ab_record),
        "source_evaluation_manifest_sha256": sha256_file(
            verified.source_release / "evaluation/EVALUATION_MANIFEST.json"
        ),
        "source_prediction_snapshot_sha256": source_manifest["prediction_snapshot_sha256"],
        "macro_points": macro_points,
        "label_selection_used": False,
        "claim_boundary": "retrospective frozen24 mechanism evidence only",
    }
    result["payload_sha256"] = sha256_bytes(canonical_json_bytes(result))
    return result, metric_rows, contrast_rows


__all__ = [
    "AB_SCHEMA_VERSION",
    "CONTRAST_REFERENCES",
    "EVALUATION_SCHEMA_VERSION",
    "REFERENCE_METHODS",
    "VerifiedAblation",
    "evaluate",
    "verify_ab",
]
