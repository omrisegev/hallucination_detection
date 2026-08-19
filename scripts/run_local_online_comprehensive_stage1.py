#!/usr/bin/env python3
"""Run S0/S1 of the frozen comprehensive Local/Online protocol.

This script fits no score with target labels.  It freezes candidate curves and
locators before calibration labels are used for thresholds and metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/gl_liu_v1/localization"))

from evidence_drop import EVIDENCE_FNS, evidence_drop_risk  # noqa: E402
from localization_metrics import step_drop_scores  # noqa: E402
from scripts.run_global_local_online_architecture_v2 import (  # noqa: E402
    _best_threshold,
    _cell_path,
    _peak_locator,
    _processbench,
    _safe_ap,
    _safe_auc,
    _zapply,
    _zfit,
    fit_registered_global,
    fit_registered_local,
    load_rows,
)
from spectral_utils.local_online_comprehensive import (  # noqa: E402
    PreparedTrace,
    fit_references,
    fit_trajectory_head_prepared,
    local_candidate_roster,
    prepare_trace,
)


# A verification replay must not be able to overwrite the artifacts it exists to
# verify.  Setting LOCAL_ONLINE_V1_OUT redirects every write to a scratch
# directory, which turns reproducing the frozen run into a comparison instead of
# a mutation.  Unset, the default keeps the original behaviour.
OUT = Path(os.environ["LOCAL_ONLINE_V1_OUT"]).resolve() if os.environ.get(
    "LOCAL_ONLINE_V1_OUT") else ROOT / "results/local_online_comprehensive_v1"

# The gated document is the snapshot the run actually read, recovered byte-exact
# and kept immutable.  The editable `LOCAL_ONLINE_COMPREHENSIVE_V1.md` was
# revised after the run and before it was committed, so it hashes to something
# else (b5991a89...); pointing the gate at it would either fail forever or, if
# the constant below were "fixed" to match, stop checking anything at all.  The
# constant is the run's recorded protocol hash and does not move.
PROTOCOL = ROOT / "docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md"
PROTOCOL_SHA256 = "c921b0d446eebd4611c4426168c30410741997ea2c6d23238e5d22b83e8d1e5b"
SEED = 20260816
BOOTSTRAP = 2000
CELLS = (("qwen3_4b", "gsm8k"), ("qwen3_4b", "math"))
LOCATORS = ("peak", "persistent_q90_3", "step_top5mean")

COMPETITOR_ROOTS = {
    "qwen_prm": ROOT / "dataset_cache/four_localization/pb_prm_qwen25math7b_full",
    "qwen72b_critic": ROOT / "dataset_cache/four_localization/pb_critic_qwen72b_full",
    "qwen3_judge_control": ROOT / "dataset_cache/four_localization/pb_uprm_baseline_qwen3_8b_full",
}
COMPETITOR_PATTERNS = {
    "qwen_prm": "pb_prm_{family}.pkl",
    "qwen72b_critic": "pb_critic_{family}.pkl",
    "qwen3_judge_control": "pb_uprm_base_{family}.pkl",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stage_partition(family: str, unit: str) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"local-online-v1|{family}|{unit}".encode()).digest()[:8],
        "big",
    ) % 100
    if value < 40:
        return "calibration"
    if value < 60:
        return "development"
    if value < 80:
        return "architecture"
    return "audit"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # `write_text` without an encoding uses the *locale* codec, so the same
    # report is UTF-8 on macOS and cp1252 on Windows. That is how the frozen
    # run's em-dashes came back as mojibake on a replay: the numbers were
    # identical and the bytes were not.
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _token_to_step(token: int, row: Mapping[str, Any]) -> int:
    for index, span in enumerate(row.get("step_token_spans") or ()):
        if span is not None and int(span[0]) <= token < int(span[1]):
            return int(index)
    return -1


def _persistent_reference(curves: Sequence[np.ndarray]) -> float:
    sampled = []
    for curve in curves:
        curve = np.asarray(curve, dtype=float)
        indexes = np.linspace(0, len(curve) - 1, 32, dtype=int)
        sampled.append(curve[indexes])
    return float(np.quantile(np.concatenate(sampled), 0.90))


def _persistent_locator(curve: np.ndarray, row: Mapping[str, Any], threshold: float) -> int:
    hits = np.asarray(curve, dtype=float) > float(threshold)
    for index in range(2, len(hits)):
        if bool(hits[index - 2:index + 1].all()):
            return _token_to_step(index - 2, row)
    return _peak_locator(curve, row)


def _step_top5_locator(curve: np.ndarray, row: Mapping[str, Any]) -> int:
    curve = np.asarray(curve, dtype=float)
    scores = []
    for span in row.get("step_token_spans") or ():
        if span is None:
            scores.append(float("nan"))
            continue
        values = curve[max(0, int(span[0])):min(len(curve), int(span[1]))]
        values = values[np.isfinite(values)]
        if not len(values):
            scores.append(float("nan"))
            continue
        k = min(5, len(values))
        scores.append(float(np.mean(np.partition(values, -k)[-k:])))
    return int(np.nanargmax(scores)) if np.isfinite(scores).any() else -1


def _locators(
    curves: Sequence[np.ndarray], rows: Sequence[Mapping[str, Any]],
    kind: str, persistent_threshold: float,
) -> np.ndarray:
    if kind == "peak":
        return np.asarray([_peak_locator(curve, row) for curve, row in zip(curves, rows)])
    if kind == "persistent_q90_3":
        return np.asarray([
            _persistent_locator(curve, row, persistent_threshold)
            for curve, row in zip(curves, rows)
        ])
    return np.asarray([
        _step_top5_locator(curve, row) for curve, row in zip(curves, rows)
    ])


def _evaluate_curve_system(
    name: str,
    family: str,
    calibration_rows: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
    calibration_curves: Sequence[np.ndarray],
    development_curves: Sequence[np.ndarray],
    *,
    access_tier: str,
    fidelity: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records, metrics = [], []
    cal_labels = np.asarray([int(row["label"]) for row in calibration_rows])
    dev_labels = np.asarray([int(row["label"]) for row in development_rows])
    cal_detector = np.asarray([float(np.nanmax(curve)) for curve in calibration_curves])
    dev_detector = np.asarray([float(np.nanmax(curve)) for curve in development_curves])
    persistent_threshold = _persistent_reference(calibration_curves)
    for locator_name in LOCATORS:
        cal_locator = _locators(
            calibration_curves, calibration_rows, locator_name, persistent_threshold
        )
        dev_locator = _locators(
            development_curves, development_rows, locator_name, persistent_threshold
        )
        threshold, calibration_f1 = _best_threshold(
            cal_detector, cal_locator, cal_labels
        )
        prediction = np.where(dev_detector > threshold, dev_locator, -1)
        result = _processbench(prediction, dev_labels)
        error = dev_labels != -1
        mae = float(np.mean(np.abs(dev_locator[error] - dev_labels[error]))) if error.any() else float("nan")
        late = float(np.mean(dev_locator[error] > dev_labels[error])) if error.any() else float("nan")
        key = f"{name}__{locator_name}"
        metrics.append({
            "candidate": key,
            "base_candidate": name,
            "locator": locator_name,
            "family": family,
            "task": "local",
            "primary": result["f1"],
            **result,
            "detector_auroc": _safe_auc(dev_labels != -1, dev_detector),
            "detector_auprc": _safe_ap(dev_labels != -1, dev_detector),
            "mean_absolute_step_error": mae,
            "late_location_rate": late,
            "threshold": threshold,
            "calibration_f1": calibration_f1,
            "n": len(dev_labels),
            "access_tier": access_tier,
            "fidelity": fidelity,
        })
        records.extend({
            "candidate": key,
            "base_candidate": name,
            "locator_kind": locator_name,
            "family": family,
            "unit": row["_unit"],
            "target": int(target),
            "score": float(score),
            "locator": int(locator),
            "prediction": int(pred),
            "access_tier": access_tier,
            "fidelity": fidelity,
        } for row, target, score, locator, pred in zip(
            development_rows, dev_labels, dev_detector, dev_locator, prediction
        ))
    return records, metrics


def _mindgap(
    calibration_rows: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
    family: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def score(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        detector, locator = [], []
        for row in rows:
            evidence = EVIDENCE_FNS["shannon"](row, 20)
            detector.append(evidence_drop_risk(evidence, M=5, ema_span=5))
            step = step_drop_scores(evidence, row["step_token_spans"], ema_span=5)
            locator.append(int(np.nanargmax(step)) if np.isfinite(step).any() else -1)
        return np.asarray(detector), np.asarray(locator)

    cal_detector, cal_locator = score(calibration_rows)
    dev_detector, dev_locator = score(development_rows)
    cal_labels = np.asarray([int(row["label"]) for row in calibration_rows])
    dev_labels = np.asarray([int(row["label"]) for row in development_rows])
    threshold, calibration_f1 = _best_threshold(cal_detector, cal_locator, cal_labels)
    prediction = np.where(dev_detector > threshold, dev_locator, -1)
    result = _processbench(prediction, dev_labels)
    metrics = [{
        "candidate": "mind_the_gap",
        "base_candidate": "mind_the_gap",
        "locator": "paper_evidence_drop",
        "family": family,
        "task": "local",
        "primary": result["f1"],
        **result,
        "detector_auroc": _safe_auc(dev_labels != -1, dev_detector),
        "detector_auprc": _safe_ap(dev_labels != -1, dev_detector),
        "threshold": threshold,
        "calibration_f1": calibration_f1,
        "n": len(dev_labels),
        "access_tier": "A",
        "fidelity": "protocol_reproduction",
    }]
    records = [{
        "candidate": "mind_the_gap",
        "base_candidate": "mind_the_gap",
        "locator_kind": "paper_evidence_drop",
        "family": family,
        "unit": row["_unit"],
        "target": int(target),
        "score": float(score),
        "locator": int(locator),
        "prediction": int(pred),
        "access_tier": "A",
        "fidelity": "protocol_reproduction",
    } for row, target, score, locator, pred in zip(
        development_rows, dev_labels, dev_detector, dev_locator, prediction
    )]
    return records, metrics


def _global_local_incumbent(
    name: str,
    family: str,
    calibration_rows: Sequence[Mapping[str, Any]],
    development_rows: Sequence[Mapping[str, Any]],
    global_model: Any,
    local_model: Any,
    local_curves_cal: Sequence[np.ndarray],
    local_curves_dev: Sequence[np.ndarray],
    weight: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cal_global = np.asarray([global_model.score(row) for row in calibration_rows])
    dev_global = np.asarray([global_model.score(row) for row in development_rows])
    cal_local = np.asarray([float(np.max(curve)) for curve in local_curves_cal])
    dev_local = np.asarray([float(np.max(curve)) for curve in local_curves_dev])
    detector_cal = weight * _zapply(cal_global, _zfit(cal_global)) + (1.0 - weight) * _zapply(cal_local, _zfit(cal_local))
    detector_dev = weight * _zapply(dev_global, _zfit(cal_global)) + (1.0 - weight) * _zapply(dev_local, _zfit(cal_local))
    locator_cal = np.asarray([_peak_locator(curve, row) for curve, row in zip(local_curves_cal, calibration_rows)])
    locator_dev = np.asarray([_peak_locator(curve, row) for curve, row in zip(local_curves_dev, development_rows)])
    labels_cal = np.asarray([int(row["label"]) for row in calibration_rows])
    labels_dev = np.asarray([int(row["label"]) for row in development_rows])
    threshold, calibration_f1 = _best_threshold(detector_cal, locator_cal, labels_cal)
    prediction = np.where(detector_dev > threshold, locator_dev, -1)
    result = _processbench(prediction, labels_dev)
    metric = {
        "candidate": name,
        "base_candidate": name,
        "locator": "peak",
        "family": family,
        "task": "local",
        "primary": result["f1"],
        **result,
        "detector_auroc": _safe_auc(labels_dev != -1, detector_dev),
        "detector_auprc": _safe_ap(labels_dev != -1, detector_dev),
        "threshold": threshold,
        "calibration_f1": calibration_f1,
        "n": len(labels_dev),
        "access_tier": "A",
        "fidelity": "registered_incumbent_replay",
    }
    records = [{
        "candidate": name,
        "base_candidate": name,
        "locator_kind": "peak",
        "family": family,
        "unit": row["_unit"],
        "target": int(target),
        "score": float(score),
        "locator": int(locator),
        "prediction": int(pred),
        "access_tier": "A",
        "fidelity": "registered_incumbent_replay",
    } for row, target, score, locator, pred in zip(
        development_rows, labels_dev, detector_dev, locator_dev, prediction
    )]
    return records, [metric]


def _tier_b(
    family: str, development_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wanted = {row["_unit"]: row for row in development_rows}
    records, metrics = [], []
    for candidate, root in COMPETITOR_ROOTS.items():
        path = root / COMPETITOR_PATTERNS[candidate].format(family=family)
        with path.open("rb") as handle:
            data = pickle.load(handle)
        lookup = {str(item.get("id", key)): item for key, item in data.items()}
        if set(wanted) - set(lookup):
            raise RuntimeError(f"{candidate}/{family}: incomplete ID join")
        labels = np.asarray([int(wanted[unit]["label"]) for unit in sorted(wanted)])
        predictions = np.asarray([int(lookup[unit]["prediction"]) for unit in sorted(wanted)])
        result = _processbench(predictions, labels)
        fidelity = "exact_local_competitor_run" if candidate != "qwen72b_critic" else "critic_protocol_different_model"
        metrics.append({
            "candidate": candidate,
            "base_candidate": candidate,
            "locator": "native_prediction",
            "family": family,
            "task": "local",
            "primary": result["f1"],
            **result,
            "n": len(labels),
            "access_tier": "B",
            "fidelity": fidelity,
        })
        records.extend({
            "candidate": candidate,
            "base_candidate": candidate,
            "locator_kind": "native_prediction",
            "family": family,
            "unit": unit,
            "target": int(target),
            "locator": int(prediction),
            "prediction": int(prediction),
            "access_tier": "B",
            "fidelity": fidelity,
        } for unit, target, prediction in zip(sorted(wanted), labels, predictions))
    return records, metrics


def _aggregate(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for candidate in sorted({row["candidate"] for row in metrics}):
        rows = [row for row in metrics if row["candidate"] == candidate]
        output.append({
            "candidate": candidate,
            "base_candidate": rows[0]["base_candidate"],
            "locator": rows[0]["locator"],
            "primary": float(np.mean([float(row["primary"]) for row in rows])),
            "families": len(rows),
            "access_tier": rows[0]["access_tier"],
            "fidelity": rows[0]["fidelity"],
        })
    return output


def _bootstrap_interval(
    records: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> tuple[float, float, float, int, int]:
    families = sorted({row["family"] for row in records if row["candidate"] in {candidate, reference}})
    prepared = []
    point_deltas = []
    for family in families:
        relevant = [row for row in records if row["family"] == family and row["candidate"] in {candidate, reference}]
        lookup = {
            method: {row["unit"]: row for row in relevant if row["candidate"] == method}
            for method in (candidate, reference)
        }
        units = sorted(set(lookup[candidate]) & set(lookup[reference]))
        if not units:
            continue
        prepared.append((units, lookup))
        point_deltas.append(
            _processbench([lookup[candidate][unit]["prediction"] for unit in units], [lookup[candidate][unit]["target"] for unit in units])["f1"]
            - _processbench([lookup[reference][unit]["prediction"] for unit in units], [lookup[reference][unit]["target"] for unit in units])["f1"]
        )
    rng = np.random.default_rng(SEED + sum(ord(char) for char in candidate + reference))
    draws = []
    for _ in range(BOOTSTRAP):
        deltas = []
        for units, lookup in prepared:
            sampled = [units[index] for index in rng.integers(0, len(units), len(units))]
            left = _processbench([lookup[candidate][unit]["prediction"] for unit in sampled], [lookup[candidate][unit]["target"] for unit in sampled])["f1"]
            right = _processbench([lookup[reference][unit]["prediction"] for unit in sampled], [lookup[reference][unit]["target"] for unit in sampled])["f1"]
            deltas.append(left - right)
        draws.append(float(np.mean(deltas)))
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(np.mean(point_deltas)), float(low), float(high), int(sum(value > 0 for value in point_deltas)), int(sum(value < 0 for value in point_deltas))


def _stage0() -> None:
    rows = [
        {"task": "local", "method": "Mind the Gap", "value": 0.249631, "access_tier": "A", "scope": "ProcessBench full four-subset macro", "status": "direct published-method reproduction"},
        {"task": "local", "method": "GL-LIU v1", "value": 0.312456, "access_tier": "A", "scope": "Qwen ProcessBench macro", "status": "frozen incumbent"},
        {"task": "local", "method": "maximum token entropy", "value": 0.315011, "access_tier": "A", "scope": "Llama ProcessBench macro", "status": "transparent incumbent bar"},
        {"task": "local", "method": "Step-272 two-head", "value": 0.313617, "access_tier": "A", "scope": "12 scorer/family-cell macro", "status": "current architecture incumbent"},
        {"task": "local", "method": "broad-28 DUFS", "value": 0.2903, "access_tier": "A", "scope": "8-cell historical macro", "status": "failure anchor"},
        {"task": "local", "method": "Qwen2.5-72B critic", "value": 0.594003, "access_tier": "B", "scope": "same 3,400 ProcessBench rows", "status": "critic-protocol reproduction, different model"},
        {"task": "local", "method": "Qwen2.5-Math-PRM-7B", "value": 0.729354, "access_tier": "B", "scope": "same 3,400 ProcessBench rows", "status": "supervised ceiling"},
        {"task": "local", "method": "Qwen3-8B judge control", "value": 0.096441, "access_tier": "B", "scope": "same 3,400 ProcessBench rows", "status": "not uPRM"},
        {"task": "online", "method": "IU28", "value": (0.631693 + 0.675056) / 2, "access_tier": "A", "scope": "historical 11-cell equal-family AUROC@64/128", "status": "registered direct bar"},
        {"task": "online", "method": "DeepConf-w64", "value": (0.607238 + 0.661281) / 2, "access_tier": "A", "scope": "historical 11-cell equal-family AUROC@64/128", "status": "direct black-box bar"},
        {"task": "online", "method": "Step-272 two-head", "value": 0.607531, "access_tier": "A", "scope": "12 scorer/family-cell macro AUROC@64/128", "status": "current architecture incumbent"},
        {"task": "online", "method": "Streaming supervised probe", "value": 0.811, "access_tier": "C", "scope": "published Qwen, different data/labels/hidden-state access", "status": "context only; no delta"},
    ]
    _write_csv(OUT / "STAGE_0_BASELINES.csv", rows)
    lines = [
        "# S0 competitor baseline",
        "",
        "**Verdict: `MECHANICS_ONLY_NO_PERFORMANCE_CLAIM`.**",
        "",
        "The numbers below establish reporting bars; incompatible scopes are not subtracted.",
        "",
        "| task | method | value | tier | scope |",
        "|---|---|---:|---|---|",
    ]
    lines.extend(
        f"| {row['task']} | {row['method']} | {row['value']:.4f} | {row['access_tier']} | {row['scope']} |"
        for row in rows
    )
    lines.extend([
        "",
        "Tier A is the only same-access improvement tier. Tier B uses the same ProcessBench rows but substantially different compute. Tier C is cross-protocol context only.",
    ])
    (OUT / "STAGE_0_BASELINES.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    observed = _sha256(PROTOCOL)
    if observed != PROTOCOL_SHA256:
        raise RuntimeError(
            "frozen protocol hash mismatch\n"
            f"  file     : {PROTOCOL}\n"
            f"  expected : {PROTOCOL_SHA256}\n"
            f"  observed : {observed}\n"
            "The snapshot is checked in as binary precisely so this cannot drift; "
            "an observed hash that differs usually means the file was rewritten on "
            "checkout (core.autocrlf) rather than edited."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    _stage0()
    roster = local_candidate_roster()
    records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    started_all = time.perf_counter()

    for model_name, family in CELLS:
        rows = load_rows(_cell_path(model_name, family))
        for row in rows:
            row["_stage"] = _stage_partition(family, row["_unit"])
        calibration = [row for row in rows if row["_stage"] == "calibration"]
        development = [row for row in rows if row["_stage"] == "development"]
        print(f"S1 {model_name}/{family}: calibration={len(calibration)} development={len(development)}", flush=True)

        started = time.perf_counter()
        references = fit_references(calibration)
        prepared_cal = [prepare_trace(row, references) for row in calibration]
        prepared_dev = [prepare_trace(row, references) for row in development]
        diagnostics[f"{model_name}/{family}"] = {
            "calibration": len(calibration),
            "development": len(development),
            "references": references.as_dict(),
            "preparation_seconds": time.perf_counter() - started,
            "heads": {},
        }

        global_model = fit_registered_global(calibration)
        registered_local = fit_registered_local(calibration)
        registered_cal_curves = [registered_local.curve(row) for row in calibration]
        registered_dev_curves = [registered_local.curve(row) for row in development]
        item_records, item_metrics = _evaluate_curve_system(
            "l_core5", family, calibration, development,
            registered_cal_curves, registered_dev_curves,
            access_tier="A", fidelity="historical_core5_replay",
        )
        records.extend(item_records); metrics.extend(item_metrics)
        item_records, item_metrics = _global_local_incumbent(
            "gl_liu_v1_replay", family, calibration, development,
            global_model, registered_local, registered_cal_curves,
            registered_dev_curves, 0.75,
        )
        records.extend(item_records); metrics.extend(item_metrics)

        raw9_curves = None
        for name, (representation, operators) in roster.items():
            head_started = time.perf_counter()
            head = fit_trajectory_head_prepared(
                prepared_cal,
                name=name,
                representation=representation,
                operators=operators,
            )
            cal_curves = [head.curve_from_level(item.representations[representation]) for item in prepared_cal]
            dev_curves = [head.curve_from_level(item.representations[representation]) for item in prepared_dev]
            item_records, item_metrics = _evaluate_curve_system(
                name, family, calibration, development, cal_curves, dev_curves,
                access_tier="A", fidelity="frozen_retrospective_candidate",
            )
            records.extend(item_records); metrics.extend(item_metrics)
            diagnostics[f"{model_name}/{family}"]["heads"][name] = {
                **head.diagnostics,
                "fit_and_score_seconds": time.perf_counter() - head_started,
            }
            if name == "l_raw9__level":
                raw9_curves = (cal_curves, dev_curves)
            print(f"  {name} done", flush=True)

        if raw9_curves is None:
            raise RuntimeError("raw9 incumbent missing")
        item_records, item_metrics = _global_local_incumbent(
            "step272_twohead_replay", family, calibration, development,
            global_model, None, raw9_curves[0], raw9_curves[1], 0.50,
        )
        records.extend(item_records); metrics.extend(item_metrics)

        max_entropy_cal = [np.asarray(row["token_entropies"], dtype=float) for row in calibration]
        max_entropy_dev = [np.asarray(row["token_entropies"], dtype=float) for row in development]
        item_records, item_metrics = _evaluate_curve_system(
            "max_entropy", family, calibration, development,
            max_entropy_cal, max_entropy_dev,
            access_tier="A", fidelity="transparent_same_trace_baseline",
        )
        records.extend(item_records); metrics.extend(item_metrics)

        item_records, item_metrics = _mindgap(calibration, development, family)
        records.extend(item_records); metrics.extend(item_metrics)
        item_records, item_metrics = _tier_b(family, development)
        records.extend(item_records); metrics.extend(item_metrics)

    aggregate = _aggregate(metrics)
    incumbent_names = {
        "gl_liu_v1_replay", "step272_twohead_replay", "mind_the_gap",
        "max_entropy__peak", "max_entropy__persistent_q90_3", "max_entropy__step_top5mean",
    }
    direct = [row for row in aggregate if row["candidate"] in incumbent_names]
    reference = max(direct, key=lambda row: row["primary"])["candidate"]
    candidates = [row for row in aggregate if row["base_candidate"].startswith("l_")]
    numerical_best_any = max(candidates, key=lambda row: row["primary"])
    family_primary = {
        (row["candidate"], row["family"]): float(row["primary"])
        for row in metrics
    }
    incumbent_families = {
        row["family"]: float(row["primary"])
        for row in metrics if row["candidate"] == "step272_twohead_replay"
    }
    promotable = [
        row for row in candidates
        if all(
            family_primary[(row["candidate"], family)] >= incumbent - 0.010
            for family, incumbent in incumbent_families.items()
        )
    ]
    rejected_family_margin = sorted(
        row["candidate"] for row in candidates if row not in promotable
    )
    numerical_best = max(promotable, key=lambda row: row["primary"]) if promotable else None
    intervals = []
    for row in aggregate:
        if row["access_tier"] != "A" or row["candidate"] == reference:
            continue
        delta, low, high, wins, losses = _bootstrap_interval(
            records, row["candidate"], reference
        )
        intervals.append({
            "candidate": row["candidate"], "reference": reference,
            "delta": delta, "ci_low": low, "ci_high": high,
            "family_wins": wins, "family_losses": losses,
        })

    best_intervals = {}
    for row in promotable:
        delta, low, high, wins, losses = _bootstrap_interval(
            records, row["candidate"], numerical_best["candidate"]
        ) if row["candidate"] != numerical_best["candidate"] else (0.0, 0.0, 0.0, 0, 0)
        best_intervals[row["candidate"]] = (delta, low, high)
    eligible = [
        row for row in promotable
        if row["primary"] >= numerical_best["primary"] - 0.005
        and best_intervals[row["candidate"]][1] <= 0.0 <= best_intervals[row["candidate"]][2]
    ] if numerical_best is not None else []
    def cost(row: Mapping[str, Any]) -> tuple[int, int, str]:
        base = row["base_candidate"]
        if base == "l_core5":
            representation, operators = "core5", ("level",)
        else:
            representation, operators = roster[base]
        widths = {"core5": 5, "raw7": 7, "raw9": 9, "family6": 6, "broad28": 28}
        locator_cost = {"peak": 0, "persistent_q90_3": 1, "step_top5mean": 1}
        return widths[representation] * len(operators), locator_cost[row["locator"]], row["candidate"]
    if numerical_best is None:
        selected = next(row for row in aggregate if row["candidate"] == "step272_twohead_replay")
    else:
        selected = min(eligible, key=cost) if eligible else numerical_best
    selected_interval = next((row for row in intervals if row["candidate"] == selected["candidate"]), None)
    verdict = "PARITY_WITH_DIRECT_COMPETITOR"
    if selected_interval and selected_interval["ci_low"] > 0:
        verdict = "IMPROVES_DIRECT_COMPETITOR"
    elif selected["primary"] < next(row["primary"] for row in aggregate if row["candidate"] == reference) - 0.005:
        verdict = "REGRESSES_DIRECT_COMPETITOR"

    _write_csv(OUT / "STAGE_1_LOCAL_PER_QUESTION.csv", records)
    _write_csv(OUT / "STAGE_1_LOCAL_CELL_METRICS.csv", metrics)
    _write_csv(OUT / "STAGE_1_LOCAL_AGGREGATE.csv", aggregate)
    _write_csv(OUT / "STAGE_1_LOCAL_INTERVALS.csv", intervals)
    _write_json(OUT / "STAGE_1_LOCAL_DIAGNOSTICS.json", diagnostics)
    selection = {
        "verdict": verdict,
        "selected": selected,
        "direct_reference": reference,
        "numerical_best": numerical_best,
        "numerical_best_before_family_guard": numerical_best_any,
        "rejected_by_step272_family_margin": rejected_family_margin,
        "rule": "first require no family worse than Step-272 by >0.010; then simplest within 0.005 of promotable numerical best with paired interval including zero",
        "protocol_sha256": PROTOCOL_SHA256,
        "score_sha256": _sha256(OUT / "STAGE_1_LOCAL_PER_QUESTION.csv"),
    }
    _write_json(OUT / "STAGE_1_LOCAL_SELECTION.json", selection)

    lookup_interval = {row["candidate"]: row for row in intervals}
    lines = [
        "# S1 Local feature and locator screen",
        "",
        f"**Verdict: `{verdict}`.**",
        "",
        f"Direct Tier-A reference on the same development rows: `{reference}`.",
        f"Frozen S1 selection: `{selected['candidate']}`.",
        "",
        "| method | F1 | tier | delta vs direct bar | 95% CI |",
        "|---|---:|---|---:|---|",
    ]
    for row in sorted(aggregate, key=lambda item: item["primary"], reverse=True):
        interval = lookup_interval.get(row["candidate"])
        delta = "—" if interval is None else f"{interval['delta']:+.4f}"
        ci = "—" if interval is None else f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
        lines.append(f"| {row['candidate']} | {row['primary']:.4f} | {row['access_tier']} | {delta} | {ci} |")
    lines.extend([
        "",
        "Tier-B critic/PRM rows are same-row compute ceilings and are not used as same-access deltas. The full report keeps every candidate; selection does not hide losing variants.",
    ])
    (OUT / "STAGE_1_LOCAL.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    _write_json(OUT / "RUN_MANIFEST.json", {
        "status": "STAGE_1_COMPLETE_STAGE_2_PENDING",
        "protocol": str(PROTOCOL),
        "protocol_sha256": PROTOCOL_SHA256,
        "stage0_sha256": _sha256(OUT / "STAGE_0_BASELINES.csv"),
        "stage1_selection_sha256": _sha256(OUT / "STAGE_1_LOCAL_SELECTION.json"),
        "new_inference": False,
        "gpu_hours": 0,
        "drive_mutation": False,
        "elapsed_seconds": time.perf_counter() - started_all,
    })
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
