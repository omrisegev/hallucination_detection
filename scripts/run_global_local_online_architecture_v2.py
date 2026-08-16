#!/usr/bin/env python3
"""Run the frozen CPU-only three-output architecture search.

Candidate score construction is completed and hashed before target fields are
used for thresholds or metrics.  The only selection cells are Qwen3-4B GSM8K
and MATH.  All other cells receive the frozen identities without adaptation
other than their declared unlabeled calibration fit and calibrated threshold.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import pickle
import time
import tracemalloc
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from spectral_utils.feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from spectral_utils.multitask_trajectory import (
    FIT_POSITIONS,
    GLOBAL_HEADS,
    HEAD_FEATURES,
    LOCAL_HEADS,
    ONLINE_HEADS,
    ChannelReference,
    FrozenIUHead,
    causal_states,
    equal_positions,
    fit_channel_reference,
    fit_iu_head,
    stable_partition,
    truncate_row,
)
from spectral_utils.online_convergence import (
    causal_raw_prefix_matrix,
    fit_frozen_prefix_iu,
)
from spectral_utils.online_localization_fusion import causal_trace_features
from spectral_utils.repeated_measurement_reliability import FixedMixedV2Transformer
from spectral_utils.token_feature_views import _cusum_abs_series, _sw_var_series
from spectral_utils.upcr import upcr_fit


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/global_local_online_architecture_v2"
PROTOCOL = ROOT / "docs/experiments/GLOBAL_LOCAL_ONLINE_ARCHITECTURE_V2.md"
SEED = 20260816
BOOTSTRAP = 2000
BUDGETS = (16, 32, 64, 128, 256, 512)
DEV_CELLS = {("qwen3_4b", "gsm8k"), ("qwen3_4b", "math")}
MODELS = ("qwen3_4b", "qwen3_8b", "llama31_8b")
FAMILIES = ("gsm8k", "math", "olympiadbench", "omnimath")

CELL_ROOTS = {
    "qwen3_4b": ROOT / "cache/localization/processbench/pb_qwen3_4b",
    "qwen3_8b": ROOT / "cache/localization/processbench/pb_qwen3_8b",
    "llama31_8b": ROOT / "dataset_cache/repgrid/pb_llama31_8b",
}

NEW_GLOBAL = tuple(GLOBAL_HEADS)
NEW_LOCAL = tuple(LOCAL_HEADS)
NEW_ONLINE = tuple(ONLINE_HEADS)
GLOBAL_CANDIDATES = NEW_GLOBAL + ("g_registered_mixed",)
LOCAL_CANDIDATES = NEW_LOCAL + ("l_registered_core5",)
ONLINE_CANDIDATES = NEW_ONLINE + ("o_iu28_registered",)

HEAD_COST = {
    **{name: len(HEAD_FEATURES[name]) for name in HEAD_FEATURES},
    "g_registered_mixed": 30,
    "l_registered_core5": 5,
    "o_iu28_registered": 28,
}
STATE_COST = {
    **{name: 0 for name in NEW_GLOBAL},
    "g_registered_mixed": 0,
    "l_level9": 9,
    "l_onset9": 18,
    "l_level_onset18": 27,
    "l_registered_core5": 12,
    "o_level_ewma18": 18,
    "o_level_ewma_onset27": 27,
    "o_ewma_area_persist27": 36,
    "o_iu28_registered": 28,
}

ORDINARY_FIT = {
    "loss": "l2",
    "exclusion": False,
    "difficulty_gate": False,
    "simple_avg_fallback": False,
    "recompute_after_exclusion": False,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
    "n_components": 2,
    "auto_components": False,
}
REGISTERED_LOCAL_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score_hash(values: Sequence[np.ndarray] | np.ndarray) -> str:
    if isinstance(values, np.ndarray):
        packed = np.asarray(values, dtype="<f8").reshape(-1)
    else:
        packed = np.concatenate([np.asarray(value, dtype="<f8").reshape(-1) for value in values])
    return hashlib.sha256(packed.tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _cell_path(model: str, family: str) -> Path:
    return CELL_ROOTS[model] / f"processbench_{family}.pkl"


def preflight_inventory() -> list[dict[str, Any]]:
    """Validate all twelve caches and scorer-shared IDs before outcome scoring."""

    required = {
        "token_entropies", "token_spilled_energies", "token_logsumexp",
        "top_k_logprobs", "gen_token_ids", "step_token_spans", "label",
        "final_answer_correct", "id",
    }
    inventory, ids_by_family = [], {}
    for model_name in MODELS:
        for family in FAMILIES:
            path = _cell_path(model_name, family)
            if not path.exists():
                raise FileNotFoundError(path)
            rows = load_rows(path)
            missing = sorted({name for row in rows for name in required if row.get(name) is None})
            if missing:
                raise RuntimeError(f"{model_name}/{family} missing required telemetry: {missing}")
            ids = tuple(row["_unit"] for row in rows)
            partitions = tuple(row["_partition"] for row in rows)
            if family in ids_by_family:
                reference_ids, reference_partitions = ids_by_family[family]
                if ids != reference_ids or partitions != reference_partitions:
                    raise RuntimeError(f"shared-ID/split mismatch for {model_name}/{family}")
            else:
                ids_by_family[family] = (ids, partitions)
            inventory.append({
                "model": model_name, "family": family, "path": str(path),
                "bytes": path.stat().st_size, "mtime": path.stat().st_mtime,
                "rows": len(rows),
                "calibration": sum(row["_partition"] == "calibration" for row in rows),
                "evaluation": sum(row["_partition"] == "evaluation" for row in rows),
                "role": "roster_selection" if (model_name, family) in DEV_CELLS else "retrospective_nonselection",
                "id_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
                "shared_id_split_verified": True,
            })
    return inventory


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    output = []
    for key in sorted(cache, key=str):
        row = cache[key]
        if row.get("align_diag", {}).get("problems"):
            continue
        copied = dict(row)
        copied["_unit"] = str(row.get("id", key))
        copied["_partition"] = stable_partition(copied["_unit"])
        output.append(copied)
    return output


def _split(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict], list[dict]]:
    calibration = [dict(row) for row in rows if row["_partition"] == "calibration"]
    evaluation = [dict(row) for row in rows if row["_partition"] == "evaluation"]
    return calibration, evaluation


def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def _processbench(prediction: Sequence[int], target: Sequence[int]) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=int)
    target = np.asarray(target, dtype=int)
    error, clean = target != -1, target == -1
    acc_error = float(np.mean(prediction[error] == target[error])) if error.any() else float("nan")
    acc_clean = float(np.mean(prediction[clean] == -1)) if clean.any() else float("nan")
    f1 = (
        2.0 * acc_error * acc_clean / (acc_error + acc_clean)
        if np.isfinite(acc_error) and np.isfinite(acc_clean) and acc_error + acc_clean > 0
        else 0.0
    )
    tol1 = float(np.mean(
        (prediction[error] != -1) & (np.abs(prediction[error] - target[error]) <= 1)
    )) if error.any() else float("nan")
    return {
        "f1": f1,
        "exact_error": acc_error,
        "clean_abstention": acc_clean,
        "within_one": tol1,
    }


def _best_threshold(
    risk: Sequence[float], locator: Sequence[int], labels: Sequence[int]
) -> tuple[float, float]:
    risk = np.asarray(risk, dtype=float)
    locator = np.asarray(locator, dtype=int)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-risk, kind="mergesort")
    r, p, y = risk[order], locator[order], labels[order]
    n_error, n_clean = max(int(np.sum(y != -1)), 1), max(int(np.sum(y == -1)), 1)
    hit = ((y != -1) & (p == y)).astype(int)
    clean_flag = (y == -1).astype(int)
    cum_hit = np.r_[0, np.cumsum(hit)]
    cum_clean_flag = np.r_[0, np.cumsum(clean_flag)]
    acc_error = cum_hit / n_error
    acc_clean = (n_clean - cum_clean_flag) / n_clean
    quality = np.divide(
        2 * acc_error * acc_clean,
        acc_error + acc_clean,
        out=np.zeros_like(acc_error, dtype=float),
        where=(acc_error + acc_clean) > 0,
    )
    count = int(np.flatnonzero(quality == np.max(quality))[0])
    if count == 0:
        threshold = float("inf")
    elif count == len(r):
        threshold = float("-inf")
    else:
        threshold = float((r[count - 1] + r[count]) / 2.0)
    return threshold, float(quality[count])


def _token_to_step(token: int, row: Mapping[str, Any]) -> int:
    spans = row.get("step_token_spans") or ()
    for index, span in enumerate(spans):
        if span is not None and int(span[0]) <= int(token) < int(span[1]):
            return int(index)
    return int(len(spans) - 1) if spans else -1


def _peak_locator(curve: np.ndarray, row: Mapping[str, Any]) -> int:
    curve = np.asarray(curve, dtype=float)
    return _token_to_step(int(np.nanargmax(curve)), row) if np.isfinite(curve).any() else -1


def _curve_reference(curves: Sequence[np.ndarray]) -> tuple[float, float, float]:
    sampled = []
    for curve in curves:
        curve = np.asarray(curve, dtype=float)
        sampled.append(curve[equal_positions(len(curve), FIT_POSITIONS)])
    values = np.concatenate(sampled)
    mean, std = float(np.mean(values)), max(float(np.std(values)), 1e-12)
    threshold = float(np.quantile((values - mean) / std, 0.90))
    return mean, std, threshold


def _persistent_locator(
    curve: np.ndarray,
    row: Mapping[str, Any],
    reference: tuple[float, float, float],
) -> int:
    mean, std, threshold = reference
    standardized = (np.asarray(curve, dtype=float) - mean) / std
    hits = standardized > threshold
    for index in range(2, len(hits)):
        if bool(hits[index - 2:index + 1].all()):
            return _token_to_step(index - 2, row)
    return _peak_locator(curve, row)


def _zfit(values: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    return float(np.mean(values)), max(float(np.std(values)), 1e-12)


def _zapply(values: Sequence[float], fit: tuple[float, float]) -> np.ndarray:
    return (np.asarray(values, dtype=float) - fit[0]) / fit[1]


@dataclass
class RegisteredGlobal:
    names: tuple[str, ...]
    transformer: FixedMixedV2Transformer
    weights: np.ndarray
    diagnostics: dict[str, Any]

    def score(self, row: Mapping[str, Any], budget: int | None = None) -> float:
        values = causal_trace_features(row, budget)
        raw = np.asarray([[values.get(name, np.nan) for name in self.names]])
        confidence = self.transformer.transform(raw)
        return float(-(confidence @ self.weights)[0])


def fit_registered_global(rows: Sequence[Mapping[str, Any]]) -> RegisteredGlobal:
    feature_rows = [causal_trace_features(row, None) for row in rows]
    names, columns, availability = [], [], {}
    for name in CONFIDENCE_FEATURE_SIGNS_V1:
        if name == "trace_length":
            continue
        values = np.asarray([item.get(name, np.nan) for item in feature_rows])
        finite = np.isfinite(values)
        availability[name] = float(np.mean(finite))
        if availability[name] < 0.70 or not finite.any():
            continue
        clean = np.where(finite, values, np.median(values[finite]))
        if np.std(clean) < 1e-8 or np.mean(clean == np.median(clean)) > 0.40:
            continue
        names.append(name)
        columns.append(clean)
    raw = np.column_stack(columns)
    transformer = FixedMixedV2Transformer.fit(raw, names)
    confidence = transformer.training_output
    fitted = upcr_fit(confidence.T, **ORDINARY_FIT)
    weights = np.asarray(fitted.w, dtype=float)
    score, anchor = confidence @ weights, confidence.mean(axis=1)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if np.isfinite(correlation) and correlation < 0:
        weights = -weights
    return RegisteredGlobal(tuple(names), transformer, weights, {
        "labels_seen_during_fit": False,
        "feature_names": names,
        "availability": availability,
        "orientation_correlation": correlation,
        "g2_hat": float(fitted.g2_hat),
    })


def _registered_core_matrix(row: Mapping[str, Any], budget: int | None = None) -> np.ndarray:
    if budget is not None:
        row = truncate_row(row, budget)
    entropy = np.asarray(row["token_entropies"], dtype=float)
    spilled = np.asarray(row.get("token_spilled_energies", entropy), dtype=float)[:len(entropy)]
    if len(spilled) < len(entropy):
        spilled = np.pad(spilled, (0, len(entropy) - len(spilled)), mode="edge")
    return np.column_stack([
        entropy,
        _sw_var_series(entropy),
        _cusum_abs_series(entropy),
        _sw_var_series(spilled),
        _cusum_abs_series(spilled),
    ])


@dataclass
class RegisteredLocal:
    mean: np.ndarray
    std: np.ndarray
    derived: np.ndarray
    weights: np.ndarray
    flipped: bool
    diagnostics: dict[str, Any]

    def curve(self, row: Mapping[str, Any], budget: int | None = None) -> np.ndarray:
        raw = _registered_core_matrix(row, budget)
        standardized = (raw - self.mean) / self.std
        score = (standardized * self.derived) @ self.weights
        return -score if self.flipped else score


def fit_registered_local(rows: Sequence[Mapping[str, Any]]) -> RegisteredLocal:
    sampled = []
    for row in rows:
        matrix = _registered_core_matrix(row)
        sampled.append(matrix[equal_positions(len(matrix), FIT_POSITIONS)])
    raw = np.vstack(sampled)
    mean, std = raw.mean(axis=0), raw.std(axis=0)
    std = np.where(std > 1e-8, std, 1.0)
    standardized = (raw - mean) / std
    first = upcr_fit(standardized.T, **REGISTERED_LOCAL_FIT)
    derived = np.sign(first.rho_hat_full)
    derived[derived == 0] = 1.0
    oriented = standardized * derived
    fitted = upcr_fit(oriented.T, **REGISTERED_LOCAL_FIT)
    weights = np.asarray(fitted.w, dtype=float)
    score = oriented @ weights
    correlation = float(np.corrcoef(score, standardized[:, 0])[0, 1])
    flipped = bool(np.isfinite(correlation) and correlation < 0)
    return RegisteredLocal(mean, std, derived, weights, flipped, {
        "labels_seen_during_fit": False,
        "n_fit_rows": int(len(raw)),
        "historical_core_replay": True,
        "full_trace_cusum_curve_is_not_suffix_invariant": True,
        "orientation_correlation": correlation,
    })


@dataclass
class CellModels:
    reference: ChannelReference
    global_heads: dict[str, Any]
    local_heads: dict[str, Any]
    online_heads: dict[str, Any]
    efficiency: list[dict[str, Any]]


def fit_cell_models(
    calibration: Sequence[Mapping[str, Any]],
    *,
    global_names: Sequence[str],
    local_names: Sequence[str],
    online_names: Sequence[str],
) -> CellModels:
    efficiency = []
    started = time.perf_counter()
    reference = fit_channel_reference(calibration)
    efficiency.append({"component": "primitive_reference", "seconds": time.perf_counter() - started})
    global_heads, local_heads, online_heads = {}, {}, {}
    for name in global_names:
        tracemalloc.start(); started = time.perf_counter()
        model = fit_registered_global(calibration) if name == "g_registered_mixed" else fit_iu_head(calibration, reference, name)
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        global_heads[name] = model
        efficiency.append({"component": name, "seconds": time.perf_counter() - started, "python_peak_bytes": peak})
    for name in local_names:
        tracemalloc.start(); started = time.perf_counter()
        model = fit_registered_local(calibration) if name == "l_registered_core5" else fit_iu_head(calibration, reference, name)
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        local_heads[name] = model
        efficiency.append({"component": name, "seconds": time.perf_counter() - started, "python_peak_bytes": peak})
    for name in online_names:
        tracemalloc.start(); started = time.perf_counter()
        model = fit_frozen_prefix_iu(calibration, include_elapsed_length=False) if name == "o_iu28_registered" else fit_iu_head(calibration, reference, name)
        _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
        online_heads[name] = model
        efficiency.append({"component": name, "seconds": time.perf_counter() - started, "python_peak_bytes": peak})
    return CellModels(reference, global_heads, local_heads, online_heads, efficiency)


def _global_score(model: Any, reference: ChannelReference, row: Mapping[str, Any], budget: int | None = None) -> float:
    if isinstance(model, FrozenIUHead):
        return model.score_global(row, reference, upto=budget)
    return model.score(row, budget)


def _local_curve(model: Any, reference: ChannelReference, row: Mapping[str, Any], budget: int | None = None) -> np.ndarray:
    if isinstance(model, FrozenIUHead):
        curve = model.score_curve(row if budget is None else truncate_row(row, budget), reference)
    else:
        curve = model.curve(row, budget)
    return np.asarray(curve, dtype=float)


def _online_curve(model: Any, reference: ChannelReference, row: Mapping[str, Any]) -> np.ndarray:
    if isinstance(model, FrozenIUHead):
        return model.score_curve(row, reference)
    raw, _ = causal_raw_prefix_matrix(row, None, include_elapsed_length=False)
    return np.asarray(model.risk(raw), dtype=float)


def _online_score(model: Any, reference: ChannelReference, row: Mapping[str, Any], budget: int | None) -> float:
    if isinstance(model, FrozenIUHead):
        curve = model.score_curve(row if budget is None else truncate_row(row, budget), reference)
        return float(curve[-1])
    raw, _ = causal_raw_prefix_matrix(row, budget, include_elapsed_length=False)
    return float(np.max(model.risk(raw)))


def score_head_screen(
    model_name: str,
    family: str,
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    models: CellModels,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Return per-question rows, metrics, and frozen score objects for selection."""

    records, metrics = [], []
    frozen: dict[str, Any] = {"global": {}, "local": {}, "online": {}}
    for name, model in models.global_heads.items():
        by_split = {}
        for split_name, rows in (("calibration", calibration), ("evaluation", evaluation)):
            scores = np.asarray([_global_score(model, models.reference, row) for row in rows])
            labels = np.asarray([int(not bool(row["final_answer_correct"])) for row in rows])
            by_split[split_name] = (rows, labels, scores)
            records.extend({
                "model": model_name, "family": family, "split": split_name,
                "unit": row["_unit"], "task": "global", "candidate": name,
                "budget": "final", "target": int(target), "score": float(score),
            } for row, target, score in zip(rows, labels, scores))
        rows, labels, scores = by_split["evaluation"]
        metrics.append({
            "model": model_name, "family": family, "task": "global", "candidate": name,
            "primary": _safe_auc(labels, scores), "auroc": _safe_auc(labels, scores),
            "auprc": _safe_ap(labels, scores), "n": len(labels),
        })
        frozen["global"][name] = by_split

    for name, model in models.local_heads.items():
        by_split = {}
        for split_name, rows in (("calibration", calibration), ("evaluation", evaluation)):
            curves = [_local_curve(model, models.reference, row) for row in rows]
            locators = np.asarray([_peak_locator(curve, row) for curve, row in zip(curves, rows)])
            detectors = np.asarray([float(np.max(curve)) for curve in curves])
            labels = np.asarray([int(row["label"]) for row in rows])
            by_split[split_name] = (rows, labels, curves, detectors, locators)
        _, cal_labels, _, cal_detector, cal_locator = by_split["calibration"]
        threshold, calibration_f1 = _best_threshold(cal_detector, cal_locator, cal_labels)
        rows, labels, curves, detectors, locators = by_split["evaluation"]
        prediction = np.where(detectors > threshold, locators, -1)
        scored = _processbench(prediction, labels)
        records.extend({
            "model": model_name, "family": family, "split": "evaluation",
            "unit": row["_unit"], "task": "local", "candidate": name,
            "budget": "final", "target": int(target), "score": float(detector),
            "locator": int(locator), "prediction": int(pred),
        } for row, target, detector, locator, pred in zip(rows, labels, detectors, locators, prediction))
        metrics.append({
            "model": model_name, "family": family, "task": "local", "candidate": name,
            "primary": scored["f1"], **scored, "threshold": threshold,
            "calibration_f1": calibration_f1, "n": len(labels),
        })
        frozen["local"][name] = {**by_split, "threshold": threshold, "prediction": prediction}

    for name, model in models.online_heads.items():
        by_split = {}
        for split_name, rows in (("calibration", calibration), ("evaluation", evaluation)):
            scores_by_budget = {}
            for budget in BUDGETS:
                eligible = [row for row in rows if len(row["token_entropies"]) > budget]
                scores = np.asarray([_online_score(model, models.reference, row, budget) for row in eligible])
                labels = np.asarray([int(not bool(row["final_answer_correct"])) for row in eligible])
                scores_by_budget[budget] = (eligible, labels, scores)
                records.extend({
                    "model": model_name, "family": family, "split": split_name,
                    "unit": row["_unit"], "task": "online", "candidate": name,
                    "budget": budget, "target": int(target), "score": float(score),
                } for row, target, score in zip(eligible, labels, scores))
            by_split[split_name] = scores_by_budget
        primary_values = []
        for budget in BUDGETS:
            _, labels, scores = by_split["evaluation"][budget]
            auc, ap = _safe_auc(labels, scores), _safe_ap(labels, scores)
            metrics.append({
                "model": model_name, "family": family, "task": "online", "candidate": name,
                "budget": budget, "primary": auc if budget in (64, 128) else float("nan"),
                "auroc": auc, "auprc": ap, "n": len(labels),
            })
            if budget in (64, 128) and np.isfinite(auc):
                primary_values.append(auc)
        metrics.append({
            "model": model_name, "family": family, "task": "online_primary", "candidate": name,
            "budget": "64_128", "primary": float(np.mean(primary_values)), "n": 0,
        })
        frozen["online"][name] = by_split
    return records, metrics, frozen


def _aggregate_head_metrics(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for task, candidates in (
        ("global", GLOBAL_CANDIDATES), ("local", LOCAL_CANDIDATES),
        ("online_primary", ONLINE_CANDIDATES),
    ):
        for candidate in candidates:
            values = [
                float(row["primary"]) for row in metrics
                if row["task"] == task and row["candidate"] == candidate
                and (row["model"], row["family"]) in DEV_CELLS
                and np.isfinite(float(row["primary"]))
            ]
            output.append({
                "task": task.removesuffix("_primary"), "candidate": candidate,
                "primary": float(np.mean(values)) if values else float("nan"),
                "dev_cells": len(values), "features": HEAD_COST[candidate],
                "state_scalars": STATE_COST[candidate],
            })
    return output


def _paired_head_interval(
    records: Sequence[Mapping[str, Any]], task: str, candidate: str, reference: str
) -> tuple[float, float]:
    """Paired question bootstrap across the two frozen selection cells."""

    if candidate == reference:
        return 0.0, 0.0
    cells = sorted(DEV_CELLS)
    prepared = []
    for model_name, family in cells:
        rows = [
            row for row in records
            if row["model"] == model_name and row["family"] == family
            and row["task"] == task and row["split"] == "evaluation"
            and row["candidate"] in {candidate, reference}
            and (task != "online" or int(row["budget"]) in (64, 128))
        ]
        units = sorted({row["unit"] for row in rows})
        by_method = {
            method: {(row["unit"], str(row["budget"])): row for row in rows if row["candidate"] == method}
            for method in (candidate, reference)
        }
        prepared.append((units, by_method))

    def metric(method: str, units: Sequence[str], lookup: Mapping[tuple[str, str], Mapping[str, Any]]) -> float:
        if task == "global":
            rows = [lookup[(unit, "final")] for unit in units]
            return _safe_auc([row["target"] for row in rows], [row["score"] for row in rows])
        if task == "local":
            rows = [lookup[(unit, "final")] for unit in units]
            return _processbench([row["prediction"] for row in rows], [row["target"] for row in rows])["f1"]
        values = []
        for budget in (64, 128):
            selected = [lookup[(unit, str(budget))] for unit in units if (unit, str(budget)) in lookup]
            auc = _safe_auc([row["target"] for row in selected], [row["score"] for row in selected])
            if np.isfinite(auc):
                values.append(auc)
        return float(np.mean(values)) if values else float("nan")

    rng = np.random.default_rng(SEED + sum(ord(char) for char in candidate + reference + task))
    draws = []
    for _ in range(BOOTSTRAP):
        deltas = []
        for units, by_method in prepared:
            sampled = [units[index] for index in rng.integers(0, len(units), len(units))]
            left = metric(candidate, sampled, by_method[candidate])
            right = metric(reference, sampled, by_method[reference])
            if np.isfinite(left) and np.isfinite(right):
                deltas.append(left - right)
        if deltas:
            draws.append(float(np.mean(deltas)))
    if not draws:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def _select_heads(
    aggregate: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    selected, ledger = {}, []
    for task in ("global", "local", "online"):
        rows = [row for row in aggregate if row["task"] == task and np.isfinite(row["primary"])]
        best = max(rows, key=lambda row: row["primary"])
        enriched = []
        for row in rows:
            ci_low, ci_high = _paired_head_interval(
                records, task, row["candidate"], best["candidate"]
            )
            enriched.append({**row, "delta_vs_best": row["primary"] - best["primary"], "ci_low": ci_low, "ci_high": ci_high})
        eligible = [
            row for row in enriched
            if row["primary"] >= best["primary"] - 0.005
            and row["ci_low"] <= 0.0 <= row["ci_high"]
        ]
        chosen = min(eligible, key=lambda row: (row["features"], row["state_scalars"], row["candidate"]))
        selected[task] = chosen["candidate"]
        for row in enriched:
            ledger.append({
                **row, "best": best["candidate"], "within_0p005": row in eligible,
                "selected": row["candidate"] == chosen["candidate"],
            })
    return {"selected": selected, "ledger": ledger, "rule": "lowest cost within 0.005 of dev best when paired 95% interval includes zero"}


def _fit_selected_cell(
    calibration: Sequence[Mapping[str, Any]], selection: Mapping[str, str]
) -> CellModels:
    return fit_cell_models(
        calibration,
        global_names=(selection["global"],),
        local_names=(selection["local"],),
        online_names=(selection["online"],),
    )


def _selected_outputs(
    rows: Sequence[Mapping[str, Any]], models: CellModels, selection: Mapping[str, str]
) -> dict[str, Any]:
    global_model = models.global_heads[selection["global"]]
    local_model = models.local_heads[selection["local"]]
    online_model = models.online_heads[selection["online"]]
    global_final = np.asarray([_global_score(global_model, models.reference, row) for row in rows])
    local_curves = [_local_curve(local_model, models.reference, row) for row in rows]
    online_curves = [_online_curve(online_model, models.reference, row) for row in rows]
    global_prefix, online_prefix, local_prefix = {}, {}, {}
    for budget in BUDGETS:
        global_prefix[budget] = np.asarray([
            _global_score(global_model, models.reference, row, budget)
            if len(row["token_entropies"]) > budget else np.nan for row in rows
        ])
        online_prefix[budget] = np.asarray([
            _online_score(online_model, models.reference, row, budget)
            if len(row["token_entropies"]) > budget else np.nan for row in rows
        ])
        local_prefix[budget] = np.asarray([
            float(np.max(_local_curve(local_model, models.reference, row, budget)))
            if len(row["token_entropies"]) > budget else np.nan for row in rows
        ])
    return {
        "global_final": global_final,
        "local_curves": local_curves,
        "online_curves": online_curves,
        "global_prefix": global_prefix,
        "online_prefix": online_prefix,
        "local_prefix": local_prefix,
    }


def _architecture_configs() -> list[dict[str, Any]]:
    output = []
    for locator in ("peak", "persistent_onset"):
        output.append({"architecture": "a_one_shared", "weight": 1.0, "locator": locator})
    for architecture in ("a_two_global_local", "a_three_independent"):
        for weight in (0.0, 0.25, 0.50, 0.75, 1.0):
            for locator in ("peak", "persistent_onset"):
                output.append({"architecture": architecture, "weight": weight, "locator": locator})
    return output


def _architecture_cell(
    model_name: str,
    family: str,
    calibration: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
    cal_output: Mapping[str, Any],
    eval_output: Mapping[str, Any],
    configs: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records, metrics, frozen = [], [], {}
    cal_global_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in calibration])
    eval_global_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in evaluation])
    cal_local_target = np.asarray([int(row["label"]) for row in calibration])
    eval_local_target = np.asarray([int(row["label"]) for row in evaluation])

    cal_g_fit = _zfit(cal_output["global_final"])
    cal_l_max = np.asarray([float(np.max(curve)) for curve in cal_output["local_curves"]])
    eval_l_max = np.asarray([float(np.max(curve)) for curve in eval_output["local_curves"]])
    cal_l_fit = _zfit(cal_l_max)
    cal_o_fit = _zfit([float(curve[-1]) for curve in cal_output["online_curves"]])

    for config in configs:
        architecture = config["architecture"]
        weight = float(config["weight"])
        locator_name = config["locator"]
        key = f"{architecture}__w{weight:.2f}__{locator_name}"

        if architecture == "a_one_shared":
            cal_global = np.asarray([float(curve[-1]) for curve in cal_output["online_curves"]])
            eval_global = np.asarray([float(curve[-1]) for curve in eval_output["online_curves"]])
            cal_curves, eval_curves = cal_output["online_curves"], eval_output["online_curves"]
            cal_detector, eval_detector = cal_global, eval_global
        else:
            cal_global, eval_global = cal_output["global_final"], eval_output["global_final"]
            cal_curves, eval_curves = cal_output["local_curves"], eval_output["local_curves"]
            cal_detector = weight * _zapply(cal_global, cal_g_fit) + (1.0 - weight) * _zapply(cal_l_max, cal_l_fit)
            eval_detector = weight * _zapply(eval_global, cal_g_fit) + (1.0 - weight) * _zapply(eval_l_max, cal_l_fit)

        curve_reference = _curve_reference(cal_curves)
        locator_fn = (
            (lambda curve, row: _peak_locator(curve, row))
            if locator_name == "peak" else
            (lambda curve, row: _persistent_locator(curve, row, curve_reference))
        )
        cal_locator = np.asarray([locator_fn(curve, row) for curve, row in zip(cal_curves, calibration)])
        eval_locator = np.asarray([locator_fn(curve, row) for curve, row in zip(eval_curves, evaluation)])
        threshold, calibration_f1 = _best_threshold(cal_detector, cal_locator, cal_local_target)
        prediction = np.where(eval_detector > threshold, eval_locator, -1)
        local_metric = _processbench(prediction, eval_local_target)
        global_auc = _safe_auc(eval_global_target, eval_global)

        budget_metrics = []
        online_rows = []
        for budget in BUDGETS:
            eligible = np.asarray([len(row["token_entropies"]) > budget for row in evaluation])
            if architecture == "a_one_shared" or architecture == "a_three_independent":
                score = eval_output["online_prefix"][budget]
            else:
                score = (
                    weight * _zapply(eval_output["global_prefix"][budget], cal_g_fit)
                    + (1.0 - weight) * _zapply(eval_output["local_prefix"][budget], cal_l_fit)
                )
            labels, selected_score = eval_global_target[eligible], np.asarray(score)[eligible]
            auc, ap = _safe_auc(labels, selected_score), _safe_ap(labels, selected_score)
            budget_metrics.append((budget, auc, ap, int(eligible.sum())))
            online_rows.extend({
                "model": model_name, "family": family, "unit": row["_unit"],
                "architecture": key, "task": "online", "budget": budget,
                "target": int(target), "score": float(value),
            } for row, target, value in zip(np.asarray(evaluation, dtype=object)[eligible], labels, selected_score))
        online_primary = float(np.mean([auc for budget, auc, _, _ in budget_metrics if budget in (64, 128) and np.isfinite(auc)]))

        metrics.extend([
            {"model": model_name, "family": family, "architecture": key, "base_architecture": architecture, "weight": weight, "locator": locator_name, "task": "global", "primary": global_auc, "auroc": global_auc, "auprc": _safe_ap(eval_global_target, eval_global), "n": len(evaluation)},
            {"model": model_name, "family": family, "architecture": key, "base_architecture": architecture, "weight": weight, "locator": locator_name, "task": "local", "primary": local_metric["f1"], **local_metric, "threshold": threshold, "calibration_f1": calibration_f1, "n": len(evaluation)},
            {"model": model_name, "family": family, "architecture": key, "base_architecture": architecture, "weight": weight, "locator": locator_name, "task": "online", "primary": online_primary, "n": 0},
        ])
        metrics.extend({
            "model": model_name, "family": family, "architecture": key, "base_architecture": architecture,
            "weight": weight, "locator": locator_name, "task": "online_budget", "budget": budget,
            "primary": auc, "auroc": auc, "auprc": ap, "n": n,
        } for budget, auc, ap, n in budget_metrics)
        records.extend({
            "model": model_name, "family": family, "unit": row["_unit"],
            "architecture": key, "task": "global", "budget": "final",
            "target": int(target), "score": float(score),
        } for row, target, score in zip(evaluation, eval_global_target, eval_global))
        records.extend({
            "model": model_name, "family": family, "unit": row["_unit"],
            "architecture": key, "task": "local", "budget": "final",
            "target": int(target), "score": float(score), "locator": int(locator),
            "prediction": int(pred),
        } for row, target, score, locator, pred in zip(evaluation, eval_local_target, eval_detector, eval_locator, prediction))
        records.extend(online_rows)
        frozen[key] = {
            "threshold": threshold,
            "curve_reference": curve_reference,
            "score_hash": _score_hash(np.r_[eval_global, eval_detector, prediction.astype(float)]),
        }
    return records, metrics, frozen


def _aggregate_architecture(metrics: Sequence[Mapping[str, Any]], *, dev_only: bool) -> list[dict[str, Any]]:
    keys = sorted({row["architecture"] for row in metrics if row["task"] in {"global", "local", "online"}})
    output = []
    for key in keys:
        example = next(row for row in metrics if row["architecture"] == key)
        item = {
            "architecture": key,
            "base_architecture": example["base_architecture"],
            "weight": example["weight"],
            "locator": example["locator"],
        }
        for task in ("global", "local", "online"):
            values = [
                float(row["primary"]) for row in metrics
                if row["architecture"] == key and row["task"] == task
                and (not dev_only or (row["model"], row["family"]) in DEV_CELLS)
                and np.isfinite(float(row["primary"]))
            ]
            item[task] = float(np.mean(values)) if values else float("nan")
        output.append(item)
    return output


def _architecture_complexity(name: str) -> int:
    if name == "a_one_shared":
        return 1
    if name == "a_two_global_local":
        return 2
    return 3


def _paired_architecture_interval(
    records: Sequence[Mapping[str, Any]], task: str, candidate: str, reference: str
) -> tuple[float, float]:
    if candidate == reference:
        return 0.0, 0.0
    prepared = []
    for model_name, family in sorted(DEV_CELLS):
        rows = [
            row for row in records
            if row["model"] == model_name and row["family"] == family
            and row["task"] == task and row["architecture"] in {candidate, reference}
            and (task != "online" or int(row["budget"]) in (64, 128))
        ]
        units = sorted({row["unit"] for row in rows})
        by_method = {
            method: {(row["unit"], str(row["budget"])): row for row in rows if row["architecture"] == method}
            for method in (candidate, reference)
        }
        prepared.append((units, by_method))

    def metric(method: str, units: Sequence[str], lookup: Mapping[tuple[str, str], Mapping[str, Any]]) -> float:
        if task == "global":
            selected = [lookup[(unit, "final")] for unit in units]
            return _safe_auc([row["target"] for row in selected], [row["score"] for row in selected])
        if task == "local":
            selected = [lookup[(unit, "final")] for unit in units]
            return _processbench([row["prediction"] for row in selected], [row["target"] for row in selected])["f1"]
        values = []
        for budget in (64, 128):
            selected = [lookup[(unit, str(budget))] for unit in units if (unit, str(budget)) in lookup]
            auc = _safe_auc([row["target"] for row in selected], [row["score"] for row in selected])
            if np.isfinite(auc):
                values.append(auc)
        return float(np.mean(values)) if values else float("nan")

    rng = np.random.default_rng(SEED + sum(ord(char) for char in candidate + reference + task))
    draws = []
    for _ in range(BOOTSTRAP):
        deltas = []
        for units, by_method in prepared:
            sampled = [units[index] for index in rng.integers(0, len(units), len(units))]
            left = metric(candidate, sampled, by_method[candidate])
            right = metric(reference, sampled, by_method[reference])
            if np.isfinite(left) and np.isfinite(right):
                deltas.append(left - right)
        if deltas:
            draws.append(float(np.mean(deltas)))
    if not draws:
        return float("nan"), float("nan")
    return tuple(float(value) for value in np.quantile(draws, (0.025, 0.975)))


def _select_architectures(
    aggregate: Sequence[Mapping[str, Any]], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    margins = {"global": 0.010, "local": 0.010, "online": 0.015}
    best = {task: max(row[task] for row in aggregate) for task in margins}
    eligible = [
        row for row in aggregate
        if all(row[task] >= best[task] - margins[task] for task in margins)
    ]
    if not eligible:
        # The preregistered independent architecture preserves each selected
        # task head.  If the intersection is empty, retain its strongest Local
        # configuration and report the failed joint margin explicitly.
        pool = [row for row in aggregate if row["base_architecture"] == "a_three_independent"]
        eligible = [max(pool, key=lambda row: row["local"])]
        fallback = "no architecture met all three margins; used strongest three-head Local configuration"
    else:
        fallback = None
    simplest = min(
        eligible,
        key=lambda row: (
            _architecture_complexity(row["base_architecture"]),
            abs(float(row["weight"]) - 0.5),
            row["architecture"],
        ),
    )
    comparisons = []
    materially_better = []
    for row in eligible:
        intervals = {}
        for task in margins:
            low, high = _paired_architecture_interval(
                records, task, row["architecture"], simplest["architecture"]
            )
            intervals[task] = {"low": low, "high": high}
        comparisons.append({"architecture": row["architecture"], "reference": simplest["architecture"], "intervals": intervals})
        if any(intervals[task]["low"] > 0.0 for task in margins):
            materially_better.append(row)
    chosen = min(
        materially_better,
        key=lambda row: (_architecture_complexity(row["base_architecture"]), row["architecture"]),
    ) if materially_better else simplest
    best_by_base = {}
    for base in ("a_one_shared", "a_two_global_local", "a_three_independent"):
        pool = [row for row in aggregate if row["base_architecture"] == base]
        base_best = {task: max(row[task] for row in pool) for task in margins}
        survivors = [row for row in pool if all(row[task] >= base_best[task] - margins[task] for task in margins)]
        if not survivors:
            survivors = [max(pool, key=lambda row: row["local"])]
        best_by_base[base] = min(survivors, key=lambda row: (abs(float(row["weight"]) - 0.5), row["architecture"]))
    return {
        "selected": chosen,
        "best_by_base": best_by_base,
        "panel_bests": best,
        "margins": margins,
        "joint_survivors": [row["architecture"] for row in eligible],
        "simplest_survivor": simplest["architecture"],
        "paired_vs_simplest": comparisons,
        "fallback": fallback,
    }


def main() -> None:
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    OUT.mkdir(parents=True, exist_ok=True)
    print("[preflight] validating twelve telemetry cells and shared-ID splits", flush=True)
    inventory = preflight_inventory()
    _write_csv(OUT / "CACHE_INVENTORY.csv", inventory)
    head_records, head_metrics, efficiency = [], [], []
    dev_payload = {}

    # Stage 1: score the complete frozen head roster only on selection cells.
    for model_name, family in sorted(DEV_CELLS):
        path = _cell_path(model_name, family)
        rows = load_rows(path)
        calibration, evaluation = _split(rows)
        print(f"[heads] {model_name}/{family}: {len(calibration)} cal, {len(evaluation)} eval", flush=True)
        models = fit_cell_models(
            calibration,
            global_names=GLOBAL_CANDIDATES,
            local_names=LOCAL_CANDIDATES,
            online_names=ONLINE_CANDIDATES,
        )
        records, metrics, frozen = score_head_screen(
            model_name, family, calibration, evaluation, models
        )
        head_records.extend(records); head_metrics.extend(metrics)
        efficiency.extend({"model": model_name, "family": family, "stage": "head_screen", **row} for row in models.efficiency)
        dev_payload[(model_name, family)] = (calibration, evaluation, models, frozen)

    head_aggregate = _aggregate_head_metrics(head_metrics)
    head_selection = _select_heads(head_aggregate, head_records)
    _write_csv(OUT / "HEAD_PER_QUESTION.csv", head_records)
    _write_csv(OUT / "HEAD_METRICS.csv", head_metrics)
    _write_csv(OUT / "HEAD_AGGREGATE.csv", head_aggregate)
    _write_json(OUT / "HEAD_SELECTION.json", head_selection)
    print("[selection]", head_selection["selected"], flush=True)

    # Stage 2: refit only the selected heads and cross the frozen architectures.
    architecture_records, architecture_metrics, architecture_frozen = [], [], {}
    configs = _architecture_configs()
    for key, (calibration, evaluation, _, _) in dev_payload.items():
        model_name, family = key
        selected_models = _fit_selected_cell(calibration, head_selection["selected"])
        cal_output = _selected_outputs(calibration, selected_models, head_selection["selected"])
        eval_output = _selected_outputs(evaluation, selected_models, head_selection["selected"])
        records, metrics, frozen = _architecture_cell(
            model_name, family, calibration, evaluation, cal_output, eval_output, configs
        )
        architecture_records.extend(records); architecture_metrics.extend(metrics)
        architecture_frozen[f"{model_name}/{family}"] = frozen
        efficiency.extend({"model": model_name, "family": family, "stage": "architecture", **row} for row in selected_models.efficiency)

    architecture_aggregate = _aggregate_architecture(architecture_metrics, dev_only=True)
    architecture_selection = _select_architectures(architecture_aggregate, architecture_records)
    _write_csv(OUT / "ARCHITECTURE_DEV_PER_QUESTION.csv", architecture_records)
    _write_csv(OUT / "ARCHITECTURE_DEV_METRICS.csv", architecture_metrics)
    _write_csv(OUT / "ARCHITECTURE_DEV_AGGREGATE.csv", architecture_aggregate)
    _write_json(OUT / "ARCHITECTURE_SELECTION.json", architecture_selection)
    _write_json(OUT / "ARCHITECTURE_SCORE_FREEZE.json", architecture_frozen)
    print("[architecture]", architecture_selection["selected"]["architecture"], flush=True)

    # Stage 3: apply only the frozen per-head identities and one frozen config
    # per head-count family to every non-selection cell.
    frozen_configs = [
        {
            "architecture": row["base_architecture"],
            "weight": row["weight"],
            "locator": row["locator"],
        }
        for row in architecture_selection["best_by_base"].values()
    ]
    transfer_records, transfer_metrics = [], []
    for model_name in MODELS:
        for family in FAMILIES:
            if (model_name, family) in DEV_CELLS:
                continue
            path = _cell_path(model_name, family)
            rows = load_rows(path)
            calibration, evaluation = _split(rows)
            print(f"[transfer] {model_name}/{family}: {len(calibration)} cal, {len(evaluation)} eval", flush=True)
            models = _fit_selected_cell(calibration, head_selection["selected"])
            cal_output = _selected_outputs(calibration, models, head_selection["selected"])
            eval_output = _selected_outputs(evaluation, models, head_selection["selected"])
            records, metrics, _ = _architecture_cell(
                model_name, family, calibration, evaluation, cal_output, eval_output, frozen_configs
            )
            transfer_records.extend(records); transfer_metrics.extend(metrics)
            efficiency.extend({"model": model_name, "family": family, "stage": "transfer", **row} for row in models.efficiency)

    all_architecture_records = architecture_records + transfer_records
    all_architecture_metrics = architecture_metrics + transfer_metrics
    all_aggregate = _aggregate_architecture(all_architecture_metrics, dev_only=False)
    _write_csv(OUT / "ARCHITECTURE_PER_QUESTION.csv", all_architecture_records)
    _write_csv(OUT / "ARCHITECTURE_METRICS.csv", all_architecture_metrics)
    _write_csv(OUT / "ARCHITECTURE_AGGREGATE.csv", all_aggregate)
    _write_csv(OUT / "EFFICIENCY.csv", efficiency)
    _write_csv(OUT / "CACHE_INVENTORY.csv", inventory)
    _write_json(OUT / "RUN_MANIFEST.json", {
        "status": "HEAD_ARCHITECTURE_COMPLETE_FUSION_PENDING",
        "seed": SEED,
        "bootstrap": BOOTSTRAP,
        "protocol": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "head_selection_sha256": _sha256(OUT / "HEAD_SELECTION.json"),
        "architecture_selection_sha256": _sha256(OUT / "ARCHITECTURE_SELECTION.json"),
        "candidate_roster": {
            "global": GLOBAL_CANDIDATES, "local": LOCAL_CANDIDATES,
            "online": ONLINE_CANDIDATES, "architectures": configs,
        },
        "labels_seen_during_score_fit": False,
        "all_results_retrospective": True,
        "new_inference": False,
        "gpu_hours": 0,
        "drive_mutation": False,
    })
    print(f"[done] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
