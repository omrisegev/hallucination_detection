"""Shared, label-free scoring and reporting helpers for paper-aligned benchmarks.

This module deliberately has no dataset-specific label loader.  A task adapter
builds a feature matrix, calls :func:`fit_spectral_scores`, freezes the returned
scores, and only then evaluates them.  Keeping this boundary explicit prevents
benchmark labels from leaking into U-PCR, IU-PCR, DUFS gates, or the graph.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .laplacian_upcr import build_graph_from_features, dufs_soft_gates, laplacian_iu_path
from .upcr import upcr_fit


DEPLOYED_FIT = {
    "loss": "l2",
    "exclusion": True,
    "difficulty_gate": False,
    "simple_avg_fallback": True,
    "recompute_after_exclusion": True,
    "g2_projection_k": 1,
    "scale_ratio": 0.25,
}
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
DUFS_LAMBDA = 0.1


@dataclass(frozen=True)
class ProtocolSignature:
    """Fields that must match before two scores may share a headline row."""

    dataset: str
    model: str
    split: str
    prediction_unit: str
    metric: str
    grader: str = ""
    sample_count: int | None = None

    def compatibility_key(self) -> tuple[str, ...]:
        return (
            self.dataset, self.model, self.split, self.prediction_unit,
            self.metric, self.grader,
        )


def assert_protocol_match(left: ProtocolSignature, right: ProtocolSignature) -> None:
    if left.compatibility_key() != right.compatibility_key():
        differing = [
            name for name in ("dataset", "model", "split", "prediction_unit", "metric", "grader")
            if getattr(left, name) != getattr(right, name)
        ]
        raise ValueError("protocol mismatch: " + ", ".join(differing))


def standardize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Median-impute and z-score columns without using labels."""

    X = np.asarray(values, dtype=float)
    if X.ndim != 2 or X.shape[0] < 3 or X.shape[1] < 3:
        raise ValueError("feature matrix must have shape (samples>=3, features>=3)")
    columns, keep, means, scales = [], [], [], []
    for index in range(X.shape[1]):
        raw = X[:, index]
        finite = np.isfinite(raw)
        if not finite.any():
            continue
        median = float(np.median(raw[finite]))
        clean = np.where(finite, raw, median)
        scale = float(clean.std())
        if scale < 1e-8 or float(np.mean(clean == median)) > 0.95:
            continue
        keep.append(index)
        means.append(float(clean.mean()))
        scales.append(scale)
        columns.append((clean - clean.mean()) / scale)
    if len(columns) < 3:
        raise ValueError(f"only {len(columns)} non-degenerate features remain")
    return np.column_stack(columns), np.asarray(keep), np.asarray(means), np.asarray(scales)


def _orient(score: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, bool]:
    score = np.asarray(score, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    corr = np.corrcoef(score, anchor)[0, 1]
    flipped = bool(np.isfinite(corr) and corr < 0)
    return (-score if flipped else score), flipped


def fit_spectral_scores(
    values: np.ndarray,
    *,
    feature_names: Sequence[str],
    risk_anchor: np.ndarray | None = None,
    dufs_seeds: Sequence[int] = DUFS_SEEDS,
    dufs_epochs: int = DUFS_EPOCHS,
    k: int = DUFS_K,
    lambda_: float = DUFS_LAMBDA,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the frozen three-solver progression to one label-free matrix.

    Every input column must already be oriented so that larger means more risk.
    The output orientation is anchored to the mean input risk (or an explicit
    label-free anchor), never to the evaluation labels.
    """

    X, keep, mean, scale = standardize(values)
    kept_names = [str(feature_names[int(index)]) for index in keep]
    F = X.T
    anchor = X.mean(axis=1) if risk_anchor is None else np.asarray(risk_anchor, dtype=float)
    if anchor.shape != (X.shape[0],):
        raise ValueError("risk_anchor must contain one value per sample")

    deployed = upcr_fit(F, **DEPLOYED_FIT)
    gates, gate_diag = dufs_soft_gates(
        F, seeds=tuple(int(seed) for seed in dufs_seeds),
        epochs=int(dufs_epochs), return_history=True,
    )
    graph = build_graph_from_features(F, gates=gates, k=int(k))
    path = laplacian_iu_path(F, (0.0, float(lambda_)), graph=graph)
    if not np.array_equal(path[0.0].w, path[0.0].baseline.w):
        raise AssertionError("lambda=0 must exactly equal IU-PCR")

    raw = {
        "deployed_upcr": deployed.w @ F,
        "iu_pcr": path[0.0].w @ F,
        "dufs_liu_pcr": path[float(lambda_)].w @ F,
    }
    scores, flips = {}, {}
    for method, score in raw.items():
        scores[method], flips[method] = _orient(score, anchor)
        if not np.isfinite(scores[method]).all():
            raise ValueError(f"non-finite score from {method}")
    diagnostics = {
        "labels_seen_during_fit": False,
        "input_feature_names": list(feature_names),
        "kept_feature_names": kept_names,
        "dropped_feature_names": [
            str(name) for index, name in enumerate(feature_names) if index not in set(keep.tolist())
        ],
        "standardization_mean": mean.tolist(),
        "standardization_scale": scale.tolist(),
        "orientation_flipped": flips,
        "dufs": _jsonable(gate_diag),
        "dufs_gates": np.asarray(gates).tolist(),
        "deployed_weights": np.asarray(deployed.w).tolist(),
        "iu_weights": np.asarray(path[0.0].w).tolist(),
        "dufs_liu_weights": np.asarray(path[float(lambda_)].w).tolist(),
        "graph": _jsonable(path[float(lambda_)].diagnostics),
        "lambda_zero_exact": True,
        "k": int(k),
        "lambda": float(lambda_),
        "dufs_seeds": list(dufs_seeds),
        "dufs_epochs": int(dufs_epochs),
    }
    return scores, diagnostics


def fit_spectral_model(
    values: np.ndarray,
    *,
    feature_names: Sequence[str],
    risk_anchor: np.ndarray | None = None,
    dufs_seeds: Sequence[int] = DUFS_SEEDS,
    dufs_epochs: int = DUFS_EPOCHS,
    k: int = DUFS_K,
    lambda_: float = DUFS_LAMBDA,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit on a bounded label-free population and return an applicable model."""

    X, keep, mean, scale = standardize(values)
    F = X.T
    anchor = X.mean(axis=1) if risk_anchor is None else np.asarray(risk_anchor, dtype=float)
    deployed = upcr_fit(F, **DEPLOYED_FIT)
    gates, gate_diag = dufs_soft_gates(
        F, seeds=tuple(int(seed) for seed in dufs_seeds),
        epochs=int(dufs_epochs), return_history=True,
    )
    graph = build_graph_from_features(F, gates=gates, k=int(k))
    path = laplacian_iu_path(F, (0.0, float(lambda_)), graph=graph)
    if not np.array_equal(path[0.0].w, path[0.0].baseline.w):
        raise AssertionError("lambda=0 must exactly equal IU-PCR")
    weights = {
        "deployed_upcr": np.asarray(deployed.w),
        "iu_pcr": np.asarray(path[0.0].w),
        "dufs_liu_pcr": np.asarray(path[float(lambda_)].w),
    }
    flips = {}
    for method, weight in weights.items():
        _, flips[method] = _orient(weight @ F, anchor)
    model = {
        "feature_names": list(feature_names),
        "keep": keep.tolist(),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "weights": {key: value.tolist() for key, value in weights.items()},
        "flips": flips,
    }
    diagnostics = {
        "labels_seen_during_fit": False,
        "kept_feature_names": [str(feature_names[int(index)]) for index in keep],
        "dufs": _jsonable(gate_diag),
        "dufs_gates": np.asarray(gates).tolist(),
        "graph": _jsonable(path[float(lambda_)].diagnostics),
        "lambda_zero_exact": True,
        "k": int(k), "lambda": float(lambda_),
        "dufs_seeds": list(dufs_seeds), "dufs_epochs": int(dufs_epochs),
    }
    return model, diagnostics


def apply_spectral_model(values: np.ndarray, model: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Apply a frozen model; this function has no label argument by design."""

    X = np.asarray(values, dtype=float)
    keep = np.asarray(model["keep"], dtype=int)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    selected = X[:, keep]
    selected = np.where(np.isfinite(selected), selected, mean[None, :])
    F = ((selected - mean[None, :]) / scale[None, :]).T
    output = {}
    for method, raw_weight in model["weights"].items():
        score = np.asarray(raw_weight, dtype=float) @ F
        if model["flips"].get(method):
            score = -score
        output[str(method)] = np.asarray(score, dtype=float)
    return output


def score_hash(score_map: Mapping[str, np.ndarray], ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for method in sorted(score_map):
        digest.update(method.encode("utf-8"))
        digest.update(np.asarray(ids, dtype="U").tobytes())
        digest.update(np.asarray(score_map[method], dtype="<f8").tobytes())
    return digest.hexdigest()


def binary_metrics(labels: Sequence[int | bool], scores: Sequence[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    s = np.asarray(scores, dtype=float)
    if len(y) != len(s) or len(np.unique(y)) != 2:
        raise ValueError("binary metrics require aligned scores and both label classes")
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }


def grouped_bootstrap_binary(
    labels: Sequence[int | bool],
    scores: Mapping[str, Sequence[float]],
    groups: Sequence[str],
    *,
    draws: int = 1000,
    seed: int = 20260811,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Use identical group resamples for every method in a protocol."""

    y = np.asarray(labels, dtype=int)
    g = np.asarray(groups, dtype=str)
    unique = np.unique(g)
    member = {group: np.flatnonzero(g == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = {method: {"auroc": [], "auprc": []} for method in scores}
    arrays = {method: np.asarray(value, dtype=float) for method, value in scores.items()}
    for _ in range(int(draws)):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        index = np.concatenate([member[group] for group in chosen])
        if len(np.unique(y[index])) != 2:
            continue
        for method, score in arrays.items():
            metric = binary_metrics(y[index], score[index])
            for name in values[method]:
                values[method][name].append(metric[name])
    return {
        method: {
            metric: tuple(float(value) for value in np.quantile(sample, (0.025, 0.975)))
            for metric, sample in metric_values.items()
        }
        for method, metric_values in values.items()
    }


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or not path.stat().st_size:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def bar_chart(rows: Sequence[Mapping[str, Any]], *, metric: str, title: str) -> str:
    """Small self-contained horizontal SVG; values are read from result rows."""

    usable = [row for row in rows if row.get("metric") == metric and row.get("value") not in (None, "")]
    if not usable:
        return f"<div class='empty'>No matched {esc(metric)} values are ready.</div>"
    width, left, right, row_h = 760, 230, 60, 34
    height = 52 + row_h * len(usable)
    plot = width - left - right
    maximum = max(1.0, max(float(row["value"]) for row in usable))
    parts = [f"<svg class='chart' viewBox='0 0 {width} {height}' role='img' aria-label='{esc(title)}'>",
             f"<text x='12' y='22' class='chart-title'>{esc(title)}</text>"]
    for index, row in enumerate(usable):
        y = 40 + index * row_h
        value = float(row["value"])
        bar = max(0.0, value / maximum * plot)
        role = str(row.get("role", "ours"))
        css = "ceiling" if "ceiling" in role else ("published" if "published" in role else "ours")
        parts.extend([
            f"<text x='{left-8}' y='{y+16}' text-anchor='end'>{esc(row.get('method',''))}</text>",
            f"<rect x='{left}' y='{y}' width='{bar:.2f}' height='21' rx='4' class='bar-{css}'/>",
            f"<text x='{min(left+bar+7,width-45):.2f}' y='{y+16}'>{value:.3f}</text>",
        ])
    parts.append("</svg>")
    return "".join(parts)


def forbid_cross_task_macro(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail if a generated row tries to average incompatible prediction units."""

    for row in rows:
        if str(row.get("scope", "")).lower() in {"global", "all_tasks", "suite"}:
            raise ValueError("cross-task macro averaging is forbidden")
        subgroup = str(row.get("subgroup", "")).lower()
        protocol_id = str(row.get("protocol_id", ""))
        if ("suite" in subgroup or "cross-task" in subgroup) and "macro" in subgroup:
            raise ValueError("cross-task macro averaging is forbidden")
        if subgroup == "four-subset macro" and protocol_id != "localization-processbench-first-error":
            raise ValueError("four-subset macro is only valid inside ProcessBench")
