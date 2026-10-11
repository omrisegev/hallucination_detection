"""Shared Global-Local-Online IU regression utilities.

The module implements the bounded dynamic Online-head roster frozen in
``docs/experiments/GLOBAL_LOCAL_ONLINE_IU_V1.md``.  It consumes long-form,
causally generated CUSUM and sliding-window-variance trajectories.  Fitting is
label-blind by construction: target fields may be present in input records for
later evaluation, but no fit routine reads them.

All public scores are risk-oriented (larger means more likely error).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .online_convergence import IU_FIT
from .upcr import upcr_fit


EPS = 1e-12
COMPONENTS = ("cusum_max", "sw_var_peak")
CANDIDATE_FEATURES = {
    "dyn_level4_iu": (
        "cusum_current", "cusum_running_max",
        "swvar_current", "swvar_running_max",
    ),
    "dyn_persist6_iu": (
        "cusum_current", "cusum_positive_area", "cusum_run_fraction",
        "swvar_current", "swvar_positive_area", "swvar_run_fraction",
    ),
    "dyn_change6_iu": (
        "cusum_current", "cusum_slope", "cusum_recovery",
        "swvar_current", "swvar_slope", "swvar_recovery",
    ),
}


def _finite_float(value: Any) -> float | None:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return None
    return output if np.isfinite(output) else None


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _trace_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row.get("group", "")), int(row["unit_index"])


def _component_points(
    score_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Join the two registered component scores at each causal monitor point."""

    joined: dict[tuple[str, int, int, bool], dict[str, Any]] = {}
    for row in score_rows:
        method = str(row.get("method", ""))
        if method not in COMPONENTS:
            continue
        score = _finite_float(row.get("score"))
        if score is None:
            continue
        key = (
            *_trace_key(row),
            int(row["budget"]),
            _bool(row.get("is_final", False)),
        )
        point = joined.setdefault(key, {
            "group": key[0],
            "unit_index": key[1],
            "budget": key[2],
            "is_final": key[3],
        })
        point[method] = score

    output: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for key, point in joined.items():
        # Missing families are explicit and deterministic: their standardized
        # value is the label-free reference centre (zero after scaling).
        for component in COMPONENTS:
            point.setdefault(component, None)
        output.setdefault((key[0], key[1]), []).append(point)
    for points in output.values():
        points.sort(key=lambda point: (int(point["budget"]), bool(point["is_final"])))
    return output


def _robust_reference(values: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0.0, 1.0
    centre = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(values))
    return centre, max(scale, 1e-8)


def _dynamic_rows(
    points_by_trace: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    centres: Mapping[str, float],
    scales: Mapping[str, float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for trace_key in sorted(points_by_trace):
        state = {
            component: {
                "running_max": -np.inf,
                "positive_sum": 0.0,
                "run": 0,
                "previous": None,
                "previous_budget": None,
            }
            for component in COMPONENTS
        }
        for step_index, point in enumerate(points_by_trace[trace_key], start=1):
            features: dict[str, float] = {}
            for component, prefix in (("cusum_max", "cusum"), ("sw_var_peak", "swvar")):
                value = point.get(component)
                z = 0.0 if value is None else (
                    (float(value) - float(centres[component])) / float(scales[component])
                )
                item = state[component]
                previous = item["previous"]
                previous_budget = item["previous_budget"]
                item["running_max"] = max(float(item["running_max"]), z)
                item["positive_sum"] = float(item["positive_sum"]) + max(z, 0.0)
                item["run"] = int(item["run"]) + 1 if z > 0.0 else 0
                if previous is None or previous_budget is None:
                    slope = 0.0
                else:
                    elapsed = max(1, int(point["budget"]) - int(previous_budget))
                    slope = (z - float(previous)) * (64.0 / elapsed)
                features[f"{prefix}_current"] = z
                features[f"{prefix}_running_max"] = float(item["running_max"])
                features[f"{prefix}_positive_area"] = float(item["positive_sum"]) / step_index
                features[f"{prefix}_run_fraction"] = int(item["run"]) / step_index
                features[f"{prefix}_slope"] = slope
                features[f"{prefix}_recovery"] = z - float(item["running_max"])
                item["previous"] = z
                item["previous_budget"] = int(point["budget"])
            output.append({**point, "features": features})
    return output


def _equal_trace_sample(
    dynamic_rows: Sequence[Mapping[str, Any]], rows_per_trace: int,
) -> list[Mapping[str, Any]]:
    by_trace: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in dynamic_rows:
        by_trace.setdefault(_trace_key(row), []).append(row)
    sampled: list[Mapping[str, Any]] = []
    for key in sorted(by_trace):
        rows = by_trace[key]
        indexes = np.linspace(0, len(rows) - 1, int(rows_per_trace), dtype=int)
        sampled.extend(rows[index] for index in indexes)
    return sampled


@dataclass(frozen=True)
class FrozenDynamicOnlineHead:
    """Label-blind ordinary-IU head over a fixed dynamic coordinate roster."""

    name: str
    feature_names: tuple[str, ...]
    component_centres: Mapping[str, float]
    component_scales: Mapping[str, float]
    feature_keep: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    weights: np.ndarray
    diagnostics: Mapping[str, Any]

    def score_rows(
        self, score_rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        points = _component_points(score_rows)
        dynamic = _dynamic_rows(points, self.component_centres, self.component_scales)
        source_lookup = {}
        for source_row in score_rows:
            if str(source_row.get("method")) not in COMPONENTS:
                continue
            source_key = (
                *_trace_key(source_row), int(source_row["budget"]),
                _bool(source_row.get("is_final", False)),
            )
            source_lookup.setdefault(source_key, source_row)
        output = []
        for row in dynamic:
            raw = np.asarray([row["features"][name] for name in self.feature_names], dtype=float)
            selected = raw[self.feature_keep]
            standardized = (selected - self.feature_mean) / self.feature_std
            score = float(standardized @ self.weights)
            key = (
                *_trace_key(row), int(row["budget"]), bool(row["is_final"])
            )
            source = source_lookup[key]
            # Metadata and labels are copied only after the score has been
            # computed. They never enter the feature path above.
            output.append({
                "unit_index": int(source["unit_index"]),
                "trace_id": str(source.get("trace_id", source["unit_index"])),
                "group": str(source.get("group", "")),
                "label_error": int(source.get("label_error", 0)),
                "trace_length": int(source.get("trace_length", source["budget"])),
                "budget": int(source["budget"]),
                "is_final": _bool(source.get("is_final", False)),
                "method": self.name,
                "score": score,
            })
        return output


def fit_dynamic_online_head(
    score_rows: Sequence[Mapping[str, Any]],
    candidate: str,
    *,
    rows_per_trace: int = 6,
) -> FrozenDynamicOnlineHead:
    """Fit one frozen candidate without reading correctness labels."""

    if candidate not in CANDIDATE_FEATURES:
        raise KeyError(f"unregistered dynamic candidate: {candidate}")
    points = _component_points(score_rows)
    values = {component: [] for component in COMPONENTS}
    for trace_points in points.values():
        for point in trace_points:
            for component in COMPONENTS:
                value = point.get(component)
                if value is not None:
                    values[component].append(float(value))
    references = {component: _robust_reference(values[component]) for component in COMPONENTS}
    centres = {component: pair[0] for component, pair in references.items()}
    scales = {component: pair[1] for component, pair in references.items()}
    dynamic = _dynamic_rows(points, centres, scales)
    sampled = _equal_trace_sample(dynamic, rows_per_trace)
    feature_names = tuple(CANDIDATE_FEATURES[candidate])
    raw = np.asarray([
        [row["features"][name] for name in feature_names] for row in sampled
    ], dtype=float)
    mean_all = raw.mean(axis=0)
    std_all = raw.std(axis=0)
    keep = np.isfinite(mean_all) & np.isfinite(std_all) & (std_all > 1e-8)
    if int(keep.sum()) < 3:
        raise ValueError(f"{candidate} has fewer than three non-degenerate coordinates")
    mean = mean_all[keep]
    std = std_all[keep]
    standardized = (raw[:, keep] - mean[None, :]) / std[None, :]
    fitted = upcr_fit(standardized.T, **IU_FIT)
    weights = np.asarray(fitted.w, dtype=float)
    score = standardized @ weights
    anchor = standardized.mean(axis=1)
    corr = (
        float(np.corrcoef(score, anchor)[0, 1])
        if score.std() > EPS and anchor.std() > EPS else float("nan")
    )
    flipped = bool(np.isfinite(corr) and corr < 0)
    if flipped:
        weights = -weights
    kept_names = [name for name, use in zip(feature_names, keep) if use]
    diagnostics = {
        "candidate": candidate,
        "feature_names": list(feature_names),
        "kept_feature_names": kept_names,
        "dropped_feature_names": [name for name, use in zip(feature_names, keep) if not use],
        "feature_count": int(keep.sum()),
        "rows_per_trace": int(rows_per_trace),
        "fit_trace_count": int(len(points)),
        "fit_row_count": int(len(sampled)),
        "component_centres": centres,
        "component_scales": scales,
        "orientation_correlation": corr,
        "orientation_flipped": flipped,
        "labels_seen_during_fit": False,
        "lambda": 0.0,
        "iu_fit": dict(IU_FIT),
        "g2_hat": float(fitted.g2_hat),
        "projection_residual": float(fitted.proj_residual),
        "persistent_state_scalars_per_trace": {
            "dyn_level4_iu": 6,
            "dyn_persist6_iu": 10,
            "dyn_change6_iu": 8,
        }[candidate],
        "update_complexity": "O(1) per monitor observation",
    }
    return FrozenDynamicOnlineHead(
        candidate, feature_names, centres, scales, keep, mean, std, weights, diagnostics
    )


@dataclass(frozen=True)
class GlobalLocalOnlineOutput:
    """One-method interface required by the cross-task regression harness."""

    global_risk: float
    token_risk: np.ndarray
    first_onset_token: int | None
    prefix_trajectory: tuple[tuple[int, float], ...]
    declaration_trajectory: tuple[tuple[int, int | None], ...]
    runtime_seconds: float
    peak_memory_bytes: int
    feature_count: int
    persistent_state_scalars: int


__all__ = [
    "CANDIDATE_FEATURES",
    "COMPONENTS",
    "FrozenDynamicOnlineHead",
    "GlobalLocalOnlineOutput",
    "fit_dynamic_online_head",
]
