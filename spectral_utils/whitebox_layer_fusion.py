"""Label-free feature contracts and fusion helpers for white-box layer views.

The module deliberately separates three phases:

``validate_and_join``
    Audits a raw replication-grid cache against a ``layer-lens-v1`` sidecar and
    returns a :class:`LayerCell` whose records no longer contain correctness
    labels.

``extract_*``
    Builds a :class:`FeatureMatrix` from inner-state tensors only.  Every
    feature is oriented so that a larger value has the declared *risk*
    interpretation, and the sole global orientation anchor is final-residual
    target-token NLL.

``fit_*``
    Fits label-free controls and the repository's maintained U-PCR, IU-PCR,
    DUFS-LIU-PCR, dependency-aware, and matched hierarchical solvers.  These
    functions intentionally expose no label-like argument.  Labels should be
    opened only after scores have been frozen, via
    :func:`load_evaluation_labels` or another evaluator-owned loader.

Arrays use the public convention ``samples x features`` in
:class:`FeatureMatrix`; the underlying spectral solvers receive the transposed
``features x samples`` matrix they expect.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .dependency_fusion import sparse_upcr_fit
from .fusion_utils import lsml_continuous
from .laplacian_upcr import (
    IU_FIT_DEFAULTS,
    build_graph_from_features,
    dufs_soft_gates,
    laplacian_iu_path,
)
from .paper_benchmark_suite import (
    fit_spectral_scores as canonical_fit_spectral_scores,
    standardize as canonical_standardize,
)
from .upcr import upcr_fit
from .upcr_clustered import upcr_clustered_fit


ALL_LAYERS = tuple(range(32))
SPACED_LAYERS = (0, 4, 9, 13, 18, 22, 27, 31)
LATE_LAYERS = tuple(range(24, 32))
FIXED_BANDS = (
    tuple(range(0, 8)),
    tuple(range(8, 16)),
    tuple(range(16, 24)),
    tuple(range(24, 32)),
)
CORE_METRICS = ("lens_H", "lens_logp_tgt", "lens_logp_top1", "lens_kl_final")
REQUIRED_MODULES = ("attn", "mlp", "resid")

DEPLOYED_UPCR_FIT = {
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

_EPS = 1e-12
_DEGENERATE_STD = 1e-8
_ROW_ID = re.compile(r"^(?P<problem>[^:]+):(?P<candidate>[0-9]+)$")
_FORBIDDEN_FIELD_NAMES = {"label", "labels", "y", "target", "targets"}


def all_layers(n_layers: int) -> tuple[int, ...]:
    """Return every layer for an architecture without assuming L=32."""

    if int(n_layers) < 1:
        raise ValueError("n_layers must be positive")
    return tuple(range(int(n_layers)))


def spaced_layers(n_layers: int, count: int = 8) -> tuple[int, ...]:
    """Frozen architecture-relative evenly spaced layer subset."""

    if int(n_layers) < 1 or int(count) < 1 or int(count) > int(n_layers):
        raise ValueError("count must be between one and n_layers")
    return tuple(int(value) for value in np.rint(
        np.linspace(0, int(n_layers) - 1, int(count))
    ).astype(int))


def late_layers(n_layers: int, count: int = 8) -> tuple[int, ...]:
    """Frozen architecture-relative final-layer subset."""

    if int(n_layers) < 1 or int(count) < 1 or int(count) > int(n_layers):
        raise ValueError("count must be between one and n_layers")
    return tuple(range(int(n_layers) - int(count), int(n_layers)))


def fixed_bands(n_layers: int, count: int = 4) -> tuple[tuple[int, ...], ...]:
    """Split depth into fixed contiguous architecture-relative bands."""

    if int(n_layers) < int(count) or int(count) < 1:
        raise ValueError("band count must be positive and no larger than n_layers")
    return tuple(
        tuple(int(value) for value in band)
        for band in np.array_split(np.arange(int(n_layers)), int(count))
    )


def _readonly(array: Any, *, dtype: Any | None = None) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).strip().lower() in _FORBIDDEN_FIELD_NAMES:
                return True
            if _contains_forbidden_key(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(child) for child in value)
    return False


@dataclass(frozen=True)
class LayerCell:
    """One validated layer-view cell with correctness labels removed.

    ``records`` remain in row-id order and contain the original sidecar arrays.
    They are not copied to float64, which keeps the large float16 sidecars
    memory-efficient.  The record mappings themselves are shallow copies made
    by :func:`validate_and_join`, so removing ``label`` cannot mutate the
    loaded sidecar object.
    """

    cell_id: str
    row_ids: tuple[str, ...]
    problem_ids: tuple[str, ...]
    n_gen_tokens: np.ndarray
    records: tuple[Mapping[str, Any], ...]
    modules: tuple[str, ...]
    n_layers: int
    projection_dim: int
    covariance_rank: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.row_ids)
        if not self.cell_id:
            raise ValueError("cell_id must not be empty")
        if n < 3:
            raise ValueError("LayerCell needs at least three candidate rows")
        if len(set(self.row_ids)) != n:
            raise ValueError("row_ids must be unique")
        if len(self.problem_ids) != n or len(self.records) != n:
            raise ValueError("row_ids, problem_ids, and records must align")
        tokens = np.asarray(self.n_gen_tokens, dtype=int)
        if tokens.shape != (n,) or np.any(tokens <= 0):
            raise ValueError("n_gen_tokens must be a positive vector aligned to rows")
        if self.n_layers < 1 or self.projection_dim < 1 or self.covariance_rank < 1:
            raise ValueError("layer, projection, and covariance dimensions must be positive")
        if set(REQUIRED_MODULES) - set(self.modules):
            raise ValueError("modules must contain attn, mlp, and resid")
        for row_id, record in zip(self.row_ids, self.records):
            if _contains_forbidden_key(record):
                raise ValueError(f"LayerCell record {row_id!r} still contains a label field")
        if _contains_forbidden_key(self.provenance):
            raise ValueError("LayerCell provenance must not contain label fields")
        object.__setattr__(self, "row_ids", tuple(str(value) for value in self.row_ids))
        object.__setattr__(self, "problem_ids", tuple(str(value) for value in self.problem_ids))
        object.__setattr__(self, "records", tuple(self.records))
        object.__setattr__(self, "modules", tuple(str(value) for value in self.modules))
        object.__setattr__(self, "n_gen_tokens", _readonly(tokens, dtype=int))

    @property
    def n_samples(self) -> int:
        return len(self.row_ids)

    @property
    def protocol_signature(self) -> str:
        payload = {
            "cell_id": self.cell_id,
            "row_ids_sha256": hashlib.sha256(
                "\n".join(self.row_ids).encode("utf-8")
            ).hexdigest(),
            "n_samples": self.n_samples,
            "n_layers": self.n_layers,
            "projection_dim": self.projection_dim,
            "covariance_rank": self.covariance_rank,
            "model": self.provenance.get("model", ""),
            "version": self.provenance.get("version", ""),
            "projection_seed": self.provenance.get("proj_seed"),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class FeatureMatrix:
    """A finite, risk-oriented, label-free matrix passed to fusion methods."""

    values: np.ndarray
    feature_names: tuple[str, ...]
    risk_anchor: np.ndarray
    groups: tuple[str, ...]
    protocol_signature: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2 or values.shape[0] < 3 or values.shape[1] < 1:
            raise ValueError("values must have shape (samples>=3, features>=1)")
        if not np.isfinite(values).all():
            raise ValueError("FeatureMatrix values must be finite")
        names = tuple(str(value) for value in self.feature_names)
        groups = tuple(str(value) for value in self.groups)
        if len(names) != values.shape[1] or len(groups) != values.shape[1]:
            raise ValueError("feature_names and groups must align to value columns")
        if len(set(names)) != len(names):
            raise ValueError("feature_names must be unique")
        anchor = np.asarray(self.risk_anchor, dtype=float)
        if anchor.shape != (values.shape[0],) or not np.isfinite(anchor).all():
            raise ValueError("risk_anchor must be a finite vector aligned to samples")
        if float(np.std(anchor)) < _DEGENERATE_STD:
            raise ValueError("risk_anchor is degenerate")
        if not str(self.protocol_signature):
            raise ValueError("protocol_signature must not be empty")
        if _contains_forbidden_key(self.metadata):
            raise ValueError("FeatureMatrix metadata must not contain label fields")
        object.__setattr__(self, "values", _readonly(values, dtype=float))
        object.__setattr__(self, "risk_anchor", _readonly(anchor, dtype=float))
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "protocol_signature", str(self.protocol_signature))

    @property
    def n_samples(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.values.shape[1])


def assert_same_protocol(*matrices: FeatureMatrix) -> None:
    """Reject comparisons that do not contain the same ordered candidate cohort."""

    if not matrices:
        raise ValueError("at least one FeatureMatrix is required")
    expected = matrices[0].protocol_signature
    mismatched = [index for index, matrix in enumerate(matrices) if matrix.protocol_signature != expected]
    if mismatched:
        raise ValueError(f"protocol signature mismatch at matrices {mismatched}")


def _raw_candidate(raw_cache: Mapping[Any, Any], row_id: str) -> Mapping[str, Any]:
    match = _ROW_ID.match(str(row_id))
    if not match:
        raise ValueError(f"invalid row id {row_id!r}; expected 'problem:candidate'")
    problem = match.group("problem")
    candidate_index = int(match.group("candidate"))
    problem_lookup = {str(key): value for key, value in raw_cache.items()}
    if problem not in problem_lookup:
        raise KeyError(f"raw cache has no problem {problem!r}")
    entry = problem_lookup[problem]
    candidates = entry.get("candidates") if isinstance(entry, Mapping) else None
    if not isinstance(candidates, (list, tuple)):
        raise ValueError(f"raw problem {problem!r} has no candidate list")
    if candidate_index >= len(candidates):
        raise IndexError(f"raw problem {problem!r} has no candidate {candidate_index}")
    candidate = candidates[candidate_index]
    if not isinstance(candidate, Mapping):
        raise ValueError(f"raw candidate {row_id!r} is not a mapping")
    return candidate


def _expected_raw_ids(raw_cache: Mapping[Any, Any]) -> set[str]:
    output: set[str] = set()
    for raw_key, entry in raw_cache.items():
        candidates = entry.get("candidates") if isinstance(entry, Mapping) else None
        if not isinstance(candidates, (list, tuple)):
            raise ValueError(f"raw problem {raw_key!r} has no candidate list")
        output.update(f"{raw_key}:{index}" for index in range(len(candidates)))
    return output


def _numeric_row_sort(row_id: str) -> tuple[int, int | str, int]:
    match = _ROW_ID.match(str(row_id))
    if not match:
        raise ValueError(f"invalid sidecar row key {row_id!r}")
    problem = match.group("problem")
    try:
        return 0, int(problem), int(match.group("candidate"))
    except ValueError:
        return 1, problem, int(match.group("candidate"))


def validate_and_join(
    raw_cache: Mapping[Any, Any],
    sidecar: Mapping[str, Any],
    *,
    cell_id: str,
    expected_model: str | None = None,
    expected_version: str = "layer-lens-v1",
    expected_n_layers: int = 32,
    expected_hidden_size: int = 4096,
    expected_projection_dim: int = 256,
    expected_covariance_rank: int = 16,
    require_complete: bool = True,
    final_residual_kl_tolerance: float = 1e-6,
    exclude_invalid: bool = False,
    require_geometry_finite: bool = True,
) -> tuple[LayerCell, dict[str, Any]]:
    """Validate and exactly join a nested raw cache to its layer sidecar.

    Correctness is inspected solely to assert equality between the two source
    artifacts.  It is never returned in the :class:`LayerCell` or audit.  The
    raw cache must be opened again by the evaluator after score freezing.
    """

    if not isinstance(raw_cache, Mapping) or not isinstance(sidecar, Mapping):
        raise TypeError("raw_cache and sidecar must be mappings")
    meta = sidecar.get("_meta")
    if not isinstance(meta, Mapping):
        raise ValueError("sidecar is missing a mapping-valued _meta")
    checks = {
        "version": (meta.get("version"), expected_version),
        "n_layers": (meta.get("n_layers"), expected_n_layers),
        "hidden_size": (meta.get("hidden_size"), expected_hidden_size),
        "proj_dim": (meta.get("proj_dim"), expected_projection_dim),
        "cov_eigs_r": (meta.get("cov_eigs_r"), expected_covariance_rank),
    }
    if expected_model is not None:
        checks["model"] = (meta.get("model"), expected_model)
    mismatches = [f"{name}={actual!r} (expected {expected!r})"
                  for name, (actual, expected) in checks.items() if actual != expected]
    if mismatches:
        raise ValueError("sidecar metadata mismatch: " + "; ".join(mismatches))
    if require_complete and meta.get("complete") is not True:
        raise ValueError("sidecar _meta.complete is not true")
    modules = tuple(str(value) for value in meta.get("modules", ()))
    quantities = tuple(str(value) for value in meta.get("quantities", ()))
    missing_modules = set(REQUIRED_MODULES) - set(modules)
    missing_quantities = set(CORE_METRICS) - set(quantities)
    if missing_modules or missing_quantities:
        raise ValueError(
            f"sidecar contract missing modules={sorted(missing_modules)} "
            f"quantities={sorted(missing_quantities)}"
        )
    resid_index = modules.index("resid")

    row_keys = [str(key) for key in sidecar if str(key) != "_meta"]
    expected_ids = _expected_raw_ids(raw_cache)
    if set(row_keys) != expected_ids:
        missing = sorted(expected_ids - set(row_keys))[:5]
        extra = sorted(set(row_keys) - expected_ids)[:5]
        raise ValueError(f"raw/sidecar row-id mismatch; missing={missing}, extra={extra}")
    row_keys.sort(key=_numeric_row_sort)

    records: list[Mapping[str, Any]] = []
    included_row_ids: list[str] = []
    problem_ids: list[str] = []
    token_counts: list[int] = []
    excluded_rows: list[dict[str, str]] = []
    nonfinite_geometry_counts = {"cov_eigs": 0, "hid_proj": 0}
    max_final_kl = 0.0
    for row_id in row_keys:
        record = sidecar[row_id]
        if not isinstance(record, Mapping):
            raise ValueError(f"sidecar row {row_id!r} is not a mapping")
        raw_candidate = _raw_candidate(raw_cache, row_id)
        if "label" not in raw_candidate or "label" not in record:
            raise ValueError(f"row {row_id!r} lacks a label needed for join validation")
        if int(bool(raw_candidate["label"])) != int(bool(record["label"])):
            raise ValueError(f"label mismatch at row {row_id!r}")
        n_tokens = int(record.get("n_gen_tokens", -1))
        gen_ids = raw_candidate.get("gen_token_ids")
        if n_tokens <= 0 or gen_ids is None or len(gen_ids) != n_tokens:
            reason = (
                f"generated-token length mismatch: sidecar={n_tokens}, "
                f"raw={None if gen_ids is None else len(gen_ids)}"
            )
            if exclude_invalid:
                excluded_rows.append({"row_id": row_id, "reason": reason})
                continue
            raise ValueError(f"{reason} at row {row_id!r}")
        token_entropies = raw_candidate.get("token_entropies")
        if token_entropies is not None and len(token_entropies) != n_tokens:
            reason = (
                f"entropy-token length mismatch: sidecar={n_tokens}, "
                f"raw={len(token_entropies)}"
            )
            if exclude_invalid:
                excluded_rows.append({"row_id": row_id, "reason": reason})
                continue
            raise ValueError(f"{reason} at row {row_id!r}")

        expected_shapes = {
            "lens_H": (len(modules), expected_n_layers, n_tokens),
            "lens_logp_tgt": (len(modules), expected_n_layers, n_tokens),
            "lens_logp_top1": (len(modules), expected_n_layers, n_tokens),
            "lens_kl_final": (len(modules), expected_n_layers, n_tokens),
            "resid_norm": (expected_n_layers, n_tokens),
            "cov_eigs": (expected_n_layers, expected_covariance_rank),
            "hid_proj": (expected_n_layers, expected_projection_dim),
        }
        for name, shape in expected_shapes.items():
            if name not in record:
                raise ValueError(f"sidecar row {row_id!r} is missing {name}")
            array = np.asarray(record[name])
            if array.shape != shape:
                raise ValueError(
                    f"sidecar row {row_id!r} {name} has shape {array.shape}, expected {shape}"
                )
            nonfinite_count = int(np.size(array) - np.count_nonzero(np.isfinite(array)))
            if nonfinite_count:
                if name in nonfinite_geometry_counts and not require_geometry_finite:
                    nonfinite_geometry_counts[name] += nonfinite_count
                else:
                    raise ValueError(f"sidecar row {row_id!r} {name} contains non-finite values")
        final_kl = np.asarray(record["lens_kl_final"], dtype=float)[resid_index, -1]
        max_final_kl = max(max_final_kl, float(np.max(np.abs(final_kl))))
        sanitized = {key: value for key, value in record.items()
                     if str(key).strip().lower() not in _FORBIDDEN_FIELD_NAMES}
        records.append(sanitized)
        included_row_ids.append(row_id)
        problem_ids.append(row_id.rsplit(":", 1)[0])
        token_counts.append(n_tokens)
    if max_final_kl > float(final_residual_kl_tolerance):
        raise ValueError(
            f"final residual KL identity failed: max={max_final_kl:.6g} > "
            f"{final_residual_kl_tolerance:.6g}"
        )

    provenance = dict(meta)
    provenance.update({"cell_id": cell_id, "join_contract": "problem:candidate"})
    cell = LayerCell(
        cell_id=cell_id,
        row_ids=tuple(included_row_ids),
        problem_ids=tuple(problem_ids),
        n_gen_tokens=np.asarray(token_counts, dtype=int),
        records=tuple(records),
        modules=modules,
        n_layers=expected_n_layers,
        projection_dim=expected_projection_dim,
        covariance_rank=expected_covariance_rank,
        provenance=provenance,
    )
    audit = {
        "cell_id": cell_id,
        "n_rows": cell.n_samples,
        "n_joined_rows": cell.n_samples,
        "n_source_rows": len(row_keys),
        "n_excluded_rows": len(excluded_rows),
        "excluded_rows": excluded_rows,
        "nonfinite_geometry_counts": nonfinite_geometry_counts,
        "core_lens_and_residual_tensors_finite": True,
        "geometry_tensors_finite": not any(nonfinite_geometry_counts.values()),
        "n_problems": len(set(problem_ids)),
        "min_tokens": int(np.min(token_counts)),
        "max_tokens": int(np.max(token_counts)),
        "labels_compared_and_equal": len(row_keys),
        "max_final_residual_kl": max_final_kl,
        "model": meta.get("model"),
        "version": meta.get("version"),
        "n_layers": expected_n_layers,
        "hidden_size": expected_hidden_size,
        "projection_dim": expected_projection_dim,
        "covariance_rank": expected_covariance_rank,
        "projection_seed": meta.get("proj_seed"),
        "complete": bool(meta.get("complete")),
        "protocol_signature": cell.protocol_signature,
    }
    return cell, audit


def entropy_agreement_gate(
    raw_cache: Mapping[Any, Any],
    cell: LayerCell,
    *,
    assume_comparable_domains: bool = False,
    median_error_limit: float = 0.02,
    first_token_median_limit: float = 0.05,
    token_error_limit: float = 0.05,
    min_fraction_within_limit: float = 0.90,
) -> dict[str, Any]:
    """Compare nested raw entropy traces with final-residual lens entropy.

    This function repairs the *loader* bug in job 183956 by descending into
    every problem's ``candidates`` list.  It is not, by itself, a valid Gate B
    for the archived artifacts: ``token_entropies`` were saved after the
    generation temperature/top-k/top-p warp and top-15 renormalization, whereas
    ``lens_H`` is an unwarped full-vocabulary entropy.  The numerical summary is
    still useful for diagnosing that domain mismatch, but promotion fails
    closed unless a caller explicitly certifies comparable domains (used only
    by controlled fixtures).  The live reference pilot reconstructs ordinary
    logits, reapplies the warpers, and performs the valid comparison.
    """

    resid_index = cell.modules.index("resid")
    absolute_errors: list[np.ndarray] = []
    first_errors: list[float] = []
    missing: list[str] = []
    length_mismatch: list[str] = []
    for row_id, record in zip(cell.row_ids, cell.records):
        raw = _raw_candidate(raw_cache, row_id)
        entropy = raw.get("token_entropies")
        if entropy is None:
            missing.append(row_id)
            continue
        observed = np.asarray(entropy, dtype=float)
        reconstructed = np.asarray(record["lens_H"], dtype=float)[resid_index, -1]
        if observed.shape != reconstructed.shape:
            length_mismatch.append(row_id)
            continue
        errors = np.abs(observed - reconstructed)
        absolute_errors.append(errors)
        first_errors.append(float(errors[0]))
    if not absolute_errors:
        return {
            "done": False,
            "pass": False,
            "numeric_thresholds_pass": False,
            "comparable_domains": bool(assume_comparable_domains),
            "reason": "no comparable token_entropies traces",
            "n_rows": cell.n_samples,
            "n_compared_rows": 0,
            "missing_rows": missing,
            "length_mismatch_rows": length_mismatch,
        }
    pooled = np.concatenate(absolute_errors)
    median_error = float(np.median(pooled))
    first_median = float(np.median(first_errors))
    fraction = float(np.mean(pooled <= float(token_error_limit)))
    complete = not missing and not length_mismatch and len(absolute_errors) == cell.n_samples
    numeric_thresholds_pass = bool(
        complete
        and median_error <= float(median_error_limit)
        and first_median <= float(first_token_median_limit)
        and fraction >= float(min_fraction_within_limit)
    )
    passed = bool(assume_comparable_domains and numeric_thresholds_pass)
    reasons = []
    if not assume_comparable_domains:
        reasons.append(
            "entropy domains are not comparable: raw trace is sampling-warped/top-15; "
            "sidecar lens_H is unwarped/full-vocabulary"
        )
    if not complete:
        reasons.append("not every nested candidate was comparable")
    if median_error > median_error_limit:
        reasons.append("median absolute error exceeded limit")
    if first_median > first_token_median_limit:
        reasons.append("median first-token error exceeded limit")
    if fraction < min_fraction_within_limit:
        reasons.append("fraction within token limit was too small")
    return {
        "done": True,
        "pass": passed,
        "numeric_thresholds_pass": numeric_thresholds_pass,
        "comparable_domains": bool(assume_comparable_domains),
        "reason": "ok" if passed else "; ".join(reasons),
        "n_rows": cell.n_samples,
        "n_compared_rows": len(absolute_errors),
        "n_compared_tokens": int(pooled.size),
        "missing_rows": missing,
        "length_mismatch_rows": length_mismatch,
        "median_absolute_error": median_error,
        "median_first_token_error": first_median,
        "fraction_within_token_limit": fraction,
        "thresholds": {
            "median_absolute_error": float(median_error_limit),
            "median_first_token_error": float(first_token_median_limit),
            "token_absolute_error": float(token_error_limit),
            "fraction_within_token_limit": float(min_fraction_within_limit),
        },
    }


def load_evaluation_labels(
    raw_cache: Mapping[Any, Any], row_ids: Sequence[str], *, hallucination_is_one: bool = True
) -> np.ndarray:
    """Evaluator-only label loader; call after the score-freeze checkpoint."""

    correctness = np.asarray(
        [int(bool(_raw_candidate(raw_cache, str(row_id))["label"])) for row_id in row_ids],
        dtype=int,
    )
    output = 1 - correctness if hallucination_is_one else correctness
    return _readonly(output, dtype=int)


def _layers(layers: Sequence[int], n_layers: int) -> tuple[int, ...]:
    output = tuple(int(value) for value in layers)
    if not output or len(set(output)) != len(output):
        raise ValueError("layers must be a nonempty sequence of unique indices")
    if any(value < 0 or value >= n_layers for value in output):
        raise ValueError(f"layer indices must be in [0, {n_layers - 1}]")
    return output


def _band(layer: int, n_layers: int = 32) -> str:
    for index, band in enumerate(fixed_bands(n_layers)):
        if layer in band:
            return f"band_{index}_{band[0]:02d}_{band[-1]:02d}"
    raise ValueError(f"layer {layer} is outside the frozen {n_layers}-layer bands")


def _metric_values(cell: LayerCell, metric: str, module: str) -> np.ndarray:
    module_index = cell.modules.index(module)
    direction = -1.0 if metric in {"lens_logp_tgt", "lens_logp_top1"} else 1.0
    rows = []
    for record in cell.records:
        trace = np.asarray(record[metric], dtype=float)[module_index]
        rows.append(direction * np.mean(trace, axis=1))
    return np.asarray(rows, dtype=float)


def _risk_anchor(cell: LayerCell) -> np.ndarray:
    return _metric_values(cell, "lens_logp_tgt", "resid")[:, -1]


def _drop_degenerate_columns(
    values: np.ndarray, names: Sequence[str], groups: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...], tuple[str, ...], list[str]]:
    kept, dropped = [], []
    for index, name in enumerate(names):
        column = np.asarray(values[:, index], dtype=float)
        if not np.isfinite(column).all() or float(np.std(column)) < _DEGENERATE_STD:
            dropped.append(str(name))
        else:
            kept.append(index)
    if not kept:
        raise ValueError("every extracted feature was degenerate")
    return (
        np.asarray(values[:, kept], dtype=float),
        tuple(str(names[index]) for index in kept),
        tuple(str(groups[index]) for index in kept),
        dropped,
    )


def _feature_matrix(
    cell: LayerCell,
    values: np.ndarray,
    names: Sequence[str],
    groups: Sequence[str],
    *,
    contract: str,
    metadata: Mapping[str, Any] | None = None,
    drop_degenerate: bool = True,
) -> FeatureMatrix:
    dropped: list[str] = []
    if drop_degenerate:
        values, kept_names, kept_groups, dropped = _drop_degenerate_columns(
            np.asarray(values, dtype=float), names, groups
        )
    else:
        kept_names, kept_groups = tuple(names), tuple(groups)
    meta = dict(metadata or {})
    meta.update({
        "contract": contract,
        "cell_id": cell.cell_id,
        "nominal_feature_count": len(names),
        "dropped_degenerate_features": dropped,
        "risk_anchor": "resid.final_layer.target_token_nll",
    })
    return FeatureMatrix(
        values=values,
        feature_names=kept_names,
        risk_anchor=_risk_anchor(cell),
        groups=kept_groups,
        protocol_signature=cell.protocol_signature,
        metadata=meta,
    )


def extract_resid_core(
    cell: LayerCell, layers: Sequence[int] | None = None
) -> FeatureMatrix:
    """Build one residual-stream expert per selected layer.

    Each expert is the equal mean of the available cell-standardized token
    means for entropy, target NLL, top-1 surprisal, and KL-to-final.  A
    mechanically degenerate component is dropped only from that layer's mean;
    notably, final residual KL is expected to be identically zero.
    """

    selected = _layers(all_layers(cell.n_layers) if layers is None else layers,
                       cell.n_layers)
    metric_matrices = {
        metric: _metric_values(cell, metric, "resid") for metric in CORE_METRICS
    }
    columns, names, groups = [], [], []
    components_by_feature: dict[str, list[str]] = {}
    dropped_components: list[str] = []
    for layer in selected:
        components = []
        used = []
        for metric in CORE_METRICS:
            raw = metric_matrices[metric][:, layer]
            std = float(np.std(raw))
            component_name = f"resid.{metric}.layer_{layer:02d}"
            if not np.isfinite(raw).all() or std < _DEGENERATE_STD:
                dropped_components.append(component_name)
                continue
            components.append((raw - float(np.mean(raw))) / std)
            used.append(metric)
        if not components:
            raise ValueError(f"layer {layer} has no non-degenerate core components")
        name = f"resid_core.layer_{layer:02d}"
        columns.append(np.mean(np.column_stack(components), axis=1))
        names.append(name)
        groups.append(_band(layer, cell.n_layers))
        components_by_feature[name] = used
    return _feature_matrix(
        cell,
        np.column_stack(columns),
        names,
        groups,
        contract=f"resid-core-{len(selected)}",
        metadata={
            "layers": list(selected),
            "component_directions": {
                "lens_H": "+risk",
                "lens_logp_tgt": "negated_to_nll",
                "lens_logp_top1": "negated_to_surprisal",
                "lens_kl_final": "+risk",
            },
            "components_by_feature": components_by_feature,
            "dropped_degenerate_components": dropped_components,
        },
        drop_degenerate=True,
    )


def extract_lens_grid(
    cell: LayerCell, layers: Sequence[int] | None = None
) -> FeatureMatrix:
    """Build the oriented 3-module x 4-metric grid for selected layers.

    This is also the label-free source for layer-oracle *diagnostic* curves;
    evaluators can score its named columns after the score-freeze boundary
    without reimplementing feature extraction inline.
    """

    selected = _layers(all_layers(cell.n_layers) if layers is None else layers,
                       cell.n_layers)
    columns, names, groups = [], [], []
    for module in REQUIRED_MODULES:
        for metric in CORE_METRICS:
            matrix = _metric_values(cell, metric, module)
            group = f"{module}.{metric}"
            for layer in selected:
                columns.append(matrix[:, layer])
                names.append(f"{group}.layer_{layer:02d}")
                groups.append(group)
    return _feature_matrix(
        cell,
        np.column_stack(columns),
        names,
        groups,
        contract=f"lens-{len(REQUIRED_MODULES) * len(CORE_METRICS) * len(selected)}",
        metadata={
            "layers": list(selected),
            "modules": list(REQUIRED_MODULES),
            "metrics": list(CORE_METRICS),
            "grouping": "module_x_metric",
        },
        drop_degenerate=True,
    )


def extract_lens96(
    cell: LayerCell, layers: Sequence[int] | None = None
) -> FeatureMatrix:
    """Build the nominal 3-module x 4-metric x 8-layer lens contract."""

    return extract_lens_grid(
        cell, layers=spaced_layers(cell.n_layers) if layers is None else layers
    )


def extract_trilens_entropy(
    cell: LayerCell, layers: Sequence[int] | None = None
) -> FeatureMatrix:
    """TriLens-compatible 3 x L entropy contract.

    The capture contains the MHSA write, FFN write, and residual-stream
    logit-lens entropies.  We freeze token-mean readout because the paper does
    not specify the exact fixed-token reduction used in its released text.
    """

    selected = _layers(all_layers(cell.n_layers) if layers is None else layers,
                       cell.n_layers)
    columns, names, groups = [], [], []
    for module in REQUIRED_MODULES:
        values = _metric_values(cell, "lens_H", module)
        for layer in selected:
            columns.append(values[:, layer])
            names.append(f"trilens_entropy.{module}.layer_{layer:02d}")
            groups.append(module)
    return _feature_matrix(
        cell, np.column_stack(columns), names, groups,
        contract=f"trilens-entropy-{3 * len(selected)}",
        metadata={
            "layers": list(selected),
            "modules": list(REQUIRED_MODULES),
            "readout": "mean_over_generated_tokens_frozen_approximation",
            "fidelity": "feature-faithful; paper-exact token readout unavailable",
        },
    )


def extract_dola_kl_proxy(
    cell: LayerCell, layers: Sequence[int] | None = None
) -> FeatureMatrix:
    """Residual depth-wise KL-to-final proxy for the DoLa/JSD detector arm."""

    selected = _layers(all_layers(cell.n_layers) if layers is None else layers,
                       cell.n_layers)
    matrix = _metric_values(cell, "lens_kl_final", "resid")
    columns = [matrix[:, layer] for layer in selected]
    names = [f"dola_kl_proxy.resid.layer_{layer:02d}" for layer in selected]
    groups = [_band(layer, cell.n_layers) for layer in selected]
    return _feature_matrix(
        cell, np.column_stack(columns), names, groups,
        contract=f"dola-kl-proxy-{len(selected)}",
        metadata={
            "layers": list(selected),
            "fidelity": "proxy: saved KL-to-final, whereas the TriLens DoLa-style arm uses JSD",
            "final_layer_degenerate_by_identity": True,
        },
    )


def extract_haloscope_projection(
    cell: LayerCell, layer: int | None = None
) -> FeatureMatrix:
    """Fixed-layer mean-token JL projection used by the direct HaloScope proxy."""

    selected = int(cell.n_layers // 2 if layer is None else layer)
    _layers((selected,), cell.n_layers)
    values = np.asarray(
        [np.asarray(record["hid_proj"], dtype=float)[selected] for record in cell.records],
        dtype=float,
    )
    names = [f"haloscope_jl.layer_{selected:02d}.dim_{index:03d}"
             for index in range(values.shape[1])]
    groups = ["haloscope_projection"] * values.shape[1]
    return _feature_matrix(
        cell, values, names, groups,
        contract=f"haloscope-direct-jl-middle-k4",
        metadata={
            "layer": selected,
            "readout": "mean_over_generated_tokens_random_projection_dim_256",
            "fidelity": "direct-membership proxy; not HaloScope pseudo-label classifier",
        },
    )


def fit_haloscope_direct_proxy(
    matrix: FeatureMatrix, *, k: int = 4
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit HaloScope's direct top-subspace membership score without labels."""

    values = np.asarray(matrix.values, dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    kept = min(int(k), len(singular), vt.shape[0])
    if kept < 1:
        raise ValueError("HaloScope proxy needs at least one singular direction")
    projection = centered @ vt[:kept].T
    score = np.mean(projection ** 2 * singular[:kept][None, :], axis=1)
    score = _anchor_orient(score, matrix.risk_anchor)[0]
    return np.asarray(score, dtype=float), {
        "labels_seen_during_fit": False,
        "k": kept,
        "singular_values": singular[:kept].tolist(),
        "orientation_anchor": "final-layer residual target-token NLL",
        "fidelity": matrix.metadata.get("fidelity"),
    }


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= _EPS:
        return 0.0
    return float(1.0 - np.dot(left, right) / denominator)


def _normalized_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / (np.linalg.norm(left) + np.linalg.norm(right) + _EPS))


def _covariance_summaries(eigenvalues: np.ndarray) -> tuple[float, float, float]:
    eig = np.maximum(np.asarray(eigenvalues, dtype=float), 0.0)
    total = float(np.sum(eig))
    if total <= _EPS:
        return 0.0, 0.0, 0.0
    probability = eig / total
    entropy = float(-np.sum(probability * np.log(probability + _EPS)))
    effective_rank = float(np.exp(entropy))
    top_share = float(np.max(probability))
    # The two rank/diversity features are negated so larger consistently means
    # greater concentration / less representational diversity / more risk.
    return top_share, -effective_rank, -entropy


def extract_geometry(
    cell: LayerCell, layers: Sequence[int] | None = None
) -> FeatureMatrix:
    """Extract only orthogonal-rotation-invariant representation summaries."""

    selected = _layers(all_layers(cell.n_layers) if layers is None else layers,
                       cell.n_layers)
    n = cell.n_samples
    columns: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []

    def add(name: str, group: str, values: Sequence[float]) -> None:
        columns.append(np.asarray(values, dtype=float))
        names.append(name)
        groups.append(group)

    for layer in selected:
        if layer != cell.n_layers - 1:
            add(
                f"geometry.hidden_cos_to_final.layer_{layer:02d}",
                "geometry.hidden_to_final",
                [_cosine_distance(np.asarray(record["hid_proj"])[layer],
                                  np.asarray(record["hid_proj"])[-1])
                 for record in cell.records],
            )
            add(
                f"geometry.hidden_dist_to_final.layer_{layer:02d}",
                "geometry.hidden_to_final",
                [_normalized_distance(np.asarray(record["hid_proj"])[layer],
                                      np.asarray(record["hid_proj"])[-1])
                 for record in cell.records],
            )
            add(
                f"geometry.resid_norm_convergence.layer_{layer:02d}",
                "geometry.resid_norm",
                [abs(math.log((float(np.mean(np.asarray(record["resid_norm"])[layer])) + _EPS)
                              / (float(np.mean(np.asarray(record["resid_norm"])[-1])) + _EPS)))
                 for record in cell.records],
            )
        if layer > 0:
            add(
                f"geometry.hidden_cos_adjacent.layer_{layer:02d}",
                "geometry.hidden_adjacent",
                [_cosine_distance(np.asarray(record["hid_proj"])[layer - 1],
                                  np.asarray(record["hid_proj"])[layer])
                 for record in cell.records],
            )
            add(
                f"geometry.hidden_dist_adjacent.layer_{layer:02d}",
                "geometry.hidden_adjacent",
                [_normalized_distance(np.asarray(record["hid_proj"])[layer - 1],
                                      np.asarray(record["hid_proj"])[layer])
                 for record in cell.records],
            )
        cov = np.asarray([record["cov_eigs"][layer] for record in cell.records], dtype=float)
        summaries = np.asarray([_covariance_summaries(row) for row in cov], dtype=float)
        add(f"geometry.cov_top_share.layer_{layer:02d}", "geometry.covariance", summaries[:, 0])
        add(f"geometry.cov_neg_effective_rank.layer_{layer:02d}", "geometry.covariance", summaries[:, 1])
        add(f"geometry.cov_neg_spectral_entropy.layer_{layer:02d}", "geometry.covariance", summaries[:, 2])
    if not columns or n != len(columns[0]):
        raise ValueError("geometry extraction produced no aligned features")
    return _feature_matrix(
        cell,
        np.column_stack(columns),
        names,
        groups,
        contract="representation-geometry-invariant",
        metadata={
            "layers": list(selected),
            "rotation_invariant": True,
            "directions": {
                "distances_and_convergence": "+risk",
                "cov_top_share": "+risk",
                "effective_rank_and_entropy": "negated_to_concentration_risk",
            },
        },
        drop_degenerate=True,
    )


def residualize_token_length(
    matrix: FeatureMatrix, n_gen_tokens: Sequence[int]
) -> FeatureMatrix:
    """Remove each feature's unlabeled linear dependence on ``log1p(T)``."""

    lengths = np.asarray(n_gen_tokens, dtype=float)
    if lengths.shape != (matrix.n_samples,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("n_gen_tokens must be a positive vector aligned to samples")
    predictor = np.column_stack([np.ones(matrix.n_samples), np.log1p(lengths)])

    def residualize(values: np.ndarray) -> np.ndarray:
        beta = np.linalg.lstsq(predictor, values, rcond=None)[0]
        return values - predictor @ beta

    values = np.column_stack([residualize(matrix.values[:, index])
                              for index in range(matrix.n_features)])
    anchor = residualize(matrix.risk_anchor)
    meta = dict(matrix.metadata)
    meta.update({
        "token_length_residualized": True,
        "token_length_transform": "log1p",
        "token_length_fit": "unlabeled_ols_with_intercept",
    })
    return FeatureMatrix(
        values=values,
        feature_names=matrix.feature_names,
        risk_anchor=anchor,
        groups=matrix.groups,
        protocol_signature=matrix.protocol_signature,
        metadata=meta,
    )


def _standardize(matrix: FeatureMatrix) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Use the repository's frozen label-free standardization everywhere."""

    X, keep, means, scales = canonical_standardize(matrix.values)
    diagnostics = {
        "kept_feature_names": [matrix.feature_names[index] for index in keep],
        "dropped_feature_names": [matrix.feature_names[index]
                                  for index in range(matrix.n_features) if index not in set(keep)],
        "mean": means.tolist(),
        "scale": scales.tolist(),
        "implementation": "spectral_utils.paper_benchmark_suite.standardize",
    }
    return X, np.asarray(keep, dtype=int), diagnostics


def _anchor_orient(score: np.ndarray, anchor: np.ndarray) -> tuple[np.ndarray, bool]:
    score = np.asarray(score, dtype=float)
    anchor = np.asarray(anchor, dtype=float)
    if score.shape != anchor.shape or not np.isfinite(score).all():
        raise ValueError("score and risk anchor must be aligned and finite")
    if float(np.std(score)) < _DEGENERATE_STD:
        raise ValueError("fusion returned a degenerate score")
    corr = float(np.corrcoef(score, anchor)[0, 1])
    if not np.isfinite(corr):
        raise ValueError("score orientation correlation is not finite")
    flipped = corr < 0
    return (-score if flipped else score), bool(flipped)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def fit_controls(
    matrix: FeatureMatrix, *, n_gen_tokens: Sequence[int] | None = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit final-NLL, optional token-length, equal-mean, and PC1 controls."""

    X, keep, standardization = _standardize(matrix)
    anchor = matrix.risk_anchor
    raw_scores: dict[str, np.ndarray] = {
        "final_layer_nll": np.asarray(anchor, dtype=float),
        "equal_mean": np.mean(X, axis=1),
    }
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    raw_scores["pc1"] = X @ vh[0]
    if n_gen_tokens is not None:
        lengths = np.asarray(n_gen_tokens, dtype=float)
        if lengths.shape != (matrix.n_samples,) or np.any(lengths <= 0):
            raise ValueError("n_gen_tokens must be positive and aligned")
        raw_scores["token_length"] = np.log1p(lengths)
    scores, flips = {}, {}
    for name, raw in raw_scores.items():
        if name == "final_layer_nll":
            scores[name], flips[name] = np.asarray(raw, dtype=float), False
        else:
            scores[name], flips[name] = _anchor_orient(raw, anchor)
    diagnostics = {
        "labels_seen_during_fit": False,
        "protocol_signature": matrix.protocol_signature,
        "standardization": standardization,
        "kept_column_indices": keep.tolist(),
        "orientation_flipped": flips,
        "pc1_explained_variance_fraction": float(
            np.linalg.svd(X, compute_uv=False)[0] ** 2 / (np.sum(X ** 2) + _EPS)
        ),
    }
    return scores, diagnostics


def _fit_solver(
    F: np.ndarray,
    solver: str,
    *,
    dufs_seeds: Sequence[int],
    dufs_epochs: int,
    k: int,
    lambda_: float,
    dufs_gates: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return ``(weights, diagnostics)`` for one already-standardized matrix."""

    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[1] < 3 or not np.isfinite(F).all():
        raise ValueError("solver input must have shape (features, samples>=3) and be finite")
    if F.shape[0] < 3:
        weights = np.ones(F.shape[0], dtype=float) / F.shape[0]
        return weights, {"solver": solver, "fallback": "equal_mean_fewer_than_3_views"}
    if solver == "upcr":
        fit = upcr_fit(F, **DEPLOYED_UPCR_FIT)
        return np.asarray(fit.w, dtype=float), {
            "solver": solver,
            "weights": fit.w,
            "rho_hat_full": fit.rho_hat_full,
            "keep": fit.keep,
            "abstained": fit.abstained,
            "used_simple_average": fit.used_simple_average,
            "fit_meta": fit.meta,
        }
    if solver == "iu_pcr":
        fit = upcr_fit(F, **IU_FIT_DEFAULTS)
        return np.asarray(fit.w, dtype=float), {
            "solver": solver,
            "weights": fit.w,
            "rho_hat_full": fit.rho_hat_full,
            "keep": fit.keep,
            "fit_meta": fit.meta,
        }
    if solver != "dufs_liu_pcr":
        raise ValueError("solver must be 'upcr', 'iu_pcr', or 'dufs_liu_pcr'")
    if dufs_gates is None:
        gates, gate_diagnostics = dufs_soft_gates(
            F, seeds=tuple(int(seed) for seed in dufs_seeds), epochs=int(dufs_epochs)
        )
    else:
        gates = np.asarray(dufs_gates, dtype=float)
        if gates.shape != (F.shape[0],) or not np.isfinite(gates).all() or np.any(gates < 0):
            raise ValueError("dufs_gates must be finite, nonnegative, and aligned to features")
        gate_diagnostics = {"source": "provided", "effective_feature_count": float(
            np.sum(gates) ** 2 / (np.sum(gates ** 2) + _EPS)
        )}
    graph = build_graph_from_features(F, gates=gates, k=int(k))
    path = laplacian_iu_path(F, (0.0, float(lambda_)), graph=graph)
    zero = path[0.0]
    if not np.array_equal(zero.w, zero.baseline.w):
        raise AssertionError("lambda=0 must exactly equal IU-PCR")
    chosen = path[float(lambda_)]
    return np.asarray(chosen.w, dtype=float), {
        "solver": solver,
        "weights": chosen.w,
        "iu_weights": zero.w,
        "lambda_zero_exact": True,
        "lambda": float(lambda_),
        "k": int(k),
        "dufs_seeds": [int(seed) for seed in dufs_seeds],
        "dufs_epochs": int(dufs_epochs),
        "dufs_gates": gates,
        "dufs": gate_diagnostics,
        "graph": chosen.diagnostics,
    }


def fit_core_spectral(
    matrix: FeatureMatrix,
    *,
    methods: Sequence[str] = ("upcr", "iu_pcr", "dufs_liu_pcr"),
    dufs_seeds: Sequence[int] = DUFS_SEEDS,
    dufs_epochs: int = DUFS_EPOCHS,
    k: int = DUFS_K,
    lambda_: float = DUFS_LAMBDA,
    dufs_gates: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the registered three-solver progression to one frozen matrix."""

    allowed = {"upcr", "iu_pcr", "dufs_liu_pcr"}
    requested = tuple(str(method) for method in methods)
    if not requested or set(requested) - allowed:
        raise ValueError(f"methods must be a nonempty subset of {sorted(allowed)}")
    if dufs_gates is None:
        canonical_scores, diagnostics = canonical_fit_spectral_scores(
            matrix.values,
            feature_names=matrix.feature_names,
            risk_anchor=matrix.risk_anchor,
            dufs_seeds=dufs_seeds,
            dufs_epochs=dufs_epochs,
            k=k,
            lambda_=lambda_,
        )
        name_map = {
            "upcr": "deployed_upcr",
            "iu_pcr": "iu_pcr",
            "dufs_liu_pcr": "dufs_liu_pcr",
        }
        scores = {
            method: np.asarray(canonical_scores[name_map[method]], dtype=float)
            for method in requested
        }
        diagnostics = dict(diagnostics)
        diagnostics.update({
            "protocol_signature": matrix.protocol_signature,
            "canonical_implementation": "spectral_utils.paper_benchmark_suite.fit_spectral_scores",
            "requested_methods": list(requested),
        })
        return scores, diagnostics

    # A provided gate vector is retained for controlled synthetic diagnostics;
    # the registered benchmark never uses this escape hatch.
    X, keep, standardization = _standardize(matrix)
    F = X.T
    scores, fits, flips = {}, {}, {}
    for method in requested:
        weights, diagnostic = _fit_solver(
            F,
            method,
            dufs_seeds=dufs_seeds,
            dufs_epochs=dufs_epochs,
            k=k,
            lambda_=lambda_,
            dufs_gates=dufs_gates if method == "dufs_liu_pcr" else None,
        )
        scores[method], flips[method] = _anchor_orient(weights @ F, matrix.risk_anchor)
        fits[method] = _jsonable(diagnostic)
    diagnostics = {
        "labels_seen_during_fit": False,
        "protocol_signature": matrix.protocol_signature,
        "input_feature_names": list(matrix.feature_names),
        "kept_feature_names": [matrix.feature_names[index] for index in keep],
        "kept_column_indices": keep.tolist(),
        "standardization": standardization,
        "orientation_flipped": flips,
        "fits": fits,
        "canonical_implementation": False,
        "diagnostic_only_provided_gates": True,
    }
    return scores, diagnostics


def _encode_groups(groups: Sequence[str], expected_length: int) -> tuple[np.ndarray, list[str]]:
    if len(groups) != expected_length:
        raise ValueError("groups must align to feature columns")
    order: list[str] = []
    lookup: dict[str, int] = {}
    encoded = []
    for raw in groups:
        value = str(raw)
        if value not in lookup:
            lookup[value] = len(order)
            order.append(value)
        encoded.append(lookup[value])
    return np.asarray(encoded, dtype=int), order


def fit_dependency_methods(
    matrix: FeatureMatrix,
    *,
    methods: Sequence[str] = ("su_pcr", "lsml_continuous", "clustered_upcr"),
    clustered_groups: Sequence[str] | None = None,
    lsml_groups: Sequence[str] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit registered dependency controls on an identical standardized matrix."""

    allowed = {"su_pcr", "lsml_continuous", "clustered_upcr"}
    requested = tuple(str(method) for method in methods)
    if not requested or set(requested) - allowed:
        raise ValueError(f"methods must be a nonempty subset of {sorted(allowed)}")
    X, keep, standardization = _standardize(matrix)
    if X.shape[1] < 3:
        raise ValueError("dependency fusion requires at least three features")
    F = X.T
    kept_groups = tuple(matrix.groups[index] for index in keep)
    scores, fits, flips = {}, {}, {}

    if "su_pcr" in requested:
        fit = sparse_upcr_fit(F)
        scores["su_pcr"], flips["su_pcr"] = _anchor_orient(fit.w_pcr @ F, matrix.risk_anchor)
        fits["su_pcr"] = _jsonable({
            "weights": fit.w_pcr,
            "rho_hat": fit.rho_hat,
            "g2_hat": fit.g2_hat,
            "projection_residual": fit.projection_residual,
            "decomposition": fit.decomposition,
            "pcr_eigenvalues": fit.pcr_eigenvalues,
        })

    if "lsml_continuous" in requested:
        supplied = None
        group_names: list[str] | None = None
        if lsml_groups is not None:
            if len(lsml_groups) != matrix.n_features:
                raise ValueError("lsml_groups must align to the original FeatureMatrix")
            selected = tuple(str(lsml_groups[index]) for index in keep)
            supplied, group_names = _encode_groups(selected, F.shape[0])
        raw, meta = lsml_continuous(
            *[F[index] for index in range(F.shape[0])],
            groups=supplied,
            compute_score_matrix=False,
            loading_scale="complete",
        )
        scores["lsml_continuous"], flips["lsml_continuous"] = _anchor_orient(
            raw, matrix.risk_anchor
        )
        fits["lsml_continuous"] = _jsonable({"meta": meta, "supplied_group_names": group_names})

    if "clustered_upcr" in requested:
        source_groups = tuple(clustered_groups) if clustered_groups is not None else matrix.groups
        if len(source_groups) != matrix.n_features:
            raise ValueError("clustered_groups must align to the original FeatureMatrix")
        selected = tuple(str(source_groups[index]) for index in keep)
        encoded, group_names = _encode_groups(selected, F.shape[0])
        fit, identifiability = upcr_clustered_fit(
            F,
            groups=encoded,
            require_identifiable=True,
            **DEPLOYED_UPCR_FIT,
        )
        scores["clustered_upcr"], flips["clustered_upcr"] = _anchor_orient(
            fit.w @ F, matrix.risk_anchor
        )
        fits["clustered_upcr"] = _jsonable({
            "weights": fit.w,
            "group_names": group_names,
            "groups": encoded,
            "identifiability": identifiability,
            "fit_meta": fit.meta,
        })

    diagnostics = {
        "labels_seen_during_fit": False,
        "protocol_signature": matrix.protocol_signature,
        "kept_feature_names": [matrix.feature_names[index] for index in keep],
        "kept_groups": list(kept_groups),
        "kept_column_indices": keep.tolist(),
        "standardization": standardization,
        "orientation_flipped": flips,
        "fits": fits,
    }
    return scores, diagnostics


def fit_hierarchical(
    matrix: FeatureMatrix,
    solver: str,
    *,
    groups: Sequence[str] | None = None,
    dufs_seeds: Sequence[int] = DUFS_SEEDS,
    dufs_epochs: int = DUFS_EPOCHS,
    k: int = DUFS_K,
    lambda_: float = DUFS_LAMBDA,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the same solver within fixed groups and again across virtual experts.

    ``solver`` is exactly one of ``upcr``, ``iu_pcr``, or
    ``dufs_liu_pcr``.  This covers both four-band ``resid-core-32`` grouping
    and the twelve module-by-metric groups carried by ``lens-96``.
    """

    if solver not in {"upcr", "iu_pcr", "dufs_liu_pcr"}:
        raise ValueError("solver must be 'upcr', 'iu_pcr', or 'dufs_liu_pcr'")
    source_groups = tuple(groups) if groups is not None else matrix.groups
    if len(source_groups) != matrix.n_features:
        raise ValueError("groups must align to the original FeatureMatrix")
    X, keep, standardization = _standardize(matrix)
    if X.shape[1] < 3:
        raise ValueError("hierarchical fusion requires at least three features")
    selected_groups = tuple(str(source_groups[index]) for index in keep)
    encoded, group_names = _encode_groups(selected_groups, X.shape[1])
    if len(group_names) < 3:
        raise ValueError("hierarchical fusion requires at least three groups")
    F = X.T

    virtual_rows, inner_diagnostics = [], []
    inner_weights: list[np.ndarray] = []
    inner_flips: list[bool] = []
    inner_scales: list[float] = []
    group_indices: list[np.ndarray] = []
    for group_index, group_name in enumerate(group_names):
        indices = np.flatnonzero(encoded == group_index)
        group_indices.append(indices)
        weights, diagnostic = _fit_solver(
            F[indices],
            solver,
            dufs_seeds=dufs_seeds,
            dufs_epochs=dufs_epochs,
            k=k,
            lambda_=lambda_,
        )
        raw = weights @ F[indices]
        oriented, flipped = _anchor_orient(raw, matrix.risk_anchor)
        scale = float(np.std(oriented))
        standardized = (oriented - float(np.mean(oriented))) / scale
        virtual_rows.append(standardized)
        inner_weights.append(weights)
        inner_flips.append(flipped)
        inner_scales.append(scale)
        inner_diagnostics.append({
            "group": group_name,
            "feature_indices_after_drop": indices.tolist(),
            "feature_names": [matrix.feature_names[int(keep[index])] for index in indices],
            "orientation_flipped": flipped,
            "score_scale": scale,
            "fit": _jsonable(diagnostic),
        })

    virtual_F = np.asarray(virtual_rows, dtype=float)
    outer_weights, outer_diagnostic = _fit_solver(
        virtual_F,
        solver,
        dufs_seeds=dufs_seeds,
        dufs_epochs=dufs_epochs,
        k=k,
        lambda_=lambda_,
    )
    raw_outer = outer_weights @ virtual_F
    score, outer_flipped = _anchor_orient(raw_outer, matrix.risk_anchor)

    folded_kept = np.zeros(F.shape[0], dtype=float)
    outer_sign = -1.0 if outer_flipped else 1.0
    for group_index, indices in enumerate(group_indices):
        inner_sign = -1.0 if inner_flips[group_index] else 1.0
        folded_kept[indices] = (
            outer_sign
            * outer_weights[group_index]
            * inner_sign
            * inner_weights[group_index]
            / inner_scales[group_index]
        )
    folded_full = np.zeros(matrix.n_features, dtype=float)
    folded_full[keep] = folded_kept
    diagnostics = {
        "labels_seen_during_fit": False,
        "protocol_signature": matrix.protocol_signature,
        "solver": solver,
        "group_names": group_names,
        "groups_after_drop": encoded.tolist(),
        "n_groups": len(group_names),
        "standardization": standardization,
        "kept_column_indices": keep.tolist(),
        "inner": inner_diagnostics,
        "outer": _jsonable(outer_diagnostic),
        "outer_weights": outer_weights.tolist(),
        "outer_orientation_flipped": outer_flipped,
        "folded_feature_weights": folded_full.tolist(),
    }
    return score, diagnostics


def assert_no_label_fitting_signatures() -> None:
    """Mechanical leakage gate used by the benchmark and its unit tests."""

    fitting_functions = (
        fit_controls,
        fit_core_spectral,
        fit_dependency_methods,
        fit_hierarchical,
        fit_haloscope_direct_proxy,
    )
    failures = {}
    for function in fitting_functions:
        names = {name.lower() for name in inspect.signature(function).parameters}
        overlap = sorted(names & _FORBIDDEN_FIELD_NAMES)
        if overlap:
            failures[function.__name__] = overlap
    if failures:
        raise AssertionError(f"label-like fitting parameters found: {failures}")


__all__ = [
    "ALL_LAYERS",
    "SPACED_LAYERS",
    "LATE_LAYERS",
    "FIXED_BANDS",
    "all_layers",
    "spaced_layers",
    "late_layers",
    "fixed_bands",
    "CORE_METRICS",
    "DEPLOYED_UPCR_FIT",
    "DUFS_SEEDS",
    "DUFS_EPOCHS",
    "DUFS_K",
    "DUFS_LAMBDA",
    "LayerCell",
    "FeatureMatrix",
    "assert_same_protocol",
    "validate_and_join",
    "entropy_agreement_gate",
    "load_evaluation_labels",
    "extract_resid_core",
    "extract_lens_grid",
    "extract_lens96",
    "extract_trilens_entropy",
    "extract_dola_kl_proxy",
    "extract_haloscope_projection",
    "fit_haloscope_direct_proxy",
    "extract_geometry",
    "residualize_token_length",
    "fit_controls",
    "fit_core_spectral",
    "fit_dependency_methods",
    "fit_hierarchical",
    "assert_no_label_fitting_signatures",
]
