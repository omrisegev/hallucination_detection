"""Evidence-aware views of the frozen 30-feature mixed-v2 contract.

The answer is fixed across RAG conditions.  This module extracts the same 30
global trace features from every condition, fits the mixed-v2 transform on
full-context rows only, and reuses that coordinate system for no-context and
leave-one-chunk-out traces.  It accepts no labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .dufs_liu_feature_contract import dufs_liu_mixed_v2_matrix
from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from .feature_utils import extract_all_features
from .ragtruth_evidence_contrast import FeatureTable, RagDataset
from .repeated_measurement_reliability import FixedMixedV2Transformer
from .repgrid_scoring import (
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)


ORIGINAL_FEATURES = tuple(CONFIDENCE_FEATURE_SIGNS_V1)
CONTRACT_VERSION = "ragtruth-original30-evidence-aware-v1-2026-08-10"
EPS = 1e-12


@dataclass(frozen=True)
class MixedV2EvidenceTensor:
    """Ragged condition tensor in raw and shared mixed-v2 coordinates."""

    response_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    task_types: tuple[str, ...]
    sources: tuple[str, ...]
    response_lengths: np.ndarray
    context_lengths: np.ndarray
    chunk_counts: np.ndarray
    feature_names: tuple[str, ...]
    raw_full: np.ndarray
    raw_noctx: np.ndarray
    raw_loo: tuple[np.ndarray, ...]
    loo_indexes: tuple[np.ndarray, ...]
    mixed_full: np.ndarray
    mixed_noctx: np.ndarray
    mixed_loo: tuple[np.ndarray, ...]
    transformer: FixedMixedV2Transformer
    exact_full_contract_error: float


@dataclass(frozen=True)
class VariantMatrix:
    """One response-by-view fusion matrix with auditable column provenance."""

    name: str
    values: np.ndarray
    feature_names: tuple[str, ...]
    block_names: tuple[str, ...]
    base_features: tuple[str, ...]
    permutable: tuple[bool, ...]


def trace_original30(trace: Any) -> dict[str, float]:
    """Extract exactly the canonical 30 raw features from one condition."""
    values = extract_all_features(
        trace.entropy,
        spilled_energies=-np.asarray(trace.target_logprob, dtype=float),
        allow_short=True,
    ) or {}
    values.update(energy_features_from_logsumexp(trace.logsumexp))
    top = {"ids": trace.top_ids, "logprobs": trace.top_logprobs}
    values.update(logprob_features(top))
    values.update(logprob_features_extended(top))
    return {name: float(values.get(name, np.nan)) for name in ORIGINAL_FEATURES}


def feature_availability(dataset: RagDataset) -> list[dict[str, Any]]:
    """Audit all 30 features on the response-level LOO cohort."""
    counts: dict[tuple[str, str], list[int]] = {}
    for response in dataset.responses:
        loo_names = sorted(
            (name for name in response.conditions if name.startswith("loo_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        if not loo_names:
            continue
        for group, names in (
            ("full", ("full",)),
            ("noctx", ("noctx",)),
            ("loo", tuple(loo_names)),
        ):
            for condition in names:
                row = trace_original30(response.conditions[condition])
                for feature in ORIGINAL_FEATURES:
                    key = (group, feature)
                    item = counts.setdefault(key, [0, 0])
                    item[1] += 1
                    item[0] += int(np.isfinite(row[feature]))
    rows = []
    for group in ("full", "noctx", "loo"):
        for feature in ORIGINAL_FEATURES:
            finite, total = counts.get((group, feature), [0, 0])
            rows.append({
                "condition_group": group,
                "feature": feature,
                "finite": int(finite),
                "total": int(total),
                "availability": float(finite / total) if total else 0.0,
                "fully_available": bool(total and finite == total),
            })
    return rows


def _row_vector(trace: Any) -> np.ndarray:
    row = trace_original30(trace)
    return np.asarray([row[name] for name in ORIGINAL_FEATURES], dtype=float)


def build_mixed_v2_evidence_tensor(dataset: RagDataset) -> MixedV2EvidenceTensor:
    """Build a label-free tensor and fail closed on any missing raw feature."""
    responses = [
        response for response in dataset.responses
        if any(name.startswith("loo_") for name in response.conditions)
    ]
    responses.sort(key=lambda item: int(item.response_id))
    if not responses:
        raise ValueError("the dataset contains no leave-one-out responses")

    raw_full: list[np.ndarray] = []
    raw_noctx: list[np.ndarray] = []
    raw_loo: list[np.ndarray] = []
    loo_indexes: list[np.ndarray] = []
    missing: list[str] = []
    for response in responses:
        full = _row_vector(response.conditions["full"])
        noctx = _row_vector(response.conditions["noctx"])
        loo_names = sorted(
            (name for name in response.conditions if name.startswith("loo_")),
            key=lambda name: int(name.split("_", 1)[1]),
        )
        loo = np.vstack([_row_vector(response.conditions[name]) for name in loo_names])
        for condition, values in (("full", full), ("noctx", noctx)):
            for index in np.flatnonzero(~np.isfinite(values)):
                missing.append(
                    f"{response.response_id}/{condition}/{ORIGINAL_FEATURES[int(index)]}"
                )
        for row_index, values in enumerate(loo):
            for index in np.flatnonzero(~np.isfinite(values)):
                missing.append(
                    f"{response.response_id}/{loo_names[row_index]}/"
                    f"{ORIGINAL_FEATURES[int(index)]}"
                )
        raw_full.append(full)
        raw_noctx.append(noctx)
        raw_loo.append(loo)
        loo_indexes.append(np.asarray([
            int(name.split("_", 1)[1]) for name in loo_names
        ], dtype=int))
    if missing:
        preview = ", ".join(missing[:12])
        raise ValueError(
            f"{len(missing)} original-30 values are unavailable; no imputation is "
            f"allowed. First entries: {preview}"
        )

    full_matrix = np.vstack(raw_full)
    noctx_matrix = np.vstack(raw_noctx)
    transformer = FixedMixedV2Transformer.fit(full_matrix, ORIGINAL_FEATURES)
    reference, reference_names, _ = dufs_liu_mixed_v2_matrix(
        full_matrix, ORIGINAL_FEATURES
    )
    if reference_names != ORIGINAL_FEATURES:
        raise RuntimeError("the frozen mixed-v2 feature order changed")
    exact_error = float(np.max(np.abs(reference - transformer.training_output)))
    if exact_error > 1e-10:
        raise RuntimeError(
            f"shared transform does not reproduce full mixed-v2: {exact_error:.3g}"
        )
    transformed_loo = tuple(transformer.transform(values) for values in raw_loo)
    return MixedV2EvidenceTensor(
        response_ids=tuple(response.response_id for response in responses),
        source_ids=tuple(response.source_id for response in responses),
        task_types=tuple(response.task_type for response in responses),
        sources=tuple(response.source for response in responses),
        response_lengths=np.asarray([
            len(response.conditions["full"].token_ids) for response in responses
        ], dtype=int),
        context_lengths=np.asarray([
            response.conditions["full"].prompt_len for response in responses
        ], dtype=int),
        chunk_counts=np.asarray([len(values) for values in raw_loo], dtype=int),
        feature_names=ORIGINAL_FEATURES,
        raw_full=full_matrix,
        raw_noctx=noctx_matrix,
        raw_loo=tuple(raw_loo),
        loo_indexes=tuple(loo_indexes),
        mixed_full=transformer.training_output,
        mixed_noctx=transformer.transform(noctx_matrix),
        mixed_loo=transformed_loo,
        transformer=transformer,
        exact_full_contract_error=exact_error,
    )


def _block(
    values: np.ndarray,
    block: str,
    names: tuple[str, ...],
    *,
    permutable: bool,
) -> tuple[np.ndarray, list[str], list[str], list[str], list[bool]]:
    return (
        np.asarray(values, dtype=float),
        [f"{block}::{name}" for name in names],
        [block] * len(names),
        list(names),
        [bool(permutable)] * len(names),
    )


def _combine(name: str, blocks: list[tuple[Any, ...]]) -> VariantMatrix:
    values = np.column_stack([item[0] for item in blocks])
    feature_names = tuple(value for item in blocks for value in item[1])
    block_names = tuple(value for item in blocks for value in item[2])
    base_features = tuple(value for item in blocks for value in item[3])
    permutable = tuple(value for item in blocks for value in item[4])
    if values.shape[1] != len(feature_names) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid matrix for {name}")
    return VariantMatrix(
        name, values, feature_names, block_names, base_features, permutable
    )


def build_variant_matrices(
    tensor: MixedV2EvidenceTensor,
    ec_table: FeatureTable,
) -> dict[str, VariantMatrix]:
    """Construct the four frozen comparison matrices from the same base tensor."""
    names = tensor.feature_names
    full = _block(tensor.mixed_full, "full", names, permutable=False)
    noctx_delta = tensor.mixed_full - tensor.mixed_noctx
    noctx = _block(noctx_delta, "noctx_drop", names, permutable=True)

    maxima, top_two, positive_means, negative_std = [], [], [], []
    for full_row, loo_rows in zip(tensor.mixed_full, tensor.mixed_loo):
        drops = full_row[None, :] - loo_rows
        ordered = np.sort(drops, axis=0)
        largest = ordered[-1]
        second = ordered[-2] if len(ordered) > 1 else ordered[-1]
        positive = drops > 0.0
        positive_count = positive.sum(axis=0)
        positive_sum = np.where(positive, drops, 0.0).sum(axis=0)
        maxima.append(largest)
        top_two.append(0.5 * (largest + second))
        positive_means.append(np.divide(
            positive_sum,
            positive_count,
            out=np.zeros_like(positive_sum),
            where=positive_count > 0,
        ))
        negative_std.append(-drops.std(axis=0))

    loo_blocks = [
        _block(np.vstack(maxima), "loo_max_drop", names, permutable=True),
        _block(np.vstack(top_two), "loo_top2_mean_drop", names, permutable=True),
        _block(np.vstack(positive_means), "loo_positive_mean_drop", names, permutable=True),
        _block(np.vstack(negative_std), "loo_negative_std", names, permutable=True),
    ]
    variants = {
        "original30_full": _combine("original30_full", [full]),
        "original30_noctx": _combine("original30_noctx", [full, noctx]),
        "original30_loo": _combine(
            "original30_loo", [full, noctx, *loo_blocks]
        ),
    }

    ec_lookup = {sample_id: index for index, sample_id in enumerate(ec_table.sample_ids)}
    if set(tensor.response_ids) != set(ec_lookup):
        raise ValueError("EC-full-v1 and original-30 LOO cohorts disagree")
    ec_values = np.vstack([
        ec_table.values[ec_lookup[response_id]] for response_id in tensor.response_ids
    ])
    ec_intrinsic_count = 4
    ec_intrinsic_names = tuple(ec_table.feature_names[:ec_intrinsic_count])
    ec_evidence_names = tuple(ec_table.feature_names[ec_intrinsic_count:])
    ec_blocks = [
        _block(
            ec_values[:, :ec_intrinsic_count],
            "ec_intrinsic",
            ec_intrinsic_names,
            permutable=False,
        ),
        _block(
            ec_values[:, ec_intrinsic_count:],
            "ec_evidence",
            ec_evidence_names,
            permutable=True,
        ),
    ]
    variants["hybrid"] = _combine(
        "hybrid", [full, noctx, *loo_blocks, *ec_blocks]
    )
    return variants


def permute_evidence_blocks(
    variant: VariantMatrix,
    task_types: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Break within-response evidence pairing while preserving task marginals."""
    task_types = np.asarray(task_types).astype(str)
    if len(task_types) != len(variant.values):
        raise ValueError("task vector and variant rows disagree")
    output = variant.values.copy()
    rng = np.random.default_rng(int(seed))
    blocks = sorted({
        block for block, flag in zip(variant.block_names, variant.permutable) if flag
    })
    for block in blocks:
        columns = np.flatnonzero(np.asarray([
            name == block for name in variant.block_names
        ], dtype=bool))
        for task in sorted(set(task_types)):
            rows = np.flatnonzero(task_types == task)
            output[np.ix_(rows, columns)] = output[np.ix_(rng.permutation(rows), columns)]
    return output


def condition_matrices(
    tensor: MixedV2EvidenceTensor,
) -> Mapping[str, np.ndarray]:
    """Thirty-column views used only to compare DUFS gates by condition."""
    return {
        "full": tensor.mixed_full,
        "noctx": tensor.mixed_noctx,
        "loo_mean": np.vstack([values.mean(axis=0) for values in tensor.mixed_loo]),
    }


def flatten_loo(
    values: tuple[np.ndarray, ...],
    indexes: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return concatenated values, response offsets and original LOO indexes."""
    offsets = np.zeros(len(values) + 1, dtype=int)
    offsets[1:] = np.cumsum([len(item) for item in values])
    return np.vstack(values), offsets, np.concatenate(indexes)


__all__ = [
    "CONTRACT_VERSION",
    "ORIGINAL_FEATURES",
    "MixedV2EvidenceTensor",
    "VariantMatrix",
    "build_mixed_v2_evidence_tensor",
    "build_variant_matrices",
    "condition_matrices",
    "feature_availability",
    "flatten_loo",
    "permute_evidence_blocks",
    "trace_original30",
]
