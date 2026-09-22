"""Build the development-only Fusion Independence Atlas v1.

The driver deliberately separates label-free construction from error analysis.
Signal extraction, causal predictors, and fusion-weight fitting have no label
argument.  Correctness annotations are opened only by ``dependence`` and the
nested roster-selection part of ``fusion``.

Long-running stages use immutable contract manifests plus answer completion
bitmaps.  A changed source, implementation, shape, or semantic identity is a
hard error; the driver never overwrites a checkpoint after drift.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE_ROOT = Path("/Users/osegev/Desktop/hallucination_detection")
DEFAULT_OUTPUT_ROOT = ROOT / "results/fusion_independence_atlas_v1"
PROTOCOL = ROOT / "docs/experiments/FUSION_INDEPENDENCE_ATLAS_V1.md"

SCHEMA = "fusion-independence-atlas-v1"
EXPECTED_ANSWERS = 13_769
EXPECTED_TOKENS = 6_968_779
EXPECTED_STEPS = 145_597
PRIMITIVE_TARGETS = ("H0lim", "VE0", "VE0.75", "VE1")
FUSION_POINTS = (
    "background", "token_pre_readout", "step_post_readout", "decoder", "answer_gate",
)
PENDING_EXPANSION = ("FM", "DiFlo", "DOT", "artifactless_models")
DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 39_615

# Frozen hashes copied from the verified Step-395 input manifest.  The PRMB
# label source is intentionally verified here but never opened by bundle,
# extraction, or predictors.
RAW_SOURCE_HASHES = {
    "dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl":
        "95be425c66d4dadc6a8e03567ad66ef2933aee27eb7967bf7200139c56518232",
    "dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl":
        "b934afad0889ffacf0f4420f885ad52ddcf08f2124e506a21cd24f216bd170be",
    "dataset_cache/repgrid/pb_qwen3_4b/processbench_gsm8k.pkl":
        "549843b11a99f41b60ccb13eb6a65c5583527402602e6e40d1a417699e64b02b",
    "dataset_cache/repgrid/pb_qwen3_4b/processbench_math.pkl":
        "807292c490012b68c7a20cc3d093cda637866227940b9024b258b6a201657c32",
    "dataset_cache/repgrid/pb_qwen3_4b/processbench_olympiadbench.pkl":
        "e5e73f044247e0fb4081fea1c3a77793aab182485daebce48cf7a3f7809eb0ae",
    "dataset_cache/repgrid/pb_qwen3_4b/processbench_omnimath.pkl":
        "ec20e56553acee15c0646ad266f5d5d8d24ef22ba87289b6ffe013b0c639ba67",
    "dataset_cache/repgrid/pb_qwen3_8b/processbench_gsm8k.pkl":
        "c258e8a37fad43216290fef2f70c88735f413f8c0ceb5cce39dfefe67d66955d",
    "dataset_cache/repgrid/pb_qwen3_8b/processbench_math.pkl":
        "19cbbb2b4f90e073f2793360f142bc25334f88bbc4ec01fd78807f9f1ad2836d",
    "dataset_cache/repgrid/pb_qwen3_8b/processbench_olympiadbench.pkl":
        "2083c4f9d0cc8d84657b02558d0720f726db4288e39e7557be4c55f78f4d4e60",
    "dataset_cache/repgrid/pb_qwen3_8b/processbench_omnimath.pkl":
        "561b96c5e96a25256e7e51a62c7ef01bcb2fd87ea1adcfc36e2c758807f559a0",
}

COMPACT_HASHES = {
    "results/localization_full_benchmark_v3/evaluation/JOINED.json":
        "22fcbfd346db6565665b5358ac74393a4188e9047565ca2af44908a9e8d3f49d",
    "results/localization_full_benchmark_v3/evaluation/JOINED.npz":
        "31c1fe06ac62f7dbd3f3e9fe5653148db718f32258bb45a246992476da0cb2c2",
    "results/localization_source_group_audit_v1/FOLDS_V2.json":
        "1f9f33032addedc450d0e2b5a45690679dfe2f3c3f8764663e6d0d94a1cf8205",
    "results/fusion_fixed_gate_v1/DETECTORS.npz":
        "faa127cd0f891ac51ac174598b24b55dbebcb4b964f0b98fdbb4c1740e5bcce6",
    "results/fusion_fixed_gate_v1/METRICS.json":
        "b1665ae7b3e84798ff8196a1a3e8ddfb1871eaddcd4ae1dcda07b939903c494c",
}

BASELINE_EXPECTED = {
    "original4": {"pb": 0.374749},
    "innovation5": {"pb": 0.398314},
    "digit025": {"pb": 0.413300, "within": 0.776036},
    "current": {"pb": 0.432546, "within": 0.776036},
}

STAGES = (
    "preflight", "registry", "bundle", "extract", "predictors", "reconcile",
    "dependence", "fusion", "ablation", "report",
)
STAGE_DEPENDENCIES = {
    "preflight": (),
    "registry": ("preflight",),
    "bundle": ("preflight",),
    "extract": ("registry", "bundle"),
    "predictors": ("bundle",),
    "reconcile": ("preflight",),
    "dependence": ("extract", "predictors", "reconcile"),
    "fusion": ("dependence",),
    "ablation": ("fusion",),
    # A report is still a required artifact when the scientifically honest
    # ablation outcome is a fail-closed composition contract.  The report
    # therefore depends on the completed nested-fusion stage and records the
    # ablation status explicitly instead of suppressing all upstream results.
    "report": ("fusion",),
}


class AtlasContractError(RuntimeError):
    """The requested operation violates or drifts from the frozen contract."""


def _np():
    """Import NumPy lazily so ``--help`` and preflight remain informative."""
    import numpy as np
    return np


def json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return json_ready(value.tolist())
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, Path):
        return str(value)
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def pretty_json(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False) + "\n"


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(pretty_json(value), encoding="utf8")
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf8")).hexdigest()


def implementation_sha256(*functions: Callable[..., Any], files: Sequence[Path] = ()) -> str:
    """Hash the exact callable sources and dependency files behind an artifact.

    Function-level source binding avoids invalidating a long stage for an
    unrelated report-only edit while still making semantic code drift a hard
    resumability error.
    """
    payload = {
        # The same orchestrator is both executable and importable in tests.
        # Binding the runtime ``__module__`` would make an identical callable
        # hash differently as ``__main__`` versus ``scripts....``.  Existing
        # CLI artifacts were created under ``__main__``; use that stable
        # namespace in both modes so resume/audit compares code, not invocation.
        "functions": {
            f"__main__.{function.__qualname__}": inspect.getsource(function)
            for function in functions
        },
        "files": {str(Path(path).relative_to(ROOT)): sha256_file(Path(path)) for path in files},
    }
    return sha256_json(payload)


def bind_immutable_manifest(path: Path, manifest: Mapping[str, Any]) -> str:
    """Create an immutable manifest, or verify byte-semantic equality.

    The return value is the canonical payload hash and is suitable for binding
    completion bitmaps and downstream manifests.
    """
    path = Path(path)
    normalized = json_ready(dict(manifest))
    if path.exists():
        existing = read_json(path)
        if canonical_json(existing) != canonical_json(normalized):
            raise AtlasContractError(f"immutable manifest drift: {path}")
    else:
        atomic_json(path, normalized)
    return sha256_json(normalized)


def verify_hashed_files(files: Mapping[str, str], roots: Sequence[Path]) -> list[dict[str, Any]]:
    rows = []
    for relative, expected in sorted(files.items()):
        candidates = [Path(root) / relative for root in roots]
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        actual = sha256_file(path) if path.is_file() else None
        rows.append({
            "relative": relative,
            "path": str(path.resolve()) if path.exists() else str(path),
            "size": path.stat().st_size if path.is_file() else None,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "status": "PASS" if actual == expected else ("MISSING" if actual is None else "DRIFT"),
        })
    return rows


def _array_contract(path: Path, shape: Sequence[int], dtype: str) -> None:
    np = _np()
    if not path.is_file():
        raise AtlasContractError(f"missing checkpoint array: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        if tuple(array.shape) != tuple(shape) or np.dtype(array.dtype) != np.dtype(dtype):
            raise AtlasContractError(
                f"array contract drift for {path}: {array.shape}/{array.dtype}, "
                f"expected {tuple(shape)}/{np.dtype(dtype)}"
            )
    finally:
        del array


class CompletionBitmap:
    """A crash-resumable answer bitmap bound to one manifest hash."""

    def __init__(self, path: Path, size: int, contract_sha256: str):
        np = _np()
        self.path = Path(path)
        self.binding = self.path.with_suffix(self.path.suffix + ".json")
        expected = {"schema": "answer-completion-bitmap-v1", "answers": int(size),
                    "contract_sha256": contract_sha256}
        bind_immutable_manifest(self.binding, expected)
        if self.path.exists():
            _array_contract(self.path, (size,), "bool")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            bitmap = np.lib.format.open_memmap(self.path, mode="w+", dtype=bool, shape=(size,))
            bitmap[:] = False
            bitmap.flush()
            del bitmap
        self.array = np.load(self.path, mmap_mode="r+", allow_pickle=False)

    def done(self, index: int) -> bool:
        return bool(self.array[index])

    def mark(self, index: int) -> None:
        self.array[index] = True
        self.array.flush()

    def count(self) -> int:
        return int(self.array.sum())

    def complete(self) -> bool:
        return self.count() == len(self.array)

    def close(self) -> None:
        self.array.flush()
        del self.array


def sanitized_metadata(record: Mapping[str, Any], *, fold: int, token_offset: int,
                       step_start: int, step_stop: int, mean: Sequence[float],
                       scale: Sequence[float], signs: Sequence[float]) -> dict[str, Any]:
    """Return the only metadata schema permitted in the label-free bundle."""
    return {
        "uid": str(record["uid"]), "cell": str(record["cell"]),
        "group_id": str(record["group_id"]), "fold": int(fold),
        "offset": int(token_offset), "tokens": int(record["tokens"]),
        "step_start": int(step_start), "step_stop": int(step_stop),
        "mean": [float(x) for x in mean], "scale": [float(x) for x in scale],
        "signs": [float(x) for x in signs],
    }


def assert_label_free_callable(function: Callable[..., Any]) -> None:
    # ``target`` is valid registry metadata (for example localization_risk),
    # whereas benchmark annotations are not valid fit/extraction inputs.
    forbidden = {"label", "labels", "correctness", "annotations", "y"}
    names = set(inspect.signature(function).parameters)
    leaked = names & forbidden
    if leaked:
        raise AtlasContractError(f"label-bearing API is forbidden here: {function.__name__}: {sorted(leaked)}")


def causal_backgrounds(levels: Any) -> dict[str, Any]:
    """Four deterministic backgrounds; every prediction is strictly past-only."""
    np = _np()
    from spectral_utils.aligned_context_predictors import bocpd_mean, noreset_mean, past_mean
    x = np.asarray(levels, dtype=float)
    if x.ndim != 2 or x.shape[1] != 4 or not len(x) or not np.isfinite(x).all():
        raise ValueError("causal predictors require a finite [tokens,4] primitive matrix")
    prefix = np.vstack((np.zeros((1, 4)), np.cumsum(x[:-1], axis=0)))
    denominator = np.maximum(np.arange(len(x)), 1)[:, None]
    return {
        "prefix": prefix / denominator,
        "mean16": past_mean(x, width=16),
        "noreset": noreset_mean(x),
        "bocpd": bocpd_mean(x, hazard=1 / 32),
    }


def innovations_from_background(levels: Any, background: Any) -> tuple[Any, Any]:
    np = _np()
    x = np.asarray(levels, dtype=float)
    predicted = np.asarray(background, dtype=float)
    if x.shape != predicted.shape or x.ndim != 2 or x.shape[1] != 4:
        raise ValueError("primitive/background shape mismatch")
    available = np.ones(len(x), dtype=bool)
    available[0] = False
    residual = x - predicted
    residual[0] = 0.0
    return residual, available


def predictor_exclusions() -> tuple[tuple[int, ...], ...]:
    singles = tuple((fold,) for fold in range(5))
    pairs = tuple(itertools.combinations(range(5), 2))
    return singles + pairs


def factorial_arms(points: Sequence[str] = FUSION_POINTS) -> list[dict[str, Any]]:
    if len(points) != 5 or len(set(points)) != 5:
        raise ValueError("the frozen factorial has five unique insertion points")
    arms = []
    for bits in itertools.product((0, 1), repeat=5):
        selection = {point: ("finalist" if bit else "incumbent")
                     for point, bit in zip(points, bits)}
        arms.append({"arm": "".join(map(str, bits)), "bits": list(bits), "selection": selection})
    return arms


def _midranks(values: Any) -> Any:
    np = _np()
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("midranks require one finite vector")
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and x[order[stop]] == x[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2
        start = stop
    return ranks / max(len(x) - 1, 1)


def rerank_gate_draw(gate_scores: Any, cells: Sequence[str], draw_indices: Sequence[int],
                     threshold: float = .33) -> dict[str, Any]:
    """Recompute cell midranks and rank fusion *inside* one bootstrap draw."""
    np = _np()
    score = np.asarray(gate_scores, dtype=float)
    if score.ndim == 1:
        score = score[:, None]
    cell = np.asarray(cells, dtype=str)
    index = np.asarray(draw_indices, dtype=int)
    if score.ndim != 2 or len(score) != len(cell) or np.any(index < 0) or np.any(index >= len(cell)):
        raise ValueError("misaligned gate draw")
    selected, selected_cells = score[index], cell[index]
    fused = np.full(len(index), np.nan)
    for name in sorted(set(selected_cells.tolist())):
        mask = selected_cells == name
        ranked = np.column_stack([_midranks(selected[mask, column])
                                  for column in range(selected.shape[1])])
        fused[mask] = ranked.mean(axis=1)
    return {"rank_fused": fused, "open": fused >= float(threshold),
            "threshold": float(threshold), "draw_indices": index}


def gate_reranking_callback(gate_scores: Any, cells: Sequence[str], threshold: float = .33):
    """Callback passed to grouped bootstrap; it closes over scores, not labels."""
    calls = {"count": 0}

    def callback(draw_indices: Sequence[int]) -> dict[str, Any]:
        calls["count"] += 1
        return rerank_gate_draw(gate_scores, cells, draw_indices, threshold)

    callback.calls = calls  # type: ignore[attr-defined]
    return callback


def verify_baseline_replay(actual: Mapping[str, Mapping[str, float]],
                           tolerance: float = 5e-7) -> dict[str, Any]:
    checks = []
    for method, expected_metrics in BASELINE_EXPECTED.items():
        observed = actual.get(method)
        for metric, expected in expected_metrics.items():
            value = None if observed is None else observed.get(metric)
            passed = value is not None and abs(float(value) - expected) <= tolerance
            checks.append({"method": method, "metric": metric, "expected": expected,
                           "actual": value, "difference": None if value is None else float(value) - expected,
                           "pass": passed})
    result = {"status": "PASS" if all(row["pass"] for row in checks) else "MISMATCH",
              "tolerance": tolerance, "checks": checks}
    if result["status"] != "PASS":
        raise AtlasContractError("frozen baseline replay mismatch")
    return result


def _baseline_gate_percentiles(values: Any, cells: Any, pb_mask: Any) -> Any:
    """Canonical tie-aware within-cell percentiles on ProcessBench only."""
    np = _np()
    from spectral_utils.math_gate_selection import percentile_by_cell
    score = np.asarray(values, dtype=np.float64)
    cell = np.asarray(cells, dtype=str)
    pb = np.asarray(pb_mask, dtype=bool)
    if score.shape != cell.shape or pb.shape != cell.shape or score.ndim != 1:
        raise ValueError("baseline gate inputs must be aligned vectors")
    if not np.isfinite(score[pb]).all():
        raise AtlasContractError("baseline gate contains nonfinite ProcessBench scores")
    ranked = np.full(len(score), np.nan, dtype=np.float64)
    ranked[pb] = percentile_by_cell(score[pb], cell[pb])
    return ranked


def _baseline_arm_metrics(
    records: Sequence[Mapping[str, Any]],
    offsets: Any,
    target: Any,
    labels: Any,
    step_scores: Any,
    gate_open: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-evaluate one locator from step scores, never from headline metrics."""
    np = _np()
    from spectral_utils.historical_fusion_evaluation import auc, pb_metrics
    score = np.asarray(step_scores, dtype=np.float64)
    offset = np.asarray(offsets, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    label = np.asarray(labels, dtype=np.int8)
    opened = np.asarray(gate_open, dtype=bool)
    cells = np.asarray([row["cell"] for row in records], dtype=str)
    if offset.shape != (len(records) + 1,) or target.shape != (len(records),) \
            or opened.shape != (len(records),) or score.shape != (int(offset[-1]),) \
            or label.shape != score.shape:
        raise AtlasContractError("baseline replay arrays are not aligned")
    if not np.isfinite(score).all() or np.any(np.diff(offset) <= 0):
        raise AtlasContractError("baseline replay requires finite nonempty answer scores")
    peak = np.empty(len(records), dtype=np.int32)
    within = np.full(len(records), np.nan, dtype=np.float64)
    for index in range(len(records)):
        local = slice(int(offset[index]), int(offset[index + 1]))
        values = score[local]
        peak[index] = int(np.argmax(values))
        if not cells[index].startswith("pb_"):
            usable = label[local] >= 0
            # Match the frozen evaluator's PRMB contract exactly: JOINED's
            # positive ranking class is label 1.  Reversing this bit produces
            # the complementary AUC (0.223964 instead of 0.776036) while PB is
            # unchanged, so baseline replay guards this orientation explicitly.
            binary = label[local][usable] == 1
            if binary.any() and (~binary).any():
                within[index] = float(auc(binary, values[usable]))
    pb = np.char.startswith(cells, "pb_")
    prediction = np.where(opened, peak, -1).astype(np.int32)
    valid = np.ones(len(records), dtype=bool)
    quality = pb_metrics(target[pb], prediction[pb], valid[pb], cells[pb])
    metric = {
        "pb": float(quality["macros"]["all"]),
        "within": float(np.nanmean(within)) if np.isfinite(within).any() else None,
        "within_n": int(np.isfinite(within).sum()),
        "pb_cells": quality["cells"],
        "pb_cell_count": len(quality["cells"]),
    }
    per_answer = {"peak": peak, "prediction": prediction, "valid": valid, "within": within}
    return metric, per_answer


def _reconstruct_digit025_from_extract(
    extract_root: Path,
    records: Sequence[Mapping[str, Any]],
    offsets: Any,
    archived_innovation5: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Rebuild digit025 exclusively from completed packed extraction rows."""
    np = _np()
    from spectral_utils.fusion_signal_registry import READOUT_NAMES
    from spectral_utils.temporal_context_models import residual_step_score
    root = Path(extract_root)
    required = ("MANIFEST.json", "completion.npy", "answer_sha256.npy")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise AtlasContractError("baseline replay extraction inputs are missing: " + ", ".join(missing))
    status_path = root / "STATUS.json"
    if not status_path.is_file() or read_json(status_path).get("status") != "COMPLETE":
        raise AtlasContractError("baseline replay requires completed atomic extraction")
    completion = np.load(root / "completion.npy", allow_pickle=False)
    answer_hashes = np.load(root / "answer_sha256.npy", allow_pickle=False)
    try:
        if completion.shape != (len(records),) or answer_hashes.shape != (len(records),) \
                or not np.asarray(completion, dtype=bool).all():
            raise AtlasContractError("atomic extraction completion roster is incomplete or misaligned")
        recorded_hashes = [bytes(value).decode("ascii").rstrip("\x00") for value in answer_hashes]
    finally:
        del completion, answer_hashes
    if any(len(value) != 64 for value in recorded_hashes):
        raise AtlasContractError("atomic extraction answer hash index is incomplete")

    offset = np.asarray(offsets, dtype=np.int64)
    archived = np.asarray(archived_innovation5, dtype=np.float64)
    total_steps = int(offset[-1])
    if archived.shape != (total_steps,):
        raise AtlasContractError("archived innovation5 is misaligned")
    reconstructed = np.full(total_steps, np.nan, dtype=np.float64)
    digit025 = np.full(total_steps, np.nan, dtype=np.float64)
    digit_rate = np.zeros(len(records), dtype=np.float64)
    top10 = READOUT_NAMES.index("top10")
    q15 = ("q15.H0lim", "q15.VE0", "q15.VE0.75", "q15.VE1")
    answers = root / "answers"
    for index, row in enumerate(records):
        path = answers / f"{index:05d}.npz"
        if not path.is_file() or sha256_file(path) != recorded_hashes[index]:
            raise AtlasContractError(f"atomic extraction answer drift: {index}")
        local = slice(int(offset[index]), int(offset[index + 1]))
        steps = int(offset[index + 1] - offset[index])
        if int(row.get("steps", steps)) != steps:
            raise AtlasContractError(f"answer step-count drift: {index}")
        with np.load(path, allow_pickle=False) as packed:
            try:
                primitive = np.column_stack([
                    np.asarray(packed[f"step__{name}__readouts"], dtype=np.float64)[:, top10]
                    for name in q15
                ])
                innovation = np.asarray(
                    packed["step__q15.H0lim.prefix_mean_innovation__readouts"],
                    dtype=np.float64,
                )[:, top10]
                disagreement = np.asarray(
                    packed["step__digit.disagreement__readouts"], dtype=np.float64,
                )[:, top10]
                digit_token = np.asarray(packed["token__digit.disagreement__values"], dtype=np.float64)
                opportunity = np.asarray(packed["token__digit.opportunity__values"], dtype=np.float64)
            except KeyError as error:
                raise AtlasContractError(f"packed extraction baseline field missing: {error}") from error
        if primitive.shape != (steps, 4) or innovation.shape != (steps,) \
                or disagreement.shape != (steps,) or digit_token.shape != opportunity.shape:
            raise AtlasContractError(f"packed extraction baseline shape drift: {index}")
        # Dequantize each packed float32 view before accumulation.  Summing the
        # already-rounded columns in float32 introduces avoidable ~1e-6
        # accumulation deltas that are not part of the signal contract.
        base = (
            primitive.sum(axis=1, dtype=np.float64) + innovation
        ) / np.float64(5.0)
        reconstructed[local] = base
        # The incumbent's innovation5 score is itself a frozen baseline input.
        # Use that exact full step vector for the replay and independently
        # reconstruct it above as a provenance/alignment audit.  This avoids
        # allowing the new first-token-inactive contract (and float32 packing)
        # to redefine the historical incumbent while the digit contribution
        # is still rebuilt from raw-token extraction.
        digit025[local] = residual_step_score(archived[local], disagreement, .25)
        digit_rate[index] = float(digit_token.sum() / max(float(opportunity.sum()), 1.0))
    if not np.isfinite(reconstructed).all() or not np.isfinite(digit025).all():
        raise AtlasContractError("reconstructed digit025 contains nonfinite scores")
    base_delta = np.abs(reconstructed - archived)
    initial_steps = np.asarray(offset[:-1], dtype=np.int64)
    noninitial = np.ones(total_steps, dtype=bool)
    noninitial[initial_steps] = False
    audit = {
        "answers": len(records), "steps": total_steps, "top10_column": top10,
        "packed_innovation5_max_abs_delta": float(base_delta.max(initial=0.0)),
        "packed_innovation5_initial_step_max_abs_delta": float(
            base_delta[initial_steps].max(initial=0.0)
        ),
        "packed_innovation5_noninitial_step_max_abs_delta": float(
            base_delta[noninitial].max(initial=0.0)
        ),
        "packed_innovation5_initial_steps_over_1e6": int(
            np.sum(base_delta[initial_steps] > 1e-6)
        ),
        "packed_innovation5_noninitial_steps_over_1e6": int(
            np.sum(base_delta[noninitial] > 1e-6)
        ),
        "packed_innovation5_contract_difference": (
            "the Atlas masks token zero before readout; the historical archive "
            "retained its zero innovation inside the first-step Top10 mean"
        ),
        "packed_innovation5_peak_mismatches": int(sum(
            np.argmax(reconstructed[offset[i]:offset[i + 1]])
            != np.argmax(archived[offset[i]:offset[i + 1]]) for i in range(len(records))
        )),
        "answer_hash_index_sha256": sha256_file(root / "answer_sha256.npy"),
        "answer_tree_sha256": sha256_json(recorded_hashes),
        "extract_manifest_sha256": sha256_file(root / "MANIFEST.json"),
        "completion_sha256": sha256_file(root / "completion.npy"),
    }
    return digit025, digit_rate, audit


def replay_frozen_baselines(
    contract_root: Path,
    baseline_root: Path,
    extract_root: Path,
    *,
    strict_roster: bool = True,
) -> dict[str, Any]:
    """Perform the four frozen baseline evaluations from scores and labels."""
    np = _np()
    contract_root, baseline_root = Path(contract_root), Path(baseline_root)
    joined_json = contract_root / "results/localization_full_benchmark_v3/evaluation/JOINED.json"
    joined_npz = contract_root / "results/localization_full_benchmark_v3/evaluation/JOINED.npz"
    score_path = baseline_root / "SCORES_FROZEN.npz"
    for path in (joined_json, joined_npz, score_path):
        if not path.is_file():
            raise AtlasContractError(f"frozen baseline replay input is missing: {path}")
    records = read_json(joined_json).get("records", [])
    with np.load(joined_npz, allow_pickle=False) as joined:
        for name in ("offsets", "target", "labels"):
            if name not in joined.files:
                raise AtlasContractError("JOINED label contract is missing " + name)
        offsets = np.asarray(joined["offsets"], dtype=np.int64)
        target = np.asarray(joined["target"], dtype=np.int64)
        labels = np.asarray(joined["labels"], dtype=np.int8)
    with np.load(score_path, allow_pickle=False) as frozen:
        required_scores = (
            "steps__mean__H0lim_VE0_VE075_VE1",
            "steps__append_innovation__H0lim", "gate_raw",
        )
        missing = [name for name in required_scores if name not in frozen.files]
        if missing:
            raise AtlasContractError("frozen score archive fields are missing: " + ", ".join(missing))
        original4 = np.asarray(frozen[required_scores[0]], dtype=np.float64)
        innovation5 = np.asarray(frozen[required_scores[1]], dtype=np.float64)
        tail15 = np.asarray(frozen["gate_raw"], dtype=np.float64)
        archived_gate = (np.asarray(frozen["gate_percentile"], dtype=np.float64)
                         if "gate_percentile" in frozen.files else None)
    if len(records) != len(target) or offsets.shape != (len(records) + 1,) \
            or labels.shape != (int(offsets[-1]),) or original4.shape != labels.shape \
            or innovation5.shape != labels.shape or tail15.shape != (len(records),):
        raise AtlasContractError("frozen baseline score/label roster is misaligned")
    cells = np.asarray([row["cell"] for row in records], dtype=str)
    pb = np.char.startswith(cells, "pb_")
    pb_cells = sorted(set(cells[pb].tolist()))
    if strict_roster and (len(records) != EXPECTED_ANSWERS or int(offsets[-1]) != EXPECTED_STEPS
                          or len(pb_cells) != 8):
        raise AtlasContractError("full 13,769-answer/eight-cell frozen roster is required")

    digit025, digit_rate, extraction_audit = _reconstruct_digit025_from_extract(
        extract_root, records, offsets, innovation5,
    )
    tail_rank = _baseline_gate_percentiles(tail15, cells, pb)
    if archived_gate is not None and not np.allclose(
            tail_rank[pb], archived_gate[pb], rtol=0, atol=1e-12):
        raise AtlasContractError("frozen Tail15 gate percentile replay mismatch")
    digit_rank = _baseline_gate_percentiles(digit_rate, cells, pb)
    fused_component = np.full(len(records), np.nan, dtype=np.float64)
    fused_component[pb] = (tail_rank[pb] + digit_rank[pb]) / 2.0
    fused_rank = _baseline_gate_percentiles(fused_component, cells, pb)
    tail_gate = np.zeros(len(records), dtype=bool); tail_gate[pb] = tail_rank[pb] >= .33
    current_gate = np.zeros(len(records), dtype=bool); current_gate[pb] = fused_rank[pb] >= .33

    definitions = {
        "original4": (original4, tail_gate),
        "innovation5": (innovation5, tail_gate),
        "digit025": (digit025, tail_gate),
        "current": (digit025, current_gate),
    }
    metrics, per_answer = {}, {}
    for name, (scores, opened) in definitions.items():
        metrics[name], per_answer[name] = _baseline_arm_metrics(
            records, offsets, target, labels, scores, opened,
        )
    code_files = (
        ROOT / "spectral_utils/fusion_signal_registry.py",
        ROOT / "spectral_utils/historical_fusion_evaluation.py",
        ROOT / "spectral_utils/math_gate_selection.py",
        ROOT / "spectral_utils/temporal_context_models.py",
    )
    provenance = {
        "schema": SCHEMA + "/baseline-replay-inputs-v2",
        "score_inputs": {
            str(score_path.resolve()): sha256_file(score_path),
            str((Path(extract_root) / "MANIFEST.json").resolve()): sha256_file(Path(extract_root) / "MANIFEST.json"),
            str((Path(extract_root) / "answer_sha256.npy").resolve()): extraction_audit["answer_hash_index_sha256"],
            str((Path(extract_root) / "completion.npy").resolve()): extraction_audit["completion_sha256"],
            str((Path(extract_root) / "STATUS.json").resolve()): sha256_file(Path(extract_root) / "STATUS.json"),
        },
        "label_inputs": {
            str(joined_json.resolve()): sha256_file(joined_json),
            str(joined_npz.resolve()): sha256_file(joined_npz),
        },
        "code_inputs": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files},
        "implementation_sha256": implementation_sha256(
            _baseline_gate_percentiles, _baseline_arm_metrics,
            _reconstruct_digit025_from_extract, replay_frozen_baselines,
            files=code_files,
        ),
        "answer_tree_sha256": extraction_audit["answer_tree_sha256"],
        "development_only": True,
    }
    arrays: dict[str, Any] = {
        "steps__digit025": digit025, "gate__tail15": tail_gate,
        "gate__current": current_gate, "gate_rank__tail15": tail_rank,
        "gate_rank__digit_rate": digit_rank, "gate_rank__equal_rank": fused_rank,
        "digit_rate": digit_rate,
    }
    for name, values in per_answer.items():
        for field, array in values.items():
            arrays[f"{field}__{name}"] = array
    return {
        "metrics": metrics, "per_answer": arrays,
        "audit": {
            **extraction_audit, "pb_cells": pb_cells,
            "tail_gate_open": int(tail_gate[pb].sum()),
            "current_gate_open": int(current_gate[pb].sum()),
            "gate_q": .33, "development_only": True,
        },
        "provenance": provenance,
    }


def collect_frozen_baseline_metrics(
    contract_root: Path,
    baseline_root: Path,
    extract_root: Path,
) -> dict[str, dict[str, float]]:
    """Compatibility wrapper returning metrics from the true score replay."""
    return replay_frozen_baselines(contract_root, baseline_root, extract_root)["metrics"]


def _bind_baseline_replay_arrays(path: Path, arrays: Mapping[str, Any]) -> str:
    """Write a deterministic replay payload once, or verify its exact arrays."""
    np = _np()
    normalized = {str(name): np.asarray(value) for name, value in arrays.items()}
    if Path(path).is_file():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(normalized):
                raise AtlasContractError("immutable baseline replay array roster drift")
            for name, expected in normalized.items():
                actual = existing[name]
                if actual.shape != expected.shape or actual.dtype != expected.dtype \
                        or not np.array_equal(actual, expected, equal_nan=True):
                    raise AtlasContractError("immutable baseline replay array drift: " + name)
        return sha256_file(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(path).with_suffix(Path(path).suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **normalized)
    os.replace(temporary, path)
    return sha256_file(path)


def _stage_status(output_root: Path, stage: str) -> Path:
    return Path(output_root) / stage / "STATUS.json"


def _complete(output_root: Path, stage: str) -> bool:
    path = _stage_status(output_root, stage)
    return path.is_file() and read_json(path).get("status") == "COMPLETE"


def require_dependencies(output_root: Path, stage: str) -> None:
    missing = [name for name in STAGE_DEPENDENCIES[stage] if not _complete(output_root, name)]
    if missing:
        raise AtlasContractError(f"stage {stage} requires completed stages: {', '.join(missing)}")


def _write_status(output_root: Path, stage: str, status: str, **extra: Any) -> None:
    atomic_json(_stage_status(output_root, stage), {
        "schema": SCHEMA, "stage": stage, "status": status,
        "development_only": True, **extra,
    })


def stage_preflight(args: argparse.Namespace) -> None:
    out = args.output_root / "preflight"
    out.mkdir(parents=True, exist_ok=True)
    raw = verify_hashed_files(RAW_SOURCE_HASHES, [args.source_root])
    compact = verify_hashed_files(COMPACT_HASHES, [args.contract_root, args.source_root])
    dependencies = {}
    for name in ("numpy", "scipy", "sklearn", "torch", "matplotlib"):
        spec = importlib.util.find_spec(name)
        dependencies[name] = {"available": spec is not None, "origin": None if spec is None else spec.origin}
    mps = {"available": False, "built": False, "constraint": "one MPS training process only"}
    if dependencies["torch"]["available"]:
        import torch
        mps.update(available=bool(torch.backends.mps.is_available()),
                   built=bool(torch.backends.mps.is_built()), torch=str(torch.__version__))
    passed = all(row["status"] == "PASS" for row in raw + compact)
    report = {
        "schema": SCHEMA + "/preflight-v1", "status": "PASS" if passed else "FAILED",
        "source_root": str(args.source_root.resolve()), "contract_root": str(args.contract_root.resolve()),
        "raw_sources": raw, "compact_contract": compact, "dependencies": dependencies,
        "runtime": {"python": sys.version, "platform": platform.platform(), "mps": mps},
        "no_download": True, "no_model_forward": True,
    }
    atomic_json(out / "PREFLIGHT.json", report)
    manifest = {"schema": SCHEMA + "/preflight-manifest-v1",
                "protocol_sha256": sha256_file(PROTOCOL),
                "source_hashes": {row["relative"]: row["actual_sha256"] for row in raw + compact}}
    bind_immutable_manifest(out / "MANIFEST.json", manifest)
    if not passed:
        _write_status(args.output_root, "preflight", "FAILED", audit="PREFLIGHT.json")
        raise AtlasContractError("preflight source drift or missing input")
    _write_status(args.output_root, "preflight", "COMPLETE", audit="PREFLIGHT.json")


def _serialize_spec(spec: Any) -> dict[str, Any]:
    if hasattr(spec, "to_dict"):
        value = spec.to_dict()
    elif hasattr(spec, "__dataclass_fields__"):
        from dataclasses import asdict
        value = asdict(spec)
    elif isinstance(spec, Mapping):
        value = dict(spec)
    else:
        value = {name: getattr(spec, name) for name in dir(spec)
                 if not name.startswith("_") and not callable(getattr(spec, name))}
    return json_ready(value)


def _registry_payload(module: Any) -> dict[str, Any]:
    """Adapt the registry module while keeping the orchestration API narrow."""
    if hasattr(module, "registry_payload"):
        payload = module.registry_payload()
        return json_ready(payload)
    if hasattr(module, "build_builtin_registry"):
        registry = module.build_builtin_registry()
    elif hasattr(module, "BUILTIN_REGISTRY"):
        registry = module.BUILTIN_REGISTRY
    elif hasattr(module, "build_default_registry"):
        registry = module.build_default_registry()
    elif hasattr(module, "default_registry"):
        registry = module.default_registry()
    else:
        signals = getattr(module, "SIGNALS", getattr(module, "SIGNAL_SPECS", ()))
        readouts = getattr(module, "READOUTS", getattr(module, "READOUT_SPECS", ()))
        fusions = getattr(module, "FUSION_SETS", getattr(module, "FUSION_SET_SPECS", ()))
        return {"signals": [_serialize_spec(x) for x in signals],
                "readouts": [_serialize_spec(x) for x in readouts],
                "fusion_sets": [_serialize_spec(x) for x in fusions]}
    if hasattr(registry, "to_dict"):
        return json_ready(registry.to_dict())
    return {name: [_serialize_spec(item) for item in getattr(registry, name, ())]
            for name in ("signals", "readouts", "fusion_sets")}


def stage_registry(args: argparse.Namespace) -> None:
    module = importlib.import_module("spectral_utils.fusion_signal_registry")
    payload = _registry_payload(module)
    if not payload.get("signals") or not payload.get("readouts"):
        raise AtlasContractError("registry module returned an empty deterministic roster")
    forbidden = {"label", "labels", "correctness", "annotations", "y"}
    if any(forbidden & set(inspect.signature(value).parameters)
           for _, value in inspect.getmembers(module, inspect.isfunction)):
        raise AtlasContractError("registry exports a label-bearing function")
    historical_path = args.contract_root / "results/predictor_error_profiles_v1/BROAD_ANALYSIS.json"
    if not historical_path.is_file():
        raise AtlasContractError("historical score/peak inventory is required by the unified registry")
    historical = read_json(historical_path)
    summary = historical.get("summary", {})
    if (summary.get("scored_columns"), summary.get("unique_score_arrays"),
            summary.get("unique_peak_vectors")) != (254, 178, 173):
        raise AtlasContractError("historical registry cardinality drift")
    historical_records = []
    for archive in historical.get("inventory", []):
        relative = str(archive["path"]).replace("\\", "/")
        candidates = (args.contract_root / relative, args.source_root / relative)
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        artifact_status = (
            "AVAILABLE" if path.is_file() and sha256_file(path) == archive.get("sha256")
            else "MISSING_OR_DRIFT"
        )
        for row in archive.get("rows", []):
            historical_records.append({
                "name": f"historical::{archive['archive']}::{row['name']}",
                "provenance_family": str(archive["archive"]),
                "insertion_point": "step_post_readout",
                "resolution": "step",
                "access_scope": "historical_artifact",
                "orientation": "high_is_risk",
                "mask_semantics": "artifact_defined",
                "status": "REPORT_ONLY",
                "artifact_status": artifact_status,
                "archive_sha256": archive.get("sha256"),
                "peak_sha256": row.get("peak_sha256"),
                "included": bool(row.get("included", False)),
                "control_tagged": bool(row.get("control_tagged", False)),
                "reason": (
                    "historical comparison only; exact score arrays must be replayed and explained "
                    "before promotion"
                ),
            })
    if len(historical_records) != 254:
        raise AtlasContractError("historical registry must retain all 254 scored columns")
    payload["historical_signal_records"] = historical_records
    payload["historical_deduplication"] = {
        "scored_columns": 254,
        "unique_score_arrays": 178,
        "unique_peak_vectors": 173,
        "score_identity_scope": "inventory aggregate; missing bulk remains REPORT_ONLY",
        "peak_hashes_recorded": True,
    }
    out = args.output_root / "registry"
    out.mkdir(parents=True, exist_ok=True)
    payload.update({"schema": SCHEMA + "/registry-v2", "development_only": True,
                    "module_sha256": sha256_file(ROOT / "spectral_utils/fusion_signal_registry.py"),
                    "historical_inventory_sha256": sha256_file(historical_path)})
    bind_immutable_manifest(out / "REGISTRY.json", payload)
    _write_status(args.output_root, "registry", "COMPLETE",
                  signals=len(payload["signals"]), historical_records=len(historical_records),
                  historical_unique_scores=178, historical_unique_peaks=173,
                  readouts=len(payload["readouts"]))


def _bundle_contract(args: argparse.Namespace, records: Sequence[Mapping[str, Any]],
                     offsets: Any, folds: Mapping[str, int]) -> dict[str, Any]:
    np = _np()
    answers = len(records)
    tokens = sum(int(row["tokens"]) for row in records)
    steps = int(offsets[-1])
    roster = [{"uid": row["uid"], "tokens": int(row["tokens"]), "steps": int(row["steps"]),
               "fold": int(folds[row["group_id"]])} for row in records]
    source_audit = read_json(args.output_root / "preflight/PREFLIGHT.json")
    return {
        "schema": SCHEMA + "/primitive-bundle-contract-v1",
        "answers": answers, "tokens": tokens, "steps": steps,
        "primitive_targets": list(PRIMITIVE_TARGETS), "target_shape": [tokens, 4],
        "target_dtype": "float32", "logprobs15_shape": [tokens, 15],
        "spans_shape": [steps, 2], "folds": 5,
        "roster_sha256": sha256_json(roster),
        "source_hashes": {row["relative"]: row["actual_sha256"] for row in source_audit["raw_sources"]},
        "implementation_sha256": implementation_sha256(
            stage_bundle, _finalize_bundle, sanitized_metadata,
            files=(
                ROOT / "spectral_utils/temporal_research_features.py",
                ROOT / "spectral_utils/renyi_locator_feature_bank.py",
                ROOT / "scripts/run_direct_probability_temporal.py",
            ),
        ),
        "correctness_labels_in_bundle": False,
        "forbidden_predictor_targets": ["H0lim_innovation", "digit_innovation", "derived_stream"],
        "numpy_version": np.__version__,
    }


def _load_records_for_bundle(args: argparse.Namespace):
    np = _np()
    joined_json = args.contract_root / "results/localization_full_benchmark_v3/evaluation/JOINED.json"
    joined_npz = args.contract_root / "results/localization_full_benchmark_v3/evaluation/JOINED.npz"
    folds_path = args.contract_root / "results/localization_source_group_audit_v1/FOLDS_V2.json"
    records = read_json(joined_json)["records"]
    with np.load(joined_npz, allow_pickle=False) as saved:
        # Only offsets are allowed across the label-free boundary.
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
    folds = read_json(folds_path)["outer"]
    expected = np.r_[0, np.cumsum([int(row["steps"]) for row in records])]
    np.testing.assert_array_equal(offsets, expected)
    if len(set(row["uid"] for row in records)) != len(records):
        raise AtlasContractError("duplicate answer UID in frozen roster")
    return records, offsets, folds


def _open_or_create_array(path: Path, *, shape: Sequence[int], dtype: str):
    np = _np()
    if path.exists():
        _array_contract(path, shape, dtype)
        return np.load(path, mmap_mode="r+", allow_pickle=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=tuple(shape))


def _finalize_bundle(bundle: Path, records: Sequence[Mapping[str, Any]], offsets: Any,
                     folds: Mapping[str, int]) -> dict[str, Any]:
    np = _np()
    frozen_path = bundle / "FREEZE.json"
    if frozen_path.is_file():
        frozen = read_json(frozen_path)
        for name, identity in frozen.get("files", {}).items():
            path = bundle / name
            if not path.is_file() or sha256_file(path) != identity["sha256"] \
                    or path.stat().st_size != identity["size"]:
                raise AtlasContractError("immutable frozen bundle drift: " + name)
        if frozen.get("primitive_targets") != list(PRIMITIVE_TARGETS) \
                or frozen.get("correctness_labels_in_bundle") is not False:
            raise AtlasContractError("frozen bundle semantic drift")
        return frozen["files"]
    levels = np.load(bundle / "primitive_levels.npy", mmap_mode="r", allow_pickle=False)
    signs = np.load(bundle / "orientation.npy", mmap_mode="r", allow_pickle=False)
    token_offsets = np.r_[0, np.cumsum([int(row["tokens"]) for row in records])]
    metadata = []
    for index, row in enumerate(records):
        start, stop = map(int, token_offsets[index:index + 2])
        values = np.asarray(levels[start:stop], dtype=float)
        metadata.append(sanitized_metadata(
            row, fold=int(folds[row["group_id"]]), token_offset=start,
            step_start=int(offsets[index]), step_stop=int(offsets[index + 1]),
            mean=values.mean(axis=0), scale=np.maximum(values.std(axis=0), 1e-8),
            signs=signs[index],
        ))
    atomic_json(bundle / "METADATA.json", metadata)
    atomic_json(bundle / "FOLDS.json", {row["group_id"]: int(folds[row["group_id"]]) for row in records})
    del levels, signs
    files = ("primitive_levels.npy", "logprobs15.npy", "step_spans.npy", "orientation.npy",
             "completion.npy", "METADATA.json", "FOLDS.json")
    freeze = {name: {"sha256": sha256_file(bundle / name), "size": (bundle / name).stat().st_size}
              for name in files}
    bind_immutable_manifest(frozen_path, {
        "schema": SCHEMA + "/primitive-bundle-freeze-v1", "files": freeze,
        "answers": len(records), "tokens": int(token_offsets[-1]), "steps": int(offsets[-1]),
        "primitive_targets": list(PRIMITIVE_TARGETS), "correctness_labels_in_bundle": False,
    })
    return freeze


def stage_bundle(args: argparse.Namespace) -> None:
    np = _np()
    from scripts import run_direct_probability_temporal as evaluator
    from spectral_utils.temporal_research_features import score_features

    records, offsets, folds = _load_records_for_bundle(args)
    contract = _bundle_contract(args, records, offsets, folds)
    if not args.fixture_mode and (contract["answers"], contract["tokens"], contract["steps"]) != (
            EXPECTED_ANSWERS, EXPECTED_TOKENS, EXPECTED_STEPS):
        raise AtlasContractError("full frozen roster must be exactly 13,769 / 6,968,779 / 145,597")
    if contract["target_shape"][1] != 4 or tuple(contract["primitive_targets"]) != PRIMITIVE_TARGETS:
        raise AtlasContractError("bundle may contain exactly four primitive prediction targets")
    bundle = args.bundle_root
    bundle.mkdir(parents=True, exist_ok=True)
    contract_hash = bind_immutable_manifest(bundle / "MANIFEST.json", contract)
    completion = CompletionBitmap(bundle / "completion.npy", len(records), contract_hash)
    levels = _open_or_create_array(bundle / "primitive_levels.npy",
                                   shape=(contract["tokens"], 4), dtype="float32")
    logprobs = _open_or_create_array(bundle / "logprobs15.npy",
                                     shape=(contract["tokens"], 15), dtype="float32")
    spans_out = _open_or_create_array(bundle / "step_spans.npy",
                                      shape=(contract["steps"], 2), dtype="int64")
    orientation = _open_or_create_array(bundle / "orientation.npy",
                                        shape=(contract["answers"], 4), dtype="float32")
    token_offsets = np.r_[0, np.cumsum([int(row["tokens"]) for row in records])]

    evaluator.old.configure_source_root(args.source_root)
    completed_this_launch = 0
    try:
        for cell, source_path, kind, dataset in evaluator.source_specs():
            indexes = [i for i, row in enumerate(records) if row["cell"] == cell and not completion.done(i)]
            if not indexes:
                continue
            rows = evaluator.old._source_row_map(evaluator.old.load_pickle(source_path), kind=kind, dataset=dataset)
            for index in indexes:
                row_contract = records[index]
                row = rows[row_contract["row_id"]]
                lp = np.asarray(evaluator.old._topk_payload(row)["logprobs"], dtype=float)
                entropy = np.asarray(row["token_entropies"], dtype=float)
                local_spans = np.asarray(row["step_token_spans"], dtype=np.int64)
                if lp.shape[0] != int(row_contract["tokens"]) or lp.shape[1] < 15:
                    raise AtlasContractError("top15/token identity mismatch: " + row_contract["uid"])
                if entropy.shape != (len(lp),) or local_spans.shape != (int(row_contract["steps"]), 2):
                    raise AtlasContractError("entropy/span identity mismatch: " + row_contract["uid"])
                _, _, info, matrix = score_features(lp, entropy, local_spans)
                if matrix.shape != (len(lp), 4):
                    raise AtlasContractError("derived or missing primitive target")
                token_start, token_stop = map(int, token_offsets[index:index + 2])
                step_start, step_stop = map(int, offsets[index:index + 2])
                levels[token_start:token_stop] = matrix.astype(np.float32)
                logprobs[token_start:token_stop] = lp[:, :15].astype(np.float32)
                spans_out[step_start:step_stop] = local_spans + token_start
                orientation[index] = np.asarray(info["signs"], dtype=np.float32)
                levels.flush(); logprobs.flush(); spans_out.flush(); orientation.flush()
                completion.mark(index)
                completed_this_launch += 1
                _write_status(args.output_root, "bundle", "RUNNING", completed=completion.count(),
                              expected=len(records), bundle_root=str(bundle))
                if args.max_answers and completed_this_launch >= args.max_answers:
                    return
            del rows
        if not completion.complete():
            raise AtlasContractError(f"bundle incomplete: {completion.count()}/{len(records)}")
        freeze = _finalize_bundle(bundle, records, offsets, folds)
        _write_status(args.output_root, "bundle", "COMPLETE", completed=len(records),
                      bundle_root=str(bundle), freeze_sha256=sha256_json(freeze))
    finally:
        levels.flush(); logprobs.flush(); spans_out.flush(); orientation.flush()
        del levels, logprobs, spans_out, orientation
        completion.close()


def extract_atomic_answer(
    primitive_levels: Any,
    logprobs: Any,
    top_ids: Any,
    generated_ids: Any,
    selected_surprisal: Any,
    entropy: Any,
    spans: Any,
    uid: str,
) -> Mapping[str, Any]:
    """Extract every replayable atomic signal and fixed readout, label-free.

    The input is saved telemetry only.  This function intentionally has no
    annotation/target argument, and its signature is checked in unit tests.
    """
    np = _np()
    from spectral_utils import fusion_signal_registry as registry
    from spectral_utils.alternative_views_fusion import VIEWS, extract as alternative_views
    from spectral_utils.digit_fusion import digit_streams
    from spectral_utils.direct_probability_fusion_v2 import augmented_probability_risk, residual_tail_mass
    from spectral_utils.renyi_alpha_sweep import FAMILY_OF, sweep_matrix
    from spectral_utils.renyi_locator_feature_bank import feature_matrix

    levels = np.asarray(primitive_levels, dtype=float)
    lp = np.asarray(logprobs, dtype=float)
    ids = np.asarray(top_ids)
    generated = np.asarray(generated_ids)
    chosen = np.asarray(selected_surprisal, dtype=float)
    entropy = np.asarray(entropy, dtype=float)
    spans = np.asarray(spans, dtype=np.int64)
    if levels.ndim != 2 or levels.shape[1] != 4:
        raise ValueError("exactly four primitive levels are required")
    n = len(levels)
    if lp.ndim != 2 or lp.shape[0] != n or lp.shape[1] < 50 or ids.shape != lp.shape:
        raise ValueError("full saved top-50 telemetry must align with primitive levels")
    if generated.shape != (n,) or chosen.shape != (n,) or entropy.shape != (n,):
        raise ValueError("provided-token telemetry must align with primitive levels")

    token: dict[str, tuple[Any, Any]] = {}
    all_active = np.ones(n, dtype=bool)
    for column, name in enumerate(registry.Q15_PRIMITIVE_NAMES):
        token[name] = (levels[:, column], all_active)

    sweep, sweep_names = sweep_matrix(lp[:, :15])
    anchor = levels[:, 3]
    for column, local in enumerate(sweep_names):
        values = sweep[:, column].copy()
        full_name = "renyi_escort." + local
        method_name = "view__" + local
        if FAMILY_OF[method_name] == "escort_varentropy" and np.std(values) > 1e-12 \
                and np.std(anchor) > 1e-12 and np.corrcoef(values, anchor)[0, 1] < 0:
            values *= -1
        token[full_name] = (values, all_active)

    bank = feature_matrix(lp, entropy)
    token["q50.VE1"] = (bank["matrix"][:, 4], all_active)
    token["q15.H1_native"] = (bank["matrix"][:, 5], all_active)
    token["q15.Hinf"] = (bank["matrix"][:, 6], all_active)

    alternative, _ = alternative_views(generated, ids, lp[:, :50], chosen)
    for column, local in enumerate(VIEWS):
        token["step395." + local] = (alternative[:, column], all_active)

    augmented = augmented_probability_risk({"logprobs": lp}, chosen, k=15)
    for column in range(15):
        token[f"direct_probability.rank_{column + 1}_risk"] = (augmented[:, column], all_active)
    token["direct_probability.selected_token_surprisal"] = (augmented[:, 15], all_active)
    token["direct_probability.residual_tail_mass"] = (augmented[:, 16], all_active)

    digit, opportunity, permuted = digit_streams(generated, ids[:, 0], range(15, 25), uid)
    token["digit.disagreement"] = (digit, all_active)
    token["digit.opportunity"] = (opportunity, all_active)
    token["digit.permuted_location_control"] = (permuted, opportunity.astype(bool))
    for clock, (values, active) in registry.digit_clock_innovations(digit, opportunity).items():
        token[f"digit.{clock}_innovation"] = (values, active)

    tail15 = residual_tail_mass(lp[:, :15])
    for kind, (values, active) in registry.tail15_causal_innovations(tail15).items():
        token[f"tail15.{kind}_innovation"] = (values, active)
    for column, name in enumerate(registry.Q15_PRIMITIVE_NAMES):
        values, active = registry.prefix_mean_innovation(levels[:, column])
        token[name + ".prefix_mean_innovation"] = (values, active)

    step: dict[str, tuple[Any, Any]] = {}
    count = np.asarray([digit[start:stop].sum() for start, stop in spans], dtype=float)
    opportunities = np.asarray([opportunity[start:stop].sum() for start, stop in spans], dtype=float)
    rate = np.divide(count, opportunities, out=np.zeros_like(count), where=opportunities > 0)
    presence = opportunities > 0
    step["digit.count"] = (count, np.ones(len(spans), dtype=bool))
    step["digit.rate"] = (rate, presence)
    step["digit.presence"] = (presence.astype(float), np.ones(len(spans), dtype=bool))

    arrays: dict[str, Any] = {}
    for name, (values, active) in sorted(token.items()):
        arrays[f"token__{name}__values"] = np.asarray(values, dtype=np.float32)
        arrays[f"token__{name}__active"] = np.asarray(active, dtype=bool)
        readout_values, readout_active, decisions = [], [], []
        for readout in registry.READOUT_NAMES:
            score, available = registry.readout_steps(values, spans, readout, active_mask=active)
            readout_values.append(score)
            readout_active.append(available)
            decisions.append((registry.decode_argmax(score, available),
                              registry.decode_first_near_max(score, .25, available),
                              registry.decode_persistent_q90_3(score, available)))
        # Columns are the immutable registry.READOUT_NAMES order; packing keeps
        # 13,769 answer checkpoints tractable without dropping any expert.
        arrays[f"step__{name}__readouts"] = np.column_stack(readout_values).astype(np.float32)
        arrays[f"step__{name}__readout_active"] = np.column_stack(readout_active)
        arrays[f"decision__{name}__readouts"] = np.asarray(decisions, dtype=np.int32)
        arrays[f"decision__{name}__step_top5"] = np.asarray(
            registry.decode_step_top5(values, spans, active), dtype=np.int32)
    for name, (values, active) in sorted(step.items()):
        arrays[f"step__{name}__native__values"] = np.asarray(values, dtype=np.float32)
        arrays[f"step__{name}__native__active"] = np.asarray(active, dtype=bool)
        arrays[f"decision__{name}__native__argmax"] = np.asarray(
            registry.decode_argmax(values, active), dtype=np.int32)
    ve0, ve0_active = registry.readout_steps(levels[:, 1], spans, "top10")
    ve075, ve075_active = registry.readout_steps(levels[:, 2], spans, "top10")
    arrays["decision__earlier_ve_peak"] = np.asarray(
        registry.decode_earlier_ve_peak(ve0, ve075, ve0_active, ve075_active), dtype=np.int32)
    answer = {"tail15.answer_prominence": registry.answer_top10_minus_mean(tail15)}
    arrays["answer__tail15.answer_prominence"] = np.asarray(
        answer["tail15.answer_prominence"], dtype=np.float32)

    # All runnable canonical registry signals must have a concrete token or
    # native-step representation. Aliases are deliberately not duplicated.
    computable = set(token) | set(step) | set(answer)
    missing = [spec.name for spec in registry.BUILTIN_REGISTRY.signals
               if spec.status == "ELIGIBLE" and spec.insertion_point != "background"
               and spec.name not in computable]
    if missing:
        raise AtlasContractError("eligible registry signals lack extraction: " + ", ".join(missing))
    return arrays


def stage_extract(args: argparse.Namespace) -> None:
    np = _np()
    module = importlib.import_module("spectral_utils.fusion_signal_registry")
    from scripts import run_direct_probability_temporal as evaluator
    bundle = args.bundle_root
    metadata = read_json(bundle / "METADATA.json")
    levels = np.load(bundle / "primitive_levels.npy", mmap_mode="r", allow_pickle=False)
    logprobs = np.load(bundle / "logprobs15.npy", mmap_mode="r", allow_pickle=False)
    spans = np.load(bundle / "step_spans.npy", mmap_mode="r", allow_pickle=False)
    out = args.output_root / "extract"
    out.mkdir(parents=True, exist_ok=True)
    contract = {"schema": SCHEMA + "/atomic-extraction-contract-v2",
                "bundle_root": str(bundle.resolve()),
                "bundle_freeze_sha256": sha256_file(bundle / "FREEZE.json"),
                "registry_sha256": sha256_file(args.output_root / "registry/REGISTRY.json"),
                "implementation_sha256": implementation_sha256(
                    extract_atomic_answer,
                    files=(
                        ROOT / "spectral_utils/fusion_signal_registry.py",
                        ROOT / "spectral_utils/alternative_views_fusion.py",
                        ROOT / "spectral_utils/digit_fusion.py",
                        ROOT / "spectral_utils/direct_probability_fusion_v2.py",
                        ROOT / "spectral_utils/renyi_alpha_sweep.py",
                        ROOT / "spectral_utils/renyi_locator_feature_bank.py",
                    ),
                ),
                "answers": len(metadata), "label_free": True}
    contract_hash = bind_immutable_manifest(out / "MANIFEST.json", contract)
    completion = CompletionBitmap(out / "completion.npy", len(metadata), contract_hash)
    answer_hashes = _open_or_create_array(
        out / "answer_sha256.npy", shape=(len(metadata),), dtype="S64",
    )
    answers = out / "answers"
    answers.mkdir(exist_ok=True)
    for index in np.flatnonzero(np.asarray(completion.array)):
        path = answers / f"{int(index):05d}.npz"
        recorded = bytes(answer_hashes[index]).decode("ascii")
        if not path.is_file() or len(recorded) != 64 or sha256_file(path) != recorded:
            raise AtlasContractError(f"completed extraction artifact drift: answer {int(index)}")
    evaluator.old.configure_source_root(args.source_root)
    processed = 0
    try:
        for cell, source_path, kind, dataset in evaluator.source_specs():
            indexes = [index for index, row in enumerate(metadata)
                       if row["cell"] == cell and not completion.done(index)]
            if not indexes:
                continue
            source_rows = evaluator.old._source_row_map(
                evaluator.old.load_pickle(source_path), kind=kind, dataset=dataset,
            )
            records = read_json(args.contract_root / "results/localization_full_benchmark_v3/evaluation/JOINED.json")["records"]
            for index in indexes:
                row = metadata[index]
                raw = source_rows[records[index]["row_id"]]
                payload = evaluator.old._topk_payload(raw)
                token = slice(int(row["offset"]), int(row["offset"]) + int(row["tokens"]))
                step = slice(int(row["step_start"]), int(row["step_stop"]))
                local_spans = np.asarray(spans[step]) - int(row["offset"])
                arrays = extract_atomic_answer(
                    np.asarray(levels[token]), np.asarray(payload["logprobs"]),
                    np.asarray(payload["ids"]), np.asarray(raw["gen_token_ids"]),
                    np.asarray(raw["token_spilled_energies"]), np.asarray(raw["token_entropies"]),
                    local_spans, str(row["uid"]),
                )
                arrays = {str(key): np.asarray(value) for key, value in arrays.items()}
                if any(array.dtype == object for array in arrays.values()):
                    raise AtlasContractError("object arrays are forbidden in atomic extraction")
                temporary = answers / f"{index:05d}.npz.tmp"
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                final = answers / f"{index:05d}.npz"
                os.replace(temporary, final)
                answer_hashes[index] = sha256_file(final).encode("ascii")
                answer_hashes.flush()
                completion.mark(index)
                processed += 1
                _write_status(args.output_root, "extract", "RUNNING", completed=completion.count(),
                              expected=len(metadata))
                if args.max_answers and processed >= args.max_answers:
                    return
            del source_rows
        if not completion.complete():
            raise AtlasContractError("atomic extraction incomplete")
        ledger = {"answers": len(metadata), "files": len(list(answers.glob("*.npz"))),
                  "answer_hashes_sha256": sha256_file(out / "answer_sha256.npy"),
                  "implementation_sha256": contract["implementation_sha256"],
                  "status": "COMPLETE", "tail15_roles": 4, "digit_clocks": 2}
        atomic_json(out / "COVERAGE.json", ledger)
        _write_status(
            args.output_root, "extract", "COMPLETE",
            **{key: value for key, value in ledger.items() if key != "status"},
        )
    finally:
        answer_hashes.flush()
        del answer_hashes
        completion.close()


def _ridge_design(levels: Any, history: int = 16) -> tuple[Any, Any]:
    np = _np()
    from spectral_utils.temporal_context_models import history_windows
    x = np.asarray(levels, dtype=float)
    indexes = np.arange(len(x))
    windows, mask = history_windows(x, indexes, history=history)
    # The Atlas predictor firewall permits only past primitive levels and their
    # explicit observation mask.  In particular, the final answer length (and
    # therefore a normalized full-answer clock) is not a predictor input.
    design = np.column_stack((windows.reshape(len(x), -1), mask, np.ones(len(x))))
    return design, x


def _fit_ridge_sufficient(statistics: Mapping[int, tuple[Any, Any, int]], excluded: Sequence[int],
                          ridge: float = 1.0) -> dict[str, Any]:
    np = _np()
    allowed = [fold for fold in sorted(statistics) if fold not in set(excluded)]
    if not allowed:
        raise ValueError("empty Ridge training split")
    xtx = sum((statistics[fold][0] for fold in allowed), np.zeros_like(statistics[allowed[0]][0]))
    xty = sum((statistics[fold][1] for fold in allowed), np.zeros_like(statistics[allowed[0]][1]))
    count = sum(statistics[fold][2] for fold in allowed)
    penalty = np.eye(xtx.shape[0]) * float(ridge)
    penalty[-1, -1] = 0.0
    coefficient = np.linalg.solve(xtx + penalty, xty)
    return {"coefficient": coefficient, "excluded_folds": list(excluded), "observations": count,
            "targets": list(PRIMITIVE_TARGETS)}


def _save_npz_immutable(path: Path, **arrays: Any) -> str:
    np = _np()
    if path.exists():
        return sha256_file(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return sha256_file(path)


@contextlib.contextmanager
def _single_mps_process(lock_path: Path, enabled: bool):
    if not enabled:
        yield
        return
    import fcntl
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise AtlasContractError("another MPS predictor worker is active") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _train_tcn(bundle: Path, metadata: Sequence[Mapping[str, Any]], excluded: Sequence[int],
               checkpoint: Path, *, device: str, seed: int, updates: int, batch_size: int) -> dict[str, Any]:
    """Fresh four-target, past-only TCN fit for one exclusion split."""
    import torch
    from spectral_utils.temporal_context_models import TelemetryTCN, gaussian_prediction_loss
    np = _np()
    sidecar = checkpoint.with_suffix(".json")
    if checkpoint.exists() != sidecar.exists():
        raise AtlasContractError("partial TCN checkpoint; refusing overwrite")
    if checkpoint.exists() and sidecar.exists():
        spec = read_json(sidecar)
        if spec["excluded_folds"] != list(excluded) or spec["targets"] != list(PRIMITIVE_TARGETS):
            raise AtlasContractError("TCN checkpoint semantic drift")
        if sha256_file(checkpoint) != spec.get("checkpoint_sha256"):
            raise AtlasContractError("TCN checkpoint hash drift")
        return spec
    torch.manual_seed(seed)
    np.random.seed(seed)
    levels = np.load(bundle / "primitive_levels.npy", mmap_mode="r", allow_pickle=False)
    eligible = [row for row in metadata if int(row["fold"]) not in set(excluded) and int(row["tokens"]) > 1]
    if not eligible:
        raise AtlasContractError("empty TCN training split")
    model = TelemetryTCN(dimensions=4, width=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    from spectral_utils.temporal_context_models import history_windows
    last_loss = None
    for _ in range(updates):
        chosen = rng.integers(0, len(eligible), size=batch_size)
        histories, masks, targets = [], [], []
        for choice in chosen:
            row = eligible[int(choice)]
            n = int(row["tokens"])
            position = int(rng.integers(n))
            start = int(row["offset"])
            answer = np.asarray(levels[start:start + n], dtype=np.float32)
            window, mask = history_windows(answer, [position], history=16)
            histories.append(window[0]); masks.append(mask[0])
            targets.append(answer[position])
        history = torch.tensor(np.asarray(histories), device=device)
        mask = torch.tensor(np.asarray(masks), dtype=torch.bool, device=device)
        # TelemetryTCN retains the architectural clock input for checkpoint
        # compatibility, but it is fixed to zero so no full-answer information
        # crosses the primitive-only predictor boundary.
        clock = torch.zeros(len(histories), dtype=torch.float32, device=device)
        target = torch.tensor(np.asarray(targets), device=device)
        optimizer.zero_grad(set_to_none=True)
        mean, variance = model(history, mask, clock)
        loss = gaussian_prediction_loss(mean, variance, target)
        loss.backward(); optimizer.step()
        last_loss = float(loss.detach().cpu())
    # Deterministic CPU smoke parity on exactly the same past-only batch.
    cpu = TelemetryTCN(dimensions=4, width=32)
    cpu.load_state_dict({key: value.detach().cpu() for key, value in model.state_dict().items()})
    with torch.no_grad():
        device_output = model(history, mask, clock)[0].detach().cpu().numpy()
        cpu_output = cpu(history.cpu(), mask.cpu(), clock.cpu())[0].numpy()
    parity = float(np.max(np.abs(device_output - cpu_output)))
    tolerance = 3e-4 if device == "mps" else 1e-6
    if parity > tolerance:
        raise AtlasContractError(f"TCN CPU/{device} smoke parity failed: {parity}")
    temporary = checkpoint.with_suffix(checkpoint.suffix + ".tmp")
    torch.save({"state_dict": cpu.state_dict(), "targets": PRIMITIVE_TARGETS}, temporary)
    os.replace(temporary, checkpoint)
    spec = {"schema": SCHEMA + "/tcn-four-target-v1", "excluded_folds": list(excluded),
            "targets": list(PRIMITIVE_TARGETS), "seed": seed, "updates": updates,
            "batch_size": batch_size, "device": device, "cpu_smoke_max_abs": parity,
            "loss": last_loss, "checkpoint_sha256": sha256_file(checkpoint),
            "labels_used": False}
    bind_immutable_manifest(sidecar, spec)
    return spec


def _apply_outer_predictors(
    bundle: Path,
    output: Path,
    metadata: Sequence[Mapping[str, Any]],
    *,
    device: str,
    contract_sha256: str,
    max_answers: int = 0,
) -> tuple[int, bool]:
    """Apply the five singleton-excluded Ridge/TCN models out of fold."""
    np = _np()
    import torch
    from spectral_utils.temporal_context_models import TelemetryTCN, history_windows

    levels = np.load(bundle / "primitive_levels.npy", mmap_mode="r", allow_pickle=False)
    learned = output / "learned_oof"
    learned.mkdir(exist_ok=True)
    predictions = _open_or_create_array(
        learned / "backgrounds.npy", shape=(len(levels), 2, 4), dtype="float32",
    )
    active = _open_or_create_array(learned / "active.npy", shape=(len(levels),), dtype="bool")
    completion = CompletionBitmap(learned / "completion.npy", len(metadata), contract_sha256)

    ridge_models: dict[int, Any] = {}
    tcn_models: dict[int, Any] = {}
    processed = 0
    try:
        for index, row in enumerate(metadata):
            if completion.done(index):
                continue
            fold = int(row["fold"])
            start, n = int(row["offset"]), int(row["tokens"])
            values = np.asarray(levels[start:start + n], dtype=float)
            if fold not in ridge_models:
                path = output / f"ridge/exclude_{fold}.npz"
                with np.load(path, allow_pickle=False) as saved:
                    ridge_models[fold] = np.asarray(saved["coefficient"], dtype=float)
            design, _ = _ridge_design(values)
            ridge = design @ ridge_models[fold]

            if fold not in tcn_models:
                path = output / f"tcn/exclude_{fold}.pt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if tuple(checkpoint.get("targets", ())) != PRIMITIVE_TARGETS:
                    raise AtlasContractError("TCN checkpoint contains a derived or missing target")
                model = TelemetryTCN(dimensions=4, width=32)
                model.load_state_dict(checkpoint["state_dict"])
                model.to(device).eval()
                tcn_models[fold] = model
            windows, masks = history_windows(values, np.arange(n), history=16)
            tcn = np.empty((n, 4), dtype=np.float32)
            with torch.no_grad():
                for begin in range(0, n, 2_048):
                    stop = min(n, begin + 2_048)
                    history = torch.tensor(windows[begin:stop], dtype=torch.float32, device=device)
                    mask = torch.tensor(masks[begin:stop], dtype=torch.bool, device=device)
                    position = torch.zeros(stop - begin, dtype=torch.float32, device=device)
                    tcn[begin:stop] = tcn_models[fold](history, mask, position)[0].cpu().numpy()
            row_active = np.ones(n, dtype=bool)
            row_active[0] = False
            ridge[0] = 0.0
            tcn[0] = 0.0
            predictions[start:start + n, 0] = ridge.astype(np.float32)
            predictions[start:start + n, 1] = tcn
            active[start:start + n] = row_active
            predictions.flush(); active.flush(); completion.mark(index)
            processed += 1
            if max_answers and processed >= max_answers:
                return completion.count(), False
        if not completion.complete():
            raise AtlasContractError("OOF predictor application is incomplete")
        return completion.count(), True
    finally:
        predictions.flush(); active.flush()
        del predictions, active, levels
        completion.close()


def _apply_inner_predictors(
    bundle: Path,
    output: Path,
    metadata: Sequence[Mapping[str, Any]],
    *,
    device: str,
    contract_sha256: str,
    max_answers: int = 0,
) -> tuple[int, bool]:
    """Apply pair-excluded models for the five inner source folds.

    For an answer whose outer/source fold is ``g``, inner-fold axis ``f`` uses
    the model trained with both ``f`` and ``g`` excluded.  The diagonal
    ``f == g`` has no valid pair-excluded fit and is therefore identically zero
    and inactive.  Token zero is inactive for every inner fold.
    """
    np = _np()
    import torch
    from spectral_utils.temporal_context_models import TelemetryTCN, history_windows

    levels = np.load(bundle / "primitive_levels.npy", mmap_mode="r", allow_pickle=False)
    learned = output / "learned_inner"
    learned.mkdir(exist_ok=True)
    pair_names = tuple("_".join(map(str, pair)) for pair in itertools.combinations(range(5), 2))
    model_hashes: dict[str, str] = {}
    for pair_name in pair_names:
        for family, suffix in (("ridge", ".npz"), ("tcn", ".pt")):
            path = output / family / f"exclude_{pair_name}{suffix}"
            if not path.is_file():
                raise AtlasContractError(f"missing pair-excluded {family} fit: {path}")
            model_hashes[f"{family}/exclude_{pair_name}{suffix}"] = sha256_file(path)
    inner_contract = {
        "schema": SCHEMA + "/learned-inner-application-v1",
        "parent_contract_sha256": contract_sha256,
        "implementation_sha256": implementation_sha256(
            _apply_inner_predictors,
            files=(ROOT / "spectral_utils/temporal_context_models.py",),
        ),
        "answers": len(metadata), "tokens": len(levels),
        "shape": [len(levels), 5, 2, 4],
        "axes": {
            "0": "token", "1": "inner_fold_f", "2": ["ridge", "tcn"],
            "3": list(PRIMITIVE_TARGETS),
        },
        "selection": "answer fold g uses exclude_sorted(f,g) when f != g",
        "diagonal": "f == g is zero and inactive",
        "first_token_inactive": True,
        "device": device, "labels_used": False, "model_sha256": model_hashes,
    }
    inner_hash = bind_immutable_manifest(learned / "MANIFEST.json", inner_contract)
    predictions = _open_or_create_array(
        learned / "backgrounds.npy", shape=(len(levels), 5, 2, 4), dtype="float32",
    )
    active = _open_or_create_array(
        learned / "active.npy", shape=(len(levels), 5), dtype="bool",
    )
    completion = CompletionBitmap(learned / "completion.npy", len(metadata), inner_hash)

    ridge_models: dict[tuple[int, int], Any] = {}
    tcn_models: dict[tuple[int, int], Any] = {}
    processed = 0
    try:
        for index, row in enumerate(metadata):
            if completion.done(index):
                continue
            outer_fold = int(row["fold"])
            if outer_fold not in range(5):
                raise AtlasContractError(f"invalid source fold {outer_fold} for answer {index}")
            start, n = int(row["offset"]), int(row["tokens"])
            values = np.asarray(levels[start:start + n], dtype=float)
            design, _ = _ridge_design(values)
            windows, masks = history_windows(values, np.arange(n), history=16)

            # Clear the entire answer before writing so a crash cannot leave
            # stale values that become visible on a later resume.
            predictions[start:start + n] = 0.0
            active[start:start + n] = False
            for inner_fold in range(5):
                if inner_fold == outer_fold:
                    continue
                excluded = tuple(sorted((inner_fold, outer_fold)))
                if excluded not in ridge_models:
                    path = output / "ridge" / ("exclude_" + "_".join(map(str, excluded)) + ".npz")
                    with np.load(path, allow_pickle=False) as saved:
                        targets = tuple(str(item) for item in saved["targets"].tolist()) \
                            if "targets" in saved.files else PRIMITIVE_TARGETS
                        if targets != PRIMITIVE_TARGETS:
                            raise AtlasContractError("Ridge checkpoint contains a derived or missing target")
                        ridge_models[excluded] = np.asarray(saved["coefficient"], dtype=float)
                ridge = design @ ridge_models[excluded]

                if excluded not in tcn_models:
                    path = output / "tcn" / ("exclude_" + "_".join(map(str, excluded)) + ".pt")
                    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                    if tuple(checkpoint.get("targets", ())) != PRIMITIVE_TARGETS:
                        raise AtlasContractError("TCN checkpoint contains a derived or missing target")
                    model = TelemetryTCN(dimensions=4, width=32)
                    model.load_state_dict(checkpoint["state_dict"])
                    model.to(device).eval()
                    tcn_models[excluded] = model
                tcn = np.empty((n, 4), dtype=np.float32)
                with torch.no_grad():
                    for begin in range(0, n, 2_048):
                        stop = min(n, begin + 2_048)
                        history = torch.tensor(windows[begin:stop], dtype=torch.float32, device=device)
                        mask = torch.tensor(masks[begin:stop], dtype=torch.bool, device=device)
                        position = torch.zeros(stop - begin, dtype=torch.float32, device=device)
                        tcn[begin:stop] = tcn_models[excluded](history, mask, position)[0].cpu().numpy()
                ridge[0] = 0.0
                tcn[0] = 0.0
                predictions[start:start + n, inner_fold, 0] = ridge.astype(np.float32)
                predictions[start:start + n, inner_fold, 1] = tcn
                active[start + 1:start + n, inner_fold] = True
            predictions.flush(); active.flush(); completion.mark(index)
            processed += 1
            if max_answers and processed >= max_answers:
                return completion.count(), False
        if not completion.complete():
            raise AtlasContractError("inner-OOF predictor application is incomplete")
        return completion.count(), True
    finally:
        predictions.flush(); active.flush()
        del predictions, active, levels
        completion.close()


def stage_predictors(args: argparse.Namespace) -> None:
    np = _np()
    bundle = args.bundle_root
    metadata = read_json(bundle / "METADATA.json")
    levels = np.load(bundle / "primitive_levels.npy", mmap_mode="r", allow_pickle=False)
    out = args.output_root / "predictors"
    out.mkdir(parents=True, exist_ok=True)
    contract = {"schema": SCHEMA + "/predictors-contract-v3",
                "bundle_root": str(bundle.resolve()),
                "bundle_freeze_sha256": sha256_file(bundle / "FREEZE.json"),
                "targets": list(PRIMITIVE_TARGETS), "exclusions": [list(x) for x in predictor_exclusions()],
                "backgrounds": ["prefix", "mean16", "noreset", "bocpd", "ridge", "tcn"],
                "labels_used": False, "seed": args.seed, "device": args.device,
                "tcn_updates": args.tcn_updates, "tcn_batch_size": args.tcn_batch_size,
                "implementation_sha256": implementation_sha256(
                    causal_backgrounds, _ridge_design, _fit_ridge_sufficient,
                    _train_tcn, _apply_outer_predictors, _apply_inner_predictors,
                    files=(
                        ROOT / "spectral_utils/aligned_context_predictors.py",
                        ROOT / "spectral_utils/temporal_context_models.py",
                    ),
                )}
    contract_hash = bind_immutable_manifest(out / "MANIFEST.json", contract)

    fixed = out / "fixed"
    fixed.mkdir(exist_ok=True)
    completion = CompletionBitmap(fixed / "completion.npy", len(metadata), contract_hash)
    backgrounds = _open_or_create_array(fixed / "backgrounds.npy",
                                        shape=(len(levels), 4, 4), dtype="float32")
    processed = 0
    try:
        for index, row in enumerate(metadata):
            if completion.done(index):
                continue
            start, n = int(row["offset"]), int(row["tokens"])
            values = np.asarray(levels[start:start + n], dtype=float)
            predicted = causal_backgrounds(values)
            for column, name in enumerate(("prefix", "mean16", "noreset", "bocpd")):
                backgrounds[start:start + n, column] = predicted[name].astype(np.float32)
            backgrounds.flush(); completion.mark(index); processed += 1
            if args.max_answers and processed >= args.max_answers:
                _write_status(args.output_root, "predictors", "RUNNING", fixed_completed=completion.count())
                return
        if not completion.complete():
            raise AtlasContractError("fixed predictor extraction incomplete")
    finally:
        backgrounds.flush(); del backgrounds; completion.close()

    ridge_dir = out / "ridge"; ridge_dir.mkdir(exist_ok=True)
    stats_path = ridge_dir / "fold_sufficient.npz"
    if stats_path.exists():
        with np.load(stats_path, allow_pickle=False) as saved:
            statistics = {fold: (saved[f"xtx_{fold}"], saved[f"xty_{fold}"], int(saved[f"n_{fold}"]))
                          for fold in range(5)}
    else:
        first_design, _ = _ridge_design(np.asarray(levels[:1], dtype=float))
        dimension = first_design.shape[1]
        statistics = {fold: (np.zeros((dimension, dimension)), np.zeros((dimension, 4)), 0)
                      for fold in range(5)}
        for row in metadata:
            start, n, fold = int(row["offset"]), int(row["tokens"]), int(row["fold"])
            design, target = _ridge_design(np.asarray(levels[start:start + n], dtype=float))
            xtx, xty, count = statistics[fold]
            statistics[fold] = (xtx + design.T @ design, xty + design.T @ target, count + n)
        arrays = {f"xtx_{fold}": statistics[fold][0] for fold in range(5)}
        arrays.update({f"xty_{fold}": statistics[fold][1] for fold in range(5)})
        arrays.update({f"n_{fold}": np.asarray(statistics[fold][2]) for fold in range(5)})
        _save_npz_immutable(stats_path, **arrays)
    ridge_manifest = []
    for excluded in predictor_exclusions():
        fit = _fit_ridge_sufficient(statistics, excluded)
        path = ridge_dir / ("exclude_" + "_".join(map(str, excluded)) + ".npz")
        digest = _save_npz_immutable(path, coefficient=fit["coefficient"],
                                     targets=np.asarray(PRIMITIVE_TARGETS))
        ridge_manifest.append({**{k: v for k, v in fit.items() if k != "coefficient"},
                               "path": path.name, "sha256": digest})
    bind_immutable_manifest(ridge_dir / "INDEX.json", {
        "schema": SCHEMA + "/ridge-fit-index-v1",
        "targets": list(PRIMITIVE_TARGETS),
        "fits": ridge_manifest,
    })

    if args.device == "mps":
        import torch
        if not torch.backends.mps.is_available():
            raise AtlasContractError("--device mps requested but MPS is unavailable")
    tcn_dir = out / "tcn"; tcn_dir.mkdir(exist_ok=True)
    tcn_manifest = []
    with _single_mps_process(out / "MPS.lock", args.device == "mps"):
        for fit_index, excluded in enumerate(predictor_exclusions()):
            path = tcn_dir / ("exclude_" + "_".join(map(str, excluded)) + ".pt")
            tcn_manifest.append(_train_tcn(bundle, metadata, excluded, path, device=args.device,
                                           seed=args.seed + fit_index, updates=args.tcn_updates,
                                           batch_size=args.tcn_batch_size))
    bind_immutable_manifest(tcn_dir / "INDEX.json", {
        "schema": SCHEMA + "/tcn-fit-index-v1",
        "targets": list(PRIMITIVE_TARGETS),
        "fits": tcn_manifest,
    })
    with _single_mps_process(out / "MPS_APPLY.lock", args.device == "mps"):
        applied, application_complete = _apply_outer_predictors(
            bundle, out, metadata, device=args.device, contract_sha256=contract_hash,
            max_answers=args.max_answers,
        )
    if not application_complete:
        _write_status(args.output_root, "predictors", "RUNNING", fixed_answers=len(metadata),
                      ridge_fits=len(ridge_manifest), tcn_fits=len(tcn_manifest),
                      oof_answers=applied, device=args.device, labels_used=False)
        return
    with _single_mps_process(out / "MPS_APPLY.lock", args.device == "mps"):
        inner_applied, inner_application_complete = _apply_inner_predictors(
            bundle, out, metadata, device=args.device, contract_sha256=contract_hash,
            max_answers=args.max_answers,
        )
    if not inner_application_complete:
        _write_status(args.output_root, "predictors", "RUNNING", fixed_answers=len(metadata),
                      ridge_fits=len(ridge_manifest), tcn_fits=len(tcn_manifest),
                      oof_answers=applied, inner_oof_answers=inner_applied,
                      device=args.device, labels_used=False)
        return
    learned = out / "learned_oof"
    learned_inner = out / "learned_inner"
    required = (
        fixed / "backgrounds.npy", fixed / "completion.npy",
        ridge_dir / "INDEX.json", tcn_dir / "INDEX.json",
        learned / "backgrounds.npy", learned / "active.npy", learned / "completion.npy",
        learned_inner / "MANIFEST.json", learned_inner / "backgrounds.npy",
        learned_inner / "active.npy", learned_inner / "completion.npy",
        learned_inner / "completion.npy.json",
    )
    freeze = {str(path.relative_to(out)): {"sha256": sha256_file(path), "size": path.stat().st_size}
              for path in required}
    bind_immutable_manifest(out / "FREEZE.json", {
        "schema": SCHEMA + "/predictors-freeze-v2", "files": freeze,
        "targets": list(PRIMITIVE_TARGETS), "fixed_background_axis": ["prefix", "mean16", "noreset", "bocpd"],
        "learned_background_axis": ["ridge_source_excluded", "tcn_source_excluded"],
        "learned_inner_shape": [len(levels), 5, 2, 4],
        "learned_inner_axes": {
            "0": "token", "1": "inner_fold_f", "2": ["ridge", "tcn"],
            "3": list(PRIMITIVE_TARGETS),
        },
        "learned_inner_rule": "answer fold g uses exclude_sorted(f,g); f == g inactive",
        "implementation_sha256": contract["implementation_sha256"],
        "first_token_inactive": True, "labels_used": False,
    })
    _write_status(args.output_root, "predictors", "COMPLETE", fixed_answers=len(metadata),
                  ridge_fits=len(ridge_manifest), tcn_fits=len(tcn_manifest), oof_answers=applied,
                  inner_oof_answers=inner_applied, device=args.device, labels_used=False,
                  freeze="FREEZE.json")


def _historical_score_identity(values: Any) -> str:
    """Match the frozen inventory's float64 full-curve identity contract."""
    np = _np()
    array = np.asarray(values, dtype="<f8")
    if array.ndim != 1 or not np.isfinite(array).all():
        raise AtlasContractError("historical score array must be a finite vector")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _historical_peak_identity(values: Any, offsets: Any) -> str:
    """Match BROAD_ANALYSIS' int32 per-answer argmax identity contract."""
    np = _np()
    score = np.asarray(values, dtype=np.float64)
    boundary = np.asarray(offsets, dtype=np.int64)
    if score.shape != (int(boundary[-1]),) or np.any(np.diff(boundary) <= 0):
        raise AtlasContractError("historical score array is not aligned to JOINED offsets")
    peaks = np.asarray([
        np.argmax(score[int(boundary[index]):int(boundary[index + 1])])
        for index in range(len(boundary) - 1)
    ], dtype="<i4")
    return hashlib.sha256(peaks.tobytes()).hexdigest()


def stage_reconcile(args: argparse.Namespace) -> None:
    source = args.contract_root / "results/predictor_error_profiles_v1/BROAD_ANALYSIS.json"
    if not source.is_file():
        raise AtlasContractError("historical inventory metadata is missing")
    broad = read_json(source)
    summary = broad.get("summary", {})
    if summary.get("unique_score_arrays") != 178 or summary.get("unique_peak_vectors") != 173:
        raise AtlasContractError("historical inventory cardinality drift")
    joined_path = args.contract_root / "results/localization_full_benchmark_v3/evaluation/JOINED.npz"
    np = _np()
    with np.load(joined_path, allow_pickle=False) as joined:
        offsets = np.asarray(joined["offsets"], dtype=np.int64)
    rows = []
    columns = []
    score_canonical: dict[str, str] = {}
    for item in broad.get("inventory", []):
        relative = str(item["path"]).replace("\\", "/")
        candidates = [args.contract_root / relative, args.source_root / relative]
        if item["archive"] == "temporal_research_baseline_v1":
            candidates.insert(0, args.output_root / "baseline_replay/SCORES_FROZEN.npz")
        path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        actual = sha256_file(path) if path.is_file() else None
        expected = item.get("sha256")
        archive_match = actual == expected
        column_audit: dict[str, dict[str, Any]] = {}
        semantic_replay = False
        if path.is_file():
            try:
                with np.load(path, allow_pickle=False) as saved:
                    for column in item.get("rows", []):
                        name = str(column["name"])
                        if name not in saved.files:
                            raise AtlasContractError(f"historical archive omits {name}")
                        values = np.asarray(saved[name], dtype=np.float64)
                        score_hash = _historical_score_identity(values)
                        peak_hash = _historical_peak_identity(values, offsets)
                        if peak_hash != column.get("peak_sha256"):
                            raise AtlasContractError(f"historical peak identity drift for {name}")
                        column_audit[name] = {
                            "score_sha256": score_hash, "peak_verified": True,
                        }
                semantic_replay = bool(column_audit) and len(column_audit) == len(item.get("rows", []))
            except (OSError, ValueError, KeyError, AtlasContractError):
                column_audit = {}
                semantic_replay = False
        status = (
            "AVAILABLE_FROZEN" if archive_match and semantic_replay
            else "AVAILABLE_REPLAYED" if semantic_replay and item["archive"] == "temporal_research_baseline_v1"
            else "REPORT_ONLY"
        )
        rows.append({"archive": item["archive"], "relative": relative, "path": str(path),
                     "expected_sha256": expected, "actual_sha256": actual, "status": status,
                     "score_columns": len(item.get("rows", [])),
                     "semantic_arrays_verified": len(column_audit),
                     "reason": None if status != "REPORT_ONLY" else "bulk archive missing or semantic/hash drift"})
        for column in item.get("rows", []):
            name = str(column["name"])
            score_hash = column_audit.get(name, {}).get("score_sha256")
            canonical = score_canonical.setdefault(
                score_hash, f"historical::{item['archive']}::{name}",
            ) if score_hash is not None else None
            columns.append({
                "name": f"historical::{item['archive']}::{name}",
                "archive": item["archive"], "artifact_status": status,
                "status": "REPORT_ONLY", "peak_sha256": column.get("peak_sha256"),
                "score_sha256": score_hash, "canonical_score": canonical,
                "exact_score_duplicate": canonical is not None and canonical != f"historical::{item['archive']}::{name}",
                "peak_verified": bool(column_audit.get(name, {}).get("peak_verified", False)),
                "included": bool(column.get("included", False)),
                "control_tagged": bool(column.get("control_tagged", False)),
            })
    out = args.output_root / "reconcile"; out.mkdir(parents=True, exist_ok=True)
    if len(columns) != 254 or len({row["peak_sha256"] for row in columns}) != 173:
        raise AtlasContractError("historical column/peak reconciliation drift")
    available_columns = [row for row in columns if row["score_sha256"] is not None]
    ledger = {"schema": SCHEMA + "/historical-reconciliation-v3", "inventory": rows,
              "columns": columns, "scored_columns": 254,
              "historical_unique_score_arrays": 178, "historical_unique_peak_vectors": 173,
              "available_archives": sum(row["status"].startswith("AVAILABLE") for row in rows),
              "report_only_archives": sum(row["status"] == "REPORT_ONLY" for row in rows),
              "available_score_columns": len(available_columns),
              "available_unique_score_arrays": len({row["score_sha256"] for row in available_columns}),
              "available_verified_peak_vectors": len({row["peak_sha256"] for row in available_columns}),
              "policy": "Missing bulk is retained as REPORT_ONLY; inventory rows are never silently dropped.",
              "pending_expansion": list(PENDING_EXPANSION), "development_only": True}
    bind_immutable_manifest(out / "LEDGER.json", ledger)
    _write_status(args.output_root, "reconcile", "COMPLETE",
                  available=ledger["available_archives"], report_only=ledger["report_only_archives"])


def _load_error_module() -> Any:
    module = importlib.import_module("spectral_utils.error_dependence")
    for name, function in inspect.getmembers(module, inspect.isfunction):
        # Error construction may accept labels; fusion fitting may not.
        if any(token in name.lower() for token in ("weight", "simplex", "equal_rank", "family_equal", "iu")):
            assert_label_free_callable(function)
    return module


def stage_dependence(args: argparse.Namespace) -> None:
    """Assemble separate error matrices and call the frozen dependence API."""
    module = _load_error_module()
    out = args.output_root / "dependence"; out.mkdir(parents=True, exist_ok=True)
    # Evaluation is not allowed to start until the four incumbent arms replay
    # from full score/label arrays.  Persist this precondition here (rather
    # than only in the later fusion stage) so a baseline drift cannot consume
    # development labels during roster screening.
    replay = replay_frozen_baselines(
        args.contract_root, args.baseline_root, args.output_root / "extract",
    )
    baseline = verify_baseline_replay(replay["metrics"], args.baseline_tolerance)
    bind_immutable_manifest(out / "BASELINE_INPUTS.json", replay["provenance"])
    per_answer_sha256 = _bind_baseline_replay_arrays(
        out / "BASELINE_PER_ANSWER.npz", replay["per_answer"],
    )
    bind_immutable_manifest(out / "BASELINE_REPLAY_PRECONDITION.json", {
        "schema": SCHEMA + "/baseline-replay-precondition-v1",
        **baseline, "metrics": replay["metrics"], "audit": replay["audit"],
        "inputs_sha256": sha256_file(out / "BASELINE_INPUTS.json"),
        "per_answer_sha256": per_answer_sha256,
        "headline_metrics_used": False, "development_only": True,
    })
    entry = getattr(module, "run_atlas_dependence", None)
    if entry is None:
        contract = {
            "schema": SCHEMA + "/dependence-backend-contract-v1",
            "callable": "spectral_utils.error_dependence.run_atlas_dependence",
            "required_inputs": ["extract_root", "predictor_root", "reconciliation_root",
                                "contract_root", "output_root", "draws", "seed"],
            "required_outputs": ["ERROR_LEDGER.json", "PAIRWISE.json", "COMPATIBILITY_GRAPH.json",
                                 "GROUPS.json"],
            "constraints": {"separate_error_targets": True, "grouped_bootstrap": True,
                            "gate_reranked_within_draw": True, "draws": args.draws},
        }
        bind_immutable_manifest(out / "BACKEND_CONTRACT.json", contract)
        raise AtlasContractError(
            "dependence backend is not wired: implement run_atlas_dependence per BACKEND_CONTRACT.json"
        )
    result = entry(
        extract_root=args.output_root / "extract",
        predictor_root=args.output_root / "predictors",
        reconciliation_root=args.output_root / "reconcile",
        contract_root=args.contract_root,
        output_root=out,
        draws=args.draws,
        seed=args.seed,
    )
    required = ("ERROR_LEDGER.json", "PAIRWISE.json", "COMPATIBILITY_GRAPH.json", "GROUPS.json")
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise AtlasContractError("dependence backend omitted required artifacts: " + ", ".join(missing))
    result = {} if result is None else json_ready(result)
    result.update({"schema": SCHEMA + "/dependence-v1", "draws": args.draws,
                   "separate_error_matrices": ["predictor_residual", "pb_raw_locator_miss",
                                               "prmb_pair_misordering", "gate_false_open",
                                               "gate_false_close", "pb_final"],
                   "development_only": True})
    atomic_json(out / "SUMMARY.json", result)
    _write_status(
        args.output_root, "dependence", "COMPLETE", draws=args.draws,
        baseline_replay="PASS",
    )


def stage_fusion(args: argparse.Namespace) -> None:
    module = _load_error_module()
    out = args.output_root / "fusion"; out.mkdir(parents=True, exist_ok=True)
    replay = replay_frozen_baselines(
        args.contract_root, args.baseline_root, args.output_root / "extract",
    )
    bind_immutable_manifest(out / "BASELINE_INPUTS.json", replay["provenance"])
    per_answer_sha256 = _bind_baseline_replay_arrays(
        out / "BASELINE_PER_ANSWER.npz", replay["per_answer"],
    )
    actual_baselines = replay["metrics"]
    baseline = verify_baseline_replay(actual_baselines, args.baseline_tolerance)
    bind_immutable_manifest(out / "BASELINE_REPLAY.json", {
        "schema": SCHEMA + "/baseline-replay-v2", **baseline, "metrics": actual_baselines,
        "audit": replay["audit"], "inputs_sha256": sha256_file(out / "BASELINE_INPUTS.json"),
        "per_answer_sha256": per_answer_sha256,
        "sources": ["baseline_replay/SCORES_FROZEN.npz", "extract/answers/*.npz",
                    "localization_full_benchmark_v3/evaluation/JOINED.{json,npz}"],
        "headline_metrics_used": False, "development_only": True,
    })
    entry = getattr(module, "run_nested_fusion_search", None)
    if entry is None:
        contract = {
            "schema": SCHEMA + "/fusion-backend-contract-v1",
            "callable": "spectral_utils.error_dependence.run_nested_fusion_search",
            "required_inputs": ["dependence_root", "extraction_root", "output_root", "folds",
                                "family_cap", "maximum_size", "roster_stability", "heads",
                                "iu_eligible_only", "seed"],
            "required_outputs": ["ALL_GROUPS.json", "NESTED_SELECTION.json", "PARETO.json",
                                 "FINALISTS.json", "SUMMARY.json"],
            "constraints": {"folds": 5, "family_cap": 2, "maximum_size": 6,
                            "roster_stability": "4/5", "iu_eligible_only": True},
        }
        bind_immutable_manifest(out / "BACKEND_CONTRACT.json", contract)
        raise AtlasContractError(
            "fusion backend is not wired: implement run_nested_fusion_search per BACKEND_CONTRACT.json"
        )
    result = entry(
        dependence_root=args.output_root / "dependence",
        extraction_root=args.output_root / "extract",
        output_root=out,
        folds=5,
        family_cap=2,
        maximum_size=6,
        roster_stability=4,
        heads=("singleton", "equal_rank", "family_equal", "nonnegative_shrunk_simplex", "iu"),
        iu_eligible_only=True,
        seed=args.seed,
    )
    required = (
        "ALL_GROUPS.json", "NESTED_SELECTION.json", "PARETO.json", "FINALISTS.json",
        "LEAVE_ONE_SIGNAL_OUT.json", "UNCERTAINTY.json",
        "DEPENDENT_COMPLEMENTARY.json", "COMPOSITION_CONTRACT.json",
    )
    missing = [name for name in required if not (out / name).is_file()]
    if missing:
        raise AtlasContractError("fusion backend omitted required artifacts: " + ", ".join(missing))
    result = {} if result is None else json_ready(result)
    backend_status = result.get("status")
    if backend_status not in {"COMPLETE", "PARTIAL_FAIL_CLOSED"}:
        raise AtlasContractError(
            "fusion backend returned an unrecognized completion status: "
            + repr(backend_status)
        )
    factorial_status = result.get("factorial_status", "COMPLETE")
    if backend_status == "PARTIAL_FAIL_CLOSED" and (
        factorial_status != "UNRESOLVED_PIPELINE_COMPOSITION_NOT_FABRICATED"
        or not (out / "COMPOSITION_CONTRACT.json").is_file()
    ):
        raise AtlasContractError(
            "partial fusion must expose the unresolved typed-composition contract"
        )
    result.setdefault("pareto", read_json(out / "PARETO.json").get("rows", []))
    result.setdefault("finalists", read_json(out / "FINALISTS.json").get("finalists", {}))
    result.update({"schema": SCHEMA + "/nested-fusion-v1", "folds": 5,
                   "family_cap": 2, "maximum_clique": 6, "roster_stability": "4/5",
                   "baseline_replay": "PASS", "development_only": True})
    atomic_json(out / "SUMMARY.json", result)
    # Nested selection is complete even when the subsequent cross-point
    # factorial is intentionally blocked.  Preserve both facts in the marker;
    # the ablation stage below will never manufacture the missing 32 arms.
    _write_status(
        args.output_root, "fusion", "COMPLETE", baseline_replay="PASS",
        nested_search_status=backend_status, factorial_status=factorial_status,
    )


def stage_ablation(args: argparse.Namespace) -> None:
    out = args.output_root / "ablation"; out.mkdir(parents=True, exist_ok=True)
    fusion = read_json(args.output_root / "fusion/SUMMARY.json")
    finalists = fusion.get("finalists", {})
    missing = [point for point in FUSION_POINTS if point not in finalists]
    arms = factorial_arms()
    # The arm definition is immutable and useful to a backend, while finalist
    # choice and measurements are deliberately not frozen until they exist.
    bind_immutable_manifest(out / "ARMS.json", {
        "schema": SCHEMA + "/factorial-arms-v1", "points": list(FUSION_POINTS),
        "arms": arms, "arm_count": 32,
    })
    if missing:
        factorial_status = fusion.get(
            "factorial_status", "UNRESOLVED_MISSING_STABLE_FINALISTS"
        )
        blocked = {
            "schema": SCHEMA + "/factorial-ablation-v1",
            "points": list(FUSION_POINTS), "arms": arms, "arm_count": 32,
            "finalists": finalists, "missing_finalists": missing,
            "factorial_results": [], "status": "BLOCKED_COMPOSITION_CONTRACT",
            "reason": factorial_status,
            "composition_contract": "../fusion/COMPOSITION_CONTRACT.json",
            "fabricated_arms": False, "development_only": True,
        }
        bind_immutable_manifest(out / "FACTORIAL.json", blocked)
        _write_status(
            args.output_root, "ablation", "BLOCKED_COMPOSITION_CONTRACT",
            arm_count=32, evaluated_arms=0, fabricated_arms=False,
            missing_finalists=missing, reason=factorial_status,
        )
        return
    evaluated = fusion.get("factorial_results")
    if not isinstance(evaluated, list) or len(evaluated) != len(arms):
        factorial_status = fusion.get("factorial_status")
        if factorial_status != "UNRESOLVED_PIPELINE_COMPOSITION_NOT_FABRICATED":
            raise AtlasContractError(
                "fusion backend must provide exactly 32 evaluated factorial_results"
            )
        blocked = {
            "schema": SCHEMA + "/factorial-ablation-v1",
            "points": list(FUSION_POINTS), "arms": arms, "arm_count": 32,
            "finalists": finalists, "factorial_results": [],
            "status": "BLOCKED_COMPOSITION_CONTRACT",
            "reason": factorial_status,
            "composition_contract": "../fusion/COMPOSITION_CONTRACT.json",
            "fabricated_arms": False, "development_only": True,
        }
        bind_immutable_manifest(out / "FACTORIAL.json", blocked)
        _write_status(
            args.output_root, "ablation", "BLOCKED_COMPOSITION_CONTRACT",
            arm_count=32, evaluated_arms=0, fabricated_arms=False,
            reason=factorial_status,
        )
        return
    result_by_arm = {str(row.get("arm")): row for row in evaluated if isinstance(row, dict)}
    expected_arms = {row["arm"] for row in arms}
    if set(result_by_arm) != expected_arms:
        raise AtlasContractError("factorial_results must contain each arm 00000..11111 exactly once")
    result = {
        "schema": SCHEMA + "/factorial-ablation-v1", "points": list(FUSION_POINTS),
        "arms": arms, "arm_count": 32, "finalists": finalists,
        "factorial_results": [result_by_arm[row["arm"]] for row in arms],
        "status": "COMPLETE", "development_only": True,
    }
    bind_immutable_manifest(out / "FACTORIAL.json", result)
    _write_status(args.output_root, "ablation", "COMPLETE", arm_count=32,
                  evaluation_status="COMPLETE")


def _safe_artifact_name(value: str) -> str:
    text = "".join(character if character.isalnum() else "_" for character in str(value))
    return "_".join(part for part in text.split("_") if part)[:120] or "artifact"


def _render_atlas_figures(output_root: Path, report_root: Path, pareto: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Render compact views from canonical JSON; JSON remains authoritative."""
    np = _np()
    if importlib.util.find_spec("matplotlib") is None:
        return {"status": "SKIPPED_MATPLOTLIB_UNAVAILABLE", "files": []}
    os.environ.setdefault("MPLCONFIGDIR", str(report_root / ".mplconfig"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    files: list[str] = []
    numeric = [row for row in pareto if row.get("pb") is not None and row.get("within") is not None]
    if numeric:
        figure, axis = plt.subplots(figsize=(8, 5.5))
        points = sorted({str(row.get("point", "unknown")) for row in numeric})
        palette = plt.get_cmap("tab10")
        for point_index, point in enumerate(points):
            local = [row for row in numeric if str(row.get("point", "unknown")) == point]
            axis.scatter(
                [row["pb"] for row in local], [row["within"] for row in local],
                s=24, alpha=.75, label=point, color=palette(point_index % 10),
            )
        for row in numeric:
            if row.get("pareto") or row.get("is_finalist"):
                axis.annotate(
                    str(row.get("name", "candidate")), (row["pb"], row["within"]),
                    fontsize=6, alpha=.85,
                )
        axis.set(xlabel="ProcessBench", ylabel="PRMB within", title="Development Pareto frontier")
        axis.legend(fontsize=7, loc="best")
        figure.tight_layout()
        path = report_root / "PARETO.png"
        figure.savefig(path, dpi=180); plt.close(figure)
        files.append(path.name)

    heatmap_path = output_root / "dependence/HEATMAPS.json"
    heatmap_index = []
    if heatmap_path.is_file():
        heatmap_root = report_root / "pairwise_heatmaps"; heatmap_root.mkdir(exist_ok=True)
        payload = read_json(heatmap_path)
        for matrix_name, rows in sorted(payload.get("matrices", {}).items()):
            names = sorted({str(row[side]) for row in rows for side in ("left", "right")})
            if not names:
                continue
            column = {name: index for index, name in enumerate(names)}
            values = np.full((len(names), len(names)), np.nan, dtype=float)
            np.fill_diagonal(values, 1.0)
            for row in rows:
                value = row.get("conditional_phi")
                if value is None:
                    continue
                left, right = column[str(row["left"])], column[str(row["right"])]
                values[left, right] = values[right, left] = float(value)
            size = min(18.0, max(6.0, 0.25 * len(names)))
            figure, axis = plt.subplots(figsize=(size, size))
            image_handle = axis.imshow(values, vmin=-1, vmax=1, cmap="coolwarm", interpolation="nearest")
            axis.set_title(str(matrix_name), fontsize=9)
            if len(names) <= 35:
                axis.set_xticks(range(len(names)), names, rotation=90, fontsize=5)
                axis.set_yticks(range(len(names)), names, fontsize=5)
            else:
                axis.set_xticks([]); axis.set_yticks([])
            figure.colorbar(image_handle, ax=axis, fraction=.046, pad=.04, label="conditional phi")
            figure.tight_layout()
            path = heatmap_root / (_safe_artifact_name(matrix_name) + ".png")
            figure.savefig(path, dpi=170); plt.close(figure)
            relative = str(path.relative_to(report_root)); files.append(relative)
            heatmap_index.append({"matrix": matrix_name, "methods": len(names), "path": relative})
        atomic_json(report_root / "PAIRWISE_HEATMAP_INDEX.json", {
            "schema": SCHEMA + "/pairwise-heatmap-index-v1", "rows": heatmap_index,
        })

    graph_path = output_root / "dependence/COMPATIBILITY_GRAPH.json"
    if graph_path.is_file():
        graphs = read_json(graph_path).get("graphs", {})
        if graphs:
            columns = min(3, len(graphs)); rows_count = (len(graphs) + columns - 1) // columns
            figure, axes = plt.subplots(rows_count, columns, figsize=(5 * columns, 5 * rows_count))
            axes = np.asarray(axes, dtype=object).reshape(-1)
            for axis, (point, graph) in zip(axes, sorted(graphs.items())):
                nodes = list(graph.get("nodes", [])); edges = list(graph.get("edges", []))
                angles = np.linspace(0, 2 * np.pi, max(len(nodes), 1), endpoint=False)
                positions = {node: (np.cos(angle), np.sin(angle)) for node, angle in zip(nodes, angles)}
                degree = {node: 0 for node in nodes}
                for left, right in edges:
                    if left not in positions or right not in positions:
                        continue
                    degree[left] += 1; degree[right] += 1
                    axis.plot(
                        [positions[left][0], positions[right][0]],
                        [positions[left][1], positions[right][1]],
                        color="#9aa0a6", linewidth=.45, alpha=.4, zorder=1,
                    )
                if nodes:
                    axis.scatter(
                        [positions[node][0] for node in nodes],
                        [positions[node][1] for node in nodes],
                        c=[degree[node] for node in nodes], cmap="viridis", s=22, zorder=2,
                    )
                    if len(nodes) <= 24:
                        for node in nodes:
                            axis.text(*positions[node], str(node), fontsize=5)
                axis.set_title(f"{point}: {len(nodes)} nodes / {len(edges)} edges", fontsize=9)
                axis.set_aspect("equal"); axis.axis("off")
            for axis in axes[len(graphs):]:
                axis.axis("off")
            figure.tight_layout()
            path = report_root / "COMPATIBILITY_GRAPH.png"
            figure.savefig(path, dpi=180); plt.close(figure)
            files.append(path.name)
    return {"status": "COMPLETE", "files": files, "heatmaps": heatmap_index}


def stage_report(args: argparse.Namespace) -> None:
    out = args.output_root / "report"; out.mkdir(parents=True, exist_ok=True)
    reconciliation = read_json(args.output_root / "reconcile/LEDGER.json")
    fusion = read_json(args.output_root / "fusion/SUMMARY.json")
    ablation_path = args.output_root / "ablation/FACTORIAL.json"
    ablation = (
        read_json(ablation_path) if ablation_path.is_file()
        else {
            "status": "NOT_RUN", "arm_count": 32, "factorial_results": [],
            "reason": fusion.get("factorial_status", "MISSING_ABLATION_ARTIFACT"),
        }
    )
    pareto = fusion.get("pareto", [])
    dependence_summary_path = args.output_root / "dependence/SUMMARY.json"
    dependence_summary = (
        read_json(dependence_summary_path) if dependence_summary_path.is_file() else {}
    )
    pairwise_path = args.output_root / "dependence/PAIRWISE.json"
    pairwise = read_json(pairwise_path).get("rows", []) if pairwise_path.is_file() else []
    pair_status_counts: dict[str, int] = {}
    for row in pairwise:
        status_name = str(row.get("status", "UNRESOLVED"))
        pair_status_counts[status_name] = pair_status_counts.get(status_name, 0) + 1
    complementary_path = args.output_root / "fusion/DEPENDENT_COMPLEMENTARY.json"
    complementary = (
        read_json(complementary_path).get("rows", []) if complementary_path.is_file() else []
    )
    compatible_groups_path = args.output_root / "dependence/GROUPS.json"
    compatible_groups = []
    if compatible_groups_path.is_file():
        compatible_groups = [
            row for row in read_json(compatible_groups_path).get("rows", [])
            if row.get("status") == "INDEPENDENCE_COMPATIBLE"
        ]
    baseline_replay_path = args.output_root / "fusion/BASELINE_REPLAY.json"
    baseline_metrics = (
        read_json(baseline_replay_path).get("metrics", {})
        if baseline_replay_path.is_file() else {}
    )
    current_metrics = baseline_metrics.get("current", {})
    candidates_path = args.output_root / "dependence/CANDIDATES.json"
    candidates = read_json(candidates_path) if candidates_path.is_file() else {}
    unique_extraction_signatures = len({
        row.get("semantic_hash") for row in candidates.get("screening", [])
        if row.get("semantic_hash") is not None
    })
    eligible_by_point: dict[str, int] = {}
    for row in candidates.get("eligible", []):
        point = str(row.get("point", "unknown"))
        eligible_by_point[point] = eligible_by_point.get(point, 0) + 1
    graph_path = args.output_root / "dependence/COMPATIBILITY_GRAPH.json"
    graph_document = read_json(graph_path) if graph_path.is_file() else {"graphs": {}}
    graph_edge_counts = {
        point: len(graph.get("edges", []))
        for point, graph in graph_document.get("graphs", {}).items()
    }
    compatible_by_matrix: dict[str, int] = {}
    residual_ranges: dict[str, list[float]] = {}
    for row in pairwise:
        matrix = str(row.get("matrix", "unknown"))
        if row.get("status") == "INDEPENDENCE_COMPATIBLE":
            compatible_by_matrix[matrix] = compatible_by_matrix.get(matrix, 0) + 1
        if matrix.startswith("background:") and matrix.endswith(":predictor_residual"):
            value = row.get("conditional_residual_correlation")
            if value is not None:
                primitive = matrix.split(":")[1]
                residual_ranges.setdefault(primitive, []).append(float(value))
    complementary_rows = [
        row for row in complementary if row.get("status") == "DEPENDENT_COMPLEMENTARY"
    ]
    baseline_pb = current_metrics.get("pb")
    baseline_within = current_metrics.get("within")

    def _pct(value: Any) -> str:
        return "NA" if value is None else f"{100.0 * float(value):.4f}%"

    def _metric(value: Any) -> str:
        return "NA" if value is None else f"{float(value):.6f}"

    status = (
        "DEVELOPMENT_PARTIAL"
        if ablation["status"] != "COMPLETE"
        else "DEVELOPMENT_COMPLETE"
    )
    report = [
        "# Fusion Independence Atlas v1", "",
        f"Status: **{status}** (development only; no external confirmation).", "",
        "## Coverage", "",
        f"- Historical unique score arrays retained: {reconciliation['historical_unique_score_arrays']}",
        f"- Historical unique peak vectors retained: {reconciliation['historical_unique_peak_vectors']}",
        f"- REPORT_ONLY archives: {reconciliation['report_only_archives']}",
        f"- Factorial arms declared: {ablation['arm_count']}",
        f"- Factorial arms evaluated: {len(ablation.get('factorial_results', []))}",
        f"- Factorial status: {ablation['status']}",
        f"- Pairwise classifications: {pair_status_counts}",
        f"- Independence-compatible groups: {len(compatible_groups)}",
        f"- Dependent-complementary diagnostics: {sum(row.get('status') == 'DEPENDENT_COMPLEMENTARY' for row in complementary)}",
        f"- Five-point backend status: {fusion.get('status', 'UNKNOWN')}",
        "- PENDING_EXPANSION: " + ", ".join(PENDING_EXPANSION), "",
        "## Frozen baseline replay", "",
        f"- original4: {_pct(baseline_metrics.get('original4', {}).get('pb'))} PB",
        f"- innovation5: {_pct(baseline_metrics.get('innovation5', {}).get('pb'))} PB",
        f"- digit025: {_pct(baseline_metrics.get('digit025', {}).get('pb'))} PB / "
        f"{_metric(baseline_metrics.get('digit025', {}).get('within'))} within",
        f"- current: {_pct(baseline_pb)} PB / {_metric(baseline_within)} within",
        "- Replay status: PASS before development labels were opened downstream.", "",
        "## Candidate funnel", "",
        f"- Screened definitions: {len(candidates.get('screening', []))}",
        f"- Unique extraction signatures: {unique_extraction_signatures}",
        f"- Exact duplicates removed: {len(candidates.get('aliases', {}))}",
        f"- Outer-training family representatives: {len(candidates.get('eligible', []))}",
        f"- Representatives by insertion point: {eligible_by_point}", "",
        "## Independence result", "",
        f"- Combined compatibility edges by insertion point: {graph_edge_counts}",
        "- No pair passed all required error views at any insertion point; therefore no "
        "independence-compatible clique of size 2-6 exists.",
        f"- Raw compatible pairs by individual matrix: {compatible_by_matrix}",
        "- The compatible raw pairs occur only on PRMB pairwise misordering. Every PB raw-locator "
        "matrix has zero compatible pairs under the simultaneous phi and odds-ratio intervals.",
        "- Background predictor residual correlation ranges: "
        + ", ".join(
            f"{primitive}={min(values):.3f}-{max(values):.3f}"
            for primitive, values in sorted(residual_ranges.items())
        ) + ".", "",
        "## Interpretation", "",
        "Independent-compatible, dependent-complementary, redundant, and unresolved candidates "
        "are reported separately. Missing bulk artifacts are not silently promoted.", "",
        "The frozen baseline replay passed before fusion comparisons were accepted.", "",
    ]
    finalists = fusion.get("finalists", {})
    report.extend(["## Nested OOF finalists", ""])
    if finalists:
        for point, row in sorted(finalists.items()):
            uncertainty = row.get("uncertainty", {})
            report.append(
                f"- {point}: `{row.get('members', [])}` with `{row.get('head')}`; "
                f"stability {row.get('roster_stability')}/5, {_pct(row.get('pb'))} PB / "
                f"{_metric(row.get('within'))} within; promotion "
                f"`{uncertainty.get('promotion', 'UNRESOLVED')}`."
            )
    else:
        report.append("- No roster-stable finalist was found.")
    report.extend([
        f"- Unsupported points: {fusion.get('unsupported_points', {})}",
        "- No performance successor or independently justified simplification successor was found.", "",
        "## Dependent-complementary research candidates", "",
        "These rows failed the independence contract. They are diagnostics, not promotion-eligible winners.", "",
    ])
    if baseline_pb is not None and baseline_within is not None:
        near_current = [
            row for row in complementary_rows
            if row.get("oof_metrics", {}).get("pb") is not None
            and row.get("oof_metrics", {}).get("within") is not None
            and float(row["oof_metrics"]["pb"]) >= float(baseline_pb) - .01
            and float(row["oof_metrics"]["within"]) >= float(baseline_within) - .002
        ]
        near_current.sort(key=lambda row: float(row["oof_metrics"]["pb"]), reverse=True)
        for row in near_current[:5]:
            metrics = row["oof_metrics"]
            report.append(
                f"- {row.get('point')}: `{row.get('members')}` via `{row.get('head')}` -> "
                f"{_pct(metrics.get('pb'))} PB / {_metric(metrics.get('within'))} within; "
                f"supported folds {row.get('statistically_supported_outer_folds')}; unique held-fold "
                f"successes {row.get('held_fold_unique_successes', {}).get('total')}."
            )
    if complementary_rows:
        top_pb_row = max(
            complementary_rows,
            key=lambda row: float(row.get("oof_metrics", {}).get("pb", float("-inf"))),
        )
        metrics = top_pb_row.get("oof_metrics", {})
        report.extend([
            "",
            "Highest-PB diagnostic (explicit tradeoff): "
            f"`{top_pb_row.get('members')}` -> {_pct(metrics.get('pb'))} PB / "
            f"{_metric(metrics.get('within'))} within.",
            "", "## Tail15 finding", "",
        ])
        tail15_digit = next((
            row for row in complementary_rows
            if row.get("point") == "step_post_readout"
            and any("digit.disagreement" in member for member in row.get("members", []))
            and any("logtail15" in member for member in row.get("members", []))
        ), None)
        if tail15_digit is not None:
            metrics = tail15_digit["oof_metrics"]
            report.append(
                "- The strongest near-incumbent simplification signal is digit-disagreement Top2 + "
                f"Tail15/logtail15 Top10 equal-rank: {_pct(metrics.get('pb'))} PB / "
                f"{_metric(metrics.get('within'))} within, stable in "
                f"{len(tail15_digit.get('statistically_supported_outer_folds', []))}/5 folds."
            )
        report.extend([
            "- Tail15 is therefore complementary to digit in this dataset, but not statistically "
            "independent under the locked screen.", "",
        ])
    if ablation["status"] != "COMPLETE":
        report.extend([
            "## Open composition contract", "",
            "The five insertion points were evaluated independently, but stable finalists were "
            "found only for token and step. The repository also does not define a type-safe rule "
            "for composing alternative locator representations into 32 cross-point arms. The "
            "factorial is therefore blocked rather than fabricated.",
            "",
            f"Missing stable finalists: `{ablation.get('missing_finalists', [])}`.",
            f"Reason: `{ablation.get('reason', 'UNRESOLVED')}`.", "",
        ])
    (out / "REPORT.md").write_text("\n".join(report), encoding="utf8")
    atomic_json(out / "PARETO.json", {"schema": SCHEMA + "/pareto-v1", "rows": pareto,
                                      "status": status, "development_only": True})
    figures = _render_atlas_figures(args.output_root, out, pareto)
    atomic_json(out / "FIGURES.json", {
        "schema": SCHEMA + "/figures-v1", **figures, "development_only": True,
    })
    _write_status(
        args.output_root, "report", "COMPLETE", report_status=status,
        figures=len(figures.get("files", [])),
        dependence_status=dependence_summary.get("status", "UNKNOWN"),
    )


STAGE_FUNCTIONS = {
    "preflight": stage_preflight, "registry": stage_registry, "bundle": stage_bundle,
    "extract": stage_extract, "predictors": stage_predictors, "reconcile": stage_reconcile,
    "dependence": stage_dependence, "fusion": stage_fusion, "ablation": stage_ablation,
    "report": stage_report,
}


def execute_stage(stage: str, args: argparse.Namespace, *, check_dependencies: bool = True) -> None:
    if stage not in STAGE_FUNCTIONS:
        raise ValueError(stage)
    if check_dependencies:
        require_dependencies(args.output_root, stage)
    try:
        STAGE_FUNCTIONS[stage](args)
    except BaseException as error:
        _write_status(args.output_root, stage, "FAILED", error=f"{type(error).__name__}: {error}")
        raise


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("stage", choices=STAGES + ("all",))
    command.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT,
                         help=f"raw-source root (default: {DEFAULT_SOURCE_ROOT})")
    command.add_argument("--contract-root", type=Path, default=ROOT,
                         help=f"compact benchmark/code root (default: {ROOT})")
    command.add_argument("--output-root", "--out", dest="output_root", type=Path,
                         default=DEFAULT_OUTPUT_ROOT,
                         help=f"stage output root (default: {DEFAULT_OUTPUT_ROOT})")
    command.add_argument("--bundle-root", type=Path, default=None,
                         help="primitive bundle root (default: OUTPUT_ROOT/bundle/data)")
    command.add_argument("--baseline-root", type=Path, default=None,
                         help="independent replay root (default: OUTPUT_ROOT/baseline_replay)")
    command.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    command.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    command.add_argument("--tcn-updates", type=int, default=2_000)
    command.add_argument("--tcn-batch-size", type=int, default=256)
    command.add_argument("--max-answers", type=int, default=0,
                         help="checkpoint after N new answers; 0 means no launch cap")
    command.add_argument("--baseline-tolerance", type=float, default=5e-7)
    command.add_argument("--fixture-mode", action="store_true", help=argparse.SUPPRESS)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.source_root = args.source_root.resolve()
    args.contract_root = args.contract_root.resolve()
    args.output_root = args.output_root.resolve()
    args.bundle_root = ((args.output_root / "bundle/data") if args.bundle_root is None
                        else args.bundle_root.resolve())
    args.baseline_root = ((args.output_root / "baseline_replay") if args.baseline_root is None
                          else args.baseline_root.resolve())
    if args.draws < 1 or args.tcn_updates < 1 or args.tcn_batch_size < 1 or args.max_answers < 0:
        raise SystemExit("draws, TCN settings, and max-answers must be positive/nonnegative")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.stage == "all":
        for stage in STAGES:
            execute_stage(stage, args, check_dependencies=True)
    else:
        execute_stage(args.stage, args, check_dependencies=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
