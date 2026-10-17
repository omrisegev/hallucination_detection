"""No-new-label infrastructure for the automatic group-free IU program.

This module contains only Phase-A0 structural utilities.  It deliberately does
not import the manual SpecRaGE provenance registry and none of its public fit or
audit functions accepts correctness labels.  The frozen mixed-v2 input contract
does, however, include signs and transforms selected in earlier labelled work;
callers must not describe that inherited input contract as label-naive.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import pickle
from typing import Iterable, Mapping, Sequence

import numpy as np

from .dufs_liu_feature_contract import (
    FEATURE_TRANSFORMS,
    dufs_liu_mixed_v2_from_bundle,
)
from .feature_contract import CONFIDENCE_FEATURE_SIGNS_V1
from .feature_utils import (
    FEAT_NAMES,
    compute_cusum_residuals,
    compute_hurst_exponent,
    compute_permutation_entropy,
    compute_spectral_features,
    compute_spilled_energy_features,
    compute_stft_features,
    compute_time_domain,
    extract_all_features,
)
from .repgrid_scoring import (
    ENERGY_FEATS,
    LOGPROB_FEATS,
    LOGPROB_FEATS_EXT,
    energy_features_from_logsumexp,
    logprob_features,
    logprob_features_extended,
)


PHASE_VERSION = "automatic-group-free-iu-a0-v1-2026-08-13"
PROCESSBENCH_MODEL_DIRS = (
    "pb_qwen3_4b",
    "pb_qwen3_8b",
    "pb_llama31_8b",
)
PROCESSBENCH_SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
REQUIRED_TELEMETRY = (
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
)
PROHIBITED_TARGET_KEYS = frozenset({
    "label",
    "labels",
    "final_answer_correct",
    "first_error",
    "error_label",
})


@dataclass(frozen=True)
class FactorialWorld:
    """Synthetic crossed-measurement world with evaluator-only latent targets."""

    environments: tuple[dict, ...]
    feature_names: tuple[str, ...]
    channels: tuple[str, ...]
    operators: tuple[str, ...]
    target_loading: np.ndarray
    difficulty_loading: np.ndarray
    seed: int
    environment_specific_target: bool


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_feature_names() -> tuple[str, ...]:
    """Return the 30-feature contract from the extractor-owned registries."""

    names = tuple(FEAT_NAMES + ENERGY_FEATS + LOGPROB_FEATS + LOGPROB_FEATS_EXT)
    if len(names) != 30 or len(set(names)) != len(names):
        raise RuntimeError("the mixed-v2 extractor registry is not a unique 30-feature pool")
    if set(names) != set(CONFIDENCE_FEATURE_SIGNS_V1):
        raise RuntimeError("extractor and orientation registries disagree")
    return names


def _function_record(function) -> dict:
    signature = inspect.signature(function)
    defaults = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
        and isinstance(parameter.default, (str, int, float, bool, type(None)))
    }
    source = Path(inspect.getsourcefile(function) or "unknown")
    try:
        line = int(inspect.getsourcelines(function)[1])
    except (OSError, TypeError):
        line = None
    return {
        "module": function.__module__,
        "function": function.__name__,
        "source_file": f"spectral_utils/{source.name}",
        "source_line": line,
        "default_parameters": defaults,
    }


def _operator_record(name: str) -> tuple[str, object]:
    stem = name.removesuffix("_spilled").removesuffix("_energy")
    if name == "trace_length":
        return "length", extract_all_features
    if stem == "epr":
        return "mean_level", (
            compute_spilled_energy_features
            if name.endswith(("_spilled", "_energy"))
            else extract_all_features
        )
    if stem == "min":
        return "minimum", (
            energy_features_from_logsumexp
            if name.endswith("_energy")
            else compute_spilled_energy_features
        )
    if stem == "sw_var_peak":
        return "sliding_variance_peak", (
            compute_time_domain
            if name == "sw_var_peak"
            else compute_spilled_energy_features
        )
    if stem == "cusum_max":
        return "cusum_magnitude", (
            compute_cusum_residuals
            if name == "cusum_max"
            else compute_spilled_energy_features
        )
    if name == "cusum_shift_idx":
        return "cusum_location", compute_cusum_residuals
    if name in {"spectral_entropy", "low_band_power", "high_band_power",
                "hl_ratio", "dominant_freq", "spectral_centroid"}:
        return {
            "spectral_entropy": "fft_entropy",
            "low_band_power": "fft_low_band",
            "high_band_power": "fft_high_band",
            "hl_ratio": "fft_band_ratio",
            "dominant_freq": "fft_peak_location",
            "spectral_centroid": "fft_centroid",
        }[name], compute_spectral_features
    if name in {"stft_max_high_power", "stft_spectral_entropy"}:
        return {
            "stft_max_high_power": "stft_high_band_peak",
            "stft_spectral_entropy": "stft_frame_entropy",
        }[name], compute_stft_features
    if name == "rpdi":
        return "tail_to_global_mean_ratio", compute_time_domain
    if name == "pe_mean":
        return "permutation_entropy_mean", compute_permutation_entropy
    if name == "hurst_exponent":
        return "rescaled_range_hurst", compute_hurst_exponent
    topk = {
        "mean_top1_logprob": ("top1_mean", logprob_features),
        "logprob_margin": ("top1_top2_margin", logprob_features),
        "mean_logprob_entropy": ("topk_shannon_entropy", logprob_features),
        "varentropy": ("topk_varentropy", logprob_features_extended),
        "renyi_entropy_2": ("topk_renyi2", logprob_features_extended),
        "topk_tail_mass": ("topk_tail_mass", logprob_features_extended),
    }
    if name in topk:
        return topk[name]
    raise KeyError(f"no computation operator can be derived for {name}")


def derive_feature_dag(feature_names: Sequence[str] | None = None) -> list[dict]:
    """Register label-blind feature metadata without manual feature families.

    Source streams come from extractor-owned registries.  The operator taxonomy
    is the explicit, handwritten mapping in ``_operator_record``; inspected
    function signatures record implementation provenance and defaults but do
    not infer the taxonomy.
    """

    names = canonical_feature_names() if feature_names is None else tuple(feature_names)
    entropy_names = set(FEAT_NAMES[:16])
    sampled_names = set(FEAT_NAMES[16:])
    energy_names = set(ENERGY_FEATS)
    topk_names = set(LOGPROB_FEATS + LOGPROB_FEATS_EXT)
    output = []
    for index, name in enumerate(names):
        if name in entropy_names:
            source_stream = "generated_token_ids" if name == "trace_length" else "token_entropies"
        elif name in sampled_names:
            source_stream = "token_spilled_energies"
        elif name in energy_names:
            source_stream = "token_logsumexp"
        elif name in topk_names:
            source_stream = "top_k_logprobs"
        else:
            raise KeyError(f"feature {name} is not owned by an extractor registry")
        operator, function = _operator_record(name)
        output.append({
            "feature_index": index,
            "feature_name": name,
            "source_stream": source_stream,
            "operator": operator,
            "post_transform": FEATURE_TRANSFORMS.get(name, "raw"),
            "confidence_sign": int(CONFIDENCE_FEATURE_SIGNS_V1[name]),
            "implementation": _function_record(function),
            "derivation": "extractor_registry_plus_explicit_operator_taxonomy",
            "operator_taxonomy_source": (
                "handwritten label-blind _operator_record mapping"
            ),
            "function_signature_role": (
                "records implementation provenance and scalar defaults only"
            ),
            "inherited_input_contract": (
                "mixed-v2 post-transforms and confidence signs were frozen in "
                "earlier label-informed development"
            ),
            "manual_provenance_registry_used": False,
        })
    return output


def _manifest_observation_count(manifest: Mapping) -> int | None:
    cells = manifest.get("cells")
    if not isinstance(cells, list) or not cells:
        return None
    counts = []
    for cell in cells:
        n = cell.get("n_problems")
        if n is None:
            continue
        counts.append(int(n) * int(manifest.get("k", 1)))
    return int(sum(counts)) if counts else None


def audit_source_environments(
    bundle_path: str | Path,
    cache_root: str | Path,
    source_cells: Iterable[str],
) -> dict:
    """Audit the strict source roster without indexing any label sidecar."""

    bundle_path = Path(bundle_path)
    cache_root = Path(cache_root)
    source_cells = tuple(source_cells)
    data = np.load(bundle_path, allow_pickle=True)
    canonical = canonical_feature_names()
    rows = []
    presence = np.zeros((len(source_cells), len(canonical)), dtype=bool)
    for cell_index, cell in enumerate(source_cells):
        required = [f"{cell}__{suffix}" for suffix in ("V", "pool", "hand_signs")]
        missing_keys = [key for key in required if key not in data.files]
        if missing_keys:
            raise RuntimeError(f"bundle is missing label-free keys for {cell}: {missing_keys}")
        stored = np.asarray(data[f"{cell}__V"], dtype=float)
        names = tuple(str(value) for value in data[f"{cell}__pool"])
        legacy = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
        matrix, kept_names, _ = dufs_liu_mixed_v2_from_bundle(stored, names, legacy)
        feature_lookup = {name: index for index, name in enumerate(canonical)}
        for name in kept_names:
            presence[cell_index, feature_lookup[name]] = True
        manifest_path = cache_root / cell / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_observations = _manifest_observation_count(manifest)
        rows.append({
            "cell": cell,
            "model": manifest.get("model"),
            "dataset": manifest.get("dataset"),
            "split": manifest.get("split"),
            "temperature": (manifest.get("temps") or [None])[0],
            "responses_per_problem": manifest.get("k"),
            "bundle_samples": int(matrix.shape[0]),
            "manifest_observations": manifest_observations,
            "bundle_retention_fraction": (
                float(matrix.shape[0] / manifest_observations)
                if manifest_observations else None
            ),
            "active_feature_count": int(matrix.shape[1]),
            "active_features": list(kept_names),
            "missing_features": [name for name in canonical if name not in kept_names],
            "manifest_sha256": sha256_file(manifest_path),
            "labels_accessed": False,
        })
    pair_counts = presence.astype(int).T @ presence.astype(int)
    feature_rows = [
        {
            "feature_name": name,
            "source_cell_count": int(presence[:, index].sum()),
            "source_cell_fraction": float(presence[:, index].mean()),
            "missing_cells": [
                cell for cell, present in zip(source_cells, presence[:, index]) if not present
            ],
        }
        for index, name in enumerate(canonical)
    ]
    pair_rows = [
        {
            "feature_a": canonical[i],
            "feature_b": canonical[j],
            "joint_source_cells": int(pair_counts[i, j]),
        }
        for i in range(len(canonical))
        for j in range(i, len(canonical))
    ]
    return {
        "version": PHASE_VERSION,
        "bundle_sha256": sha256_file(bundle_path),
        "bundle_contains_label_sidecars": any(key.endswith("__labels") for key in data.files),
        "label_sidecars_indexed": False,
        "source_cells": list(source_cells),
        "environment_rows": rows,
        "feature_rows": feature_rows,
        "pair_rows": pair_rows,
        "presence_matrix": presence,
        "minimum_pair_coverage": int(pair_counts.min()),
        "maximum_pair_coverage": int(pair_counts.max()),
    }


def resolve_local_lfs_object(path: str | Path, repo_root: str | Path) -> tuple[Path, dict]:
    """Resolve a local Git-LFS pointer without changing the working tree."""

    path = Path(path)
    repo_root = Path(repo_root)
    prefix = path.read_bytes()[:256]
    if not prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return path, {
            "storage": "working_tree_file",
            "path": str(path),
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    lines = prefix.decode("ascii").splitlines()
    oid = next(line.split("sha256:", 1)[1] for line in lines if line.startswith("oid sha256:"))
    declared_size = int(next(line.split()[1] for line in lines if line.startswith("size ")))
    obj = repo_root / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not obj.exists():
        raise FileNotFoundError(f"local Git-LFS object is unavailable for {path}")
    actual_hash = sha256_file(obj)
    if actual_hash != oid or obj.stat().st_size != declared_size:
        raise RuntimeError(f"Git-LFS object integrity failure for {path}")
    return obj, {
        "storage": "local_git_lfs_object",
        "pointer_path": str(path),
        "object_path": str(obj),
        "size": declared_size,
        "sha256": actual_hash,
    }


def _safe_processbench_rows(path: Path) -> dict[str, dict]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"ProcessBench cache must be a dict: {path}")
    safe = {}
    for key, row in raw.items():
        if not isinstance(row, dict):
            raise TypeError(f"ProcessBench row must be a dict: {path}/{key}")
        row_id = str(row.get("id", key))
        content = json.dumps(
            {
                "problem": row.get("problem"),
                "steps": row.get("steps"),
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        safe[row_id] = {
            "content_sha256": sha256_text(content),
            "telemetry_complete": all(name in row for name in REQUIRED_TELEMETRY),
            "token_count": len(row.get("token_entropies", ())),
        }
    return safe


def audit_processbench_pairing(cache_root: str | Path, repo_root: str | Path) -> dict:
    """Verify exact cross-model ProcessBench pairs without reading targets."""

    cache_root = Path(cache_root)
    repo_root = Path(repo_root)
    model_rows = []
    subset_results = []
    total_exact = 0
    for subset in PROCESSBENCH_SUBSETS:
        by_model = {}
        for model_dir in PROCESSBENCH_MODEL_DIRS:
            pointer = cache_root / model_dir / f"processbench_{subset}.pkl"
            resolved, storage = resolve_local_lfs_object(pointer, repo_root)
            safe = _safe_processbench_rows(resolved)
            by_model[model_dir] = safe
            model_rows.append({
                "subset": subset,
                "model_view": model_dir,
                "row_count": len(safe),
                "complete_telemetry_rows": int(sum(
                    record["telemetry_complete"] for record in safe.values()
                )),
                "storage": storage["storage"],
                "artifact_sha256": storage["sha256"],
                "artifact_size": storage["size"],
                "labels_accessed": False,
            })
        sets = [set(rows) for rows in by_model.values()]
        shared = set.intersection(*sets)
        union = set.union(*sets)
        exact = sum(
            len({by_model[model][row_id]["content_sha256"]
                 for model in PROCESSBENCH_MODEL_DIRS}) == 1
            for row_id in shared
        )
        telemetry_complete = sum(
            all(by_model[model][row_id]["telemetry_complete"]
                for model in PROCESSBENCH_MODEL_DIRS)
            for row_id in shared
        )
        total_exact += exact
        subset_results.append({
            "subset": subset,
            "model_view_count": len(by_model),
            "union_ids": len(union),
            "shared_ids": len(shared),
            "exact_content_matches": int(exact),
            "complete_telemetry_pairs": int(telemetry_complete),
            "pairing_fraction": float(len(shared) / max(len(union), 1)),
            "exact_content_fraction": float(exact / max(len(shared), 1)),
        })
    return {
        "version": PHASE_VERSION,
        "model_views": list(PROCESSBENCH_MODEL_DIRS),
        "subsets": subset_results,
        "model_rows": model_rows,
        "total_exact_pairs": int(total_exact),
        "labels_accessed": False,
        "target_keys_explicitly_excluded": sorted(PROHIBITED_TARGET_KEYS),
    }


def simulate_factorial_world(
    *,
    seed: int = 20260813,
    n_environments: int = 8,
    n_samples: int = 600,
    n_channels: int = 5,
    n_operators: int = 6,
    missing_fraction: float = 0.12,
    environment_specific_target: bool = False,
) -> FactorialWorld:
    """Generate an incomplete channel-by-operator measurement system."""

    if n_environments < 2 or n_samples < 20 or n_channels < 2 or n_operators < 2:
        raise ValueError("the simulator requires multiple environments and crossed factors")
    if not 0 <= missing_fraction < 0.5:
        raise ValueError("missing_fraction must lie in [0, 0.5)")
    rng = np.random.default_rng(int(seed))
    channels = tuple(f"channel_{index}" for index in range(n_channels))
    operators = tuple(f"operator_{index}" for index in range(n_operators))
    names = tuple(f"{channel}__{operator}" for channel in channels for operator in operators)
    channel_index = np.repeat(np.arange(n_channels), n_operators)
    operator_index = np.tile(np.arange(n_operators), n_channels)
    channel_target = rng.normal(0.75, 0.25, n_channels)
    operator_target = rng.normal(0.0, 0.12, n_operators)
    target_loading = channel_target[channel_index] + operator_target[operator_index]
    operator_difficulty = rng.normal(0.8, 0.25, n_operators)
    channel_difficulty = rng.normal(0.0, 0.12, n_channels)
    difficulty_loading = (
        operator_difficulty[operator_index] + channel_difficulty[channel_index]
    )
    nuisance_loading = rng.normal(0.0, 1.0, (2, len(names)))
    environments = []
    missing_count = int(round(missing_fraction * len(names)))
    for environment in range(n_environments):
        target = rng.choice(np.asarray([-1.0, 1.0]), size=n_samples)
        difficulty = rng.normal(size=n_samples)
        nuisance = rng.normal(size=(n_samples, 2))
        local_target = target_loading.copy()
        if environment_specific_target:
            local_target += rng.normal(0.0, 0.35, len(names))
        nuisance_scale = rng.uniform(0.4, 1.8, 2)
        matrix = (
            target[:, None] * local_target[None, :]
            + difficulty[:, None] * difficulty_loading[None, :]
            + (nuisance * nuisance_scale[None, :]) @ nuisance_loading
            + rng.normal(0.0, 0.65, (n_samples, len(names)))
        )
        # One exact duplicate is a deliberate multiplicity stressor.
        matrix[:, -1] = matrix[:, 0]
        available = np.ones(len(names), dtype=bool)
        if missing_count:
            candidates = np.arange(1, len(names) - 1)
            available[rng.choice(candidates, size=missing_count, replace=False)] = False
        environments.append({
            "environment_id": f"environment_{environment}",
            "matrix": matrix[:, available],
            "feature_names": tuple(name for name, keep in zip(names, available) if keep),
            "target": target,
            "difficulty": difficulty,
            "target_loading": local_target,
            "available_mask": available,
        })
    return FactorialWorld(
        environments=tuple(environments),
        feature_names=names,
        channels=channels,
        operators=operators,
        target_loading=target_loading,
        difficulty_loading=difficulty_loading,
        seed=int(seed),
        environment_specific_target=bool(environment_specific_target),
    )


def factorial_world_diagnostics(world: FactorialWorld) -> dict:
    duplicate_errors = []
    missing_fractions = []
    target_correlations = []
    difficulty_correlations = []
    target_loading_cosines = []
    reference = world.target_loading
    for environment in world.environments:
        available = np.asarray(environment["available_mask"], dtype=bool)
        matrix = np.asarray(environment["matrix"], dtype=float)
        names = environment["feature_names"]
        lookup = {name: index for index, name in enumerate(names)}
        if world.feature_names[0] in lookup and world.feature_names[-1] in lookup:
            duplicate_errors.append(float(np.max(np.abs(
                matrix[:, lookup[world.feature_names[0]]]
                - matrix[:, lookup[world.feature_names[-1]]]
            ))))
        missing_fractions.append(float(1.0 - available.mean()))
        target = np.asarray(environment["target"], dtype=float)
        difficulty = np.asarray(environment["difficulty"], dtype=float)
        target_correlations.extend(np.corrcoef(target, matrix, rowvar=False)[0, 1:])
        difficulty_correlations.extend(np.corrcoef(difficulty, matrix, rowvar=False)[0, 1:])
        local = np.asarray(environment["target_loading"], dtype=float)
        target_loading_cosines.append(float(
            reference @ local / (np.linalg.norm(reference) * np.linalg.norm(local))
        ))
    return {
        "version": PHASE_VERSION,
        "seed": world.seed,
        "environment_count": len(world.environments),
        "feature_count": len(world.feature_names),
        "channel_count": len(world.channels),
        "operator_count": len(world.operators),
        "environment_specific_target": world.environment_specific_target,
        "maximum_duplicate_error": max(duplicate_errors, default=None),
        "mean_missing_fraction": float(np.mean(missing_fractions)),
        "median_absolute_target_feature_correlation": float(
            np.median(np.abs(target_correlations))
        ),
        "median_absolute_difficulty_feature_correlation": float(
            np.median(np.abs(difficulty_correlations))
        ),
        "minimum_target_loading_cosine": float(min(target_loading_cosines)),
    }


__all__ = [
    "PHASE_VERSION",
    "PROCESSBENCH_MODEL_DIRS",
    "PROCESSBENCH_SUBSETS",
    "FactorialWorld",
    "audit_processbench_pairing",
    "audit_source_environments",
    "canonical_feature_names",
    "derive_feature_dag",
    "factorial_world_diagnostics",
    "resolve_local_lfs_object",
    "sha256_file",
    "simulate_factorial_world",
]
