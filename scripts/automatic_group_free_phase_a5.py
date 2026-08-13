#!/usr/bin/env python3
"""Freeze and execute the bounded A5 sealed nuisance-first synthetic gate.

``prepare`` records the immutable source/protocol/test/configuration boundary.
``run-nuisance`` verifies that boundary, executes exactly the 100 registered
world-8 sealed repetitions, and writes an append-only closure/pass artifact.
It does not load retrospective labels or the multi-gigabyte real-data caches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import sklearn


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.a5_simulation import (  # noqa: E402
    ALPHAS,
    NUISANCE_WORLD_INDEX,
    PENALTIES,
    SEALED_REPETITIONS,
    run_synthetic_repetition,
    sealed_world_seed,
    simulate_synthetic_world,
    duplicate_pair_diagnostics,
)


VERSION = "automatic-group-free-iu-a5-v1-2026-08-14"
DEFAULT_OUT = REPO / "results" / "automatic_group_free_phase_a5_v1"
PROTOCOL = REPO / "docs" / "experiments" / "AUTOMATIC_GROUP_FREE_IU_PHASE_A5_V1.md"
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_NAMESPACE = 529_000
STATIC_SOURCE_FILES = (
    "docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A5_V1.md",
    "scripts/automatic_group_free_phase_a5.py",
    "scripts/test_anchored_sparse_latent_mixture.py",
    "scripts/test_a5_target_free_data.py",
    "scripts/test_a5_simulation.py",
    "scripts/test_automatic_group_free_phase_a5.py",
)


def source_files() -> tuple[str, ...]:
    # Importing any spectral_utils submodule executes the package initializer.
    # Hash the entire local package (and detect additions/removals), not merely
    # a hand-maintained approximation to its eager transitive import closure.
    package = tuple(
        str(path.relative_to(REPO))
        for path in sorted((REPO / "spectral_utils").glob("*.py"))
    )
    return STATIC_SOURCE_FILES + package


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _exclusive_json(path: Path, payload) -> None:
    """Write one immutable JSON artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise RuntimeError(f"refusing to overwrite sealed artifact: {path.name}")
    data = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def _source_hashes() -> dict[str, str]:
    return {name: sha256_file(REPO / name) for name in source_files()}


def _seed_hash() -> str:
    payload = "\n".join(
        str(sealed_world_seed(NUISANCE_WORLD_INDEX, repetition))
        for repetition in range(SEALED_REPETITIONS)
    ) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prepare(out: str | Path) -> dict:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("A5 prepare requires a new or empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    boundary = {
        "version": VERSION,
        "status": "FROZEN_BEFORE_ANY_SEALED_A5_RESULT",
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_sha256": _source_hashes(),
        "runtime_versions": _runtime_versions(),
        "configuration": {
            "penalties": list(PENALTIES),
            "alphas": list(ALPHAS),
            "selection_rule": "paired_one_standard_error_then_smallest_alpha_larger_penalty",
            "first_sealed_world": NUISANCE_WORLD_INDEX,
            "repetitions": SEALED_REPETITIONS,
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_namespace": BOOTSTRAP_NAMESPACE,
            "sealed_seed_sha256": _seed_hash(),
            "real_cache_accessed": False,
            "retrospective_labels_accessed": False,
        },
    }
    _exclusive_json(out / "A5_BOUNDARY.json", boundary)
    report = out / "BOUNDARY_REPORT.md"
    with report.open("x", encoding="utf-8") as handle:
        handle.write(
            "# A5 frozen boundary\n\n"
            "No sealed synthetic, real-cache, or retrospective-label result was "
            "opened while creating this boundary. World 8 is the registered first "
            "sealed hard stop.\n"
        )
    return boundary


def load_and_verify_boundary(out: str | Path) -> dict:
    out = Path(out)
    with (out / "A5_BOUNDARY.json").open(encoding="utf-8") as handle:
        boundary = json.load(handle)
    if boundary.get("version") != VERSION:
        raise RuntimeError("A5 boundary version mismatch")
    if boundary.get("status") != "FROZEN_BEFORE_ANY_SEALED_A5_RESULT":
        raise RuntimeError("A5 boundary status mismatch")
    if boundary.get("protocol_sha256") != sha256_file(PROTOCOL):
        raise RuntimeError("A5 protocol changed after freeze")
    if boundary.get("source_sha256") != _source_hashes():
        raise RuntimeError("A5 source changed after freeze")
    if boundary.get("runtime_versions") != _runtime_versions():
        raise RuntimeError("A5 numerical runtime changed after freeze")
    configuration = boundary.get("configuration", {})
    expected = {
        "penalties": list(PENALTIES), "alphas": list(ALPHAS),
        "selection_rule": "paired_one_standard_error_then_smallest_alpha_larger_penalty",
        "first_sealed_world": NUISANCE_WORLD_INDEX,
        "repetitions": SEALED_REPETITIONS,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_namespace": BOOTSTRAP_NAMESPACE,
        "sealed_seed_sha256": _seed_hash(),
        "real_cache_accessed": False,
        "retrospective_labels_accessed": False,
    }
    if configuration != expected:
        raise RuntimeError("A5 configuration changed after freeze")
    return boundary


def _bootstrap_lower(values: np.ndarray, gate_name: str) -> float:
    digest = hashlib.sha256(gate_name.encode("utf-8")).digest()
    seed = BOOTSTRAP_NAMESPACE + int.from_bytes(digest[:4], "big")
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for index in range(BOOTSTRAP_DRAWS):
        draws[index] = np.mean(values[rng.integers(0, len(values), len(values))])
    return float(np.quantile(draws, 0.025))


def _scheduled_record(world: int, repetition: int, **extra) -> dict:
    return {
        "world": int(world), "repetition": int(repetition),
        "seed": sealed_world_seed(world, repetition), **extra,
    }


def _run_registered_repetition(world: int, repetition: int, *, deletion_count=0,
                               duplicate_variant=None) -> dict:
    seed = sealed_world_seed(world, repetition)
    try:
        result = run_synthetic_repetition(
            simulate_synthetic_world(world, seed, duplicate_variant=duplicate_variant),
            deletion_count=deletion_count,
        )
        return _scheduled_record(
            world, repetition, usable=True, duplicate_variant=duplicate_variant,
            deletion_count=int(deletion_count), result=result,
        )
    except Exception as error:
        return _scheduled_record(
            world, repetition, usable=False, duplicate_variant=duplicate_variant,
            deletion_count=int(deletion_count), failure_type=type(error).__name__,
            failure=str(error),
        )


def _gate_harm(records: list[dict]) -> dict:
    values = np.asarray([
        record["result"]["candidate_minus_iu"] for record in records
    ], dtype=float)
    return {
        "mean": float(np.mean(values)), "fifth_percentile": float(np.quantile(values, .05)),
        "pass": bool(np.mean(values) >= -.005 and np.quantile(values, .05) >= -.02),
    }


def remaining_schedule() -> tuple[tuple[int, int, str | None, int], ...]:
    output = []
    for world in (1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
        variants = ("exact", "near") if world == 6 else (None,)
        deletions = (0, 1, 2, 3) if world in (3, 4) else (0,)
        for variant in variants:
            for deletion in deletions:
                output.extend(
                    (world, repetition, variant, deletion)
                    for repetition in range(SEALED_REPETITIONS)
                )
    return tuple(output)


def summarize_remaining(records: list[dict], duplicates: list[dict]) -> dict:
    expected = list(remaining_schedule())
    observed = [(
        value["world"], value["repetition"], value["duplicate_variant"],
        value["deletion_count"], value.get("seed")
    ) for value in records]
    expected_with_seeds = [
        (*value, sealed_world_seed(value[0], value[1])) for value in expected
    ]
    if observed != expected_with_seeds:
        raise ValueError("remaining summary requires the exact ordered sealed schedule")
    duplicate_expected = [
        (variant, repetition, sealed_world_seed(6, repetition))
        for variant in ("exact", "near")
        for repetition in range(SEALED_REPETITIONS)
    ]
    if [(x["variant"], x["repetition"], x.get("seed")) for x in duplicates] != duplicate_expected:
        raise ValueError("duplicate diagnostics require exact paired seed schedule")
    failures = [value for value in records if not value["usable"]]
    if failures:
        return {
            "scheduled_records": len(records), "failure_count": len(failures),
            "gate_pass": False, "verdict": "CLOSE_NUMERICAL_NONCONVERGENCE",
        }

    def subset(world, *, variant=None, deletion=0):
        return [value for value in records if value["world"] == world
                and value["duplicate_variant"] == variant
                and value["deletion_count"] == deletion]

    favorable = subset(1)
    final = np.asarray([x["result"]["final_cos2_target"] for x in favorable])
    correction = np.asarray([x["result"]["correction_cos2_target"] for x in favorable])
    support = np.asarray([x["result"]["support_f1"] for x in favorable])
    candidate = np.asarray([x["result"]["candidate_minus_iu"] for x in favorable])
    oracle = np.asarray([x["result"]["oracle_minus_iu"] for x in favorable])
    favorable_gate = {
        "final_median": float(np.median(final)), "final_p05": float(np.quantile(final, .05)),
        "correction_median": float(np.median(correction)),
        "correction_p05": float(np.quantile(correction, .05)),
        "support_f1_mean": float(np.mean(support)), "oracle_minus_iu_mean": float(np.mean(oracle)),
        "captured_gap_ratio": float(np.mean(candidate) / max(np.mean(oracle), 1e-12)),
        "candidate_minus_iu_lower": _bootstrap_lower(candidate, "a5-favorable-candidate-minus-iu"),
    }
    favorable_gate["pass"] = bool(
        favorable_gate["final_median"] >= .8 and favorable_gate["final_p05"] >= .5
        and favorable_gate["correction_median"] >= .6
        and favorable_gate["correction_p05"] >= .25
        and favorable_gate["support_f1_mean"] >= .6
        and favorable_gate["oracle_minus_iu_mean"] >= .01
        and favorable_gate["captured_gap_ratio"] >= .5
        and favorable_gate["candidate_minus_iu_lower"] > 0
    )
    identity = {}
    for world in (2, 10):
        values = subset(world)
        alpha_zero = sum(x["result"]["selected_alpha"] == 0 for x in values)
        errors = [x["result"]["fallback_error"] for x in values
                  if x["result"]["fallback_error"] is not None]
        identity[str(world)] = {
            "alpha_zero_count": alpha_zero,
            "maximum_fallback_error": max(errors, default=0.0),
            "pass": bool(alpha_zero >= 90 and max(errors, default=0.0) < 1e-10),
        }
    harm = {}
    for world in (3, 4):
        for deletion in (0, 1, 2, 3):
            harm[f"world_{world}_deletion_{deletion}"] = _gate_harm(
                subset(world, deletion=deletion)
            )
    for world in (5, 7, 9):
        harm[f"world_{world}"] = _gate_harm(subset(world))
    for variant in ("exact", "near"):
        harm[f"world_6_{variant}"] = _gate_harm(subset(6, variant=variant))

    duplicate_failures = [x for x in duplicates if not x["usable"]]
    duplicate_gate = {}
    for variant in ("exact", "near"):
        values = [x["result"] for x in duplicates if x["variant"] == variant and x["usable"]]
        if duplicate_failures or len(values) != SEALED_REPETITIONS:
            duplicate_gate[variant] = {"pass": False, "failure_count": len(duplicate_failures)}
            continue
        masses = [x["median_combined_mass_ratio"] for x in values]
        ranks = [x["median_score_spearman"] for x in values]
        alpha_differences = [x["selected_alpha_absolute_difference"] for x in values]
        rank_threshold = .999999 if variant == "exact" else .995
        duplicate_gate[variant] = {
            "median_correction_mass_ratio": float(np.median(masses)),
            "median_selected_score_spearman": float(np.median(ranks)),
            "maximum_selected_alpha_difference": float(np.max(alpha_differences)),
            "pass": bool(
                np.median(masses) <= 1.10 and np.median(ranks) >= rank_threshold
                and (variant == "exact" or np.max(alpha_differences) <= .125)
            ),
        }
    all_pass = bool(
        favorable_gate["pass"] and all(x["pass"] for x in identity.values())
        and all(x["pass"] for x in harm.values())
        and all(x["pass"] for x in duplicate_gate.values())
    )
    verdict = "PASS_ALL_SYNTHETIC_GATES" if all_pass else (
        "CLOSE_SYNTHETIC_NO_HEADROOM"
        if favorable_gate["oracle_minus_iu_mean"] < .01
        else "CLOSE_SYNTHETIC_MISSPECIFICATION"
    )
    return {
        "scheduled_records": len(records), "failure_count": 0,
        "favorable": favorable_gate, "identity": identity, "harm": harm,
        "duplicates": duplicate_gate, "gate_pass": all_pass,
        "verdict": verdict,
    }


def summarize_nuisance(records: list[dict]) -> dict:
    expected_seeds = [
        sealed_world_seed(NUISANCE_WORLD_INDEX, repetition)
        for repetition in range(SEALED_REPETITIONS)
    ]
    observed_seeds = [value.get("seed") for value in records]
    if len(records) != SEALED_REPETITIONS or observed_seeds != expected_seeds:
        raise ValueError("nuisance summary requires the exact ordered 100-seed schedule")
    failures = [value for value in records if not value.get("usable", False)]
    if failures:
        implementation_failures = [
            value for value in failures
            if value.get("failure_class") == "implementation_invalid"
        ]
        return {
            "world": NUISANCE_WORLD_INDEX,
            "repetitions": len(records),
            "usable_repetitions": len(records) - len(failures),
            "failure_count": len(failures),
            "implementation_failure_count": len(implementation_failures),
            "gate_pass": False,
            "verdict": ("INVALID_IMPLEMENTATION" if implementation_failures
                        else "CLOSE_NUMERICAL_NONCONVERGENCE"),
            "real_cache_accessed": False,
            "retrospective_labels_accessed": False,
        }
    final_passes = sum(bool(value["target_preferred_final"]) for value in records)
    correction_passes = sum(
        bool(value["target_preferred_correction"]) for value in records
    )
    deltas = np.asarray([value["candidate_minus_iu"] for value in records], dtype=float)
    lower = _bootstrap_lower(deltas, "a5-nuisance-candidate-minus-iu")
    pass_gate = bool(final_passes >= 90 and correction_passes >= 90 and lower >= 0.0)
    return {
        "world": NUISANCE_WORLD_INDEX,
        "repetitions": len(records),
        "usable_repetitions": len(records),
        "failure_count": 0,
        "target_preferred_final_count": final_passes,
        "target_preferred_correction_count": correction_passes,
        "candidate_minus_iu_mean": float(np.mean(deltas)),
        "candidate_minus_iu_bootstrap_95_lower": lower,
        "gate_pass": pass_gate,
        "verdict": ("PASS_NUISANCE_ANTI_REPACKAGING_GATE" if pass_gate
                    else "CLOSE_NUISANCE_REPACKAGING"),
        "real_cache_accessed": False,
        "retrospective_labels_accessed": False,
    }


def run_nuisance(out: str | Path) -> dict:
    out = Path(out)
    load_and_verify_boundary(out)
    result_path = out / "A5_NUISANCE_COMPLETE.json"
    records_path = out / "nuisance_repetitions.json"
    if result_path.exists():
        raise RuntimeError("sealed nuisance completion already exists")
    boundary_hash = sha256_file(out / "A5_BOUNDARY.json")
    checkpoint = out / "nuisance_checkpoints"
    checkpoint.mkdir(exist_ok=True)
    records = []
    for repetition in range(SEALED_REPETITIONS):
        seed = sealed_world_seed(NUISANCE_WORLD_INDEX, repetition)
        checkpoint_path = checkpoint / f"{repetition:03d}.json"
        if checkpoint_path.exists():
            record = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if (record.get("world") != NUISANCE_WORLD_INDEX
                    or record.get("seed") != seed
                    or record.get("boundary_sha256") != boundary_hash):
                raise RuntimeError("sealed nuisance checkpoint provenance mismatch")
        else:
            try:
                record = run_synthetic_repetition(
                    simulate_synthetic_world(NUISANCE_WORLD_INDEX, seed)
                )
                record["usable"] = True
            except Exception as error:
                registered = (
                    isinstance(error, RuntimeError)
                    and str(error).startswith("CLOSE_")
                )
                record = {
                    "world": NUISANCE_WORLD_INDEX, "seed": seed, "usable": False,
                    "failure_class": ("registered_numerical_close" if registered
                                      else "implementation_invalid"),
                    "failure_type": type(error).__name__, "failure": str(error),
                }
            record["boundary_sha256"] = boundary_hash
            _exclusive_json(checkpoint_path, record)
        records.append(record)
    summary = summarize_nuisance(records)
    if records_path.exists():
        persisted = json.loads(records_path.read_text(encoding="utf-8"))
        if persisted != records:
            raise RuntimeError("sealed nuisance aggregate disagrees with checkpoints")
    else:
        _exclusive_json(records_path, records)
    summary["boundary_sha256"] = boundary_hash
    summary["repetitions_sha256"] = sha256_file(records_path)
    _exclusive_json(result_path, summary)
    return summary


def verify_nuisance_artifacts(out: str | Path) -> dict:
    out = Path(out)
    load_and_verify_boundary(out)
    records_path = out / "nuisance_repetitions.json"
    result_path = out / "A5_NUISANCE_COMPLETE.json"
    records = json.loads(records_path.read_text(encoding="utf-8"))
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    expected = summarize_nuisance(records)
    expected.update({
        "boundary_sha256": sha256_file(out / "A5_BOUNDARY.json"),
        "repetitions_sha256": sha256_file(records_path),
    })
    if stored != expected:
        raise RuntimeError("nuisance completion does not reproduce from frozen records")
    return stored


def run_remaining(out: str | Path) -> dict:
    """Scaffold only; S1b needs a separately frozen boundary after S1a PASS."""
    raise RuntimeError("S1B_NOT_FROZEN: prepare an independently reviewed S1b boundary")
    # The unreachable implementation below is deliberately source-frozen as a
    # review scaffold, not authorized by the S1a boundary.
    # pragma: no cover
    out = Path(out)
    load_and_verify_boundary(out)
    nuisance_path = out / "A5_NUISANCE_COMPLETE.json"
    if not nuisance_path.exists():
        raise RuntimeError("nuisance hard-stop artifact is missing")
    nuisance = json.loads(nuisance_path.read_text(encoding="utf-8"))
    if not nuisance.get("gate_pass"):
        raise RuntimeError("A5 closed at the nuisance hard stop")
    if (nuisance.get("boundary_sha256") != sha256_file(out / "A5_BOUNDARY.json")
            or nuisance.get("repetitions_sha256")
            != sha256_file(out / "nuisance_repetitions.json")):
        raise RuntimeError("nuisance artifact provenance mismatch")
    records_path = out / "remaining_repetitions.json"
    duplicates_path = out / "duplicate_diagnostics.json"
    result_path = out / "A5_SYNTHETIC_COMPLETE.json"
    for path in (records_path, duplicates_path, result_path):
        if path.exists() or path.with_name(path.name + ".tmp").exists():
            raise RuntimeError("sealed continuation output is not empty")
    records = []
    for world, repetition, variant, deletion in remaining_schedule():
        records.append(_run_registered_repetition(
            world, repetition, deletion_count=deletion, duplicate_variant=variant,
        ))
    duplicate_records = []
    for variant in ("exact", "near"):
        for repetition in range(SEALED_REPETITIONS):
            seed = sealed_world_seed(6, repetition)
            try:
                result = duplicate_pair_diagnostics(
                    simulate_synthetic_world(1, seed),
                    simulate_synthetic_world(6, seed, duplicate_variant=variant),
                )
                duplicate_records.append({
                    "variant": variant, "repetition": repetition, "seed": seed,
                    "usable": True, "result": result,
                })
            except Exception as error:
                duplicate_records.append({
                    "variant": variant, "repetition": repetition, "seed": seed,
                    "usable": False, "failure_type": type(error).__name__,
                    "failure": str(error),
                })
    summary = summarize_remaining(records, duplicate_records)
    _exclusive_json(records_path, records)
    _exclusive_json(duplicates_path, duplicate_records)
    summary.update({
        "boundary_sha256": sha256_file(out / "A5_BOUNDARY.json"),
        "nuisance_complete_sha256": sha256_file(nuisance_path),
        "remaining_repetitions_sha256": sha256_file(records_path),
        "duplicate_diagnostics_sha256": sha256_file(duplicates_path),
        "real_cache_accessed": False, "retrospective_labels_accessed": False,
    })
    _exclusive_json(result_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=(
            "prepare", "verify", "verify-nuisance", "run-nuisance", "run-remaining"
        )
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare(args.out)
    elif args.command == "verify":
        payload = load_and_verify_boundary(args.out)
    elif args.command == "run-nuisance":
        payload = run_nuisance(args.out)
    elif args.command == "verify-nuisance":
        payload = verify_nuisance_artifacts(args.out)
    else:
        payload = run_remaining(args.out)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
