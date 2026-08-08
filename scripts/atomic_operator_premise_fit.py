#!/usr/bin/env python3
"""Fit and freeze the label-free atomic-operator premise audit.

Correctness arrays are never read by this program. Evaluation is isolated in
``atomic_operator_premise_report.py`` and can run only after all artifacts have
been hashed.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import json
import os
import platform
import sys
import time
import types

import numpy as np


REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.inscope_cells import GROUP, INSCOPE  # noqa: E402
from spectral_utils.atomic_operator_audit import audit_cell  # noqa: E402
from spectral_utils.specrage_views import fixed_stable_from_bundle  # noqa: E402


VERSION = "atomic-operator-premise-v2-2026-08-07"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_OUT = os.path.join(REPO, "results", "atomic_operator_premise_audit_v2")

GRAPH_KS = (7, 15, 30)
PRIMARY_GRAPH_K = 15
LAMBDAS = (0.3, 1.0, 3.0)
PRIMARY_LAMBDA = 1.0
DUPLICATE_THRESHOLD = 0.95
DUPLICATE_SENSITIVITIES = (0.90, 0.99)
SUBSAMPLES = 40
SAMPLE_FRACTION = 0.80
SAMPLE_CAP = 1500
PERMUTATION_COUNT = 16
CONVERGENCE_CHECKPOINTS = (4, 8, 12, 20, 30, 40)

CONTINUATION_GATES = {
    "all primary cell associations and quartile contrasts are defined": "undefined count = 0",
    "median within-cell Spearman > 0": "> 0",
    "family-bootstrap association lower > 0": "> 0",
    "feature-identity permutation p <= 0.05": "<= 0.05",
    "eight-family association sign-flip p <= 0.05": "<= 0.05",
    "positive top-minus-bottom in at least 6 of 8 families": ">= 6 of 8",
    "partial-association family-bootstrap lower > 0": "> 0",
    "partial Freedman-Lane p <= 0.05": "<= 0.05",
    "eight-family partial sign-flip p <= 0.05": "<= 0.05",
    "median abs(proxy, ridge-distance Spearman) < 0.8": "< 0.8",
    "top-proxy atomic family-bootstrap AUROC lower > 0": "> 0 pp",
    "top-proxy atomic family sign-flip p <= 0.05": "<= 0.05",
    "top-proxy atomic improves at least 14 of 24 cells": ">= 14 of 24",
    "top-proxy atomic worst loss no worse than -2pp": ">= -2 pp",
    "oracle atomic family-bootstrap AUROC lower > 0": "> 0 pp",
}

SOURCE_FILES = (
    "scripts/atomic_operator_premise_fit.py",
    "scripts/atomic_operator_premise_report.py",
    "scripts/test_atomic_operator_premise.py",
    "scripts/inscope_cells.py",
    "spectral_utils/atomic_operator_audit.py",
    "spectral_utils/answer_span.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/feature_contract.py",
    "docs/experiments/FROZEN_ATOMIC_OPERATOR_PREMISE_AUDIT.md",
)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "not-installed"


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def registered_parameters() -> dict:
    return {
        "graph_ks": list(GRAPH_KS),
        "primary_graph_k": PRIMARY_GRAPH_K,
        "lambdas": list(LAMBDAS),
        "primary_lambda": PRIMARY_LAMBDA,
        "duplicate_threshold": DUPLICATE_THRESHOLD,
        "duplicate_sensitivities": list(DUPLICATE_SENSITIVITIES),
        "subsamples": SUBSAMPLES,
        "sample_fraction": SAMPLE_FRACTION,
        "sample_cap": SAMPLE_CAP,
        "permutation_count": PERMUTATION_COUNT,
        "convergence_checkpoints": list(CONVERGENCE_CHECKPOINTS),
    }


def cell_input_hash(matrix, names) -> str:
    matrix = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(matrix.shape).encode("utf-8"))
    digest.update(matrix.tobytes())
    digest.update(json.dumps(list(names), separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def bundle_cells(data) -> set[str]:
    suffixes = ("__V", "__pool", "__hand_signs")
    cells = set()
    for key in data.files:
        for suffix in suffixes:
            if key.endswith(suffix):
                cells.add(key[: -len(suffix)])
                break
    return cells


def validate_bundle(data) -> tuple[str, ...]:
    expected = set(INSCOPE)
    observed = bundle_cells(data)
    if observed != expected:
        raise RuntimeError(
            f"bundle roster differs: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    required = ("V", "pool", "hand_signs")
    missing = [
        f"{cell}__{suffix}" for cell in INSCOPE for suffix in required
        if f"{cell}__{suffix}" not in data.files
    ]
    if missing:
        raise RuntimeError("bundle is missing label-free arrays: " + ", ".join(missing))
    if len(INSCOPE) != 24:
        raise RuntimeError("registered roster must contain exactly 24 cells")
    return tuple(INSCOPE)


def run_definition(bundle: str, fit_bundle: str, cells: tuple[str, ...], scientific_run: bool) -> dict:
    payload = {
        "version": VERSION,
        "scientific_run": bool(scientific_run),
        "bundle": os.path.relpath(bundle, REPO),
        "bundle_sha256": sha256_file(bundle),
        "label_free_fit_bundle": os.path.relpath(fit_bundle, REPO),
        "label_free_fit_bundle_sha256": sha256_file(fit_bundle),
        "cells": list(cells),
        "domains": {cell: GROUP[cell] for cell in cells},
        "feature_contract": "fixed_stable_v1",
        "feature_contract_history": (
            "label-free fitting conditional on a contract historically developed "
            "using these retrospective cells"
        ),
        "atomic_graph_tie_policy": (
            "collapse exact ties to count-weighted unique-value quotient nodes; "
            "invalidate fewer than three unique values"
        ),
        "primary_proxy": (
            "median_crossfit_alignment * sqrt(operator_reproducibility * "
            "rank_change_reproducibility) * bounded_relative_actuation"
        ),
        "parameters": registered_parameters(),
        "continuation_gates": dict(CONTINUATION_GATES),
        "source_sha256": {
            path: sha256_file(os.path.join(REPO, path)) for path in SOURCE_FILES
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dependency_versions": {
            name: dependency_version(name)
            for name in ("scipy", "scikit-learn", "matplotlib")
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def expected_score_keys(m: int) -> set[str]:
    keys = {
        "feature_names", "sample_index", "run_fingerprint",
        "input_cell_sha256", "iu_pcr",
    }
    for lambda_ in LAMBDAS:
        keys.add(f"ridge__lambda_{lambda_:g}")
    for graph_k in GRAPH_KS:
        for lambda_ in LAMBDAS:
            keys.add(f"atomic__k_{graph_k}__lambda_{lambda_:g}")
            keys.add(f"uniform_atomic__k_{graph_k}__lambda_{lambda_:g}")
    return keys


def valid_checkpoint(
    score_path: str,
    diagnostic_path: str,
    cell: str,
    *,
    run_fingerprint: str,
    input_hash: str,
    feature_names,
    parameters: dict,
):
    if not os.path.exists(score_path) or not os.path.exists(diagnostic_path):
        return None
    try:
        with np.load(score_path, allow_pickle=False) as scores:
            names = tuple(str(value) for value in scores["feature_names"])
            if set(scores.files) != expected_score_keys(len(names)):
                return None
            if names != tuple(feature_names):
                return None
            if str(scores["run_fingerprint"].item()) != run_fingerprint:
                return None
            if str(scores["input_cell_sha256"].item()) != input_hash:
                return None
            n = len(scores["sample_index"])
            if not np.array_equal(scores["sample_index"], np.arange(n, dtype=np.int64)):
                return None
            for key in scores.files:
                if key in {
                    "feature_names", "sample_index", "run_fingerprint",
                    "input_cell_sha256",
                }:
                    continue
                value = np.asarray(scores[key])
                expected_shape = (len(names), n) if key.startswith("atomic__") else (n,)
                if value.shape != expected_shape or not np.isfinite(value).all():
                    return None
        with open(diagnostic_path, encoding="utf-8") as handle:
            diagnostic = json.load(handle)
        if (
            diagnostic.get("cell") != cell
            or diagnostic.get("run_fingerprint") != run_fingerprint
            or diagnostic.get("input_cell_sha256") != input_hash
            or tuple(diagnostic.get("feature_names", ())) != names
            or diagnostic.get("parameters") != parameters
            or len(diagnostic.get("feature_records", [])) != len(names)
        ):
            return None
    except Exception:
        return None
    return {
        "cell": cell,
        "score_file": os.path.relpath(score_path, os.path.dirname(os.path.dirname(score_path))),
        "score_sha256": sha256_file(score_path),
        "diagnostic_file": os.path.relpath(
            diagnostic_path, os.path.dirname(os.path.dirname(diagnostic_path))
        ),
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "n_samples": n,
        "n_features": len(names),
        "input_cell_sha256": input_hash,
    }


def ensure_label_free_bundle(source_bundle: str, out_dir: str) -> str:
    """Materialize and verify a fit input containing no correctness arrays."""
    path = os.path.join(out_dir, "LABEL_FREE_FIT_INPUT.npz")
    with np.load(source_bundle, allow_pickle=True) as source:
        validate_bundle(source)
        arrays = {
            f"{cell}__{suffix}": np.asarray(source[f"{cell}__{suffix}"])
            for cell in INSCOPE for suffix in ("V", "pool", "hand_signs")
        }
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as existing:
            if set(existing.files) != set(arrays):
                raise RuntimeError("existing label-free fit bundle has wrong keys")
            for key, value in arrays.items():
                if not np.array_equal(existing[key], value):
                    raise RuntimeError(f"existing label-free fit bundle differs: {key}")
    else:
        temporary = path + ".tmp.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    with np.load(path, allow_pickle=True) as stripped:
        validate_bundle(stripped)
        forbidden = [key for key in stripped.files if "label" in key.lower() or "target" in key.lower()]
        if forbidden:
            raise RuntimeError("label-free fit bundle contains forbidden keys: " + ", ".join(forbidden))
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug-cell", choices=INSCOPE)
    args = parser.parse_args()

    bundle = os.path.abspath(args.bundle)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "diagnostics"), exist_ok=True)
    fit_bundle = ensure_label_free_bundle(bundle, out_dir)
    data = np.load(fit_bundle, allow_pickle=True)
    all_cells = validate_bundle(data)
    cells = (args.debug_cell,) if args.debug_cell else all_cells
    scientific_run = args.debug_cell is None
    definition = run_definition(bundle, fit_bundle, cells, scientific_run)
    definition_path = os.path.join(out_dir, "RUN_DEFINITION.json")
    if os.path.exists(definition_path):
        with open(definition_path, encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != definition:
            raise RuntimeError("run definition changed; use a new output directory")
    else:
        write_json(definition_path, definition)

    started = time.time()
    manifest = []
    for position, cell in enumerate(cells, start=1):
        score_path = os.path.join(out_dir, "scores", f"{cell}.npz")
        diagnostic_path = os.path.join(out_dir, "diagnostics", f"{cell}.json")
        stored = np.asarray(data[f"{cell}__V"], dtype=float)
        names = tuple(str(name) for name in data[f"{cell}__pool"])
        legacy = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
        matrix, stable_names = fixed_stable_from_bundle(stored, names, legacy)
        input_hash = cell_input_hash(matrix, stable_names)
        checkpoint = valid_checkpoint(
            score_path,
            diagnostic_path,
            cell,
            run_fingerprint=definition["run_fingerprint"],
            input_hash=input_hash,
            feature_names=stable_names,
            parameters=registered_parameters(),
        ) if args.resume else None
        if checkpoint is not None:
            print(f"[{position}/{len(cells)}] resume {cell}", flush=True)
            manifest.append(checkpoint)
            continue
        print(f"[{position}/{len(cells)}] fit {cell}", flush=True)
        stamp = time.time()
        scores, diagnostics = audit_cell(
            matrix.T,
            stable_names,
            cell=cell,
            graph_ks=GRAPH_KS,
            primary_graph_k=PRIMARY_GRAPH_K,
            lambdas=LAMBDAS,
            primary_lambda=PRIMARY_LAMBDA,
            duplicate_threshold=DUPLICATE_THRESHOLD,
            duplicate_sensitivities=DUPLICATE_SENSITIVITIES,
            subsamples=SUBSAMPLES,
            sample_fraction=SAMPLE_FRACTION,
            sample_cap=SAMPLE_CAP,
            permutation_count=PERMUTATION_COUNT,
            convergence_checkpoints=CONVERGENCE_CHECKPOINTS,
        )
        scores["run_fingerprint"] = np.asarray(definition["run_fingerprint"])
        scores["input_cell_sha256"] = np.asarray(input_hash)
        diagnostics["domain"] = GROUP[cell]
        diagnostics["run_fingerprint"] = definition["run_fingerprint"]
        diagnostics["input_cell_sha256"] = input_hash
        diagnostics["feature_names"] = list(stable_names)
        diagnostics["runtime_seconds"] = float(time.time() - stamp)
        np.savez_compressed(score_path, **scores)
        write_json(diagnostic_path, diagnostics)
        checkpoint = valid_checkpoint(
            score_path,
            diagnostic_path,
            cell,
            run_fingerprint=definition["run_fingerprint"],
            input_hash=input_hash,
            feature_names=stable_names,
            parameters=registered_parameters(),
        )
        if checkpoint is None:
            raise RuntimeError(f"written checkpoint did not validate: {cell}")
        manifest.append(checkpoint)
        print(
            f"[{position}/{len(cells)}] done {cell} "
            f"({diagnostics['runtime_seconds']:.1f}s)",
            flush=True,
        )

    complete = {
        "version": VERSION,
        "scientific_run": scientific_run,
        "run_fingerprint": definition["run_fingerprint"],
        "n_cells": len(cells),
        "runtime_seconds": float(time.time() - started),
        "artifact_manifest": manifest,
    }
    write_json(os.path.join(out_dir, "FIT_COMPLETE.json"), complete)
    print(
        f"FIT COMPLETE: {len(cells)} cells in {complete['runtime_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
