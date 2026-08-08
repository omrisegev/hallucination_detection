#!/usr/bin/env python3
"""Fit and freeze label-free repeated cross-view diffusion scores."""

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
from spectral_utils.repeated_cross_view_diffusion import (  # noqa: E402
    fit_repeated_cross_view_paths,
)
from spectral_utils.specrage_views import fixed_stable_from_bundle  # noqa: E402


VERSION = "repeated-cross-view-ad-iu-pcr-v1-2026-08-07"
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_REFERENCE = os.path.join(REPO, "results", "frozen_24cell_benchmark", "scores")
DEFAULT_OUT = os.path.join(REPO, "results", "repeated_cross_view_diffusion_v1")

PARTITION_COUNT = 16
MIN_FRACTION = 0.30
DEPENDENCY_THRESHOLD = 0.15
PRIMARY_K = 7
SENSITIVITY_KS = (5, 11)
LAMBDAS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
PRIMARY_LAMBDA = 0.1
PREFIX_COUNTS = (4, 8, 16)

SOURCE_FILES = (
    "scripts/repeated_cross_view_fit.py",
    "scripts/repeated_cross_view_report.py",
    "scripts/test_repeated_cross_view_diffusion.py",
    "scripts/inscope_cells.py",
    "spectral_utils/repeated_cross_view_diffusion.py",
    "spectral_utils/cross_view_graph.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/upcr.py",
    "spectral_utils/specrage_views.py",
    "spectral_utils/feature_contract.py",
    "docs/experiments/FROZEN_REPEATED_CROSS_VIEW_DIFFUSION.md",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dependency_version(name):
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "not-installed"


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def parameters():
    return {
        "partition_count": PARTITION_COUNT,
        "min_fraction": MIN_FRACTION,
        "dependency_threshold": DEPENDENCY_THRESHOLD,
        "primary_k": PRIMARY_K,
        "sensitivity_ks": list(SENSITIVITY_KS),
        "lambdas": list(LAMBDAS),
        "primary_lambda": PRIMARY_LAMBDA,
        "prefix_counts": list(PREFIX_COUNTS),
    }


def cell_input_hash(matrix, names):
    values = np.ascontiguousarray(np.asarray(matrix, dtype=np.float64))
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(values.tobytes())
    digest.update(json.dumps(list(names), separators=(",", ":")).encode("utf-8"))
    return digest.hexdigest()


def validate_bundle(data):
    observed = {key[:-3] for key in data.files if key.endswith("__V")}
    if observed != set(INSCOPE):
        raise RuntimeError(
            f"bundle roster mismatch: missing={sorted(set(INSCOPE)-observed)}, "
            f"extra={sorted(observed-set(INSCOPE))}"
        )
    for cell in INSCOPE:
        for suffix in ("V", "pool", "hand_signs"):
            if f"{cell}__{suffix}" not in data.files:
                raise RuntimeError(f"missing label-free input {cell}__{suffix}")
    return tuple(INSCOPE)


def ensure_label_free_bundle(bundle, out_dir):
    path = os.path.join(out_dir, "LABEL_FREE_FIT_INPUT.npz")
    if not os.path.exists(path):
        with np.load(bundle, allow_pickle=True) as source:
            validate_bundle(source)
            arrays = {
                f"{cell}__{suffix}": source[f"{cell}__{suffix}"]
                for cell in INSCOPE for suffix in ("V", "pool", "hand_signs")
            }
        temporary = path + ".tmp.npz"
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    with np.load(path, allow_pickle=True) as stripped:
        validate_bundle(stripped)
        forbidden = [
            key for key in stripped.files
            if "label" in key.lower() or "target" in key.lower()
        ]
        if forbidden or len(stripped.files) != 72:
            raise RuntimeError("invalid physically stripped fit bundle")
    return path


def run_definition(bundle, fit_bundle, reference_dir, cells, scientific_run):
    payload = {
        "version": VERSION,
        "scientific_run": bool(scientific_run),
        "bundle": os.path.relpath(bundle, REPO),
        "bundle_sha256": sha256_file(bundle),
        "label_free_fit_bundle": os.path.relpath(fit_bundle, REPO),
        "label_free_fit_bundle_sha256": sha256_file(fit_bundle),
        "reference_score_dir": os.path.relpath(reference_dir, REPO),
        "reference_score_sha256": {
            cell: sha256_file(os.path.join(reference_dir, f"{cell}.npz"))
            for cell in cells
        },
        "cells": list(cells),
        "domains": {cell: GROUP[cell] for cell in cells},
        "feature_contract": "fixed_stable_v1",
        "parameters": parameters(),
        "source_sha256": {
            relative: sha256_file(os.path.join(REPO, relative))
            for relative in SOURCE_FILES
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dependencies": {
            name: dependency_version(name)
            for name in ("scipy", "scikit-learn", "matplotlib")
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def validate_checkpoint(path, diagnostic_path, *, fingerprint, input_hash, names):
    if not os.path.exists(path) or not os.path.exists(diagnostic_path):
        return None
    try:
        with np.load(path, allow_pickle=False) as scores:
            if tuple(map(str, scores["feature_names"])) != tuple(names):
                return None
            if not np.array_equal(scores["sample_index"], np.arange(len(scores["iu_pcr"]))):
                return None
            if str(scores["run_fingerprint"]) != fingerprint:
                return None
            if str(scores["input_cell_sha256"]) != input_hash:
                return None
            for key in scores.files:
                value = scores[key]
                if value.dtype.kind not in "US" and not np.isfinite(value).all():
                    return None
        with open(diagnostic_path, encoding="utf-8") as handle:
            diagnostic = json.load(handle)
        if diagnostic.get("run_fingerprint") != fingerprint:
            return None
        if diagnostic.get("input_cell_sha256") != input_hash:
            return None
        if diagnostic.get("parameters") != parameters():
            return None
    except Exception:
        return None
    root = os.path.dirname(os.path.dirname(path))
    return {
        "cell": diagnostic["cell"],
        "score_file": os.path.relpath(path, root),
        "score_sha256": sha256_file(path),
        "diagnostic_file": os.path.relpath(diagnostic_path, root),
        "diagnostic_sha256": sha256_file(diagnostic_path),
        "input_cell_sha256": input_hash,
        "n_samples": int(diagnostic["n_samples"]),
        "n_features": int(diagnostic["n_features"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--reference-dir", default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug-cell", choices=INSCOPE)
    args = parser.parse_args()

    bundle = os.path.abspath(args.bundle)
    reference_dir = os.path.abspath(args.reference_dir)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "diagnostics"), exist_ok=True)
    fit_bundle = ensure_label_free_bundle(bundle, out_dir)
    data = np.load(fit_bundle, allow_pickle=True)
    all_cells = validate_bundle(data)
    cells = (args.debug_cell,) if args.debug_cell else all_cells
    scientific_run = args.debug_cell is None
    definition = run_definition(
        bundle, fit_bundle, reference_dir, cells, scientific_run
    )
    definition_path = os.path.join(out_dir, "RUN_DEFINITION.json")
    if os.path.exists(definition_path):
        with open(definition_path, encoding="utf-8") as handle:
            if json.load(handle) != definition:
                raise RuntimeError("run definition changed; use a fresh output directory")
    else:
        write_json(definition_path, definition)

    started = time.time()
    manifest = []
    for position, cell in enumerate(cells, 1):
        stored = np.asarray(data[f"{cell}__V"], dtype=float)
        names = tuple(map(str, data[f"{cell}__pool"]))
        signs = np.asarray(data[f"{cell}__hand_signs"], dtype=float)
        matrix, stable_names = fixed_stable_from_bundle(stored, names, signs)
        input_hash = cell_input_hash(matrix, stable_names)
        score_path = os.path.join(out_dir, "scores", f"{cell}.npz")
        diagnostic_path = os.path.join(out_dir, "diagnostics", f"{cell}.json")
        checkpoint = validate_checkpoint(
            score_path, diagnostic_path, fingerprint=definition["run_fingerprint"],
            input_hash=input_hash, names=stable_names,
        ) if args.resume else None
        if checkpoint is not None:
            print(f"[{position}/{len(cells)}] resume {cell}", flush=True)
            manifest.append(checkpoint)
            continue

        print(f"[{position}/{len(cells)}] fit {cell}", flush=True)
        stamp = time.time()
        scores, diagnostics = fit_repeated_cross_view_paths(
            matrix.T,
            stable_names,
            cell=cell,
            partition_count=PARTITION_COUNT,
            min_fraction=MIN_FRACTION,
            dependency_threshold=DEPENDENCY_THRESHOLD,
            primary_k=PRIMARY_K,
            sensitivity_ks=SENSITIVITY_KS,
            lambdas=LAMBDAS,
            primary_lambda=PRIMARY_LAMBDA,
            prefix_counts=PREFIX_COUNTS,
        )
        reference_path = os.path.join(reference_dir, f"{cell}.npz")
        with np.load(reference_path, allow_pickle=False) as reference:
            if tuple(map(str, reference["feature_names"])) != tuple(stable_names):
                raise RuntimeError(f"reference feature contract mismatch: {cell}")
            if not np.array_equal(reference["sample_index"], scores["sample_index"]):
                raise RuntimeError(f"reference sample index mismatch: {cell}")
            if not np.allclose(reference["iu_pcr"], scores["iu_pcr"], atol=1e-9):
                raise RuntimeError(f"IU-PCR reference mismatch: {cell}")
            scores["deployed_upcr"] = np.asarray(
                reference["deployed_upcr"], dtype=np.float64
            )
            scores["dufs_liu"] = np.asarray(
                reference["dufs_liu__lambda_0p1"], dtype=np.float64
            )
            scores["raw_uniform_liu"] = np.asarray(
                reference["raw_uniform_liu__lambda_0p1"], dtype=np.float64
            )
        scores["run_fingerprint"] = np.asarray(definition["run_fingerprint"])
        scores["input_cell_sha256"] = np.asarray(input_hash)
        diagnostics.update({
            "domain": GROUP[cell],
            "run_fingerprint": definition["run_fingerprint"],
            "input_cell_sha256": input_hash,
            "reference_score_sha256": definition["reference_score_sha256"][cell],
            "parameters": parameters(),
            "n_samples": int(matrix.shape[0]),
            "n_features": int(matrix.shape[1]),
            "runtime_seconds": float(time.time() - stamp),
        })
        np.savez_compressed(score_path, **scores)
        write_json(diagnostic_path, diagnostics)
        checkpoint = validate_checkpoint(
            score_path, diagnostic_path, fingerprint=definition["run_fingerprint"],
            input_hash=input_hash, names=stable_names,
        )
        if checkpoint is None:
            raise RuntimeError(f"written checkpoint did not validate: {cell}")
        manifest.append(checkpoint)
        print(
            f"[{position}/{len(cells)}] done {cell} "
            f"({diagnostics['runtime_seconds']:.2f}s)", flush=True
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
    print(f"FIT COMPLETE: {len(cells)} cells in {complete['runtime_seconds']:.1f}s")


if __name__ == "__main__":
    main()
