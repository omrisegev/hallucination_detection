#!/usr/bin/env python3
"""Develop one answer-level gate on the historical 15-cell math panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.inscope_cells import MATH_CELLS
from spectral_utils import gate_feature_readout as features
from spectral_utils import math_gate_selection as selection


OUT = ROOT / "results/math_gate_development_v1"
PROTOCOL = ROOT / "docs/experiments/MATH_GATE_DEVELOPMENT_V1.md"
EXPECTED_ANSWERS = 18_614
EXPECTED_CORRECT = 11_013
EXPECTED_ERROR = 7_601

SOURCES = {
    "ars_gsm8k_r1distill8b": {
        "file": "raw_gsm8k_T0.0.pkl", "family": "gsm8k", "size": 189353848,
        "sha256": "ae33ac6139828c1a69fb8887b950cc956323707d537d10c20bebf1108d8f2dc8",
        "remote": "cluster_results/repgrid/ars_gsm8k_r1distill8b/raw_gsm8k_T0.0.pkl",
    },
    "internalstates_gsm8k_qwen25_7b": {
        "file": "raw_gsm8k_T0.8.pkl", "family": "gsm8k", "size": 146217844,
        "sha256": "7ff68214158c740a88baf3959dff6484b68f43f40e6f6096ca6163fb64b5f82c",
        "remote": "cluster_results/repgrid/internalstates_gsm8k_qwen25_7b/raw_gsm8k_T0.8.pkl",
    },
    "lapeigvals_gsm8k_llama3b": {
        "file": "raw_gsm8k_T1.0.pkl", "family": "gsm8k", "size": 261092570,
        "sha256": "595310c86caf867978e261d563de1af60ca782e8981491fa0fba646524af1270",
        "remote": "cluster_results/repgrid/lapeigvals_gsm8k_llama3b/raw_gsm8k_T1.0.pkl",
    },
    "lapeigvals_gsm8k_llama8b": {
        "file": "raw_gsm8k_T1.0.pkl", "family": "gsm8k", "size": 111563114,
        "sha256": "6ec52c7af5306a48464ab58f96d5b3f31029a064ccfc2943c049275b9383aa88",
        "remote": "cluster_results/repgrid/lapeigvals_gsm8k_llama8b/raw_gsm8k_T1.0.pkl",
    },
    "lapeigvals_gsm8k_mistral24b": {
        "file": "raw_gsm8k_T1.0.pkl", "family": "gsm8k", "size": 285037844,
        "sha256": "881dfabbbe48a2af4d756483c6800f6d1674d04b13a1664b9765f52e7a36f23c",
        "remote": "cluster_results/repgrid/lapeigvals_gsm8k_mistral24b/raw_gsm8k_T1.0.pkl",
    },
    "lapeigvals_gsm8k_nemo": {
        "file": "raw_gsm8k_T1.0.pkl", "family": "gsm8k", "size": 302055723,
        "sha256": "4b772b6a47d5b070511b863aebbec7fdbf25ab5b58a2f4d13de18805c133eff0",
        "remote": "cluster_results/repgrid/lapeigvals_gsm8k_nemo/raw_gsm8k_T1.0.pkl",
    },
    "lapeigvals_gsm8k_phi35": {
        "file": "raw_gsm8k_T1.0.pkl", "family": "gsm8k", "size": 344193497,
        "sha256": "0cb4b31f13fb59f34f786db1d931f1d99daee71a0cfa10402e523a859477dde8",
        "remote": "cluster_results/repgrid/lapeigvals_gsm8k_phi35/raw_gsm8k_T1.0.pkl",
    },
    "noise_gsm8k_mistral7b": {
        "file": "raw_gsm8k_T1.0.pkl", "family": "gsm8k", "size": 373510793,
        "sha256": "9c80391981959c8196f0816db0caca818e4828a5cfbcd31bafb4e9271af93ccf",
        "remote": "cluster_results/repgrid/noise_gsm8k_mistral7b/raw_gsm8k_T1.0.pkl",
    },
    "noise_gsm8k_phi3mini": {
        "file": "raw_gsm8k_T1.0.pkl", "family": "gsm8k", "size": 297124222,
        "sha256": "10ccb627b29021b9f20757b500285404215097c858cfcea0b2e20c89f8fae5d5",
        "remote": "cluster_results/repgrid/noise_gsm8k_phi3mini/raw_gsm8k_T1.0.pkl",
    },
    "math500_dsmath7b": {
        "file": "raw_math500_T1.5.pkl", "family": "math500", "size": 134163160,
        "sha256": "dc03941c3713227f0afaacc6f60b634c6b907d69d676a5201fd8ea7621d71f86",
        "remote": "cluster_results/regen/math500_dsmath7b/raw_math500_T1.5.pkl",
    },
    "math500_qwenmath7b": {
        "file": "raw_math500_T1.5.pkl", "family": "math500", "size": 399256006,
        "sha256": "34e4c6c7c23b2694f75f72ba247cb00d00f82437c6e6a201369173c6c83c5e8b",
        "remote": "cluster_results/regen/math500_qwenmath7b/raw_math500_T1.5.pkl",
    },
    "math500_r1distill8b": {
        "file": "raw_math500_T1.5.pkl", "family": "math500", "size": 395029741,
        "sha256": "da2da9c73f6d3a5c658dad2d3f9d4348ac9ee5272c2cf64eb87157cfdd40ba6c",
        "remote": "cluster_results/regen/math500_r1distill8b/raw_math500_T1.5.pkl",
    },
    "math500_r1distill8b_mn4096": {
        "file": "raw_math500_T1.5.pkl", "family": "math500", "size": 684761665,
        "sha256": "7d1f8376f6d25004ca532af89a28bbc446c4a36025d877a89a2c07b8d122face",
        "remote": "cluster_results/regen/math500_r1distill8b_mn4096/raw_math500_T1.5.pkl",
    },
    "trace_gsm8k_llama8b_k10": {
        "file": "raw_gsm8k_T1.0.pkl", "family": "gsm8k", "size": 1119977097,
        "sha256": "eae1924f5b7790e74da5721b6e99ef1792627a8abf6400eee0ae42a605aa4281",
        "remote": "cluster_results/regen/trace_gsm8k_llama8b_k10/raw_gsm8k_T1.0.pkl",
    },
    "trace_math500_qwenmath15b_k10": {
        "file": "raw_math500_T1.0.pkl", "family": "math500", "size": 1283184154,
        "sha256": "fff54f9e7078316f936ea933061b04d53367b297bf26c51088c6c9567882c0c3",
        "remote": "cluster_results/regen/trace_math500_qwenmath15b_k10/raw_math500_T1.0.pkl",
    },
}


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def default_raw_root() -> Path:
    candidates = (
        ROOT / "local_cache/math_gate_selection_v1/raw",
        ROOT.parents[1] / "local_cache/math_gate_selection_v1/raw",
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def source_path(raw_root: Path, cell: str) -> Path:
    return raw_root / cell / SOURCES[cell]["file"]


def audit_sources(raw_root: Path) -> dict:
    if tuple(MATH_CELLS) != tuple(SOURCES):
        raise ValueError("source registry does not exactly match frozen math roster")
    rows = []
    for cell in MATH_CELLS:
        spec = SOURCES[cell]
        path = source_path(raw_root, cell)
        print("[audit]", cell, flush=True)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != spec["size"]:
            raise ValueError(f"source size mismatch for {cell}: {size} != {spec['size']}")
        digest = sha256_file(path)
        if digest != spec["sha256"]:
            raise ValueError(f"source SHA-256 mismatch for {cell}")
        manifest_path = ROOT / "dataset_cache/repgrid" / cell / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if int(manifest.get("logprob_top_k", 0)) != 50:
            raise ValueError(f"{cell} manifest does not declare logprob_top_k=50")
        rows.append({
            "cell": cell,
            "family": spec["family"],
            "path": str(path.resolve()),
            "remote": "gdrive:hallucination_detection/" + spec["remote"],
            "size": size,
            "sha256": digest,
            "drive_sha256_prechecked": spec["sha256"],
            "drive_size_prechecked": spec["size"],
            "drive_matches_lfs_pointer": True,
            "manifest": str(manifest_path.resolve()),
            "logprob_top_k": 50,
        })
    result = {
        "schema": "math-gate-source-audit-v1",
        "status": "PASS",
        "cells": len(rows),
        "total_bytes": sum(row["size"] for row in rows),
        "sources": rows,
    }
    atomic_json(OUT / "SOURCE_AUDIT.json", result)
    return result


def _sorted_problem_ids(payload) -> list:
    try:
        return sorted(payload, key=lambda value: int(value))
    except (TypeError, ValueError):
        return sorted(payload, key=str)


def extract_cell(raw_root: Path, cell: str) -> dict:
    checkpoint = OUT / "cells" / f"{cell}.npz"
    if checkpoint.is_file():
        with np.load(checkpoint, allow_pickle=False) as saved:
            if tuple(saved["names"].astype(str)) != features.TEMPORAL_METHODS:
                raise ValueError("checkpoint feature roster drift: " + cell)
            return {key: saved[key] for key in saved.files}

    path = source_path(raw_root, cell)
    print("[load]", cell, path, flush=True)
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    matrix, labels, problem_ids, candidate_ids, uids, lengths = [], [], [], [], [], []
    h1_max_abs = 0.0
    count = sum(len(payload[problem]["candidates"]) for problem in payload)
    completed = 0
    for problem in _sorted_problem_ids(payload):
        candidates = payload[problem]["candidates"]
        for candidate_id, candidate in enumerate(candidates):
            topk = candidate.get("top_k_logprobs")
            if not isinstance(topk, dict) or topk.get("logprobs") is None:
                raise ValueError(f"missing primary top-k payload: {cell}/{problem}/{candidate_id}")
            logprobs = np.asarray(topk["logprobs"], dtype=np.float64)
            entropy = np.asarray(candidate["token_entropies"], dtype=np.float64)
            if logprobs.ndim != 2 or logprobs.shape[1] != 50 or len(logprobs) != len(entropy):
                raise ValueError(f"telemetry alignment mismatch: {cell}/{problem}/{candidate_id}")
            signal_bank = features.token_signals(logprobs, entropy)
            h1_max_abs = max(
                h1_max_abs,
                float(np.max(np.abs(signal_bank["entropy_native"] - signal_bank["q15_H1"]))),
            )
            row = []
            for signal in features.SIGNAL_NAMES:
                readouts = features.apply_temporal_readouts(signal_bank[signal])
                row.extend(readouts[name] for name in features.TEMPORAL_READOUT_NAMES)
            matrix.append(row)
            labels.append(int(not bool(candidate.get("label", False))))
            problem_ids.append(int(problem))
            candidate_ids.append(int(candidate_id))
            uids.append(f"{cell}::{problem}::{candidate_id}")
            lengths.append(len(entropy))
            completed += 1
            if completed % 250 == 0 or completed == count:
                print(f"[extract] {cell} {completed}/{count}", flush=True)
    del payload
    arrays = {
        "X_raw": np.asarray(matrix, dtype=np.float64),
        "y_error": np.asarray(labels, dtype=np.int8),
        "problem_id": np.asarray(problem_ids, dtype=np.int64),
        "candidate_id": np.asarray(candidate_ids, dtype=np.int64),
        "uid": np.asarray(uids),
        "token_length": np.asarray(lengths, dtype=np.int64),
        "names": np.asarray(features.TEMPORAL_METHODS),
        "native_h1_max_abs": np.asarray(h1_max_abs),
    }
    if arrays["X_raw"].shape != (count, len(features.TEMPORAL_METHODS)):
        raise ValueError("feature matrix shape mismatch: " + cell)
    if not np.isfinite(arrays["X_raw"]).all():
        raise ValueError("nonfinite gate detector: " + cell)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(checkpoint, **arrays)
    return arrays


def extract(raw_root: Path) -> dict:
    source_audit = audit_sources(raw_root)
    cells = []
    for cell in MATH_CELLS:
        arrays = extract_cell(raw_root, cell)
        cells.append((cell, arrays))
    combined = {
        "X_raw": np.concatenate([arrays["X_raw"] for _, arrays in cells]),
        "y_error": np.concatenate([arrays["y_error"] for _, arrays in cells]),
        "cell": np.concatenate([
            np.full(len(arrays["y_error"]), cell) for cell, arrays in cells
        ]),
        "family": np.concatenate([
            np.full(len(arrays["y_error"]), SOURCES[cell]["family"])
            for cell, arrays in cells
        ]),
        "problem_id": np.concatenate([arrays["problem_id"] for _, arrays in cells]),
        "candidate_id": np.concatenate([arrays["candidate_id"] for _, arrays in cells]),
        "uid": np.concatenate([arrays["uid"] for _, arrays in cells]),
        "token_length": np.concatenate([arrays["token_length"] for _, arrays in cells]),
        "names": np.asarray(features.TEMPORAL_METHODS),
    }
    y = combined["y_error"]
    if len(y) != EXPECTED_ANSWERS or int(y.sum()) != EXPECTED_ERROR:
        raise ValueError(
            f"math population mismatch: n={len(y)} error={int(y.sum())}; "
            f"expected {EXPECTED_ANSWERS}/{EXPECTED_ERROR}"
        )
    if len(set(combined["uid"].astype(str))) != len(y):
        raise ValueError("duplicate math answer identity")
    archive = OUT / "FEATURES_FROZEN.npz"
    np.savez_compressed(archive, **combined)
    per_cell = []
    for cell, arrays in cells:
        cell_y = arrays["y_error"]
        per_cell.append({
            "cell": cell,
            "family": SOURCES[cell]["family"],
            "answers": int(len(cell_y)),
            "correct": int(len(cell_y) - cell_y.sum()),
            "error": int(cell_y.sum()),
            "error_rate": float(cell_y.mean()),
            "native_h1_max_abs": float(arrays["native_h1_max_abs"]),
            "min_tokens": int(arrays["token_length"].min()),
            "median_tokens": float(np.median(arrays["token_length"])),
            "max_tokens": int(arrays["token_length"].max()),
        })
    result = {
        "schema": "math-gate-feature-archive-v1",
        "status": "FEATURES_FROZEN",
        "archive": str(archive.resolve()),
        "archive_sha256": sha256_file(archive),
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": sha256_file(PROTOCOL),
        "source_audit_sha256": sha256_file(OUT / "SOURCE_AUDIT.json"),
        "answers": len(y),
        "correct": int(len(y) - y.sum()),
        "error": int(y.sum()),
        "signals": len(features.SIGNAL_NAMES),
        "readouts": len(features.TEMPORAL_READOUT_NAMES),
        "candidates": len(features.TEMPORAL_METHODS),
        "cells": per_cell,
        "source_audit": source_audit["status"],
    }
    atomic_json(OUT / "FEATURE_MANIFEST.json", result)
    print("[freeze]", archive, result["archive_sha256"], flush=True)
    return result


def select() -> dict:
    archive = OUT / "FEATURES_FROZEN.npz"
    manifest = json.loads((OUT / "FEATURE_MANIFEST.json").read_text(encoding="utf8"))
    if sha256_file(archive) != manifest["archive_sha256"]:
        raise ValueError("frozen feature archive hash mismatch")
    with np.load(archive, allow_pickle=False) as saved:
        matrix = np.asarray(saved["X_raw"], dtype=np.float64)
        y = np.asarray(saved["y_error"], dtype=np.int8)
        cells = saved["cell"].astype(str)
        families = saved["family"].astype(str)
        names = saved["names"].astype(str).tolist()
    print("[select] single candidates", len(names), flush=True)
    report, scores = selection.develop_fusions(y, matrix, names, cells, families)
    atomic_json(OUT / "SINGLE_RESULTS.json", {
        "schema": "math-gate-single-screen-v1",
        "results": report.pop("single_ranking"),
    })
    atomic_json(OUT / "SELECTION.json", report)
    np.savez_compressed(OUT / "OOF_SCORES.npz", **scores)
    frozen = {
        "schema": "math-gate-frozen-v1",
        "status": "FROZEN_BEFORE_PROCESSBENCH",
        **report["frozen_gate"],
        "feature_names": names,
        "math_feature_archive_sha256": manifest["archive_sha256"],
        "single_results_sha256": sha256_file(OUT / "SINGLE_RESULTS.json"),
        "selection_sha256": sha256_file(OUT / "SELECTION.json"),
        "protocol_sha256": manifest["protocol_sha256"],
        "selection_population": "15 historical math/reasoning development cells",
        "processbench_labels_seen": False,
        "not_external_confirmation": True,
    }
    atomic_json(OUT / "FROZEN_GATE.json", frozen)
    freeze_record = {
        "schema": "math-gate-freeze-record-v1",
        "status": "PASS",
        "frozen_gate": str((OUT / "FROZEN_GATE.json").resolve()),
        "frozen_gate_sha256": sha256_file(OUT / "FROZEN_GATE.json"),
        "written_before_processbench_evaluation": True,
    }
    atomic_json(OUT / "FREEZE_RECORD.json", freeze_record)
    selected = report["arms"][report["winner"]]["result"]["selected"]
    print(
        "[winner]", report["winner"], "q", frozen["q"],
        "family_macro_f1", selected["family_macro"]["macro_f1"],
        flush=True,
    )
    return freeze_record


def preflight(raw_root: Path) -> dict:
    from scripts.test_math_gate_development import run as unit_run

    unit = unit_run()
    missing = [str(source_path(raw_root, cell)) for cell in MATH_CELLS if not source_path(raw_root, cell).is_file()]
    result = {
        "schema": "math-gate-development-preflight-v1",
        "status": "PASS" if not missing and unit["status"] == "PASS" else "BLOCKED",
        "raw_root": str(raw_root.resolve()),
        "missing": missing,
        "unit": unit,
        "python": sys.executable,
        "no_packages_installed": True,
    }
    atomic_json(OUT / "PREFLIGHT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=default_raw_root())
    parser.add_argument("--stage", choices=("preflight", "extract", "select", "all"), default="all")
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    check = preflight(raw_root)
    print(json.dumps(check, indent=2), flush=True)
    if check["status"] != "PASS":
        raise SystemExit(2)
    if args.stage in ("extract", "all"):
        extract(raw_root)
    if args.stage in ("select", "all"):
        select()


if __name__ == "__main__":
    main()
