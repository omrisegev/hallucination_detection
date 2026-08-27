#!/usr/bin/env python3
"""Fit/evaluate the CIW cross-scale token-response input layer."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.ciw_cross_scale_localization import fit_cross_scale_token_head
from spectral_utils.reconstruction_benchmark.io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_bytes,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (
    empirical_midrank,
    load_prepared_localization_cell,
    validate_fit_manifest,
)
from spectral_utils.reconstruction_benchmark.localization_fit import (
    load_localization_score_bundle,
)


METHODS = {
    "iu": "ciw_cross_scale_token_response_v1",
    "su": "ciw_cross_scale_su_token_response_v1",
}
CELLS = (
    "processbench_gsm8k_qwen3_4b", "processbench_math_qwen3_4b",
    "processbench_olympiadbench_qwen3_4b", "processbench_omnimath_qwen3_4b",
    "processbench_gsm8k_qwen3_8b", "processbench_math_qwen3_8b",
    "processbench_olympiadbench_qwen3_8b", "processbench_omnimath_qwen3_8b",
    "processbench_gsm8k_llama31_8b", "processbench_math_llama31_8b",
    "processbench_olympiadbench_llama31_8b", "processbench_omnimath_llama31_8b",
    "prmbench_response_qwen3_8b",
)
SOURCE_FILES = (
    "spectral_utils/ciw_cross_scale_localization.py",
    "spectral_utils/dependency_fusion.py",
    "spectral_utils/upcr.py",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/localization_contract.py",
    "spectral_utils/reconstruction_benchmark/localization_fit.py",
    "scripts/reconstruction_benchmark/run_ciw_cross_scale_localization.py",
    "scripts/reconstruction_benchmark/run_ciw_localization.py",
)


def _payload_sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _load_ciw(ciw_root: Path, cell_id: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record_path = ciw_root / "cells" / cell_id / "RECORD.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    payload = dict(record)
    digest = payload.pop("payload_sha256", None)
    if digest != _payload_sha(payload):
        raise RuntimeError(f"{cell_id}: CIW response record payload failed")
    score_path = record_path.parent / str(record["score_path"])
    if sha256_file(score_path) != record["score_sha256"]:
        raise RuntimeError(f"{cell_id}: CIW response score hash failed")
    return record, load_npz_no_pickle(score_path)


def fit(ciw_root: Path, localization_release: Path, out: Path, *, fusion: str) -> None:
    method = METHODS[fusion]
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    input_root = localization_release / "build_A/localization/inputs"
    manifest = validate_fit_manifest(input_root / "MANIFEST.json", input_root=input_root)
    records_by_cell = {str(row["cell_id"]): row for row in manifest["cells"]}
    if not set(CELLS).issubset(records_by_cell):
        raise RuntimeError("localization input manifest lacks a required cell")

    frozen_localization_root = localization_release / "build_A/localization/fit/cells"
    records: list[dict[str, Any]] = []
    for index, cell_id in enumerate(CELLS, start=1):
        input_record = records_by_cell[cell_id]
        input_path = input_root / str(input_record["artifact_path"])
        cell = load_prepared_localization_cell(input_path, input_record)
        ciw_record, ciw = _load_ciw(ciw_root, cell_id)
        ciw_rows = np.asarray(ciw["row_ids"]).astype(str)
        if not np.array_equal(ciw_rows, np.asarray(cell.row_ids)):
            raise RuntimeError(f"{cell_id}: CIW response and token rows differ")

        old_record_path = frozen_localization_root / cell_id / "RECORD.json"
        _old_record, old = load_localization_score_bundle(old_record_path)
        if not np.array_equal(np.asarray(old["row_ids"]).astype(str), ciw_rows):
            raise RuntimeError(f"{cell_id}: frozen token-IU rows differ")
        response_risk = np.asarray(ciw["score"], dtype=np.float64)
        result = fit_cross_scale_token_head(
            cell.token_confidence,
            cell.token_offsets,
            cell.segment_starts,
            cell.segment_ends,
            cell.row_ids,
            response_risk,
            fusion=fusion,
        )
        response_rank = empirical_midrank(response_risk)
        step_rank = empirical_midrank(result.step_risk)
        expanded_response = np.repeat(response_rank, np.diff(cell.segment_offsets))
        combined = np.sqrt(expanded_response * step_rank)
        if combined.shape != result.step_risk.shape or not np.isfinite(combined).all():
            raise RuntimeError(f"{cell_id}: combined score is invalid")

        target = out / "cells" / cell_id
        target.mkdir(parents=True, exist_ok=False)
        score_path = target / "scores.npz"
        score_sha = atomic_write_npz(score_path, {
            "row_ids": np.asarray(cell.row_ids, dtype="<U80"),
            "segment_offsets": np.asarray(cell.segment_offsets, dtype="<i8"),
            "step_scores": combined.astype("<f8"),
            "ciw_response_score": response_risk.astype("<f8"),
            "token_step_score": result.step_risk.astype("<f8"),
            "baseline_token_step_score": np.asarray(old["token_step_score"], dtype="<f8"),
            "reliability": result.reliability.astype("<f8"),
            "gate": result.gate.astype("<f8"),
        })
        record = {
            "schema_version": "ciw-cross-scale-localization-cell-v1",
            "method_id": method,
            "cell_id": cell_id,
            "model_id": ciw_record["model_id"],
            "slice_id": ciw_record["slice_id"],
            "n_rows": len(cell.row_ids),
            "n_tokens": len(cell.token_confidence),
            "n_steps": len(combined),
            "formula": "geomean(CIW-response-rank, cross-scale-innovation-token-IU29-rank)",
            "input_layer": dict(result.diagnostics),
            "labels_opened_during_fit": False,
            "targets_accessed_during_fit": False,
            "prepared_input_sha256": sha256_file(input_path),
            "ciw_record_sha256": sha256_file(ciw_root / "cells" / cell_id / "RECORD.json"),
            "localization_record_sha256": sha256_file(old_record_path),
            "score_path": "scores.npz",
            "score_sha256": score_sha,
        }
        record["payload_sha256"] = _payload_sha(record)
        record_path = target / "RECORD.json"
        atomic_write_json(record_path, record)
        records.append({
            "cell_id": cell_id,
            "record_path": record_path.relative_to(out).as_posix(),
            "record_sha256": sha256_file(record_path),
            "score_sha256": score_sha,
        })
        print(f"fit {cell_id} ({index}/{len(CELLS)})", flush=True)

    freeze = {
        "schema_version": "ciw-cross-scale-localization-score-freeze-v1",
        "method_id": method,
        "expected_cells": list(CELLS),
        "n_cells": len(records),
        "all_expected_scores_present": len(records) == len(CELLS),
        "labels_opened_during_fit": False,
        "targets_accessed_during_fit": False,
        "input_manifest_sha256": sha256_file(input_root / "MANIFEST.json"),
        "ciw_external_freeze_sha256": sha256_file(ciw_root / "SCORE_FREEZE_MANIFEST.json"),
        "source_snapshot": [
            {"path": name, "sha256": sha256_file(ROOT / name)} for name in SOURCE_FILES
        ],
        "records": records,
    }
    freeze["payload_sha256"] = _payload_sha(freeze)
    atomic_write_json(out / "SCORE_FREEZE_MANIFEST.json", freeze)
    print(json.dumps({"status": "PASS", "stage": "fit", "cells": len(records)}, indent=2))


def _load_legacy_evaluator():
    path = ROOT / "scripts/reconstruction_benchmark/run_ciw_localization.py"
    spec = importlib.util.spec_from_file_location("_ciw_localization_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the frozen localization evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(fit_root: Path, localization_release: Path, out: Path) -> None:
    freeze = json.loads((fit_root / "SCORE_FREEZE_MANIFEST.json").read_text(encoding="utf-8"))
    method = str(freeze.get("method_id", ""))
    if method not in set(METHODS.values()):
        raise RuntimeError("unknown cross-scale localization method")
    module = _load_legacy_evaluator()
    module.METHOD = method
    module.evaluate(fit_root, localization_release, out)

    # The numerical evaluator is deliberately reused unchanged.  Replace only
    # its legacy prose and rebind the evaluation manifest to the final report.
    report = out / "REPORT.md"
    text = report.read_text(encoding="utf-8")
    text = text.replace(
        "# CIW-DEEM localization transfer",
        "# CIW cross-scale token-response localization",
    ).replace(
        "CIW response risk is combined with the already frozen token-IU29 step risk. No new token model is fitted.",
        "A target-free CIW-style input layer removes predictable whole-response variation from each token stream before the unchanged token IU-PCR head. The frozen CIW response risk is then combined with the corrected step risk.",
    ).replace(
        "This is an application transfer result, not a new method-selection test.",
        "This is a new application-method experiment; its score freeze was completed before the existing localization targets were opened.",
    )
    report.write_text(text, encoding="utf-8")
    manifest_path = out / "EVALUATION_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "ciw-cross-scale-localization-evaluation-v1"
    manifest["method_id"] = method
    manifest["outputs"]["REPORT.md"] = sha256_file(report)
    manifest.pop("payload_sha256", None)
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(manifest_path, manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    fit_parser = sub.add_parser("fit")
    fit_parser.add_argument("--ciw-root", required=True)
    fit_parser.add_argument("--localization-release", required=True)
    fit_parser.add_argument("--out-dir", required=True)
    fit_parser.add_argument("--fusion", choices=tuple(METHODS), default="iu")
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--fit-dir", required=True)
    eval_parser.add_argument("--localization-release", required=True)
    eval_parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.stage == "fit":
        fit(
            Path(args.ciw_root).resolve(),
            Path(args.localization_release).resolve(),
            Path(args.out_dir).resolve(),
            fusion=args.fusion,
        )
    else:
        evaluate(Path(args.fit_dir).resolve(), Path(args.localization_release).resolve(), Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
