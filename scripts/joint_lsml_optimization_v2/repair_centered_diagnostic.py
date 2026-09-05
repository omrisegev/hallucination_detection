"""Add a corrected diagnostic beside frozen outputs; never change old files.

Uses saved float32 order-statistic measurements (promoted to float64), not a
bit-exact rerun of the original float64 producer. No outcomes or label files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from spectral_utils.joint_lsml_integrity import file_sha256
from spectral_utils.trajectory_reducer import ORDERSTAT_K, center_within_answers, fit_orderstat_weights


def repair_fold(root, cell, outer):
    directory = root / "structure" / cell / f"outer{outer}"
    target = directory / "CENTERED_DIAGNOSTIC_REPAIR.json"
    if not (directory / "COMPLETE.json").is_file() or target.exists():
        raise RuntimeError(f"incomplete fold or existing correction: {directory}")
    folds_path = root / "folds/folds.json"
    cells_path = root / "cells" / f"{cell}.npz"
    matrix_path = directory / "moduleb.npz"
    folds = json.loads(folds_path.read_text(encoding="utf-8"))
    panel = "processbench" if cell.startswith("pb_") else "prmbench"
    assignment = folds[panel]["outer"]
    with np.load(cells_path, allow_pickle=False) as bundle:
        groups = bundle["group_ids"].astype(str)
    if any(g not in assignment for g in groups):
        raise RuntimeError("group missing from frozen fold assignment")
    with np.load(matrix_path, allow_pickle=False) as bundle:
        matrix = np.asarray(bundle["orderstats"], dtype=np.float64)
        lengths = bundle["lengths"]
        owners = bundle["step_rows"]
    train = np.asarray([assignment[g] != outer for g in groups])[owners]
    admitted = train & (lengths >= ORDERSTAT_K)
    centered = center_within_answers(matrix[admitted], owners[admitted])
    try:
        weights, meta = fit_orderstat_weights(centered, lengths[admitted])
        result = {"status": "FITTED", "weights": weights.tolist(), "metadata": meta}
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        result = {"status": "UNAVAILABLE", "reason": str(error), "weights": None}
    result.update({
        "created_utc": datetime.now(timezone.utc).isoformat(), "cell": cell, "outer": outer,
        "labels_accessed": False, "diagnostic_only": True,
        "centering": "per answer, using full-length outer-training steps only",
        "precision": "saved float32 orderstats promoted to float64",
        "input_sha256": {p.relative_to(root).as_posix(): file_sha256(p)
                         for p in (folds_path, cells_path, matrix_path)},
        "repair_source_sha256": file_sha256(Path(__file__)),
    })
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.results_root.resolve()
    if (root / "evaluation").exists() or (root / "INTEGRITY_RECORD_V2.json").exists():
        raise RuntimeError("diagnostic correction must precede evaluation and late freeze")
    cells = [f"pb_{s}_{m}" for s in ("gsm8k", "math", "olympiadbench", "omnimath") for m in ("q4", "q8")]
    cells.append("prmbench_qwen3_8b")
    # Preflight the complete campaign before writing the first correction.
    for cell in cells:
        for outer in range(5):
            folder = root / "structure" / cell / f"outer{outer}"
            if not (folder / "COMPLETE.json").is_file() or (folder / "CENTERED_DIAGNOSTIC_REPAIR.json").exists():
                raise RuntimeError(f"campaign incomplete or already amended: {folder}")
    for cell in cells:
        for outer in range(5):
            result = repair_fold(root, cell, outer)
            print(cell, outer, result["status"], flush=True)


if __name__ == "__main__":
    main()
