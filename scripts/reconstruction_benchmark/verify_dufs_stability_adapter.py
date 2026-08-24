#!/usr/bin/env python3
"""Verify the isolated Stable-DUFS adapter against the historical a2.dufs arm.

This is a target-free implementation-fidelity check.  It deliberately runs the
legacy omnibus selector in a separate verification process, where its
label-informed GOOD-5 control may be materialized, and compares only the
``a2.dufs`` readout with the isolated adapter used by the scientific fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    load_npz_no_pickle,
)
from spectral_utils.selectors.a2_groupfs import (  # noqa: E402
    a2_groupfs,
    dufs_pf_cell_rng,
    dufs_stability_selection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-cell", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arrays = load_npz_no_pickle(args.prepared_cell)
    allowed = {"X_confidence", "feature_names", "family_ids", "row_ids", "row_index"}
    if set(arrays) != allowed:
        raise RuntimeError("prepared cell has unexpected members")
    matrix = np.asarray(arrays["X_confidence"], dtype=np.float64)
    names = tuple(str(value) for value in arrays["feature_names"].tolist())
    cell = SimpleNamespace(p=matrix.shape[1], V=matrix, pool=names)

    isolated = dufs_stability_selection(
        matrix, dufs_pf_cell_rng(args.cell_id, args.domain, seed=0)
    )
    legacy_rows = a2_groupfs(
        cell, dufs_pf_cell_rng(args.cell_id, args.domain, seed=0)
    )
    legacy_matches = [row for row in legacy_rows if row.get("variant") == "a2.dufs"]
    if len(legacy_matches) != 1:
        raise RuntimeError("legacy family did not return exactly one a2.dufs row")
    legacy = legacy_matches[0]

    checks = {
        "selected_columns_equal": bool(
            np.array_equal(
                np.asarray(isolated["cols"], dtype=np.int64),
                np.asarray(legacy["cols"], dtype=np.int64),
            )
        ),
        "fallback_equal": bool(isolated.get("fallback", False))
        == bool(legacy.get("fallback", False)),
        "lambda_equal": isolated["diag"].get("lambda_dufs")
        == legacy["diag"].get("lambda_dufs"),
        "gate_means_equal": isolated["diag"].get("feat_gate_means")
        == legacy["diag"].get("feat_gate_means"),
    }
    payload = {
        "schema_version": "reconstruction-dufs-stability-adapter-verification-v1",
        "cell_id": args.cell_id,
        "domain": args.domain,
        "prepared_cell": str(args.prepared_cell.resolve()),
        "labels_used": False,
        "isolated_selected_columns": [int(value) for value in isolated["cols"]],
        "legacy_selected_columns": [int(value) for value in legacy["cols"]],
        "checks": checks,
        "pass": all(checks.values()),
    }
    if not payload["pass"]:
        raise RuntimeError(json.dumps(payload, indent=2, sort_keys=True))
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
