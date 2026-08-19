#!/usr/bin/env python3
"""Verify the complete six-cell white-box data audit after ``prepare``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


EXPECTED = {
    "gsm8k_t1.0": (500, 496, 500, 496, {"1": 500}, {"1": 496}),
    "triviaqa_t1.0": (500, 500, 500, 500, {"1": 500}, {"1": 500}),
    "sciq_t1.0": (1000, 1000, 1000, 1000, {"1": 1000}, {"1": 1000}),
    "truthfulqa_t0.5": (8170, 8170, 817, 817, {"10": 817}, {"10": 817}),
    "squadv2_t0.5": (10000, 10000, 1000, 1000, {"10": 1000}, {"10": 1000}),
    "nq_open_t0.5": (10000, 10000, 1000, 1000, {"10": 1000}, {"10": 1000}),
}
EXPECTED_EXCLUDED = {"20:0", "359:0", "423:0", "450:0"}


def verify(results_dir: Path) -> None:
    audit = json.loads((results_dir / "data_audit.json").read_text())
    source = json.loads((results_dir / "SOURCE_FREEZE_MANIFEST.json").read_text())
    assert audit["n_cells"] == 6
    assert audit["n_source_rows"] == 30170
    assert audit["n_evaluable_rows"] == 30166
    assert audit["n_excluded_rows"] == 4
    assert set(audit["cells"]) == set(EXPECTED)
    for cell, expected in EXPECTED.items():
        row = audit["cells"][cell]
        source_rows, valid_rows, source_groups, valid_groups, source_mult, valid_mult = expected
        assert row["n_source_rows"] == source_rows, (cell, row["n_source_rows"])
        assert row["n_rows"] == valid_rows, (cell, row["n_rows"])
        assert row["source_n_problems"] == source_groups
        assert row["n_problems"] == valid_groups
        assert row["source_candidate_multiplicity"] == source_mult
        assert row["valid_candidate_multiplicity"] == valid_mult
        assert row["labels_compared_and_equal"] == source_rows
        assert row["all_registered_tensor_shapes_valid"] is True
        assert row["all_numeric_values_finite"] is True
        assert row["labels_equal_between_raw_and_sidecar"] is True
        assert row["token_lengths_equal_for_evaluable_rows"] is True
        assert row["source_rows_globally_namespaced_unique"] is True
        assert row["valid_rows_globally_namespaced_unique"] is True
        assert all(contract["finite"] for contract in row["feature_contracts"].values())
    excluded = {item["row_id"] for item in audit["cells"]["gsm8k_t1.0"]["excluded_rows"]}
    assert excluded == EXPECTED_EXCLUDED
    assert len(source["sources"]) == 12
    assert source["n_source_rows"] == 30170
    assert source["n_evaluable_rows"] == 30166
    for item in source["sources"]:
        assert item["remote_path"].startswith("gdrive:hallucination_detection/")
        assert item["remote_modification_time"].endswith("Z")
        assert item["remote_hash_algorithm"] == "sha256"
        assert item["remote_hash"] == item["local_sha256"]
        assert item["remote_size"] == item["local_size"]
        assert item["remote_local_sha256_equal"] is True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "whitebox_layer_fusion_v1",
    )
    args = parser.parse_args()
    verify(args.results_dir.resolve())
    print("white-box full data audit: PASS")


if __name__ == "__main__":
    main()
