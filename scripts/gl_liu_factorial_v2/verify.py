#!/usr/bin/env python3
"""Verify frozen-v1 score equality and the factorial result contract."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HASH_PAIRS = (
    ("detectors", "global_iu", "detectors", "answer_iu_mixed"),
    ("detectors", "global_dufs", "detectors", "answer_dufs_liu_mixed"),
    ("token_curves", "local_temporal_core", "token_curves", "token_temporal_liu_l0p3"),
    ("token_curves", "local_dufs_core", "token_curves", "token_dufs_liu_l0p3"),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--frozen-dir", required=True)
    args = parser.parse_args()
    results = Path(args.results_dir)
    frozen = Path(args.frozen_dir)

    records = []
    for current_path in sorted((results / "diagnostics").glob("*.json")):
        frozen_path = frozen / "diagnostics" / (
            current_path.stem + "__diagnostics.json"
        )
        current = json.loads(current_path.read_text())["hashes_before_labels"]
        reference = json.loads(frozen_path.read_text())["hashes_before_labels"]
        checks = {}
        for new_group, new_name, old_group, old_name in HASH_PAIRS:
            checks[new_name] = current[new_group][new_name] == reference[old_group][old_name]
        records.append({"cell": current_path.stem, "checks": checks, "all_equal": all(checks.values())})

    with (results / "system_macros.csv").open() as handle:
        macros = list(csv.DictReader(handle))
    all_rows = {row["system"]: row for row in macros if row["group"] == "all_8_cells"}
    headline = {
        "mindgap_control_f1": float(all_rows["mindgap_control"]["f1"]),
        "gl_liu_v1_reproduced_f1": float(
            all_rows["global_dufs__local_temporal_core"]["f1"]
        ),
        "unified_core_f1": float(all_rows["global_dufs__local_dufs_core"]["f1"]),
        "unified_broad_f1": float(all_rows["global_dufs__local_dufs_broad"]["f1"]),
    }
    output = {
        "all_frozen_hashes_equal": all(record["all_equal"] for record in records),
        "n_cells_checked": len(records),
        "hash_checks": records,
        "headline": headline,
    }
    if not output["all_frozen_hashes_equal"] or len(records) != 8:
        raise SystemExit(json.dumps(output, indent=2))
    (results / "REPRODUCTION_CHECK.json").write_text(json.dumps(output, indent=2))
    print(results / "REPRODUCTION_CHECK.json")


if __name__ == "__main__":
    main()
