#!/usr/bin/env python3
"""Mechanically refresh S3 cost fields from already-frozen score artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_local_online_comprehensive_stage1 import OUT, _sha256  # noqa: E402
from scripts.run_local_online_comprehensive_stage3 import (  # noqa: E402
    _aggregate_architecture,
    _architecture_configs,
    _write_csv,
    _write_json,
)


def _read(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    metrics = _read(OUT / "STAGE_3_ARCHITECTURE_METRICS.csv")
    intervals = _read(OUT / "STAGE_3_ARCHITECTURE_INTERVALS.csv")
    aggregate = _aggregate_architecture(metrics, _architecture_configs())
    _write_csv(OUT / "STAGE_3_ARCHITECTURE_AGGREGATE.csv", aggregate)

    selection_path = OUT / "STAGE_3_ARCHITECTURE_SELECTION.json"
    selection = json.loads(selection_path.read_text())
    best_local = max(row["local"] for row in aggregate)
    best_online = max(row["online"] for row in aggregate)
    survivors = [
        row for row in aggregate
        if row["local"] >= best_local - 0.010
        and row["online"] >= best_online - 0.015
    ]
    simplest = min(
        survivors,
        key=lambda row: (
            row["head_count"], row["coordinate_count"],
            row["fusion_terms"], row["candidate"],
        ),
    )
    selected_name = selection["selected"]["candidate"]
    selection["selected"] = next(
        row for row in aggregate if row["candidate"] == selected_name
    )
    selection["simplest_survivor"] = simplest
    selection["survivors"] = [row["candidate"] for row in survivors]
    selection["cost_rule"] = (
        "physical heads (Local always required for the locator), then "
        "coordinates, then nonzero fusion terms"
    )
    _write_json(selection_path, selection)

    interval_lookup = {
        (row["candidate"], row["task"]): row for row in intervals
    }
    lines = [
        "# S3 fusion and joint architecture",
        "",
        f"**Verdict: `{selection['verdict']}`.**",
        "",
        f"Same-matrix fusion selection: Local `{selection['local_fusion']}`, Online `{selection['online_fusion']}`.",
        f"Joint architecture selection: `{selected_name}`.",
        f"Direct references: Local `{selection['local_reference']}`, Online `{selection['online_reference']}`.",
        "",
        "A Local head is counted whenever a step locator is emitted, even if its detector coefficient is zero.",
        "",
        "| architecture | Local F1 | delta | Local 95% CI | Online AUROC | delta | Online 95% CI | heads | coords |",
        "|---|---:|---:|---|---:|---:|---|---:|---:|",
    ]
    for row in sorted(
        aggregate, key=lambda item: item["local"] + item["online"], reverse=True
    ):
        li = interval_lookup[(row["candidate"], "local")]
        oi = interval_lookup[(row["candidate"], "online")]
        lines.append(
            f"| {row['candidate']} | {row['local']:.4f} | {float(li['delta']):+.4f} | "
            f"[{float(li['ci_low']):+.4f}, {float(li['ci_high']):+.4f}] | "
            f"{row['online']:.4f} | {float(oi['delta']):+.4f} | "
            f"[{float(oi['ci_low']):+.4f}, {float(oi['ci_high']):+.4f}] | "
            f"{row['head_count']} | {row['coordinate_count']} |"
        )
    lines.extend([
        "",
        "Tier-B critic and PRM metrics remain in the machine-readable Local table and are not treated as same-access deltas.",
    ])
    (OUT / "STAGE_3_ARCHITECTURE.md").write_text("\n".join(lines) + "\n")

    manifest_path = OUT / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["stage3_architecture_selection_sha256"] = _sha256(selection_path)
    _write_json(manifest_path, manifest)
    print(json.dumps(selection["selected"], indent=2))


if __name__ == "__main__":
    main()
