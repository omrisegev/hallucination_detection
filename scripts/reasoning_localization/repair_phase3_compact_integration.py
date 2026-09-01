#!/usr/bin/env python3
"""Repair the registered H0 alias and complete the interrupted P3 integration."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json  # noqa: E402
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization.build_reasoning_localization_report import REPORTING  # noqa: E402
from scripts.reasoning_localization.register_phase3_compact_fusion import EXPERIMENT  # noqa: E402

OLD = "P3_H0_REFERENCE"
NEW = "P2C_F6_TOP10_REFERENCE"


def repair_csv(path: Path) -> int:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle)); fields = list(rows[0])
    changed = 0
    for row in rows:
        if row.get("experiment_id") != EXPERIMENT:
            continue
        for field in ("variant_id", "left_variant_id", "right_variant_id"):
            if row.get(field) == OLD:
                row[field] = NEW; changed += 1
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    return changed


def main() -> None:
    changed = repair_csv(p1.PROGRAM_ROOT/"METRICS_LONG.csv") + repair_csv(p1.PROGRAM_ROOT/"CONTRASTS_LONG.csv")
    cp = p1.PROGRAM_ROOT/"CLAIMS.json"; claims = json.loads(cp.read_text())
    claim = next(row for row in claims["claims"] if row["claim_id"] == "CLAIM_P3_OUTER_IU_HARM")
    claim["statistical_summary"]["bound_basis"] = "Preregistered P3 practical-benefit bound and zero-harm directional verdict."
    atomic_write_json(cp, claims)
    build = REPORTING.prepare_build(p1.PROGRAM_ROOT, REPO); REPORTING.write_build(p1.PROGRAM_ROOT, build)
    print(json.dumps({"status":"REPAIRED","alias_cells_changed":changed,"report_sha256":build.manifest["output"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
