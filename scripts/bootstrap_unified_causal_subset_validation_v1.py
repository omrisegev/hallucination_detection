#!/usr/bin/env python3
"""Paired source-question bootstrap for a completed subset validation run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import types

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [str(ROOT / "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.run_unified_causal_iu_v1 import (  # noqa: E402
    _write_csv,
    _write_json,
    grouped_bootstrap_comparisons,
)


MARGINS = {"global": 0.010, "localization": 0.010, "early": 0.015}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--control", required=True)
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260818)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records_path = args.run / "VALIDATION_RECORDS.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(records_path)
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line]
    candidates = sorted({str(record["candidate"]) for record in records})
    if args.control not in candidates:
        raise ValueError("control is absent from validation records")
    output, rows = {}, []
    for candidate in candidates:
        if candidate == args.control:
            continue
        paired = [
            {**record, "outer_fold": 0}
            for record in records
            if str(record["candidate"]) in {candidate, args.control}
        ]
        result = grouped_bootstrap_comparisons(
            paired,
            repeats=args.repeats,
            seed=args.seed,
            primary_candidate=candidate,
        )
        comparison = result["comparisons"][args.control]
        output[candidate] = comparison
        improves = []
        noninferior = []
        for task in ("global", "localization", "early"):
            delta = comparison[task]["delta"]
            low, high = comparison[task]["ci95"]
            rows.append({
                "candidate": candidate,
                "control": args.control,
                "task": task,
                "delta": delta,
                "ci95_low": low,
                "ci95_high": high,
                "margin": MARGINS[task],
                "ci_positive": low > 0.0,
                "ci_noninferior": low > -MARGINS[task],
                "bootstrap_repeats": args.repeats,
            })
            improves.append(low > 0.0)
            noninferior.append(low > -MARGINS[task])
        output[candidate]["promotion_gate"] = {
            "positive_ci_on_at_least_one_task": any(improves),
            "noninferior_ci_on_all_tasks": all(noninferior),
            "passes": any(improves) and all(noninferior),
        }
    payload = {
        "unit": "dataset-qualified source question",
        "control": args.control,
        "repeats": args.repeats,
        "seed": args.seed,
        "comparisons": output,
    }
    _write_json(args.run / "BOOTSTRAP.json", payload)
    _write_csv(args.run / "BOOTSTRAP.csv", rows)
    lines = [
        "# Frozen scorer-model validation bootstrap",
        "",
        f"Control: `{args.control}`. Paired source-question bootstrap, {args.repeats} replicates.",
        "",
        "| candidate | ΔGlobal [95% CI] | ΔLocalization [95% CI] | ΔEarly [95% CI] | gate |",
        "|---|---:|---:|---:|:---:|",
    ]
    for candidate in candidates:
        if candidate == args.control:
            continue
        current = output[candidate]
        cells = []
        for task in ("global", "localization", "early"):
            value = current[task]
            cells.append(
                f"{value['delta']:+.4f} [{value['ci95'][0]:+.4f}, {value['ci95'][1]:+.4f}]"
            )
        lines.append(
            f"| {candidate} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{'PASS' if current['promotion_gate']['passes'] else 'fail'} |"
        )
    lines.extend([
        "",
        "The gate requires a positive lower CI on at least one task and lower CIs above "
        "the frozen noninferiority margins (-0.010 Global, -0.010 Localization, -0.015 Early) "
        "on all remaining tasks.",
        "",
    ])
    (args.run / "BOOTSTRAP_REPORT.md").write_text("\n".join(lines))
    print(args.run / "BOOTSTRAP_REPORT.md")


if __name__ == "__main__":
    main()
