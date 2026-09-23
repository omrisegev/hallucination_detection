"""Estimate telemetry-only costs from completed timing jobs; no quality metrics."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.external_generalization.budget import cost_estimate
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="Directories containing compact TIMING.json and TOKENIZATION.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in args.runs:
        timing = json.loads((run / "TIMING.json").read_text())
        tokenization = json.loads((run / "TOKENIZATION.json").read_text())
        if not timing["complete"] or timing["quality_evaluated"]:
            raise ValueError("requires completed timing-only run")
        if timing["identity"] != tokenization["identity"]:
            raise ValueError("timing/tokenization identities disagree")
        result = cost_estimate(tokenization["lengths"], timing["measurements"])
        result.update(run=run.name, job_id=timing["job_id"], identity=timing["identity"],
                      load_seconds=timing["load_seconds"],
                      timing_sha256=file_hash(run / "TIMING.json"),
                      tokenization_sha256=file_hash(run / "TOKENIZATION.json"))
        result["estimated_gpu_hours_with_load"] = result["estimated_gpu_hours"] + timing["load_seconds"]/3600
        rows.append(result)
    atomic_json(args.out, {"scope": "TELEMETRY_COLLECTION_ONLY", "cells": rows,
                "estimated_gpu_hours": sum(r["estimated_gpu_hours_with_load"] for r in rows),
                "estimated_storage_bytes": sum(r["storage_bytes"] for r in rows),
                "comparator_generation_included": False, "bootstrap_or_feature_cpu_included": False,
                "model_download_and_environment_startup_included": False,
                "full_run_approved": False,
                "limitations": ["length-based projection from <=12 examples per cell, not a guaranteed cap",
                                "separately budget comparator generation, model preparation and startup retries"]})
    print(args.out)


if __name__ == "__main__":
    main()
