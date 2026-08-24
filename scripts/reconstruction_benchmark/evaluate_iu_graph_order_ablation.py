#!/usr/bin/env python3
"""Verify A/B and evaluate the frozen IU graph-order ablation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.iu_graph_order_evaluation import (  # noqa: E402
    evaluate,
    verify_ab,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-release", type=Path, required=True)
    parser.add_argument("--output-release", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20000)
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    output_release = args.output_release.resolve()
    source_release = args.source_release.resolve()
    evaluation_dir = output_release / "evaluation"
    if evaluation_dir.exists():
        raise FileExistsError(f"evaluation already exists: {evaluation_dir}")
    config_path = REPO / "configs/reconstruction_benchmark_v1/iu_graph_order_ablation_v1.json"
    verified = verify_ab(
        output_release=output_release,
        source_release=source_release,
        config_path=config_path,
    )
    if args.draws != 20000:
        raise ValueError("scientific v1 evaluation requires exactly 20,000 draws")
    result, metrics, contrasts = evaluate(verified, draws=args.draws)

    temporary = Path(tempfile.mkdtemp(prefix=".evaluation.", dir=output_release))
    try:
        ab_sha = atomic_write_json(temporary / "SCORE_AB_VERIFICATION.json", dict(verified.ab_record))
        result_sha = atomic_write_json(temporary / "EVALUATION.json", result)
        _write_csv(temporary / "metrics_long.csv", metrics)
        _write_csv(temporary / "contrasts_long.csv", contrasts)
        manifest = {
            "schema_version": "iu-graph-order-evaluation-manifest-v1",
            "status": result["status"],
            "headline_status": result["headline_status"],
            "score_ab_sha256": ab_sha,
            "evaluation_sha256": result_sha,
            "metrics_sha256": sha256_file(temporary / "metrics_long.csv"),
            "contrasts_sha256": sha256_file(temporary / "contrasts_long.csv"),
            "bootstrap_draws": args.draws,
            "n_cells": result["n_cells"],
            "n_new_arms": result["n_new_arms"],
        }
        atomic_write_json(temporary / "EVALUATION_MANIFEST.json", manifest)
        os.replace(temporary, evaluation_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(json.dumps({
        "status": result["status"],
        "headline_status": result["headline_status"],
        "evaluation_dir": str(evaluation_dir),
        "macro_points": result["macro_points"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
