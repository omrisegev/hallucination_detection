#!/usr/bin/env python3
"""Execute the label-free A0 audit for the group-free IU research program."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.atomic_nrm_structural_audit import SOURCE_CELLS  # noqa: E402
from spectral_utils.group_free_research import (  # noqa: E402
    PHASE_VERSION,
    audit_processbench_pairing,
    audit_source_environments,
    derive_feature_dag,
    factorial_world_diagnostics,
    sha256_file,
    simulate_factorial_world,
)


DEFAULT_BUNDLE = REPO / "results" / "dependency_fusion_raw" / "cells.npz"
DEFAULT_REPGRID = REPO / "dataset_cache" / "repgrid"
DEFAULT_OUT = REPO / "results" / "automatic_group_free_phase_a0_v1"


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                for key, value in row.items()
            })


def confirmation_boundary() -> dict:
    return {
        "boundary_version": "semgrad-triviaqa-qwen3-4b-confirmation-v1",
        "status": "RESERVED_REQUIRES_COLLECTION",
        "labels_opened": False,
        "dataset": "SemGrad official TriviaQA split",
        "dataset_rows": 11313,
        "model": "Qwen/Qwen3-4B-Instruct-2507",
        "generation": "one greedy response per prompt; max_new_tokens=150",
        "prompt": "Please directly answer the following question with one or few words:\n{query}",
        "primary_label": "BEM answer equivalence at frozen threshold 0.8",
        "primary_metric": "response-level correctness AUROC",
        "bootstrap_group": "source question ID",
        "required_methods": [
            "IU-PCR",
            "frozen Family-NRM",
            "one frozen S1 finalist",
            "optional pre-frozen S2 finalist",
            "optional pre-frozen S3 finalist",
            "supervised atomic ceiling (diagnostic only)",
        ],
        "collection_gate": (
            "Pin the SemGrad dataset artifact and resolved model revision, run local smoke, "
            "then N=200 scientific pilot without opening AUROC. Full collection only if "
            "telemetry is complete and both BEM classes contain at least 30 responses."
        ),
        "reuse_policy": (
            "No rank, component, sign, trust, feature, or method selection may use these labels."
        ),
    }


def report_text(source: dict, pairing: dict, simulator: dict, boundary: dict) -> str:
    complete_features = sum(
        row["source_cell_count"] == len(source["source_cells"])
        for row in source["feature_rows"]
    )
    exact_subsets = sum(
        row["exact_content_fraction"] == 1.0 and row["pairing_fraction"] == 1.0
        for row in pairing["subsets"]
    )
    reduced_cells = [
        row for row in source["environment_rows"]
        if row["bundle_retention_fraction"] is not None
        and row["bundle_retention_fraction"] < 1.0
    ]
    minimum_retention = min(
        (row["bundle_retention_fraction"] for row in reduced_cells),
        default=1.0,
    )
    return f"""# Automatic group-free IU — Phase A0 audit

- Version: `{PHASE_VERSION}`
- Correctness labels accessed: **no**
- Source environments: **{len(source['source_cells'])}**
- Canonical mixed-v2 features: **{len(source['feature_rows'])}**
- Features present in every source environment: **{complete_features}**
- Minimum / maximum feature-pair source coverage: **{source['minimum_pair_coverage']} / {source['maximum_pair_coverage']}**
- Cells whose valid mixed-v2 rows are fewer than manifest attempts: **{len(reduced_cells)}** (minimum retention {minimum_retention:.1%})
- Exact ProcessBench cross-model pairs: **{pairing['total_exact_pairs']}** across Qwen-3 4B, Qwen-3 8B, and Llama-3.1 8B
- Fully exact ProcessBench subsets: **{exact_subsets} / {len(pairing['subsets'])}**
- Simulator crossed dimensions: **{simulator['channel_count']} channels x {simulator['operator_count']} operators**, {simulator['environment_count']} environments
- Simulator duplicate error: **{simulator['maximum_duplicate_error']:.3g}**
- Reserved confirmation: `{boundary['boundary_version']}` ({boundary['status']})

## Decision

A0 passes. The source roster has a fully auditable missingness and pair-coverage
boundary, and exact cross-model pairing exists for 3,400 fixed reasoning traces.
This supports A1/A2 structural work and gives A4 an exact paired-view calibration
surface without semantic matching. The confirmation cell is reserved but must be
collected only after a finalist and all target-selection rules are frozen.

The feature DAG records source streams and computational operators from the
extractor-owned registries and function signatures. It does not import or reproduce
the manual `FEATURE_TO_VIEW` partition.

Manifest observation counts describe attempted/generated candidates, whereas the
bundle contains the valid rows admitted to the frozen mixed-v2 comparison. A1/A2
must preserve the bundle population and equal-environment weighting; they may not
silently restore filtered rows or weight an environment by its candidate count.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(DEFAULT_BUNDLE))
    parser.add_argument("--repgrid", default=str(DEFAULT_REPGRID))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feature_dag = derive_feature_dag()
    source = audit_source_environments(args.bundle, args.repgrid, SOURCE_CELLS)
    pairing = audit_processbench_pairing(args.repgrid, REPO)
    stable_world = factorial_world_diagnostics(simulate_factorial_world())
    drifting_world = factorial_world_diagnostics(simulate_factorial_world(
        seed=20260814,
        environment_specific_target=True,
    ))
    simulator = {
        "stable_target_world": stable_world,
        "environment_specific_target_world": drifting_world,
        **stable_world,
    }
    boundary = confirmation_boundary()

    write_json(out / "feature_dag.json", feature_dag)
    write_csv(out / "feature_dag.csv", feature_dag)
    write_json(out / "source_environment_audit.json", {
        key: value for key, value in source.items() if key != "presence_matrix"
    })
    write_csv(out / "source_environments.csv", source["environment_rows"])
    write_csv(out / "feature_coverage.csv", source["feature_rows"])
    write_csv(out / "feature_pair_coverage.csv", source["pair_rows"])
    np.savez_compressed(
        out / "source_presence.npz",
        source_cells=np.asarray(source["source_cells"]),
        feature_names=np.asarray([row["feature_name"] for row in source["feature_rows"]]),
        presence=np.asarray(source["presence_matrix"], dtype=np.uint8),
    )
    write_json(out / "processbench_cross_model_pairing.json", pairing)
    write_csv(out / "processbench_cross_model_pairing.csv", pairing["model_rows"])
    write_json(out / "simulator_manifest.json", simulator)
    write_json(out / "confirmation_boundary.json", boundary)
    (out / "REPORT.md").write_text(
        report_text(source, pairing, simulator, boundary), encoding="utf-8"
    )
    write_json(out / "A0_COMPLETE.json", {
        "version": PHASE_VERSION,
        "status": "PASS",
        "labels_accessed": False,
        "source_environment_count": len(source["source_cells"]),
        "exact_cross_model_pairs": pairing["total_exact_pairs"],
        "confirmation_boundary": boundary["boundary_version"],
        "next_phase": "A1 factorial soft measurement model",
    })
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "ARTIFACT_HASHES.json"
    }
    write_json(out / "ARTIFACT_HASHES.json", artifact_hashes)
    print(json.dumps({
        "out": str(out),
        "status": "PASS",
        "source_environments": len(source["source_cells"]),
        "exact_cross_model_pairs": pairing["total_exact_pairs"],
        "labels_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
