#!/usr/bin/env python3
"""Formal offline harness for the pure inner-state distributed candidate."""

from __future__ import annotations

import argparse
from pathlib import Path

import scripts.whitebox_depth_consensus_experiment as harness
import scripts.whitebox_depth_distributed_consensus_experiment as distributed_harness
from scripts.whitebox_depth_tail_consensus_experiment import augment_external_baselines
from spectral_utils.whitebox_depth_distributed_pure import (
    COMPONENTS,
    REGISTRY,
    VERSION,
    extract_depth_distributed_pure,
    registry_hash,
)


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "whitebox_depth_distributed_pure_v1"
SOURCE_FILES = (
    "scripts/whitebox_depth_distributed_pure_experiment.py",
    "scripts/whitebox_depth_distributed_consensus_experiment.py",
    "scripts/whitebox_depth_tail_consensus_experiment.py",
    "scripts/whitebox_depth_consensus_experiment.py",
    "spectral_utils/whitebox_depth_distributed_pure.py",
    "spectral_utils/whitebox_depth_distributed_consensus.py",
    "spectral_utils/whitebox_depth_tail_organic_consensus.py",
    "spectral_utils/whitebox_depth_tail_consensus.py",
    "spectral_utils/whitebox_depth_consensus.py",
    "spectral_utils/whitebox_depth_metrics.py",
    "spectral_utils/whitebox_depth_token_metrics.py",
    "spectral_utils/whitebox_layer_fusion.py",
    "spectral_utils/paper_benchmark_suite.py",
    "spectral_utils/upcr.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/selectors/a2_groupfs.py",
)


def configure_harness() -> None:
    harness.RESULTS = RESULTS
    harness.VERSION = VERSION
    harness.COMPONENTS = COMPONENTS
    harness.REGISTRY = REGISTRY
    harness.registry_hash = registry_hash
    harness.extract_depth_consensus = extract_depth_distributed_pure
    harness.best_single_layer = distributed_harness.strongest_atomic_layer_view
    harness.SOURCE_FILES = SOURCE_FILES
    harness.REPORT_TITLE = "Pure white-box distributed-depth consensus"
    harness.DISCOVERY_SELECTION_HISTORY = (
        "Pure inner-state candidate selected retrospectively after the final hybrid and the "
        "strengthened atomic-view oracle exposed a 0.00015 AUROC gap. A compact label-free "
        "depth-summary screen selected mean KL-to-final with two views forced from each depth "
        "quartile. No cell is independent confirmation."
    )
    harness.DISCOVERY_METHOD_DESCRIPTION = (
        "This thirteen-expert U-PCR matrix is entirely internal-state based. It combines "
        "tail signals forced across depth bands, a true per-layer organic hierarchy, a "
        "lens-96 hierarchical DUFS expert, all-layer target uncertainty summaries, and a "
        "depth-spread mean KL-to-final component. It contains no generation-entropy or other "
        "output-only expert."
    )
    harness.DISPLAY = dict(harness.DISPLAY)
    harness.DISPLAY["upcr"] = "Pure distributed-depth · deployed U-PCR"
    harness.DISPLAY["equal_mean"] = "Pure distributed-depth · equal mean"
    harness.DISPLAY["iu_pcr"] = "Pure distributed-depth · IU-PCR"
    harness.DISPLAY["dufs_liu_pcr"] = "Pure distributed-depth · DUFS-LIU-PCR"
    harness.DISPLAY["best_single_layer"] = "Strongest atomic internal layer/view · evaluation oracle"


def main() -> None:
    configure_harness()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit", "evaluate", "report", "all"), nargs="?", default="all")
    parser.add_argument("--cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    distributed_harness.ACTIVE_CACHE = args.cache
    if args.phase in ("prepare", "all"):
        harness.prepare(args.cache, args.results)
    if args.phase in ("fit", "all"):
        harness.fit(args.results)
    if args.phase in ("evaluate", "all"):
        harness.evaluate(args.cache, args.results)
        augment_external_baselines(args.results)
    if args.phase in ("report", "all"):
        augment_external_baselines(args.results)
        harness.report(args.results)


if __name__ == "__main__":
    main()
