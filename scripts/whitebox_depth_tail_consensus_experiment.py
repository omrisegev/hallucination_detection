#!/usr/bin/env python3
"""Formal offline harness for the retrospective depth-tail consensus candidate.

The phase boundary, manifests, grouped evaluation, and HTML generator are reused
from ``whitebox_depth_consensus_experiment``.  This adapter replaces only the
frozen feature registry and augments the sorted headline with independent
published-method approximations already evaluated by the source benchmark.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import scripts.whitebox_depth_consensus_experiment as harness
from spectral_utils.whitebox_depth_tail_consensus import (
    COMPONENTS,
    REGISTRY,
    VERSION,
    extract_depth_tail_consensus,
    registry_hash,
)


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "whitebox_depth_tail_consensus_v1"
SOURCE_FILES = (
    "scripts/whitebox_depth_tail_consensus_experiment.py",
    "scripts/whitebox_depth_consensus_experiment.py",
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

EXTERNAL_BASELINES = {
    "generation_entropy_mean": "Generation entropy · output baseline",
    "final_layer_nll": "Final-layer target NLL · output baseline",
    "dola_kl_supervised_lr": "DoLa KL trajectory · grouped L2 probe",
    "dola_kl_equal_mean": "DoLa KL trajectory · equal mean proxy",
    "spilled_energy_eq8_mean_proxy": "Spilled Energy Eq. 8 · mean proxy",
    "haloscope_direct_proxy": "HaloScope · direct projection proxy",
}


def configure_harness() -> None:
    harness.RESULTS = RESULTS
    harness.VERSION = VERSION
    harness.COMPONENTS = COMPONENTS
    harness.REGISTRY = REGISTRY
    harness.registry_hash = registry_hash
    harness.extract_depth_consensus = extract_depth_tail_consensus
    harness.SOURCE_FILES = SOURCE_FILES
    harness.REPORT_TITLE = "White-box depth-tail consensus"
    harness.DISCOVERY_SELECTION_HISTORY = (
        "Selected retrospectively after a 245-component token-tail screen and a compact "
        "combination audit on these same 13 eligible cells. The chosen combination was "
        "preferred for performance on both the original six-cell and architecture-cell "
        "subsets; neither subset is independent confirmation because both had appeared in "
        "earlier experiment history."
    )
    harness.DISCOVERY_METHOD_DESCRIPTION = (
        "The seven-column matrix combines the earlier four depth-consensus signals with "
        "maximum top-1 surprisal, maximum target-token NLL, and maximum entropy excess over "
        "top-1 surprisal. The first two additions force two views from every depth quartile; "
        "the third uses eight anchor-reliable views. Correctness labels are unavailable to "
        "feature selection and fitting."
    )
    harness.DISPLAY = dict(harness.DISPLAY)
    harness.DISPLAY["upcr"] = "Depth-tail consensus · deployed U-PCR"
    harness.DISPLAY["equal_mean"] = "Depth-tail consensus · equal mean"
    harness.DISPLAY["iu_pcr"] = "Depth-tail consensus · IU-PCR"
    harness.DISPLAY["dufs_liu_pcr"] = "Depth-tail consensus · DUFS-LIU-PCR"
    harness.DISPLAY["best_single_layer"] = "Best single module/metric/layer · evaluation oracle"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def augment_external_baselines(results: Path) -> None:
    """Add independent-method rows without changing candidate selection or scores."""

    headline_path = results / "headline_summary.csv"
    rows = read_csv(headline_path)
    present = {row["method"] for row in rows}
    source_headline = read_csv(harness.BASE_RESULTS / "headline_summary.csv")
    source_unique = {}
    for row in source_headline:
        method = row["method"]
        if method in EXTERNAL_BASELINES and method != "dola_kl_supervised_lr":
            source_unique[method] = row
    for method, display in EXTERNAL_BASELINES.items():
        if method in present or method == "dola_kl_supervised_lr":
            continue
        source = source_unique[method]
        rows.append({
            "method": method,
            "display_method": display,
            "n_cells": len(harness.PRIMARY_CELLS),
            "macro_auroc": source["macro_auroc"],
            "macro_auroc_ci_low": source["macro_auroc_ci_low"],
            "macro_auroc_ci_high": source["macro_auroc_ci_high"],
            "macro_auprc": source["macro_auprc"],
            "macro_auprc_ci_low": source["macro_auprc_ci_low"],
            "macro_auprc_ci_high": source["macro_auprc_ci_high"],
            "label_use": "none; source-benchmark proxy",
        })

    if "dola_kl_supervised_lr" not in present:
        per_cell = read_csv(harness.BASE_RESULTS / "per_cell_metrics.csv")
        selected = [
            row for row in per_cell
            if row["method"] == "dola_kl_supervised_lr" and row["cell"] in harness.PRIMARY_CELLS
        ]
        if len(selected) != len(harness.PRIMARY_CELLS):
            raise RuntimeError("DoLa grouped-probe source roster is incomplete")
        rows.append({
            "method": "dola_kl_supervised_lr",
            "display_method": EXTERNAL_BASELINES["dola_kl_supervised_lr"],
            "n_cells": len(selected),
            "macro_auroc": sum(float(row["auroc"]) for row in selected) / len(selected),
            "macro_auroc_ci_low": "nan",
            "macro_auroc_ci_high": "nan",
            "macro_auprc": sum(float(row["auprc"]) for row in selected) / len(selected),
            "macro_auprc_ci_low": "nan",
            "macro_auprc_ci_high": "nan",
            "label_use": "5-fold grouped supervised; source-benchmark approximation",
        })
    rows.sort(key=lambda row: float(row["macro_auroc"]), reverse=True)
    write_csv(headline_path, rows)


def main() -> None:
    configure_harness()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit", "evaluate", "report", "all"), nargs="?", default="all")
    parser.add_argument("--cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.phase in ("prepare", "all"):
        harness.prepare(args.cache, args.results)
    if args.phase in ("fit", "all"):
        harness.fit(args.results)
    if args.phase in ("evaluate", "all"):
        harness.evaluate(args.cache, args.results)
        augment_external_baselines(args.results)
    if args.phase in ("report", "all"):
        # Idempotent when report is regenerated after a separate evaluation phase.
        augment_external_baselines(args.results)
        harness.report(args.results)


if __name__ == "__main__":
    main()
