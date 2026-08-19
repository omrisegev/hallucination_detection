#!/usr/bin/env python3
"""Formal harness for the strongest retrospective depth-tail hybrid."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

import scripts.whitebox_depth_consensus_experiment as harness
from scripts.whitebox_depth_tail_consensus_experiment import augment_external_baselines
from spectral_utils.whitebox_depth_tail_organic_consensus import (
    COMPONENTS as WHITEBOX_COMPONENTS,
    REGISTRY as WHITEBOX_REGISTRY,
    VERSION as WHITEBOX_VERSION,
    extract_depth_tail_organic_consensus,
)
from spectral_utils.whitebox_layer_fusion import FeatureMatrix


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "whitebox_depth_tail_organic_consensus_v1"
VERSION = "whitebox-depth-tail-organic-hybrid-v1-2026-08-14"
OUTPUT_COMPONENT = "generation_entropy_mean"
COMPONENTS = WHITEBOX_COMPONENTS + (OUTPUT_COMPONENT,)
REGISTRY = {
    **WHITEBOX_REGISTRY,
    OUTPUT_COMPONENT: {
        "source": "raw output token_entropies",
        "formula": "mean token entropy over the generated answer",
        "role": "standard output-level uncertainty control added to the white-box consensus",
        "label_use": "none",
    },
}
SOURCE_FILES = (
    "scripts/whitebox_depth_tail_organic_consensus_experiment.py",
    "scripts/whitebox_depth_tail_consensus_experiment.py",
    "scripts/whitebox_depth_consensus_experiment.py",
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


def registry_hash() -> str:
    import json

    payload = json.dumps(REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_hybrid(cell: object) -> FeatureMatrix:
    whitebox = extract_depth_tail_organic_consensus(cell)
    by_id = {name: name for name in harness.CELLS}
    by_id.update({str(spec["cell_id"]): name for name, spec in harness.CELLS.items()})
    if whitebox.metadata.get("labels_seen_during_fit") is not False:
        raise RuntimeError("white-box extractor did not attest label-free fitting")
    try:
        cell_name = by_id[str(cell.cell_id)]
    except KeyError as exc:
        raise RuntimeError(f"unknown benchmark cell id: {cell.cell_id}") from exc
    rows_path = harness.BASE_RESULTS / "prepared" / f"{cell_name}__rows.npz"
    with np.load(rows_path, allow_pickle=False) as rows:
        expected_ids = tuple(str(value) for value in rows["row_ids"].tolist())
    if expected_ids != tuple(cell.row_ids):
        raise RuntimeError("raw-output baseline row order differs from validated sidecar order")
    output = harness.load_feature_matrix(
        harness.BASE_RESULTS / "prepared" / f"{cell_name}__raw_output_baselines.npz"
    )
    if output.n_samples != whitebox.n_samples or not np.allclose(
        output.risk_anchor, whitebox.risk_anchor, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("raw-output baseline does not share the white-box matrix and anchor contract")
    entropy_index = output.feature_names.index(OUTPUT_COMPONENT)
    entropy = np.asarray(output.values[:, entropy_index], dtype=float)
    values = np.column_stack([whitebox.values, entropy])
    return FeatureMatrix(
        values=values,
        feature_names=COMPONENTS,
        risk_anchor=np.asarray(whitebox.risk_anchor, dtype=float),
        groups=COMPONENTS,
        protocol_signature=hashlib.sha256(
            f"{whitebox.protocol_signature}:{VERSION}:{registry_hash()}:{output.protocol_signature}".encode()
        ).hexdigest(),
        metadata={
            "version": VERSION,
            "registry_sha256": registry_hash(),
            "analysis_role": "retrospective_discovery_candidate",
            "labels_seen_during_fit": False,
            "whitebox_metadata": whitebox.metadata,
            "output_contract": output.metadata,
            "output_protocol_signature": output.protocol_signature,
            "component_fits": {
                **dict(whitebox.metadata.get("component_fits", {})),
                OUTPUT_COMPONENT: {
                    "labels_seen_during_fit": False,
                    "source": "frozen raw-output-baselines FeatureMatrix",
                    "feature_name": OUTPUT_COMPONENT,
                },
            },
        },
    )


def configure_harness() -> None:
    harness.RESULTS = RESULTS
    harness.VERSION = VERSION
    harness.COMPONENTS = COMPONENTS
    harness.REGISTRY = REGISTRY
    harness.registry_hash = registry_hash
    harness.extract_depth_consensus = extract_hybrid
    harness.SOURCE_FILES = SOURCE_FILES
    harness.REPORT_TITLE = "White-box depth-tail organic consensus"
    harness.DISCOVERY_SELECTION_HISTORY = (
        "Selected retrospectively after the token-tail screen, the registered layer-organic "
        "grouping requested by the project, and a small audit of output/dependency additions "
        "on these same 13 cells. No cell is independent confirmation."
    )
    harness.DISCOVERY_METHOD_DESCRIPTION = (
        "The final nine-expert matrix contains seven depth-tail signals, one hierarchical "
        "layer-organic expert, and ordinary generation entropy. In the organic expert each "
        "transformer layer is a group whose three internal features are maximum target NLL, "
        "maximum top-1 surprisal, and entropy excess over top-1 surprisal. All orientation, "
        "selection, hierarchy fitting, and outer U-PCR fitting are label-free."
    )
    harness.DISPLAY = dict(harness.DISPLAY)
    harness.DISPLAY["upcr"] = "Depth-tail organic hybrid · deployed U-PCR"
    harness.DISPLAY["equal_mean"] = "Depth-tail organic hybrid · equal mean"
    harness.DISPLAY["iu_pcr"] = "Depth-tail organic hybrid · IU-PCR"
    harness.DISPLAY["dufs_liu_pcr"] = "Depth-tail organic hybrid · DUFS-LIU-PCR"
    harness.DISPLAY["best_single_layer"] = "Best single module/metric/layer · evaluation oracle"


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
        augment_external_baselines(args.results)
        harness.report(args.results)


if __name__ == "__main__":
    main()
