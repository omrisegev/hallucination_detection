#!/usr/bin/env python3
"""Formal benchmark for the final retrospective distributed-depth candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

import scripts.whitebox_depth_consensus_experiment as harness
from scripts.whitebox_depth_tail_consensus_experiment import augment_external_baselines
from spectral_utils.whitebox_depth_distributed_consensus import (
    COMPONENTS as WHITEBOX_COMPONENTS,
    REGISTRY as WHITEBOX_REGISTRY,
    extract_depth_distributed_consensus,
)
from spectral_utils.whitebox_depth_metrics import extract_prediction_revision
from spectral_utils.whitebox_depth_token_metrics import extract_resid_entropy_burst
from spectral_utils.whitebox_layer_fusion import FeatureMatrix, extract_lens_grid


REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results" / "whitebox_depth_distributed_consensus_v1"
VERSION = "whitebox-depth-distributed-hybrid-v1-2026-08-14"
OUTPUT_COMPONENT = "generation_entropy_mean"
COMPONENTS = WHITEBOX_COMPONENTS + (OUTPUT_COMPONENT,)
REGISTRY = {
    **WHITEBOX_REGISTRY,
    OUTPUT_COMPONENT: {
        "source": "raw output token_entropies",
        "formula": "mean token entropy over the generated answer",
        "role": "standard output-level uncertainty control",
        "label_use": "none",
    },
}
SOURCE_FILES = (
    "scripts/whitebox_depth_distributed_consensus_experiment.py",
    "scripts/whitebox_depth_tail_consensus_experiment.py",
    "scripts/whitebox_depth_consensus_experiment.py",
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
ACTIVE_CACHE = harness.DEFAULT_CACHE


def registry_hash() -> str:
    payload = json.dumps(REGISTRY, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def extract_final_matrix(cell: Any) -> FeatureMatrix:
    whitebox = extract_depth_distributed_consensus(cell)
    by_id = {name: name for name in harness.CELLS}
    by_id.update({str(spec["cell_id"]): name for name, spec in harness.CELLS.items()})
    try:
        cell_name = by_id[str(cell.cell_id)]
    except KeyError as exc:
        raise RuntimeError(f"unknown benchmark cell id: {cell.cell_id}") from exc
    with np.load(
        harness.BASE_RESULTS / "prepared" / f"{cell_name}__rows.npz", allow_pickle=False
    ) as rows:
        expected_ids = tuple(str(value) for value in rows["row_ids"].tolist())
    if expected_ids != tuple(cell.row_ids):
        raise RuntimeError("generation-entropy row order differs from the validated sidecar order")
    output = harness.load_feature_matrix(
        harness.BASE_RESULTS / "prepared" / f"{cell_name}__raw_output_baselines.npz"
    )
    if output.n_samples != whitebox.n_samples or not np.allclose(
        output.risk_anchor, whitebox.risk_anchor, rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("output entropy does not share the white-box row/anchor contract")
    entropy = np.asarray(
        output.values[:, output.feature_names.index(OUTPUT_COMPONENT)], dtype=float
    )
    return FeatureMatrix(
        values=np.column_stack([whitebox.values, entropy]),
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
            "component_fits": {
                **dict(whitebox.metadata.get("component_fits", {})),
                OUTPUT_COMPONENT: {
                    "labels_seen_during_fit": False,
                    "source": "frozen raw-output-baselines FeatureMatrix",
                },
            },
        },
    )


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def strongest_atomic_layer_view(
    cell_name: str, y: np.ndarray
) -> tuple[np.ndarray, str, float, float]:
    """Evaluation-only oracle over every atomic layer view used by the candidate.

    View direction is still anchored without labels.  Labels select only the
    best view inside this deliberately optimistic diagnostic comparator.
    """

    spec = harness.CELLS[cell_name]
    raw = _load_pickle(ACTIVE_CACHE / spec["raw"])
    sidecar = _load_pickle(ACTIVE_CACHE / spec["sidecar"])
    joined, _audit = harness.validate_and_join(
        raw,
        sidecar,
        cell_id=cell_name,
        expected_model=spec.get("model", "meta-llama/Llama-3.1-8B-Instruct"),
        expected_n_layers=int(spec.get("n_layers", 32)),
        expected_hidden_size=int(spec.get("hidden_size", 4096)),
        exclude_invalid=True,
        require_geometry_finite=False,
    )
    if len(y) != joined.n_samples:
        raise RuntimeError("oracle labels do not align to the validated cell")
    lens = extract_lens_grid(joined)
    revision = extract_prediction_revision(joined)
    burst = extract_resid_entropy_burst(joined)
    matrices = [lens.values, revision.values, burst.values]
    names = list(lens.feature_names) + list(revision.feature_names) + list(burst.feature_names)
    families = {
        "max_entropy": [],
        "max_target_nll": [],
        "max_top1_surprisal": [],
        "max_entropy_excess_top1": [],
        "max_target_gap": [],
        "mean_entropy_excess_top1": [],
    }
    for record in joined.records:
        entropy = np.asarray(record["lens_H"], dtype=np.float32)
        top1 = -np.asarray(record["lens_logp_top1"], dtype=np.float32)
        target = -np.asarray(record["lens_logp_tgt"], dtype=np.float32)
        families["max_entropy"].append(np.max(entropy, axis=-1))
        families["max_target_nll"].append(np.max(target, axis=-1))
        families["max_top1_surprisal"].append(np.max(top1, axis=-1))
        families["max_entropy_excess_top1"].append(np.max(entropy - top1, axis=-1))
        families["max_target_gap"].append(np.max(target - top1, axis=-1))
        families["mean_entropy_excess_top1"].append(np.mean(entropy - top1, axis=-1))
    for family, raw_values in families.items():
        values = np.asarray(raw_values, dtype=np.float32)
        matrices.extend([values.reshape(joined.n_samples, -1), values.mean(axis=1)])
        names.extend(
            f"{family}.module_{module}.layer_{layer:02d}"
            for module in range(3)
            for layer in range(joined.n_layers)
        )
        names.extend(f"{family}.module_mean.layer_{layer:02d}" for layer in range(joined.n_layers))
    values = np.column_stack(matrices)
    best: tuple[float, str, np.ndarray, float] = (-np.inf, "", np.empty(0), float("nan"))
    for index, name in enumerate(names):
        score = np.asarray(values[:, index], dtype=float)
        correlation = float(spearmanr(score, lens.risk_anchor).statistic)
        if correlation < 0.0:
            score = -score
        auroc, auprc = harness.metric_pair(y, score)
        if auroc > best[0]:
            best = (auroc, name, score, auprc)
    del raw, sidecar, joined, matrices, values
    return np.asarray(best[2]), str(best[1]), float(best[0]), float(best[3])


def configure_harness() -> None:
    harness.RESULTS = RESULTS
    harness.VERSION = VERSION
    harness.COMPONENTS = COMPONENTS
    harness.REGISTRY = REGISTRY
    harness.registry_hash = registry_hash
    harness.extract_depth_consensus = extract_final_matrix
    harness.best_single_layer = strongest_atomic_layer_view
    harness.SOURCE_FILES = SOURCE_FILES
    harness.REPORT_TITLE = "White-box distributed-depth consensus"
    harness.DISCOVERY_SELECTION_HISTORY = (
        "Final retrospective candidate selected after the token-tail, layer-organic, lens-96, "
        "and compact combination screens on these same 13 cells. The atomic-view oracle was "
        "then expanded to cover every raw layer readout used by the candidate. No cell is an "
        "independent confirmation set."
    )
    harness.DISCOVERY_METHOD_DESCRIPTION = (
        "The outer U-PCR matrix has thirteen label-free experts: twelve white-box summaries "
        "plus generation entropy. Its white-box evidence includes depth-band tail consensus, "
        "a real per-layer organic hierarchy, a lens-96 hierarchical DUFS expert, all-layer "
        "maximum target NLL, a four-band target gap, and an eight-layer entropy-excess view. "
        "The comparator is the strongest evaluation-selected atomic module/metric/layer view "
        "across every raw readout used here."
    )
    harness.DISPLAY = dict(harness.DISPLAY)
    harness.DISPLAY["upcr"] = "Distributed-depth hybrid · deployed U-PCR"
    harness.DISPLAY["equal_mean"] = "Distributed-depth hybrid · equal mean"
    harness.DISPLAY["iu_pcr"] = "Distributed-depth hybrid · IU-PCR"
    harness.DISPLAY["dufs_liu_pcr"] = "Distributed-depth hybrid · DUFS-LIU-PCR"
    harness.DISPLAY["best_single_layer"] = "Strongest atomic internal layer/view · evaluation oracle"


def main() -> None:
    global ACTIVE_CACHE

    configure_harness()
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "fit", "evaluate", "report", "all"), nargs="?", default="all")
    parser.add_argument("--cache", type=Path, default=harness.DEFAULT_CACHE)
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    ACTIVE_CACHE = args.cache
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
