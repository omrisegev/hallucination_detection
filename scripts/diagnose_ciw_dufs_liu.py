#!/usr/bin/env python3
"""Target-free CIW input comparison for ordinary IU-PCR and DUFS-LIU."""

from __future__ import annotations

from collections import defaultdict
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.adapted_dufs import adapted_dufs_soft_gates
from spectral_utils.deem_b3_contract_ablation import prepare_arm
from spectral_utils.deem_b3_unsupervised_input_gate import build_gate_map
from spectral_utils.laplacian_upcr import (
    build_graph_from_features,
    graph_diagnostics,
    laplacian_iu_path,
)
from spectral_utils.residual_graph_deem import (
    atomic_write_json,
    canonical_sha256,
    sha256_file,
)
from spectral_utils.residual_graph_deem_data import (
    load_registry,
    load_target_free_bundle,
)


DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
DUFS_K = 7
LIU_LAMBDA = 0.1
REPRESENTATIONS = ("D1_INPUT", "CIW_INPUT")
METHODS = ("IU_PCR", "DUFS_LIU")


def ciw_transform(X: np.ndarray, gate_map) -> np.ndarray:
    prediction = X @ gate_map.prediction_matrix.T
    innovation = (X - prediction) / gate_map.innovation_scale
    gate = 0.5 * gate_map.reliability
    mask = np.zeros_like(X)
    mask[:, gate_map.core_indices] = gate[gate_map.core_indices]
    return (1.0 - mask) * X + mask * innovation


def orient_scores(
    iu_score: np.ndarray,
    liu_score: np.ndarray,
    anchor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    correlation = float(spearmanr(iu_score, anchor).statistic)
    if not np.isfinite(correlation) or abs(correlation) <= 1e-6:
        raise RuntimeError("IU-PCR risk orientation is ambiguous")
    orientation = 1.0 if correlation > 0 else -1.0
    return orientation * iu_score, orientation * liu_score, abs(correlation)


def family_risk_anchor(X: np.ndarray, prepared) -> np.ndarray:
    family_means = []
    for indices in prepared.groups.values():
        usable = [
            index for index in indices
            if prepared.feature_names[index] not in prepared.anchor_exclusions
        ]
        if usable:
            family_means.append(X[:, usable].mean(axis=1))
    if not family_means:
        raise RuntimeError("no family coordinates available for orientation")
    return np.mean(family_means, axis=0)


def fit_representation(X: np.ndarray, prepared) -> tuple[dict, dict]:
    F = np.asarray(X, dtype=np.float64).T
    gates, gate_diag = adapted_dufs_soft_gates(
        F, seeds=DUFS_SEEDS, epochs=DUFS_EPOCHS
    )
    graph = build_graph_from_features(F, gates=gates, k=DUFS_K)
    path = laplacian_iu_path(F, (0.0, LIU_LAMBDA), graph=graph)
    iu_score = np.asarray(path[0.0].w @ F, dtype=np.float64)
    liu_score = np.asarray(path[LIU_LAMBDA].w @ F, dtype=np.float64)
    anchor = family_risk_anchor(X, prepared)
    iu_score, liu_score, anchor_rho = orient_scores(iu_score, liu_score, anchor)
    diagnostics = {
        "anchor_abs_spearman": anchor_rho,
        "dufs_mean_probability": float(gate_diag["mean_probability"]),
        "dufs_effective_feature_count": float(
            gate_diag["effective_feature_count"]
        ),
        "dufs_mean_seed_std": float(gate_diag["mean_seed_std"]),
        "graph": graph_diagnostics(graph),
        "liu_weight_cosine_vs_iu": float(
            path[LIU_LAMBDA].diagnostics["weight_cosine_vs_iu"]
        ),
        "iu_score_sd": float(np.std(iu_score)),
        "liu_score_sd": float(np.std(liu_score)),
    }
    return {"IU_PCR": iu_score, "DUFS_LIU": liu_score}, diagnostics


def aggregate(rows: list[dict], representation: str, method: str) -> dict:
    selected = [
        row for row in rows
        if row["representation"] == representation and row["method"] == method
    ]
    by_family_auroc: dict[str, list[float]] = defaultdict(list)
    by_family_auprc: dict[str, list[float]] = defaultdict(list)
    for row in selected:
        by_family_auroc[row["dataset_family"]].append(row["auroc"])
        by_family_auprc[row["dataset_family"]].append(row["auprc"])
    return {
        "cell_macro_auroc": float(np.mean([row["auroc"] for row in selected])),
        "cell_macro_auprc": float(np.mean([row["auprc"] for row in selected])),
        "equal_family_auroc": float(np.mean([
            np.mean(values) for values in by_family_auroc.values()
        ])),
        "equal_family_auprc": float(np.mean([
            np.mean(values) for values in by_family_auprc.values()
        ])),
        "family_auroc": {
            family: float(np.mean(values))
            for family, values in sorted(by_family_auroc.items())
        },
    }


def main() -> None:
    bundle_dir = ROOT / "local_cache/deem_b3_moe_v1/bundles"
    sidecar_dir = ROOT / "local_cache/deem_b3_moe_v1/label_sidecars"
    out_dir = ROOT / "local_cache/deem_b3_moe_v1/ciw_dufs_liu_v1"
    score_dir = out_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(
        ROOT / "configs/residual_graph_deem_24cell_v1_registry.json"
    )

    cells = {}
    frozen_files = []
    for registered in registry["cells"]:
        cell_id = str(registered["cell_id"])
        bundle = load_target_free_bundle(bundle_dir / f"{cell_id}.npz")
        prepared = prepare_arm(
            bundle.X_raw, bundle.feature_names, "D1_TRANSFORM_ONLY"
        )
        gate_map = build_gate_map(
            prepared,
            bundle.group_ids,
            bundle.raw_trace_length,
            "M1_ROOK_STATIC_R2",
        )
        matrices = {
            "D1_INPUT": prepared.X_risk,
            "CIW_INPUT": ciw_transform(prepared.X_risk, gate_map),
        }
        arrays = {"row_ids": np.asarray(bundle.row_ids)}
        diagnostics = {}
        for representation, matrix in matrices.items():
            scores, representation_diag = fit_representation(
                matrix, prepared
            )
            diagnostics[representation] = representation_diag
            for method, score in scores.items():
                arrays[f"{representation}__{method}"] = score
        score_path = score_dir / f"{cell_id}.npz"
        np.savez_compressed(score_path, **arrays)
        metadata = {
            "cell_id": cell_id,
            "dataset_family": bundle.dataset_family,
            "bundle_sha256": bundle.bundle_sha256,
            "ordered_row_id_sha256": canonical_sha256(list(bundle.row_ids)),
            "diagnostics": diagnostics,
            "labels_loaded_during_fit": False,
        }
        metadata_path = score_dir / f"{cell_id}.json"
        atomic_write_json(metadata_path, metadata)
        frozen_files.extend([score_path, metadata_path])
        cells[cell_id] = bundle

    freeze = {
        "schema": "ciw_dufs_liu_score_freeze_v1",
        "source_sha256": sha256_file(Path(__file__)),
        "representations": list(REPRESENTATIONS),
        "methods": list(METHODS),
        "settings": {
            "dufs_seeds": list(DUFS_SEEDS),
            "dufs_epochs": DUFS_EPOCHS,
            "dufs_k": DUFS_K,
            "liu_lambda": LIU_LAMBDA,
        },
        "files": {
            str(path.relative_to(out_dir)): sha256_file(path)
            for path in sorted(frozen_files)
        },
        "labels_loaded": False,
    }
    freeze["content_sha256"] = canonical_sha256(freeze)
    atomic_write_json(out_dir / "SCORE_FREEZE.json", freeze)

    labels = importlib.import_module("spectral_utils.residual_graph_deem_labels")
    rows = []
    for cell_id, bundle in cells.items():
        y = labels.join_labels_by_id(
            bundle, labels.load_label_sidecar(sidecar_dir / f"{cell_id}.npz")
        ).astype(int)
        with np.load(score_dir / f"{cell_id}.npz", allow_pickle=False) as data:
            for representation in REPRESENTATIONS:
                for method in METHODS:
                    score = np.asarray(
                        data[f"{representation}__{method}"], dtype=float
                    )
                    rows.append({
                        "cell_id": cell_id,
                        "dataset_family": bundle.dataset_family,
                        "representation": representation,
                        "method": method,
                        "auroc": float(roc_auc_score(y, score)),
                        "auprc": float(average_precision_score(y, score)),
                    })

    summary = {
        f"{representation}__{method}": aggregate(rows, representation, method)
        for representation in REPRESENTATIONS
        for method in METHODS
    }
    for representation in REPRESENTATIONS:
        liu = summary[f"{representation}__DUFS_LIU"]
        iu = summary[f"{representation}__IU_PCR"]
        summary[f"{representation}__DUFS_MINUS_IU"] = {
            metric: liu[metric] - iu[metric]
            for metric in (
                "cell_macro_auroc",
                "cell_macro_auprc",
                "equal_family_auroc",
                "equal_family_auprc",
            )
        }
    for method in METHODS:
        ciw = summary[f"CIW_INPUT__{method}"]
        d1 = summary[f"D1_INPUT__{method}"]
        summary[f"CIW_MINUS_D1__{method}"] = {
            metric: ciw[metric] - d1[metric]
            for metric in (
                "cell_macro_auroc",
                "cell_macro_auprc",
                "equal_family_auroc",
                "equal_family_auprc",
            )
        }
    atomic_write_json(out_dir / "PER_CELL.json", rows)
    atomic_write_json(out_dir / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
