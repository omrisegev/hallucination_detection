#!/usr/bin/env python3
"""Evaluate frozen seed-0 B3 contract ablations after strict preflight."""

from __future__ import annotations

from collections import defaultdict
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_contract_ablation import ARMS  # noqa: E402
from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256, sha256_file  # noqa: E402
from spectral_utils.residual_graph_deem_data import load_registry, load_target_free_bundle  # noqa: E402


def main() -> None:
    run_dir = ROOT / "local_cache/deem_b3_moe_v1/contract_ablation_seed0"
    baseline_dir = ROOT / "local_cache/deem_b3_moe_v1/b3_frozen"
    bundle_dir = ROOT / "local_cache/deem_b3_moe_v1/bundles"
    sidecar_dir = ROOT / "local_cache/deem_b3_moe_v1/label_sidecars"
    out_dir = ROOT / "local_cache/deem_b3_moe_v1/contract_ablation_seed0_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    definition = json.loads((run_dir / "RUN_DEFINITION.json").read_text())
    freeze = json.loads((run_dir / "SCORE_FREEZE.json").read_text())
    for value in (definition, freeze):
        copy = dict(value); expected = copy.pop("content_sha256"); assert canonical_sha256(copy) == expected
    assert definition["sources"]["runner"] == sha256_file(ROOT / "scripts/run_deem_b3_contract_ablation.py")
    assert definition["sources"]["core"] == sha256_file(ROOT / "spectral_utils/deem_b3_contract_ablation.py")
    assert len(freeze["records"]) == 72 and not freeze["labels_accessed_during_fit"]
    record_map = {(row["cell_id"], row["arm"]): row for row in freeze["records"]}
    bundles, scores = {}, {}
    for cell_row in registry["cells"]:
        cell = str(cell_row["cell_id"]); bundle = load_target_free_bundle(bundle_dir / f"{cell}.npz")
        bundles[cell], scores[cell] = bundle, {}
        with np.load(baseline_dir / "fits" / cell / "B3__seed0.npz", allow_pickle=False) as data:
            scores[cell]["B3"] = np.asarray(data["score"], dtype=float)
        for arm in ARMS:
            record = record_map[(cell, arm)]; path = run_dir / record["npz"]
            assert sha256_file(path) == record["npz_sha256"]
            with np.load(path, allow_pickle=False) as data:
                assert tuple(str(x) for x in data["row_id"].tolist()) == bundle.row_ids
                scores[cell][arm] = np.asarray(data["score"], dtype=float)
    pre = {"schema": "contract_ablation_pre_label", "all_72_fits_verified": True, "labels_imported": False}
    pre["content_sha256"] = canonical_sha256(pre); atomic_write_json(out_dir / "PRE_LABEL_FREEZE.json", pre)
    labels = importlib.import_module("spectral_utils.residual_graph_deem_labels")
    deltas: dict[str, dict[str, list[float]]] = {arm: defaultdict(list) for arm in ARMS}
    cells = []
    for cell, bundle in bundles.items():
        y = labels.join_labels_by_id(bundle, labels.load_label_sidecar(sidecar_dir / f"{cell}.npz"))
        base = float(roc_auc_score(y, scores[cell]["B3"]))
        row = {"cell_id": cell, "dataset_family": bundle.dataset_family, "B3": base}
        for arm in ARMS:
            value = float(roc_auc_score(y, scores[cell][arm])); delta = value - base
            row[arm] = value; row[arm + "_delta"] = delta; deltas[arm][bundle.dataset_family].append(delta)
        cells.append(row)
    summary = {}
    for arm in ARMS:
        family = {name: float(np.mean(values)) for name, values in deltas[arm].items()}
        cell_values = [row[arm + "_delta"] for row in cells]
        summary[arm] = {
            "equal_family_delta": float(np.mean(list(family.values()))),
            "cell_macro_delta": float(np.mean(cell_values)),
            "family_delta": family,
            "wins_ties_losses": [
                int(sum(x > 0.0005 for x in cell_values)),
                int(sum(abs(x) <= 0.0005 for x in cell_values)),
                int(sum(x < -0.0005 for x in cell_values)),
            ],
            "worst_cell": float(min(cell_values)),
        }
    atomic_write_json(out_dir / "SUMMARY.json", summary)
    atomic_write_json(out_dir / "PER_CELL.json", cells)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
