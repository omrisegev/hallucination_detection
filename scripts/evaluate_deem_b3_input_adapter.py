#!/usr/bin/env python3
"""Evaluate the frozen structured input-adapter seed-0 screen."""

from __future__ import annotations

from collections import defaultdict
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from spectral_utils.deem_b3_input_adapter import ARMS  # noqa: E402
from spectral_utils.residual_graph_deem import atomic_write_json, canonical_sha256, sha256_file  # noqa: E402
from spectral_utils.residual_graph_deem_data import load_registry, load_target_free_bundle  # noqa: E402


def main() -> None:
    run_dir = ROOT / "local_cache/deem_b3_moe_v1/input_adapter_seed0"
    ablation_dir = ROOT / "local_cache/deem_b3_moe_v1/contract_ablation_seed0"
    bundle_dir = ROOT / "local_cache/deem_b3_moe_v1/bundles"
    sidecar_dir = ROOT / "local_cache/deem_b3_moe_v1/label_sidecars"
    out_dir = ROOT / "local_cache/deem_b3_moe_v1/input_adapter_seed0_eval"; out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry(ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    definition = json.loads((run_dir / "RUN_DEFINITION.json").read_text()); freeze = json.loads((run_dir / "SCORE_FREEZE.json").read_text())
    for value in (definition, freeze): copy=dict(value); expected=copy.pop("content_sha256"); assert canonical_sha256(copy)==expected
    assert definition["source_hashes"]["runner"] == sha256_file(ROOT / "scripts/run_deem_b3_input_adapter.py")
    assert definition["source_hashes"]["adapter"] == sha256_file(ROOT / "spectral_utils/deem_b3_input_adapter.py")
    assert len(freeze["records"]) == 72 and not freeze["labels_accessed_during_fit"]
    records = {(row["cell_id"], row["arm"]): row for row in freeze["records"]}
    bundles, scores, health = {}, {}, []
    for row in registry["cells"]:
        cell=str(row["cell_id"]); bundle=load_target_free_bundle(bundle_dir/f"{cell}.npz"); bundles[cell]=bundle; scores[cell]={}
        with np.load(ablation_dir/"fits"/cell/"D1_TRANSFORM_ONLY__seed0.npz",allow_pickle=False) as data: scores[cell]["D1_BASE"]=np.asarray(data["score"])
        for arm in ARMS:
            record=records[(cell,arm)]; path=run_dir/record["npz"]; assert sha256_file(path)==record["npz_sha256"]
            meta=json.loads((run_dir/record["json"]).read_text()); copy=dict(meta); expected=copy.pop("content_sha256"); assert canonical_sha256(copy)==expected
            with np.load(path,allow_pickle=False) as data:
                assert tuple(str(x) for x in data["row_id"].tolist())==bundle.row_ids
                scores[cell][arm]=np.asarray(data["score"])
            health.append({"cell_id":cell,"arm":arm,**meta["health"],**meta["basis_diagnostics"]})
    pre={"schema":"input_adapter_pre_label","all_72_fits_verified":True,"labels_imported":False}; pre["content_sha256"]=canonical_sha256(pre); atomic_write_json(out_dir/"PRE_LABEL_FREEZE.json",pre)
    labels=importlib.import_module("spectral_utils.residual_graph_deem_labels")
    deltas={arm:defaultdict(list) for arm in ARMS}; per_cell=[]
    for cell,bundle in bundles.items():
        y=labels.join_labels_by_id(bundle,labels.load_label_sidecar(sidecar_dir/f"{cell}.npz")); base=float(roc_auc_score(y,scores[cell]["D1_BASE"])); record={"cell_id":cell,"dataset_family":bundle.dataset_family,"D1_BASE":base}
        for arm in ARMS:
            value=float(roc_auc_score(y,scores[cell][arm])); delta=value-base; record[arm]=value; record[arm+"_delta"]=delta; deltas[arm][bundle.dataset_family].append(delta)
        per_cell.append(record)
    summary={}
    for arm in ARMS:
        family={name:float(np.mean(values)) for name,values in deltas[arm].items()}; values=[row[arm+"_delta"] for row in per_cell]
        summary[arm]={
            "equal_family_delta_vs_D1":float(np.mean(list(family.values()))),"cell_macro_delta_vs_D1":float(np.mean(values)),
            "family_delta":family,"wins_ties_losses":[int(sum(x>.0005 for x in values)),int(sum(abs(x)<=.0005 for x in values)),int(sum(x<-.0005 for x in values))],
            "worst_cell":float(min(values)),"mean_input_correction_sd":float(np.mean([row["input_correction_sd"] for row in health if row["arm"]==arm])),
            "mean_offblock_covariance_fraction":float(np.mean([row["offblock_covariance_fraction"] for row in health if row["arm"]==arm])),
        }
    atomic_write_json(out_dir/"SUMMARY.json",summary); atomic_write_json(out_dir/"PER_CELL.json",per_cell); atomic_write_json(out_dir/"HEALTH.json",health)
    print(json.dumps(summary,indent=2,sort_keys=True))


if __name__=="__main__": main()
