"""Inventory existing L-SML/PRMScore rows; no scoring or fitting."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
W = ROOT / ".worktrees"
DIRS = {
    "cumulative": W/"cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2",
    "ct7_profiles": W/"cumulative-vote-fusion-v2/results/cumulative_vote_fusion_v2/ct7_profiles_v1",
    "readout": W/"readout-quickest-detection-v1/results/readout_family_v1",
    "evidence": W/"readout-quickest-detection-v1/results/step_evidence_v1",
    "ct7_token": W/"lsml-ct7-levers-run/results/ct7_token_lsml_v1",
    "window": W/"lsml-ct7-levers-run/results/window_representation_b3_v1/fusion",
}


def main():
    rows, hashes = [], {}
    def read(p):
        data = p.read_bytes(); hashes[str(p.relative_to(ROOT)).replace("\\", "/")] = hashlib.sha256(data).hexdigest()
        return json.loads(data.decode("utf8"))
    for bundle,p in DIRS.items():
        score = read(p/"PRMSCORE.json")
        mf = next((p/f for f in ("PRM_RANKING_METRICS.json","PRM_METRICS.json") if (p/f).exists()),p/"RESULTS.json")
        metrics = read(mf)
        if mf.name == "RESULTS.json": metrics = metrics["prm"]
        uf = p/("CT7_CONTRASTS.json" if bundle == "ct7_profiles" else "UNCERTAINTY.json")
        contrasts = read(uf)["contrasts"]
        for name,result in score.items():
            if "lsml" not in name and not name.startswith("T_C"): continue
            equal = name.replace("continuous_lsml","equal").replace("binary_lsml","equal").replace("token_lsml","token_equal").replace("T_C","T_E").replace("window_lsml","window_equal")
            if bundle == "evidence" and name.startswith("evidence"): equal += "_std"
            def value(m,c): return score.get(m,{}).get(c,{}).get("prmscore")
            auc,eqauc = metrics.get(name,{}).get("within_auc"),metrics.get(equal,{}).get("within_auc")
            rows.append({"bundle":bundle,"directory":str(p.relative_to(ROOT)).replace("\\","/"),"method":name,
                         "same_bank_equal":equal if equal in metrics else None,
                         "answers":metrics.get(name,{}).get("answers"),"eligible":metrics.get(name,{}).get("eligible"),
                         "within_auc":auc,"equal_within_auc":eqauc,"delta_within_auc":None if auc is None or eqauc is None else auc-eqauc,
                         "q80":value(name,"quantile_0.8"),"equal_q80":value(equal,"quantile_0.8"),
                         "inner":value(name,"inner_selected"),"equal_inner":value(equal,"inner_selected"),
                         "uses_truth_for_readout_selection":name.startswith("selected"),
                         "is_shuffle_control":"shuffle" in name,
                         "paired_within_auc":[c for c in contrasts if c["endpoint"]=="prm_within_auc" and c["a"]==name and c["b"] in (equal,equal.removesuffix("_std"))]})
    assert len(rows)==67
    out=ROOT/"results/prmbench_lsml_month_inventory_v1"
    out.mkdir(parents=True,exist_ok=False)
    payload={"scope":"67 saved L-SML rows in six unique recent result bundles; includes inherited duplicates/shuffle controls, not 67 independent experiments",
             "rows":sorted(rows,key=lambda r:r["q80"] or 0,reverse=True),"source_sha256":hashes}
    (out/"INVENTORY.json").write_text(json.dumps(payload,indent=2),encoding="utf8")
    print(f"Indexed {len(rows)} rows from {len(DIRS)} bundles; preserved exact source hashes.")


if __name__ == "__main__": main()
