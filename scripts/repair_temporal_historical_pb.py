"""Recompute all 47 historical PB bundles into a new, source-preserving release.

This replays PB scoring/fitting only. Historical PRMB fields are retained with
their previous provenance, not claimed as independently recomputed here.
"""
from __future__ import annotations
import argparse
import gc
import io
import json
from pathlib import Path
import sqlite3
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from scripts import run_temporal_research_baseline as baseline
from scripts import run_renyi_position_temporal_fusion as common
from scripts import run_fusion_input_normalization_ablation_v1 as normalization
from spectral_utils import renyi_position_fusion as position
from spectral_utils import renyi_locator_feature_bank as banks
from spectral_utils.pb_prediction_bundle import prediction_bundle
from spectral_utils.math_gate_selection import percentile_by_cell


def prepare(source, records, con, out):
    done = {r[0] for r in con.execute("SELECT idx FROM features")}
    for cell, path, kind, dataset in baseline.evaluator.source_specs():
        if kind != "pb": continue
        ids = [i for i,r in enumerate(records) if r["cell"] == cell and i not in done]
        if not ids: continue
        rows = baseline.evaluator.old._source_row_map(baseline.evaluator.old.load_pickle(path), kind=kind, dataset=dataset)
        print("[prepare]",cell,len(ids),flush=True)
        for i in ids:
            row = rows[records[i]["row_id"]]
            lp = np.asarray(baseline.evaluator.old._topk_payload(row)["logprobs"],float)
            f = position.feature_bank(lp)
            allf = banks.feature_matrix(lp,np.asarray(row["token_entropies"],float))
            spans=np.asarray(row["step_token_spans"],int)
            blob=common.packed(z=f["z"],mean=f["mean"],scale=f["scale"],spans=spans,
                               matrix=allf["matrix"],anchor=allf["anchor"])
            con.execute("INSERT INTO features VALUES (?,?)",(i,blob))
            if (i+1)%100==0:con.commit()
        con.commit();del rows;gc.collect()


def run(source, baseout, out):
    if baseline.read_json(baseout/"BASELINE_REPLAY.json")["status"] != "PASS":
        raise ValueError("independent baseline replay must pass first")
    # Re-verify bytes before this separate historical model replay.
    audit=baseline.audit_inputs(source,out)
    records,joined=baseline.load_contract(source)
    cells=np.array([r["cell"] for r in records]);pb=np.char.startswith(cells,"pb_")
    ids=np.flatnonzero(pb);pb_records=[records[i] for i in ids]
    folds=baseline.read_json(baseline.evaluator.old.FOLDS)["outer"]
    fold={i:int(folds[r["group_id"]]) for i,r in enumerate(records)}
    metadata=common.training_metadata(records,fold)
    with np.load(baseout/"SCORES_FROZEN.npz",allow_pickle=False) as f:
        gate_raw=f["gate_raw"]
    opened=percentile_by_cell(gate_raw[pb],cells[pb])>=.33
    code_paths=[Path(__file__),ROOT/"spectral_utils/renyi_position_fusion.py",ROOT/"spectral_utils/renyi_locator_feature_bank.py"]
    manifest=dict(schema="pb-report-repair-v2",inputs=audit,baseline_freeze=baseline.read_json(baseout/"SCORE_FREEZE.json"),
                  code={str(p.relative_to(ROOT)):common.sha256_file(p) for p in code_paths},
                  scope="PB independent replay; prior PRMB fields retained with historical provenance")
    if (out/"MANIFEST.json").exists() and baseline.read_json(out/"MANIFEST.json")!=manifest:
        raise ValueError("repair checkpoint manifest drift")
    common.atomic_json(out/"MANIFEST.json",manifest)
    con=sqlite3.connect(out/"FEATURES.sqlite")
    con.execute("CREATE TABLE IF NOT EXISTS features(idx INTEGER PRIMARY KEY,payload BLOB NOT NULL)")
    con.execute("CREATE TABLE IF NOT EXISTS scored(stage TEXT,idx INTEGER,peaks TEXT,PRIMARY KEY(stage,idx))")
    try:
        prepare(source,records,con,out)
        new={}
        sources={
            "fusion_input_normalization_ablation_v1":baseline.read_json(ROOT/"results/fusion_input_normalization_ablation_v1/METRICS.json"),
            "renyi_locator_feature_bank_v1":baseline.read_json(ROOT/"results/renyi_locator_feature_bank_v1/METRICS.json"),
            "renyi_locator_integrated_replay_v1":baseline.read_json(ROOT/"results/renyi_locator_integrated_replay_v1/METRICS.json"),
        }
        # Full original method rosters and exact original functions; no new tuning.
        all_peaks={}
        for transform in normalization.TRANSFORMS:
            cache=out/("PEAKS_"+transform+".npz")
            if cache.exists():
                with np.load(cache,allow_pickle=False) as f: all_peaks.update({k:f[k] for k in f.files})
                continue
            priors={}
            for cell in sorted(set(cells[pb])):
                cell_ids=[int(i) for i in ids if cells[i]==cell]
                statistics={i:position.regional_statistics(normalization.transform_bank(normalization.read_feature(con,i),transform),records[i]["uid"]) for i in cell_ids}
                for excluded in common.excluded_sets(metadata,cell_ids,cell):
                    value,_=common.fit_prior(statistics,metadata,cell,excluded)
                    priors[common.model_key(cell,excluded)]=value
                print("[prior]",transform,cell,flush=True)
            peaks={transform+"__"+s:np.full(len(ids),-1,int) for s in normalization.SOLVERS}
            completed={int(i):json.loads(p) for i,p in con.execute("SELECT idx,peaks FROM scored WHERE stage=?",(transform,))}
            for j,i in enumerate(ids):
                if int(i) in completed:
                    for name,value in completed[int(i)].items(): peaks[name][j]=value
                    continue
                arrays=normalization.read_feature(con,int(i))
                bank=normalization.transform_bank(arrays,transform)
                values,health,_=position.score_answer({"z":bank,"singles":{}},arrays["spans"],
                    priors[common.model_key(records[i]["cell"],(fold[i],))],records[i]["uid"],methods=normalization.SOLVERS)
                for solver in normalization.SOLVERS:
                    if health[solver]["status"]!="OK" or not np.isfinite(values[solver]).all():
                        raise ValueError(f"historical fit failed: {i} {transform} {solver}")
                    peaks[transform+"__"+solver][j]=int(np.argmax(values[solver]))
                con.execute("INSERT INTO scored VALUES (?,?,?)",(transform,int(i),json.dumps({n:int(v[j]) for n,v in peaks.items()})))
                if (j+1)%50==0 or baseline.STOP:con.commit()
                if baseline.STOP:raise InterruptedError("checkpoint saved on termination")
                if (j+1)%250==0:
                    print("[norm]",transform,j+1,len(ids),flush=True)
                    common.atomic_json(out/"RUN_STATE.json",dict(status="REPLAYING",stage=transform,completed=j+1,expected=len(ids)))
            con.commit()
            with cache.with_suffix(".tmp").open("wb") as f:np.savez_compressed(f,**peaks)
            cache.with_suffix(".tmp").replace(cache);all_peaks.update(peaks)
        bank_cache=out/"PEAKS_banks.npz"
        if bank_cache.exists():
            with np.load(bank_cache,allow_pickle=False) as f: bank_peaks={k:f[k] for k in f.files}
        else:
            bank_peaks={m:np.full(len(ids),-1,int) for m in banks.METHODS}
            completed={int(i):json.loads(p) for i,p in con.execute("SELECT idx,peaks FROM scored WHERE stage='banks'")}
            for j,i in enumerate(ids):
                if int(i) in completed:
                    for name,value in completed[int(i)].items():bank_peaks[name][j]=value
                    continue
                arrays=normalization.read_feature(con,int(i))
                for bank in banks.BANKS:
                    for solver in banks.SOLVERS:
                        scores,_=banks.score_bank(arrays,arrays["spans"],bank,solver)
                        bank_peaks[bank.name+"__"+solver][j]=int(np.argmax(scores))
                con.execute("INSERT INTO scored VALUES (?,?,?)",("banks",int(i),json.dumps({n:int(v[j]) for n,v in bank_peaks.items()})))
                if (j+1)%50==0 or baseline.STOP:con.commit()
                if baseline.STOP:raise InterruptedError("checkpoint saved on termination")
                if (j+1)%500==0: print("[banks]",j+1,len(ids),flush=True)
            con.commit()
            with bank_cache.with_suffix(".tmp").open("wb") as f:np.savez_compressed(f,**bank_peaks)
            bank_cache.with_suffix(".tmp").replace(bank_cache)
        all_peaks.update(bank_peaks)
        ledger=[]
        for experiment,original in sources.items():
            result=json.loads(json.dumps(original))
            for name,previous in original["metrics"].items():
                peaks=all_peaks[name]
                bundle,pred,valid=prediction_bundle(joined["target"][pb],cells[pb],peaks,np.ones(len(ids),bool),opened)
                delta=bundle["pb_all8"]-previous["pb_all8"]
                corrected={**previous,**bundle}
                result["metrics"][name]=corrected
                ledger.append(dict(experiment=experiment,method=name,headline_delta=delta,
                    changed_fields=[k for k,v in bundle.items() if previous.get(k)!=v],
                    old_suppressed=previous.get("pb_correct_peaks_suppressed"),
                    corrected_suppressed=bundle["pb_correct_peaks_suppressed"]))
            result["pb_correction_v2"]={"source_metrics_sha256":common.sha256_file(ROOT/"results"/experiment/"METRICS.json"),
                "pb_from_independent_raw_replay":True,"prmb_independently_recomputed":False,"original_preserved":True}
            common.atomic_json(out/experiment/"METRICS.json",result)
        if len(ledger)!=47: raise ValueError("expected exactly 47 repaired bundles")
        changed=[row for row in ledger if abs(row["headline_delta"])>1e-12]
        common.atomic_json(out/"REPAIR_REVIEW.json",dict(status="HEADLINE_DRIFT" if changed else "PASS",count=len(ledger),
            changed_headlines=changed,ledger=ledger,originals_preserved=True,
            interpretation="A reporting repair is confirmed only when independently recomputed headlines agree."))
        common.atomic_json(out/"RUN_STATE.json",dict(status="COMPLETE_REVIEWED" if not changed else "NEEDS_INVESTIGATION",corrected=len(ledger)))
    finally: con.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root",type=Path,required=True)
    p.add_argument("--baseline",type=Path,default=ROOT/"results/temporal_research_baseline_v1")
    p.add_argument("--out",type=Path,default=ROOT/"results/temporal_historical_pb_repair_v2")
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    signal.signal(signal.SIGTERM,baseline.stop_handler)
    try:
        with threadpool_limits(limits=1):run(a.source_root,a.baseline,a.out)
    except BaseException as e:
        common.atomic_json(a.out/"RUN_STATE.json",dict(status="FAILED",error=f"{type(e).__name__}: {e}"));raise


if __name__=="__main__":main()
