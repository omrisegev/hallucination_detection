#!/usr/bin/env python3
"""Run frozen C7/C8 scorer-family transfer on four Llama ProcessBench cells."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json, atomic_write_npz, load_npz_no_pickle, sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import (  # noqa: E402
    load_prepared_localization_cell, validate_fit_manifest,
)
from scripts.reasoning_localization import run_phase1_baseline as p1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_c1 as c1  # noqa: E402
from scripts.reasoning_localization import run_phase2_atomic_remaining as atomic  # noqa: E402
from scripts.reasoning_localization import run_phase2_reducer as p2r  # noqa: E402


VARIANTS = ("P2C_C7_EDIS_LLAMA4", "P2C_C8_INNOV_LLAMA4")
SOURCE_VARIANT = {VARIANTS[0]:"C7_EDIS_ONSET", VARIANTS[1]:"C8_SELF_INNOV"}
TOP10 = "P2C_ENTROPY_TOP10_LLAMA4"
TOP5 = "P2C_ENTROPY_TOP5_LLAMA4"
FAMILY6 = "P2C_FAMILY6_TOP5_LLAMA4"
IU29_PARENT = "P2C_IU29_TOP10_LLAMA4"
LLAMA_MODEL = "llama31_8b"
LLAMA_CELLS = tuple(f"processbench_{family}_{LLAMA_MODEL}" for family in p1.FAMILIES)
ROOT = p1.PROGRAM_ROOT / "phase_2/confirmation_llama4"
PRIMARY_FAMILY_SIZE = 4
BENEFIT = 0.005
HARM = -0.005
HARD_WORST_CELL = -0.030
PROMOTION_WORST_CELL = -0.020
COMPONENT_BOUND = -0.010


class ConfirmationError(RuntimeError):
    pass


def output_root(variant: str) -> Path:
    return ROOT / variant.lower()


def registry_path(variant: str) -> Path:
    return ROOT / f"{variant}_EXECUTION_REGISTRY.json"


def require_sources(registry: Mapping[str, Any]) -> None:
    for source in registry["frozen_sources"]:
        path = Path(source["path"])
        if not path.is_file() or sha256_file(path) != source["sha256"]:
            raise ConfirmationError(f"frozen source changed or missing: {source['role']}")


def load_registry(path: Path, release: Path, variant: str) -> dict[str, Any]:
    registry = json.loads(path.read_text())
    required = {"schema":"reasoning-localization-p2-confirmation-execution-v1",
                "status":"FROZEN_BEFORE_RUN","variant_id":variant,
                "llama_cells":list(LLAMA_CELLS),"primary_family_size":PRIMARY_FAMILY_SIZE,
                "bootstrap_draws":p1.BOOTSTRAP_DRAWS,"bootstrap_seed":p1.BOOTSTRAP_SEED}
    for key,value in required.items():
        if registry.get(key) != value:
            raise ConfirmationError(f"registry mismatch: {key}")
    if Path(registry["release_root"]).resolve() != release.resolve():
        raise ConfirmationError("release root mismatch")
    if registry["runner_sha256"] != sha256_file(Path(__file__).resolve()):
        raise ConfirmationError("runner changed after freeze")
    require_sources(registry)
    return registry


def load_family6(cell_id: str, cell: Any) -> tuple[np.ndarray,np.ndarray,float]:
    path = p1.PROGRAM_ROOT / "phase_1/r2_family6_top5_current/score_freeze/cells" / cell_id / "scores.npz"
    arrays = load_npz_no_pickle(path)
    if tuple(arrays["row_ids"].astype(str)) != tuple(cell.row_ids) or not np.array_equal(arrays["segment_offsets"], cell.segment_offsets):
        raise ConfirmationError(f"{cell_id}: family6 alignment mismatch")
    local = np.asarray(arrays["local_step_scores"], dtype=np.float64)
    combined = np.asarray(arrays["combined_step_scores"], dtype=np.float64)
    reconstructed = p1.combine_with_common_detector(cell, local)
    return local, combined, float(np.max(np.abs(combined-reconstructed)))


def load_phase1_top5(cell_id: str, cell: Any) -> tuple[np.ndarray,np.ndarray]:
    path = p1.PROGRAM_ROOT / "phase_1/r1_entropy_top5/score_freeze/cells" / cell_id / "scores.npz"
    arrays = load_npz_no_pickle(path)
    if tuple(arrays["row_ids"].astype(str)) != tuple(cell.row_ids) or not np.array_equal(arrays["segment_offsets"], cell.segment_offsets):
        raise ConfirmationError(f"{cell_id}: Phase-1 top-five alignment mismatch")
    return (np.asarray(arrays["local_step_scores"], dtype=np.float64),
            np.asarray(arrays["combined_step_scores"], dtype=np.float64))


def freeze_scores(variant: str, release: Path, output: Path, registry: Mapping[str, Any]) -> dict[str,Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {variant}")
    score_root = output / "score_freeze"
    score_root.mkdir(parents=True, exist_ok=False)
    input_root = release / "build_A/localization/inputs"
    manifest_path = input_root / "MANIFEST.json"
    manifest = validate_fit_manifest(manifest_path, input_root=input_root)
    by_cell = {str(row["cell_id"]):row for row in manifest["cells"]}
    records=[]; family6_alias=0.0; top5_local_alias=0.0; top5_combined_alias=0.0; suffix_error=0.0; iu29_alias=0.0
    for position,cell_id in enumerate(LLAMA_CELLS,1):
        source=by_cell[cell_id]; input_path=input_root/source["artifact_path"]
        cell=load_prepared_localization_cell(input_path,source)
        entropy=atomic.primitive_risks(cell)["entropy"]
        top10_local=p1.topk_step_mean(entropy,cell.segment_starts,cell.segment_ends,k=10)
        top5_local=p1.topk_step_mean(entropy,cell.segment_starts,cell.segment_ends,k=5)
        phase1_top5_local,phase1_top5_combined=load_phase1_top5(cell_id,cell)
        top5_local_alias=max(top5_local_alias,float(np.max(np.abs(top5_local-phase1_top5_local))))
        top5_combined=p1.combine_with_common_detector(cell,top5_local)
        top5_combined_alias=max(top5_combined_alias,float(np.max(np.abs(top5_combined-phase1_top5_combined))))
        family6_local,family6_combined,alias=load_family6(cell_id,cell)
        family6_alias=max(family6_alias,alias)
        source_variant=SOURCE_VARIANT[variant]
        candidate_local,extra,diagnostics,cell_suffix=atomic.score_candidate(source_variant,cell)
        suffix_error=max(suffix_error,cell_suffix)
        arrays={
            "row_ids":np.asarray(cell.row_ids,dtype="<U80"),
            "segment_offsets":np.asarray(cell.segment_offsets,dtype="<i8"),
            "segment_lengths":np.asarray(cell.segment_ends-cell.segment_starts,dtype="<i8"),
            f"{TOP10}__local":np.asarray(top10_local,dtype="<f8"),
            f"{TOP10}__combined":np.asarray(p1.combine_with_common_detector(cell,top10_local),dtype="<f8"),
            f"{TOP5}__local":np.asarray(top5_local,dtype="<f8"),
            f"{TOP5}__combined":np.asarray(top5_combined,dtype="<f8"),
            f"{FAMILY6}__local":np.asarray(family6_local,dtype="<f8"),
            f"{FAMILY6}__combined":np.asarray(family6_combined,dtype="<f8"),
            f"{variant}__local":np.asarray(candidate_local,dtype="<f8"),
            f"{variant}__combined":np.asarray(p1.combine_with_common_detector(cell,candidate_local),dtype="<f8"),
        }
        if variant == VARIANTS[1]:
            parent_local=np.asarray(extra[atomic.EXTRA_PARENT["C8_SELF_INNOV"]],dtype=np.float64)
            arrays[f"{IU29_PARENT}__local"]=parent_local
            arrays[f"{IU29_PARENT}__combined"]=p1.combine_with_common_detector(cell,parent_local)
            strict_local,_strict_combined,_record=p1._strict_r3_scores(release,cell)
            audit=np.asarray(extra["__C8_IU29_STEPMAX_AUDIT"],dtype=np.float64)
            iu29_alias=max(iu29_alias,float(np.max(np.abs(audit-strict_local))))
        target=score_root/"cells"/cell_id; target.mkdir(parents=True,exist_ok=False)
        score_path=target/"scores.npz"; score_sha=atomic_write_npz(score_path,arrays)
        record={"schema":"reasoning-localization-p2-confirmation-cell-v1","variant_id":variant,"cell_id":cell_id,
                "model_id":str(cell.model_id),"slice_id":str(cell.slice_id),"population_id":str(cell.population_id),
                "n_rows":len(cell.row_ids),"n_steps":len(candidate_local),"prepared_input":str(input_path),
                "prepared_input_sha256":sha256_file(input_path),"score_file":"scores.npz","score_sha256":score_sha,
                "labels_seen_during_fit":False,"targets_accessed_during_fit":False,"diagnostics":dict(diagnostics)}
        record["payload_sha256"]=c1.payload_sha(record); atomic_write_json(target/"RECORD.json",record)
        records.append({"cell_id":cell_id,"record_path":f"cells/{cell_id}/RECORD.json","record_sha256":sha256_file(target/"RECORD.json"),"score_sha256":score_sha})
        print(f"score-freeze {variant}: {cell_id} ({position}/4)",flush=True)
    require_sources(registry)
    freeze={"schema":"reasoning-localization-p2-confirmation-score-freeze-v1","status":"COMPLETE","variant_id":variant,
            "llama_cells":list(LLAMA_CELLS),"labels_seen_during_fit":False,"targets_accessed_during_fit":False,
            "family6_combined_alias_max_abs_error":family6_alias,
            "phase1_top5_local_alias_max_abs_error":top5_local_alias,
            "phase1_top5_combined_alias_max_abs_error":top5_combined_alias,
            "suffix_invariance_max_abs_error":suffix_error,
            "iu29_stepmax_alias_max_abs_error":iu29_alias if variant==VARIANTS[1] else None,
            "input_manifest_sha256":sha256_file(manifest_path),"execution_registry_sha256":sha256_file(Path(registry["registry_path"])),
            "runner_sha256":sha256_file(Path(__file__).resolve()),"environment":{"python":sys.version,"platform":platform.platform(),"numpy":np.__version__},
            "records":records}
    freeze["payload_sha256"]=c1.payload_sha(freeze); atomic_write_json(score_root/"SCORE_FREEZE_MANIFEST.json",freeze)
    return freeze


def verified(output: Path, freeze: Mapping[str,Any]) -> dict[str,Any]:
    result={}
    for item in freeze["records"]:
        record_path=output/"score_freeze"/item["record_path"]
        if sha256_file(record_path)!=item["record_sha256"]: raise ConfirmationError("record changed after freeze")
        record=json.loads(record_path.read_text()); score_path=record_path.parent/record["score_file"]
        if sha256_file(score_path)!=item["score_sha256"]: raise ConfirmationError("score changed after freeze")
        result[item["cell_id"]]={"record":record,"arrays":load_npz_no_pickle(score_path)}
    return result


def rows_for_arm(scores: Mapping[str,Any], labels: Mapping[str,Mapping[str,tuple[str,int]]], arm: str) -> list[dict[str,Any]]:
    rows=[]
    for cell_id in LLAMA_CELLS:
        record,arrays=scores[cell_id]["record"],scores[cell_id]["arrays"]
        row_ids=tuple(arrays["row_ids"].astype(str)); offsets=np.asarray(arrays["segment_offsets"],dtype=np.int64)
        lengths=np.asarray(arrays["segment_lengths"],dtype=np.int64); values=np.asarray(arrays[f"{arm}__combined"],dtype=np.float64)
        if set(row_ids)!=set(labels[cell_id]): raise ConfirmationError("score/label population mismatch")
        for index,row_id in enumerate(row_ids):
            lo,hi=map(int,offsets[index:index+2]); group_id,first_error=labels[cell_id][row_id]
            rows.append({"row_id":row_id,"group_id":group_id,"slice_id":record["slice_id"],"cell_id":cell_id,
                         "model_id":LLAMA_MODEL,"first_error":first_error,"step_scores":values[lo:hi].tolist(),"step_lengths":lengths[lo:hi].tolist()})
    return rows


def evaluate_arm(arm: str, rows: Sequence[Mapping[str,Any]], evaluation: Any) -> dict[str,Any]:
    result=evaluation.crossfit_processbench_threshold(list(rows)); source={str(row["row_id"]):row for row in rows}
    assignments=evaluation.assign_processbench_folds(list(rows)); cutpoints=p2r._length_cutpoints(rows,assignments)
    decisions=[]
    for row in result["decisions"]:
        parent=source[str(row["row_id"])]; target=int(parent["first_error"]); prediction=int(row["prediction_step"])
        decisions.append({"arm_id":arm,"model_id":LLAMA_MODEL,"cell_id":parent["cell_id"],"slice_id":parent["slice_id"],
            "row_id":row["row_id"],"group_id":parent["group_id"],"fold":int(row["fold"]),"true_first_error":target,"prediction_step":prediction,
            "true_error_step_length":int(parent["step_lengths"][target]) if target>=0 else None,
            "true_error_length_stratum":p2r._stratum(int(parent["step_lengths"][target]),cutpoints[str(row["fold"])]) if target>=0 else "CLEAN",
            "selected_step_length":int(parent["step_lengths"][prediction]) if prediction>=0 else None})
    by_cell=[]
    for family,metrics in result["metrics"]["per_subset"].items():
        by_cell.append({"arm_id":arm,"model_id":LLAMA_MODEL,"slice_id":family,"cell_id":f"processbench_{family}_{LLAMA_MODEL}",
                        **{metric:metrics[metric] for metric in p1.PB_METRICS},"n_examples":metrics["n_examples"],"n_error":metrics["n_error"],"n_clean":metrics["n_clean"]})
    samples=p1._bootstrap_pb_panel(decisions,(LLAMA_MODEL,)); panels=[]
    for metric in p1.PB_METRICS:
        values=np.asarray(samples[metric],dtype=np.float64)
        panels.append({"arm_id":arm,"population_id":"current_llama4_scorer_transfer","metric_id":metric,
                       "value":float(np.mean([float(row[metric]) for row in by_cell])),"ci_low":float(np.quantile(values,.025)),"ci_high":float(np.quantile(values,.975)),
                       "n_rows":sum(int(row["n_examples"]) for row in by_cell),"n_groups":3400})
    return {"decisions":decisions,"by_cell":by_cell,"samples":samples,"panels":panels,"ledgers":result["calibration_ledgers"]}


def contrast(left: Mapping[str,Any],right: Mapping[str,Any],variant: str,comparator: str,metric: str,*,primary: bool) -> dict[str,Any]:
    rc={row["cell_id"]:row for row in right["by_cell"]}; lp=float(np.mean([float(row[metric]) for row in left["by_cell"]])); rp=float(np.mean([float(rc[row["cell_id"]][metric]) for row in left["by_cell"]]))
    draws=np.asarray(left["samples"][metric])-np.asarray(right["samples"][metric]); q=.025/PRIMARY_FAMILY_SIZE if primary and metric=="official_macro_f1" else .025
    cells={row["cell_id"]:float(row[metric])-float(rc[row["cell_id"]][metric]) for row in left["by_cell"]}; eps=1e-12
    return {"contrast_id":f"llama::{variant}::{comparator}::{metric}","left_variant_id":variant,"right_variant_id":comparator,
            "metric_id":"macro_f1" if metric=="official_macro_f1" else metric,"source_metric_id":metric,"delta":lp-rp,
            "ci_low":float(np.quantile(draws,q)),"ci_high":float(np.quantile(draws,1-q)),"wins":sum(v>eps for v in cells.values()),
            "ties":sum(abs(v)<=eps for v in cells.values()),"losses":sum(v<-eps for v in cells.values()),"worst_unit_delta":min(cells.values()),"worst_unit_id":min(cells,key=cells.get),
            "multiplicity_family_size":PRIMARY_FAMILY_SIZE if primary and metric=="official_macro_f1" else 1,
            "inference":"Bonferroni simultaneous percentile interval across four C7/C8 Llama primary contrasts" if primary and metric=="official_macro_f1" else "unadjusted paired mechanism diagnostic"}


def evaluate(variant: str, release: Path, output: Path, registry: Mapping[str,Any], freeze: Mapping[str,Any]) -> dict[str,Any]:
    require_sources(registry); scores=verified(output,freeze); labels=p1._load_pb_labels(release)
    evaluation=importlib.import_module("spectral_utils.reconstruction_benchmark.localization_evaluation")
    arms=(TOP10,TOP5,FAMILY6,variant)+( (IU29_PARENT,) if variant==VARIANTS[1] else () )
    results={arm:evaluate_arm(arm,rows_for_arm(scores,labels,arm),evaluation) for arm in arms}
    contrasts=[]
    for comparator in (TOP10,TOP5,FAMILY6)+( (IU29_PARENT,) if variant==VARIANTS[1] else () ):
        primary=comparator in (TOP10,TOP5)
        contrasts.extend(contrast(results[variant],results[comparator],variant,comparator,metric,primary=primary) for metric in p1.PB_METRICS)
    primary={row["right_variant_id"]:row for row in contrasts if row["metric_id"]=="macro_f1" and row["right_variant_id"] in (TOP10,TOP5)}
    by={(row["right_variant_id"],row["metric_id"]):row for row in contrasts}
    technical=(freeze["family6_combined_alias_max_abs_error"]>1e-12
               or freeze["phase1_top5_local_alias_max_abs_error"]>1e-12
               or freeze["phase1_top5_combined_alias_max_abs_error"]>1e-12
               or freeze["suffix_invariance_max_abs_error"]>1e-12
               or (variant==VARIANTS[1] and freeze["iu29_stepmax_alias_max_abs_error"]>1e-12))
    robustness=min(float(row["worst_unit_delta"]) for row in primary.values())<HARD_WORST_CELL
    promotion=not technical and not robustness and all(float(row["delta"])>=BENEFIT and float(row["ci_low"])>BENEFIT and int(row["wins"])+int(row["ties"])>=3 and float(row["worst_unit_delta"])>=PROMOTION_WORST_CELL and float(by[(comp,"first_error_exact")]["delta"])>=COMPONENT_BOUND and float(by[(comp,"clean_abstention_accuracy")]["delta"])>=COMPONENT_BOUND for comp,row in primary.items())
    hard=technical or robustness
    gates=[{"gate_id":"P2C_SCORE_FREEZE","status":"PASS","observed":4,"required":"4 Llama cells","detail":"all arms froze before labels"},
           {"gate_id":"P2C_LABEL_FIREWALL","status":"PASS","observed":"labels opened after score freeze","required":"no fit-side labels","detail":"fixed transfer scorers"},
           {"gate_id":"P2C_FAMILY6_ALIAS","status":"PASS" if freeze["family6_combined_alias_max_abs_error"]<=1e-12 else "HARD_FAIL","observed":freeze["family6_combined_alias_max_abs_error"],"required":"<=1e-12","detail":"frozen Phase-1 family6 comparator"},
           {"gate_id":"P2C_TOP5_ALIAS","status":"PASS" if max(freeze["phase1_top5_local_alias_max_abs_error"],freeze["phase1_top5_combined_alias_max_abs_error"])<=1e-12 else "HARD_FAIL","observed":max(freeze["phase1_top5_local_alias_max_abs_error"],freeze["phase1_top5_combined_alias_max_abs_error"]),"required":"<=1e-12","detail":"recomputed top-five aliases frozen Phase-1 local and combined scores"},
           {"gate_id":f"{variant}_SUFFIX_INVARIANCE","status":"PASS" if freeze["suffix_invariance_max_abs_error"]<=1e-12 else "HARD_FAIL","observed":freeze["suffix_invariance_max_abs_error"],"required":"<=1e-12","detail":"fixed-map causal replay"},
           {"gate_id":f"{variant}_WORST_CELL_HARD","status":"HARD_FAIL" if robustness else "PASS","observed":min(float(row["worst_unit_delta"]) for row in primary.values()),"required":f">={HARD_WORST_CELL}","detail":"required-reference worst cell"},
           {"gate_id":f"{variant}_PROMOTION","status":"PASS" if promotion else "FAIL","observed":str(promotion).lower(),"required":"all transfer promotion gates","detail":"eligibility for PRMBench and Phase 3"}]
    eval_root=output/"evaluation"; eval_root.mkdir(parents=True,exist_ok=False)
    c1.write_csv(eval_root/"PROCESSBENCH_DECISIONS.csv",[row for arm in arms for row in results[arm]["decisions"]])
    c1.write_csv(eval_root/"PROCESSBENCH_BY_CELL.csv",[row for arm in arms for row in results[arm]["by_cell"]])
    c1.write_csv(eval_root/"PROCESSBENCH_PANELS.csv",[row for arm in arms for row in results[arm]["panels"]])
    atomic_write_npz(eval_root/"PROCESSBENCH_BOOTSTRAP_SAMPLES.npz",{f"{arm}__{metric}":values for arm in arms for metric,values in results[arm]["samples"].items()})
    atomic_write_json(eval_root/"CALIBRATION_LEDGERS.json",{"schema":"reasoning-localization-p2-confirmation-calibration-v1","arms":{arm:results[arm]["ledgers"] for arm in arms}})
    c1.write_csv(eval_root/"PAIRWISE_CONTRASTS.csv",contrasts); c1.write_csv(eval_root/"GATES.csv",gates)
    c1.write_csv(eval_root/"STEP_LENGTH_STRATA.csv",p2r._length_strata(results[variant]["decisions"],results[variant]["by_cell"]))
    flips,flip_summary=c1.prediction_flips(results[variant]["decisions"],results[FAMILY6]["decisions"])
    c1.write_csv(eval_root/"FAMILY6_PREDICTION_FLIPS.csv",flips); c1.write_csv(eval_root/"FAMILY6_PREDICTION_FLIP_SUMMARY.csv",flip_summary)
    panel=next(row for row in results[variant]["panels"] if row["metric_id"]=="official_macro_f1")
    status="HARD_FAIL" if hard else "COMPLETE"
    summary={"schema":"reasoning-localization-p2-confirmation-evaluation-v1","variant_id":variant,"status":status,"promotion_gate_passed":promotion,
             "macro_f1":panel["value"],"macro_f1_ci":[panel["ci_low"],panel["ci_high"]],"primary_contrasts":primary,
             "family6_macro_contrast":next(row for row in contrasts if row["right_variant_id"]==FAMILY6 and row["metric_id"]=="macro_f1"),
             "bootstrap_draws":p1.BOOTSTRAP_DRAWS,"bootstrap_seed":p1.BOOTSTRAP_SEED,"peak_memory_bytes":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    summary["payload_sha256"]=c1.payload_sha(summary); atomic_write_json(eval_root/"SUMMARY.json",summary)
    outputs=[path.name for path in sorted(eval_root.iterdir())]
    manifest={"schema":"reasoning-localization-p2-confirmation-evaluation-manifest-v1","variant_id":variant,"status":status,
              "score_freeze_sha256":sha256_file(output/"score_freeze/SCORE_FREEZE_MANIFEST.json"),"execution_registry_sha256":sha256_file(Path(registry["registry_path"])),
              "outputs":[{"path":name,"sha256":sha256_file(eval_root/name),"bytes":(eval_root/name).stat().st_size} for name in outputs]}
    manifest["payload_sha256"]=c1.payload_sha(manifest); atomic_write_json(eval_root/"EVALUATION_MANIFEST.json",manifest)
    return summary


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--variant",choices=VARIANTS,required=True); parser.add_argument("--release",type=Path,default=p1.DEFAULT_RELEASE)
    args=parser.parse_args(); variant=args.variant; release=args.release.resolve(); reg_path=registry_path(variant).resolve(); output=output_root(variant).resolve()
    registry=load_registry(reg_path,release,variant); registry["registry_path"]=str(reg_path); started=time.perf_counter()
    freeze=freeze_scores(variant,release,output,registry); summary=evaluate(variant,release,output,registry,freeze)
    run={"schema":"reasoning-localization-p2-confirmation-run-v1","variant_id":variant,"status":summary["status"],"execution_registry_sha256":sha256_file(reg_path),
         "runner_sha256":sha256_file(Path(__file__).resolve()),"score_freeze_manifest_sha256":sha256_file(output/"score_freeze/SCORE_FREEZE_MANIFEST.json"),
         "evaluation_manifest_sha256":sha256_file(output/"evaluation/EVALUATION_MANIFEST.json"),"elapsed_seconds":time.perf_counter()-started,"summary":summary}
    run["payload_sha256"]=c1.payload_sha(run); atomic_write_json(output/"RUN_MANIFEST.json",run); print(json.dumps(run,indent=2))


if __name__=="__main__": main()
