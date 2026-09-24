"""Red-team B: raw full-population coverage; no aggregate quality calculation.

Requires explicit parent GO and ALL_CELLS_SEALED before predictions/annotations.
Standard library only, sequential I/O, one CPU. Never imports the scorer/evaluator.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "results/lsml_external_generalization_v1"
EVAL = BASE / "evaluation"
OUT = Path(__file__).resolve().parent
PRIVATE = ROOT / "scratch/external_generalization_private"
CELLS = {"hard2verify_qwen3_8b": "hard2verify", "socratic_qwen3_8b": "socratic", "socratic_qwq32b": "socratic"}
ARMS = ("frozen_lsml", "frozen_equal", "frozen_partition_equal", "local_lsml", "local_equal", "local_partition_equal", "ct7")
LOCAL = tuple(x for x in ARMS if x.startswith("local_"))
PAIRS = [(v + "_lsml", v + "_" + c) for v in ("frozen", "local") for c in ("equal", "partition_equal")] + [(v + "_lsml", "ct7") for v in ("frozen", "local")]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for k, v in pairs:
        check(k not in result, "duplicate JSON object key: " + k)
        result[k] = v
    return result


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf8")).hexdigest()


def write(name, value):
    p = OUT/name
    check(not p.exists(), "refuse overwrite audit artifact: " + str(p))
    p.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorized-go", action="store_true", help="Pass only after explicit parent GO")
    args = parser.parse_args()
    check(args.authorized_go, "No parent GO supplied; no predictions/annotations opened")
    start = time.monotonic()
    marker_path = EVAL/"ALL_CELLS_SEALED.json"
    marker = load(marker_path)
    check(set(marker["seals"]) == set(CELLS), "not all three prediction cells sealed")
    seals = {cell: load(EVAL/cell/"SEAL.json") for cell in CELLS}
    for cell in CELLS:
        check(seals[cell] == marker["seals"][cell], "marker/seal discrepancy: " + cell)
        check((EVAL/cell/"PREDICTIONS.json").is_file(), "missing sealed predictions: " + cell)

    freeze = load(EVAL/"METHOD_FREEZE.json")
    analysis_path = EVAL/"ANALYSIS_FREEZE.json"
    analysis = load(analysis_path)
    check(marker["analysis_freeze_sha256"] == sha(analysis_path), "all-cell analysis freeze identity mismatch")
    check(analysis["method_freeze_sha256"] == sha(EVAL/"METHOD_FREEZE.json"), "analysis/method freeze mismatch")
    check(analysis["draws"] == 100000 and analysis["seed"] == 20260924 and analysis["primary_family"] == 18, "registered analysis parameters differ")
    for rel, h in analysis["files"].items():
        check(sha(ROOT/Path(rel)) == h, "analysis code/prose changed: " + rel)
    bundle_path = EVAL/"source/BUNDLE.json"
    bundle = load(bundle_path)
    bundle_hash = sha(bundle_path)
    check(marker["bundle_sha256"] == bundle_hash, "all-cell bundle identity mismatch")
    check(set(bundle["arms"]) == set(ARMS), "frozen seven-arm roster differs")
    check(bundle["fit_folds"] == [0, 1, 2, 3] and bundle["calibration_fold"] == 4, "source fit/calibration scope differs")
    check(set(bundle["thresholds"]) == set(ARMS), "threshold roster differs")
    for rel, h in freeze["artifacts"].items():
        check(sha(EVAL/Path(rel)) == h, "frozen artifact hash mismatch: " + rel)
    for rel, h in freeze["code"].items():
        check(sha(ROOT/Path(rel)) == h, "frozen package code changed: " + rel)
    execution = load(EVAL/"CPU_EXECUTION.json")
    check(execution["limit"] == 0, "execution remains a feasibility/smoke run")
    identity = execution["identity"]
    check(identity["bundle"] == bundle_hash, "CPU execution bundle differs")
    package = ROOT/"spectral_utils/external_generalization"
    actual_code = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob("*.py"))}
    check(actual_code == identity["code"], "recorded CPU code differs from available package")
    expected_code = {str(Path(k).relative_to("spectral_utils/external_generalization")): v for k, v in freeze["code"].items()}
    check(actual_code == expected_code, "CPU package differs from METHOD_FREEZE")
    run_identity = digest(identity)
    source_inputs = load(EVAL/"source/INPUTS.json")
    check(sha(EVAL/"source/INPUTS.json") == bundle["inputs_sha256"], "source input manifest changed")
    source_hashes = {}
    for name, info in source_inputs["paths"].items():
        actual = sha(info["path"])
        check(actual == info["sha256"], "source artifact changed: " + name)
        source_hashes[name] = actual
    collection_protocol = load(BASE/"COLLECTION_PROTOCOL.json")
    collection_protocol_hash = sha(BASE/"COLLECTION_PROTOCOL.json")

    population = EVAL/"independent_population"
    pop_audit, pop_lists = load(population/"AUDIT.json"), load(population/"UID_LISTS.json")
    pop_rows = {r["uid"]: r for r in load(population/"ROWS.json")}
    for info in pop_audit["input_provenance"]:
        check(sha(info["path"]) == info["sha256"], "independently audited population input changed")
    # All seals and source identities have passed before labels are opened.
    answers, gold = {}, {}
    for bench in sorted(set(CELLS.values())):
        a = load(PRIVATE/"inputs"/bench/"answers.json")
        g = load(PRIVATE/"inputs/evaluator_only"/(bench+".json"))
        check(len(a) == len({x["uid"] for x in a}), "duplicate input UIDs")
        check(len(g) == len({x["uid"] for x in g}), "duplicate annotation UIDs")
        answers[bench], gold[bench] = {x["uid"]: x for x in a}, {x["uid"]: x for x in g}
        check(set(answers[bench]) == set(gold[bench]) == set(pop_lists[bench]["all_uids"]), "population UID set mismatch")

    results, details = {}, []
    for cell, bench in CELLS.items():
        print("AUDIT_CELL_START", cell, flush=True)
        predictions = load(EVAL/cell/"PREDICTIONS.json")
        seal = seals[cell]
        check(seal["prediction_sha256"] == digest(predictions), "prediction digest/seal mismatch: " + cell)
        check(seal["lock_sha256"] == bundle_hash, "prediction seal bundle mismatch")
        check(set(seal["arms"]) == set(ARMS) and seal["answers"] == len(predictions), "seal coverage mismatch")
        check(set(predictions) == set(answers[bench]), "missing/extra sealed prediction UID")
        records_dir = PRIVATE/"evaluation_archives"/cell/"records"
        telemetry_files = list(records_dir.glob("*.record.json"))
        check(len(telemetry_files) == len(predictions), "telemetry file count differs")
        check({p.name for p in telemetry_files} == {digest(u)+".record.json" for u in predictions}, "telemetry UID filenames differ")
        collection = load(BASE/"full"/cell/"MANIFEST.json")
        check(collection["answers"] == len(predictions), "collection manifest answer count differs")
        expected_model = "Qwen/QwQ-32B" if cell == "socratic_qwq32b" else "Qwen/Qwen3-8B"
        check(collection["identity"]["model"] == expected_model, "telemetry backbone differs from cell")
        check(collection["identity"]["revision"] == collection_protocol["models"][expected_model], "telemetry model revision differs")
        check(collection["identity"]["mode"] == "full", "telemetry is only feasibility/smoke")
        check(collection["identity"]["protocol_sha256"] == collection_protocol_hash, "collection protocol differs")
        check(set(collection["identity"]["selected_uids"]) == set(predictions), "collection selected UID set differs")
        check(collection["identity"]["answers_sha256"] == sha(PRIVATE/"inputs"/bench/"answers.json"), "collection input answers identity differs")
        raw_run_identity = digest(collection["identity"])
        shard_records = {}
        shard_paths = list((EVAL/cell).glob("shard_*/*.record.json"))
        for shard in {p.parent for p in shard_paths}:
            check(not (shard/"WRITER.lock").exists(), "prediction writer still active")
            check(load(shard/"RUN.json")["identity"] == run_identity, "shard run identity differs")
        for path in shard_paths:
            rec = load(path)
            uid = rec["uid"]
            check(uid not in shard_records, "duplicate shard UID")
            check(path.name == digest(uid)+".record.json", "prediction filename/UID mismatch")
            check(rec["run_identity"] == run_identity, "per-answer CPU identity mismatch")
            check(rec["payload"] == predictions[uid], "sealed output differs from immutable checkpoint")
            shard_records[uid] = sha(path)
        check(set(shard_records) == set(predictions), "shard/seal UID mismatch")
        count = Counter(answers=len(predictions), registered_answers=len(answers[bench]))
        fallback_reasons = Counter()
        contrasts = {a+"__vs__"+b: Counter() for a,b in PAIRS}
        telemetry_ledger = {}
        groups, disjoint_groups, native_groups = set(), set(), set()
        for j, (uid, row) in enumerate(predictions.items()):
            inp, annotation, source_row = answers[bench][uid], gold[bench][uid], pop_rows[uid]
            n = len(inp["steps"])
            check(n == len(annotation["correct"]) == len(annotation["include"]) == len(row["nonempty"]), "step alignment differs: " + uid)
            check(set(row["arms"]) == set(row["scores"]) == set(row["predictions"]) == set(ARMS), "arm roster mismatch")
            check(len(row["arms"]) == len(ARMS), "duplicate arm entry")
            nonempty = row["nonempty"]
            check(all(type(v) is bool for v in nonempty), "nonboolean nonempty mask")
            check(nonempty == [bool(s) for s in inp["steps"]], "empty-text/nonempty mask mismatch")
            rawpath = records_dir/(digest(uid)+".record.json")
            raw_hash = sha(rawpath)
            check(raw_hash == row["telemetry_sha256"], "scored telemetry SHA mismatch: " + uid)
            telemetry_ledger[uid] = raw_hash
            raw = load(rawpath)
            check(raw["uid"] == uid and raw["run_identity"] == raw_run_identity, "telemetry UID/collection identity mismatch")
            telemetry = raw["payload"]["telemetry"]
            spans = telemetry["step_token_spans"]
            tokens = len(telemetry["gen_token_ids"])
            check(tokens == row["tokens"], "telemetry/scoring token count mismatch")
            check(len(spans) == n and [b>a for a,b in spans] == nonempty, "telemetry step-mask mismatch")
            check(all(0 <= a <= b <= tokens for a,b in spans), "out-of-range telemetry span")
            check(all(spans[k][1] <= spans[k+1][0] for k in range(n-1)), "overlapping/nonchronological step spans")
            native = row["local"]["native"]
            check(type(native) is bool, "nonboolean local native flag")
            if not native:
                check(row["local"]["fallback"] == "chosen_surprisal", "unexpected local fallback")
                check(all(row["scores"][arm] == row["scores"][LOCAL[0]] for arm in LOCAL), "local fallback scores differ")
                fallback_reasons[row["local"].get("reason", "unrecorded")] += 1
            else:
                check(row["local"].get("fallback") is None, "native row also reports fallback")
            group = source_row["bootstrap_group"]
            groups.add(group)
            disjoint = not source_row["component_development_overlap"]
            if disjoint:
                disjoint_groups.add(group)
            if native:
                native_groups.add(group)
            include = annotation["include"]
            check(all(type(v) is bool for v in include), "nonboolean inclusion mask")
            included = sum(include)
            eligible_native = sum(k and v for k,v in zip(include,nonempty)) if native else 0
            eligible_ranking = [v for v,k,ne in zip(annotation["correct"],include,nonempty) if k and ne]
            count.update(steps=n, included_steps=included, empty_steps=n-sum(nonempty),
                         scored_steps=sum(nonempty), tokens=tokens, local_native_answers=native,
                         local_fallback_answers=not native, matched_native_included_steps=eligible_native,
                         disjoint_answers=disjoint, disjoint_included_steps=included if disjoint else 0,
                         mixed_label_nonempty_answers=len(set(eligible_ranking)) == 2,
                         all_arm_answer_records=len(ARMS), all_arm_step_predictions=n*len(ARMS))
            for arm in ARMS:
                scores, decisions = row["scores"][arm], row["predictions"][arm]
                check(len(scores) == len(decisions) == n, "arm step count differs")
                for k, (s,pred,valid) in enumerate(zip(scores,decisions,nonempty)):
                    check(type(pred) is int and pred in (0,1), "invalid binary prediction")
                    if valid:
                        check(type(s) in (int,float) and math.isfinite(s), "nonfinite native score")
                        check(pred == int(s < bundle["thresholds"][arm]), "prediction differs from frozen threshold")
                    else:
                        check(s is None and pred == 0, "empty-step policy violated")
            for left,right in PAIRS:
                differences = [x != y for x,y in zip(row["predictions"][left],row["predictions"][right])]
                c = contrasts[left+"__vs__"+right]
                c.update(matched_answers=1, matched_included_steps=included,
                         matched_native_answers=native, matched_native_included_steps=eligible_native,
                         matched_disjoint_answers=disjoint, matched_disjoint_included_steps=included if disjoint else 0,
                         changed_answers=any(differences), changed_steps=sum(differences),
                         local_fallback_changed_steps=sum(differences) if left.startswith("local_") and not native else 0)
            details.append({"cell":cell,"uid":uid,"group":group,"steps":n,"nonempty_steps":sum(nonempty),
                            "native":native,"disjoint":disjoint,"tokens":tokens})
            if (j+1)%500 == 0:
                print("AUDIT_PROGRESS", cell, j+1, "/",len(predictions), flush=True)
        check(count["steps"] == collection["totals"]["steps"], "collection/scoring step totals differ")
        check(count["tokens"] == collection["totals"]["tokens"], "collection/scoring token totals differ")
        check(count["empty_steps"] == collection["totals"]["empty_steps"], "collection/scoring empty totals differ")
        check(count["disjoint_answers"] == len(pop_lists[bench]["observed_development_disjoint_uids"]), "disjoint cohort differs from independent audit")
        results[cell] = {"counts":dict(count),"source_groups":len(groups),"disjoint_groups":len(disjoint_groups),
                         "matched_native_groups":len(native_groups),"fallback_reasons":dict(fallback_reasons),
                         "comparisons":{k:dict(v) for k,v in contrasts.items()},
                         "prediction_sha256":sha(EVAL/cell/"PREDICTIONS.json"),"seal_sha256":sha(EVAL/cell/"SEAL.json"),
                         "collection_manifest_sha256":sha(BASE/"full"/cell/"MANIFEST.json"),
                         "telemetry_hash_ledger_digest":digest(telemetry_ledger),"shard_hash_ledger_digest":digest(shard_records)}
        write(cell+"_HASHES.json", {"telemetry":telemetry_ledger,"prediction_shards":shard_records})
        print("AUDIT_CELL_PASS",cell,dict(count),flush=True)
    check(sum(r["counts"]["answers"] for r in results.values()) == 6190, "not all6190 answer/backbone records checked")
    report = {"status":"PASS","created_utc":datetime.now(timezone.utc).isoformat(),"n_checked":6190,"n_total":6190,
              "quality_reports_read":False,"quality_metrics_computed":False,"processes":1,"results":results,
              "source_input_hashes":source_hashes,"bundle_sha256":bundle_hash,"method_freeze_sha256":sha(EVAL/"METHOD_FREEZE.json"),
              "all_cells_sealed_sha256":sha(marker_path),"cpu_execution_sha256":sha(EVAL/"CPU_EXECUTION.json"),
              "analysis_freeze_sha256":sha(analysis_path),"analysis_files_checked":len(analysis["files"]),
              "population_audit_sha256":sha(population/"AUDIT.json"),"audit_script_sha256":sha(Path(__file__)),
              "collection_protocol_sha256":collection_protocol_hash,
              "execution_lock_sha256_observed":sha(ROOT/"docs/experiments/LSML_EXTERNAL_EVALUATION_LOCK_20260924.md"),
              "wall_seconds":time.monotonic()-start,
              "limits":["Coverage/provenance audit only; estimator fidelity, label semantics and quality metrics require independent other checks.",
                        "Per-cell seals bind source bundle; ALL_CELLS_SEALED additionally binds ANALYSIS_FREEZE, which pins the runner/evaluator, bootstrap helper, label-sanity code, prose lock and METHOD_FREEZE.",
                        "All three local arms share fallback scores; per-arm source-calibrated thresholds may produce different fallback decisions.",
                        "Native comparisons use the SAME local-native and nonempty mask for every arm; they are conditional diagnostics, not full-population replacements.",
                        "Observed-disjoint cohorts exclude known normalized-text/source components; underlying unseen-source/pretraining independence remains unproven.",
                        "No supervised comparator inference or historical final-answer benchmark transfer is validated by this coverage audit."]}
    write("AUDIT.json", report)
    write("ANSWER_COVERAGE.json", details)
    print("FULL_COVERAGE_PASS",6190,"of",6190,"wall_seconds",round(time.monotonic()-start,1),flush=True)


if __name__ == "__main__":
    main()
