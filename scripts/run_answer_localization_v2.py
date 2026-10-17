"""Prepare, score and evaluate the registered representation pilot separately."""
from __future__ import annotations

import os
for _thread_option in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_option] = "1"

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LIVE = Path("C:/Users/omris/TAU/hd_jlsml_v2_wt/results/joint_lsml_optimization_v2")
OUT = ROOT / "results/answer_localization_representation_pilot_v1"
RELEASE = "localization-cached-v1-20260907"
PROTOCOL = ROOT / "docs/experiments/ANSWER_LOCALIZATION_REPRESENTATION_PILOT_V1.md"
CAPSULE = ROOT / "local_cache/short_cycle01_code"
sys.path.insert(0, str(CAPSULE))
import spectral_utils
spectral_utils.__path__.append(str(ROOT / "spectral_utils"))
from spectral_utils.answer_localization_v2 import ARM_IDS, REPS, json_safe, score_answer

PILOT_CELLS = ("prmbench_qwen3_8b", "pb_gsm8k_q8", "pb_math_q8", "pb_olympiadbench_q8", "pb_omnimath_q8")
BINS = ((64, 255), (256, 1023), (1024, 2048))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def source_files():
    paths = {Path(__file__).resolve(), PROTOCOL}
    for name, module in sys.modules.items():
        if name.startswith("spectral_utils") and getattr(module, "__file__", None):
            paths.add(Path(module.__file__).resolve())
    return {str(p): sha(p) for p in sorted(paths)}


def verify_sources(prepared):
    for filename, expected in prepared["source_hashes"].items():
        if sha(filename) != expected:
            raise RuntimeError(f"FROZEN_SOURCE_DRIFT: {filename}")


def prepare():
    if (OUT / "PREPARED.json").exists():
        prepared = load_json(OUT / "PREPARED.json")
        verify_sources(prepared)
        print("Preparation already frozen; use the existing release.", flush=True)
        return
    (OUT / "inputs").mkdir(parents=True, exist_ok=True)
    complete, selected = {}, []
    for cell_path in sorted((LIVE / "cells").glob("*.npz")):
        cell = cell_path.stem
        with np.load(cell_path, allow_pickle=False) as bundle:
            ids, groups = bundle["row_ids"].astype(str), bundle["group_ids"].astype(str)
            offsets, step_offsets = bundle["token_offsets"], bundle["step_row_offsets"]
            lengths = np.diff(offsets)
            if len(ids) != len(set(ids)):
                raise ValueError("DUPLICATE_ROW_IDS")
            records = [{"row": i, "row_id": str(row_id), "group_id": str(groups[i]),
                        "tokens": int(lengths[i]), "steps": int(step_offsets[i+1]-step_offsets[i])}
                       for i, row_id in enumerate(ids)]
            label_path = LIVE / "labels" / f"{cell}_labels.npz"
            complete[cell] = {"rows": records, "telemetry_path": str(cell_path), "telemetry_sha256": sha(cell_path),
                              "label_path": str(label_path), "label_opaque_sha256": sha(label_path),
                              "role": "development_pilot" if cell in PILOT_CELLS else "registered_later_replication"}
            if cell not in PILOT_CELLS:
                continue
            chosen, used_groups = [], set()
            for lower, upper in BINS:
                candidates = [r for r in records if lower <= r["tokens"] <= upper]
                candidates.sort(key=lambda r: (digest_text(f"{RELEASE}/{cell}/{r['group_id']}"),
                                               digest_text(r["row_id"])))
                count = 0
                for record in candidates:
                    if record["group_id"] in used_groups:
                        continue
                    chosen.append({**record, "length_bin": [lower, upper]})
                    used_groups.add(record["group_id"])
                    count += 1
                    if count == 4:
                        break
            # Decode raw data once per cell, then persist only selected telemetry.
            raw = bundle["raw"]
            for record in chosen:
                i = record["row"]
                lo, hi = map(int, offsets[[i, i+1]])
                a, b = map(int, step_offsets[[i, i+1]])
                uid = f"{cell}__{digest_text(record['row_id'])[:16]}"
                path = OUT / "inputs" / f"{uid}.npz"
                np.savez_compressed(path, raw=raw[lo:hi],
                                    step_starts=bundle["step_starts"][a:b]-lo,
                                    step_ends=bundle["step_ends"][a:b]-lo)
                selected.append({**record, "cell": cell, "uid": uid, "input_sha256": sha(path)})
            del raw
            print(f"prepared {cell}: {len(chosen)} answers", flush=True)
    save_json(OUT / "RELEASE.json", {"release_id": RELEASE, "exposure": "development_previously_evaluated_by_v2",
                                      "cells": complete, "labels_decoded": False})
    prepared = {"release_id": RELEASE, "pilot_id": OUT.name, "selected": selected,
                "arms": ARM_IDS, "source_hashes": source_files(),
                "release_manifest_sha256": sha(OUT / "RELEASE.json"), "labels_decoded": False,
                "selection": "4 unique groups per frozen length bin and cell; no label stratification",
                "single_answer_sign_convention": "negative entropy anchor; learned within-answer coordinate signs",
                "legacy_lane": "borrowed historical feature calibration, original fit admission",
                "workers_cap": 3}
    save_json(OUT / "PREPARED.json", prepared)
    print(f"Frozen cohort: {len(selected)} answers. No labels decoded.", flush=True)


def process_one(record, prepared_hash):
    uid = record["uid"]
    destination = OUT / "scores" / f"{uid}.npz"
    meta_path = destination.with_suffix(".json")
    if meta_path.exists():
        meta = load_json(meta_path)
        if meta["prepared_sha256"] == prepared_hash and destination.exists() and sha(destination) == meta["array_sha256"]:
            return {"uid": uid, "state": "VERIFIED_RESUME", "seconds": meta["elapsed_seconds"]}
        raise RuntimeError(f"INVALID_EXISTING_CHECKPOINT: {uid}")
    input_path = OUT / "inputs" / f"{uid}.npz"
    if sha(input_path) != record["input_sha256"]:
        raise RuntimeError(f"INPUT_DRIFT: {uid}")
    started = time.monotonic()
    with np.load(input_path, allow_pickle=False) as bundle:
        arrays, report = score_answer(bundle["raw"], bundle["step_starts"], bundle["step_ends"],
                                      f"{RELEASE}/{record['cell']}/{record['row_id']}")
    np.savez_compressed(destination, **arrays)
    meta = {**record, "report": report, "prepared_sha256": prepared_hash,
            "array_sha256": sha(destination), "labels_decoded": False,
            "elapsed_seconds": time.monotonic() - started}
    save_json(meta_path, meta)
    return {"uid": uid, "state": "SCORED", "seconds": meta["elapsed_seconds"],
            "valid": sum(m.get("valid", False) for m in report["methods"].values())}


def scores(workers):
    prepared = load_json(OUT / "PREPARED.json")
    verify_sources(prepared)
    if not 1 <= workers <= prepared["workers_cap"]:
        raise ValueError("worker cap is three")
    (OUT / "scores").mkdir(exist_ok=True)
    prepared_hash = sha(OUT / "PREPARED.json")
    started = time.monotonic()
    save_json(OUT / "RUN_STATE.json", {"pid": os.getpid(), "state": "RUNNING", "completed": 0,
                                       "total": len(prepared["selected"]), "start_unix": time.time()})
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(process_one, record, prepared_hash) for record in prepared["selected"]]
        for number, future in enumerate(concurrent.futures.as_completed(tasks), 1):
            result = future.result()
            print(f"{number}/{len(tasks)} {json.dumps(result)}", flush=True)
            save_json(OUT / "RUN_STATE.json", {"pid": os.getpid(), "state": "RUNNING", "completed": number,
                                               "total": len(tasks), "elapsed_seconds": time.monotonic()-started})
    verify_sources(prepared)
    paths = sorted((OUT / "scores").glob("*.npz")) + sorted((OUT / "scores").glob("*.json"))
    if len(paths) != 2 * len(prepared["selected"]):
        raise RuntimeError("SCORE_COUNT_MISMATCH")
    save_json(OUT / "SCORES_FROZEN.json", {"prepared_sha256": prepared_hash,
                                           "files": {str(p): sha(p) for p in paths},
                                           "labels_decoded": False, "elapsed_seconds": time.monotonic()-started})
    save_json(OUT / "RUN_STATE.json", {"pid": os.getpid(), "state": "COMPLETE", "completed": len(tasks),
                                       "total": len(tasks), "elapsed_seconds": time.monotonic()-started})
    print("All scores and decisions frozen; evaluation remains separate.", flush=True)


def safe_auc(y, score):
    return float(roc_auc_score(y, score)) if len(y) and len(np.unique(y)) == 2 else None


def prm_metric(rows, arm):
    available = [r for r in rows if r["cell"].startswith("prm") and r["valid"].get(arm)]
    if not available:
        return {"answers": 0, "auroc": None, "within_answer_auc": None, "mixed_answers": 0}
    within = [safe_auc(r["target"], r["scores"][arm]) for r in available]
    within = [v for v in within if v is not None]
    return {"answers": len(available), "auroc": safe_auc(np.concatenate([r["target"] for r in available]),
                                                         np.concatenate([r["scores"][arm] for r in available])),
            "within_answer_auc": float(np.mean(within)) if within else None, "mixed_answers": len(within)}


def pb_metric(rows, arm):
    cells = sorted({r["cell"] for r in rows if r["cell"].startswith("pb_")})
    result = {}
    for cell in cells:
        subset = [r for r in rows if r["cell"] == cell]
        clean, erroneous = [r for r in subset if r["target"] == -1], [r for r in subset if r["target"] != -1]
        def success(row):
            return row["decision_valid"].get(arm, False) and row["predictions"].get(arm) == row["target"]
        ca = sum(map(success, clean))/len(clean) if clean else None
        ea = sum(map(success, erroneous))/len(erroneous) if erroneous else None
        f1 = None if ca is None or ea is None else (2*ca*ea/(ca+ea) if ca+ea else 0.)
        result[cell] = {"answers": len(subset), "clean": len(clean), "erroneous": len(erroneous),
                        "clean_accuracy": ca, "error_exact_accuracy": ea, "f1": f1,
                        "valid_decisions": sum(r["decision_valid"].get(arm, False) for r in subset)}
    f1s = [x["f1"] for x in result.values()]
    return {"cells": result, "macro_f1": float(np.mean(f1s)) if f1s and None not in f1s else None}


def paired_intervals(rows, left, right, draws=1000):
    common_prm = [r for r in rows if r["cell"].startswith("prm") and r["valid"].get(left) and r["valid"].get(right)]
    pb_rows = [r for r in rows if r["cell"].startswith("pb_")]
    def index_groups(records):
        strata = {}
        for row in records:
            strata.setdefault(row["cell"], {}).setdefault(row["group_id"], []).append(row)
        return strata
    def sample(strata, rng):
        output = []
        for groups in strata.values():
            ids = sorted(groups)
            for index in rng.integers(len(ids), size=len(ids)):
                output.extend(groups[ids[index]])
        return output
    prm_groups, pb_groups = index_groups(common_prm), index_groups(pb_rows)
    rng = np.random.default_rng(2026090706)
    prm_delta, pb_delta = [], []
    for _ in range(draws):
        if common_prm:
            chosen = sample(prm_groups, rng)
            a, b = prm_metric(chosen, left)["auroc"], prm_metric(chosen, right)["auroc"]
            if a is not None and b is not None:
                prm_delta.append(a-b)
        chosen = sample(pb_groups, rng)
        a, b = pb_metric(chosen, left)["macro_f1"], pb_metric(chosen, right)["macro_f1"]
        if a is not None and b is not None:
            pb_delta.append(a-b)
    return {"unit": "source group, stratified by cell", "seed": 2026090706,
            "prm_common_valid_ci95": np.quantile(prm_delta, [.025, .975]).tolist() if prm_delta else None,
            "prm_valid_draws": len(prm_delta),
            "pb_all_population_ci95": np.quantile(pb_delta, [.025, .975]).tolist() if pb_delta else None,
            "pb_valid_draws": len(pb_delta)}


def evaluate():
    prepared = load_json(OUT / "PREPARED.json")
    verify_sources(prepared)
    frozen = load_json(OUT / "SCORES_FROZEN.json")
    if frozen["prepared_sha256"] != sha(OUT / "PREPARED.json") or frozen["labels_decoded"] is not False:
        raise RuntimeError("INVALID_SCORE_FREEZE")
    for filename, expected in frozen["files"].items():
        if sha(filename) != expected:
            raise RuntimeError(f"SCORE_DRIFT: {filename}")
    release = load_json(OUT / "RELEASE.json")
    if sha(OUT / "RELEASE.json") != prepared["release_manifest_sha256"]:
        raise RuntimeError("RELEASE_DRIFT")
    labels, positions = {}, {}
    for cell in PILOT_CELLS:
        info = release["cells"][cell]
        if sha(info["label_path"]) != info["label_opaque_sha256"]:
            raise RuntimeError("LABEL_FILE_DRIFT")
        with np.load(info["label_path"], allow_pickle=False) as bundle:
            labels[cell] = {k: bundle[k] for k in bundle.files}
        ids = list(map(str, labels[cell]["row_ids"]))
        if len(ids) != len(set(ids)):
            raise RuntimeError("DUPLICATE_LABEL_IDS")
        positions[cell] = {rid: i for i, rid in enumerate(ids)}
    rows, fit_counts = [], {}
    for record in prepared["selected"]:
        cell, uid = record["cell"], record["uid"]
        meta = load_json(OUT / "scores" / f"{uid}.json")
        i = positions[cell][record["row_id"]]
        if cell.startswith("prm"):
            a, b = map(int, labels[cell]["step_flag_offsets"][[i, i+1]])
            target = labels[cell]["step_error_flags"][a:b]
        else:
            target = int(labels[cell]["first_error"][i])
        row = {**record, "target": target, "scores": {}, "valid": {}, "decision_valid": {}, "predictions": {}}
        with np.load(OUT / "scores" / f"{uid}.npz", allow_pickle=False) as arrays:
            for arm in ARM_IDS:
                detail = meta["report"]["methods"].get(arm, {"status": "MISSING"})
                fit_counts.setdefault(arm, {})[detail["status"]] = fit_counts.setdefault(arm, {}).get(detail["status"], 0) + 1
                key = f"{arm}__step"
                row["valid"][arm] = bool(detail.get("valid") and key in arrays.files)
                row["decision_valid"][arm] = bool(row["valid"][arm] and detail.get("readout_valid"))
                if key in arrays.files:
                    values = arrays[key]
                    if len(values) != record["steps"] or not np.isfinite(values).all():
                        raise RuntimeError("INVALID_STEP_SCORES")
                    if cell.startswith("prm") and len(values) != len(target):
                        raise RuntimeError("TARGET_STEP_ALIGNMENT")
                    row["scores"][arm] = values
                row["predictions"][arm] = detail.get("readout", {}).get("prediction")
        rows.append(row)
    summary = {arm: {"prm": prm_metric(rows, arm), "pb": pb_metric(rows, arm)} for arm in ARM_IDS}
    pairs = [("moments27_local8__iu", "legacy_fixed32__iu"),
             ("global30_local32__iu", "legacy_fixed32__iu"),
             ("moments27_local32__iu", "global30_local32__iu"),
             ("moments27_local8__iu", "moments27_local32__iu"),
             ("moments27_local8__iu", "moments27_local8__equal"),
             ("moments27_local8__iu", "entropy_mean_w8")]
    for rep in REPS[1:]:
        pairs += [(f"{rep}__joint_graph010", f"{rep}__{other}") for other in ("joint_lambda0", "joint_graph_permuted", "iu")]
    paired = {}
    for left, right in pairs:
        common = [r for r in rows if r["valid"].get(left) and r["valid"].get(right)]
        paired[f"{left}_minus_{right}"] = {"common_ids": [r["uid"] for r in common],
                                            "left_prm": prm_metric(common, left), "right_prm": prm_metric(common, right),
                                            "left_pb": pb_metric(rows, left), "right_pb": pb_metric(rows, right),
                                            "uncertainty": paired_intervals(rows, left, right)}
        print(f"paired bootstrap completed: {left} minus {right}", flush=True)
    save_json(OUT / "EVALUATION.json", {"status": "RETROSPECTIVE_DEVELOPMENT_PILOT", "labels_decoded": True,
                                        "score_freeze_sha256": sha(OUT / "SCORES_FROZEN.json"),
                                        "metrics": summary, "fit_counts": fit_counts, "paired": paired,
                                        "rows": rows})
    print(json.dumps({arm: {"prm": result["prm"], "pb_macro": result["pb"]["macro_f1"]}
                      for arm, result in summary.items()}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("prepare", "scores", "evaluate"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    {"prepare": prepare, "scores": lambda: scores(args.workers), "evaluate": evaluate}[args.phase]()


if __name__ == "__main__":
    main()
