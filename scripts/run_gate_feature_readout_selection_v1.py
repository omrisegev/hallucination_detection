"""Select one q=.3 answer-gate feature/readout for the frozen q15 locator."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import importlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_direct_probability_temporal as evaluator
from scripts import run_renyi_position_temporal_fusion as base
from spectral_utils import gate_feature_readout as model
from spectral_utils import fixed_gate_readout as gate
from spectral_utils.historical_fusion_evaluation import pb_metrics


OUT = ROOT / "results/gate_feature_readout_selection_v1"
PROTOCOL = ROOT / "docs/experiments/GATE_FEATURE_READOUT_SELECTION_V1.md"
UNIFORM = ROOT / "results/uniform_multiscale_fusion_v1"
FINALIST = ROOT / "results/selected_q15_finalist_replay_v1"
FIXED_GATE = ROOT / "results/fusion_fixed_gate_v1"
EXPECTED_ANSWERS = 13_769
EXPECTED_PB = 6_800
EXPECTED_STEPS = 145_597
Q = 0.3
BOOTSTRAP_DRAWS = 10_000


def configure():
    manifest = json.loads((UNIFORM / "MANIFEST.json").read_text(encoding="utf8"))
    source = Path(manifest["source_root"])
    contract_root = Path(manifest["contract_root"])
    base.configure_sources(source, contract_root)
    return source, contract_root


def required_inputs() -> list[Path]:
    source, contract_root = configure()
    paths = [
        PROTOCOL,
        Path(__file__),
        ROOT / "spectral_utils/gate_feature_readout.py",
        ROOT / "scripts/test_gate_feature_readout.py",
        evaluator.old.BENCH / "evaluation/JOINED.json",
        evaluator.old.BENCH / "evaluation/JOINED.npz",
        evaluator.old.FOLDS,
        FIXED_GATE / "DETECTORS.npz",
        FIXED_GATE / "METRICS.json",
        FINALIST / "SCORES_FROZEN.npz",
        FINALIST / "RESULT_REVIEW.json",
    ]
    paths.extend(path for cell, path, kind, _ in evaluator.source_specs() if kind == "pb")
    return list(dict.fromkeys(Path(path).resolve() for path in paths))


def preflight() -> dict:
    paths = required_inputs()
    missing = [str(path) for path in paths if not path.is_file()]
    pointers = [str(path) for path in paths if path.is_file() and base.is_lfs_pointer(path)]
    units = importlib.import_module("scripts.test_gate_feature_readout").run()
    finalist_review = None
    if (FINALIST / "RESULT_REVIEW.json").is_file():
        finalist_review = json.loads((FINALIST / "RESULT_REVIEW.json").read_text(encoding="utf8"))["status"]
    return {
        "schema": "gate-feature-readout-preflight-v1",
        "status": "PASS" if not missing and not pointers and units["status"] == finalist_review == "PASS" else "BLOCKED",
        "required": len(paths),
        "missing": missing,
        "lfs_pointers": pointers,
        "unit_review": units,
        "finalist_review": finalist_review,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def load_metadata():
    configure()
    records = json.loads(
        (evaluator.old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8")
    )["records"]
    if len(records) != EXPECTED_ANSWERS or len({row["uid"] for row in records}) != len(records):
        raise ValueError("frozen answer roster mismatch")
    folds = json.loads(evaluator.old.FOLDS.read_text(encoding="utf8"))["outer"]
    outer = np.asarray([int(folds[row["group_id"]]) for row in records], dtype=np.int64)
    pb = np.asarray([row["cell"].startswith("pb_") for row in records])
    if int(pb.sum()) != EXPECTED_PB or set(outer[pb]) != set(range(5)):
        raise ValueError("frozen PB/fold coverage mismatch")
    return records, outer, pb


def smoke_selection(records, outer, pb):
    buckets = defaultdict(list)
    for index in np.flatnonzero(pb):
        buckets[(records[index]["cell"], int(outer[index]))].append(int(index))
    selected = sorted(members[0] for _, members in sorted(buckets.items()))
    if len(selected) != 40:
        raise ValueError("smoke must cover eight cells by five folds")
    return selected


def manifest(selected, smoke):
    hashes = {}
    for path in required_inputs():
        print("[hash]", path, flush=True)
        hashes[str(path)] = base.sha256_file(path)
    return {
        "schema": "gate-feature-readout-selection-v1",
        "status": "FROZEN",
        "selected_ids": list(map(int, selected)),
        "smoke": bool(smoke),
        "signals": list(model.SIGNAL_NAMES),
        "readouts": list(model.READOUT_NAMES),
        "methods": list(model.METHODS),
        "baseline": model.BASELINE,
        "locator": "selected_q15_raw_per_view_top10, bitwise frozen",
        "gate_rule": "other-fold global detector quantile q=.3; detector >= threshold opens",
        "selection": "max PB all-eight macro; exact tie worst cell then readout preference then lexical",
        "selection_uses_development_labels": True,
        "hashes": hashes,
    }


def connect(output: Path, freeze: dict):
    output.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(output / "CHECKPOINT.sqlite")
    con.execute("pragma journal_mode=WAL")
    con.execute("create table if not exists manifest(payload text not null)")
    con.execute("create table if not exists detectors(idx integer primary key,uid text not null,payload blob not null)")
    previous = con.execute("select payload from manifest").fetchone()
    if previous and json.loads(previous[0]) != base.json_ready(freeze):
        con.close()
        raise ValueError("gate checkpoint manifest drift; use a fresh output")
    if not previous:
        con.execute("insert into manifest values(?)", (base.dumps(freeze),))
        con.commit()
    return con


def extract(con, records, selected):
    done = {index for index, in con.execute("select idx from detectors")}
    expected = set(map(int, selected))
    fixed = np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False)["entropy_mean"]
    for cell, path, kind, dataset in evaluator.source_specs():
        indices = [index for index in selected if records[index]["cell"] == cell and index not in done]
        if not indices:
            continue
        if kind != "pb":
            raise ValueError("gate extraction must remain PB-only")
        print("[load]", cell, len(indices), flush=True)
        rows = evaluator.old._source_row_map(evaluator.old.load_pickle(path), kind=kind, dataset=dataset)
        for index in indices:
            record = records[index]
            row = rows[record["row_id"]]
            logprobs = np.asarray(evaluator.old._topk_payload(row)["logprobs"], dtype=np.float64)
            entropy = np.asarray(row["token_entropies"], dtype=np.float64)
            spans = np.asarray(row["step_token_spans"], dtype=np.int64)
            if logprobs.shape != (len(entropy), 50) or spans.shape != (int(record["steps"]), 2):
                raise ValueError("raw gate alignment mismatch: " + record["uid"])
            detectors = model.answer_detectors(logprobs, entropy, spans)
            if detectors[model.BASELINE] != float(fixed[index]):
                raise ValueError("entropy-mean baseline did not replay bitwise: " + record["uid"])
            with con:
                con.execute(
                    "insert into detectors values(?,?,?)",
                    (index, record["uid"], base.packed(**{name: np.asarray(value) for name, value in detectors.items()})),
                )
            done.add(index)
            if len(done) % 100 == 0 or len(expected) == 40:
                print("[extract]", len(done), "/", len(expected), flush=True)
        del rows
        gc.collect()
    if done != expected:
        raise ValueError("gate extraction roster mismatch")


def materialize(con, records, selected):
    output = {name: np.full(len(records), np.nan) for name in model.METHODS}
    for index, uid, blob in con.execute("select idx,uid,payload from detectors order by idx"):
        if uid != records[index]["uid"] or index not in set(selected):
            raise ValueError("gate detector identity mismatch")
        values = base.unpacked(blob)
        for name in model.METHODS:
            output[name][index] = float(values[name])
    if any(not np.isfinite(value[selected]).all() for value in output.values()):
        raise ValueError("gate detector archive contains a gap")
    return output


def freeze_detectors(output, detectors, freeze):
    path = output / "DETECTORS_FROZEN.npz"
    temporary = path.with_suffix(".npz.tmp")
    temporary.write_bytes(base.packed(**detectors))
    temporary.replace(path)
    record = {
        "schema": "gate-detector-freeze-v1",
        "status": "DETECTORS_FROZEN_BEFORE_TARGET_EVALUATION",
        "path": str(path),
        "sha256": base.sha256_file(path),
        "manifest_sha256": base.sha256_file(output / "MANIFEST.json"),
        "selection_not_performed_during_extraction": True,
    }
    base.atomic_json(output / "FROZEN_DETECTORS.json", record)
    return record


def _auc_by_cell(detector, target, cells):
    result = {}
    for cell in sorted(set(cells)):
        mask = cells == cell
        result[cell] = evaluator.old.auc(target[mask] >= 0, detector[mask])
    result["mean"] = float(np.mean(list(result.values())))
    return result


def choose(metrics):
    readout_preference = {"token_top10": 0, "mean_step_top10": 1, "token_mean": 2}
    ranked = []
    for name in model.METHODS:
        value = metrics[name]
        readout = next(item for item in model.READOUT_NAMES if name.endswith("__" + item))
        worst = min(cell["f1"] for cell in value["cells"].values())
        ranked.append((-value["macros"]["all"], -worst, readout_preference[readout], name))
    ranked.sort()
    point_winner = ranked[0][3]
    selected = point_winner if metrics[point_winner]["macros"]["all"] > metrics[model.BASELINE]["macros"]["all"] else model.BASELINE
    return {
        "schema": "gate-feature-readout-selection-v1",
        "rule": "max all-eight macro; exact tie worst-cell, readout preference, lexical; promote only over baseline point",
        "point_winner": point_winner,
        "selected": selected,
        "baseline": model.BASELINE,
        "ranking": [
            {"method": name, "pb_all8": metrics[name]["macros"]["all"], "worst_cell": -worst}
            for _, worst, _, name in ranked
        ],
        "development_selection": True,
        "not_model_transfer_confirmation": True,
    }


def evaluate(output, records, outer, pb, detectors):
    frozen = json.loads((output / "FROZEN_DETECTORS.json").read_text(encoding="utf8"))
    if base.sha256_file(Path(frozen["path"])) != frozen["sha256"]:
        raise ValueError("frozen gate detector hash changed")
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        target = np.asarray(saved["target"], dtype=np.int64)
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
    with np.load(FINALIST / "SCORES_FROZEN.npz", allow_pickle=False) as saved:
        step_score = np.asarray(saved["steps__selected_q15_raw_per_view_top10"], dtype=np.float64)
    if len(step_score) != EXPECTED_STEPS:
        raise ValueError("frozen finalist step vector mismatch")
    peak = np.full(len(records), -1, dtype=np.int64)
    valid = np.zeros(len(records), dtype=bool)
    for index in np.flatnonzero(pb):
        value = step_score[offsets[index]:offsets[index + 1]]
        if len(value) and np.isfinite(value).all():
            valid[index] = True
            peak[index] = int(np.argmax(value))

    cells = np.asarray([row["cell"] for row in records])[pb]
    groups = np.asarray([row["group_id"] for row in records])[pb]
    target_pb = target[pb]
    metrics = {}
    predictions = {}
    for name in model.METHODS:
        detector = detectors[name][pb]
        prediction, gate_valid, thresholds = gate.nested_gate(
            detector, peak[pb], valid[pb], target_pb, cells, outer[pb], "quantile", q=Q
        )
        summary = pb_metrics(target_pb, prediction, gate_valid, cells)
        erroneous = target_pb >= 0
        clean = target_pb < 0
        metrics[name] = {
            "macros": summary["macros"],
            "cells": summary["cells"],
            "clean_accuracy": float(np.mean(prediction[clean] == -1)),
            "error_exact_accuracy": float(np.mean(prediction[erroneous] == target_pb[erroneous])),
            "error_called_clean": int(np.sum(erroneous & (prediction == -1))),
            "clean_false_alarm": int(np.sum(clean & (prediction != -1))),
            "exact_peak_suppressed": int(np.sum(erroneous & (peak[pb] == target_pb) & (prediction == -1))),
            "thresholds": {str(key): float(value) for key, value in thresholds.items()},
            "detector_auc": _auc_by_cell(detector, target_pb, cells),
            "answers": int(len(target_pb)),
        }
        predictions[name] = (prediction, gate_valid)

    selection = choose(metrics)
    selected = selection["selected"]
    selected_prediction, selected_valid = predictions[selected]
    baseline_prediction, baseline_valid = predictions[model.BASELINE]
    bootstrap = gate.paired_group_bootstrap(
        target_pb, cells, groups,
        selected_valid, selected_prediction,
        baseline_valid, baseline_prediction,
        draws=BOOTSTRAP_DRAWS,
    )
    selected_modes = gate.error_modes(target_pb, selected_prediction, peak[pb], selected_valid, cells)
    baseline_modes = gate.error_modes(target_pb, baseline_prediction, peak[pb], baseline_valid, cells)
    errors = {
        "selected": selected,
        "baseline": model.BASELINE,
        "selected_modes": selected_modes,
        "baseline_modes": baseline_modes,
        "prediction_disagreements": int(np.sum(selected_prediction != baseline_prediction)),
        "selected_only_correct": int(np.sum((selected_prediction == target_pb) & (baseline_prediction != target_pb))),
        "baseline_only_correct": int(np.sum((baseline_prediction == target_pb) & (selected_prediction != target_pb))),
    }
    base.atomic_json(output / "METRICS.json", {"schema": "gate-feature-readout-metrics-v1", "metrics": metrics})
    base.atomic_json(output / "SELECTION.json", selection)
    base.atomic_json(output / "SELECTED_CONTRAST.json", {"scope": "post-selection descriptive", "selected_minus_baseline": bootstrap})
    base.atomic_json(output / "ERROR_ANALYSIS.json", errors)
    with (output / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ("method", "pb_all8", "pb_q4", "pb_q8", "worst_cell", "clean_accuracy", "error_exact_accuracy", "detector_auc_mean")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in model.METHODS:
            value = metrics[name]
            writer.writerow({
                "method": name,
                "pb_all8": value["macros"]["all"],
                "pb_q4": value["macros"]["q4"],
                "pb_q8": value["macros"]["q8"],
                "worst_cell": min(cell["f1"] for cell in value["cells"].values()),
                "clean_accuracy": value["clean_accuracy"],
                "error_exact_accuracy": value["error_exact_accuracy"],
                "detector_auc_mean": value["detector_auc"]["mean"],
            })
    review = {
        "schema": "gate-feature-readout-result-review-v1",
        "status": "PASS",
        "answers": len(records),
        "pb_answers": int(pb.sum()),
        "candidates": len(model.METHODS),
        "entropy_mean_bitwise_reproduced": True,
        "same_gate_definition_all_cells": True,
        "q_fixed": Q,
        "detectors_frozen_before_target_evaluation": True,
        "selection_uses_development_labels": True,
        "selected": selected,
        "detector_sha256": frozen["sha256"],
    }
    base.atomic_json(output / "RESULT_REVIEW.json", review)
    base.atomic_json(output / "RUN_STATE.json", {"status": "COMPLETE_REVIEWED", "pb_answers": int(pb.sum())})
    print("selected", selected)
    print("baseline_pb_all8", metrics[model.BASELINE]["macros"]["all"])
    print("selected_pb_all8", metrics[selected]["macros"]["all"])
    print("selected_minus_baseline", base.dumps(bootstrap))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "extract", "evaluate"), default="preflight")
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    check = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    base.atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise FileNotFoundError(base.dumps(check))
    if args.phase == "preflight":
        print(base.dumps(check))
        return
    if args.phase in ("extract", "evaluate") and not args.allow_full:
        raise SystemExit("full gate phases require --allow-full after smoke")

    records, outer, pb = load_metadata()
    selected = smoke_selection(records, outer, pb) if args.phase == "smoke" else list(map(int, np.flatnonzero(pb)))
    output = OUT / "smoke" if args.phase == "smoke" else OUT
    freeze = manifest(selected, args.phase == "smoke")
    base.atomic_json(output / "MANIFEST.json", freeze)
    con = connect(output, freeze)
    try:
        if args.phase in ("smoke", "extract"):
            with threadpool_limits(limits=1):
                extract(con, records, selected)
            detectors = materialize(con, records, selected)
            frozen = freeze_detectors(output, detectors, freeze)
            if args.phase == "smoke":
                base.atomic_json(output / "SMOKE_REVIEW.json", {
                    "status": "PASS", "answers": len(selected), "cells": 8, "folds": 5,
                    "entropy_mean_bitwise_reproduced": True, "detector_sha256": frozen["sha256"],
                })
                print(base.dumps({"status": "SMOKE_PASS", "answers": len(selected)}))
            else:
                base.atomic_json(output / "RUN_STATE.json", {"status": "DETECTORS_FROZEN", "pb_answers": len(selected)})
                print(base.dumps({"status": frozen["status"], "sha256": frozen["sha256"]}))
            return
        detectors = materialize(con, records, selected)
        evaluate(output, records, outer, pb, detectors)
    finally:
        con.close()


if __name__ == "__main__":
    main()
