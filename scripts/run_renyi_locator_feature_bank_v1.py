#!/usr/bin/env python3
"""Run the frozen factorial H1/Hinf/VE1-support locator-bank experiment."""
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
from scripts import run_integrated_q15_tail15_gate_replay_v1 as integration
from scripts import run_renyi_position_temporal_fusion as base
from scripts import run_tail15_localization_q_v1 as frozen_gate
from spectral_utils import renyi_locator_feature_bank as model


OUT = ROOT / "results/renyi_locator_feature_bank_v1"
PROTOCOL = ROOT / "docs/experiments/RENYI_LOCATOR_FEATURE_BANK_V1.md"
SOURCE = ROOT / "results/renyi_position_temporal_fusion_v1"
REFERENCE = ROOT / "results/selected_q15_finalist_replay_v1/SCORES_FROZEN.npz"
EXPECTED_ANSWERS = 13_769
EXPECTED_STEPS = 145_597
Q = 0.33
BOOTSTRAP_DRAWS = 10_000
PB_MARGIN = 0.002
PRM_MARGIN = 0.002


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(base.json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def required_inputs() -> list[Path]:
    manifest = json.loads((SOURCE / "MANIFEST.json").read_text(encoding="utf8"))
    source = Path(manifest["source_root"])
    contract = Path(manifest["contract_root"])
    base.configure_sources(source, contract)
    paths = [
        PROTOCOL,
        Path(__file__),
        ROOT / "spectral_utils/renyi_locator_feature_bank.py",
        ROOT / "scripts/test_renyi_locator_feature_bank_v1.py",
        SOURCE / "MANIFEST.json",
        SOURCE / "RESULT_REVIEW.json",
        REFERENCE,
        ROOT / "results/selected_q15_finalist_replay_v1/RESULT_REVIEW.json",
        ROOT / "results/tail15_localization_q_v1/FROZEN_METHOD.json",
        ROOT / "results/tail15_localization_q_v1/RESULT_REVIEW.json",
        ROOT / "results/gate_feature_readout_selection_v1/DETECTORS_FROZEN.npz",
        evaluator.old.BENCH / "evaluation/JOINED.json",
        evaluator.old.BENCH / "evaluation/JOINED.npz",
        evaluator.old.FOLDS,
        evaluator.old.PRMB_LABELS,
        *[path for _, path, _, _ in evaluator.source_specs()],
    ]
    return list(dict.fromkeys(path.resolve() for path in paths))


def preflight() -> dict:
    paths = required_inputs()
    missing = [str(path) for path in paths if not path.is_file()]
    pointers = [str(path) for path in paths if path.is_file() and base.is_lfs_pointer(path)]
    unit = importlib.import_module("scripts.test_renyi_locator_feature_bank_v1").run()
    checks = {}
    if not missing:
        gate = json.loads((ROOT / "results/tail15_localization_q_v1/FROZEN_METHOD.json").read_text())
        checks = {
            "source_review": json.loads((SOURCE / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "locator_review": json.loads((ROOT / "results/selected_q15_finalist_replay_v1/RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "gate_review": json.loads((ROOT / "results/tail15_localization_q_v1/RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "gate_q": gate["processbench_gate"]["q"] == Q,
            "baseline_registered": model.BASELINE in model.METHODS,
            "factorial_roster": len(model.BANKS) == 8 and len(model.METHODS) == 24,
            "joint_lsml_excluded": all("lsml" not in name for name in model.METHODS),
        }
    return {
        "schema": "renyi-locator-feature-bank-preflight-v1",
        "status": "PASS" if not missing and not pointers and unit["status"] == "PASS" and all(checks.values()) else "BLOCKED",
        "missing": missing,
        "lfs_pointers": pointers,
        "checks": checks,
        "unit": unit,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def load_contract():
    manifest = json.loads((SOURCE / "MANIFEST.json").read_text(encoding="utf8"))
    records, joined, fold = base.load_contract(Path(manifest["source_root"]), Path(manifest["contract_root"]))
    if len(records) != EXPECTED_ANSWERS or int(joined["offsets"][-1]) != EXPECTED_STEPS:
        raise ValueError("frozen answer/step roster mismatch")
    return manifest, records, joined, fold


def smoke_selection(records: list[dict], fold: dict[int, int]) -> list[int]:
    buckets = defaultdict(list)
    for index, row in enumerate(records):
        buckets[(row["cell"], fold[index])].append(index)
    selected = sorted(index for key in sorted(buckets) for index in buckets[key][:2])
    if {records[index]["cell"] for index in selected} != {row["cell"] for row in records}:
        raise ValueError("smoke selection did not cover every cell")
    return selected


def build_manifest(source_manifest: dict, selected: list[int], smoke: bool) -> dict:
    hashes = {}
    for path in required_inputs():
        print("[hash]", path, flush=True)
        hashes[str(path)] = base.sha256_file(path)
    return {
        "schema": "renyi-locator-feature-bank-v1",
        "status": "PROTOCOL_FROZEN",
        "source_root": source_manifest["source_root"],
        "contract_root": source_manifest["contract_root"],
        "population": {"selected_answers": len(selected), "full_answers": EXPECTED_ANSWERS, "full_steps": EXPECTED_STEPS},
        "smoke": smoke,
        "selected_ids": selected,
        "features": list(model.ALL_FEATURES),
        "banks": [{"name": bank.name, "features": [model.ALL_FEATURES[index] for index in bank.indices]} for bank in model.BANKS],
        "solvers": list(model.SOLVERS),
        "methods": list(model.METHODS),
        "baseline": model.BASELINE,
        "gate": {"feature": "tail15_mass__token_top10", "q": Q, "one_uniform_q": True},
        "selection": {"rule": "minimum worst normalized PB/PRMB regret", "pb_margin": PB_MARGIN, "prm_margin": PRM_MARGIN},
        "joint_lsml": "deferred: user-approved cost staging and structurally inadmissible four-view baseline",
        "development_only": True,
        "hashes": hashes,
    }


def connect(path: Path, manifest: dict) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("pragma journal_mode=WAL")
    con.execute("create table if not exists manifest(payload text not null)")
    con.execute("create table if not exists scores(idx integer primary key,uid text not null,payload blob not null,info text not null)")
    old = con.execute("select payload from manifest").fetchone()
    if old and json.loads(old[0]) != base.json_ready(manifest):
        con.close()
        raise ValueError("checkpoint manifest mismatch; use a fresh output")
    if not old:
        con.execute("insert into manifest values(?)", (base.dumps(manifest),))
        con.commit()
    return con


def score_answers(con: sqlite3.Connection, selected: list[int], records: list[dict], offsets: np.ndarray) -> None:
    done = {index for index, in con.execute("select idx from scores")}
    with np.load(REFERENCE, allow_pickle=False) as saved:
        reference = np.asarray(saved["steps__selected_q15_raw_per_view_top10"], dtype=np.float64)
    identity_max = 0.0
    for cell, path, kind, dataset in evaluator.source_specs():
        indices = [index for index in selected if records[index]["cell"] == cell and index not in done]
        if not indices:
            continue
        print("[load]", cell, len(indices), flush=True)
        rows = evaluator.old._source_row_map(evaluator.old.load_pickle(path), kind=kind, dataset=dataset)
        for index in indices:
            record = records[index]
            row = rows[record["row_id"]]
            logprobs = np.asarray(evaluator.old._topk_payload(row)["logprobs"], dtype=np.float64)
            entropy = np.asarray(row["token_entropies"], dtype=np.float64)
            spans = np.asarray(row["step_token_spans"], dtype=np.int64)
            if logprobs.shape != (record["tokens"], 50) or entropy.shape != (record["tokens"],) or spans.shape != (record["steps"], 2):
                raise ValueError("raw token/step alignment mismatch: " + record["uid"])
            features = model.feature_matrix(logprobs, entropy)
            values, diagnostics = {}, {}
            for bank in model.BANKS:
                for solver in model.SOLVERS:
                    name = bank.name + "__" + solver
                    values[name], diagnostics[name] = model.score_bank(features, spans, bank, solver)
            start, stop = offsets[index:index + 2]
            delta = float(np.max(np.abs(values[model.BASELINE] - reference[start:stop])))
            identity_max = max(identity_max, delta)
            if delta > 2e-8:
                raise ValueError(f"baseline identity failed for {record['uid']}: {delta}")
            summary = {
                "baseline_identity_max_abs": delta,
                "feature_signs": features["signs"],
                "feature_correlations": features["correlations"],
                "local_iu": {
                    name: {key: value for key, value in info.items() if key in {"active", "beta", "condition", "orientation_flipped", "anchor_correlation"}}
                    for name, info in diagnostics.items() if name.endswith("answer_z_local_iu")
                },
            }
            with con:
                con.execute("insert into scores values(?,?,?,?)", (index, record["uid"], base.packed(**values), base.dumps(summary)))
            done.add(index)
            if len(done) % 100 == 0:
                print("[score]", len(done), "/", len(selected), "identity", identity_max, flush=True)
        del rows
        gc.collect()
    if done != set(selected):
        raise ValueError("score checkpoint roster mismatch")


def materialize(con: sqlite3.Connection, selected: list[int], records: list[dict], offsets: np.ndarray):
    total = int(offsets[-1]) if len(selected) == len(records) else sum(records[index]["steps"] for index in selected)
    if len(selected) != len(records):
        output = {name: [] for name in model.METHODS}
        identity = []
        for index in selected:
            uid, blob, info = con.execute("select uid,payload,info from scores where idx=?", (index,)).fetchone()
            if uid != records[index]["uid"]:
                raise ValueError("uid mismatch")
            values = base.unpacked(blob)
            for name in model.METHODS:
                output[name].append(values[name])
            identity.append(json.loads(info)["baseline_identity_max_abs"])
        return {name: np.concatenate(values) for name, values in output.items()}, {"max_abs": max(identity), "answers": len(identity)}
    scores = {name: np.full(total, np.nan, dtype=np.float64) for name in model.METHODS}
    identities = []
    for index, uid, blob, info in con.execute("select idx,uid,payload,info from scores order by idx"):
        if uid != records[index]["uid"]:
            raise ValueError("uid mismatch")
        values = base.unpacked(blob)
        start, stop = offsets[index:index + 2]
        for name in model.METHODS:
            scores[name][start:stop] = values[name]
        identities.append(json.loads(info)["baseline_identity_max_abs"])
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("nonfinite or incomplete score materialization")
    return scores, {"max_abs": max(identities), "answers": len(identities)}


def apply_gate(records, joined, scores, data):
    offsets = np.asarray(joined["offsets"], dtype=np.int64)
    pb = np.asarray(data["pb"], dtype=bool)
    opened = np.asarray(data["tail_top10"] >= Q, dtype=bool)
    metrics, predictions = {}, {}
    for name, flat in scores.items():
        peak, valid = integration.peaks(flat, offsets, pb)
        peak_pb, valid_pb = peak[pb], valid[pb]
        prediction = np.where(opened & valid_pb, peak_pb, -1)
        summary = integration.summarize(data["target"], data["cells"], prediction, valid_pb)
        error = data["target"] >= 0
        summary["raw_exact"] = float(np.mean(peak_pb[error] == data["target"][error]))
        metrics[name] = summary
        predictions[name] = (prediction, valid_pb)
    return metrics, predictions


def select_method(metrics: dict) -> dict:
    best_pb = max(metrics[name]["pb_all8"] for name in model.METHODS)
    best_prm = max(metrics[name]["prm_within"] for name in model.METHODS)
    solver_order = {"raw_step_equal": 0, "scale_step_equal": 1, "answer_z_local_iu": 2}
    rows = []
    for name in model.METHODS:
        bank_name, solver = name.rsplit("__", 1)
        bank = model.BANK_BY_NAME[bank_name]
        regret = max((best_pb - metrics[name]["pb_all8"]) / PB_MARGIN, (best_prm - metrics[name]["prm_within"]) / PRM_MARGIN)
        rows.append((float(regret), len(bank.indices), 0 if bank.ve1_support == 15 else 1, solver_order[solver], name))
    rows.sort()
    return {
        "schema": "renyi-locator-feature-bank-selection-v1",
        "rule": "minimum worst normalized regret; ties fewer features, q15, raw, scale, local-IU",
        "best_pb_all8": best_pb,
        "best_prm_within": best_prm,
        "selected": rows[0][4],
        "baseline": model.BASELINE,
        "ranking": [{"method": name, "regret": regret, "views": views} for regret, views, _, _, name in rows],
        "development_only": True,
    }


def paired_pb(data, predictions, first: str, second: str) -> dict:
    groups, inverse = np.unique(data["groups"], return_inverse=True)
    rng = np.random.default_rng(2026091451)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        counts = np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
        weight = counts[inverse].astype(np.float64)
        values = [integration.pb_metrics(data["target"], *predictions[name], data["cells"], weights=weight)["macros"]["all"] for name in (first, second)]
        draws.append(values[0] - values[1])
    draws = np.asarray(draws)
    return {"draws": BOOTSTRAP_DRAWS, "ci_level": 0.95, "mean": float(draws.mean()), "interval": np.quantile(draws, [0.025, 0.975]).tolist()}


def paired_prm(records, per, first: str, second: str) -> dict:
    prm = np.asarray([not row["cell"].startswith("pb_") for row in records], dtype=bool)
    groups = np.asarray([row["group_id"] for row in records])
    common = prm & np.isfinite(per[first]["within"]) & np.isfinite(per[second]["within"])
    unique, inverse = np.unique(groups[common], return_inverse=True)
    difference = per[first]["within"][common] - per[second]["within"][common]
    sums = np.bincount(inverse, weights=difference, minlength=len(unique))
    sizes = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(2026091452)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        count = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        draws.append(float(count @ sums / (count @ sizes)))
    return {"draws": BOOTSTRAP_DRAWS, "ci_level": 0.95, "point": float(difference.mean()), "interval": np.quantile(draws, [0.025, 0.975]).tolist(), "answers": int(common.sum()), "groups": len(unique)}


def union_diagnostic(records, joined, per) -> dict:
    cells = np.asarray([row["cell"] for row in records])
    pb = np.char.startswith(cells, "pb_")
    error = pb & (joined["target"] >= 0)
    exact = np.column_stack([per[name]["peak"] == joined["target"] for name in model.METHODS])
    baseline = exact[:, model.METHODS.index(model.BASELINE)]
    union = exact.any(axis=1)
    return {
        "label_using_diagnostic_only": True,
        "error_answers": int(error.sum()),
        "baseline_exact": int(np.sum(error & baseline)),
        "union_exact": int(np.sum(error & union)),
        "additional_union_hits": int(np.sum(error & union & ~baseline)),
    }


def write_outputs(result: dict, scores: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "SCORES_FROZEN.npz", **{"steps__" + name: value for name, value in scores.items()})
    result["score_sha256"] = base.sha256_file(OUT / "SCORES_FROZEN.npz")
    atomic_json(OUT / "METRICS.json", result)
    fields = ("method", "pb_all8", "pb_raw_exact", "prm_within", "prm_fold_auc", "prm_pooled_oof_descriptive", "prmscore_q08", "regret")
    regret = {row["method"]: row["regret"] for row in result["selection"]["ranking"]}
    with (OUT / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in model.METHODS:
            value = result["metrics"][name]
            writer.writerow({"method": name, "pb_all8": value["pb_all8"], "pb_raw_exact": value["pb_raw_exact"], "prm_within": value["prm_within"], "prm_fold_auc": value["prm_fold_auc"], "prm_pooled_oof_descriptive": value["prm_pooled_oof_descriptive"], "prmscore_q08": value["prmscore_q08"], "regret": regret[name]})
    review = {
        "schema": "renyi-locator-feature-bank-result-review-v1",
        "status": "PASS",
        "answers": EXPECTED_ANSWERS,
        "steps": EXPECTED_STEPS,
        "methods": len(model.METHODS),
        "baseline_identity": result["baseline_identity"],
        "same_method_on_both_benchmarks": True,
        "gate_reused_q": Q,
        "no_fallback": True,
        "score_sha256": result["score_sha256"],
        "selection_before_integrated_replay": True,
        "development_only": True,
    }
    atomic_json(OUT / "SELECTION.json", result["selection"])
    atomic_json(OUT / "RESULT_REVIEW.json", review)
    atomic_json(OUT / "RUN_STATE.json", {"status": "COMPLETE_REVIEWED", "answers": EXPECTED_ANSWERS, "selected": result["selection"]["selected"]})


def evaluate(records, joined, scores):
    metrics, per = evaluator.evaluate_arrays(records, joined, scores, fold_auc=True)
    gate_data = frozen_gate.prepare()
    pb_metrics, predictions = apply_gate(records, joined, scores, gate_data)
    cells = np.asarray([row["cell"] for row in records])
    prm_answer = ~np.char.startswith(cells, "pb_")
    prm_steps = np.repeat(prm_answer, np.diff(joined["offsets"])) & (joined["labels"] >= 0)
    for name in model.METHODS:
        metrics[name]["pb_all8"] = pb_metrics[name]["macros"]["all"]
        metrics[name]["pb_q4"] = pb_metrics[name]["macros"]["q4"]
        metrics[name]["pb_q8"] = pb_metrics[name]["macros"]["q8"]
        metrics[name]["pb_raw_exact"] = pb_metrics[name]["raw_exact"]
        metrics[name]["pb_clean_accuracy"] = pb_metrics[name]["clean_accuracy"]
        metrics[name]["pb_error_exact_accuracy"] = pb_metrics[name]["error_exact_accuracy"]
        metrics[name]["prm_pooled_oof_descriptive"] = evaluator.old.auc(joined["labels"][prm_steps] == 1, scores[name][prm_steps])
    selection = select_method(metrics)
    selected = selection["selected"]
    contrast = {
        "selected_minus_baseline": {
            "selected": selected,
            "baseline": model.BASELINE,
            "pb_point": metrics[selected]["pb_all8"] - metrics[model.BASELINE]["pb_all8"],
            "pb_bootstrap": paired_pb(gate_data, predictions, selected, model.BASELINE),
            "prm_within_point": metrics[selected]["prm_within"] - metrics[model.BASELINE]["prm_within"],
            "prm_bootstrap": paired_prm(records, per, selected, model.BASELINE),
        }
    }
    return metrics, per, selection, contrast, union_diagnostic(records, joined, per)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "smoke", "full", "evaluate"), default="preflight")
    args = parser.parse_args()
    check = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise RuntimeError(base.dumps(check))
    if args.phase == "preflight":
        print(base.dumps(check))
        return
    source_manifest, records, joined, fold = load_contract()
    selected = smoke_selection(records, fold) if args.phase == "smoke" else list(range(len(records)))
    smoke = args.phase == "smoke"
    manifest = build_manifest(source_manifest, selected, smoke)
    path = OUT / ("SMOKE_CHECKPOINT.sqlite" if smoke else "CHECKPOINT.sqlite")
    con = connect(path, manifest)
    try:
        with threadpool_limits(limits=1):
            if args.phase != "evaluate":
                score_answers(con, selected, records, np.asarray(joined["offsets"], dtype=np.int64))
            scores, identity = materialize(con, selected, records, np.asarray(joined["offsets"], dtype=np.int64))
    finally:
        con.close()
    if smoke:
        atomic_json(OUT / "SMOKE.json", {"status": "PASS_FEASIBILITY_ONLY", "answers": len(selected), "steps": len(next(iter(scores.values()))), "methods": len(scores), "baseline_identity": identity})
        print(base.dumps(json.loads((OUT / "SMOKE.json").read_text())))
        return
    atomic_json(OUT / "MANIFEST.json", manifest)
    metrics, per, selection, contrasts, union = evaluate(records, joined, scores)
    result = {
        "schema": "renyi-locator-feature-bank-metrics-v1",
        "status": "COMPLETE",
        "metrics": metrics,
        "selection": selection,
        "contrasts": contrasts,
        "union_diagnostic": union,
        "baseline_identity": identity,
        "gate": {"feature": "tail15_mass__token_top10", "q": Q},
        "joint_lsml_run": False,
        "development_only": True,
    }
    write_outputs(result, scores)
    print("method,pb_all8,prm_within,prmscore,regret")
    regrets = {row["method"]: row["regret"] for row in selection["ranking"]}
    for name in model.METHODS:
        print(name, metrics[name]["pb_all8"], metrics[name]["prm_within"], metrics[name]["prmscore_q08"], regrets[name], sep=",")
    print("SELECTED", selection["selected"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "RUN_STATE.json", {"status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", "error": f"{type(error).__name__}: {error}"})
        raise
