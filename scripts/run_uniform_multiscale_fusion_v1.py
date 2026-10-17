"""Run the frozen benchmark-uniform q15/q50 multiscale fusion experiment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gc
import importlib
import json
from itertools import combinations
from pathlib import Path
import sqlite3
import sys

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_direct_probability_temporal as evaluator
from scripts import run_probability_normalization_ablation_v1 as previous
from scripts import run_renyi_position_temporal_fusion as base
from spectral_utils import uniform_multiscale_fusion as model
from spectral_utils.varentropy_expansion_supervised import step_labels


OUT = ROOT / "results/uniform_multiscale_fusion_v1"
PROTOCOL = ROOT / "docs/experiments/UNIFORM_MULTISCALE_FUSION_V1.md"
EXPECTED_ANSWERS = 13_769
BOOTSTRAP_DRAWS = 10_000
METHODS = tuple(
    f"{support}_{arm}"
    for support in model.SUPPORTS
    for arm in ("raw_equal", "scale_equal", "scale_natural", "scale_simplex", "scale_simplex_centered")
) + ("reference_equal4_token_raw",)
EXTERNAL_METHODS = tuple(
    name for name in METHODS
    if "scale_" in name
)
CANDIDATES = tuple(
    f"{support}_{arm}"
    for support in model.SUPPORTS
    for arm in ("raw_equal", "scale_equal", "scale_simplex")
)
PRIMARY = (
    ("q50_raw_equal", "q15_raw_equal"),
    ("ms8_raw_equal", "q15_raw_equal"),
    ("q50_scale_equal", "q15_scale_equal"),
    ("ms8_scale_equal", "q15_scale_equal"),
    ("q15_scale_simplex", "q15_scale_equal"),
    ("q50_scale_simplex", "q50_scale_equal"),
    ("ms8_scale_simplex", "ms8_scale_equal"),
    ("ms8_scale_simplex", "ms8_scale_simplex_centered"),
)
PRIMARY_CI = 1.0 - 0.05 / len(PRIMARY)
REFERENCE_PATHS = {
    "position": ROOT / "results/renyi_position_temporal_fusion_v1/SCORES.npz",
    "experiment1": ROOT / "results/probability_normalization_ablation_v1/SCORES_FROZEN.npz",
}


def required_inputs(source: Path, contract_root: Path) -> list[Path]:
    base.configure_sources(source, contract_root)
    paths = [
        evaluator.old.BENCH / "evaluation/JOINED.json",
        evaluator.old.BENCH / "evaluation/JOINED.npz",
        evaluator.old.FIXED_GATE / "DETECTORS.npz",
        evaluator.old.FIXED_GATE / "METRICS.json",
        evaluator.old.FOLDS,
        evaluator.old.PRMB_LABELS,
        *[path for _, path, _, _ in evaluator.source_specs()],
        *REFERENCE_PATHS.values(),
    ]
    return list(dict.fromkeys(Path(path).resolve() for path in paths))


def preflight(source: Path, contract_root: Path) -> dict:
    paths = required_inputs(source, contract_root)
    missing = [str(path) for path in paths if not path.is_file()]
    pointers = [str(path) for path in paths if path.is_file() and base.is_lfs_pointer(path)]
    units = importlib.import_module("scripts.test_uniform_multiscale_fusion").run()
    return {
        "schema": "uniform-multiscale-fusion-preflight-v1",
        "status": "PASS" if not missing and not pointers and units["status"] == "PASS" else "BLOCKED",
        "source_root": str(source),
        "contract_root": str(contract_root),
        "required": len(paths),
        "missing": missing,
        "lfs_pointers": pointers,
        "unit_review": units,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def load_contract(source: Path, contract_root: Path):
    check = preflight(source, contract_root)
    if check["status"] != "PASS":
        raise FileNotFoundError(base.dumps(check))
    base.configure_sources(source, contract_root)
    records = json.loads((evaluator.old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8"))["records"]
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
    if len(records) != EXPECTED_ANSWERS or len({row["uid"] for row in records}) != EXPECTED_ANSWERS:
        raise ValueError("frozen answer roster mismatch")
    np.testing.assert_array_equal(offsets, np.r_[0, np.cumsum([int(row["steps"]) for row in records])])
    folds = json.loads(evaluator.old.FOLDS.read_text(encoding="utf8"))["outer"]
    fold = {index: int(folds[row["group_id"]]) for index, row in enumerate(records)}
    return records, offsets, fold, check


def smoke_selection(records: list[dict], fold: dict[int, int]) -> list[int]:
    buckets = defaultdict(list)
    for index, row in enumerate(records):
        buckets[(row["cell"], fold[index])].append(index)
    selected = sorted(index for key in sorted(buckets) for index in buckets[key][:8])
    if len({records[i]["cell"] for i in selected}) != len({row["cell"] for row in records}):
        raise ValueError("smoke did not cover every cell")
    return selected


def manifest(source: Path, contract_root: Path, selected: list[int], smoke: bool) -> dict:
    hashes = {}
    for path in required_inputs(source, contract_root):
        print("[hash]", path, flush=True)
        hashes[str(path)] = base.sha256_file(path)
    dependencies = [Path(__file__), ROOT / "spectral_utils/uniform_multiscale_fusion.py",
                    ROOT / "scripts/test_uniform_multiscale_fusion.py", PROTOCOL]
    hashes.update({str(path): base.sha256_file(path) for path in dependencies})
    return {
        "schema": "uniform-multiscale-fusion-v1",
        "source_root": str(source),
        "contract_root": str(contract_root),
        "selected_ids": list(map(int, selected)),
        "smoke": bool(smoke),
        "methods": list(METHODS),
        "external_methods": list(EXTERNAL_METHODS),
        "candidates": list(CANDIDATES),
        "primary_pairs": [list(pair) for pair in PRIMARY],
        "feature_names": list(model.FEATURE_NAMES),
        "fit_scope": "one shared PB+PRMB model; equal panel/cell/group/answer; other-fold labels",
        "readout": "per-view Top10 token mean, then step-level fusion",
        "simplex": {"epsilon": model.WEIGHT_EPSILON, "ridge": model.RIDGE, "solver": "SLSQP"},
        "gate": "frozen mean entropy q=.3",
        "prmscore": "q=.8 with two-fold-excluded external models",
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "primary_ci": PRIMARY_CI},
        "hashes": hashes,
    }


def connect(output: Path, freeze: dict) -> sqlite3.Connection:
    output.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(output / "CHECKPOINT.sqlite")
    con.execute("pragma journal_mode=WAL")
    con.execute("create table if not exists manifest(payload text not null)")
    con.execute("create table if not exists features(idx integer primary key,uid text not null,payload blob not null,info text not null)")
    con.execute("create table if not exists models(key text primary key,payload blob not null,info text not null)")
    con.execute("create table if not exists scores(idx integer primary key,uid text not null,payload blob not null,info text not null)")
    old = con.execute("select payload from manifest").fetchone()
    if old and json.loads(old[0]) != base.json_ready(freeze):
        con.close()
        raise ValueError("checkpoint manifest mismatch; use a fresh output")
    if not old:
        con.execute("insert into manifest values(?)", (base.dumps(freeze),))
        con.commit()
    return con


def extract(con: sqlite3.Connection, selected: list[int], records: list[dict]) -> None:
    done = {index for index, in con.execute("select idx from features")}
    detector, _ = evaluator.old._gate_contract(records)
    for cell, path, kind, dataset in evaluator.source_specs():
        indices = [i for i in selected if records[i]["cell"] == cell and i not in done]
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
            if logprobs.shape != (len(entropy), 50) or spans.shape != (record["steps"], 2):
                raise ValueError("raw token/step alignment mismatch: " + record["uid"])
            if int(record["tokens"]) != len(logprobs):
                raise ValueError("frozen token count mismatch: " + record["uid"])
            if kind == "pb":
                np.testing.assert_allclose(entropy.mean(), detector[index], atol=1e-12, rtol=0)
            bank = model.feature_bank(logprobs)
            steps = model.step_bank(bank["bank"], spans)
            payload = base.packed(steps=steps, mean=bank["mean"], second=bank["second"])
            info = {
                "uid": record["uid"],
                "q15_frozen_identity_max": bank["q15_frozen_identity_max"],
                "orientation": bank["orientation"],
                "tokens": len(logprobs),
                "steps": len(spans),
            }
            with con:
                con.execute("insert into features values(?,?,?,?)", (index, record["uid"], payload, base.dumps(info)))
            done.add(index)
            if len(done) % 100 == 0:
                print("[extract]", len(done), "/", len(selected), flush=True)
        del rows
        gc.collect()
    if done != set(selected):
        raise ValueError("feature checkpoint roster mismatch")


def load_features(con: sqlite3.Connection, selected: list[int]):
    steps, stats = {}, {}
    for index in selected:
        row = con.execute("select payload from features where idx=?", (index,)).fetchone()
        if row is None:
            raise ValueError("missing feature row")
        value = base.unpacked(row[0])
        steps[index] = value["steps"]
        stats[index] = {"mean": value["mean"], "second": value["second"]}
    return steps, stats


def training_labels(records, offsets, selected, joined):
    result = {}
    for index in selected:
        if records[index]["cell"].startswith("pb_"):
            result[index] = step_labels("pb", records[index]["steps"], target=joined["target"][index])
        else:
            sl = slice(offsets[index], offsets[index + 1])
            result[index] = step_labels("prm", records[index]["steps"], labels=joined["labels"][sl])
    return result


def exclusion_sets(fold: dict[int, int], selected: list[int]) -> list[tuple[int, ...]]:
    folds = sorted({fold[i] for i in selected})
    return [(value,) for value in folds] + list(combinations(folds, 2))


def model_key(excluded: tuple[int, ...]) -> str:
    return "exclude_" + "_".join(map(str, excluded))


def fit_models(con, records, offsets, fold, selected, joined):
    saved = {key for key, in con.execute("select key from models")}
    steps, stats = load_features(con, selected)
    labels = training_labels(records, offsets, selected, joined)
    for excluded in exclusion_sets(fold, selected):
        key = model_key(excluded)
        if key in saved:
            continue
        train = [index for index in selected if fold[index] not in excluded]
        fitted = model.fit_global_scale(stats, records, train)
        arrays = {"mean": fitted.mean, "scale": fitted.scale}
        fit_info = {}
        for support, columns in model.SUPPORTS.items():
            weight, info = model.fit_simplex(steps, labels, records, train, columns, fitted.scale)
            arrays["weights__" + support] = weight
            fit_info[support] = info
        excluded_groups = {records[i]["group_id"] for i in selected if fold[i] in excluded}
        if excluded_groups.intersection(fitted.training_groups):
            raise ValueError("excluded source group entered global model")
        info = {
            "excluded_folds": list(map(int, excluded)),
            "training_groups": list(fitted.training_groups),
            "training_answers": fitted.training_answers,
            "panels": list(fitted.panels),
            "cells": list(fitted.cells),
            "one_shared_model": True,
            "label_access": "other-fold supervised step labels",
            "fits": fit_info,
        }
        with con:
            con.execute("insert into models values(?,?,?)", (key, base.packed(**arrays), base.dumps(info)))
        print("[fit]", key, len(train), "answers", flush=True)


def restore_models(con):
    arrays, info = {}, {}
    for key, blob, text in con.execute("select key,payload,info from models"):
        arrays[key] = base.unpacked(blob)
        info[key] = json.loads(text)
    return arrays, info


def score_methods(steps: np.ndarray, fitted: dict, reference: np.ndarray) -> dict[str, np.ndarray]:
    output = {}
    scale = fitted["scale"]
    for support, columns in model.SUPPORTS.items():
        equal = np.full(len(columns), 1.0 / len(columns))
        natural = model.natural_weights(columns, scale)
        simplex = fitted["weights__" + support]
        output[support + "_raw_equal"] = steps[:, columns].mean(axis=1)
        output[support + "_scale_equal"] = model.score(steps, columns, scale, equal)
        output[support + "_scale_natural"] = model.score(steps, columns, scale, natural)
        output[support + "_scale_simplex"] = model.score(steps, columns, scale, simplex)
        output[support + "_scale_simplex_centered"] = model.score(steps, columns, scale, simplex, centered=True)
    output["reference_equal4_token_raw"] = np.asarray(reference, dtype=np.float64)
    if set(output) != set(METHODS):
        raise AssertionError("method roster drift")
    return output


def score_all(con, records, offsets, fold, selected):
    done = {index for index, in con.execute("select idx from scores")}
    fitted, info = restore_models(con)
    with np.load(REFERENCE_PATHS["experiment1"], allow_pickle=False) as saved:
        reference = np.asarray(saved["steps__equal4_raw"], dtype=np.float64)
    all_folds = sorted({fold[i] for i in selected})
    for index in selected:
        if index in done:
            continue
        steps = base.unpacked(con.execute("select payload from features where idx=?", (index,)).fetchone()[0])["steps"]
        outer_key = model_key((fold[index],))
        start, stop = offsets[index:index + 2]
        payload = score_methods(steps, fitted[outer_key], reference[start:stop])
        contexts = {"outer": outer_key}
        if not records[index]["cell"].startswith("pb_"):
            for calibration_fold in all_folds:
                if calibration_fold == fold[index]:
                    continue
                key = model_key(tuple(sorted((fold[index], calibration_fold))))
                nested = score_methods(steps, fitted[key], reference[start:stop])
                suffix = "__inner_for_" + str(calibration_fold)
                payload.update({name + suffix: nested[name] for name in EXTERNAL_METHODS})
                contexts[suffix] = key
        if any(not np.isfinite(value).all() for value in payload.values()):
            raise ValueError("nonfinite score: " + records[index]["uid"])
        with con:
            con.execute("insert into scores values(?,?,?,?)", (index, records[index]["uid"], base.packed(**payload), base.dumps({"contexts": contexts})))
        done.add(index)
        if len(done) % 100 == 0:
            print("[score]", len(done), "/", len(selected), flush=True)
    if done != set(selected):
        raise ValueError("score checkpoint roster mismatch")


def reference_arrays():
    with np.load(REFERENCE_PATHS["position"], allow_pickle=False) as position, \
         np.load(REFERENCE_PATHS["experiment1"], allow_pickle=False) as exp1:
        return {
            "q15__H0lim": np.asarray(position["steps__view__H0lim"]),
            "q15__ve0": np.asarray(position["steps__view__ve0"]),
            "q15__ve0.75": np.asarray(position["steps__view__ve0.75"]),
            "q15__ve1": np.asarray(position["steps__view__ve1"]),
            "q50__ve0.75": np.asarray(exp1["steps__ve075_q50_raw"]),
            "q50__ve1": np.asarray(exp1["steps__ve1_q50_raw"]),
            "reference_equal4_token_raw": np.asarray(exp1["steps__equal4_raw"]),
        }


def score_review(con, records, offsets, fold, selected):
    references = reference_arrays()
    identity = {name: 0.0 for name in references}
    q15_token_identity = 0.0
    natural_ordering = {support: True for support in model.SUPPORTS}
    centered_ordering = {support: True for support in model.SUPPORTS}
    fitted, model_info = restore_models(con)
    context_count = 0
    for key, arrays in fitted.items():
        info = model_info[key]
        assert info["one_shared_model"] and set(info["panels"]) == {"pb", "prm"}
        excluded = set(info["excluded_folds"])
        excluded_groups = {records[i]["group_id"] for i in selected if fold[i] in excluded}
        if excluded_groups.intersection(info["training_groups"]):
            raise ValueError("held group entered model " + key)
        for support in model.SUPPORTS:
            weight = arrays["weights__" + support]
            np.testing.assert_allclose(weight.sum(), 1.0, atol=1e-10, rtol=0)
            if np.any(weight < 0) or np.any((weight > 0) & (weight < model.WEIGHT_EPSILON - 1e-10)):
                raise ValueError("simplex/epsilon contract failed")
    for index in selected:
        feature_blob, feature_info = con.execute("select payload,info from features where idx=?", (index,)).fetchone()
        score_blob, score_info = con.execute("select payload,info from scores where idx=?", (index,)).fetchone()
        features = base.unpacked(feature_blob)
        values = base.unpacked(score_blob)
        q15_token_identity = max(q15_token_identity, float(json.loads(feature_info)["q15_frozen_identity_max"]))
        start, stop = offsets[index:index + 2]
        for j, name in enumerate(model.FEATURE_NAMES):
            if name in references:
                identity[name] = max(identity[name], float(np.max(np.abs(features["steps"][:, j] - references[name][start:stop]))))
        identity["reference_equal4_token_raw"] = max(identity["reference_equal4_token_raw"], float(np.max(np.abs(values["reference_equal4_token_raw"] - references["reference_equal4_token_raw"][start:stop]))))
        for support in model.SUPPORTS:
            natural_ordering[support] &= bool(np.array_equal(
                np.argsort(values[support + "_raw_equal"], kind="stable"),
                np.argsort(values[support + "_scale_natural"], kind="stable"),
            ))
            centered_ordering[support] &= bool(np.array_equal(
                np.argsort(values[support + "_scale_simplex"], kind="stable"),
                np.argsort(values[support + "_scale_simplex_centered"], kind="stable"),
            ))
        for context, key in json.loads(score_info)["contexts"].items():
            expected = {fold[index]}
            if context != "outer":
                expected.add(int(context.rsplit("_", 1)[-1]))
            if set(model_info[key]["excluded_folds"]) != expected:
                raise ValueError("wrong model exclusion context")
            context_count += 1
    if max(identity.values()) > 2e-8 or q15_token_identity > 2e-8:
        raise ValueError("frozen representation identity failed")
    if not all(natural_ordering.values()) or not all(centered_ordering.values()):
        raise ValueError("registered affine ordering control failed")
    return {
        "schema": "uniform-multiscale-score-review-v1",
        "status": "PASS",
        "answers": len(selected),
        "identity_max_abs_step_error": identity,
        "q15_token_bank_identity_max": q15_token_identity,
        "raw_vs_natural_ordering": natural_ordering,
        "full_vs_centered_ordering": centered_ordering,
        "global_model_contexts_reviewed": context_count,
        "held_source_groups_excluded": True,
        "one_model_across_benchmarks": True,
        "labels_used_only_for_other_fold_simplex_fit": True,
    }


def materialize_scores(con, offsets, records):
    scores = {name: np.full(int(offsets[-1]), np.nan) for name in METHODS}
    predictions = {}
    for index, uid, blob in con.execute("select idx,uid,payload from scores order by idx"):
        if uid != records[index]["uid"]:
            raise ValueError("score uid mismatch")
        values = base.unpacked(blob)
        predictions[index] = values
        start, stop = offsets[index:index + 2]
        for name in METHODS:
            scores[name][start:stop] = values[name]
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("full score materialization has a gap")
    return scores, predictions


def freeze_scores(output, scores, freeze, review):
    path = output / "SCORES_FROZEN.npz"
    temporary = path.with_suffix(".npz.tmp")
    temporary.write_bytes(base.packed(**{"steps__" + name: value for name, value in scores.items()}))
    temporary.replace(path)
    record = {
        "schema": "uniform-multiscale-score-freeze-v1",
        "status": "OOF_SCORES_FROZEN_BEFORE_AGGREGATE_EVALUATION",
        "score_path": str(path),
        "score_sha256": base.sha256_file(path),
        "manifest_sha256": base.sha256_file(output / "MANIFEST.json"),
        "protocol_sha256": freeze["hashes"][str(PROTOCOL)],
        "labels_used_for_other_fold_simplex_fit": True,
        "aggregate_metrics_not_computed_in_fit_score_phase": True,
        "review": review,
    }
    base.atomic_json(output / "FROZEN_SCORES.json", record)
    return record


def calibration_thresholds(predictions, scores, records, offsets, fold):
    thresholds = {name: {} for name in METHODS}
    coverage = []
    prm = [i for i, row in enumerate(records) if not row["cell"].startswith("pb_")]
    for held in sorted({fold[i] for i in prm}):
        train = [i for i in prm if fold[i] != held]
        held_groups = {records[i]["group_id"] for i in prm if fold[i] == held}
        for name in METHODS:
            vectors = []
            groups = set()
            for index in train:
                start, stop = offsets[index:index + 2]
                value = predictions[index].get(name + "__inner_for_" + str(held), scores[name][start:stop])
                vectors.append(value)
                groups.add(records[index]["group_id"])
            if groups.intersection(held_groups):
                raise ValueError("PRMScore held group leakage")
            thresholds[name][str(held)] = float(np.quantile(np.concatenate(vectors), 0.8))
            coverage.append({"method": name, "held_fold": held, "answers": len(train), "groups": len(groups), "nested_external_fit": name in EXTERNAL_METHODS})
    return thresholds, coverage


def choose_candidate(metrics):
    best_pb = max(metrics[name]["pb_all8"] for name in CANDIDATES)
    best_prm = max(metrics[name]["prm_within"] for name in CANDIDATES)
    preference = {"scale_equal": 0, "scale_simplex": 1, "raw_equal": 2}
    rows = []
    for name in CANDIDATES:
        regret = max((best_pb - metrics[name]["pb_all8"]) / 0.005,
                     (best_prm - metrics[name]["prm_within"]) / 0.002)
        support = name.split("_", 1)[0]
        arm = name[len(support) + 1:]
        rows.append((float(regret), len(model.SUPPORTS[support]), preference[arm], name))
    rows.sort()
    return {
        "schema": "uniform-multiscale-selection-v1",
        "rule": "minimum worst normalized regret; ties fewer views, equal, simplex, raw, lexical",
        "best_pb": best_pb,
        "best_prm_within": best_prm,
        "selected": rows[0][3],
        "ranking": [{"method": name, "regret": regret, "views": views} for regret, views, _, name in rows],
        "development_only": True,
    }


def error_analysis(records, joined, offsets, per, selection):
    chosen = selection["selected"]
    cells = np.array([row["cell"] for row in records])
    pb = np.char.startswith(cells, "pb_")
    error = pb & (joined["target"] >= 0)
    comparisons = {}
    for reference in ("q15_raw_equal", "q50_raw_equal", "ms8_scale_equal", "reference_equal4_token_raw"):
        a = error & (per[chosen]["peak"] == joined["target"])
        b = error & (per[reference]["peak"] == joined["target"])
        mask = np.isfinite(per[chosen]["within"]) & np.isfinite(per[reference]["within"])
        delta = per[chosen]["within"][mask] - per[reference]["within"][mask]
        comparisons[reference] = {
            "pb_only_selected_exact": int(np.sum(a & ~b)),
            "pb_only_reference_exact": int(np.sum(~a & b)),
            "pb_both_exact": int(np.sum(a & b)),
            "prm_common_answers": int(mask.sum()),
            "prm_selected_better": int(np.sum(delta > 0)),
            "prm_reference_better": int(np.sum(delta < 0)),
            "prm_tied": int(np.sum(delta == 0)),
            "prm_mean_within_delta": float(delta.mean()),
        }
    return {"schema": "uniform-multiscale-error-analysis-v1", "selected": chosen, "comparisons": comparisons}


def evaluate(output, con, records, offsets, fold):
    frozen = json.loads((output / "FROZEN_SCORES.json").read_text(encoding="utf8"))
    if frozen["status"] != "OOF_SCORES_FROZEN_BEFORE_AGGREGATE_EVALUATION":
        raise ValueError("OOF scores were not frozen")
    score_path = Path(frozen["score_path"])
    if base.sha256_file(score_path) != frozen["score_sha256"]:
        raise ValueError("frozen score hash changed")
    with np.load(score_path, allow_pickle=False) as saved:
        scores = {name: np.asarray(saved["steps__" + name], dtype=np.float64) for name in METHODS}
    _, predictions = materialize_scores(con, offsets, records)
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        joined = {key: saved[key] for key in saved.files}
    thresholds, coverage = calibration_thresholds(predictions, scores, records, offsets, fold)
    metrics, per = evaluator.evaluate_arrays(records, joined, scores, calibration_thresholds=thresholds, fold_auc=True)
    cells = np.array([row["cell"] for row in records])
    prm_answer = ~np.char.startswith(cells, "pb_")
    prm_step = np.repeat(prm_answer, np.diff(offsets))
    for name, flat in scores.items():
        valid = prm_step & np.repeat(per[name]["valid"], np.diff(offsets)) & (joined["labels"] >= 0)
        metrics[name]["prm_pooled_oof_descriptive"] = evaluator.old.auc(joined["labels"][valid] == 1, flat[valid])
    contrasts = evaluator.paired_bootstrap(records, joined, per, draws=BOOTSTRAP_DRAWS,
        pairs=list(PRIMARY), primary_pairs=set(PRIMARY), primary_ci=PRIMARY_CI)
    for first, second in PRIMARY:
        contrasts[first + "_minus_" + second]["pb_delta"] = metrics[first]["pb_all8"] - metrics[second]["pb_all8"]
    selection = choose_candidate(metrics)
    errors = error_analysis(records, joined, offsets, per, selection)
    base.atomic_json(output / "CALIBRATION.json", {"thresholds": thresholds, "coverage": coverage})
    base.atomic_json(output / "METRICS.json", {"schema": "uniform-multiscale-metrics-v1", "scope": "adaptive full development; not confirmation", "metrics": metrics})
    base.atomic_json(output / "CONTRASTS.json", contrasts)
    base.atomic_json(output / "SELECTION.json", selection)
    base.atomic_json(output / "ERROR_ANALYSIS.json", errors)
    _, model_info = restore_models(con)
    base.atomic_json(output / "WEIGHTS.json", {"schema": "uniform-multiscale-weights-v1", "models": model_info})
    with (output / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        fields = ("method", "pb_all8", "pb_raw_exact", "prm_within", "prm_fold_auc", "prm_pooled_oof_descriptive", "prmscore_q08", "valid_answers")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in METHODS:
            writer.writerow({"method": name, **{field: metrics[name][field] for field in fields[1:]}})
    base.atomic_json(output / "RESULT_REVIEW.json", {
        "schema": "uniform-multiscale-result-review-v1", "status": "PASS",
        "score_sha256": frozen["score_sha256"], "aggregate_evaluation_after_score_freeze": True,
        "supervised_other_fold_fit_declared": True, "benchmark_specific_method_selection": False,
        "selected_by_frozen_joint_rule": selection["selected"], "development_only": True,
    })
    base.atomic_json(output / "RUN_STATE.json", {"status": "COMPLETE", "completed": len(records), "expected": len(records)})
    print("method,pb_all8,pb_raw_exact,prm_within,prm_fold_auc,prm_pooled_oof,prmscore", flush=True)
    for name in METHODS:
        value = metrics[name]
        print(
            ",".join(
                map(
                    str,
                    (
                        name,
                        value["pb_all8"],
                        value["pb_raw_exact"],
                        value["prm_within"],
                        value["prm_fold_auc"],
                        value["prm_pooled_oof_descriptive"],
                        value["prmscore_q08"],
                    ),
                )
            ),
            flush=True,
        )
    print("selected", selection["selected"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("preflight", "smoke", "fit-score", "evaluate"), default="preflight")
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    source, contract_root = args.source_root.resolve(), args.contract_root.resolve()
    records, offsets, fold, check = load_contract(source, contract_root)
    base.atomic_json(OUT / "PREFLIGHT.json", check)
    if args.phase == "preflight":
        print(base.dumps(check))
        return
    if args.phase in ("fit-score", "evaluate") and not args.allow_full:
        raise SystemExit("full phases require --allow-full after smoke")
    selected = smoke_selection(records, fold) if args.phase == "smoke" else list(range(len(records)))
    output = OUT / "smoke" if args.phase == "smoke" else OUT
    freeze = manifest(source, contract_root, selected, args.phase == "smoke")
    base.atomic_json(output / "MANIFEST.json", freeze)
    con = connect(output, freeze)
    try:
        if args.phase in ("smoke", "fit-score"):
            with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
                joined = {"labels": np.asarray(saved["labels"]), "target": np.asarray(saved["target"])}
            with threadpool_limits(limits=1):
                extract(con, selected, records)
                fit_models(con, records, offsets, fold, selected, joined)
                score_all(con, records, offsets, fold, selected)
                review = score_review(con, records, offsets, fold, selected)
            base.atomic_json(output / "SCORE_REVIEW.json", review)
            if args.phase == "smoke":
                base.atomic_json(output / "SMOKE_REVIEW.json", {"status": "PASS", "answers": len(selected), "score_review": review})
                print(base.dumps({"status": "SMOKE_PASS", "answers": len(selected)}))
                return
            scores, _ = materialize_scores(con, offsets, records)
            frozen = freeze_scores(output, scores, freeze, review)
            base.atomic_json(output / "RUN_STATE.json", {"status": "SCORES_FROZEN", "completed": len(records), "expected": len(records)})
            print(base.dumps({"status": frozen["status"], "sha256": frozen["score_sha256"]}))
            return
        evaluate(output, con, records, offsets, fold)
    finally:
        con.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        base.atomic_json(OUT / "RUN_STATE.json", {"status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", "error": f"{type(error).__name__}: {error}"})
        raise
