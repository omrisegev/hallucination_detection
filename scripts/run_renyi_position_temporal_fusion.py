"""Resumable Renyi/varentropy fusion with whole-answer position-varying weights.

The four token views and all hyperparameters are frozen in
docs/experiments/RENYI_POSITION_TEMPORAL_FUSION_V1.md.  External fits receive
only feature moments and source-group/fold metadata; benchmark labels never
enter a fit API.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import inspect
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import traceback

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_direct_probability_temporal as evaluator
from spectral_utils import renyi_position_fusion as model


OUT = ROOT / "results/renyi_position_temporal_fusion_v1"
EXPECTED_ANSWERS = 13_769
SMOKE_ANSWERS = 27
BOOTSTRAP_DRAWS = 10_000
PRIMARY_CI = 1.0 - 0.05 / 3.0
INVARIANT_METHODS = model.SINGLE_METHODS + ("equal", "local_iu")
REFERENCE_FILES = (
    ROOT / "results/varentropy_contribution_fusion_v1/SCORES.npz",
    ROOT / "results/direct_probability_temporal_v3/SCORES.npz",
)
REFERENCE_KEYS = {
    "reference__ve1_raw": (REFERENCE_FILES[0], "steps__k15__raw"),
    "reference__ve1_iu": (REFERENCE_FILES[0], "steps__k15__iu"),
    "reference__entropy": (REFERENCE_FILES[0], "steps__entropy"),
    "reference__direct_iu": (REFERENCE_FILES[0], "steps__direct_iu"),
    "reference__temporal_current_iu": (REFERENCE_FILES[1], "steps__current__iu"),
    "reference__temporal_lag8_iu": (REFERENCE_FILES[1], "steps__lag8__iu"),
}


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def dumps(value):
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(dumps(value) + "\n", encoding="utf8")
    os.replace(temporary, path)


def packed(**arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def unpacked(blob):
    with np.load(io.BytesIO(blob), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def is_lfs_pointer(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024:
        return False
    return path.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")


def configure_sources(source, contract_root):
    """Use large raw data from source and committed gate/folds from contract root."""
    evaluator.old.configure_source_root(source)
    benchmark = contract_root / "results/localization_full_benchmark_v3"
    gate = contract_root / "results/fusion_fixed_gate_v1"
    folds = contract_root / "results/localization_source_group_audit_v1/FOLDS_V2.json"
    if not (evaluator.old.BENCH / "evaluation/JOINED.json").exists():
        evaluator.old.BENCH = benchmark
    if not (evaluator.old.FIXED_GATE / "DETECTORS.npz").exists():
        evaluator.old.FIXED_GATE = gate
    if not evaluator.old.FOLDS.exists():
        evaluator.old.FOLDS = folds


def required_inputs(source, contract_root):
    configure_sources(source, contract_root)
    paths = [
        evaluator.old.BENCH / "evaluation/JOINED.json",
        evaluator.old.BENCH / "evaluation/JOINED.npz",
        evaluator.old.FIXED_GATE / "DETECTORS.npz",
        evaluator.old.FIXED_GATE / "METRICS.json",
        evaluator.old.FOLDS,
        evaluator.old.PRMB_LABELS,
    ]
    paths.extend(path for _, path, _, _ in evaluator.source_specs())
    paths.extend(REFERENCE_FILES)
    return list(dict.fromkeys(Path(path).resolve() for path in paths))


def preflight(source, contract_root):
    paths = required_inputs(source, contract_root)
    missing = [str(path) for path in paths if not path.is_file()]
    pointers = [str(path) for path in paths if path.is_file() and is_lfs_pointer(path)]
    return dict(
        status="PASS" if not missing and not pointers else "BLOCKED_INPUTS",
        source_root=str(source),
        contract_root=str(contract_root),
        required=len(paths),
        missing=missing,
        git_lfs_pointers_without_payload=pointers,
        note="No artifact was regenerated or substituted.",
    )


def load_contract(source, contract_root):
    check = preflight(source, contract_root)
    if check["status"] != "PASS":
        raise FileNotFoundError(dumps(check))
    configure_sources(source, contract_root)
    records = json.loads(
        (evaluator.old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8")
    )["records"]
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        joined = {key: saved[key] for key in saved.files}
    if len(records) != EXPECTED_ANSWERS or len({row["uid"] for row in records}) != EXPECTED_ANSWERS:
        raise ValueError("frozen benchmark roster mismatch")
    expected_offsets = np.concatenate([[0], np.cumsum([row["steps"] for row in records])])
    np.testing.assert_array_equal(joined["offsets"], expected_offsets)
    folds = json.loads(evaluator.old.FOLDS.read_text(encoding="utf8"))["outer"]
    fold = {index: int(folds[row["group_id"]]) for index, row in enumerate(records)}
    return records, joined, fold


def smoke_selection(records):
    selected = []
    for cell in sorted({row["cell"] for row in records}):
        ordered = sorted(
            (index for index, row in enumerate(records) if row["cell"] == cell),
            key=lambda index: (records[index]["tokens"], records[index]["uid"]),
        )
        picks = sorted({0, len(ordered) // 2, min(len(ordered) - 1, int(0.95 * len(ordered)))})
        if len(picks) != 3:
            raise ValueError("smoke cell cannot supply three distinct length picks: " + cell)
        selected.extend(ordered[pick] for pick in picks)
    if len(selected) != SMOKE_ANSWERS:
        raise ValueError(f"expected {SMOKE_ANSWERS} smoke answers, got {len(selected)}")
    return sorted(selected)


def training_metadata(records, fold):
    return {
        index: dict(
            uid=str(row["uid"]),
            cell=str(row["cell"]),
            group_id=str(row["group_id"]),
            fold=int(fold[index]),
        )
        for index, row in enumerate(records)
    }


def group_weights(metadata, ids):
    groups = {}
    for index in ids:
        groups.setdefault(metadata[index]["group_id"], []).append(index)
    if not groups:
        raise ValueError("cannot weight an empty training set")
    weights = {}
    for members in groups.values():
        for index in members:
            weights[index] = 1.0 / len(groups) / len(members)
    np.testing.assert_allclose(sum(weights.values()), 1.0, atol=1e-12, rtol=0)
    return weights


def model_key(cell, excluded):
    return cell + "__exclude_" + "_".join(map(str, sorted(set(excluded))))


def excluded_sets(metadata, selected, cell):
    folds = sorted({metadata[index]["fold"] for index in selected if metadata[index]["cell"] == cell})
    result = [(fold,) for fold in folds]
    if not cell.startswith("pb_"):
        result.extend((first, second) for position, first in enumerate(folds) for second in folds[position + 1 :])
    return result


def fit_prior(statistics, metadata, cell, excluded):
    """Fit one exclusion-safe model without accepting records or labels."""
    allowed = {"uid", "cell", "group_id", "fold"}
    if any(set(row) != allowed for row in metadata.values()):
        raise ValueError("prior metadata must contain exactly uid/cell/group_id/fold")
    excluded = tuple(sorted(set(map(int, excluded))))
    if len(excluded) not in (1, 2):
        raise ValueError("one or two held folds are required")
    cell_ids = sorted(index for index in statistics if metadata[index]["cell"] == cell)
    train_ids = [index for index in cell_ids if metadata[index]["fold"] not in excluded]
    held_ids = [index for index in cell_ids if metadata[index]["fold"] in excluded]
    train_groups = sorted({metadata[index]["group_id"] for index in train_ids})
    held_groups = sorted({metadata[index]["group_id"] for index in held_ids})
    if not train_ids or set(train_groups).intersection(held_groups):
        raise ValueError("empty or source-group-overlapping external training set")
    weights = group_weights(metadata, train_ids)
    names = {"real_second", "real_mean", "shuffle_second", "shuffle_mean"}
    aggregate = {
        "real_second": np.zeros((model.BINS, len(model.FEATURE_NAMES), len(model.FEATURE_NAMES))),
        "real_mean": np.zeros((model.BINS, len(model.FEATURE_NAMES))),
        "shuffle_second": np.zeros((model.BINS, len(model.FEATURE_NAMES), len(model.FEATURE_NAMES))),
        "shuffle_mean": np.zeros((model.BINS, len(model.FEATURE_NAMES))),
    }
    digest = hashlib.sha256()
    for index in train_ids:
        if set(statistics[index]) != names:
            raise ValueError("saved regional-statistics roster mismatch")
        digest.update(str(index).encode() + b"\0" + np.float64(weights[index]).tobytes())
        for name in sorted(names):
            value = np.asarray(statistics[index][name], dtype=np.float64)
            if value.shape != aggregate[name].shape or not np.isfinite(value).all():
                raise ValueError("invalid regional statistics")
            aggregate[name] += weights[index] * value
            digest.update(np.ascontiguousarray(value).tobytes())
    fitted, fit_info = model.fit_external_model(aggregate, len(train_groups))
    info = dict(
        cell=cell,
        excluded_folds=list(excluded),
        training_ids=train_ids,
        training_groups=train_groups,
        excluded_groups=held_groups,
        training_answers=len(train_ids),
        training_group_count=len(train_groups),
        answer_weights={str(index): weights[index] for index in train_ids},
        fit_input_sha256=digest.hexdigest(),
        fit_api_accepts_no_labels=True,
        fits=fit_info,
    )
    return fitted, info


def flatten_model(fitted):
    arrays = {}
    for method in (
        "external_iu_static",
        "external_iu_position_mean",
        "external_iu_position",
        "external_iu_position_shuffled",
    ):
        for name in ("coefficients", "intercept"):
            arrays[method + "__" + name] = fitted[method][name]
    arrays["covariance__real"] = fitted["covariance"]["real"]
    arrays["covariance__shuffle"] = fitted["covariance"]["shuffle"]
    return arrays


def unflatten_model(arrays):
    fitted = {}
    for method in (
        "external_iu_static",
        "external_iu_position_mean",
        "external_iu_position",
        "external_iu_position_shuffled",
    ):
        fitted[method] = {
            "coefficients": arrays[method + "__coefficients"],
            "intercept": arrays[method + "__intercept"],
        }
    fitted["covariance"] = {
        "real": arrays["covariance__real"],
        "shuffle": arrays["covariance__shuffle"],
    }
    return fitted


def manifest(source, contract_root, selected, smoke):
    check = preflight(source, contract_root)
    if check["status"] != "PASS":
        raise FileNotFoundError(dumps(check))
    hashes = {}
    for path in required_inputs(source, contract_root):
        print("[hash]", path, flush=True)
        hashes[str(path)] = sha256_file(path)
    dependencies = [
        Path(__file__),
        ROOT / "spectral_utils/renyi_position_fusion.py",
        ROOT / "scripts/test_renyi_position_fusion.py",
        ROOT / "scripts/test_renyi_position_driver.py",
        ROOT / "docs/experiments/RENYI_POSITION_TEMPORAL_FUSION_V1.md",
    ]
    hashes.update({str(path): sha256_file(path) for path in dependencies})
    return dict(
        schema="renyi-position-temporal-fusion-v1",
        source_root=str(source),
        contract_root=str(contract_root),
        smoke=bool(smoke),
        selected_ids=list(map(int, selected)),
        methods=list(model.METHODS),
        features=list(model.FEATURE_NAMES),
        bins=model.BINS,
        borrow_alpha=model.BORROW_ALPHA,
        fit_scope="other-answer moments; equal source groups then answers; held folds excluded",
        position="fractional whole-answer clock; step boundaries only in Top10",
        gate="frozen mean entropy q=.3",
        prmscore="q=.8 with nested two-fold exclusion for prior-dependent methods",
        bootstrap=dict(draws=BOOTSTRAP_DRAWS, primary_ci=PRIMARY_CI, secondary_ci=0.95),
        primary_pairs=list(model.PRIMARY),
        hashes=hashes,
    )


def connect(output, freeze):
    output.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(output / "CHECKPOINT.sqlite")
    con.execute("pragma journal_mode=WAL")
    for statement in (
        "create table if not exists manifest(payload text not null)",
        "create table if not exists features(idx integer primary key,uid text not null,payload blob not null)",
        "create table if not exists stats(idx integer primary key,uid text not null,payload blob not null)",
        "create table if not exists models(key text primary key,payload blob not null,info text not null)",
        "create table if not exists scores(idx integer primary key,payload blob not null,health text not null,weightmaps blob not null)",
    ):
        con.execute(statement)
    previous = con.execute("select payload from manifest").fetchone()
    if previous and json.loads(previous[0]) != json_ready(freeze):
        stored = json.loads(previous[0])
        driver_key = str(Path(__file__))
        driver_test_key = str(ROOT / "scripts/test_renyi_position_driver.py")

        def without_driver(value):
            clean = dict(value)
            clean.pop("driver_resume", None)
            clean["hashes"] = {
                key: item for key, item in clean["hashes"].items() if key not in (driver_key, driver_test_key)
            }
            return clean

        old_comparable = without_driver(stored)
        new_comparable = without_driver(json_ready(freeze))
        evaluation_only_correction = False
        if old_comparable != new_comparable:
            old_without_ci = json.loads(dumps(old_comparable))
            new_without_ci = json.loads(dumps(new_comparable))
            old_without_ci["bootstrap"].pop("primary_ci", None)
            new_without_ci["bootstrap"].pop("primary_ci", None)
            score_count = con.execute("select count(*) from scores").fetchone()[0]
            evaluation_only_correction = (
                old_without_ci == new_without_ci
                and stored["bootstrap"]["primary_ci"] == 0.975
                and np.isclose(freeze["bootstrap"]["primary_ci"], 1.0 - 0.05 / 3.0)
                and score_count in (0, len(freeze["selected_ids"]))
            )
            if not evaluation_only_correction:
                con.close()
                raise ValueError("checkpoint code/input/protocol manifest changed; use a fresh output directory")
        history = list(stored.get("driver_resume", []))
        history.append(
            dict(
                previous_sha256=stored["hashes"].get(driver_key),
                current_sha256=freeze["hashes"][driver_key],
                previous_driver_test_sha256=stored["hashes"].get(driver_test_key),
                current_driver_test_sha256=freeze["hashes"][driver_test_key],
                reason=(
                    "evaluation-only correction to the frozen 98.333% primary interval; scores unchanged"
                    if evaluation_only_correction
                    else "driver-only numerical assertion/resume handling change; source and model hashes unchanged"
                ),
            )
        )
        freeze["driver_resume"] = history
        con.execute("update manifest set payload=?", (dumps(freeze),))
        con.commit()
        print("[resume] accepted a driver-only change; all source/model/protocol hashes match", flush=True)
    if not previous:
        con.execute("insert into manifest values(?)", (dumps(freeze),))
        con.commit()
    return con


def feature_payload(features, spans):
    arrays = dict(
        z=features["z"],
        spans=np.asarray(spans, dtype=np.int64),
        mean=features["mean"],
        scale=features["scale"],
        single_signs=features["single_signs"],
        fusion_signs=features["fusion_signs"],
    )
    arrays.update({"single__" + name: value for name, value in features["singles"].items()})
    return arrays


def restore_features(arrays):
    return dict(
        z=arrays["z"],
        singles={name: arrays["single__" + name] for name in model.SINGLE_METHODS},
    )


def extract(con, selected, records):
    done = {index for index, in con.execute("select idx from features")}
    selected_set = set(selected)
    detector, _ = evaluator.old._gate_contract(records)
    for cell, path, kind, dataset in evaluator.source_specs():
        indices = [index for index in selected if records[index]["cell"] == cell and index not in done]
        if not indices:
            continue
        print("[load]", cell, len(indices), flush=True)
        rows = evaluator.old._source_row_map(evaluator.old.load_pickle(path), kind=kind, dataset=dataset)
        for index in indices:
            row_meta = records[index]
            row = rows[row_meta["row_id"]]
            logprobs = np.asarray(evaluator.old._topk_payload(row)["logprobs"], dtype=np.float64)
            entropy = np.asarray(row["token_entropies"], dtype=np.float64)
            spans = np.asarray(row["step_token_spans"], dtype=np.int64)
            if logprobs.shape != (len(entropy), 50) or spans.shape != (row_meta["steps"], 2):
                raise ValueError("raw token/step alignment mismatch: " + row_meta["uid"])
            if row_meta["tokens"] != len(logprobs):
                raise ValueError("JOINED/raw token count mismatch: " + row_meta["uid"])
            if kind == "pb":
                np.testing.assert_allclose(entropy.mean(), detector[index], atol=1e-12, rtol=0)
            features = model.feature_bank(logprobs)
            statistics = model.regional_statistics(features["z"], row_meta["uid"])
            with con:
                con.execute(
                    "insert into features values(?,?,?)",
                    (index, row_meta["uid"], packed(**feature_payload(features, spans))),
                )
                con.execute(
                    "insert into stats values(?,?,?)",
                    (index, row_meta["uid"], packed(**statistics)),
                )
            done.add(index)
            if len(done) % 100 == 0 or len(selected) == SMOKE_ANSWERS:
                print("[extract]", len(done), "/", len(selected), flush=True)
        del rows
        gc.collect()
    if done != selected_set:
        raise ValueError("extracted feature roster differs from the frozen selection")


def load_cell_stats(con, metadata, selected, cell):
    return {
        index: unpacked(con.execute("select payload from stats where idx=?", (index,)).fetchone()[0])
        for index in selected
        if metadata[index]["cell"] == cell
    }


def train(con, metadata, selected):
    saved = {key for key, in con.execute("select key from models")}
    for cell in sorted({metadata[index]["cell"] for index in selected}):
        statistics = load_cell_stats(con, metadata, selected, cell)
        for excluded in excluded_sets(metadata, selected, cell):
            key = model_key(cell, excluded)
            if key in saved:
                continue
            fitted, info = fit_prior(statistics, metadata, cell, excluded)
            with con:
                con.execute(
                    "insert into models values(?,?,?)",
                    (key, packed(**flatten_model(fitted)), dumps(info)),
                )
            saved.add(key)
            print("[fit]", key, info["training_answers"], "answers", flush=True)


def replay_exclusions(con, metadata, selected):
    expected = set()
    reviewed = []
    for cell in sorted({metadata[index]["cell"] for index in selected}):
        statistics = load_cell_stats(con, metadata, selected, cell)
        for excluded in excluded_sets(metadata, selected, cell):
            key = model_key(cell, excluded)
            expected.add(key)
            row = con.execute("select payload,info from models where key=?", (key,)).fetchone()
            if row is None:
                raise ValueError("missing fitted prior: " + key)
            saved, saved_info = unflatten_model(unpacked(row[0])), json.loads(row[1])
            replay, replay_info = fit_prior(statistics, metadata, cell, excluded)
            if saved_info != replay_info:
                raise ValueError("fit provenance replay mismatch: " + key)
            for name, value in flatten_model(replay).items():
                np.testing.assert_array_equal(flatten_model(saved)[name], value)
            reviewed.append(dict(key=key, training_groups=len(replay_info["training_groups"])))
    actual = {key for key, in con.execute("select key from models")}
    if actual != expected:
        raise ValueError("unexpected or missing model keys")
    return dict(status="PASS", models=reviewed, all_exclusions_recomputed=True)


def label_firewall_review(con, records, fold, metadata, selected):
    poisoned = [
        {**row, "label": "FORBIDDEN", "labels": ["FORBIDDEN"], "target": "FORBIDDEN"}
        for row in records
    ]
    rebuilt = training_metadata(poisoned, fold)
    if rebuilt != metadata:
        raise ValueError("record labels reached prior metadata")
    reviews = []
    categories = set()
    for key, blob, info_text in con.execute("select key,payload,info from models order by key"):
        info = json.loads(info_text)
        category = ("pb" if info["cell"].startswith("pb_") else "prm", len(info["excluded_folds"]))
        if category in categories:
            continue
        statistics = load_cell_stats(con, metadata, selected, info["cell"])
        held = [
            index for index in statistics if metadata[index]["fold"] in info["excluded_folds"]
        ]
        for index in held:
            statistics[index] = {name: value + 12_345.0 for name, value in statistics[index].items()}
        replay, replay_info = fit_prior(
            statistics, rebuilt, info["cell"], tuple(info["excluded_folds"])
        )
        if replay_info != info:
            raise ValueError("held labels/features changed prior provenance")
        saved = unflatten_model(unpacked(blob))
        for name, value in flatten_model(replay).items():
            np.testing.assert_array_equal(flatten_model(saved)[name], value)
        categories.add(category)
        reviews.append(dict(category=list(category), key=key, held_answers=len(held), bitwise_unchanged=True))
    return dict(
        status="PASS",
        fit_metadata_fields=["uid", "cell", "group_id", "fold"],
        fit_api_accepts_no_labels=True,
        reviews=reviews,
    )


def score(con, metadata, selected):
    done = {index for index, in con.execute("select idx from scores")}
    priors = {
        key: unflatten_model(unpacked(blob))
        for key, blob in con.execute("select key,payload from models")
    }
    prm_folds = sorted({metadata[index]["fold"] for index in selected if not metadata[index]["cell"].startswith("pb_")})
    for index in selected:
        if index in done:
            continue
        row = metadata[index]
        arrays = unpacked(con.execute("select payload from features where idx=?", (index,)).fetchone()[0])
        features, spans = restore_features(arrays), arrays["spans"]
        outer_key = model_key(row["cell"], (row["fold"],))
        values, health, maps = model.score_answer(features, spans, priors[outer_key], row["uid"])
        payload = dict(values)
        all_health = dict(outer=health, contexts={"outer": outer_key})
        if not row["cell"].startswith("pb_"):
            for calibration_fold in prm_folds:
                if calibration_fold == row["fold"]:
                    continue
                excluded = tuple(sorted((row["fold"], calibration_fold)))
                key = model_key(row["cell"], excluded)
                nested, nested_health, _ = model.score_answer(
                    features,
                    spans,
                    priors[key],
                    row["uid"],
                    methods=model.EXTERNAL_METHODS,
                )
                suffix = "__inner_for_" + str(calibration_fold)
                payload.update({name + suffix: value for name, value in nested.items()})
                all_health["contexts"][suffix] = key
                all_health[suffix] = nested_health
        with con:
            con.execute(
                "insert into scores values(?,?,?,?)",
                (index, packed(**payload), dumps(all_health), packed(**maps)),
            )
        done.add(index)
        if len(done) % 50 == 0 or len(selected) == SMOKE_ANSWERS:
            print("[score]", len(done), "/", len(selected), flush=True)


def score_review(con, metadata, selected):
    roster = set(selected)
    for table in ("features", "stats", "scores"):
        if {index for index, in con.execute("select idx from " + table)} != roster:
            raise ValueError("incomplete atomic coverage in " + table)
    failures = []
    for index, score_blob, health_text, maps_blob in con.execute(
        "select idx,payload,health,weightmaps from scores order by idx"
    ):
        values, health, maps = unpacked(score_blob), json.loads(health_text), unpacked(maps_blob)
        feature = unpacked(con.execute("select payload from features where idx=?", (index,)).fetchone()[0])
        steps = len(feature["spans"])
        expected = set(model.METHODS)
        row = metadata[index]
        if not row["cell"].startswith("pb_"):
            folds = sorted({metadata[item]["fold"] for item in selected if metadata[item]["cell"] == row["cell"]})
            expected.update(
                name + "__inner_for_" + str(fold)
                for fold in folds
                if fold != row["fold"]
                for name in model.EXTERNAL_METHODS
            )
        if set(values) != expected or set(maps) != set(model.METHODS):
            raise ValueError("outer/inner score roster mismatch")
        for name, vector in values.items():
            if vector.shape != (steps,):
                raise ValueError("official step roster changed")
            if not np.isfinite(vector).all():
                failures.append(dict(index=index, uid=row["uid"], method=name))
        for name, weight_map in maps.items():
            if weight_map.shape != (model.BINS, len(model.FEATURE_NAMES)):
                raise ValueError("invalid outer weight-map shape")
        for context, key in health["contexts"].items():
            info = json.loads(con.execute("select info from models where key=?", (key,)).fetchone()[0])
            if row["group_id"] in info["training_groups"]:
                raise ValueError("held answer source group entered its prior")
            expected_folds = [row["fold"]] if context == "outer" else sorted(
                (row["fold"], int(context.split("_")[-1]))
            )
            if info["excluded_folds"] != expected_folds:
                raise ValueError("nested score used the wrong excluded-fold prior")
    return dict(
        status="PASS" if not failures else "FAIL",
        answers=len(selected),
        failures=failures,
        finite_outer_and_inner_required=True,
        all_source_group_exclusions_audited=True,
    )


def load_references(total_steps):
    opened = {}
    result = {}
    try:
        for name, (path, key) in REFERENCE_KEYS.items():
            if path not in opened:
                opened[path] = np.load(path, allow_pickle=False)
            result[name] = np.asarray(opened[path][key], dtype=np.float64)
            if result[name].shape != (total_steps,) or not np.isfinite(result[name]).all():
                raise ValueError("invalid frozen reference " + name)
    finally:
        for saved in opened.values():
            saved.close()
    return result


def build_calibration(predictions, scores, records, joined, fold):
    offsets = joined["offsets"]
    methods = set(model.EXTERNAL_METHODS)
    thresholds = {name: {} for name in scores}
    coverage = []
    for held_fold in sorted(set(fold.values())):
        train = [
            index
            for index, row in enumerate(records)
            if not row["cell"].startswith("pb_") and fold[index] != held_fold
        ]
        excluded_groups = {
            row["group_id"]
            for index, row in enumerate(records)
            if not row["cell"].startswith("pb_") and fold[index] == held_fold
        }
        for name in scores:
            vectors, used = [], []
            for index in train:
                start, stop = offsets[index : index + 2]
                vector = (
                    predictions[index][name + "__inner_for_" + str(held_fold)]
                    if name in methods
                    else scores[name][start:stop]
                )
                if np.isfinite(vector).all():
                    vectors.append(vector)
                    used.append(index)
            groups = {records[index]["group_id"] for index in used}
            if groups.intersection(excluded_groups) or not vectors:
                raise ValueError("empty or source-group-overlapping PRMScore calibration")
            thresholds[name][str(held_fold)] = float(np.quantile(np.concatenate(vectors), 0.8))
            coverage.append(
                dict(
                    method=name,
                    outer_fold=held_fold,
                    answers=len(used),
                    groups=len(groups),
                    nested_prior=name in methods,
                )
            )
    return thresholds, coverage


def contrast_pairs():
    pairs = list(model.PRIMARY)
    pairs.extend(
        [
            ("external_iu_position", "external_iu_static"),
            ("external_iu_position", "external_iu_position_shuffled"),
            ("local_shrink_position", "local_iu"),
            ("local_shrink_position", "local_shrink_pooled"),
            ("local_shrink_position", "local_shrink_position_shuffled"),
            ("equal", "view__ve1"),
            ("local_iu", "view__ve1"),
        ]
    )
    pairs.extend((name, "view__ve1") for name in model.EXTERNAL_METHODS)
    return list(dict.fromkeys(pairs))


def summarize_weightmaps(con):
    """Compact label-free summary of how each outer map changes over position."""
    totals = {
        name: np.zeros((model.BINS, len(model.FEATURE_NAMES)), dtype=np.float64)
        for name in model.METHODS
    }
    drift = {name: [] for name in model.METHODS}
    answers = 0
    for blob, in con.execute("select weightmaps from scores order by idx"):
        maps = unpacked(blob)
        answers += 1
        for name in model.METHODS:
            value = np.asarray(maps[name], dtype=np.float64)
            if value.shape != totals[name].shape or not np.isfinite(value).all():
                raise ValueError("weight-map summary received an invalid map")
            totals[name] += value
            drift[name].append(float(np.linalg.norm(value - value.mean(axis=0), axis=1).mean()))
    if not answers:
        raise ValueError("weight-map summary requires scored answers")
    methods = {}
    for name in model.METHODS:
        mean = totals[name] / answers
        methods[name] = dict(
            mean_by_bin=mean,
            first_bin=mean[0],
            middle_bin=mean[model.BINS // 2],
            last_bin=mean[-1],
            mean_answer_position_drift=float(np.mean(drift[name])),
            q90_answer_position_drift=float(np.quantile(drift[name], 0.9)),
        )
    return dict(
        schema="renyi-position-weight-summary-v1",
        scope="outer predictions; label-free descriptive average over all answers",
        answers=answers,
        feature_order=list(model.FEATURE_NAMES),
        methods=methods,
    )


def evaluate(con, records, joined, fold, score_audit, exclusion_audit, firewall_audit, output):
    offsets, total = joined["offsets"], int(joined["offsets"][-1])
    scores = {name: np.full(total, np.nan) for name in model.METHODS}
    predictions = {}
    for index, blob in con.execute("select idx,payload from scores order by idx"):
        predictions[index] = unpacked(blob)
        start, stop = offsets[index : index + 2]
        for name in model.METHODS:
            scores[name][start:stop] = predictions[index][name]
    references = load_references(total)
    scores.update(references)
    np.testing.assert_allclose(scores["view__ve1"], scores["reference__ve1_raw"], atol=1e-12, rtol=1e-10)
    thresholds, calibration = build_calibration(predictions, scores, records, joined, fold)
    metrics, per = evaluator.evaluate_arrays(
        records, joined, scores, calibration_thresholds=thresholds, fold_auc=True
    )
    prm = np.array([not row["cell"].startswith("pb_") for row in records])
    step_prm = np.repeat(prm, np.diff(offsets))
    for name, flat in scores.items():
        valid = step_prm & np.repeat(per[name]["valid"], np.diff(offsets)) & (joined["labels"] >= 0)
        metrics[name]["prm_pooled"] = evaluator.old.auc(joined["labels"][valid] == 1, flat[valid])
    pairs = contrast_pairs()
    contrasts = evaluator.paired_bootstrap(
        records,
        joined,
        per,
        draws=BOOTSTRAP_DRAWS,
        pairs=pairs,
        primary_pairs=set(model.PRIMARY),
        primary_ci=PRIMARY_CI,
    )
    for first, second in pairs:
        contrasts[first + "_minus_" + second]["pb_delta"] = (
            metrics[first]["pb_all8"] - metrics[second]["pb_all8"]
        )
    np.savez_compressed(output / "SCORES.npz", **{"steps__" + name: value for name, value in scores.items()})
    weight_summary = summarize_weightmaps(con)
    atomic_json(output / "CALIBRATION.json", dict(thresholds=thresholds, coverage=calibration))
    atomic_json(
        output / "METRICS.json",
        dict(schema="renyi-position-temporal-fusion-v1", metrics=metrics, weight_summary=weight_summary),
    )
    atomic_json(output / "CONTRASTS.json", contrasts)
    atomic_json(
        output / "RESULT_REVIEW.json",
        dict(
            status="PASS" if score_audit["status"] == "PASS" else "FAIL",
            answers=len(records),
            ve1_reference_reproduced=True,
            score_audit=score_audit,
            exclusion_audit=exclusion_audit,
            label_firewall=firewall_audit,
            performance_thresholds_applied=False,
        ),
    )
    with (output / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("method", "pb_all8", "pb_q4", "pb_q8", "prm_within", "prm_pooled", "prmscore_q08", "valid_answers"),
        )
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"method": name, **{key: values[key] for key in writer.fieldnames[1:]}})


def run_units():
    core = importlib.import_module("scripts.test_renyi_position_fusion").run()
    driver = importlib.import_module("scripts.test_renyi_position_driver").run()
    return dict(status="PASS", core=core, driver=driver)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    source, contract_root = args.source_root.resolve(), args.contract_root.resolve()
    output = OUT / "smoke" if args.phase == "smoke" else OUT
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "UNIT_REVIEW.json", run_units())
    check = preflight(source, contract_root)
    atomic_json(output / "PREFLIGHT.json", check)
    if args.phase == "preflight":
        print(dumps(check))
        return
    if check["status"] != "PASS":
        raise FileNotFoundError("preflight blocked: " + dumps(check))
    if args.phase == "full" and not args.allow_full:
        raise SystemExit("full run requires --allow-full after a passing real-data smoke")
    records, joined, fold = load_contract(source, contract_root)
    selected = smoke_selection(records) if args.phase == "smoke" else list(range(len(records)))
    freeze = manifest(source, contract_root, selected, args.phase == "smoke")
    con = connect(output, freeze)
    atomic_json(output / "MANIFEST.json", freeze)
    try:
        metadata = training_metadata(records, fold)
        with threadpool_limits(limits=1):
            extract(con, selected, records)
            train(con, metadata, selected)
            exclusion_audit = replay_exclusions(con, metadata, selected)
            firewall_audit = label_firewall_review(con, records, fold, metadata, selected)
            score(con, metadata, selected)
            score_audit = score_review(con, metadata, selected)
        atomic_json(output / "EXCLUSION_REVIEW.json", exclusion_audit)
        atomic_json(output / "LABEL_FIREWALL_REVIEW.json", firewall_audit)
        atomic_json(output / "SCORE_REVIEW.json", score_audit)
        if args.phase == "smoke":
            if score_audit["status"] != "PASS":
                raise ValueError("real-data smoke produced nonfinite predictions")
            atomic_json(
                output / "SMOKE_REVIEW.json",
                dict(
                    status="PASS",
                    answers=len(selected),
                    purpose="mechanics, numerical health and exclusion checks only; no ranking",
                    performance_thresholds_applied=False,
                ),
            )
        else:
            smoke_review = json.loads((OUT / "smoke/SMOKE_REVIEW.json").read_text(encoding="utf8"))
            if smoke_review.get("status") != "PASS" or smoke_review.get("answers") != SMOKE_ANSWERS:
                raise ValueError("full run requires the completed real-data 27-answer smoke")
            evaluate(con, records, joined, fold, score_audit, exclusion_audit, firewall_audit, output)
        atomic_json(output / "RUN_STATE.json", dict(status="COMPLETE", phase=args.phase, answers=len(selected)))
    finally:
        con.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        target = OUT / ("smoke" if "--phase" in sys.argv and "smoke" in sys.argv else "")
        target.mkdir(parents=True, exist_ok=True)
        atomic_json(
            target / "RUN_STATE.json",
            dict(status="FAILED", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()),
        )
        raise
