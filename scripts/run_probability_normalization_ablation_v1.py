"""Run the frozen probability-normalization and retained-mass ablation v1.

Scoring and evaluation are separate phases.  The score phase reads features,
step spans and fold/source metadata only, freezes a hash-bound score archive,
and exits.  The evaluate phase verifies that archive before opening benchmark
targets through the existing frozen evaluator.
"""
from __future__ import annotations

import argparse
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
from spectral_utils import probability_normalization_ablation as model


OUT = ROOT / "results/probability_normalization_ablation_v1"
PROTOCOL = ROOT / "docs/experiments/PROBABILITY_NORMALIZATION_ABLATION_V1.md"
EXPECTED_ANSWERS = 13_769
SMOKE_ANSWERS = 27
BOOTSTRAP_DRAWS = 10_000
PRIMARY_CI = 1.0 - 0.05 / len(model.PRIMARY)
REFERENCE_PATHS = {
    "varentropy": ROOT / "results/varentropy_contribution_fusion_v1/SCORES.npz",
    "position": ROOT / "results/renyi_position_temporal_fusion_v1/SCORES.npz",
}
REFERENCE_KEYS = {
    "ve1_q15_raw": ("varentropy", "steps__k15__raw"),
    "ve1_q50_raw": ("varentropy", "steps__k50__raw"),
    "ve075_q15_raw": ("position", "steps__view__ve0.75"),
    "equal4_answer_z": ("position", "steps__equal"),
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
    units = importlib.import_module("scripts.test_probability_normalization_ablation").run()
    return dict(
        schema="probability-normalization-ablation-preflight-v1",
        status="PASS" if not missing and not pointers and units["status"] == "PASS" else "BLOCKED",
        source_root=str(source),
        contract_root=str(contract_root),
        required=len(paths),
        missing=missing,
        lfs_pointers=pointers,
        unit_review=units,
        no_packages_installed=True,
    )


def load_score_contract(source: Path, contract_root: Path):
    check = preflight(source, contract_root)
    if check["status"] != "PASS":
        raise FileNotFoundError(base.dumps(check))
    base.configure_sources(source, contract_root)
    records = json.loads(
        (evaluator.old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8")
    )["records"]
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        offsets = np.asarray(saved["offsets"], dtype=np.int64)
    if len(records) != EXPECTED_ANSWERS or len({row["uid"] for row in records}) != EXPECTED_ANSWERS:
        raise ValueError("frozen answer roster mismatch")
    expected_offsets = np.concatenate([[0], np.cumsum([int(row["steps"]) for row in records])])
    np.testing.assert_array_equal(offsets, expected_offsets)
    folds = json.loads(evaluator.old.FOLDS.read_text(encoding="utf8"))["outer"]
    fold = {index: int(folds[row["group_id"]]) for index, row in enumerate(records)}
    return records, offsets, fold, check


def manifest(source: Path, contract_root: Path, selected: list[int], smoke: bool) -> dict:
    hashes = {}
    for path in required_inputs(source, contract_root):
        print("[hash]", path, flush=True)
        hashes[str(path)] = base.sha256_file(path)
    dependencies = [
        Path(__file__),
        ROOT / "spectral_utils/probability_normalization_ablation.py",
        ROOT / "scripts/test_probability_normalization_ablation.py",
        PROTOCOL,
    ]
    hashes.update({str(path): base.sha256_file(path) for path in dependencies})
    return dict(
        schema="probability-normalization-ablation-v1",
        source_root=str(source),
        contract_root=str(contract_root),
        selected_ids=list(map(int, selected)),
        smoke=bool(smoke),
        methods=list(model.METHODS),
        local_methods=list(model.LOCAL_METHODS),
        external_methods=list(model.EXTERNAL_METHODS),
        primary_pairs=[list(pair) for pair in model.PRIMARY],
        fit_scope="same-cell other-fold moments; equal source groups then equal answers",
        readout="Top10 token mean per official step",
        gate="frozen mean entropy q=.3",
        prmscore="q=.8; two-fold exclusion for external standardizers",
        bootstrap={"draws": BOOTSTRAP_DRAWS, "primary_ci": PRIMARY_CI},
        hashes=hashes,
    )


def connect(output: Path, freeze: dict) -> sqlite3.Connection:
    output.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(output / "CHECKPOINT.sqlite")
    con.execute("pragma journal_mode=WAL")
    con.execute("create table if not exists manifest(payload text not null)")
    con.execute(
        "create table if not exists features(idx integer primary key,uid text not null,payload blob not null,info text not null)"
    )
    con.execute(
        "create table if not exists models(key text primary key,payload blob not null,info text not null)"
    )
    con.execute(
        "create table if not exists scores(idx integer primary key,uid text not null,payload blob not null,info text not null)"
    )
    previous = con.execute("select payload from manifest").fetchone()
    if previous and json.loads(previous[0]) != base.json_ready(freeze):
        con.close()
        raise ValueError("checkpoint code/input/protocol manifest mismatch; use a fresh output")
    if not previous:
        con.execute("insert into manifest values(?)", (base.dumps(freeze),))
        con.commit()
    return con


def smoke_selection(records: list[dict]) -> list[int]:
    return base.smoke_selection(records)


def _feature_payload(bank: dict, spans: np.ndarray) -> dict[str, np.ndarray]:
    payload = {
        "spans": np.asarray(spans, dtype=np.int64),
        "oriented4": np.asarray(bank["oriented4"], dtype=np.float64),
        "oriented5": np.asarray(bank["oriented5"], dtype=np.float64),
        "m4_mean": np.asarray(bank["moments4"]["mean"], dtype=np.float64),
        "m4_second": np.asarray(bank["moments4"]["second"], dtype=np.float64),
        "m5_mean": np.asarray(bank["moments5"]["mean"], dtype=np.float64),
        "m5_second": np.asarray(bank["moments5"]["second"], dtype=np.float64),
    }
    for name, token in bank["local"].items():
        payload["step__" + name] = model.step_readout(token, spans)
    return payload


def extract(con: sqlite3.Connection, selected: list[int], records: list[dict]) -> None:
    done = {index for index, in con.execute("select idx from features")}
    expected = set(selected)
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
            info = dict(
                uid=record["uid"],
                tokens=len(logprobs),
                invariance_error=bank["invariance_error"],
                tail15_mean=float(np.mean(bank["tail15"])),
                tail15_min=float(np.min(bank["tail15"])),
                tail15_max=float(np.max(bank["tail15"])),
                orientation=bank["orientation"],
            )
            with con:
                con.execute(
                    "insert into features values(?,?,?,?)",
                    (index, record["uid"], base.packed(**_feature_payload(bank, spans)), base.dumps(info)),
                )
            done.add(index)
            if len(done) % 100 == 0 or len(selected) == SMOKE_ANSWERS:
                print("[extract]", len(done), "/", len(selected), flush=True)
        del rows
        gc.collect()
    if done != expected:
        raise ValueError("feature checkpoint roster mismatch")


def _stats_for_cell(con: sqlite3.Connection, metadata: dict, selected: list[int], cell: str) -> dict:
    output = {}
    for index in selected:
        if metadata[index]["cell"] != cell:
            continue
        blob = con.execute("select payload from features where idx=?", (index,)).fetchone()[0]
        arrays = base.unpacked(blob)
        output[index] = {
            "m4_mean": arrays["m4_mean"],
            "m4_second": arrays["m4_second"],
            "m5_mean": arrays["m5_mean"],
            "m5_second": arrays["m5_second"],
        }
    return output


def _model_key(cell: str, excluded: tuple[int, ...]) -> str:
    return base.model_key(cell, excluded)


def fit_models(con: sqlite3.Connection, metadata: dict, selected: list[int]) -> None:
    saved = {key for key, in con.execute("select key from models")}
    for cell in sorted({metadata[index]["cell"] for index in selected}):
        statistics = _stats_for_cell(con, metadata, selected, cell)
        group_ids = {index: metadata[index]["group_id"] for index in statistics}
        for excluded in base.excluded_sets(metadata, selected, cell):
            key = _model_key(cell, excluded)
            if key in saved:
                continue
            train = [index for index in statistics if metadata[index]["fold"] not in excluded]
            fit4 = model.fit_grouped_standardizer(statistics, group_ids, train, "m4")
            fit5 = model.fit_grouped_standardizer(statistics, group_ids, train, "m5")
            arrays = dict(mean4=fit4.mean, scale4=fit4.scale, mean5=fit5.mean, scale5=fit5.scale)
            info = dict(
                cell=cell,
                excluded_folds=list(map(int, excluded)),
                training_groups=list(fit4.training_groups),
                training_answers=fit4.training_answers,
                equal_group_equal_answer=True,
            )
            if fit4.training_groups != fit5.training_groups or fit4.training_answers != fit5.training_answers:
                raise AssertionError("four/five-view fit roster differs")
            with con:
                con.execute("insert into models values(?,?,?)", (key, base.packed(**arrays), base.dumps(info)))
            saved.add(key)
            print("[fit]", key, fit4.training_answers, "answers", flush=True)


def _restore_standardizer(arrays: dict, width: int, info: dict) -> model.Standardizer:
    suffix = "4" if width == 4 else "5"
    return model.Standardizer(
        mean=np.asarray(arrays["mean" + suffix], dtype=np.float64),
        scale=np.asarray(arrays["scale" + suffix], dtype=np.float64),
        training_groups=tuple(info["training_groups"]),
        training_answers=int(info["training_answers"]),
    )


def _external_steps(feature: dict, fitted: dict, info: dict) -> dict[str, np.ndarray]:
    standard4 = _restore_standardizer(fitted, 4, info)
    standard5 = _restore_standardizer(fitted, 5, info)
    return {
        "equal4_fold_global_z": model.step_readout(
            model.global_equal_score(feature["oriented4"], standard4), feature["spans"]
        ),
        "equal5_tail_fold_global_z": model.step_readout(
            model.global_equal_score(feature["oriented5"], standard5), feature["spans"]
        ),
    }


def score(con: sqlite3.Connection, metadata: dict, selected: list[int]) -> None:
    done = {index for index, in con.execute("select idx from scores")}
    fitted = {}
    fitted_info = {}
    for key, blob, info_text in con.execute("select key,payload,info from models"):
        fitted[key] = base.unpacked(blob)
        fitted_info[key] = json.loads(info_text)
    prm_folds = sorted(
        {metadata[index]["fold"] for index in selected if not metadata[index]["cell"].startswith("pb_")}
    )
    for index in selected:
        if index in done:
            continue
        row = metadata[index]
        blob = con.execute("select payload from features where idx=?", (index,)).fetchone()[0]
        feature = base.unpacked(blob)
        payload = {name: feature["step__" + name] for name in model.LOCAL_METHODS}
        contexts = {}
        outer_key = _model_key(row["cell"], (row["fold"],))
        payload.update(_external_steps(feature, fitted[outer_key], fitted_info[outer_key]))
        contexts["outer"] = outer_key
        if not row["cell"].startswith("pb_"):
            for calibration_fold in prm_folds:
                if calibration_fold == row["fold"]:
                    continue
                excluded = tuple(sorted((row["fold"], calibration_fold)))
                key = _model_key(row["cell"], excluded)
                nested = _external_steps(feature, fitted[key], fitted_info[key])
                suffix = "__inner_for_" + str(calibration_fold)
                payload.update({name + suffix: values for name, values in nested.items()})
                contexts[suffix] = key
        if any(not np.isfinite(values).all() for values in payload.values()):
            raise ValueError("nonfinite score: " + row["uid"])
        with con:
            con.execute(
                "insert into scores values(?,?,?,?)",
                (index, row["uid"], base.packed(**payload), base.dumps({"contexts": contexts})),
            )
        done.add(index)
        if len(done) % 100 == 0 or len(selected) == SMOKE_ANSWERS:
            print("[score]", len(done), "/", len(selected), flush=True)


def _reference_arrays() -> dict[str, np.ndarray]:
    opened = {name: np.load(path, allow_pickle=False) for name, path in REFERENCE_PATHS.items()}
    try:
        return {
            method: np.asarray(opened[source][key], dtype=np.float64)
            for method, (source, key) in REFERENCE_KEYS.items()
        }
    finally:
        for saved in opened.values():
            saved.close()


def score_review(
    con: sqlite3.Connection,
    records: list[dict],
    offsets: np.ndarray,
    metadata: dict,
    selected: list[int],
) -> dict:
    roster = set(selected)
    for table in ("features", "scores"):
        if {index for index, in con.execute("select idx from " + table)} != roster:
            raise ValueError("incomplete " + table + " roster")
    references = _reference_arrays()
    identity_max = {name: 0.0 for name in REFERENCE_KEYS}
    invariance_max = {}
    tail_means = []
    affine_ordering = {"raw_vs_center": True, "scale_vs_answer_z": True}
    reviewed_contexts = 0
    for index in selected:
        feature_blob, feature_info_text = con.execute(
            "select payload,info from features where idx=?", (index,)
        ).fetchone()
        score_blob, score_info_text = con.execute(
            "select payload,info from scores where idx=?", (index,)
        ).fetchone()
        feature = base.unpacked(feature_blob)
        values = base.unpacked(score_blob)
        feature_info = json.loads(feature_info_text)
        score_info = json.loads(score_info_text)
        start, stop = offsets[index : index + 2]
        for name, reference in references.items():
            difference = float(np.max(np.abs(values[name] - reference[start:stop])))
            identity_max[name] = max(identity_max[name], difference)
        for name, value in feature_info["invariance_error"].items():
            invariance_max[name] = max(invariance_max.get(name, 0.0), float(value))
        tail_means.append(float(feature_info["tail15_mean"]))
        affine_ordering["raw_vs_center"] &= bool(
            np.array_equal(np.argsort(values["equal4_raw"]), np.argsort(values["equal4_center_only"]))
        )
        affine_ordering["scale_vs_answer_z"] &= bool(
            np.array_equal(np.argsort(values["equal4_scale_only"]), np.argsort(values["equal4_answer_z"]))
        )
        for context, key in score_info["contexts"].items():
            info = json.loads(con.execute("select info from models where key=?", (key,)).fetchone()[0])
            if records[index]["group_id"] in info["training_groups"]:
                raise ValueError("held source group entered global standardizer")
            expected = [metadata[index]["fold"]]
            if context != "outer":
                expected.append(int(context.rsplit("_", 1)[-1]))
            if info["excluded_folds"] != sorted(expected):
                raise ValueError("wrong fold exclusion in " + key)
            reviewed_contexts += 1
    if max(identity_max.values()) > 1e-9:
        raise ValueError("standalone/fusion identity control drifted")
    if max(invariance_max.values()) > 2e-8:
        raise ValueError("proper raw-head/conditional-head varentropy identity failed")
    if not all(affine_ordering.values()):
        raise ValueError("registered affine step-ordering invariance failed")
    return dict(
        schema="probability-normalization-score-review-v1",
        status="PASS",
        answers=len(selected),
        identity_max_abs_step_error=identity_max,
        raw_head_vs_conditional_head_max_abs_token_error=invariance_max,
        affine_step_ordering=affine_ordering,
        global_fit_contexts_reviewed=reviewed_contexts,
        held_source_groups_excluded=True,
        tail15_answer_mean_summary={
            "mean": float(np.mean(tail_means)),
            "q10": float(np.quantile(tail_means, 0.1)),
            "median": float(np.median(tail_means)),
            "q90": float(np.quantile(tail_means, 0.9)),
        },
        benchmark_targets_not_used_by_scoring_api=True,
    )


def materialize_scores(con: sqlite3.Connection, offsets: np.ndarray, records: list[dict]) -> tuple[dict, dict]:
    total = int(offsets[-1])
    scores = {name: np.full(total, np.nan, dtype=np.float64) for name in model.METHODS}
    predictions = {}
    for index, uid, blob in con.execute("select idx,uid,payload from scores order by idx"):
        if uid != records[index]["uid"]:
            raise ValueError("score checkpoint uid mismatch")
        values = base.unpacked(blob)
        predictions[index] = values
        start, stop = offsets[index : index + 2]
        for name in model.METHODS:
            scores[name][start:stop] = values[name]
    if any(not np.isfinite(values).all() for values in scores.values()):
        raise ValueError("full score materialization has a gap")
    return scores, predictions


def freeze_scores(output: Path, scores: dict, manifest_value: dict, review: dict) -> dict:
    path = output / "SCORES_FROZEN.npz"
    blob = base.packed(**{"steps__" + name: values for name, values in scores.items()})
    temporary = path.with_suffix(".npz.tmp")
    temporary.write_bytes(blob)
    temporary.replace(path)
    record = dict(
        schema="probability-normalization-score-freeze-v1",
        status="SCORES_FROZEN_BEFORE_EVALUATION",
        score_path=str(path),
        score_sha256=base.sha256_file(path),
        manifest_sha256=base.sha256_file(output / "MANIFEST.json"),
        protocol_sha256=manifest_value["hashes"][str(PROTOCOL)],
        methods=list(model.METHODS),
        review=review,
    )
    base.atomic_json(output / "FROZEN_SCORES.json", record)
    return record


def calibration_thresholds(
    predictions: dict,
    scores: dict,
    records: list[dict],
    offsets: np.ndarray,
    fold: dict,
) -> tuple[dict, list[dict]]:
    thresholds = {name: {} for name in scores}
    coverage = []
    prm = [index for index, row in enumerate(records) if not row["cell"].startswith("pb_")]
    for held_fold in sorted({fold[index] for index in prm}):
        train = [index for index in prm if fold[index] != held_fold]
        excluded_groups = {records[index]["group_id"] for index in prm if fold[index] == held_fold}
        for name in scores:
            vectors = []
            used = []
            for index in train:
                start, stop = offsets[index : index + 2]
                vector = (
                    predictions[index][name + "__inner_for_" + str(held_fold)]
                    if name in model.EXTERNAL_METHODS
                    else scores[name][start:stop]
                )
                if np.isfinite(vector).all():
                    vectors.append(vector)
                    used.append(index)
            groups = {records[index]["group_id"] for index in used}
            if not vectors or groups.intersection(excluded_groups):
                raise ValueError("invalid PRMScore calibration coverage")
            thresholds[name][str(held_fold)] = float(np.quantile(np.concatenate(vectors), 0.8))
            coverage.append(
                dict(
                    method=name,
                    held_fold=held_fold,
                    answers=len(used),
                    groups=len(groups),
                    nested_external_fit=name in model.EXTERNAL_METHODS,
                )
            )
    return thresholds, coverage


def evaluate(
    output: Path,
    con: sqlite3.Connection,
    records: list[dict],
    offsets: np.ndarray,
    fold: dict,
) -> None:
    frozen = json.loads((output / "FROZEN_SCORES.json").read_text(encoding="utf8"))
    if frozen["status"] != "SCORES_FROZEN_BEFORE_EVALUATION":
        raise ValueError("score archive was not frozen")
    score_path = Path(frozen["score_path"])
    if base.sha256_file(score_path) != frozen["score_sha256"]:
        raise ValueError("frozen score archive hash changed")
    with np.load(score_path, allow_pickle=False) as saved:
        scores = {name: np.asarray(saved["steps__" + name], dtype=np.float64) for name in model.METHODS}
    _, predictions = materialize_scores(con, offsets, records)
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        joined = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(joined["offsets"], offsets)
    thresholds, coverage = calibration_thresholds(predictions, scores, records, offsets, fold)
    metrics, per = evaluator.evaluate_arrays(
        records, joined, scores, calibration_thresholds=thresholds, fold_auc=True
    )
    cells = np.array([row["cell"] for row in records])
    prm_answer = ~np.char.startswith(cells, "pb_")
    prm_step = np.repeat(prm_answer, np.diff(offsets))
    for name, flat in scores.items():
        valid = prm_step & np.repeat(per[name]["valid"], np.diff(offsets)) & (joined["labels"] >= 0)
        metrics[name]["prm_pooled"] = evaluator.old.auc(joined["labels"][valid] == 1, flat[valid])
    secondary = [
        ("equal4_center_only", "equal4_raw"),
        ("equal4_answer_z", "equal4_scale_only"),
        ("ve075_q15_raw", "ve1_q15_raw"),
        ("tail15_raw", "ve075_q15_raw"),
    ]
    pairs = list(model.PRIMARY) + secondary
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
    base.atomic_json(output / "CALIBRATION.json", {"thresholds": thresholds, "coverage": coverage})
    base.atomic_json(
        output / "METRICS.json",
        dict(
            schema="probability-normalization-ablation-metrics-v1",
            scope="opened development population; diagnostic, not confirmation",
            metrics=metrics,
        ),
    )
    base.atomic_json(output / "CONTRASTS.json", contrasts)
    with (output / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        fields = (
            "method",
            "pb_all8",
            "pb_raw_exact",
            "prm_within",
            "prm_pooled",
            "prmscore_q08",
            "valid_answers",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in model.METHODS:
            writer.writerow({"method": name, **{field: metrics[name][field] for field in fields[1:]}})
    base.atomic_json(
        output / "RESULT_REVIEW.json",
        dict(
            schema="probability-normalization-ablation-result-review-v1",
            status="PASS",
            score_sha256=frozen["score_sha256"],
            scores_verified_before_labels=True,
            identity_review=frozen["review"],
            performance_thresholds_applied=False,
            development_labels_opened_only_in_evaluate_phase=True,
        ),
    )
    base.atomic_json(
        output / "RUN_STATE.json",
        {"status": "COMPLETE", "completed": len(records), "expected": len(records)},
    )
    print("method,pb_all8,pb_raw_exact,prm_within,prm_pooled,prmscore_q08", flush=True)
    for name in model.METHODS:
        values = metrics[name]
        print(
            ",".join(
                str(value)
                for value in (
                    name,
                    values["pb_all8"],
                    values["pb_raw_exact"],
                    values["prm_within"],
                    values["prm_pooled"],
                    values["prmscore_q08"],
                )
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, default=ROOT)
    parser.add_argument("--phase", choices=("preflight", "smoke", "score", "evaluate"), default="preflight")
    parser.add_argument("--allow-full", action="store_true")
    args = parser.parse_args()
    source = args.source_root.resolve()
    contract_root = args.contract_root.resolve()
    records, offsets, fold, check = load_score_contract(source, contract_root)
    base.atomic_json(OUT / "PREFLIGHT.json", check)
    if args.phase == "preflight":
        print(base.dumps(check))
        return
    if args.phase in ("score", "evaluate") and not args.allow_full:
        raise SystemExit("full score/evaluate requires --allow-full after a passing real-data smoke")
    selected = smoke_selection(records) if args.phase == "smoke" else list(range(len(records)))
    output = OUT / "smoke" if args.phase == "smoke" else OUT
    freeze = manifest(source, contract_root, selected, args.phase == "smoke")
    base.atomic_json(output / "MANIFEST.json", freeze)
    con = connect(output, freeze)
    metadata = base.training_metadata(records, fold)
    try:
        if args.phase in ("smoke", "score"):
            with threadpool_limits(limits=1):
                extract(con, selected, records)
                fit_models(con, metadata, selected)
                score(con, metadata, selected)
                review = score_review(con, records, offsets, metadata, selected)
            base.atomic_json(output / "SCORE_REVIEW.json", review)
            if args.phase == "smoke":
                base.atomic_json(
                    output / "SMOKE_REVIEW.json",
                    {"status": "PASS", "answers": len(selected), "score_review": review},
                )
                print(base.dumps({"status": "SMOKE_PASS", "answers": len(selected)}))
                return
            scores, _ = materialize_scores(con, offsets, records)
            frozen = freeze_scores(output, scores, freeze, review)
            base.atomic_json(
                output / "RUN_STATE.json",
                {"status": "SCORES_FROZEN", "completed": len(records), "expected": len(records)},
            )
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
        base.atomic_json(
            OUT / "RUN_STATE.json",
            {"status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED", "error": f"{type(error).__name__}: {error}"},
        )
        raise
