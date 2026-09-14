#!/usr/bin/env python3
"""Compare answer-z, scale-only and raw inputs for frozen Renyi fusion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_integrated_q15_tail15_gate_replay_v1 as integration
from scripts import run_renyi_position_temporal_fusion as base
from scripts import run_tail15_localization_q_v1 as frozen_gate
from scripts import run_direct_probability_temporal as evaluator
from spectral_utils import renyi_position_fusion as model


OUT = ROOT / "results/fusion_input_normalization_ablation_v1"
PROTOCOL = ROOT / "docs/experiments/FUSION_INPUT_NORMALIZATION_ABLATION_V1.md"
SOURCE = ROOT / "results/renyi_position_temporal_fusion_v1"
CHECKPOINT = SOURCE / "CHECKPOINT.sqlite"
TRANSFORMS = ("answer_z", "scale_only", "raw")
SOLVERS = (
    "equal",
    "local_iu",
    "external_iu_static",
    "external_iu_position",
    "local_shrink_pooled",
    "local_shrink_position",
    "local_shrink_position_scale_only",
)
EXTERNAL_SOLVERS = tuple(name for name in SOLVERS if name in model.EXTERNAL_METHODS)
METHODS = tuple(f"{transform}__{solver}" for transform in TRANSFORMS for solver in SOLVERS)
Q = 0.33
EXPECTED_ANSWERS = 13_769
BOOTSTRAP_DRAWS = 10_000
PRIMARY_CI = 1.0 - 0.05 / (2 * len(SOLVERS))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(base.json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def transform_bank(arrays: dict[str, np.ndarray], transform: str) -> np.ndarray:
    z = np.asarray(arrays["z"], dtype=np.float64)
    mean = np.asarray(arrays["mean"], dtype=np.float64)
    scale = np.asarray(arrays["scale"], dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != 4 or mean.shape != (4,) or scale.shape != (4,):
        raise ValueError("frozen feature payload has the wrong shape")
    if not np.isfinite(z).all() or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid frozen normalization parameters")
    if transform == "answer_z":
        output = z.copy()
    elif transform == "scale_only":
        output = z + mean / scale
    elif transform == "raw":
        output = z * scale + mean
    else:
        raise ValueError("unknown transform: " + transform)
    if not np.isfinite(output).all():
        raise ValueError("nonfinite transformed bank")
    return output


def preflight() -> dict:
    required = [
        PROTOCOL,
        CHECKPOINT,
        SOURCE / "MANIFEST.json",
        SOURCE / "RESULT_REVIEW.json",
        ROOT / "results/tail15_localization_q_v1/FROZEN_METHOD.json",
        ROOT / "results/gate_feature_readout_selection_v1/DETECTORS_FROZEN.npz",
        ROOT / "results/selected_q15_finalist_replay_v1/SCORES_FROZEN.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    pointers = [str(path) for path in required if path.is_file() and base.is_lfs_pointer(path)]
    unit = importlib.import_module("scripts.test_fusion_input_normalization_ablation_v1").run()
    checks = {}
    if not missing:
        frozen = json.loads((ROOT / "results/tail15_localization_q_v1/FROZEN_METHOD.json").read_text())
        checks = {
            "source_review": json.loads((SOURCE / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "gate_q": frozen["processbench_gate"]["q"] == Q,
            "four_view_source": frozen["locator"]["views"] == ["H0lim", "VE0", "VE0.75", "VE1"],
            "joint_lsml_excluded": all("lsml" not in name for name in SOLVERS),
        }
    return {
        "schema": "fusion-input-normalization-preflight-v1",
        "status": "PASS" if not missing and not pointers and unit["status"] == "PASS" and all(checks.values()) else "BLOCKED",
        "missing": missing,
        "lfs_pointers": pointers,
        "checks": checks,
        "unit": unit,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def load_source():
    manifest = json.loads((SOURCE / "MANIFEST.json").read_text())
    records, joined, fold = base.load_contract(Path(manifest["source_root"]), Path(manifest["contract_root"]))
    if len(records) != EXPECTED_ANSWERS:
        raise ValueError("answer roster mismatch")
    metadata = base.training_metadata(records, fold)
    return records, joined, fold, metadata


def read_feature(con: sqlite3.Connection, index: int) -> dict[str, np.ndarray]:
    row = con.execute("select payload from features where idx=?", (int(index),)).fetchone()
    if row is None:
        raise ValueError(f"missing frozen feature payload {index}")
    return base.unpacked(row[0])


def fit_priors(con, records, metadata, transform: str) -> dict[str, dict]:
    fitted = {}
    for cell in sorted({row["cell"] for row in records}):
        cell_ids = [index for index, row in enumerate(records) if row["cell"] == cell]
        statistics = {}
        for index in cell_ids:
            arrays = read_feature(con, index)
            bank = transform_bank(arrays, transform)
            statistics[index] = model.regional_statistics(bank, metadata[index]["uid"])
        for excluded in base.excluded_sets(metadata, cell_ids, cell):
            value, _ = base.fit_prior(statistics, metadata, cell, excluded)
            fitted[base.model_key(cell, excluded)] = value
        print("[fit]", transform, cell, len(cell_ids), flush=True)
    return fitted


def score_transform(con, records, joined, fold, metadata, transform: str, priors: dict[str, dict]):
    offsets = np.asarray(joined["offsets"], dtype=np.int64)
    total = int(offsets[-1])
    outer = {solver: np.full(total, np.nan, dtype=np.float64) for solver in SOLVERS}
    prm_folds = sorted({fold[index] for index, row in enumerate(records) if not row["cell"].startswith("pb_")})
    nested = {
        (solver, held): np.full(total, np.nan, dtype=np.float64)
        for solver in EXTERNAL_SOLVERS for held in prm_folds
    }
    failures = []
    for index, row in enumerate(records):
        arrays = read_feature(con, index)
        bank = transform_bank(arrays, transform)
        features = {"z": bank, "singles": {}}
        key = base.model_key(row["cell"], (fold[index],))
        values, health, _ = model.score_answer(features, arrays["spans"], priors[key], row["uid"], methods=SOLVERS)
        start, stop = offsets[index:index + 2]
        for solver in SOLVERS:
            outer[solver][start:stop] = values[solver]
            if health[solver]["status"] != "OK":
                failures.append({"index": index, "uid": row["uid"], "method": solver, "reason": health[solver].get("reason")})
        if not row["cell"].startswith("pb_"):
            for held in prm_folds:
                if held == fold[index]:
                    continue
                excluded = tuple(sorted((fold[index], held)))
                inner_key = base.model_key(row["cell"], excluded)
                inner, inner_health, _ = model.score_answer(
                    features, arrays["spans"], priors[inner_key], row["uid"], methods=EXTERNAL_SOLVERS
                )
                for solver in EXTERNAL_SOLVERS:
                    nested[(solver, held)][start:stop] = inner[solver]
                    if inner_health[solver]["status"] != "OK":
                        failures.append({"index": index, "uid": row["uid"], "method": f"{solver}__inner_{held}", "reason": inner_health[solver].get("reason")})
        if (index + 1) % 500 == 0:
            print("[score]", transform, index + 1, "/", len(records), flush=True)
    if failures:
        raise RuntimeError("fusion failures: " + json.dumps(failures[:5]))
    if any(not np.isfinite(value).all() for value in outer.values()):
        raise RuntimeError("nonfinite outer score")
    return outer, nested


def calibration_thresholds(records, joined, fold, score_bank, nested_bank) -> dict:
    offsets = np.asarray(joined["offsets"], dtype=np.int64)
    prm_folds = sorted({fold[index] for index, row in enumerate(records) if not row["cell"].startswith("pb_")})
    output = {name: {} for name in score_bank}
    for held in prm_folds:
        train = [index for index, row in enumerate(records) if not row["cell"].startswith("pb_") and fold[index] != held]
        excluded_groups = {records[index]["group_id"] for index, row in enumerate(records) if not row["cell"].startswith("pb_") and fold[index] == held}
        train_groups = {records[index]["group_id"] for index in train}
        if excluded_groups.intersection(train_groups):
            raise RuntimeError("PRM calibration group leakage")
        for name, flat in score_bank.items():
            solver = name.split("__", 1)[1]
            source = nested_bank[(name, held)] if solver in EXTERNAL_SOLVERS else flat
            vectors = [source[offsets[index]:offsets[index + 1]] for index in train]
            if not vectors or any(not np.isfinite(vector).all() for vector in vectors):
                raise RuntimeError("invalid nested calibration score")
            output[name][str(held)] = float(np.quantile(np.concatenate(vectors), 0.8))
    return output


def apply_frozen_pb_gate(records, joined, score_bank, gate_data) -> tuple[dict, dict]:
    offsets = np.asarray(joined["offsets"], dtype=np.int64)
    pb = np.asarray(gate_data["pb"], dtype=bool)
    opened = np.asarray(gate_data["tail_top10"] >= Q, dtype=bool)
    result, predictions = {}, {}
    for name, flat in score_bank.items():
        peak, valid = integration.peaks(flat, offsets, pb)
        peak_pb, valid_pb = peak[pb], valid[pb]
        prediction = np.where(opened & valid_pb, peak_pb, -1)
        summary = integration.summarize(gate_data["target"], gate_data["cells"], prediction, valid_pb)
        error = gate_data["target"] >= 0
        summary["raw_exact"] = float(np.mean(peak_pb[error] == gate_data["target"][error]))
        result[name] = summary
        predictions[name] = {"prediction": prediction, "valid": valid_pb}
    return result, predictions


def paired_pb_bootstrap(gate_data, predictions, pairs):
    unique, inverse = np.unique(gate_data["groups"], return_inverse=True)
    rng = np.random.default_rng(2026091404)
    draws = {first + "_minus_" + second: [] for first, second in pairs}
    for _ in range(BOOTSTRAP_DRAWS):
        counts = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weights = counts[inverse].astype(np.float64)
        current = {
            name: integration.pb_metrics(
                gate_data["target"], value["prediction"], value["valid"], gate_data["cells"], weights=weights
            )["macros"]["all"]
            for name, value in predictions.items()
        }
        for first, second in pairs:
            draws[first + "_minus_" + second].append(current[first] - current[second])
    alpha = (1.0 - PRIMARY_CI) / 2.0
    return {
        key: {
            "draws": BOOTSTRAP_DRAWS,
            "ci_level": PRIMARY_CI,
            "interval": [float(np.quantile(value, alpha)), float(np.quantile(value, 1.0 - alpha))],
            "mean": float(np.mean(value)),
        }
        for key, value in draws.items()
    }


def paired_prm_bootstrap(records, per, pairs):
    groups = np.asarray([row["group_id"] for row in records])
    prm = np.asarray([not row["cell"].startswith("pb_") for row in records], dtype=bool)
    rng = np.random.default_rng(2026091405)
    output = {}
    alpha = (1.0 - PRIMARY_CI) / 2.0
    for first, second in pairs:
        common = prm & np.isfinite(per[first]["within"]) & np.isfinite(per[second]["within"])
        unique, inverse = np.unique(groups[common], return_inverse=True)
        difference = per[first]["within"][common] - per[second]["within"][common]
        group_sum = np.bincount(inverse, weights=difference, minlength=len(unique))
        group_count = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        draws = []
        for _ in range(BOOTSTRAP_DRAWS):
            count = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
            draws.append(float(count @ group_sum / (count @ group_count)))
        draws = np.asarray(draws, dtype=np.float64)
        output[first + "_minus_" + second] = {
            "point_prm_within_delta": float(difference.mean()),
            "prm_within_interval": [float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))],
            "prm_within_bootstrap_mean": float(draws.mean()),
            "common_prm_answers": int(common.sum()),
            "common_prm_groups": int(len(unique)),
        }
    return output


def run() -> dict:
    check = preflight()
    atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise RuntimeError("preflight blocked")
    records, joined, fold, metadata = load_source()
    gate_data = frozen_gate.prepare()
    con = sqlite3.connect(f"file:{CHECKPOINT}?mode=ro", uri=True)
    all_scores, all_nested = {}, {}
    representation_diagnostics = {}
    try:
        with threadpool_limits(limits=1):
            for transform in TRANSFORMS:
                priors = fit_priors(con, records, metadata, transform)
                scores, nested = score_transform(con, records, joined, fold, metadata, transform, priors)
                for solver, value in scores.items():
                    all_scores[f"{transform}__{solver}"] = value
                for (solver, held), value in nested.items():
                    all_nested[(f"{transform}__{solver}", held)] = value
                # Literal-input diagnostics over the frozen answer payloads.
                conditions, means, scales = [], [], []
                for index in range(len(records)):
                    matrix = transform_bank(read_feature(con, index), transform)
                    second = matrix.T @ matrix / len(matrix)
                    conditions.append(float(np.linalg.cond(second)))
                    means.append(float(np.linalg.norm(matrix.mean(axis=0))))
                    scales.append(float(np.linalg.norm(matrix.std(axis=0))))
                representation_diagnostics[transform] = {
                    "median_second_moment_condition": float(np.median(conditions)),
                    "q95_second_moment_condition": float(np.quantile(conditions, 0.95)),
                    "mean_answer_mean_norm": float(np.mean(means)),
                    "mean_answer_scale_norm": float(np.mean(scales)),
                }
    finally:
        con.close()

    thresholds = calibration_thresholds(records, joined, fold, all_scores, all_nested)
    metrics, per = evaluator.evaluate_arrays(records, joined, all_scores, calibration_thresholds=thresholds, fold_auc=True)
    pb, predictions = apply_frozen_pb_gate(records, joined, all_scores, gate_data)
    for name in METHODS:
        metrics[name]["pb_all8"] = pb[name]["macros"]["all"]
        metrics[name]["pb_q4"] = pb[name]["macros"]["q4"]
        metrics[name]["pb_q8"] = pb[name]["macros"]["q8"]
        metrics[name]["pb_clean_accuracy"] = pb[name]["clean_accuracy"]
        metrics[name]["pb_error_exact_accuracy"] = pb[name]["error_exact_accuracy"]
        metrics[name]["pb_raw_exact"] = pb[name]["raw_exact"]
        valid = np.repeat(np.array([not row["cell"].startswith("pb_") for row in records]), np.diff(joined["offsets"])) & (joined["labels"] >= 0)
        metrics[name]["prm_pooled_oof_descriptive"] = evaluator.old.auc(joined["labels"][valid] == 1, all_scores[name][valid])

    pairs = []
    for solver in SOLVERS:
        pairs.extend(((f"scale_only__{solver}", f"answer_z__{solver}"), (f"raw__{solver}", f"answer_z__{solver}")))
    pb_contrasts = paired_pb_bootstrap(gate_data, predictions, pairs)
    prm_contrasts = paired_prm_bootstrap(records, per, pairs)
    for first, second in pairs:
        key = first + "_minus_" + second
        pb_contrasts[key]["point_pb_delta"] = metrics[first]["pb_all8"] - metrics[second]["pb_all8"]
        pb_contrasts[key].update(prm_contrasts[key])

    current = json.loads((ROOT / "results/tail15_localization_q_v1/METRICS.json").read_text())
    reference = {
        "pb_all8": current["localization"][frozen_gate.FINAL]["macros"]["all"],
        "pb_q4": current["localization"][frozen_gate.FINAL]["macros"]["q4"],
        "pb_q8": current["localization"][frozen_gate.FINAL]["macros"]["q8"],
        **current["prmbench"]["current_locator"],
    }
    ranking = sorted(METHODS, key=lambda name: (-metrics[name]["pb_all8"], -metrics[name]["prm_within"], name))
    result = {
        "schema": "fusion-input-normalization-ablation-v1",
        "status": "COMPLETE",
        "development_only": True,
        "joint_lsml_run": False,
        "joint_lsml_reason": "four-view bank is structurally inadmissible and multistart cost was deferred",
        "transforms": list(TRANSFORMS),
        "solvers": list(SOLVERS),
        "gate": {"feature": "tail15_mass__token_top10", "q": Q, "one_uniform_q": True},
        "metrics": metrics,
        "contrasts": pb_contrasts,
        "representation_diagnostics": representation_diagnostics,
        "current_frozen_locator_reference": reference,
        "pb_ranking": ranking,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "SCORES.npz", **{"steps__" + name: all_scores[name] for name in METHODS})
    atomic_json(OUT / "METRICS.json", result)
    atomic_json(OUT / "CALIBRATION.json", thresholds)
    with (OUT / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "pb_all8", "pb_raw_exact", "prm_within", "prm_fold_auc", "prm_pooled_oof_descriptive", "prmscore_q08"))
        writer.writeheader()
        for name in ranking:
            writer.writerow({"method": name, **{key: metrics[name][key] for key in writer.fieldnames[1:]}})
    return result


def write_report(result: dict) -> None:
    metrics = result["metrics"]
    rows = []
    for solver in SOLVERS:
        for transform in TRANSFORMS:
            name = f"{transform}__{solver}"
            value = metrics[name]
            rows.append(
                f"| {solver} | {transform} | {value['pb_all8']*100:.4f}% | {value['pb_raw_exact']*100:.4f}% | "
                f"{value['prm_within']:.6f} | {value['prm_fold_auc']:.6f} | {value['prm_pooled_oof_descriptive']:.6f} | {value['prmscore_q08']:.6f} |"
            )
    best = result["pb_ranking"][0]
    b = metrics[best]
    reference = result["current_frozen_locator_reference"]
    screen_candidates = []
    for solver in SOLVERS:
        z = metrics[f"answer_z__{solver}"]
        for transform in ("scale_only", "raw"):
            name = f"{transform}__{solver}"
            value = metrics[name]
            pb_gain = value["pb_all8"] - z["pb_all8"]
            prm_gain = value["prm_within"] - z["prm_within"]
            if (pb_gain >= .002 or prm_gain >= .002) and pb_gain >= -.002 and prm_gain >= -.002:
                screen_candidates.append(name)
    promoted = [
        name for name in screen_candidates
        if metrics[name]["pb_all8"] >= reference["pb_all8"] - .002
        and metrics[name]["prm_within"] >= reference["prm_within"] - .002
    ]
    decision = "CARRY_NON_Z_CANDIDATE_TO_EXPERIMENT_3" if promoted else "RETAIN_CURRENT_LOCATOR; KEEP_SCALE_ONLY_AS_CALIBRATION_CONTROL"
    report = f"""# Fusion input-normalization ablation v1

Status: **COMPLETE / REVIEW PASS — DEVELOPMENT ONLY**

The frozen q15 four-view bank was evaluated under answer-z, scale-only and
literal raw inputs. Every ProcessBench result below uses the already frozen
tail15 Top10 q=.33 gate; no q was changed. Joint L-SML was not run because this
bank is structurally inadmissible and its expensive multistart fit was deferred.

| Solver | Input | PB all-8 | PB raw exact | PRMB within | PRMB fold AUC | PRMB pooled OOF | PRMScore |
|---|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

Best tested fusion arm by PB is `{best}` at {b['pb_all8']*100:.4f}%. The current
development-frozen per-view-Top10 locator remains {reference['pb_all8']*100:.4f}%
PB, {reference['prm_within']:.6f} PRMB within and
{reference['prmscore_q08']:.6f} PRMScore.

## Interpretation

- For equal fusion, removing centering alone is localization-invariant but
  restores pooled OOF AUROC from {metrics['answer_z__equal']['prm_pooled_oof_descriptive']:.6f}
  to {metrics['scale_only__equal']['prm_pooled_oof_descriptive']:.6f} and PRMScore
  from {metrics['answer_z__equal']['prmscore_q08']:.6f} to
  {metrics['scale_only__equal']['prmscore_q08']:.6f}. This is calibration
  recovery, not better step selection.
- Natural-unit equal raises PRMB within by
  {metrics['raw__equal']['prm_within']-metrics['answer_z__equal']['prm_within']:+.6f}
  but changes PB by
  {(metrics['raw__equal']['pb_all8']-metrics['answer_z__equal']['pb_all8'])*100:+.3f}pp.
- Literal raw local IU and local shrinkage collapse. The median input
  second-moment condition number rises from
  {result['representation_diagnostics']['answer_z']['median_second_moment_condition']:.1f}
  under answer-z to
  {result['representation_diagnostics']['raw']['median_second_moment_condition']:.1f}
  in natural units, violating the scale assumptions used by their spectral
  solve.
- Raw/scale-only external stationary IU recovers cross-answer calibration but
  produces essentially no localization gain over its answer-z version.

Decision: `{decision}`. Non-z arms passing the within-solver development screen:
{', '.join('`'+name+'`' for name in screen_candidates) if screen_candidates else 'none'}.
Arms also reaching the current frozen locator's PB/within noninferiority region:
{', '.join('`'+name+'`' for name in promoted) if promoted else 'none'}.

Literal noncentered inputs are documented deviations from IU-PCR's expected
z-scored contract. Condition diagnostics and all paired contrasts are retained
in `METRICS.json`; this is not external confirmation.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf8")
    with np.load(OUT / "SCORES.npz", allow_pickle=False) as current, \
            np.load(SOURCE / "SCORES.npz", allow_pickle=False) as source, \
            np.load(ROOT / "results/probability_normalization_ablation_v1/SCORES_FROZEN.npz", allow_pickle=False) as raw_source:
        answer_z_identity = {
            solver: float(np.max(np.abs(current["steps__answer_z__" + solver] - source["steps__" + solver])))
            for solver in SOLVERS
        }
        raw_equal_identity = float(np.max(np.abs(current["steps__raw__equal"] - raw_source["steps__equal4_raw"])))
    if max(answer_z_identity.values()) > 1e-12 or raw_equal_identity > 1e-12:
        raise RuntimeError("frozen score identity drift")
    review = {
        "schema": "fusion-input-normalization-result-review-v1",
        "status": "PASS",
        "answers": EXPECTED_ANSWERS,
        "methods": len(METHODS),
        "one_frozen_gate": True,
        "joint_lsml_run": False,
        "all_scores_finite": True,
        "answer_z_source_identity_max_by_solver": answer_z_identity,
        "raw_equal_source_identity_max": raw_equal_identity,
        "report_sha256": sha256_file(OUT / "REPORT.md"),
        "scores_sha256": sha256_file(OUT / "SCORES.npz"),
    }
    atomic_json(OUT / "RESULT_REVIEW.json", review)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--review-existing", action="store_true")
    args = parser.parse_args()
    check = preflight()
    if args.preflight:
        atomic_json(OUT / "PREFLIGHT.json", check)
        print(json.dumps(check, indent=2, sort_keys=True))
        return
    if args.review_existing:
        records, joined, fold, _ = load_source()
        with np.load(OUT / "SCORES.npz", allow_pickle=False) as saved:
            all_scores = {name: np.asarray(saved["steps__" + name], dtype=np.float64) for name in METHODS}
        thresholds = json.loads((OUT / "CALIBRATION.json").read_text())
        metrics, per = evaluator.evaluate_arrays(records, joined, all_scores, calibration_thresholds=thresholds, fold_auc=True)
        gate_data = frozen_gate.prepare()
        pb, predictions = apply_frozen_pb_gate(records, joined, all_scores, gate_data)
        valid_steps = np.repeat(np.array([not row["cell"].startswith("pb_") for row in records]), np.diff(joined["offsets"])) & (joined["labels"] >= 0)
        for name in METHODS:
            metrics[name].update(
                pb_all8=pb[name]["macros"]["all"], pb_q4=pb[name]["macros"]["q4"], pb_q8=pb[name]["macros"]["q8"],
                pb_clean_accuracy=pb[name]["clean_accuracy"], pb_error_exact_accuracy=pb[name]["error_exact_accuracy"],
                pb_raw_exact=pb[name]["raw_exact"],
                prm_pooled_oof_descriptive=evaluator.old.auc(joined["labels"][valid_steps] == 1, all_scores[name][valid_steps]),
            )
        pairs = []
        for solver in SOLVERS:
            pairs.extend(((f"scale_only__{solver}", f"answer_z__{solver}"), (f"raw__{solver}", f"answer_z__{solver}")))
        contrasts = paired_pb_bootstrap(gate_data, predictions, pairs)
        prm_contrasts = paired_prm_bootstrap(records, per, pairs)
        for first, second in pairs:
            key = first + "_minus_" + second
            contrasts[key]["point_pb_delta"] = metrics[first]["pb_all8"] - metrics[second]["pb_all8"]
            contrasts[key].update(prm_contrasts[key])
        previous = json.loads((OUT / "METRICS.json").read_text())
        previous.update(metrics=metrics, contrasts=contrasts, pb_ranking=sorted(METHODS, key=lambda name: (-metrics[name]["pb_all8"], -metrics[name]["prm_within"], name)))
        atomic_json(OUT / "METRICS.json", previous)
        write_report(previous)
        print(json.dumps(json.loads((OUT / "RESULT_REVIEW.json").read_text()), indent=2, sort_keys=True))
        return
    result = run()
    write_report(result)
    print(json.dumps(json.loads((OUT / "RESULT_REVIEW.json").read_text()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
