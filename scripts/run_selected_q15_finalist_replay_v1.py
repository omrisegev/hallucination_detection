"""Replay the selected q15 raw per-view-Top10 finalist and fixed baselines."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_direct_probability_temporal as evaluator
from scripts import run_renyi_position_temporal_fusion as base


OUT = ROOT / "results/selected_q15_finalist_replay_v1"
PROTOCOL = ROOT / "docs/experiments/SELECTED_Q15_FINALIST_REPLAY_V1.md"
POSITION = ROOT / "results/renyi_position_temporal_fusion_v1"
UNIFORM = ROOT / "results/uniform_multiscale_fusion_v1"
EXPECTED_ANSWERS = 13_769
EXPECTED_STEPS = 145_597
BOOTSTRAP_DRAWS = 10_000
METHODS = (
    "selected_q15_raw_per_view_top10",
    "original_static_fusion_before_top10",
    "original_position_equal_z",
    "original_position_local_shrink",
)
PAIRS = tuple((METHODS[0], name) for name in METHODS[1:])
PRIMARY_CI = 1.0 - 0.05 / len(PAIRS)


def compose_q15_raw(views: list[np.ndarray]) -> np.ndarray:
    """Average four finite aligned step views without centering or scaling."""
    if len(views) != 4:
        raise ValueError("the finalist requires exactly four q15 views")
    arrays = [np.asarray(value, dtype=np.float64) for value in views]
    if any(value.ndim != 1 for value in arrays):
        raise ValueError("each q15 view must be one-dimensional")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("q15 views must have identical shapes")
    if any(not np.isfinite(value).all() for value in arrays):
        raise ValueError("q15 views must be finite")
    return np.mean(np.column_stack(arrays), axis=1)


def required_inputs() -> list[Path]:
    return [
        PROTOCOL,
        Path(__file__),
        POSITION / "SCORES.npz",
        POSITION / "CALIBRATION.json",
        POSITION / "RESULT_REVIEW.json",
        UNIFORM / "SCORES_FROZEN.npz",
        UNIFORM / "CALIBRATION.json",
        UNIFORM / "FROZEN_SCORES.json",
        UNIFORM / "MANIFEST.json",
        UNIFORM / "RESULT_REVIEW.json",
    ]


def preflight() -> dict:
    missing = [str(path) for path in required_inputs() if not path.is_file()]
    position_review = None
    uniform_review = None
    if not missing:
        position_review = json.loads((POSITION / "RESULT_REVIEW.json").read_text(encoding="utf8"))["status"]
        uniform_review = json.loads((UNIFORM / "RESULT_REVIEW.json").read_text(encoding="utf8"))["status"]
    status = "PASS" if not missing and position_review == uniform_review == "PASS" else "BLOCKED"
    return {
        "schema": "selected-q15-finalist-preflight-v1",
        "status": status,
        "missing": missing,
        "position_source_review": position_review,
        "uniform_source_review": uniform_review,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def load_contract():
    manifest = json.loads((UNIFORM / "MANIFEST.json").read_text(encoding="utf8"))
    source = Path(manifest["source_root"])
    contract_root = Path(manifest["contract_root"])
    base.configure_sources(source, contract_root)
    records = json.loads(
        (evaluator.old.BENCH / "evaluation/JOINED.json").read_text(encoding="utf8")
    )["records"]
    with np.load(evaluator.old.BENCH / "evaluation/JOINED.npz", allow_pickle=False) as saved:
        joined = {name: np.asarray(saved[name]) for name in saved.files}
    if len(records) != EXPECTED_ANSWERS or int(joined["offsets"][-1]) != EXPECTED_STEPS:
        raise ValueError("frozen answer/step roster mismatch")
    folds = json.loads(evaluator.old.FOLDS.read_text(encoding="utf8"))["outer"]
    fold = {index: int(folds[row["group_id"]]) for index, row in enumerate(records)}
    return manifest, records, joined, fold


def load_scores() -> tuple[dict[str, np.ndarray], dict]:
    with np.load(POSITION / "SCORES.npz", allow_pickle=False) as position, np.load(
        UNIFORM / "SCORES_FROZEN.npz", allow_pickle=False
    ) as uniform:
        views = [
            np.asarray(position["steps__view__H0lim"], dtype=np.float64),
            np.asarray(position["steps__view__ve0"], dtype=np.float64),
            np.asarray(position["steps__view__ve0.75"], dtype=np.float64),
            np.asarray(position["steps__view__ve1"], dtype=np.float64),
        ]
        finalist = compose_q15_raw(views)
        frozen_selected = np.asarray(uniform["steps__q15_raw_equal"], dtype=np.float64)
        scores = {
            METHODS[0]: finalist,
            METHODS[1]: np.asarray(uniform["steps__reference_equal4_token_raw"], dtype=np.float64),
            METHODS[2]: np.asarray(position["steps__equal"], dtype=np.float64),
            METHODS[3]: np.asarray(position["steps__local_shrink_position"], dtype=np.float64),
        }
    if any(value.shape != (EXPECTED_STEPS,) or not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("invalid finalist replay score array")
    difference = np.abs(finalist - frozen_selected)
    review = {
        "finalist_exactly_reconstructs_frozen_q15_raw_equal": bool(np.array_equal(finalist, frozen_selected)),
        "finalist_max_abs_reconstruction_error": float(difference.max()),
        "views": ["H0lim", "VE0", "VE0.75", "VE1"],
        "support": "q15",
        "readout": "per-view Top10 then raw equal step fusion",
        "centering": "none",
        "scaling": "natural units retained",
    }
    if not review["finalist_exactly_reconstructs_frozen_q15_raw_equal"]:
        raise ValueError("finalist does not reproduce the frozen selected score")
    return scores, review


def load_thresholds() -> dict[str, dict]:
    position = json.loads((POSITION / "CALIBRATION.json").read_text(encoding="utf8"))["thresholds"]
    uniform = json.loads((UNIFORM / "CALIBRATION.json").read_text(encoding="utf8"))["thresholds"]
    return {
        METHODS[0]: uniform["q15_raw_equal"],
        METHODS[1]: uniform["reference_equal4_token_raw"],
        METHODS[2]: position["equal"],
        METHODS[3]: position["local_shrink_position"],
    }


def freeze_scores(scores: dict[str, np.ndarray], manifest: dict) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "SCORES_FROZEN.npz"
    temporary = path.with_suffix(".npz.tmp")
    temporary.write_bytes(base.packed(**{"steps__" + name: value for name, value in scores.items()}))
    temporary.replace(path)
    record = {
        "schema": "selected-q15-finalist-score-freeze-v1",
        "status": "SCORES_FROZEN_BEFORE_REPLAY_EVALUATION",
        "score_path": str(path),
        "score_sha256": base.sha256_file(path),
        "manifest_sha256": base.sha256_file(OUT / "MANIFEST.json"),
        "aggregate_metrics_computed_after_freeze": True,
        "development_replay_only": True,
    }
    base.atomic_json(OUT / "FROZEN_SCORES.json", record)
    return record


def evaluate(records, joined, fold, scores, thresholds):
    metrics, per = evaluator.evaluate_arrays(
        records, joined, scores, calibration_thresholds=thresholds, fold_auc=True
    )
    cells = np.asarray([row["cell"] for row in records])
    prm_answer = ~np.char.startswith(cells, "pb_")
    prm_step = np.repeat(prm_answer, np.diff(joined["offsets"]))
    for name, value in scores.items():
        valid = prm_step & np.repeat(per[name]["valid"], np.diff(joined["offsets"])) & (joined["labels"] >= 0)
        metrics[name]["prm_pooled_oof_descriptive"] = evaluator.old.auc(
            joined["labels"][valid] == 1, value[valid]
        )
    contrasts = evaluator.paired_bootstrap(
        records,
        joined,
        per,
        draws=BOOTSTRAP_DRAWS,
        pairs=list(PAIRS),
        primary_pairs=set(PAIRS),
        primary_ci=PRIMARY_CI,
    )
    for first, second in PAIRS:
        contrasts[first + "_minus_" + second]["pb_delta"] = (
            metrics[first]["pb_all8"] - metrics[second]["pb_all8"]
        )
    return metrics, contrasts


def write_comparison(metrics):
    fields = (
        "method", "pb_all8", "pb_raw_exact", "prm_within", "prm_fold_auc",
        "prm_pooled_oof_descriptive", "prmscore_q08", "valid_answers",
    )
    with (OUT / "COMPARISON.csv").open("w", newline="", encoding="utf8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in METHODS:
            writer.writerow({"method": name, **{field: metrics[name][field] for field in fields[1:]}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "full"), default="preflight")
    args = parser.parse_args()
    check = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    base.atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise FileNotFoundError(base.dumps(check))
    if args.phase == "preflight":
        print(base.dumps(check))
        return

    source_manifest, records, joined, fold = load_contract()
    hashes = {str(path): base.sha256_file(path) for path in required_inputs()}
    manifest = {
        "schema": "selected-q15-finalist-replay-v1",
        "status": "FROZEN",
        "population": {"answers": len(records), "steps": int(joined["offsets"][-1])},
        "methods": list(METHODS),
        "pairs": [list(pair) for pair in PAIRS],
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "familywise_ci": PRIMARY_CI},
        "gate": "unchanged frozen mean entropy q=.3",
        "prmscore": "unchanged nested q=.8 thresholds from source runs",
        "source_manifest_sha256": base.sha256_file(UNIFORM / "MANIFEST.json"),
        "hashes": hashes,
        "development_replay_only": True,
    }
    base.atomic_json(OUT / "MANIFEST.json", manifest)
    scores, reconstruction = load_scores()
    frozen = freeze_scores(scores, manifest)
    thresholds = load_thresholds()
    metrics, contrasts = evaluate(records, joined, fold, scores, thresholds)
    base.atomic_json(OUT / "CALIBRATION.json", {"thresholds": thresholds, "reused_unchanged": True})
    base.atomic_json(OUT / "METRICS.json", {"schema": "selected-q15-finalist-metrics-v1", "metrics": metrics})
    base.atomic_json(OUT / "CONTRASTS.json", contrasts)
    write_comparison(metrics)
    review = {
        "schema": "selected-q15-finalist-result-review-v1",
        "status": "PASS",
        "answers": len(records),
        "steps": int(joined["offsets"][-1]),
        "score_sha256": frozen["score_sha256"],
        "reconstruction": reconstruction,
        "same_method_for_every_benchmark": True,
        "benchmark_specific_selection": False,
        "calibration_thresholds_reused_unchanged": True,
        "no_training_or_refitting": True,
        "development_replay_only": True,
    }
    base.atomic_json(OUT / "RESULT_REVIEW.json", review)
    base.atomic_json(OUT / "RUN_STATE.json", {"status": "COMPLETE_REVIEWED", "answers": len(records)})
    print("method,pb_all8,pb_raw_exact,prm_within,prm_fold_auc,prm_pooled_oof,prmscore")
    for name in METHODS:
        value = metrics[name]
        print(",".join(map(str, (
            name, value["pb_all8"], value["pb_raw_exact"], value["prm_within"],
            value["prm_fold_auc"], value["prm_pooled_oof_descriptive"], value["prmscore_q08"],
        ))))
    print("score_sha256", frozen["score_sha256"])


if __name__ == "__main__":
    main()

