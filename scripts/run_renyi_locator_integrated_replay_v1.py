#!/usr/bin/env python3
"""Independently replay the Experiment-3 candidate and final recommendation."""
from __future__ import annotations

import gc
import json
from pathlib import Path
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


OUT = ROOT / "results/renyi_locator_integrated_replay_v1"
PROTOCOL = ROOT / "docs/experiments/RENYI_LOCATOR_INTEGRATED_REPLAY_V1.md"
EXPERIMENT = ROOT / "results/renyi_locator_feature_bank_v1"
CURRENT = ROOT / "results/selected_q15_finalist_replay_v1/SCORES_FROZEN.npz"
SOURCE = ROOT / "results/renyi_position_temporal_fusion_v1"
CANDIDATE = "ve1q50__h10__hinf0__scale_step_equal"
REFERENCE = model.BASELINE
METHODS = (REFERENCE, CANDIDATE)
EXPECTED_ANSWERS = 13_769
EXPECTED_STEPS = 145_597
Q = 0.33
MARGIN = 0.002
BOOTSTRAP_DRAWS = 10_000


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(base.json_ready(value), indent=2, sort_keys=True) + "\n", encoding="utf8")
    temporary.replace(path)


def preflight() -> dict:
    required = [
        PROTOCOL,
        Path(__file__),
        ROOT / "spectral_utils/renyi_locator_feature_bank.py",
        EXPERIMENT / "SELECTION.json",
        EXPERIMENT / "SCORES_FROZEN.npz",
        EXPERIMENT / "RESULT_REVIEW.json",
        CURRENT,
        SOURCE / "MANIFEST.json",
        ROOT / "results/tail15_localization_q_v1/FROZEN_METHOD.json",
        ROOT / "results/tail15_localization_q_v1/RESULT_REVIEW.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    pointers = [str(path) for path in required if path.is_file() and base.is_lfs_pointer(path)]
    checks = {}
    if not missing:
        selection = json.loads((EXPERIMENT / "SELECTION.json").read_text())
        gate = json.loads((ROOT / "results/tail15_localization_q_v1/FROZEN_METHOD.json").read_text())
        checks = {
            "selected_candidate_exact": selection["selected"] == CANDIDATE,
            "experiment_review": json.loads((EXPERIMENT / "RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "gate_review": json.loads((ROOT / "results/tail15_localization_q_v1/RESULT_REVIEW.json").read_text())["status"] == "PASS",
            "gate_q": gate["processbench_gate"]["q"] == Q,
        }
    return {
        "schema": "renyi-locator-integrated-replay-preflight-v1",
        "status": "PASS" if not missing and not pointers and all(checks.values()) else "BLOCKED",
        "missing": missing,
        "lfs_pointers": pointers,
        "checks": checks,
        "python": sys.executable,
        "no_packages_installed": True,
    }


def load_contract():
    manifest = json.loads((SOURCE / "MANIFEST.json").read_text())
    records, joined, fold = base.load_contract(Path(manifest["source_root"]), Path(manifest["contract_root"]))
    if len(records) != EXPECTED_ANSWERS or int(joined["offsets"][-1]) != EXPECTED_STEPS:
        raise ValueError("frozen roster mismatch")
    return manifest, records, joined, fold


def recompute(records, joined) -> dict[str, np.ndarray]:
    offsets = np.asarray(joined["offsets"], dtype=np.int64)
    scores = {name: np.full(EXPECTED_STEPS, np.nan, dtype=np.float64) for name in METHODS}
    candidate_bank = model.BANK_BY_NAME[CANDIDATE.rsplit("__", 1)[0]]
    baseline_bank = model.BANK_BY_NAME[REFERENCE.rsplit("__", 1)[0]]
    for cell, path, kind, dataset in evaluator.source_specs():
        indices = [index for index, record in enumerate(records) if record["cell"] == cell]
        print("[load]", cell, len(indices), flush=True)
        rows = evaluator.old._source_row_map(evaluator.old.load_pickle(path), kind=kind, dataset=dataset)
        for index in indices:
            record = records[index]
            row = rows[record["row_id"]]
            logprobs = np.asarray(evaluator.old._topk_payload(row)["logprobs"], dtype=np.float64)
            entropy = np.asarray(row["token_entropies"], dtype=np.float64)
            spans = np.asarray(row["step_token_spans"], dtype=np.int64)
            features = model.feature_matrix(logprobs, entropy)
            baseline, _ = model.score_bank(features, spans, baseline_bank, "raw_step_equal")
            candidate, _ = model.score_bank(features, spans, candidate_bank, "scale_step_equal")
            start, stop = offsets[index:index + 2]
            scores[REFERENCE][start:stop] = baseline
            scores[CANDIDATE][start:stop] = candidate
            if (index + 1) % 500 == 0:
                print("[replay]", index + 1, "/", len(records), flush=True)
        del rows
        gc.collect()
    if any(not np.isfinite(value).all() for value in scores.values()):
        raise ValueError("replay score gap")
    return scores


def identity(scores: dict[str, np.ndarray]) -> dict:
    with np.load(EXPERIMENT / "SCORES_FROZEN.npz", allow_pickle=False) as saved, np.load(CURRENT, allow_pickle=False) as current:
        expected = {
            REFERENCE: np.asarray(current["steps__selected_q15_raw_per_view_top10"], dtype=np.float64),
            CANDIDATE: np.asarray(saved["steps__" + CANDIDATE], dtype=np.float64),
        }
    output = {}
    for name in METHODS:
        delta = np.abs(scores[name] - expected[name])
        output[name] = {"array_equal": bool(np.array_equal(scores[name], expected[name])), "max_abs": float(delta.max())}
        if output[name]["max_abs"] > 2e-8:
            raise ValueError("score identity failed: " + name)
    return output


def apply_gate(joined, scores, data):
    pb = np.asarray(data["pb"], dtype=bool)
    opened = np.asarray(data["tail_top10"] >= Q, dtype=bool)
    metrics, predictions = {}, {}
    for name in METHODS:
        peak, valid = integration.peaks(scores[name], joined["offsets"], pb)
        peak, valid = peak[pb], valid[pb]
        prediction = np.where(opened & valid, peak, -1)
        value = integration.summarize(data["target"], data["cells"], prediction, valid)
        error = data["target"] >= 0
        value["raw_exact"] = float(np.mean(peak[error] == data["target"][error]))
        metrics[name] = value
        predictions[name] = (prediction, valid)
    return metrics, predictions


def pb_bootstrap(data, predictions) -> dict:
    unique, inverse = np.unique(data["groups"], return_inverse=True)
    rng = np.random.default_rng(2026091501)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        count = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        weight = count[inverse].astype(float)
        values = [integration.pb_metrics(data["target"], *predictions[name], data["cells"], weights=weight)["macros"]["all"] for name in (CANDIDATE, REFERENCE)]
        draws.append(values[0] - values[1])
    draws = np.asarray(draws)
    return {"draws": BOOTSTRAP_DRAWS, "ci_level": 0.95, "mean": float(draws.mean()), "interval": np.quantile(draws, [0.025, 0.975]).tolist()}


def prm_bootstrap(records, per) -> dict:
    prm = np.asarray([not row["cell"].startswith("pb_") for row in records], dtype=bool)
    common = prm & np.isfinite(per[CANDIDATE]["within"]) & np.isfinite(per[REFERENCE]["within"])
    groups = np.asarray([row["group_id"] for row in records])[common]
    unique, inverse = np.unique(groups, return_inverse=True)
    difference = per[CANDIDATE]["within"][common] - per[REFERENCE]["within"][common]
    sums = np.bincount(inverse, weights=difference, minlength=len(unique))
    sizes = np.bincount(inverse, minlength=len(unique)).astype(float)
    rng = np.random.default_rng(2026091502)
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        count = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))
        draws.append(float(count @ sums / (count @ sizes)))
    draws = np.asarray(draws)
    return {"draws": BOOTSTRAP_DRAWS, "ci_level": 0.95, "point": float(difference.mean()), "mean": float(draws.mean()), "interval": np.quantile(draws, [0.025, 0.975]).tolist(), "answers": int(common.sum()), "groups": len(unique)}


def main() -> None:
    check = preflight()
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "PREFLIGHT.json", check)
    if check["status"] != "PASS":
        raise RuntimeError(base.dumps(check))
    source_manifest, records, joined, fold = load_contract()
    manifest = {
        "schema": "renyi-locator-integrated-replay-v1",
        "status": "FROZEN",
        "candidate": CANDIDATE,
        "reference": REFERENCE,
        "gate": {"feature": "tail15_mass__token_top10", "q": Q},
        "promotion": {"pb_delta_min": 0.0, "prm_within_delta_min": -MARGIN},
        "population": {"answers": len(records), "steps": int(joined["offsets"][-1])},
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "source_manifest_sha256": base.sha256_file(SOURCE / "MANIFEST.json"),
        "selection_sha256": base.sha256_file(EXPERIMENT / "SELECTION.json"),
        "protocol_sha256": base.sha256_file(PROTOCOL),
        "development_only": True,
    }
    atomic_json(OUT / "MANIFEST.json", manifest)
    with threadpool_limits(limits=1):
        scores = recompute(records, joined)
    replay_identity = identity(scores)
    archive = OUT / "SCORES_FROZEN.npz"
    np.savez_compressed(archive, **{"steps__" + name: value for name, value in scores.items()})
    freeze = {"status": "SCORES_FROZEN_BEFORE_AGGREGATE_EVALUATION", "sha256": base.sha256_file(archive), "identity": replay_identity}
    atomic_json(OUT / "FROZEN_SCORES.json", freeze)
    metrics, per = evaluator.evaluate_arrays(records, joined, scores, fold_auc=True)
    gate_data = frozen_gate.prepare()
    pb, predictions = apply_gate(joined, scores, gate_data)
    cells = np.asarray([row["cell"] for row in records])
    prm_steps = np.repeat(~np.char.startswith(cells, "pb_"), np.diff(joined["offsets"])) & (joined["labels"] >= 0)
    for name in METHODS:
        metrics[name]["pb_all8"] = pb[name]["macros"]["all"]
        metrics[name]["pb_q4"] = pb[name]["macros"]["q4"]
        metrics[name]["pb_q8"] = pb[name]["macros"]["q8"]
        metrics[name]["pb_raw_exact"] = pb[name]["raw_exact"]
        metrics[name]["pb_clean_accuracy"] = pb[name]["clean_accuracy"]
        metrics[name]["pb_error_exact_accuracy"] = pb[name]["error_exact_accuracy"]
        metrics[name]["prm_pooled_oof_descriptive"] = evaluator.old.auc(joined["labels"][prm_steps] == 1, scores[name][prm_steps])
    deltas = {
        "pb_all8": metrics[CANDIDATE]["pb_all8"] - metrics[REFERENCE]["pb_all8"],
        "prm_within": metrics[CANDIDATE]["prm_within"] - metrics[REFERENCE]["prm_within"],
        "prmscore_q08": metrics[CANDIDATE]["prmscore_q08"] - metrics[REFERENCE]["prmscore_q08"],
    }
    promoted = deltas["pb_all8"] >= 0 and deltas["prm_within"] >= -MARGIN
    recommendation = CANDIDATE if promoted else REFERENCE
    result = {
        "schema": "renyi-locator-integrated-replay-metrics-v1",
        "status": "COMPLETE",
        "metrics": metrics,
        "deltas_candidate_minus_reference": deltas,
        "bootstrap": {"pb_all8": pb_bootstrap(gate_data, predictions), "prm_within": prm_bootstrap(records, per)},
        "promotion": {"passed": promoted, "recommended_locator": recommendation, "reason": "candidate meets both frozen margins" if promoted else "candidate PRMB-within loss exceeds frozen .002 margin"},
        "integrated_recommendation": {"locator": recommendation, "gate": "tail15_mass__token_top10", "gate_q": Q},
        "replay_identity": replay_identity,
        "score_sha256": freeze["sha256"],
        "development_only": True,
    }
    atomic_json(OUT / "METRICS.json", result)
    review = {
        "schema": "renyi-locator-integrated-replay-review-v1",
        "status": "PASS",
        "answers": len(records),
        "steps": int(joined["offsets"][-1]),
        "score_identity": replay_identity,
        "scores_frozen_before_evaluation": True,
        "same_gate_both_methods": True,
        "no_tuning_in_replay": True,
        "recommended_locator": recommendation,
        "development_only": True,
    }
    atomic_json(OUT / "RESULT_REVIEW.json", review)
    atomic_json(OUT / "RUN_STATE.json", {"status": "COMPLETE_REVIEWED", "recommended_locator": recommendation})
    print(json.dumps(base.json_ready(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        OUT.mkdir(parents=True, exist_ok=True)
        atomic_json(OUT / "RUN_STATE.json", {"status": "FAILED", "error": f"{type(error).__name__}: {error}"})
        raise
