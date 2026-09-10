"""Run the frozen gray-box direct probability-rank fusion experiment.

Track A fits probability-rank fusion inside each single answer for PB/PRMB
localization. Track B aggregates those ranks per answer and fits across answers
inside each of the canonical historical 24 cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inscope_cells import CROPPED_CELLS, GROUP, INSCOPE  # noqa: E402
from scripts.build_repgrid_featcache import H16, candidate_feats  # noqa: E402
from spectral_utils.answer_span import crop_candidate  # noqa: E402
from spectral_utils.direct_probability_fusion import (  # noqa: E402
    DEFAULT_K,
    answer_rank_features,
    direct_rank_risk,
    fit_rank_fusion,
    logprob_matrix,
    step_top_mean,
    top_mean,
    topk_renormalized_varentropy,
)
from spectral_utils.dufs_liu_feature_contract import (  # noqa: E402
    dufs_liu_mixed_v2_from_bundle,
)
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402

MIND_GAP_LIB = ROOT / "scripts" / "gl_liu_v1" / "localization"
if str(MIND_GAP_LIB) not in sys.path:
    sys.path.insert(0, str(MIND_GAP_LIB))
from evidence_drop import EVIDENCE_FNS  # noqa: E402
from localization_metrics import step_drop_scores as mindgap_step_scores  # noqa: E402


SOURCE_ROOT = ROOT
BENCH: Path
FIXED_GATE: Path
FOLDS: Path
PB_DIRS: dict[str, Path]
PRMB_TELEMETRY: Path
PRMB_LABELS: Path
REPGRID: Path
REFERENCE_24: Path
HISTORICAL_BUNDLE: Path
STEP334: Path
PRMSCORE_REFERENCE: Path
OUT = ROOT / "results" / "direct_probability_fusion_v1"
METHODS = ("entropy", "rank_equal", "rank_iu", "rank_joint_lw")
FIT_METHOD = {
    "rank_equal": "equal",
    "rank_iu": "iu",
    "rank_joint_lw": "joint_lw",
}
BOOT_SEED = 2026091015
MIND_GAP_COMPARATOR = "mindgap_paper_locator_common_gate"
DISPLAY_NAMES = {
    "entropy": "Token Entropy",
    "rank_equal": "Direct Probability Fusion - Equal Weights",
    "rank_iu": "Direct Probability Fusion - IU-PCR",
    "rank_joint_lw": "Direct Probability Fusion - Joint Shrinkage",
    MIND_GAP_COMPARATOR: "Mind the Gap Locator - Common Gate",
    "varentropy": "Token Varentropy",
    "historical_iu_pcr": "Historical IU-PCR",
}


def configure_source_root(source_root: Path) -> None:
    """Bind all read-only benchmark inputs to one explicit checkout."""

    global SOURCE_ROOT, BENCH, FIXED_GATE, FOLDS, PB_DIRS
    global PRMB_TELEMETRY, PRMB_LABELS, REPGRID, REFERENCE_24
    global HISTORICAL_BUNDLE, STEP334
    global PRMSCORE_REFERENCE
    SOURCE_ROOT = source_root.resolve()
    BENCH = SOURCE_ROOT / "results" / "localization_full_benchmark_v3"
    FIXED_GATE = SOURCE_ROOT / "results" / "fusion_fixed_gate_v1"
    FOLDS = (
        SOURCE_ROOT
        / "results"
        / "localization_source_group_audit_v1"
        / "FOLDS_V2.json"
    )
    PB_DIRS = {
        "q4": SOURCE_ROOT / "dataset_cache" / "repgrid" / "pb_qwen3_4b",
        "q8": SOURCE_ROOT / "dataset_cache" / "repgrid" / "pb_qwen3_8b",
    }
    PRMB_TELEMETRY = (
        SOURCE_ROOT
        / "dataset_cache"
        / "four_localization"
        / "prmbench_qwen3_8b_telemetry_full"
        / "prmbench_telemetry.pkl"
    )
    PRMB_LABELS = (
        SOURCE_ROOT
        / "dataset_cache"
        / "four_localization"
        / "prmbench_qwen25math7b_full"
        / "prmbench_prm.pkl"
    )
    REPGRID = SOURCE_ROOT / "dataset_cache" / "repgrid"
    REFERENCE_24 = (
        SOURCE_ROOT
        / "results"
        / "hard_filter_dufs_liu_24cell"
        / "per_cell_metrics.csv"
    )
    HISTORICAL_BUNDLE = SOURCE_ROOT / "results" / "dependency_fusion_raw" / "cells.npz"
    STEP334 = SOURCE_ROOT / "results" / "token_level_readout_v1" / "METRICS.json"
    PRMSCORE_REFERENCE = SOURCE_ROOT / "results" / "prmscore_adapter_v1" / "METRICS.json"


configure_source_root(ROOT)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_inputs(k: int) -> dict[str, Any]:
    """Fail closed if the audited source or benchmark contract has drifted."""

    audit_path = OUT / "DATA_AUDIT.json"
    if not audit_path.exists():
        raise FileNotFoundError(
            "Run audit_direct_probability_data_v1.py in this worktree before scoring"
        )
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    summary = audit.get("summary", {})
    if Path(audit.get("source_root", "")).resolve() != SOURCE_ROOT:
        raise ValueError("data audit belongs to a different source checkout")
    if int(audit.get("required_k", -1)) != int(k):
        raise ValueError("data audit K differs from the frozen experiment K")
    if summary.get("localization_artifacts_ready") != 9:
        raise ValueError("data audit did not approve all 9 localization artifacts")
    if summary.get("historical_cells_ready") != 24:
        raise ValueError("data audit did not approve all 24 historical cells")

    required = {
        "benchmark_join_json": BENCH / "evaluation" / "JOINED.json",
        "benchmark_join_arrays": BENCH / "evaluation" / "JOINED.npz",
        "fixed_gate_detectors": FIXED_GATE / "DETECTORS.npz",
        "fixed_gate_metrics": FIXED_GATE / "METRICS.json",
        "source_group_folds": FOLDS,
        "historical_24_reference": REFERENCE_24,
        "historical_24_bundle": HISTORICAL_BUNDLE,
        "token_level_reference": STEP334,
        "supervised_prmscore_reference": PRMSCORE_REFERENCE,
        "data_audit": audit_path,
        "source_snapshot": (
            ROOT
            / "docs"
            / "experiments"
            / "DIRECT_PROBABILITY_FUSION_V1_SOURCE_SNAPSHOT.json"
        ),
        "input_freeze": (
            ROOT
            / "docs"
            / "experiments"
            / "DIRECT_PROBABILITY_FUSION_V1_INPUT_FREEZE.json"
        ),
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing frozen inputs: " + ", ".join(missing))
    # Every large raw cache was hashed by the preflight audit. Recheck those
    # hashes before scoring so a changed cache cannot pass on row counts alone.
    audited_items = [*audit["localization"], *audit["historical_24"]]
    audited_items.append(audit["prmbench_label_join"])
    audited_hashes: dict[str, str] = {}
    for item in audited_items:
        relative = str(item["artifact"])
        if relative in audited_hashes:
            continue
        path = SOURCE_ROOT / Path(relative)
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(f"audited input changed after preflight: {relative}")
        audited_hashes[relative] = actual

    freeze = json.loads(required["input_freeze"].read_text(encoding="utf-8-sig"))
    frozen_hashes: dict[str, str] = {}
    for relative, expected in freeze["source_files"].items():
        path = SOURCE_ROOT / Path(relative)
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen benchmark input changed: {relative}")
        frozen_hashes[relative] = actual

    hashes = {name: sha256_file(path) for name, path in required.items()}
    joined = json.loads(required["benchmark_join_json"].read_text(encoding="utf-8"))
    if len(joined.get("records", [])) != 13_769:
        raise ValueError("localization benchmark roster is not the frozen 13,769 rows")
    return {
        "source_root": str(SOURCE_ROOT),
        "worktree_root": str(ROOT),
        "audited_localization_artifacts": 9,
        "audited_historical_cells": 24,
        "localization_rows": 13_769,
        "hashes": hashes,
        "audited_raw_hashes": audited_hashes,
        "frozen_source_hashes": frozen_hashes,
    }


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return float("nan")
    return float(
        (rankdata(scores)[labels].sum() - positive * (positive + 1) / 2)
        / (positive * negative)
    )


def _source_row_map(payload: Any, *, kind: str, dataset: str | None = None) -> dict[str, dict]:
    values = payload.values() if isinstance(payload, dict) else payload
    output: dict[str, dict] = {}
    for row in values:
        if not isinstance(row, dict):
            continue
        if kind == "pb":
            key = f"{dataset}::{row.get('id')}"
        else:
            key = str(row.get("idx"))
        if key in output:
            raise ValueError(f"duplicate source row id {key}")
        output[key] = row
    return output


def _topk_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Return the saved top-K payload without changing its probability mass."""

    for key in ("top_k_logprobs", "top_k_logprobs_raw"):
        value = row.get(key)
        if isinstance(value, dict) and value.get("logprobs") is not None:
            return value
    raise ValueError("row has no saved top-K log-probability matrix")


def _local_token_scores(
    row: dict, k: int
) -> tuple[dict[str, np.ndarray], dict[str, dict], dict[str, float]]:
    method_seconds: dict[str, float] = {}
    started = time.perf_counter()
    logprobs = logprob_matrix(_topk_payload(row), k=k)
    entropy = np.asarray(row["token_entropies"], dtype=float)
    if entropy.shape != (len(logprobs),) or not np.isfinite(entropy).all():
        raise ValueError("entropy/top-k token alignment failure")
    risk = direct_rank_risk(logprobs)
    scores: dict[str, np.ndarray] = {"entropy": entropy}
    method_seconds["entropy"] = time.perf_counter() - started
    diagnostics: dict[str, dict] = {}
    for name, method in FIT_METHOD.items():
        started = time.perf_counter()
        try:
            fit = fit_rank_fusion(risk, method=method, anchor=entropy)
            scores[name] = fit.score
            diagnostics[name] = {
                "fallback": fit.fallback,
                "alpha": fit.alpha,
                "kept_ranks": int(len(fit.kept_ranks)),
                "weights": fit.weights.tolist(),
                "orientation_flipped": fit.orientation_flipped,
            }
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            # Explicit coverage-preserving fallback. It is counted and reported;
            # the baseline is never presented as successful probability fusion.
            scores[name] = entropy.copy()
            diagnostics[name] = {
                "fallback": f"entropy:{type(exc).__name__}:{exc}",
                "alpha": None,
                "kept_ranks": 0,
                "weights": [0.0] * k,
                "orientation_flipped": False,
            }
        method_seconds[name] = time.perf_counter() - started
    return scores, diagnostics, method_seconds


def _mind_gap_step_scores(row: dict[str, Any]) -> np.ndarray:
    """Replay the project's paper-form Mind the Gap locator on one answer."""

    evidence = EVIDENCE_FNS["shannon"](row, 20)
    return np.asarray(
        mindgap_step_scores(evidence, row["step_token_spans"], ema_span=5),
        dtype=float,
    )


def _gate_contract(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    detector = np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False)["entropy_mean"]
    metrics = json.loads((FIXED_GATE / "METRICS.json").read_text(encoding="utf-8"))
    thresholds = {
        int(key): float(value)
        for key, value in metrics["arms"]["dual__iu"]["rows"]
        ["entropy_mean|quantile_0.3"]["thresholds"].items()
    }
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    outer = np.asarray(
        [int(folds["outer"].get(record["group_id"], -1)) for record in records], dtype=int
    )
    row_thresholds = np.asarray([thresholds.get(int(value), np.nan) for value in outer])
    if detector.shape != (len(records),) or not np.isfinite(row_thresholds).all():
        raise ValueError("fixed gate does not cover every benchmark row")
    return detector, row_thresholds


def _localization_metrics(
    records: list[dict],
    joined: np.lib.npyio.NpzFile,
    step_scores: dict[str, np.ndarray],
    mind_gap_scores: np.ndarray,
    fallbacks: dict[str, Counter],
    alphas: dict[str, list[float]],
    weights: dict[str, list[np.ndarray]],
    runtimes: dict[str, float],
    bootstrap: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels = joined["labels"]
    offsets = joined["offsets"]
    target = joined["target"]
    cells = np.asarray([record["cell"] for record in records])
    groups = np.asarray([record["group_id"] for record in records])
    pb = np.asarray([cell.startswith("pb_") for cell in cells])
    detector, thresholds = _gate_contract(records)
    per: dict[str, dict[str, np.ndarray]] = {}
    results: dict[str, Any] = {}
    predictions_out: dict[str, np.ndarray] = {}

    for name in METHODS:
        flat = step_scores[name]
        valid = np.ones(len(records), dtype=bool)
        peak = np.full(len(records), -1, dtype=int)
        within = np.full(len(records), np.nan)
        for index in range(len(records)):
            score = flat[offsets[index] : offsets[index + 1]]
            if len(score) == 0 or not np.isfinite(score).all():
                valid[index] = False
                continue
            peak[index] = int(np.argmax(score))
            if not pb[index]:
                truth = labels[offsets[index] : offsets[index + 1]]
                usable = truth >= 0
                if (truth[usable] == 1).any() and (truth[usable] == 0).any():
                    within[index] = auc(truth[usable] == 1, score[usable])

        local_mask = np.zeros(len(labels), dtype=bool)
        for index in np.flatnonzero(valid & ~pb):
            local_mask[offsets[index] : offsets[index + 1]] = True
        local_mask &= labels >= 0
        pb_valid = pb & valid & np.isfinite(detector) & np.isfinite(thresholds)
        prediction = np.where(detector >= thresholds, peak, -1)
        prediction[~pb_valid] = -1
        metrics = pb_metrics(
            target[pb], prediction[pb], pb_valid[pb], cells[pb]
        )
        error = pb & (target >= 0) & valid
        clean = pb & (target < 0) & valid
        exact = peak[error] == target[error]
        difference = peak[error] - target[error]
        results[name] = {
            "pb_all8": float(metrics["macros"]["all"]),
            "pb_q4": float(metrics["macros"]["q4"]),
            "pb_q8": float(metrics["macros"]["q8"]),
            "pb_cells": {
                cell: float(value["f1"]) for cell, value in metrics["cells"].items()
            },
            "pb_raw_exact": float(np.mean(exact)),
            "pb_within_one": float(np.mean(np.abs(difference) <= 1)),
            "pb_early": int(np.sum(difference < 0)),
            "pb_late": int(np.sum(difference > 0)),
            "pb_exact_count": int(np.sum(exact)),
            "pb_error_answers": int(np.sum(error)),
            "pb_clean_accuracy": float(np.mean(prediction[clean] == -1)),
            "prm_within": float(np.nanmean(within)),
            "prm_pooled": auc(labels[local_mask] == 1, flat[local_mask]),
            "valid_answers": int(valid.sum()),
            "coverage": float(valid.mean()),
            "fallbacks": dict(fallbacks.get(name, Counter())),
            "mean_alpha": float(np.mean(alphas[name])) if alphas.get(name) else None,
            "mean_weights": (
                np.mean(np.stack(weights[name]), axis=0).tolist() if weights.get(name) else None
            ),
            "fit_seconds": float(runtimes.get(name, 0.0)),
        }
        per[name] = {"within": within, "prediction": prediction, "valid": pb_valid}
        predictions_out[f"prediction__{name}"] = prediction

    # Same-contract external comparator: the paper-form Mind the Gap step
    # locator receives the exact same frozen entropy q=0.3 no-error gate.
    # This is deliberately named as a common-gate replay, not as native SLA.
    comparator_valid = np.zeros(len(records), dtype=bool)
    comparator_peak = np.full(len(records), -1, dtype=int)
    for index in np.flatnonzero(pb):
        score = mind_gap_scores[offsets[index] : offsets[index + 1]]
        if len(score) and np.isfinite(score).any():
            comparator_valid[index] = True
            comparator_peak[index] = int(np.nanargmax(score))
    comparator_pb_valid = pb & comparator_valid & np.isfinite(detector) & np.isfinite(thresholds)
    comparator_prediction = np.where(detector >= thresholds, comparator_peak, -1)
    comparator_prediction[~comparator_pb_valid] = -1
    comparator_metrics = pb_metrics(
        target[pb], comparator_prediction[pb], comparator_pb_valid[pb], cells[pb]
    )
    comparator_error = pb & (target >= 0) & comparator_valid
    comparator_clean = pb & (target < 0) & comparator_valid
    comparator_difference = comparator_peak[comparator_error] - target[comparator_error]
    comparator_result = {
        "pb_all8": float(comparator_metrics["macros"]["all"]),
        "pb_q4": float(comparator_metrics["macros"]["q4"]),
        "pb_q8": float(comparator_metrics["macros"]["q8"]),
        "pb_cells": {
            cell: float(value["f1"])
            for cell, value in comparator_metrics["cells"].items()
        },
        "pb_raw_exact": float(np.mean(comparator_difference == 0)),
        "pb_within_one": float(np.mean(np.abs(comparator_difference) <= 1)),
        "pb_early": int(np.sum(comparator_difference < 0)),
        "pb_late": int(np.sum(comparator_difference > 0)),
        "pb_exact_count": int(np.sum(comparator_difference == 0)),
        "pb_error_answers": int(np.sum(comparator_error)),
        "pb_clean_accuracy": float(np.mean(comparator_prediction[comparator_clean] == -1)),
        "valid_answers": int(comparator_valid[pb].sum()),
        "coverage": float(comparator_valid[pb].mean()),
        "access": "paper-form locator; frozen mean-entropy q=0.3 gate",
        "native_sla": False,
        "fit_seconds": float(runtimes.get(MIND_GAP_COMPARATOR, 0.0)),
    }
    per[MIND_GAP_COMPARATOR] = {
        "within": np.full(len(records), np.nan),
        "prediction": comparator_prediction,
        "valid": comparator_pb_valid,
    }
    predictions_out[f"prediction__{MIND_GAP_COMPARATOR}"] = comparator_prediction

    # Exact continuity check against the accepted Step334 entropy row.
    step334 = json.loads(STEP334.read_text(encoding="utf-8"))
    old = step334["results"]["token_entropy"]
    for current, previous in (
        (results["entropy"]["pb_all8"], old["pb_all8"]),
        (results["entropy"]["prm_within"], old["prm_within"]),
        (results["entropy"]["prm_pooled"], old["prm_pooled"]),
    ):
        if not np.isclose(current, previous, atol=1e-12, rtol=0.0):
            raise AssertionError(f"Step334 entropy continuity failed: {current} != {previous}")

    # Official PRMScore adapter, same cross-fold q0.8 rule as Step334.
    label_rows = list(load_pickle(PRMB_LABELS).values())
    label_map = {str(row["idx"]): row for row in label_rows}
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    outer = np.asarray([int(folds["outer"].get(record["group_id"], -1)) for record in records])
    prm_indexes = np.flatnonzero(~pb)
    for name in METHODS:
        flat = step_scores[name]
        predicted: dict[int, np.ndarray] = {}
        for fold in sorted(set(outer[prm_indexes].tolist())):
            train = [index for index in prm_indexes if outer[index] != fold]
            test = [index for index in prm_indexes if outer[index] == fold]
            threshold = float(
                np.quantile(
                    np.concatenate(
                        [flat[offsets[index] : offsets[index + 1]] for index in train]
                    ),
                    0.8,
                )
            )
            for index in test:
                predicted[index] = (
                    ~(flat[offsets[index] : offsets[index + 1]] >= threshold)
                ).astype(int)
        evaluation = prmbench_evaluate(
            [
                {"idx": records[index]["row_id"], "labels": value.astype(int).tolist()}
                for index, value in predicted.items()
            ],
            [label_map[str(records[index]["row_id"])] for index in predicted],
        )
        results[name]["prmscore_q08"] = float(
            0.5 * evaluation["total"]["f1"]
            + 0.5 * evaluation["total"]["negative_f1"]
        )
    if not np.isclose(
        results["entropy"]["prmscore_q08"],
        old["prmscore_q08"],
        atol=1e-12,
        rtol=0.0,
    ):
        raise AssertionError(
            "Step334 entropy PRMScore continuity failed: "
            f"{results['entropy']['prmscore_q08']} != {old['prmscore_q08']}"
        )

    comparisons = (
        ("rank_joint_lw", "entropy"),
        ("rank_iu", "entropy"),
    )
    pb_only_comparisons = (
        ("rank_iu", MIND_GAP_COMPARATOR),
        ("rank_joint_lw", MIND_GAP_COMPARATOR),
    )
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    draws = {
        f"{left}_minus_{right}": {"pb": [], "within": []}
        for left, right in comparisons
    }
    draws.update(
        {
            f"{left}_minus_{right}": {"pb": []}
            for left, right in pb_only_comparisons
        }
    )
    for draw_index in range(int(bootstrap)):
        sampled = rng.integers(0, len(unique_groups), len(unique_groups))
        group_weight = np.bincount(sampled, minlength=len(unique_groups))[inverse].astype(float)
        for left, right in comparisons:
            key = f"{left}_minus_{right}"
            common = np.isfinite(per[left]["within"]) & np.isfinite(per[right]["within"])
            draws[key]["within"].append(
                float(
                    np.average(
                        per[left]["within"][common] - per[right]["within"][common],
                        weights=group_weight[common],
                    )
                )
            )
            left_pb = pb_metrics(
                target[pb],
                per[left]["prediction"][pb],
                per[left]["valid"][pb],
                cells[pb],
                weights=group_weight[pb],
            )["macros"]["all"]
            right_pb = pb_metrics(
                target[pb],
                per[right]["prediction"][pb],
                per[right]["valid"][pb],
                cells[pb],
                weights=group_weight[pb],
            )["macros"]["all"]
            draws[key]["pb"].append(float(left_pb - right_pb))
        for left, right in pb_only_comparisons:
            key = f"{left}_minus_{right}"
            left_pb = pb_metrics(
                target[pb],
                per[left]["prediction"][pb],
                per[left]["valid"][pb],
                cells[pb],
                weights=group_weight[pb],
            )["macros"]["all"]
            right_pb = pb_metrics(
                target[pb],
                per[right]["prediction"][pb],
                per[right]["valid"][pb],
                cells[pb],
                weights=group_weight[pb],
            )["macros"]["all"]
            draws[key]["pb"].append(float(left_pb - right_pb))
        if (draw_index + 1) % 1000 == 0:
            print(f"[localization bootstrap] {draw_index + 1}/{bootstrap}", flush=True)
    contrasts = {}
    for left, right in comparisons:
        key = f"{left}_minus_{right}"
        contrasts[key] = {
            "pb_delta": float(results[left]["pb_all8"] - results[right]["pb_all8"]),
            "pb_ci_97_5": np.percentile(draws[key]["pb"], [1.25, 98.75]).tolist(),
            "prm_within_delta": float(results[left]["prm_within"] - results[right]["prm_within"]),
            "prm_within_ci_97_5": np.percentile(
                draws[key]["within"], [1.25, 98.75]
            ).tolist(),
        }
    all_pb_results = {**results, MIND_GAP_COMPARATOR: comparator_result}
    for left, right in pb_only_comparisons:
        key = f"{left}_minus_{right}"
        contrasts[key] = {
            "pb_delta": float(
                all_pb_results[left]["pb_all8"] - all_pb_results[right]["pb_all8"]
            ),
            "pb_ci_97_5": np.percentile(draws[key]["pb"], [1.25, 98.75]).tolist(),
            "prm_within_delta": None,
            "prm_within_ci_97_5": None,
        }
    prm_reference = json.loads(PRMSCORE_REFERENCE.read_text(encoding="utf-8"))
    frozen_references = {
        "token_varentropy": step334["results"]["token_varentropy"],
        "token_feature_fusion_iu": step334["results"]["token_iu9"],
        "window_feature_fusion_iu": step334["results"]["window_iu27_top10"],
        "supervised_math_prm": prm_reference["qwen25_math_prm_7b_rewards>=0.5"],
    }
    return {
        "methods": results,
        "comparators": {MIND_GAP_COMPARATOR: comparator_result},
        "frozen_references": frozen_references,
        "contrasts": contrasts,
    }, predictions_out


def run_localization(k: int, bootstrap: int) -> dict[str, Any]:
    print("[localization] loading frozen benchmark", flush=True)
    joined_json = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    joined = np.load(BENCH / "evaluation" / "JOINED.npz", allow_pickle=False)
    records = joined_json["records"]
    offsets = joined["offsets"]
    step_scores = {
        name: np.full(int(offsets[-1]), np.nan, dtype=float) for name in METHODS
    }
    mind_gap_scores = np.full(int(offsets[-1]), np.nan, dtype=float)
    fallbacks = {name: Counter() for name in FIT_METHOD}
    alphas: dict[str, list[float]] = {name: [] for name in FIT_METHOD}
    weights: dict[str, list[np.ndarray]] = {name: [] for name in FIT_METHOD}
    runtimes = {name: 0.0 for name in (*METHODS, MIND_GAP_COMPARATOR)}
    audited_detector = np.load(FIXED_GATE / "DETECTORS.npz", allow_pickle=False)[
        "entropy_mean"
    ]
    if audited_detector.shape != (len(records),):
        raise ValueError("fixed gate detector length differs from the frozen roster")

    source_specs = [
        (model, dataset, directory / f"processbench_{dataset}.pkl")
        for model, directory in PB_DIRS.items()
        for dataset in ("gsm8k", "math", "olympiadbench", "omnimath")
    ]
    for model, dataset, path in source_specs:
        print(f"[localization] source {dataset} {model}", flush=True)
        source = _source_row_map(load_pickle(path), kind="pb", dataset=dataset)
        indexes = [
            index
            for index, record in enumerate(records)
            if record["cell"] == f"pb_{dataset}_{model}"
        ]
        for index in indexes:
            row = source[records[index]["row_id"]]
            token_scores, diagnostics, method_seconds = _local_token_scores(row, k)
            if not np.isclose(
                float(np.mean(row["token_entropies"])),
                float(audited_detector[index]),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(f"fixed gate row order mismatch for {records[index]['uid']}")
            spans = np.asarray(row["step_token_spans"], dtype=int)
            if spans.shape != (records[index]["steps"], 2):
                raise ValueError(f"step-span mismatch for {records[index]['uid']}")
            for name in METHODS:
                step_scores[name][offsets[index] : offsets[index + 1]] = step_top_mean(
                    token_scores[name], spans[:, 0], spans[:, 1], count=10
                )
            comparator_started = time.perf_counter()
            comparator = _mind_gap_step_scores(row)
            runtimes[MIND_GAP_COMPARATOR] += time.perf_counter() - comparator_started
            if comparator.shape != (records[index]["steps"],):
                raise ValueError(f"Mind the Gap step mismatch for {records[index]['uid']}")
            mind_gap_scores[offsets[index] : offsets[index + 1]] = comparator
            for name, diagnostic in diagnostics.items():
                if diagnostic["fallback"]:
                    fallbacks[name][diagnostic["fallback"]] += 1
                if diagnostic["alpha"] is not None:
                    alphas[name].append(float(diagnostic["alpha"]))
                weights[name].append(np.asarray(diagnostic["weights"], dtype=float))
            for name, seconds in method_seconds.items():
                runtimes[name] += float(seconds)
        del source

    print("[localization] source prmbench", flush=True)
    source = _source_row_map(load_pickle(PRMB_TELEMETRY), kind="prm")
    for index, record in enumerate(records):
        if record["cell"] != "prmbench_qwen3_8b":
            continue
        row = source[record["row_id"]]
        token_scores, diagnostics, method_seconds = _local_token_scores(row, k)
        if not np.isclose(
            float(np.mean(row["token_entropies"])),
            float(audited_detector[index]),
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError(f"fixed gate row order mismatch for {record['uid']}")
        spans = np.asarray(row["step_token_spans"], dtype=int)
        if spans.shape != (record["steps"], 2):
            raise ValueError(f"step-span mismatch for {record['uid']}")
        for name in METHODS:
            step_scores[name][offsets[index] : offsets[index + 1]] = step_top_mean(
                token_scores[name], spans[:, 0], spans[:, 1], count=10
            )
        for name, diagnostic in diagnostics.items():
            if diagnostic["fallback"]:
                fallbacks[name][diagnostic["fallback"]] += 1
            if diagnostic["alpha"] is not None:
                alphas[name].append(float(diagnostic["alpha"]))
            weights[name].append(np.asarray(diagnostic["weights"], dtype=float))
        for name, seconds in method_seconds.items():
            runtimes[name] += float(seconds)
    del source

    for name, values in step_scores.items():
        if not np.isfinite(values).all():
            missing = int((~np.isfinite(values)).sum())
            raise ValueError(f"{name} has {missing} unfilled/nonfinite step scores")
    metrics, predictions = _localization_metrics(
        records,
        joined,
        step_scores,
        mind_gap_scores,
        fallbacks,
        alphas,
        weights,
        runtimes,
        bootstrap,
    )
    np.savez_compressed(
        OUT / "LOCALIZATION_SCORES.npz",
        **{f"steps__{name}": values for name, values in step_scores.items()},
        steps__mindgap_paper_locator=mind_gap_scores,
        **predictions,
    )
    payload = {
        "schema": "direct-probability-localization-v1",
        "k": k,
        "token_readout": "top-10 mean within each reasoning step",
        "gate": "frozen foldwise mean entropy q=0.3",
        "mind_gap_comparator": (
            "paper-form step locator with the same frozen entropy q=0.3 gate; "
            "not native erroneous-trace SLA"
        ),
        "n_answers": len(records),
        "n_steps": int(offsets[-1]),
        **metrics,
    }
    write_json(OUT / "LOCALIZATION.json", payload)
    return payload


def _historical_source(cell: str) -> Path:
    matches = sorted((REPGRID / cell).glob("raw_*.pkl"))
    if len(matches) != 1:
        raise ValueError(f"{cell}: expected one raw pkl, found {len(matches)}")
    return matches[0]


def _historical_references() -> dict[str, dict[str, str]]:
    with REFERENCE_24.open(newline="", encoding="utf-8-sig") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["contract"] == "mixed_v2"
            and row["filter"] == "full"
            and row["solver"] == "iu_pcr"
        ]
    references = {row["cell"]: row for row in rows}
    if len(rows) != 24 or set(references) != set(INSCOPE):
        raise ValueError(
            "historical IU-PCR reference must contain exactly the 24 in-scope "
            "mixed_v2/full/iu_pcr rows"
        )
    return references


def _matched_historical_candidates(
    payload: dict[Any, Any], cell: str, expected_n: int
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Replay the exact complete-case population used to build repgrid_cells.pkl."""

    raw: list[tuple[str, dict[str, Any]]] = []
    crop = cell in CROPPED_CELLS
    for problem_id in sorted(payload, key=lambda value: int(value)):
        for candidate in payload[problem_id]["candidates"]:
            prepared = crop_candidate(candidate) if crop else candidate
            raw.append((str(problem_id), prepared))
    if len(raw) == int(expected_n) or crop:
        return [candidate for _, candidate in raw], np.asarray(
            [problem_id for problem_id, _ in raw], dtype=str
        )

    candidates: list[dict[str, Any]] = []
    problem_ids: list[str] = []
    for problem_id, prepared in raw:
        features = candidate_feats(prepared, allow_short=False)
        complete = all(np.isfinite(features.get(feature, np.nan)) for feature in H16)
        if not complete:
            continue
        candidates.append(prepared)
        problem_ids.append(problem_id)
    return candidates, np.asarray(problem_ids, dtype=str)


def _weighted_auc_batch(
    labels: np.ndarray,
    scores: np.ndarray,
    group_index: np.ndarray,
    group_counts: np.ndarray,
) -> np.ndarray:
    """Weighted AUROC for many fixed-score grouped-bootstrap draws."""

    order = np.argsort(scores, kind="mergesort")
    ordered_scores = np.asarray(scores, dtype=float)[order]
    ordered_labels = np.asarray(labels, dtype=bool)[order]
    weights = np.asarray(group_counts[:, group_index[order]], dtype=float)
    positive = weights * ordered_labels[None, :]
    negative = weights * (~ordered_labels)[None, :]
    negative_before = np.cumsum(negative, axis=1) - negative
    numerator = np.sum(positive * negative_before, axis=1)

    # Replace the stable-sort ordering inside exact score ties by half credit.
    boundaries = np.flatnonzero(np.diff(ordered_scores) != 0) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, len(order)]
    for start, end in zip(starts, ends):
        if end - start <= 1:
            continue
        positive_block = positive[:, start:end].sum(axis=1)
        negative_block = negative[:, start:end].sum(axis=1)
        before = negative_before[:, start]
        old = np.sum(
            positive[:, start:end] * negative_before[:, start:end], axis=1
        )
        numerator += positive_block * (before + 0.5 * negative_block) - old

    denominator = positive.sum(axis=1) * negative.sum(axis=1)
    result = np.full(len(group_counts), np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def _paired_group_auc_bootstrap(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
    batch_size: int = 128,
) -> np.ndarray:
    """Paired candidate AUROC delta under canonical problem-group resampling."""

    _, group_index = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    n_groups = int(group_index.max()) + 1
    if n_groups < 2:
        raise ValueError("historical grouped bootstrap needs at least two problem groups")
    rng = np.random.default_rng(seed)
    output: list[np.ndarray] = []
    for start in range(0, int(draws), int(batch_size)):
        size = min(int(batch_size), int(draws) - start)
        counts = rng.multinomial(
            n_groups, np.full(n_groups, 1.0 / n_groups), size=size
        )
        output.append(
            _weighted_auc_batch(labels, left, group_index, counts)
            - _weighted_auc_batch(labels, right, group_index, counts)
        )
    values = np.concatenate(output)
    if np.isfinite(values).sum() < 0.99 * int(draws):
        raise ValueError("too many single-class historical bootstrap draws")
    return values[np.isfinite(values)]


def run_historical(k: int, bootstrap: int) -> dict[str, Any]:
    references = _historical_references()
    historical_bundle = np.load(HISTORICAL_BUNDLE, allow_pickle=True)
    cells_out: dict[str, Any] = {}
    score_arrays: dict[str, np.ndarray] = {}
    primary_bootstrap: list[np.ndarray] = []
    total_runtime = {name: 0.0 for name in (*METHODS, "varentropy")}
    total_fallbacks = {name: Counter() for name in METHODS}
    for position, cell in enumerate(INSCOPE, start=1):
        print(f"[historical] {position:02d}/{len(INSCOPE)} {cell}", flush=True)
        payload = load_pickle(_historical_source(cell))
        historical_correct = np.asarray(
            historical_bundle[f"{cell}__labels"], dtype=bool
        )
        expected_labels = ~historical_correct
        candidates, problem_ids = _matched_historical_candidates(
            payload, cell, len(expected_labels)
        )
        if len(candidates) != len(expected_labels):
            raise ValueError(
                f"{cell}: matched population has {len(candidates)} candidates; "
                f"historical IU-PCR has {len(expected_labels)}"
            )
        rank_rows = []
        entropy = []
        varentropy = []
        for candidate in candidates:
            logprobs = logprob_matrix(_topk_payload(candidate), k=k)
            rank_risk = direct_rank_risk(logprobs)
            rank_rows.append(answer_rank_features(rank_risk, count=10))
            token_entropy = np.asarray(candidate["token_entropies"], dtype=float)
            if len(token_entropy) != len(logprobs):
                raise ValueError(f"{cell}: entropy/top-k length mismatch")
            entropy.append(top_mean(token_entropy, count=10))
            varentropy.append(top_mean(topk_renormalized_varentropy(logprobs), count=10))
        X = np.asarray(rank_rows, dtype=float)
        entropy_array = np.asarray(entropy, dtype=float)
        method_runtime: dict[str, float] = {}
        method_scores: dict[str, np.ndarray] = {"entropy": entropy_array}
        diagnostics: dict[str, Any] = {}
        for name, method in FIT_METHOD.items():
            method_started = time.perf_counter()
            fit = fit_rank_fusion(X, method=method, anchor=entropy_array)
            method_runtime[name] = time.perf_counter() - method_started
            method_scores[name] = fit.score
            diagnostics[name] = {
                "alpha": fit.alpha,
                "fallback": fit.fallback,
                "kept_ranks": int(len(fit.kept_ranks)),
                "weights": fit.weights.tolist(),
                "orientation_flipped": fit.orientation_flipped,
            }
            total_runtime[name] += method_runtime[name]
            if fit.fallback:
                total_fallbacks[name][fit.fallback] += 1
        method_runtime["entropy"] = 0.0
        method_runtime["varentropy"] = 0.0

        # Labels are opened only after every new fusion score is frozen in memory.
        raw_labels = np.asarray(
            [not bool(candidate.get("label", False)) for candidate in candidates],
            dtype=bool,
        )
        if not np.array_equal(raw_labels, expected_labels):
            raise ValueError(f"{cell}: raw labels do not match the frozen historical annotation")
        label_array = expected_labels

        stored = np.asarray(historical_bundle[f"{cell}__V"], dtype=float)
        names = tuple(str(value) for value in historical_bundle[f"{cell}__pool"])
        hand_signs = np.asarray(
            historical_bundle[f"{cell}__hand_signs"], dtype=float
        )
        mixed, _, _ = dufs_liu_mixed_v2_from_bundle(stored, names, hand_signs)
        historical_matrix = np.asarray(mixed.T, dtype=float)
        reference_fit = upcr_fit(historical_matrix, **dict(IU_FIT_DEFAULTS))
        # The historical bundle uses 1=correct. This experiment consistently
        # uses 1=hallucination, so flip both the target and the reference score.
        # AUROC is therefore exactly preserved while all visible scores retain
        # the common convention high=risk.
        historical_iu_score = -np.asarray(
            reference_fit.w @ historical_matrix, dtype=float
        )
        if historical_iu_score.shape != label_array.shape:
            raise ValueError(f"{cell}: historical IU-PCR score length mismatch")

        method_auc = {name: auc(label_array, score) for name, score in method_scores.items()}
        method_auc["varentropy"] = auc(label_array, np.asarray(varentropy, dtype=float))
        reference = references[cell]
        reference_iu_pcr = float(reference["auroc"])
        replayed_reference = auc(label_array, historical_iu_score)
        if not np.isclose(replayed_reference, reference_iu_pcr, atol=1e-12, rtol=0.0):
            raise AssertionError(
                f"{cell}: historical IU-PCR replay {replayed_reference} != {reference_iu_pcr}"
            )
        cell_bootstrap = _paired_group_auc_bootstrap(
            label_array,
            method_scores["rank_iu"],
            historical_iu_score,
            problem_ids,
            draws=bootstrap,
            seed=BOOT_SEED + 100 + position,
        )
        primary_bootstrap.append(cell_bootstrap)
        cells_out[cell] = {
            "domain": GROUP[cell],
            "n_answers": len(candidates),
            "n_problems": len(set(problem_ids)),
            "hallucination_rate": float(label_array.mean()),
            "auroc": method_auc,
            "historical_iu_pcr": reference_iu_pcr,
            "direct_iu_minus_historical_iu_pcr_ci97_5": np.percentile(
                cell_bootstrap, [1.25, 98.75]
            ).tolist(),
            "rank_joint_lw_minus_historical_iu_pcr": float(
                method_auc["rank_joint_lw"] - reference_iu_pcr
            ),
            "diagnostics": diagnostics,
            "fit_seconds": method_runtime,
        }
        safe_cell = cell.replace("-", "_")
        score_arrays[f"{safe_cell}__label"] = label_array.astype(np.int8)
        for name, score in method_scores.items():
            score_arrays[f"{safe_cell}__{name}"] = np.asarray(score, dtype=np.float32)
        score_arrays[f"{safe_cell}__historical_iu_pcr"] = historical_iu_score.astype(
            np.float32
        )
        del payload, candidates

    macro = {}
    for name in (*METHODS, "varentropy"):
        all_values = np.asarray([cells_out[cell]["auroc"][name] for cell in INSCOPE])
        qa_values = np.asarray(
            [cells_out[cell]["auroc"][name] for cell in INSCOPE if GROUP[cell] == "QA"]
        )
        math_values = np.asarray(
            [cells_out[cell]["auroc"][name] for cell in INSCOPE if GROUP[cell] == "math"]
        )
        macro[name] = {
            "all24": float(np.mean(all_values)),
            "qa9": float(np.mean(qa_values)),
            "math15": float(np.mean(math_values)),
        }
    historical_iu_pcr = np.asarray(
        [cells_out[cell]["historical_iu_pcr"] for cell in INSCOPE], dtype=float
    )
    macro["historical_iu_pcr"] = {
        "all24": float(np.mean(historical_iu_pcr)),
        "qa9": float(
            np.mean([cells_out[cell]["historical_iu_pcr"] for cell in INSCOPE if GROUP[cell] == "QA"])
        ),
        "math15": float(
            np.mean([cells_out[cell]["historical_iu_pcr"] for cell in INSCOPE if GROUP[cell] == "math"])
        ),
    }

    comparisons = (
        ("rank_joint_lw", "entropy"),
        ("rank_iu", "entropy"),
        ("rank_joint_lw", "rank_iu"),
        ("rank_iu", "historical_iu_pcr"),
        ("rank_joint_lw", "historical_iu_pcr"),
    )
    vectors = {
        name: np.asarray(
            [
                cells_out[cell]["historical_iu_pcr"]
                if name == "historical_iu_pcr"
                else cells_out[cell]["auroc"][name]
                for cell in INSCOPE
            ],
            dtype=float,
        )
        for name in set(sum(([left, right] for left, right in comparisons), []))
    }
    rng = np.random.default_rng(BOOT_SEED + 1)
    contrasts = {}
    indexes = rng.integers(0, len(INSCOPE), size=(int(bootstrap), len(INSCOPE)))
    for left, right in comparisons:
        delta = vectors[left] - vectors[right]
        draws = delta[indexes].mean(axis=1)
        contrasts[f"{left}_minus_{right}"] = {
            "delta": float(delta.mean()),
            "cell_bootstrap_ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
            "wins": int(np.sum(delta > 0)),
            "ties": int(np.sum(np.isclose(delta, 0.0, atol=1e-12))),
            "losses": int(np.sum(delta < 0)),
        }
    # Primary matched contrast: group bootstrap within every cell, then paired
    # cell bootstrap for the 24-cell macro. This preserves repeated answers to
    # the same canonical problem and carries within-cell uncertainty forward.
    primary_matrix = np.stack(primary_bootstrap, axis=0)
    cell_indexes = rng.integers(
        0, len(INSCOPE), size=(int(bootstrap), len(INSCOPE))
    )
    draw_indexes = np.arange(int(bootstrap))[:, None]
    hierarchical = primary_matrix[cell_indexes, draw_indexes].mean(axis=1)
    primary_key = "rank_iu_minus_historical_iu_pcr"
    contrasts[primary_key]["hierarchical_group_ci97_5"] = np.percentile(
        hierarchical, [1.25, 98.75]
    ).tolist()
    contrasts[primary_key]["bootstrap_unit"] = (
        "canonical problem groups within cells, paired cells for the macro"
    )
    np.savez_compressed(OUT / "HISTORICAL_24_SCORES.npz", **score_arrays)
    result = {
        "schema": "direct-probability-historical24-v1",
        "k": k,
        "answer_aggregation": "top-10 token mean separately for each probability rank",
        "fit_population": (
            "all complete-case answers in the historical mixed_v2 population of each cell; "
            "labels excluded from fusion"
        ),
        "matched_population": (
            "exact complete-case candidate rows used by historical mixed_v2/full/iu_pcr"
        ),
        "cells": cells_out,
        "macro": macro,
        "contrasts": contrasts,
        "coverage": {name: 1.0 for name in (*METHODS, "varentropy", "historical_iu_pcr")},
        "fallbacks": {name: dict(total_fallbacks.get(name, Counter())) for name in METHODS},
        "fit_seconds": total_runtime,
    }
    write_json(OUT / "HISTORICAL_24.json", result)
    return result


def render_report(localization: dict[str, Any], historical: dict[str, Any]) -> str:
    lines = [
        "# Direct probability-rank fusion v1",
        "",
        "Gray-box only. K=15 vocabulary ranks; top-10 is the separate token readout.",
        "",
        "## One-answer localization",
        "",
        "| method | PB all8 | PB Q4 | PB Q8 | PRMB within | PRMScore | fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in METHODS:
        row = localization["methods"][name]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {100*row['pb_all8']:.2f}% | {100*row['pb_q4']:.2f}% | "
            f"{100*row['pb_q8']:.2f}% | {row['prm_within']:.4f} | "
            f"{row['prmscore_q08']:.4f} | {sum(row['fallbacks'].values())} |"
        )
    mind_gap = localization["comparators"][MIND_GAP_COMPARATOR]
    lines.append(
        f"| {DISPLAY_NAMES[MIND_GAP_COMPARATOR]} | {100*mind_gap['pb_all8']:.2f}% | "
        f"{100*mind_gap['pb_q4']:.2f}% | {100*mind_gap['pb_q8']:.2f}% | n/a | n/a | 0 |"
    )
    lines.extend(["", "Primary paired contrasts (97.5% intervals):", ""])
    for name, row in localization["contrasts"].items():
        description = (
            f"- `{name}`: PB {100*row['pb_delta']:+.2f} pp "
            f"[{100*row['pb_ci_97_5'][0]:+.2f}, {100*row['pb_ci_97_5'][1]:+.2f}]"
        )
        if row["prm_within_delta"] is not None:
            description += (
                f"; PRMB within {row['prm_within_delta']:+.4f} "
                f"[{row['prm_within_ci_97_5'][0]:+.4f}, "
                f"{row['prm_within_ci_97_5'][1]:+.4f}]"
            )
        lines.append(description + ".")
    lines.extend(
        [
            "",
            "## Complete-answer detection: historical 24 cells",
            "",
            "| method | all24 macro AUROC | QA9 | math15 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (*METHODS, "varentropy", "historical_iu_pcr"):
        row = historical["macro"][name]
        lines.append(
            f"| {DISPLAY_NAMES[name]} | {row['all24']:.4f} | {row['qa9']:.4f} | {row['math15']:.4f} |"
        )
    lines.extend(["", "Paired cell-level contrasts:", ""])
    for name, row in historical["contrasts"].items():
        lines.append(
            f"- `{name}`: {row['delta']:+.4f} "
            f"[{row['cell_bootstrap_ci95'][0]:+.4f}, {row['cell_bootstrap_ci95'][1]:+.4f}], "
            f"{row['wins']}W/{row['ties']}T/{row['losses']}L."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "DEEM is not part of this primary run. Its soft input is a probability over the target "
            "class from each base learner, while these columns are alternative vocabulary ranks. "
            "A direct-rank DEEM adapter is a separate nonlinear experiment and should be attempted "
            "only after this run establishes that the rank representation itself carries useful signal.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("localization", "historical", "all"), default="all")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.k != DEFAULT_K:
        raise ValueError("v1 is frozen to K=15; no K search is allowed")
    configure_source_root(args.source_root)
    OUT = args.out.resolve()
    try:
        OUT.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("experiment outputs must stay inside the isolated worktree") from exc
    OUT.mkdir(parents=True, exist_ok=True)
    input_contract = validate_frozen_inputs(args.k)
    started = time.time()
    localization = None
    historical = None
    if args.stage in ("localization", "all"):
        localization = run_localization(args.k, args.bootstrap)
    elif (OUT / "LOCALIZATION.json").exists():
        localization = json.loads((OUT / "LOCALIZATION.json").read_text(encoding="utf-8"))
    if args.stage in ("historical", "all"):
        historical = run_historical(args.k, args.bootstrap)
    elif (OUT / "HISTORICAL_24.json").exists():
        historical = json.loads((OUT / "HISTORICAL_24.json").read_text(encoding="utf-8"))
    if localization is not None and historical is not None:
        (OUT / "REPORT.md").write_text(
            render_report(localization, historical), encoding="utf-8"
        )
    manifest = {
        "schema": "direct-probability-fusion-run-v1",
        "stage": args.stage,
        "k": args.k,
        "bootstrap": args.bootstrap,
        "elapsed_seconds": time.time() - started,
        "protocol": "docs/experiments/DIRECT_PROBABILITY_FUSION_V1.md",
        "input_contract": input_contract,
        "source_sha256": {
            "core": sha256_file(ROOT / "spectral_utils" / "direct_probability_fusion.py"),
            "driver": sha256_file(Path(__file__)),
            "protocol": sha256_file(ROOT / "docs" / "experiments" / "DIRECT_PROBABILITY_FUSION_V1.md"),
        },
    }
    write_json(OUT / "RUN_MANIFEST.json", manifest)
    print(f"[complete] {OUT} ({manifest['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
