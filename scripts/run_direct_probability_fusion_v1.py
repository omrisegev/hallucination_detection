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
from spectral_utils.historical_fusion_evaluation import pb_metrics  # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402


BENCH = ROOT / "results" / "localization_full_benchmark_v3"
FIXED_GATE = ROOT / "results" / "fusion_fixed_gate_v1"
FOLDS = ROOT / "results" / "localization_source_group_audit_v1" / "FOLDS_V2.json"
PB_DIRS = {
    "q4": ROOT / "dataset_cache" / "repgrid" / "pb_qwen3_4b",
    "q8": ROOT / "dataset_cache" / "repgrid" / "pb_qwen3_8b",
}
PRMB_TELEMETRY = (
    ROOT
    / "dataset_cache"
    / "four_localization"
    / "prmbench_qwen3_8b_telemetry_full"
    / "prmbench_telemetry.pkl"
)
PRMB_LABELS = (
    ROOT
    / "dataset_cache"
    / "four_localization"
    / "prmbench_qwen25math7b_full"
    / "prmbench_prm.pkl"
)
REPGRID = ROOT / "dataset_cache" / "repgrid"
REFERENCE_24 = ROOT / "results" / "benchmark_standing.csv"
STEP334 = ROOT / "results" / "token_level_readout_v1" / "METRICS.json"
OUT = ROOT / "results" / "direct_probability_fusion_v1"
METHODS = ("entropy", "rank_equal", "rank_iu", "rank_joint_lw")
FIT_METHOD = {
    "rank_equal": "equal",
    "rank_iu": "iu",
    "rank_joint_lw": "joint_lw",
}
BOOT_SEED = 2026091015


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


def _local_token_scores(row: dict, k: int) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    logprobs = logprob_matrix(row["top_k_logprobs"], k=k)
    entropy = np.asarray(row["token_entropies"], dtype=float)
    if entropy.shape != (len(logprobs),) or not np.isfinite(entropy).all():
        raise ValueError("entropy/top-k token alignment failure")
    risk = direct_rank_risk(logprobs)
    scores: dict[str, np.ndarray] = {"entropy": entropy}
    diagnostics: dict[str, dict] = {}
    for name, method in FIT_METHOD.items():
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
    return scores, diagnostics


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
    fallbacks: dict[str, Counter],
    alphas: dict[str, list[float]],
    weights: dict[str, list[np.ndarray]],
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
        }
        per[name] = {"within": within, "prediction": prediction, "valid": pb_valid}
        predictions_out[f"prediction__{name}"] = prediction

    # Exact continuity check against the accepted Step334 entropy row.
    old = json.loads(STEP334.read_text(encoding="utf-8"))["results"]["token_entropy"]
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

    comparisons = (
        ("rank_joint_lw", "entropy"),
        ("rank_iu", "entropy"),
    )
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(BOOT_SEED)
    draws = {f"{left}_minus_{right}": {"pb": [], "within": []} for left, right in comparisons}
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
    return {"methods": results, "contrasts": contrasts}, predictions_out


def run_localization(k: int, bootstrap: int) -> dict[str, Any]:
    print("[localization] loading frozen benchmark", flush=True)
    joined_json = json.loads((BENCH / "evaluation" / "JOINED.json").read_text(encoding="utf-8"))
    joined = np.load(BENCH / "evaluation" / "JOINED.npz", allow_pickle=False)
    records = joined_json["records"]
    offsets = joined["offsets"]
    step_scores = {
        name: np.full(int(offsets[-1]), np.nan, dtype=float) for name in METHODS
    }
    fallbacks = {name: Counter() for name in FIT_METHOD}
    alphas: dict[str, list[float]] = {name: [] for name in FIT_METHOD}
    weights: dict[str, list[np.ndarray]] = {name: [] for name in FIT_METHOD}

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
            token_scores, diagnostics = _local_token_scores(row, k)
            spans = np.asarray(row["step_token_spans"], dtype=int)
            if spans.shape != (records[index]["steps"], 2):
                raise ValueError(f"step-span mismatch for {records[index]['uid']}")
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
        del source

    print("[localization] source prmbench", flush=True)
    source = _source_row_map(load_pickle(PRMB_TELEMETRY), kind="prm")
    for index, record in enumerate(records):
        if record["cell"] != "prmbench_qwen3_8b":
            continue
        row = source[record["row_id"]]
        token_scores, diagnostics = _local_token_scores(row, k)
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
    del source

    for name, values in step_scores.items():
        if not np.isfinite(values).all():
            missing = int((~np.isfinite(values)).sum())
            raise ValueError(f"{name} has {missing} unfilled/nonfinite step scores")
    metrics, predictions = _localization_metrics(
        records, joined, step_scores, fallbacks, alphas, weights, bootstrap
    )
    np.savez_compressed(
        OUT / "LOCALIZATION_SCORES.npz",
        **{f"steps__{name}": values for name, values in step_scores.items()},
        **predictions,
    )
    payload = {
        "schema": "direct-probability-localization-v1",
        "k": k,
        "token_readout": "top-10 mean within each reasoning step",
        "gate": "frozen foldwise mean entropy q=0.3",
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
        return {row["cell"]: row for row in csv.DictReader(handle)}


def run_historical(k: int, bootstrap: int) -> dict[str, Any]:
    references = _historical_references()
    cells_out: dict[str, Any] = {}
    score_arrays: dict[str, np.ndarray] = {}
    for position, cell in enumerate(INSCOPE, start=1):
        print(f"[historical] {position:02d}/{len(INSCOPE)} {cell}", flush=True)
        payload = load_pickle(_historical_source(cell))
        candidates: list[dict] = []
        problem_ids: list[str] = []
        for problem_id in sorted(payload, key=str):
            for candidate in payload[problem_id]["candidates"]:
                candidates.append(
                    crop_candidate(candidate) if cell in CROPPED_CELLS else candidate
                )
                problem_ids.append(str(problem_id))
        rank_rows = []
        entropy = []
        varentropy = []
        labels = []
        for candidate in candidates:
            logprobs = logprob_matrix(candidate["top_k_logprobs"], k=k)
            rank_risk = direct_rank_risk(logprobs)
            rank_rows.append(answer_rank_features(rank_risk, count=10))
            token_entropy = np.asarray(candidate["token_entropies"], dtype=float)
            if len(token_entropy) != len(logprobs):
                raise ValueError(f"{cell}: entropy/top-k length mismatch")
            entropy.append(top_mean(token_entropy, count=10))
            varentropy.append(top_mean(topk_renormalized_varentropy(logprobs), count=10))
            labels.append(not bool(candidate.get("label", False)))
        X = np.asarray(rank_rows, dtype=float)
        entropy_array = np.asarray(entropy, dtype=float)
        label_array = np.asarray(labels, dtype=bool)
        method_scores: dict[str, np.ndarray] = {"entropy": entropy_array}
        diagnostics: dict[str, Any] = {}
        for name, method in FIT_METHOD.items():
            fit = fit_rank_fusion(X, method=method, anchor=entropy_array)
            method_scores[name] = fit.score
            diagnostics[name] = {
                "alpha": fit.alpha,
                "fallback": fit.fallback,
                "kept_ranks": int(len(fit.kept_ranks)),
                "weights": fit.weights.tolist(),
                "orientation_flipped": fit.orientation_flipped,
            }
        method_auc = {name: auc(label_array, score) for name, score in method_scores.items()}
        method_auc["varentropy"] = auc(label_array, np.asarray(varentropy, dtype=float))
        reference = references[cell]
        reference_upcr = float(reference["upcr.rho"])
        cells_out[cell] = {
            "domain": GROUP[cell],
            "n_answers": len(candidates),
            "n_problems": len(set(problem_ids)),
            "hallucination_rate": float(label_array.mean()),
            "auroc": method_auc,
            "historical_upcr_rho": reference_upcr,
            "rank_joint_lw_minus_historical_upcr": float(
                method_auc["rank_joint_lw"] - reference_upcr
            ),
            "diagnostics": diagnostics,
        }
        safe_cell = cell.replace("-", "_")
        score_arrays[f"{safe_cell}__label"] = label_array.astype(np.int8)
        for name, score in method_scores.items():
            score_arrays[f"{safe_cell}__{name}"] = np.asarray(score, dtype=np.float32)
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
    historical_upcr = np.asarray(
        [cells_out[cell]["historical_upcr_rho"] for cell in INSCOPE], dtype=float
    )
    macro["historical_upcr_rho"] = {
        "all24": float(np.mean(historical_upcr)),
        "qa9": float(
            np.mean([cells_out[cell]["historical_upcr_rho"] for cell in INSCOPE if GROUP[cell] == "QA"])
        ),
        "math15": float(
            np.mean([cells_out[cell]["historical_upcr_rho"] for cell in INSCOPE if GROUP[cell] == "math"])
        ),
    }

    comparisons = (
        ("rank_joint_lw", "entropy"),
        ("rank_iu", "entropy"),
        ("rank_joint_lw", "rank_iu"),
        ("rank_joint_lw", "historical_upcr_rho"),
    )
    vectors = {
        name: np.asarray(
            [
                cells_out[cell]["historical_upcr_rho"]
                if name == "historical_upcr_rho"
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
    np.savez_compressed(OUT / "HISTORICAL_24_SCORES.npz", **score_arrays)
    result = {
        "schema": "direct-probability-historical24-v1",
        "k": k,
        "answer_aggregation": "top-10 token mean separately for each probability rank",
        "fit_population": "all answers within each cell; labels excluded from fusion",
        "cells": cells_out,
        "macro": macro,
        "contrasts": contrasts,
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
            f"| `{name}` | {100*row['pb_all8']:.2f}% | {100*row['pb_q4']:.2f}% | "
            f"{100*row['pb_q8']:.2f}% | {row['prm_within']:.4f} | "
            f"{row['prmscore_q08']:.4f} | {sum(row['fallbacks'].values())} |"
        )
    lines.extend(["", "Primary paired contrasts (97.5% intervals):", ""])
    for name, row in localization["contrasts"].items():
        lines.append(
            f"- `{name}`: PB {100*row['pb_delta']:+.2f} pp "
            f"[{100*row['pb_ci_97_5'][0]:+.2f}, {100*row['pb_ci_97_5'][1]:+.2f}]; "
            f"PRMB within {row['prm_within_delta']:+.4f} "
            f"[{row['prm_within_ci_97_5'][0]:+.4f}, {row['prm_within_ci_97_5'][1]:+.4f}]."
        )
    lines.extend(
        [
            "",
            "## Complete-answer detection: historical 24 cells",
            "",
            "| method | all24 macro AUROC | QA9 | math15 |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (*METHODS, "varentropy", "historical_upcr_rho"):
        row = historical["macro"][name]
        lines.append(
            f"| `{name}` | {row['all24']:.4f} | {row['qa9']:.4f} | {row['math15']:.4f} |"
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
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.k != DEFAULT_K:
        raise ValueError("v1 is frozen to K=15; no K search is allowed")
    OUT = args.out.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
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
