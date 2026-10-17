#!/usr/bin/env python3
"""Audit pooling-scope OOF results and bootstrap ProcessBench contrasts."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "scripts"))
from run_claude_feature_bank_v1 import load_roster, write_json  # noqa: E402


RESULT_DIR = ROOT / "results" / "claude_feature_bank_pooling_scope_v1"
SCOPES = ("global", "model", "dataset", "cell")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predictions(step_scores: np.ndarray, offsets: np.ndarray, gate_open: np.ndarray) -> np.ndarray:
    peaks = np.empty(len(offsets) - 1, dtype=np.int32)
    for answer, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        local = step_scores[int(start):int(stop)]
        if not np.isfinite(local).all():
            raise ValueError(f"non-finite scores for answer {answer}")
        peaks[answer] = int(np.argmax(local))
    return np.where(gate_open, peaks, -1)


def weighted_macro_f1(
    target: np.ndarray,
    prediction: np.ndarray,
    cells: np.ndarray,
    weights: np.ndarray,
) -> float:
    correct = prediction == target
    values = []
    for cell in sorted(np.unique(cells)):
        mask = cells == cell
        clean = mask & (target == -1)
        error = mask & (target >= 0)
        clean_total = float(weights[clean].sum())
        error_total = float(weights[error].sum())
        if clean_total <= 0 or error_total <= 0:
            return float("nan")
        clean_accuracy = float(weights[clean & correct].sum() / clean_total)
        error_accuracy = float(weights[error & correct].sum() / error_total)
        denominator = clean_accuracy + error_accuracy
        values.append(2 * clean_accuracy * error_accuracy / denominator if denominator else 0.0)
    return float(np.mean(values))


def interval(samples: np.ndarray, point: float) -> dict[str, Any]:
    finite = samples[np.isfinite(samples)]
    return {
        "point": float(point),
        "ci95": [float(item) for item in np.quantile(finite, [0.025, 0.975])],
        "bootstrap_mean": float(finite.mean()),
        "probability_gt_zero": float(np.mean(finite > 0)),
        "valid_draws": int(len(finite)),
    }


def bootstrap(
    target: np.ndarray,
    cells: np.ndarray,
    groups: np.ndarray,
    predictions_by_name: dict[str, np.ndarray],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    names = sorted(predictions_by_name)
    points = {
        name: weighted_macro_f1(target, prediction, cells, np.ones(len(target)))
        for name, prediction in predictions_by_name.items()
    }
    contrasts: dict[str, tuple[str, str]] = {}
    for scope in ("model", "dataset", "cell"):
        contrasts[f"{scope}_l_sml_minus_global_l_sml"] = (f"{scope}__l_sml", "global__l_sml")
    for scope in SCOPES:
        contrasts[f"{scope}_l_sml_minus_{scope}_equal"] = (f"{scope}__l_sml", f"{scope}__equal")
    samples = {name: np.full(draws, np.nan, dtype=float) for name in contrasts}
    for draw in range(draws):
        sampled = rng.integers(0, len(unique_groups), size=len(unique_groups))
        group_counts = np.bincount(sampled, minlength=len(unique_groups))
        answer_weights = group_counts[inverse].astype(float)
        scores = {
            name: weighted_macro_f1(target, predictions_by_name[name], cells, answer_weights)
            for name in names
        }
        for contrast, (left, right) in contrasts.items():
            samples[contrast][draw] = scores[left] - scores[right]
    return {
        "unit": "source_group",
        "paired": True,
        "draws": draws,
        "seed": seed,
        "unique_groups": int(len(unique_groups)),
        "contrasts": {
            name: interval(values, points[left] - points[right])
            for name, values in samples.items()
            for left, right in [contrasts[name]]
        },
    }


def partition_stability(fits: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for fit in fits:
        grouped[(str(fit["scope"]), str(fit["scope_key"]))].append(fit)
    by_scope: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_key = {}
    for (scope, key), local in sorted(grouped.items()):
        pairwise = []
        for left, right in itertools.combinations(local, 2):
            left_groups = np.asarray(left["lsml_meta"]["groups"], dtype=int)
            right_groups = np.asarray(right["lsml_meta"]["groups"], dtype=int)
            ari = float(adjusted_rand_score(left_groups, right_groups))
            pairwise.append({
                "folds": [str(left["fold"]), str(right["fold"])],
                "adjusted_rand_index": ari,
                "exact": bool(np.array_equal(left_groups, right_groups)),
            })
        summary = {
            "scope": scope,
            "scope_key": key,
            "pairs": len(pairwise),
            "mean_ari": float(np.mean([row["adjusted_rand_index"] for row in pairwise])),
            "min_ari": float(np.min([row["adjusted_rand_index"] for row in pairwise])),
            "exact_pairs": int(sum(row["exact"] for row in pairwise)),
            "pairwise": pairwise,
        }
        by_key[f"{scope}:{key}"] = summary
        by_scope[scope].extend(pairwise)
    scope_summary = {
        scope: {
            "pairs": len(rows),
            "mean_ari": float(np.mean([row["adjusted_rand_index"] for row in rows])),
            "min_ari": float(np.min([row["adjusted_rand_index"] for row in rows])),
            "exact_pairs": int(sum(row["exact"] for row in rows)),
        }
        for scope, rows in sorted(by_scope.items())
    }
    return {"by_scope": scope_summary, "by_scope_key": by_key}


def run(*, draws: int, seed: int) -> Path:
    result_path = RESULT_DIR / "RESULTS.json"
    score_path = RESULT_DIR / "OOF_SCORES.npz"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    _, data = load_roster()
    offsets = np.asarray(data["offsets"], dtype=int)
    cells = np.asarray(data["cells"], dtype=str)
    pb = np.char.startswith(cells, "pb_")
    with np.load(score_path, allow_pickle=False) as loaded:
        gate_open = loaded["gate_open"].copy()
        finite = {
            name: bool(np.isfinite(loaded[name]).all())
            for name in loaded.files
            if name != "gate_score"
        }
        predictions_by_name = {
            f"{scope}__{method}": predictions(
                loaded[f"{scope}__{method}"], offsets, gate_open
            )[pb]
            for scope in SCOPES
            for method in ("l_sml", "equal")
        }
    bootstrap_result = bootstrap(
        np.asarray(data["target"])[pb],
        cells[pb],
        np.asarray(data["groups"], dtype=str)[pb],
        predictions_by_name,
        draws=draws,
        seed=seed,
    )
    k_counts = Counter(int(fit["lsml_meta"]["K"]) for fit in result["fits"])
    audit = {
        "schema": "claude-feature-bank-pooling-scope-v1/audit",
        "quality_result": False,
        "files": {
            "RESULTS.json": sha256(result_path),
            "OOF_SCORES.npz": sha256(score_path),
        },
        "arrays_finite": finite,
        "fits": len(result["fits"]),
        "degenerate_fits": int(sum(bool(fit["lsml_meta"]["degenerate"]) for fit in result["fits"])),
        "K_counts": {str(key): int(value) for key, value in sorted(k_counts.items())},
        "partition_stability": partition_stability(result["fits"]),
        "processbench_bootstrap": bootstrap_result,
    }
    path = RESULT_DIR / "AUDIT.json"
    write_json(path, audit)
    return path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026091801)
    args = parser.parse_args(argv)
    print(run(draws=args.draws, seed=args.seed))


if __name__ == "__main__":
    main()
