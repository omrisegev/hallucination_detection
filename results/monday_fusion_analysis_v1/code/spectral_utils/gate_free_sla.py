"""Gate-free Step-level Localization Accuracy on the frozen 13,769-answer population.

The metric is Mind-the-Gap's native Step-Level Accuracy at tolerance 0, over ERRONEOUS answers
only, with the prediction taken as the within-answer argmax of a risk-ascending step score. It
is "gate-free" in the sense the paper uses and that Omri asked for: no no-error decision enters
anywhere, so a gate can neither rescue nor mask a locator.

Validated against a real reported number, not only a fixture: replaying the Step-422 saved
out-of-fold scores through `sla_by_cell` reproduces the published 35.92 (token CONT L-SML) and
32.59 (token equal mean) with a maximum per-cell difference of 0.0, and the same 4,442 erroneous
ProcessBench answers. `scripts/validate_gate_free_sla.py` re-runs that check.

Three things this module exists to stop repeating:

* the PRMBench label trap - `target` is a -2 sentinel there, so the ProcessBench shortcut would
  silently mark all 6,969 PRMBench answers clean;
* the pooled trap - every statistic is per cell, because on this population a per-cell 0.68-0.78
  can pool to 0.46 (a full Simpson reversal, measured in Stage 1a of the white-box line);
* the fold trap - folds are keyed by SOURCE GROUP, and a group must never straddle folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: the frozen roster this module is defined against
N_ANSWERS = 13769
N_STEPS = 145597
N_GROUPS = 3483
EXPECTED_CELL_COUNTS = {
    "pb_gsm8k_q4": 400, "pb_gsm8k_q8": 400,
    "pb_math_q4": 1000, "pb_math_q8": 1000,
    "pb_olympiadbench_q4": 1000, "pb_olympiadbench_q8": 1000,
    "pb_omnimath_q4": 1000, "pb_omnimath_q8": 1000,
    "prmbench_qwen3_8b": 6969,
}


def load_roster(evaluation_dir: Path | str) -> dict:
    """Load the frozen v3 roster from a `localization_full_benchmark_v3/evaluation` directory.

    Takes a path because the roster is DATA and lives outside this worktree. Importing code
    across worktrees is silent version drift; reading a frozen data file across them is not.
    """
    base = Path(evaluation_dir)
    records = json.loads((base / "JOINED.json").read_text(encoding="utf-8"))["records"]
    with np.load(base / "JOINED.npz", allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if len(records) != N_ANSWERS:
        raise ValueError(f"expected {N_ANSWERS} records, found {len(records)}")
    if int(arrays["offsets"][-1]) != N_STEPS:
        raise ValueError(f"expected {N_STEPS} steps, found {int(arrays['offsets'][-1])}")
    cells = np.array([r["cell"] for r in records])
    counts = {cell: int((cells == cell).sum()) for cell in EXPECTED_CELL_COUNTS}
    if counts != EXPECTED_CELL_COUNTS:
        raise ValueError(f"cell counts drifted: {counts}")
    return {
        "cells": cells,
        "group_id": np.array([r["group_id"] for r in records]),
        "row_id": np.array([r["row_id"] for r in records]),
        "labels": arrays["labels"],
        "offsets": arrays["offsets"].astype(np.int64),
        "target": arrays["target"].astype(np.int64),
    }


def load_folds(folds_path: Path | str, group_id: np.ndarray) -> np.ndarray:
    """Per-answer outer fold, from the canonical source-group fold release.

    The fold key is the SOURCE GROUP, not the answer: PRMBench carries several perturbed
    versions of one source question and ProcessBench repeats identical questions under
    different answer ids, so folding by answer would put the same question on both sides.
    """
    outer = json.loads(Path(folds_path).read_text(encoding="utf-8"))["outer"]
    missing = sorted({g for g in group_id.tolist() if g not in outer})
    if missing:
        raise ValueError(f"{len(missing)} group ids absent from the fold release, "
                         f"first: {missing[:3]}")
    folds = np.array([outer[g] for g in group_id.tolist()], dtype=np.int64)
    straddling = {g for g in set(group_id.tolist())
                  if len(set(folds[group_id == g].tolist())) > 1}
    if straddling:
        raise ValueError(f"{len(straddling)} source groups straddle folds")
    return folds


def group_codes(group_id) -> tuple[np.ndarray, int]:
    """Integer source-group codes over the UNION of both benchmarks.

    66 group ids span both, so coding them separately would resample one question twice
    independently and understate the paired uncertainty.
    """
    uniques, codes = np.unique(np.asarray(group_id), return_inverse=True)
    return codes.astype(np.int64), int(len(uniques))


def first_error_step(labels, offsets, target, cells) -> np.ndarray:
    """Index of the first erroneous step in each answer, or -1 when the answer is clean.

    The two benchmarks encode this differently and neither can be derived from the other.
    ProcessBench stores the index in ``target`` (-1 = clean) and carries NO per-step labels.
    PRMBench carries per-step labels and a -2 ``target`` sentinel - so reading ``target`` for
    PRMBench, the obvious shortcut, would mark all 6,969 of its answers clean.
    """
    labels = np.asarray(labels)
    offsets = np.asarray(offsets, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    pb = np.char.startswith(np.asarray(cells).astype(str), "pb_")
    out = np.full(len(pb), -1, dtype=np.int64)
    out[pb] = target[pb]
    for i in np.flatnonzero(~pb):
        seg = labels[offsets[i]:offsets[i + 1]]
        hit = np.flatnonzero(seg == 1)
        out[i] = int(hit[0]) if hit.size else -1
    return out


def argmax_predictions(scores, offsets) -> np.ndarray:
    """Within-answer argmax of a RISK-ASCENDING step score. Ties go to the EARLIEST step.

    Earliest by construction rather than by `np.argmax`'s incidental first-occurrence: the
    target is the FIRST error, so on a tie the earlier step is the better-calibrated guess, and
    saying so stops the result depending on array order. Tied peaks are real here - the v3
    forensics found them worth over a point of SLA.

    Vectorised via reduceat because candidate sweeps call this hundreds of times.
    """
    scores = np.asarray(scores, dtype=np.float64)
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    peak = np.repeat(np.maximum.reduceat(scores, offsets[:-1]), counts)
    within = np.arange(len(scores)) - np.repeat(offsets[:-1], counts)
    masked = np.where(scores >= peak, within, np.iinfo(np.int64).max)
    return np.minimum.reduceat(masked, offsets[:-1])


def chance_by_cell(truth, offsets, cells) -> dict:
    """Uniform-guess SLA per cell: the mean of 1/n_steps over that cell's erroneous answers.

    Computed rather than quoted. The project's 16.58% anchor is an eight-ProcessBench-cell
    number and does not describe PRMBench, whose answers are longer.
    """
    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    err = np.asarray(truth) >= 0
    cells = np.asarray(cells)
    out = {p: float(np.mean(1.0 / counts[err & (cells == p)]))
           for p in sorted(set(cells.tolist()))}
    out["POOLED"] = float(np.mean(1.0 / counts[err]))
    return out


def shared_group_weights(n_groups: int, draws: int, seed: int) -> np.ndarray:
    """One multinomial weight matrix reused across every arm and cell - this is the pairing."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=draws).astype(float)


def sla_by_cell(pred, truth, codes, cells, n_groups, weights=None) -> dict:
    """Gate-free SLA per cell and pooled, with paired source-group bootstrap draws.

    Erroneous answers only: a clean answer has no first error to localize, so including them
    would blend the locator with a no-error decision the arm never makes.

    The bootstrap aggregates to GROUP totals before drawing. Expanding the [draws, n_groups]
    weight matrix to one column per answer would build a 10,000 x 10,477 array per panel
    (838 MB); summing hits and counts per group makes each draw a matrix-vector product.
    """
    truth = np.asarray(truth)
    err = truth >= 0
    hit = (np.asarray(pred) == truth) & err
    cells = np.asarray(cells)
    out: dict[str, dict] = {}
    for panel in sorted(set(cells.tolist())) + ["POOLED"]:
        mask = err if panel == "POOLED" else (err & (cells == panel))
        if not mask.any():
            out[panel] = {"point": float("nan"), "n": 0, "draws": None}
            continue
        entry = {"point": float(hit[mask].mean()), "n": int(mask.sum())}
        if weights is not None:
            g_hit = np.bincount(codes[mask], weights=hit[mask].astype(np.float64),
                                minlength=n_groups)
            g_n = np.bincount(codes[mask], minlength=n_groups).astype(np.float64)
            num, den = weights @ g_hit, weights @ g_n
            entry["draws"] = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
        out[panel] = entry
    return out


def paired_intervals(left: dict, right: dict, alpha: float = 0.05) -> dict:
    """Paired delta intervals between two ``sla_by_cell`` results sharing one weight matrix."""
    out = {}
    for panel in left:
        a, b = left[panel], right[panel]
        if a.get("draws") is None or b.get("draws") is None:
            out[panel] = None
            continue
        delta = a["draws"] - b["draws"]
        finite = delta[np.isfinite(delta)]
        out[panel] = {
            "point": a["point"] - b["point"],
            "low": float(np.quantile(finite, alpha / 2)),
            "high": float(np.quantile(finite, 1 - alpha / 2)),
            "probability_positive": float(np.mean(finite > 0)),
            "finite_draws": int(finite.size),
        }
    return out


def pb_mean(result: dict) -> float:
    """Unweighted mean over the eight ProcessBench cells.

    PRMBench is never folded in: it is one cell holding more than half the population, its
    answers carry several error steps, and its chance rate differs - averaging it with the
    eight would be a category error, not a summary.
    """
    return float(np.mean([v["point"] for k, v in result.items() if k.startswith("pb_")]))
