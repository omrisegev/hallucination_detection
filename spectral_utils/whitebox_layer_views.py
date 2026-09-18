"""White-box per-layer field on the localization population: loader, labels, geometry, statistics.

Plan of record: docs/experiments/WHITEBOX_LAYER_VIEWS_LOCALIZATION_V1.md (v4).

This module deliberately does NOT go through
``spectral_utils.whitebox_layer_fusion.validate_and_join``. That path hard-requires all four
``(3, L, T)`` lens tensors plus ``gen_token_ids`` and per-candidate labels, because
``_feature_matrix`` attaches a ``risk_anchor`` read from ``lens_logp_tgt``. Satisfying it would
force pulling the full 5.5 GB token-level capture. The rotation-invariance contract we actually
need lives in three pure primitives, and those are imported directly and unmodified:

    _cosine_distance, _normalized_distance, _covariance_summaries

so the admissible-summary definition stays byte-identical to the registered one.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .whitebox_layer_fusion import (
    _cosine_distance,
    _normalized_distance,
    _covariance_summaries,
    all_layers,
)

__all__ = [
    "load_joined",
    "answer_error_label",
    "assert_label_contract",
    "n_steps_per_answer",
    "group_codes",
    "geometry_summaries",
    "auc_plan",
    "weighted_auc",
    "shared_group_weights",
    "auc_by_cell",
    "paired_auc_intervals",
]

# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------

EVALUATION = Path("results/localization_full_benchmark_v3/evaluation")

#: The nine cells, and the answer counts the extraction manifests agree on.
EXPECTED_CELL_COUNTS = {
    "pb_gsm8k_q4": 400,
    "pb_gsm8k_q8": 400,
    "pb_math_q4": 1000,
    "pb_math_q8": 1000,
    "pb_olympiadbench_q4": 1000,
    "pb_olympiadbench_q8": 1000,
    "pb_omnimath_q4": 1000,
    "pb_omnimath_q8": 1000,
    "prmbench_qwen3_8b": 6969,
}

N_ANSWERS = 13769
N_STEPS = 145597


def load_joined(root: Path | str = ".") -> dict:
    """Load the frozen v3 roster: per-answer records plus the step-level arrays.

    Returns a dict with ``records`` (list of 13,769 dicts), ``cells`` (str array),
    ``group_id`` (str array), and the npz arrays ``labels``, ``offsets``, ``target``.
    """
    base = Path(root) / EVALUATION
    with open(base / "JOINED.json", encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload["records"]
    arrays = np.load(base / "JOINED.npz", allow_pickle=False)

    if len(records) != N_ANSWERS:
        raise ValueError(f"expected {N_ANSWERS} records, found {len(records)}")
    offsets = arrays["offsets"]
    if int(offsets[-1]) != N_STEPS:
        raise ValueError(f"expected {N_STEPS} steps, found {int(offsets[-1])}")

    cells = np.array([r["cell"] for r in records])
    counts = {cell: int((cells == cell).sum()) for cell in EXPECTED_CELL_COUNTS}
    if counts != EXPECTED_CELL_COUNTS:
        raise ValueError(f"cell counts drifted: {counts}")

    return {
        "records": records,
        "arms": payload["arms"],
        "cells": cells,
        "group_id": np.array([r["group_id"] for r in records]),
        "row_id": np.array([r["row_id"] for r in records]),
        "labels": arrays["labels"],
        "offsets": offsets,
        "target": arrays["target"],
        "scores": arrays["scores"],
        "valid": arrays["valid"],
    }


def n_steps_per_answer(offsets: np.ndarray) -> np.ndarray:
    """Steps per answer. Identical to ``records[i]['steps']``; ``diff`` is the cheaper source."""
    return np.diff(np.asarray(offsets)).astype(np.int64)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def assert_label_contract(labels, offsets, target, cells) -> dict:
    """Assert the ProcessBench / PRMBench label partition the answer-level label depends on.

    The two label sources cannot be cross-checked against one another: every ProcessBench step
    carries ``labels == -2`` (no per-step labels exist there), so a "both sources agree" check
    would fail on all 4,442 PB error answers by construction. What is checkable for free is the
    partition itself, which is what drifted in Step 313.
    """
    labels = np.asarray(labels)
    target = np.asarray(target)
    cells = np.asarray(cells)
    offsets = np.asarray(offsets)

    pb = np.char.startswith(cells.astype(str), "pb_")
    steps = n_steps_per_answer(offsets)
    pb_steps = np.repeat(pb, steps)

    failures = []
    if not (labels[pb_steps] == -2).all():
        failures.append("ProcessBench steps carry per-step labels (expected all -2)")
    if not (labels[~pb_steps] != -2).all():
        failures.append("PRMBench steps missing per-step labels")
    if not (target[~pb] == -2).all():
        failures.append("ProcessBench-style target leaked into PRMBench rows")
    if not ((target[pb] >= -1) & (target[pb] < steps[pb])).all():
        failures.append("ProcessBench target out of range [-1, n_steps)")
    if failures:
        raise ValueError("label contract drifted: " + "; ".join(failures))

    return {
        "pb_answers": int(pb.sum()),
        "prm_answers": int((~pb).sum()),
        "pb_steps": int(pb_steps.sum()),
        "prm_steps": int((~pb_steps).sum()),
    }


def answer_error_label(labels, offsets, target, cells) -> np.ndarray:
    """Answer-level "contains an error", for both benchmarks.

    ``target >= 0`` alone is WRONG: all 6,969 PRMBench rows carry ``target == -2``, a
    not-applicable sentinel, so that rule scores every PRMBench answer negative. ProcessBench
    uses ``-1`` for clean and ``>= 0`` for the first-error step; PRMBench has no target and its
    per-step ``labels`` are the only source.
    """
    labels = np.asarray(labels)
    offsets = np.asarray(offsets)
    target = np.asarray(target)
    cells = np.asarray(cells)

    pb = np.char.startswith(cells.astype(str), "pb_")
    any_err = np.add.reduceat((labels == 1).astype(np.int64), offsets[:-1]) > 0
    return np.where(pb, target >= 0, any_err)


def group_codes(group_id) -> tuple[np.ndarray, int]:
    """Integer source-group codes over the UNION of both benchmarks.

    66 group ids appear in both ProcessBench and PRMBench cells, so a per-family coding would
    double-count them. Returns ``(codes, n_groups)`` with ``n_groups == 3483``.
    """
    unique, codes = np.unique(np.asarray(group_id), return_inverse=True)
    return codes.astype(np.int64), int(len(unique))


# ---------------------------------------------------------------------------
# Rotation-invariant geometry
# ---------------------------------------------------------------------------

def geometry_summaries(
    cov_eigs: np.ndarray,
    hid_proj: np.ndarray,
    resid_norm_mean: np.ndarray,
    layers=None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Rotation-invariant per-answer geometry, on the registered ``extract_geometry`` contract.

    Inputs are stacked over answers:
        cov_eigs        [N, L, R]
        hid_proj        [N, L, P]   cast to float64 here; float16 at rest is fine, float16
                                    accumulation in norms/dots is not
        resid_norm_mean [N, L]      per-layer token-mean of ||x_l,t||

    "Final" always means the architectural last layer ``L-1``, never the last selected layer,
    matching the registered implementation.
    """
    cov_eigs = np.asarray(cov_eigs, dtype=np.float64)
    hid_proj = np.asarray(hid_proj, dtype=np.float64)
    resid = np.asarray(resid_norm_mean, dtype=np.float64)

    n, n_layers = hid_proj.shape[0], hid_proj.shape[1]
    if cov_eigs.shape[:2] != (n, n_layers) or resid.shape != (n, n_layers):
        raise ValueError("cov_eigs / hid_proj / resid_norm_mean disagree on shape")

    selected = tuple(all_layers(n_layers) if layers is None else layers)
    eps = 1e-12
    columns: list[np.ndarray] = []
    names: list[str] = []
    groups: list[str] = []

    def add(name: str, group: str, values) -> None:
        columns.append(np.asarray(values, dtype=np.float64))
        names.append(name)
        groups.append(group)

    for layer in selected:
        if layer != n_layers - 1:
            add(f"geometry.hidden_cos_to_final.layer_{layer:02d}", "geometry.hidden_to_final",
                [_cosine_distance(hid_proj[i, layer], hid_proj[i, -1]) for i in range(n)])
            add(f"geometry.hidden_dist_to_final.layer_{layer:02d}", "geometry.hidden_to_final",
                [_normalized_distance(hid_proj[i, layer], hid_proj[i, -1]) for i in range(n)])
            add(f"geometry.resid_norm_convergence.layer_{layer:02d}", "geometry.resid_norm",
                np.abs(np.log((resid[:, layer] + eps) / (resid[:, -1] + eps))))
        if layer > 0:
            add(f"geometry.hidden_cos_adjacent.layer_{layer:02d}", "geometry.hidden_adjacent",
                [_cosine_distance(hid_proj[i, layer - 1], hid_proj[i, layer]) for i in range(n)])
            add(f"geometry.hidden_dist_adjacent.layer_{layer:02d}", "geometry.hidden_adjacent",
                [_normalized_distance(hid_proj[i, layer - 1], hid_proj[i, layer]) for i in range(n)])
        summaries = np.asarray([_covariance_summaries(cov_eigs[i, layer]) for i in range(n)],
                               dtype=np.float64)
        add(f"geometry.cov_top_share.layer_{layer:02d}", "geometry.covariance", summaries[:, 0])
        add(f"geometry.cov_neg_effective_rank.layer_{layer:02d}", "geometry.covariance",
            summaries[:, 1])
        add(f"geometry.cov_neg_spectral_entropy.layer_{layer:02d}", "geometry.covariance",
            summaries[:, 2])

    values = np.column_stack(columns)
    return values, names, groups


# ---------------------------------------------------------------------------
# AUROC and the paired source-group bootstrap
# ---------------------------------------------------------------------------
# Vendored verbatim from spectral_utils/historical_fusion_evaluation.py (repo root, branch
# codex/token-local-fusion-optimization-v1 @ 72d8235b4), which is the estimator the frozen v3
# benchmark used. Vendored rather than imported across worktrees: the root checkout sits on a
# different branch at a different commit, and importing across trees is silent version drift.
# The repo already carries three byte-identical copies of these two functions.
# Equality with sklearn.roc_auc_score is asserted in scripts/smoke_whitebox_layer_views.py.

def auc_plan(y, scores, groups):
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    starts = np.r_[0, np.flatnonzero(s[1:] != s[:-1]) + 1]
    return np.asarray(y, bool)[order], np.asarray(groups)[order], starts


def weighted_auc(plan, weights):
    y, groups, starts = plan
    w = weights[groups]
    pos = np.add.reduceat(w * y, starts)
    neg = np.add.reduceat(w * (~y), starts)
    p, n = pos.sum(), neg.sum()
    return float(np.dot(pos, np.cumsum(neg) - .5 * neg) / (p * n)) if p and n else np.nan


def shared_group_weights(n_groups: int, draws: int, seed: int) -> np.ndarray:
    """One multinomial weight matrix reused across every arm, cell and the pooled panel.

    Sharing the draw is what makes the comparison paired.
    """
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=draws).astype(float)


def auc_by_cell(y, scores, codes, cells, n_groups, weights=None) -> dict:
    """Point AUROC per cell plus pooled, and the bootstrap draws when ``weights`` is given.

    Groups absent from a cell simply contribute zero rows, so the global ``n_groups``-length
    weight vector is indexed directly rather than re-coded per cell.
    """
    y = np.asarray(y, bool)
    scores = np.asarray(scores, dtype=np.float64)
    cells = np.asarray(cells)
    ones = np.ones(n_groups)

    out: dict[str, dict] = {}
    panels = sorted(set(cells.tolist())) + ["POOLED"]
    for panel in panels:
        mask = np.ones(len(y), bool) if panel == "POOLED" else (cells == panel)
        if not mask.any() or y[mask].all() or not y[mask].any():
            out[panel] = {"point": float("nan"), "n": int(mask.sum()), "draws": None}
            continue
        plan = auc_plan(y[mask], scores[mask], codes[mask])
        entry = {"point": weighted_auc(plan, ones), "n": int(mask.sum()),
                 "positives": int(y[mask].sum())}
        if weights is not None:
            entry["draws"] = np.array([weighted_auc(plan, w) for w in weights])
        out[panel] = entry
    return out


def paired_auc_intervals(left: dict, right: dict, alpha: float = 0.05) -> dict:
    """Paired delta intervals between two ``auc_by_cell`` results sharing one weight matrix."""
    out = {}
    for panel in left:
        a, b = left[panel], right[panel]
        if a.get("draws") is None or b.get("draws") is None:
            out[panel] = None
            continue
        delta = a["draws"] - b["draws"]
        finite = delta[np.isfinite(delta)]
        if finite.size < 0.95 * delta.size:
            out[panel] = {"status": "TOO_MANY_DEGENERATE_DRAWS",
                          "finite": int(finite.size), "draws": int(delta.size)}
            continue
        out[panel] = {
            "point": a["point"] - b["point"],
            "low": float(np.quantile(finite, alpha / 2)),
            "high": float(np.quantile(finite, 1 - alpha / 2)),
            "probability_positive": float(np.mean(finite > 0)),
            "finite_draws": int(finite.size),
        }
    return out
