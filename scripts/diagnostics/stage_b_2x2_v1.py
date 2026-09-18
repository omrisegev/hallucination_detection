#!/usr/bin/env python
"""Stage B: standardization axis x fusion stage, on one fixed bank and one fold set.

Pre-registration and its two amendments:
`docs/experiments/TOKEN_PROBABILITY_FUSION_V1_STAGE_B_PREREGISTRATION.md`.
Every prediction there was recorded before this file was written.

  C1  pooled       x fuse BEFORE readout   the published arm; must replay exactly
  C2  answer-local x fuse BEFORE readout   aligns the fit axis with the readout axis
  C3  pooled       x fuse AFTER  readout
  C4  answer-local x fuse AFTER  readout   <- CT7's architecture, on this bank

C4 is labelled deliberately: answer-local standardization with fusion applied after a
per-channel step readout is exactly what CT7 does. That is what makes this 2x2 a bridge
between the two lines rather than four arbitrary arms.

One code path serves all four cells. A cell differs only in (a) which matrix enters --
token rows for BEFORE, per-channel step readouts for AFTER -- and (b) whether that
matrix is standardized within the answer first. The donor-fold standardizer and the
L-SML fit are then identical in every cell. On the answer-local cells the donor
standardizer comes out near (0, 1) by construction; it is kept rather than replaced by
an identity so that no cell gets a different code path.

Development-only: the fold structure is source-disjoint but this population has been
inspected before.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
TOKENS = RES / "TOKEN_MATRICES.npz"
CHANNELS = RES / "DERIVATIVE_CHANNELS.npz"
PUBLISHED_OOF = ROOT / "results" / "claude_feature_bank_token_lsml_v1" / "OOF_SCORES.npz"
OUT = RES / "STAGE_B_2X2.json"

BOOTSTRAP_DRAWS = 10_000
SEED = 20260918
TOKEN_CAP = 60_000
SHORT, LONG = ("gsm8k", "math"), ("olympiadbench", "omnimath")

from spectral_utils.claude_feature_bank_v1 import (  # noqa: E402
    FEATURE_NAMES, fit_l_sml_weights, fit_token_standardizer, fuse_token_matrix,
)
from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gate_isolation_v1", Path(__file__).with_name("gate_isolation_token_lsml_v1.py"))
_canon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_canon)
peaks_of, sla_gate_free = _canon.peaks_of, _canon.sla_gate_free


def step_top10(values: np.ndarray, spans: np.ndarray, k: int = 10) -> np.ndarray:
    out = np.empty(len(spans))
    for i, (a, b) in enumerate(spans):
        seg = values[a:b]
        kk = min(k, len(seg))
        out[i] = np.partition(seg, len(seg) - kk)[-kk:].mean()
    return out


def answer_local(matrix: np.ndarray) -> np.ndarray:
    """Standardize each channel within this answer. Zero variance -> zero column."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    return np.where(std > 1e-8, (matrix - mean) / np.where(std > 1e-8, std, 1.0), 0.0)


def conditional_participation_ratio(views: np.ndarray, valid: np.ndarray, y: np.ndarray) -> float:
    """(sum lambda)^2 / sum(lambda^2) of the within-label-centred view correlation.

    The project's 1.80 / 2.46 / 2.83 are this quantity. It is NOT the weight-spread
    statistic that the runner calls effective_rank; see amendment 1.
    """
    m = views[valid].copy()
    for cls in (True, False):
        sel = y == cls
        if sel.any():
            m[sel] -= m[sel].mean(0)
    keep = m.std(0) > 1e-12
    if keep.sum() < 2:
        return float(keep.sum())
    lam = np.maximum(np.linalg.eigvalsh(np.corrcoef(m[:, keep].T)), 0.0)
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def weight_ipr(weights: np.ndarray) -> float:
    """1 / sum((|w_i| / sum|w|)^2). Ceiling = n channels. Sign-blind (amendment 1)."""
    w = np.abs(np.asarray(weights, float))
    return float(1.0 / np.sum((w / w.sum()) ** 2))


def _group_draws(groups, mask, rng, draws):
    order = np.argsort(groups[mask], kind="stable")
    flat = np.flatnonzero(mask)[order]
    _, starts, counts = np.unique(groups[flat], return_index=True, return_counts=True)
    for _ in range(draws):
        d = rng.integers(0, len(starts), size=len(starts))
        take, base = counts[d], starts[d]
        ends = np.cumsum(take)
        within = np.arange(int(ends[-1])) - np.repeat(ends - take, take)
        yield flat[np.repeat(base, take) + within]


def interval(diff: np.ndarray) -> dict:
    d = diff[np.isfinite(diff)]
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return {"point_pp": float(100 * d.mean()), "ci95_pp": [float(100 * lo), float(100 * hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def run_cell(matrices: list[np.ndarray], spans: list[np.ndarray] | None,
             folds: np.ndarray, offsets: np.ndarray) -> dict:
    """Fit per fold on donors, score held-out. spans=None means the rows ARE steps."""
    total = int(offsets[-1])
    l_sml = np.full(total, np.nan)
    equal = np.full(total, np.nan)
    views = np.full((total, matrices[0].shape[1]), np.nan)
    fits = []
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        std = fit_token_standardizer((matrices[i] for i in train), cap=TOKEN_CAP)
        w, meta = fit_l_sml_weights((matrices[i] for i in train), std)
        for i in test:
            fl, fe = fuse_token_matrix(matrices[i], std, weights=w)
            a, b = int(offsets[i]), int(offsets[i + 1])
            if spans is None:
                l_sml[a:b], equal[a:b] = fl, fe
                views[a:b] = (matrices[i] - std["mean"]) / std["std"]
            else:
                l_sml[a:b] = step_top10(fl, spans[i])
                equal[a:b] = step_top10(fe, spans[i])
                for c in range(matrices[i].shape[1]):
                    views[a:b, c] = step_top10(
                        (matrices[i][:, c] - std["mean"][c]) / std["std"][c], spans[i])
        fits.append({"fold": str(fold), "weights": w.tolist(), "weight_ipr": weight_ipr(w),
                     "K": int(meta["K"]), "train_answers": int(len(train))})
    if not (np.isfinite(l_sml).all() and np.isfinite(equal).all()):
        raise ValueError("cell produced non-finite step scores")
    return {"l_sml": l_sml, "equal": equal, "views": views, "fits": fits}


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
        labels = np.asarray(z["labels"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)
    pb = np.char.startswith(cells, "pb_")

    with np.load(TOKENS, allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)
    with np.load(CHANNELS, allow_pickle=False) as z:
        level = z["level"].copy()

    n = len(records)
    tok_mats = [tokens[tok_off[i]:tok_off[i + 1]] for i in range(n)]
    # step_token_spans are already 0-based WITHIN the answer, and were cached verbatim.
    # Subtracting the answer's token offset would shift them into nonsense.
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n)]
    lvl_mats = [level[offsets[i]:offsets[i + 1]] for i in range(n)]
    for i in (0, n // 2, n - 1):  # cheap guard: spans must index inside their own answer
        s = spans[i]
        if s[0, 0] != 0 or s[-1, 1] != len(tok_mats[i]):
            raise ValueError(f"answer {i}: spans {s[0, 0]}..{s[-1, 1]} vs {len(tok_mats[i])} tokens")

    design = {
        "C1_pooled_before": (tok_mats, spans),
        "C2_answerlocal_before": ([answer_local(m) for m in tok_mats], spans),
        "C3_pooled_after": (lvl_mats, None),
        "C4_answerlocal_after": ([answer_local(m) for m in lvl_mats], None),
    }

    report = {"schema": "token-probability-fusion-v1-stage-b", "development_only": True,
              "cells": {}, "note_c4": "C4 is CT7's architecture applied to this bank"}
    out_scores: dict[str, np.ndarray] = {}

    for name, (mats, sp) in design.items():
        print(f"[cell] {name}", flush=True)
        res = run_cell(mats, sp, folds, offsets)
        out_scores[f"{name}__l_sml"] = res["l_sml"]
        out_scores[f"{name}__equal"] = res["equal"]
        prm = np.repeat(np.char.startswith(cells, "prmbench_"), np.diff(offsets))
        valid = prm & (labels >= 0)
        report["cells"][name] = {
            "fits": res["fits"],
            "weight_ipr_mean": float(np.mean([f["weight_ipr"] for f in res["fits"]])),
            "chosen_surprisal_weight_by_fold":
                [f["weights"][FEATURE_NAMES.index("chosen_surprisal")] for f in res["fits"]],
            "conditional_participation_ratio":
                conditional_participation_ratio(res["views"], valid, labels[valid] == 1),
        }
        for rule in ("l_sml", "equal"):
            per = sla_gate_free(peaks_of(res[rule], offsets), target, cells)
            report["cells"][name][f"sla_{rule}"] = per
            report["cells"][name][f"mean_sla_{rule}"] = float(np.mean([v["sla"] for v in per.values()]))

    # ---- C1 must replay the published arm ----------------------------------
    with np.load(PUBLISHED_OOF, allow_pickle=False) as z:
        pub_l, pub_e = z["l_sml"].copy(), z["equal"].copy()
    report["c1_replay"] = {
        "l_sml_max_abs_diff": float(np.max(np.abs(out_scores["C1_pooled_before__l_sml"] - pub_l))),
        "equal_max_abs_diff": float(np.max(np.abs(out_scores["C1_pooled_before__equal"] - pub_e))),
        "l_sml_peak_agreement": float(np.mean(
            peaks_of(out_scores["C1_pooled_before__l_sml"], offsets) == peaks_of(pub_l, offsets))),
    }

    # ---- paired intervals: L-SML minus equal, inside each cell --------------
    rng = np.random.default_rng(SEED)
    acc = {f"{c}__{r}": [] for c in design for r in ("l_sml", "equal")}
    peaks = {k: peaks_of(v, offsets) for k, v in out_scores.items()}
    for i in _group_draws(groups, pb & (target >= 0), rng, BOOTSTRAP_DRAWS):
        t, c = target[i], cells[i]
        for k, p in peaks.items():
            per = sla_gate_free(p[i], t, c)
            acc[k].append(float(np.mean([v["sla"] for v in per.values()])))
    draws = {k: np.asarray(v) for k, v in acc.items()}
    report["intervals_lsml_minus_equal"] = {
        c: interval(draws[f"{c}__l_sml"] - draws[f"{c}__equal"]) for c in design}
    report["intervals_vs_C1_lsml"] = {
        c: interval(draws[f"{c}__l_sml"] - draws["C1_pooled_before__l_sml"])
        for c in design if c != "C1_pooled_before"}

    np.savez_compressed(RES / "STAGE_B_SCORES.npz", **out_scores)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    # ------------------------------------------------------------------ console
    print()
    print("=" * 96)
    print("STAGE B 2x2 -- gate-free mean SLA over the eight ProcessBench cells")
    print("=" * 96)
    print(f"{'cell':26s} {'L-SML':>8s} {'equal':>8s} {'L-SML-equal':>28s} {'cond PR':>9s} {'w-IPR':>7s}")
    for c in design:
        b = report["cells"][c]
        iv = report["intervals_lsml_minus_equal"][c]
        flag = "*" if iv["excludes_zero"] else " "
        print(f"{c:26s} {100*b['mean_sla_l_sml']:8.2f} {100*b['mean_sla_equal']:8.2f} "
              f"{iv['point_pp']:+7.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]{flag}"
              f" {b['conditional_participation_ratio']:9.2f} {b['weight_ipr_mean']:7.2f}")
    print()
    print("C1 replay vs published OOF:", {k: round(v, 12) for k, v in report["c1_replay"].items()})
    print()
    print("chosen_surprisal weight by fold (amendment 1: sign matters, the IPR cannot see it)")
    for c in design:
        w = report["cells"][c]["chosen_surprisal_weight_by_fold"]
        print(f"  {c:26s} " + " ".join(f"{v:+.3f}" for v in w) +
              f"   non-negative in {sum(v >= 0 for v in w)}/5 folds")
    print()
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
