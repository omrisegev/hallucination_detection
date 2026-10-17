#!/usr/bin/env python
"""Item 4, stage 1 (2026-09-23): the window-representation measurement of atlas handoff B3.

Builds the declared ten-view moment bank on 8-token windows for every answer, measures the
conditional participation ratio (the 1.80 statistic) at window level and at step level after two
readouts, beside a per-channel within-answer token shuffle (an UPPER reference: it makes the
channels independent, so its PR sits near p), the effective sample size of the window series and
one marginal within-answer participation rank (to bridge the old 3.6-type figures).

Gate, pre-registered in docs/experiments/WINDOW_REPRESENTATION_B3_V1.md: the fusion stage runs
only if the real step-level conditional PR (Top10 readout) is >= 3.0 AND the shuffled reference is
above it by the declared margin. Labels are used for the measurement only (as in Step 414).

    python -B scripts/diagnostics/window_pr_measurement_v1.py --config configs/window_representation_b3_v1.json
    python -B scripts/diagnostics/window_pr_measurement_v1.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
import ct7_levers_common as L  # noqa: E402

L.ensure_spectral_package()
from spectral_utils.ct7_token_streams import STREAMS as CT7_STREAMS, despike_step0, masked_step_top10  # noqa: E402
from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402
from spectral_utils.window_localization import tokens_to_official_steps, windows_to_tokens  # noqa: E402
from spectral_utils.window_moment_bank import (  # noqa: E402
    conditional_participation_ratio, moment_bank, window_labels, window_plan,
)

SCHEMA = "window-representation-b3-v1"
ELEVEN = ["q15_H1", "q15_VE1", "chosen_surprisal", "logprob_margin", "true_tail50", "energy_level",
          "energy_innovation", "top15_turnover", "top50_js", "dominant_freq16", "bocpd_p0"]
BASE_STREAMS = ["q15_H1", "q15_VE1", "logprob_margin", "true_tail50", "energy_innovation", "bocpd_residual", "chosen_std_excess"]
DEFAULT_VIEWS = [["q15_H1", "level"], ["q15_H1", "sd"], ["q15_H1", "slope"], ["q15_VE1", "level"], ["q15_VE1", "sd"],
                 ["chosen_std_excess", "level"], ["logprob_margin", "level"], ["true_tail50", "level"],
                 ["energy_innovation", "level"], ["bocpd_residual", "level"]]
WIDTH, STRIDE, SHUFFLE_SEED, PR_GATE, SHUFFLE_MARGIN, MIN_FIT_FOR_RANK = 8, 1, 20260918, 3.0, 0.5, 12
CT7_ANCHOR_PR, CT7_ANCHOR_TOL = 1.80, 0.01


def effective_n(series, max_lag=64):
    """n / (1 + 2 sum of positive autocorrelations), truncated at the first non-positive lag
    (copy of scripts/diagnostics/readout_shape_and_neff_v1.py::effective_n)."""
    x = np.asarray(series, float); n = len(x)
    if n < 4 or x.std() < 1e-12:
        return float(n)
    x = x - x.mean(); denom = float((x * x).sum()); total = 0.0
    for lag in range(1, min(max_lag, n - 1) + 1):
        rho = float((x[:-lag] * x[lag:]).sum() / denom)
        if rho <= 0:
            break
        total += rho
    return float(n / (1 + 2 * total))


def base_streams(d, eleven_npz, ct7_npz):
    """Per answer [T x 7] base token streams (five from the eleven-bank, two from CT7's streams,
    the chosen-token stream despiked), plus the per-answer step spans."""
    e = np.load(eleven_npz); c = np.load(ct7_npz)
    assert list(e["channels"]) == ELEVEN and list(c["channels"]) == list(CT7_STREAMS)
    toff = np.asarray(e["token_offsets"], int); assert np.array_equal(toff, c["token_offsets"])
    spans = np.asarray(e["step_spans"], int); assert np.array_equal(spans, c["step_spans"])
    et = e["tokens"]; ct = c["tokens"]; cv = np.asarray(c["valid"], bool)
    mats, sp = [], []
    for i in range(d.n):
        ta, tb = toff[i:i + 2]; a, b = d.off[i:i + 2]
        x7 = despike_step0(ct[ta:tb].astype(float), cv[ta:tb], spans[a:b], [6])
        x = np.column_stack([et[ta:tb, [0, 1, 3, 4, 6]].astype(float), x7[:, 5], x7[:, 6]])
        mats.append(x); sp.append(spans[a:b])
    return mats, sp


def answer_views(x, sp, views, *, rng=None):
    """Window bank of one answer (optionally on per-channel shuffled tokens)."""
    if rng is not None:
        x = np.column_stack([rng.permutation(x[:, j]) for j in range(x.shape[1])])
    plan = window_plan(len(x), WIDTH, STRIDE)
    V, cols = moment_bank(x, BASE_STREAMS, plan, [tuple(v) for v in views])
    return plan, V, cols


def measure(d, mats, sp, views, *, shuffle: bool):
    rng = np.random.default_rng(SHUFFLE_SEED) if shuffle else None
    total = int(d.off[-1]); p = len(views)
    step_mean = np.full((total, p), np.nan); step_top = np.full((total, p), np.nan)
    win_rows, win_y = [], []; neff = []; prank = []; uncovered = 0; fit_counts = []
    for i in range(d.n):
        x = mats[i]; a, b = d.off[i:i + 2]
        if len(x) < WIDTH:
            uncovered += 1; continue
        plan, V, _ = answer_views(x, sp[i], views, rng=rng)
        fit = V[plan.fit_indices]; fit_counts.append(len(fit))
        for j in range(p):
            tok = windows_to_tokens(plan, V[:, j])
            step_mean[a:b, j] = tokens_to_official_steps(tok, sp[i][:, 0], sp[i][:, 1])
            step_top[a:b, j] = masked_step_top10(tok, np.ones(len(tok), bool), sp[i])
        if d.prm[i]:
            y, inside = window_labels(plan, sp[i], d.labels[a:b])
            if inside.any():
                z = V[inside] - V.mean(0); sd = V.std(0); z = z / np.where(sd > 1e-12, sd, 1.0)
                win_rows.append(z); win_y.append(y[inside])
        if len(fit) >= MIN_FIT_FOR_RANK:
            neff.append([effective_n(fit[:, j]) / len(fit) for j in range(p)])
            zf = (fit - fit.mean(0)) / np.where(fit.std(0) > 1e-12, fit.std(0), 1.0); zf -= zf.mean(0)
            s = np.linalg.svd(zf, compute_uv=False); eig = s * s / len(fit)
            prank.append(float(eig.sum() ** 2 / np.square(eig).sum()))
    prm_steps = np.repeat(d.prm, np.diff(d.off))
    out = {"uncovered_answers_below_width": uncovered, "fit_windows_median": float(np.median(fit_counts)) if fit_counts else None}
    for name, S in (("step_topk10", step_top), ("step_overlap_mean", step_mean)):
        Z = masked_answer_standardize(np.nan_to_num(S), np.isfinite(S), d.off)
        ok = prm_steps & np.isfinite(S).all(1)
        out["pr_" + name] = conditional_participation_ratio(Z[ok], d.labels[ok] == 1)
        out["n_" + name] = int(ok.sum())
    if win_rows:
        W = np.vstack(win_rows); Y = np.concatenate(win_y)
        out["pr_window_level"] = conditional_participation_ratio(W, Y == 1); out["n_windows_labelled"] = int(len(W))
    out["neff_over_n_median_per_view"] = np.median(np.array(neff), axis=0).tolist() if neff else None
    out["marginal_participation_rank_median"] = float(np.median(prank)) if prank else None
    return out


def synthetic_inputs(tmp: Path):
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
    from ct7_token_lsml_v1 import synthetic_token_data
    d = synthetic_token_data(tmp)
    c = np.load(tmp / "CT7_TOKEN_MATRICES.npz"); toff = c["token_offsets"]; rng = np.random.default_rng(5)
    tokens = rng.standard_normal((int(toff[-1]), 11)).astype(np.float32)
    for i in range(d.n):
        t = d.target[i]
        if t >= 0:
            a, b = c["step_spans"][d.off[i] + t]; tokens[toff[i] + a:toff[i] + b] += 1.0
    np.savez(tmp / "TOKEN_MATRICES.npz", tokens=tokens, token_offsets=toff, step_spans=c["step_spans"], channels=np.asarray(ELEVEN, dtype=str))
    return d


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(); started = time.perf_counter()
    if args.dry_run:
        tmp = Path(tempfile.mkdtemp(prefix="window_pr_dry_")); d = synthetic_inputs(tmp)
        views = DEFAULT_VIEWS; eleven, ct7t = tmp / "TOKEN_MATRICES.npz", tmp / "CT7_TOKEN_MATRICES.npz"; out = d.out; anchor = None
    else:
        d = L.light_dataset(args.config); paths = d.c["paths"]; views = d.c.get("views", DEFAULT_VIEWS); out = d.out
        eleven, ct7t = Path(paths["tokens"]), Path(paths["ct7_tokens"])
        L.run_freeze(out, [Path(__file__), L.ROOT / "scripts/experiments/ct7_levers_common.py", L.ROOT / "spectral_utils/window_moment_bank.py",
                           L.ROOT / "spectral_utils/ct7_token_streams.py", Path(args.config).resolve()],
                     [Path(paths[k]) for k in ("roster", "joined", "folds", "ct7", "tokens", "ct7_tokens", "profiles") if k in paths],
                     {"schema": SCHEMA, "stage": "measurement", "views": views, "development_only": True})
        profiles = L.load_ct7_profiles(d, paths["profiles"], paths.get("profile_validation"))
        prm_steps = np.repeat(d.prm, np.diff(d.off))
        anchor = conditional_participation_ratio(profiles[prm_steps], d.labels[prm_steps] == 1)
        assert abs(anchor - CT7_ANCHOR_PR) < CT7_ANCHOR_TOL, f"CT7 anchor PR {anchor} is not {CT7_ANCHOR_PR}"
    mats, sp = base_streams(d, eleven, ct7t)
    real = measure(d, mats, sp, views, shuffle=False)
    shuffled = measure(d, mats, sp, views, shuffle=True)
    gate = bool(real["pr_step_topk10"] >= PR_GATE and shuffled["pr_step_topk10"] >= real["pr_step_topk10"] + SHUFFLE_MARGIN)
    record = {"schema": SCHEMA, "development_only": True, "note": L.DEVELOPMENT_NOTE, "views": views, "width": WIDTH, "stride": STRIDE,
              "ct7_anchor_pr": anchor, "real": real, "shuffled_reference": shuffled,
              "gate": {"rule": f"real step-level PR (Top10 readout) >= {PR_GATE} and shuffled >= real + {SHUFFLE_MARGIN}", "passed": gate},
              "gate_passed": gate, "seconds": time.perf_counter() - started, "dry_run": bool(args.dry_run)}
    L.dump(out / "WINDOW_PR.json", record)
    print(json.dumps({k: record[k] for k in ("ct7_anchor_pr", "real", "shuffled_reference", "gate")}, indent=1))
    print("GATE_PASSED" if gate else "GATE_NOT_PASSED", "written:", out / "WINDOW_PR.json", f"({time.perf_counter() - started:.1f}s)")


if __name__ == "__main__":
    main()
