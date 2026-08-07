#!/usr/bin/env python
"""
score_processbench.py — step-level localization: reproduce "Mind the Gap" Table 3, with our arm
under the identical protocol.

Local CPU only; consumes the `processbench_<subset>.pkl` files written by
`cluster/run_teacher_forced.py`. Emits one CSV per subset plus a diagnostics JSON.

=============================================================================================
THE AVAILABILITY GATE RUNS BEFORE ANY SCORE, AND CAN DECLARE THE STEP ARM UNSCOREABLE
=============================================================================================
Our features have hard length floors: `compute_spectral_features` returns None below 8 tokens and
`compute_stft_features` returns **0.0, not NaN**, below 32 — so between 8 and 32 tokens several
views are constants that `np.isfinite` accepts and that would enter the fusion as information-free
columns. `our_arm.degenerate_features` NaNs them per step, which is the honest treatment, but it
means a corpus of short steps leaves a genuinely smaller pool.

ProcessBench steps are frequently short. So this script prints the step-length distribution and
the per-view availability FIRST, and if fewer than `MIN_VIEWS` views are finite on at least
`MIN_AVAIL` of steps, the step-as-item arm is recorded as UNSCOREABLE for that subset — with the
reason — instead of producing a number off three degenerate columns. The token arm has no such
floor (a 32-token window spans step boundaries) and carries the row in that case.

That is a property of the data, not a failure of the method, and the report says so either way.

THREE ARMS, AND ONLY ONE OF THEM KEEPS THE FEATURES AT THEIR NATIVE SCALE
-------------------------------------------------------------------------
  * `upcr_positional` — **the arm to read first.** Every window-based feature is computed ONCE
                    over the whole trace with its deployed parameters, and localization is the
                    position where the series peaks. This recovers the argmax that
                    `feature_utils` computes and discards (`sw_var_peak` on line 141; the rolling
                    permutation entropy; the per-frame STFT power) — and which the pool already
                    exposes exactly once, as `cusum_shift_idx`. See `positional_views`.
  * `upcr_step`   — U-PCR fitted on the pooled-STEPS population and read per step. Mirrors the
                    maintained `upcr_rho_oriented` bit-for-bit, but re-runs trace-scale
                    statistics on ~25-token slices, where `sw_var_peak` has ten window positions
                    instead of hundreds. Kept for comparison, not as the headline.
  * `upcr_token`  — the step fit's frozen parameters replayed over 32-token sliding windows, then
                    the paper's own EMA -> flux -> worst-drop pipeline. Same scale objection as
                    `upcr_step`, and it exists mainly because a window crossing step boundaries
                    survives a short-step corpus that starves the step arm.

Usage:
    python scripts/localization/score_processbench.py <dir> [--out-dir results/localization]
    python scripts/localization/score_processbench.py <dir> --subsets gsm8k --stride 4
"""
import argparse
import csv
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from evidence_drop import DEFAULTS, EVIDENCE_FNS                          # noqa: E402
from localization_metrics import NO_ERROR, evaluate, step_avg_scores, step_drop_scores  # noqa: E402
from our_arm import (                                                     # noqa: E402
    CANONICAL_POOL, step_feature_availability, step_feature_rows, upcr_arm_fit,
)
from positional_views import fit_positional_arm                           # noqa: E402
from positional_views import step_risk as positional_step_risk            # noqa: E402
from token_trace import STRIDE_DEFAULT, W_DEFAULT, trace_row              # noqa: E402

# The availability gate. 3 is `prepare_cell`'s own `min_size` and L-SML's information floor
# (Step 205: exactly zero information at 3 views, numerically undetermined at 4); 0.70 is the
# fraction of steps a view must be measurable on to count as a view at all. Both are declared
# here, before any data is read, so the gate cannot be relaxed to fit a result.
MIN_VIEWS = 3
MIN_AVAIL = 0.70

ALPHA_DEFAULT = 0.10


def load_subset(path):
    with open(path, "rb") as f:
        cache = pickle.load(f)
    return [cache[k] for k in sorted(cache)]


def diagnostics(rows):
    """Everything that decides whether the numbers below mean anything."""
    step_lens = [n for r in rows for n in r["align_diag"]["step_token_lengths"]]
    misaligned = [i for i, r in enumerate(rows) if r["align_diag"]["problems"]]
    labels = np.array([r["label"] for r in rows], dtype=int)
    return {
        "n_rows": len(rows),
        "n_with_error": int((labels != NO_ERROR).sum()),
        "n_all_correct": int((labels == NO_ERROR).sum()),
        "n_rows_misaligned": len(misaligned),
        "misaligned_idx": misaligned[:20],
        "n_steps_total": len(step_lens),
        "median_step_tokens": float(np.median(step_lens)) if step_lens else 0.0,
        "p10_step_tokens": float(np.percentile(step_lens, 10)) if step_lens else 0.0,
        "p90_step_tokens": float(np.percentile(step_lens, 90)) if step_lens else 0.0,
        "frac_steps_lt_8_tokens": float(np.mean([n < 8 for n in step_lens])) if step_lens else 0.0,
        "frac_steps_lt_32_tokens": float(np.mean([n < 32 for n in step_lens])) if step_lens else 0.0,
        "mean_tokens": float(np.mean([len(r["token_entropies"]) for r in rows])) if rows else 0.0,
    }


def pooled_step_features(rows):
    """(per-row list of step feature dicts, flat pooled feature dict, index map).

    The fit population is every step of every row pooled together — U-PCR needs one population to
    estimate its covariance structure on, and a per-row fit on ~12 steps would be meaningless.
    """
    per_row = [step_feature_rows(r, feat_names=CANONICAL_POOL) for r in rows]
    flat = [s for r in per_row for s in r]
    fd = {f: np.array([s.get(f, np.nan) for s in flat], dtype=float) for f in CANONICAL_POOL}
    return per_row, fd, [len(r) for r in per_row]


def unpool(values, lengths):
    """Split a flat per-step array back into one array per row."""
    out, i = [], 0
    for n in lengths:
        out.append(np.asarray(values[i:i + n], dtype=float))
        i += n
    return out


def window_fit_population(rows, W, stride, max_windows=20000, seed=0):
    """Pooled 32-token-window features, for when the pooled-STEP population cannot support a fit.

    This is the fallback that keeps the token arm alive on a short-step corpus. It is genuinely a
    different fit population from the step arm's, so whenever it is used the two arms stop being
    the same detector — `score_subset` records `token_fit_population` and the report must say so
    rather than presenting the curve as the table's own detector.

    Windows are drawn from rows in a shuffled order and capped, because a fit does not get better
    past ~20k samples and the full corpus would be ~2M windows.
    """
    from token_trace import window_features

    rng = np.random.default_rng(seed)
    flat, n_rows = [], 0
    for i in rng.permutation(len(rows)):
        r = rows[int(i)]
        _, feats = window_features(r.get("token_entropies"), r.get("token_spilled_energies"),
                                   W, max(int(stride), 4))
        flat.extend(feats)
        n_rows += 1
        if len(flat) >= max_windows:
            break
    fd = {f: np.array([s.get(f, np.nan) for s in flat], dtype=float) for f in CANONICAL_POOL}
    return fd, len(flat), n_rows


def baseline_step_scores(rows, ema_span, top_k):
    """The six paper methods at STEP level: {shannon, logtoku, ln_s} x {avg, drop}."""
    out = {}
    for base, fn in EVIDENCE_FNS.items():
        drop_rows, avg_rows, ok = [], [], True
        for r in rows:
            try:
                ev = fn(r, top_k)
            except (KeyError, TypeError, ValueError, IndexError):
                ok = False
                break
            drop_rows.append(step_drop_scores(ev, r["step_token_spans"], ema_span=ema_span))
            avg_rows.append(step_avg_scores(ev, r["step_token_spans"]))
        if ok:
            out[f"{base}_drop"] = drop_rows
            out[f"{base}_avg"] = avg_rows
    return out


def score_subset(path, out_dir, alpha, n_splits, seed, W, stride, ema_span, top_k):
    subset = os.path.basename(path).replace("processbench_", "").replace(".pkl", "")
    rows_all = load_subset(path)
    diag = diagnostics(rows_all)

    print(f"\n=== {subset} ===")
    print(f"  {diag['n_rows']} rows | {diag['n_with_error']} with an annotated error step | "
          f"{diag['n_all_correct']} all-correct")
    print(f"  steps: {diag['n_steps_total']} total, median {diag['median_step_tokens']:.0f} tok "
          f"(p10 {diag['p10_step_tokens']:.0f}, p90 {diag['p90_step_tokens']:.0f}) | "
          f"<8 tok {diag['frac_steps_lt_8_tokens']:.1%}  <32 tok "
          f"{diag['frac_steps_lt_32_tokens']:.1%}")
    if diag["n_rows_misaligned"]:
        print(f"  EXCLUDING {diag['n_rows_misaligned']} row(s) that failed the step-alignment "
              f"gate — their step spans do not cover the chain")

    rows = [r for r in rows_all if not r["align_diag"]["problems"]]
    if len(rows) < 20:
        print(f"  REJECT: only {len(rows)} aligned rows")
        return None, {**diag, "verdict": "REJECT", "reason": "fewer than 20 aligned rows"}
    labels = np.array([r["label"] for r in rows], dtype=int)

    # ── availability gate, BEFORE any score ──────────────────────────────────
    t0 = time.time()
    per_row, fd, lengths = pooled_step_features(rows)
    avail = step_feature_availability(per_row, CANONICAL_POOL)
    usable = sorted([f for f, a in avail.items() if a >= MIN_AVAIL], key=CANONICAL_POOL.index)
    print(f"  step features extracted in {time.time()-t0:.0f}s — "
          f"{len(usable)}/{len(CANONICAL_POOL)} views finite on >={MIN_AVAIL:.0%} of steps")
    thin = [(f, avail[f]) for f in CANONICAL_POOL if 0.0 < avail[f] < MIN_AVAIL]
    if thin:
        print("    below the bar: " + ", ".join(f"{f} {a:.0%}" for f, a in thin))

    scores, notes = {}, {}
    arm = None
    if len(usable) < MIN_VIEWS:
        notes["upcr_step"] = (f"UNSCOREABLE: only {len(usable)} view(s) finite on "
                              f">={MIN_AVAIL:.0%} of steps, below the floor of {MIN_VIEWS}")
        print(f"  STEP ARM UNSCOREABLE — {notes['upcr_step']}")
    else:
        arm = upcr_arm_fit(fd, labels=None)          # label-free; see upcr_arm_fit's docstring
        if arm is None:
            notes["upcr_step"] = "UNSCOREABLE: prepare_cell refused the pooled-step population"
            print(f"  STEP ARM UNSCOREABLE — {notes['upcr_step']}")
        else:
            print(f"  U-PCR fit on {arm.n} pooled steps: pool={len(arm.pool)}/"
                  f"{len(CANONICAL_POOL)} kept={arm.n_kept} anchor={arm.anchor_name} "
                  f"imputed={arm.n_imputed}")
            scores["upcr_step"] = unpool(arm.risk, lengths)

    # ── token arm: no step-length floor, so it survives what the step arm cannot ─────
    # When the step fit exists, the token arm REUSES it and the two are the same detector at two
    # resolutions. When it does not — which is exactly the short-step case the token arm was
    # built for — falling back to "no token arm either" would throw the row away for the one
    # reason the row most needs it. So the arm is fitted on the window population instead, and
    # `token_fit_population` records which happened. The distinction is reported, never elided.
    token_fit = None
    arm_token = arm
    if arm is None:
        t0 = time.time()
        fd_win, n_win, n_used = window_fit_population(rows, W, stride)
        arm_token = upcr_arm_fit(fd_win, labels=None)
        if arm_token is None:
            notes["upcr_token"] = "UNSCOREABLE: the window population also refused a fit"
            print(f"  TOKEN ARM UNSCOREABLE — {notes['upcr_token']}")
        else:
            token_fit = "windows"
            print(f"  step fit unavailable -> U-PCR fitted on {n_win} windows from {n_used} rows "
                  f"in {time.time()-t0:.0f}s: pool={len(arm_token.pool)}/{len(CANONICAL_POOL)} "
                  f"kept={arm_token.n_kept} anchor={arm_token.anchor_name}")
            notes["upcr_token"] = ("fitted on the WINDOW population, not the step population — "
                                   "this arm and the step arm are NOT the same detector here")
    else:
        token_fit = "steps"

    if arm_token is not None:
        t0 = time.time()
        scores["upcr_token"] = [trace_row(r, arm_token, W=W, stride=stride,
                                          ema_span=ema_span)["step_risk"] for r in rows]
        print(f"  token trace (W={W}, stride={stride}, fit on {token_fit}) over {len(rows)} "
              f"rows in {time.time()-t0:.0f}s")

    # ── the positional arm: the series the extractors already compute and discard ────
    # This is the arm that keeps each feature at its NATIVE trace scale. The other two re-run
    # trace-scale statistics on step slices / 32-token windows, where `sw_var_peak` has ten
    # window positions instead of hundreds and stops being a peak statistic at all. Here the
    # sliding-window variance, the CUSUM excursion, the rolling permutation entropy and the
    # per-frame STFT power are computed once over the whole trace with the DEPLOYED window
    # parameters, and localization is read off where they peak — which is the argmax
    # `feature_utils` computes and throws away (and already exposes, once, as
    # `cusum_shift_idx`).
    t0 = time.time()
    parm = fit_positional_arm(rows)
    if parm is None:
        notes["upcr_positional"] = "UNSCOREABLE: fewer than 3 usable positional views"
        print(f"  POSITIONAL ARM UNSCOREABLE — {notes['upcr_positional']}")
    else:
        scores["upcr_positional"] = [positional_step_risk(r, parm) for r in rows]
        print(f"  positional arm: fitted on {parm.n:,} pooled tokens in {time.time()-t0:.0f}s, "
              f"pool={len(parm.pool)} kept={parm.n_kept} anchor={parm.anchor_name}")

    # ── the six paper methods ────────────────────────────────────────────────
    scores.update(baseline_step_scores(rows, ema_span, top_k))

    # ── metrics ──────────────────────────────────────────────────────────────
    recs = []
    for name, by_row in scores.items():
        res = evaluate(by_row, labels, alpha=alpha, n_splits=n_splits, seed=seed)
        recs.append({"subset": subset, "method": name, **res})
    recs.sort(key=lambda r: (-r["sla"] if np.isfinite(r["sla"]) else 1.0))

    hdr = (f"{'method':16s} {'SLA':>14s} {'SLA +/-1':>14s} {'PB F1':>14s} "
           f"{'acc_err':>8s} {'acc_ok':>8s}")
    print("\n  " + hdr)
    print("  " + "-" * len(hdr))
    for r in recs:
        print(f"  {r['method']:16s} {r['sla']*100:8.2f}+/-{r['sla_sd']*100:4.2f} "
              f"{r['sla_tol1']*100:8.2f}+/-{r['sla_tol1_sd']*100:4.2f} "
              f"{r['f1']*100:8.2f}+/-{r['f1_sd']*100:4.2f} "
              f"{r['acc_erroneous']*100:7.1f} {r['acc_correct']*100:7.1f}")

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{subset}__processbench.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted(recs[0]))
        w.writeheader()
        w.writerows(recs)

    diag.update({
        "verdict": "SCORED", "n_rows_scored": len(rows), "alpha": alpha,
        "availability": {f: round(a, 4) for f, a in avail.items()},
        "usable_views": usable, "min_views": MIN_VIEWS, "min_avail": MIN_AVAIL,
        "notes": notes, "token_fit_population": token_fit,
        "arm_positional": None if parm is None else {
            "pool": parm.pool, "n_kept": parm.n_kept, "anchor": parm.anchor_name,
            "n_tokens_fit": int(parm.n)},
        "same_detector": token_fit == "steps",
        "arm": None if arm is None else {
            "pool": arm.pool, "n_kept": arm.n_kept, "anchor": arm.anchor_name,
            "n_steps_fit": arm.n, "dropped": arm.dropped,
        },
        "arm_token": None if arm_token is None else {
            "pool": arm_token.pool, "n_kept": arm_token.n_kept,
            "anchor": arm_token.anchor_name, "n_fit": arm_token.n,
        },
        "W": W, "stride": stride, "ema_span": ema_span,
    })
    with open(os.path.join(out_dir, f"{subset}__diagnostics.json"), "w") as f:
        json.dump(diag, f, indent=2, default=str)
    print(f"\n  -> {csv_path}")
    return recs, diag


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cell", help="directory holding processbench_<subset>.pkl")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "results", "localization",
                                                      "processbench"))
    ap.add_argument("--subsets", default=None, help="comma-separated; default all found")
    ap.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    ap.add_argument("--n-splits", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=W_DEFAULT, dest="W")
    ap.add_argument("--stride", type=int, default=STRIDE_DEFAULT)
    ap.add_argument("--ema-span", type=int, default=DEFAULTS["ema_span"])
    ap.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.cell, "processbench_*.pkl")))
    if a.subsets:
        want = {s.strip() for s in a.subsets.split(",")}
        paths = [p for p in paths
                 if os.path.basename(p).replace("processbench_", "").replace(".pkl", "") in want]
    if not paths:
        raise SystemExit(f"no processbench_*.pkl under {a.cell}")

    for p in paths:
        score_subset(p, a.out_dir, a.alpha, a.n_splits, a.seed, a.W, a.stride,
                     a.ema_span, a.top_k)


if __name__ == "__main__":
    main()
