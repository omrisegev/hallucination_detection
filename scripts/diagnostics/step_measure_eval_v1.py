#!/usr/bin/env python
"""Replace only the step-measurement stage: 2x2 of {raw, whitened} x {Top-K, best window}.

Scope, fixed by Omri on 2026-09-19: the fusion is NOT reopened. The eleven-channel bank,
the pooled standardizer and the L-SML weights are taken as given, and the input to this
file is the finished fused token series cached in `C1_TOKEN_SERIES.npz`. Only the stage
that turns the tokens inside a step into one number is varied, and anything it fits is
fitted **per model** over all of that model's steps.

Why this stage. Its width alone moves gate-free SLA from 20.42 at K=1 to 37.01 at K=20 --
a 16.6 pp range -- against 3 to 8 pp for every decision rule tried on top of it. It is
the expensive stage, and it is the one nobody has designed.

Two defects, one arm each, crossed:

  whitening        the fused token series has lag-1 autocorrelation 0.500 and `n_eff` of
                   22-29 per step against 55-93 raw tokens, so a flat K-window carries
                   about 3x the variance it would on independent samples. A Top-K mean
                   treats the tokens as white. The AR whitener is fitted per model, on
                   training folds only, from within-answer autocorrelations.
  contiguity       a Top-K mean takes the K largest values ANYWHERE in the step. If the
                   error is a contiguous burst, the matched statistic is the best
                   contiguous run instead. Windows never cross a step boundary.

Pre-registered before the run, in `docs/experiments/STEP_MEASURE_V1_PREREGISTRATION.md`:

  P1  If the K=10-to-20 optimum exists BECAUSE the noise is correlated, then after
      whitening the best width moves DOWN toward 1 and the peak moves UP. Falsified if
      whitening leaves the optimum where it is or lowers the peak.
  P2  If the error is a contiguous burst, the best contiguous window beats the Top-K mean
      at matched effective width. Falsified if it does not.
  P3  Per-model fitting beats one pooled whitener. Falsified if the pooled whitener is at
      or above it -- in which case the extra scope is not earning anything, exactly as it
      did not for the fusion stage (-0.22 short / -0.65 long, both covering zero).

Label-free selection reuses criterion 1 from `readout_calibration_v1.py` verbatim in
spirit: split-half odd/even token reproducibility, with the count proportionally reduced
on each half so that each half computes the same statistic as the whole. The label-using
best of the sweep is reported as a CEILING, never as a candidate.

Development-only. Anchor: raw Top-10 must replay 35.92.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
RES = ROOT / "results" / "token_probability_fusion_v1"
OUT = RES / "STEP_MEASURE.json"
PEAKS = RES / "STEP_MEASURE_PEAKS.npz"

AR_ORDER = 8            # autocorrelation is past 0.1 by lag 8 and past 0.03 by lag 40
K_GRID = (1, 3, 5, 10, 20, 40, 80)
W_GRID = (1, 2, 4, 8, 16, 32, 64)
BOOTSTRAP_DRAWS = 10_000
SEED = 20260919
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")

from spectral_utils.step_measure_v1 import (  # noqa: E402
    accumulate_autocorrelation, best_window_steps, topk_mean_steps, whiten, yule_walker,
)

_spec = importlib.util.spec_from_file_location(
    "stage_b_2x2_v1", Path(__file__).with_name("stage_b_2x2_v1.py"))
_sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sb)
_group_draws, interval = _sb._group_draws, _sb.interval


def subset_of(cell: str) -> str:
    return cell.split("_", 1)[1].rsplit("_", 1)[0] if cell.startswith("pb_") else "prm"


def measure(series: np.ndarray, spans: np.ndarray, kind: str, size: int) -> np.ndarray:
    return (topk_mean_steps(series, spans, size) if kind == "topk"
            else best_window_steps(series, spans, size))


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)
    subsets = np.asarray([subset_of(c) for c in cells], str)
    models = np.asarray(["q4" if c.endswith("_q4") else "q8" for c in cells], str)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)
    pb = np.char.startswith(cells, "pb_")

    with np.load(RES / "C1_TOKEN_SERIES.npz", allow_pickle=False) as z:
        fused = np.asarray(z["token_l_sml"], float)
        tok_off = np.asarray(z["token_offsets"], int)
    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        step_spans = np.asarray(z["step_spans"], int)

    n = len(records)
    raw = [fused[tok_off[i]:tok_off[i + 1]] for i in range(n)]
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n)]

    # ---- whiteners: per model, and one pooled control, both fold-safe ----------------
    started = time.time()
    white_per_model = [None] * n
    white_pooled = [None] * n
    fitted = {}
    for fold in np.unique(folds):
        train_all = np.flatnonzero(folds != fold)
        test = np.flatnonzero(folds == fold)
        pooled = yule_walker(accumulate_autocorrelation((raw[i] for i in train_all), AR_ORDER))
        for i in test:
            white_pooled[i] = whiten(raw[i], pooled)
        fitted[f"pooled/fold{fold}"] = {"coefficients": pooled["coefficients"].round(6).tolist(),
                                        "residual_std": pooled["residual_std"],
                                        "autocorrelation": pooled["autocorrelation"].round(6).tolist()}
        for model in ("q4", "q8"):
            train = train_all[models[train_all] == model]
            fit = yule_walker(accumulate_autocorrelation((raw[i] for i in train), AR_ORDER))
            for i in test[models[test] == model]:
                white_per_model[i] = whiten(raw[i], fit)
            fitted[f"{model}/fold{fold}"] = {
                "coefficients": fit["coefficients"].round(6).tolist(),
                "residual_std": fit["residual_std"],
                "autocorrelation": fit["autocorrelation"].round(6).tolist(),
                "train_answers": int(len(train))}
        print(f"[whiten] fold {fold}: {len(test)} answers  ({time.time() - started:.0f}s)", flush=True)
    if any(w is None for w in white_per_model) or any(w is None for w in white_pooled):
        raise ValueError("an answer was left unwhitened")

    inputs = {"raw": raw, "white_per_model": white_per_model, "white_pooled": white_pooled}

    # ---- the sweep --------------------------------------------------------------------
    rows = np.flatnonzero(pb & (target >= 0))
    t_rows, c_rows, s_rows = target[rows], cells[rows], subsets[rows]
    cell_names = sorted(set(c_rows))
    code = np.searchsorted(np.asarray(cell_names), c_rows)
    n_cells = len(cell_names)
    counts = np.bincount(code, minlength=n_cells)

    def mean_sla(hit: np.ndarray) -> float:
        return float((np.bincount(code, weights=hit, minlength=n_cells) / counts).mean())

    # The odd/even token split for criterion 1 does not depend on the arm, so it is built
    # once per (answer, source) and reused across all fourteen widths. Rebuilding it inside
    # the arm loop would repeat the same per-step slicing 42 times over.
    def split_parity(values: np.ndarray, span: np.ndarray, parity: int):
        pieces, marks, cursor = [], [], 0
        for a, b in span:
            segment = values[a:b][parity::2]
            marks.append((cursor, cursor + len(segment)))
            pieces.append(segment)
            cursor += len(segment)
        flat = np.concatenate(pieces) if cursor else np.zeros(1, dtype=float)
        return flat, np.asarray(marks, dtype=int)

    peaks: dict[str, np.ndarray] = {}
    point: dict[str, dict] = {}
    for source, series in inputs.items():
        halves = [[split_parity(series[i], spans[i], p) for p in (0, 1)] for i in rows]
        for kind, grid in (("topk", K_GRID), ("win", W_GRID)):
            for size in grid:
                name = f"{source}__{kind}{size}"
                # count halved so each half computes the same statistic as the whole
                half = max(1, int(round(size / 2)))
                chosen, repro = [], []
                for slot, i in enumerate(rows):
                    chosen.append(int(np.argmax(measure(series[i], spans[i], kind, size))))
                    picks = [int(np.argmax(measure(flat, marks, kind, half)))
                             for flat, marks in halves[slot]]
                    repro.append(int(picks[0] == picks[1]))
                p = np.asarray(chosen, int)
                peaks[name] = p
                hit = (p == t_rows).astype(float)
                per_subset = {}
                for s in SUBSETS:
                    m = s_rows == s
                    per_subset[s] = {"sla": float(hit[m].mean()),
                                     "within1": float((np.abs(p[m] - t_rows[m]) <= 1).mean())}
                point[name] = {"mean_sla": mean_sla(hit),
                               "mean_within1": float(np.mean([v["within1"] for v in per_subset.values()])),
                               "reproducibility": float(np.mean(repro)),
                               "per_subset": per_subset}
        print(f"[sweep] {source}: {len(K_GRID) + len(W_GRID)} arms  "
              f"({time.time() - started:.0f}s)", flush=True)

    anchor = 100 * point["raw__topk10"]["mean_sla"]
    print(f"\nanchor: raw Top-10 = {anchor:.2f}  (must be 35.92)")
    if abs(anchor - 35.92) > 0.005:
        raise SystemExit(f"anchor failed: {anchor:.4f}")

    # ---- intervals for the arms that carry a claim -------------------------------------
    reference = "raw__topk10"
    wanted = {reference}
    for source in inputs:
        family = [k for k in point if k.startswith(f"{source}__")]
        wanted.add(max(family, key=lambda k: point[k]["mean_sla"]))                 # ceiling
        wanted.add(max(family, key=lambda k: point[k]["reproducibility"]))          # label-free
        wanted.add(f"{source}__topk10")
    wanted = sorted(wanted)

    raw_best = max((k for k in point if k.startswith("raw__")),
                   key=lambda k: point[k]["mean_sla"])
    white_best = max((k for k in point if k.startswith("white_per_model__")),
                     key=lambda k: point[k]["mean_sla"])
    pooled_best = max((k for k in point if k.startswith("white_pooled__")),
                      key=lambda k: point[k]["mean_sla"])
    topk_best = max((k for k in point if "__topk" in k), key=lambda k: point[k]["mean_sla"])
    win_best = max((k for k in point if "__win" in k), key=lambda k: point[k]["mean_sla"])
    KEY_PAIRS = [(white_best, raw_best), (white_best, pooled_best), (win_best, topk_best)]
    wanted = sorted(set(wanted) | {raw_best, white_best, pooled_best, topk_best, win_best})

    rng = np.random.default_rng(SEED)
    acc = {k: np.empty(BOOTSTRAP_DRAWS) for k in wanted}
    row_index = np.full(len(target), -1, int)
    row_index[rows] = np.arange(len(rows))
    for d, draw in enumerate(_group_draws(groups, pb & (target >= 0), rng, BOOTSTRAP_DRAWS)):
        take = row_index[draw]
        dt, dcode = t_rows[take], code[take]
        dcounts = np.bincount(dcode, minlength=n_cells)
        present = dcounts > 0
        for k in wanted:
            hit = peaks[k][take] == dt
            acc[k][d] = (np.bincount(dcode, weights=hit, minlength=n_cells)[present]
                         / dcounts[present]).mean()

    report = {"schema": "step-measure-v1", "development_only": True,
              "scope": "step measurement only; fusion, bank and L-SML weights unchanged",
              "ar_order": AR_ORDER, "anchor_raw_topk10": anchor,
              "whiteners": fitted, "point": point,
              "intervals_vs_raw_topk10": {k: interval(acc[k] - acc[reference]) for k in wanted},
              # The decisive P1 contrast is ceiling against ceiling, not either against the
              # incumbent: does the BEST whitened width beat the BEST raw width? And P3 is
              # per-model against pooled at their own best widths.
              "key_contrasts": {
                  f"{a} minus {b}": interval(acc[a] - acc[b])
                  for a, b in KEY_PAIRS if a in acc and b in acc}}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    np.savez_compressed(PEAKS, **peaks)

    # ------------------------------------------------------------------------- console
    for kind, grid, label in (("topk", K_GRID, "Top-K mean (unordered)"),
                              ("win", W_GRID, "best contiguous window")):
        print()
        print("=" * 96)
        print(f"{label}: gate-free SLA by width, per whitening source")
        print("=" * 96)
        print(f"{'width':>7s}" + "".join(f"{s:>20s}" for s in inputs))
        for size in grid:
            cells_out = []
            for source in inputs:
                p = point[f"{source}__{kind}{size}"]
                cells_out.append(f"{100*p['mean_sla']:8.2f} (rep {p['reproducibility']:.3f})")
            print(f"{size:7d}" + "".join(f"{c:>20s}" for c in cells_out))

    print()
    print("=" * 96)
    print("ARMS THAT CARRY A CLAIM -- paired against the incumbent raw Top-10")
    print("=" * 96)
    print(f"{'arm':28s}{'SLA':>8s}{'within1':>9s}{'rep':>7s}{'vs raw Top-10 (pp)':>30s}")
    for k in wanted:
        iv = report["intervals_vs_raw_topk10"][k]
        mark = "*" if iv["excludes_zero"] else " "
        print(f"{k:28s}{100*point[k]['mean_sla']:8.2f}{100*point[k]['mean_within1']:9.2f}"
              f"{point[k]['reproducibility']:7.3f}"
              f"{iv['point_pp']:+11.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]{mark}")

    print()
    print("=" * 96)
    print("KEY CONTRASTS -- ceiling against ceiling, which is what the predictions are about")
    print("=" * 96)
    for name, iv in report["key_contrasts"].items():
        mark = "*" if iv["excludes_zero"] else " "
        print(f"{name:58s}{iv['point_pp']:+9.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]{mark}")

    print()
    print("fitted AR(8) lag-1 coefficient by model and fold (do the two models differ?)")
    for key in sorted(k for k in fitted if not k.startswith("pooled")):
        print(f"  {key:16s} a1 = {fitted[key]['coefficients'][0]:+.4f}  "
              f"rho1 = {fitted[key]['autocorrelation'][1]:+.4f}")
    print()
    print(f"written: {OUT}  ({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
