#!/usr/bin/env python
"""statistic x granularity x decision rule, on one roster, against one anchor.

Pre-registration: `docs/experiments/CHANGEPOINT_STEP_READOUT_V1_PREREGISTRATION.md`.
Every prediction and falsification condition there was written before this file existed.

Two questions, one grid.

Q1 finishes `docs/HANDOFF_MIND_THE_GAP_MECHANISM.md`: the 2x2 of {level, drop} x {argmax,
first-crossing} is missing the cell that is actually theirs -- a derivative statistic read
out by a first-crossing rule. Both single-axis cells were tested and both lost.

Q2 is Omri's: instead of the argmax of the step-score series, ask a sequential change-point
detector -- CUSUM or BOCPD -- where the first error is. Q1 is the degenerate member of Q2's
family, so they are one grid.

Primary endpoint is gate-free SLA, the Mind-the-Gap protocol: exact first-error hit rate on
ERRONEOUS ProcessBench answers only, per cell, then the mean over the eight cells. No gate
is involved anywhere in this file.

An adaptation, not a reproduction: the paper leaves SLA's token-to-step collapse undefined,
so `step_m5` and `step_worst` are OUR choices and are labelled as such. Nothing here carries
an author's name or is a candidate for the method of record.
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
OUT = RES / "CHANGEPOINT_READOUT.json"
SCORES = RES / "CHANGEPOINT_PEAKS.npz"

BOOTSTRAP_DRAWS = 10_000
SEED = 20260919
TOKEN_RMAX = 128           # BOCPD run-length cap at token granularity; exact is O(T^2)
SHORT, LONG = ("gsm8k", "math"), ("olympiadbench", "omnimath")

# Pre-registered defaults. Not tuned here; the grid below is a ceiling, not a candidate.
DEFAULT_Q = 0.90
DEFAULT_K = 0.5
DEFAULT_H = 5.0
DEFAULT_HAZARD = 1.0 / 32.0

Q_GRID = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95)
K_GRID = (0.25, 0.5, 1.0)
H_GRID = (1.0, 2.0, 3.0, 5.0, 8.0)
HAZARD_GRID = (1 / 8, 1 / 32, 1 / 128)

from spectral_utils.changepoint_step_readout_v1 import (  # noqa: E402
    argmax_readout, bocpd_filter, first_crossing_readout, page_cusum, token_index_to_step,
)

_spec = importlib.util.spec_from_file_location(
    "stage_b_2x2_v1", Path(__file__).with_name("stage_b_2x2_v1.py"))
_sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sb)
_group_draws, interval = _sb._group_draws, _sb.interval

_gspec = importlib.util.spec_from_file_location(
    "gate_isolation_v1", Path(__file__).with_name("gate_isolation_token_lsml_v1.py"))
_canon = importlib.util.module_from_spec(_gspec)
_gspec.loader.exec_module(_canon)
sla_gate_free = _canon.sla_gate_free


def subset_of(cell: str) -> str:
    return cell.split("_", 1)[1].rsplit("_", 1)[0] if cell.startswith("pb_") else "prm"


# --------------------------------------------------------------------------- the rules
def rules_for(series: np.ndarray) -> dict[str, int]:
    """Every decision rule applied to one answer's series. Returns index INTO the series.

    Computed together rather than one call per rule because CUSUM and BOCPD each cost one
    pass and are shared across every setting of their own free parameter.
    """
    out: dict[str, int] = {"argmax": argmax_readout(series)}
    for q in Q_GRID:
        out[f"first_q{q:g}"] = first_crossing_readout(series, q=q)
    for k in K_GRID:
        run = page_cusum(series, k=k)
        statistic, onset = run["statistic"], run["onset"]
        best = int(np.argmax(statistic))
        for h in H_GRID:
            hit = np.flatnonzero(statistic >= h)
            if len(hit):
                alarm = int(hit[0])
                out[f"cusum_alarm_k{k:g}_h{h:g}"] = alarm
                out[f"cusum_onset_k{k:g}_h{h:g}"] = int(onset[alarm])
                out[f"cusum_alarm_series_k{k:g}_h{h:g}"] = alarm
            else:
                out[f"cusum_alarm_k{k:g}_h{h:g}"] = best
                out[f"cusum_onset_k{k:g}_h{h:g}"] = int(onset[best])
                out[f"cusum_alarm_series_k{k:g}_h{h:g}"] = out["argmax"]
    for hazard in HAZARD_GRID:
        cap = None if len(series) <= 256 else TOKEN_RMAX
        filt = bocpd_filter(series, hazard=hazard, r_max=cap)
        out[f"bocpd_rise_hz{1 / hazard:g}"] = int(np.argmax(filt["rise"]))
        out[f"bocpd_reset_hz{1 / hazard:g}"] = int(np.argmax(filt["reset_probability"]))
    return out


DEFAULT_RULES = {
    "argmax": "argmax",
    "first_crossing": f"first_q{DEFAULT_Q:g}",
    "cusum_alarm": f"cusum_alarm_k{DEFAULT_K:g}_h{DEFAULT_H:g}",
    "cusum_onset": f"cusum_onset_k{DEFAULT_K:g}_h{DEFAULT_H:g}",
    "cusum_alarm_series_fallback": f"cusum_alarm_series_k{DEFAULT_K:g}_h{DEFAULT_H:g}",
    "bocpd_rise": f"bocpd_rise_hz{1 / DEFAULT_HAZARD:g}",
    "bocpd_reset": f"bocpd_reset_hz{1 / DEFAULT_HAZARD:g}",
}


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)
    subsets = np.asarray([subset_of(c) for c in cells], str)
    pb = np.char.startswith(cells, "pb_")

    with np.load(RES / "STAGE_B_SCORES.npz", allow_pickle=False) as z:
        c1_step = np.asarray(z["C1_pooled_before__l_sml"], float)
    with np.load(RES / "C1_TOKEN_SERIES.npz", allow_pickle=False) as z:
        c1_token = np.asarray(z["token_l_sml"], float)
        tok_off = np.asarray(z["token_offsets"], int)
    with np.load(RES / "EVIDENCE_DROP.npz", allow_pickle=False) as z:
        drop_token = np.asarray(z["risk_token"], float)
        drop_m5 = np.asarray(z["step_m5"], float)
        drop_worst = np.asarray(z["step_worst"], float)
        h1_corr = float(z["h1_correlation"])
    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        step_spans = np.asarray(z["step_spans"], int)

    # ---- anchor before anything else ---------------------------------------------
    rows = np.flatnonzero(pb & (target >= 0))
    anchor_peaks = np.asarray([argmax_readout(c1_step[offsets[i]:offsets[i + 1]]) for i in rows])
    per = sla_gate_free(anchor_peaks, target[rows], cells[rows])
    anchor = float(100 * np.mean([v["sla"] for v in per.values()]))
    print(f"anchor: C1 level x argmax, mean gate-free SLA = {anchor:.2f}  (must be 35.92)")
    if abs(anchor - 35.92) > 0.005:
        raise SystemExit(f"anchor failed: {anchor:.4f}")

    # ---- the grid -----------------------------------------------------------------
    # Arms: (name, granularity, per-answer series accessor). Only erroneous ProcessBench
    # answers are scored, so only those answers are visited.
    def step_series(array):
        return lambda i: array[offsets[i]:offsets[i + 1]]

    def token_series(array):
        return lambda i: array[tok_off[i]:tok_off[i + 1]]

    arms = {
        "level__step": ("step", step_series(c1_step)),
        "level__token": ("token", token_series(c1_token)),
        "drop_m5__step": ("step", step_series(drop_m5)),
        "drop_worst__step": ("step", step_series(drop_worst)),
        "drop__token": ("token", token_series(drop_token)),
    }

    started = time.time()
    peaks: dict[str, np.ndarray] = {}
    for arm, (granularity, get) in arms.items():
        collected: dict[str, list[int]] = {}
        for i in rows:
            series = get(i)
            spans = step_spans[offsets[i]:offsets[i + 1]]
            chosen = rules_for(series)
            for rule, index in chosen.items():
                step = token_index_to_step(index, spans) if granularity == "token" else index
                collected.setdefault(rule, []).append(step)
        for rule, values in collected.items():
            peaks[f"{arm}@@{rule}"] = np.asarray(values, int)
        print(f"[arm] {arm}: {len(collected)} rules  ({time.time() - started:.0f}s)", flush=True)

    # ---- point metrics ------------------------------------------------------------
    t_rows, c_rows = target[rows], cells[rows]
    s_rows = subsets[rows]
    lengths = np.diff(offsets)[rows]
    chance = {}
    for subset in SHORT + LONG:
        mask = s_rows == subset
        chance[subset] = float(np.mean(1.0 / lengths[mask]))

    def summarise(prediction: np.ndarray) -> dict:
        per_cell = sla_gate_free(prediction, t_rows, c_rows)
        per_subset = {}
        for subset in SHORT + LONG:
            mask = s_rows == subset
            per_subset[subset] = {
                "sla": float((prediction[mask] == t_rows[mask]).mean()),
                "within1": float((np.abs(prediction[mask] - t_rows[mask]) <= 1).mean()),
                "ratio_to_chance": float((prediction[mask] == t_rows[mask]).mean() / chance[subset]),
            }
        mean_sla = float(np.mean([v["sla"] for v in per_cell.values()]))
        long_short = (np.mean([per_subset[s]["sla"] for s in LONG])
                      - np.mean([per_subset[s]["sla"] for s in SHORT]))
        return {"mean_sla": mean_sla, "per_cell": per_cell, "per_subset": per_subset,
                "mean_within1": float(np.mean([v["within1"] for v in per_cell.values()])),
                "long_minus_short": float(long_short)}

    point = {name: summarise(p) for name, p in peaks.items()}

    # ---- paired intervals against the anchor, and the long-short interaction -------
    reference = "level__step@@argmax"
    if not np.array_equal(peaks[reference], anchor_peaks):
        raise SystemExit("the grid's own level x argmax arm does not match the anchor")

    # Bootstrapping every one of the ~130 rules is wasteful; the pre-registered defaults
    # and the per-arm grid winners are what carry a claim, so those are what get intervals.
    wanted = {reference}
    for arm in arms:
        for rule in DEFAULT_RULES.values():
            if f"{arm}@@{rule}" in peaks:
                wanted.add(f"{arm}@@{rule}")
        family = [n for n in peaks if n.startswith(f"{arm}@@") and n != f"{arm}@@argmax"]
        if family:
            wanted.add(max(family, key=lambda n: point[n]["mean_sla"]))
    wanted = sorted(wanted)

    # The cell and subset partitions do not depend on the arm, so they are built ONCE per
    # draw and reused. Calling `sla_gate_free` inside the arm loop instead would rebuild
    # eight string masks over 4,442 rows for each of ~15 arms on each of 10,000 draws --
    # about 5e9 string comparisons, which is hours rather than minutes.
    cell_names = sorted(set(c_rows))
    cell_code = np.searchsorted(np.asarray(cell_names), c_rows)
    n_cells = len(cell_names)
    # The interaction is defined over SUBSETS, matching the point summary: the mean of the
    # two long subsets minus the mean of the two short ones. Pooling rows instead would
    # weight OlympiadBench and Omni-MATH by their differing row counts.
    subset_names = list(SHORT + LONG)
    subset_code = np.asarray([subset_names.index(s) for s in s_rows])
    long_slots = [subset_names.index(s) for s in LONG]
    short_slots = [subset_names.index(s) for s in SHORT]

    rng = np.random.default_rng(SEED)
    acc = {name: np.empty(BOOTSTRAP_DRAWS) for name in wanted}
    inter = {name: np.empty(BOOTSTRAP_DRAWS) for name in wanted}
    row_index = np.full(len(target), -1, int)
    row_index[rows] = np.arange(len(rows))
    started_boot = time.time()
    for draw_number, draw in enumerate(_group_draws(groups, pb & (target >= 0), rng,
                                                    BOOTSTRAP_DRAWS)):
        take = row_index[draw]
        dt, code = t_rows[take], cell_code[take]
        counts = np.bincount(code, minlength=n_cells)
        present = counts > 0
        scode = subset_code[take]
        scounts = np.maximum(np.bincount(scode, minlength=len(subset_names)), 1)
        for name in wanted:
            hit = peaks[name][take] == dt
            acc[name][draw_number] = (
                np.bincount(code, weights=hit, minlength=n_cells)[present]
                / counts[present]).mean()
            rate = np.bincount(scode, weights=hit, minlength=len(subset_names)) / scounts
            inter[name][draw_number] = rate[long_slots].mean() - rate[short_slots].mean()
        if draw_number == 199:
            rate = (time.time() - started_boot) / 200
            print(f"[bootstrap] {rate * BOOTSTRAP_DRAWS / 60:.1f} min projected", flush=True)
    draws, inters = acc, inter

    report = {
        "schema": "changepoint-step-readout-v1",
        "development_only": True,
        "adaptation_note": "token-to-step collapse is undefined in the source paper; "
                           "step_m5 and step_worst are our pre-registered choices",
        "anchor_c1_mean_sla": anchor,
        "evidence_series_corr_with_bank_q15_H1": h1_corr,
        "chance_sla": chance,
        "defaults": DEFAULT_RULES,
        "point": point,
        "intervals_vs_anchor": {
            name: interval(draws[name] - draws[reference]) for name in wanted},
        "long_minus_short_interaction_vs_anchor": {
            name: interval(inters[name] - inters[reference]) for name in wanted},
    }
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    np.savez_compressed(SCORES, **peaks)

    # ------------------------------------------------------------------------ console
    print()
    print("=" * 118)
    print("PRE-REGISTERED DEFAULTS -- gate-free SLA, mean over the eight ProcessBench cells")
    print("=" * 118)
    print(f"{'arm':20s} {'rule':28s} {'SLA':>7s} {'within1':>8s} {'vs anchor (pp)':>26s} "
          f"{'gsm8k':>7s} {'math':>7s} {'olymp':>7s} {'omni':>7s}")
    for arm in arms:
        for label, rule in DEFAULT_RULES.items():
            name = f"{arm}@@{rule}"
            if name not in point:
                continue
            p = point[name]
            iv = report["intervals_vs_anchor"].get(name)
            cell = (f"{iv['point_pp']:+7.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]"
                    f"{'*' if iv['excludes_zero'] else ' '}") if iv else ""
            sub = p["per_subset"]
            print(f"{arm:20s} {label:28s} {100*p['mean_sla']:7.2f} {100*p['mean_within1']:8.2f} "
                  f"{cell:>26s} " + " ".join(f"{100*sub[s]['sla']:7.2f}"
                                             for s in ("gsm8k", "math", "olympiadbench", "omnimath")))
        print()

    print("=" * 118)
    print("LABEL-SELECTED CEILING per arm -- the best of ~130 settings. NOT a candidate.")
    print("=" * 118)
    for arm in arms:
        family = [n for n in point if n.startswith(f"{arm}@@")]
        best = max(family, key=lambda n: point[n]["mean_sla"])
        iv = report["intervals_vs_anchor"].get(best)
        tail = (f"  vs anchor {iv['point_pp']:+.2f} [{iv['ci95_pp'][0]:+.2f},{iv['ci95_pp'][1]:+.2f}]"
                f"{'*' if iv['excludes_zero'] else ''}") if iv else ""
        print(f"{arm:20s} {best.split('@@')[1]:34s} {100*point[best]['mean_sla']:7.2f}{tail}")

    print()
    print("=" * 118)
    print("RATIO TO CHANCE by subset -- does the advantage grow with chain length, as theirs does?")
    print("=" * 118)
    print(f"{'arm / rule':50s} {'gsm8k':>8s} {'math':>8s} {'olymp':>8s} {'omni':>8s} "
          f"{'long-short vs anchor (pp)':>28s}")
    for name in wanted:
        p, sub = point[name], point[name]["per_subset"]
        iv = report["long_minus_short_interaction_vs_anchor"][name]
        print(f"{name:50s} " + " ".join(f"{sub[s]['ratio_to_chance']:8.2f}"
                                        for s in ("gsm8k", "math", "olympiadbench", "omnimath"))
              + f" {iv['point_pp']:+8.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]"
                f"{'*' if iv['excludes_zero'] else ' '}")

    print()
    print(f"written: {OUT}  and  {SCORES}   ({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
