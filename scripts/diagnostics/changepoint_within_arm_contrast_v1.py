#!/usr/bin/env python
"""Omri's question, asked directly: on a FIXED series, does a change-point rule beat argmax?

The main grid compares everything against the published anchor, which answers "is this a
better localizer". That is the right primary endpoint and the answer there is no. But it
does not answer the question that was actually asked, which is about the DECISION RULE
holding the statistic fixed -- and the anchor comparison confounds the two axes, because
moving from the step readout to the token series costs far more than any rule can return.

So this file holds each series fixed and bootstraps every rule against THAT SERIES' OWN
argmax, on the same source-group draws. Nothing is refitted; the cached peaks from
CHANGEPOINT_PEAKS.npz are reused, so these contrasts are exactly the same predictions the
grid scored.

Reported per arm: the pre-registered defaults, and the arm's best rule. The best-rule row
is label-selected within that arm and is marked as a ceiling, not a candidate.
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
OUT = RES / "CHANGEPOINT_WITHIN_ARM.json"

BOOTSTRAP_DRAWS = 10_000
SEED = 20260919
ARMS = ("level__step", "level__token", "drop_m5__step", "drop_worst__step", "drop__token")
DEFAULTS = ("argmax", "first_q0.9", "cusum_alarm_k0.5_h5", "cusum_onset_k0.5_h5",
            "bocpd_rise_hz32", "bocpd_reset_hz32")

_spec = importlib.util.spec_from_file_location(
    "stage_b_2x2_v1", Path(__file__).with_name("stage_b_2x2_v1.py"))
_sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sb)
_group_draws, interval = _sb._group_draws, _sb.interval


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)
    pb = np.char.startswith(cells, "pb_")
    rows = np.flatnonzero(pb & (target >= 0))

    peaks = {}
    with np.load(RES / "CHANGEPOINT_PEAKS.npz", allow_pickle=False) as z:
        for key in z.files:
            peaks[key] = np.asarray(z[key], int)
    grid = json.loads((RES / "CHANGEPOINT_READOUT.json").read_text(encoding="utf8"))
    point = grid["point"]

    anchor = 100 * point["level__step@@argmax"]["mean_sla"]
    print(f"anchor: level x argmax = {anchor:.2f}  (must be 35.92)")
    if abs(anchor - 35.92) > 0.005:
        raise SystemExit(f"anchor failed: {anchor:.4f}")

    wanted, best_of = [], {}
    for arm in ARMS:
        family = [n for n in point if n.startswith(f"{arm}@@")]
        best = max(family, key=lambda n: point[n]["mean_sla"])
        best_of[arm] = best.split("@@")[1]
        for rule in DEFAULTS:
            if f"{arm}@@{rule}" in peaks:
                wanted.append(f"{arm}@@{rule}")
        if best not in wanted:
            wanted.append(best)
    wanted = sorted(set(wanted))

    t_rows, c_rows = target[rows], cells[rows]
    cell_names = sorted(set(c_rows))
    code = np.searchsorted(np.asarray(cell_names), c_rows)
    n_cells = len(cell_names)
    row_index = np.full(len(target), -1, int)
    row_index[rows] = np.arange(len(rows))

    rng = np.random.default_rng(SEED)
    acc = {name: np.empty(BOOTSTRAP_DRAWS) for name in wanted}
    for d, draw in enumerate(_group_draws(groups, pb & (target >= 0), rng, BOOTSTRAP_DRAWS)):
        take = row_index[draw]
        dt, dcode = t_rows[take], code[take]
        counts = np.bincount(dcode, minlength=n_cells)
        present = counts > 0
        for name in wanted:
            hit = peaks[name][take] == dt
            acc[name][d] = (np.bincount(dcode, weights=hit, minlength=n_cells)[present]
                            / counts[present]).mean()

    report = {"schema": "changepoint-within-arm-contrast-v1", "development_only": True,
              "note": "each rule against its OWN arm's argmax; the best-rule row is "
                      "label-selected within the arm and is a ceiling, not a candidate",
              "anchor_mean_sla": anchor, "arms": {}}
    print()
    print("=" * 104)
    print("EACH RULE AGAINST ITS OWN SERIES' ARGMAX -- gate-free SLA, 8 ProcessBench cells")
    print("=" * 104)
    print(f"{'arm':18s} {'rule':26s} {'SLA':>7s} {'vs this arm''s argmax (pp)':>32s}")
    for arm in ARMS:
        base = f"{arm}@@argmax"
        report["arms"][arm] = {"argmax_sla": 100 * point[base]["mean_sla"],
                               "best_rule_label_selected": best_of[arm], "rules": {}}
        rules = [r for r in DEFAULTS if f"{arm}@@{r}" in acc]
        if best_of[arm] not in rules:
            rules.append(best_of[arm])
        for rule in rules:
            name = f"{arm}@@{rule}"
            iv = interval(acc[name] - acc[base])
            report["arms"][arm]["rules"][rule] = {"sla": 100 * point[name]["mean_sla"], **iv}
            mark = "*" if iv["excludes_zero"] else " "
            tag = "  <- best in arm (ceiling)" if rule == best_of[arm] and rule != "argmax" else ""
            print(f"{arm:18s} {rule:26s} {100*point[name]['mean_sla']:7.2f} "
                  f"{iv['point_pp']:+8.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]{mark}{tag}")
        print()

    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
