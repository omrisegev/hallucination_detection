#!/usr/bin/env python
"""Isolate the GATE from the LOCATOR in the token-level L-SML experiment.

Omri, 2026-09-18.  `results/claude_feature_bank_token_lsml_v1` reports PB macro-F1
0.3408 (L-SML) vs 0.3234 (equal) and cannot be compared to CT7's 41.19, because the
two arms use DIFFERENT GATES (LOCO-5 here, frozen non-digit tail15 for CT7) and
ProcessBench macro-F1 is strongly gate-dependent.  Three separations, all computed
from the frozen OOF step scores, no GPU and no refit:

A. MIND-THE-GAP PROTOCOL (gate-free).  Chen et al. (ICML 2026) report Step-level
   Localization Accuracy per ProcessBench subset, on ERRONEOUS answers only, with no
   no-error decision at all.  Our population is exactly theirs (400/1000/1000/1000 per
   model), so this is the closest published head-to-head the project has.  This is the
   measurement that removes the gate entirely.

B. LOCO-5 THRESHOLD SWEEP.  The 0.33 opening quantile is inherited from an older arm
   and was never retuned for this locator.  Sweeping it bounds how much of the macro-F1
   deficit is a mis-set threshold rather than a worse locator.  The argmax over the
   sweep is label-selected and is reported as a CEILING, not as a candidate.

C. Per-cell decomposition of clean accuracy vs error accuracy, which says whether the
   gate is losing points by closing on erroneous answers or by opening on clean ones.

Nothing here is a new candidate and nothing modifies CT7.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"

# Chen et al., "Mind the Gap", ICML 2026, Table 3 (Step-level Localization Accuracy %).
# Their ProcessBench subset sizes match ours exactly (400/1000/1000/1000).
MIND_THE_GAP = {
    "q4": {"gsm8k": {"shannon_drop": 43.42, "shannon_avg": 27.94, "lns_avg": 31.24},
           "math": {"shannon_drop": 32.03, "shannon_avg": 24.17, "lns_avg": 28.39},
           "olympiadbench": {"shannon_drop": 43.06, "shannon_avg": 24.95, "lns_avg": 26.22},
           "omnimath": {"shannon_drop": 38.04, "shannon_avg": 23.67, "lns_avg": 23.21}},
    "q8": {"gsm8k": {"shannon_drop": 46.11, "shannon_avg": 27.66, "lns_avg": 32.34},
           "math": {"shannon_drop": 32.90, "shannon_avg": 24.62, "lns_avg": 26.91},
           "olympiadbench": {"shannon_drop": 41.52, "shannon_avg": 26.30, "lns_avg": 26.48},
           "omnimath": {"shannon_drop": 37.04, "shannon_avg": 23.40, "lns_avg": 23.35}},
}


def load():
    npz = np.load(ROSTER / "JOINED.npz", allow_pickle=True)
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    cells = np.asarray([r["cell"] for r in records], dtype=str)
    oof = np.load(os.environ["OOF_PATH"], allow_pickle=True)
    return {
        "offsets": np.asarray(npz["offsets"], dtype=int),
        "target": np.asarray(npz["target"], dtype=int),
        "cells": cells,
        "l_sml": np.asarray(oof["l_sml"], dtype=float),
        "equal": np.asarray(oof["equal"], dtype=float),
        "gate_score": np.asarray(oof["gate_score"], dtype=float),
        "gate_open": np.asarray(oof["gate_open"], dtype=bool),
    }


def peaks_of(step_scores, offsets):
    """argmax step per answer; the pure locator, with no gate."""
    n = len(offsets) - 1
    out = np.empty(n, dtype=int)
    for a in range(n):
        out[a] = int(np.argmax(step_scores[offsets[a]:offsets[a + 1]]))
    return out


def sla_gate_free(peaks, target, cells):
    """Mind-the-Gap SLA: exact first-error hit rate on ERRONEOUS answers, no gate."""
    out = {}
    for cell in sorted(set(cells)):
        if not cell.startswith("pb_"):
            continue
        mask = (cells == cell) & (target >= 0)
        if not mask.any():
            continue
        out[cell] = {"n_error": int(mask.sum()),
                     "sla": float((peaks[mask] == target[mask]).mean()),
                     "within1": float((np.abs(peaks[mask] - target[mask]) <= 1).mean())}
    return out


def chance_sla(target, cells, offsets):
    """Uniform-random step choice, the honest floor for SLA."""
    lengths = np.diff(offsets)
    out = {}
    for cell in sorted(set(cells)):
        if not cell.startswith("pb_"):
            continue
        mask = (cells == cell) & (target >= 0)
        if mask.any():
            out[cell] = float(np.mean(1.0 / lengths[mask]))
    return out


def gated_f1(peaks, target, cells, opened):
    """The registered ProcessBench adapter: per-cell F1, then macro."""
    prediction = np.where(opened, peaks, -1)
    cell_f1, rows = {}, {}
    for cell in sorted(set(cells)):
        if not cell.startswith("pb_"):
            continue
        mask = cells == cell
        clean, err = mask & (target == -1), mask & (target >= 0)
        ca = float((prediction[clean] == target[clean]).mean()) if clean.any() else None
        ea = float((prediction[err] == target[err]).mean()) if err.any() else None
        f1 = 0.0 if not (ca or ea) else 2 * ca * ea / (ca + ea)
        cell_f1[cell] = f1
        rows[cell] = {"clean_accuracy": ca, "error_exact_accuracy": ea, "f1": f1}
    return float(np.mean(list(cell_f1.values()))), rows


def main():
    d = load()
    off, tgt, cells = d["offsets"], d["target"], d["cells"]
    report = {"population": {"answers": int(len(tgt)), "steps": int(off[-1]),
                             "pb_answers": int(np.char.startswith(cells, "pb_").sum())}}

    arms = {"token_l_sml": d["l_sml"], "token_equal_mean": d["equal"]}
    peaks = {name: peaks_of(scores, off) for name, scores in arms.items()}

    # --- A. gate-free Mind-the-Gap protocol ------------------------------------
    report["chance_sla"] = chance_sla(tgt, cells, off)
    report["sla_gate_free"] = {name: sla_gate_free(p, tgt, cells) for name, p in peaks.items()}
    report["sla_gated"] = {}
    for name, p in peaks.items():
        gated = np.where(d["gate_open"], p, -1)
        report["sla_gated"][name] = {
            cell: float((gated[(cells == cell) & (tgt >= 0)] == tgt[(cells == cell) & (tgt >= 0)]).mean())
            for cell in sorted(set(cells)) if cell.startswith("pb_")}
    report["mind_the_gap_published"] = MIND_THE_GAP

    # --- B. LOCO-5 threshold sweep --------------------------------------------
    pb = np.char.startswith(cells, "pb_")
    sweep = {}
    grid = np.round(np.arange(0.00, 1.001, 0.01), 3)
    for name, p in peaks.items():
        curve = []
        for thr in grid:
            opened = pb & (d["gate_score"] >= thr)
            macro, _ = gated_f1(p, tgt, cells, opened)
            curve.append({"threshold": float(thr), "macro_f1": macro,
                          "opened": int(opened[pb].sum())})
        best = max(curve, key=lambda r: r["macro_f1"])
        at_033 = min(curve, key=lambda r: abs(r["threshold"] - 0.33))
        sweep[name] = {"curve": curve, "best_label_selected_ceiling": best, "registered_0.33": at_033}
    report["loco5_threshold_sweep"] = sweep

    # --- C. per-cell clean vs error decomposition at the registered threshold ---
    report["per_cell_at_registered_threshold"] = {}
    for name, p in peaks.items():
        macro, rows = gated_f1(p, tgt, cells, pb & (d["gate_score"] >= 0.33))
        report["per_cell_at_registered_threshold"][name] = {"macro_f1": macro, "cells": rows}

    out = ROOT / "results" / "claude_feature_bank_token_lsml_v1" / "GATE_ISOLATION.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1), encoding="utf8")

    # --- console summary -------------------------------------------------------
    print("=" * 78)
    print("A. MIND-THE-GAP PROTOCOL: per-subset SLA on ERRONEOUS answers, NO GATE")
    print("=" * 78)
    print(f"{'cell':26s} {'n_err':>6s} {'L-SML':>8s} {'equal':>8s} {'chance':>8s} "
          f"{'MtG-drop':>9s} {'MtG-avg':>8s}")
    for cell in sorted(report["sla_gate_free"]["token_l_sml"]):
        model = "q4" if cell.endswith("q4") else "q8"
        subset = cell[3:-3]
        mtg = MIND_THE_GAP[model].get(subset, {})
        a = report["sla_gate_free"]["token_l_sml"][cell]
        b = report["sla_gate_free"]["token_equal_mean"][cell]
        print(f"{cell:26s} {a['n_error']:6d} {100*a['sla']:8.2f} {100*b['sla']:8.2f} "
              f"{100*report['chance_sla'][cell]:8.2f} {mtg.get('shannon_drop', float('nan')):9.2f} "
              f"{mtg.get('shannon_avg', float('nan')):8.2f}")
    mean_l = np.mean([v["sla"] for v in report["sla_gate_free"]["token_l_sml"].values()])
    mean_e = np.mean([v["sla"] for v in report["sla_gate_free"]["token_equal_mean"].values()])
    print(f"{'MEAN over 8 cells':26s} {'':6s} {100*mean_l:8.2f} {100*mean_e:8.2f}")

    print()
    print("=" * 78)
    print("B. LOCO-5 THRESHOLD SWEEP (label-selected ceiling, NOT a candidate)")
    print("=" * 78)
    for name in arms:
        s = sweep[name]
        r0, rb = s["registered_0.33"], s["best_label_selected_ceiling"]
        print(f"{name:18s} registered thr=0.33 -> macro {100*r0['macro_f1']:.2f} "
              f"(open {r0['opened']})   best thr={rb['threshold']:.2f} -> "
              f"macro {100*rb['macro_f1']:.2f} (open {rb['opened']})   "
              f"headroom {100*(rb['macro_f1']-r0['macro_f1']):+.2f} pp")

    print()
    print("=" * 78)
    print("C. WHERE THE GATE LOSES: clean vs error accuracy at thr=0.33")
    print("=" * 78)
    print(f"{'cell':26s} {'clean_acc':>10s} {'err_acc':>9s} {'gate-free SLA':>14s} {'gate cost':>10s}")
    rows = report["per_cell_at_registered_threshold"]["token_l_sml"]["cells"]
    for cell in sorted(rows):
        free = report["sla_gate_free"]["token_l_sml"][cell]["sla"]
        print(f"{cell:26s} {100*rows[cell]['clean_accuracy']:10.2f} "
              f"{100*rows[cell]['error_exact_accuracy']:9.2f} {100*free:14.2f} "
              f"{100*(rows[cell]['error_exact_accuracy']-free):10.2f}")
    print()
    print(f"written: {out}")


if __name__ == "__main__":
    main()
