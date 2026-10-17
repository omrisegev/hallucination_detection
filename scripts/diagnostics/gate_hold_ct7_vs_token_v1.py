#!/usr/bin/env python
"""Stage A of the token-probability line: hold the gate fixed, both directions.

docs/HANDOFF_TOKEN_PROBABILITIES.md sec. 5.4.  The handoff's own table carries a
warning next to it -- "Do not compare the 41.19 and the 34.08" -- because CT7 is
scored under its frozen non-digit tail15 gate and the token-level arm under
LOCO-5 at 0.33, and ProcessBench macro-F1 is strongly gate-dependent.  The
gate-free column is the comparable one and CT7's entry in it was missing.  This
script fills that entry and, separately, crosses the two locators with the two
gates so that every comparison printed here holds one of the two fixed.

Nothing is refitted.  Both locators are read from frozen per-step score vectors:

  CT7            results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz
                 restored from the Drive backup of 2026-09-17, not re-derived.
                 Its FROZEN_CANDIDATE_CT7.json is byte-identical to master's.
  token L-SML    results/claude_feature_bank_token_lsml_v1/OOF_SCORES.npz
  token equal    the same file

The scoring functions are imported from the Step-422 diagnostic rather than
re-implemented, so the SLA, chance floor and ProcessBench macro-F1 adapter are
by construction the same code that produced the numbers in the handoff.

Development-only.  LOCO-5's membership was historically label-selected, and the
whole cached population has been inspected before.  No result here is a
publication confirmation.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ROSTER = ROOT / "results" / "localization_full_benchmark_v3" / "evaluation"
CT7 = ROOT / "results" / "chosen_token_calibration_v1" / "CT7_DEV_SCORES.npz"
TOKEN = ROOT / "results" / "claude_feature_bank_token_lsml_v1" / "OOF_SCORES.npz"
OUT = ROOT / "results" / "token_probability_fusion_v1" / "GATE_HOLD_STAGE_A.json"

BOOTSTRAP_DRAWS = 10_000
SEED = 20260918

# Import the canonical scorer instead of restating it.  Mirroring by hand is how
# a "same protocol" claim quietly stops being true.
_spec = importlib.util.spec_from_file_location(
    "gate_isolation_v1", Path(__file__).with_name("gate_isolation_token_lsml_v1.py")
)
_canon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_canon)
peaks_of, sla_gate_free, chance_sla, gated_f1 = (
    _canon.peaks_of,
    _canon.sla_gate_free,
    _canon.chance_sla,
    _canon.gated_f1,
)


def load() -> dict:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as npz:
        offsets = np.asarray(npz["offsets"], dtype=int)
        target = np.asarray(npz["target"], dtype=int)
    with np.load(CT7, allow_pickle=False) as npz:
        ct7_scores = np.asarray(npz["step_scores"], dtype=float)
        ct7_gate = np.asarray(npz["gate"], dtype=bool)
    with np.load(TOKEN, allow_pickle=False) as npz:
        l_sml = np.asarray(npz["l_sml"], dtype=float)
        equal = np.asarray(npz["equal"], dtype=float)
        loco5_gate = np.asarray(npz["gate_open"], dtype=bool)

    n_answers, n_steps = len(target), int(offsets[-1])
    for name, arr, want in (
        ("CT7 step_scores", ct7_scores, n_steps),
        ("token l_sml", l_sml, n_steps),
        ("token equal", equal, n_steps),
        ("CT7 gate", ct7_gate, n_answers),
        ("LOCO-5 gate", loco5_gate, n_answers),
    ):
        if len(arr) != want:
            raise ValueError(f"{name} has {len(arr)} entries, roster wants {want}")

    return {
        "offsets": offsets,
        "target": target,
        "cells": np.asarray([r["cell"] for r in records], dtype=str),
        "groups": np.asarray([r["group_id"] for r in records], dtype=str),
        "locators": {"ct7": ct7_scores, "token_l_sml": l_sml, "token_equal_mean": equal},
        "gates": {"ct7_frozen_tail15": ct7_gate, "loco5_at_0.33": loco5_gate},
    }


def mean_sla(peaks: np.ndarray, target: np.ndarray, cells: np.ndarray) -> float:
    """Mind-the-Gap headline: unweighted mean of the eight per-subset SLAs."""
    per_cell = sla_gate_free(peaks, target, cells)
    return float(np.mean([v["sla"] for v in per_cell.values()]))


def _group_bootstrap_draws(groups: np.ndarray, mask: np.ndarray, rng):
    """Resample SOURCE GROUPS with replacement; yield per-draw answer indexes.

    Answers sharing a source question are not independent, so the resampling unit
    is the group, matching the paired interval reported in the handoff.

    Members are packed into one contiguous array ordered by group, so a draw is a
    vectorised gather rather than a concatenation of ~2k slices.  Yielded lazily:
    materialising all 10,000 index arrays at once costs a few hundred MB for no gain.
    """
    order = np.argsort(groups[mask], kind="stable")
    flat = np.flatnonzero(mask)[order]
    _, starts, counts = np.unique(groups[flat], return_index=True, return_counts=True)
    n_groups = len(starts)

    for _ in range(BOOTSTRAP_DRAWS):
        draw = rng.integers(0, n_groups, size=n_groups)
        take, size = counts[draw], starts[draw]
        ends = np.cumsum(take)
        within = np.arange(int(ends[-1])) - np.repeat(ends - take, take)
        yield flat[np.repeat(size, take) + within]


def paired_interval(values_a: np.ndarray, values_b: np.ndarray) -> dict:
    diff = values_a - values_b
    finite = diff[np.isfinite(diff)]
    return {
        "mean_difference_pp": float(100 * finite.mean()),
        "ci95_pp": [float(100 * np.percentile(finite, 2.5)),
                    float(100 * np.percentile(finite, 97.5))],
        "draws_used": int(len(finite)),
        "excludes_zero": bool(np.percentile(finite, 2.5) > 0 or np.percentile(finite, 97.5) < 0),
    }


def main() -> None:
    d = load()
    off, tgt, cells, groups = d["offsets"], d["target"], d["cells"], d["groups"]
    pb = np.char.startswith(cells, "pb_")
    peaks = {name: peaks_of(scores, off) for name, scores in d["locators"].items()}

    report: dict = {
        "schema": "token-probability-fusion-v1-stage-a",
        "development_only": True,
        "refit": False,
        "ct7_source": "Drive backup 2026-09-17, restored; not re-derived",
        "population": {
            "answers": int(len(tgt)),
            "steps": int(off[-1]),
            "pb_answers": int(pb.sum()),
            "pb_erroneous": int((pb & (tgt >= 0)).sum()),
            "pb_error_source_groups": int(len(set(groups[pb & (tgt >= 0)]))),
            "pb_all_source_groups": int(len(set(groups[pb]))),
        },
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED, "unit": "source group"},
    }

    # --- A1. the missing column: gate-free SLA, every locator on one protocol ---
    report["chance_sla"] = chance_sla(tgt, cells, off)
    report["sla_gate_free"] = {n: sla_gate_free(p, tgt, cells) for n, p in peaks.items()}
    report["mean_sla_gate_free"] = {n: mean_sla(p, tgt, cells) for n, p in peaks.items()}
    report["mean_sla_gate_free"]["chance"] = float(np.mean(list(report["chance_sla"].values())))

    # --- A2. gate held fixed, both directions ----------------------------------
    grid: dict[str, dict[str, float]] = {}
    for gate_name, opened in d["gates"].items():
        grid[gate_name] = {}
        for loc_name, p in peaks.items():
            macro, rows = gated_f1(p, tgt, cells, pb & opened)
            grid[gate_name][loc_name] = macro
            report.setdefault("per_cell_gate_hold", {}).setdefault(gate_name, {})[loc_name] = rows
        grid[gate_name]["_answers_opened"] = int((pb & opened).sum())
    report["macro_f1_gate_hold"] = grid
    report["native_pairings"] = {
        "ct7": {"gate": "ct7_frozen_tail15", "macro_f1": grid["ct7_frozen_tail15"]["ct7"]},
        "token_l_sml": {"gate": "loco5_at_0.33", "macro_f1": grid["loco5_at_0.33"]["token_l_sml"]},
    }

    # --- A3. paired source-group intervals -------------------------------------
    # One pass per draw, all arms scored on the SAME resampled groups: that is what
    # makes the differences below paired rather than two independent intervals.
    rng = np.random.default_rng(SEED)
    sla_acc: dict[str, list[float]] = {n: [] for n in peaks}
    for i in _group_bootstrap_draws(groups, pb & (tgt >= 0), rng):
        t, c = tgt[i], cells[i]
        for n, p in peaks.items():
            sla_acc[n].append(mean_sla(p[i], t, c))
    sla_draws = {n: np.asarray(v) for n, v in sla_acc.items()}

    rng_f1 = np.random.default_rng(SEED + 1)
    f1_acc: dict[tuple[str, str], list[float]] = {
        (g, l): [] for g in d["gates"] for l in peaks}
    for i in _group_bootstrap_draws(groups, pb, rng_f1):
        t, c = tgt[i], cells[i]
        for gate_name, opened in d["gates"].items():
            o = opened[i]
            for loc_name, p in peaks.items():
                f1_acc[(gate_name, loc_name)].append(gated_f1(p[i], t, c, o)[0])
    f1_draws = {k: np.asarray(v) for k, v in f1_acc.items()}

    report["intervals_gate_free_mean_sla"] = {
        "token_l_sml_minus_token_equal_mean":
            paired_interval(sla_draws["token_l_sml"], sla_draws["token_equal_mean"]),
        "ct7_minus_token_l_sml":
            paired_interval(sla_draws["ct7"], sla_draws["token_l_sml"]),
        "ct7_minus_token_equal_mean":
            paired_interval(sla_draws["ct7"], sla_draws["token_equal_mean"]),
    }
    report["intervals_macro_f1_gate_held"] = {
        "under_ct7_frozen_tail15__ct7_minus_token_l_sml": paired_interval(
            f1_draws[("ct7_frozen_tail15", "ct7")],
            f1_draws[("ct7_frozen_tail15", "token_l_sml")]),
        "under_loco5_at_0.33__ct7_minus_token_l_sml": paired_interval(
            f1_draws[("loco5_at_0.33", "ct7")],
            f1_draws[("loco5_at_0.33", "token_l_sml")]),
        "under_loco5_at_0.33__token_l_sml_minus_token_equal_mean": paired_interval(
            f1_draws[("loco5_at_0.33", "token_l_sml")],
            f1_draws[("loco5_at_0.33", "token_equal_mean")]),
    }

    # ------------------------------------------------------------------ console
    order = ["ct7", "token_l_sml", "token_equal_mean"]
    print("=" * 84)
    print("A1. GATE-FREE SLA (Mind-the-Gap protocol: erroneous answers only, no gate)")
    print("=" * 84)
    print(f"{'cell':24s} {'n_err':>6s} {'CT7':>8s} {'L-SML':>8s} {'equal':>8s} {'chance':>8s} "
          f"{'MtG-drop':>9s} {'MtG-avg':>8s}")
    mtg_drop, mtg_avg = [], []
    for cell in sorted(report["sla_gate_free"]["ct7"]):
        row = [100 * report["sla_gate_free"][n][cell]["sla"] for n in order]
        published = _canon.MIND_THE_GAP["q4" if cell.endswith("q4") else "q8"].get(cell[3:-3], {})
        mtg_drop.append(published.get("shannon_drop", float("nan")))
        mtg_avg.append(published.get("shannon_avg", float("nan")))
        print(f"{cell:24s} {report['sla_gate_free']['ct7'][cell]['n_error']:6d} "
              f"{row[0]:8.2f} {row[1]:8.2f} {row[2]:8.2f} "
              f"{100 * report['chance_sla'][cell]:8.2f} {mtg_drop[-1]:9.2f} {mtg_avg[-1]:8.2f}")
    m = report["mean_sla_gate_free"]
    print(f"{'MEAN over 8 cells':24s} {'':6s} {100*m['ct7']:8.2f} {100*m['token_l_sml']:8.2f} "
          f"{100*m['token_equal_mean']:8.2f} {100*m['chance']:8.2f} "
          f"{np.mean(mtg_drop):9.2f} {np.mean(mtg_avg):8.2f}")
    print("  MtG = Chen et al., ICML 2026, Table 3, same subsets and sizes. Their score is a")
    print("  derivative (EMA then mean of the worst M drops); CT7 and the token arms are levels.")
    report["mind_the_gap_published_mean"] = {"shannon_drop": float(np.mean(mtg_drop)),
                                             "shannon_avg": float(np.mean(mtg_avg))}

    print()
    print("=" * 84)
    print("A2. PROCESSBENCH MACRO-F1 WITH THE GATE HELD FIXED (rows = gate, cols = locator)")
    print("=" * 84)
    print(f"{'gate':24s} {'opened':>7s} {'CT7':>8s} {'L-SML':>8s} {'equal':>8s}")
    for gate_name, row in grid.items():
        print(f"{gate_name:24s} {row['_answers_opened']:7d} "
              f"{100*row['ct7']:8.2f} {100*row['token_l_sml']:8.2f} "
              f"{100*row['token_equal_mean']:8.2f}")
    print("  (native pairings, i.e. the numbers in the handoff table, are on the diagonal:")
    print(f"   CT7 under its own gate {100*grid['ct7_frozen_tail15']['ct7']:.2f}, "
          f"token L-SML under LOCO-5 {100*grid['loco5_at_0.33']['token_l_sml']:.2f})")

    print()
    print("=" * 84)
    print("A3. PAIRED SOURCE-GROUP INTERVALS")
    print("=" * 84)
    for title, block in (("gate-free mean SLA", report["intervals_gate_free_mean_sla"]),
                         ("macro-F1, gate held", report["intervals_macro_f1_gate_held"])):
        print(f"-- {title}")
        for name, iv in block.items():
            flag = "excludes zero" if iv["excludes_zero"] else "includes zero"
            print(f"   {name:56s} {iv['mean_difference_pp']:+7.2f} pp "
                  f"[{iv['ci95_pp'][0]:+6.2f}, {iv['ci95_pp'][1]:+6.2f}]  {flag}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    print()
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
