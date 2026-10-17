#!/usr/bin/env python
"""Length axis, attack 1: decompose Step 420's length-calibration ablation by subset.

Step 420 measured that neutralising the step-length coupling costs 8.16 PB points
(CT7 41.19 -> LX7 33.03) and that adding an explicit log-length view back recovers
only 3.35 (LX8 36.38).  Both are MACRO numbers.  The per-subset decomposition was
never computed, and the mean is exactly the thing that hid the length asymmetry in
the first place.

The question this answers, and the only one it answers: **is the 8.16 uniform, or
is it negative on the short-chain subsets and positive on the long-chain ones?**

  uniform            -> the calibrated readout is a dead tool; close it.
  short-neg/long-pos -> it is a length-CONDITIONAL mechanism, not a failed tool,
                        and the deficit against Chen et al. has a candidate repair.

Nothing is refitted and no arm is new.  The five Step-420 arms are rebuilt from the
frozen banks and must reproduce that step's published macro numbers before any new
number is printed (see GATE below).  Step 420's own caveat still binds: length is
real evidence, not only bias -- the first error genuinely is the longest step in
29.7% of erroneous answers against 15.5% chance -- so LX7 (no length) and LX8
(length restored as an explicit view) are always reported together.

Inputs, all restored from the Drive backup of 2026-09-17, none re-derived:
  results/length_explicit_ct7_v1/bank/*.npz      top10 / calibrated / counts, 5 streams
  results/length_explicit_ct7_v1/bocpd.npz       the BOCPD residual channel
  results/chosen_token_calibration_v1/extracted_sufficient/*.npz
  results/chosen_token_calibration_v1/CT7_DEV_SCORES.npz    frozen CT7 + its gate
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
LEN_DIR = ROOT / "results" / "length_explicit_ct7_v1"
CT7_DIR = ROOT / "results" / "chosen_token_calibration_v1"
OUT = ROOT / "results" / "token_probability_fusion_v1" / "LENGTH_AXIS_BY_SUBSET.json"

BOOTSTRAP_DRAWS = 10_000
SEED = 20260918

SHORT = ("gsm8k", "math")
LONG = ("olympiadbench", "omnimath")

# Step 420's published macro PB, results/length_explicit_ct7_v1/RESULTS.json.
PUBLISHED = {"CT7": 0.41188745848863717, "CT7+LEN": 0.41401021582525155,
             "LEN": 0.35142017339315323, "LX7": 0.330261931046284,
             "LX8": 0.363799273930601}
GATE_TOL = 5e-4  # the rebuild goes through float64; the frozen store is float32

from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402
from spectral_utils.frozen_locator_ct7 import despiked_chosen_token_z  # noqa: E402
from spectral_utils.chosen_token_calibration import SUFFICIENT  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "gate_isolation_v1", Path(__file__).with_name("gate_isolation_token_lsml_v1.py"))
_canon = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_canon)
peaks_of, sla_gate_free, gated_f1 = _canon.peaks_of, _canon.sla_gate_free, _canon.gated_f1


def astd(v: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    if v.ndim == 1:
        return masked_answer_standardize(v[:, None], np.isfinite(v)[:, None], offsets)[:, 0]
    return masked_answer_standardize(np.nan_to_num(v), np.isfinite(v), offsets)


def load_sufficient(offsets: np.ndarray) -> np.ndarray:
    """Mirror analyze_chosen_token_step_tests_v2.load_folder, pointed at the restored dir."""
    width = len(SUFFICIENT)
    x = np.full((offsets[-1], width), np.nan)
    done = np.zeros(len(offsets) - 1, bool)
    for path in sorted((CT7_DIR / "extracted_sufficient").glob("*.npz")):
        with np.load(path) as z:
            idx, vals = z["indexes"], z["values"]
            cursor = 0
            for i in idx:
                a, b = offsets[i:i + 2]
                x[a:b] = vals[cursor:cursor + b - a]
                cursor += b - a
                done[i] = True
    if not done.all():
        raise ValueError("incomplete extracted_sufficient")
    return x


def build_arms(offsets: np.ndarray) -> dict[str, np.ndarray]:
    total = int(offsets[-1])
    top = np.full((total, 5), np.nan)
    cal = np.zeros((total, 5))
    cnt = np.zeros((total, 5))
    done = np.zeros(len(offsets) - 1, bool)
    for path in sorted((LEN_DIR / "bank").glob("*.npz")):
        with np.load(path) as z:
            idx, ztop, zcal, zcnt = z["indexes"], z["top10"], z["calibrated"], z["counts"]
            cursor = 0
            for i in idx:
                a, b = offsets[i:i + 2]
                n = b - a
                top[a:b], cal[a:b], cnt[a:b] = ztop[cursor:cursor + n], zcal[cursor:cursor + n], zcnt[cursor:cursor + n]
                cursor += n
                done[i] = True
    if not done.all():
        raise ValueError("incomplete length bank")
    with np.load(LEN_DIR / "bocpd.npz") as z:
        aux_top, aux_cal = z["top10"].copy(), z["calibrated"].copy()

    suff = load_sufficient(offsets)
    token_view = despiked_chosen_token_z(suff, offsets)
    loglen = astd(np.log(suff[:, 0]), offsets)

    ct7_views = np.column_stack([astd(top, offsets), astd(aux_top, offsets), token_view])
    cal_std = masked_answer_standardize(cal, cnt > 0, offsets)
    lx7_views = np.column_stack([cal_std, astd(aux_cal, offsets), token_view])

    return {
        "CT7": ct7_views.mean(1),
        "CT7+LEN": np.column_stack([ct7_views, loglen]).mean(1),
        "LX7": lx7_views.mean(1),
        "LX8": np.column_stack([lx7_views, loglen]).mean(1),
        "LEN": loglen,
    }


def _group_draws(groups, mask, rng):
    order = np.argsort(groups[mask], kind="stable")
    flat = np.flatnonzero(mask)[order]
    _, starts, counts = np.unique(groups[flat], return_index=True, return_counts=True)
    for _ in range(BOOTSTRAP_DRAWS):
        draw = rng.integers(0, len(starts), size=len(starts))
        take, base = counts[draw], starts[draw]
        ends = np.cumsum(take)
        within = np.arange(int(ends[-1])) - np.repeat(ends - take, take)
        yield flat[np.repeat(base, take) + within]


def interval(diff: np.ndarray) -> dict:
    d = diff[np.isfinite(diff)]
    lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
    return {"point_pp": float(100 * d.mean()), "ci95_pp": [float(100 * lo), float(100 * hi)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def subset_of(cell: str) -> str:
    return cell[3:-3]


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets, target = np.asarray(z["offsets"], int), np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], dtype=str)
    groups = np.asarray([r["group_id"] for r in records], dtype=str)
    with np.load(CT7_DIR / "CT7_DEV_SCORES.npz", allow_pickle=False) as z:
        ct7_frozen, gate = z["step_scores"].copy(), z["gate"].copy().astype(bool)
    pb = np.char.startswith(cells, "pb_")

    arms = build_arms(offsets)

    # The rebuild exists to GATE the reconstruction, not to be reported. It differs from
    # the frozen store at 1e-6 (float32 round-trip), which is enough to flip argmax on a
    # near-tied step and move a per-cell SLA by a few tenths. Report the frozen CT7, so
    # that every CT7 number here is identical to the one Stage A published.
    gates = {"ct7_rebuild_max_abs_diff": float(np.max(np.abs(arms["CT7"] - ct7_frozen)))}
    rebuilt_peaks = {a: peaks_of(s, offsets) for a, s in arms.items()}
    rebuilt_macro = {a: gated_f1(p, target, cells, pb & gate)[0] for a, p in rebuilt_peaks.items()}
    for arm, want in PUBLISHED.items():
        got = rebuilt_macro[arm]
        gates[f"macro_{arm}"] = {"rebuilt": got, "published": want, "abs_diff": abs(got - want)}
        if abs(got - want) > GATE_TOL:
            raise SystemExit(
                f"GATE FAILED: {arm} rebuilt {100*got:.4f} vs published {100*want:.4f}; "
                "no per-subset number written.")
    print("rebuild gate PASS -- all five Step-420 macro numbers reproduced "
          f"(max |diff| {100*max(g['abs_diff'] for g in list(gates.values())[1:]):.4f} pp, "
          f"CT7 step-score max |diff| {gates['ct7_rebuild_max_abs_diff']:.2e})\n")

    # Past the gate, CT7 is the frozen vector so every CT7 number matches Stage A exactly.
    arms["CT7"] = ct7_frozen
    peaks = dict(rebuilt_peaks, CT7=peaks_of(ct7_frozen, offsets))
    macro = dict(rebuilt_macro, CT7=gated_f1(peaks["CT7"], target, cells, pb & gate)[0])

    # ------------------------------------------------- per-subset decomposition
    sla = {a: sla_gate_free(p, target, cells) for a, p in peaks.items()}
    per_cell_macro = {a: {c: r["f1"] for c, r in gated_f1(p, target, cells, pb & gate)[1].items()}
                      for a, p in peaks.items()}

    report = {
        "schema": "token-probability-fusion-v1-length-axis-by-subset",
        "development_only": True,
        "refit": False,
        "question": "is the 8.16 pp cost of length calibration uniform, or length-conditional?",
        "gates": gates,
        "macro_pb": macro,
        "sla_gate_free": sla,
        "per_cell_macro_pb": per_cell_macro,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED, "unit": "source group"},
    }

    # ------------------------------------------------------- the decisive contrast
    rng = np.random.default_rng(SEED)
    acc: dict[str, list] = {a: [] for a in arms}
    for i in _group_draws(groups, pb & (target >= 0), rng):
        t, c = target[i], cells[i]
        for a, p in peaks.items():
            # Key by FULL cell name. Keying by subset_of() collapses q4 and q8 into one
            # entry and silently keeps only the second, halving the population.
            per = {k: v["sla"] for k, v in sla_gate_free(p[i], t, c).items()}
            short = np.mean([v for k, v in per.items() if subset_of(k) in SHORT])
            lng = np.mean([v for k, v in per.items() if subset_of(k) in LONG])
            acc[a].append((short, lng))
    draws = {a: np.asarray(v) for a, v in acc.items()}

    report["interaction"] = {}
    for arm in ("LX7", "LX8", "CT7+LEN"):
        d_short = draws[arm][:, 0] - draws["CT7"][:, 0]
        d_long = draws[arm][:, 1] - draws["CT7"][:, 1]
        report["interaction"][f"{arm} - CT7"] = {
            "short_chain": interval(d_short),
            "long_chain": interval(d_long),
            "long_minus_short": interval(d_long - d_short),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    # -------------------------------------------------------------- console
    order = ["CT7", "LX7", "LX8", "CT7+LEN", "LEN"]
    print("=" * 92)
    print("GATE-FREE SLA BY SUBSET (erroneous answers, no gate) -- the decomposition of the 8.16")
    print("=" * 92)
    print(f"{'cell':24s} " + "".join(f"{a:>9s}" for a in order) + f"{'LX7-CT7':>10s}{'LX8-CT7':>10s}")
    for cell in sorted(sla["CT7"]):
        row = [100 * sla[a][cell]["sla"] for a in order]
        print(f"{cell:24s} " + "".join(f"{v:9.2f}" for v in row) +
              f"{row[1]-row[0]:10.2f}{row[2]-row[0]:10.2f}")
    for label, group in (("SHORT (gsm8k, math)", SHORT), ("LONG (olympiad, omni)", LONG)):
        row = [100 * np.mean([v["sla"] for k, v in sla[a].items() if subset_of(k) in group]) for a in order]
        print(f"{label:24s} " + "".join(f"{v:9.2f}" for v in row) +
              f"{row[1]-row[0]:10.2f}{row[2]-row[0]:10.2f}")
    print()
    print(f"{'macro PB (gated)':24s} " + "".join(f"{100*macro[a]:9.2f}" for a in order))

    print()
    print("=" * 92)
    print("IS THE COST LENGTH-CONDITIONAL?  paired source-group intervals on the SLA change")
    print("=" * 92)
    for arm, block in report["interaction"].items():
        print(f"-- {arm}")
        for key in ("short_chain", "long_chain", "long_minus_short"):
            iv = block[key]
            flag = "excludes zero" if iv["excludes_zero"] else "includes zero"
            print(f"   {key:18s} {iv['point_pp']:+7.2f} pp "
                  f"[{iv['ci95_pp'][0]:+6.2f}, {iv['ci95_pp'][1]:+6.2f}]  {flag}")
    print()
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()
