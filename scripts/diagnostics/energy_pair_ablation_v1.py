#!/usr/bin/env python
"""Is the energy pair worth two slots in the bank -- and would a swap be better?

Step 423 observed that L-SML gives `energy_level` and `energy_innovation` weights that
are equal and opposite, so their joint contribution is one difference direction whose
split between the two members is not identified by the data. This file checks the
measured facts behind that and then asks the practical question directly, by refitting.

The measured facts, before any ablation:
  * the two fitted weights sum to EXACTLY zero in all five folds, to six decimals;
  * they are the LARGEST magnitude in the bank, 0.409 against q15_H1's 0.373;
  * the two channels correlate -0.886 after risk orientation;
  * `energy_level` is -E (full-vocabulary logsumexp, risk-signed) and
    `energy_innovation` is E - prefix_mean(E), so the difference L-SML lands on is
    -2E + prefix_mean(E): mostly the raw level, with a prefix-mean correction.

Variants, each a full five-fold refit -- standardizer, L-SML weights, token fusion,
Top-10 step readout, argmax, gate-free SLA:

  full11                   the published arm. Must replay 35.92 exactly.
  drop_energy_innovation   ten channels, keep the level
  drop_energy_level        ten channels, keep the innovation
  drop_both_energy         nine channels
  explicit_difference      ten channels, the pair replaced by their raw difference,
                           which makes the gauge direction explicit and costs one slot
  swap_innovation_for_drop eleven channels, `energy_innovation` replaced by the
                           Mind-the-Gap evidence-drop channel
  add_drop_to_full         twelve channels, the drop channel ADDED, which separates
                           "the drop helps" from "the innovation hurts"

The drop channel is the only spare feature available without new inference: the
renormalised top-20 negative entropy, EMA span 5, first difference, risk-oriented. It
correlates 0.9996 with our own q15_H1 at the level, so what it adds is the derivative.

Development-only. Nothing here is a candidate until it is confirmed on untouched data.
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
OUT = RES / "ENERGY_PAIR_ABLATION.json"

TOKEN_CAP = 60_000
BOOTSTRAP_DRAWS = 10_000
SEED = 20260919
READOUT_K = 10

from spectral_utils.claude_feature_bank_v1 import (  # noqa: E402
    FEATURE_NAMES, fit_l_sml_weights, fit_token_standardizer, fuse_token_matrix,
)

_spec = importlib.util.spec_from_file_location(
    "stage_b_2x2_v1", Path(__file__).with_name("stage_b_2x2_v1.py"))
_sb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sb)
_group_draws, interval = _sb._group_draws, _sb.interval


def step_topk(values: np.ndarray, spans: np.ndarray, k: int = READOUT_K) -> np.ndarray:
    out = np.empty(len(spans))
    for i, (a, b) in enumerate(spans):
        seg = values[a:b]
        kk = min(k, len(seg))
        out[i] = np.partition(seg, len(seg) - kk)[-kk:].mean()
    return out


def main() -> None:
    records = json.load(open(ROSTER / "JOINED.json", encoding="utf8"))["records"]
    with np.load(ROSTER / "JOINED.npz", allow_pickle=False) as z:
        offsets = np.asarray(z["offsets"], int)
        target = np.asarray(z["target"], int)
    cells = np.asarray([r["cell"] for r in records], str)
    groups = np.asarray([r["group_id"] for r in records], str)
    outer = json.load(open(ROSTER / "FOLDS_V2.json", encoding="utf8"))["outer"]
    folds = np.asarray([outer[r["group_id"]] for r in records], int)
    pb = np.char.startswith(cells, "pb_")

    with np.load(RES / "TOKEN_MATRICES.npz", allow_pickle=False) as z:
        tokens = z["tokens"].astype(float)
        tok_off = np.asarray(z["token_offsets"], int)
        step_spans = np.asarray(z["step_spans"], int)
    with np.load(RES / "EVIDENCE_DROP.npz", allow_pickle=False) as z:
        drop = np.asarray(z["risk_token"], float)

    names = list(FEATURE_NAMES)
    iL, iI = names.index("energy_level"), names.index("energy_innovation")
    keep_all = list(range(len(names)))

    # Column recipes. Each returns (matrix, channel names) for the whole population; the
    # per-answer slices are taken afterwards so every variant shares one code path.
    difference = (tokens[:, iL] - tokens[:, iI])[:, None]
    variants = {
        "full11": (tokens, names),
        "drop_energy_innovation": (tokens[:, [j for j in keep_all if j != iI]],
                                   [n for j, n in enumerate(names) if j != iI]),
        "drop_energy_level": (tokens[:, [j for j in keep_all if j != iL]],
                              [n for j, n in enumerate(names) if j != iL]),
        "drop_both_energy": (tokens[:, [j for j in keep_all if j not in (iL, iI)]],
                             [n for j, n in enumerate(names) if j not in (iL, iI)]),
        "explicit_difference": (
            np.hstack([tokens[:, [j for j in keep_all if j not in (iL, iI)]], difference]),
            [n for j, n in enumerate(names) if j not in (iL, iI)] + ["energy_difference"]),
        "swap_innovation_for_drop": (
            np.hstack([tokens[:, [j for j in keep_all if j != iI]], drop[:, None]]),
            [n for j, n in enumerate(names) if j != iI] + ["evidence_drop"]),
        "add_drop_to_full": (np.hstack([tokens, drop[:, None]]), names + ["evidence_drop"]),
    }

    n = len(records)
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n)]
    total_steps = int(offsets[-1])
    rows = np.flatnonzero(pb & (target >= 0))
    cell_names = sorted(set(cells[rows]))
    code = np.searchsorted(np.asarray(cell_names), cells[rows])
    n_cells = len(cell_names)

    def mean_sla(peaks: np.ndarray) -> float:
        hit = peaks == target[rows]
        counts = np.bincount(code, minlength=n_cells)
        return float((np.bincount(code, weights=hit, minlength=n_cells) / counts).mean())

    started = time.time()
    peaks_of: dict[str, np.ndarray] = {}
    detail: dict[str, dict] = {}
    for name, (matrix, channels) in variants.items():
        mats = [matrix[tok_off[i]:tok_off[i + 1]] for i in range(n)]
        step_l = np.full(total_steps, np.nan)
        step_e = np.full(total_steps, np.nan)
        fits = []
        for fold in np.unique(folds):
            train = np.flatnonzero(folds != fold)
            std = fit_token_standardizer((mats[i] for i in train), cap=TOKEN_CAP)
            weights, _ = fit_l_sml_weights((mats[i] for i in train), std)
            fits.append({"fold": int(fold), "weights": dict(zip(channels, weights.round(5).tolist()))})
            for i in np.flatnonzero(folds == fold):
                fused, equal = fuse_token_matrix(mats[i], std, weights=weights)
                step_l[offsets[i]:offsets[i + 1]] = step_topk(fused, spans[i])
                step_e[offsets[i]:offsets[i + 1]] = step_topk(equal, spans[i])
        if not (np.isfinite(step_l).all() and np.isfinite(step_e).all()):
            raise ValueError(f"{name}: non-finite step scores")
        for rule, series in (("l_sml", step_l), ("equal", step_e)):
            p = np.asarray([int(np.argmax(series[offsets[i]:offsets[i + 1]])) for i in rows])
            peaks_of[f"{name}__{rule}"] = p
            detail.setdefault(name, {})[f"sla_{rule}"] = 100 * mean_sla(p)
        detail[name]["channels"] = len(channels)
        detail[name]["fits"] = fits
        print(f"[variant] {name:26s} n_ch {len(channels):2d}  L-SML {detail[name]['sla_l_sml']:6.2f}  "
              f"equal {detail[name]['sla_equal']:6.2f}   ({time.time() - started:.0f}s)", flush=True)
        del mats

    anchor = detail["full11"]["sla_l_sml"]
    print(f"\nanchor: full11 L-SML = {anchor:.2f}  (must be 35.92)")
    if abs(anchor - 35.92) > 0.005:
        raise SystemExit(f"anchor failed: {anchor:.4f}")

    # ---- paired intervals against the full eleven-channel bank ---------------------
    rng = np.random.default_rng(SEED)
    keys = sorted(peaks_of)
    acc = {k: np.empty(BOOTSTRAP_DRAWS) for k in keys}
    row_index = np.full(len(target), -1, int)
    row_index[rows] = np.arange(len(rows))
    for d, draw in enumerate(_group_draws(groups, pb & (target >= 0), rng, BOOTSTRAP_DRAWS)):
        take = row_index[draw]
        dt, dcode = target[rows][take], code[take]
        counts = np.bincount(dcode, minlength=n_cells)
        present = counts > 0
        for k in keys:
            hit = peaks_of[k][take] == dt
            acc[k][d] = (np.bincount(dcode, weights=hit, minlength=n_cells)[present]
                         / counts[present]).mean()

    report = {"schema": "energy-pair-ablation-v1", "development_only": True,
              "readout": f"Top-{READOUT_K} step mean, argmax", "anchor_full11": anchor,
              "variants": detail,
              "intervals_vs_full11": {
                  k: interval(acc[k] - acc["full11__l_sml"]) for k in keys},
              # Within each variant, how much is L-SML still worth over equal weighting?
              # This is the quantity Step 422 headlined at +3.32 pp on the full bank, and
              # a redundancy repair that equal fusion cannot do for itself would inflate
              # it. Reported per variant so the two explanations can be told apart.
              "intervals_l_sml_minus_equal": {
                  name: interval(acc[f"{name}__l_sml"] - acc[f"{name}__equal"])
                  for name in variants}}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")
    np.savez_compressed(RES / "ENERGY_PAIR_PEAKS.npz", **peaks_of)

    print()
    print("=" * 94)
    print("ENERGY PAIR ABLATION -- gate-free SLA, Top-10 argmax, 8 ProcessBench cells")
    print("=" * 94)
    print(f"{'variant':26s}{'ch':>4s}{'L-SML':>8s}{'equal':>8s}{'L-SML vs full11 (pp)':>30s}")
    for name in variants:
        iv = report["intervals_vs_full11"][f"{name}__l_sml"]
        mark = "*" if iv["excludes_zero"] else " "
        print(f"{name:26s}{detail[name]['channels']:4d}{detail[name]['sla_l_sml']:8.2f}"
              f"{detail[name]['sla_equal']:8.2f}"
              f"{iv['point_pp']:+11.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]{mark}")
    print()
    print("=" * 94)
    print("HOW MUCH IS L-SML STILL WORTH OVER EQUAL WEIGHTING, INSIDE EACH VARIANT?")
    print("=" * 94)
    print(f"{'variant':26s}{'L-SML':>8s}{'equal':>8s}{'L-SML - equal (pp)':>30s}")
    for name in variants:
        iv = report["intervals_l_sml_minus_equal"][name]
        mark = "*" if iv["excludes_zero"] else " "
        print(f"{name:26s}{detail[name]['sla_l_sml']:8.2f}{detail[name]['sla_equal']:8.2f}"
              f"{iv['point_pp']:+11.2f} [{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]{mark}")
    print()
    print(f"written: {OUT}  ({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
