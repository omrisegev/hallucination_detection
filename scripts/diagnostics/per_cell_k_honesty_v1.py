#!/usr/bin/env python
"""Does per-cell K tuning survive an honest split, or is the per-cell optimum noise?

The fine sweep produced a per-cell argmax of 15, 12, 9, 16, 30, 37, 28, 56 -- short cells
peaking near 12-16 and long ones near 28-37. Before deriving any metric from that
pattern, it has to survive the obvious objection: each argmax is the maximum of 64
correlated values estimated on 200-760 answers, where the standard error of a proportion
near 0.5 is about 2-3.5 pp. Maximising over a noisy curve manufactures an optimum whether
or not one exists.

The honest test splits the SOURCE GROUPS of each cell in half, picks K on one half and
scores the other, and repeats over many random splits:

  global_fixed     one K for every cell, chosen on the selection half
  per_cell_tuned   a separate K per cell, chosen on the selection half
  oracle_per_cell  a separate K per cell chosen on the EVALUATION half -- label-using,
                   and the ceiling that per-cell tuning is being measured against
  K=10             what we use today

If per_cell_tuned does not beat global_fixed out of sample, the per-cell pattern is not
usable however clean it looks in-sample, and no metric should be derived from it.

Splitting by source group, never by answer, so a question cannot appear on both sides.
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
OUT = RES / "PER_CELL_K_HONESTY.json"
MAX_K = 64
N_SPLITS = 400
SEED = 20260919

_b = importlib.util.spec_from_file_location("stage_b", Path(__file__).with_name("stage_b_2x2_v1.py"))
SB = importlib.util.module_from_spec(_b)
_b.loader.exec_module(SB)
_s = importlib.util.spec_from_file_location("shape", Path(__file__).with_name("readout_shape_and_neff_v1.py"))
SH = importlib.util.module_from_spec(_s)
_s.loader.exec_module(SH)

from spectral_utils.digitfree_broad50 import masked_answer_standardize  # noqa: E402


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
    n = len(records)
    spans = [step_spans[offsets[i]:offsets[i + 1]] for i in range(n)]
    mats = [tokens[tok_off[i]:tok_off[i + 1]] for i in range(n)]

    print("[fit] token-level L-SML on donor folds, once", flush=True)
    fused = [None] * n
    for fold in np.unique(folds):
        train = np.flatnonzero(folds != fold)
        std = SB.fit_token_standardizer((mats[i] for i in train), cap=SB.TOKEN_CAP)
        w, _ = SB.fit_l_sml_weights((mats[i] for i in train), std)
        for i in np.flatnonzero(folds == fold):
            fused[i], _ = SB.fuse_token_matrix(mats[i], std, weights=w)

    print("[sweep] hit matrix over every integer K", flush=True)
    grid = np.zeros((int(offsets[-1]), MAX_K))
    for i in range(n):
        a = offsets[i]
        for s, (lo, hi) in enumerate(spans[i]):
            grid[a + s] = SH.running_topk(fused[i][lo:hi], MAX_K)
    for k in range(MAX_K):
        grid[:, k] = masked_answer_standardize(
            grid[:, k][:, None], np.isfinite(grid[:, k])[:, None], offsets)[:, 0]

    eligible = np.flatnonzero(pb & (target >= 0))
    hits = np.zeros((len(eligible), MAX_K), dtype=bool)
    for r, i in enumerate(eligible):
        a, b = offsets[i], offsets[i + 1]
        hits[r] = np.argmax(grid[a:b], axis=0) == target[i]
    cell_of = cells[eligible]
    group_of = groups[eligible]
    names = sorted(set(cell_of))
    cell_mask = {c: cell_of == c for c in names}

    def mean_over_cells(mask: np.ndarray, k_by_cell: dict[str, int]) -> float:
        return float(np.mean([hits[mask & cell_mask[c], k_by_cell[c] - 1].mean()
                              for c in names]))

    rng = np.random.default_rng(SEED)
    uniq = np.unique(group_of)
    acc = {k: [] for k in ("K=10", "global_fixed", "per_cell_tuned", "oracle_per_cell")}
    picks = {c: [] for c in names}
    for _ in range(N_SPLITS):
        perm = rng.permutation(len(uniq))
        left = set(uniq[perm[: len(uniq) // 2]])
        sel = np.array([g in left for g in group_of])
        evl = ~sel
        if not (sel.any() and evl.any()):
            continue
        # global K chosen on the selection half
        gcurve = np.array([np.mean([hits[sel & cell_mask[c], k].mean() for c in names])
                           for k in range(MAX_K)])
        gk = int(np.argmax(gcurve)) + 1
        # per-cell K chosen on the selection half
        pk = {c: int(np.argmax(hits[sel & cell_mask[c]].mean(axis=0))) + 1 for c in names}
        # per-cell K chosen on the EVALUATION half: label-using ceiling
        ok = {c: int(np.argmax(hits[evl & cell_mask[c]].mean(axis=0))) + 1 for c in names}
        acc["K=10"].append(mean_over_cells(evl, {c: 10 for c in names}))
        acc["global_fixed"].append(mean_over_cells(evl, {c: gk for c in names}))
        acc["per_cell_tuned"].append(mean_over_cells(evl, pk))
        acc["oracle_per_cell"].append(mean_over_cells(evl, ok))
        for c in names:
            picks[c].append(pk[c])
    draws = {k: np.asarray(v) for k, v in acc.items()}

    def interval(d):
        return {"mean_pp": float(100 * d.mean()),
                "ci95_pp": [float(100 * np.percentile(d, 2.5)),
                            float(100 * np.percentile(d, 97.5))],
                "excludes_zero": bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0)}

    report = {"schema": "token-probability-fusion-v1-per-cell-k-honesty",
              "splits": int(len(draws["K=10"])),
              "out_of_sample": {k: float(100 * v.mean()) for k, v in draws.items()},
              "contrasts": {
                  "per_cell_tuned - global_fixed": interval(draws["per_cell_tuned"] - draws["global_fixed"]),
                  "global_fixed - K=10": interval(draws["global_fixed"] - draws["K=10"]),
                  "per_cell_tuned - K=10": interval(draws["per_cell_tuned"] - draws["K=10"]),
                  "oracle_per_cell - per_cell_tuned": interval(draws["oracle_per_cell"] - draws["per_cell_tuned"]),
              },
              "selected_K": {c: {"median": float(np.median(picks[c])),
                                 "iqr": [float(np.percentile(picks[c], 25)),
                                         float(np.percentile(picks[c], 75))]} for c in names}}
    OUT.write_text(json.dumps(report, indent=1), encoding="utf8")

    print()
    print("=" * 92)
    print(f"OUT-OF-SAMPLE, {report['splits']} source-group half splits")
    print("=" * 92)
    for k, v in report["out_of_sample"].items():
        print(f"  {k:22s} {v:6.2f}")
    print()
    for name, iv in report["contrasts"].items():
        flag = "excludes zero" if iv["excludes_zero"] else "includes zero"
        print(f"  {name:38s} {iv['mean_pp']:+6.2f} pp "
              f"[{iv['ci95_pp'][0]:+6.2f},{iv['ci95_pp'][1]:+6.2f}]  {flag}")
    print()
    print("  K selected per cell across splits (median [IQR]) -- how stable is the pattern?")
    for c in names:
        s = report["selected_K"][c]
        print(f"    {c:22s} {s['median']:5.0f}  [{s['iqr'][0]:.0f}, {s['iqr'][1]:.0f}]")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
