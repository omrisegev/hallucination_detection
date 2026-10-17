"""Two corrections the 1a result demands before it can be read.

1. SELECTION INFLATION. `geometry_oracle` is the max of 283 per-cell AUROCs chosen with labels,
   with orientation granted. The max of 283 noisy statistics is biased upward even under no
   signal, so the raw 0.68-0.78 is not comparable to a single pre-specified baseline. Estimate
   the ceiling under the null by permuting the labels WITHIN each cell and recomputing the same
   best-of-283-with-orientation.

2. THE REAL INCUMBENT. The kill rule named `n_steps` and `locator_max`. That was wrong for the
   gate: CT7's production gate is a frozen tail15 statistic, NOT a threshold on the locator
   composite (that is the token arm's LOCO-5 gate). `ct7_gate` is therefore the incumbent a gate
   improvement actually has to beat, and it must be in the contrast set.

    python scripts/run_whitebox_kill_test_1a_followup.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spectral_utils.whitebox_layer_views import (  # noqa: E402
    answer_error_label,
    auc_by_cell,
    auc_plan,
    geometry_summaries,
    group_codes,
    load_joined,
    paired_auc_intervals,
    shared_group_weights,
    weighted_auc,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results" / "whitebox_layer_views_localization_v1"
CT7 = ROOT.parent / "token-probability-fusion-v1" / "results" / "chosen_token_calibration_v1" / "CT7_DEV_SCORES.npz"
PERMS = 200
DRAWS = 10000
SEED = 20260919


def best_of_bank(y, values, codes, ones) -> float:
    """Max over columns of the orientation-granted AUROC - exactly the oracle's rule."""
    best = 0.0
    for j in range(values.shape[1]):
        col = values[:, j]
        if col.std() < 1e-12:
            continue
        a = weighted_auc(auc_plan(y, col, codes), ones)
        if not np.isfinite(a):
            continue
        best = max(best, max(a, 1.0 - a))
    return best


def main() -> int:
    t0 = time.time()
    data = load_joined(ROOT)
    z = np.load(BUNDLE / "ANSWER_LEVEL.npz", allow_pickle=False)
    cells, offsets = data["cells"], data["offsets"]
    y = answer_error_label(data["labels"], offsets, data["target"], cells)
    codes, ng = group_codes(data["group_id"])
    values, names, _ = geometry_summaries(z["cov_eigs"], z["hid_proj"], z["resid_norm_mean"])
    ones = np.ones(ng)
    panels = sorted(set(cells.tolist()))
    rng = np.random.default_rng(SEED)

    print(f"[null] best-of-{values.shape[1]} under {PERMS} within-cell label permutations")
    null = {}
    for panel in panels:
        m = cells == panel
        yv, vv, cv = y[m], values[m], codes[m]
        draws = np.empty(PERMS)
        for p in range(PERMS):
            draws[p] = best_of_bank(rng.permutation(yv), vv, cv, ones)
        null[panel] = {
            "median": float(np.median(draws)),
            "p95": float(np.quantile(draws, 0.95)),
            "max": float(draws.max()),
        }
        print(f"  {panel:<22} null median {null[panel]['median']:.4f}  "
              f"p95 {null[panel]['p95']:.4f}  max {null[panel]['max']:.4f}  "
              f"[{time.time() - t0:.0f}s]")

    print("\n[contrast] geometry_equal vs the REAL incumbent, ct7_gate")
    ct7 = np.load(CT7, allow_pickle=False)
    std = values.std(axis=0)
    keep = std > 1e-12
    zsc = (values[:, keep] - values[:, keep].mean(axis=0)) / std[keep]
    arms = {
        "geometry_equal": zsc.mean(axis=1),
        "ct7_gate": ct7["gate"].astype(np.float64),
    }
    weights = shared_group_weights(ng, DRAWS, SEED + 1)
    res = {k: auc_by_cell(y, v, codes, cells, ng, weights=weights) for k, v in arms.items()}
    iv = paired_auc_intervals(res["geometry_equal"], res["ct7_gate"])

    payload = {"null_best_of_bank": null, "perms": PERMS,
               "geometry_equal_vs_ct7_gate": iv,
               "columns": int(values.shape[1]), "seconds": round(time.time() - t0, 1)}
    (BUNDLE / "KILL_TEST_1A_FOLLOWUP.json").write_text(json.dumps(payload, indent=1),
                                                       encoding="utf-8")

    prior = json.loads((BUNDLE / "KILL_TEST_1A.json").read_text(encoding="utf-8"))
    print("\n=== oracle against its own selection null ===")
    print(f"{'cell':<22}{'oracle':>9}{'null p95':>10}{'excess':>9}   verdict")
    for panel in panels:
        o = prior["oracle_pick"][panel]["auc"]
        p95 = null[panel]["p95"]
        print(f"{panel:<22}{o:>9.4f}{p95:>10.4f}{o - p95:>+9.4f}   "
              f"{'above null' if o > p95 else 'WITHIN NULL'}")

    print("\n=== geometry_equal - ct7_gate (the real incumbent) ===")
    print(f"{'cell':<22}{'delta':>9}{'low':>9}{'high':>9}   verdict")
    for panel in panels + ["POOLED"]:
        e = iv.get(panel)
        if not e or "point" not in e:
            continue
        flag = "geometry wins" if e["low"] > 0 else ("ct7 wins" if e["high"] < 0 else "tie")
        print(f"{panel:<22}{e['point']:>+9.4f}{e['low']:>+9.4f}{e['high']:>+9.4f}   {flag}")
    print(f"\n{time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
