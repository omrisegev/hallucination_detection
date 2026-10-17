"""Stage 1a kill test: does answer-level white-box geometry beat the trivial incumbents?

One binary answer-level question - does this answer contain an error - by AUROC, per cell and
pooled, with a paired source-group bootstrap over the 3,483 groups.

Pre-declared kill rule (plan section 6):
    if invariant geometry does not beat BOTH n_steps AND the locator maximum on the per-cell
    stratified AUROC, with a paired interval excluding zero, the GATE branch ends. The locator
    branch (1b) is unaffected.

The arm is deliberately given its best shot. `geometry_oracle` is the best single geometry
column PER CELL, selected with labels - an ORACLE CEILING, never a candidate. A kill test should
be generous to the thing it might kill: if even the oracle ceiling loses, the branch is dead
without ambiguity. `geometry_equal` is the honest label-free arm (the registered contract
already orients every column to +risk, so an equal mean of z-scored columns needs no labels).

Pooled AUROC is reported but DECIDES NOTHING: PRMBench is one cell, over half the population,
with a base rate 21 points above ProcessBench, so anything separating the benchmarks earns
pooled AUROC with no within-cell signal.

    python scripts/run_whitebox_kill_test_1a.py
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
    assert_label_contract,
    auc_by_cell,
    geometry_summaries,
    group_codes,
    load_joined,
    paired_auc_intervals,
    shared_group_weights,
)

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "results" / "whitebox_layer_views_localization_v1"
CT7 = ROOT.parent / "token-probability-fusion-v1" / "results" / "chosen_token_calibration_v1" / "CT7_DEV_SCORES.npz"
DRAWS = 10000
SEED = 20260919


def main() -> int:
    t0 = time.time()
    print("[load] roster + bundle")
    data = load_joined(ROOT)
    z = np.load(BUNDLE / "ANSWER_LEVEL.npz", allow_pickle=False)
    cells, offsets = data["cells"], data["offsets"]
    assert_label_contract(data["labels"], offsets, data["target"], cells)
    y = answer_error_label(data["labels"], offsets, data["target"], cells)
    codes, ng = group_codes(data["group_id"])
    if not np.array_equal(z["row_id"].astype(str), data["row_id"]):
        raise SystemExit("bundle row order does not match the roster")
    print(f"  {len(y)} answers, {int(y.sum())} positives, {ng} source groups")

    print("[geometry] rotation-invariant summaries")
    values, names, _ = geometry_summaries(z["cov_eigs"], z["hid_proj"], z["resid_norm_mean"])
    print(f"  {values.shape[1]} columns in {time.time() - t0:.0f}s")

    print("[baselines]")
    ct7 = np.load(CT7, allow_pickle=False)
    step_scores = ct7["step_scores"]
    locator_max = np.array([step_scores[offsets[i]:offsets[i + 1]].max()
                            for i in range(len(y))])
    arms = {
        "n_steps": np.diff(offsets).astype(np.float64),
        "locator_max": locator_max,
        "ct7_gate": ct7["gate"].astype(np.float64),
    }

    # Label-free arm: the contract already orients every column to +risk, so an equal mean of
    # globally z-scored columns needs no labels.
    std = values.std(axis=0)
    keep = std > 1e-12
    zsc = (values[:, keep] - values[:, keep].mean(axis=0)) / std[keep]
    arms["geometry_equal"] = zsc.mean(axis=1)

    print("[oracle] best single geometry column per cell (LABEL-SELECTED CEILING)")
    panels = sorted(set(cells.tolist()))
    oracle = np.zeros(len(y))
    oracle_pick = {}
    ones = np.ones(ng)
    from spectral_utils.whitebox_layer_views import auc_plan, weighted_auc
    for panel in panels:
        m = cells == panel
        best, best_auc = None, -1.0
        for j in range(values.shape[1]):
            col = values[m, j]
            if col.std() < 1e-12:
                continue
            a = weighted_auc(auc_plan(y[m], col, codes[m]), ones)
            a = max(a, 1.0 - a)  # ceiling: orientation granted, see caveat in the report
            if a > best_auc:
                best, best_auc = j, a
        sgn = 1.0
        raw = weighted_auc(auc_plan(y[m], values[m, best], codes[m]), ones)
        if raw < 0.5:
            sgn = -1.0
        oracle[m] = sgn * values[m, best]
        oracle_pick[panel] = {"column": names[best], "auc": best_auc}
        print(f"  {panel:<22} {best_auc:.4f}  {names[best]}")
    arms["geometry_oracle"] = oracle

    print(f"[bootstrap] {DRAWS} shared draws over {ng} groups")
    weights = shared_group_weights(ng, DRAWS, SEED)
    results = {name: auc_by_cell(y, score, codes, cells, ng, weights=weights)
               for name, score in arms.items()}

    contrasts = {
        "geometry_oracle - n_steps": ("geometry_oracle", "n_steps"),
        "geometry_oracle - locator_max": ("geometry_oracle", "locator_max"),
        "geometry_equal - n_steps": ("geometry_equal", "n_steps"),
        "geometry_equal - locator_max": ("geometry_equal", "locator_max"),
    }
    intervals = {label: paired_auc_intervals(results[a], results[b])
                 for label, (a, b) in contrasts.items()}

    # --- the pre-declared kill rule, on the per-cell stratified statistic ---
    pb_panels = [p for p in panels if p.startswith("pb_")]
    verdict = {}
    for label in ("geometry_oracle", "geometry_equal"):
        beats = {}
        for base in ("n_steps", "locator_max"):
            iv = intervals[f"{label} - {base}"]
            wins = [p for p in panels if iv[p] and iv[p].get("low", -1) > 0]
            beats[base] = {"cells_won": wins, "n_won": len(wins), "n_panels": len(panels)}
        verdict[label] = beats

    payload = {
        "population": {"answers": int(len(y)), "positives": int(y.sum()), "groups": ng},
        "geometry_columns": int(values.shape[1]),
        "oracle_pick": oracle_pick,
        "auc": {name: {p: {k: v for k, v in e.items() if k != "draws"}
                       for p, e in res.items()} for name, res in results.items()},
        "intervals": intervals,
        "kill_rule": verdict,
        "draws": DRAWS, "seed": SEED,
        "seconds": round(time.time() - t0, 1),
        "notes": {
            "pooled": "descriptive only; PRMBench is one cell with a 21pp higher base rate",
            "oracle": "label-selected per cell, orientation granted; a CEILING, not a candidate",
        },
    }
    out = BUNDLE / "KILL_TEST_1A.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")

    print("\n=== per-cell AUROC ===")
    header = f"{'cell':<22}" + "".join(f"{a:>17}" for a in arms)
    print(header)
    for p in panels + ["POOLED"]:
        row = f"{p:<22}"
        for a in arms:
            row += f"{results[a][p]['point']:>17.4f}"
        print(row)
    print("\n=== kill rule (cells where the interval excludes zero, of 9) ===")
    for label, beats in verdict.items():
        print(f"  {label:<18} beats n_steps in {beats['n_steps']['n_won']}/9, "
              f"beats locator_max in {beats['locator_max']['n_won']}/9")
    print(f"\nwrote {out}  ({payload['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
