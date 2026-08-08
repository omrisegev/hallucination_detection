#!/usr/bin/env python
"""
build_examples.py — pick worked examples out of a cell and freeze them to JSON for the report.

Separated from the report renderer on purpose: choosing WHICH trace to show is a decision that
must be reproducible and auditable, and re-deriving it inside a rendering pass would hide it. The
selection rule is fixed here and printed on every run.

SELECTION RULE (fixed before looking at any score)
--------------------------------------------------
Among rows with at least `MIN_STEPS` measurable steps:
  * the **wrong** panel is the incorrect answer whose peak step risk is highest — the detector's
    most confident catch, which is what a reader wants to see first;
  * the **right** panel is the correct answer whose peak step risk is LOWEST — the cleanest
    contrast, so the two panels differ in what the detector says and not in trace length;
  * a **miss** panel is included when one exists: an incorrect answer whose peak step risk is
    below the median of correct answers, i.e. a case the detector does not catch.
Showing only the catch would be picking the evidence. The miss panel is not optional.

Usage:
    python scripts/localization/build_examples.py <cell_dir_or_pkl> [--out results/localization]
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (REPO, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from worked_example import as_row, build_example, fit_step_arm  # noqa: E402

MIN_STEPS = 5
MAX_ROWS_FIT = 100000   # the whole cell: peak step risk is now free, only the 3 panels cost time


def resolve_pkl(path):
    if os.path.isfile(path):
        return path
    hits = sorted(glob.glob(os.path.join(path, "raw_*.pkl")))
    if not hits:
        raise SystemExit(f"no raw_*.pkl under {path}")
    return hits[0]


def jsonable(o):
    if isinstance(o, np.ndarray):
        return [None if not np.isfinite(v) else round(float(v), 6) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cell")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "localization"))
    ap.add_argument("--max-rows", type=int, default=MAX_ROWS_FIT)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    pkl = resolve_pkl(a.cell)
    cell_name = os.path.basename(os.path.dirname(pkl))
    with open(pkl, "rb") as f:
        data = pickle.load(f)

    rows, meta = [], []
    for idx in sorted(data):
        q = data[idx].get("question", "")
        for ci, cand in enumerate(data[idx]["candidates"]):
            if "token_offsets" not in cand:
                continue
            cand = {**cand, "question": q}
            r = as_row(cand)
            if sum(1 for s in r["step_token_spans"] if s and s[1] - s[0] >= 8) < MIN_STEPS:
                continue
            rows.append(r)
            meta.append({"idx": int(idx), "cand": ci})
    print(f"{cell_name}: {len(rows)} rows with >= {MIN_STEPS} measurable steps "
          f"(of {sum(len(v['candidates']) for v in data.values())} candidates)")

    keep = rows[:a.max_rows]
    arm, per_row = fit_step_arm(keep)
    if arm is None:
        raise SystemExit("U-PCR declined the pooled-step population for this cell")
    print(f"  U-PCR fit on {arm.n} pooled steps from {len(keep)} answers: "
          f"pool={len(arm.pool)} kept={arm.n_kept} anchor={arm.anchor_name}")

    # Peak step risk comes straight off the pooled fit — one array, already computed. Routing it
    # through `build_example` instead would run a full token trace per row (~1.5 s), which is
    # what previously forced the separability sample down to a few hundred answers and left it
    # with only 19 incorrect ones. The token trace is needed for the PANELS, not the histogram.
    risk_by_row, i = [], 0
    for steps in per_row:
        risk_by_row.append(np.asarray(arm.risk[i:i + len(steps)], dtype=float))
        i += len(steps)
    peaks = np.array([np.nanmax(r) if r.size and np.isfinite(r).any() else np.nan
                      for r in risk_by_row])
    correct = [bool(r["answer_correct"]) for r in keep]
    correct = np.asarray(correct)
    ok = np.isfinite(peaks)

    wrong_i = int(np.flatnonzero(ok & ~correct)[np.argmax(peaks[ok & ~correct])]) \
        if (ok & ~correct).any() else None
    right_i = int(np.flatnonzero(ok & correct)[np.argmin(peaks[ok & correct])]) \
        if (ok & correct).any() else None
    med_correct = float(np.median(peaks[ok & correct])) if (ok & correct).any() else np.nan
    miss_pool = np.flatnonzero(ok & ~correct & (peaks < med_correct))
    miss_i = int(miss_pool[np.argmin(peaks[miss_pool])]) if miss_pool.size else None

    print(f"  peak step risk: incorrect median "
          f"{np.median(peaks[ok & ~correct]):.3f}, correct median {med_correct:.3f}")
    print(f"  panels -> caught={wrong_i} clean={right_i} missed={miss_i} "
          f"({miss_pool.size} incorrect answers fall below the correct-answer median)")

    panels = {}
    for name, i in (("caught", wrong_i), ("clean", right_i), ("missed", miss_i)):
        if i is None:
            continue
        ex = build_example(keep[i], arm, stride=a.stride)
        ex["source"] = {"cell": cell_name, **meta[i], "panel": name}
        panels[name] = jsonable(ex)

    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{cell_name}__examples.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "cell": cell_name,
            "selection_rule": {"min_steps": MIN_STEPS, "max_rows_fit": a.max_rows,
                               "stride": a.stride},
            "arm": {"pool": arm.pool, "n_kept": arm.n_kept, "anchor": arm.anchor_name,
                    "n_steps_fit": int(arm.n), "n_answers_fit": len(keep)},
            "peak_risk": {"incorrect_median": float(np.median(peaks[ok & ~correct])),
                          "correct_median": med_correct,
                          "n_incorrect": int((ok & ~correct).sum()),
                          "n_correct": int((ok & correct).sum()),
                          "n_missed_below_correct_median": int(miss_pool.size)},
            "separability": {"incorrect": jsonable(peaks[ok & ~correct]),
                             "correct": jsonable(peaks[ok & correct])},
            "panels": panels,
        }, f, indent=1)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
