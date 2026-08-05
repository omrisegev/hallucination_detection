#!/usr/bin/env python3
"""SECONDARY diagnostic — Study C of ``SPEC_SOLVER_MECHANISM_STUDY.md``.

WHY THIS EXISTS
---------------
`deem_deep_soft` failed on 27 of 30 attempts in the registered sweep, every one with the same
`ValueError: method returned a non-finite or constant score`.  That exception is raised by
`orient_score` *after* `fit_deem_score` has already returned, so the fit completes and
`save_method_record`'s `except` branch discards the whole `DeemRunResult` — including
`model.history_`, which is exactly what a diagnosis needs.

This probe calls `fit_deem_score` DIRECTLY, before any orientation, and keeps everything the
runner throws away.  It never edits the hashed adapter or the hashed runner.

TIMING: the reviewer required this to run only AFTER the registered sweep exits — a concurrent
fit competes for CPU threads and memory bandwidth and can affect the registered run's timing and
stability.  The gate is asserted at startup and cannot be skipped without an explicit flag.

STOPPING DECISIONS ARE SEPARATE (review item): hard categorical DEEM and repaired soft DEEM are
different methods, so poor hard performance must not veto a successfully repaired soft model.

Usage:
    python scripts/deem_soft_collapse_probe.py --data-dir local_cache
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import GROUP, INSCOPE                                   # noqa: E402
from spectral_utils.deem_adapter import fit_deem_score                     # noqa: E402
from run_dependency_fusion_experiment import (                             # noqa: E402
    DEEM_BASE, dataset_family, derive_oriented_matrix, load_cells,
)

OUT = os.path.join(REPO, "results", "deem_probe")
STUDY = os.path.join(REPO, "results", "dependency_fusion_study")

# ---- registered constants (SPEC_SOLVER_MECHANISM_STUDY.md §5) --------------------------------
COLLAPSED_SD = 1e-6
HEALTHY_SD = 1e-3
COMPLETION_MIN = 0.90
MEANINGFUL_GAIN_PP = 1.0
SEEDS = (0, 1, 2, 3, 4)
GRID_LR = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
GRID_EPOCHS = (100, 300, 1000)
N_PILOT_CELLS = 3
DEFAULT_LR, DEFAULT_EPOCHS = 1e-3, 100


def health(sd):
    """The three-way classification, calibrated so the sweep's three soft 'successes'
    (sigma ~ 1e-8) are collapsed rather than healthy."""
    if not np.isfinite(sd) or sd < COLLAPSED_SD:
        return "collapsed"
    if sd < HEALTHY_SD:
        return "degenerate_nonconstant"
    return "healthy"


def sweep_is_finished(quiet_seconds=600):
    """Gate: no new checkpoint line for >= 10 min and the summary written."""
    records = os.path.join(STUDY, "records.jsonl")
    summary = os.path.join(STUDY, "summary.json")
    if not os.path.exists(records):
        return False, "records.jsonl missing"
    quiet = time.time() - os.path.getmtime(records)
    if quiet < quiet_seconds:
        return False, f"records.jsonl written {quiet:.0f}s ago (< {quiet_seconds}s quiet period)"
    if not os.path.exists(summary):
        return False, "summary.json not written — the sweep did not reach its reporting stage"
    if os.path.getmtime(summary) < os.path.getmtime(records):
        return False, "summary.json is older than records.jsonl"
    return True, f"quiet for {quiet:.0f}s, summary.json present"


def failing_cells():
    """Cells with at least one deem_deep_soft failure, from the checkpoint itself."""
    path = os.path.join(STUDY, "records.jsonl")
    bad = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("arm") != "deem_deep_soft":
                continue
            if rec.get("status") != "ok":
                bad.setdefault(rec["cell"], 0)
                bad[rec["cell"]] += 1
    return bad


def one_fit(F_a, seed, config):
    """Direct call — no orientation, so nothing raises on a constant score and the history
    survives.  Preregistered fallback if `model.fit()` itself starts raising: instrument DEEM
    directly.  That path is reported, never silently taken."""
    started = time.time()
    rec = {"seed": int(seed), "learning_rate": config.learning_rate,
           "epochs": config.epochs, "input_mode": config.input_mode}
    try:
        res = fit_deem_score(F_a.T, seed=seed, config=config, verbose=False)
        score = np.asarray(res.score, dtype=float)
        sd = float(np.std(score))
        hist = res.history or {}
        finite_history = all(
            np.all(np.isfinite(np.asarray(v, dtype=float)))
            for v in hist.values() if isinstance(v, (list, tuple)) and v
            and isinstance(v[0], (int, float)))
        rec.update({
            "status": "fit_returned", "score_sd": sd, "health": health(sd),
            "score_n_unique": int(len(np.unique(score))),
            "score_min": float(score.min()), "score_max": float(score.max()),
            "finite_history": bool(finite_history),
            "history_keys": "|".join(sorted(hist)),
            "class_map": json.dumps(res.class_map),
            "n_epochs_recorded": int(max((len(v) for v in hist.values()
                                          if isinstance(v, (list, tuple))), default=0)),
            "score": score,
        })
        for key, val in hist.items():
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], (int, float)):
                arr = np.asarray(val, dtype=float)
                rec[f"history_{key}_first"] = float(arr[0])
                rec[f"history_{key}_last"] = float(arr[-1])
                rec[f"history_{key}_finite"] = bool(np.all(np.isfinite(arr)))
    except Exception as exc:                                    # model.fit() itself failed
        rec.update({"status": "fit_raised", "health": "fit_raised",
                    "error_type": type(exc).__name__, "error": str(exc)[:300],
                    "score_sd": float("nan"), "score": None,
                    "note": "fit_deem_score lost the model object; direct DEEM instrumentation "
                            "is the preregistered fallback for this case"})
    rec["runtime_seconds"] = time.time() - started
    return rec


def cross_seed_stability(records):
    """Median pairwise |Spearman| between seeds' scores — label-free tie-breaker step 3."""
    scores = [r["score"] for r in records if r.get("score") is not None
              and np.std(r["score"]) > 1e-12]
    if len(scores) < 2:
        return float("nan")
    vals = []
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            vals.append(abs(float(spearmanr(scores[i], scores[j]).statistic)))
    return float(np.median(vals)) if vals else float("nan")


def hard_deem_decision():
    """Decision 2, read off the finished registered study.  Independent of the soft decision."""
    path = os.path.join(STUDY, "per_cell.csv")
    if not os.path.exists(path):
        return {"available": False}
    by_arm = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            arena, _, arm = row["arm"].partition(".")
            if arena == "full":
                by_arm.setdefault(arm, {})[row["cell"]] = float(row["auc"])
    out = {"available": True, "arms": {}}
    ref = by_arm.get("iu_pcr", {})
    for arm in ("deem_irbm_hard_ensemble", "deem_deep_hard_ensemble"):
        cand = by_arm.get(arm, {})
        common = sorted(set(ref) & set(cand))
        if len(common) < 3:
            out["arms"][arm] = {"n_cells": len(common), "verdict": "insufficient cells"}
            continue
        d = np.array([100 * (cand[c] - ref[c]) for c in common])
        fams = [dataset_family(c) for c in common]
        fam_list = sorted(set(fams))
        idx = {f: np.flatnonzero(np.array(fams) == f) for f in fam_list}
        rng = np.random.default_rng(20260805)
        boot = [d[np.concatenate([idx[fam_list[k]] for k in
                                  rng.integers(0, len(fam_list), len(fam_list))])].mean()
                for _ in range(10000)]
        hi = float(np.percentile(boot, 97.5))
        no_advantage = bool(d.mean() <= 0 and hi < MEANINGFUL_GAIN_PP)
        out["arms"][arm] = {
            "n_cells": len(common), "mean_delta_pp": float(d.mean()),
            "family_blocked_ci_hi_pp": hi,
            "shows_no_meaningful_advantage_over_iu": no_advantage,
            "verdict": ("abandon hard DEEM — no meaningful advantage over IU (this rules out the "
                        "preregistered +1.0pp gain; it does NOT prove inferiority)"
                        if no_advantage else "hard DEEM retains a possible meaningful gain"),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--out-dir", default=OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ignore-sweep-gate", action="store_true",
                        help="explicit override; the gate exists because a concurrent fit "
                             "competes with the registered sweep")
    args = parser.parse_args()

    finished, why = sweep_is_finished()
    print(f"sweep gate: {'PASS' if finished else 'BLOCKED'} — {why}")
    if not finished and not args.ignore_sweep_gate:
        raise SystemExit("refusing to run while the registered sweep may still be live")

    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    cells = load_cells(os.path.abspath(args.data_dir))

    bad = failing_cells()
    candidates = [c for c in INSCOPE if c in cells and c in bad]
    if not candidates:
        print("no deem_deep_soft failures in the checkpoint — nothing to repair")
        candidates = [c for c in INSCOPE if c in cells]
    # three failing cells with smallest n, ties alphabetical — deterministic and label-free
    pilot = sorted(candidates, key=lambda c: (len(cells[c]["labels"]), c))[:N_PILOT_CELLS]
    print(f"pilot cells (smallest n among failing): "
          + ", ".join(f"{c} (n={len(cells[c]['labels'])})" for c in pilot))

    base = replace(DEEM_BASE, input_mode="soft", use_preprocessing=True, device=args.device)
    rows = []

    # ---- 1. reproduce the failure and keep what the runner discarded ---------------------
    print("\nreproducing the registered configuration, retaining history:")
    for ck in pilot:
        F, _, _ = derive_oriented_matrix(cells[ck])
        for seed in SEEDS:
            rec = one_fit(F, seed, base)
            rec.update({"cell": ck, "family": dataset_family(ck), "stage": "reproduce"})
            rows.append(rec)
            print(f"  {ck:28s} seed={seed} sd={rec['score_sd']:.3e} {rec['health']}", flush=True)
    write_csv(os.path.join(args.out_dir, "per_fit.csv"), rows)

    # ---- 2. the preregistered grid, seed 0, label-free selection only ---------------------
    print(f"\nconfiguration grid ({len(GRID_LR)}x{len(GRID_EPOCHS)}), seed 0, label-free:")
    grid = []
    for lr in GRID_LR:
        for ep in GRID_EPOCHS:
            cfg = replace(base, learning_rate=float(lr), epochs=int(ep))
            recs = []
            for ck in pilot:
                F, _, _ = derive_oriented_matrix(cells[ck])
                rec = one_fit(F, 0, cfg)
                rec.update({"cell": ck, "family": dataset_family(ck), "stage": "grid"})
                rows.append(rec)
                recs.append(rec)
            healthy = [r for r in recs if r["health"] == "healthy"]
            grid.append({
                "learning_rate": lr, "epochs": ep, "n_cells": len(recs),
                "completion_rate": len(healthy) / max(len(recs), 1),
                "all_finite_history": int(all(r.get("finite_history", False) for r in recs)),
                "cross_seed_stability": float("nan"),          # filled for the winner only
                "deviation_from_default": abs(np.log10(lr / DEFAULT_LR))
                                          + abs(np.log10(ep / DEFAULT_EPOCHS)),
                "median_score_sd": float(np.nanmedian([r["score_sd"] for r in recs])),
            })
            print(f"  lr={lr:<7g} epochs={ep:<5d} completion={grid[-1]['completion_rate']:.2f} "
                  f"median sd={grid[-1]['median_score_sd']:.3e}", flush=True)
    write_csv(os.path.join(args.out_dir, "grid.csv"), grid)

    # ---- 3. deterministic label-free tie-breaker ------------------------------------------
    trace = []
    pool = [g for g in grid]
    best_completion = max(g["completion_rate"] for g in pool)
    pool = [g for g in pool if g["completion_rate"] == best_completion]
    trace.append(f"step 1 completion rate == {best_completion:.2f}: {len(pool)} configs remain")
    finite = [g for g in pool if g["all_finite_history"]]
    if finite:
        pool = finite
    trace.append(f"step 2 finite objective required: {len(pool)} configs remain")
    if len(pool) > 1:
        for g in pool:                    # step 3 needs the seeds, so it runs only on survivors
            cfg = replace(base, learning_rate=float(g["learning_rate"]),
                          epochs=int(g["epochs"]))
            recs = []
            for ck in pilot:
                F, _, _ = derive_oriented_matrix(cells[ck])
                for seed in SEEDS[1:]:
                    rec = one_fit(F, seed, cfg)
                    rec.update({"cell": ck, "family": dataset_family(ck), "stage": "tiebreak"})
                    rows.append(rec)
                    recs.append(rec)
            g["cross_seed_stability"] = cross_seed_stability(recs)
        best_stab = np.nanmax([g["cross_seed_stability"] for g in pool])
        if np.isfinite(best_stab):
            pool = [g for g in pool if g["cross_seed_stability"] >= best_stab - 1e-12]
        trace.append(f"step 3 cross-seed stability == {best_stab:.4f}: {len(pool)} remain")
    pool = sorted(pool, key=lambda g: (g["deviation_from_default"], g["learning_rate"],
                                       g["epochs"]))
    winner = pool[0]
    trace.append(f"step 4 smallest deviation from the registered default: "
                 f"lr={winner['learning_rate']:g}, epochs={winner['epochs']}")
    print("\ntie-breaker trace (AUROC never consulted):")
    for line in trace:
        print(f"  {line}")

    # ---- 4. the two SEPARATE stopping decisions -------------------------------------------
    repaired = [r for r in rows if r["stage"] in ("grid", "tiebreak")
                and r.get("learning_rate") == winner["learning_rate"]
                and r.get("epochs") == winner["epochs"]]
    completion = (sum(1 for r in repaired if r["health"] == "healthy") / max(len(repaired), 1))
    soft = {
        "winner": {"learning_rate": winner["learning_rate"], "epochs": winner["epochs"]},
        "completion_rate": completion, "required": COMPLETION_MIN,
        "n_fits": len(repaired),
        "repair_succeeded": bool(completion >= COMPLETION_MIN),
        "verdict": ("soft DEEM repaired — run its predefined evaluation, regardless of how hard "
                    "DEEM performs" if completion >= COMPLETION_MIN else
                    "abandon soft DEEM — repair did not reach 90% healthy completion"),
    }
    hard = hard_deem_decision()

    write_csv(os.path.join(args.out_dir, "per_fit.csv"), rows)
    write_csv(os.path.join(args.out_dir, "grid.csv"), grid)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "status": "SECONDARY diagnostic, not a registered arm",
            "spec": "SPEC_SOLVER_MECHANISM_STUDY.md §5",
            "sweep_gate": why,
            "definitions": {"collapsed_sd": COLLAPSED_SD, "healthy_sd": HEALTHY_SD,
                            "completion_min": COMPLETION_MIN,
                            "meaningful_gain_pp": MEANINGFUL_GAIN_PP,
                            "primary_comparison": "ensemble AUROC (H3 is registered on "
                                                  "full.deem_deep_soft_ensemble); mean per-seed "
                                                  "AUROC reported alongside, never substituted"},
            "pilot_cells": pilot, "grid": {"learning_rate": list(GRID_LR),
                                           "epochs": list(GRID_EPOCHS)},
            "tiebreaker_trace": trace,
            "decision_soft_deem": soft,
            "decision_hard_deem": hard,
            "decisions_are_independent": "hard-DEEM performance does not veto a repaired soft "
                                         "DEEM; they are different methods",
            "base_config": asdict(base),
            "runtime_seconds": time.time() - started,
        }, handle, indent=2, sort_keys=True, default=str)

    print(f"\nDECISION 1 (soft DEEM): {soft['verdict']}")
    if hard.get("available"):
        for arm, info in hard["arms"].items():
            print(f"DECISION 2 ({arm}): {info.get('verdict')}")
    print(f"\nwrote {args.out_dir}  ({time.time() - started:.1f}s)")


def write_csv(path, rows):
    if not rows:
        return
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen and k != "score":
                seen.add(k); fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
