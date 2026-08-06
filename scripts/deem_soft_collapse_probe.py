#!/usr/bin/env python3
"""SECONDARY diagnostic — Study C of ``SPEC_SOLVER_MECHANISM_STUDY.md``.

WHY THIS EXISTS
---------------
`deem_deep_soft` failed on 27 of 30 attempts in the registered sweep, every one with the same
`ValueError: method returned a non-finite or constant score`.  That exception is raised by
`orient_score` *after* `fit_deem_score` has already returned, so the fit completes and
`save_method_record`'s `except` branch discards the whole `DeemRunResult` — including
`model.history_`, which is exactly what a diagnosis needs.

This probe mirrors the pinned adapter's constructor and injects a DEEM trainer callback, keeping
everything the runner throws away.  A two-epoch gate proves that the instrumented path produces
the hashed adapter's exact score before the diagnostic begins.  It never edits the hashed adapter
or the hashed runner.

TIMING: the reviewer required this to run only AFTER the registered sweep exits — a concurrent
fit competes for CPU threads and memory bandwidth and can affect the registered run's timing and
stability.  The gate is asserted at startup and cannot be bypassed.

STOPPING DECISIONS ARE SEPARATE (review item): hard categorical DEEM and repaired soft DEEM are
different methods, so poor hard performance must not veto a successfully repaired soft model.

Usage:
    python scripts/deem_soft_collapse_probe.py --data-dir local_cache
"""

import argparse
import csv
from importlib import metadata
import json
import os
import random
import sys
import time
from dataclasses import asdict, replace

import numpy as np
from scipy.stats import spearmanr

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (REPO, os.path.join(REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inscope_cells import INSCOPE                                          # noqa: E402
from spectral_utils.deem_adapter import (                                  # noqa: E402
    DEEM_PINNED_VERSION, _aligned_probabilities, _jsonable,
    continuous_to_deem_hard, continuous_to_deem_soft, fit_deem_score,
)
from run_dependency_fusion_experiment import (                             # noqa: E402
    DEEM_BASE, dataset_family, derive_oriented_matrix, evaluate_score, load_cells, orient_score,
)
from run_step227_studies import sweep_done                                  # noqa: E402

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


def health(sd, finite_objective=True):
    """The three-way classification, calibrated so the sweep's three soft 'successes'
    (sigma ~ 1e-8) are collapsed rather than healthy."""
    if not finite_objective:
        return "nonfinite_objective"
    if not np.isfinite(sd) or sd < COLLAPSED_SD:
        return "collapsed"
    if sd < HEALTHY_SD:
        return "degenerate_nonconstant"
    return "healthy"


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


class EpochDiagnostics:
    """DEEM trainer callback that records collapse indicators after every epoch."""

    def __init__(self, predictions):
        self.predictions = np.asarray(predictions)
        self.rows = []
        self.last_finite_epoch = None
        self.last_finite_state = None

    def on_epoch_end(self, epoch, trainer, metrics):
        import torch

        model = trainer.model
        was_training = bool(model.training)
        model.eval()
        try:
            with torch.no_grad():
                x = torch.as_tensor(self.predictions, dtype=torch.float32, device=trainer.device)
                x = model.preprocess(x)
                processed = model.apply_multinomial_layer(x)
                probs = model.calc_hidden_probs(processed).squeeze(1)
                score = (probs[:, 1]
                         if probs.ndim == 2 and probs.shape[1] > 1 else probs.reshape(-1))
                output_sd = float(score.std(unbiased=False).detach().cpu())
                sparse_zero_fraction = float((processed == 0).float().mean().detach().cpu())
                dead = (processed == 0).all(dim=0)
                sparse_dead_unit_fraction = float(dead.float().mean().detach().cpu())
        finally:
            if was_training:
                model.train()

        param_sq, grad_sq = 0.0, 0.0
        for parameter in model.parameters():
            value = parameter.detach()
            param_sq += float((value * value).sum().cpu())
            if parameter.grad is not None:
                grad = parameter.grad.detach()
                grad_sq += float((grad * grad).sum().cpu())
        row = {
            "epoch": int(epoch), "loss": float(metrics.get("loss", float("nan"))),
            "lr": float(metrics.get("lr", float("nan"))),
            "output_sd": output_sd, "parameter_norm": float(np.sqrt(param_sq)),
            "gradient_norm": float(np.sqrt(grad_sq)),
            "sparsemax_zero_fraction": sparse_zero_fraction,
            "sparsemax_dead_unit_fraction": sparse_dead_unit_fraction,
        }
        finite = all(np.isfinite(v) for k, v in row.items() if k != "epoch")
        row["finite"] = bool(finite)
        self.rows.append(row)
        if finite:
            self.last_finite_epoch = int(epoch)
            self.last_finite_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        return True


def _instrumented_deem_fit(X, seed, config, verbose=False):
    """Mirror the pinned adapter while retaining the fitted model and epoch diagnostics."""
    try:
        package_version = metadata.version("deem")
        from deem import DEEM
        from deem.core.training import RBMTrainer
    except Exception as exc:
        raise RuntimeError(
            'DEEM is unavailable; install with `pip install -e ".[dependency-experiment]"`'
        ) from exc
    if config.strict_version and package_version != DEEM_PINNED_VERSION:
        raise RuntimeError(
            f"DEEM version drift: found {package_version}, expected {DEEM_PINNED_VERSION}"
        )

    X = np.asarray(X, dtype=float)
    predictions = (continuous_to_deem_hard(X) if config.input_mode == "hard"
                   else continuous_to_deem_soft(X))
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = DEEM(
        n_classes=2,
        hidden_dim=int(config.hidden_dim),
        cd_k=int(config.cd_k),
        deterministic=bool(config.deterministic),
        learning_rate=float(config.learning_rate),
        momentum=float(config.momentum),
        epochs=int(config.epochs),
        batch_size=min(int(config.batch_size), len(X)),
        device=config.device,
        auto_hyperparameters=False,
        random_state=seed,
        use_preprocessing=bool(config.use_preprocessing),
        preprocessing_layers=int(config.preprocessing_layers),
        preprocessing_activation=config.preprocessing_activation,
        preprocessing_init=config.preprocessing_init,
        sampler_steps=int(config.sampler_steps),
        sampler_oh_mode=(config.input_mode == "soft"),
        use_weighted=bool(config.use_weighted),
        init_method=config.init_method,
    )

    callback = EpochDiagnostics(predictions)
    original_fit = RBMTrainer.fit

    def fit_with_diagnostics(trainer, *args, **kwargs):
        callbacks = list(kwargs.pop("callbacks", None) or [])
        callbacks.append(callback)
        return original_fit(trainer, *args, callbacks=callbacks, **kwargs)

    RBMTrainer.fit = fit_with_diagnostics
    fit_error = None
    try:
        model.fit(predictions, verbose=bool(verbose))
    except Exception as exc:
        fit_error = exc
    finally:
        RBMTrainer.fit = original_fit

    if fit_error is not None:
        fit_error.epoch_diagnostics = callback
        raise fit_error

    try:
        aligned, mapping = _aligned_probabilities(model, predictions)
    except Exception as exc:
        exc.epoch_diagnostics = callback
        raise
    return {
        "score": aligned[:, 1].copy(),
        "aligned_probabilities": aligned,
        "class_map": mapping,
        "package_version": package_version,
        "history": _jsonable(getattr(model, "history_", {})),
        "epoch_diagnostics": callback,
    }


def _artifact_stem(stage, cell_key, seed, config):
    lr = f"{float(config.learning_rate):.0e}".replace("+", "")
    safe_cell = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cell_key)
    return f"{stage}__{safe_cell}__lr{lr}__ep{int(config.epochs)}__seed{int(seed)}"


def adapter_equivalence_gate(F_a, base_config):
    """Prove the instrumented constructor produces the hashed adapter's exact score."""
    gate_config = replace(base_config, epochs=2)
    instrumented = _instrumented_deem_fit(F_a.T, 0, gate_config, verbose=False)["score"]
    canonical = fit_deem_score(F_a.T, seed=0, config=gate_config, verbose=False).score
    error = float(np.max(np.abs(np.asarray(instrumented) - np.asarray(canonical))))
    if error > 1e-10:
        raise SystemExit(
            f"DEEM adapter-equivalence gate failed: max score difference {error:.3e}"
        )
    return error


def one_fit(F_a, cell_key, stage, seed, config, artifact_dir):
    """Fit directly and persist the evidence needed to diagnose a collapsed output."""
    started = time.time()
    rec = {"seed": int(seed), "learning_rate": config.learning_rate,
           "epochs": config.epochs, "input_mode": config.input_mode,
           "cell": cell_key, "family": dataset_family(cell_key), "stage": stage}
    stem = _artifact_stem(stage, cell_key, seed, config)
    os.makedirs(artifact_dir, exist_ok=True)
    diagnostics = None
    try:
        res = _instrumented_deem_fit(F_a.T, seed=seed, config=config, verbose=False)
        diagnostics = res["epoch_diagnostics"]
        score = np.asarray(res["score"], dtype=float)
        sd = float(np.std(score))
        hist = res["history"] or {}
        loss = np.asarray(hist.get("loss", []), dtype=float)
        finite_history = bool(len(loss) == int(config.epochs) and np.isfinite(loss).all())
        array_path = os.path.join(artifact_dir, stem + ".npz")
        np.savez_compressed(array_path, score=score,
                            aligned_probabilities=np.asarray(res["aligned_probabilities"]))
        diag_path = os.path.join(artifact_dir, stem + ".json")
        with open(diag_path, "w", encoding="utf-8") as handle:
            json.dump({"history": hist, "epochs": diagnostics.rows}, handle,
                      indent=2, sort_keys=True, default=str)
        rec.update({
            "status": "fit_returned", "score_sd": sd,
            "health": health(sd, finite_history),
            "score_n_unique": int(len(np.unique(score))),
            "score_min": float(score.min()), "score_max": float(score.max()),
            "finite_history": bool(finite_history),
            "history_keys": "|".join(sorted(hist)),
            "class_map": json.dumps(res["class_map"]),
            "n_epochs_recorded": int(max((len(v) for v in hist.values()
                                          if isinstance(v, (list, tuple))), default=0)),
            "score": score, "array_artifact": os.path.relpath(array_path, REPO),
            "diagnostic_artifact": os.path.relpath(diag_path, REPO),
            "last_finite_epoch": diagnostics.last_finite_epoch,
        })
        if diagnostics.rows:
            for key in ("output_sd", "parameter_norm", "gradient_norm",
                        "sparsemax_zero_fraction", "sparsemax_dead_unit_fraction"):
                vals = np.asarray([row[key] for row in diagnostics.rows], dtype=float)
                rec[f"epoch_{key}_first"] = float(vals[0])
                rec[f"epoch_{key}_last"] = float(vals[-1])
                rec[f"epoch_{key}_min"] = float(np.nanmin(vals))
                rec[f"epoch_{key}_max"] = float(np.nanmax(vals))
        for key, val in hist.items():
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], (int, float)):
                arr = np.asarray(val, dtype=float)
                rec[f"history_{key}_first"] = float(arr[0])
                rec[f"history_{key}_last"] = float(arr[-1])
                rec[f"history_{key}_finite"] = bool(np.all(np.isfinite(arr)))
    except Exception as exc:
        diagnostics = getattr(exc, "epoch_diagnostics", diagnostics)
        rec.update({"status": "fit_raised", "health": "fit_raised",
                    "error_type": type(exc).__name__, "error": str(exc)[:300],
                    "score_sd": float("nan"), "score": None})
    if (diagnostics is not None and diagnostics.last_finite_state is not None
            and rec.get("health") != "healthy"):
        try:
            import torch
            checkpoint_path = os.path.join(artifact_dir, stem + "__last_finite.pt")
            torch.save({"epoch": diagnostics.last_finite_epoch,
                        "model_state_dict": diagnostics.last_finite_state,
                        "config": asdict(config)}, checkpoint_path)
            rec["last_finite_checkpoint"] = os.path.relpath(checkpoint_path, REPO)
            rec["last_finite_epoch"] = diagnostics.last_finite_epoch
        except Exception as exc:
            rec["checkpoint_error"] = f"{type(exc).__name__}: {exc}"[:300]
    rec["runtime_seconds"] = time.time() - started
    return rec


def cross_seed_stability(records):
    """Median within-cell, cross-seed |Spearman|; never compare different datasets."""
    vals = []
    for cell_key in sorted({r["cell"] for r in records}):
        cell_records = sorted(
            (r for r in records if r["cell"] == cell_key and r.get("score") is not None
             and np.std(r["score"]) > 1e-12),
            key=lambda r: int(r["seed"]),
        )
        for i in range(len(cell_records)):
            for j in range(i + 1, len(cell_records)):
                a, b = cell_records[i]["score"], cell_records[j]["score"]
                if len(a) == len(b):
                    vals.append(abs(float(spearmanr(a, b).statistic)))
    return float(np.nanmedian(vals)) if vals else float("nan")


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
        fams_arr = np.asarray(fams)
        family_delta = np.array([
            float(np.mean(d[fams_arr == family])) for family in fam_list
        ])
        rng = np.random.default_rng(20260805)
        boot = [family_delta[rng.integers(0, len(fam_list), len(fam_list))].mean()
                for _ in range(10000)]
        hi = float(np.percentile(boot, 97.5))
        no_advantage = bool(family_delta.mean() <= 0 and hi < MEANINGFUL_GAIN_PP)
        out["arms"][arm] = {
            "n_cells": len(common), "n_families": len(fam_list),
            "cell_mean_delta_pp": float(d.mean()),
            "family_macro_delta_pp": float(family_delta.mean()),
            "family_blocked_ci_hi_pp": hi,
            "shows_no_meaningful_advantage_over_iu": no_advantage,
            "verdict": ("abandon hard DEEM — no meaningful advantage over IU (this rules out the "
                        "preregistered +1.0pp gain; it does NOT prove inferiority)"
                        if no_advantage else "hard DEEM retains a possible meaningful gain"),
        }
    return out


def family_delta_summary(reference, candidate, *, seed=20260805):
    """Equal-family paired comparison used by the repaired-soft evaluation."""
    common = sorted(set(reference) & set(candidate))
    if not common:
        return {"n_cells": 0, "n_families": 0, "available": False}
    cell_delta = np.array([100.0 * (candidate[c] - reference[c]) for c in common])
    families = np.array([dataset_family(c) for c in common])
    fam_list = sorted(set(families.tolist()))
    family_delta = np.array([
        float(np.mean(cell_delta[families == family])) for family in fam_list
    ])
    rng = np.random.default_rng(seed)
    boot = np.empty(10000)
    for i in range(len(boot)):
        boot[i] = family_delta[rng.integers(0, len(family_delta), len(family_delta))].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "available": True, "n_cells": len(common), "n_families": len(fam_list),
        "cells": common, "cell_mean_delta_pp": float(cell_delta.mean()),
        "family_macro_delta_pp": float(family_delta.mean()),
        "family_bootstrap_ci95_low_pp": float(lo),
        "family_bootstrap_ci95_high_pp": float(hi),
        "wins": int(np.sum(cell_delta > 0)), "losses": int(np.sum(cell_delta < 0)),
    }


def iu_reference_scores():
    path = os.path.join(STUDY, "per_cell.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("arm") == "full.iu_pcr":
                out[row["cell"]] = float(row["auc"])
    return out


def evaluate_repaired_soft(cells, config, out_dir, artifact_dir):
    """Run the frozen, label-free-selected configuration on all cells and five seeds."""
    print("\nfull repaired-soft evaluation (all cells x five seeds):", flush=True)
    seed_rows, cell_rows, ensemble_scores = [], [], {}
    for ck in [c for c in INSCOPE if c in cells]:
        F, _, _ = derive_oriented_matrix(cells[ck])
        oriented_scores = []
        for seed in SEEDS:
            rec = one_fit(F, ck, "evaluation", seed, config, artifact_dir)
            if rec.get("health") == "healthy" and rec.get("score") is not None:
                oriented, flipped = orient_score(rec["score"], cells[ck]["anchor"])
                rec["auc"] = evaluate_score(cells[ck]["labels"], oriented)
                rec["anchor_flipped"] = int(flipped)
                rec["oriented_score"] = oriented
                oriented_scores.append(oriented)
            seed_rows.append(rec)
            print(f"  {ck:28s} seed={seed} {rec['health']}", flush=True)
        healthy = [r for r in seed_rows if r["cell"] == ck and r["health"] == "healthy"]
        row = {"cell": ck, "family": dataset_family(ck), "n_seeds": len(SEEDS),
               "n_healthy_seeds": len(healthy),
               "seed_completion_rate": len(healthy) / len(SEEDS)}
        if len(oriented_scores) == len(SEEDS):
            ensemble = np.mean(oriented_scores, axis=0)
            row["ensemble_auc"] = evaluate_score(cells[ck]["labels"], ensemble)
            row["mean_seed_auc"] = float(np.mean([r["auc"] for r in healthy]))
            row["seed_auc_sd"] = float(np.std([r["auc"] for r in healthy]))
            ensemble_scores[ck] = row["ensemble_auc"]
        else:
            row.update({"ensemble_auc": float("nan"), "mean_seed_auc": float("nan"),
                        "seed_auc_sd": float("nan")})
        cell_rows.append(row)
        write_csv(os.path.join(out_dir, "evaluation_seeds.csv"), seed_rows)
        write_csv(os.path.join(out_dir, "evaluation_per_cell.csv"), cell_rows)

    seed_completion = (sum(r["health"] == "healthy" for r in seed_rows)
                       / max(len(seed_rows), 1))
    ensemble_completion = len(ensemble_scores) / max(len(cell_rows), 1)
    comparison = family_delta_summary(iu_reference_scores(), ensemble_scores)
    return {
        "seed_completion_rate": float(seed_completion),
        "ensemble_cell_completion_rate": float(ensemble_completion),
        "n_seed_fits": len(seed_rows), "n_ensemble_cells": len(ensemble_scores),
        "completion_required": COMPLETION_MIN,
        "evaluation_succeeded": bool(seed_completion >= COMPLETION_MIN
                                     and ensemble_completion >= COMPLETION_MIN),
        "ensemble_vs_iu": comparison,
    }, seed_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(REPO, "local_cache"))
    parser.add_argument("--out-dir", default=OUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sweep-pid", type=int, default=None,
                        help="PID of the registered sweep; a live PID always blocks this probe")
    args = parser.parse_args()

    finished, why = sweep_done(args.sweep_pid)
    print(f"sweep gate: {'PASS' if finished else 'BLOCKED'} — {why}")
    if not finished:
        raise SystemExit("refusing to run while the registered sweep may still be live")

    started = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    artifact_dir = os.path.join(args.out_dir, "artifacts")
    cells = load_cells(os.path.abspath(args.data_dir))

    bad = failing_cells()
    candidates = [c for c in INSCOPE if c in cells and c in bad]
    if not candidates:
        print("no deem_deep_soft failures in the checkpoint — nothing to repair")
        candidates = [c for c in INSCOPE if c in cells]
    # three failing cells with smallest n, ties alphabetical — deterministic and label-free
    pilot = sorted(candidates, key=lambda c: (len(cells[c]["labels"]), c))[:N_PILOT_CELLS]
    print("pilot cells (smallest n among failing): "
          + ", ".join(f"{c} (n={len(cells[c]['labels'])})" for c in pilot))

    base = replace(DEEM_BASE, input_mode="soft", use_preprocessing=True, device=args.device)
    rows = []

    gate_F, _, _ = derive_oriented_matrix(cells[pilot[0]])
    adapter_error = adapter_equivalence_gate(gate_F, base)
    print(f"DEEM adapter-equivalence gate: max |score difference|={adapter_error:.3e}")

    # ---- 1. reproduce the failure and keep what the runner discarded ---------------------
    print("\nreproducing the registered configuration, retaining history:")
    for ck in pilot:
        F, _, _ = derive_oriented_matrix(cells[ck])
        for seed in SEEDS:
            rec = one_fit(F, ck, "reproduce", seed, base, artifact_dir)
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
                rec = one_fit(F, ck, "grid", 0, cfg, artifact_dir)
                rows.append(rec)
                recs.append(rec)
            healthy = [r for r in recs if r["health"] == "healthy"]
            score_sds = np.asarray([r["score_sd"] for r in recs], dtype=float)
            finite_sds = score_sds[np.isfinite(score_sds)]
            grid.append({
                "learning_rate": lr, "epochs": ep, "n_cells": len(recs),
                "completion_rate": len(healthy) / max(len(recs), 1),
                "all_finite_history": int(all(r.get("finite_history", False) for r in recs)),
                "cross_seed_stability": float("nan"),          # filled for the winner only
                "deviation_from_default": abs(np.log10(lr / DEFAULT_LR))
                                          + abs(np.log10(ep / DEFAULT_EPOCHS)),
                "median_score_sd": (float(np.median(finite_sds))
                                    if finite_sds.size else float("nan")),
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
    pool = [g for g in pool if g["all_finite_history"]]
    trace.append(f"step 2 finite objective required: {len(pool)} configs remain")
    winner = None
    if pool:
        # Step 3 always runs all remaining configurations on seeds 1-4.  Even a unique
        # seed-0 winner must satisfy the promised five-seed completion gate.
        for g in pool:
            cfg = replace(base, learning_rate=float(g["learning_rate"]),
                          epochs=int(g["epochs"]))
            recs = [r for r in rows if r["stage"] == "grid" and r["seed"] == 0
                    and r["learning_rate"] == g["learning_rate"]
                    and r["epochs"] == g["epochs"]]
            for ck in pilot:
                F, _, _ = derive_oriented_matrix(cells[ck])
                for seed in SEEDS[1:]:
                    rec = one_fit(F, ck, "tiebreak", seed, cfg, artifact_dir)
                    rows.append(rec)
                    recs.append(rec)
            g["cross_seed_stability"] = cross_seed_stability(recs)
        finite_stability = [g["cross_seed_stability"] for g in pool
                            if np.isfinite(g["cross_seed_stability"])]
        best_stab = max(finite_stability) if finite_stability else float("nan")
        if finite_stability:
            pool = [g for g in pool if g["cross_seed_stability"] >= best_stab - 1e-12]
            trace.append(f"step 3 cross-seed stability == {best_stab:.4f}: {len(pool)} remain")
        else:
            trace.append("step 3 cross-seed stability unavailable: all survivors retained")
        pool = sorted(pool, key=lambda g: (g["deviation_from_default"], g["learning_rate"],
                                           g["epochs"]))
        winner = pool[0]
        trace.append(f"step 4 smallest deviation from the registered default: "
                     f"lr={winner['learning_rate']:g}, epochs={winner['epochs']}")
    else:
        trace.append("step 3/4: no configuration had a finite objective on every pilot cell")
    print("\ntie-breaker trace (AUROC never consulted):")
    for line in trace:
        print(f"  {line}")

    # ---- 4. the two SEPARATE stopping decisions -------------------------------------------
    repaired = ([r for r in rows if r["stage"] in ("grid", "tiebreak")
                 and r.get("learning_rate") == winner["learning_rate"]
                 and r.get("epochs") == winner["epochs"]] if winner else [])
    completion = (sum(1 for r in repaired if r["health"] == "healthy") / max(len(repaired), 1))
    soft = {
        "winner": ({"learning_rate": winner["learning_rate"], "epochs": winner["epochs"]}
                   if winner else None),
        "completion_rate": completion, "required": COMPLETION_MIN,
        "n_fits": len(repaired),
        "expected_pilot_fits": len(pilot) * len(SEEDS),
        "repair_succeeded": bool(winner and len(repaired) == len(pilot) * len(SEEDS)
                                 and completion >= COMPLETION_MIN),
        "verdict": ("soft DEEM repaired — run its predefined evaluation, regardless of how hard "
                    "DEEM performs" if winner and len(repaired) == len(pilot) * len(SEEDS)
                    and completion >= COMPLETION_MIN else
                    "abandon soft DEEM — repair did not reach 90% healthy completion"),
    }
    hard = hard_deem_decision()

    full_evaluation = {"run": False, "reason": "pilot repair did not pass"}
    if soft["repair_succeeded"]:
        winner_config = replace(base, learning_rate=float(winner["learning_rate"]),
                                epochs=int(winner["epochs"]))
        full_evaluation, evaluation_rows = evaluate_repaired_soft(
            cells, winner_config, args.out_dir, artifact_dir,
        )
        full_evaluation["run"] = True
        rows.extend(evaluation_rows)
        if not full_evaluation["evaluation_succeeded"]:
            soft["verdict"] = ("abandon soft DEEM — pilot passed but full evaluation did not "
                               "reach 90% seed and ensemble-cell completion")
        else:
            soft["verdict"] = "soft DEEM fit reliably; interpret the frozen full evaluation"

    write_csv(os.path.join(args.out_dir, "per_fit.csv"), rows)
    write_csv(os.path.join(args.out_dir, "grid.csv"), grid)
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "status": "SECONDARY diagnostic, not a registered arm",
            "spec": "SPEC_SOLVER_MECHANISM_STUDY.md §5",
            "sweep_gate": why,
            "adapter_equivalence_gate_max_abs_error": adapter_error,
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
            "full_soft_evaluation": full_evaluation,
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
            if k not in seen and k not in ("score", "oriented_score"):
                seen.add(k); fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
