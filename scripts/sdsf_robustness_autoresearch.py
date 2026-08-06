#!/usr/bin/env python3
"""Disjoint synthetic research loop for stabilized SDSF.

This is an autoresearch-style loop with scientific guardrails: the benchmark,
candidate order, utility function, development seeds, sealed-validation seeds,
and promotion rule are fixed in this file.  It never invokes git and never
changes its own benchmark.  Correctness labels are used only after a candidate
has produced a frozen score.

Usage
-----
    python3 scripts/sdsf_robustness_autoresearch.py
    python3 scripts/sdsf_robustness_autoresearch.py --quick
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import types

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Loading a narrow numerical experiment must not require the repository's
# optional model-serving dependencies (notably torch).  A namespace package is
# enough for the relative imports used by the spectral modules.
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.dependency_fusion import (                        # noqa: E402
    regularized_covariance_weights,
    sparse_upcr_fit,
)
from spectral_utils.robust_dependency_fusion import (                 # noqa: E402
    bootstrap_reliability,
    diagonal_shrinkage,
    stability_shrunk_weights,
)


VERSION = "sdsf-robustness-v3-2026-08-05"
DEFAULT_OUT = os.path.join(REPO, "results", "sdsf_robustness_v3")
N_TEST = 5000
N_BOOTSTRAP_CI = 5000

SPARSE_FIT = dict(
    scale_ratio=0.25,
    rank=2,
    n_components=2,
    g2_projection_components=1,
    threshold_multiplier=1.0,
    max_iter=100,
    inner_completion_iter=40,
    decomposition_tol=1e-8,
    max_sparse_fraction=None,
    target_condition=100.0,
)

# Ordered before the development run.  Order is retained in the ledger so the
# running-best curve is an honest record of the search, including regressions.
CANDIDATES = (
    {"name": "sdsf_cond100", "kind": "sdsf", "condition": 100.0},
    {"name": "sdsf_cond50", "kind": "sdsf", "condition": 50.0},
    {"name": "sdsf_cond20", "kind": "sdsf", "condition": 20.0},
    {"name": "diag25_cond50", "kind": "stable", "tau": 0.0,
     "covariance_shrinkage": 0.25, "condition": 50.0},
    {"name": "diag50_cond50", "kind": "stable", "tau": 0.0,
     "covariance_shrinkage": 0.50, "condition": 50.0},
    {"name": "rho_boot_tau0.5", "kind": "stable", "tau": 0.5,
     "covariance_shrinkage": 0.0, "condition": 50.0},
    {"name": "rho_boot_tau1", "kind": "stable", "tau": 1.0,
     "covariance_shrinkage": 0.0, "condition": 50.0},
    {"name": "rho_boot_tau2", "kind": "stable", "tau": 2.0,
     "covariance_shrinkage": 0.0, "condition": 50.0},
    {"name": "joint_tau1_diag25", "kind": "stable", "tau": 1.0,
     "covariance_shrinkage": 0.25, "condition": 50.0},
    {"name": "blend_half_joint_su", "kind": "blend", "source": "joint_tau1_diag25",
     "alpha": 0.5},
)

WORLDS = {
    "clean_gaussian": {"n_train": 1000, "dependency": "clean", "sampling": "gaussian"},
    "sparse_small": {"n_train": 350, "dependency": "sparse", "sampling": "gaussian"},
    "sparse_gaussian": {"n_train": 3000, "dependency": "sparse", "sampling": "gaussian"},
    "sparse_heavy_t4": {"n_train": 3000, "dependency": "sparse", "sampling": "t4"},
    "sparse_contaminated": {"n_train": 3000, "dependency": "sparse",
                            "sampling": "contaminated"},
    "dense_blocks": {"n_train": 3000, "dependency": "dense", "sampling": "gaussian"},
}

PRIMARY_WORLDS = ("sparse_gaussian", "sparse_heavy_t4", "sparse_contaminated")
STRESS_WORLDS = ("sparse_small", "dense_blocks")


def stable_seed(*parts):
    payload = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2 ** 32)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def population(dependency, m=14):
    """Tenzer additive signal plus controlled feature-error dependence."""
    g2 = 0.18
    a = np.linspace(-0.02, 0.02, m)
    rho = g2 + a
    C = g2 * np.ones((m, m)) + a[:, None] + a[None, :]
    np.fill_diagonal(C, 1.0)
    S = np.zeros_like(C)
    if dependency == "sparse":
        for i, j, value in ((0, 7, 0.50), (2, 10, -0.50),
                            (4, 12, 0.50), (5, 13, -0.50)):
            S[i, j] = S[j, i] = value
    elif dependency == "dense":
        for block, value in ((range(7), 0.12), (range(7, 14), -0.10)):
            block = list(block)
            for left, i in enumerate(block):
                for j in block[left + 1:]:
                    S[i, j] = S[j, i] = value
    elif dependency != "clean":
        raise ValueError(dependency)
    C = C + S
    joint = np.block([[C, rho[:, None]], [rho[None, :], np.ones((1, 1))]])
    if np.linalg.eigvalsh(joint).min() <= 1e-10:
        raise RuntimeError("invalid planted covariance")
    return joint


def draw_world(world, repetition, phase):
    config = WORLDS[world]
    rng = np.random.default_rng(stable_seed(VERSION, phase, world, repetition))
    joint = population(config["dependency"])
    n_train = int(config["n_train"])
    total = n_train + N_TEST
    raw = rng.multivariate_normal(np.zeros(joint.shape[0]), joint, size=total)
    if config["sampling"] == "t4":
        # One scale per row preserves the elliptical correlation structure but
        # creates finite-variance, infinite-fourth-moment observations.
        scale = np.sqrt(rng.chisquare(4, size=total) / 4.0)
        raw = raw / scale[:, None]
    train, test = raw[:n_train].copy(), raw[n_train:].copy()
    if config["sampling"] == "contaminated":
        # Training-only cell contamination: 2% rows, four random feature views,
        # six population SDs.  The test population stays clean.
        count = max(1, int(round(0.02 * n_train)))
        rows = rng.choice(n_train, size=count, replace=False)
        for row in rows:
            cols = rng.choice(14, size=4, replace=False)
            train[row, cols] += rng.choice([-6.0, 6.0], size=4)
    center = train[:, :-1].mean(axis=0)
    scale = train[:, :-1].std(axis=0)
    X_train = (train[:, :-1] - center) / scale
    X_test = (test[:, :-1] - center) / scale
    return X_train, X_test, (test[:, -1] > 0.0).astype(int)


def orient(weight, X, anchor):
    weight = np.asarray(weight, dtype=float)
    corr = np.corrcoef(X @ weight, anchor)[0, 1]
    if not np.isfinite(corr):
        raise ValueError("constant candidate score")
    return -weight if corr < 0 else weight


def candidate_weights(candidate, fit, boot, cache):
    kind = candidate["kind"]
    if kind == "sdsf":
        weight, _ = regularized_covariance_weights(
            fit.decomposition.structured_cov, fit.rho_hat,
            target_condition=candidate["condition"],
        )
        return weight, {}
    if kind == "stable":
        weight, diag = stability_shrunk_weights(
            fit, boot, tau=candidate["tau"], preserve_components=2,
            covariance_shrinkage=candidate["covariance_shrinkage"],
            target_condition=candidate["condition"],
        )
        return weight, diag
    if kind == "blend":
        source = cache[candidate["source"]]
        alpha = float(candidate["alpha"])
        return alpha * source + (1.0 - alpha) * fit.w_pcr, {"alpha": alpha}
    raise ValueError(kind)


def run_repetition(world, repetition, phase, n_boot, score_names=None):
    X_train, X_test, labels = draw_world(world, repetition, phase)
    F = X_train.T
    fit = sparse_upcr_fit(F, **SPARSE_FIT)
    boot = bootstrap_reliability(
        F, fit, n_boot=n_boot,
        seed=stable_seed(VERSION, phase, world, repetition, "bootstrap"),
        fit_kwargs=SPARSE_FIT,
    )
    anchor = X_train[:, 0]
    su = orient(fit.w_pcr, X_train, anchor)
    auc_su = roc_auc_score(labels, X_test @ su)
    cache = {}
    row = {
        "phase": phase, "world": world, "repetition": repetition,
        "auc_su_pcr": float(auc_su),
        "sparse_fraction": fit.decomposition.sparse_fraction,
        "bootstrap_success": boot.n_successful / boot.n_requested,
    }
    score_names = None if score_names is None else set(score_names)
    for candidate in CANDIDATES:
        weight, diag = candidate_weights(candidate, fit, boot, cache)
        cache[candidate["name"]] = weight
        if score_names is not None and candidate["name"] not in score_names:
            continue
        weight = orient(weight, X_train, anchor)
        auc = float(roc_auc_score(labels, X_test @ weight))
        row[f"auc_{candidate['name']}"] = auc
        row[f"delta_{candidate['name']}"] = auc - auc_su
        if "tail_kappa_mean" in diag:
            row[f"tail_kappa_{candidate['name']}"] = diag["tail_kappa_mean"]
    return row


def bootstrap_ci(values, name):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(VERSION, "ci", name))
    means = np.empty(N_BOOTSTRAP_CI)
    for start in range(0, N_BOOTSTRAP_CI, 500):
        size = min(500, N_BOOTSTRAP_CI - start)
        pick = rng.integers(0, len(values), size=(size, len(values)))
        means[start:start + size] = values[pick].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def summarize_candidate(rows, candidate, phase):
    name = candidate["name"]
    by_world = {}
    for world in WORLDS:
        values = np.array([r[f"delta_{name}"] for r in rows if r["world"] == world])
        lo, hi = bootstrap_ci(values, f"{phase}_{world}_{name}")
        by_world[world] = {
            "mean": float(values.mean()), "p05": float(np.quantile(values, 0.05)),
            "ci_low": lo, "ci_high": hi,
            "wins": int(np.sum(values > 0)), "n": len(values),
        }
    primary = np.array([
        r[f"delta_{name}"] for r in rows if r["world"] in PRIMARY_WORLDS
    ])
    stress = np.array([
        r[f"delta_{name}"] for r in rows if r["world"] in STRESS_WORLDS
    ])
    clean = by_world["clean_gaussian"]["mean"]
    # Utility rewards broad improvement and explicitly prices the lower tail.
    # All terms are AUROC fractions.  It is a ranking criterion, not a claim.
    utility = (
        float(primary.mean()) + 0.25 * float(stress.mean())
        - 0.50 * max(0.0, -float(np.quantile(primary, 0.05)))
        - 0.50 * max(0.0, -clean)
    )
    primary_lo, primary_hi = bootstrap_ci(primary, f"{phase}_primary_{name}")
    return {
        "phase": phase, "candidate": name, "utility": utility,
        "primary_mean": float(primary.mean()), "primary_ci_low": primary_lo,
        "primary_ci_high": primary_hi,
        "primary_p05": float(np.quantile(primary, 0.05)),
        "stress_mean": float(stress.mean()), "clean_mean": clean,
        "worlds": by_world,
    }


def render_report(summary):
    lines = [
        "# SDSF robustness research loop v3", "",
        f"Decision: **{summary['decision']}**.", "",
        "The loop compares every candidate with fixed-orientation SU-PCR. Positive deltas "
        "mean higher held-out AUROC. Development selected candidates by a frozen robust "
        "utility; only those candidates were opened on disjoint validation seeds.", "",
        "## Development ledger", "",
        "| step | candidate | utility | primary mean | primary p05 | clean | promoted |", 
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["development"]:
        lines.append(
            f"| {row['step']} | `{row['candidate']}` | {row['utility']:+.5f} | "
            f"{row['primary_mean']:+.5f} | {row['primary_p05']:+.5f} | "
            f"{row['clean_mean']:+.5f} | {'yes' if row['promoted'] else ''} |"
        )
    lines += ["", "## Sealed validation", "",
              "| candidate | vs SU-PCR [95% CI] | vs current SDSF [95% CI] | primary p05 | clean | decision |",
              "|---|---:|---:|---:|---:|:---:|"]
    for row in summary["validation"]:
        lines.append(
            f"| `{row['candidate']}` | {row['primary_mean']:+.5f} "
            f"[{row['primary_ci_low']:+.5f}, {row['primary_ci_high']:+.5f}] | "
            f"{row['vs_current_sdsf_mean']:+.5f} "
            f"[{row['vs_current_sdsf_ci_low']:+.5f}, "
            f"{row['vs_current_sdsf_ci_high']:+.5f}] | "
            f"{row['primary_p05']:+.5f} | {row['clean_mean']:+.5f} | "
            f"**{'PASS' if row['pass'] else 'FAIL'}** |"
        )
    lines += [
        "", "## Frozen validation gates", "",
        "A candidate passes only if it improves both SU-PCR and the current SDSF, both "
        "paired 95% CI lower bounds are non-negative, its 5th percentile is at least "
        "-2 AUROC points, and its clean-world mean is at least -0.5 points. Dense "
        "dependence and small-sample "
        "results are mandatory stress reports, not promotion gates.", "",
        "## Interpretation boundary", "",
        "Synthetic success establishes only that the stabilization mechanism works in the "
        "declared covariance worlds. It does not establish improvement on hallucination "
        "detection. A failed candidate must not be repaired using validation labels; a new "
        "hypothesis requires a new version and seed namespace.", "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--dev-repeats", type=int, default=12)
    parser.add_argument("--validation-repeats", type=int, default=24)
    parser.add_argument("--n-boot", type=int, default=10)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    dev_repeats = 2 if args.quick else args.dev_repeats
    validation_repeats = 3 if args.quick else args.validation_repeats
    n_boot = 3 if args.quick else args.n_boot
    started = time.time()

    dev_rows = []
    for world in WORLDS:
        for repetition in range(dev_repeats):
            dev_rows.append(run_repetition(world, repetition, "development", n_boot))
            print(f"development {world} {repetition + 1}/{dev_repeats}", flush=True)
    dev_summary = [summarize_candidate(dev_rows, c, "development") for c in CANDIDATES]
    ranked = sorted(dev_summary, key=lambda row: row["utility"], reverse=True)
    promoted_names = {row["candidate"] for row in ranked[:3]}
    running_best = -float("inf")
    ledger = []
    for step, row in enumerate(dev_summary, 1):
        running_best = max(running_best, row["utility"])
        ledger.append({**row, "step": step, "running_best": running_best,
                       "promoted": row["candidate"] in promoted_names})

    validation_rows = []
    # The implementation evaluates all candidates per fit to keep the numerical
    # path identical, but only pre-promoted candidates are summarized or judged.
    for world in WORLDS:
        for repetition in range(validation_repeats):
            validation_rows.append(run_repetition(
                world, repetition, "sealed_validation", n_boot,
                score_names=promoted_names,
            ))
            print(f"validation {world} {repetition + 1}/{validation_repeats}", flush=True)
    candidate_by_name = {c["name"]: c for c in CANDIDATES}
    validation = []
    for name in [r["candidate"] for r in ranked[:3]]:
        row = summarize_candidate(validation_rows, candidate_by_name[name], "sealed_validation")
        versus_current = np.array([
            r[f"auc_{name}"] - r["auc_sdsf_cond100"]
            for r in validation_rows if r["world"] in PRIMARY_WORLDS
        ])
        current_lo, current_hi = bootstrap_ci(
            versus_current, f"sealed_validation_{name}_minus_current_sdsf",
        )
        row["vs_current_sdsf_mean"] = float(versus_current.mean())
        row["vs_current_sdsf_ci_low"] = current_lo
        row["vs_current_sdsf_ci_high"] = current_hi
        row["pass"] = bool(
            name != "sdsf_cond100"
            and row["primary_mean"] > 0.0
            and row["primary_ci_low"] >= 0.0
            and row["primary_p05"] >= -0.020
            and row["clean_mean"] >= -0.005
            and row["vs_current_sdsf_mean"] > 0.0
            and row["vs_current_sdsf_ci_low"] >= 0.0
        )
        validation.append(row)
    passing = [row for row in validation if row["pass"]]
    decision = "ADVANCE_BEST_TO_REAL_DATA" if passing else "STOP_AND_REVISE"
    best = max(passing, key=lambda row: row["utility"])["candidate"] if passing else None
    summary = {
        "version": VERSION, "decision": decision, "best_candidate": best,
        "runtime_seconds": time.time() - started,
        "config": {"worlds": WORLDS, "primary_worlds": PRIMARY_WORLDS,
                   "stress_worlds": STRESS_WORLDS, "dev_repeats": dev_repeats,
                   "validation_repeats": validation_repeats, "n_boot": n_boot,
                   "candidates": CANDIDATES},
        "development": ledger, "validation": validation,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "development_replicates.csv"), dev_rows)
    write_csv(os.path.join(args.out_dir, "validation_replicates.csv"), validation_rows)
    write_csv(os.path.join(args.out_dir, "candidate_ledger.csv"), [
        {k: v for k, v in row.items() if k != "worlds"} for row in ledger
    ])
    write_csv(os.path.join(args.out_dir, "validation_summary.csv"), [
        {k: v for k, v in row.items() if k != "worlds"} for row in validation
    ])
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(render_report(summary))
    print(f"Decision: {decision}; best={best}")
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()
