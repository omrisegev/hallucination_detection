#!/usr/bin/env python3
"""Disjoint research cycle for covariance-aware U-PCR reliability estimation.

The candidate registry, worlds, seed namespaces, promotion rule, and decision
gates are fixed here and in SPEC_DEPENDENCY_AWARE_RHO_V5.md.  Estimators never
receive correctness labels.  A synthetic winner is frozen before an optional
retrospective replay on the tracked derived-feature bundle.
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
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.dependency_aware_rho import (                    # noqa: E402
    CANDIDATE_METHODS,
    estimate_dependency_aware_rho,
)
from spectral_utils.dependency_fusion import sparse_upcr_fit          # noqa: E402
from spectral_utils.feature_contract import (                         # noqa: E402
    LEGACY_FEATURE_SIGNS,
    SCHEMA_VERSION,
    confidence_oriented_matrix,
    consensus_anchor,
)


VERSION = "dependency-aware-rho-v5-2026-08-06"
DEFAULT_OUT = os.path.join(REPO, "results", "dependency_aware_rho_v5")
DEFAULT_BUNDLE = os.path.join(REPO, "results", "dependency_fusion_raw", "cells.npz")
DEFAULT_MANIFEST = os.path.join(
    REPO, "results", "dependency_fusion_raw", "cells_manifest.csv",
)
N_TEST = 4000
N_CI_BOOT = 10000
MOMENT_MAX_CONDITION = 100.0

SPARSE_FIT = dict(
    scale_ratio=0.25, rank=2, n_components=2,
    g2_projection_components=1, threshold_multiplier=1.0,
    max_iter=100, inner_completion_iter=40, decomposition_tol=1e-8,
    max_sparse_fraction=None, target_condition=100.0,
)

# Registry order is itself recorded in the development convergence ledger.
CANDIDATES = tuple(CANDIDATE_METHODS)
WORLDS = {
    "clean_gaussian": {"n_train": 1000, "dependency": "clean", "sampling": "gaussian"},
    "sparse_gaussian": {"n_train": 2500, "dependency": "sparse", "sampling": "gaussian"},
    "sparse_small": {"n_train": 350, "dependency": "sparse", "sampling": "gaussian"},
    "sparse_t5": {"n_train": 1800, "dependency": "sparse", "sampling": "t5"},
    "sparse_mixed_kurtosis": {
        "n_train": 1200, "dependency": "sparse", "sampling": "mixed_kurtosis",
    },
    "sparse_contaminated": {
        "n_train": 1200, "dependency": "sparse", "sampling": "contaminated",
    },
}
PRIMARY_WORLDS = (
    "sparse_gaussian", "sparse_small", "sparse_t5", "sparse_mixed_kurtosis",
)
STRESS_WORLDS = ("sparse_contaminated",)
KNOWN_FAMILIES = (
    "triviaqa", "coqa", "hotpotqa", "sciq", "nq_open", "squad_v2",
    "truthfulqa", "gsm8k", "math500", "gpqa", "webq", "humaneval",
)
REAL_GATES = {
    "mean_delta_pp": 1.0,
    "ci95_low_pp": 0.0,
    "qa_delta_pp": -0.5,
    "math_delta_pp": -0.5,
    "family_macro_delta_pp": 0.0,
}


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
    """Joint feature/latent covariance satisfying the additive U-PCR model."""
    g2 = 0.18
    a = np.linspace(-0.02, 0.02, m)
    rho = g2 + a
    C = g2 * np.ones((m, m)) + a[:, None] + a[None, :]
    np.fill_diagonal(C, 1.0)
    sparse = np.zeros_like(C)
    if dependency == "sparse":
        for i, j, value in (
            (0, 7, 0.50), (2, 10, -0.50), (4, 12, 0.50), (5, 13, -0.50),
        ):
            sparse[i, j] = sparse[j, i] = value
    elif dependency != "clean":
        raise ValueError(dependency)
    C = C + sparse
    joint = np.block([[C, rho[:, None]], [rho[None, :], np.ones((1, 1))]])
    if np.linalg.eigvalsh(joint).min() <= 1e-10:
        raise RuntimeError("invalid planted joint covariance")
    return joint, rho


def _unit_variance_t(rng, df, size):
    return rng.normal(size=size) * np.sqrt((df - 2.0) / rng.chisquare(df, size=size))


def draw_joint(rng, joint, total, sampling):
    if sampling in {"gaussian", "contaminated", "t5"}:
        raw = rng.multivariate_normal(np.zeros(joint.shape[0]), joint, size=total)
        if sampling == "t5":
            raw *= np.sqrt(3.0 / rng.chisquare(5, size=total))[:, None]
        return raw
    if sampling == "mixed_kurtosis":
        d = joint.shape[0]
        sources = rng.normal(size=(total, d))
        sources[:, 0] = _unit_variance_t(rng, 5, total)
        sources[:, 1] = _unit_variance_t(rng, 7, total)
        mixture = rng.normal(size=total)
        rare = rng.random(total) < 0.02
        mixture[rare] = rng.normal(scale=5.0, size=int(np.sum(rare)))
        sources[:, 2] = mixture / np.sqrt(0.98 + 0.02 * 25.0)
        return sources @ np.linalg.cholesky(joint).T
    raise ValueError(sampling)


def draw_world(world, repetition, phase):
    config = WORLDS[world]
    rng = np.random.default_rng(stable_seed(VERSION, phase, world, repetition))
    joint, rho = population(config["dependency"])
    n_train = int(config["n_train"])
    raw = draw_joint(rng, joint, n_train + N_TEST, config["sampling"])
    train, test = raw[:n_train].copy(), raw[n_train:].copy()
    if config["sampling"] == "contaminated":
        # Unequal training-moment precision: fixed views receive rare shocks.
        count = max(1, int(round(0.02 * n_train)))
        rows = rng.choice(n_train, size=count, replace=False)
        cols = np.array([0, 1, 2, 3])
        train[np.ix_(rows, cols)] += rng.choice(
            [-7.0, 7.0], size=(count, len(cols)),
        )
    center = train[:, :-1].mean(axis=0)
    scale = train[:, :-1].std(axis=0)
    if np.any(scale < 1e-10):
        raise RuntimeError("degenerate synthetic training feature")
    X_train = (train[:, :-1] - center) / scale
    X_test = (test[:, :-1] - center) / scale
    labels = (test[:, -1] > 0.0).astype(int)
    # Cov((X-center)/scale, Y) is unchanged by centering and divided by scale.
    return X_train, X_test, labels, rho / scale


def orientation(weight, X, anchor):
    correlation = float(np.corrcoef(X @ weight, anchor)[0, 1])
    if not np.isfinite(correlation):
        raise RuntimeError("constant synthetic score")
    return -np.asarray(weight) if correlation < 0 else np.asarray(weight)


def fit_candidates(F, methods):
    fit = sparse_upcr_fit(F, **SPARSE_FIT)
    results = {}
    for method in methods:
        results[method] = estimate_dependency_aware_rho(
            F, fit.covariance, fit.decomposition.low_rank, fit.var_y,
            method=method, n_components=SPARSE_FIT["n_components"],
            projection_components=SPARSE_FIT["g2_projection_components"],
            moment_max_condition=MOMENT_MAX_CONDITION,
        )
    if "ols" in results:
        rho_error = np.linalg.norm(results["ols"].rho_hat - fit.rho_hat)
        weight_error = np.linalg.norm(results["ols"].w_pcr - fit.w_pcr)
        if rho_error > 1e-8 or weight_error > 1e-8:
            raise RuntimeError(
                f"OLS control does not reproduce SU-PCR: rho={rho_error}, w={weight_error}"
            )
    return fit, results


def run_synthetic_repetition(world, repetition, phase, methods):
    X_train, X_test, labels, rho_true = draw_world(world, repetition, phase)
    _, results = fit_candidates(X_train.T, methods)
    anchor = X_train[:, 0]
    row = {"phase": phase, "world": world, "repetition": repetition}
    for method, result in results.items():
        weight = orientation(result.w_pcr, X_train, anchor)
        row[f"auc_{method}"] = float(roc_auc_score(labels, X_test @ weight))
        row[f"rho_nrmse_{method}"] = float(
            np.linalg.norm(result.rho_hat - rho_true)
            / (np.linalg.norm(rho_true) + 1e-12)
        )
        row[f"rho_cosine_{method}"] = float(
            np.dot(result.rho_hat, rho_true)
            / ((np.linalg.norm(result.rho_hat) * np.linalg.norm(rho_true)) + 1e-12)
        )
        row[f"g2_{method}"] = result.g2_hat
        row[f"pair_rmse_{method}"] = result.pair_residual_rmse
        row[f"moment_condition_{method}"] = result.moment_condition_regularized
    baseline_rmse = row["rho_nrmse_ols"]
    baseline_auc = row["auc_ols"]
    for method in results:
        row[f"auc_delta_{method}"] = row[f"auc_{method}"] - baseline_auc
        row[f"rho_relative_reduction_{method}"] = (
            baseline_rmse - row[f"rho_nrmse_{method}"]
        ) / (baseline_rmse + 1e-12)
    return row


def bootstrap_ci(values, name, n_boot=N_CI_BOOT):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(stable_seed(VERSION, "ci", name))
    stats = np.empty(n_boot)
    for start in range(0, n_boot, 1000):
        size = min(1000, n_boot - start)
        picks = rng.integers(0, len(values), size=(size, len(values)))
        stats[start:start + size] = values[picks].mean(axis=1)
    return tuple(float(x) for x in np.quantile(stats, [0.025, 0.975]))


def summarize_synthetic(rows, method, phase):
    primary = [row for row in rows if row["world"] in PRIMARY_WORLDS]
    rho = np.asarray([row[f"rho_relative_reduction_{method}"] for row in primary])
    auc = np.asarray([row[f"auc_delta_{method}"] for row in primary])
    clean = np.asarray([
        row[f"auc_delta_{method}"] for row in rows if row["world"] == "clean_gaussian"
    ])
    stress = np.asarray([
        row[f"auc_delta_{method}"] for row in rows if row["world"] in STRESS_WORLDS
    ])
    rho_lo, rho_hi = bootstrap_ci(rho, f"{phase}_{method}_rho")
    auc_lo, auc_hi = bootstrap_ci(auc, f"{phase}_{method}_auc")
    utility = (
        float(rho.mean()) + 2.0 * float(auc.mean())
        - 2.0 * max(0.0, -float(clean.mean()) - 0.005)
        - 0.5 * max(0.0, -float(np.quantile(auc, 0.05)) - 0.020)
    )
    worlds = {}
    for world in WORLDS:
        selected = [row for row in rows if row["world"] == world]
        worlds[world] = {
            "rho_relative_reduction": float(np.mean([
                row[f"rho_relative_reduction_{method}"] for row in selected
            ])),
            "auc_delta": float(np.mean([
                row[f"auc_delta_{method}"] for row in selected
            ])),
        }
    return {
        "phase": phase, "candidate": method, "utility": utility,
        "primary_rho_reduction": float(rho.mean()),
        "primary_rho_ci_low": rho_lo, "primary_rho_ci_high": rho_hi,
        "primary_auc_delta": float(auc.mean()),
        "primary_auc_ci_low": auc_lo, "primary_auc_ci_high": auc_hi,
        "primary_auc_p05": float(np.quantile(auc, 0.05)),
        "clean_auc_delta": float(clean.mean()),
        "stress_auc_delta": float(stress.mean()),
        "worlds": worlds,
    }


def synthetic_pass(summary):
    return bool(
        summary["candidate"] != "ols"
        and summary["primary_rho_reduction"] > 0.0
        and summary["primary_rho_ci_low"] >= 0.0
        and summary["primary_auc_delta"] > 0.0
        and summary["primary_auc_ci_low"] >= -0.001
        and summary["clean_auc_delta"] >= -0.005
        and summary["primary_auc_p05"] >= -0.020
    )


def render_convergence_svg(path, ledger):
    """Dependency-free static visualization of the honest development trace."""
    width, height = 840, 430
    left, right, top, bottom = 82, 34, 42, 78
    values = [row["utility"] for row in ledger] + [row["running_best"] for row in ledger]
    lo, hi = min(values), max(values)
    padding = max((hi - lo) * 0.12, 0.01)
    lo, hi = lo - padding, hi + padding

    def x(index):
        return left + index * (width - left - right) / max(len(ledger) - 1, 1)

    def y(value):
        return top + (hi - value) * (height - top - bottom) / (hi - lo)

    ticks = np.linspace(lo, hi, 6)
    candidate_points = " ".join(
        f"{x(i):.1f},{y(row['utility']):.1f}" for i, row in enumerate(ledger)
    )
    best_points = " ".join(
        f"{x(i):.1f},{y(row['running_best']):.1f}" for i, row in enumerate(ledger)
    )
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="430" viewBox="0 0 840 430">',
        '<rect width="840" height="430" fill="white"/>',
        '<text x="420" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">v5 development convergence</text>',
        f'<rect x="{left}" y="{top}" width="{width-left-right}" height="{height-top-bottom}" fill="none" stroke="#777"/>',
    ]
    for tick in ticks:
        yy = y(float(tick))
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{left-9}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{tick:+.3f}</text>')
    lines.extend([
        f'<polyline points="{candidate_points}" fill="none" stroke="#377eb8" stroke-width="2"/>',
        f'<polyline points="{best_points}" fill="none" stroke="#e41a1c" stroke-width="3"/>',
    ])
    for i, row in enumerate(ledger):
        xx = x(i)
        lines.append(f'<circle cx="{xx:.1f}" cy="{y(row["utility"]):.1f}" r="4" fill="#377eb8"/>')
        lines.append(f'<text x="{xx:.1f}" y="{height-bottom+18}" text-anchor="end" transform="rotate(-28 {xx:.1f} {height-bottom+18})" font-family="sans-serif" font-size="11">{row["candidate"]}</text>')
    lines.extend([
        f'<text x="{(left+width-right)/2:.1f}" y="419" text-anchor="middle" font-family="sans-serif" font-size="12">fixed candidate order</text>',
        f'<text x="18" y="{(top+height-bottom)/2:.1f}" text-anchor="middle" transform="rotate(-90 18 {(top+height-bottom)/2:.1f})" font-family="sans-serif" font-size="12">frozen development utility</text>',
        '<line x1="590" y1="25" x2="620" y2="25" stroke="#377eb8" stroke-width="2"/><text x="626" y="29" font-family="sans-serif" font-size="11">candidate</text>',
        '<line x1="700" y1="25" x2="730" y2="25" stroke="#e41a1c" stroke-width="3"/><text x="736" y="29" font-family="sans-serif" font-size="11">running best</text>',
        '</svg>',
    ])
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def family(cell):
    return next((name for name in KNOWN_FAMILIES if name in cell), cell)


def load_real_cells(bundle, manifest_path):
    with open(manifest_path, newline="", encoding="utf-8") as handle:
        manifest = {row["cell"]: row for row in csv.DictReader(handle)}
    data = np.load(bundle, allow_pickle=True)
    keys = sorted({name.rsplit("__", 1)[0] for name in data.files})
    cells = {}
    for key in keys:
        names = [str(name) for name in data[f"{key}__pool"]]
        historical = np.asarray(data[f"{key}__hand_signs"], dtype=float)
        expected = np.asarray([LEGACY_FEATURE_SIGNS[name] for name in names], dtype=float)
        if not np.array_equal(historical, expected):
            raise RuntimeError(f"{key}: feature reconstruction failed")
        raw = np.asarray(data[f"{key}__V"], dtype=float) * historical
        matrix, kept, _ = confidence_oriented_matrix(raw, names, stable=True)
        cells[key] = {
            "matrix": matrix, "kept": kept,
            "labels": np.asarray(data[f"{key}__labels"], dtype=int),
            "domain": manifest[key]["domain"], "family": family(key),
        }
    return cells


def oriented_auc(weight, matrix, labels, anchor):
    score = matrix @ np.asarray(weight, dtype=float)
    correlation = float(np.corrcoef(score, anchor)[0, 1])
    if not np.isfinite(correlation):
        raise RuntimeError("constant real score")
    if correlation < 0:
        score = -score
    return float(roc_auc_score(labels, score))


def real_contrast(rows, candidate):
    delta = np.asarray([row["auc_candidate"] - row["auc_ols"] for row in rows])
    lo, hi = bootstrap_ci(delta, f"real_{candidate}_minus_ols", n_boot=20000)
    domains = {
        domain: float(np.mean([
            row["auc_candidate"] - row["auc_ols"]
            for row in rows if row["domain"] == domain
        ])) for domain in ("QA", "math")
    }
    families = sorted({row["family"] for row in rows})
    family_delta = np.asarray([
        np.mean([
            row["auc_candidate"] - row["auc_ols"]
            for row in rows if row["family"] == fam
        ]) for fam in families
    ])
    return {
        "candidate": candidate,
        "mean_delta_pp": float(100 * delta.mean()),
        "ci95_low_pp": float(100 * lo), "ci95_high_pp": float(100 * hi),
        "median_delta_pp": float(100 * np.median(delta)),
        "qa_delta_pp": float(100 * domains["QA"]),
        "math_delta_pp": float(100 * domains["math"]),
        "family_macro_delta_pp": float(100 * family_delta.mean()),
        "wins": int(np.sum(delta > 0)), "losses": int(np.sum(delta < 0)),
        "worst_cell_delta_pp": float(100 * np.min(delta)),
    }


def run_real_replay(candidate, bundle, manifest_path):
    rows = []
    cells = load_real_cells(bundle, manifest_path)
    for index, (key, cell) in enumerate(cells.items()):
        F = cell["matrix"].T
        _, results = fit_candidates(F, ("ols", candidate))
        anchor = consensus_anchor(cell["matrix"])
        rows.append({
            "cell": key, "family": cell["family"], "domain": cell["domain"],
            "n": len(cell["labels"]), "m": F.shape[0],
            "auc_ols": oriented_auc(
                results["ols"].w_pcr, cell["matrix"], cell["labels"], anchor,
            ),
            "auc_candidate": oriented_auc(
                results[candidate].w_pcr, cell["matrix"], cell["labels"], anchor,
            ),
            "rho_cosine_to_ols": float(
                np.dot(results[candidate].rho_hat, results["ols"].rho_hat)
                / (np.linalg.norm(results[candidate].rho_hat)
                   * np.linalg.norm(results["ols"].rho_hat) + 1e-12)
            ),
            "candidate_pair_rmse": results[candidate].pair_residual_rmse,
            "candidate_moment_condition": results[candidate].moment_condition_regularized,
        })
        print(f"real replay {index + 1:02d}/{len(cells)} {key}", flush=True)
    contrast = real_contrast(rows, candidate)
    gates = [
        {"gate": key, "observed": contrast[key], "threshold": threshold,
         "pass": bool(contrast[key] >= threshold)}
        for key, threshold in REAL_GATES.items()
    ]
    return rows, contrast, gates, all(gate["pass"] for gate in gates)


def render_report(summary):
    lines = [
        "# Dependency-aware U-PCR reliability cycle v5", "",
        f"Decision: **{summary['decision']}**.", "",
        "This cycle changes only the pair-equation reliability solve. Every arm uses "
        "the same sparse decomposition, g2 interval, and two-component PCR final solver.",
        "", "## Development convergence", "",
        "| step | candidate | utility | rho error reduction | AUROC delta | running best | promoted |",
        "|---:|---|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["development"]:
        lines.append(
            f"| {row['step']} | `{row['candidate']}` | {row['utility']:+.4f} | "
            f"{100*row['primary_rho_reduction']:+.2f}% | "
            f"{100*row['primary_auc_delta']:+.3f} pp | {row['running_best']:+.4f} | "
            f"{'yes' if row['promoted'] else ''} |"
        )
    lines += [
        "", "![Development convergence](convergence.svg)", "",
        "## Sealed synthetic validation", "",
        "| candidate | rho reduction [95% CI] | AUROC delta [95% CI] | clean | p05 | result |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in summary["validation"]:
        lines.append(
            f"| `{row['candidate']}` | {100*row['primary_rho_reduction']:+.2f}% "
            f"[{100*row['primary_rho_ci_low']:+.2f}, {100*row['primary_rho_ci_high']:+.2f}] | "
            f"{100*row['primary_auc_delta']:+.3f} "
            f"[{100*row['primary_auc_ci_low']:+.3f}, {100*row['primary_auc_ci_high']:+.3f}] pp | "
            f"{100*row['clean_auc_delta']:+.3f} pp | "
            f"{100*row['primary_auc_p05']:+.3f} pp | "
            f"**{'PASS' if row['pass'] else 'FAIL'}** |"
        )
    if summary.get("real_replay"):
        real = summary["real_replay"]
        contrast = real["contrast"]
        lines += [
            "", "## Frozen real-artifact replay", "",
            f"Candidate `{real['candidate']}` versus OLS SU-PCR: "
            f"**{contrast['mean_delta_pp']:+.2f}** points "
            f"(95% CI [{contrast['ci95_low_pp']:+.2f}, "
            f"{contrast['ci95_high_pp']:+.2f}]), W/L "
            f"{contrast['wins']}/{contrast['losses']}, family macro "
            f"{contrast['family_macro_delta_pp']:+.2f} points.", "",
            "| gate | observed | threshold | result |", "|---|---:|---:|:---:|",
        ]
        for gate in real["gates"]:
            lines.append(
                f"| `{gate['gate']}` | {gate['observed']:+.3f} | "
                f">= {gate['threshold']:+.3f} | "
                f"**{'PASS' if gate['pass'] else 'FAIL'}** |"
            )
    else:
        lines += [
            "", "## Real replay", "",
            "Not run: no candidate crossed all frozen synthetic gates. This is the "
            "registered stop rule, not missing output.",
        ]
    lines += ["", "## Scientific conclusion", "", summary["conclusion"], ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--dev-repeats", type=int, default=8)
    parser.add_argument("--validation-repeats", type=int, default=16)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    dev_repeats = 2 if args.quick else args.dev_repeats
    validation_repeats = 3 if args.quick else args.validation_repeats
    started = time.time()

    dev_rows = []
    for world in WORLDS:
        for repetition in range(dev_repeats):
            dev_rows.append(run_synthetic_repetition(
                world, repetition, "development", CANDIDATES,
            ))
            print(f"development {world} {repetition + 1}/{dev_repeats}", flush=True)
    dev_summaries = [
        summarize_synthetic(dev_rows, method, "development")
        for method in CANDIDATES if method != "ols"
    ]
    ranked = sorted(dev_summaries, key=lambda row: row["utility"], reverse=True)
    promoted = tuple(row["candidate"] for row in ranked[:3])
    running_best = -float("inf")
    ledger = []
    for step, row in enumerate(dev_summaries, 1):
        running_best = max(running_best, row["utility"])
        ledger.append({
            **row, "step": step, "running_best": running_best,
            "promoted": row["candidate"] in promoted,
        })

    validation_rows = []
    score_methods = ("ols",) + promoted
    for world in WORLDS:
        for repetition in range(validation_repeats):
            validation_rows.append(run_synthetic_repetition(
                world, repetition, "sealed_validation", score_methods,
            ))
            print(
                f"sealed validation {world} {repetition + 1}/{validation_repeats}",
                flush=True,
            )
    validation = []
    for method in promoted:
        row = summarize_synthetic(validation_rows, method, "sealed_validation")
        row["pass"] = synthetic_pass(row)
        validation.append(row)
    passing = [row for row in validation if row["pass"]]
    frozen_winner = (
        max(passing, key=lambda row: row["utility"])["candidate"] if passing else None
    )

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "development_replicates.csv"), dev_rows)
    write_csv(os.path.join(args.out_dir, "validation_replicates.csv"), validation_rows)
    write_csv(os.path.join(args.out_dir, "candidate_ledger.csv"), [
        {key: value for key, value in row.items() if key != "worlds"} for row in ledger
    ])
    write_csv(os.path.join(args.out_dir, "validation_summary.csv"), [
        {key: value for key, value in row.items() if key != "worlds"}
        for row in validation
    ])
    write_csv(os.path.join(args.out_dir, "convergence.csv"), [
        {"step": row["step"], "candidate": row["candidate"],
         "utility": row["utility"], "running_best": row["running_best"]}
        for row in ledger
    ])
    render_convergence_svg(os.path.join(args.out_dir, "convergence.svg"), ledger)

    real_summary = None
    if frozen_winner is not None and os.path.exists(args.bundle):
        real_rows, contrast, gates, real_pass = run_real_replay(
            frozen_winner, args.bundle, args.manifest,
        )
        write_csv(os.path.join(args.out_dir, "real_per_cell.csv"), real_rows)
        real_summary = {
            "candidate": frozen_winner, "contrast": contrast,
            "gates": gates, "pass": bool(real_pass),
        }

    if frozen_winner is None:
        decision = "STOP_SYNTHETIC_HYPOTHESIS_REJECTED"
        conclusion = (
            "Covariance-aware moment weighting did not satisfy the preregistered "
            "mechanism-and-no-harm criteria on disjoint synthetic validation. The "
            "correlated pair equations are real, but correcting their sampling "
            "covariance is not a demonstrated improvement to U-PCR under these worlds. "
            "The real artifact was intentionally not opened for candidate selection."
        )
    elif real_summary and real_summary["pass"]:
        decision = "MEETS_RETROSPECTIVE_REAL_GATE_NEEDS_EXTERNAL_CONFIRMATION"
        conclusion = (
            "The frozen covariance-aware rho estimator passed the planted mechanism "
            "test and the retrospective real contribution gate. This is promising but "
            "not confirmatory: the 24-cell artifact informed prior development, so a "
            "new dataset/model family is required before a contribution claim."
        )
    else:
        decision = "SYNTHETIC_MECHANISM_ONLY_REAL_GATE_FAILED"
        conclusion = (
            "The covariance-aware rho estimator improved known-truth synthetic "
            "reliability under the frozen gates, but did not deliver the required "
            "retrospective real AUROC improvement over SU-PCR. The mechanism may be "
            "statistically valid yet practically filtered out by two-component PCR; "
            "it must not be tuned on these 24 label vectors."
        )

    summary = {
        "version": VERSION, "decision": decision,
        "frozen_winner": frozen_winner,
        "feature_schema": SCHEMA_VERSION,
        "runtime_seconds": time.time() - started,
        "config": {
            "worlds": WORLDS, "primary_worlds": PRIMARY_WORLDS,
            "stress_worlds": STRESS_WORLDS, "candidates": CANDIDATES,
            "moment_max_condition": MOMENT_MAX_CONDITION,
            "dev_repeats": dev_repeats,
            "validation_repeats": validation_repeats,
            "real_gates": REAL_GATES,
        },
        "development": ledger, "validation": validation,
        "real_replay": real_summary, "conclusion": conclusion,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(jsonable(summary), handle, indent=2, sort_keys=True)
    with open(os.path.join(args.out_dir, "REPORT.md"), "w", encoding="utf-8") as handle:
        handle.write(render_report(summary))
    print(f"Decision: {decision}; frozen winner={frozen_winner}")
    print(f"Outputs: {args.out_dir}")


if __name__ == "__main__":
    main()

