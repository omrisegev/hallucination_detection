#!/usr/bin/env python3
"""Run Phase A1: label-blind factorial measurement reconstruction.

All model selection uses only the hash-defined structural-training cells and
masked covariance reconstruction.  The seven audit environments are not used
until one configuration per method family has been frozen to disk.  No
correctness-label key or detector metric is read anywhere in this script.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.atomic_nrm_structural_audit import (  # noqa: E402
    DEFAULT_BUNDLE,
    SOURCE_CELLS,
    load_spaces,
)
from spectral_utils.contribution_subspace import fit_contribution_transform  # noqa: E402
from spectral_utils.factorial_measurement import (  # noqa: E402
    FactorialConfiguration,
    augment_correlated_duplicate,
    covariance_from_residuals,
    fit_factorial_measurement,
    masked_feature_reconstruction_rows,
    pooled_mean_reconstruction_rows,
    reconstruction_mse,
    select_configuration,
    soft_quotient_weights,
    subspace_stability,
)
from spectral_utils.group_free_research import (  # noqa: E402
    canonical_feature_names,
    derive_feature_dag,
    sha256_file,
    simulate_factorial_world,
)


VERSION = "automatic-group-free-iu-a1-v1-2026-08-13"
DEFAULT_OUT = REPO / "results" / "automatic_group_free_phase_a1_v1"
RANKS = (1, 2, 3, 4, 6)
RIDGES = (0.01, 0.1, 1.0)
HYBRID_ALPHAS = (0.25, 0.5, 0.75)
RANDOM_PARTITION_DRAWS = 32
BOOTSTRAP_DRAWS = 2000
SEED = 20260813
STABILITY_GATE = 0.80
NEAR_DUPLICATE_MASS_GATE = 1.10


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_split(cell: str) -> str:
    bucket = int(hashlib.sha256(cell.encode("utf-8")).hexdigest()[:8], 16) % 3
    return "audit" if bucket == 0 else "structural_train"


def source_covariances() -> tuple[np.ndarray, tuple[str, ...], list[dict]]:
    feature_rows, spaces, _ = load_spaces(DEFAULT_BUNDLE)
    roster = canonical_feature_names()
    covariances = []
    rows = []
    for cell, feature_row, space in zip(SOURCE_CELLS, feature_rows, spaces):
        indices = np.arange(len(space.baseline_score), dtype=int)
        transform = fit_contribution_transform(space, indices)
        _, residuals = transform.apply(space.baseline_score, space.contributions)
        covariance = covariance_from_residuals(residuals, space.families, roster)
        covariances.append(covariance)
        rows.append({
            "cell": cell,
            "split": audit_split(cell),
            "n_samples": feature_row["n_samples"],
            "n_features": feature_row["n_features"],
            "active_residual_features": int(np.isfinite(np.diag(covariance)).sum()),
            "labels_accessed": False,
        })
    return np.asarray(covariances), roster, rows


def configuration_grid(kind: str) -> list[FactorialConfiguration]:
    alphas = HYBRID_ALPHAS if kind == "hybrid" else (0.5,)
    maximum_rank = 4 if kind == "channel" else max(RANKS)
    return [
        FactorialConfiguration(kind, rank, interaction, ridge, alpha=alpha)
        for rank in RANKS if rank <= maximum_rank
        for interaction in (False, True)
        for ridge in RIDGES
        for alpha in alphas
    ]


def evaluate_configuration(
    train_covariances: np.ndarray,
    audit_covariances: np.ndarray,
    audit_ids: list[str],
    roster: tuple[str, ...],
    dag: list[dict],
    configuration: FactorialConfiguration,
    method: str,
) -> tuple[list[dict], object]:
    fit = fit_factorial_measurement(
        train_covariances, roster, dag, configuration
    )
    rows = []
    for environment, covariance in zip(audit_ids, audit_covariances):
        for row in masked_feature_reconstruction_rows(fit, covariance, environment):
            rows.append({"method": method, **row})
    return rows, fit


def keyed_errors(rows: list[dict]) -> tuple[list[tuple[str, str, str]], np.ndarray]:
    ordered = sorted(rows, key=lambda row: (
        row["environment"], row["held_out_feature"], row["partner_feature"]
    ))
    keys = [(
        row["environment"], row["held_out_feature"], row["partner_feature"]
    ) for row in ordered]
    errors = np.asarray([row["squared_error"] for row in ordered], dtype=float)
    return keys, errors


def grouped_bootstrap_delta(
    candidate_rows: list[dict],
    baseline_rows: list[dict],
    *,
    seed: int,
) -> dict:
    candidate_keys, candidate = keyed_errors(candidate_rows)
    baseline_keys, baseline = keyed_errors(baseline_rows)
    if candidate_keys != baseline_keys:
        raise RuntimeError("paired reconstruction rows do not align")
    groups = np.asarray([key[0] for key in candidate_keys], dtype=object)
    unique = np.asarray(sorted(set(groups)), dtype=object)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in sampled])
        draws[draw] = float(np.mean(candidate[indices] - baseline[indices]))
    delta = candidate - baseline
    return {
        "delta_mse": float(np.mean(delta)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
        "bootstrap_groups": len(unique),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "seed": int(seed),
    }


def append_dag_duplicate(dag: list[dict], source_name: str, duplicate_name: str) -> list[dict]:
    source = next(row for row in dag if row["feature_name"] == source_name)
    return dag + [{**source, "feature_name": duplicate_name, "feature_index": len(dag)}]


def duplicate_stress(
    train_covariances: np.ndarray,
    roster: tuple[str, ...],
    dag: list[dict],
    configuration: FactorialConfiguration,
) -> dict:
    fit = fit_factorial_measurement(train_covariances, roster, dag, configuration)
    weights, _ = soft_quotient_weights(fit)
    source_index = int(np.argmax(weights))
    source_name = roster[source_index]
    duplicate_name = source_name + "__automatic_duplicate"
    augmented_names = roster + (duplicate_name,)
    augmented_dag = append_dag_duplicate(dag, source_name, duplicate_name)

    exact_covariances = augment_correlated_duplicate(
        train_covariances, source_index, correlation=1.0
    )
    exact_fit = fit_factorial_measurement(
        exact_covariances, augmented_names, augmented_dag, configuration
    )
    exact_weights, exact_diagnostics = soft_quotient_weights(exact_fit)
    exact_mass = float(exact_weights[source_index] + exact_weights[-1])
    exact_error = float(abs(exact_mass - weights[source_index]))

    near_covariances = augment_correlated_duplicate(
        train_covariances, source_index, correlation=0.999
    )
    near_fit = fit_factorial_measurement(
        near_covariances, augmented_names, augmented_dag, configuration
    )
    near_weights, _ = soft_quotient_weights(near_fit)
    near_mass = float(near_weights[source_index] + near_weights[-1])
    return {
        "source_feature": source_name,
        "original_mass": float(weights[source_index]),
        "exact_duplicate_combined_mass": exact_mass,
        "exact_duplicate_mass_error": exact_error,
        "exact_duplicate_class_count": len(exact_diagnostics["duplicate_classes"]),
        "near_duplicate_correlation": 0.999,
        "near_duplicate_combined_mass": near_mass,
        "near_duplicate_mass_ratio": float(near_mass / weights[source_index]),
        "uses_labels": False,
    }


def permutation_repeatability_stress(
    train_covariances: np.ndarray,
    audit_covariances: np.ndarray,
    audit_ids: list[str],
    roster: tuple[str, ...],
    dag: list[dict],
    configuration: FactorialConfiguration,
    reference_rows: list[dict],
) -> dict:
    """Verify named predictions under feature permutation and deterministic refit."""

    rng = np.random.default_rng(SEED + 77)
    permutation = rng.permutation(len(roster))
    permuted_roster = tuple(roster[index] for index in permutation)
    permuted_train = train_covariances[:, permutation][:, :, permutation]
    permuted_audit = audit_covariances[:, permutation][:, :, permutation]
    permuted_rows, _ = evaluate_configuration(
        permuted_train,
        permuted_audit,
        audit_ids,
        permuted_roster,
        dag,
        configuration,
        "permuted",
    )
    repeated_rows, _ = evaluate_configuration(
        train_covariances,
        audit_covariances,
        audit_ids,
        roster,
        dag,
        configuration,
        "repeat",
    )

    def predictions(rows):
        return {
            (row["environment"], row["held_out_feature"], row["partner_feature"]):
            float(row["prediction"])
            for row in rows
        }

    reference = predictions(reference_rows)
    permuted = predictions(permuted_rows)
    repeated = predictions(repeated_rows)
    if reference.keys() != permuted.keys() or reference.keys() != repeated.keys():
        raise RuntimeError("permutation/repeatability reconstruction keys disagree")
    permutation_error = max(
        abs(reference[key] - permuted[key]) for key in reference
    )
    repeatability_error = max(
        abs(reference[key] - repeated[key]) for key in reference
    )
    return {
        "feature_order_permutation_max_prediction_error": float(permutation_error),
        "deterministic_repeatability_max_prediction_error": float(repeatability_error),
        "seed": SEED + 77,
        "uses_labels": False,
    }


def simulator_dag(world) -> list[dict]:
    rows = []
    for index, name in enumerate(world.feature_names):
        channel, operator = name.split("__", 1)
        rows.append({
            "feature_index": index,
            "feature_name": name,
            "source_stream": channel,
            "operator": operator,
        })
    return rows


def simulator_check(configuration: FactorialConfiguration) -> dict:
    world = simulate_factorial_world(seed=SEED)
    covariances = []
    for environment in world.environments:
        covariances.append(covariance_from_residuals(
            environment["matrix"],
            environment["feature_names"],
            world.feature_names,
        ))
    covariances = np.asarray(covariances)
    train = covariances[:6]
    audit = covariances[6:]
    fit = fit_factorial_measurement(
        train, world.feature_names, simulator_dag(world), configuration
    )
    candidate_rows = []
    pooled_rows = []
    for index, covariance in enumerate(audit):
        environment = f"simulator_audit_{index}"
        candidate_rows.extend(masked_feature_reconstruction_rows(
            fit, covariance, environment
        ))
        pooled_rows.extend(pooled_mean_reconstruction_rows(
            train, covariance, world.feature_names, environment
        ))
    weights, _ = soft_quotient_weights(fit)
    duplicate = fit.duplicate_classes
    duplicate_class = next(
        group for group in duplicate if 0 in group
    )
    return {
        "candidate_mse": reconstruction_mse(candidate_rows),
        "pooled_mean_mse": reconstruction_mse(pooled_rows),
        "candidate_beats_pooled": reconstruction_mse(candidate_rows) < reconstruction_mse(pooled_rows),
        "detected_duplicate_class": list(duplicate_class),
        "duplicate_class_mass": float(weights[np.asarray(duplicate_class)].sum()),
        "environment_specific_target": world.environment_specific_target,
        "uses_evaluator_latents_for_selection": False,
        "uses_labels": False,
    }


def summary_row(method: str, rows: list[dict], configuration=None) -> dict:
    payload = {
        "method": method,
        "mse": reconstruction_mse(rows),
        "rmse": float(np.sqrt(reconstruction_mse(rows))),
        "prediction_rows": len(rows),
    }
    if configuration is not None:
        payload.update(configuration.payload())
    return payload


def main() -> None:
    out = DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    covariances, roster, source_rows = source_covariances()
    dag = derive_feature_dag(roster)
    train_mask = np.asarray([row["split"] == "structural_train" for row in source_rows])
    audit_mask = ~train_mask
    train = covariances[train_mask]
    audit = covariances[audit_mask]
    train_ids = [row["cell"] for row in source_rows if row["split"] == "structural_train"]
    audit_ids = [row["cell"] for row in source_rows if row["split"] == "audit"]

    write_csv(out / "source_split.csv", source_rows)
    boundary = {
        "version": VERSION,
        "split_rule": "sha256(cell) first 32 bits modulo 3; bucket 0 is audit",
        "structural_train_cells": train_ids,
        "audit_cells": audit_ids,
        "feature_roster": list(roster),
        "rank_grid": list(RANKS),
        "ridge_grid": list(RIDGES),
        "hybrid_alpha_grid": list(HYBRID_ALPHAS),
        "random_partition_draws": RANDOM_PARTITION_DRAWS,
        "stability_gate": STABILITY_GATE,
        "near_duplicate_mass_gate": NEAR_DUPLICATE_MASS_GATE,
        "selection_metric": "masked feature-by-environment covariance MSE",
        "source_bundle_sha256": sha256_file(DEFAULT_BUNDLE),
        "code_sha256": {
            "script": sha256_file(Path(__file__)),
            "factorial_measurement_module": sha256_file(
                REPO / "spectral_utils" / "factorial_measurement.py"
            ),
        },
        "labels_accessed": False,
    }
    write_json(out / "SELECTION_BOUNDARY.json", boundary)

    selected = {}
    search_rows = []
    for kind in ("pca", "channel", "operator", "factorial", "hybrid"):
        configuration, rows = select_configuration(
            train, train_ids, roster, dag, configuration_grid(kind)
        )
        selected[kind] = configuration
        search_rows.extend({"method_family": kind, **row} for row in rows)
        print(json.dumps({
            "selected_family": kind,
            "configuration": configuration.payload(),
            "cv_mse": rows[0]["cv_mse"],
        }), flush=True)
    write_csv(out / "configuration_search.csv", search_rows)
    frozen = {
        "version": VERSION,
        "selected_configurations": {
            kind: configuration.payload() for kind, configuration in selected.items()
        },
        "primary_candidate": "hybrid",
        "selection_used_audit_environments": False,
        "selection_used_correctness_labels": False,
    }
    write_json(out / "FROZEN_SELECTION.json", frozen)

    detailed_rows = []
    summaries = []
    method_rows = {}
    for kind, configuration in selected.items():
        rows, _ = evaluate_configuration(
            train, audit, audit_ids, roster, dag, configuration, kind
        )
        method_rows[kind] = rows
        detailed_rows.extend(rows)
        summaries.append(summary_row(kind, rows, configuration))

    pooled_rows = []
    for environment, covariance in zip(audit_ids, audit):
        pooled_rows.extend({"method": "pooled_mean", **row}
                           for row in pooled_mean_reconstruction_rows(
                               train, covariance, roster, environment
                           ))
    method_rows["pooled_mean"] = pooled_rows
    detailed_rows.extend(pooled_rows)
    summaries.append(summary_row("pooled_mean", pooled_rows))

    random_summaries = []
    random_rows_by_seed = []
    primary = selected["hybrid"]
    for draw in range(RANDOM_PARTITION_DRAWS):
        configuration = FactorialConfiguration(
            "random",
            primary.rank,
            primary.interaction,
            primary.ridge,
            random_seed=SEED + draw,
        )
        rows, _ = evaluate_configuration(
            train, audit, audit_ids, roster, dag, configuration, f"random_{draw:02d}"
        )
        random_rows_by_seed.append(rows)
        random_summaries.append(summary_row(f"random_{draw:02d}", rows, configuration))
    random_summaries.sort(key=lambda row: row["mse"])
    write_csv(out / "random_partition_summary.csv", random_summaries)
    median_random = random_rows_by_seed[
        int(random_summaries[len(random_summaries) // 2]["random_seed"] - SEED)
    ]

    primary_rows = method_rows["hybrid"]
    pca_delta = grouped_bootstrap_delta(
        primary_rows, method_rows["pca"], seed=SEED + 101
    )
    random_delta = grouped_bootstrap_delta(
        primary_rows, median_random, seed=SEED + 102
    )
    stability = subspace_stability(train, roster, dag, primary)
    stability_rows = [
        {"held_out_cell": cell, "projector_overlap": overlap}
        for cell, overlap in zip(train_ids, stability)
    ]
    stress = duplicate_stress(train, roster, dag, primary)
    invariance = permutation_repeatability_stress(
        train, audit, audit_ids, roster, dag, primary, primary_rows
    )
    stress.update(invariance)
    simulator = simulator_check(primary)

    fifth_percentile_random = float(np.quantile(
        [row["mse"] for row in random_summaries], 0.05
    ))
    gates = {
        "beats_pooled_pca_ci": pca_delta["ci_upper"] < 0.0,
        "beats_random_partition_ci": random_delta["ci_upper"] < 0.0,
        "beats_random_partition_fifth_percentile": (
            reconstruction_mse(primary_rows) < fifth_percentile_random
        ),
        "minimum_loo_projector_overlap": min(stability),
        "stability_pass": min(stability) >= STABILITY_GATE,
        "exact_duplicate_mass_error": stress["exact_duplicate_mass_error"],
        "exact_duplicate_pass": stress["exact_duplicate_mass_error"] < 1e-10,
        "near_duplicate_mass_ratio": stress["near_duplicate_mass_ratio"],
        "near_duplicate_pass": stress["near_duplicate_mass_ratio"] <= NEAR_DUPLICATE_MASS_GATE,
        "feature_order_permutation_pass": (
            stress["feature_order_permutation_max_prediction_error"] < 1e-10
        ),
        "deterministic_repeatability_pass": (
            stress["deterministic_repeatability_max_prediction_error"] < 1e-12
        ),
        "simulator_reconstruction_pass": simulator["candidate_beats_pooled"],
    }
    passed = all([
        gates["beats_pooled_pca_ci"],
        gates["beats_random_partition_ci"],
        gates["beats_random_partition_fifth_percentile"],
        gates["stability_pass"],
        gates["exact_duplicate_pass"],
        gates["near_duplicate_pass"],
        gates["feature_order_permutation_pass"],
        gates["deterministic_repeatability_pass"],
        gates["simulator_reconstruction_pass"],
    ])
    decision = "PASS" if passed else "CLOSE_AS_DETECTOR_BASIS"

    write_csv(out / "audit_reconstruction_summary.csv", summaries)
    write_csv(out / "audit_reconstruction_rows.csv", detailed_rows)
    write_csv(out / "stability.csv", stability_rows)
    write_json(out / "stress_tests.json", stress)
    write_json(out / "simulator_results.json", simulator)
    decision_payload = {
        "version": VERSION,
        "decision": decision,
        "primary_configuration": primary.payload(),
        "primary_audit_mse": reconstruction_mse(primary_rows),
        "pca_audit_mse": reconstruction_mse(method_rows["pca"]),
        "hard_factorial_audit_mse": reconstruction_mse(method_rows["factorial"]),
        "pooled_mean_audit_mse": reconstruction_mse(pooled_rows),
        "random_mse_fifth_percentile": fifth_percentile_random,
        "random_mse_median": float(np.median([row["mse"] for row in random_summaries])),
        "paired_delta_vs_pca": pca_delta,
        "paired_delta_vs_median_random": random_delta,
        "gates": gates,
        "labels_accessed": False,
        "next_phase": "A2 multi-environment joint block diagonalization",
    }
    write_json(out / "A1_COMPLETE.json", decision_payload)

    report = f"""# Automatic group-free IU — Phase A1 factorial measurement model

- Version: `{VERSION}`
- Correctness labels accessed: **no**
- Structural train / untouched structural audit environments: **{len(train_ids)} / {len(audit_ids)}**
- Feature roster: **{len(roster)}**, with NaN-preserving incomplete coverage
- Primary basis: **hybrid soft factorial**, selected only by training-cell LOEO reconstruction
- Frozen configuration: `{json.dumps(primary.payload(), sort_keys=True)}`
- Audit RMSE — hybrid / pooled PCA / hard factorial / pooled mean: **{np.sqrt(reconstruction_mse(primary_rows)):.6f} / {np.sqrt(reconstruction_mse(method_rows['pca'])):.6f} / {np.sqrt(reconstruction_mse(method_rows['factorial'])):.6f} / {np.sqrt(reconstruction_mse(pooled_rows)):.6f}**
- Paired MSE delta vs pooled PCA, grouped 95% CI: **{pca_delta['delta_mse']:.6g} [{pca_delta['ci_lower']:.6g}, {pca_delta['ci_upper']:.6g}]**
- Paired MSE delta vs median random partition, grouped 95% CI: **{random_delta['delta_mse']:.6g} [{random_delta['ci_lower']:.6g}, {random_delta['ci_upper']:.6g}]**
- Random-partition fifth-percentile MSE: **{fifth_percentile_random:.6g}**
- Leave-one-training-environment projector overlap: min **{min(stability):.6f}**, median **{np.median(stability):.6f}**
- Exact duplicate mass error: **{stress['exact_duplicate_mass_error']:.3g}**
- Near-duplicate combined/original mass ratio at rho=0.999: **{stress['near_duplicate_mass_ratio']:.6f}**
- Feature-order permutation / repeatability max error: **{stress['feature_order_permutation_max_prediction_error']:.3g} / {stress['deterministic_repeatability_max_prediction_error']:.3g}**
- Simulator candidate / pooled MSE: **{simulator['candidate_mse']:.6g} / {simulator['pooled_mean_mse']:.6g}**

## Decision

**{decision}**. The route passes only if the predeclared hybrid representation
beats pooled PCA and cardinality-matched random partitions on the hash-held-out
environments, remains stable under environment deletion, conserves exact-
duplicate mass, controls a near duplicate, and beats the pooled simulator
baseline. No detector AUROC, correctness target, Family-NRM direction, or
supervised atomic direction participated in selection or in this decision.

Regardless of this A1 decision, Phase A2 proceeds on the raw atomic residual
covariances. A1 may be used inside A3 only when the result above is PASS.
"""
    (out / "REPORT.md").write_text(report, encoding="utf-8")

    artifacts = {}
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != "ARTIFACT_HASHES.json":
            artifacts[path.name] = sha256_file(path)
    write_json(out / "ARTIFACT_HASHES.json", artifacts)
    print(json.dumps({
        "out": str(out),
        "decision": decision,
        "primary_audit_mse": decision_payload["primary_audit_mse"],
        "pca_audit_mse": decision_payload["pca_audit_mse"],
        "labels_accessed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
