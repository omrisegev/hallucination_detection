#!/usr/bin/env python3
"""Evaluation-only Stage B for frozen Residual-Graph DEEM scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.graph_topology import (  # noqa: E402
    exact_length_permutations,
    propensity_crt_permutations,
)
from spectral_utils.residual_graph_deem import (  # noqa: E402
    ARM_SPECS,
    SEEDS,
    ResidualGraphDeemError,
    atomic_write_json,
    canonical_sha256,
    family_index_map,
    normalized_rayleigh,
    sha256_file,
    symmetric_normalized_laplacian,
)
from spectral_utils.residual_graph_deem_data import (  # noqa: E402
    load_registry,
    load_target_free_bundle,
)
from spectral_utils.residual_graph_deem_labels import (  # noqa: E402
    join_labels_by_id,
    load_label_sidecar,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def verify_score_freeze(run_dir: Path) -> dict:
    path = run_dir / "SCORE_FREEZE_MANIFEST.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete" or value.get("debug"):
        raise ResidualGraphDeemError("evaluator refuses incomplete/debug score freeze")
    unhashed = dict(value)
    expected = unhashed.pop("content_sha256", None)
    if canonical_sha256(unhashed) != expected:
        raise ResidualGraphDeemError("score-freeze content hash mismatch")
    for artifact in value.get("artifacts", []):
        candidate = run_dir / artifact["path"]
        if not candidate.is_file() or sha256_file(candidate) != artifact["sha256"]:
            raise ResidualGraphDeemError(f"score artifact hash mismatch: {candidate}")
    return value


def lambda_token(value: float) -> str:
    return str(float(value)).replace(".", "p")


def load_seed_score(run_dir: Path, cell: str, stem: str) -> np.ndarray:
    metadata_path = run_dir / "fits" / cell / f"{stem}.json"
    array_path = run_dir / "fits" / cell / f"{stem}.npz"
    if not metadata_path.is_file() or not array_path.is_file():
        raise ResidualGraphDeemError(f"missing fit artifact: {cell}/{stem}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    health = metadata.get("health", {})
    healthy = health.get("healthy", metadata.get("healthy", False))
    if metadata.get("status") != "complete" or not healthy:
        raise ResidualGraphDeemError(f"unhealthy/incomplete fit: {cell}/{stem}")
    if sha256_file(array_path) != metadata.get("array_sha256"):
        raise ResidualGraphDeemError(f"fit metadata/array hash mismatch: {cell}/{stem}")
    with np.load(array_path, allow_pickle=False) as data:
        return np.asarray(data["score"], dtype=float)


def load_csr(path: Path):
    with np.load(path, allow_pickle=False) as data:
        required = {"graph_data", "graph_indices", "graph_indptr", "graph_shape"}
        if not required.issubset(data.files):
            raise ResidualGraphDeemError(f"missing serialized sparse graph: {path}")
        return sparse.csr_matrix(
            (data["graph_data"], data["graph_indices"], data["graph_indptr"]),
            shape=tuple(int(value) for value in data["graph_shape"]),
        )


def fit_metadata(run_dir: Path, cell: str, stem: str) -> dict:
    path = run_dir / "fits" / cell / f"{stem}.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = value.get("content_sha256")
    if expected is not None:
        unhashed = dict(value); unhashed.pop("content_sha256", None)
        if canonical_sha256(unhashed) != expected:
            raise ResidualGraphDeemError(f"fit metadata content mismatch: {path}")
    return value


def ensemble(run_dir: Path, cell: str, arm: str, lambda_: float | None = None) -> tuple[np.ndarray, dict]:
    scores = []
    for seed in SEEDS:
        stem = arm
        if lambda_ is not None:
            stem += f"__lambda{lambda_token(lambda_)}"
        stem += f"__seed{seed}"
        scores.append(load_seed_score(run_dir, cell, stem))
    matrix = np.asarray(scores, dtype=float)
    correlations = []
    from scipy.stats import spearmanr
    for left in range(5):
        for right in range(left + 1, 5):
            correlations.append(abs(float(spearmanr(matrix[left], matrix[right]).statistic)))
    return matrix.mean(axis=0), {
        "seed_score_sd_mean": float(np.mean(matrix.std(axis=0))),
        "median_abs_spearman": float(np.median(correlations)),
        "minimum_abs_spearman": float(np.min(correlations)),
    }


def metrics(target: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    y = np.asarray(target, dtype=int)
    s = np.asarray(score, dtype=float)
    if len(np.unique(y)) != 2 or s.shape != y.shape or not np.isfinite(s).all():
        raise ResidualGraphDeemError("invalid target/score for evaluation")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def method_scores(run_dir: Path, cell: str, nominations: dict) -> tuple[dict[str, np.ndarray], dict]:
    output, stability = {}, {}
    for arm in ("B0", "B1", "B2", "B3"):
        output[arm], stability[arm] = ensemble(run_dir, cell, arm)
    for arm in ("G0", "G1", "G2", "G3"):
        output[arm], stability[arm] = ensemble(run_dir, cell, arm, nominations["target"])
    output["G4"], stability["G4"] = ensemble(run_dir, cell, "G4", nominations["nuisance"])
    output["G5"], stability["G5"] = ensemble(run_dir, cell, "G5", nominations["family"])
    return output, stability


def aggregate(per_cell: list[dict], method: str, metric_name: str = "auroc") -> dict:
    selected = [row for row in per_cell if row["method"] == method]
    by_family = defaultdict(list)
    for row in selected:
        by_family[row["dataset_family"]].append(float(row[metric_name]))
    family_means = {family: float(np.mean(values)) for family, values in by_family.items()}
    qa = [row[metric_name] for row in selected if row["task_type"] == "QA"]
    math = [row[metric_name] for row in selected if row["task_type"] == "math"]
    return {
        "method": method,
        "metric": metric_name,
        "equal_family_macro": float(np.mean(list(family_means.values()))),
        "qa_macro": float(np.mean(qa)),
        "math_macro": float(np.mean(math)),
        "worst_cell": float(min(row[metric_name] for row in selected)),
        "worst_family": float(min(family_means.values())),
        "family_means": family_means,
    }


def family_bootstrap(
    per_cell: list[dict],
    reference: str,
    candidate: str,
    *,
    draws: int = 10_000,
    seed: int = 20260821,
) -> dict:
    values = defaultdict(dict)
    for row in per_cell:
        if row["method"] in {reference, candidate}:
            values[row["dataset_family"]].setdefault(row["method"], []).append(float(row["auroc"]))
    families = sorted(values)
    family_delta = {
        family: float(np.mean(values[family][candidate]) - np.mean(values[family][reference]))
        for family in families
    }
    generator = np.random.Generator(np.random.PCG64(seed))
    distribution = np.empty(draws, dtype=float)
    for draw in range(draws):
        sampled = generator.choice(families, size=len(families), replace=True)
        cell_deltas = []
        for family in sampled:
            ref = np.asarray(values[family][reference])
            cand = np.asarray(values[family][candidate])
            indexes = generator.integers(0, len(ref), size=len(ref))
            cell_deltas.append(float(np.mean(cand[indexes] - ref[indexes])))
        distribution[draw] = float(np.mean(cell_deltas))
    observed = float(np.mean(list(family_delta.values())))
    return {
        "reference": reference,
        "candidate": candidate,
        "observed": observed,
        "lower": float(np.quantile(distribution, 0.025)),
        "upper": float(np.quantile(distribution, 0.975)),
        "family_delta": family_delta,
        "leave_one_family_out": {
            omitted: float(np.mean([value for family, value in family_delta.items() if family != omitted]))
            for omitted in families
        },
        "distribution": distribution,
    }


def win_table(per_cell: list[dict], reference: str, candidate: str, tolerance: float = 0.0005) -> dict:
    lookup = {(row["cell_id"], row["method"]): row["auroc"] for row in per_cell}
    delta = [lookup[(cell, candidate)] - lookup[(cell, reference)]
             for cell in sorted({row["cell_id"] for row in per_cell})]
    return {
        "wins": int(sum(value > tolerance for value in delta)),
        "ties": int(sum(abs(value) <= tolerance for value in delta)),
        "losses": int(sum(value < -tolerance for value in delta)),
        "worst_cell_delta": float(min(delta)),
    }


def blocked_group_draws(targets, bundles, *, B: int, seed: int) -> dict[str, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(seed))
    groups_by_stratum = defaultdict(list)
    for cell, bundle in bundles.items():
        y = targets[cell]
        for group in sorted(set(bundle.group_ids)):
            indexes = np.flatnonzero(np.asarray(bundle.group_ids) == group)
            groups_by_stratum[(bundle.dataset_family, len(indexes))].append((cell, indexes, y[indexes].copy()))
    output = {cell: np.empty((len(targets[cell]), B), dtype=np.int8) for cell in targets}
    for draw in range(B):
        for entries in groups_by_stratum.values():
            permutation = generator.permutation(len(entries))
            for destination, source_position in zip(entries, permutation):
                dest_cell, dest_indices, _ = destination
                _, _, source_values = entries[int(source_position)]
                output[dest_cell][dest_indices, draw] = source_values
    return output


def conditional_draws(targets, bundles, *, B: int, seed: int = 20260821):
    exact_by_cell, crt_by_cell, diagnostics = {}, {}, {}
    for index, (cell, y) in enumerate(targets.items()):
        exact_by_cell[cell], exact_diag = exact_length_permutations(
            y, bundles[cell].raw_trace_length, permutations=B, seed=seed + index
        )
        crt_by_cell[cell], crt_diag = propensity_crt_permutations(
            y, bundles[cell].raw_trace_length, permutations=B, seed=seed + 100 + index
        )
        diagnostics[cell] = {"exact": exact_diag, "crt": crt_diag}
    blocked = blocked_group_draws(targets, bundles, B=B, seed=seed + 200)
    return {"exact": exact_by_cell, "crt": crt_by_cell, "family_group": blocked}, diagnostics


def graph_catalog(run_dir: Path, cells, nominations: dict) -> dict:
    output = {}
    for cell in cells:
        current = {"B0_LINEAR": []}
        current["B0_LINEAR"].append(load_csr(run_dir / "graphs" / cell / "b0_one_dimensional_k7.npz"))
        for arm in ("G0", "G1", "G2", "G3", "G4"):
            mechanism = "nuisance" if arm == "G4" else "target"
            token = lambda_token(nominations[mechanism])
            current[arm] = [
                load_csr(run_dir / "fits" / cell / f"{arm}__lambda{token}__seed{seed}.npz")
                for seed in SEEDS
            ]
        output[cell] = current
    return output


def graph_smoothness(graph, values) -> float:
    vector = np.asarray(values, dtype=float)
    if float(np.var(vector)) <= 1e-12:
        return 0.0
    return float(1.0 - normalized_rayleigh(vector, symmetric_normalized_laplacian(graph)))


def geometry_rows(targets, bundles, graphs, draws) -> list[dict]:
    output = []
    for cell, roles in graphs.items():
        y = targets[cell]
        length = np.log1p(bundles[cell].raw_trace_length)
        linear = roles["B0_LINEAR"][0]
        linear_effect = {}
        for null_name in ("exact", "crt"):
            observed = graph_smoothness(linear, y)
            null = np.asarray([
                graph_smoothness(linear, draws[null_name][cell][:, index])
                for index in range(draws[null_name][cell].shape[1])
            ])
            linear_effect[null_name] = float(observed - np.mean(null))
        for role, members in roles.items():
            for seed_position, graph in enumerate(members):
                observed = graph_smoothness(graph, y)
                row = {
                    "cell_id": cell,
                    "dataset_family": bundles[cell].dataset_family,
                    "task_type": bundles[cell].task_type,
                    "graph_role": role,
                    "seed": int(seed_position if role != "B0_LINEAR" else 0),
                    "target_rayleigh": float(1.0 - observed),
                    "length_rayleigh": float(normalized_rayleigh(
                        length, symmetric_normalized_laplacian(graph)
                    )),
                }
                effects = []
                for null_name in ("exact", "crt"):
                    null = np.asarray([
                        graph_smoothness(graph, draws[null_name][cell][:, index])
                        for index in range(draws[null_name][cell].shape[1])
                    ])
                    effect = float(observed - np.mean(null))
                    row[f"target_effect_{null_name}"] = effect
                    row[f"target_p_{null_name}"] = float((1 + np.sum(null >= observed)) / (len(null) + 1))
                    row[f"advantage_vs_linear_{null_name}"] = float(effect - linear_effect[null_name])
                    effects.append(effect)
                row["min_conditional_effect"] = float(min(effects))
                output.append(row)
    return output


def whole_search_null(
    targets, bundles, scores_by_cell, graphs, draws, *, B: int,
) -> dict:
    candidate_methods = ("G0", "G1", "G2", "G3", "G4", "G5")

    def statistic(target_map):
        rows = []
        for cell, y in target_map.items():
            for method in ("B3",) + candidate_methods:
                try:
                    auc = float(roc_auc_score(y, scores_by_cell[cell][method]))
                except ValueError:
                    auc = 0.5
                rows.append({
                    "cell_id": cell, "dataset_family": bundles[cell].dataset_family,
                    "task_type": bundles[cell].task_type, "method": method, "auroc": auc,
                })
        base = aggregate(rows, "B3")["equal_family_macro"]
        components = {
            f"auroc::{method}": aggregate(rows, method)["equal_family_macro"] - base
            for method in candidate_methods
        }
        graph_values = defaultdict(lambda: defaultdict(list))
        for cell, roles in graphs.items():
            for role, members in roles.items():
                graph_values[role][bundles[cell].dataset_family].extend(
                    graph_smoothness(graph, target_map[cell]) for graph in members
                )
        components.update({
            f"geometry::{role}": float(np.mean([
                np.mean(values) for values in family_values.values()
            ]))
            for role, family_values in graph_values.items()
        })
        winner = max(components, key=components.get)
        return float(components[winner]), winner, components

    observed, observed_winner, observed_components = statistic(targets)
    nulls = {"exact": np.empty(B), "crt": np.empty(B), "family_group": np.empty(B)}
    winner_counts = {name: defaultdict(int) for name in nulls}
    for draw in range(B):
        for null_name in nulls:
            value, winner, _ = statistic({
                cell: draws[null_name][cell][:, draw] for cell in targets
            })
            nulls[null_name][draw] = value
            winner_counts[null_name][winner] += 1
    return {
        "B": B,
        "observed_max_delta": observed,
        "observed_winner": observed_winner,
        "observed_components": observed_components,
        "p_values": {
            name: float((1 + np.sum(values >= observed)) / (B + 1))
            for name, values in nulls.items()
        },
        "null_maxima": nulls,
        "null_winner_counts": winner_counts,
    }


def collect_fit_diagnostics(run_dir: Path, cells, nominations: dict):
    per_fit, reconstruction, graph_rows, gate_rows = [], [], [], []
    for cell in cells:
        for path in sorted((run_dir / "fits" / cell).glob("*.json")):
            if path.name == "CELL_COMPLETE.json":
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            health = value.get("health", {})
            extras = value.get("extras", {})
            row = {
                "cell_id": cell, "arm_id": value.get("arm_id", value.get("stem", "").split("__")[0]),
                "stem": value.get("stem", path.stem), "seed": value.get("seed"),
                "status": value.get("status"), "healthy": health.get("healthy", value.get("healthy")),
                "posterior_sd": health.get("posterior_sd", value.get("score_sd")),
                "runtime_seconds": health.get("runtime_seconds"),
                "mala_acceptance_mean": health.get("mala_acceptance_mean"),
                "nuisance_variance_min": health.get("nuisance_variance_min"),
                "nuisance_whitening_max_abs": health.get("nuisance_whitening_max_abs"),
                "logit_nuisance_dependence": health.get("logit_nuisance_dependence"),
                "is_headline": extras.get("is_headline", False),
                "alias_of": value.get("alias_of"), "array_sha256": value.get("array_sha256"),
                "graph_sha256": value.get("graph_sha256"),
                "gate_sha256": extras.get("gate_sha256"),
            }
            per_fit.append(row)
            if "contribution_reconstruction_max_abs" in health:
                reconstruction.append({
                    "cell_id": cell, "arm_id": row["arm_id"], "stem": row["stem"],
                    "seed": row["seed"], "max_abs_error": health["contribution_reconstruction_max_abs"],
                    "pass": float(health["contribution_reconstruction_max_abs"]) <= 1e-8,
                })
            graph = extras.get("graph_health")
            artifact = extras.get("fold_artifact") or {}
            if graph and extras.get("is_headline"):
                graph_rows.append({
                    "cell_id": cell, "arm_id": row["arm_id"], "seed": row["seed"],
                    "largest_component_fraction": graph.get("largest_component_fraction"),
                    "isolated_fraction": graph.get("isolated_fraction"),
                    "n_components": graph.get("n_components"), "n_edges": graph.get("n_edges"),
                    "healthy": graph.get("healthy"),
                    "fold_artifact_healthy": artifact.get("healthy"),
                    "same_fold_edge_p": artifact.get("same_fold_edge_p"),
                    "fold_predictability_p": artifact.get("fold_predictability_p"),
                })
            gates = extras.get("gate_weights")
            diagnostics = extras.get("residual_gate_diagnostics") or extras.get("raw_gate_diagnostics")
            if gates is not None and extras.get("is_headline"):
                with np.load(run_dir / "fits" / cell / f"{row['stem']}.npz", allow_pickle=False) as data:
                    names = [str(item) for item in data["feature_names"].tolist()]
                for feature, weight in zip(names, gates):
                    gate_rows.append({
                        "cell_id": cell, "arm_id": row["arm_id"], "seed": row["seed"],
                        "feature_name": feature, "gate_weight": float(weight),
                        "effective_feature_count": diagnostics.get("effective_feature_count"),
                        "median_seed_cosine": diagnostics.get("median_seed_cosine"),
                        "family_mass": json.dumps(diagnostics.get("family_mass", {}), sort_keys=True),
                    })
    return per_fit, reconstruction, graph_rows, gate_rows


def residual_diagnostics(run_dir: Path, bundles, cells) -> list[dict]:
    output = []
    for cell in cells:
        bundle = bundles[cell]
        length = np.log1p(bundle.raw_trace_length)
        for seed in SEEDS:
            path = run_dir / "crossfit" / cell / f"seed{seed}.npz"
            with np.load(path, allow_pickle=False) as data:
                logit = np.asarray(data["logit"], dtype=float)
                contributions = np.asarray(data["contributions"], dtype=float)
                residuals = np.asarray(data["residuals"], dtype=float)
                folds = np.asarray(data["folds"], dtype=int)
            def median_abs_corr(matrix, vector):
                values = [abs(float(spearmanr(matrix[:, index], vector).statistic)) for index in range(matrix.shape[1])]
                return float(np.nanmedian(values))
            groups = np.asarray(bundle.group_ids)
            disjoint = all(
                not set(groups[folds == fold]).intersection(set(groups[folds != fold]))
                for fold in range(5)
            )
            output.append({
                "cell_id": cell, "seed": seed, "n_features": residuals.shape[1],
                "raw_contribution_abs_spearman_length": median_abs_corr(contributions, length),
                "residual_abs_spearman_length": median_abs_corr(residuals, length),
                "raw_contribution_abs_spearman_logit": median_abs_corr(contributions, logit),
                "residual_abs_spearman_logit": median_abs_corr(residuals, logit),
                "residual_mean_abs": float(np.mean(np.abs(residuals))),
                "residual_sd_mean": float(np.mean(np.std(residuals, axis=0))),
                "group_disjoint_folds": disjoint,
            })
    return output


def nuisance_diagnostics(run_dir: Path, cells, nominations: dict) -> list[dict]:
    token_target = lambda_token(nominations["target"])
    token_nuisance = lambda_token(nominations["nuisance"])
    output = []

    def dependence(U, values):
        centered_u = U - U.mean(axis=0, keepdims=True)
        centered_y = values - np.mean(values)
        coef = np.linalg.lstsq(centered_u, centered_y, rcond=None)[0]
        fitted = centered_u @ coef
        return float(np.sum(fitted * fitted) / (np.sum(centered_y * centered_y) + 1e-12))

    for cell in cells:
        for seed in SEEDS:
            g4_path = run_dir / "fits" / cell / f"G4__lambda{token_nuisance}__seed{seed}.npz"
            with np.load(g4_path, allow_pickle=False) as data:
                U = np.asarray(data["state__nuisance::U"], dtype=float)
                g4_logit = np.asarray(data["logit"], dtype=float)
            with np.load(run_dir / "fits" / cell / f"B3__seed{seed}.npz", allow_pickle=False) as data:
                b3_logit = np.asarray(data["logit"], dtype=float)
            with np.load(run_dir / "fits" / cell / f"G3__lambda{token_target}__seed{seed}.npz", allow_pickle=False) as data:
                g3_logit = np.asarray(data["logit"], dtype=float)
            covariance = U.T @ U / max(len(U) - 1, 1)
            output.append({
                "cell_id": cell, "seed": seed,
                "nuisance_variance_min": float(np.min(np.var(U, axis=0))),
                "whitening_max_abs": float(np.max(np.abs(covariance - np.eye(U.shape[1])))),
                "dependence_B3": dependence(U, b3_logit),
                "dependence_G3": dependence(U, g3_logit),
                "dependence_G4": dependence(U, g4_logit),
            })
    return output


def evaluate_controls(run_dir, cells, targets, bundles, nominations, per_cell):
    headline = float(nominations["target"])
    controls = ("length_only", "node_permuted", "random_gate", "family_permuted", "posterior_permuted")
    control_rows = []
    lambda_zero_pass = True
    lambda_zero_rows = []
    for cell in cells:
        for arm in ("G0", "G1", "G2", "G3", "G4", "G5"):
            for seed in SEEDS:
                b3 = load_seed_score(run_dir, cell, f"B3__seed{seed}")
                zero_path = run_dir / "fits" / cell / f"{arm}__lambda0p0__seed{seed}.npz"
                with np.load(zero_path, allow_pickle=False) as data:
                    zero = np.asarray(data["score"], dtype=float)
                    graph_absent = not any(name.startswith("graph_") for name in data.files)
                exact = bool(np.array_equal(b3, zero) and graph_absent)
                lambda_zero_pass &= exact
                lambda_zero_rows.append({"cell_id": cell, "arm": arm, "seed": seed, "exact": exact})
        for control in controls:
            members = [
                load_seed_score(
                    run_dir, cell,
                    f"G3__lambda{lambda_token(headline)}__{control}__seed{seed}",
                ) for seed in SEEDS
            ]
            score_value = np.mean(members, axis=0)
            auc, ap = metrics(targets[cell], score_value)
            control_rows.append({
                "cell_id": cell, "dataset_family": bundles[cell].dataset_family,
                "task_type": bundles[cell].task_type, "method": control,
                "auroc": auc, "auprc": ap,
            })
    summary = {}
    b3_macro = aggregate(per_cell, "B3")["equal_family_macro"]
    g3_macro = aggregate(per_cell, "G3")["equal_family_macro"]
    failures = {"lambda_zero": not lambda_zero_pass}
    for control in controls:
        macro = aggregate(control_rows, control)["equal_family_macro"]
        summary[control] = {
            "equal_family_auroc": macro,
            "equal_family_delta_vs_B3": float(macro - b3_macro),
            "G3_advantage": float(g3_macro - macro),
        }
        failures[control] = bool(g3_macro <= macro)
    for name, arm in (("uniform", "G2"), ("raw", "G1")):
        macro = aggregate(per_cell, arm)["equal_family_macro"]
        summary[name] = {
            "equal_family_auroc": macro,
            "equal_family_delta_vs_B3": float(macro - b3_macro),
            "G3_advantage": float(g3_macro - macro),
        }
    return {
        "lambda_zero_exact": lambda_zero_pass,
        "lambda_zero_rows": lambda_zero_rows,
        "per_cell": control_rows, "summary": summary, "failures": failures,
        "all_required_pass": not any(failures.values()),
    }


def geometry_advantage_summary(geometry: list[dict], *, draws: int = 10_000) -> dict:
    by_cell = defaultdict(lambda: defaultdict(list))
    family_by_cell = {}
    for row in geometry:
        if row["graph_role"] in {"B0_LINEAR", "G3"}:
            by_cell[row["cell_id"]][row["graph_role"]].append(float(row["min_conditional_effect"]))
            family_by_cell[row["cell_id"]] = row["dataset_family"]
    cell_delta = {
        cell: float(np.mean(values["G3"]) - np.mean(values["B0_LINEAR"]))
        for cell, values in by_cell.items() if values["G3"] and values["B0_LINEAR"]
    }
    by_family = defaultdict(list)
    for cell, value in cell_delta.items():
        by_family[family_by_cell[cell]].append(value)
    family_delta = {family: float(np.mean(values)) for family, values in by_family.items()}
    generator = np.random.Generator(np.random.PCG64(20260821))
    families = sorted(family_delta)
    distribution = np.empty(draws)
    for index in range(draws):
        sampled = generator.choice(families, size=len(families), replace=True)
        distribution[index] = float(np.mean([family_delta[family] for family in sampled]))
    return {
        "observed": float(np.mean(list(family_delta.values()))),
        "lower": float(np.quantile(distribution, .025)),
        "upper": float(np.quantile(distribution, .975)),
        "family_delta": family_delta,
        "cell_delta": cell_delta,
    }


def decide(per_cell, summaries, comparisons, null, stability, graph_rows, controls, geometry, nuisance) -> dict:
    b0 = next(row for row in summaries if row["method"] == "B0" and row["metric"] == "auroc")
    b3 = next(row for row in summaries if row["method"] == "B3" and row["metric"] == "auroc")
    b3_vs_b0 = comparisons[("B0", "B3")]
    gate_a = bool(
        b3_vs_b0["lower"] > -0.0025
        and b3["qa_macro"] >= b0["qa_macro"] - 0.005
        and b3["math_macro"] >= b0["math_macro"] - 0.005
        and min(stability[cell]["B3"]["median_abs_spearman"] for cell in stability) >= 0.90
    )
    graph_gates = {}
    for method in ("G3", "G4"):
        summary = next(row for row in summaries if row["method"] == method and row["metric"] == "auroc")
        comparison = comparisons[("B3", method)]
        wins = win_table(per_cell, "B3", method)
        by_cell_health = defaultdict(list)
        for row in graph_rows:
            if row["arm_id"] == method:
                by_cell_health[row["cell_id"]].append(
                    str(row["healthy"]).lower() == "true"
                    and str(row["fold_artifact_healthy"]).lower() == "true"
                )
        healthy_cells = {
            cell for cell, values in by_cell_health.items()
            if len(values) == 5 and all(values)
        }
        graph_gates[method] = bool(
            comparison["observed"] >= 0.005 and comparison["lower"] > 0
            and summary["qa_macro"] >= b3["qa_macro"] - 0.005
            and summary["math_macro"] >= b3["math_macro"] - 0.005
            and wins["wins"] + wins["ties"] >= 14 and wins["worst_cell_delta"] >= -0.02
            and len(healthy_cells) >= 22
        )
    null_pass = all(value <= 0.05 for value in null["p_values"].values())
    residual_specificity = bool(
        graph_gates["G3"]
        and comparisons[("G1", "G3")]["observed"] >= 0.0025
        and comparisons[("G2", "G3")]["observed"] >= 0.0025
        and null_pass and controls["all_required_pass"]
    )
    geometry_exact = bool(null["p_values"]["exact"] <= .05)
    geometry_crt = bool(null["p_values"]["crt"] <= .05)
    linear = geometry_advantage_summary(geometry)
    linear_advantage = float(linear["observed"])
    gate_c = bool(
        residual_specificity and geometry_exact and geometry_crt
        and linear_advantage >= .02 and float(linear["lower"]) > 0
    )
    nuisance_diagnostics_pass = bool(nuisance) and all(
        row["nuisance_variance_min"] > 1e-6
        and row["whitening_max_abs"] <= .10
        and row["dependence_G4"] < row["dependence_B3"]
        and row["dependence_G4"] < row["dependence_G3"]
        for row in nuisance
    )
    gate_d = bool(
        graph_gates["G4"]
        and (
            comparisons[("G3", "G4")]["observed"] >= 0.0025
            or (
                nuisance_diagnostics_pass
                and comparisons[("G3", "G4")]["observed"] >= -0.0025
            )
        )
        and null_pass and controls["all_required_pass"] and nuisance_diagnostics_pass
    )
    mechanical_failure = bool(not controls["all_required_pass"])
    if mechanical_failure:
        decision = "MECHANICAL_OR_CONTROL_FAILURE_INVALIDATES_DEEM_GRAPH_AUDIT"
    elif not gate_a:
        decision = "DEEM_BASELINE_NOT_STABLE_STOP"
    elif b3_vs_b0["observed"] <= 0:
        decision = "NO_DEEM_ADVANTAGE_ON_ORIGINAL_24"
    elif gate_c or gate_d:
        decision = "INTERNAL_RESIDUAL_GRAPH_DEEM_CANDIDATE_AWAITING_NEW_GLOBAL_VALIDATION"
    elif residual_specificity and linear_advantage < .02:
        decision = "RESIDUAL_GRAPH_IS_LINEAR_DIRECTION_NOT_MANIFOLD"
    elif graph_gates["G3"] or graph_gates["G4"] or not controls["all_required_pass"]:
        decision = "RESIDUAL_GRAPH_MECHANISM_NOT_SUPPORTED"
    else:
        decision = "STABLE_DEEM_REPLACEMENT_WITHOUT_GRAPH_GAIN"
    return {
        "primary_decision": decision,
        "gate_a": gate_a,
        "gate_b_g3": graph_gates["G3"],
        "gate_b_g4": graph_gates["G4"],
        "gate_c": gate_c,
        "gate_d": gate_d,
        "graph_healthy_cells_g3": sum(
            len(values) == 5 and all(values) for cell, values in (
                (cell, [str(row["healthy"]).lower() == "true" and str(row["fold_artifact_healthy"]).lower() == "true"
                        for row in graph_rows if row["arm_id"] == "G3" and row["cell_id"] == cell])
                for cell in {row["cell_id"] for row in graph_rows if row["arm_id"] == "G3"}
            )
        ),
        "graph_healthy_cells_g4": sum(
            len(values) == 5 and all(values) for cell, values in (
                (cell, [str(row["healthy"]).lower() == "true" and str(row["fold_artifact_healthy"]).lower() == "true"
                        for row in graph_rows if row["arm_id"] == "G4" and row["cell_id"] == cell])
                for cell in {row["cell_id"] for row in graph_rows if row["arm_id"] == "G4"}
            )
        ),
        "controls_pass": bool(controls["all_required_pass"]),
        "geometry_exact_pass": geometry_exact,
        "geometry_crt_pass": geometry_crt,
        "mean_min_advantage_vs_linear": linear_advantage,
        "linear_advantage_lower": float(linear["lower"]),
        "nuisance_diagnostics_pass": nuisance_diagnostics_pass,
        "advance_core": bool(gate_a and not mechanical_failure),
        "advance_graph": bool(gate_a and not mechanical_failure and (gate_c or gate_d)),
        "eligible_for_B999": bool(gate_a and (gate_c or gate_d) and null_pass),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--sidecar-dir", type=Path, required=True)
    parser.add_argument("--phase0-complete", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/residual_graph_deem_24cell_v1_registry.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--B", type=int, choices=(199, 999), default=199)
    parser.add_argument("--promotion-decision", type=Path,
                        help="required B=199 DECISION.json when requesting B=999")
    args = parser.parse_args()
    run_dir, out = args.run_dir.resolve(), args.out_dir.resolve()
    freeze = verify_score_freeze(run_dir)
    registry = load_registry(args.registry)
    phase0 = json.loads(args.phase0_complete.read_text(encoding="utf-8"))
    if phase0.get("status") != "pass" or phase0.get("smoke"):
        raise SystemExit("evaluation requires a full passing Phase-0 freeze")
    if args.B == 999:
        if args.promotion_decision is None:
            raise SystemExit("B=999 requires --promotion-decision from the B=199 evaluation")
        promoted = json.loads(args.promotion_decision.read_text(encoding="utf-8"))
        if not promoted.get("eligible_for_B999"):
            raise SystemExit("B=999 forbidden: the B=199 decision did not pass every promotion gate")
    cells = [cell["cell_id"] for cell in registry["cells"]]
    if sorted(freeze["cells"]) != sorted(cells):
        raise ResidualGraphDeemError("score freeze does not contain exact 24-cell roster")
    bundles, targets, scores_by_cell, stability = {}, {}, {}, {}
    per_cell = []
    for cell in cells:
        bundle = load_target_free_bundle(args.bundle_dir / f"{cell}.npz")
        sidecar = load_label_sidecar(args.sidecar_dir / f"{cell}.npz")
        y_h = join_labels_by_id(bundle, sidecar)
        scores, stable = method_scores(run_dir, cell, phase0["nominated_lambdas"])
        bundles[cell], targets[cell], scores_by_cell[cell], stability[cell] = bundle, y_h, scores, stable
        for method, score in scores.items():
            auroc, auprc = metrics(y_h, score)
            per_cell.append({
                "cell_id": cell, "dataset_family": bundle.dataset_family,
                "task_type": bundle.task_type, "method": method,
                "auroc": auroc, "auprc": auprc, "n": len(y_h),
                "n_hallucination": int(y_h.sum()),
            })
    per_fit, reconstruction, graph_rows, gate_rows = collect_fit_diagnostics(
        run_dir, cells, phase0["nominated_lambdas"]
    )
    residual_rows = residual_diagnostics(run_dir, bundles, cells)
    nuisance_rows = nuisance_diagnostics(run_dir, cells, phase0["nominated_lambdas"])
    sensitivity_rows = []
    for cell in cells:
        for sensitivity, arm, stems in (
            ("stable_inventory_minus4", "B0", [f"SENSITIVITY__stable_inventory_minus4__B0__seed{seed}" for seed in SEEDS]),
            ("stable_inventory_minus4", "B3", [f"SENSITIVITY__stable_inventory_minus4__B3__seed{seed}" for seed in SEEDS]),
            ("stable_inventory_minus4", "G3", [f"SENSITIVITY__stable_inventory_minus4__G3__seed{seed}" for seed in SEEDS]),
            *[
                (f"k{k}", "G3", [f"SENSITIVITY__k{k}__G3__lambda{lambda_token(phase0['nominated_lambdas']['target'])}__seed{seed}" for seed in SEEDS])
                for k in (5, 10, 15)
            ],
        ):
            value = np.mean([load_seed_score(run_dir, cell, stem) for stem in stems], axis=0)
            auroc, auprc = metrics(targets[cell], value)
            sensitivity_rows.append({
                "cell_id": cell, "dataset_family": bundles[cell].dataset_family,
                "task_type": bundles[cell].task_type, "sensitivity": sensitivity,
                "method": arm, "auroc": auroc, "auprc": auprc,
            })
    lambda_rows = []
    for cell in cells:
        for arm in ("G0", "G1", "G2", "G3", "G4", "G5"):
            for lambda_ in (0.0, .01, .03, .1, .3, 1.0):
                value, _ = ensemble(run_dir, cell, arm, lambda_)
                auroc, auprc = metrics(targets[cell], value)
                lambda_rows.append({
                    "cell_id": cell, "dataset_family": bundles[cell].dataset_family,
                    "task_type": bundles[cell].task_type, "method": arm,
                    "lambda": lambda_, "auroc": auroc, "auprc": auprc,
                    "headline": float(lambda_) == float(
                        phase0["nominated_lambdas"]["nuisance" if arm == "G4" else "family" if arm == "G5" else "target"]
                    ),
                })
    methods = [spec.arm_id for spec in ARM_SPECS]
    summaries = [aggregate(per_cell, method, metric) for method in methods for metric in ("auroc", "auprc")]
    comparisons = {}
    for reference, candidate in (("B0", "B3"), ("B3", "G2"), ("B3", "G3"),
                                 ("B3", "G4"), ("G1", "G3"), ("G2", "G3"),
                                 ("G3", "G4")):
        comparisons[(reference, candidate)] = family_bootstrap(per_cell, reference, candidate)
    draws, conditional_diagnostics = conditional_draws(targets, bundles, B=args.B)
    graphs = graph_catalog(run_dir, cells, phase0["nominated_lambdas"])
    geometry = geometry_rows(targets, bundles, graphs, draws)
    null = whole_search_null(targets, bundles, scores_by_cell, graphs, draws, B=args.B)
    null["conditional_diagnostics"] = conditional_diagnostics
    controls = evaluate_controls(
        run_dir, cells, targets, bundles, phase0["nominated_lambdas"], per_cell
    )
    decision = decide(
        per_cell, summaries, comparisons, null, stability, graph_rows,
        controls, geometry, nuisance_rows,
    )
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "PER_CELL.csv", per_cell)
    write_csv(out / "PER_FIT.csv", per_fit)
    write_csv(out / "CONTRIBUTION_RECONSTRUCTION.csv", reconstruction)
    write_csv(out / "RESIDUAL_DIAGNOSTICS.csv", residual_rows)
    write_csv(out / "NUISANCE_DIAGNOSTICS.csv", nuisance_rows)
    write_csv(out / "GRAPH_HEALTH.csv", graph_rows)
    write_csv(out / "GATE_STABILITY.csv", gate_rows)
    write_csv(out / "CONDITIONAL_GEOMETRY.csv", geometry)
    write_csv(out / "SENSITIVITY.csv", sensitivity_rows)
    write_csv(out / "LAMBDA_SENSITIVITY.csv", lambda_rows)
    flat_summary = [{key: value for key, value in row.items() if key != "family_means"} for row in summaries]
    write_csv(out / "FAMILY_SUMMARY.csv", flat_summary)
    comparison_rows = []
    for value in comparisons.values():
        comparison_rows.append({key: item for key, item in value.items() if key not in {"distribution", "family_delta", "leave_one_family_out"}})
    write_csv(out / "PAIRWISE_COMPARISONS.csv", comparison_rows)
    atomic_write_json(out / "BOOTSTRAP.json", {f"{a}_vs_{b}": value for (a, b), value in comparisons.items()})
    atomic_write_json(out / "WHOLE_SEARCH_NULL.json", null)
    atomic_write_json(out / "CONTROLS.json", controls)
    atomic_write_json(out / "SEED_STABILITY.json", stability)
    atomic_write_json(out / "DECISION.json", decision)
    atomic_write_json(out / "EVALUATION_COMPLETE.json", {
        "status": "complete", "B": args.B, "decision": decision,
        "score_freeze_sha256": sha256_file(run_dir / "SCORE_FREEZE_MANIFEST.json"),
        "sidecar_set": str(args.sidecar_dir),
        "nominated_lambdas": phase0["nominated_lambdas"],
    })
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
