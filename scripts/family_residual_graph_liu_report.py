#!/usr/bin/env python3
"""Open labels only after fit freeze and evaluate Family-residual graph LIU."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.family_residual_graph_liu_fit import (  # noqa: E402
    DEFAULT_OUT,
    VERSION,
    canonical_hash,
    score_key,
    sha256_file,
    write_json,
)
from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family,
)
from scripts.inscope_cells import INSCOPE  # noqa: E402


MIN_POSITIVES = 20
BOOTSTRAPS = 20000
DEFAULT_KEY = score_key("cs", 0.5, 0.5, 7, 0.1, 1.0)
INCUMBENT_KEY = score_key("u2", 0.0, 0.5, 7, 0.1)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_and_freeze(fit_dir, bundle):
    """Verify the label-free artifacts and write a label-opening barrier."""
    definition = json.loads((fit_dir / "RUN_DEFINITION.json").read_text())
    complete = json.loads((fit_dir / "FIT_COMPLETE.json").read_text())
    configs = json.loads((fit_dir / "CONFIG_INDEX.json").read_text())
    if definition["version"] != VERSION or complete["version"] != VERSION:
        raise RuntimeError("fit/report version mismatch")
    if not complete["scientific_run"]:
        raise RuntimeError("debug scores cannot be reported")
    if complete["roster"] != list(INSCOPE):
        raise RuntimeError("fit did not freeze the registered roster")
    if definition["bundle_sha256"] != sha256_file(bundle):
        raise RuntimeError("bundle changed since label-free fitting")
    if complete["definition_hash"] != definition["definition_hash"]:
        raise RuntimeError("run definition hash mismatch")
    definition_payload = dict(definition)
    recorded_definition_hash = definition_payload.pop("definition_hash")
    if canonical_hash(definition_payload) != recorded_definition_hash:
        raise RuntimeError("run definition payload is not self-consistent")
    complete_payload = dict(complete)
    recorded_manifest_hash = complete_payload.pop("manifest_hash")
    if canonical_hash(complete_payload) != recorded_manifest_hash:
        raise RuntimeError("fit completion manifest is not self-consistent")
    current_sources = {
        "fit_script": REPO / "scripts" / "family_residual_graph_liu_fit.py",
        "report_script": REPO / "scripts" / "family_residual_graph_liu_report.py",
        "controls_script": REPO / "scripts" / "family_residual_graph_liu_controls.py",
        "prmbench_script": REPO / "scripts" / "family_residual_graph_liu_prmbench.py",
        "hle_script": REPO / "scripts" / "family_residual_graph_liu_hle.py",
        "hard_filter_contract_script": REPO / "scripts" / "hard_filter_dufs_liu_benchmark.py",
        "inscope_roster_script": REPO / "scripts" / "inscope_cells.py",
        "dufs_trainer_module": REPO / "spectral_utils" / "selectors" / "a2_groupfs.py",
        "transfer_contract_script": REPO / "scripts" / "leverage_balanced_processbench_transfer.py",
        "prmbench_loader_script": REPO / "scripts" / "neutral_residual_mode_prmbench_confirmation.py",
        "hle_loader_script": REPO / "scripts" / "neutral_residual_mode_hle_confirmation.py",
        "core_module": REPO / "spectral_utils" / "family_residual_graph.py",
        "contribution_module": REPO / "spectral_utils" / "contribution_subspace.py",
        "graph_topology_module": REPO / "spectral_utils" / "graph_topology.py",
        "laplacian_module": REPO / "spectral_utils" / "laplacian_upcr.py",
        "upcr_module": REPO / "spectral_utils" / "upcr.py",
        "feature_contract_module": REPO / "spectral_utils" / "dufs_liu_feature_contract.py",
        "family_registry_module": REPO / "spectral_utils" / "specrage_views.py",
        "fusion_utils_module": REPO / "spectral_utils" / "fusion_utils.py",
        "base_feature_contract_module": REPO / "spectral_utils" / "feature_contract.py",
        "feature_utils_module": REPO / "spectral_utils" / "feature_utils.py",
        "repgrid_scoring_module": REPO / "spectral_utils" / "repgrid_scoring.py",
        "spec": REPO / "docs" / "experiments" / "FAMILY_RESIDUAL_GRAPH_LIU_V3.md",
        "base_v2_spec": REPO / "docs" / "experiments" / "FAMILY_RESIDUAL_GRAPH_LIU_V2.md",
    }
    source_checks = {
        name: sha256_file(path) == definition["sources"].get(name)
        for name, path in current_sources.items()
    }
    if not all(source_checks.values()):
        raise RuntimeError(f"fitted source changed: {source_checks}")
    nrm_comparator = (
        REPO / "results" / "neutral_residual_mode_cs_iu_v1"
        / "cell_results.csv"
    )
    if sha256_file(nrm_comparator) != definition["comparators"].get(
        "family_nrm_cell_results_sha256"
    ):
        raise RuntimeError("Family-NRM comparator changed")
    for cell in INSCOPE:
        path = fit_dir / "scores" / f"{cell}.npz"
        if sha256_file(path) != complete["score_hashes"].get(cell):
            raise RuntimeError(f"score hash mismatch for {cell}")
        diagnostic_path = fit_dir / "diagnostics" / f"{cell}.json"
        if sha256_file(diagnostic_path) != complete["diagnostic_hashes"].get(cell):
            raise RuntimeError(f"diagnostic hash mismatch for {cell}")
    if sha256_file(fit_dir / "CONFIG_INDEX.json") != complete.get(
        "config_index_sha256"
    ):
        raise RuntimeError("configuration index hash mismatch")
    freeze = {
        "version": VERSION,
        "definition_hash": definition["definition_hash"],
        "fit_manifest_hash": complete["manifest_hash"],
        "bundle_sha256": definition["bundle_sha256"],
        "score_hashes": complete["score_hashes"],
        "diagnostic_hashes": complete["diagnostic_hashes"],
        "config_index_sha256": complete["config_index_sha256"],
        "configuration_count": len(configs),
        "labels_opened_only_after_this_manifest": True,
    }
    freeze["freeze_hash"] = canonical_hash(freeze)
    path = fit_dir / "SCORE_FREEZE_MANIFEST.json"
    if path.exists():
        if json.loads(path.read_text()) != freeze:
            raise RuntimeError("existing score freeze differs")
    else:
        write_json(path, freeze)
    return configs, freeze


def selectable(config):
    if config.get("readout") not in {"u2", "cs"}:
        return False
    if float(config.get("lambda", 0)) <= 0:
        return False
    eta = float(config.get("eta", 0))
    beta = float(config.get("beta", 1))
    return eta > 0 and eta * (1 - beta) > 0


def graph_health_registry(fit_dir):
    """Freeze label-free numerical validity before hyperparameter search."""
    status = {}
    counts = {}
    for cell in INSCOPE:
        diagnostics = json.loads(
            (fit_dir / "diagnostics" / f"{cell}.json").read_text()
        )
        for row in diagnostics["graphs"]:
            key = (
                str(row["topology"]), float(row["eta"]),
                float(row["beta"]), int(row["k"]),
            )
            healthy = bool(
                bool(row["all_edge_weights_finite"])
                and float(row["minimum_edge_weight"]) >= 0.0
                and float(row["isolated_fraction"]) == 0.0
                and np.isfinite(float(row["degree_min"]))
                and float(row["degree_min"]) > 0
                and np.isfinite(float(row["degree_mean"]))
                and float(row["degree_mean"]) > 0
                and float(row["graph_symmetry_error"]) <= 1e-12
            )
            status[key] = status.get(key, True) and healthy
            counts[key] = counts.get(key, 0) + 1
    if not all(count == len(INSCOPE) for count in counts.values()):
        raise RuntimeError("graph-health registry is incomplete")
    return {key for key, healthy in status.items() if healthy}, {
        "registered_graph_settings": len(status),
        "eligible_graph_settings": int(sum(status.values())),
        "criterion": (
            "symmetric finite nonnegative weights, finite positive minimum/mean "
            "degree, and zero isolated nodes in all 24 cells; connectivity is "
            "a separate mechanism gate"
        ),
    }


def load_metrics(fit_dir, bundle, configs, healthy_graphs):
    candidates = tuple(
        key for key, value in configs.items()
        if selectable(value) and (
            str(value["topology"]), float(value["eta"]),
            float(value["beta"]), int(value["k"])
        ) in healthy_graphs
    )
    mechanism_keys = {
        key for key, value in configs.items()
        if value.get("readout") in {"u2", "cs"}
        and float(value.get("lambda", 0)) > 0
    }
    needed = mechanism_keys | {"iu", "cardinality", DEFAULT_KEY, INCUMBENT_KEY}
    rows = []
    auc = {}
    auprc = {}
    with np.load(bundle, allow_pickle=True) as data:
        for cell in INSCOPE:
            labels = np.asarray(data[f"{cell}__labels"], dtype=int)
            positives = int(labels.sum())
            if positives < MIN_POSITIVES:
                continue
            with np.load(fit_dir / "scores" / f"{cell}.npz") as scores:
                if not np.array_equal(
                    scores["sample_index"], np.arange(len(labels))
                ):
                    raise RuntimeError(f"sample alignment failed for {cell}")
                for key in needed:
                    values = np.asarray(scores[key], dtype=float)
                    auc[(cell, key)] = float(roc_auc_score(labels, values))
                    if key in {"iu", "cardinality", DEFAULT_KEY, INCUMBENT_KEY}:
                        auprc[(cell, key)] = float(
                            average_precision_score(labels, values)
                        )
            rows.append({
                "cell": cell,
                "dataset_family": family(cell),
                "n": len(labels),
                "n_correct": positives,
                "iu_auroc": auc[(cell, "iu")],
                "default_auroc": auc[(cell, DEFAULT_KEY)],
                "default_delta_pp": 100 * (
                    auc[(cell, DEFAULT_KEY)] - auc[(cell, "iu")]
                ),
                "incumbent_auroc": auc[(cell, INCUMBENT_KEY)],
                "incumbent_delta_pp": 100 * (
                    auc[(cell, INCUMBENT_KEY)] - auc[(cell, "iu")]
                ),
            })
    return rows, auc, auprc, candidates


def family_deltas(cells, keys, auc):
    grouped = {}
    for cell in cells:
        grouped.setdefault(family(cell), []).append(cell)
    output = {key: {} for key in keys}
    for key in keys:
        for group, members in grouped.items():
            output[key][group] = float(np.mean([
                auc[(cell, key)] - auc[(cell, "iu")] for cell in members
            ]))
    return output


def complexity_tuple(key, config, mean_delta):
    eta = float(config["eta"])
    beta = float(config["beta"])
    active = int(eta < 1) + int(eta * beta > 0) + int(eta * (1 - beta) > 0)
    return (
        active,
        0 if config["topology"] == "union" else 1,
        0 if config["readout"] == "u2" else 1,
        float(config["lambda"]),
        abs(int(config["k"]) - 7),
        abs(eta - 0.5) + abs(beta - 0.5),
        abs(float(config.get("trust_factor", 1.0)) - 1.0),
        -mean_delta,
        key,
    )


def select_one_se(keys, training_families, deltas, configs):
    stats = {}
    for key in keys:
        values = np.asarray([deltas[key][g] for g in training_families])
        stats[key] = {
            "mean": float(np.mean(values)),
            "se": float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else 0.0,
            "worst": float(np.min(values)),
        }
    best = max(keys, key=lambda key: (stats[key]["mean"], key))
    threshold = stats[best]["mean"] - stats[best]["se"]
    eligible = [key for key in keys if stats[key]["mean"] >= threshold]
    tail_safe = [key for key in eligible if stats[key]["worst"] >= -0.005]
    pool = tail_safe or eligible
    chosen = min(
        pool,
        key=lambda key: complexity_tuple(key, configs[key], stats[key]["mean"]),
    )
    return chosen, {
        "best_key": best,
        "best_mean": stats[best]["mean"],
        "best_se": stats[best]["se"],
        "one_se_threshold": threshold,
        "eligible_count": len(eligible),
        "tail_safe_count": len(tail_safe),
        "chosen_training_mean": stats[chosen]["mean"],
        "chosen_training_worst": stats[chosen]["worst"],
    }


def matched_key(config, readout, *, eta=None, beta=None, trust_factor=None):
    eta = float(config["eta"] if eta is None else eta)
    beta = float(config["beta"] if beta is None else beta)
    if readout == "cs":
        trust = float(
            config.get("trust_factor", 1.0)
            if trust_factor is None else trust_factor
        )
        return score_key(
            "cs", eta, beta, config["k"], config["lambda"], trust,
            topology=config["topology"],
        )
    return score_key(
        readout, eta, beta, config["k"], config["lambda"],
        topology=config["topology"],
    )


def bootstrap_ci(values, seed=20260822):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(BOOTSTRAPS, len(values)), replace=True)
    means = np.mean(draws, axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def nrm_family_deltas(cells):
    path = REPO / "results" / "neutral_residual_mode_cs_iu_v1" / "cell_results.csv"
    by_cell = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["regime"] == "original_lofo":
                by_cell[row["cell"]] = float(row["nrm_delta_pp"]) / 100.0
    output = {}
    for group in sorted({family(cell) for cell in cells}):
        members = [cell for cell in cells if family(cell) == group]
        if not all(cell in by_cell for cell in members):
            raise RuntimeError(f"NRM comparator missing cells for {group}")
        output[group] = float(np.mean([by_cell[cell] for cell in members]))
    return output


def mean_delta_for_family(key, group, cells, auc):
    members = [cell for cell in cells if family(cell) == group]
    return float(np.mean([
        auc[(cell, key)] - auc[(cell, "iu")] for cell in members
    ]))


def attach_outer_auprc(outer_rows, cells, fit_dir, bundle):
    """Evaluate only the eight frozen held-family choices on secondary AUPRC."""
    family_values = []
    with np.load(bundle, allow_pickle=True) as data:
        for row in outer_rows:
            held = row["held_family"]
            deltas = []
            for cell in cells:
                if family(cell) != held:
                    continue
                labels = np.asarray(data[f"{cell}__labels"], dtype=int)
                with np.load(fit_dir / "scores" / f"{cell}.npz") as scores:
                    deltas.append(
                        average_precision_score(labels, scores[row["selected_key"]])
                        - average_precision_score(labels, scores["iu"])
                    )
            value = float(np.mean(deltas))
            row["held_auprc_delta_pp"] = 100 * value
            family_values.append(value)
    return np.asarray(family_values, dtype=float)


def selected_graph_health(fit_dir, config):
    rows = []
    for cell in INSCOPE:
        diagnostics = json.loads(
            (fit_dir / "diagnostics" / f"{cell}.json").read_text()
        )
        matches = [row for row in diagnostics["graphs"] if (
            str(row["topology"]) == str(config["topology"])
            and
            float(row["eta"]) == float(config["eta"])
            and float(row["beta"]) == float(config["beta"])
            and int(row["k"]) == int(config["k"])
        )]
        if len(matches) != 1:
            raise RuntimeError(f"selected graph diagnostic mismatch for {cell}")
        rows.append(matches[0])
    mechanism_healthy = [
        float(row["largest_component_fraction"]) >= 0.90
        and float(row["isolated_fraction"]) <= 0.05
        for row in rows
    ]
    return {
        "connected_cells": int(sum(row["n_components"] == 1 for row in rows)),
        "n_cells": len(rows),
        "minimum_degree": float(min(row["degree_min"] for row in rows)),
        "minimum_algebraic_connectivity": float(min(
            row["algebraic_connectivity"] for row in rows
        )),
        "maximum_component_excess_fraction": float(max(
            (row["n_components"] - 1) / row["n_nodes"] for row in rows
        )),
        "maximum_isolated_fraction": float(max(
            row["isolated_fraction"] for row in rows
        )),
        "minimum_largest_component_fraction": float(min(
            row["largest_component_fraction"] for row in rows
        )),
        "mechanism_healthy_cells": int(sum(mechanism_healthy)),
        "mechanism_healthy_cell_fraction": float(np.mean(mechanism_healthy)),
        "mechanism_health_criterion": (
            "largest component >=90% and isolated fraction <=5%; at least "
            "90% of cells required for mechanism promotion"
        ),
        "all_symmetric": bool(all(
            row["graph_symmetry_error"] <= 1e-12 for row in rows
        )),
    }


def nested_selector_summary(keys, families, deltas, configs, *, seed):
    held_values = []
    selections = []
    for held in families:
        training = [group for group in families if group != held]
        chosen, _ = select_one_se(keys, training, deltas, configs)
        held_values.append(deltas[chosen][held])
        selections.append(chosen)
    values = np.asarray(held_values, dtype=float)
    final_key, final_diag = select_one_se(keys, families, deltas, configs)
    return {
        "candidate_count": len(keys),
        "nested_delta_vs_iu_pp": 100 * float(np.mean(values)),
        "nested_delta_vs_iu_ci_pp": [
            100 * x for x in bootstrap_ci(values, seed=seed)
        ],
        "positive_families": int(np.sum(values > 0)),
        "worst_family_pp": 100 * float(np.min(values)),
        "selected_keys_by_fold": selections,
        "final_key": final_key,
        "final_config": configs[final_key],
        "final_selection_diagnostics": final_diag,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--fit-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    configs, freeze = verify_and_freeze(args.fit_dir, args.bundle)
    healthy_graphs, health_filter = graph_health_registry(args.fit_dir)
    cell_rows, auc, _, all_candidates = load_metrics(
        args.fit_dir, args.bundle, configs, healthy_graphs
    )
    candidates = tuple(
        key for key in all_candidates if configs[key]["topology"] == "union"
    )
    adaptive_candidates = tuple(
        key for key in all_candidates if configs[key]["topology"] == "adaptive"
    )
    if not candidates:
        raise RuntimeError("no healthy union-kNN candidate survived prefilter")
    if not adaptive_candidates:
        raise RuntimeError("no healthy adaptive-kNN candidate survived prefilter")
    cells = [row["cell"] for row in cell_rows]
    families = sorted({family(cell) for cell in cells})
    deltas = family_deltas(cells, all_candidates, auc)
    nrm = nrm_family_deltas(cells)

    outer_rows = []
    outer_deltas = []
    full_vs_u2 = []
    full_vs_dufs_same = []
    for held in families:
        training = [group for group in families if group != held]
        chosen, selection_diag = select_one_se(
            candidates, training, deltas, configs
        )
        config = configs[chosen]
        held_delta = deltas[chosen][held]
        same_graph_u2 = matched_key(config, "u2")
        same_graph_cs = matched_key(config, "cs", trust_factor=1.0)
        dufs_same = matched_key(config, config["readout"], eta=0.0, beta=0.5)
        row = {
            "held_family": held,
            "selected_key": chosen,
            **config,
            "held_delta_pp": 100 * held_delta,
            "held_nrm_delta_pp": 100 * nrm[held],
            "held_recovery_fraction": held_delta / nrm[held]
            if abs(nrm[held]) > 1e-12 else "",
            "same_graph_u2_delta_pp": 100 * mean_delta_for_family(
                same_graph_u2, held, cells, auc
            ),
            "same_graph_cs_t1_delta_pp": 100 * mean_delta_for_family(
                same_graph_cs, held, cells, auc
            ),
            "dufs_same_readout_delta_pp": 100 * mean_delta_for_family(
                dufs_same, held, cells, auc
            ),
            **selection_diag,
        }
        outer_rows.append(row)
        outer_deltas.append(held_delta)
        full_vs_u2.append(
            held_delta - mean_delta_for_family(same_graph_u2, held, cells, auc)
        )
        full_vs_dufs_same.append(
            held_delta - mean_delta_for_family(dufs_same, held, cells, auc)
        )

    final_key, final_diag = select_one_se(candidates, families, deltas, configs)
    topology_sensitivities = {
        "adaptive_only": nested_selector_summary(
            adaptive_candidates, families, deltas, configs, seed=20260827
        ),
        "union_plus_adaptive": nested_selector_summary(
            all_candidates, families, deltas, configs, seed=20260828
        ),
    }
    outer_auprc = attach_outer_auprc(
        outer_rows, cells, args.fit_dir, args.bundle
    )
    default_family = family_deltas(cells, (DEFAULT_KEY,), auc)[DEFAULT_KEY]
    incumbent_family = family_deltas(cells, (INCUMBENT_KEY,), auc)[INCUMBENT_KEY]
    outer = np.asarray(outer_deltas)
    nrm_values = np.asarray([nrm[group] for group in families])
    d50 = outer - 0.5 * nrm_values
    d30 = outer - 0.3 * nrm_values
    selected_health = selected_graph_health(
        args.fit_dir, configs[final_key]
    )
    summary = {
        "version": VERSION,
        "design": "nested leave-one-dataset-family-out retrospective development",
        "n_cells": len(cells),
        "n_dataset_families": len(families),
        "candidate_count": len(candidates),
        "all_topology_candidate_count": len(all_candidates),
        "graph_health_prefilter": health_filter,
        "score_freeze_hash": freeze["freeze_hash"],
        "nested_delta_vs_iu_pp": 100 * float(np.mean(outer)),
        "nested_delta_vs_iu_ci_pp": [100 * x for x in bootstrap_ci(outer)],
        "nested_auprc_delta_vs_iu_pp": 100 * float(np.mean(outer_auprc)),
        "nested_auprc_delta_vs_iu_ci_pp": [
            100 * x for x in bootstrap_ci(outer_auprc, seed=20260826)
        ],
        "positive_outer_families": int(np.sum(outer > 0)),
        "worst_outer_family_pp": 100 * float(np.min(outer)),
        "nonzero_selected_folds": int(sum(
            float(row["lambda"]) > 0 for row in outer_rows
        )),
        "fixed_default_delta_pp": 100 * float(np.mean(list(default_family.values()))),
        "fixed_default_ci_pp": [
            100 * x for x in bootstrap_ci(list(default_family.values()), seed=20260823)
        ],
        "incumbent_dufs_liu_delta_pp": 100 * float(
            np.mean(list(incumbent_family.values()))
        ),
        "family_nrm_delta_pp": 100 * float(np.mean(nrm_values)),
        "nrm_recovery_fraction": float(np.mean(outer) / np.mean(nrm_values)),
        "d50_point_pp": 100 * float(np.mean(d50)),
        "d50_ci_pp": [100 * x for x in bootstrap_ci(d50, seed=20260824)],
        "d30_point_pp": 100 * float(np.mean(d30)),
        "d30_ci_pp": [100 * x for x in bootstrap_ci(d30, seed=20260825)],
        "full_minus_same_graph_u2_pp": 100 * float(np.mean(full_vs_u2)),
        "full_minus_dufs_same_readout_pp": 100 * float(
            np.mean(full_vs_dufs_same)
        ),
        "promotion_gates": {
            "ci_lower_gt_zero": bootstrap_ci(outer)[0] > 0,
            "point_at_least_0p10pp": 100 * float(np.mean(outer)) >= 0.10,
            "six_of_eight_positive": int(np.sum(outer > 0)) >= 6,
            "worst_at_least_minus_0p50pp": 100 * float(np.min(outer)) >= -0.50,
            "six_nonzero_folds": int(sum(
                float(row["lambda"]) > 0 for row in outer_rows
            )) >= 6,
            "d30_lower_nonnegative": bootstrap_ci(d30, seed=20260825)[0] >= 0,
        },
        "selected_graph_health": selected_health,
        "connectivity_mechanism_gate_pass": (
            selected_health["mechanism_healthy_cells"] >= 22
        ),
        "topology_rescue_sensitivities": topology_sensitivities,
    }
    summary["promotion_pass"] = all(summary["promotion_gates"].values())
    frozen = {
        "version": VERSION,
        "score_freeze_hash": freeze["freeze_hash"],
        "selected_key": final_key,
        "selected_config": configs[final_key],
        "selection_diagnostics": final_diag,
        "fixed_default_key": DEFAULT_KEY,
        "nested_outer": outer_rows,
        "topology_rescue_sensitivities": topology_sensitivities,
    }
    frozen["selection_hash"] = canonical_hash(frozen)
    selection_path = args.fit_dir / "FROZEN_SELECTION.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text()) != frozen:
            raise RuntimeError("frozen selection already exists and differs")
    else:
        write_json(selection_path, frozen)
    write_json(args.fit_dir / "RESULT.json", summary)
    write_csv(args.fit_dir / "cell_summary.csv", cell_rows)
    write_csv(args.fit_dir / "nested_outer.csv", outer_rows)
    write_csv(args.fit_dir / "summary.csv", [{
        key: json.dumps(value) if isinstance(value, (dict, list)) else value
        for key, value in summary.items()
    }])

    ci = summary["nested_delta_vs_iu_ci_pp"]
    gate_lines = "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}"
        for name, passed in summary["promotion_gates"].items()
    )
    adaptive = topology_sensitivities["adaptive_only"]
    combined = topology_sensitivities["union_plus_adaptive"]
    report = f"""# Family-residual graph LIU v3 — post-audit development report

## Result

The nested leave-dataset-family-out procedure changed AUROC versus ordinary
IU-PCR by **{summary['nested_delta_vs_iu_pp']:+.3f}pp** (equal-family bootstrap
95% CI **[{ci[0]:+.3f}, {ci[1]:+.3f}]pp**), with
{summary['positive_outer_families']}/{len(families)} positive held-out families
and worst-family change {summary['worst_outer_family_pp']:+.3f}pp.

This primary is the strict self-safe **union-kNN bug repair**. The separately
reported adaptive-only sensitivity changed AUROC by
{adaptive['nested_delta_vs_iu_pp']:+.3f}pp (CI
[{adaptive['nested_delta_vs_iu_ci_pp'][0]:+.3f},
{adaptive['nested_delta_vs_iu_ci_pp'][1]:+.3f}]pp); allowing union and adaptive
to compete changed it by {combined['nested_delta_vs_iu_pp']:+.3f}pp (CI
[{combined['nested_delta_vs_iu_ci_pp'][0]:+.3f},
{combined['nested_delta_vs_iu_ci_pp'][1]:+.3f}]pp). These are retrospective
topology-rescue sensitivities, not new confirmation.

The fixed label-free default changed AUROC by
{summary['fixed_default_delta_pp']:+.3f}pp. The corrected union-kNN
DUFS-coordinate arm changed
it by {summary['incumbent_dufs_liu_delta_pp']:+.3f}pp, while frozen Family-NRM's
reference change was {summary['family_nrm_delta_pp']:+.3f}pp.  The nested
procedure recovered {100 * summary['nrm_recovery_fraction']:.1f}% of the NRM
point gain; `D_0.5` was {summary['d50_point_pp']:+.3f}pp and `D_0.3` was
{summary['d30_point_pp']:+.3f}pp.

Secondary AUPRC changed by
{summary['nested_auprc_delta_vs_iu_pp']:+.3f}pp under the same held-family
choices (95% family bootstrap
[{summary['nested_auprc_delta_vs_iu_ci_pp'][0]:+.3f},
{summary['nested_auprc_delta_vs_iu_ci_pp'][1]:+.3f}]pp).

## Mechanism attribution

Across held-out families, the selected arm minus the matched hybrid-graph U2
arm was {summary['full_minus_same_graph_u2_pp']:+.3f}pp.  Its advantage over
the same readout on the ordinary DUFS graph was
{summary['full_minus_dufs_same_readout_pp']:+.3f}pp.  Direct score diffusion is
evaluated only after selection in the fixed-control phase and cannot affect
this hyperparameter search.

The selected graph met the registered connectivity mechanism criterion in
{summary['selected_graph_health']['mechanism_healthy_cells']}/
{summary['selected_graph_health']['n_cells']} cells. Connectivity did not
filter utility HPO; it is a separate connectivity mechanism gate:
**{'PASS' if summary['connectivity_mechanism_gate_pass'] else 'FAIL'}**.
This connectivity gate is necessary but not sufficient for mechanism
promotion; the post-selection matched and permutation controls are still
required.

## Promotion gates

{gate_lines}

Utility promotion: **{'PASS' if summary['promotion_pass'] else 'FAIL'}**.

## Frozen development finalist

`{final_key}`

This is a retrospective development estimate: all fits and scores were frozen
without labels, but labels selected the final configuration across the original
eight dataset families.  Existing external datasets can test frozen transfer,
not prospective confirmation.
"""
    (args.fit_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
