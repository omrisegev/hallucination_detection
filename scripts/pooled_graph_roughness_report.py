#!/usr/bin/env python3
"""Label-facing nested report for Pooled Graph-Roughness Direction V1."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr, ttest_1samp, wilcoxon
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.hard_filter_dufs_liu_benchmark import (  # noqa: E402
    DEFAULT_BUNDLE,
    family as dataset_family,
)
from scripts.pooled_graph_roughness_fit import (  # noqa: E402
    DEFAULT_OUT,
    ELIGIBLE_CELLS,
    LAMBDAS,
    TRUST_FACTORS,
    VERSION,
    candidate_key,
    candidates,
    canonical_hash,
    run_definition,
    sha256_file,
    source_hashes,
    write_json,
)
from spectral_utils.pooled_graph_roughness import direction_cosine  # noqa: E402


BOOTSTRAPS = 200_000
BOOTSTRAP_SEED = 20260822
TAIL_FLOOR = -0.005


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_fit(out: Path, bundle: Path) -> dict:
    definition = json.loads((out / "RUN_DEFINITION.json").read_text())
    current = run_definition(bundle)
    if definition != current:
        raise RuntimeError("run definition or frozen source hash changed")
    if definition["version"] != VERSION:
        raise RuntimeError("run definition version changed")
    complete = json.loads((out / "FIT_COMPLETE.json").read_text())
    payload = dict(complete)
    recorded = payload.pop("manifest_hash")
    if canonical_hash(payload) != recorded:
        raise RuntimeError("fit manifest is not self-consistent")
    if complete["definition_hash"] != definition["definition_hash"]:
        raise RuntimeError("fit/run-definition mismatch")
    if complete.get("version") != VERSION:
        raise RuntimeError("fit manifest version changed")
    if complete.get("labels_accessed_by_fit") is not False:
        raise RuntimeError("fit manifest does not certify label-free fitting")
    if complete.get("target_fields_received_by_fit") != []:
        raise RuntimeError("fit manifest received target fields")
    if set(complete["state_hashes"]) != set(ELIGIBLE_CELLS):
        raise RuntimeError("state hash roster changed")
    if set(complete["score_hashes"]) != set(ELIGIBLE_CELLS):
        raise RuntimeError("score hash roster changed")
    for cell, expected in complete["state_hashes"].items():
        if sha256_file(out / "states" / f"{cell}.npz") != expected:
            raise RuntimeError(f"state hash changed: {cell}")
    for cell, expected in complete["score_hashes"].items():
        if sha256_file(out / "scores" / f"{cell}.npz") != expected:
            raise RuntimeError(f"score hash changed: {cell}")
    if sha256_file(out / "CALIBRATIONS.json") != complete["calibrations_sha256"]:
        raise RuntimeError("calibration hash changed")
    if sha256_file(out / "DIAGNOSTICS.json") != complete["diagnostics_sha256"]:
        raise RuntimeError("diagnostic hash changed")
    return complete


def load_after_freeze(out: Path, bundle: Path):
    scores = {}
    labels = {}
    with np.load(bundle, allow_pickle=True) as data:
        for cell in ELIGIBLE_CELLS:
            labels[cell] = np.asarray(data[f"{cell}__labels"], dtype=int)
            with np.load(out / "scores" / f"{cell}.npz") as stored:
                group = dataset_family(cell)
                other_groups = sorted({
                    dataset_family(value) for value in ELIGIBLE_CELLS
                    if dataset_family(value) != group
                })
                expected = {"iu", "sample_index"}
                for lambda_, trust in candidates():
                    key = candidate_key(lambda_, trust)
                    expected.update((f"full__{key}", f"outer__{key}"))
                    expected.update(
                        f"inner={outer_group}__{key}"
                        for outer_group in other_groups
                    )
                if set(stored.files) != expected:
                    missing = sorted(expected - set(stored.files))
                    extra = sorted(set(stored.files) - expected)
                    raise RuntimeError(
                        f"score registry mismatch for {cell}: "
                        f"missing={missing}, extra={extra}"
                    )
                sample_index = np.asarray(stored["sample_index"], dtype=int)
                if not np.array_equal(sample_index, np.arange(len(sample_index))):
                    raise RuntimeError(f"sample index changed: {cell}")
                scores[cell] = {
                    key: np.asarray(stored[key], dtype=float)
                    for key in stored.files if key != "sample_index"
                }
                if any(
                    value.ndim != 1 or not np.isfinite(value).all()
                    for value in scores[cell].values()
                ):
                    raise RuntimeError(f"invalid score array: {cell}")
            if scores[cell]["iu"].shape != labels[cell].shape:
                raise RuntimeError(f"score/label shape mismatch: {cell}")
            if not np.all(np.isin(labels[cell], (0, 1))):
                raise RuntimeError(f"invalid labels: {cell}")
    return scores, labels


def metric_delta(y, baseline, candidate, metric):
    if metric == "auroc":
        fn = roc_auc_score
    elif metric == "auprc":
        fn = average_precision_score
    else:
        raise ValueError(metric)
    return float(fn(y, candidate) - fn(y, baseline))


def group_value(scores, labels, group, score_name, metric="auroc"):
    cells = [cell for cell in ELIGIBLE_CELLS if dataset_family(cell) == group]
    return float(np.mean([
        metric_delta(
            labels[cell], scores[cell]["iu"], scores[cell][score_name], metric
        )
        for cell in cells
    ]))


def candidate_values(scores, labels, validation_groups, *, outer_group=None):
    output = {}
    for lambda_, trust in candidates():
        key = candidate_key(lambda_, trust)
        values = {}
        for group in validation_groups:
            prefix = "outer" if outer_group is None else f"inner={outer_group}"
            values[group] = group_value(
                scores, labels, group, f"{prefix}__{key}", "auroc"
            )
        output[(lambda_, trust)] = values
    return output


def candidate_summary(candidate, values, groups):
    vector = np.asarray([values[candidate][group] for group in groups])
    return {
        "lambda": float(candidate[0]),
        "trust_factor": float(candidate[1]),
        "mean": float(np.mean(vector)),
        "se": float(np.std(vector, ddof=1) / np.sqrt(len(vector))),
        "worst": float(np.min(vector)),
        "values": {group: float(value) for group, value in zip(groups, vector)},
    }


def choose_one_se(values, groups):
    summaries = {
        candidate: candidate_summary(candidate, values, groups)
        for candidate in values
    }
    best = max(
        summaries,
        key=lambda candidate: (
            summaries[candidate]["mean"], -candidate[1], -candidate[0]
        ),
    )
    threshold = summaries[best]["mean"] - summaries[best]["se"]
    eligible = [
        candidate for candidate in summaries
        if summaries[candidate]["mean"] >= threshold - 1e-15
    ]
    tail_safe = [
        candidate for candidate in eligible
        if summaries[candidate]["worst"] >= TAIL_FLOOR
    ]
    pool = tail_safe if tail_safe else eligible
    selected = min(
        pool,
        key=lambda candidate: (
            candidate[1], candidate[0], -summaries[candidate]["mean"]
        ),
    )
    return selected, {
        "policy": "one_se_then_tail_then_min_trust_lambda",
        "best": summaries[best],
        "threshold": float(threshold),
        "eligible_count": len(eligible),
        "tail_safe_count": len(tail_safe),
        "selected": summaries[selected],
    }


def choose_max_mean(values, groups):
    summaries = {
        candidate: candidate_summary(candidate, values, groups)
        for candidate in values
    }
    selected = max(
        summaries,
        key=lambda candidate: (
            summaries[candidate]["mean"], -candidate[1], -candidate[0]
        ),
    )
    return selected, {
        "policy": "max_mean",
        "selected": summaries[selected],
    }


def nested_rows(scores, labels, selector):
    groups = tuple(sorted({dataset_family(cell) for cell in ELIGIBLE_CELLS}))
    rows = []
    for held in groups:
        training = tuple(group for group in groups if group != held)
        values = candidate_values(
            scores, labels, training, outer_group=held
        )
        selected, selection_diag = selector(values, training)
        key = candidate_key(*selected)
        held_auc = group_value(
            scores, labels, held, f"outer__{key}", "auroc"
        )
        held_ap = group_value(
            scores, labels, held, f"outer__{key}", "auprc"
        )
        rows.append({
            "held_group": held,
            "lambda": selected[0],
            "trust_factor": selected[1],
            "candidate_key": key,
            "held_delta_auroc": held_auc,
            "held_delta_auroc_pp": 100 * held_auc,
            "held_delta_auprc": held_ap,
            "held_delta_auprc_pp": 100 * held_ap,
            "selection": selection_diag,
        })
    return rows


def load_nrm_reference(scores, labels):
    path = (
        REPO / "results" / "neutral_residual_mode_cs_iu_v1"
        / "cell_results.csv"
    )
    output = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["regime"] != "original_lofo":
                continue
            cell = row["cell"]
            if cell not in ELIGIBLE_CELLS:
                continue
            current_iu = roc_auc_score(labels[cell], scores[cell]["iu"])
            recorded_iu = float(row["iu_auroc"])
            if abs(current_iu - recorded_iu) > 1e-12:
                raise RuntimeError(f"IU/Family-NRM comparator drift: {cell}")
            output[cell] = float(row["nrm_auroc"]) - recorded_iu
    if set(output) != set(ELIGIBLE_CELLS):
        raise RuntimeError("Family-NRM reference roster mismatch")
    groups = tuple(sorted({dataset_family(cell) for cell in ELIGIBLE_CELLS}))
    return np.asarray([
        np.mean([
            output[cell] for cell in ELIGIBLE_CELLS
            if dataset_family(cell) == group
        ]) for group in groups
    ], dtype=float)


def bootstrap_summary(new, nrm):
    new = np.asarray(new, dtype=float)
    nrm = np.asarray(nrm, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(new), size=(BOOTSTRAPS, len(new)))
    new_draw = new[indices].mean(axis=1)
    nrm_draw = nrm[indices].mean(axis=1)

    def interval(values):
        return [
            100 * float(np.quantile(values, .025)),
            100 * float(np.quantile(values, .975)),
        ]

    return {
        "draws": BOOTSTRAPS,
        "seed": BOOTSTRAP_SEED,
        "delta_ci_pp": interval(new_draw),
        "probability_positive": float(np.mean(new_draw > 0)),
        "d30_pp": 100 * float(np.mean(new - .3 * nrm)),
        "d30_ci_pp": interval(new_draw - .3 * nrm_draw),
        "d50_pp": 100 * float(np.mean(new - .5 * nrm)),
        "d50_ci_pp": interval(new_draw - .5 * nrm_draw),
        "d100_pp": 100 * float(np.mean(new - nrm)),
        "d100_ci_pp": interval(new_draw - nrm_draw),
    }


def exact_sign_flip_pvalue(values):
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    draws = np.asarray([
        np.mean(values * np.asarray(signs))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    return float(np.mean(draws >= observed - 1e-15))


def calibration_key(excluded, lambda_):
    text = "+".join(sorted(excluded)) if excluded else "none"
    return f"exclude={text}__lambda={lambda_:g}"


def write_once(path: Path, payload):
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != payload:
            raise RuntimeError(f"refusing to overwrite changed frozen file: {path}")
        return
    write_json(path, payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    complete = verify_fit(args.out, args.bundle)
    # Correctness arrays are opened only after every fit/source hash verifies.
    scores, labels = load_after_freeze(args.out, args.bundle)
    groups = tuple(sorted({dataset_family(cell) for cell in ELIGIBLE_CELLS}))
    calibrations = json.loads((args.out / "CALIBRATIONS.json").read_text())

    primary_rows = nested_rows(scores, labels, choose_one_se)
    max_mean_rows = nested_rows(scores, labels, choose_max_mean)
    primary = np.asarray([row["held_delta_auroc"] for row in primary_rows])
    primary_ap = np.asarray([row["held_delta_auprc"] for row in primary_rows])
    max_mean = np.asarray([row["held_delta_auroc"] for row in max_mean_rows])
    nrm = load_nrm_reference(scores, labels)
    bootstrap = bootstrap_summary(primary, nrm)

    full_values = candidate_values(scores, labels, groups)
    final, final_diag = choose_one_se(full_values, groups)
    max_final, max_final_diag = choose_max_mean(full_values, groups)
    final_cal = calibrations[calibration_key((), final[0])]
    max_final_cal = calibrations[calibration_key((), max_final[0])]
    outer_directions = [
        np.asarray(calibrations[calibration_key(
            (row["held_group"],), row["lambda"]
        )]["direction"], dtype=float)
        for row in primary_rows
    ]
    cosines = [
        direction_cosine(outer_directions[left], outer_directions[right])
        for left in range(len(outer_directions))
        for right in range(left + 1, len(outer_directions))
    ]
    pearson = pearsonr(primary, nrm)
    spearman = spearmanr(primary, nrm)
    try:
        wilcoxon_p = float(wilcoxon(
            primary, alternative="greater", zero_method="wilcox"
        ).pvalue)
    except ValueError:
        wilcoxon_p = 1.0
    t_result = ttest_1samp(primary, popmean=0.0, alternative="greater")
    t_se = float(np.std(primary, ddof=1) / np.sqrt(len(primary)))
    t_critical = 2.364624251
    t_ci = [
        100 * float(np.mean(primary) - t_critical * t_se),
        100 * float(np.mean(primary) + t_critical * t_se),
    ]

    cell_rows = []
    selected_by_group = {row["held_group"]: row for row in primary_rows}
    max_by_group = {row["held_group"]: row for row in max_mean_rows}
    for cell in ELIGIBLE_CELLS:
        group = dataset_family(cell)
        key = selected_by_group[group]["candidate_key"]
        max_key = max_by_group[group]["candidate_key"]
        y = labels[cell]
        iu_auc = float(roc_auc_score(y, scores[cell]["iu"]))
        candidate_auc = float(roc_auc_score(y, scores[cell][f"outer__{key}"]))
        max_auc = float(roc_auc_score(y, scores[cell][f"outer__{max_key}"]))
        cell_rows.append({
            "cell": cell,
            "group": group,
            "n": len(y),
            "n_correct": int(np.sum(y)),
            "iu_auroc": iu_auc,
            "primary_auroc": candidate_auc,
            "primary_delta_pp": 100 * (candidate_auc - iu_auc),
            "max_mean_auroc": max_auc,
            "max_mean_delta_pp": 100 * (max_auc - iu_auc),
            "primary_candidate": key,
            "max_mean_candidate": max_key,
        })
    write_csv(args.out / "cell_results.csv", cell_rows)
    write_csv(args.out / "nested_outer.csv", [{
        key: value for key, value in row.items() if key != "selection"
    } for row in primary_rows])

    gates = {
        "bootstrap_lower_positive": bootstrap["delta_ci_pp"][0] > 0,
        "point_gain_at_least_0_10pp": 100 * float(np.mean(primary)) >= .10,
        "positive_families_at_least_6_of_8": int(np.sum(primary > 0)) >= 6,
        "worst_family_at_least_minus_0_50pp": 100 * float(np.min(primary)) >= -.50,
        "point_nrm_recovery_at_least_50pct": (
            float(np.mean(primary) / np.mean(nrm)) >= .5
        ),
        "d30_lower_nonnegative": bootstrap["d30_ci_pp"][0] >= 0,
        "minimum_outer_direction_cosine_at_least_0_80": min(cosines) >= .8,
        "graph_attribution_controls": None,
    }
    selection = {
        "version": VERSION,
        "fit_manifest_hash": complete["manifest_hash"],
        "policy": final_diag["policy"],
        "selected_config": {
            "lambda": final[0], "trust_factor": final[1], "k": 7,
            "topology": "union", "coordinates": "family_residuals",
        },
        "direction": final_cal["direction"],
        "pooled_A": final_cal["A"],
        "pooled_c": final_cal["c"],
        "selection_diagnostics": final_diag,
        "max_mean_sensitivity_config": {
            "lambda": max_final[0], "trust_factor": max_final[1],
        },
        "max_mean_sensitivity_direction": max_final_cal["direction"],
        "retrospective": True,
    }
    selection["selection_hash"] = canonical_hash(selection)
    write_once(args.out / "FROZEN_SELECTION.json", selection)

    result = {
        "version": VERSION,
        "status": "RETROSPECTIVE_RECOVERY_FOUND_MECHANISM_CONTROLS_PENDING",
        "claim_boundary": (
            "strict nested reconstruction on opened development families; "
            "not independent confirmation"
        ),
        "n_cells": len(ELIGIBLE_CELLS),
        "n_groups": len(groups),
        "primary": {
            "selector": "one_se",
            "delta_auroc_pp": 100 * float(np.mean(primary)),
            "delta_auprc_pp": 100 * float(np.mean(primary_ap)),
            "ci_auroc_pp": bootstrap["delta_ci_pp"],
            "positive_groups": int(np.sum(primary > 0)),
            "worst_group_pp": 100 * float(np.min(primary)),
            "nrm_delta_pp": 100 * float(np.mean(nrm)),
            "nrm_recovery_fraction": float(np.mean(primary) / np.mean(nrm)),
            "group_values_pp": {
                group: 100 * float(value) for group, value in zip(groups, primary)
            },
            "outer_rows": primary_rows,
        },
        "max_mean_sensitivity": {
            "delta_auroc_pp": 100 * float(np.mean(max_mean)),
            "positive_groups": int(np.sum(max_mean > 0)),
            "worst_group_pp": 100 * float(np.min(max_mean)),
            "nrm_recovery_fraction": float(np.mean(max_mean) / np.mean(nrm)),
            "group_values_pp": {
                group: 100 * float(value) for group, value in zip(groups, max_mean)
            },
            "outer_rows": max_mean_rows,
        },
        "bootstrap": bootstrap,
        "inference_sensitivities": {
            "two_sided_t_interval_pp": t_ci,
            "one_sided_t_p": float(t_result.pvalue),
            "one_sided_wilcoxon_p": wilcoxon_p,
            "exact_one_sided_mean_sign_flip_p": exact_sign_flip_pvalue(primary),
        },
        "direction_stability": {
            "pairwise_min_cosine": float(np.min(cosines)),
            "pairwise_mean_cosine": float(np.mean(cosines)),
        },
        "nrm_profile_similarity": {
            "pearson": float(pearson.statistic),
            "pearson_p": float(pearson.pvalue),
            "spearman": float(spearman.statistic),
            "spearman_p": float(spearman.pvalue),
            "sign_agreement": int(np.sum(np.sign(primary) == np.sign(nrm))),
        },
        "final_selection": selection,
        "gates": gates,
        "all_predictive_gates_pass": all(
            value for key, value in gates.items()
            if key != "graph_attribution_controls"
        ),
    }
    write_json(args.out / "RESULT.json", result)
    lines = [
        "# Pooled Graph-Roughness Direction V1", "",
        "**Retrospective reconstruction; mechanism controls pending.**", "",
        f"Primary strict nested one-SE: **{result['primary']['delta_auroc_pp']:+.3f}pp** "
        f"AUROC (equal-family bootstrap 95% CI "
        f"[{bootstrap['delta_ci_pp'][0]:+.3f}, {bootstrap['delta_ci_pp'][1]:+.3f}]pp), "
        f"{result['primary']['positive_groups']}/8 positive and "
        f"{100*result['primary']['nrm_recovery_fraction']:.1f}% of the frozen "
        "Family-NRM gain recovered.", "",
        f"Nested max-mean HPO sensitivity: **{result['max_mean_sensitivity']['delta_auroc_pp']:+.3f}pp**, "
        f"{result['max_mean_sensitivity']['positive_groups']}/8 positive.", "",
        "| held dataset family | primary ΔAUROC (pp) | max-mean ΔAUROC (pp) |",
        "|---|---:|---:|",
    ]
    for group, primary_value, max_value in zip(groups, primary, max_mean):
        lines.append(f"| `{group}` | {100*primary_value:+.3f} | {100*max_value:+.3f} |")
    lines += [
        "", f"Outer-direction cosine: min {min(cosines):.3f}, mean {np.mean(cosines):.3f}.",
        f"The registered `D_0.30` lower-bound gate is "
        f"{'PASS' if gates['d30_lower_nonnegative'] else 'FAIL'}: "
        f"{bootstrap['d30_pp']:+.3f}pp "
        f"[{bootstrap['d30_ci_pp'][0]:+.3f}, {bootstrap['d30_ci_pp'][1]:+.3f}]pp.",
        "", "The direction and every candidate score were fitted and hashed before this "
        "report opened labels. However, the graph, pooling rule, and selector were designed "
        "after the development outcomes were already known, so the result is discovery-level. "
        "PRMBench/HLE and ProcessBench/SemGrad are also known-outcome stress tests.", "",
    ]
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "primary_delta_pp": result["primary"]["delta_auroc_pp"],
        "primary_ci_pp": result["primary"]["ci_auroc_pp"],
        "nrm_recovery_pct": 100 * result["primary"]["nrm_recovery_fraction"],
        "max_mean_delta_pp": result["max_mean_sensitivity"]["delta_auroc_pp"],
        "selection_hash": selection["selection_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
