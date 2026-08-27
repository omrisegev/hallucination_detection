#!/usr/bin/env python3
"""Post-report corrections for Graph Geometry Selection Research V1.

This audit deliberately consumes only the frozen development JSON/CSV report
artifacts.  It never opens the original feature archive or any correctness
array.  It corrects two reporting semantics without changing the frozen fit:

1. geometry-oracle regret is computed against a donor-calibration-policy-
   matched oracle, separately for one-SE and max-mean;
2. geometry-search optimism is the matched difference in differences
   ``(searched_inner-fixed_inner) - (searched_outer-fixed_outer)``.

It also records that the legacy ``lambda_is_a_cross_parameter`` column in
``actuator_arms.csv`` actually encodes ``lambda_is_full_parameter``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Callable, Iterable, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


VERSION = "graph-geometry-selection-postreport-audit-v1-2026-08-23"
DEFAULT_DEVELOPMENT = (
    REPO / "results" / "graph_geometry_selection_research_v1"
    / "development_fit"
)
DEFAULT_OUT = (
    REPO / "results" / "graph_geometry_selection_research_v1"
    / "postreport_audit"
)
TAIL_FLOOR = -0.005
TOLERANCE = 1e-12

Candidate = tuple[str, float, float]
Selector = Callable[[dict[Candidate, dict[str, float]], Sequence[str], dict[str, int]], tuple[Candidate, dict]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def write_json(path: Path, payload) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_embedded_hash(payload: dict, field: str) -> None:
    content = dict(payload)
    recorded = content.pop(field)
    if canonical_hash(content) != recorded:
        raise RuntimeError(f"embedded hash mismatch: {field}")


def candidate_summary(
    candidate: Candidate,
    values: dict[Candidate, dict[str, float]],
    groups: Sequence[str],
) -> dict:
    vector = np.asarray([values[candidate][group] for group in groups], dtype=float)
    return {
        "geometry_id": candidate[0],
        "lambda": float(candidate[1]),
        "trust_factor": float(candidate[2]),
        "mean": float(np.mean(vector)),
        "se": float(np.std(vector, ddof=1) / np.sqrt(len(vector))),
        "worst": float(np.min(vector)),
    }


def choose_one_se(
    values: dict[Candidate, dict[str, float]],
    groups: Sequence[str],
    priority: dict[str, int],
) -> tuple[Candidate, dict]:
    summaries = {
        candidate: candidate_summary(candidate, values, groups)
        for candidate in values
    }
    best = max(
        summaries,
        key=lambda candidate: (
            summaries[candidate]["mean"],
            -candidate[2],
            -candidate[1],
            -priority[candidate[0]],
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
            candidate[2],
            candidate[1],
            -summaries[candidate]["mean"],
            priority[candidate[0]],
        ),
    )
    return selected, {
        "selected": summaries[selected],
        "best": summaries[best],
        "threshold": float(threshold),
        "eligible_count": len(eligible),
        "tail_safe_count": len(tail_safe),
    }


def choose_max_mean(
    values: dict[Candidate, dict[str, float]],
    groups: Sequence[str],
    priority: dict[str, int],
) -> tuple[Candidate, dict]:
    summaries = {
        candidate: candidate_summary(candidate, values, groups)
        for candidate in values
    }
    selected = max(
        summaries,
        key=lambda candidate: (
            summaries[candidate]["mean"],
            -candidate[2],
            -candidate[1],
            -priority[candidate[0]],
        ),
    )
    return selected, {"selected": summaries[selected]}


def matched_did(
    searched_inner: float,
    fixed_inner: float,
    searched_outer: float,
    fixed_outer: float,
) -> float:
    return (searched_inner - fixed_inner) - (searched_outer - fixed_outer)


def parse_bool(value: object) -> bool:
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"not a serialized boolean: {value!r}")


def corrected_actuator_semantics(rows: Iterable[dict]) -> list[dict]:
    corrected = []
    for row in rows:
        if "lambda_is_a_cross_parameter" not in row:
            raise RuntimeError("legacy actuator semantic column is absent")
        legacy = parse_bool(row["lambda_is_a_cross_parameter"])
        expected = row["actuator"] == "full"
        if legacy != expected:
            raise RuntimeError(
                "legacy actuator column does not encode full-arm lambda presence"
            )
        corrected.append({
            "geometry_id": row["geometry_id"],
            "selector": row["selector"],
            "trust_class": row["trust_class"],
            "actuator": row["actuator"],
            "legacy_field_name": "lambda_is_a_cross_parameter",
            "legacy_value": legacy,
            "corrected_field_name": "lambda_is_full_parameter",
            "lambda_is_full_parameter": expected,
            "cross_lambda_parameter": None,
        })
    return corrected


def exact_sign_flip_pvalue(values: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    observed = float(np.mean(values))
    draws = np.asarray([
        np.mean(values * np.asarray(signs, dtype=float))
        for signs in itertools.product((-1.0, 1.0), repeat=len(values))
    ])
    return float(np.mean(draws >= observed - 1e-15))


def summarize_pp(values: Sequence[float]) -> dict:
    vector = np.asarray(values, dtype=float)
    return {
        "mean_pp": 100 * float(np.mean(vector)),
        "positive_groups": int(np.sum(vector > 0)),
        "worst_group_pp": 100 * float(np.min(vector)),
        "best_group_pp": 100 * float(np.max(vector)),
        "exact_one_sided_sign_flip_p": exact_sign_flip_pvalue(vector),
    }


def load_candidate_index(
    path: Path,
    *,
    eligible_cells: Sequence[str],
    cell_groups: dict[str, str],
) -> dict:
    index = {}
    seen_cells = set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        forbidden = {"label", "labels", "target", "correctness"}
        if forbidden.intersection(name.lower() for name in (reader.fieldnames or [])):
            raise RuntimeError("candidate metric artifact contains a raw-target column")
        for row in reader:
            cell = row["cell"]
            if cell not in cell_groups:
                raise RuntimeError(f"unexpected cell in candidate metrics: {cell}")
            if row["group"] != cell_groups[cell]:
                raise RuntimeError(f"cell/group registry mismatch: {cell}")
            seen_cells.add(cell)
            if row["actuator"] != "full" or row["prefix"].startswith("nodeperm="):
                continue
            candidate = (
                row["geometry_id"],
                float(row["lambda"]),
                float(row["trust_factor"]),
            )
            key = (cell, row["prefix"], candidate)
            if key in index:
                raise RuntimeError(f"duplicate candidate metric row: {key}")
            index[key] = float(row["delta_auroc"])
    if seen_cells != set(eligible_cells):
        raise RuntimeError("candidate metric cell roster changed")
    return index


def group_delta(
    index: dict,
    cells_by_group: dict[str, tuple[str, ...]],
    group: str,
    prefix: str,
    candidate: Candidate,
) -> float:
    return float(np.mean([
        index[(cell, prefix, candidate)] for cell in cells_by_group[group]
    ]))


def candidate_values(
    index: dict,
    cells_by_group: dict[str, tuple[str, ...]],
    validation_groups: Sequence[str],
    *,
    held: str,
    geometries: Sequence[str],
    lambdas: Sequence[float],
    trusts: Sequence[float],
) -> dict[Candidate, dict[str, float]]:
    prefix = f"inner={held}"
    return {
        (geometry, float(lambda_), float(trust)): {
            group: group_delta(
                index, cells_by_group, group, prefix,
                (geometry, float(lambda_), float(trust)),
            )
            for group in validation_groups
        }
        for geometry in geometries
        for lambda_ in lambdas
        for trust in trusts
    }


def outer_delta(
    index: dict,
    cells_by_group: dict[str, tuple[str, ...]],
    held: str,
    candidate: Candidate,
) -> float:
    return group_delta(index, cells_by_group, held, "outer", candidate)


def policy_fold(
    *,
    policy_name: str,
    selector: Selector,
    held: str,
    groups: Sequence[str],
    geometries: Sequence[str],
    lambdas: Sequence[float],
    trusts: Sequence[float],
    priority: dict[str, int],
    index: dict,
    cells_by_group: dict[str, tuple[str, ...]],
) -> tuple[list[dict], dict]:
    training = tuple(group for group in groups if group != held)
    values = candidate_values(
        index,
        cells_by_group,
        training,
        held=held,
        geometries=geometries,
        lambdas=lambdas,
        trusts=trusts,
    )
    supervised, supervised_diag = selector(values, training, priority)
    fixed_values = {
        candidate: value for candidate, value in values.items()
        if candidate[0] == "residual_union_k7"
    }
    fixed, fixed_diag = selector(fixed_values, training, priority)

    per_geometry = []
    for geometry in geometries:
        geometry_values = {
            candidate: value for candidate, value in values.items()
            if candidate[0] == geometry
        }
        candidate, diagnostics = selector(geometry_values, training, priority)
        per_geometry.append({
            "candidate": candidate,
            "inner_mean": diagnostics["selected"]["mean"],
            "outer": outer_delta(index, cells_by_group, held, candidate),
        })
    oracle = max(
        per_geometry,
        key=lambda row: (
            row["outer"], -priority[row["candidate"][0]],
        ),
    )
    oracle_candidate = oracle["candidate"]
    oracle_outer = oracle["outer"]
    methods = (
        ("fixed_residual_union_k7", fixed, fixed_diag["selected"]["mean"]),
        ("supervised_geometry_selector", supervised, supervised_diag["selected"]["mean"]),
        ("held_family_geometry_oracle", oracle_candidate, oracle["inner_mean"]),
    )
    rows = []
    for method, candidate, inner_mean in methods:
        held_delta = outer_delta(index, cells_by_group, held, candidate)
        regret = oracle_outer - held_delta
        if regret < -TOLERANCE:
            raise RuntimeError(
                f"negative policy-matched regret: {policy_name}/{held}/{method}: {regret}"
            )
        rows.append({
            "policy": policy_name,
            "held_group": held,
            "method": method,
            "geometry_id": candidate[0],
            "lambda": candidate[1],
            "trust_factor": candidate[2],
            "inner_selected_mean_pp": 100 * inner_mean,
            "held_delta_auroc_pp": 100 * held_delta,
            "oracle_geometry_id": oracle_candidate[0],
            "oracle_lambda": oracle_candidate[1],
            "oracle_trust_factor": oracle_candidate[2],
            "oracle_delta_auroc_pp": 100 * oracle_outer,
            "policy_matched_regret_pp": max(0.0, 100 * regret),
            "oracle_geometry_agreement": candidate[0] == oracle_candidate[0],
        })
    return rows, {
        "fixed_candidate": fixed,
        "fixed_inner": fixed_diag["selected"]["mean"],
        "fixed_outer": outer_delta(index, cells_by_group, held, fixed),
        "searched_candidate": supervised,
        "searched_inner": supervised_diag["selected"]["mean"],
        "searched_outer": outer_delta(index, cells_by_group, held, supervised),
    }


def fixed_strength_fold(
    *,
    held: str,
    geometries: Sequence[str],
    priority: dict[str, int],
    index: dict,
    cells_by_group: dict[str, tuple[str, ...]],
    intrinsic: dict,
) -> tuple[list[dict], dict]:
    context = intrinsic["contexts"][f"outer_held={held}"]
    fixed_lambda = float(context["fixed_lambda"])
    fixed_trust = float(context["fixed_trust"])
    candidates = [
        (geometry, fixed_lambda, fixed_trust) for geometry in geometries
    ]
    oracle_candidate = max(
        candidates,
        key=lambda candidate: (
            outer_delta(index, cells_by_group, held, candidate),
            -priority[candidate[0]],
        ),
    )
    label_free = (
        context["selected_geometry"], fixed_lambda, fixed_trust,
    )
    canonical = ("residual_union_k7", fixed_lambda, fixed_trust)
    oracle_outer = outer_delta(index, cells_by_group, held, oracle_candidate)
    rows = []
    for method, candidate in (
        ("canonical_fixed_strength", canonical),
        ("intrinsic_label_free", label_free),
        ("held_family_geometry_oracle", oracle_candidate),
    ):
        held_delta = outer_delta(index, cells_by_group, held, candidate)
        regret = oracle_outer - held_delta
        if regret < -TOLERANCE:
            raise RuntimeError(
                f"negative fixed-strength regret: {held}/{method}: {regret}"
            )
        rows.append({
            "policy": "intrinsic_fixed_strength",
            "held_group": held,
            "method": method,
            "geometry_id": candidate[0],
            "lambda": candidate[1],
            "trust_factor": candidate[2],
            "inner_selected_mean_pp": None,
            "held_delta_auroc_pp": 100 * held_delta,
            "oracle_geometry_id": oracle_candidate[0],
            "oracle_lambda": oracle_candidate[1],
            "oracle_trust_factor": oracle_candidate[2],
            "oracle_delta_auroc_pp": 100 * oracle_outer,
            "policy_matched_regret_pp": max(0.0, 100 * regret),
            "oracle_geometry_agreement": candidate[0] == oracle_candidate[0],
        })
    return rows, {
        "candidate": label_free,
        "outer": outer_delta(index, cells_by_group, held, label_free),
    }


def summarize_oracle_rows(rows: list[dict]) -> list[dict]:
    output = []
    keys = sorted({(row["policy"], row["method"]) for row in rows})
    for policy, method in keys:
        selected = [
            row for row in rows
            if row["policy"] == policy and row["method"] == method
        ]
        regrets = np.asarray(
            [row["policy_matched_regret_pp"] for row in selected], dtype=float
        )
        if np.min(regrets) < -100 * TOLERANCE:
            raise RuntimeError(f"negative regret survived summary: {policy}/{method}")
        output.append({
            "policy": policy,
            "method": method,
            "mean_delta_auroc_pp": float(np.mean([
                row["held_delta_auroc_pp"] for row in selected
            ])),
            "mean_policy_matched_regret_pp": float(np.mean(regrets)),
            "minimum_policy_matched_regret_pp": float(np.min(regrets)),
            "oracle_geometry_agreement_count": int(sum(
                bool(row["oracle_geometry_agreement"]) for row in selected
            )),
            "held_group_count": len(selected),
        })
    return output


def original_method_values(path: Path) -> dict[tuple[str, str], float]:
    output = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[(row["method"], row["held_group"])] = (
                float(row["held_delta_auroc_pp"])
            )
    return output


def audit(development: Path, out: Path) -> dict:
    development = Path(development).resolve()
    out = Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite post-report audit: {out}")
    out.mkdir(parents=True)

    required = {
        "fit_complete": development / "FIT_COMPLETE.json",
        "run_definition": development / "RUN_DEFINITION.json",
        "development_result": development / "RESULT.json",
        "label_free_selection": development / "FROZEN_LABELFREE_SELECTION.json",
        "candidate_metrics": development / "candidate_cell_metrics.csv",
        "selector_results": development / "selector_results.csv",
        "actuator_arms": development / "actuator_arms.csv",
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen report artifact {name}: {path}")
    input_hashes_before = {name: sha256_file(path) for name, path in required.items()}

    complete = read_json(required["fit_complete"])
    definition = read_json(required["run_definition"])
    development_result = read_json(required["development_result"])
    intrinsic = read_json(required["label_free_selection"])
    verify_embedded_hash(complete, "manifest_hash")
    verify_embedded_hash(definition, "definition_hash")
    if complete["definition_hash"] != definition["definition_hash"]:
        raise RuntimeError("frozen fit/run-definition mismatch")
    if development_result["fit_manifest_hash"] != complete["manifest_hash"]:
        raise RuntimeError("development result/fit mismatch")
    if (
        sha256_file(required["label_free_selection"])
        != complete["label_free_selection_sha256"]
    ):
        raise RuntimeError("frozen label-free selection hash mismatch")
    if complete.get("labels_accessed_by_fit") is not False:
        raise RuntimeError("frozen fit does not certify label non-access")

    eligible_cells = tuple(definition["eligible_cells"])
    cell_groups = {
        cell: definition["dataset_families"][cell] for cell in eligible_cells
    }
    groups = tuple(sorted(set(cell_groups.values())))
    cells_by_group = {
        group: tuple(cell for cell in eligible_cells if cell_groups[cell] == group)
        for group in groups
    }
    if len(groups) != 8:
        raise RuntimeError("post-report audit expects the frozen eight-family roster")
    geometry_order = tuple(
        item["geometry_id"] for item in definition["geometries"]
    )
    priority = {geometry: rank for rank, geometry in enumerate(geometry_order)}
    selector_geometries = tuple(complete["selector_geometry_ids"])
    phase_a_geometries = tuple(
        geometry for geometry in definition["phase_a_geometry_ids"]
        if geometry in selector_geometries
    )
    lambdas = tuple(float(value) for value in definition["lambda_grid"])
    trusts = tuple(
        float(value) for value in definition["trust_classes"]["canonical"]
    )
    index = load_candidate_index(
        required["candidate_metrics"],
        eligible_cells=eligible_cells,
        cell_groups=cell_groups,
    )

    selectors: dict[str, Selector] = {
        "one_se": choose_one_se,
        "max_mean": choose_max_mean,
    }
    oracle_rows: list[dict] = []
    fold_states: dict[tuple[str, str], dict] = {}
    for policy_name, selector in selectors.items():
        for held in groups:
            rows, state = policy_fold(
                policy_name=policy_name,
                selector=selector,
                held=held,
                groups=groups,
                geometries=selector_geometries,
                lambdas=lambdas,
                trusts=trusts,
                priority=priority,
                index=index,
                cells_by_group=cells_by_group,
            )
            oracle_rows.extend(rows)
            fold_states[(policy_name, held)] = state
    for held in groups:
        rows, _ = fixed_strength_fold(
            held=held,
            geometries=selector_geometries,
            priority=priority,
            index=index,
            cells_by_group=cells_by_group,
            intrinsic=intrinsic,
        )
        oracle_rows.extend(rows)
    oracle_summaries = summarize_oracle_rows(oracle_rows)

    original = original_method_values(required["selector_results"])
    method_mapping = {
        ("one_se", "fixed_residual_union_k7"): "canonical_fixed_one_se",
        ("one_se", "supervised_geometry_selector"): "supervised_geometry_one_se",
        ("max_mean", "fixed_residual_union_k7"): "fixed_max_mean",
        ("max_mean", "supervised_geometry_selector"): "supervised_geometry_max_mean",
        ("intrinsic_fixed_strength", "intrinsic_label_free"): "intrinsic_label_free",
    }
    max_reconstruction_error = 0.0
    for row in oracle_rows:
        mapped = method_mapping.get((row["policy"], row["method"]))
        if mapped is None:
            continue
        error = abs(
            row["held_delta_auroc_pp"] - original[(mapped, row["held_group"])]
        )
        max_reconstruction_error = max(max_reconstruction_error, error)
    if max_reconstruction_error > 1e-10:
        raise RuntimeError(
            f"frozen selector reconstruction mismatch: {max_reconstruction_error}pp"
        )

    did_rows = []
    did_summaries = []
    banks = {
        "phase_a_four_geometry": phase_a_geometries,
        "phase_b_selector_bank": selector_geometries,
    }
    for bank_name, geometries in banks.items():
        for policy_name, selector in selectors.items():
            values = []
            for held in groups:
                _, searched = policy_fold(
                    policy_name=policy_name,
                    selector=selector,
                    held=held,
                    groups=groups,
                    geometries=geometries,
                    lambdas=lambdas,
                    trusts=trusts,
                    priority=priority,
                    index=index,
                    cells_by_group=cells_by_group,
                )
                value = matched_did(
                    searched["searched_inner"],
                    searched["fixed_inner"],
                    searched["searched_outer"],
                    searched["fixed_outer"],
                )
                values.append(value)
                did_rows.append({
                    "bank": bank_name,
                    "policy": policy_name,
                    "held_group": held,
                    "fixed_geometry_id": searched["fixed_candidate"][0],
                    "searched_geometry_id": searched["searched_candidate"][0],
                    "fixed_inner_pp": 100 * searched["fixed_inner"],
                    "searched_inner_pp": 100 * searched["searched_inner"],
                    "inner_search_gain_pp": 100 * (
                        searched["searched_inner"] - searched["fixed_inner"]
                    ),
                    "fixed_outer_pp": 100 * searched["fixed_outer"],
                    "searched_outer_pp": 100 * searched["searched_outer"],
                    "outer_search_effect_pp": 100 * (
                        searched["searched_outer"] - searched["fixed_outer"]
                    ),
                    "matched_selection_optimism_did_pp": 100 * value,
                })
            did_summaries.append({
                "bank": bank_name,
                "policy": policy_name,
                **summarize_pp(values),
                "group_values_pp": [100 * float(value) for value in values],
            })

    full_tuple_rows = []
    full_tuple_values = []
    all_candidates = [
        (geometry, lambda_, trust)
        for geometry in selector_geometries
        for lambda_ in lambdas
        for trust in trusts
    ]
    for held in groups:
        candidate = max(
            all_candidates,
            key=lambda item: (
                outer_delta(index, cells_by_group, held, item),
                -item[2], -item[1], -priority[item[0]],
            ),
        )
        value = outer_delta(index, cells_by_group, held, candidate)
        full_tuple_values.append(value)
        full_tuple_rows.append({
            "scope": "held_family_full_tuple_ceiling_not_geometry_regret",
            "held_group": held,
            "geometry_id": candidate[0],
            "lambda": candidate[1],
            "trust_factor": candidate[2],
            "held_delta_auroc_pp": 100 * value,
            "policy_matched_regret_pp": None,
        })
        error = abs(
            100 * value
            - original[("held_family_full_tuple_ceiling", held)]
        )
        if error > 1e-10:
            raise RuntimeError(f"full-tuple ceiling reconstruction mismatch: {held}")

    with required["actuator_arms"].open(newline="", encoding="utf-8") as handle:
        actuator_rows = corrected_actuator_semantics(list(csv.DictReader(handle)))

    write_csv(out / "policy_matched_oracle_rows.csv", oracle_rows)
    write_csv(out / "policy_matched_oracle_summaries.csv", oracle_summaries)
    write_csv(out / "selection_optimism_did.csv", did_rows)
    write_csv(out / "selection_optimism_did_summaries.csv", did_summaries)
    write_csv(out / "full_tuple_ceiling.csv", full_tuple_rows)
    write_csv(out / "actuator_arms_semantic_correction.csv", actuator_rows)
    csv_names = (
        "policy_matched_oracle_rows.csv",
        "policy_matched_oracle_summaries.csv",
        "selection_optimism_did.csv",
        "selection_optimism_did_summaries.csv",
        "full_tuple_ceiling.csv",
        "actuator_arms_semantic_correction.csv",
    )
    csv_hashes = {name: sha256_file(out / name) for name in csv_names}
    input_hashes_after = {name: sha256_file(path) for name, path in required.items()}
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("a frozen development input changed during the audit")

    minimum_regret = min(
        float(row["policy_matched_regret_pp"]) for row in oracle_rows
    )
    primary_did = next(
        row for row in did_summaries
        if row["bank"] == "phase_b_selector_bank"
        and row["policy"] == "max_mean"
    )
    policy_summary_index = {
        f"{row['policy']}__{row['method']}": row for row in oracle_summaries
    }
    result = {
        "version": VERSION,
        "status": "PASS",
        "scope": {
            "source": "frozen development JSON/CSV report artifacts only",
            "raw_labels_opened": False,
            "raw_feature_archive_opened": False,
            "frozen_fit_or_report_sources_modified": False,
        },
        "input_hashes": input_hashes_before,
        "source_hashes": {
            "audit_script": sha256_file(Path(__file__)),
            "test_script": sha256_file(
                REPO / "scripts" / "test_graph_geometry_selection_postreport_audit.py"
            ),
        },
        "output_csv_hashes": csv_hashes,
        "reconstruction": {
            "maximum_abs_outer_method_error_pp": max_reconstruction_error,
            "frozen_selector_methods_reproduced": True,
            "full_tuple_ceiling_reproduced": True,
        },
        "policy_matched_oracles": {
            "policies": ["one_se", "max_mean", "intrinsic_fixed_strength"],
            "negative_regret_prohibited": True,
            "minimum_regret_pp": minimum_regret,
            "all_regrets_nonnegative": minimum_regret >= -100 * TOLERANCE,
            "summaries": policy_summary_index,
        },
        "selection_optimism": {
            "definition": (
                "(searched_inner-fixed_inner)-(searched_outer-fixed_outer)"
            ),
            "primary_bank": "phase_b_selector_bank",
            "primary_policy": "max_mean",
            "primary": primary_did,
            "all_summaries": did_summaries,
            "legacy_raw_gap_pp": development_result["phase_b"]["effects"][
                "supervised_max_inner_minus_outer_optimism"
            ]["mean_pp"],
            "legacy_raw_gap_is_not_a_matched_optimism_estimand": True,
        },
        "full_tuple_ceiling": {
            "scope": "separately named optimism ceiling; no geometry-regret field",
            **summarize_pp(full_tuple_values),
        },
        "actuator_csv_semantic_correction": {
            "legacy_field": "lambda_is_a_cross_parameter",
            "legacy_field_is_misnamed": True,
            "verified_meaning": "lambda_is_full_parameter",
            "rows_verified": len(actuator_rows),
            "cross_lambda_parameter": None,
        },
        "bounded_finding": (
            "GEOMETRY_SEARCH_SELECTION_OPTIMISM"
            if primary_did["mean_pp"] > 0
            else "NO_POSITIVE_MATCHED_SELECTION_OPTIMISM"
        ),
    }
    result["result_payload_sha256"] = canonical_hash(result)
    write_json(out / "RESULT.json", result)

    one_oracle = policy_summary_index["one_se__held_family_geometry_oracle"]
    max_oracle = policy_summary_index["max_mean__held_family_geometry_oracle"]
    intrinsic_summary = policy_summary_index[
        "intrinsic_fixed_strength__intrinsic_label_free"
    ]
    intrinsic_oracle = policy_summary_index[
        "intrinsic_fixed_strength__held_family_geometry_oracle"
    ]
    report_lines = [
        "# Graph Geometry Selection Research V1 — post-report audit",
        "",
        "**Status: `PASS`.** This audit read only frozen development JSON/CSV "
        "artifacts; it opened neither raw labels nor the feature archive.",
        "",
        "## Corrected oracle semantics",
        "",
        "Held-family geometry oracles now use donor-selected correction strength "
        "under the same policy as the method being compared. One-SE and max-mean "
        "are separate estimands; the full-tuple held-label ceiling is not called "
        "geometry regret.",
        "",
        "| policy | method | mean ΔAUROC (pp) | matched oracle regret (pp) | agreement |",
        "|---|---|---:|---:|---:|",
    ]
    for row in oracle_summaries:
        report_lines.append(
            f"| `{row['policy']}` | `{row['method']}` | "
            f"{row['mean_delta_auroc_pp']:+.6f} | "
            f"{row['mean_policy_matched_regret_pp']:+.6f} | "
            f"{row['oracle_geometry_agreement_count']}/{row['held_group_count']} |"
        )
    report_lines += [
        "",
        f"The one-SE policy-matched geometry oracle is **{one_oracle['mean_delta_auroc_pp']:+.6f}pp**; "
        f"the max-mean policy-matched geometry oracle is **{max_oracle['mean_delta_auroc_pp']:+.6f}pp**.",
        f"The intrinsic selector is **{intrinsic_summary['mean_delta_auroc_pp']:+.6f}pp** versus its "
        f"fixed-strength geometry oracle at **{intrinsic_oracle['mean_delta_auroc_pp']:+.6f}pp**.",
        "",
        "All policy-matched regret values are nonnegative by construction and "
        "checked at runtime. The separately named full-tuple ceiling is "
        f"**{100*np.mean(full_tuple_values):+.6f}pp**.",
        "",
        "## Corrected selection optimism",
        "",
        "The matched estimand is `(searched_inner − fixed_inner) − "
        "(searched_outer − fixed_outer)`.",
        "",
        f"For the Phase-B selector bank under max-mean it is "
        f"**{primary_did['mean_pp']:+.6f}pp** "
        f"({primary_did['positive_groups']}/8 positive families; exact one-sided "
        f"sign-flip p={primary_did['exact_one_sided_sign_flip_p']:.6f}).",
        f"The earlier **{result['selection_optimism']['legacy_raw_gap_pp']:+.6f}pp** number was the "
        "searched arm's raw inner-minus-outer gap, not the matched search-optimism estimand.",
        "",
        "## Actuator CSV semantic correction",
        "",
        "The legacy `lambda_is_a_cross_parameter` column is misnamed. Its values "
        "were verified on every row to mean `lambda_is_full_parameter`; cross has "
        "no lambda parameter. The corrected interpretation is emitted in "
        "`actuator_arms_semantic_correction.csv`.",
        "",
        "## Finding",
        "",
        f"`{result['bounded_finding']}` remains supported after using the matched "
        "difference-in-differences. The geometry-oracle headroom must be quoted "
        "with its selector policy, and the held-label full-tuple result remains "
        "an optimism ceiling only.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    manifest = {
        "version": VERSION,
        "input_hashes": input_hashes_before,
        "outputs": {
            name: sha256_file(out / name)
            for name in ("RESULT.json", "REPORT.md", *csv_names)
        },
        "source_hashes": result["source_hashes"],
    }
    manifest["manifest_payload_sha256"] = canonical_hash(manifest)
    write_json(out / "MANIFEST.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = audit(args.development, args.out)
    print(json.dumps({
        "status": result["status"],
        "bounded_finding": result["bounded_finding"],
        "matched_selection_optimism_did_pp": result["selection_optimism"][
            "primary"
        ]["mean_pp"],
        "minimum_policy_matched_regret_pp": result[
            "policy_matched_oracles"
        ]["minimum_regret_pp"],
        "result_payload_sha256": result["result_payload_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
