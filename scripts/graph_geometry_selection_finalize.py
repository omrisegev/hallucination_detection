#!/usr/bin/env python3
"""Finalize and verify Graph Geometry Selection Research V1 artifacts.

This phase is report-only.  It reads already-frozen JSON/CSV artifacts, never
opens the development feature/label bundle or an external target cache, and
registers the complete scientific report surface in REPORT_COMPLETE.json.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "graph_geometry_selection_research_v1"
DEV = ROOT / "development_fit"
POST = ROOT / "postreport_audit"
PLOTS = ROOT / "plots"
TRANSFER = ROOT / "external_transfer"
AUDITS = ROOT / "audits"
VERSION = "graph-geometry-selection-research-final-v1-2026-08-23"
DECISION = "GEOMETRY_SEARCH_SELECTION_OPTIMISM"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict) -> str:
    data = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def verify_self_hash(path: Path, field: str) -> dict:
    payload = read_json(path)
    unhashed = dict(payload)
    recorded = unhashed.pop(field, None)
    if not isinstance(recorded, str) or canonical_hash(unhashed) != recorded:
        raise RuntimeError(f"self-hash failure: {path}:{field}")
    return payload


def require_close(value: float, expected: float, label: str) -> None:
    if abs(float(value) - expected) > 1e-12:
        raise RuntimeError(f"{label} changed: {value} != {expected}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_audit(path: Path) -> dict:
    payload = read_json(path)
    hash_field = next(
        (field for field in ("audit_hash", "result_hash", "manifest_hash")
         if isinstance(payload.get(field), str)),
        None,
    )
    if hash_field is None:
        raise RuntimeError(f"audit has no self-hash: {path}")
    unhashed = dict(payload)
    recorded = unhashed.pop(hash_field)
    if canonical_hash(unhashed) != recorded:
        raise RuntimeError(f"audit self-hash failure: {path}")
    return payload


def one(rows: list[dict], **criteria) -> dict:
    found = [
        row for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    if len(found) != 1:
        raise RuntimeError(f"expected one row for {criteria}, found {len(found)}")
    return found[0]


def verify_plot_manifest() -> dict:
    manifest = verify_self_hash(PLOTS / "PLOT_MANIFEST.json", "manifest_hash")
    if sha256_file(Path(manifest["generator_path"])) != manifest["generator_sha256"]:
        raise RuntimeError("plot generator changed")
    for registry_name in ("input_hashes", "postreport_audit_input_hashes"):
        for item in manifest.get(registry_name, {}).values():
            path = Path(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"plot input changed: {path}")
    for formats in manifest["outputs"].values():
        for item in formats.values():
            path = Path(item["path"])
            if sha256_file(path) != item["sha256"]:
                raise RuntimeError(f"plot changed: {path}")
    return manifest


def verify_external_fit() -> tuple[dict, dict]:
    fit = verify_self_hash(TRANSFER / "FIT_MANIFEST.json", "manifest_hash")
    isolation = verify_self_hash(
        TRANSFER / "isolated" / "ISOLATION_MANIFEST.json", "manifest_hash"
    )
    if not fit.get("canonical_scores_exactly_reproduced"):
        raise RuntimeError("external canonical scores were not reproduced")
    if fit.get("labels_accessed_by_fit") is not False:
        raise RuntimeError("external score fit accessed labels")
    if fit.get("target_fields_physically_present_in_fit_input") is not False:
        raise RuntimeError("external score fit input contains targets")
    if fit.get("target_fields_received_by_fit") != []:
        raise RuntimeError("external score fit received target fields")
    if set(fit["methods"]) != {
        "iu", "canonical", "label_free", "supervised_one_se",
        "supervised_max_mean", "canonical_cross", "label_free_cross",
        "supervised_one_se_cross", "supervised_max_mean_cross",
    }:
        raise RuntimeError("external method registry changed")
    for cell, digest in fit["score_file_hashes"].items():
        path = TRANSFER / "scores" / f"{cell}.npz"
        if sha256_file(path) != digest:
            raise RuntimeError(f"external frozen score file changed: {cell}")
    if isolation.get("target_fields_physically_present_in_score_fit_inputs") is not False:
        raise RuntimeError("external isolation target-free assertion failed")
    return fit, isolation


def validate_inputs() -> dict:
    fit = verify_self_hash(DEV / "FIT_COMPLETE.json", "manifest_hash")
    run = verify_self_hash(DEV / "RUN_DEFINITION.json", "definition_hash")
    label_free = verify_self_hash(
        DEV / "FROZEN_LABELFREE_SELECTION.json", "selection_hash"
    )
    transfer_selection = verify_self_hash(
        DEV / "FROZEN_TRANSFER_SELECTIONS.json", "selection_hash"
    )
    post = read_json(POST / "RESULT.json")
    post_unhashed = dict(post)
    post_hash = post_unhashed.pop("result_payload_sha256", None)
    if canonical_hash(post_unhashed) != post_hash or post.get("status") != "PASS":
        raise RuntimeError("post-report audit failed")
    verify_self_hash(POST / "MANIFEST.json", "manifest_payload_sha256")
    plots = verify_plot_manifest()
    external_fit, external_isolation = verify_external_fit()
    external = read_json(TRANSFER / "RESULT.json")
    if external.get("status") != "RETROSPECTIVE_STRESS_TEST_COMPLETE":
        raise RuntimeError("external transfer report incomplete")
    if external.get("canonical_scores_exactly_reproduced") is not True:
        raise RuntimeError("external report lost canonical identity")
    if external.get("scores_verified_before_outcome_access") is not True:
        raise RuntimeError("external outcomes opened before score verification")
    provenance = verify_audit(AUDITS / "PROVENANCE_FINAL.json")
    mechanism = verify_audit(AUDITS / "MECHANISM_FINAL.json")
    if provenance.get("status") != "PASS":
        raise RuntimeError("final provenance audit did not pass")
    if mechanism.get("status") not in {"PASS", "PASS_WITH_CAVEATS"}:
        raise RuntimeError("final mechanism audit did not pass")
    outcome_manifest = provenance["external_outcome_artifact_manifest"]
    unhashed_outcome_manifest = dict(outcome_manifest)
    recorded_outcome_manifest = unhashed_outcome_manifest.pop(
        "manifest_hash", None
    )
    if canonical_hash(unhashed_outcome_manifest) != recorded_outcome_manifest:
        raise RuntimeError("external outcome artifact manifest self-hash failed")
    for relative, item in outcome_manifest["artifacts"].items():
        path = TRANSFER / relative
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item[
            "size_bytes"
        ]:
            raise RuntimeError(f"external outcome artifact changed: {relative}")

    result = read_json(DEV / "RESULT.json")
    anchors = result["anchors"]
    require_close(
        anchors["canonical_one_se_pp"], 0.25147679442711046,
        "canonical one-SE anchor",
    )
    require_close(
        anchors["fixed_max_mean_pp"], 0.449629196668661,
        "fixed max-mean anchor",
    )
    require_close(
        anchors["legacy_v1_separate_reproduction_pp"],
        0.4516058351238263,
        "legacy V1 anchor",
    )
    if fit.get("labels_accessed_by_fit") is not False:
        raise RuntimeError("development fit accessed labels")
    if fit.get("target_fields_physically_present_in_fit_input") is not False:
        raise RuntimeError("development fit input was not physically target-free")
    if fit.get("candidate_score_count") != 75348:
        raise RuntimeError("development candidate score registry changed")
    if result.get("provisional_decision") != DECISION:
        raise RuntimeError("development decision changed")
    if not post["policy_matched_oracles"]["all_regrets_nonnegative"]:
        raise RuntimeError("policy-matched oracle regret is invalid")
    if transfer_selection.get("held_family_oracle_exported") is not False:
        raise RuntimeError("oracle leaked into transfer")
    if transfer_selection.get("fit_manifest_hash") != fit["manifest_hash"]:
        raise RuntimeError("development fit/transfer selection linkage changed")
    label_free_sha = sha256_file(DEV / "FROZEN_LABELFREE_SELECTION.json")
    if fit.get("label_free_selection_sha256") != label_free_sha or (
        transfer_selection.get("fit_label_free_selection_sha256")
        != label_free_sha
    ):
        raise RuntimeError("label-free selection SHA linkage changed")
    external_selection_path = TRANSFER / "FROZEN_TRANSFER_SELECTIONS.json"
    if sha256_file(external_selection_path) != external_fit["selection_sha256"]:
        raise RuntimeError("external transfer selection file changed")
    external_selection = verify_self_hash(external_selection_path, "selection_hash")
    if external_selection["selection_hash"] != transfer_selection["selection_hash"]:
        raise RuntimeError("development/external transfer selection hash changed")
    return {
        "fit": fit,
        "run": run,
        "label_free": label_free,
        "transfer_selection": transfer_selection,
        "post": post,
        "plots": plots,
        "external_fit": external_fit,
        "external_isolation": external_isolation,
        "external": external,
        "provenance": provenance,
        "mechanism": mechanism,
        "development": result,
    }


def collect_results(data: dict) -> dict:
    dev = data["development"]
    post = data["post"]
    phase_rows = read_csv(DEV / "phase_a_factorial.csv")
    contrast_rows = read_csv(DEV / "phase_a_contrasts.csv")
    actuator_rows = read_csv(DEV / "actuator_paired_summaries.csv")
    node_rows = read_csv(DEV / "node_permutation_outcome_summaries.csv")

    def arm(capacity: str, selector: str, trust: str) -> dict:
        row = one(
            phase_rows, capacity=capacity, selector=selector,
            trust_class=trust,
        )
        return {
            "mean_pp": float(row["mean_pp"]),
            "ci_pp": json.loads(row["ci_pp"]),
            "positive_groups": int(row["positive_groups"]),
            "worst_group_pp": float(row["worst_group_pp"]),
        }

    def contrast(name: str) -> dict:
        row = one(contrast_rows, contrast=name)
        return {
            "mean_pp": float(row["mean_pp"]),
            "ci_pp": json.loads(row["ci_pp"]),
            "positive_groups": int(row["positive_groups"]),
            "worst_group_pp": float(row["worst_group_pp"]),
        }

    def paired(geometry: str, selector: str = "one_se") -> dict:
        row = one(
            actuator_rows, geometry_id=geometry, selector=selector,
            trust_class="canonical",
        )
        return {
            "mean_pp": float(row["mean_pp"]),
            "ci_pp": json.loads(row["ci_pp"]),
            "positive_groups": int(row["positive_groups"]),
            "sign_flip_p": float(row["full_minus_cross_sign_flip_p"]),
        }

    def node(geometry: str, actuator: str = "full") -> dict:
        row = one(node_rows, geometry_id=geometry, actuator=actuator)
        return {
            "real_delta_pp": float(row["real_delta_pp"]),
            "permutation_mean_delta_pp": float(row["permutation_mean_delta_pp"]),
            "real_minus_permutation_mean_pp": float(
                row["real_minus_permutation_mean_mean_pp"]
            ),
            "randomization_p": float(row["randomization_p_greater_or_equal"]),
        }

    summaries = post["policy_matched_oracles"]["summaries"]
    diagnostics = read_json(DEV / "ACTUATOR_DIAGNOSTICS.json")["contexts"][
        "all_source"
    ]
    external_rows = {}
    for domain, panel in data["external"]["summaries"].items():
        external_rows[domain] = {
            name: panel["methods"][name]["equal_group_delta_pp"]
            for name in (
                "canonical", "label_free", "supervised_one_se",
                "supervised_max_mean", "family_nrm", "canonical_cross",
                "label_free_cross",
            )
        }

    return {
        "anchors_pp": dev["anchors"],
        "phase_a": {
            "fixed_one_se_canonical": arm("fixed", "one_se", "canonical"),
            "fixed_max_mean_canonical": arm("fixed", "max_mean", "canonical"),
            "searched_one_se_canonical": arm("searched", "one_se", "canonical"),
            "searched_max_mean_canonical": arm("searched", "max_mean", "canonical"),
            "geometry_effect_one_se_canonical": contrast(
                "searched_minus_fixed__one_se__canonical"
            ),
            "geometry_effect_max_mean_canonical": contrast(
                "searched_minus_fixed__max_mean__canonical"
            ),
            "selector_effect_fixed_canonical": contrast(
                "selector_max_minus_one_se__fixed__canonical"
            ),
            "trust_v1_minus_canonical_fixed_one_se": contrast(
                "trust_v1_minus_canonical__fixed__one_se"
            ),
            "trust_v1_minus_canonical_fixed_max_mean": contrast(
                "trust_v1_minus_canonical__fixed__max_mean"
            ),
        },
        "policy_matched_geometry": summaries,
        "selection_optimism": post["selection_optimism"],
        "full_tuple_ceiling": post["full_tuple_ceiling"],
        "actuator": {
            "canonical_full_minus_cross": paired("residual_union_k7"),
            "adaptive_full_minus_cross": paired("residual_adaptive_k7"),
            "canonical_max_mean_full_minus_cross": paired(
                "residual_union_k7", "max_mean"
            ),
            "adaptive_max_mean_full_minus_cross": paired(
                "residual_adaptive_k7", "max_mean"
            ),
            "canonical_cross_delta_pp": dev["controls"]["matched_controls_pp"][
                "cross_only"
            ],
            "canonical_direction_cosine_lambda_0_03": diagnostics[
                "residual_union_k7"
            ]["full_vs_cross_direction_cosine_by_lambda"]["0.03"],
            "adaptive_direction_cosine_lambda_0_03": diagnostics[
                "residual_adaptive_k7"
            ]["full_vs_cross_direction_cosine_by_lambda"]["0.03"],
            "canonical_cbar_leave_source_cosine_min": diagnostics[
                "residual_union_k7"
            ]["leave_one_source_c_cosine_min"],
            "adaptive_cbar_leave_source_cosine_min": diagnostics[
                "residual_adaptive_k7"
            ]["leave_one_source_c_cosine_min"],
            "canonical_cbar_node_separation_ratio": diagnostics[
                "residual_union_k7"
            ]["real_minus_permuted_mean_separation_ratio"],
            "adaptive_cbar_node_separation_ratio": diagnostics[
                "residual_adaptive_k7"
            ]["real_minus_permuted_mean_separation_ratio"],
            "lambda_is_a_cross_parameter_legacy_field_corrected": True,
            "cross_lambda_parameter": None,
        },
        "node_permutation_controls": {
            "canonical": node("residual_union_k7"),
            "adaptive": node("residual_adaptive_k7"),
            "contribution": node("contribution_union_k7"),
            "historical_canonical": dev["controls"]["node_permutation_null"],
            "full_attribution_gate_passed": dev["controls"][
                "complete_registered_attribution_passed"
            ],
        },
        "external_transfer_pp": external_rows,
    }


def final_result(data: dict, findings: dict) -> dict:
    post = data["post"]
    payload = {
        "version": VERSION,
        "status": "COMPLETE",
        "decision": DECISION,
        "decision_basis": (
            "Neither label-free nor donor-label geometry selection beat its "
            "capacity-matched fixed-geometry comparator on held families; "
            "held-label policy-matched oracles show real but unidentified "
            "geometry headroom, while the enlarged bank has positive matched "
            "inner-minus-outer selection optimism."
        ),
        "answers": {
            "deployably_useful_new_geometry_found": False,
            "useful_geometry_exists_conditionally": True,
            "useful_label_free_selector_found": False,
            "useful_supervised_donor_selector_found": False,
            "geometry_headroom_not_identified_without_held_labels": True,
            "selection_optimism_supported": True,
            "mechanism_description": "conservative pooled graph cross-gradient",
            "quadratic_curvature_is_primary_mechanism": False,
        },
        "findings": findings,
        "fit_report_boundary": {
            "development_fit_physically_target_free": True,
            "development_scores_verified_before_labels": 75348,
            "external_fit_physically_target_free": True,
            "external_scores_verified_before_labels": True,
            "held_family_oracle_exported_to_transfer": False,
            "no_su_rho_or_su_covariance_cleaning_arms": True,
        },
        "claim_boundary": {
            "development": (
                "Retrospective held-family comparison conditional on the "
                "previously outcome-informed frozen mixed-v2/confidence "
                "orientation contract; not end-to-end unseen-family validation."
            ),
            "transfer": (
                "Score-frozen stress test on historically opened ProcessBench, "
                "SemGrad, PRMBench, and HLE; not independent confirmation."
            ),
        },
        "audit_status": {
            "mechanism": data["mechanism"]["status"],
            "provenance": data["provenance"]["status"],
            "postreport_correction": post["status"],
        },
        "key_hashes": {
            "development_fit_manifest": data["fit"]["manifest_hash"],
            "development_run_definition": data["run"]["definition_hash"],
            "label_free_selection": data["label_free"]["selection_hash"],
            "transfer_selection": data["transfer_selection"]["selection_hash"],
            "external_isolation_manifest": data["external_isolation"][
                "manifest_hash"
            ],
            "external_fit_manifest": data["external_fit"]["manifest_hash"],
            "plot_manifest": data["plots"]["manifest_hash"],
            "postreport_result": post["result_payload_sha256"],
        },
    }
    payload["result_hash"] = canonical_hash(payload)
    return payload


def fmt(value: float) -> str:
    return f"{float(value):+.3f}"


def make_report(result: dict) -> str:
    fnd = result["findings"]
    phase = fnd["phase_a"]
    policy = fnd["policy_matched_geometry"]
    optimism = fnd["selection_optimism"]
    actuator = fnd["actuator"]
    external = fnd["external_transfer_pp"]
    tick = chr(96)
    lines = [
        "# Graph Geometry Selection Research V1 — final synthesis",
        "",
        f"**Final decision: {tick}{DECISION}{tick}.**",
        "",
        "The study found geometry headroom, but neither the label-free selector nor the donor-label selector identified it reliably on held families. The apparent gain from the historical +0.251pp to roughly +0.450pp is almost entirely a selector/correction-strength effect, not a graph-search effect. The enlarged Phase-B bank improves inner selection more than outer performance, so the bounded conclusion is selection optimism rather than a promoted geometry.",
        "",
        "## Controlled factorial and anchors",
        "",
        "| estimand | mean paired delta (pp) | interpretation |",
        "|---|---:|---|",
        f"| fixed union-k7, one-SE/canonical | {fmt(phase['fixed_one_se_canonical']['mean_pp'])} | exact +0.251pp anchor |",
        f"| fixed union-k7, max-mean/canonical | {fmt(phase['fixed_max_mean_canonical']['mean_pp'])} | exact +0.450pp anchor |",
        f"| max-mean minus one-SE, fixed/canonical | {fmt(phase['selector_effect_fixed_canonical']['mean_pp'])} | selector/correction-strength effect |",
        f"| graph search minus fixed, one-SE/canonical | {fmt(phase['geometry_effect_one_se_canonical']['mean_pp'])} | small, interval crosses zero |",
        f"| graph search minus fixed, max-mean/canonical | {fmt(phase['geometry_effect_max_mean_canonical']['mean_pp'])} | negligible |",
        f"| V1 trust minus canonical, fixed/one-SE | {fmt(phase['trust_v1_minus_canonical_fixed_one_se']['mean_pp'])} | expanded trust hurts guarded selection |",
        f"| V1 trust minus canonical, fixed/max-mean | {fmt(phase['trust_v1_minus_canonical_fixed_max_mean']['mean_pp'])} | no observed effect |",
        "",
        "The exact legacy five-lambda searched/V1 result is +0.451606pp and equals the common-eight-lambda matched arm. Expanded and V1 trust grids select the same observed maxima for max-mean; they add no gain there.",
        "",
        "![Controlled factorial forest](plots/plot_01_factorial_forest.png)",
        "",
        "## Selector result and geometry headroom",
        "",
        "| policy | method | mean delta (pp) | matched held-geometry oracle (pp) | regret (pp) |",
        "|---|---|---:|---:|---:|",
    ]
    rows = [
        ("fixed strength", "canonical union-k7", "intrinsic_fixed_strength__canonical_fixed_strength", "intrinsic_fixed_strength__held_family_geometry_oracle"),
        ("fixed strength", "label-free adaptive-k7", "intrinsic_fixed_strength__intrinsic_label_free", "intrinsic_fixed_strength__held_family_geometry_oracle"),
        ("one-SE", "supervised donor selector", "one_se__supervised_geometry_selector", "one_se__held_family_geometry_oracle"),
        ("max-mean", "fixed union-k7", "max_mean__fixed_residual_union_k7", "max_mean__held_family_geometry_oracle"),
        ("max-mean", "supervised donor selector", "max_mean__supervised_geometry_selector", "max_mean__held_family_geometry_oracle"),
    ]
    for policy_name, method_name, key, oracle_key in rows:
        item = policy[key]
        oracle = policy[oracle_key]
        lines.append(
            f"| {policy_name} | {method_name} | {fmt(item['mean_delta_auroc_pp'])} | "
            f"{fmt(oracle['mean_delta_auroc_pp'])} | "
            f"{fmt(item['mean_policy_matched_regret_pp'])} |"
        )
    lines += [
        "",
        f"The separately scoped held-label full-tuple ceiling is {fmt(fnd['full_tuple_ceiling']['mean_pp'])}pp; it is not a deployable method and is not used as a geometry-regret reference. The matched Phase-B max-mean optimism difference-in-differences is {fmt(optimism['primary']['mean_pp'])}pp (5/8 families; one-sided exact sign-flip p={optimism['primary']['exact_one_sided_sign_flip_p']:.6f}). Under one-SE it is +0.251355pp (6/8; p=0.042969).",
        "",
        "![Policy-matched selector regret](plots/plot_05_policy_matched_selector_regret.png)",
        "",
        "## Actuator and controls",
        "",
        f"For canonical union-k7, cross-only is +{fnd['actuator']['canonical_cross_delta_pp']:.3f}pp and conservative one-SE full minus cross is only {fmt(actuator['canonical_full_minus_cross']['mean_pp'])}pp. For label-free adaptive-k7, full minus cross is {fmt(actuator['adaptive_full_minus_cross']['mean_pp'])}pp. At lambda=0.03, cos(full, -cbar) is {actuator['canonical_direction_cosine_lambda_0_03']:.6f} (canonical) and {actuator['adaptive_direction_cosine_lambda_0_03']:.6f} (adaptive). Cross-only has no lambda because score normalization fixes the requested correction SD; only direction is identified.",
        "",
        f"The cbar signal is stable to leaving out a source family (minimum cosine {actuator['canonical_cbar_leave_source_cosine_min']:.3f} canonical; {actuator['adaptive_cbar_leave_source_cosine_min']:.3f} adaptive) and separated from the 20 node-permutation cbar null by ratios {actuator['canonical_cbar_node_separation_ratio']:.2f} and {actuator['adaptive_cbar_node_separation_ratio']:.2f}. Outcome controls at fixed lambda=.03/trust=.5 give adaptive real-minus-permutation-mean +{fnd['node_permutation_controls']['adaptive']['real_minus_permutation_mean_pp']:.3f}pp (randomization p={fnd['node_permutation_controls']['adaptive']['randomization_p']:.6f}). Aggressive max-mean activates more curvature (full-minus-cross {fmt(actuator['canonical_max_mean_full_minus_cross']['mean_pp'])}pp canonical and {fmt(actuator['adaptive_max_mean_full_minus_cross']['mean_pp'])}pp adaptive), but both paired intervals cross zero. The complete attribution gate also fails because the canonical arm is not separated from the DUFS graph control. For the conservative promoted-policy question, the accurate mechanism label is pooled graph cross-gradient, not quadratic graph solve.",
        "",
        "## Frozen retrospective transfer",
        "",
        "| opened domain | canonical | label-free | supervised one-SE | supervised max-mean | Family-NRM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    order = [
        "processbench_llama", "processbench_qwen", "semgrad", "prmbench", "hle"
    ]
    for domain in order:
        item = external[domain]
        lines.append(
            f"| {domain.replace('_', ' ')} | {fmt(item['canonical'])} | "
            f"{fmt(item['label_free'])} | {fmt(item['supervised_one_se'])} | "
            f"{fmt(item['supervised_max_mean'])} | {fmt(item['family_nrm'])} |"
        )
    lines += [
        "",
        "The frozen label-free arm is better than canonical on four of five opened domains, but it remains negative versus IU on PRMBench and trails canonical on HLE. Supervised arms are heterogeneous. Because all five domains were historically opened and the development comparison did not beat fixed union-k7, this stress test does not promote the selector.",
        "",
        "![Frozen retrospective transfer](external_transfer/plot_07_frozen_transfer.png)",
        "",
        "## Provenance and claim boundary",
        "",
        "The present development fit consumed an exact-whitelist, physically target-free archive. All 75,348 candidate/full/cross/node-control scores were frozen and independently reconstructed before outcomes were loaded. The external fit similarly consumed only isolated telemetry plus row identifiers, reproduced every canonical external score exactly in 16/16 cells, and froze all method scores before its report opened outcomes. No SU-rho or SU covariance-cleaning arm was included.",
        "",
        "The historical canonical fit never indexed, decoded, or passed any label array into graph construction, calibration, or scoring, and its emitted state/score artifacts are label-free and hash-consistent. Its input archive nevertheless physically contained 24 label members and its provenance hash read the archive bytes. Historical separation was therefore logical member whitelisting, not physical input isolation; the present study repairs that boundary.",
        "",
        "Outer LOFO is strict only for the new graph/selector stage conditional on the frozen mixed-v2 and confidence-orientation contract, which was itself developed using these eight families. Development and transfer findings are retrospective, not end-to-end unseen-family confirmation.",
        "",
        "Final independent audits: audits/MECHANISM_FINAL.md and audits/PROVENANCE_FINAL.md. Corrected policy-matched oracle semantics: postreport_audit/REPORT.md. Artifact closure: REPORT_COMPLETE.json.",
        "",
    ]
    return "\n".join(lines)


def report_paths() -> list[Path]:
    paths: list[Path] = []
    for directory in (
        ROOT / "anchor_reproduction", DEV, ROOT / "label_free_input", POST,
        PLOTS, TRANSFER, AUDITS,
    ):
        if not directory.exists():
            raise RuntimeError(f"missing report directory: {directory}")
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path == ROOT / "REPORT_COMPLETE.json":
                continue
            if "states" in path.parts or "score_basis" in path.parts:
                continue
            if path.suffix.lower() in {".json", ".csv", ".md", ".png", ".pdf"}:
                paths.append(path)
            elif path.name == "cells_target_free.npz":
                paths.append(path)
    paths += [
        ROOT / "FINAL_REPORT.md",
        ROOT / "FINAL_RESULT.json",
        REPO / "docs" / "experiments" / "GRAPH_GEOMETRY_SELECTION_RESEARCH_V1.md",
        REPO / "spectral_utils" / "graph_geometry_selection.py",
        REPO / "scripts" / "build_graph_geometry_label_free_bundle.py",
        REPO / "scripts" / "graph_geometry_selection_fit.py",
        REPO / "scripts" / "graph_geometry_selection_report.py",
        REPO / "scripts" / "graph_geometry_selection_transfer.py",
        REPO / "scripts" / "graph_geometry_selection_postreport_audit.py",
        REPO / "scripts" / "graph_geometry_selection_plots.py",
        Path(__file__),
        REPO / "scripts" / "test_graph_geometry_selection.py",
        REPO / "scripts" / "test_graph_geometry_selection_transfer.py",
        REPO / "scripts" / "test_graph_geometry_selection_postreport_audit.py",
        REPO / "PROGRESS.md",
        REPO / "HISTORY.md",
        REPO / "Research_Directions.md",
        REPO / "GLOSSARY.md",
        REPO / "spectral_utils" / "glossary.py",
        REPO / "scripts" / "build_glossary.py",
    ]
    unique = {path.resolve(): path for path in paths}
    return sorted(unique, key=lambda path: str(path.relative_to(REPO)))


def build_manifest(final: dict) -> dict:
    files = {}
    for path in report_paths():
        if not path.exists():
            raise RuntimeError(f"registered artifact missing: {path}")
        files[str(path.relative_to(REPO))] = sha256_file(path)
    payload = {
        "version": VERSION,
        "status": "COMPLETE",
        "decision": final["decision"],
        "final_result_hash": final["result_hash"],
        "fit_score_banks_registered_transitively": True,
        "fit_score_bank_note": (
            "Development FIT_COMPLETE/SCORE_HASHES and external FIT_MANIFEST/"
            "SCORE_HASHES authenticate the omitted bulk NPZ banks."
        ),
        "registered_artifact_count": len(files),
        "artifact_hashes": files,
    }
    payload["manifest_hash"] = canonical_hash(payload)
    return payload


def finalize() -> None:
    data = validate_inputs()
    findings = collect_results(data)
    final = final_result(data, findings)
    write_json(ROOT / "FINAL_RESULT.json", final)
    (ROOT / "FINAL_REPORT.md").write_text(
        make_report(final), encoding="utf-8"
    )
    manifest = build_manifest(final)
    write_json(ROOT / "REPORT_COMPLETE.json", manifest)
    verify()
    print(json.dumps({
        "status": final["status"],
        "decision": final["decision"],
        "result_hash": final["result_hash"],
        "report_manifest_hash": manifest["manifest_hash"],
        "registered_artifacts": manifest["registered_artifact_count"],
    }, indent=2))


def verify() -> None:
    validate_inputs()
    final = verify_self_hash(ROOT / "FINAL_RESULT.json", "result_hash")
    manifest = verify_self_hash(ROOT / "REPORT_COMPLETE.json", "manifest_hash")
    if manifest.get("final_result_hash") != final["result_hash"]:
        raise RuntimeError("final result/manifest linkage changed")
    for relative, digest in manifest["artifact_hashes"].items():
        path = REPO / relative
        if sha256_file(path) != digest:
            raise RuntimeError(f"registered final artifact changed: {relative}")
    if manifest.get("registered_artifact_count") != len(
        manifest["artifact_hashes"]
    ):
        raise RuntimeError("registered artifact count changed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("finalize", "verify"))
    args = parser.parse_args()
    if args.phase == "finalize":
        finalize()
    else:
        verify()
        print("graph geometry selection final verification: PASS")


if __name__ == "__main__":
    main()
