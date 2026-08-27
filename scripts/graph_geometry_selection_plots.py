#!/usr/bin/env python3
"""Static development plots for Graph Geometry Selection Research V1.

This script is deliberately downstream of the frozen development fit and
outcome report.  It never imports or mutates the fit/report implementation.
Every required input is pinned by SHA-256 and semantically cross-checked before
the first figure is created.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


MPLCONFIG = Path("/tmp/hallucination_geometry_plots_mpl")
MPLCONFIG.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIG)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


REPO = Path(__file__).resolve().parents[1]
VERSION = "graph-geometry-selection-static-plots-v1-2026-08-23"
DEFAULT_DEVELOPMENT = (
    REPO / "results" / "graph_geometry_selection_research_v1"
    / "development_fit"
)
DEFAULT_OUT = (
    REPO / "results" / "graph_geometry_selection_research_v1" / "plots"
)
DEFAULT_POSTREPORT_AUDIT = (
    REPO / "results" / "graph_geometry_selection_research_v1"
    / "postreport_audit"
)

EXPECTED_INPUT_HASHES = {
    "RESULT.json": "a14f178a0cf068c5e1ccb326d9c00d294f38a1961e46e8284c8ccbe98483d15d",
    "PROVENANCE_AUDIT.json": "096939c9fea8468297767004c4b3a45673dcfa75bf13778febdb32facb142823",
    "FIT_COMPLETE.json": "6130ea9585ae6cc7ba85272320be13a9912dcf1570b47209cd350293d4f8307e",
    "phase_a_factorial.csv": "6e80a24c0178bc2706ad2c903e1553cce9aff2bcd8e8056323574e02c19dc07e",
    "phase_a_contrasts.csv": "d78a70d8c5d4ed99fe5d3f85bfc9e6513089d4fe9bc3e927e586cf3f6e9b084e",
    "selector_results.csv": "7465dcdbfe6877449b88c02dd1a22a331a714bc49b4f70a13e986ee8fde0fc07",
    "geometry_family_matrix.csv": "6dfd8d695c4a061c1b33589662de75f80148b63a62ecbaa7a0935a48888fb2eb",
    "intrinsic_diagnostics.csv": "8bee63ea0439401d514742146fd1dd85772458f9360ffc35d08e36e7b46213d7",
    "GRAPH_DIVERSITY.json": "bb42dfd8272e45b6d9e029f18e55c13032396b1e417c027f7ff151900860bbb8",
}
EXPECTED_POSTREPORT_HASHES = {
    "MANIFEST.json": "7962cb5c7a4038ede89c6e7b2e01c80390edc29d841ee8957497cdf242e89d1f",
    "RESULT.json": "f54d0072839af4c5e4d5851842dcd5b09912c1fe38ca464a2118434fec45e2b0",
    "policy_matched_oracle_rows.csv": "f616e899f8ebd677d274b287ad92f8fbd7521810c8a6a6f91be5ca9e6ca578db",
    "policy_matched_oracle_summaries.csv": "86a7c82da9a035f5e3791279489c826f25ee65096db7a2347c2e4d4b36b4eef5",
    "full_tuple_ceiling.csv": "279165095255d41ecdea21b673e2b001068984702f7208b2ed571d333f756d09",
}

GROUP_ORDER = (
    "gsm8k", "hotpotqa", "math500", "nq_open",
    "sciq", "squad_v2", "triviaqa", "truthfulqa",
)
GEOMETRY_ORDER = (
    "residual_union_k7",
    "residual_union_k5",
    "residual_union_k15",
    "residual_adaptive_k7",
    "residual_mutual_k7",
    "residual_cosine_union_k7",
    "residual_shrinkage_mahalanobis_union_k7",
    "contribution_union_k7",
)
DIVERSITY_ORDER = GEOMETRY_ORDER + ("dufs_union_k7",)

GEOMETRY_LABELS = {
    "residual_union_k7": "residual union k7",
    "residual_union_k5": "residual union k5",
    "residual_union_k15": "residual union k15",
    "residual_adaptive_k7": "residual adaptive k7",
    "residual_mutual_k7": "residual mutual k7",
    "residual_cosine_union_k7": "residual cosine k7",
    "residual_shrinkage_mahalanobis_union_k7": "residual shrink-Maha k7",
    "contribution_union_k7": "contribution union k7",
    "dufs_union_k7": "DUFS union k7",
}
GROUP_LABELS = {
    "gsm8k": "GSM8K",
    "hotpotqa": "HotpotQA",
    "math500": "MATH-500",
    "nq_open": "NQ-Open",
    "sciq": "SciQ",
    "squad_v2": "SQuAD-v2",
    "triviaqa": "TriviaQA",
    "truthfulqa": "TruthfulQA",
}
METHOD_COLORS = {
    "canonical_fixed_one_se": "#3366cc",
    "fixed_max_mean": "#6a51a3",
    "intrinsic_label_free": "#109618",
    "supervised_geometry_one_se": "#ff9900",
    "supervised_geometry_max_mean": "#dc3912",
    "held_family_geometry_oracle": "#111111",
    "held_family_full_tuple_ceiling": "#777777",
}
METHOD_LABELS = {
    "canonical_fixed_one_se": "canonical fixed / one-SE",
    "fixed_max_mean": "fixed / max-mean",
    "intrinsic_label_free": "intrinsic label-free",
    "supervised_geometry_one_se": "supervised geometry / one-SE",
    "supervised_geometry_max_mean": "supervised geometry / max-mean",
    "held_family_geometry_oracle": "held-family geometry oracle",
    "held_family_full_tuple_ceiling": "held-family full-tuple ceiling",
}

PLOT_FILES = {
    "01_factorial_forest": "plot_01_factorial_forest",
    "02_held_family_paired": "plot_02_held_family_paired_lines",
    "03_geometry_family_heatmap": "plot_03_geometry_by_held_family",
    "04_intrinsic_scatter": "plot_04_intrinsic_vs_held_performance",
    "05_selector_regret": "plot_05_policy_matched_selector_regret",
    "06_graph_diversity": "plot_06_graph_diversity_edge_overlap",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(payload) -> str:
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_json_cell(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON-valued CSV cell: {value}") from error


def verify_inputs(root: Path) -> tuple[dict, dict]:
    hashes = {}
    for name, expected in EXPECTED_INPUT_HASHES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"required frozen plot input missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"frozen plot input mismatch: {name}; "
                f"expected {expected}, observed {observed}"
            )
        hashes[name] = observed

    result = read_json(root / "RESULT.json")
    audit = read_json(root / "PROVENANCE_AUDIT.json")
    complete = read_json(root / "FIT_COMPLETE.json")
    if result.get("version") != "graph-geometry-selection-research-v1-2026-08-23":
        raise RuntimeError("development result version changed")
    if result.get("status") != "DEVELOPMENT_COMPLETE_TRANSFER_PENDING":
        raise RuntimeError("development result is not complete")
    if result.get("provisional_decision") != "GEOMETRY_SEARCH_SELECTION_OPTIMISM":
        raise RuntimeError("development decision changed")
    if result.get("fit_manifest_hash") != complete.get("manifest_hash"):
        raise RuntimeError("development result/fit manifest mismatch")
    if result.get("candidate_scores_verified_before_labels") != complete.get(
        "candidate_score_count"
    ):
        raise RuntimeError("candidate score-verification count mismatch")
    required_audit = {
        "fit_input_target_fields_physically_present": False,
        "report_first_opened_outcomes_after_all_score_hashes_verified": True,
        "retrospective": True,
        "su_arms_present": False,
    }
    for key, expected in required_audit.items():
        if audit.get(key) is not expected:
            raise RuntimeError(f"postreport provenance audit failed closed: {key}")
    isolation = audit.get("new_fit_physical_isolation", {})
    if not (
        isolation.get("all_score_hashes_matched") is True
        and isolation.get("labels_available_to_verifier") is False
        and isolation.get("candidate_scores_verified_before_label_open")
        == complete.get("candidate_score_count")
    ):
        raise RuntimeError("postreport physical-isolation audit failed")
    context = {
        "result": result,
        "audit": audit,
        "complete": complete,
        "factorial": read_csv(root / "phase_a_factorial.csv"),
        "contrasts": read_csv(root / "phase_a_contrasts.csv"),
        "selectors": read_csv(root / "selector_results.csv"),
        "matrix": read_csv(root / "geometry_family_matrix.csv"),
        "intrinsic": read_csv(root / "intrinsic_diagnostics.csv"),
        "diversity": read_json(root / "GRAPH_DIVERSITY.json"),
    }
    semantic_checks(context)
    return context, hashes


def verify_postreport_audit(root: Path, context: dict) -> dict[str, str]:
    hashes = {}
    for name, expected in EXPECTED_POSTREPORT_HASHES.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(
                f"required policy-matching audit input missing: {path}"
            )
        observed = sha256_file(path)
        if observed != expected:
            raise RuntimeError(
                f"postreport audit input mismatch: {name}; "
                f"expected {expected}, observed {observed}"
            )
        hashes[name] = observed
    manifest = read_json(root / "MANIFEST.json")
    audit_result = read_json(root / "RESULT.json")
    if manifest.get("version") != "graph-geometry-selection-postreport-audit-v1-2026-08-23":
        raise RuntimeError("postreport audit manifest version changed")
    if audit_result.get("version") != manifest.get("version"):
        raise RuntimeError("postreport audit result/manifest version mismatch")
    if audit_result.get("status") != "PASS":
        raise RuntimeError("postreport policy audit did not pass")
    if audit_result.get("bounded_finding") != "GEOMETRY_SEARCH_SELECTION_OPTIMISM":
        raise RuntimeError("postreport bounded finding changed")
    if audit_result.get("scope", {}).get("raw_labels_opened") is not False:
        raise RuntimeError("postreport audit unexpectedly opened raw labels")
    if audit_result.get("scope", {}).get("frozen_fit_or_report_sources_modified") is not False:
        raise RuntimeError("postreport audit modified a frozen source")
    if audit_result["input_hashes"]["development_result"] != EXPECTED_INPUT_HASHES[
        "RESULT.json"
    ]:
        raise RuntimeError("postreport/development result provenance mismatch")
    if audit_result["input_hashes"]["fit_complete"] != EXPECTED_INPUT_HASHES[
        "FIT_COMPLETE.json"
    ]:
        raise RuntimeError("postreport/development fit provenance mismatch")
    if audit_result["input_hashes"]["selector_results"] != EXPECTED_INPUT_HASHES[
        "selector_results.csv"
    ]:
        raise RuntimeError("postreport/development selector provenance mismatch")
    for filename in (
        "RESULT.json", "policy_matched_oracle_rows.csv",
        "policy_matched_oracle_summaries.csv", "full_tuple_ceiling.csv",
    ):
        if manifest["outputs"].get(filename) != hashes[filename]:
            raise RuntimeError(f"postreport manifest output mismatch: {filename}")

    policy_rows = read_csv(root / "policy_matched_oracle_rows.csv")
    policy_summaries = read_csv(root / "policy_matched_oracle_summaries.csv")
    full_tuple = read_csv(root / "full_tuple_ceiling.csv")
    if len(policy_rows) != 72 or len(policy_summaries) != 9:
        raise RuntimeError("postreport policy-matched oracle roster changed")
    if set(row["policy"] for row in policy_rows) != {
        "intrinsic_fixed_strength", "one_se", "max_mean",
    }:
        raise RuntimeError("postreport policy registry changed")
    if any(float(row["policy_matched_regret_pp"]) < -1e-12 for row in policy_rows):
        raise RuntimeError("postreport audit contains negative policy-matched regret")
    if len(full_tuple) != len(GROUP_ORDER) or any(
        row["scope"] != "held_family_full_tuple_ceiling_not_geometry_regret"
        for row in full_tuple
    ):
        raise RuntimeError("full-tuple ceiling scope changed")

    summary = {
        (row["policy"], row["method"]): row for row in policy_summaries
    }
    required_regrets = {
        ("intrinsic_fixed_strength", "canonical_fixed_strength"):
            0.20054449902643495,
        ("intrinsic_fixed_strength", "intrinsic_label_free"):
            0.23220066905537728,
        ("one_se", "fixed_residual_union_k7"): 0.28562334658008326,
        ("one_se", "supervised_geometry_selector"): 0.313478045340633,
        ("max_mean", "fixed_residual_union_k7"): 0.27321025600484505,
        ("max_mean", "supervised_geometry_selector"): 0.28615711944020056,
    }
    for key, expected in required_regrets.items():
        observed = float(summary[key]["mean_policy_matched_regret_pp"])
        if abs(observed - expected) > 1e-12:
            raise RuntimeError(f"policy-matched regret changed: {key}")
    if abs(audit_result["full_tuple_ceiling"]["mean_pp"] - 1.0408366486045157) > 1e-12:
        raise RuntimeError("full-tuple diagnostic ceiling changed")
    context.update({
        "postreport_result": audit_result,
        "policy_rows": policy_rows,
        "policy_summaries": policy_summaries,
        "full_tuple_rows": full_tuple,
    })
    return hashes


def semantic_checks(context: dict) -> None:
    result = context["result"]
    factorial = context["factorial"]
    contrasts = context["contrasts"]
    selectors = context["selectors"]
    matrix = context["matrix"]
    intrinsic = context["intrinsic"]
    diversity = context["diversity"]

    if len(factorial) != 12 or {
        (row["capacity"], row["selector"], row["trust_class"])
        for row in factorial
    } != {
        (capacity, selector, trust)
        for capacity in ("fixed", "searched")
        for selector in ("one_se", "max_mean")
        for trust in ("canonical", "v1", "expanded")
    }:
        raise RuntimeError("Phase-A factorial is incomplete")
    if len(contrasts) != 23 or set(row["kind"] for row in contrasts) != {
        "selector_main", "geometry_capacity_main", "trust_main", "interaction",
    }:
        raise RuntimeError("Phase-A contrast registry is incomplete")
    canonical = next(
        row for row in factorial
        if row["capacity"] == "fixed"
        and row["selector"] == "one_se"
        and row["trust_class"] == "canonical"
    )
    fixed_max = next(
        row for row in factorial
        if row["capacity"] == "fixed"
        and row["selector"] == "max_mean"
        and row["trust_class"] == "canonical"
    )
    anchors = result["anchors"]
    if abs(float(canonical["mean_pp"]) - anchors["canonical_one_se_pp"]) > 1e-12:
        raise RuntimeError("canonical factorial/RESULT anchor mismatch")
    if abs(float(fixed_max["mean_pp"]) - anchors["fixed_max_mean_pp"]) > 1e-12:
        raise RuntimeError("max-mean factorial/RESULT anchor mismatch")

    methods = set(METHOD_LABELS)
    if len(selectors) != 8 * len(methods):
        raise RuntimeError("selector result roster is incomplete")
    if set(row["method"] for row in selectors) != methods:
        raise RuntimeError("selector method registry changed")
    for method in methods:
        if set(
            row["held_group"] for row in selectors if row["method"] == method
        ) != set(GROUP_ORDER):
            raise RuntimeError(f"held-family selector roster changed: {method}")

    if len(matrix) != len(GROUP_ORDER) * len(GEOMETRY_ORDER):
        raise RuntimeError("geometry-by-family matrix is incomplete")
    if set(row["held_group"] for row in matrix) != set(GROUP_ORDER):
        raise RuntimeError("geometry matrix family registry changed")
    if set(row["geometry_id"] for row in matrix) != set(GEOMETRY_ORDER):
        raise RuntimeError("geometry matrix registry changed")
    outer_intrinsic = [
        row for row in intrinsic if row["context"].startswith("outer_held=")
    ]
    if len(outer_intrinsic) != len(matrix):
        raise RuntimeError("outer intrinsic diagnostic roster is incomplete")
    if any(row["valid"] not in ("True", "False") for row in intrinsic):
        raise RuntimeError("intrinsic validity encoding changed")

    if diversity.get("candidate_geometry_count") != len(DIVERSITY_ORDER):
        raise RuntimeError("diversity candidate count changed")
    if diversity.get("effective_geometry_count") != len(DIVERSITY_ORDER):
        raise RuntimeError("effective geometry count changed")
    if diversity.get("duplicate_of") != {}:
        raise RuntimeError("unexpected graph deduplication changed Plot 06")
    if set(diversity.get("active_geometry_ids", ())) != set(DIVERSITY_ORDER):
        raise RuntimeError("diversity geometry registry changed")
    for left_index, left in enumerate(DIVERSITY_ORDER):
        for right in DIVERSITY_ORDER[left_index:]:
            if not _pair(diversity, left, right):
                raise RuntimeError(f"missing diversity pair: {left}/{right}")


def _pair(diversity: dict, left: str, right: str) -> dict:
    pairs = diversity["pairwise"]
    return pairs.get(f"{left}__vs__{right}") or pairs.get(
        f"{right}__vs__{left}"
    )


def add_boundary_footer(fig) -> None:
    fig.text(
        0.995, 0.003,
        "Retrospective opened-family development · physically target-free fit · no SU arms",
        ha="right", va="bottom", fontsize=7, color="#555555",
    )


def save_figure(fig, out: Path, stem: str) -> dict:
    add_boundary_footer(fig)
    fig.tight_layout(rect=(0, 0.025, 1, 1))
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"
    fig.savefig(png, dpi=190, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {
        "png": {"path": str(png.resolve()), "sha256": sha256_file(png)},
        "pdf": {"path": str(pdf.resolve()), "sha256": sha256_file(pdf)},
    }


def contrast_label(row: dict) -> str:
    name = row["contrast"]
    if name.startswith("selector_max_minus_one_se__"):
        _, capacity, trust = name.split("__")
        return f"max-mean − one-SE | {capacity}, {trust} trust"
    if name.startswith("searched_minus_fixed__"):
        _, selector, trust = name.split("__")
        return f"search − fixed | {selector.replace('_', '-')}, {trust} trust"
    if name.startswith("trust_"):
        prefix, capacity, selector = name.rsplit("__", 2)
        grid = "V1" if "trust_v1" in prefix else "expanded"
        return f"{grid} − canonical trust | {capacity}, {selector.replace('_', '-')}"
    if name.startswith("geometry_x_selector__"):
        trust = name.split("__", 1)[1]
        return f"geometry × selector | {trust} trust"
    return name.replace("__", " | ").replace("_", " ")


def plot_factorial(context: dict, out: Path) -> dict:
    kind_order = {
        "selector_main": 0,
        "geometry_capacity_main": 1,
        "trust_main": 2,
        "interaction": 3,
    }
    rows = sorted(
        context["contrasts"],
        key=lambda row: (kind_order[row["kind"]], contrast_label(row)),
    )
    means = np.asarray([float(row["mean_pp"]) for row in rows])
    intervals = np.asarray([parse_json_cell(row["ci_pp"]) for row in rows])
    y = np.arange(len(rows))[::-1]
    colors = {
        "selector_main": "#6a51a3",
        "geometry_capacity_main": "#109618",
        "trust_main": "#3366cc",
        "interaction": "#dc3912",
    }
    fig, axis = plt.subplots(figsize=(11.5, 10.5))
    for index, row in enumerate(rows):
        axis.errorbar(
            means[index], y[index],
            xerr=np.asarray([[
                means[index] - intervals[index, 0]
            ], [
                intervals[index, 1] - means[index]
            ]]),
            fmt="o", capsize=3, markersize=5,
            color=colors[row["kind"]], ecolor=colors[row["kind"]],
        )
        axis.text(
            intervals[index, 1] + 0.025, y[index], f"{means[index]:+.3f}",
            va="center", fontsize=7.5,
        )
    axis.axvline(0, color="black", linewidth=0.9)
    axis.set_yticks(y, [contrast_label(row) for row in rows], fontsize=8)
    axis.set_xlabel("Paired outer-family AUROC contrast (pp)")
    axis.set_title(
        "Plot 01 · Controlled factorial forest\n"
        "Selector, geometry-capacity, trust-grid, and interaction contrasts"
    )
    axis.grid(axis="x", alpha=0.22)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
               markeredgecolor=color, label=kind.replace("_", " "))
        for kind, color in colors.items()
    ]
    axis.legend(handles=handles, ncol=2, fontsize=8, loc="lower right")
    anchors = context["result"]["anchors"]
    axis.text(
        0.01, 0.01,
        f"Anchors: one-SE fixed {anchors['canonical_one_se_pp']:+.3f}pp; "
        f"max-mean fixed {anchors['fixed_max_mean_pp']:+.3f}pp",
        transform=axis.transAxes, fontsize=8, ha="left", va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )
    return save_figure(fig, out, PLOT_FILES["01_factorial_forest"])


def selector_index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["method"], row["held_group"]): row for row in rows}


def plot_held_family(context: dict, out: Path) -> dict:
    index = selector_index(context["selectors"])
    x = np.arange(len(GROUP_ORDER))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, sharey=True)
    panels = (
        (
            axes[0],
            (
                "canonical_fixed_one_se",
                "intrinsic_label_free",
                "supervised_geometry_one_se",
            ),
            "One-SE / frozen intrinsic policy",
        ),
        (
            axes[1],
            ("fixed_max_mean", "supervised_geometry_max_mean"),
            "Max-mean policy",
        ),
    )
    for axis, methods, title in panels:
        for method in methods:
            values = [
                float(index[(method, group)]["held_delta_auroc_pp"])
                for group in GROUP_ORDER
            ]
            axis.plot(
                x, values, marker="o", linewidth=2,
                color=METHOD_COLORS[method], label=METHOD_LABELS[method],
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_ylabel("Held-family ΔAUROC (pp)")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)
        axis.legend(fontsize=8, ncol=len(methods), loc="best")
    axes[1].set_xticks(x, [GROUP_LABELS[group] for group in GROUP_ORDER], rotation=25)
    fig.suptitle(
        "Plot 02 · Strict outer-LOFO paired held-family performance",
        fontsize=13,
    )
    return save_figure(fig, out, PLOT_FILES["02_held_family_paired"])


def plot_geometry_heatmap(context: dict, out: Path) -> dict:
    lookup = {
        (row["geometry_id"], row["held_group"]):
        100 * float(row["held_delta_auroc"])
        for row in context["matrix"]
    }
    values = np.asarray([
        [lookup[(geometry, group)] for group in GROUP_ORDER]
        for geometry in GEOMETRY_ORDER
    ])
    limit = max(0.1, float(np.max(np.abs(values))))
    fig, axis = plt.subplots(figsize=(13, 7.5))
    image = axis.imshow(
        values, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto"
    )
    axis.set_xticks(
        np.arange(len(GROUP_ORDER)),
        [GROUP_LABELS[group] for group in GROUP_ORDER], rotation=30, ha="right",
    )
    axis.set_yticks(
        np.arange(len(GEOMETRY_ORDER)),
        [GEOMETRY_LABELS[geometry] for geometry in GEOMETRY_ORDER],
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "white" if abs(values[row, column]) > 0.55 * limit else "black"
            axis.text(
                column, row, f"{values[row, column]:+.2f}",
                ha="center", va="center", fontsize=7, color=color,
            )
    selector = selector_index(context["selectors"])
    overlays = (
        ("intrinsic_label_free", "D", "#109618", -0.20, -0.22, "label-free"),
        ("supervised_geometry_one_se", "o", "#ff9900", -0.08, 0.22, "supervised one-SE"),
        ("supervised_geometry_max_mean", "s", "#dc3912", 0.13, 0.22, "supervised max-mean"),
        ("held_family_geometry_oracle", "*", "#111111", 0.25, -0.20, "geometry oracle"),
    )
    row_index = {geometry: index for index, geometry in enumerate(GEOMETRY_ORDER)}
    for method, marker, color, dx, dy, _ in overlays:
        for column, group in enumerate(GROUP_ORDER):
            geometry = selector[(method, group)]["geometry_id"]
            axis.scatter(
                column + dx, row_index[geometry] + dy,
                marker=marker, s=55 if marker != "*" else 85,
                facecolor=color, edgecolor="white", linewidth=0.8,
                clip_on=True,
            )
    handles = [
        Line2D([0], [0], marker=marker, color="none", markerfacecolor=color,
               markeredgecolor="white", markersize=8, label=label)
        for _, marker, color, _, _, label in overlays
    ]
    axis.legend(handles=handles, ncol=4, fontsize=8, loc="upper center",
                bbox_to_anchor=(0.5, -0.16))
    colorbar = fig.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
    colorbar.set_label("Held-family ΔAUROC (pp); donor-selected calibration")
    axis.set_title(
        "Plot 03 · Geometry × held-family surface with selector/oracle overlays"
    )
    return save_figure(fig, out, PLOT_FILES["03_geometry_family_heatmap"])


def plot_intrinsic_scatter(context: dict, out: Path) -> dict:
    held_lookup = {
        (row["held_group"], row["geometry_id"]):
        100 * float(row["held_delta_auroc"])
        for row in context["matrix"]
    }
    joined = []
    for row in context["intrinsic"]:
        if not row["context"].startswith("outer_held="):
            continue
        held = row["context"].split("=", 1)[1]
        joined.append({
            **row,
            "held_group": held,
            "held_delta_pp": held_lookup[(held, row["geometry_id"])],
            "selected_bool": row["selected"] == "True",
        })
    metrics = (
        ("minimum_perturbation_stability", "Minimum perturbation stability", True),
        ("minimum_direction_cosine", "Minimum leave-source direction cosine", True),
        ("moment_dispersion", "Moment dispersion (lower preferred)", False),
        ("predicted_roughness_decrease", "Predicted roughness decrease", True),
    )
    palette = plt.get_cmap("tab10")
    geometry_colors = {
        geometry: palette(index) for index, geometry in enumerate(GEOMETRY_ORDER)
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharey=True)
    for axis, (field, label, _) in zip(axes.flat, metrics):
        x = np.asarray([float(row[field]) for row in joined])
        y = np.asarray([row["held_delta_pp"] for row in joined])
        rho = float(spearmanr(x, y).statistic)
        for row in joined:
            selected = row["selected_bool"]
            axis.scatter(
                float(row[field]), row["held_delta_pp"],
                marker="*" if selected else "o",
                s=90 if selected else 34,
                facecolor=geometry_colors[row["geometry_id"]],
                edgecolor="black" if selected else "white",
                linewidth=1.0 if selected else 0.5,
                alpha=1.0 if selected else 0.65,
            )
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set_xlabel(label)
        axis.set_title(f"Spearman ρ = {rho:+.2f}", fontsize=10)
        axis.grid(alpha=0.18)
    axes[0, 0].set_ylabel("Held-family ΔAUROC (pp)")
    axes[1, 0].set_ylabel("Held-family ΔAUROC (pp)")
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=geometry_colors[geometry], markersize=7,
               label=GEOMETRY_LABELS[geometry])
        for geometry in GEOMETRY_ORDER
    ]
    handles.append(Line2D(
        [0], [0], marker="*", color="none", markerfacecolor="white",
        markeredgecolor="black", markersize=10, label="intrinsic selection",
    ))
    fig.legend(handles=handles, ncol=3, fontsize=7.5, loc="lower center",
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Plot 04 · Intrinsic diagnostics do not reliably identify held-family geometry",
        fontsize=13,
    )
    fig.subplots_adjust(bottom=0.18)
    return save_figure(fig, out, PLOT_FILES["04_intrinsic_scatter"])


def _plot_regret_panel(axis, method_rows, title, oracle_mean_pp) -> None:
    """Plot already policy-matched per-family regret vectors."""

    x = np.arange(len(method_rows))
    family_gaps = np.column_stack([
        np.asarray(row["gaps"], dtype=float) for row in method_rows
    ])
    if len(method_rows) > 1:
        for gaps in family_gaps:
            axis.plot(x, gaps, color="#bdbdbd", alpha=0.65, linewidth=1)
            axis.scatter(x, gaps, color="#969696", alpha=0.75, s=18)
    else:
        jitter = np.linspace(-0.055, 0.055, len(GROUP_ORDER))
        axis.scatter(
            x[0] + jitter, family_gaps[:, 0], color="#969696",
            alpha=0.78, s=24,
        )
    means = np.mean(family_gaps, axis=0)
    if len(method_rows) > 1:
        axis.plot(x, means, color="black", linewidth=2.2, zorder=4)
    for position, mean, row in zip(x, means, method_rows):
        axis.scatter(
            position, mean, marker="D", s=72, zorder=5,
            color=row["color"], edgecolor="black", linewidth=0.8,
        )
        axis.text(position, mean + 0.045, f"{mean:+.3f}", ha="center",
                  va="bottom", fontsize=8.5, fontweight="bold")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(
        x, [row["label"] for row in method_rows], rotation=18, ha="right"
    )
    axis.set_ylabel("Policy-matched geometry-oracle regret (pp)")
    axis.set_title(f"{title}\noracle mean ΔAUROC {oracle_mean_pp:+.3f}pp")
    axis.grid(axis="y", alpha=0.22)


def plot_selector_regret(context: dict, out: Path) -> dict:
    policy_index = {
        (row["policy"], row["method"], row["held_group"]): row
        for row in context["policy_rows"]
    }
    def audit_gaps(policy, method):
        return [
            float(policy_index[(policy, method, group)][
                "policy_matched_regret_pp"
            ])
            for group in GROUP_ORDER
        ]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.4), sharey=True)
    _plot_regret_panel(
        axes[0],
        (
            {
                "label": "canonical\nfixed strength",
                "gaps": audit_gaps(
                    "intrinsic_fixed_strength", "canonical_fixed_strength"
                ),
                "color": METHOD_COLORS["canonical_fixed_one_se"],
            },
            {
                "label": "intrinsic label-free",
                "gaps": audit_gaps(
                    "intrinsic_fixed_strength", "intrinsic_label_free"
                ),
                "color": METHOD_COLORS["intrinsic_label_free"],
            },
        ),
        "Fixed-strength intrinsic policy",
        0.4520212934535454,
    )
    _plot_regret_panel(
        axes[1],
        (
            {
                "label": "fixed union k7",
                "gaps": audit_gaps("one_se", "fixed_residual_union_k7"),
                "color": METHOD_COLORS["canonical_fixed_one_se"],
            },
            {
                "label": "supervised donor selector",
                "gaps": audit_gaps(
                    "one_se", "supervised_geometry_selector"
                ),
                "color": METHOD_COLORS["supervised_geometry_one_se"],
            },
        ),
        "One-SE geometry policy",
        0.5371001410071937,
    )
    _plot_regret_panel(
        axes[2],
        (
            {
                "label": "fixed union k7",
                "gaps": audit_gaps("max_mean", "fixed_residual_union_k7"),
                "color": METHOD_COLORS["fixed_max_mean"],
            },
            {
                "label": "supervised donor selector",
                "gaps": audit_gaps(
                    "max_mean", "supervised_geometry_selector"
                ),
                "color": METHOD_COLORS["supervised_geometry_max_mean"],
            },
        ),
        "Max-mean geometry policy",
        0.7228394526735061,
    )
    axes[2].text(
        0.5, 0.975,
        "Full-tuple held-label ceiling: +1.041pp\n"
        "separate optimism ceiling — not this oracle",
        transform=axes[2].transAxes, ha="center", va="top", fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.35", "facecolor": "white",
            "edgecolor": "#777777", "linestyle": "--", "alpha": 0.92,
        },
    )
    fig.suptitle(
        "Plot 05 · Policy-matched selector regret and geometry headroom",
        fontsize=13,
    )
    return save_figure(fig, out, PLOT_FILES["05_selector_regret"])


def diversity_matrix(diversity: dict, field: str) -> np.ndarray:
    matrix = np.empty((len(DIVERSITY_ORDER), len(DIVERSITY_ORDER)), dtype=float)
    for left_index, left in enumerate(DIVERSITY_ORDER):
        for right_index, right in enumerate(DIVERSITY_ORDER):
            matrix[left_index, right_index] = float(_pair(
                diversity, left, right
            )[field])
    return matrix


def plot_diversity(context: dict, out: Path) -> dict:
    diversity = context["diversity"]
    edge = diversity_matrix(diversity, "edge_jaccard_mean")
    operator = diversity_matrix(diversity, "operator_cosine_mean")
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.3))
    panels = (
        (axes[0], edge, "viridis", 0.0, 1.0, "Mean weighted-edge Jaccard"),
        (axes[1], operator, "magma", float(np.min(operator)), 1.0,
         "Mean trace-normalized operator cosine"),
    )
    for axis, matrix, cmap, vmin, vmax, title in panels:
        image = axis.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax)
        labels = [GEOMETRY_LABELS[name] for name in DIVERSITY_ORDER]
        axis.set_xticks(np.arange(len(labels)), labels, rotation=55, ha="right",
                        fontsize=8)
        axis.set_yticks(np.arange(len(labels)), labels, fontsize=8)
        for row in range(len(labels)):
            for column in range(len(labels)):
                normalized = (matrix[row, column] - vmin) / max(vmax - vmin, 1e-12)
                color = "white" if normalized < 0.28 or normalized > 0.82 else "black"
                axis.text(column, row, f"{matrix[row, column]:.2f}",
                          ha="center", va="center", fontsize=6.5, color=color)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
    fig.suptitle(
        "Plot 06 · Effective graph diversity and edge overlap\n"
        f"{diversity['effective_geometry_count']}/{diversity['candidate_geometry_count']} "
        "geometries survive target-free deduplication",
        fontsize=13,
    )
    return save_figure(fig, out, PLOT_FILES["06_graph_diversity"])


def verify_existing_plot_manifest(out: Path) -> dict:
    path = out / "PLOT_MANIFEST.json"
    if not path.is_file():
        raise FileNotFoundError(
            "selective Plot-05 refresh requires the prior plot manifest"
        )
    manifest = read_json(path)
    payload = dict(manifest)
    recorded = payload.pop("manifest_hash", None)
    if recorded is None or canonical_hash(payload) != recorded:
        raise RuntimeError("prior plot manifest is not self-consistent")
    if set(manifest.get("outputs", {})) != set(PLOT_FILES):
        raise RuntimeError("prior plot output registry changed")
    for name, formats in manifest["outputs"].items():
        if set(formats) != {"png", "pdf"}:
            raise RuntimeError(f"prior plot format registry changed: {name}")
        for row in formats.values():
            output_path = Path(row["path"])
            if not output_path.is_file() or sha256_file(output_path) != row["sha256"]:
                raise RuntimeError(f"prior plot artifact changed: {output_path}")
    return dict(manifest["outputs"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument(
        "--postreport-audit", type=Path, default=DEFAULT_POSTREPORT_AUDIT
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--refresh-plot-05", action="store_true",
        help="verify all prior plots, then replace only corrected Plot 05",
    )
    args = parser.parse_args()
    context, input_hashes = verify_inputs(args.development)
    postreport_hashes = verify_postreport_audit(args.postreport_audit, context)
    existing = list(args.out.iterdir()) if args.out.exists() else []
    if existing:
        if not args.refresh_plot_05:
            raise FileExistsError(
                f"refusing to overwrite existing plot artifacts: {args.out}"
            )
        outputs = verify_existing_plot_manifest(args.out)
        outputs["05_selector_regret"] = plot_selector_regret(context, args.out)
    else:
        if args.refresh_plot_05:
            raise FileNotFoundError(
                "--refresh-plot-05 requires an existing complete plot set"
            )
        args.out.mkdir(parents=True, exist_ok=True)
        outputs = {
            "01_factorial_forest": plot_factorial(context, args.out),
            "02_held_family_paired": plot_held_family(context, args.out),
            "03_geometry_family_heatmap": plot_geometry_heatmap(context, args.out),
            "04_intrinsic_scatter": plot_intrinsic_scatter(context, args.out),
            "05_selector_regret": plot_selector_regret(context, args.out),
            "06_graph_diversity": plot_diversity(context, args.out),
        }
    manifest = {
        "version": VERSION,
        "development_root": str(args.development.resolve()),
        "plot_root": str(args.out.resolve()),
        "generator_path": str(Path(__file__).resolve()),
        "generator_sha256": sha256_file(Path(__file__)),
        "mplconfigdir": str(MPLCONFIG),
        "input_hashes": {
            name: {
                "path": str((args.development / name).resolve()),
                "sha256": digest,
            }
            for name, digest in input_hashes.items()
        },
        "postreport_audit_root": str(args.postreport_audit.resolve()),
        "postreport_audit_input_hashes": {
            name: {
                "path": str((args.postreport_audit / name).resolve()),
                "sha256": digest,
            }
            for name, digest in postreport_hashes.items()
        },
        "outputs": outputs,
        "semantic_checks": {
            "development_status": context["result"]["status"],
            "provisional_decision": context["result"]["provisional_decision"],
            "candidate_scores_verified_before_labels": context["result"][
                "candidate_scores_verified_before_labels"
            ],
            "physical_fit_isolation": True,
            "retrospective": True,
            "su_arms_present": False,
            "effective_geometry_count": context["diversity"][
                "effective_geometry_count"
            ],
            "policy_matched_oracle_audit_status": context[
                "postreport_result"
            ]["status"],
            "intrinsic_fixed_strength_regrets_pp": {
                "canonical": 0.20054449902643495,
                "label_free": 0.23220066905537728,
            },
            "one_se_regrets_pp": {
                "canonical": 0.28562334658008326,
                "supervised": 0.313478045340633,
            },
            "max_mean_regrets_pp": {
                "fixed": 0.27321025600484505,
                "supervised": 0.28615711944020056,
            },
            "full_tuple_ceiling_pp_separate_not_oracle": 1.0408366486045157,
        },
    }
    manifest["manifest_hash"] = canonical_hash(manifest)
    manifest_path = args.out / "PLOT_MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({
        "status": "PLOTS_COMPLETE",
        "plots": len(outputs),
        "files": 2 * len(outputs),
        "manifest": str(manifest_path),
        "manifest_hash": manifest["manifest_hash"],
    }, indent=2))


if __name__ == "__main__":
    main()
