#!/usr/bin/env python3
"""Retrospective cross-domain audit selecting CB-CS-IU as the next candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "cardinality_balanced_cs_iu_v1"
DEVELOPMENT_RESULTS = (
    REPO / "results" / "leverage_balanced_cs_iu_v1" / "cell_results.csv"
)
TRANSFER_RESULTS = (
    REPO
    / "results"
    / "leverage_balanced_processbench_transfer_v1"
    / "cell_results.csv"
)
VERSION = "cardinality-balanced-cs-iu-selection-v1-2026-08-12"
BOOTSTRAP_DRAWS = 20000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize(rows, domain, group_key, method, reference):
    groups = sorted({row[group_key] for row in rows})
    cell_delta = np.asarray([
        float(row[f"{method}_auroc"])
        - float(row[f"{reference}_auroc"])
        for row in rows
    ])
    group_delta = np.asarray([
        np.mean([
            float(row[f"{method}_auroc"])
            - float(row[f"{reference}_auroc"])
            for row in rows
            if row[group_key] == group
        ])
        for group in groups
    ])
    seed = int(hashlib.sha256(
        f"{VERSION}:{domain}:{method}:{reference}".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    draws = group_delta[
        rng.integers(
            0,
            len(group_delta),
            size=(BOOTSTRAP_DRAWS, len(group_delta)),
        )
    ].mean(axis=1)
    return {
        "version": VERSION,
        "domain": domain,
        "method": method,
        "reference": reference,
        "n_cells": len(rows),
        "n_groups": len(groups),
        "cell_macro_delta_pp": float(100 * np.mean(cell_delta)),
        "equal_group_delta_pp": float(100 * np.mean(group_delta)),
        "equal_group_ci_low_pp": float(100 * np.quantile(draws, 0.025)),
        "equal_group_ci_high_pp": float(100 * np.quantile(draws, 0.975)),
        "wins": int(np.sum(cell_delta > 0)),
        "losses": int(np.sum(cell_delta < 0)),
        "ties": int(np.sum(cell_delta == 0)),
        "worst_cell_delta_pp": float(100 * np.min(cell_delta)),
    }


def write_csv(path, rows):
    fields = list(rows[0])
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pp(value):
    return f"{float(value):+.3f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    development = [
        row for row in read_csv(DEVELOPMENT_RESULTS)
        if int(row["n_positive"]) >= 20
    ]
    transfer = [
        row for row in read_csv(TRANSFER_RESULTS)
        if row["split"] == "confirmation"
        and row["target"] == "reasoning_error_present"
    ]
    comparisons = (
        ("cardinality", "iu"),
        ("leverage_balanced", "iu"),
        ("dufs_liu", "iu"),
        ("cardinality", "leverage_balanced"),
        ("cardinality", "dufs_liu"),
    )
    rows = []
    for domain, source, group_key in (
        ("original_23_cell", development, "dataset_family"),
        ("processbench_confirmation", transfer, "subset"),
    ):
        for method, reference in comparisons:
            rows.append(summarize(
                source, domain, group_key, method, reference
            ))
    write_csv(args.out / "summary.csv", rows)

    lookup = {
        (row["domain"], row["method"], row["reference"]): row
        for row in rows
    }
    report_lines = [
        "# Cardinality-balanced contribution-space IU: selection audit",
        "",
        "**Status:** retrospective cross-domain selection evidence; not a "
        "prospective confirmation of CB-CS-IU.",
        "",
        "The family-cardinality rule is positive in both domains and is the "
        "current frozen non-supervised candidate. ProcessBench selected the "
        "pivot from leverage to cardinality, so a new untouched benchmark is "
        "still required.",
        "",
        "| domain | contrast | cell delta | equal-group delta (95% CI) | W/L | worst |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['domain']} | `{row['method']} - {row['reference']}` "
            f"| {pp(row['cell_macro_delta_pp'])}pp "
            f"| {pp(row['equal_group_delta_pp'])}pp "
            f"[{pp(row['equal_group_ci_low_pp'])}, "
            f"{pp(row['equal_group_ci_high_pp'])}] "
            f"| {row['wins']}/{row['losses']} "
            f"| {pp(row['worst_cell_delta_pp'])}pp |"
        )
    original = lookup[("original_23_cell", "cardinality", "iu")]
    transfer_card = lookup[
        ("processbench_confirmation", "cardinality", "iu")
    ]
    transfer_contrast = lookup[
        (
            "processbench_confirmation",
            "cardinality",
            "leverage_balanced",
        )
    ]
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        f"On the original cells, cardinality balancing improved equal-family "
        f"AUROC by {pp(original['equal_group_delta_pp'])}pp. On the frozen "
        f"ProcessBench confirmation slice it improved equal-subset AUROC by "
        f"{pp(transfer_card['equal_group_delta_pp'])}pp and won all "
        f"{transfer_card['n_cells']} cells.",
        "",
        f"On ProcessBench it also beat leverage balancing by "
        f"{pp(transfer_contrast['equal_group_delta_pp'])}pp, with interval "
        f"[{pp(transfer_contrast['equal_group_ci_low_pp'])}, "
        f"{pp(transfer_contrast['equal_group_ci_high_pp'])}]pp. This supports "
        "family multiplicity as the more transferable nuisance observable. "
        "It does not erase selection bias: this contrast motivated the pivot.",
        "",
        "## Claim boundary",
        "",
        "The score computation was label-free and the cardinality score had "
        "already been frozen as a control before both reports. However, "
        "promoting it to the primary method happened after report inspection. "
        "CB-CS-IU is therefore ready for a pristine confirmation, not yet "
        "prospectively confirmed.",
        "",
    ])
    (args.out / "REPORT.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    definition = {
        "version": VERSION,
        "status": "retrospective_variant_selection",
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "selection": "cardinality_balanced_contribution_subspace_iu",
        "requires_pristine_external_confirmation": True,
        "sources": {
            str(DEVELOPMENT_RESULTS.relative_to(REPO)): sha256_file(
                DEVELOPMENT_RESULTS
            ),
            str(TRANSFER_RESULTS.relative_to(REPO)): sha256_file(
                TRANSFER_RESULTS
            ),
            "SPEC_CARDINALITY_BALANCED_CS_IU_V1.md": sha256_file(
                REPO / "SPEC_CARDINALITY_BALANCED_CS_IU_V1.md"
            ),
            "spectral_utils/contribution_subspace.py": sha256_file(
                REPO / "spectral_utils" / "contribution_subspace.py"
            ),
            "scripts/cardinality_balanced_candidate_audit.py": sha256_file(
                Path(__file__)
            ),
        },
    }
    (args.out / "RUN_DEFINITION.json").write_text(
        json.dumps(definition, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.out / "REPORT.md")


if __name__ == "__main__":
    main()
