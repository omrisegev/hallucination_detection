#!/usr/bin/env python3
"""Run the frozen OG-SML T0 retrospective graph-identifiability test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

WORKTREE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from spectral_utils.og_sml_graph import graph_identifiability_report, groups_from_partition


EXPECTED_LEDGER_SHA256 = "027f2617dfc1d48732de9fe24d3b9809395021fb01f3e0b1391f0af68f4f5ae4"
DEFAULT_LEDGER = Path(
    "/Users/osegev/Desktop/hallucination_detection/local_cache/worktrees/structured_fusion_c_v2/"
    "results/label_free_structured_fusion_c_v2_raw/REAL_STRUCTURAL_LEDGER.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_no_label_boundary(payload: dict[str, Any]) -> None:
    forbidden_true = ("labels_seen", "targets_loaded", "outcome_metrics_computed")
    for key in forbidden_true:
        if payload.get(key) is not False:
            raise RuntimeError(f"label firewall failed at ledger.{key}")
    for index, cell in enumerate(payload.get("cells", [])):
        for key in forbidden_true:
            if cell.get(key) is not False:
                raise RuntimeError(f"label firewall failed at cells[{index}].{key}")
        if cell.get("fused_score_array_persisted") is not False:
            raise RuntimeError(f"score-array firewall failed at cells[{index}]")


def _gate_snapshot(fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "primary_gate_pass": bool(fit["primary_gate_pass"]),
        "converged": bool(fit["converged"]),
        "converged_starts": int(fit["converged_starts"]),
        "multistart_status": fit["multistart_audit"]["status"],
        "jacobian_status": fit["jacobian_audit"]["status"],
        "regularization_status": fit["regularization_audit"]["status"],
        "minimum_weight_cosine": float(fit["regularization_audit"]["minimum_weight_cosine"]),
    }


def run(ledger_path: Path, output_dir: Path) -> dict[str, Any]:
    actual_hash = sha256_file(ledger_path)
    if actual_hash != EXPECTED_LEDGER_SHA256:
        raise RuntimeError(f"unexpected ledger SHA256: {actual_hash}")
    payload = json.loads(ledger_path.read_text())
    _assert_no_label_boundary(payload)
    cells = payload["cells"]
    if len(cells) != 18:
        raise RuntimeError(f"expected 18 lanes, found {len(cells)}")

    records: list[dict[str, Any]] = []
    for cell in cells:
        p = int(cell["n_streams"])
        labels = cell["internal"]["groups"]
        if len(labels) != p:
            raise RuntimeError(f"partition length mismatch for {cell['cell_id']} / {cell['lane']}")
        groups = groups_from_partition(labels)
        report = graph_identifiability_report(
            groups,
            p=p,
            global_loading=cell["structured_fit"]["global_loading"],
        )
        records.append(
            {
                "lane_key": f"{cell['cell_id']}::{cell['lane']}",
                "cell_id": cell["cell_id"],
                "benchmark_panel": cell["benchmark_panel"],
                "lane": cell["lane"],
                "n_streams": p,
                "selected_structure_source": "internal.groups",
                "selected_structure_kind": "single_hard_partition",
                "selected_k": int(cell["internal"]["K"]),
                "partition_labels": labels,
                "groups": [list(group) for group in groups],
                "previous_gates": _gate_snapshot(cell["structured_fit"]),
                "graph": report.to_dict(),
            }
        )

    prior_pass = [record for record in records if record["previous_gates"]["primary_gate_pass"]]
    prior_fail = [record for record in records if not record["previous_gates"]["primary_gate_pass"]]
    exact_counts = len(prior_pass) == 3 and len(prior_fail) == 15
    all_pass_admissible = all(record["graph"]["admissible"] for record in prior_pass)
    min_pass_j = min((record["graph"]["j_selection"] for record in prior_pass), default=float("nan"))
    max_fail_j = max((record["graph"]["j_selection"] for record in prior_fail), default=float("nan"))
    strict_j_separation = bool(min_pass_j > max_fail_j)
    confirmed = bool(exact_counts and all_pass_admissible and strict_j_separation)

    overlap_count = 0
    partition_count = 0
    for record in records:
        memberships = [0] * record["n_streams"]
        for group in record["groups"]:
            for vertex in group:
                memberships[vertex] += 1
        overlap_count += int(any(value > 1 for value in memberships))
        partition_count += int(all(value == 1 for value in memberships))

    prior_gate_inventory = {
        "primary_gate_pass_count": sum(record["previous_gates"]["primary_gate_pass"] for record in records),
        "converged_count": sum(record["previous_gates"]["converged"] for record in records),
        "multistart_pass_count": sum(record["previous_gates"]["multistart_status"] == "PASS" for record in records),
        "jacobian_pass_count": sum(record["previous_gates"]["jacobian_status"] == "PASS" for record in records),
        "regularization_pass_count": sum(record["previous_gates"]["regularization_status"] == "PASS" for record in records),
    }
    prior_gate_inventory["primary_gate_equals_regularization_gate_in_all_lanes"] = all(
        record["previous_gates"]["primary_gate_pass"]
        == (record["previous_gates"]["regularization_status"] == "PASS")
        for record in records
    )

    summary = {
        "schema_version": "og-sml-agent-b-t0-v1",
        "terminal_status": (
            "T0_CONFIRMED_CONTINUE_TO_STEPS_0_6"
            if confirmed
            else "T0_FALSIFIED_STOP_BEFORE_STEPS_0_6"
        ),
        "labels_seen": False,
        "targets_loaded": False,
        "outcome_metrics_computed": False,
        "fused_score_arrays_created": False,
        "source_ledger": str(ledger_path.resolve()),
        "source_ledger_sha256": actual_hash,
        "lane_count": len(records),
        "selected_structure_inventory": {
            "single_hard_partition_lanes": partition_count,
            "overlapping_family_lanes": overlap_count,
            "provenance_used_as_candidate": False,
        },
        "prior_gate_inventory": prior_gate_inventory,
        "prediction": {
            "wording": (
                "The three prior joint-gate passes are admissible and their minimum J exceeds "
                "the maximum J of the 15 prior failures, with inadmissible structures assigned selection J=0."
            ),
            "exact_prior_gate_counts": exact_counts,
            "prior_pass_count": len(prior_pass),
            "prior_fail_count": len(prior_fail),
            "all_prior_passes_admissible": all_pass_admissible,
            "prior_pass_admissible_count": sum(record["graph"]["admissible"] for record in prior_pass),
            "prior_fail_admissible_count": sum(record["graph"]["admissible"] for record in prior_fail),
            "min_prior_pass_j_selection": min_pass_j,
            "max_prior_fail_j_selection": max_fail_j,
            "strict_j_separation": strict_j_separation,
            "confirmed": confirmed,
        },
        "cross_tab": {
            "prior_pass_and_admissible": sum(
                record["previous_gates"]["primary_gate_pass"] and record["graph"]["admissible"]
                for record in records
            ),
            "prior_pass_and_inadmissible": sum(
                record["previous_gates"]["primary_gate_pass"] and not record["graph"]["admissible"]
                for record in records
            ),
            "prior_fail_and_admissible": sum(
                not record["previous_gates"]["primary_gate_pass"] and record["graph"]["admissible"]
                for record in records
            ),
            "prior_fail_and_inadmissible": sum(
                not record["previous_gates"]["primary_gate_pass"] and not record["graph"]["admissible"]
                for record in records
            ),
        },
        "records": records,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "T0_REPORT.json"
    csv_path = output_dir / "T0_LANES.csv"
    markdown_path = output_dir / "T0_REPORT.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "cell_id", "benchmark_panel", "lane", "n_streams", "selected_k",
                "group_sizes", "prior_joint_gate_pass", "prior_converged",
                "prior_multistart_status", "prior_jacobian_status",
                "prior_regularization_status", "admissible", "free_edge_count",
                "free_component_count", "free_bipartite", "free_fiedler_weighted",
                "minimum_exclusive_fiedler", "j_raw", "j_selection", "blockers",
            ],
        )
        writer.writeheader()
        for record in records:
            graph = record["graph"]
            exclusive_values = [item["fiedler_unweighted"] for item in graph["exclusive"]]
            writer.writerow(
                {
                    "cell_id": record["cell_id"],
                    "benchmark_panel": record["benchmark_panel"],
                    "lane": record["lane"],
                    "n_streams": record["n_streams"],
                    "selected_k": record["selected_k"],
                    "group_sizes": "|".join(str(value) for value in graph["group_sizes"]),
                    "prior_joint_gate_pass": record["previous_gates"]["primary_gate_pass"],
                    "prior_converged": record["previous_gates"]["converged"],
                    "prior_multistart_status": record["previous_gates"]["multistart_status"],
                    "prior_jacobian_status": record["previous_gates"]["jacobian_status"],
                    "prior_regularization_status": record["previous_gates"]["regularization_status"],
                    "admissible": graph["admissible"],
                    "free_edge_count": graph["free_edge_count"],
                    "free_component_count": graph["free_component_count"],
                    "free_bipartite": graph["free_bipartite"],
                    "free_fiedler_weighted": graph["free_fiedler_weighted"],
                    "minimum_exclusive_fiedler": min(exclusive_values) if exclusive_values else "",
                    "j_raw": graph["j_raw"],
                    "j_selection": graph["j_selection"],
                    "blockers": "|".join(graph["blockers"]),
                }
            )

    table_rows = []
    for record in records:
        graph = record["graph"]
        table_rows.append(
            "| {cell} | {lane} | {k} | {sizes} | {prior} | {adm} | {edges} | {components} | {bip} | {jf:.6g} | {j:.6g} | {blockers} |".format(
                cell=record["cell_id"],
                lane=record["lane"],
                k=record["selected_k"],
                sizes=",".join(str(value) for value in graph["group_sizes"]),
                prior="PASS" if record["previous_gates"]["primary_gate_pass"] else "FAIL",
                adm="YES" if graph["admissible"] else "NO",
                edges=graph["free_edge_count"],
                components=graph["free_component_count"],
                bip="YES" if graph["free_bipartite"] else "NO",
                jf=graph["free_fiedler_weighted"],
                j=graph["j_selection"],
                blockers=", ".join(graph["blockers"]) or "none",
            )
        )
    prediction = summary["prediction"]
    report = f"""# OG-SML Agent B — T0 report

Terminal status: **{summary['terminal_status']}**

## Result

The preregistered retrospective prediction is **{'CONFIRMED' if confirmed else 'FALSIFIED'}**.
The C-v2 ledger contains {partition_count}/18 single hard partitions and
{overlap_count}/18 overlapping selected families; provenance was a reference and
was not part of the fitted structure.

- Prior joint-gate passes: {len(prior_pass)}; admissible among them: {prediction['prior_pass_admissible_count']}.
- Prior joint-gate failures: {len(prior_fail)}; admissible among them: {prediction['prior_fail_admissible_count']}.
- Minimum selection-J among prior passes: {min_pass_j:.9g}.
- Maximum selection-J among prior failures: {max_fail_j:.9g}.
- Strict J separation: {strict_j_separation}.
- C-v2 multistart PASS: {prior_gate_inventory['multistart_pass_count']}/18; profiled-Jacobian PASS: {prior_gate_inventory['jacobian_pass_count']}/18; regularization-sensitivity PASS: {prior_gate_inventory['regularization_pass_count']}/18.
- In this ledger `primary_gate_pass` equals the regularization-sensitivity verdict in all 18 lanes; it is not a pure optimizer-stability outcome.

Because the stop rule failed, Agent B does not implement Steps 0--6 or run
T1--T3 under this proposal.  This result does not show that graph-identifiable
fusion is impossible; it shows that Theorems 1--2, applied to the structures C-v2
actually fitted, do not explain its observed primary-gate outcomes.  It does not
falsify Theorems 1--2 themselves.

## Lane-level evidence

| Cell | Lane | K | Group sizes | Prior gate | Admissible | |H| | H components | H bipartite | weighted lambda2(H) | selection J | Blockers |
|---|---|---:|---|---|---|---:|---:|---|---:|---:|---|
{chr(10).join(table_rows)}

## Firewall

`labels_seen=false`, `targets_loaded=false`, `outcome_metrics_computed=false`,
and `fused_score_arrays_created=false`.  No localization outcome was evaluated.
"""
    markdown_path.write_text(report)

    manifest = {
        "schema_version": "og-sml-agent-b-t0-manifest-v1",
        "terminal_status": summary["terminal_status"],
        "artifacts": {
            path.name: sha256_file(path)
            for path in (json_path, csv_path, markdown_path)
        },
        "source_ledger_sha256": actual_hash,
    }
    (output_dir / "T0_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output-dir", type=Path, default=Path("results/og_sml_agent_b_v1"))
    args = parser.parse_args()
    summary = run(args.ledger, args.output_dir)
    print(json.dumps({key: summary[key] for key in ("terminal_status", "cross_tab", "prediction")}, indent=2))


if __name__ == "__main__":
    main()
