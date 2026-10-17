#!/usr/bin/env python3
"""Render signed static artifacts for the Joint L-SML existing-data experiment."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.joint_lsml_localization import (  # noqa: E402
    EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD, IU_METHOD, JOINT_METHOD, METHODS,
)
from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


ROOT = REPO / "results/joint_lsml_existing_localization_v1"
STRUCTURAL = ROOT / "STRUCTURAL_LEDGER.json"
EVALUATION = ROOT / "evaluation_r2/EVALUATION_SUMMARY.json"
OUT = ROOT / "presentation"
REPORT = ROOT / "REPORT.md"
METHOD_LABELS = {
    JOINT_METHOD: "Joint L-SML",
    IU_METHOD: "IU-PCR",
    EQUAL_FAMILY_METHOD: "Equal-family",
    FIXED_FAMILY_METHOD: "Fixed-family L-SML",
}
COLORS = {
    JOINT_METHOD: "#0F766E",
    IU_METHOD: "#B45309",
    EQUAL_FAMILY_METHOD: "#475569",
    FIXED_FAMILY_METHOD: "#7C3AED",
}


def _payload_ok(data: dict[str, Any]) -> bool:
    body = {key: value for key, value in data.items() if key != "payload_sha256"}
    return payload_sha256(body) == data.get("payload_sha256")


def _save(fig: Any, stem: str) -> list[Path]:
    paths = [OUT / f"{stem}.png", OUT / f"{stem}.svg"]
    for path in paths:
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return paths


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _structural_plot(structural: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    rows = []
    for cell in structural["cells"]:
        fitted = cell["status"] == "FIT_COMPLETE"
        rows.append({
            "cell_id": cell["cell_id"],
            "panel": cell["panel"],
            "status": cell["status"],
            "K": cell.get("grouping", {}).get("K"),
            "group_sizes": "/".join(map(str, cell.get("grouping", {}).get("group_sizes", []))),
            "held_admissible_fraction": cell.get("grouping", {}).get("held_admissible_fraction"),
            "joint_misfit": cell.get("joint_fit", {}).get("relative_offdiag_misfit"),
            "hard_lsml_misfit": cell.get("hard_lsml_relative_offdiag_misfit"),
            "misfit_improvement": (
                float(cell["hard_lsml_relative_offdiag_misfit"])
                - float(cell["joint_fit"]["relative_offdiag_misfit"])
            ) if fitted else None,
            "minimum_weight_map_spearman": (
                float(cell["weight_map_agreement"]["minimum"])
            ) if fitted else None,
            "diagonal_clipped_count": (
                int(cell["joint_fit"]["diagonal_audit"]["clipped_count"])
            ) if fitted else None,
        })
    labels = [row["cell_id"].replace("processbench_", "PB ").replace("prmbench_response_", "PRM ") for row in rows]
    values = [np.nan if row["misfit_improvement"] is None else row["misfit_improvement"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.35, 1]})
    ypos = np.arange(len(rows))
    axes[0].barh(ypos, np.nan_to_num(values), color=["#0F766E" if np.isfinite(v) else "#CBD5E1" for v in values])
    axes[0].set_yticks(ypos, labels)
    axes[0].invert_yaxis()
    axes[0].axvline(0, color="#334155", linewidth=0.8)
    axes[0].set_xlabel("hard L-SML misfit − Joint misfit (higher is better)")
    axes[0].set_title("Structural fit by cell")
    for index, row in enumerate(rows):
        if row["misfit_improvement"] is None:
            axes[0].text(0.002, index, "blocked: no admissible partition", va="center", fontsize=8, color="#991B1B")
        else:
            axes[0].text(row["misfit_improvement"] + 0.001, index, f"K={row['K']} · {row['group_sizes']}", va="center", fontsize=8)
    agreement = [np.nan if row["minimum_weight_map_spearman"] is None else row["minimum_weight_map_spearman"] for row in rows]
    axes[1].barh(ypos, np.nan_to_num(agreement), color=["#2563EB" if np.isfinite(v) else "#CBD5E1" for v in agreement])
    axes[1].axvline(0.50, color="#DC2626", linestyle="--", linewidth=1.2, label="frozen guard = 0.50")
    axes[1].set_yticks(ypos, [])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("minimum donor score-map Spearman")
    axes[1].set_title("Map agreement gate")
    axes[1].legend(loc="lower right", frameon=False)
    fig.suptitle("Joint L-SML structural validation · 9 existing development cells", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "Source: hash-sealed STRUCTURAL_LEDGER.json · labels not used", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, "structural_gate_overview"), rows


def _performance_plot(evaluation: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    statistics = evaluation["PRMBench"]["paired_bootstrap"]["statistics"]
    rows = []
    for method in METHODS:
        stat = statistics[f"auroc::{method}"]
        rows.append({
            "method_id": method, "method": METHOD_LABELS[method],
            "auroc": stat["point"], "ci_low": stat["ci_low"], "ci_high": stat["ci_high"],
            "auprc": statistics[f"auprc::{method}"]["point"],
            "normalized_ap": statistics[f"normalized_ap::{method}"]["point"],
        })
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1, 1.15]})
    y = np.arange(len(rows))
    point = np.asarray([row["auroc"] for row in rows])
    low = np.asarray([row["ci_low"] for row in rows])
    high = np.asarray([row["ci_high"] for row in rows])
    axes[0].errorbar(point, y, xerr=np.vstack([point - low, high - point]), fmt="none", ecolor="#334155", capsize=3)
    axes[0].scatter(point, y, s=70, color=[COLORS[row["method_id"]] for row in rows], zorder=3)
    axes[0].set_yticks(y, [row["method"] for row in rows])
    axes[0].invert_yaxis()
    pad = max(0.003, (high.max() - low.min()) * 0.18)
    axes[0].set_xlim(low.min() - pad, high.max() + pad)
    axes[0].set_xlabel("Step-error AUROC (95% paired group bootstrap CI)")
    axes[0].set_title("Absolute PRMBench performance")

    controls = [IU_METHOD, EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD]
    deltas = [statistics[f"delta_auroc_joint_vs::{method}"] for method in controls]
    dpoint = np.asarray([row["point"] for row in deltas])
    dlow = np.asarray([row["ci_low"] for row in deltas])
    dhigh = np.asarray([row["ci_high"] for row in deltas])
    dy = np.arange(len(controls))
    axes[1].axvline(0, color="#334155", linewidth=1)
    axes[1].errorbar(dpoint, dy, xerr=np.vstack([dpoint - dlow, dhigh - dpoint]), fmt="o", color="#0F766E", ecolor="#0F766E", capsize=4)
    axes[1].set_yticks(dy, [f"vs {METHOD_LABELS[method]}" for method in controls])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Joint L-SML AUROC delta (95% paired CI)")
    axes[1].set_title("Paired contrasts")
    fig.suptitle("PRMBench retrospective opened-development evaluation", fontsize=14, fontweight="bold")
    fig.text(0.01, 0.01, "6,208 responses · 83,280 official labeled steps · 2,000 source-group bootstrap draws", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    return _save(fig, "prmbench_performance"), rows


def _family_plot(evaluation: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    families = sorted(evaluation["PRMBench"]["per_family"])
    controls = [IU_METHOD, EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD]
    matrix = np.zeros((len(families), len(controls)))
    rows = []
    for i, family in enumerate(families):
        metrics = evaluation["PRMBench"]["per_family"][family]
        for j, control in enumerate(controls):
            joint_value = metrics[f"auroc::{JOINT_METHOD}"]
            control_value = metrics[f"auroc::{control}"]
            value = (
                float(joint_value - control_value)
                if joint_value is not None and control_value is not None
                else float("nan")
            )
            matrix[i, j] = value
            rows.append({
                "error_family": family, "control_id": control,
                "joint_auroc_delta": value if np.isfinite(value) else "",
                "metric_status": metrics["metric_status"],
            })
    limit = max(0.01, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(np.arange(len(controls)), [METHOD_LABELS[method] for method in controls], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(families)), [name.replace("_", " ") for name in families])
    for i in range(len(families)):
        for j in range(len(controls)):
            label = f"{matrix[i, j]:+.3f}" if np.isfinite(matrix[i, j]) else "N/A"
            ax.text(j, i, label, ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(matrix[i, j]) and abs(matrix[i, j]) > 0.55 * limit else "#0F172A")
    fig.colorbar(image, ax=ax, label="Joint L-SML AUROC delta")
    ax.set_title("Where Joint L-SML gains or loses · PRMBench error families", fontweight="bold")
    fig.text(0.01, 0.01, "Point estimates only; family cells are descriptive and not separately adjusted", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return _save(fig, "prmbench_family_deltas"), rows


def _weight_plot(structural: dict[str, Any]) -> tuple[list[Path], list[dict[str, Any]]]:
    cell = next(row for row in structural["cells"] if row["cell_id"] == "prmbench_response_qwen3_8b")
    names = cell["preparation"]["feature_names"]
    weights = np.asarray([cell["weights"][method] for method in METHODS], dtype=float)
    labels = np.asarray(cell["grouping"]["labels"], dtype=int)
    order = np.lexsort((np.asarray(names), labels))
    matrix = weights[:, order]
    limit = max(0.01, float(np.max(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(14, 4.5))
    image = ax.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_yticks(np.arange(len(METHODS)), [METHOD_LABELS[method] for method in METHODS])
    ax.set_xticks(np.arange(len(order)), [names[index].replace("entropy_", "ent_") for index in order], rotation=70, ha="right", fontsize=7)
    boundaries = np.flatnonzero(np.diff(labels[order])) + 0.5
    for boundary in boundaries:
        ax.axvline(boundary, color="#0F172A", linewidth=1.2)
    fig.colorbar(image, ax=ax, label="normalized fusion weight")
    ax.set_title("PRMBench active-23 fusion weights, ordered by learned Joint groups", fontweight="bold")
    fig.text(0.01, 0.01, "All arms share the same active-23 inputs; vertical rules mark the three learned groups", fontsize=8, color="#475569")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    rows = []
    for method_index, method in enumerate(METHODS):
        for feature_index, name in enumerate(names):
            rows.append({
                "method_id": method, "feature": name,
                "joint_group": int(labels[feature_index]), "weight": float(weights[method_index, feature_index]),
            })
    return _save(fig, "prmbench_weight_maps"), rows


def main() -> None:
    if OUT.exists() or REPORT.exists():
        raise RuntimeError("presentation namespace already exists")
    structural = json.loads(STRUCTURAL.read_text())
    evaluation = json.loads(EVALUATION.read_text())
    if not _payload_ok(structural) or not _payload_ok(evaluation):
        raise RuntimeError("input payload hash mismatch")
    if evaluation["ProcessBench"]["status"] != "STRUCTURAL_NO_SCORE":
        raise RuntimeError("unexpected ProcessBench efficacy result")
    if evaluation["PRMBench"]["status"] != "COMPLETE":
        raise RuntimeError("PRMBench evaluation is incomplete")
    OUT.mkdir(parents=True)
    artifacts: list[Path] = []
    paths, structural_rows = _structural_plot(structural); artifacts.extend(paths)
    paths, performance_rows = _performance_plot(evaluation); artifacts.extend(paths)
    paths, family_rows = _family_plot(evaluation); artifacts.extend(paths)
    paths, weight_rows = _weight_plot(structural); artifacts.extend(paths)
    for name, rows in (
        ("structural_cells.csv", structural_rows), ("prmbench_metrics.csv", performance_rows),
        ("prmbench_family_deltas.csv", family_rows), ("prmbench_weights.csv", weight_rows),
    ):
        path = OUT / name; _write_csv(path, rows); artifacts.append(path)

    stats = evaluation["PRMBench"]["paired_bootstrap"]["statistics"]
    joint = stats[f"auroc::{JOINT_METHOD}"]
    contrasts = {control: stats[f"delta_auroc_joint_vs::{control}"] for control in (
        IU_METHOD, EQUAL_FAMILY_METHOD, FIXED_FAMILY_METHOD,
    )}
    prm_cell = next(row for row in structural["cells"] if row["cell_id"] == "prmbench_response_qwen3_8b")
    fitted_pb = sum(row["status"] == "FIT_COMPLETE" and row["panel"] == "ProcessBench" for row in structural["cells"])
    report = f"""# Joint L-SML localization on existing Qwen data

Status: `{evaluation['PRMBench']['decision_state']}` on PRMBench; `STRUCTURAL_NO_SCORE` on ProcessBench.

This is a retrospective opened-development result. It can guide method development,
but it is not a generalization result, a promotion, or a new-leader claim.

## Result

- Joint L-SML PRMBench step AUROC: **{joint['point']:.6f}** (paired 95% CI **[{joint['ci_low']:.6f}, {joint['ci_high']:.6f}]**).
- Versus matched IU-PCR: **{contrasts[IU_METHOD]['point']:+.6f}** (95% CI **[{contrasts[IU_METHOD]['ci_low']:+.6f}, {contrasts[IU_METHOD]['ci_high']:+.6f}]**).
- Versus equal-family: **{contrasts[EQUAL_FAMILY_METHOD]['point']:+.6f}** (95% CI **[{contrasts[EQUAL_FAMILY_METHOD]['ci_low']:+.6f}, {contrasts[EQUAL_FAMILY_METHOD]['ci_high']:+.6f}]**).
- Versus fixed-family continuous L-SML: **{contrasts[FIXED_FAMILY_METHOD]['point']:+.6f}** (95% CI **[{contrasts[FIXED_FAMILY_METHOD]['ci_low']:+.6f}, {contrasts[FIXED_FAMILY_METHOD]['ci_high']:+.6f}]**).
- Cohort: 6,208 error responses, 83,280 official labeled steps, 2,000 paired source-group bootstrap draws.

## Structural gate

Seven of eight ProcessBench cells fit successfully, but `processbench_math_qwen3_4b`
had no admissible partition: every candidate K left at least one group below three
features. The frozen all-eight-cell rule therefore closed the entire ProcessBench
panel before labels, with no efficacy score. PRMBench selected K={prm_cell['grouping']['K']}
with group sizes {prm_cell['grouping']['group_sizes']}; Joint misfit was
{prm_cell['joint_fit']['relative_offdiag_misfit']:.6f} versus
{prm_cell['hard_lsml_relative_offdiag_misfit']:.6f} for hard L-SML.

## Figures

### Structural validation

![Structural gate](presentation/structural_gate_overview.png)

Observation: {fitted_pb}/8 ProcessBench cells and the PRMBench cell passed; every fitted cell reduced off-diagonal misfit relative to hard L-SML.

Inference: the overlapping/global factor fit is numerically useful where the learned partition satisfies the minimum-size contract.

Limitation: the blocked PB cell prevents any ProcessBench efficacy conclusion and exposes a cardinality failure, not an accuracy failure.

### PRMBench performance

![PRMBench performance](presentation/prmbench_performance.png)

Observation: Joint scores below IU-PCR and fixed-family L-SML; both paired intervals are wholly negative, while the equal-family interval crosses zero.

Inference: the registered development state is HARM; the added structural flexibility improved covariance fit but did not improve localization ranking.

Limitation: these outcomes were opened in prior work, so even a positive interval is only development evidence.

### Error-family heterogeneity

![PRMBench family deltas](presentation/prmbench_family_deltas.png)

Observation: Joint gains most against equal-family on counterfactual/deception, but loses in several other families; `multi_solutions` has no positive steps and is N/A.

Inference: the aggregate result should not be interpreted as a uniform mechanism across error types.

Limitation: family values are descriptive point estimates; no family-specific multiplicity correction was registered.

### Learned and baseline weight maps

![PRMBench weights](presentation/prmbench_weight_maps.png)

Observation: Joint, IU-PCR, equal-family, and fixed-family heads place different mass on the same 23 retained streams; Joint is visibly more concentrated.

Inference: any efficacy difference comes from fusion structure, because preprocessing, orientation, roster, and reducer are matched.

Limitation: this experiment does not estimate the value of feature pruning itself because every arm uses active-23.

## Reducers and protocol notes

ProcessBench was frozen to detector=max token risk and locator=argmax of the fixed
top-`min(10, step_length)` mean. It is not top-5 and not top-10-percent. PRMBench
uses maximum token risk inside each official step span.

The first registered evaluator failed before metrics because it required 6,966
score IDs to equal the official 6,208 error-response IDs. R1 records and audits
the canonical opaque-ID subset join. R1 then completed the bootstrap but could
not serialize the undefined single-class `multi_solutions` family metrics. R2
uses an independently verified, numerically equivalent tie-block computation
and writes those undefined family metrics as `null`. No score or method changed.
"""
    REPORT.write_text(report, encoding="utf-8")
    artifacts.append(REPORT)
    manifest = {
        "schema": "joint-lsml-existing-localization-presentation-v1",
        "status": "COMPLETE",
        "scope": "RETROSPECTIVE_OPENED_DEVELOPMENT",
        "renderer_sha256": sha256_file(Path(__file__)),
        "structural_ledger_sha256": sha256_file(STRUCTURAL),
        "evaluation_summary_sha256": sha256_file(EVALUATION),
        "artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(artifacts)
        ],
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    atomic_write_json(OUT / "PRESENTATION_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
