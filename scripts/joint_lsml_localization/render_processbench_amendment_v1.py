#!/usr/bin/env python3
"""Render the audited ProcessBench amendment result and final manifest."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.io import atomic_write_json, sha256_file  # noqa: E402
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


ROOT = REPO / "results/joint_lsml_existing_localization_v1"
PB_ROOT = ROOT / "processbench_amendment_v1"
SUMMARY = PB_ROOT / "evaluation/EVALUATION_SUMMARY.json"
AUDIT = PB_ROOT / "INDEPENDENT_EVALUATION_RESULT_AUDIT.json"
PRESENTATION = PB_ROOT / "presentation"
REPORT = PB_ROOT / "REPORT.md"
FINAL = PB_ROOT / "FINAL_COMPLETE.json"
CANDIDATE = "joint_lsml23_hierarchical_v1_1__flat_sml_structural_fallback"
METHODS = (
    CANDIDATE,
    "iu_pcr_active23",
    "equal_family_active23",
    "fixed_family_continuous_lsml_active23",
)
LABELS = {
    CANDIDATE: "Joint-or-flat",
    "iu_pcr_active23": "IU-PCR",
    "equal_family_active23": "Equal-family",
    "fixed_family_continuous_lsml_active23": "Fixed-family L-SML",
}
COLORS = {
    CANDIDATE: "#c0392b",
    "iu_pcr_active23": "#2471a3",
    "equal_family_active23": "#7f8c8d",
    "fixed_family_continuous_lsml_active23": "#1e8449",
}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _short_cell(cell: str) -> str:
    bits = cell.replace("processbench_", "").replace("qwen3_", "Q").split("_")
    return "\n".join((bits[0].replace("olympiadbench", "olympiad").replace("omnimath", "omni"), bits[-1]))


def main() -> None:
    result = json.loads(SUMMARY.read_text())
    audit = json.loads(AUDIT.read_text())
    if audit.get("status") != "PASS" or audit.get("evaluation_summary_sha256") != sha256_file(SUMMARY):
        raise RuntimeError("independent result audit is absent or stale")
    if result.get("decision_state") != "HARM":
        raise RuntimeError("renderer is bound to the audited HARM result")
    stats = result["paired_bootstrap"]["statistics"]
    PRESENTATION.mkdir(parents=True, exist_ok=True)

    metric_rows = []
    for method in METHODS:
        row = stats[f"macro_f1::{method}"]
        metric_rows.append({
            "method_id": method,
            "label": LABELS[method],
            "macro_f1": row["point"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
        })
    _write_csv(PRESENTATION / "processbench_metrics.csv", metric_rows)

    contrast_rows = []
    for control in METHODS[1:]:
        row = stats[f"delta_candidate_vs::{control}"]
        contrast_rows.append({
            "control_id": control,
            "control_label": LABELS[control],
            "delta": row["point"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
        })
    _write_csv(PRESENTATION / "processbench_contrasts.csv", contrast_rows)

    cells = [key.split("::", 1)[1] for key in result["per_cell_oof_f1"] if key.startswith(CANDIDATE + "::")]
    cell_rows = []
    for cell in cells:
        for method in METHODS:
            cell_rows.append({
                "cell_id": cell,
                "method_id": method,
                "label": LABELS[method],
                "oof_f1": result["per_cell_oof_f1"][f"{method}::{cell}"],
                "candidate_component": "flat_sml_fallback" if cell == "processbench_math_qwen3_4b" else "joint_lsml",
            })
    _write_csv(PRESENTATION / "processbench_per_cell.csv", cell_rows)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    y = np.arange(len(METHODS))
    values = np.asarray([row["macro_f1"] for row in metric_rows], dtype=float)
    low = values - np.asarray([row["ci_low"] for row in metric_rows], dtype=float)
    high = np.asarray([row["ci_high"] for row in metric_rows], dtype=float) - values
    ax.barh(y, values, color=[COLORS[m] for m in METHODS], alpha=0.88)
    ax.errorbar(values, y, xerr=np.vstack([low, high]), fmt="none", ecolor="#17202a", capsize=4, lw=1.5)
    ax.set_yticks(y, [LABELS[m] for m in METHODS])
    ax.invert_yaxis()
    ax.set_xlim(0.22, 0.38)
    ax.set_xlabel("ProcessBench macro-F1 (equal model × equal subset)")
    ax.set_title("Joint-or-flat is below both matched spectral controls")
    for position, value in enumerate(values):
        ax.text(value + 0.003, position, f"{value:.4f}", va="center", fontsize=10)
    ax.text(0.22, len(METHODS) + 0.05, "2,000 paired source-question bootstrap draws · development data", fontsize=9, color="#566573")
    fig.tight_layout()
    fig.savefig(PRESENTATION / "processbench_performance.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    y = np.arange(len(contrast_rows))
    points = np.asarray([row["delta"] for row in contrast_rows], dtype=float)
    low = points - np.asarray([row["ci_low"] for row in contrast_rows], dtype=float)
    high = np.asarray([row["ci_high"] for row in contrast_rows], dtype=float) - points
    ax.axvline(0.0, color="#17202a", lw=1.2)
    ax.errorbar(points, y, xerr=np.vstack([low, high]), fmt="o", color="#c0392b", capsize=5, markersize=7)
    ax.set_yticks(y, [f"vs {row['control_label']}" for row in contrast_rows])
    ax.invert_yaxis()
    ax.set_xlabel("Joint-or-flat minus control macro-F1")
    ax.set_title("All three paired intervals are wholly negative")
    ax.set_xlim(-0.10, 0.015)
    for position, value in enumerate(points):
        ax.text(value + 0.003, position, f"{value:+.4f}", va="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(PRESENTATION / "processbench_contrasts.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    x = np.arange(len(cells))
    width = 0.19
    for method_position, method in enumerate(METHODS):
        values = [result["per_cell_oof_f1"][f"{method}::{cell}"] for cell in cells]
        ax.bar(x + (method_position - 1.5) * width, values, width, label=LABELS[method], color=COLORS[method])
    fallback_position = cells.index("processbench_math_qwen3_4b")
    ax.axvspan(fallback_position - 0.48, fallback_position + 0.48, color="#f5b7b1", alpha=0.22)
    ax.text(fallback_position, 0.435, "flat fallback", ha="center", va="top", fontsize=9, color="#922b21")
    ax.set_xticks(x, [_short_cell(cell) for cell in cells])
    ax.set_ylim(0.0, 0.45)
    ax.set_ylabel("OOF F1 under all-eight calibration")
    ax.set_title("The loss is not confined to the fallback cell")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    fig.savefig(PRESENTATION / "processbench_per_cell.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fitted7 = result["fitted7_selection_conditioned_equal_cell_mean_f1"]
    candidate = result["point_metrics"][CANDIDATE]
    iu = result["point_metrics"][METHODS[1]]
    equal = result["point_metrics"][METHODS[2]]
    fixed = result["point_metrics"][METHODS[3]]
    delta_iu = stats[f"delta_candidate_vs::{METHODS[1]}"]
    delta_fixed = stats[f"delta_candidate_vs::{METHODS[3]}"]
    delta_equal = stats[f"delta_candidate_vs::{METHODS[2]}"]
    fallback_f1 = result["per_cell_oof_f1"][f"{CANDIDATE}::processbench_math_qwen3_4b"]
    q8_gsm_f1 = result["per_cell_oof_f1"][f"{CANDIDATE}::processbench_gsm8k_qwen3_8b"]
    report = f"""# Joint L-SML ProcessBench amendment result

Status: `HARM`. Scope: retrospective opened development after the PRMBench result was already open.

The all-eight candidate is a disclosed coverage policy, not pure Joint L-SML: seven cells use the frozen Joint head and `processbench_math_qwen3_4b` uses the exact flat-SML alias up to global sign gauge.

## Primary result

- Joint-or-flat macro-F1: **{candidate:.6f}**.
- IU-PCR: **{iu:.6f}**; candidate delta **{delta_iu['point']:+.6f}** (descriptive paired 95% interval **[{delta_iu['ci_low']:+.6f}, {delta_iu['ci_high']:+.6f}]**).
- Fixed-family continuous L-SML: **{fixed:.6f}**; candidate delta **{delta_fixed['point']:+.6f}** (95% interval **[{delta_fixed['ci_low']:+.6f}, {delta_fixed['ci_high']:+.6f}]**).
- Equal-family: **{equal:.6f}**; candidate delta **{delta_equal['point']:+.6f}** (95% interval **[{delta_equal['ci_low']:+.6f}, {delta_equal['ci_high']:+.6f}]**).
- Population: 3,400 source questions, 6,800 paired model rows, 2,000 paired source-question bootstrap draws with threshold refit.

![ProcessBench performance](presentation/processbench_performance.png)

Observation: the candidate is below every matched control and every paired contrast is wholly negative.

Inference: the registered decision is `HARM`; the current Joint head should not replace IU-PCR or fixed-family L-SML.

Limitation: this is opened development evidence, and the candidate includes a one-cell structural fallback.

## Where the failure occurs

![Per-cell ProcessBench F1](presentation/processbench_per_cell.png)

Observation: the fallback Qwen3-4B/MATH cell is very poor at **{fallback_f1:.4f}** F1, but a pure-Joint cell, Qwen3-8B/GSM8K, is also poor at **{q8_gsm_f1:.4f}**. Across only the seven parent-admissible cells, the selection-conditioned mean is **{fitted7[CANDIDATE]:.4f}** for Joint, versus **{fitted7[METHODS[1]]:.4f}** IU-PCR and **{fitted7[METHODS[3]]:.4f}** fixed-family L-SML.

Inference: the negative result is not explained solely by the flat-SML fallback. The hierarchical cross-group weighting itself is unstable for localization on at least one fitted cell.

Limitation: the seven-cell diagnostic reuses thresholds calibrated by the all-eight procedure, has no interval, and is not fallback-independent or a complete-panel estimand.

## Paired contrasts

![ProcessBench paired contrasts](presentation/processbench_contrasts.png)

Observation: Joint loses by 7.11 F1 points to IU-PCR and 7.36 points to fixed-family L-SML; even equal-family is ahead by 1.67 points.

Inference: lower covariance misfit is not a sufficient selection criterion for a localization fusion head. The next method iteration should gate or regularize the weight map itself, not add more covariance-fit flexibility on these opened labels.

Limitation: no new variant may be selected from this population and then described as confirmation; a new algorithm needs a newly frozen protocol and fresh data for generalization.

## Reducer and historical boundary

ProcessBench uses detector=max token risk and locator=argmax of the fixed top-`min(10, step_length)` mean. It is not top-5 and not top-10-percent. The historical `0.3662` value belongs to a different H2/H3 configuration and remains an audit anchor, not the matched comparator for this active-23 experiment.

PRMBench is not reevaluated here. Its parent result remains `HARM`: pure Joint AUROC 0.669063 versus IU-PCR 0.671539 and fixed-family L-SML 0.672619.

No promotion or generalization claim is allowed.
"""
    REPORT.write_text(report, encoding="utf-8")

    artifacts = [
        SUMMARY,
        AUDIT,
        REPORT,
        *sorted(PRESENTATION.iterdir()),
        PB_ROOT / "EXECUTION_REGISTRY.json",
        PB_ROOT / "SCORE_FREEZE_MANIFEST.json",
        PB_ROOT / "POLICY_LEDGER.json",
        PB_ROOT / "INDEPENDENT_SCORE_FREEZE_AUDIT.json",
        PB_ROOT / "EVALUATION_REGISTRY.json",
        PB_ROOT / "INDEPENDENT_EVALUATION_REGISTRY_AUDIT.json",
    ]
    final = {
        "schema": "joint-lsml-processbench-amendment-final-v1",
        "status": "HARM__NO_PROMOTION",
        "evaluation_summary_sha256": sha256_file(SUMMARY),
        "independent_result_audit_sha256": sha256_file(AUDIT),
        "parent_prmbench_final_sha256": sha256_file(ROOT / "FINAL_COMPLETE.json"),
        "renderer_sha256": sha256_file(Path(__file__)),
        "artifacts": {str(path.relative_to(REPO)): sha256_file(path) for path in artifacts},
        "fresh_generalization_recommended": False,
        "promotion_allowed": False,
    }
    final["payload_sha256"] = payload_sha256(final)
    atomic_write_json(FINAL, final)


if __name__ == "__main__":
    main()
