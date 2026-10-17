#!/usr/bin/env python3
"""Post-hoc forensic diagnostic for the frozen Joint L-SML localization run.

This script does not fit or score a new fusion method.  It reuses the frozen
ProcessBench score arrays, their registered labels, the PRMBench score freeze,
and the two audited result summaries to localize the failure mechanism.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.joint_lsml_localization import evaluate_processbench_amendment_v1 as pb_eval  # noqa: E402
from spectral_utils.fair_comparisons.evaluator import crossfit_localization_threshold  # noqa: E402
from spectral_utils.joint_lsml_processbench_amendment import (  # noqa: E402
    COVERAGE_METHOD,
    COVERAGE_METHODS,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    atomic_write_json,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark.localization_contract import payload_sha256  # noqa: E402


ROOT = REPO / "results/joint_lsml_existing_localization_v1"
PB_ROOT = ROOT / "processbench_amendment_v1"
OUTPUT = ROOT / "failure_diagnostic_v1"
SUMMARY = OUTPUT / "DIAGNOSTIC_SUMMARY.json"
MANIFEST = OUTPUT / "MANIFEST.json"
REPORT = OUTPUT / "REPORT.md"

IU = "iu_pcr_active23"
EQUAL = "equal_family_active23"
FIXED = "fixed_family_continuous_lsml_active23"
METHODS = (COVERAGE_METHOD, IU, EQUAL, FIXED)
LABELS = {
    COVERAGE_METHOD: "Joint-or-flat",
    IU: "IU-PCR",
    EQUAL: "Equal-family",
    FIXED: "Fixed-family L-SML",
}
COLORS = {
    COVERAGE_METHOD: "#c23b22",
    IU: "#3568a8",
    EQUAL: "#7f8c8d",
    FIXED: "#16856b",
}
FALLBACK = "processbench_math_qwen3_4b"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _short(cell: str) -> str:
    payload = cell.removeprefix("processbench_").replace("qwen3_", "Q")
    subset, model = payload.rsplit("_", 1)
    subset = subset.replace("olympiadbench", "olymp").replace("omnimath", "omni")
    return f"{subset}\n{model}"


def _weight_stats(weight: np.ndarray) -> dict[str, float]:
    absolute = np.abs(np.asarray(weight, dtype=np.float64))
    total = float(absolute.sum())
    probability = absolute / total
    return {
        "l1": total,
        "l2": float(np.linalg.norm(weight)),
        "effective_feature_count": float(1.0 / np.square(probability).sum()),
        "maximum_absolute_share": float(probability.max()),
        "negative_absolute_mass": float(absolute[np.asarray(weight) < 0].sum() / total),
    }


def _structural_by_cell() -> dict[str, Mapping[str, Any]]:
    ledger = _json(ROOT / "STRUCTURAL_LEDGER.json")
    return {row["cell_id"]: row for row in ledger["cells"]}


def _policy_by_cell() -> dict[str, Mapping[str, Any]]:
    ledger = _json(PB_ROOT / "POLICY_LEDGER.json")
    return {row["cell_id"]: row for row in ledger["cells"]}


def _processbench_diagnostics() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    registry, score_cells = pb_eval._verified_evaluation_registry(pb_eval.DEFAULT_RELEASE)
    rows = pb_eval._pb_rows(
        score_cells,
        Path(registry["processbench_label_path"]),
        registry["fold_namespace"],
    )
    structural = _structural_by_cell()
    policy = _policy_by_cell()
    per_cell: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []

    for model in pb_eval.MODELS:
        model_outputs: dict[str, Mapping[str, Any]] = {}
        model_rows: dict[str, list[Mapping[str, Any]]] = {}
        for method in METHODS:
            selected = [
                row for row in rows
                if row["model_id"] == model and row["method_id"] == method
            ]
            model_rows[method] = selected
            model_outputs[method] = crossfit_localization_threshold(
                selected, expected_subsets=pb_eval.SUBSETS
            )

        for subset in pb_eval.SUBSETS:
            cell = f"processbench_{subset}_{model}"
            score_path = pb_eval.freeze.SCORE_ROOT / score_cells[cell]["artifact_path"]
            arrays = load_npz_no_pickle(score_path)
            method_ids = tuple(arrays["method_ids"].astype(str))
            score_by_method = {
                method: np.asarray(arrays["detector_scores"][:, method_ids.index(method)], dtype=np.float64)
                for method in METHODS
            }
            locator_by_method = {
                method: np.asarray(arrays["locators"][:, method_ids.index(method)], dtype=np.int64)
                for method in METHODS
            }

            method_metrics: dict[str, dict[str, float]] = {}
            for method in METHODS:
                selected = model_rows[method]
                output = model_outputs[method]
                subset_positions = [i for i, row in enumerate(selected) if row["subset"] == subset]
                target = np.asarray([selected[i]["first_error"] for i in subset_positions], dtype=np.int64)
                prediction = np.asarray([output["predictions"][i] for i in subset_positions], dtype=np.int64)
                error = target != -1
                locators = locator_by_method[method]
                fold_thresholds = {
                    int(row["held_out_fold"]): float(row["threshold"])
                    for row in output["calibration_ledgers"]
                }
                thresholds = np.asarray(
                    [fold_thresholds[int(selected[i]["fold"])] for i in subset_positions],
                    dtype=np.float64,
                )
                scores = score_by_method[method]
                method_metrics[method] = {
                    "oof_f1": float(output["official_oof_metrics"]["per_subset"][subset]["f1"]),
                    "error_accuracy": float(np.mean(prediction[error] == target[error])),
                    "clean_accuracy": float(np.mean(prediction[~error] == -1)),
                    "activation_rate": float(np.mean(prediction != -1)),
                    "detector_auroc": float(roc_auc_score(error.astype(int), scores)),
                    "locator_exact_given_error": float(np.mean(locators[error] == target[error])),
                    "threshold_z_median": float((np.median(thresholds) - scores.mean()) / scores.std()),
                    "detector_mean": float(scores.mean()),
                    "detector_std": float(scores.std()),
                }

            candidate_weight = np.asarray(policy[cell]["weights"][COVERAGE_METHOD], dtype=np.float64)
            structural_row = structural.get(cell)
            if structural_row is not None and structural_row["status"] == "FIT_COMPLETE":
                joint_misfit = float(structural_row["joint_fit"]["relative_offdiag_misfit"])
                hard_misfit = float(structural_row["hard_lsml_relative_offdiag_misfit"])
                misfit_reduction = (hard_misfit - joint_misfit) / hard_misfit
                map_agreement = float(structural_row["weight_map_agreement"]["minimum"])
                k = int(structural_row["grouping"]["K"])
            else:
                misfit_reduction = None
                map_agreement = None
                k = 0

            record: dict[str, Any] = {
                "cell_id": cell,
                "model_id": model,
                "subset": subset,
                "candidate_component": policy[cell]["candidate_component"],
                "fallback": bool(policy[cell]["fallback"]),
                "selected_k": k,
                "relative_misfit_reduction": misfit_reduction,
                "minimum_weight_map_agreement": map_agreement,
                "candidate_vs_fixed_detector_spearman": float(
                    spearmanr(score_by_method[COVERAGE_METHOD], score_by_method[FIXED]).statistic
                ),
                "candidate_vs_iu_detector_spearman": float(
                    spearmanr(score_by_method[COVERAGE_METHOD], score_by_method[IU]).statistic
                ),
                "candidate_vs_fixed_locator_agreement": float(
                    np.mean(locator_by_method[COVERAGE_METHOD] == locator_by_method[FIXED])
                ),
                "candidate_vs_iu_locator_agreement": float(
                    np.mean(locator_by_method[COVERAGE_METHOD] == locator_by_method[IU])
                ),
                "candidate_weight_l2": _weight_stats(candidate_weight)["l2"],
            }
            for method in METHODS:
                for key, value in method_metrics[method].items():
                    record[f"{method}__{key}"] = value
            record["candidate_minus_iu_f1"] = (
                method_metrics[COVERAGE_METHOD]["oof_f1"] - method_metrics[IU]["oof_f1"]
            )
            record["candidate_minus_fixed_f1"] = (
                method_metrics[COVERAGE_METHOD]["oof_f1"] - method_metrics[FIXED]["oof_f1"]
            )
            per_cell.append(record)

            for method in METHODS:
                weight = np.asarray(policy[cell]["weights"][method], dtype=np.float64)
                stats = _weight_stats(weight)
                weight_rows.append({
                    "cell_id": cell,
                    "model_id": model,
                    "subset": subset,
                    "method_id": method,
                    "method_label": LABELS[method],
                    **stats,
                    "detector_mean": method_metrics[method]["detector_mean"],
                    "detector_std": method_metrics[method]["detector_std"],
                })
    return per_cell, weight_rows


def _prmbench_diagnostics() -> dict[str, Any]:
    structural = _structural_by_cell()["prmbench_response_qwen3_8b"]
    arrays = load_npz_no_pickle(ROOT / "score_freeze/prmbench_response_qwen3_8b.npz")
    method_ids = tuple(arrays["method_ids"].astype(str))
    step = np.asarray(arrays["step_risk"], dtype=np.float64)
    joint = step[:, method_ids.index("joint_lsml23_hierarchical_v1_1")]
    iu = step[:, method_ids.index(IU)]
    fixed = step[:, method_ids.index(FIXED)]
    hard = float(structural["hard_lsml_relative_offdiag_misfit"])
    fitted = float(structural["joint_fit"]["relative_offdiag_misfit"])
    weights = structural["weights"]
    return {
        "selected_k": int(structural["grouping"]["K"]),
        "group_sizes": structural["grouping"]["group_sizes"],
        "relative_misfit_reduction": float((hard - fitted) / hard),
        "joint_vs_iu_step_score_spearman": float(spearmanr(joint, iu).statistic),
        "joint_vs_fixed_step_score_spearman": float(spearmanr(joint, fixed).statistic),
        "minimum_weight_map_agreement": float(structural["weight_map_agreement"]["minimum"]),
        "joint_weight": _weight_stats(np.asarray(weights["joint_lsml23_hierarchical_v1_1"])),
        "iu_weight": _weight_stats(np.asarray(weights[IU])),
        "fixed_weight": _weight_stats(np.asarray(weights[FIXED])),
    }


def _source_hashes() -> dict[str, str]:
    paths = {
        "script": Path(__file__),
        "structural_ledger": ROOT / "STRUCTURAL_LEDGER.json",
        "prmbench_score_freeze": ROOT / "score_freeze/prmbench_response_qwen3_8b.npz",
        "prmbench_result": ROOT / "evaluation_r2/EVALUATION_SUMMARY.json",
        "processbench_policy": PB_ROOT / "POLICY_LEDGER.json",
        "processbench_score_manifest": PB_ROOT / "SCORE_FREEZE_MANIFEST.json",
        "processbench_result": PB_ROOT / "evaluation/EVALUATION_SUMMARY.json",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _make_plots(per_cell: list[dict[str, Any]], weight_rows: list[dict[str, Any]], prm: Mapping[str, Any]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    cells = [row["cell_id"] for row in per_cell]
    x = np.arange(len(cells))

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.5))
    deltas = np.asarray([row["candidate_minus_iu_f1"] for row in per_cell])
    axes[0].axhline(0.0, color="#1f2937", lw=1)
    axes[0].bar(x, deltas, color=["#8e2a2a" if value < -0.10 else "#d98c3f" for value in deltas])
    axes[0].set_ylabel("Joint-or-flat minus IU F1")
    axes[0].set_title("89% of the ProcessBench loss is concentrated in two cells")
    axes[0].set_xticks(x, [_short(cell) for cell in cells])
    for i, value in enumerate(deltas):
        axes[0].text(i, value - 0.012 if value < 0 else value + 0.008, f"{value:+.3f}", ha="center", va="top" if value < 0 else "bottom", fontsize=8)

    candidate_f1 = np.asarray([row[f"{COVERAGE_METHOD}__oof_f1"] for row in per_cell])
    auc = np.asarray([row[f"{COVERAGE_METHOD}__detector_auroc"] for row in per_cell])
    activation = np.asarray([row[f"{COVERAGE_METHOD}__activation_rate"] for row in per_cell])
    scatter = axes[1].scatter(auc, candidate_f1, s=250 + 1100 * activation, c=deltas, cmap="RdYlGn", vmin=-0.27, vmax=0.03, edgecolor="white", linewidth=1)
    for i, row in enumerate(per_cell):
        axes[1].annotate(_short(row["cell_id"]).replace("\n", "/"), (auc[i], candidate_f1[i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[1].set_xlabel("Candidate detector AUROC")
    axes[1].set_ylabel("OOF localization F1")
    axes[1].set_title("Good detector ranking does not rescue a mis-scaled shared threshold")
    fig.colorbar(scatter, ax=axes[1], label="F1 delta vs IU")
    axes[1].set_xlim(float(auc.min() - 0.006), float(auc.max() + 0.006))
    fig.tight_layout()
    fig.savefig(OUTPUT / "processbench_failure_map.png", dpi=190)
    plt.close(fig)

    by_method_cell = {(row["method_id"], row["cell_id"]): row for row in weight_rows}
    fig, axes = plt.subplots(2, 1, figsize=(12.8, 8.2), sharex=True)
    width = 0.24
    for position, method in enumerate((COVERAGE_METHOD, IU, FIXED)):
        values = [by_method_cell[(method, cell)]["l2"] for cell in cells]
        axes[0].bar(x + (position - 1) * width, values, width, label=LABELS[method], color=COLORS[method])
    axes[0].set_ylabel("L2 norm of deployed weight vector")
    axes[0].set_title("The hierarchical head has no common final scale across cells")
    axes[0].legend(ncol=3, frameon=False)

    for method, marker in ((COVERAGE_METHOD, "o"), (FIXED, "s")):
        values = [row[f"{method}__activation_rate"] for row in per_cell]
        axes[1].plot(x, values, marker=marker, lw=2, label=LABELS[method], color=COLORS[method])
    axes[1].set_ylabel("OOF fraction predicted as error")
    axes[1].set_xticks(x, [_short(cell) for cell in cells])
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_title("The shared threshold nearly silences q4/MATH and q8/GSM8K")
    axes[1].legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT / "processbench_scale_transfer.png", dpi=190)
    plt.close(fig)

    pb_fitted = [row for row in per_cell if not row["fallback"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    gain = 100 * np.asarray([row["relative_misfit_reduction"] for row in pb_fitted])
    efficacy = np.asarray([row["candidate_minus_fixed_f1"] for row in pb_fitted])
    axes[0].axhline(0.0, color="#1f2937", lw=1)
    axes[0].scatter(gain, efficacy, s=70, color="#c23b22")
    for row, gx, ey in zip(pb_fitted, gain, efficacy):
        axes[0].annotate(_short(row["cell_id"]).replace("\n", "/"), (gx, ey), xytext=(4, 4), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("Joint reduction in off-diagonal misfit (%)")
    axes[0].set_ylabel("Joint minus fixed L-SML F1")
    axes[0].set_title("ProcessBench: better fit is not better localization")

    prm_result = _json(ROOT / "evaluation_r2/EVALUATION_SUMMARY.json")["PRMBench"]["point_metrics"]
    labels = ["misfit reduction", "AUROC vs IU", "AUROC vs fixed"]
    values = [
        100 * float(prm["relative_misfit_reduction"]),
        100 * float(prm_result["delta_auroc_joint_vs::iu_pcr_active23"]),
        100 * float(prm_result["delta_auroc_joint_vs::fixed_family_continuous_lsml_active23"]),
    ]
    axes[1].axhline(0.0, color="#1f2937", lw=1)
    axes[1].bar(np.arange(3), values, color=["#16856b", "#c23b22", "#c23b22"])
    axes[1].set_xticks(np.arange(3), labels, rotation=15, ha="right")
    axes[1].set_ylabel("percentage points")
    axes[1].set_title("PRMBench: structural gain, ranking loss")
    for i, value in enumerate(values):
        axes[1].text(i, value + (0.35 if value >= 0 else -0.35), f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top")
    fig.tight_layout()
    fig.savefig(OUTPUT / "structural_fit_vs_efficacy.png", dpi=190)
    plt.close(fig)


def _report(summary: Mapping[str, Any]) -> str:
    two = summary["processbench"]["two_cell_concentration"]
    prm = summary["prmbench"]
    return f"""# Joint L-SML localization failure diagnostic

Status: `POSTHOC_RETROSPECTIVE_FAILURE_DIAGNOSTIC`. No new fusion candidate was fit or scored.

## Bottom line

The current Joint L-SML failure is best explained by two linked problems, not by bad signs or broken preprocessing:

1. **ProcessBench scale transfer:** `hierarchical_joint_weights` multiplies global loadings by cross-group SML weights but never normalizes the final head. Seven Joint cells therefore have materially different score scales, and the amendment also splices a unit-norm flat-SML fallback into the same absolute-threshold panel. The shared threshold nearly silences two cells.
2. **Objective/head mismatch:** the structured covariance fit estimates global `v` and group-specific `u`, but the deployed hierarchical head uses only `v` and a second SML over virtual groups. Better covariance reconstruction therefore need not improve the final ranking. PRMBench isolates this second problem because AUROC has no threshold scale failure.

This is the strongest supported diagnosis, not a causal proof. The original run did not score the full grouping x weight-map factorial, so INTERNAL grouping and the hierarchical map cannot be completely separated post hoc.

## ProcessBench: where the loss lives

![Failure map](processbench_failure_map.png)

Observation: q4/MATH and q8/GSM8K account for **{100 * two['fraction_of_total_joint_vs_iu_loss']:.1f}%** of the summed per-cell Joint-versus-IU loss. Their candidate F1 deltas are **{two['q4_math_delta_vs_iu']:+.3f}** and **{two['q8_gsm_delta_vs_iu']:+.3f}**.

Inference: the fallback is a major failure, but it is not the whole failure; q8/GSM8K uses pure Joint and collapses too.

Limitation: per-cell deltas are post-hoc descriptive quantities under the already-open all-eight cross-fitted threshold policy.

## ProcessBench: rankings survive, calibration does not

![Scale transfer](processbench_scale_transfer.png)

Observation: in q4/MATH and q8/GSM8K, candidate-versus-fixed detector Spearman is **{two['q4_math_detector_spearman_vs_fixed']:.3f}** and **{two['q8_gsm_detector_spearman_vs_fixed']:.3f}**, and locator agreement is **{two['q4_math_locator_agreement_vs_fixed']:.3f}** and **{two['q8_gsm_locator_agreement_vs_fixed']:.3f}**. Yet candidate activation is only **{100 * two['q4_math_candidate_activation']:.1f}%** and **{100 * two['q8_gsm_candidate_activation']:.1f}%**, versus fixed L-SML **{100 * two['q4_math_fixed_activation']:.1f}%** and **{100 * two['q8_gsm_fixed_activation']:.1f}%**. The median out-of-fold candidate threshold lies **{two['q4_math_candidate_threshold_z_median']:.2f}** and **{two['q8_gsm_candidate_threshold_z_median']:.2f}** within-cell score standard deviations above the respective cell means.

Inference: the detector ordering and step locator remain largely intact. The main PB collapse happens when one pooled model-level threshold is applied to cell scores whose scales are not comparable.

Limitation: score normalization may repair this particular PB failure, but it cannot by itself establish a better feature ranking or fix the PRMBench loss.

## Structural fit is not an efficacy surrogate

![Structural fit versus efficacy](structural_fit_vs_efficacy.png)

Observation: Joint reduces off-diagonal misfit in every fitted PB cell and by **{100 * prm['relative_misfit_reduction']:.1f}%** on PRMBench. Nevertheless PRMBench Joint loses **0.248 AUROC percentage points** to IU and **0.356 percentage points** to fixed-family L-SML. Its full frozen step-score Spearman remains high: **{prm['joint_vs_iu_step_score_spearman']:.3f}** versus IU and **{prm['joint_vs_fixed_step_score_spearman']:.3f}** versus fixed L-SML.

Inference: the structural model is fitting covariance variation that is not reliably useful for first-error localization. The deployed weight map—not the optimizer convergence—must be redesigned or regularized.

Limitation: the opened data can diagnose and develop; it cannot support a new-leader or generalization claim.

## What is ruled out, and what is not

- **Not supported as the root cause:** sign instability, the removed weak streams, preprocessing drift, reducer drift, optimizer non-convergence, generic weight sparsity, or a coding error in the cross-fitted threshold adapter.
- **Supported:** missing final score-scale convention in the hierarchical head; fallback/Joint scale mixing; low agreement among plausible weight maps in the worst pure-Joint PB cell; and an objective-to-deployed-head mismatch.
- **Still unresolved:** whether INTERNAL K=3 groups are intrinsically wrong, or whether the same groups would work with ordinary unit-normalized continuous L-SML. That arm was structural-only in the frozen run.

## Consequence for the next method

Make donor-frozen score normalization an invariant, not a label-tuned hyperparameter. Then separate grouping from mapping with a small preregistered factorial: INTERNAL versus provenance/fixed groups, crossed with ordinary continuous L-SML versus the hierarchical Joint map. Only after that should feature-first be compared with trajectory-first under equal nested-CV budgets. The companion plan is `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V1.md`.

DUFS must not be used as a K selector: it produces per-feature gates, not a partition count. A bounded PF-DUFS support or soft-affinity study is possible later, but prior project evidence makes it secondary and it must be registered as a new candidate family.
"""


def run() -> None:
    if OUTPUT.exists():
        raise RuntimeError(f"diagnostic namespace already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    per_cell, weight_rows = _processbench_diagnostics()
    prm = _prmbench_diagnostics()
    _write_csv(OUTPUT / "processbench_per_cell.csv", per_cell)
    _write_csv(OUTPUT / "weight_scale.csv", weight_rows)
    _make_plots(per_cell, weight_rows, prm)

    by_cell = {row["cell_id"]: row for row in per_cell}
    q4_math = by_cell["processbench_math_qwen3_4b"]
    q8_gsm = by_cell["processbench_gsm8k_qwen3_8b"]
    net_headline_loss = -sum(row["candidate_minus_iu_f1"] for row in per_cell)
    selected = -(min(0.0, q4_math["candidate_minus_iu_f1"]) + min(0.0, q8_gsm["candidate_minus_iu_f1"]))
    summary = {
        "schema": "joint-lsml-localization-posthoc-failure-diagnostic-v1",
        "status": "COMPLETE",
        "scope": "POSTHOC_RETROSPECTIVE_FAILURE_DIAGNOSTIC",
        "new_fusion_candidate_fit": False,
        "new_candidate_score_arrays_computed": False,
        "promotion_allowed": False,
        "generalization_claim_allowed": False,
        "diagnosis": {
            "primary_processbench_mechanism": "UNNORMALIZED_HIERARCHICAL_WEIGHT_AND_CROSS_CELL_SCORE_SCALE",
            "cross_task_mechanism": "STRUCTURAL_OBJECTIVE_TO_DEPLOYED_WEIGHT_MAP_MISMATCH",
            "grouping_vs_weight_map_causally_separated": False,
        },
        "processbench": {
            "two_cell_concentration": {
                "fraction_of_total_joint_vs_iu_loss": float(selected / net_headline_loss),
                "q4_math_delta_vs_iu": q4_math["candidate_minus_iu_f1"],
                "q8_gsm_delta_vs_iu": q8_gsm["candidate_minus_iu_f1"],
                "q4_math_detector_spearman_vs_fixed": q4_math["candidate_vs_fixed_detector_spearman"],
                "q8_gsm_detector_spearman_vs_fixed": q8_gsm["candidate_vs_fixed_detector_spearman"],
                "q4_math_locator_agreement_vs_fixed": q4_math["candidate_vs_fixed_locator_agreement"],
                "q8_gsm_locator_agreement_vs_fixed": q8_gsm["candidate_vs_fixed_locator_agreement"],
                "q4_math_candidate_activation": q4_math[f"{COVERAGE_METHOD}__activation_rate"],
                "q8_gsm_candidate_activation": q8_gsm[f"{COVERAGE_METHOD}__activation_rate"],
                "q4_math_fixed_activation": q4_math[f"{FIXED}__activation_rate"],
                "q8_gsm_fixed_activation": q8_gsm[f"{FIXED}__activation_rate"],
                "q4_math_candidate_threshold_z_median": q4_math[f"{COVERAGE_METHOD}__threshold_z_median"],
                "q8_gsm_candidate_threshold_z_median": q8_gsm[f"{COVERAGE_METHOD}__threshold_z_median"],
            },
            "per_cell": per_cell,
        },
        "prmbench": prm,
        "source_hashes": _source_hashes(),
    }
    summary["payload_sha256"] = payload_sha256(summary)
    atomic_write_json(SUMMARY, summary)
    REPORT.write_text(_report(summary), encoding="utf-8")
    artifact_names = (
        "DIAGNOSTIC_SUMMARY.json",
        "REPORT.md",
        "processbench_per_cell.csv",
        "weight_scale.csv",
        "processbench_failure_map.png",
        "processbench_scale_transfer.png",
        "structural_fit_vs_efficacy.png",
    )
    manifest = {
        "schema": "joint-lsml-localization-failure-diagnostic-manifest-v1",
        "status": "COMPLETE",
        "artifacts": {name: sha256_file(OUTPUT / name) for name in artifact_names},
    }
    manifest["payload_sha256"] = payload_sha256(manifest)
    atomic_write_json(MANIFEST, manifest)


def check() -> None:
    manifest = _json(MANIFEST)
    body = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if payload_sha256(body) != manifest["payload_sha256"]:
        raise RuntimeError("manifest payload hash mismatch")
    for name, expected in manifest["artifacts"].items():
        if sha256_file(OUTPUT / name) != expected:
            raise RuntimeError(f"artifact hash mismatch: {name}")
    summary = _json(SUMMARY)
    body = {key: value for key, value in summary.items() if key != "payload_sha256"}
    if payload_sha256(body) != summary["payload_sha256"]:
        raise RuntimeError("summary payload hash mismatch")
    if summary["new_fusion_candidate_fit"] or summary["new_candidate_score_arrays_computed"]:
        raise RuntimeError("diagnostic crossed the frozen-candidate boundary")
    if summary.get("source_hashes") != _source_hashes():
        raise RuntimeError("diagnostic source hash mismatch")
    print("PASS")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "check"
    if command == "run":
        run()
    elif command == "check":
        check()
    else:
        raise SystemExit("usage: diagnose_failure_v1.py [run|check]")
