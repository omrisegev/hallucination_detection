#!/usr/bin/env python3
"""Declaration, length, missing-stream, transfer, and cost audits for v2."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import pickle
import sys
import time
import tracemalloc

import numpy as np
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.run_global_local_online_architecture_v2 as run  # noqa: E402
from spectral_utils.multitask_trajectory import causal_states  # noqa: E402
from spectral_utils.online_convergence import normalize_cache_records  # noqa: E402


OUT = run.OUT
PHASE15 = Path(
    "/private/tmp/hallucination_phase1_audit_20260816/"
    "math500_qwen7b_T1.0_run0.pkl"
)


def _two_scores(rows, output, global_fit, local_fit):
    final = 0.5 * run._zapply(output["global_final"], global_fit) + 0.5 * run._zapply(
        [np.max(curve) for curve in output["local_curves"]], local_fit
    )
    prefix = {}
    for budget in run.BUDGETS:
        prefix[budget] = (
            0.5 * run._zapply(output["global_prefix"][budget], global_fit)
            + 0.5 * run._zapply(output["local_prefix"][budget], local_fit)
        )
    return np.asarray(final), prefix


def _declaration_rows(
    model_name, family, calibration, evaluation, cal_prefix, eval_prefix,
    cal_target, eval_target,
):
    output = []
    for target_fpr in (0.05, 0.10):
        maxima = []
        for index, row in enumerate(calibration):
            if cal_target[index] != 0:
                continue
            values = [
                cal_prefix[budget][index] for budget in run.BUDGETS
                if len(row["token_entropies"]) > budget
            ]
            if values:
                maxima.append(float(np.max(values)))
        threshold = float(np.quantile(maxima, 1.0 - target_fpr, method="higher"))
        warnings, first_budgets = [], []
        for index, row in enumerate(evaluation):
            hits = [
                budget for budget in run.BUDGETS
                if len(row["token_entropies"]) > budget
                and eval_prefix[budget][index] > threshold
            ]
            warnings.append(bool(hits))
            first_budgets.append(min(hits) if hits else None)
        warnings = np.asarray(warnings, dtype=bool)
        wrong, correct = eval_target == 1, eval_target == 0
        caught = wrong & warnings
        remaining = [
            len(row["token_entropies"]) - int(budget)
            for row, use, budget in zip(evaluation, caught, first_budgets)
            if use and budget is not None
        ]
        output.append({
            "model": model_name, "family": family, "target_fpr": target_fpr,
            "threshold": threshold, "n": len(evaluation),
            "wrong_warning_coverage": float(np.mean(warnings[wrong])) if wrong.any() else float("nan"),
            "correct_ever_warning": float(np.mean(warnings[correct])) if correct.any() else float("nan"),
            "overall_warning_coverage": float(np.mean(warnings)),
            "mean_first_warning_budget": float(np.mean([value for value in first_budgets if value is not None])) if warnings.any() else float("nan"),
            "potential_tokens_remaining_on_caught_wrong": float(np.mean(remaining)) if remaining else float("nan"),
            "realized_savings": False,
        })
    return output


def _length_rows(
    model_name, family, calibration, evaluation, cal_prefix, eval_prefix,
    cal_final, eval_final, eval_target,
):
    output = []
    for budget in run.BUDGETS:
        cal_keep = np.asarray([len(row["token_entropies"]) > budget for row in calibration])
        eval_keep = np.asarray([len(row["token_entropies"]) > budget for row in evaluation])
        cal_length = np.asarray([len(row["token_entropies"]) for row in calibration])[cal_keep]
        eval_length = np.asarray([len(row["token_entropies"]) for row in evaluation])[eval_keep]
        cal_score = np.asarray(cal_prefix[budget])[cal_keep]
        eval_score = np.asarray(eval_prefix[budget])[eval_keep]
        labels = eval_target[eval_keep]
        if len(cal_score) >= 3:
            isotonic = IsotonicRegression(out_of_bounds="clip").fit(cal_length, cal_score)
            residual = eval_score - isotonic.predict(eval_length)
        else:
            residual = np.full(len(eval_score), np.nan)
        correlation = float(spearmanr(eval_score, eval_length).statistic) if len(eval_score) > 2 else float("nan")
        final_correlation = float(spearmanr(eval_score, np.asarray(eval_final)[eval_keep]).statistic) if len(eval_score) > 2 else float("nan")
        q1, q2 = np.quantile(cal_length, (1 / 3, 2 / 3)) if len(cal_length) else (0, 0)
        bands = {
            "short": eval_length <= q1,
            "medium": (eval_length > q1) & (eval_length <= q2),
            "long": eval_length > q2,
        }
        base = {
            "model": model_name, "family": family, "budget": budget,
            "length_spearman": correlation, "final_score_spearman": final_correlation,
            "raw_auroc": run._safe_auc(labels, eval_score),
            "length_residual_auroc": run._safe_auc(labels, residual),
            "n": int(eval_keep.sum()),
        }
        for band, mask in bands.items():
            base[f"{band}_n"] = int(mask.sum())
            base[f"{band}_auroc"] = run._safe_auc(labels[mask], eval_score[mask])
        output.append(base)
    return output


def _score_curve_drop(model, reference, row, channel):
    states = causal_states(row, reference)
    raw = np.column_stack([states[name] for name in model.feature_names])
    if channel != "none":
        for full_index, name in enumerate(model.feature_names):
            if name.startswith(channel + "__") and model.keep[full_index]:
                selected_index = int(np.sum(model.keep[:full_index]))
                raw[:, full_index] = model.median[selected_index]
    return model.score_matrix(raw)


def _missing_sensitivity(
    model_name, family, calibration, evaluation, models, selection,
):
    output = []
    global_model = models.global_heads[selection["global"]]
    local_model = models.local_heads[selection["local"]]
    online_model = models.online_heads[selection["online"]]
    global_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in evaluation])
    local_cal_target = np.asarray([int(row["label"]) for row in calibration])
    local_eval_target = np.asarray([int(row["label"]) for row in evaluation])

    for group in ("none", "entropy", "spilled", "logsumexp", "topk"):
        changed_rows = []
        for row in evaluation:
            changed = deepcopy(row)
            if group == "entropy":
                changed["token_entropies"] = np.full(
                    len(row["token_entropies"]), models.reference.centres[0]
                )
            elif group == "spilled":
                changed.pop("token_spilled_energies", None)
            elif group == "logsumexp":
                changed.pop("token_logsumexp", None)
            elif group == "topk":
                changed.pop("top_k_logprobs", None)
            changed_rows.append(changed)
        scores = np.asarray([run._global_score(global_model, models.reference, row) for row in changed_rows])
        output.append({
            "model": model_name, "family": family, "head": "global",
            "missing": group, "metric": "auroc", "value": run._safe_auc(global_target, scores),
        })

    for head_name, model, task in (
        (selection["local"], local_model, "local"),
        (selection["online"], online_model, "online"),
    ):
        channels = ["none"] + [name.split("__", 1)[0] for name in model.feature_names]
        channels = list(dict.fromkeys(channels))
        for channel in channels:
            cal_curves = [_score_curve_drop(model, models.reference, row, channel) for row in calibration]
            eval_curves = [_score_curve_drop(model, models.reference, row, channel) for row in evaluation]
            if task == "local":
                cal_detector = np.asarray([np.max(curve) for curve in cal_curves])
                eval_detector = np.asarray([np.max(curve) for curve in eval_curves])
                cal_locator = np.asarray([run._peak_locator(curve, row) for curve, row in zip(cal_curves, calibration)])
                eval_locator = np.asarray([run._peak_locator(curve, row) for curve, row in zip(eval_curves, evaluation)])
                threshold, _ = run._best_threshold(cal_detector, cal_locator, local_cal_target)
                prediction = np.where(eval_detector > threshold, eval_locator, -1)
                value = run._processbench(prediction, local_eval_target)["f1"]
                metric = "f1"
            else:
                values = []
                for budget in (64, 128):
                    eligible = [index for index, row in enumerate(evaluation) if len(row["token_entropies"]) > budget]
                    labels = np.asarray([global_target[index] for index in eligible])
                    scores = np.asarray([eval_curves[index][budget - 1] for index in eligible])
                    values.append(run._safe_auc(labels, scores))
                value, metric = float(np.mean(values)), "auroc_64_128"
            output.append({
                "model": model_name, "family": family, "head": task,
                "missing": channel, "metric": metric, "value": value,
            })
    return output


def _phase15(selection):
    if not PHASE15.exists():
        return {"status": "UNAVAILABLE", "path": str(PHASE15)}
    with PHASE15.open("rb") as handle:
        normalized = normalize_cache_records(pickle.load(handle))
    rows = []
    for index, item in enumerate(normalized):
        row = dict(item)
        row["_unit"] = str(row.get("_group", index))
        row["_partition"] = run.stable_partition(row["_unit"])
        row["final_answer_correct"] = bool(row["label"])
        rows.append(row)
    calibration, evaluation = run._split(rows)
    models = run._fit_selected_cell(calibration, selection)
    cal_output = run._selected_outputs(calibration, models, selection)
    eval_output = run._selected_outputs(evaluation, models, selection)
    global_fit = run._zfit(cal_output["global_final"])
    local_fit = run._zfit([np.max(curve) for curve in cal_output["local_curves"]])
    cal_final, cal_prefix = _two_scores(calibration, cal_output, global_fit, local_fit)
    eval_final, eval_prefix = _two_scores(evaluation, eval_output, global_fit, local_fit)
    cal_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in calibration])
    eval_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in evaluation])
    budgets = {}
    for budget in run.BUDGETS:
        keep = np.asarray([len(row["token_entropies"]) > budget for row in evaluation])
        budgets[str(budget)] = {
            "n": int(keep.sum()),
            "auroc": run._safe_auc(eval_target[keep], eval_prefix[budget][keep]),
            "auprc": run._safe_ap(eval_target[keep], eval_prefix[budget][keep]),
        }
    declarations = _declaration_rows(
        "phase15_qwen7b", "math500", calibration, evaluation,
        cal_prefix, eval_prefix, cal_target, eval_target,
    )
    return {
        "status": "COMPLETE_RETROSPECTIVE_ONLINE_ONLY", "path": str(PHASE15),
        "n_calibration": len(calibration), "n_evaluation": len(evaluation),
        "missing_logsumexp": True, "budgets": budgets,
        "final_auroc": run._safe_auc(eval_target, eval_final),
        "declarations": declarations,
    }


def main() -> None:
    selection = json.load(open(OUT / "HEAD_SELECTION.json", encoding="utf-8"))["selected"]
    declarations, length_rows, efficiency, diagnostics = [], [], [], {}
    for model_name in run.MODELS:
        for family in run.FAMILIES:
            rows = run.load_rows(run._cell_path(model_name, family))
            calibration, evaluation = run._split(rows)
            print(f"[sensitivity] {model_name}/{family}", flush=True)
            tracemalloc.start(); started = time.perf_counter()
            models = run._fit_selected_cell(calibration, selection)
            fit_seconds = time.perf_counter() - started
            _, fit_peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
            started = time.perf_counter()
            cal_output = run._selected_outputs(calibration, models, selection)
            eval_output = run._selected_outputs(evaluation, models, selection)
            score_seconds = time.perf_counter() - started
            # Python allocation tracing makes the many small spectral-prefix
            # operations several times slower. Fit memory remains measured;
            # scoring reports wall time and an explicit unavailable sentinel.
            score_peak = -1
            global_fit = run._zfit(cal_output["global_final"])
            local_fit = run._zfit([np.max(curve) for curve in cal_output["local_curves"]])
            cal_final, cal_prefix = _two_scores(calibration, cal_output, global_fit, local_fit)
            eval_final, eval_prefix = _two_scores(evaluation, eval_output, global_fit, local_fit)
            cal_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in calibration])
            eval_target = np.asarray([int(not bool(row["final_answer_correct"])) for row in evaluation])
            declarations.extend(_declaration_rows(
                model_name, family, calibration, evaluation,
                cal_prefix, eval_prefix, cal_target, eval_target,
            ))
            length_rows.extend(_length_rows(
                model_name, family, calibration, evaluation,
                cal_prefix, eval_prefix, cal_final, eval_final, eval_target,
            ))
            efficiency.append({
                "model": model_name, "family": family,
                "fit_seconds": fit_seconds, "score_all_three_outputs_seconds": score_seconds,
                "fit_python_peak_bytes": fit_peak, "score_python_peak_bytes": score_peak,
                "n_calibration": len(calibration), "n_evaluation": len(evaluation),
            })
            diagnostics[f"{model_name}/{family}"] = {
                "reference": models.reference.as_dict(),
                "global": getattr(models.global_heads[selection["global"]], "diagnostics", {}),
                "local": getattr(models.local_heads[selection["local"]], "diagnostics", {}),
                "online": getattr(models.online_heads[selection["online"]], "diagnostics", {}),
            }
            run._write_csv(OUT / "DECLARATION_METRICS.partial.csv", declarations)
            run._write_csv(OUT / "LENGTH_SENSITIVITY.partial.csv", length_rows)
            run._write_csv(OUT / "END_TO_END_EFFICIENCY.partial.csv", efficiency)
            run._write_json(OUT / "SELECTED_HEAD_DIAGNOSTICS.partial.json", diagnostics)

    run._write_csv(OUT / "DECLARATION_METRICS.csv", declarations)
    run._write_csv(OUT / "LENGTH_SENSITIVITY.csv", length_rows)
    run._write_csv(OUT / "END_TO_END_EFFICIENCY.csv", efficiency)
    run._write_json(OUT / "SELECTED_HEAD_DIAGNOSTICS.json", diagnostics)
    run._write_json(OUT / "PHASE15_ONLINE_TRANSFER.json", _phase15(selection))
    print(f"[done] wrote sensitivity audits to {OUT}", flush=True)


if __name__ == "__main__":
    main()
