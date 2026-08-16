#!/usr/bin/env python3
"""Run S2 of the frozen comprehensive Local/Online protocol.

Every online score is rebuilt from explicitly truncated telemetry.  Candidate
fits and feature orientation are label-blind; correctness is used only for the
declared metrics, selection guards, and warning-threshold calibration.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_global_local_online_architecture_v2 import (  # noqa: E402
    _cell_path,
    _safe_ap,
    _safe_auc,
    _zapply,
    _zfit,
    fit_registered_global,
    load_rows,
)
from scripts.run_local_online_comprehensive_stage1 import (  # noqa: E402
    OUT,
    PROTOCOL,
    PROTOCOL_SHA256,
    SEED,
    _sha256,
    _stage_partition,
)
from spectral_utils.local_online_comprehensive import (  # noqa: E402
    PreparedTrace,
    fit_references,
    fit_trajectory_head_prepared,
    online_candidate_roster,
    prepare_trace,
)
from spectral_utils.multitask_trajectory import truncate_row  # noqa: E402
from spectral_utils.online_convergence import (  # noqa: E402
    causal_raw_prefix_matrix,
    fit_frozen_prefix_iu,
)
from spectral_utils.streaming_utils import deepconf_lowest_group_conf  # noqa: E402


BUDGETS = (16, 32, 64, 128, 256, 512)
CELLS = (("qwen3_4b", "gsm8k"), ("qwen3_4b", "math"))
BOOTSTRAP = 2000
DIRECT_METHODS = (
    "mean_entropy",
    "max_entropy",
    "deepconf_w32",
    "deepconf_w64",
    "iu28_registered",
    "step272_twohead",
)
GUARD_METHODS = (
    "deepconf_w32", "deepconf_w64", "iu28_registered", "step272_twohead",
)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _candidate_scores(
    prepared: PreparedTrace,
    heads: Mapping[str, Any],
    roster: Mapping[str, tuple[str, tuple[str, ...]]],
) -> dict[str, float]:
    return {
        name: float(np.max(head.curve_from_level(
            prepared.representations[roster[name][0]]
        )))
        for name, head in heads.items()
    }


def _direct_scores(
    row: Mapping[str, Any],
    budget: int | None,
    prepared: PreparedTrace,
    *,
    iu28: Any,
    global_model: Any,
    raw9_head: Any,
    global_fit: tuple[float, float],
    local_fit: tuple[float, float],
) -> dict[str, float]:
    entropy = np.asarray(row["token_entropies"], dtype=float)
    raw28, _ = causal_raw_prefix_matrix(
        row, budget, include_elapsed_length=False
    )
    local_max = float(np.max(raw9_head.curve_from_level(
        prepared.representations["raw9"]
    )))
    global_score = float(global_model.score(row, budget))
    return {
        "mean_entropy": float(np.mean(entropy)),
        "max_entropy": float(np.max(entropy)),
        "deepconf_w32": float(-deepconf_lowest_group_conf(entropy, 32)),
        "deepconf_w64": float(-deepconf_lowest_group_conf(entropy, 64)),
        "iu28_registered": float(np.max(iu28.risk(raw28))),
        "step272_twohead": float(
            0.50 * _zapply([global_score], global_fit)[0]
            + 0.50 * _zapply([local_max], local_fit)[0]
        ),
    }


def _score_split(
    family: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    references: Any,
    heads: Mapping[str, Any],
    roster: Mapping[str, tuple[str, tuple[str, ...]]],
    *,
    iu28: Any,
    global_model: Any,
    raw9_head: Any,
    global_fit: tuple[float, float],
    local_fit: tuple[float, float],
) -> list[dict[str, Any]]:
    output = []
    for row_index, row in enumerate(rows, 1):
        target = int(not bool(row["final_answer_correct"]))
        length = len(row["token_entropies"])
        for budget in (*BUDGETS, None):
            if budget is not None and length <= budget:
                continue
            prefix = truncate_row(row, length if budget is None else budget)
            prepared = prepare_trace(prefix, references)
            scores = _candidate_scores(prepared, heads, roster)
            scores.update(_direct_scores(
                prefix, budget, prepared,
                iu28=iu28,
                global_model=global_model,
                raw9_head=raw9_head,
                global_fit=global_fit,
                local_fit=local_fit,
            ))
            output.extend({
                "family": family,
                "split": split,
                "unit": row["_unit"],
                "candidate": candidate,
                "budget": "final" if budget is None else int(budget),
                "is_final": budget is None,
                "target": target,
                "score": float(score),
                "trace_length": int(length),
                "access_tier": "A",
                "fidelity": (
                    "direct_same_trace_baseline"
                    if candidate in DIRECT_METHODS
                    else "frozen_retrospective_candidate"
                ),
            } for candidate, score in scores.items())
        if row_index % 100 == 0:
            print(f"    {split}: scored {row_index}/{len(rows)}", flush=True)
    return output


def _metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    development = [row for row in records if row["split"] == "development"]
    methods = sorted({row["candidate"] for row in development})
    families = sorted({row["family"] for row in development})
    finals = {
        (row["family"], row["candidate"], row["unit"]): row
        for row in development if row["is_final"]
    }
    for family in families:
        for candidate in methods:
            for budget in BUDGETS:
                rows = [
                    row for row in development
                    if row["family"] == family
                    and row["candidate"] == candidate
                    and not row["is_final"]
                    and int(row["budget"]) == budget
                ]
                if not rows:
                    continue
                labels = np.asarray([row["target"] for row in rows], dtype=int)
                scores = np.asarray([row["score"] for row in rows], dtype=float)
                final_scores = np.asarray([
                    finals[(family, candidate, row["unit"])]["score"]
                    for row in rows
                ], dtype=float)
                lengths = np.asarray([row["trace_length"] for row in rows], dtype=float)
                prefix_final = (
                    float(spearmanr(scores, final_scores).statistic)
                    if len(rows) >= 3 and np.std(scores) > 1e-12
                    and np.std(final_scores) > 1e-12 else float("nan")
                )
                length_corr = (
                    float(spearmanr(scores, lengths).statistic)
                    if len(rows) >= 3 and np.std(scores) > 1e-12
                    and np.std(lengths) > 1e-12 else float("nan")
                )
                output.append({
                    "family": family,
                    "candidate": candidate,
                    "budget": budget,
                    "auroc": _safe_auc(labels, scores),
                    "auprc": _safe_ap(labels, scores),
                    "spearman_prefix_final": prefix_final,
                    "spearman_score_length": length_corr,
                    "n": len(rows),
                    "n_error": int(labels.sum()),
                    "access_tier": "A",
                })
    return output


def _aggregate(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    methods = sorted({row["candidate"] for row in metrics})
    for candidate in methods:
        rows = [row for row in metrics if row["candidate"] == candidate]
        by_family_budget = {
            (row["family"], int(row["budget"])): float(row["auroc"])
            for row in rows
        }
        primary_values = [
            by_family_budget[(family, budget)]
            for family in sorted({row["family"] for row in rows})
            for budget in (64, 128)
            if np.isfinite(by_family_budget.get((family, budget), np.nan))
        ]
        budget_mean = {
            budget: float(np.nanmean([
                float(row["auroc"]) for row in rows if int(row["budget"]) == budget
            ])) for budget in BUDGETS
        }
        output.append({
            "candidate": candidate,
            "primary": float(np.mean(primary_values)),
            "families": len({row["family"] for row in rows}),
            "auroc_16": budget_mean[16],
            "auroc_32": budget_mean[32],
            "auroc_64": budget_mean[64],
            "auroc_128": budget_mean[128],
            "auroc_256": budget_mean[256],
            "auroc_512": budget_mean[512],
            "slope_32_64": budget_mean[64] - budget_mean[32],
            "slope_64_128": budget_mean[128] - budget_mean[64],
            "access_tier": "A",
            "fidelity": (
                "direct_same_trace_baseline"
                if candidate in DIRECT_METHODS
                else "frozen_retrospective_candidate"
            ),
        })
    return output


def _paired_interval(
    records: Sequence[Mapping[str, Any]], candidate: str, reference: str
) -> tuple[float, float, float, int, int]:
    relevant = [
        row for row in records
        if row["split"] == "development"
        and not row["is_final"]
        and int(row["budget"]) in (64, 128)
        and row["candidate"] in {candidate, reference}
    ]
    def metric(
        method: str,
        sampled: Sequence[str],
        lookup: Mapping[str, Mapping[tuple[str, int], Mapping[str, Any]]],
    ) -> float:
        values = []
        for budget in (64, 128):
            selected = [
                lookup[method][(unit, budget)] for unit in sampled
                if (unit, budget) in lookup[method]
            ]
            auc = _safe_auc(
                [row["target"] for row in selected],
                [row["score"] for row in selected],
            )
            if np.isfinite(auc):
                values.append(auc)
        return float(np.mean(values)) if values else float("nan")

    prepared, points = [], []
    for family in sorted({row["family"] for row in relevant}):
        rows = [row for row in relevant if row["family"] == family]
        lookup = {
            method: {
                (row["unit"], int(row["budget"])): row
                for row in rows if row["candidate"] == method
            } for method in (candidate, reference)
        }
        units = sorted({row["unit"] for row in rows})

        prepared.append((units, lookup))
        points.append(
            metric(candidate, units, lookup) - metric(reference, units, lookup)
        )
    rng = np.random.default_rng(
        SEED + sum(ord(char) for char in candidate + reference + "online")
    )
    draws = []
    for _ in range(BOOTSTRAP):
        deltas = []
        for units, lookup in prepared:
            sampled = [units[index] for index in rng.integers(0, len(units), len(units))]
            left = metric(candidate, sampled, lookup)
            right = metric(reference, sampled, lookup)
            if np.isfinite(left) and np.isfinite(right):
                deltas.append(left - right)
        if deltas:
            draws.append(float(np.mean(deltas)))
    low, high = np.quantile(draws, (0.025, 0.975))
    return (
        float(np.mean(points)), float(low), float(high),
        int(sum(value > 0 for value in points)),
        int(sum(value < 0 for value in points)),
    )


def _warning_threshold(
    rows: Sequence[Mapping[str, Any]], candidate: str, alpha: float
) -> tuple[float, float, int]:
    selected = [
        row for row in rows
        if row["split"] == "calibration"
        and row["candidate"] == candidate
        and not row["is_final"] and int(row["target"]) == 0
    ]
    maxima: dict[str, float] = {}
    for row in selected:
        maxima[row["unit"]] = max(maxima.get(row["unit"], -np.inf), float(row["score"]))
    values = np.asarray(list(maxima.values()), dtype=float)
    candidates = np.r_[np.unique(values), np.nextafter(np.max(values), np.inf)]
    for threshold in candidates:
        rate = float(np.mean(values >= threshold))
        if rate <= alpha + 1e-12:
            return float(threshold), rate, len(values)
    raise RuntimeError("warning threshold calibration failed")


def _warning_metrics(
    records: Sequence[Mapping[str, Any]], methods: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries, declarations = [], []
    for family in sorted({row["family"] for row in records}):
        family_rows = [row for row in records if row["family"] == family]
        for method in methods:
            for alpha in (0.05, 0.10):
                threshold, calibration_fpr, n_cal_clean = _warning_threshold(
                    family_rows, method, alpha
                )
                dev = [
                    row for row in family_rows
                    if row["split"] == "development"
                    and row["candidate"] == method and not row["is_final"]
                ]
                by_unit: dict[str, list[Mapping[str, Any]]] = {}
                for row in dev:
                    by_unit.setdefault(str(row["unit"]), []).append(row)
                for unit_rows in by_unit.values():
                    unit_rows.sort(key=lambda row: int(row["budget"]))
                    first = next(
                        (row for row in unit_rows if float(row["score"]) >= threshold),
                        None,
                    )
                    base = unit_rows[0]
                    declarations.append({
                        "family": family,
                        "candidate": method,
                        "target_false_warning": alpha,
                        "unit": base["unit"],
                        "target": int(base["target"]),
                        "warned": first is not None,
                        "first_warning_budget": (
                            int(first["budget"]) if first is not None else ""
                        ),
                        "potential_tokens_remaining": (
                            int(base["trace_length"] - int(first["budget"]))
                            if first is not None else 0
                        ),
                        "threshold": threshold,
                    })
                subset = [
                    row for row in declarations
                    if row["family"] == family and row["candidate"] == method
                    and float(row["target_false_warning"]) == alpha
                ]
                clean = [row for row in subset if int(row["target"]) == 0]
                error = [row for row in subset if int(row["target"]) == 1]
                warned = [row for row in subset if bool(row["warned"])]
                summaries.append({
                    "family": family,
                    "candidate": method,
                    "target_false_warning": alpha,
                    "threshold": threshold,
                    "calibration_false_warning": calibration_fpr,
                    "n_calibration_clean": n_cal_clean,
                    "development_false_warning": float(np.mean([
                        row["warned"] for row in clean
                    ])) if clean else float("nan"),
                    "development_error_coverage": float(np.mean([
                        row["warned"] for row in error
                    ])) if error else float("nan"),
                    "development_warning_precision": float(np.mean([
                        int(row["target"]) for row in warned
                    ])) if warned else float("nan"),
                    "development_overall_coverage": len(warned) / max(len(subset), 1),
                    "mean_first_warning_budget": float(np.mean([
                        int(row["first_warning_budget"]) for row in warned
                    ])) if warned else float("nan"),
                    "mean_potential_tokens_remaining": float(np.mean([
                        int(row["potential_tokens_remaining"]) for row in warned
                    ])) if warned else 0.0,
                    "n": len(subset),
                })
    return summaries, declarations


def main() -> None:
    if _sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("frozen protocol hash mismatch")
    stage1 = json.loads((OUT / "STAGE_1_LOCAL_SELECTION.json").read_text())
    if stage1["selected"]["candidate"] != "l_family6__level__step_top5mean":
        raise RuntimeError("S1 identity differs from the frozen completed selection")

    roster = online_candidate_roster()
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    started_all = time.perf_counter()
    for model_name, family in CELLS:
        rows = load_rows(_cell_path(model_name, family))
        for row in rows:
            row["_stage"] = _stage_partition(family, row["_unit"])
        calibration = [row for row in rows if row["_stage"] == "calibration"]
        development = [row for row in rows if row["_stage"] == "development"]
        print(
            f"S2 {model_name}/{family}: calibration={len(calibration)} "
            f"development={len(development)}", flush=True,
        )
        started = time.perf_counter()
        references = fit_references(calibration)
        prepared_cal = [prepare_trace(row, references) for row in calibration]
        heads = {}
        head_diagnostics = {}
        for name, (representation, operators) in roster.items():
            head = fit_trajectory_head_prepared(
                prepared_cal,
                name=name,
                representation=representation,
                operators=operators,
            )
            heads[name] = head
            head_diagnostics[name] = dict(head.diagnostics)
            print(f"  fitted {name}", flush=True)
        raw9_head = fit_trajectory_head_prepared(
            prepared_cal,
            name="step272_raw9_level",
            representation="raw9",
            operators=("level",),
        )
        iu28 = fit_frozen_prefix_iu(calibration, include_elapsed_length=False)
        global_model = fit_registered_global(calibration)
        cal_global = [global_model.score(row, None) for row in calibration]
        cal_local = [
            float(np.max(raw9_head.curve_from_level(item.representations["raw9"])))
            for item in prepared_cal
        ]
        global_fit, local_fit = _zfit(cal_global), _zfit(cal_local)
        diagnostics[f"{model_name}/{family}"] = {
            "calibration": len(calibration),
            "development": len(development),
            "references": references.as_dict(),
            "heads": head_diagnostics,
            "iu28": iu28.diagnostics,
            "global": global_model.diagnostics,
            "global_fit": global_fit,
            "local_fit": local_fit,
            "fit_seconds": time.perf_counter() - started,
            "prefix_recomputed_from_truncated_telemetry": True,
        }
        records.extend(_score_split(
            family, "calibration", calibration, references, heads, roster,
            iu28=iu28, global_model=global_model, raw9_head=raw9_head,
            global_fit=global_fit, local_fit=local_fit,
        ))
        records.extend(_score_split(
            family, "development", development, references, heads, roster,
            iu28=iu28, global_model=global_model, raw9_head=raw9_head,
            global_fit=global_fit, local_fit=local_fit,
        ))

    metrics = _metrics(records)
    aggregate = _aggregate(metrics)
    direct = [row for row in aggregate if row["candidate"] in DIRECT_METHODS]
    reference = max(direct, key=lambda row: row["primary"])["candidate"]
    candidates = [row for row in aggregate if row["candidate"].startswith("o_")]
    family_primary = {
        (candidate, family): float(np.mean([
            float(row["auroc"]) for row in metrics
            if row["candidate"] == candidate and row["family"] == family
            and int(row["budget"]) in (64, 128)
        ]))
        for candidate in {row["candidate"] for row in aggregate}
        for family in {row["family"] for row in metrics}
    }
    family_guard = {
        family: max(family_primary[(method, family)] for method in GUARD_METHODS)
        for family in {row["family"] for row in metrics}
    }
    promotable = [
        row for row in candidates
        if all(
            family_primary[(row["candidate"], family)] >= bar - 0.015
            for family, bar in family_guard.items()
        )
    ]
    rejected = sorted(row["candidate"] for row in candidates if row not in promotable)
    numerical_best_any = max(candidates, key=lambda row: row["primary"])
    numerical_best = max(promotable, key=lambda row: row["primary"]) if promotable else None

    intervals = []
    for row in aggregate:
        if row["candidate"] == reference:
            continue
        delta, low, high, wins, losses = _paired_interval(
            records, row["candidate"], reference
        )
        intervals.append({
            "candidate": row["candidate"], "reference": reference,
            "delta": delta, "ci_low": low, "ci_high": high,
            "family_wins": wins, "family_losses": losses,
        })

    best_intervals = {}
    if numerical_best is not None:
        for row in promotable:
            if row["candidate"] == numerical_best["candidate"]:
                best_intervals[row["candidate"]] = (0.0, 0.0, 0.0)
            else:
                delta, low, high, _, _ = _paired_interval(
                    records, row["candidate"], numerical_best["candidate"]
                )
                best_intervals[row["candidate"]] = (delta, low, high)
    eligible = [
        row for row in promotable
        if row["primary"] >= numerical_best["primary"] - 0.005
        and best_intervals[row["candidate"]][1] <= 0 <= best_intervals[row["candidate"]][2]
    ] if numerical_best is not None else []

    widths = {"raw9": 9, "broad28": 28, "family6": 6}
    def cost(row: Mapping[str, Any]) -> tuple[int, str]:
        representation, operators = roster[row["candidate"]]
        return widths[representation] * len(operators), row["candidate"]

    selected = (
        min(eligible, key=cost) if eligible else numerical_best
    ) if numerical_best is not None else next(
        row for row in aggregate if row["candidate"] == reference
    )
    selected_interval = next(
        (row for row in intervals if row["candidate"] == selected["candidate"]), None
    )
    reference_value = next(
        row["primary"] for row in aggregate if row["candidate"] == reference
    )
    if selected["candidate"].startswith("o_") and selected_interval and selected_interval["ci_low"] > 0:
        verdict = "IMPROVES_DIRECT_COMPETITOR"
    elif selected["candidate"].startswith("o_") and selected["primary"] >= reference_value - 0.005:
        verdict = "PARITY_WITH_DIRECT_COMPETITOR"
    else:
        verdict = "REGRESSES_DIRECT_COMPETITOR"

    declaration_methods = list(DIRECT_METHODS) + [
        selected["candidate"], numerical_best_any["candidate"]
    ]
    declaration_methods = list(dict.fromkeys(declaration_methods))
    warning_metrics, declarations = _warning_metrics(records, declaration_methods)

    _write_csv(OUT / "STAGE_2_ONLINE_PER_QUESTION.csv", records)
    _write_csv(OUT / "STAGE_2_ONLINE_CELL_METRICS.csv", metrics)
    _write_csv(OUT / "STAGE_2_ONLINE_AGGREGATE.csv", aggregate)
    _write_csv(OUT / "STAGE_2_ONLINE_INTERVALS.csv", intervals)
    _write_csv(OUT / "STAGE_2_ONLINE_WARNING_METRICS.csv", warning_metrics)
    _write_csv(OUT / "STAGE_2_ONLINE_WARNINGS.csv", declarations)
    _write_json(OUT / "STAGE_2_ONLINE_DIAGNOSTICS.json", diagnostics)
    selection = {
        "verdict": verdict,
        "selected": selected,
        "direct_reference": reference,
        "numerical_best": numerical_best,
        "numerical_best_before_family_guard": numerical_best_any,
        "rejected_by_direct_family_margin": rejected,
        "family_direct_guards": family_guard,
        "rule": "first require no family worse than the best declared direct bar by >0.015; then simplest within 0.005 of the promotable numerical best with paired interval including zero",
        "protocol_sha256": PROTOCOL_SHA256,
        "score_sha256": hashlib.sha256(
            (OUT / "STAGE_2_ONLINE_PER_QUESTION.csv").read_bytes()
        ).hexdigest(),
    }
    _write_json(OUT / "STAGE_2_ONLINE_SELECTION.json", selection)

    interval_lookup = {row["candidate"]: row for row in intervals}
    lines = [
        "# S2 causal Online feature screen",
        "",
        f"**Verdict: `{verdict}`.**",
        "",
        "Every prefix score was rebuilt from truncated telemetry. No full-trace broad curve was sliced.",
        f"Direct Tier-A reference on the same development rows: `{reference}`.",
        f"Frozen S2 selection: `{selected['candidate']}`.",
        "",
        "| method | AUROC@64/128 | delta vs direct bar | 95% CI |",
        "|---|---:|---:|---|",
    ]
    for row in sorted(aggregate, key=lambda item: item["primary"], reverse=True):
        interval = interval_lookup.get(row["candidate"])
        delta = "—" if interval is None else f"{interval['delta']:+.4f}"
        ci = "—" if interval is None else f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}]"
        lines.append(f"| {row['candidate']} | {row['primary']:.4f} | {delta} | {ci} |")
    lines.extend([
        "",
        "Warning thresholds are calibrated separately per family on clean calibration traces. Warnings are one-sided and non-withdrawable; `potential_tokens_remaining` is not a realized-savings claim.",
    ])
    (OUT / "STAGE_2_ONLINE.md").write_text("\n".join(lines) + "\n")

    manifest_path = OUT / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "status": "STAGE_2_COMPLETE_STAGE_3_PENDING",
        "stage2_selection_sha256": _sha256(OUT / "STAGE_2_ONLINE_SELECTION.json"),
        "elapsed_stage2_seconds": time.perf_counter() - started_all,
        "new_inference": False,
        "gpu_hours": 0,
        "drive_mutation": False,
    })
    _write_json(manifest_path, manifest)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
