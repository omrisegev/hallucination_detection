#!/usr/bin/env python3
"""CPU runner for the Unified Causal IU-PCR development protocol.

The runner never launches inference or mutates cluster state.  It consumes existing
ProcessBench token telemetry, keeps scorer copies grouped by source question, performs
fold-local supervised development, and writes audit-friendly JSON/CSV/Markdown artifacts.

Examples:
    # Verify the existing caches without scoring outcomes.
    python scripts/run_unified_causal_iu_v1.py preflight \
      --data-root /Users/osegev/Desktop/hallucination_detection

    # Build the full information atlas (publication setting: 200 permutations).
    python scripts/run_unified_causal_iu_v1.py atlas --permutations 200

    # Nested grouped development/evaluation.  Expensive; resumable by outer-fold files.
    python scripts/run_unified_causal_iu_v1.py full --outer-folds 5 --inner-folds 3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import types
from typing import Any, Mapping, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/experiments/UNIFIED_CAUSAL_IU_V1.md"
sys.path.insert(0, str(ROOT))
# This is a CPU-only runner.  spectral_utils.__init__ eagerly imports PyTorch, although
# setup.py correctly places torch in an optional extra.  Load submodules as a namespace so
# a CPU analysis host needs only numpy/scipy/scikit-learn.
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [str(ROOT / "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.online_convergence import (  # noqa: E402
    causal_raw_prefix_matrix,
    fit_frozen_prefix_iu,
    normalize_cache_records,
)
from spectral_utils.historical_multitask_baselines import (  # noqa: E402
    fit_historical_cell_models as _fit_historical_cell_models,
    historical_global_score as _historical_global_score,
    historical_local_curve as _historical_local_curve,
    historical_online_curve as _historical_online_curve,
)
from spectral_utils.streaming_utils import causal_trajectories  # noqa: E402
from spectral_utils.unified_causal_evaluation import (  # noqa: E402
    DEFAULT_BUDGETS,
    PRIMARY_EARLY_BUDGETS,
    assert_group_split_isolation,
    best_localization_threshold,
    build_atlas_samples,
    build_group_synergy_atlas,
    build_information_atlas,
    derive_supervised_signs,
    final_wrong,
    finalist_gate,
    heldout_logloss_gain,
    processbench_metrics,
    safe_ap,
    safe_auc,
    select_atlas_roster,
    source_group,
    token_to_step,
)
from spectral_utils.unified_causal_iu import (  # noqa: E402
    ACCUMULATOR_ROSTER,
    AccumulatorSpec,
    AccumulatorState,
    UnifiedCausalIU,
    all_feature_names,
    base_matrix,
    causal_feature_matrix,
    fit_base_reference,
)


SEED = 20260817
RUN_SCHEMA = 4
DUFS_PRIMARY_LAMBDA = 0.1
DUFS_DEFAULT_LAMBDAS = (0.1, 0.3, 1.0)
DUFS_GRAPH_K = 7
DUFS_SEEDS = (11, 23, 37)
DUFS_EPOCHS = 80
FAMILIES = ("gsm8k", "math", "olympiadbench", "omnimath")
PRIMARY_MODELS = ("qwen3_4b", "qwen3_8b")
ROBUSTNESS_MODELS = ("llama31_8b",)
MODEL_PATHS = {
    "qwen3_4b": "cache/localization/processbench/pb_qwen3_4b",
    "qwen3_8b": "cache/localization/processbench/pb_qwen3_8b",
    "llama31_8b": "dataset_cache/repgrid/pb_llama31_8b",
}
REQUIRED_FIELDS = {
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
    "step_token_spans",
    "label",
    "final_answer_correct",
}
BASELINE_REGISTRY = {
    "previous_one_shared": "live_fold_refit_of_registered_v2_heads",
    "previous_two_head": "live_fold_refit_of_registered_v2_heads",
    "iu28_without_length": "live_fit",
    "mean_entropy": "live_curve",
    "max_entropy_top5": "live_curve",
    "sw_var": "live_curve",
    "deepconf_proxy_w64": "live_curve",
    "frozen_global_iu_pcr": "live_fold_refit_of_registered_length_free_global_head",
}
HISTORICAL_BASELINE_PROVENANCE = {
    "previous_one_shared": "results/global_local_online_architecture_v2/ARCHITECTURE_PER_QUESTION.csv",
    "previous_two_head": "results/global_local_online_architecture_v2/ARCHITECTURE_PER_QUESTION.csv",
    "frozen_global_iu_pcr": "results/global_local_online_architecture_v2/HEAD_PER_QUESTION.csv",
}
ROBUSTNESS_PANEL = (
    ("math500_qwen7b_t1", "phase15", None, None, 200),
    ("processbench_gsm8k__llama31_8b", "gsm8k", "generator", "Llama-3.1-8B-Instruct", 61),
    ("processbench_gsm8k__llama3_70b", "gsm8k", "generator", "Meta-Llama-3-70B-Instruct", 31),
    ("processbench_gsm8k__qwen2_7b", "gsm8k", "generator", "Qwen2-7B-Instruct", 52),
    ("processbench_gsm8k__qwen25_15b", "gsm8k", "generator", "Qwen2.5-1.5B-Instruct", 42),
    ("processbench_math__llama31_8b", "math", "generator", "Llama-3.1-8B-Instruct", 139),
    ("processbench_math__qwen25math_7b", "math", "generator", "Qwen2.5-Math-7B-Instruct", 96),
    ("processbench_olympiadbench__llama31_8b", "olympiadbench", "generator", "Llama-3.1-8B-Instruct", 164),
    ("processbench_olympiadbench__qwen25math_7b", "olympiadbench", "generator", "Qwen2.5-Math-7B-Instruct", 147),
    ("processbench_omnimath__llama31_8b", "omnimath", "generator", "Llama-3.1-8B-Instruct", 162),
    ("processbench_omnimath__qwen25math_7b", "omnimath", "generator", "Qwen2.5-Math-7B-Instruct", 132),
)


def _finite_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else float("nan")


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(_jsonable(value), sort_keys=True)
                if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def _write_trajectories_npz(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    evidence: Sequence[np.ndarray],
    trajectories: Sequence[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Store complete ragged candidate trajectories in one compressed audit artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(values) for values in evidence], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    np.savez_compressed(
        path,
        unit=np.asarray([str(row["_unit"]) for row in rows]),
        source_group=np.asarray([str(row["_source_group"]) for row in rows]),
        model=np.asarray([str(row["model"]) for row in rows]),
        family=np.asarray([str(row["family"]) for row in rows]),
        offsets=offsets,
        evidence=np.concatenate([np.asarray(values, dtype=np.float64) for values in evidence]),
        risk=np.concatenate([
            np.asarray(values[0], dtype=np.float64) for values in trajectories
        ]),
        positive_contribution=np.concatenate([
            np.asarray(values[1], dtype=np.float64) for values in trajectories
        ]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cell_path(data_root: Path, model: str, family: str) -> Path:
    return data_root / MODEL_PATHS[model] / f"processbench_{family}.pkl"


def load_cell(path: Path, *, model: str, family: str) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    output = []
    for index, key in enumerate(sorted(cache, key=str)):
        row = cache[key]
        if row.get("align_diag", {}).get("problems"):
            continue
        copy = dict(row)
        copy["_unit"] = str(row.get("id", key))
        copy["family"] = family
        copy["model"] = model
        copy["_source_group"] = source_group(copy, index)
        output.append(copy)
    return output


def preflight(
    data_root: Path,
    models: Sequence[str],
    families: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory, rows = [], []
    group_labels: dict[str, tuple[int, int]] = {}
    for model in models:
        for family in families:
            path = _cell_path(data_root, model, family)
            if not path.exists():
                raise FileNotFoundError(path)
            current = load_cell(path, model=model, family=family)
            missing = sorted({field for row in current for field in REQUIRED_FIELDS if row.get(field) is None})
            if missing:
                raise RuntimeError(f"{model}/{family}: missing required fields {missing}")
            for row in current:
                key = row["_source_group"]
                label = (final_wrong(row), int(row["label"]))
                if key in group_labels and group_labels[key] != label:
                    raise RuntimeError(f"scorer copies disagree on labels for {key}")
                group_labels[key] = label
            inventory.append({
                "model": model,
                "family": family,
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime": path.stat().st_mtime,
                "sha256": _sha256(path),
                "rows": len(current),
                "wrong": sum(final_wrong(row) for row in current),
                "first_error": sum(int(row["label"]) >= 0 for row in current),
                "unique_source_questions": len({row["_source_group"] for row in current}),
            })
            rows.extend(current)
    expected_copies = len(models)
    copy_counts = {
        group: sum(row["_source_group"] == group for row in rows)
        for group in group_labels
    }
    if set(copy_counts.values()) != {expected_copies}:
        raise RuntimeError("source-question scorer copies are incomplete")
    return inventory, rows


def _candidate_names(limit: int | None) -> tuple[str, ...] | None:
    if limit is None:
        return None
    names = all_feature_names()
    if int(limit) < 3:
        raise ValueError("candidate limit must retain at least three coordinates")
    return tuple(names[: int(limit)])


def _feature_matrices(rows, reference, raw_matrices=None):
    return [
        causal_feature_matrix(
            row,
            reference,
            raw_base=None if raw_matrices is None else raw_matrices[index],
        ) for index, row in enumerate(rows)
    ]


def develop_roster(
    rows: Sequence[Mapping[str, Any]],
    *,
    permutations: int,
    atlas_folds: int,
    candidate_limit: int | None,
    include_synergy: bool,
    workers: int,
) -> dict[str, Any]:
    """Run the complete fold-local Information Atlas and freeze roster/signs."""

    raw_matrices = [base_matrix(row) for row in rows]
    reference = fit_base_reference(rows, raw_base_matrices=raw_matrices)
    matrices = _feature_matrices(rows, reference, raw_matrices)
    iu28_model = fit_frozen_prefix_iu(rows, include_elapsed_length=False)
    iu28_curves = []
    for row in rows:
        raw, _ = causal_raw_prefix_matrix(row, None, include_elapsed_length=False)
        iu28_curves.append(np.asarray(iu28_model.risk(raw), dtype=float))
    sample_sets = [
        build_atlas_samples(
            rows,
            reference,
            target=target,
            feature_matrices=matrices,
            iu28_curves=iu28_curves,
        ) for target in ("global", "early", "localization")
    ]
    candidates = _candidate_names(candidate_limit)
    atlas = build_information_atlas(
        sample_sets,
        permutation_repeats=permutations,
        n_splits=atlas_folds,
        candidate_names=candidates,
        n_jobs=workers,
    )
    synergy = build_group_synergy_atlas(
        sample_sets,
        atlas,
        permutation_repeats=permutations,
        n_splits=atlas_folds,
    ) if include_synergy and candidates is None else []
    atlas_for_selection = list(atlas)
    for group in synergy:
        if not group.get("family_pass"):
            continue
        for member in group["members"]:
            atlas_for_selection.append({
                "target": group["target"],
                "feature": member,
                "conditional_logloss_gain": group["conditional_logloss_gain"],
                "roster_pass": True,
                "selected_via_group": group["group"],
            })
    names = sample_sets[0].feature_names
    X_development = np.vstack([samples.X for samples in sample_sets])
    roster, redundancy = select_atlas_roster(atlas_for_selection, names, X_development)
    signs = derive_supervised_signs(sample_sets, roster)
    roster_indices = [names.index(name) for name in roster]
    ceiling = {}
    for samples in sample_sets:
        gain, folds = heldout_logloss_gain(
            samples, roster_indices, n_splits=atlas_folds, seed=SEED + 90
        )
        ceiling[samples.target] = {
            "heldout_logloss_gain": gain,
            "fold_gains": list(folds),
            "role": "supervised_nonlinear_diagnostic_ceiling",
        }
    return {
        "reference": reference,
        "matrices": matrices,
        "sample_sets": sample_sets,
        "atlas": atlas,
        "synergy": synergy,
        "roster": roster,
        "signs": signs,
        "redundancy": redundancy,
        "fallback_used": any(
            row.get("reason") == "top_three_identifiability_fallback"
            for row in redundancy
        ),
        "ceiling": ceiling,
    }


def _simulate(evidence: Sequence[float], spec: AccumulatorSpec) -> tuple[np.ndarray, np.ndarray]:
    state = AccumulatorState(spec)
    risk, contribution = [], []
    for value in evidence:
        current_risk, current_contribution = state.update(float(value))
        risk.append(current_risk)
        contribution.append(current_contribution)
    return np.asarray(risk), np.asarray(contribution)


def _evidence_curves(model: UnifiedCausalIU, rows: Sequence[Mapping[str, Any]]) -> list[np.ndarray]:
    identity = model.with_accumulator(AccumulatorSpec("identity"))
    return [
        np.asarray([step.evidence for step in identity.score_row(row).trajectory], dtype=float)
        for row in rows
    ]


def _fixed_state_bytes(model: UnifiedCausalIU) -> int:
    dimension = len(model.reference.names)
    scalar_arrays = 5 + 6  # EWMA bank plus area/count/CUSUM/mean/Page-Hinkley state.
    rolling = 64 * dimension
    bocpd = 2 * (2 * dimension * 129 + 129)
    runtime = (rolling + scalar_arrays * dimension + bocpd) * 8
    frozen = sum(np.asarray(value).nbytes for value in (
        model.reference.centres,
        model.reference.scales,
        model.feature_indices,
        model.feature_medians,
        model.feature_centres,
        model.feature_scales,
        model.feature_signs,
        model.weights,
    ))
    return int(runtime + frozen)


def _curves_for_spec(evidence_curves, spec):
    return [_simulate(curve, spec) for curve in evidence_curves]


def _warning_thresholds(rows, trajectories):
    clean_maxima = [
        float(np.max(risk))
        for row, (risk, _) in zip(rows, trajectories)
        if final_wrong(row) == 0
    ]
    if not clean_maxima:
        return float("inf"), float("inf")
    return (
        float(np.quantile(clean_maxima, 0.95, method="higher")),
        float(np.quantile(clean_maxima, 0.90, method="higher")),
    )


def _curve_metrics(
    rows: Sequence[Mapping[str, Any]],
    trajectories,
    *,
    localization_threshold: float | None,
    warning_thresholds: tuple[float, float],
    global_thresholds: tuple[float, float] | None = None,
    localization_tokens: Sequence[int] | None = None,
    localization_steps: Sequence[int] | None = None,
    terminal_scores: Sequence[float] | None = None,
    localization_scores: Sequence[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    wrong = np.asarray([final_wrong(row) for row in rows], dtype=int)
    terminal = np.asarray(
        [risk[-1] for risk, _ in trajectories]
        if terminal_scores is None else terminal_scores,
        dtype=float,
    )
    detector = np.asarray(
        terminal if localization_scores is None else localization_scores,
        dtype=float,
    )
    if len(terminal) != len(rows) or len(detector) != len(rows):
        raise ValueError("score overrides must contain one entry per row")
    if localization_tokens is None:
        loc_token = np.asarray(
            [int(np.argmax(contribution)) for _, contribution in trajectories], dtype=int
        )
    else:
        loc_token = np.asarray(localization_tokens, dtype=int)
    if localization_steps is None:
        loc_step = np.asarray(
            [token_to_step(token, row) for token, row in zip(loc_token, rows)], dtype=int
        )
    else:
        loc_step = np.asarray(localization_steps, dtype=int)
    if len(loc_token) != len(rows) or len(loc_step) != len(rows):
        raise ValueError("localization overrides must contain one entry per row")
    target_step = np.asarray([int(row["label"]) for row in rows], dtype=int)
    if localization_threshold is None:
        localization_threshold, _ = best_localization_threshold(detector, loc_step, target_step)
    prediction = np.where(detector > localization_threshold, loc_step, -1)
    clean_terminal = terminal[wrong == 0]
    if global_thresholds is None:
        global_thresholds = (
            float(np.quantile(clean_terminal, 0.95, method="higher")),
            float(np.quantile(clean_terminal, 0.90, method="higher")),
        ) if len(clean_terminal) else (float("inf"), float("inf"))
    global_operating = {}
    for label, threshold in zip(("5pct", "10pct"), global_thresholds):
        predicted = terminal > threshold
        global_operating[label] = {
            "threshold": float(threshold),
            "fpr": float(np.mean(predicted[wrong == 0])) if np.any(wrong == 0) else float("nan"),
            "tpr": float(np.mean(predicted[wrong == 1])) if np.any(wrong == 1) else float("nan"),
        }
    threshold_5, threshold_10 = warning_thresholds
    first_tokens, persistent_tokens = [], []
    for (risk, contribution), fallback in zip(trajectories, loc_token):
        first = next((index for index, value in enumerate(risk) if value > threshold_10), int(fallback))
        persistent = int(fallback)
        hits = np.asarray(risk) > threshold_10
        for index in range(2, len(hits)):
            if bool(hits[index - 2:index + 1].all()):
                persistent = index - 2
                break
        first_tokens.append(first)
        persistent_tokens.append(persistent)
    first_steps = np.asarray([
        token_to_step(token, row) for token, row in zip(first_tokens, rows)
    ])
    persistent_steps = np.asarray([
        token_to_step(token, row) for token, row in zip(persistent_tokens, rows)
    ])
    first_prediction = np.where(detector > localization_threshold, first_steps, -1)
    persistent_prediction = np.where(detector > localization_threshold, persistent_steps, -1)

    family_metrics = []
    for family in sorted({_row["family"] for _row in rows}):
        mask = np.asarray([row["family"] == family for row in rows])
        local = processbench_metrics(prediction[mask], target_step[mask])
        local_first = processbench_metrics(first_prediction[mask], target_step[mask])
        local_persistent = processbench_metrics(persistent_prediction[mask], target_step[mask])
        early, early_ap = {}, {}
        for budget in DEFAULT_BUDGETS:
            scores = np.asarray([
                risk[min(int(budget), len(risk)) - 1] for risk, _ in trajectories
            ])
            early[str(budget)] = safe_auc(wrong[mask], scores[mask])
            early_ap[str(budget)] = safe_ap(wrong[mask], scores[mask])
        family_metrics.append({
            "family": family,
            "global_auroc": safe_auc(wrong[mask], terminal[mask]),
            "global_auprc": safe_ap(wrong[mask], terminal[mask]),
            "localization_f1": local["f1"],
            "localization_exact": local["exact"],
            "localization_within_one": local["within_one"],
            "clean_abstention": local["clean_abstention"],
            "first_crossing_f1": local_first["f1"],
            "persistent_crossing_3_f1": local_persistent["f1"],
            "early": early,
            "early_auprc": early_ap,
        })
    macro = {
        "global": _finite_mean([row["global_auroc"] for row in family_metrics]),
        "localization": _finite_mean([row["localization_f1"] for row in family_metrics]),
        "early": _finite_mean([
            _finite_mean([row["early"][str(budget)] for budget in PRIMARY_EARLY_BUDGETS])
            for row in family_metrics
        ]),
    }
    localization_ablations = {
        "max_positive_contribution": macro["localization"],
        "first_crossing": _finite_mean([row["first_crossing_f1"] for row in family_metrics]),
        "persistent_crossing_3": _finite_mean([row["persistent_crossing_3_f1"] for row in family_metrics]),
    }
    clean = wrong == 0
    ever_5 = np.asarray([np.any(risk > threshold_5) for risk, _ in trajectories])
    ever_10 = np.asarray([np.any(risk > threshold_10) for risk, _ in trajectories])
    alarm_times_10 = [
        next((index for index, value in enumerate(risk) if value > threshold_10), None)
        for risk, _ in trajectories
    ]
    summary = {
        "macro": macro,
        "families": family_metrics,
        "localization_ablations": localization_ablations,
        "localization_threshold": float(localization_threshold),
        "warning_threshold_5pct": threshold_5,
        "warning_threshold_10pct": threshold_10,
        "global_fixed_fpr": global_operating,
        "global_thresholds": list(global_thresholds),
        "ever_warning_fpr_5pct": float(np.mean(ever_5[clean])) if clean.any() else float("nan"),
        "ever_warning_fpr_10pct": float(np.mean(ever_10[clean])) if clean.any() else float("nan"),
        "mean_alarm_token_10pct_on_wrong": float(np.mean([
            token for token, label in zip(alarm_times_10, wrong) if label and token is not None
        ])) if any(label and token is not None for token, label in zip(alarm_times_10, wrong)) else float("nan"),
    }
    records = []
    for row, (risk, contribution), score, local_score, token, step, pred in zip(
        rows, trajectories, terminal, detector, loc_token, loc_step, prediction
    ):
        records.append({
            "unit": row["_unit"],
            "source_group": row["_source_group"],
            "family": row["family"],
            "model": row["model"],
            "wrong": final_wrong(row),
            "target_step": int(row["label"]),
            "global_score": float(score),
            "localization_score": float(local_score),
            "localization_token": int(token),
            "localization_step": int(step),
            "prediction": int(pred),
            "first_alarm_5pct": next((i for i, value in enumerate(risk) if value > threshold_5), None),
            "first_alarm_10pct": next((i for i, value in enumerate(risk) if value > threshold_10), None),
            **{
                f"risk_at_{budget}": float(risk[min(budget, len(risk)) - 1])
                for budget in DEFAULT_BUDGETS
            },
        })
    return summary, records


def _step_top5_location(curve: Sequence[float], row: Mapping[str, Any]) -> tuple[int, int]:
    """Return the token/step selected by the registered top-five-per-step locator."""

    values = np.asarray(curve, dtype=float)
    scores: list[float] = []
    for span in row.get("step_token_spans") or ():
        if span is None:
            scores.append(float("nan"))
            continue
        start, stop = max(0, int(span[0])), min(len(values), int(span[1]))
        finite = values[start:stop]
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            scores.append(float("nan"))
            continue
        count = min(5, len(finite))
        scores.append(float(np.mean(np.partition(finite, -count)[-count:])))
    if not np.isfinite(scores).any():
        return -1, -1
    step = int(np.nanargmax(scores))
    span = row["step_token_spans"][step]
    start, stop = max(0, int(span[0])), min(len(values), int(span[1]))
    local = np.where(np.isfinite(values[start:stop]), values[start:stop], -np.inf)
    token = start + int(np.argmax(local)) if len(local) else -1
    return token, step


def _record_primary_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Compute family-macro primaries without pooling scores across outer folds."""

    family_values = []
    for family in sorted({str(record["family"]) for record in records}):
        subset = [record for record in records if str(record["family"]) == family]
        labels = np.asarray([record["wrong"] for record in subset], dtype=int)
        localization = processbench_metrics(
            [record["prediction"] for record in subset],
            [record["target_step"] for record in subset],
        )
        family_values.append({
            "global": safe_auc(labels, [record["global_score"] for record in subset]),
            "localization": float(localization["f1"]),
            "early": _finite_mean([
                safe_auc(labels, [record[f"risk_at_{budget}"] for record in subset])
                for budget in PRIMARY_EARLY_BUDGETS
            ]),
        })
    return {
        task: _finite_mean([value[task] for value in family_values])
        for task in ("global", "localization", "early")
    }


def _weighted_bootstrap_auc(
    labels: np.ndarray, scores: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Vectorized weighted AUROC for one row per bootstrap replicate."""

    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores, weights = labels[finite], scores[finite], weights[:, finite]
    order = np.argsort(scores, kind="mergesort")
    labels, scores, weights = labels[order], scores[order], weights[:, order]
    positive_total = weights @ (labels == 1)
    negative_total = weights @ (labels == 0)
    concordant = np.zeros(len(weights), dtype=float)
    negative_below = np.zeros(len(weights), dtype=float)
    for indexes in np.split(np.arange(len(scores)), np.flatnonzero(np.diff(scores)) + 1):
        block = weights[:, indexes]
        positive = block @ (labels[indexes] == 1)
        negative = block @ (labels[indexes] == 0)
        concordant += positive * (negative_below + 0.5 * negative)
        negative_below += negative
    denominator = positive_total * negative_total
    return np.divide(
        concordant,
        denominator,
        out=np.full(len(weights), np.nan, dtype=float),
        where=denominator > 0,
    )


def _weighted_bootstrap_localization(
    prediction: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    error, clean = target != -1, target == -1
    error_total, clean_total = weights @ error, weights @ clean
    exact = np.divide(
        weights @ (error & (prediction == target)),
        error_total,
        out=np.full(len(weights), np.nan),
        where=error_total > 0,
    )
    abstention = np.divide(
        weights @ (clean & (prediction == -1)),
        clean_total,
        out=np.full(len(weights), np.nan),
        where=clean_total > 0,
    )
    denominator = exact + abstention
    return np.divide(
        2.0 * exact * abstention,
        denominator,
        out=np.zeros(len(weights), dtype=float),
        where=np.isfinite(denominator) & (denominator > 0),
    )


def grouped_bootstrap_comparisons(
    records: Sequence[Mapping[str, Any]],
    *,
    repeats: int,
    seed: int = SEED,
    primary_candidate: str = "unified_causal_iu",
) -> dict[str, Any]:
    """Paired question-block bootstrap, averaged within fold before comparison.

    The two scorer-model copies of a source question are resampled as one block.  Family
    metrics are computed inside their outer fold.  The implementation uses integer block
    weights rather than copying rows, which keeps the publication 2,000-replicate audit
    practical on CPU.
    """

    methods = sorted({str(record["candidate"]) for record in records})
    if primary_candidate not in methods:
        raise ValueError(f"candidate records are missing {primary_candidate}")
    references = [method for method in methods if method != primary_candidate]
    folds = sorted({int(record["outer_fold"]) for record in records})
    by_fold_method = {
        (fold, method): [
            record for record in records
            if int(record["outer_fold"]) == fold and str(record["candidate"]) == method
        ]
        for fold in folds for method in methods
    }
    point_by_method = {
        method: {
            task: _finite_mean([
                _record_primary_metrics(by_fold_method[(fold, method)])[task]
                for fold in folds
            ])
            for task in ("global", "localization", "early")
        }
        for method in methods
    }

    count = max(0, int(repeats))
    rng = np.random.default_rng(seed)
    fold_values: dict[tuple[int, str, str], list[np.ndarray]] = {
        (fold, method, task): []
        for fold in folds for method in methods
        for task in ("global", "localization", "early")
    }
    for fold in folds:
        candidate_rows = by_fold_method[(fold, primary_candidate)]
        for family in sorted({str(record["family"]) for record in candidate_rows}):
            canonical = [record for record in candidate_rows if str(record["family"]) == family]
            groups = sorted({str(record["source_group"]) for record in canonical})
            draw_counts = rng.multinomial(
                len(groups), np.full(len(groups), 1.0 / len(groups)), size=count
            ) if count else np.empty((0, len(groups)), dtype=int)
            group_position = {group: index for index, group in enumerate(groups)}
            for method in methods:
                subset = [
                    record for record in by_fold_method[(fold, method)]
                    if str(record["family"]) == family
                ]
                method_groups = {str(record["source_group"]) for record in subset}
                if method_groups != set(groups):
                    raise AssertionError(
                        f"bootstrap pairing mismatch for fold={fold}, family={family}, method={method}"
                    )
                group_index = np.asarray([
                    group_position[str(record["source_group"])] for record in subset
                ], dtype=int)
                weights = draw_counts[:, group_index]
                labels = np.asarray([record["wrong"] for record in subset], dtype=int)
                global_values = _weighted_bootstrap_auc(
                    labels,
                    np.asarray([record["global_score"] for record in subset], dtype=float),
                    weights,
                )
                localization_values = _weighted_bootstrap_localization(
                    np.asarray([record["prediction"] for record in subset], dtype=int),
                    np.asarray([record["target_step"] for record in subset], dtype=int),
                    weights,
                )
                early_stack = np.vstack([
                    _weighted_bootstrap_auc(
                        labels,
                        np.asarray([record[f"risk_at_{budget}"] for record in subset], dtype=float),
                        weights,
                    )
                    for budget in PRIMARY_EARLY_BUDGETS
                ])
                finite = np.isfinite(early_stack)
                early_values = np.divide(
                    np.nansum(early_stack, axis=0),
                    np.sum(finite, axis=0),
                    out=np.full(count, np.nan),
                    where=np.sum(finite, axis=0) > 0,
                )
                fold_values[(fold, method, "global")].append(global_values)
                fold_values[(fold, method, "localization")].append(localization_values)
                fold_values[(fold, method, "early")].append(early_values)

    replicate_by_method: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        replicate_by_method[method] = {}
        for task in ("global", "localization", "early"):
            per_fold = []
            for fold in folds:
                values = np.vstack(fold_values[(fold, method, task)])
                valid = np.isfinite(values)
                per_fold.append(np.divide(
                    np.nansum(values, axis=0),
                    np.sum(valid, axis=0),
                    out=np.full(count, np.nan),
                    where=np.sum(valid, axis=0) > 0,
                ))
            values = np.vstack(per_fold)
            valid = np.isfinite(values)
            replicate_by_method[method][task] = np.divide(
                np.nansum(values, axis=0),
                np.sum(valid, axis=0),
                out=np.full(count, np.nan),
                where=np.sum(valid, axis=0) > 0,
            )

    comparisons = {}
    for reference in references:
        comparisons[reference] = {}
        for task in ("global", "localization", "early"):
            delta = point_by_method[primary_candidate][task] - point_by_method[reference][task]
            distribution = (
                replicate_by_method[primary_candidate][task]
                - replicate_by_method[reference][task]
            )
            finite = distribution[np.isfinite(distribution)]
            comparisons[reference][task] = {
                "delta": float(delta),
                "ci95": [
                    float(np.quantile(finite, 0.025)),
                    float(np.quantile(finite, 0.975)),
                ] if len(finite) else [float("nan"), float("nan")],
                "valid_replicates": int(len(finite)),
            }
    return {
        "repeats": count,
        "primary_candidate": primary_candidate,
        "unit": "dataset-qualified source question with scorer copies together",
        "point_by_method": point_by_method,
        "comparisons": comparisons,
    }


def _accumulator_metrics(
    fit_rows,
    validation_rows,
    model,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    fit_evidence = _evidence_curves(model, fit_rows)
    val_evidence = _evidence_curves(model, validation_rows)
    scores, details = {}, {}
    for spec in ACCUMULATOR_ROSTER:
        fit_trajectories = _curves_for_spec(fit_evidence, spec)
        val_trajectories = _curves_for_spec(val_evidence, spec)
        warning = _warning_thresholds(fit_rows, fit_trajectories)
        fit_summary, _ = _curve_metrics(
            fit_rows,
            fit_trajectories,
            localization_threshold=None,
            warning_thresholds=warning,
        )
        val_summary, _ = _curve_metrics(
            validation_rows,
            val_trajectories,
            localization_threshold=fit_summary["localization_threshold"],
            warning_thresholds=warning,
            global_thresholds=tuple(fit_summary["global_thresholds"]),
        )
        scores[spec.name] = dict(val_summary["macro"])
        details[spec.name] = {"spec": spec, "fit": fit_summary, "validation": val_summary}
    return scores, details


def _evaluate_fitted_model(
    train_rows,
    test_rows,
    model: UnifiedCausalIU,
    spec: AccumulatorSpec,
):
    """Evaluate one already-fitted fusion with train-only calibration."""

    train_evidence = _evidence_curves(model, train_rows)
    started = time.perf_counter()
    test_evidence = _evidence_curves(model, test_rows)
    score_seconds = time.perf_counter() - started
    train_trajectories = _curves_for_spec(train_evidence, spec)
    test_trajectories = _curves_for_spec(test_evidence, spec)
    warning = _warning_thresholds(train_rows, train_trajectories)
    train_summary, _ = _curve_metrics(
        train_rows,
        train_trajectories,
        localization_threshold=None,
        warning_thresholds=warning,
    )
    test_summary, records = _curve_metrics(
        test_rows,
        test_trajectories,
        localization_threshold=train_summary["localization_threshold"],
        warning_thresholds=warning,
        global_thresholds=tuple(train_summary["global_thresholds"]),
    )
    return {
        "metrics": test_summary,
        "records": records,
        "train_evidence": train_evidence,
        "test_evidence": test_evidence,
        "test_trajectories": test_trajectories,
        "warning_thresholds": warning,
        "score_seconds": score_seconds,
        "test_tokens": int(sum(len(curve) for curve in test_evidence)),
    }


def _select_accumulator(scores: Mapping[str, Mapping[str, float]]) -> tuple[str, list[dict[str, Any]]]:
    identity = scores["identity"]
    ledger = []
    for name, metric in scores.items():
        delta = {task: float(metric[task] - identity[task]) for task in identity}
        survives = (
            delta["global"] >= -0.010
            and delta["localization"] >= -0.010
            and delta["early"] >= -0.015
        )
        complexity = 0 if name == "identity" else (2 if name == "cumulative_hazard" else 1)
        ledger.append({
            "candidate": name,
            "survives": survives,
            "worst_delta": min(delta.values()),
            "complexity": complexity,
            **{f"delta_{task}": value for task, value in delta.items()},
            **{f"metric_{task}": float(metric[task]) for task in metric},
        })
    survivors = [row for row in ledger if row["survives"]]
    survivors.sort(key=lambda row: (-row["worst_delta"], row["complexity"], row["candidate"]))
    return str(survivors[0]["candidate"]), ledger


def _spec_by_name(name: str) -> AccumulatorSpec:
    return next(spec for spec in ACCUMULATOR_ROSTER if spec.name == name)


def _lambda_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _dufs_candidate(value: float) -> str:
    return f"unified_causal_dufs_l{_lambda_tag(value)}"


def _parse_dufs_lambdas(raw: str) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(item) for item in raw.split(",") if item.strip()))
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("--dufs-lambdas requires finite positive comma-separated values")
    if DUFS_PRIMARY_LAMBDA not in values:
        raise ValueError(f"--dufs-lambdas must include the frozen primary {DUFS_PRIMARY_LAMBDA}")
    return values


def _select_dufs_lambda(
    ordinary: Mapping[str, float],
    dufs_path: Mapping[float, Mapping[str, float]],
) -> tuple[float, list[dict[str, Any]]]:
    """Select lambda on inner folds only, with ordinary IU as the zero path.

    Maximin selection prevents a gain in one task from buying a regression in another.
    The preregistered regression margins remain visible in the ledger, but ordinary
    lambda=0 wins whenever every DUFS arm has a negative worst-task delta.
    """

    ledger = [{
        "candidate": "ordinary_iu",
        "lambda": 0.0,
        "survives_margins": True,
        "worst_delta": 0.0,
        "best_delta": 0.0,
        **{f"delta_{task}": 0.0 for task in ordinary},
        **{f"metric_{task}": float(value) for task, value in ordinary.items()},
    }]
    for lambda_, metrics in sorted(dufs_path.items()):
        delta = {task: float(metrics[task] - ordinary[task]) for task in ordinary}
        survives = (
            delta["global"] >= -0.010
            and delta["localization"] >= -0.010
            and delta["early"] >= -0.015
        )
        ledger.append({
            "candidate": _dufs_candidate(lambda_),
            "lambda": float(lambda_),
            "survives_margins": survives,
            "worst_delta": min(delta.values()),
            "best_delta": max(delta.values()),
            "promotion_sized_gain": max(delta.values()) >= 0.010,
            **{f"delta_{task}": value for task, value in delta.items()},
            **{f"metric_{task}": float(value) for task, value in metrics.items()},
        })
    eligible = [row for row in ledger if row["survives_margins"]]
    eligible.sort(key=lambda row: (-row["worst_delta"], row["lambda"]))
    return float(eligible[0]["lambda"]), ledger


def _grouped_splits(rows, n_splits, seed):
    labels = np.asarray([final_wrong(row) for row in rows], dtype=int)
    groups = np.asarray([row["_source_group"] for row in rows], dtype=str)
    count = min(int(n_splits), len(np.unique(groups)))
    splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(rows)), labels, groups))
    assert_group_split_isolation(splits, groups)
    return splits


def run_atlas(args, rows, out):
    developed = develop_roster(
        rows,
        permutations=args.permutations,
        atlas_folds=args.atlas_folds,
        candidate_limit=args.candidate_limit,
        include_synergy=not args.skip_synergy,
        workers=args.workers,
    )
    _write_jsonl(out / "INFORMATION_ATLAS.jsonl", developed["atlas"])
    _write_csv(out / "INFORMATION_ATLAS.csv", developed["atlas"])
    _write_jsonl(out / "GROUP_SYNERGY.jsonl", developed["synergy"])
    _write_json(out / "ROSTER.json", {
        "features": developed["roster"],
        "signs": developed["signs"],
        "redundancy": developed["redundancy"],
        "fallback_used": developed["fallback_used"],
        "supervised_nonlinear_ceiling": developed["ceiling"],
        "development_only": True,
        "candidate_limit": args.candidate_limit,
        "permutations": args.permutations,
    })
    return developed


def _load_robustness_cells(
    data_root: Path, primary_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load the frozen eleven-cell Early panel from its audited inventory."""

    inventory, output, hashes = [], [], {}
    for cell, family, filter_field, filter_value, expected_rows in ROBUSTNESS_PANEL:
        if family == "phase15":
            candidates = (
                data_root / "local_cache/math500_qwen7b_T1.0_run0.pkl",
                data_root / "cache/phase15_temperature/math500_qwen7b_T1.0_run0.pkl",
                Path("/private/tmp/hallucination_phase1_audit_20260816/math500_qwen7b_T1.0_run0.pkl"),
            )
            source = next((path for path in candidates if path.exists()), candidates[0])
        else:
            source = data_root / MODEL_PATHS["qwen3_4b"] / f"processbench_{family}.pkl"
        if not source.exists():
            raise FileNotFoundError(
                f"robustness source is missing for {cell}: {source}; stage the existing cache, do not infer"
            )
        if family == "phase15":
            with source.open("rb") as handle:
                cache = pickle.load(handle)
            records = normalize_cache_records(cache, min_tokens=1)
        else:
            # The qwen3_4b copies are already resident in the primary intersection;
            # reusing them avoids deserializing the same multi-generator cache repeatedly.
            records = [
                row for row in primary_rows
                if row["model"] == "qwen3_4b" and row["family"] == family
            ]
        filter_spec = (
            {"field": filter_field, "value": filter_value}
            if filter_field is not None else None
        )
        if filter_field is not None:
            records = [
                row for row in records
                if str(row.get(filter_field)) == str(filter_value)
            ]
        converted = []
        for index, row in enumerate(records):
            item = dict(row)
            identity = str(row.get("_trace_id", row.get("id", index)))
            item["_unit"] = f"{cell}::{identity}"
            item["_source_group"] = f"{cell}::{row.get('_group', identity)}"
            item["family"] = cell
            item["model"] = str(row.get("generator", cell))
            converted.append(item)
        if len(converted) != int(expected_rows):
            raise RuntimeError(
                f"robustness cell {cell} has {len(converted)} rows; frozen panel says {expected_rows}"
            )
        source_key = str(source)
        if source_key not in hashes:
            hashes[source_key] = _sha256(source)
        inventory.append({
            "cell": cell,
            "source": str(source.resolve()),
            "sha256": hashes[source_key],
            "rows": len(converted),
            "filter": filter_spec,
        })
        output.extend(converted)
    return inventory, output


def _final_accumulator_name(fold_summaries: Sequence[Mapping[str, Any]]) -> str:
    counts: dict[str, int] = {}
    for fold in fold_summaries:
        name = str(fold["selected_accumulator"])
        counts[name] = counts.get(name, 0) + 1
    best_count = max(counts.values())
    order = {spec.name: index for index, spec in enumerate(ACCUMULATOR_ROSTER)}
    return min(
        (name for name, count in counts.items() if count == best_count),
        key=lambda name: order[name],
    )


def _final_graph_lambda(fold_summaries: Sequence[Mapping[str, Any]]) -> float:
    values = [float(fold.get("selected_graph_lambda", 0.0)) for fold in fold_summaries]
    counts = {value: values.count(value) for value in set(values)}
    best = max(counts.values())
    return min(value for value, count in counts.items() if count == best)


def run_robustness_panel(args, primary_rows, fold_summaries, out):
    """Refit the chosen pipeline on all opened primary data; score 11 cells without tuning."""

    payload = getattr(args, "robustness_payload", None)
    inventory, robustness_rows = (
        payload if payload is not None
        else _load_robustness_cells(args.data_root, primary_rows)
    )
    developed = develop_roster(
        primary_rows,
        permutations=args.permutations,
        atlas_folds=args.atlas_folds,
        candidate_limit=args.candidate_limit,
        include_synergy=not args.skip_synergy,
        workers=args.workers,
    )
    accumulator_name = _final_accumulator_name(fold_summaries)
    ordinary_model = UnifiedCausalIU.fit(
        primary_rows,
        reference=developed["reference"],
        feature_matrices=developed["matrices"],
        feature_roster=developed["roster"],
        feature_signs=developed["signs"],
        accumulator=_spec_by_name(accumulator_name),
    )
    selected_lambda = _final_graph_lambda(fold_summaries)
    model = ordinary_model
    if selected_lambda > 0.0:
        model = UnifiedCausalIU.fit_dufs_path(
            primary_rows,
            lambdas=(selected_lambda,),
            reference=developed["reference"],
            feature_matrices=developed["matrices"],
            feature_roster=developed["roster"],
            feature_signs=developed["signs"],
            accumulator=_spec_by_name(accumulator_name),
            ordinary_model=ordinary_model,
            graph_k=DUFS_GRAPH_K,
            dufs_seeds=DUFS_SEEDS,
            dufs_epochs=args.dufs_epochs,
        )[selected_lambda]
    primary_evidence = _evidence_curves(model, primary_rows)
    primary_trajectories = _curves_for_spec(primary_evidence, model.accumulator)
    warning = _warning_thresholds(primary_rows, primary_trajectories)
    model = model.with_thresholds(*warning)

    evidence, trajectories, per_question = [], [], []
    by_cell = []
    started = time.perf_counter()
    for cell in sorted({row["family"] for row in robustness_rows}):
        cell_rows = [row for row in robustness_rows if row["family"] == cell]
        finals = [model.score_row(row) for row in cell_rows]
        cell_evidence = [
            np.asarray([step.evidence for step in final.trajectory], dtype=float)
            for final in finals
        ]
        cell_trajectories = [
            (
                np.asarray([step.risk for step in final.trajectory], dtype=float),
                np.asarray([step.contribution for step in final.trajectory], dtype=float),
            ) for final in finals
        ]
        labels = np.asarray([final_wrong(row) for row in cell_rows], dtype=int)
        global_scores = np.asarray([final.global_score for final in finals], dtype=float)
        early = {
            str(budget): safe_auc(labels, [
                final.trajectory[min(int(budget), len(final.trajectory)) - 1].risk
                for final in finals
            ]) for budget in DEFAULT_BUDGETS
        }
        by_cell.append({
            "cell": cell,
            "n": len(cell_rows),
            "error_rate": float(np.mean(labels)),
            "global_auroc": safe_auc(labels, global_scores),
            "global_auprc": safe_ap(labels, global_scores),
            "early": early,
            "early_primary": _finite_mean([
                early[str(budget)] for budget in PRIMARY_EARLY_BUDGETS
            ]),
            "ever_warning_fpr_5pct": float(np.mean([
                final.first_alarm_token_5pct is not None
                for final, label in zip(finals, labels) if label == 0
            ])) if np.any(labels == 0) else float("nan"),
            "ever_warning_fpr_10pct": float(np.mean([
                final.first_alarm_token_10pct is not None
                for final, label in zip(finals, labels) if label == 0
            ])) if np.any(labels == 0) else float("nan"),
        })
        for row, final in zip(cell_rows, finals):
            per_question.append({
                "unit": row["_unit"],
                "source_group": row["_source_group"],
                "cell": cell,
                "model": row["model"],
                "wrong": final_wrong(row),
                "global_score": float(final.global_score),
                "first_alarm_5pct": final.first_alarm_token_5pct,
                "first_alarm_10pct": final.first_alarm_token_10pct,
                **{
                    f"risk_at_{budget}": float(
                        final.trajectory[min(int(budget), len(final.trajectory)) - 1].risk
                    ) for budget in DEFAULT_BUDGETS
                },
            })
        evidence.extend(cell_evidence)
        trajectories.extend(cell_trajectories)
    elapsed = time.perf_counter() - started
    result = {
        "role": "opened-data Global/Early robustness only; never used for selection",
        "cells": by_cell,
        "macro": {
            "global": _finite_mean([row["global_auroc"] for row in by_cell]),
            "early": _finite_mean([row["early_primary"] for row in by_cell]),
        },
        "selected_accumulator": accumulator_name,
        "selected_graph_lambda": selected_lambda,
        "roster": list(developed["roster"]),
        "warning_thresholds": {"5pct": warning[0], "10pct": warning[1]},
        "efficiency": {
            "seconds": elapsed,
            "tokens": int(sum(len(values) for values in evidence)),
            "microseconds_per_token": 1e6 * elapsed / max(1, sum(len(values) for values in evidence)),
        },
        "model": model.as_dict(),
    }
    _write_json(out / "ROBUSTNESS_INVENTORY.json", inventory)
    _write_json(out / "ROBUSTNESS_11.json", result)
    _write_csv(out / "ROBUSTNESS_PER_CELL.csv", by_cell)
    _write_csv(out / "ROBUSTNESS_PER_QUESTION.csv", per_question)
    _write_trajectories_npz(
        out / "ROBUSTNESS_TRAJECTORIES.npz",
        robustness_rows,
        evidence,
        trajectories,
    )
    _write_json(out / "FINAL_MODEL.json", model.as_dict())
    return result


def run_full(args, rows, out):
    outer = _grouped_splits(rows, args.outer_folds, SEED)
    fold_summaries, per_question = [], []
    for outer_fold, (train_index, test_index) in enumerate(outer):
        fold_path = out / "folds" / f"outer_{outer_fold}.json"
        trajectory_path = out / "folds" / f"outer_{outer_fold}_trajectories.npz"
        if fold_path.exists() and not args.force:
            value = json.loads(fold_path.read_text())
            if int(value.get("run_schema", 0)) == RUN_SCHEMA and trajectory_path.exists():
                fold_summaries.append(value["summary"])
                per_question.extend(value["per_question"])
                print(f"[outer {outer_fold}] resumed", flush=True)
                continue
            print(f"[outer {outer_fold}] incompatible checkpoint; recomputing", flush=True)
        train_rows = [rows[index] for index in train_index]
        test_rows = [rows[index] for index in test_index]
        developed = develop_roster(
            train_rows,
            permutations=args.permutations,
            atlas_folds=args.atlas_folds,
            candidate_limit=args.candidate_limit,
            include_synergy=not args.skip_synergy,
            workers=args.workers,
        )
        inner_ledgers = []
        accumulator_values: dict[str, list[Mapping[str, float]]] = {
            spec.name: [] for spec in ACCUMULATOR_ROSTER
        }
        dufs_accumulator_values: dict[float, dict[str, list[Mapping[str, float]]]] = {
            lambda_: {spec.name: [] for spec in ACCUMULATOR_ROSTER}
            for lambda_ in args.dufs_lambdas
        }
        for inner_fold, (fit_index, val_index) in enumerate(
            _grouped_splits(train_rows, args.inner_folds, SEED + 100 + outer_fold)
        ):
            fit_rows = [train_rows[index] for index in fit_index]
            val_rows = [train_rows[index] for index in val_index]
            # Every label-aware choice, including roster and signs, is repeated inside the
            # inner fit partition.  The validation groups therefore influence only the
            # accumulator score used by nested selection.
            inner_developed = develop_roster(
                fit_rows,
                permutations=args.permutations,
                atlas_folds=args.atlas_folds,
                candidate_limit=args.candidate_limit,
                include_synergy=not args.skip_synergy,
                workers=args.workers,
            )
            model = UnifiedCausalIU.fit(
                fit_rows,
                reference=inner_developed["reference"],
                feature_matrices=inner_developed["matrices"],
                feature_roster=inner_developed["roster"],
                feature_signs=inner_developed["signs"],
            )
            scores, _ = _accumulator_metrics(fit_rows, val_rows, model)
            for name, value in scores.items():
                accumulator_values[name].append(value)
            inner_dufs_scores = {}
            if not args.skip_dufs:
                dufs_models = UnifiedCausalIU.fit_dufs_path(
                    fit_rows,
                    lambdas=args.dufs_lambdas,
                    reference=inner_developed["reference"],
                    feature_matrices=inner_developed["matrices"],
                    feature_roster=inner_developed["roster"],
                    feature_signs=inner_developed["signs"],
                    ordinary_model=model,
                    graph_k=DUFS_GRAPH_K,
                    dufs_seeds=DUFS_SEEDS,
                    dufs_epochs=args.dufs_epochs,
                )
                for lambda_, dufs_model in dufs_models.items():
                    lambda_scores, _ = _accumulator_metrics(
                        fit_rows, val_rows, dufs_model
                    )
                    inner_dufs_scores[str(lambda_)] = lambda_scores
                    for name, value in lambda_scores.items():
                        dufs_accumulator_values[lambda_][name].append(value)
            selected, ledger = _select_accumulator(scores)
            inner_ledgers.append({
                "inner_fold": inner_fold,
                "selected": selected,
                "roster": list(inner_developed["roster"]),
                "signs": inner_developed["signs"],
                "fallback_used": inner_developed["fallback_used"],
                "ledger": ledger,
                "dufs_scores": inner_dufs_scores,
            })
        averaged = {
            name: {
                task: _finite_mean([value[task] for value in values])
                for task in ("global", "localization", "early")
            } for name, values in accumulator_values.items()
        }
        selected_name, selection_ledger = _select_accumulator(averaged)
        selected_spec = _spec_by_name(selected_name)
        dufs_averaged = {
            lambda_: {
                task: _finite_mean([
                    value[task]
                    for value in by_accumulator[selected_name]
                ])
                for task in ("global", "localization", "early")
            }
            for lambda_, by_accumulator in dufs_accumulator_values.items()
            if by_accumulator[selected_name]
        }
        selected_lambda, fusion_selection_ledger = _select_dufs_lambda(
            averaged[selected_name], dufs_averaged
        )
        ordinary_model = UnifiedCausalIU.fit(
            train_rows,
            reference=developed["reference"],
            feature_matrices=developed["matrices"],
            feature_roster=developed["roster"],
            feature_signs=developed["signs"],
            accumulator=selected_spec,
        )
        dufs_models = (
            {} if args.skip_dufs else UnifiedCausalIU.fit_dufs_path(
                train_rows,
                lambdas=args.dufs_lambdas,
                reference=developed["reference"],
                feature_matrices=developed["matrices"],
                feature_roster=developed["roster"],
                feature_signs=developed["signs"],
                accumulator=selected_spec,
                ordinary_model=ordinary_model,
                graph_k=DUFS_GRAPH_K,
                dufs_seeds=DUFS_SEEDS,
                dufs_epochs=args.dufs_epochs,
            )
        )
        evaluated_ordinary = _evaluate_fitted_model(
            train_rows, test_rows, ordinary_model, selected_spec
        )
        evaluated_dufs = {
            lambda_: _evaluate_fitted_model(
                train_rows, test_rows, model, selected_spec
            )
            for lambda_, model in dufs_models.items()
        }
        selected_model = (
            ordinary_model if selected_lambda == 0.0 else dufs_models[selected_lambda]
        )
        selected_evaluation = (
            evaluated_ordinary if selected_lambda == 0.0
            else evaluated_dufs[selected_lambda]
        )
        _write_trajectories_npz(
            trajectory_path,
            test_rows,
            selected_evaluation["test_evidence"],
            selected_evaluation["test_trajectories"],
        )
        test_summary = selected_evaluation["metrics"]
        live_baselines = {
            **_evaluate_live_baselines(train_rows, test_rows),
            **_evaluate_registered_baselines(train_rows, test_rows),
        }
        fold_records = []
        for baseline_name, baseline in live_baselines.items():
            for record in baseline["per_question"]:
                record["outer_fold"] = outer_fold
                record["candidate"] = baseline_name
            fold_records.extend(baseline["per_question"])
        for record in evaluated_ordinary["records"]:
            record["outer_fold"] = outer_fold
            record["candidate"] = "unified_causal_iu"
            record["accumulator"] = selected_name
        fold_records.extend(evaluated_ordinary["records"])
        for lambda_, evaluation in evaluated_dufs.items():
            for record in evaluation["records"]:
                record["outer_fold"] = outer_fold
                record["candidate"] = _dufs_candidate(lambda_)
                record["accumulator"] = selected_name
                record["graph_lambda"] = float(lambda_)
            fold_records.extend(evaluation["records"])
        selected_records = []
        for original in selected_evaluation["records"]:
            record = dict(original)
            record["outer_fold"] = outer_fold
            record["candidate"] = "unified_causal_selected_fusion"
            record["accumulator"] = selected_name
            record["graph_lambda"] = float(selected_lambda)
            selected_records.append(record)
        fold_records.extend(selected_records)
        summary = {
            "outer_fold": outer_fold,
            "selected_accumulator": selected_name,
            "selected_fusion": (
                "ordinary_iu" if selected_lambda == 0.0 else "dufs_laplacian_iu_pcr"
            ),
            "selected_graph_lambda": float(selected_lambda),
            "metrics": test_summary,
            "ordinary_metrics": evaluated_ordinary["metrics"],
            "dufs_metrics": {
                str(lambda_): value["metrics"]
                for lambda_, value in evaluated_dufs.items()
            },
            "roster_size": len(developed["roster"]),
            "roster": list(developed["roster"]),
            "signs": developed["signs"],
            "redundancy": developed["redundancy"],
            "fallback_used": developed["fallback_used"],
            "inner_fallback_used": any(
                ledger["fallback_used"] for ledger in inner_ledgers
            ),
            "nonlinear_ceiling": developed["ceiling"],
            "efficiency": {
                "test_scoring_seconds": selected_evaluation["score_seconds"],
                "test_tokens": selected_evaluation["test_tokens"],
                "microseconds_per_token": (
                    1e6 * selected_evaluation["score_seconds"]
                    / max(selected_evaluation["test_tokens"], 1)
                ),
                "fixed_state_bytes": _fixed_state_bytes(selected_model),
            },
            "inner_ledgers": inner_ledgers,
            "selection_ledger": selection_ledger,
            "fusion_selection_ledger": fusion_selection_ledger,
            "inner_dufs_averaged": dufs_averaged,
            "live_baselines": {
                name: value["metrics"] for name, value in live_baselines.items()
            },
            "model": selected_model.as_dict(),
            "ordinary_model": ordinary_model.as_dict(),
        }
        _write_json(fold_path, {
            "run_schema": RUN_SCHEMA,
            "summary": summary,
            "per_question": fold_records,
        })
        _write_jsonl(out / "folds" / f"outer_{outer_fold}_atlas.jsonl", developed["atlas"])
        _write_jsonl(out / "folds" / f"outer_{outer_fold}_synergy.jsonl", developed["synergy"])
        fold_summaries.append(summary)
        per_question.extend(fold_records)
        print(f"[outer {outer_fold}] {selected_name} {test_summary['macro']}", flush=True)

    # AUROC/AUPRC are averaged per untouched fold; OOF probabilities are never pooled.
    combined_atlas, combined_synergy = [], []
    for fold in fold_summaries:
        fold_id = int(fold["outer_fold"])
        for filename, destination in (
            (f"outer_{fold_id}_atlas.jsonl", combined_atlas),
            (f"outer_{fold_id}_synergy.jsonl", combined_synergy),
        ):
            path = out / "folds" / filename
            if path.exists():
                for line in path.read_text().splitlines():
                    if line.strip():
                        destination.append({"outer_fold": fold_id, **json.loads(line)})
    _write_jsonl(out / "INFORMATION_ATLAS.jsonl", combined_atlas)
    _write_csv(out / "INFORMATION_ATLAS.csv", combined_atlas)
    _write_jsonl(out / "GROUP_SYNERGY.jsonl", combined_synergy)
    _write_json(out / "ROSTERS_PER_FOLD.json", [{
        "outer_fold": fold["outer_fold"],
        "roster": fold["roster"],
        "signs": fold["signs"],
        "redundancy": fold["redundancy"],
        "fallback_used": fold.get("fallback_used", False),
    } for fold in fold_summaries])

    aggregate = {
        task: _finite_mean([
            fold["metrics"]["macro"][task] for fold in fold_summaries
        ]) for task in ("global", "localization", "early")
    }
    ordinary_aggregate = {
        task: _finite_mean([
            fold["ordinary_metrics"]["macro"][task] for fold in fold_summaries
        ]) for task in ("global", "localization", "early")
    }
    fixed_dufs_aggregate = {
        str(lambda_): {
            task: _finite_mean([
                fold["dufs_metrics"][str(lambda_)]["macro"][task]
                for fold in fold_summaries
                if str(lambda_) in fold["dufs_metrics"]
            ])
            for task in ("global", "localization", "early")
        }
        for lambda_ in args.dufs_lambdas
        if any(str(lambda_) in fold["dufs_metrics"] for fold in fold_summaries)
    }
    fold_values = {
        task: [fold["metrics"]["macro"][task] for fold in fold_summaries]
        for task in aggregate
    }
    standard_error = {
        task: float(np.nanstd(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
        for task, values in fold_values.items()
    }
    baseline_aggregate = {}
    for name in BASELINE_REGISTRY:
        baseline_aggregate[name] = {
            task: _finite_mean([
                fold["live_baselines"][name]["macro"][task] for fold in fold_summaries
            ]) for task in ("global", "localization", "early")
        }
    incumbent_reference = {
        task: max(baseline_aggregate, key=lambda name: baseline_aggregate[name][task])
        for task in ("global", "localization", "early")
    }
    taskwise_incumbent = {
        task: baseline_aggregate[incumbent_reference[task]][task]
        for task in ("global", "localization", "early")
    }
    development_gate = finalist_gate(aggregate, taskwise_incumbent)
    fallback_folds = [
        int(fold["outer_fold"]) for fold in fold_summaries
        if fold.get("fallback_used") or fold.get("inner_fallback_used")
    ]
    development_gate["identifiability_fallback_folds"] = fallback_folds
    if fallback_folds:
        development_gate["pass"] = False
        development_gate["failure_reason"] = (
            "at least one outer/inner development fold used the top-three identifiability fallback"
        )
    grouped_bootstrap = grouped_bootstrap_comparisons(
        per_question,
        repeats=args.bootstrap_repeats,
        seed=SEED + 900,
        primary_candidate="unified_causal_selected_fusion",
    )
    grouped_bootstrap_ordinary = grouped_bootstrap_comparisons(
        per_question,
        repeats=args.bootstrap_repeats,
        seed=SEED + 900,
        primary_candidate="unified_causal_iu",
    )
    positive_ci_tasks = []
    for task, reference in incumbent_reference.items():
        interval = grouped_bootstrap["comparisons"][reference][task]["ci95"]
        if np.isfinite(interval[0]) and float(interval[0]) > 0.0:
            positive_ci_tasks.append(task)
    cluster_gate = {
        "pass": bool(development_gate["pass"] and positive_ci_tasks),
        "positive_ci_tasks": positive_ci_tasks,
        "taskwise_incumbent_reference": incumbent_reference,
        "requires_separate_confirmation_design": True,
        "authorization": "design only; this runner never launches inference or mutates cluster state",
    }
    robustness = (
        None if args.skip_robustness
        else run_robustness_panel(args, rows, fold_summaries, out)
    )
    result = {
        "aggregate": aggregate,
        "ordinary_aggregate": ordinary_aggregate,
        "fixed_dufs_aggregate": fixed_dufs_aggregate,
        "live_baseline_aggregate": baseline_aggregate,
        "taskwise_incumbent": taskwise_incumbent,
        "taskwise_incumbent_reference": incumbent_reference,
        "development_finalist_gate": development_gate,
        "grouped_bootstrap": grouped_bootstrap,
        "grouped_bootstrap_ordinary": grouped_bootstrap_ordinary,
        "cluster_confirmation_gate": cluster_gate,
        "robustness_11": (
            None if robustness is None else {
                "macro": robustness["macro"],
                "selected_accumulator": robustness["selected_accumulator"],
                "role": robustness["role"],
            }
        ),
        "fold_standard_error": standard_error,
        "folds": fold_summaries,
        "evaluation_rule": "per-test-fold metric then arithmetic fold mean; no pooled OOF AUROC",
        "baseline_registry": BASELINE_REGISTRY,
        "development_status": "all ProcessBench labels historically opened",
    }
    _write_json(out / "NESTED_CV_RESULTS.json", result)
    _write_json(out / "GROUPED_BOOTSTRAP.json", grouped_bootstrap)
    _write_json(out / "GROUPED_BOOTSTRAP_ORDINARY.json", grouped_bootstrap_ordinary)
    _write_json(out / "DECISION.json", {
        "development_finalist_gate": development_gate,
        "cluster_confirmation_gate": cluster_gate,
        "robustness_11": (
            None if robustness is None else {
                "macro": robustness["macro"],
                "role": robustness["role"],
            }
        ),
        "claim_status": "supervised development on historically opened labels",
    })
    _write_csv(out / "PER_QUESTION.csv", per_question)
    return result


def run_live_baselines(rows, fit_rows):
    """Score the access-matched live baselines on cached telemetry."""

    iu28 = fit_frozen_prefix_iu(fit_rows, include_elapsed_length=False)
    output = {}
    for row in rows:
        entropy = np.asarray(row["token_entropies"], dtype=float)
        causal = causal_trajectories(entropy, window=64, sw_window=16)
        raw, _ = causal_raw_prefix_matrix(row, None, include_elapsed_length=False)
        output[row["_source_group"] + "::" + row["model"]] = {
            "mean_entropy": causal["run_mean_ent"],
            "max_entropy_top5": causal["run_max_ent"],
            "sw_var": causal["sw_var_sofar"],
            "deepconf_proxy_w64": causal["neg_group_conf"],
            "iu28_without_length": iu28.risk(raw),
        }
    return output


def _live_baseline_trajectories(rows, iu28):
    output: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        name: [] for name in (
            "mean_entropy", "max_entropy_top5", "sw_var", "deepconf_proxy_w64",
            "iu28_without_length",
        )
    }
    for row in rows:
        entropy = np.asarray(row["token_entropies"], dtype=float)
        causal = causal_trajectories(entropy, window=64, sw_window=16)
        raw, _ = causal_raw_prefix_matrix(row, None, include_elapsed_length=False)
        iu_curve = np.asarray(iu28.risk(raw), dtype=float)
        risk_curves = {
            "mean_entropy": np.asarray(causal["run_mean_ent"], dtype=float),
            "max_entropy_top5": np.asarray(causal["run_max_ent"], dtype=float),
            "sw_var": np.asarray(causal["sw_var_sofar"], dtype=float),
            "deepconf_proxy_w64": np.asarray(causal["neg_group_conf"], dtype=float),
            "iu28_without_length": iu_curve,
        }
        contributions = {
            "mean_entropy": entropy,
            "max_entropy_top5": entropy,
            "sw_var": np.r_[risk_curves["sw_var"][0], np.maximum(0.0, np.diff(risk_curves["sw_var"]))],
            "deepconf_proxy_w64": np.r_[risk_curves["deepconf_proxy_w64"][0], np.maximum(0.0, np.diff(risk_curves["deepconf_proxy_w64"]))],
            "iu28_without_length": np.r_[iu_curve[0], np.maximum(0.0, np.diff(iu_curve))],
        }
        for name in output:
            output[name].append((risk_curves[name], np.asarray(contributions[name], dtype=float)))
    return output


def _evaluate_live_baselines(train_rows, test_rows):
    names = (
        "mean_entropy", "max_entropy_top5", "sw_var", "deepconf_proxy_w64",
        "iu28_without_length",
    )
    output = {name: {"cell_metrics": [], "per_question": []} for name in names}
    cells = sorted({(row["model"], row["family"]) for row in test_rows})
    for model, family in cells:
        fit_cell = [
            row for row in train_rows if row["model"] == model and row["family"] == family
        ]
        test_cell = [
            row for row in test_rows if row["model"] == model and row["family"] == family
        ]
        if not fit_cell or not test_cell:
            raise RuntimeError(f"missing live-baseline cell {model}/{family}")
        iu28 = fit_frozen_prefix_iu(fit_cell, include_elapsed_length=False)
        train_curves = _live_baseline_trajectories(fit_cell, iu28)
        test_curves = _live_baseline_trajectories(test_cell, iu28)
        for name in names:
            train_locations = (
                [_step_top5_location(row["token_entropies"], row) for row in fit_cell]
                if name == "max_entropy_top5" else None
            )
            test_locations = (
                [_step_top5_location(row["token_entropies"], row) for row in test_cell]
                if name == "max_entropy_top5" else None
            )
            warning = _warning_thresholds(fit_cell, train_curves[name])
            train_summary, _ = _curve_metrics(
                fit_cell,
                train_curves[name],
                localization_threshold=None,
                warning_thresholds=warning,
                localization_tokens=(
                    [location[0] for location in train_locations] if train_locations else None
                ),
                localization_steps=(
                    [location[1] for location in train_locations] if train_locations else None
                ),
            )
            test_summary, records = _curve_metrics(
                test_cell,
                test_curves[name],
                localization_threshold=train_summary["localization_threshold"],
                warning_thresholds=warning,
                global_thresholds=tuple(train_summary["global_thresholds"]),
                localization_tokens=(
                    [location[0] for location in test_locations] if test_locations else None
                ),
                localization_steps=(
                    [location[1] for location in test_locations] if test_locations else None
                ),
            )
            output[name]["cell_metrics"].append({
                "model": model,
                "family": family,
                **test_summary,
            })
            output[name]["per_question"].extend(records)
    for name in names:
        output[name]["metrics"] = {
            "macro": _record_primary_metrics(output[name]["per_question"]),
            "cells": output[name]["cell_metrics"],
        }
    return output


def _z_reference(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(np.mean(array)), max(float(np.std(array)), 1e-12)


def _historical_registered_outputs(rows, models, global_z, local_z):
    """Replay the frozen v2 architectures at their registered monitor budgets."""

    output = {name: [] for name in (
        "previous_one_shared", "previous_two_head", "frozen_global_iu_pcr",
    )}
    for row in rows:
        n = len(row["token_entropies"])
        global_final = _historical_global_score(models, row)
        local_curve = np.asarray(
            _historical_local_curve(models, row), dtype=float
        )
        online_curve = np.asarray(
            _historical_online_curve(models, row), dtype=float
        )
        local_running = np.maximum.accumulate(local_curve)
        global_budget, local_budget, online_budget = {}, {}, {}
        for budget in DEFAULT_BUDGETS:
            prefix = min(int(budget), n)
            global_budget[budget] = float(
                global_final if prefix == n else
                _historical_global_score(models, row, prefix)
            )
            local_budget[budget] = float(local_running[prefix - 1])
            online_budget[budget] = float(online_curve[prefix - 1])
        local_max = float(local_running[-1])
        two_detector = (
            0.5 * ((global_final - global_z[0]) / global_z[1])
            + 0.5 * ((local_max - local_z[0]) / local_z[1])
        )
        two_budget = {
            budget: (
                0.5 * ((global_budget[budget] - global_z[0]) / global_z[1])
                + 0.5 * ((local_budget[budget] - local_z[0]) / local_z[1])
            ) for budget in DEFAULT_BUDGETS
        }
        global_values = np.asarray([global_budget[budget] for budget in DEFAULT_BUDGETS])
        global_contribution = np.r_[global_values[0], np.maximum(0.0, np.diff(global_values))]
        global_choice = int(np.argmax(global_contribution))
        global_token = min(DEFAULT_BUDGETS[global_choice], n) - 1
        output["previous_one_shared"].append({
            "global_score": float(online_curve[-1]),
            "localization_score": float(online_curve[-1]),
            "localization_token": int(np.nanargmax(online_curve)),
            "risk": online_budget,
        })
        output["previous_two_head"].append({
            "global_score": float(global_final),
            "localization_score": float(two_detector),
            "localization_token": int(np.nanargmax(local_curve)),
            "risk": two_budget,
        })
        output["frozen_global_iu_pcr"].append({
            "global_score": float(global_final),
            "localization_score": float(global_final),
            "localization_token": int(global_token),
            "risk": global_budget,
        })
    return output


def _evaluate_registered_baselines(train_rows, test_rows):
    """Fold-local, cell-local replay of the prior one/two-head and Global IU baselines."""

    names = ("previous_one_shared", "previous_two_head", "frozen_global_iu_pcr")
    output = {name: {"cell_metrics": [], "per_question": []} for name in names}
    cells = sorted({(row["model"], row["family"]) for row in test_rows})
    for model, family in cells:
        fit_cell = [
            row for row in train_rows if row["model"] == model and row["family"] == family
        ]
        test_cell = [
            row for row in test_rows if row["model"] == model and row["family"] == family
        ]
        if not fit_cell or not test_cell:
            raise RuntimeError(f"missing historical-baseline cell {model}/{family}")
        models = _fit_historical_cell_models(fit_cell)
        global_z = _z_reference([
            _historical_global_score(models, row) for row in fit_cell
        ])
        local_z = _z_reference([
            float(np.max(_historical_local_curve(models, row)))
            for row in fit_cell
        ])
        fit_output = _historical_registered_outputs(fit_cell, models, global_z, local_z)
        test_output = _historical_registered_outputs(test_cell, models, global_z, local_z)
        target_fit = np.asarray([int(row["label"]) for row in fit_cell], dtype=int)
        for name in names:
            fit_items, test_items = fit_output[name], test_output[name]
            fit_steps = np.asarray([
                token_to_step(item["localization_token"], row)
                for item, row in zip(fit_items, fit_cell)
            ], dtype=int)
            threshold, _ = best_localization_threshold(
                [item["localization_score"] for item in fit_items], fit_steps, target_fit
            )
            records = []
            for row, item in zip(test_cell, test_items):
                step = token_to_step(item["localization_token"], row)
                prediction = step if item["localization_score"] > threshold else -1
                records.append({
                    "unit": row["_unit"],
                    "source_group": row["_source_group"],
                    "family": row["family"],
                    "model": row["model"],
                    "wrong": final_wrong(row),
                    "target_step": int(row["label"]),
                    "global_score": float(item["global_score"]),
                    "localization_score": float(item["localization_score"]),
                    "localization_token": int(item["localization_token"]),
                    "localization_step": int(step),
                    "prediction": int(prediction),
                    "first_alarm_5pct": None,
                    "first_alarm_10pct": None,
                    **{
                        f"risk_at_{budget}": float(item["risk"][budget])
                        for budget in DEFAULT_BUDGETS
                    },
                })
            cell_primary = _record_primary_metrics(records)
            output[name]["cell_metrics"].append({
                "model": model,
                "family": family,
                "macro": cell_primary,
                "localization_threshold": float(threshold),
                "warning_metrics": "not replayed; registered architecture stored only monitor budgets",
            })
            output[name]["per_question"].extend(records)
    for name in names:
        output[name]["metrics"] = {
            "macro": _record_primary_metrics(output[name]["per_question"]),
            "cells": output[name]["cell_metrics"],
        }
    return output


def write_report(out: Path, inventory, result: Mapping[str, Any] | None, args) -> None:
    smoke = bool(
        args.candidate_limit is not None
        or args.max_questions_per_family is not None
        or args.permutations < 200
        or (args.mode == "full" and args.outer_folds < 5)
        or (args.mode == "full" and args.bootstrap_repeats < 2000)
        or (args.mode == "full" and args.skip_robustness)
        or (args.mode == "full" and args.dufs_epochs < DUFS_EPOCHS)
    )
    lines = [
        "# Unified Causal IU-PCR v1",
        "",
        "**Claim status:** supervised development on historically opened ProcessBench telemetry; not untouched confirmation.",
        "" if not smoke else "**NON-REPORTABLE SMOKE/DEBUG RUN:** at least one sampling, candidate, permutation, or fold control is reduced.",
        "",
        "## Method contract",
        "",
        "One frozen causal DSP bank feeds two-component L2 IU-PCR. Ordinary IU is the zero-complexity primary; the DUFS arm changes only the sample graph and Laplacian regularization while keeping the exact roster, preprocessing, signs and accumulator. `R_T` is returned from the final online update rather than recomputed.",
        "",
        f"- Base streams: 9 primitive + 28 causalized historical token views.",
        f"- DSP coordinates before redundancy/selection: {len(all_feature_names())}.",
        f"- Information-atlas permutations: {args.permutations}.",
        f"- Candidate limit: {args.candidate_limit if args.candidate_limit is not None else 'none (full bank)'}.",
        f"- Rows actually analysed: {getattr(args, 'analysis_rows', 'unknown')} ({getattr(args, 'analysis_groups', 'unknown')} source questions).",
        f"- DUFS lambda path: {', '.join(str(value) for value in args.dufs_lambdas) if not args.skip_dufs else 'skipped'}; primary lambda {DUFS_PRIMARY_LAMBDA}; inner-fold selection only.",
        "- Grouping unit: dataset-qualified source question; scorer-model copies never cross folds.",
        "- Thresholds: clean-trace maximum over the monitored horizon at 5% and 10% calibration FPR.",
        "",
        "## Data inventory",
        "",
        "| Model | Family | Rows | Wrong | First error | SHA256 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in inventory:
        lines.append(
            f"| {row['model']} | {row['family']} | {row['rows']} | {row['wrong']} | {row['first_error']} | `{row['sha256'][:12]}` |"
        )
    lines.extend(["", "## Evaluation", ""])
    if result is None:
        lines.append("No outcome evaluation was run in this invocation.")
    else:
        aggregate = result["aggregate"]
        gate = result["development_finalist_gate"]
        cluster_gate = result["cluster_confirmation_gate"]
        lines.extend([
            "Metrics are computed inside each untouched outer fold and then averaged. Cross-fold OOF probabilities are not concatenated for AUROC.",
            "",
            "| Task | Fold-mean primary metric | Fold SE |",
            "|---|---:|---:|",
            f"| Global | {aggregate['global']:.4f} | {result['fold_standard_error']['global']:.4f} |",
            f"| Localization | {aggregate['localization']:.4f} | {result['fold_standard_error']['localization']:.4f} |",
            f"| Early (64/128) | {aggregate['early']:.4f} | {result['fold_standard_error']['early']:.4f} |",
            "",
            f"Development finalist gate: **{'PASS' if gate['pass'] else 'FAIL'}**. "
            f"Cluster-confirmation design gate: **{'PASS' if cluster_gate['pass'] else 'FAIL'}**.",
            f"Positive grouped-CI tasks: {', '.join(cluster_gate['positive_ci_tasks']) or 'none'}.",
            "",
            "### Same-roster fusion comparison",
            "",
            "| Fusion | Global | Localization | Early 64/128 |",
            "|---|---:|---:|---:|",
            f"| Ordinary IU-PCR | {result['ordinary_aggregate']['global']:.4f} | {result['ordinary_aggregate']['localization']:.4f} | {result['ordinary_aggregate']['early']:.4f} |",
        ])
        for lambda_, values in sorted(
            result.get("fixed_dufs_aggregate", {}).items(), key=lambda item: float(item[0])
        ):
            lines.append(
                f"| DUFS-LIU-PCR λ={lambda_} | {values['global']:.4f} | "
                f"{values['localization']:.4f} | {values['early']:.4f} |"
            )
        fusion_comparison = result["grouped_bootstrap"]["comparisons"].get(
            "unified_causal_iu"
        )
        if fusion_comparison is not None:
            lines.extend([
                "",
                "The selected-fusion minus ordinary-IU paired deltas are: "
                + ", ".join(
                    f"{task} {fusion_comparison[task]['delta']:+.4f} "
                    f"[{fusion_comparison[task]['ci95'][0]:+.4f}, "
                    f"{fusion_comparison[task]['ci95'][1]:+.4f}]"
                    for task in ("global", "localization", "early")
                )
                + ".",
                "",
                "Selected graph lambdas by outer fold: "
                + ", ".join(
                    f"fold {fold['outer_fold']}→{fold['selected_graph_lambda']}"
                    for fold in result["folds"]
                )
                + ".",
            ])
        if result.get("robustness_11") is None:
            lines.extend(["", "The eleven-cell opened-data robustness panel was skipped in this debug invocation."])
        else:
            robustness = result["robustness_11"]
            lines.extend([
                "",
                "### Eleven-cell Global/Early robustness",
                "",
                "This panel is scored only after all primary selections are frozen and is never fed back into selection.",
                "",
                f"- Global macro AUROC: {robustness['macro']['global']:.4f}.",
                f"- Early 64/128 macro AUROC: {robustness['macro']['early']:.4f}.",
            ])
        lines.extend([
            "",
            "### Access-matched baseline primaries",
            "",
            "| Method | Global | Localization | Early 64/128 |",
            "|---|---:|---:|---:|",
        ])
        for name, values in sorted(result["live_baseline_aggregate"].items()):
            lines.append(
                f"| {name} | {values['global']:.4f} | {values['localization']:.4f} | {values['early']:.4f} |"
            )
        classic = result["grouped_bootstrap"]["comparisons"].get("frozen_global_iu_pcr")
        if classic is not None:
            lines.extend([
                "",
                "Direct selected Unified Causal minus the fold-refit classic mixed-v2 "
                "30-feature-contract IU-PCR (with final length excluded): "
                + ", ".join(
                    f"{task} {classic[task]['delta']:+.4f} "
                    f"[{classic[task]['ci95'][0]:+.4f}, {classic[task]['ci95'][1]:+.4f}]"
                    for task in ("global", "localization", "early")
                )
                + ".",
            ])
        lines.extend([
            "",
            "### Paired grouped intervals against taskwise incumbents",
            "",
            "| Task | Reference | Delta | Grouped 95% CI |",
            "|---|---|---:|---:|",
        ])
        for task, reference in result["taskwise_incumbent_reference"].items():
            comparison = result["grouped_bootstrap"]["comparisons"][reference][task]
            lines.append(
                f"| {task} | {reference} | {comparison['delta']:+.4f} | "
                f"[{comparison['ci95'][0]:+.4f}, {comparison['ci95'][1]:+.4f}] |"
            )
    lines.extend([
        "",
        "## Baseline matrix",
        "",
        "The runner fold-refits the previous one-shared, two-head, IU28-without-length, max-entropy+top-5, `sw_var`, DeepConf-w64 proxy, and length-free Global IU-PCR methods. Fits and calibration thresholds remain cell-local while source questions define the shared outer folds. Historical artifacts are provenance references only; they are never mixed into the new fold metrics.",
        "",
        "## Promotion gate",
        "",
        "A development finalist must stay within −0.010 Global AUROC, −0.010 Localization F1, and −0.015 Early AUROC, and improve at least one task by +0.010. Fresh cluster inference remains a separate decision and requires a positive grouped 95% CI on one task with no regression on the other two.",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "atlas", "full"))
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/unified_causal_iu_v1")
    parser.add_argument("--models", default=",".join(PRIMARY_MODELS))
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--atlas-folds", type=int, default=5)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--max-questions-per-family", type=int)
    parser.add_argument("--skip-synergy", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-dufs", action="store_true")
    parser.add_argument(
        "--dufs-lambdas",
        default=",".join(str(value) for value in DUFS_DEFAULT_LAMBDAS),
    )
    parser.add_argument("--dufs-epochs", type=int, default=DUFS_EPOCHS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not PROTOCOL.exists():
        raise FileNotFoundError(PROTOCOL)
    args.out.mkdir(parents=True, exist_ok=True)
    args.dufs_lambdas = _parse_dufs_lambdas(args.dufs_lambdas)
    if args.dufs_epochs < 1:
        raise ValueError("--dufs-epochs must be positive")
    models = tuple(value for value in args.models.split(",") if value)
    families = tuple(value for value in args.families.split(",") if value)
    unknown = set(models) - set(MODEL_PATHS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    started = time.perf_counter()
    inventory, rows = preflight(args.data_root, models, families)
    if args.max_questions_per_family is not None:
        selected_groups = set()
        for family in families:
            representatives = {}
            for row in rows:
                if row["family"] == family:
                    representatives.setdefault(row["_source_group"], row)
            wanted = int(args.max_questions_per_family)
            halves = []
            for target in (0, 1):
                halves.extend(sorted(
                    group for group, row in representatives.items()
                    if final_wrong(row) == target
                )[: max(1, wanted // 2)])
            selected_groups.update(halves[:wanted])
        rows = [row for row in rows if row["_source_group"] in selected_groups]
    args.analysis_rows = len(rows)
    args.analysis_groups = len({row["_source_group"] for row in rows})
    args.robustness_payload = None
    if args.mode == "full" and not args.skip_robustness:
        args.robustness_payload = _load_robustness_cells(args.data_root, rows)
    _write_json(args.out / "INVENTORY.json", inventory)
    _write_json(args.out / "RUN_DEFINITION.json", {
        "mode": args.mode,
        "run_schema": RUN_SCHEMA,
        "protocol": str(PROTOCOL.resolve()),
        "protocol_sha256": _sha256(PROTOCOL),
        "data_root": str(args.data_root.resolve()),
        "models": models,
        "families": families,
        "seed": SEED,
        "permutations": args.permutations,
        "atlas_folds": args.atlas_folds,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "bootstrap_repeats": args.bootstrap_repeats,
        "candidate_limit": args.candidate_limit,
        "max_questions_per_family": args.max_questions_per_family,
        "analysis_rows": args.analysis_rows,
        "analysis_source_questions": args.analysis_groups,
        "skip_synergy": args.skip_synergy,
        "skip_robustness": args.skip_robustness,
        "skip_dufs": args.skip_dufs,
        "dufs_lambdas": args.dufs_lambdas,
        "dufs_primary_lambda": DUFS_PRIMARY_LAMBDA,
        "dufs_graph_k": DUFS_GRAPH_K,
        "dufs_seeds": DUFS_SEEDS,
        "dufs_epochs": args.dufs_epochs,
        "workers": args.workers,
        "base_feature_count": 37,
        "dsp_feature_count": len(all_feature_names()),
        "baselines": BASELINE_REGISTRY,
        "historical_baseline_provenance": HISTORICAL_BASELINE_PROVENANCE,
        "cluster_mutation": False,
    })
    result = None
    if args.mode == "atlas":
        run_atlas(args, rows, args.out)
    elif args.mode == "full":
        result = run_full(args, rows, args.out)
    write_report(args.out, inventory, result, args)
    _write_json(args.out / "COMPLETION.json", {
        "mode": args.mode,
        "elapsed_seconds": time.perf_counter() - started,
        "completed": True,
        "result_available": result is not None,
    })
    print(f"complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
