#!/usr/bin/env python3
"""Repeated grouped development search over Unified Causal IU-PCR subsets.

This runner is intentionally separate from ``run_unified_causal_iu_v1.py``.
It performs retrospective supervised development on existing ProcessBench
telemetry, not untouched confirmation:

1. materialize the causal 37x28 DSP bank once per grouped fold/reference;
2. compare fixed, interpretable feature rosters with ordinary IU-PCR;
3. optionally blend fold-local supervised coordinate relevance into the IU
   weights;
4. run DUFS-Laplacian IU and its lambda path only for named finalists;
5. freeze the winning development configuration and score a different scorer
   model as a robustness/generalization panel without refitting on it.

All label-aware choices are fit-partition-only inside each grouped split.
Validation labels are read only for the final report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types
from typing import Any, Mapping, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [str(ROOT / "spectral_utils")]
    sys.modules["spectral_utils"] = package

from scripts.run_unified_causal_iu_v1 import (  # noqa: E402
    FAMILIES,
    PRIMARY_EARLY_BUDGETS,
    _curve_metrics,
    _curves_for_spec,
    _grouped_splits,
    _record_primary_metrics,
    _warning_thresholds,
    _write_csv,
    _write_json,
    _write_jsonl,
    preflight,
)
from spectral_utils.unified_causal_evaluation import (  # noqa: E402
    build_atlas_samples,
    derive_supervised_signs,
    final_wrong,
    processbench_metrics,
    safe_auc,
)
from spectral_utils.unified_causal_iu import (  # noqa: E402
    BASE_NAMES,
    AccumulatorSpec,
    UnifiedCausalIU,
    all_feature_names,
    base_matrix,
    causal_feature_matrix,
    fit_base_reference,
)
from spectral_utils.unified_causal_subset_search import (  # noqa: E402
    TASKS,
    TRANSFORM_FAMILIES,
    base_mask_rosters,
    blended_multipliers,
    rank_against_control,
    reweight_model,
    structured_rosters,
    supervised_relevance,
)


SEED = 20260818
RUN_SCHEMA = 1
IDENTITY = AccumulatorSpec("identity")


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


def _stable_hash(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_floats(value: str, *, include_zero: bool = False) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(float(item) for item in value.split(",") if item.strip()))
    if not values:
        return (0.0,) if include_zero else ()
    lower = 0.0 if include_zero else np.nextafter(0.0, 1.0)
    if any(not np.isfinite(item) or item < lower for item in values):
        raise ValueError("numeric grid contains an invalid value")
    return values


def _parse_names(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def _tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _choose_groups(
    rows: Sequence[Mapping[str, Any]],
    families: Sequence[str],
    questions_per_family: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select exactly balanced source groups without separating scorer copies."""

    rows = [dict(row) for row in rows]
    if questions_per_family is None:
        return rows, {"limited": False, "selected_groups": len({row["_source_group"] for row in rows})}
    wanted = int(questions_per_family)
    if wanted < 4:
        raise ValueError("questions per family must be at least four")
    rng = np.random.default_rng(int(seed))
    selected: set[str] = set()
    ledger = []
    for family in families:
        representatives: dict[str, Mapping[str, Any]] = {}
        for row in rows:
            if str(row["family"]) == family:
                representatives.setdefault(str(row["_source_group"]), row)
        by_label = {
            label: sorted(
                group for group, row in representatives.items()
                if final_wrong(row) == label
            )
            for label in (0, 1)
        }
        for label in by_label:
            rng.shuffle(by_label[label])
        quotas = {0: wanted // 2, 1: wanted - wanted // 2}
        chosen = by_label[0][:quotas[0]] + by_label[1][:quotas[1]]
        if len(chosen) < wanted:
            remaining = sorted(set(representatives) - set(chosen))
            rng.shuffle(remaining)
            chosen.extend(remaining[: wanted - len(chosen)])
        if len(chosen) < min(wanted, len(representatives)):
            raise RuntimeError(f"failed to select requested groups for {family}")
        selected.update(chosen)
        ledger.append({
            "family": family,
            "available": len(representatives),
            "selected": len(chosen),
            "selected_clean": sum(final_wrong(representatives[group]) == 0 for group in chosen),
            "selected_wrong": sum(final_wrong(representatives[group]) == 1 for group in chosen),
        })
    output = [row for row in rows if str(row["_source_group"]) in selected]
    return output, {
        "limited": True,
        "questions_per_family": wanted,
        "selected_groups": len(selected),
        "families": ledger,
    }


def _candidate_rosters(args) -> dict[str, tuple[str, ...]]:
    if args.transform_family:
        if args.transform_family not in TRANSFORM_FAMILIES:
            raise ValueError(
                f"unknown transform family {args.transform_family!r}; "
                f"choose from {sorted(TRANSFORM_FAMILIES)}"
            )
        rosters = base_mask_rosters(TRANSFORM_FAMILIES[args.transform_family])
        rosters = {
            f"{name}_{args.transform_family}": roster
            for name, roster in rosters.items()
        }
    else:
        rosters = structured_rosters(args.stage)
    requested = _parse_names(args.candidates)
    if requested:
        unknown = set(requested) - set(rosters)
        if unknown:
            raise ValueError(f"unknown candidate rosters: {sorted(unknown)}")
        rosters = {name: rosters[name] for name in requested}
    if not rosters:
        raise ValueError("candidate roster is empty")
    return rosters


def _named_selection(value: str, available: Sequence[str]) -> set[str]:
    names = set(_parse_names(value))
    if "all" in names:
        return set(available)
    unknown = names - set(available)
    if unknown:
        raise ValueError(f"unknown named finalists: {sorted(unknown)}")
    return names


def _build_feature_matrices(rows, reference, raw_matrices):
    return [
        causal_feature_matrix(row, reference, raw_base=raw)
        for row, raw in zip(rows, raw_matrices)
    ]


def _sample_sets(rows, reference, feature_matrices):
    return [
        build_atlas_samples(
            rows,
            reference,
            target=target,
            feature_matrices=feature_matrices,
        )
        for target in TASKS
    ]


def _cached_evidence(model, matrices):
    return [
        model.evidence_from_feature_matrix(matrix)
        for matrix in matrices
    ]


def _evaluate_cached(
    fit_rows,
    test_rows,
    model,
    fit_matrices,
    test_matrices,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Identity-accumulator evaluation with fit-only operating thresholds."""

    fit_evidence = _cached_evidence(model, fit_matrices)
    test_evidence = _cached_evidence(model, test_matrices)
    fit_trajectories = _curves_for_spec(fit_evidence, IDENTITY)
    test_trajectories = _curves_for_spec(test_evidence, IDENTITY)
    warning = _warning_thresholds(fit_rows, fit_trajectories)
    fit_summary, _ = _curve_metrics(
        fit_rows,
        fit_trajectories,
        localization_threshold=None,
        warning_thresholds=warning,
    )
    test_summary, records = _curve_metrics(
        test_rows,
        test_trajectories,
        localization_threshold=fit_summary["localization_threshold"],
        warning_thresholds=warning,
        global_thresholds=tuple(fit_summary["global_thresholds"]),
    )
    return test_summary, records


def _variant_name(base: str, fusion: str, lambda_: float, alpha: float) -> str:
    name = base
    if fusion == "dufs":
        name += f"__dufs_l{_tag(lambda_)}"
    if alpha > 0.0:
        name += f"__rw_a{_tag(alpha)}"
    return name


def _fit_fold_variants(
    args,
    rosters,
    fit_rows,
    fit_matrices,
    reference,
    sample_sets,
    reweight_candidates,
    dufs_candidates,
):
    signs_all = derive_supervised_signs(sample_sets, all_feature_names(reference.names))
    variants: dict[str, UnifiedCausalIU] = {}
    configs: dict[str, dict[str, Any]] = {}
    fit_diagnostics: dict[str, Any] = {}
    for roster_name, roster in rosters.items():
        ordinary = UnifiedCausalIU.fit(
            fit_rows,
            feature_roster=roster,
            feature_signs=signs_all,
            accumulator=IDENTITY,
            positions_per_trace=args.positions_per_trace,
            reference=reference,
            feature_matrices=fit_matrices,
        )
        relevance_full = supervised_relevance(sample_sets, roster)
        relevance_lookup = dict(zip(roster, relevance_full))
        retained_relevance = np.asarray([
            relevance_lookup[name] for name in ordinary.feature_names
        ])
        alpha_grid = args.reweight_alphas if roster_name in reweight_candidates else (0.0,)
        for alpha in alpha_grid:
            model = ordinary if alpha == 0.0 else reweight_model(
                ordinary,
                blended_multipliers(retained_relevance, alpha),
                fit_rows,
                fit_matrices,
                positions_per_trace=args.positions_per_trace,
                alpha=alpha,
            )
            name = _variant_name(roster_name, "ordinary", 0.0, alpha)
            variants[name] = model
            configs[name] = {
                "candidate": name,
                "roster": roster_name,
                "fusion": "ordinary",
                "lambda": 0.0,
                "reweight_alpha": float(alpha),
                "input_features": len(roster),
                "retained_features": len(model.feature_names),
            }

        if roster_name in dufs_candidates:
            path = UnifiedCausalIU.fit_dufs_path(
                fit_rows,
                lambdas=args.dufs_lambdas,
                feature_roster=roster,
                feature_signs=signs_all,
                accumulator=IDENTITY,
                positions_per_trace=args.positions_per_trace,
                reference=reference,
                feature_matrices=fit_matrices,
                ordinary_model=ordinary,
                graph_k=args.dufs_graph_k,
                dufs_seeds=args.dufs_seeds,
                dufs_epochs=args.dufs_epochs,
            )
            for lambda_, dufs_model in path.items():
                for alpha in alpha_grid:
                    model = dufs_model if alpha == 0.0 else reweight_model(
                        dufs_model,
                        blended_multipliers(retained_relevance, alpha),
                        fit_rows,
                        fit_matrices,
                        positions_per_trace=args.positions_per_trace,
                        alpha=alpha,
                    )
                    name = _variant_name(roster_name, "dufs", lambda_, alpha)
                    variants[name] = model
                    configs[name] = {
                        "candidate": name,
                        "roster": roster_name,
                        "fusion": "dufs",
                        "lambda": float(lambda_),
                        "reweight_alpha": float(alpha),
                        "input_features": len(roster),
                        "retained_features": len(model.feature_names),
                    }
            fit_diagnostics[roster_name] = {
                "dufs": {
                    str(lambda_): _jsonable(model.diagnostics)
                    for lambda_, model in path.items()
                },
                "relevance": dict(zip(roster, relevance_full.tolist())),
            }
    return variants, configs, fit_diagnostics


def _family_primary(records: Sequence[Mapping[str, Any]], family: str) -> dict[str, float]:
    subset = [record for record in records if str(record["family"]) == str(family)]
    if not subset:
        return {task: float("nan") for task in TASKS}
    labels = np.asarray([record["wrong"] for record in subset], dtype=int)
    localization = processbench_metrics(
        [record["prediction"] for record in subset],
        [record["target_step"] for record in subset],
    )
    return {
        "global": safe_auc(labels, [record["global_score"] for record in subset]),
        "localization": float(localization["f1"]),
        "early": float(np.nanmean([
            safe_auc(labels, [record[f"risk_at_{budget}"] for record in subset])
            for budget in PRIMARY_EARLY_BUDGETS
        ])),
    }


def _aggregate_payloads(payloads, control, families):
    by_repeat: dict[tuple[int, str], list[dict[str, Any]]] = {}
    configs: dict[str, dict[str, Any]] = {}
    fold_metrics = []
    fold_family_metrics = []
    for payload in payloads:
        repeat, fold = int(payload["repeat"]), int(payload["fold"])
        configs.update(payload["variant_configs"])
        for candidate, result in payload["results"].items():
            records = [dict(record) for record in result["records"]]
            by_repeat.setdefault((repeat, candidate), []).extend(records)
            fold_metrics.append({
                "repeat": repeat,
                "fold": fold,
                "candidate": candidate,
                **{task: float(result["metrics"]["macro"][task]) for task in TASKS},
                "n_features": int(configs[candidate]["input_features"]),
                "retained_features": int(configs[candidate]["retained_features"]),
            })
            for family in families:
                fold_family_metrics.append({
                    "repeat": repeat,
                    "fold": fold,
                    "candidate": candidate,
                    "family": family,
                    **_family_primary(records, family),
                })

    repeat_metrics, family_metrics = [], []
    candidates = sorted(configs)
    repeats = sorted({repeat for repeat, _ in by_repeat})
    for repeat in repeats:
        for candidate in candidates:
            current_folds = [
                row for row in fold_metrics
                if row["repeat"] == repeat and row["candidate"] == candidate
            ]
            if not current_folds:
                continue
            repeat_metrics.append({
                "repeat": repeat,
                "candidate": candidate,
                **{
                    task: float(np.nanmean([row[task] for row in current_folds]))
                    for task in TASKS
                },
                "n_features": int(configs[candidate]["input_features"]),
            })
            for family in families:
                current_family_folds = [
                    row for row in fold_family_metrics
                    if row["repeat"] == repeat
                    and row["candidate"] == candidate
                    and row["family"] == family
                ]
                family_metrics.append({
                    "repeat": repeat,
                    "candidate": candidate,
                    "family": family,
                    **{
                        task: float(np.nanmean([
                            row[task] for row in current_family_folds
                        ])) if current_family_folds else float("nan")
                        for task in TASKS
                    },
                })

    aggregate = []
    for candidate in candidates:
        current = [row for row in repeat_metrics if row["candidate"] == candidate]
        if not current:
            continue
        aggregate.append({
            "candidate": candidate,
            **{
                task: float(np.nanmean([row[task] for row in current]))
                for task in TASKS
            },
            **{
                f"{task}_repeat_sd": float(np.nanstd([row[task] for row in current]))
                for task in TASKS
            },
            "n_features": int(configs[candidate]["input_features"]),
            "retained_features_mean": float(np.mean([
                row["retained_features"]
                for row in fold_metrics if row["candidate"] == candidate
            ])),
            "repeats": len(current),
        })
    if control not in {row["candidate"] for row in aggregate}:
        raise ValueError(f"control {control!r} is absent from evaluated variants")
    ranking = rank_against_control(aggregate, control)

    stability = []
    for candidate in candidates:
        pareto_count = top_count = survive_count = 0
        for repeat in repeats:
            current = [row for row in repeat_metrics if row["repeat"] == repeat]
            ranked = rank_against_control(current, control)
            lookup = {row["candidate"]: row for row in ranked}
            if candidate not in lookup:
                continue
            pareto_count += int(lookup[candidate]["pareto"])
            survive_count += int(lookup[candidate]["survives_noninferiority"])
            top_count += int(ranked[0]["candidate"] == candidate)
        directions = []
        for family in families:
            candidate_values = [
                row for row in family_metrics
                if row["candidate"] == candidate and row["family"] == family
            ]
            control_values = [
                row for row in family_metrics
                if row["candidate"] == control and row["family"] == family
            ]
            if not candidate_values or not control_values:
                continue
            candidate_mean = {
                task: float(np.nanmean([row[task] for row in candidate_values]))
                for task in TASKS
            }
            control_mean = {
                task: float(np.nanmean([row[task] for row in control_values]))
                for task in TASKS
            }
            directions.append(all(candidate_mean[task] >= control_mean[task] for task in TASKS))
        stability.append({
            "candidate": candidate,
            "pareto_repeat_fraction": pareto_count / max(1, len(repeats)),
            "noninferior_repeat_fraction": survive_count / max(1, len(repeats)),
            "top_repeat_fraction": top_count / max(1, len(repeats)),
            "families_nonnegative_all_tasks": int(sum(directions)),
            "families_compared": len(directions),
        })
    return {
        "configs": configs,
        "fold_metrics": fold_metrics,
        "repeat_metrics": repeat_metrics,
        "family_metrics": family_metrics,
        "aggregate": aggregate,
        "ranking": ranking,
        "stability": stability,
        "oof_records": [
            {"repeat": repeat, "candidate": candidate, **record}
            for (repeat, candidate), records in by_repeat.items()
            for record in records
        ],
    }


def _fit_named_variant(
    args,
    config,
    roster,
    rows,
    reference,
    matrices,
    sample_sets,
):
    signs = derive_supervised_signs(sample_sets, all_feature_names(reference.names))
    ordinary = UnifiedCausalIU.fit(
        rows,
        feature_roster=roster,
        feature_signs=signs,
        accumulator=IDENTITY,
        positions_per_trace=args.positions_per_trace,
        reference=reference,
        feature_matrices=matrices,
    )
    if config["fusion"] == "dufs":
        model = UnifiedCausalIU.fit_dufs_path(
            rows,
            lambdas=(float(config["lambda"]),),
            feature_roster=roster,
            feature_signs=signs,
            accumulator=IDENTITY,
            positions_per_trace=args.positions_per_trace,
            reference=reference,
            feature_matrices=matrices,
            ordinary_model=ordinary,
            graph_k=args.dufs_graph_k,
            dufs_seeds=args.dufs_seeds,
            dufs_epochs=args.dufs_epochs,
        )[float(config["lambda"])]
    else:
        model = ordinary
    alpha = float(config["reweight_alpha"])
    if alpha > 0.0:
        relevance = supervised_relevance(sample_sets, roster)
        lookup = dict(zip(roster, relevance))
        retained = np.asarray([lookup[name] for name in model.feature_names])
        model = reweight_model(
            model,
            blended_multipliers(retained, alpha),
            rows,
            matrices,
            positions_per_trace=args.positions_per_trace,
            alpha=alpha,
        )
    return model


def _run_validation(
    args,
    development_rows,
    development_raw,
    validation_rows,
    rosters,
    configs,
    requested_names,
    reference_names=None,
):
    """Freeze on development and score validation models without any refit."""

    names = list(dict.fromkeys(str(name) for name in requested_names))
    unknown = set(names) - set(configs)
    if unknown:
        raise ValueError(f"validation candidates are absent from development: {sorted(unknown)}")
    if not names:
        raise ValueError("validation requires at least one frozen candidate")

    reference = fit_base_reference(
        development_rows,
        names=reference_names or BASE_NAMES,
        positions_per_trace=args.positions_per_trace,
        raw_base_matrices=development_raw,
    )
    development_matrices = _build_feature_matrices(
        development_rows, reference, development_raw
    )
    sample_sets = _sample_sets(development_rows, reference, development_matrices)
    models = {
        name: _fit_named_variant(
            args,
            configs[name],
            rosters[str(configs[name]["roster"])],
            development_rows,
            reference,
            development_matrices,
            sample_sets,
        )
        for name in names
    }

    fit_evidence = {
        name: _cached_evidence(model, development_matrices)
        for name, model in models.items()
    }
    validation_evidence = {name: [] for name in models}
    for row in validation_rows:
        matrix = causal_feature_matrix(row, reference)
        for name, model in models.items():
            validation_evidence[name].append(model.evidence_from_feature_matrix(matrix))

    output, records = {}, []
    for name, model in models.items():
        fit_trajectories = _curves_for_spec(fit_evidence[name], IDENTITY)
        validation_trajectories = _curves_for_spec(validation_evidence[name], IDENTITY)
        warning = _warning_thresholds(development_rows, fit_trajectories)
        fit_summary, _ = _curve_metrics(
            development_rows,
            fit_trajectories,
            localization_threshold=None,
            warning_thresholds=warning,
        )
        summary, current_records = _curve_metrics(
            validation_rows,
            validation_trajectories,
            localization_threshold=fit_summary["localization_threshold"],
            warning_thresholds=warning,
            global_thresholds=tuple(fit_summary["global_thresholds"]),
        )
        output[name] = {
            "metrics": summary,
            "model": model.as_dict(),
            "fit_calibration": {
                "localization_threshold": fit_summary["localization_threshold"],
                "warning_thresholds": list(warning),
                "global_thresholds": fit_summary["global_thresholds"],
            },
        }
        records.extend({"candidate": name, **record} for record in current_records)
    return output, records


def _report(run_definition, aggregate, validation, winner, control) -> str:
    ranking = aggregate["ranking"]
    lines = [
        "# Unified Causal subset search v1",
        "",
        "Retrospective supervised development; not untouched confirmation.",
        "",
        f"- Development winner: `{winner}`",
        f"- Ranking control: `{control}`",
        f"- Repeats/folds: {run_definition['repeats']} x {run_definition['folds']}",
        f"- Development groups: {run_definition['development_groups']}",
        "",
        "## Repeated grouped-CV ranking",
        "",
        "| rank | candidate | features | Global | Localization | Early | maximin | oracle regret | Pareto |",
        "|---:|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for index, row in enumerate(ranking, 1):
        lines.append(
            f"| {index} | {row['candidate']} | {int(row['n_features'])} | "
            f"{row['global']:.4f} | {row['localization']:.4f} | {row['early']:.4f} | "
            f"{row['maximin_normalized']:.3f} | {row['max_oracle_regret']:.3f} | "
            f"{'yes' if row['pareto'] else 'no'} |"
        )
    if validation:
        lines.extend([
            "",
            "## Frozen scorer-model robustness",
            "",
            "No references, signs, IU weights, thresholds, alpha, or lambda were refit on this panel.",
            "",
            "| candidate | Global | Localization | Early |",
            "|---|---:|---:|---:|",
        ])
        for name, payload in validation.items():
            macro = payload["metrics"]["macro"]
            lines.append(
                f"| {name} | {macro['global']:.4f} | {macro['localization']:.4f} | "
                f"{macro['early']:.4f} |"
            )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "The subset, signs, relevance alpha, and DUFS lambda were all eligible for "
        "selection on previously opened labels. The scorer-model panel is therefore "
        "a robustness/validation check, not an untouched paper claim.",
        "",
    ])
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=ROOT / "results/unified_causal_subset_search_v1")
    parser.add_argument("--models", default="qwen3_8b")
    parser.add_argument("--validation-models", default="llama31_8b")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--questions-per-family", type=int, default=32)
    parser.add_argument("--stage", choices=("a", "b", "c", "d", "e"), default="a")
    parser.add_argument("--transform-family")
    parser.add_argument("--candidates", default="")
    parser.add_argument("--control", default="all37_full")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--positions-per-trace", type=int, default=32)
    parser.add_argument("--reweight-candidates", default="")
    parser.add_argument("--reweight-alphas", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--dufs-candidates", default="")
    parser.add_argument("--dufs-lambdas", default="0.03,0.1,0.3,1,3")
    parser.add_argument("--dufs-epochs", type=int, default=80)
    parser.add_argument("--dufs-seeds", default="11,23,37")
    parser.add_argument("--dufs-graph-k", type=int, default=7)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1 or args.folds < 2:
        raise ValueError("search requires at least one repeat and two folds")
    if args.positions_per_trace < 1:
        raise ValueError("positions per trace must be positive")
    args.reweight_alphas = _parse_floats(args.reweight_alphas, include_zero=True)
    if 0.0 not in args.reweight_alphas or any(value > 1.0 for value in args.reweight_alphas):
        raise ValueError("reweight alpha grid must include zero and stay in [0,1]")
    args.dufs_lambdas = _parse_floats(args.dufs_lambdas)
    args.dufs_seeds = tuple(int(value) for value in _parse_floats(args.dufs_seeds, include_zero=True))
    models = _parse_names(args.models)
    validation_models = _parse_names(args.validation_models)
    families = _parse_names(args.families)
    rosters = _candidate_rosters(args)
    reweight_candidates = _named_selection(args.reweight_candidates, rosters)
    dufs_candidates = _named_selection(args.dufs_candidates, rosters)
    if args.control not in rosters:
        if len(rosters) == 1:
            args.control = next(iter(rosters))
        else:
            raise ValueError(f"control {args.control!r} is not an ordinary roster")

    started = time.perf_counter()
    inventory, loaded_rows = preflight(args.data_root, models, families)
    development_rows, selection = _choose_groups(
        loaded_rows, families, args.questions_per_family, SEED
    )
    development_raw = [base_matrix(row) for row in development_rows]
    validation_inventory, validation_rows = ([], [])
    if not args.skip_validation and validation_models:
        validation_inventory, validation_rows = preflight(
            args.data_root, validation_models, families
        )

    run_definition = {
        "run_schema": RUN_SCHEMA,
        "claim_boundary": "retrospective supervised development; validation is not untouched confirmation",
        "data_root": str(args.data_root.resolve()),
        "models": models,
        "validation_models": validation_models,
        "families": families,
        "questions_per_family": args.questions_per_family,
        "selection": selection,
        "development_rows": len(development_rows),
        "development_groups": len({row["_source_group"] for row in development_rows}),
        "repeats": args.repeats,
        "folds": args.folds,
        "positions_per_trace": args.positions_per_trace,
        "stage": args.stage,
        "transform_family": args.transform_family,
        "rosters": {name: list(roster) for name, roster in rosters.items()},
        "control": args.control,
        "reweight_candidates": sorted(reweight_candidates),
        "reweight_alphas": args.reweight_alphas,
        "dufs_candidates": sorted(dufs_candidates),
        "dufs_lambdas": args.dufs_lambdas,
        "dufs_epochs": args.dufs_epochs,
        "dufs_seeds": args.dufs_seeds,
        "dufs_graph_k": args.dufs_graph_k,
        "seed": SEED,
        "inventory": inventory,
        "validation_inventory": validation_inventory,
    }
    run_hash = _stable_hash(run_definition)
    run_definition["run_hash"] = run_hash
    args.out.mkdir(parents=True, exist_ok=True)
    run_path = args.out / "RUN_DEFINITION.json"
    if run_path.exists() and not args.force:
        existing = json.loads(run_path.read_text())
        if existing.get("run_hash") != run_hash:
            raise RuntimeError("output directory contains a different run; use a new --out or --force")
    _write_json(run_path, run_definition)

    payloads = []
    for repeat in range(args.repeats):
        splits = _grouped_splits(development_rows, args.folds, SEED + 1009 * repeat)
        for fold, (fit_indices, test_indices) in enumerate(splits):
            checkpoint = args.out / "folds" / f"repeat_{repeat:02d}_fold_{fold:02d}.json"
            if checkpoint.exists() and not args.force:
                payload = json.loads(checkpoint.read_text())
                if payload.get("run_hash") != run_hash:
                    raise RuntimeError(f"stale checkpoint: {checkpoint}")
                payloads.append(payload)
                continue
            fold_started = time.perf_counter()
            fit_rows = [development_rows[index] for index in fit_indices]
            test_rows = [development_rows[index] for index in test_indices]
            fit_raw = [development_raw[index] for index in fit_indices]
            test_raw = [development_raw[index] for index in test_indices]
            reference = fit_base_reference(
                fit_rows,
                positions_per_trace=args.positions_per_trace,
                raw_base_matrices=fit_raw,
            )
            fit_matrices = _build_feature_matrices(fit_rows, reference, fit_raw)
            test_matrices = _build_feature_matrices(test_rows, reference, test_raw)
            sample_sets = _sample_sets(fit_rows, reference, fit_matrices)
            variants, configs, diagnostics = _fit_fold_variants(
                args,
                rosters,
                fit_rows,
                fit_matrices,
                reference,
                sample_sets,
                reweight_candidates,
                dufs_candidates,
            )
            results = {}
            for name, model in variants.items():
                summary, records = _evaluate_cached(
                    fit_rows,
                    test_rows,
                    model,
                    fit_matrices,
                    test_matrices,
                )
                results[name] = {
                    "metrics": summary,
                    "records": records,
                    "diagnostics": _jsonable(model.diagnostics),
                }
            payload = {
                "run_hash": run_hash,
                "repeat": repeat,
                "fold": fold,
                "fit_groups": len({row["_source_group"] for row in fit_rows}),
                "test_groups": len({row["_source_group"] for row in test_rows}),
                "seconds": time.perf_counter() - fold_started,
                "variant_configs": configs,
                "fit_diagnostics": diagnostics,
                "results": results,
            }
            _write_json(checkpoint, payload)
            payloads.append(payload)
            print(
                f"repeat={repeat} fold={fold} variants={len(variants)} "
                f"seconds={payload['seconds']:.1f}",
                flush=True,
            )

    aggregate = _aggregate_payloads(payloads, args.control, families)
    winner = str(aggregate["ranking"][0]["candidate"])
    _write_csv(args.out / "FOLD_METRICS.csv", aggregate["fold_metrics"])
    _write_csv(args.out / "REPEAT_METRICS.csv", aggregate["repeat_metrics"])
    _write_csv(args.out / "FAMILY_METRICS.csv", aggregate["family_metrics"])
    _write_csv(args.out / "AGGREGATE.csv", aggregate["aggregate"])
    _write_csv(args.out / "RANKING.csv", aggregate["ranking"])
    _write_csv(args.out / "STABILITY.csv", aggregate["stability"])
    _write_jsonl(args.out / "OOF_RECORDS.jsonl", aggregate["oof_records"])
    _write_json(args.out / "VARIANT_CONFIGS.json", aggregate["configs"])

    validation, validation_records = {}, []
    if validation_rows:
        validation_names = [winner]
        if args.control not in validation_names:
            validation_names.append(args.control)
        winner_base = str(aggregate["configs"][winner]["roster"])
        ordinary_name = _variant_name(winner_base, "ordinary", 0.0, 0.0)
        if ordinary_name in aggregate["configs"] and ordinary_name not in validation_names:
            validation_names.append(ordinary_name)
        validation, validation_records = _run_validation(
            args,
            development_rows,
            development_raw,
            validation_rows,
            rosters,
            aggregate["configs"],
            validation_names,
        )
        _write_json(args.out / "VALIDATION.json", validation)
        _write_jsonl(args.out / "VALIDATION_RECORDS.jsonl", validation_records)
    run_definition["elapsed_seconds"] = time.perf_counter() - started
    run_definition["winner"] = winner
    _write_json(args.out / "RUN_DEFINITION.json", run_definition)
    (args.out / "REPORT.md").write_text(
        _report(run_definition, aggregate, validation, winner, args.control)
    )
    print(f"winner={winner}")
    print(f"report={args.out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
