#!/usr/bin/env python3
"""Causal-prefix CIW-DEEM adapter for the frozen ProcessBench early lane."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectral_utils.ciw_deem import FROZEN_ARM
from spectral_utils.deem_b3_contract_ablation import PreparedArm, legacy_groups
from spectral_utils.deem_b3_unsupervised_input_gate import (
    GatedInnovationEnergy,
    fit_unsupervised_gate,
)
from spectral_utils.feature_contract import confidence_sign_vector
from spectral_utils.multitask_trajectory import truncate_row
from spectral_utils.reconstruction_benchmark.external_final_answer import CANONICAL_FEATURE_NAMES
from spectral_utils.reconstruction_benchmark.io import (
    atomic_write_json,
    atomic_write_npz,
    canonical_json_bytes,
    load_npz_no_pickle,
    sha256_file,
)
from spectral_utils.repgrid_scoring import _candidate_features, logprob_features_extended
from spectral_utils.residual_graph_deem import ContinuousDeemConfig, EPS


METHOD = "ciw_deem_causal_prefix_v1"
BUDGETS = (16, 32, 64, 128, 256, 512)
FAMILIES = ("gsm8k", "math", "olympiadbench", "omnimath")
SEEDS = (0, 1, 2, 3, 4)
REFERENCES = (
    "unified28", "iu28_no_length", "step272_two_head_global_local_w0p50_peak"
)


def _payload_sha(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _raw_features(row: Mapping[str, Any], budget: int) -> np.ndarray:
    truncated = truncate_row(row, budget)
    features = _candidate_features(truncated, allow_short=False)
    features.update(logprob_features_extended(truncated.get("top_k_logprobs")))
    values = np.asarray([features.get(name, np.nan) for name in CANONICAL_FEATURE_NAMES], dtype=float)
    if values.shape != (30,) or not np.isfinite(values).all():
        missing = [name for name, value in zip(CANONICAL_FEATURE_NAMES, values) if not np.isfinite(value)]
        raise RuntimeError(f"prefix feature extraction failed at b={budget}: {missing}")
    return values


def _d1_fit_apply(train_raw: np.ndarray, held_raw: np.ndarray) -> tuple[PreparedArm, PreparedArm]:
    names = tuple(CANONICAL_FEATURE_NAMES)
    lookup = {name: i for i, name in enumerate(names)}

    def transform_raw(X: np.ndarray) -> tuple[np.ndarray, tuple[str, ...]]:
        h15 = X[:, lookup["epr"]]
        hsaved = X[:, lookup["mean_logprob_entropy"]]
        columns, new_names = [], []
        for name in names:
            if name == "epr":
                columns.append(0.5 * (h15 + hsaved)); new_names.append("entropy_common")
            elif name == "mean_logprob_entropy":
                columns.append(hsaved - h15); new_names.append("entropy_support_delta")
            else:
                columns.append(X[:, lookup[name]]); new_names.append(name)
        return np.column_stack(columns), tuple(new_names)

    train_y, new_names = transform_raw(train_raw)
    held_y, held_names = transform_raw(held_raw)
    if held_names != new_names:
        raise AssertionError("D1 feature order drifted")
    mean, scale = train_y.mean(axis=0), train_y.std(axis=0)
    scale = np.where(scale > EPS, scale, 1.0)
    common_i = new_names.index("entropy_common")
    scale[new_names.index("entropy_support_delta")] = scale[common_i]

    def risk(Y: np.ndarray) -> np.ndarray:
        output = np.empty_like(Y)
        for index, name in enumerate(new_names):
            z = (Y[:, index] - mean[index]) / scale[index]
            if name in {"entropy_common", "entropy_support_delta"}:
                output[:, index] = z
            else:
                output[:, index] = -z * float(confidence_sign_vector((name,))[0])
        return output

    legacy_names = tuple(
        "epr" if name == "entropy_common" else
        "mean_logprob_entropy" if name == "entropy_support_delta" else name
        for name in new_names
    )
    groups = legacy_groups(legacy_names)
    exclusions = frozenset({"entropy_support_delta"})
    return (
        PreparedArm(risk(train_y), new_names, groups, exclusions, mean, scale),
        PreparedArm(risk(held_y), new_names, groups, exclusions, mean, scale),
    )


def _restore_model(prepared: PreparedArm, result: Any, gate_map: Any, seed: int) -> GatedInnovationEnergy:
    import torch
    model = GatedInnovationEnergy(prepared, ContinuousDeemConfig(), seed, gate_map, FROZEN_ARM)
    state = result.state
    with torch.no_grad():
        model.a.copy_(torch.as_tensor(state["a"], dtype=torch.float64))
        model.b.copy_(torch.as_tensor(state["b"], dtype=torch.float64))
        for prefix, values in (("w", model.w), ("W", model.W), ("d", model.d), ("V", model.V), ("e", model.e)):
            for family, parameter in values.items():
                parameter.copy_(torch.as_tensor(state[f"{prefix}::{family}"], dtype=torch.float64))
    return model


def _fit_family(task: tuple[str, list[dict[str, Any]]]) -> dict[str, Any]:
    family, rows = task
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass
    output: dict[tuple[str, int], float] = {}
    audits = []
    for budget in BUDGETS:
        donors = [row for row in rows if row["partition"] == "calibration" and len(row["token_entropies"]) > budget]
        held = [row for row in rows if row["partition"] == "evaluation" and len(row["token_entropies"]) > budget]
        train_raw = np.vstack([_raw_features(row, budget) for row in donors])
        held_raw = np.vstack([_raw_features(row, budget) for row in held])
        prepared, held_prepared = _d1_fit_apply(train_raw, held_raw)
        group_ids = [str(row["source_question_id"]) for row in donors]
        lengths = np.full(len(donors), float(budget))
        seed_scores = []
        seed_health = []
        reliability = None
        for seed in SEEDS:
            result, gate_map = fit_unsupervised_gate(
                prepared, group_ids, lengths, FROZEN_ARM,
                seed=seed, config=ContinuousDeemConfig(),
            )
            if not result.health.get("healthy", False):
                raise RuntimeError(f"{family}/b{budget}/s{seed}: unhealthy CIW")
            model = _restore_model(prepared, result, gate_map, seed)
            import torch
            with torch.no_grad():
                logit, _atomic, _families = model.logit(
                    torch.as_tensor(held_prepared.X_risk, dtype=torch.float64)
                )
            aligned = result.orientation * logit.numpy()
            score = 1.0 / (1.0 + np.exp(-np.clip(aligned, -700, 700)))
            seed_scores.append(score)
            seed_health.append(dict(result.health))
            if reliability is None:
                reliability = gate_map.reliability.copy()
            elif not np.array_equal(reliability, gate_map.reliability):
                raise RuntimeError("CIW prefix reliability changed across seeds")
        score = np.mean(np.stack(seed_scores, axis=1), axis=1)
        for row, value in zip(held, score):
            output[(str(row["row_id"]), budget)] = float(value)
        audits.append({
            "family": family, "budget": budget, "n_calibration": len(donors),
            "n_evaluation": len(held), "mean_reliability": float(np.mean(reliability)),
            "health": seed_health,
        })
    return {
        "status": "COMPLETE", "family": family,
        "scores": [(row_id, budget, value) for (row_id, budget), value in sorted(output.items())],
        "audits": audits,
    }


def fit(prefix_release: Path, out: Path, jobs: int) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    fit_input_path = prefix_release / "prefix/A/inputs/FIT_INPUT.pkl"
    expected_path = prefix_release / "prefix/A/inputs/EXPECTED_SCORES.npz"
    with fit_input_path.open("rb") as handle:
        fit_input = pickle.load(handle)
    if fit_input.get("target_fields_present") is not False:
        raise RuntimeError("prefix fit input is not target-free")
    expected = load_npz_no_pickle(expected_path)
    tasks = [(family, list(fit_input["rows_by_family"][family])) for family in FAMILIES]
    results = []
    with ProcessPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(_fit_family, task): task[0] for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"completed prefix family {result['family']} ({len(results)}/4)", flush=True)
    score_map = {
        (row_id, int(budget)): float(value)
        for result in results for row_id, budget, value in result["scores"]
    }
    keys = list(zip(expected["row_id"].astype(str), expected["budget"].astype(int)))
    if set(score_map) != set(keys):
        raise RuntimeError(f"prefix CIW roster mismatch: scores={len(score_map)}, expected={len(keys)}")
    score_path = out / "SCORES.npz"
    score_sha = atomic_write_npz(score_path, {
        "row_id": np.asarray(expected["row_id"]),
        "family": np.asarray(expected["family"]),
        "budget": np.asarray(expected["budget"], dtype="<i2"),
        METHOD: np.asarray([score_map[key] for key in keys], dtype="<f8"),
    })
    freeze = {
        "schema_version": "ciw-prefix-score-freeze-v1",
        "method_id": METHOD,
        "n_observations": len(keys),
        "budgets": list(BUDGETS),
        "families": list(FAMILIES),
        "seeds": list(SEEDS),
        "causal_rule": "all feature streams are truncated to the registered budget",
        "fit_scope": "family-and-budget calibration partition only",
        "labels_opened_during_fit": False,
        "targets_accessed_during_fit": False,
        "fit_input_sha256": sha256_file(fit_input_path),
        "expected_roster_sha256": sha256_file(expected_path),
        "score_sha256": score_sha,
        "audits": sorted([audit for result in results for audit in result["audits"]], key=lambda x: (x["family"], x["budget"])),
    }
    freeze["payload_sha256"] = _payload_sha(freeze)
    atomic_write_json(out / "SCORE_FREEZE_MANIFEST.json", freeze)
    print(json.dumps({"status": "PASS", "stage": "fit", "observations": len(keys)}, indent=2))


def evaluate(prefix_release: Path, fit_root: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    freeze_path = fit_root / "SCORE_FREEZE_MANIFEST.json"
    freeze = json.loads(freeze_path.read_text())
    body = dict(freeze); digest = body.pop("payload_sha256", None)
    if digest != _payload_sha(body) or not (
        freeze.get("labels_opened_during_fit") is False
        and freeze.get("targets_accessed_during_fit") is False
        and freeze.get("n_observations") == 9277
    ):
        raise RuntimeError("CIW prefix freeze failed")
    score_path = fit_root / "SCORES.npz"
    if sha256_file(score_path) != freeze["score_sha256"]:
        raise RuntimeError("CIW prefix scores changed")
    ciw = load_npz_no_pickle(score_path)

    # Labels are opened only after the complete CIW score freeze is verified.
    labeled_path = prefix_release / "prefix/A/evaluation/LABELED_SCORES.npz"
    labeled = load_npz_no_pickle(labeled_path)
    for key in ("row_id", "family", "budget"):
        if not np.array_equal(ciw[key], labeled[key]):
            raise RuntimeError(f"prefix labeled roster differs: {key}")
    y = np.asarray(labeled["label"], dtype=np.int8)
    methods = (METHOD,) + REFERENCES
    rows = []
    for budget in BUDGETS:
        mask_budget = np.asarray(labeled["budget"], dtype=int) == budget
        for method in methods:
            scores = np.asarray(ciw[METHOD] if method == METHOD else labeled[method], dtype=float)
            by_family = []
            for family in FAMILIES:
                mask = mask_budget & (labeled["family"].astype(str) == family)
                by_family.append((
                    roc_auc_score(y[mask], scores[mask]),
                    average_precision_score(y[mask], scores[mask]),
                ))
            rows.append({
                "budget": budget, "method_id": method,
                "equal_subset_auroc": float(np.mean([value[0] for value in by_family])),
                "equal_subset_auprc": float(np.mean([value[1] for value in by_family])),
                "n": int(mask_budget.sum()),
            })
    import csv
    with (out / "METRICS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    lines = [
        "# CIW-DEEM causal early detection", "",
        "Each CIW model is fitted without labels on calibration prefixes and applied to held evaluation prefixes at the same token budget.", "",
        "| Budget | CIW AUROC | B3 unavailable | IU28 AUROC | Unified28 AUROC | Step272 AUROC | CIW AUPRC |",
        "|---:|---:|:---:|---:|---:|---:|---:|",
    ]
    for budget in BUDGETS:
        values = {row["method_id"]: row for row in rows if row["budget"] == budget}
        lines.append(
            f"| {budget} | {values[METHOD]['equal_subset_auroc']:.6f} | yes | "
            f"{values['iu28_no_length']['equal_subset_auroc']:.6f} | "
            f"{values['unified28']['equal_subset_auroc']:.6f} | "
            f"{values['step272_two_head_global_local_w0p50_peak']['equal_subset_auroc']:.6f} | "
            f"{values[METHOD]['equal_subset_auprc']:.6f} |"
        )
    lines += ["", "B3 has no registered causal-prefix model, so it is not a valid comparator in this lane."]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "ciw-prefix-evaluation-v1",
        "scores_verified_before_labels": True,
        "fit_freeze_sha256": sha256_file(freeze_path),
        "outputs": {name: sha256_file(out / name) for name in ("METRICS.csv", "REPORT.md")},
    }
    manifest["payload_sha256"] = _payload_sha(manifest)
    atomic_write_json(out / "EVALUATION_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", "stage": "evaluate", "rows": len(rows)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    fit_p = sub.add_parser("fit")
    fit_p.add_argument("--prefix-release", required=True)
    fit_p.add_argument("--out-dir", required=True)
    fit_p.add_argument("--jobs", type=int, default=4)
    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--prefix-release", required=True)
    eval_p.add_argument("--fit-dir", required=True)
    eval_p.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.stage == "fit":
        fit(Path(args.prefix_release).resolve(), Path(args.out_dir).resolve(), args.jobs)
    else:
        evaluate(Path(args.prefix_release).resolve(), Path(args.fit_dir).resolve(), Path(args.out_dir).resolve())


if __name__ == "__main__":
    main()
