#!/usr/bin/env python3
"""Frozen telemetry-only PRMBench confirmation for NRM-CS-IU v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.leverage_balanced_processbench_transfer import (  # noqa: E402
    mixed_v2_matrix,
)
from spectral_utils.contribution_subspace import (  # noqa: E402
    NeutralResidualModeCalibration,
    cardinality_balanced_contribution_score,
    neutral_residual_mode_iu_fit,
)


VERSION = "neutral-residual-mode-prmbench-confirmation-v1-2026-08-13"
CANDIDATE_VERSION = "neutral-residual-mode-cs-iu-v1-2026-08-13"
DEFAULT_OUT = REPO / "results" / "neutral_residual_mode_prmbench_v1"
DEFAULT_RAW = (
    REPO / "dataset_cache" / "four_localization"
    / "prmbench_qwen3_8b_telemetry_full" / "prmbench_telemetry.pkl"
)
DEFAULT_CALIBRATION = (
    REPO / "results" / "neutral_residual_mode_cs_iu_v1"
    / "FROZEN_CALIBRATION.json"
)
BAD_IDS = frozenset({
    "confidence_confidence_prm_train_p1_303",
    "deception_deception_prm_test_p1_87",
    "step_contradiction_step_contradiction_prm_test_p2_991",
})
TELEMETRY_KEYS = (
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
)
BOOTSTRAP_DRAWS = 5000


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def source_paths(raw, calibration):
    return {
        "script": Path(__file__),
        "spec": (
            REPO / "SPEC_NEUTRAL_RESIDUAL_MODE_PRMBENCH_CONFIRMATION_V1.md"
        ),
        "module": REPO / "spectral_utils" / "contribution_subspace.py",
        "calibration": Path(calibration),
        "raw_prmbench": Path(raw),
    }


def load_calibration(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != CANDIDATE_VERSION:
        raise RuntimeError("unexpected frozen calibration version")
    if payload.get("uses_labels") is not False:
        raise RuntimeError("calibration is not declared label-free")
    return NeutralResidualModeCalibration(
        families=tuple(payload["families"]),
        direction=np.asarray(payload["direction"], dtype=float),
        residual_covariance=np.asarray(
            payload["residual_covariance"], dtype=float
        ),
        pair_counts=np.asarray(payload["pair_counts"], dtype=int),
        eigenvalues=np.asarray(payload["eigenvalues"], dtype=float),
        selected_index=int(payload["selected_index"]),
        diagnostics=dict(payload["diagnostics"]),
    )


def telemetry_payload(row):
    payload = {name: row.get(name) for name in TELEMETRY_KEYS}
    if set(payload) != set(TELEMETRY_KEYS):
        raise RuntimeError("telemetry whitelist changed")
    return payload


def ordered_eligible_rows(cache):
    selected, excluded = [], set()
    for row_key in sorted(cache, key=lambda value: int(value)):
        row = cache[row_key]
        row_id = str(row["idx"])
        if row_id in BAD_IDS:
            excluded.add(row_id)
            continue
        selected.append((int(row_key), row_id, str(row["source_idx"]), row))
    if excluded != BAD_IDS:
        raise RuntimeError(
            f"fixed PRMBench exclusions do not match the cache: {excluded}"
        )
    if len(selected) != 6966:
        raise RuntimeError(f"unexpected eligible PRMBench count: {len(selected)}")
    return selected


def score_phase(args):
    args.out.mkdir(parents=True, exist_ok=True)
    score_path = args.out / "FROZEN_SCORES.npz"
    manifest_path = args.out / "FIT_MANIFEST.json"
    if score_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "frozen score artifacts already exist; use a new output directory"
        )
    with Path(args.raw).open("rb") as handle:
        cache = pickle.load(handle)
    selected = ordered_eligible_rows(cache)
    telemetry = [telemetry_payload(row) for _, _, _, row in selected]
    F, feature_names, availability, contract = mixed_v2_matrix(telemetry)
    calibration = load_calibration(args.calibration)
    fitted = neutral_residual_mode_iu_fit(F, feature_names, calibration)
    cardinality = cardinality_balanced_contribution_score(
        fitted.contribution_space, fitted.baseline.w
    )
    if not np.allclose(
        fitted.neutral.baseline_score,
        cardinality.baseline_score,
        atol=1e-12,
    ):
        raise RuntimeError("IU baseline mismatch between frozen methods")
    scores = {
        "row_keys": np.asarray([row[0] for row in selected], dtype=int),
        "row_ids": np.asarray([row[1] for row in selected]),
        "source_ids": np.asarray([row[2] for row in selected]),
        "iu_correctness_score": fitted.neutral.baseline_score,
        "nrm_correctness_score": fitted.score,
        "cardinality_correctness_score": cardinality.score,
        "feature_names": np.asarray(feature_names),
        "families": np.asarray(fitted.contribution_space.families),
        "nrm_effective_weights": fitted.effective_weights,
        "nrm_intercept": np.asarray(fitted.intercept),
        "calibration_direction": calibration.direction,
    }
    numeric = [
        value for key, value in scores.items()
        if key.endswith("score") or key.endswith("weights")
    ]
    if not all(np.isfinite(value).all() for value in numeric):
        raise RuntimeError("non-finite frozen score or weight")
    np.savez_compressed(score_path, **scores)
    hashes = {
        name: sha256_file(path)
        for name, path in source_paths(args.raw, args.calibration).items()
    }
    hashes["scores"] = sha256_file(score_path)
    manifest = {
        "version": VERSION,
        "candidate_version": CANDIDATE_VERSION,
        "phase": "telemetry_only_score_fit",
        "hashes": hashes,
        "n_raw_rows": len(cache),
        "n_rows": len(selected),
        "excluded_ids": sorted(BAD_IDS),
        "n_features": int(F.shape[0]),
        "feature_names": list(feature_names),
        "families": list(fitted.contribution_space.families),
        "telemetry_keys": list(TELEMETRY_KEYS),
        "score_payload_keys": sorted(scores),
        "labels_used": False,
        "target_fields_received_by_fusion": [],
        "forbidden_target_fields": [
            "classification", "category", "error_steps"
        ],
        "availability": availability,
        "feature_contract": contract,
        "calibration_diagnostics": calibration.diagnostics,
        "nrm_diagnostics": fitted.neutral.diagnostics,
        "cardinality_diagnostics": cardinality.diagnostics,
        "decomposition_reconstruction_error": (
            fitted.contribution_space.diagnostics["reconstruction_error"]
        ),
    }
    write_json(manifest_path, manifest)
    print(json.dumps({
        "phase": manifest["phase"],
        "n_rows": manifest["n_rows"],
        "n_features": manifest["n_features"],
        "families": manifest["families"],
        "excluded_ids": manifest["excluded_ids"],
        "labels_used": False,
        "scores_sha256": hashes["scores"],
    }, indent=2, allow_nan=False))


def verify_frozen(args):
    manifest_path = args.out / "FIT_MANIFEST.json"
    score_path = args.out / "FROZEN_SCORES.npz"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_paths = source_paths(args.raw, args.calibration)
    expected_paths["scores"] = score_path
    actual = {name: sha256_file(path) for name, path in expected_paths.items()}
    checks = {
        name: actual[name] == manifest["hashes"].get(name)
        for name in actual
    }
    checks.update({
        "candidate_version": (
            manifest.get("candidate_version") == CANDIDATE_VERSION
        ),
        "fixed_exclusions": set(manifest.get("excluded_ids", [])) == BAD_IDS,
        "labels_excluded": manifest.get("labels_used") is False,
        "no_target_fields": not manifest.get(
            "target_fields_received_by_fusion"
        ),
    })
    if not all(checks.values()):
        raise RuntimeError(f"frozen score verification failed: {checks}")
    return manifest, checks


def paired_group_bootstrap(y, candidate, baseline, source_ids):
    y = np.asarray(y, dtype=int)
    source_ids = np.asarray(source_ids).astype(str)
    unique, group_index = np.unique(source_ids, return_inverse=True)
    group_count = len(unique)
    seed = int(hashlib.sha256(
        f"{VERSION}:source-group:nrm-vs-iu".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    probability = np.full(group_count, 1.0 / group_count)
    for draw in range(BOOTSTRAP_DRAWS):
        counts = rng.multinomial(group_count, probability)
        weights = counts[group_index]
        deltas[draw] = (
            roc_auc_score(y, candidate, sample_weight=weights)
            - roc_auc_score(y, baseline, sample_weight=weights)
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
        "n_source_groups": group_count,
        "low_pp": float(100 * np.quantile(deltas, 0.025)),
        "median_pp": float(100 * np.quantile(deltas, 0.5)),
        "high_pp": float(100 * np.quantile(deltas, 0.975)),
        "probability_positive": float(np.mean(deltas > 0)),
    }


def report_phase(args):
    manifest, hash_checks = verify_frozen(args)
    score_path = args.out / "FROZEN_SCORES.npz"
    with np.load(score_path, allow_pickle=False) as stored:
        row_keys = stored["row_keys"].astype(int)
        row_ids = stored["row_ids"].astype(str)
        source_ids = stored["source_ids"].astype(str)
        iu = stored["iu_correctness_score"].astype(float)
        nrm = stored["nrm_correctness_score"].astype(float)
        cardinality = stored["cardinality_correctness_score"].astype(float)

    # This is the first point at which the frozen workflow reads target fields.
    with Path(args.raw).open("rb") as handle:
        cache = pickle.load(handle)
    selected = ordered_eligible_rows(cache)
    if [row[0] for row in selected] != row_keys.tolist():
        raise RuntimeError("row keys do not align with frozen scores")
    if [row[1] for row in selected] != row_ids.tolist():
        raise RuntimeError("row IDs do not align with frozen scores")
    if [row[2] for row in selected] != source_ids.tolist():
        raise RuntimeError("source IDs do not align with frozen scores")
    classifications = np.asarray([
        str(row[3]["classification"]) for row in selected
    ])
    y = (classifications == "correct").astype(int)
    if len(np.unique(y)) != 2:
        raise RuntimeError("invalid PRMBench correctness target")

    methods = {"iu": iu, "nrm": nrm, "cardinality": cardinality}
    metrics = {
        name: {
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
        }
        for name, score in methods.items()
    }
    delta_pp = 100 * (metrics["nrm"]["auroc"] - metrics["iu"]["auroc"])
    bootstrap = paired_group_bootstrap(y, nrm, iu, source_ids)
    by_error_class = {}
    correct = classifications == "correct"
    for error_class in sorted(set(classifications) - {"correct"}):
        mask = correct | (classifications == error_class)
        local_y = y[mask]
        iu_auc = float(roc_auc_score(local_y, iu[mask]))
        nrm_auc = float(roc_auc_score(local_y, nrm[mask]))
        by_error_class[error_class] = {
            "n": int(np.sum(mask)),
            "n_error": int(np.sum(classifications == error_class)),
            "iu_auroc": iu_auc,
            "nrm_auroc": nrm_auc,
            "nrm_delta_pp": float(100 * (nrm_auc - iu_auc)),
        }

    numerical_pass = bool(
        manifest["decomposition_reconstruction_error"] < 1e-10
        and abs(manifest["nrm_diagnostics"][
            "baseline_correction_covariance"
        ]) < 1e-10
        and manifest["nrm_diagnostics"][
            "weight_reconstruction_error"
        ] < 1e-10
        and all(np.isfinite(score).all() for score in methods.values())
    )
    gates = [
        {
            "name": "frozen score/source hashes verify",
            "pass": bool(all(hash_checks.values())),
        },
        {
            "name": "telemetry-only fit payload",
            "pass": bool(
                manifest["labels_used"] is False
                and not manifest["target_fields_received_by_fusion"]
            ),
        },
        {"name": "numerical invariants", "pass": numerical_pass},
        {
            "name": "positive overall PRMBench point delta",
            "pass": bool(delta_pp > 0),
        },
        {
            "name": "positive source-group 95% lower bound",
            "pass": bool(bootstrap["low_pp"] > 0),
        },
    ]
    status = "PASS" if all(gate["pass"] for gate in gates) else "FAIL"
    result = {
        "version": VERSION,
        "candidate_version": CANDIDATE_VERSION,
        "status": status,
        "prediction_unit": "complete reasoning response",
        "label_protocol": "official PRMBench response classification",
        "n": len(y),
        "n_correct": int(np.sum(y)),
        "n_error": int(np.sum(y == 0)),
        "metrics": metrics,
        "nrm_vs_iu_delta_pp": float(delta_pp),
        "paired_source_group_bootstrap": bootstrap,
        "by_error_class": by_error_class,
        "gates": gates,
        "raw_hash": sha256_file(args.raw),
    }
    write_json(args.out / "RESULT.json", result)

    def signed(value):
        return f"{float(value):+.3f}"

    lines = [
        "# Frozen NRM-CS-IU confirmation on PRMBench/Qwen3-8B",
        "",
        f"**Decision: {status}.** NRM changed response-level correctness AUROC "
        f"by **{signed(delta_pp)}pp** versus IU; the paired source-group 95% "
        f"interval is [{signed(bootstrap['low_pp'])}, "
        f"{signed(bootstrap['high_pp'])}]pp.",
        "",
        "| method | AUROC | AUPRC |",
        "|---|---:|---:|",
    ]
    for name in ("iu", "nrm", "cardinality"):
        lines.append(
            f"| `{name}` | {metrics[name]['auroc']:.6f} "
            f"| {metrics[name]['auprc']:.6f} |"
        )
    lines.extend(["", "## Pre-registered gates", ""])
    for gate in gates:
        lines.append(
            f"- **{'PASS' if gate['pass'] else 'FAIL'} — {gate['name']}**"
        )
    lines.extend([
        "",
        "## Error-class diagnostics",
        "",
        "| error class vs correct | error N | IU AUROC | NRM AUROC | delta |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, row in by_error_class.items():
        lines.append(
            f"| {name} | {row['n_error']} | {row['iu_auroc']:.6f} "
            f"| {row['nrm_auroc']:.6f} "
            f"| {signed(row['nrm_delta_pp'])}pp |"
        )
    lines.extend([
        "",
        "## Scope",
        "",
        "This is a response-level correct-versus-error adaptation.  It is not "
        "PRMBench's official step-level metric.  Exactly the three readiness-"
        "identified alignment defects were excluded before scoring.",
        "",
    ])
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("score", "report", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--calibration", type=Path, default=DEFAULT_CALIBRATION
    )
    args = parser.parse_args()
    if args.phase == "score":
        score_phase(args)
    elif args.phase == "verify":
        _, checks = verify_frozen(args)
        print(json.dumps(checks, indent=2, sort_keys=True))
    else:
        report_phase(args)


if __name__ == "__main__":
    main()
