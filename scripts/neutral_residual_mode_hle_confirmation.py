#!/usr/bin/env python3
"""Frozen telemetry-only HLE confirmation for NRM-CS-IU v1."""

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


VERSION = "neutral-residual-mode-hle-confirmation-v1-2026-08-13"
CANDIDATE_VERSION = "neutral-residual-mode-cs-iu-v1-2026-08-13"
DEFAULT_OUT = REPO / "results" / "neutral_residual_mode_hle_v1"
DEFAULT_RAW = (
    REPO / "dataset_cache" / "four_localization" / "hle_full"
    / "raw_hle_T0.0.pkl"
)
DEFAULT_LABELS = (
    REPO / "local_cache" / "data_readiness"
    / "hle_codex_5p6_sol_xhigh.jsonl"
)
DEFAULT_JUDGE_MANIFEST = (
    REPO / "results" / "data_readiness_2026_08_11"
    / "hle_codex_5p6_sol_xhigh_manifest.json"
)
DEFAULT_CALIBRATION = (
    REPO / "results" / "neutral_residual_mode_cs_iu_v1"
    / "FROZEN_CALIBRATION.json"
)
TELEMETRY_KEYS = (
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
)
BOOTSTRAP_DRAWS = 20000


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


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def source_paths(raw, calibration):
    return {
        "script": Path(__file__),
        "spec": REPO / "SPEC_NEUTRAL_RESIDUAL_MODE_CS_IU_V1.md",
        "module": REPO / "spectral_utils" / "contribution_subspace.py",
        "calibration": Path(calibration),
        "raw_hle": Path(raw),
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


def telemetry_payload(candidate):
    payload = {name: candidate.get(name) for name in TELEMETRY_KEYS}
    if set(payload) != set(TELEMETRY_KEYS):
        raise RuntimeError("telemetry whitelist changed")
    return payload


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
    row_keys = sorted(cache, key=lambda value: int(value))
    telemetry = []
    for row_key in row_keys:
        candidates = cache[row_key].get("candidates")
        if not candidates:
            raise RuntimeError(f"missing candidate for HLE row {row_key}")
        telemetry.append(telemetry_payload(candidates[0]))
    F, feature_names, availability, contract = mixed_v2_matrix(telemetry)
    calibration = load_calibration(args.calibration)
    fitted = neutral_residual_mode_iu_fit(
        F, feature_names, calibration
    )
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
        "row_keys": np.asarray([int(value) for value in row_keys], dtype=int),
        "iu_correctness_score": fitted.neutral.baseline_score,
        "nrm_correctness_score": fitted.score,
        "cardinality_correctness_score": cardinality.score,
        "feature_names": np.asarray(feature_names),
        "families": np.asarray(fitted.contribution_space.families),
        "nrm_effective_weights": fitted.effective_weights,
        "nrm_intercept": np.asarray(fitted.intercept),
        "calibration_direction": calibration.direction,
    }
    if not all(np.isfinite(value).all() for key, value in scores.items()
               if key.endswith("score") or key.endswith("weights")):
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
        "n_rows": len(row_keys),
        "n_features": int(F.shape[0]),
        "feature_names": list(feature_names),
        "families": list(fitted.contribution_space.families),
        "telemetry_keys": list(TELEMETRY_KEYS),
        "score_payload_keys": sorted(scores),
        "labels_used": False,
        "target_fields_received_by_fusion": [],
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
    checks["candidate_version"] = (
        manifest.get("candidate_version") == CANDIDATE_VERSION
    )
    checks["labels_excluded"] = manifest.get("labels_used") is False
    checks["no_target_fields"] = not manifest.get(
        "target_fields_received_by_fusion"
    )
    if not all(checks.values()):
        raise RuntimeError(f"frozen score verification failed: {checks}")
    return manifest, checks


def paired_stratified_bootstrap(y, candidate, baseline):
    y = np.asarray(y, dtype=int)
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    seed = int(hashlib.sha256(
        f"{VERSION}:hle:nrm-vs-iu".encode()
    ).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    deltas = np.empty(BOOTSTRAP_DRAWS, dtype=float)
    for draw in range(BOOTSTRAP_DRAWS):
        indices = np.concatenate([
            rng.choice(positive, len(positive), replace=True),
            rng.choice(negative, len(negative), replace=True),
        ])
        labels = y[indices]
        deltas[draw] = (
            roc_auc_score(labels, candidate[indices])
            - roc_auc_score(labels, baseline[indices])
        )
    return {
        "draws": BOOTSTRAP_DRAWS,
        "seed": seed,
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
        iu = stored["iu_correctness_score"].astype(float)
        nrm = stored["nrm_correctness_score"].astype(float)
        cardinality = stored["cardinality_correctness_score"].astype(float)

    # This is the first point at which the frozen workflow reads target labels.
    labels = read_jsonl(args.labels)
    if [int(row["row_key"]) for row in labels] != row_keys.tolist():
        raise RuntimeError("HLE label sidecar does not align with frozen scores")
    y = np.asarray([row["correct"] == "yes" for row in labels], dtype=int)
    if not np.all(np.isin(y, (0, 1))) or len(np.unique(y)) != 2:
        raise RuntimeError("invalid HLE correctness labels")

    with Path(args.raw).open("rb") as handle:
        raw = pickle.load(handle)
    answer_types = np.asarray([
        str(raw[int(key)]["gold_row"].get("answer_type", "unknown"))
        for key in row_keys
    ])
    methods = {
        "iu": iu,
        "nrm": nrm,
        "cardinality": cardinality,
    }
    metrics = {
        name: {
            "auroc": float(roc_auc_score(y, score)),
            "auprc": float(average_precision_score(y, score)),
        }
        for name, score in methods.items()
    }
    delta_pp = 100 * (metrics["nrm"]["auroc"] - metrics["iu"]["auroc"])
    bootstrap = paired_stratified_bootstrap(y, nrm, iu)
    by_type = {}
    for answer_type in sorted(set(answer_types)):
        mask = answer_types == answer_type
        local_y = y[mask]
        if len(np.unique(local_y)) < 2:
            continue
        by_type[answer_type] = {
            "n": int(np.sum(mask)),
            "n_correct": int(np.sum(local_y)),
            "iu_auroc": float(roc_auc_score(local_y, iu[mask])),
            "nrm_auroc": float(roc_auc_score(local_y, nrm[mask])),
            "nrm_delta_pp": float(100 * (
                roc_auc_score(local_y, nrm[mask])
                - roc_auc_score(local_y, iu[mask])
            )),
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
        {"name": "positive HLE point delta", "pass": bool(delta_pp > 0)},
        {
            "name": "positive paired 95% lower bound",
            "pass": bool(bootstrap["low_pp"] > 0),
        },
    ]
    status = "PASS" if all(gate["pass"] for gate in gates) else "FAIL"
    result = {
        "version": VERSION,
        "candidate_version": CANDIDATE_VERSION,
        "status": status,
        "label_protocol": (
            "interim gpt-5.6-sol xhigh Codex judge; not paper-faithful HLE"
        ),
        "n": len(y),
        "n_correct": int(np.sum(y)),
        "metrics": metrics,
        "nrm_vs_iu_delta_pp": float(delta_pp),
        "paired_stratified_bootstrap": bootstrap,
        "by_answer_type": by_type,
        "gates": gates,
        "label_hashes": {
            "labels": sha256_file(args.labels),
            "judge_manifest": sha256_file(args.judge_manifest),
        },
    }
    write_json(args.out / "RESULT.json", result)

    def signed(value):
        return f"{float(value):+.3f}"

    lines = [
        "# Frozen NRM-CS-IU confirmation on HLE/Qwen2.5-72B",
        "",
        f"**Decision: {status}.** NRM changed correctness AUROC by "
        f"**{signed(delta_pp)}pp** versus IU; the paired stratified 95% "
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
    lines.extend([
        "",
        "## Pre-registered gates",
        "",
    ])
    for gate in gates:
        lines.append(
            f"- **{'PASS' if gate['pass'] else 'FAIL'} — {gate['name']}**"
        )
    lines.extend([
        "",
        "## Answer-type diagnostics",
        "",
        "| answer type | N | correct | IU AUROC | NRM AUROC | delta |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name, row in by_type.items():
        lines.append(
            f"| {name} | {row['n']} | {row['n_correct']} "
            f"| {row['iu_auroc']:.6f} | {row['nrm_auroc']:.6f} "
            f"| {signed(row['nrm_delta_pp'])}pp |"
        )
    lines.extend([
        "",
        "## Limitation",
        "",
        "HLE has only 68 judged-correct answers here.  Labels come from one "
        "interim Codex judge rather than HLE's original GPT-4o protocol, so "
        "this is an independent-example/model confirmation under the stated "
        "judge, not a paper-faithful HLE result.",
        "",
    ])
    (args.out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("score", "report", "verify"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--judge-manifest", type=Path, default=DEFAULT_JUDGE_MANIFEST
    )
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
