#!/usr/bin/env python3
"""Run the frozen ProcessBench IU-PCR latent-state localization experiment.

The command is deliberately split into two processes:

``fit``
    Reads only token telemetry fields, fits every label-free score, writes
    numeric score arrays, and freezes their SHA-256 hashes.  It never reads a
    row's ``label`` or ``step_token_spans`` value.

``evaluate``
    Requires the frozen artifacts, verifies every hash, and only then reads
    ProcessBench labels and step spans.  It maps frozen token predictions to
    steps and applies the repository's exact 100 repeated 50/50 calibration
    protocol through ``evaluate_two_stage``.

This is an exploratory comparison because the ProcessBench labels have already
been inspected by earlier project experiments.  No result produced here may be
used to tune another variant on the same eight cells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
GL_ROOT = ROOT / "scripts" / "gl_liu_v1"
GL_LOC = GL_ROOT / "localization"
sys.path[:0] = [str(ROOT), str(GL_ROOT), str(GL_LOC)]

# Import the frozen factorial implementation first.  Its GL-LIU dependency
# installs the repository's local spectral_utils package path explicitly.
from scripts.gl_liu_factorial_v2.run import (  # noqa: E402
    CORE_TOKEN_VIEWS,
    LOCAL_LAMBDA,
    _apply_arm,
    _fit_one_arm,
    _fit_row_order,
    _prepare_token_matrix,
)
from scripts.gl_liu_v1.run import fit_answer_mixed  # noqa: E402
from scripts.gl_liu_v1.two_stage_localization import (  # noqa: E402
    best_threshold,
    evaluate_two_stage,
)
from evidence_drop import EVIDENCE_FNS, evidence_drop_risk  # noqa: E402
from localization_metrics import NO_ERROR, processbench_f1, sla, step_drop_scores  # noqa: E402
from spectral_utils.laplacian_upcr import IU_FIT_DEFAULTS  # noqa: E402
from spectral_utils.latent_state_localizer import (  # noqa: E402
    DEFAULT_MIN_SEED_LOCATOR_AGREEMENT,
    DEFAULT_SEEDS,
    apply_latent_state_fit,
    fit_upcr_initialized_hmm,
    model_to_dict,
)
from spectral_utils.streaming_utils import anchor_orient  # noqa: E402
from spectral_utils.token_feature_views import build_token_channels  # noqa: E402
from spectral_utils.upcr import upcr_fit  # noqa: E402


MODELS = ("qwen3_4b", "qwen3_8b")
SUBSETS = ("gsm8k", "math", "olympiadbench", "omnimath")
DEV = {("qwen3_4b", "gsm8k"), ("qwen3_4b", "math")}
N_SPLITS = 100
SPLIT_SEED = 0
ERROR_ALIGNMENT_RADIUS = 50
VERSION = "processbench-latent-state-v1-2026-08-11"

SOURCE_FILES = (
    "scripts/processbench_latent_state_v1/run.py",
    "scripts/processbench_latent_state_v1/report.py",
    "spectral_utils/latent_state_localizer.py",
    "scripts/gl_liu_factorial_v2/run.py",
    "scripts/gl_liu_v1/run.py",
    "scripts/gl_liu_v1/evaluate_answer_level.py",
    "scripts/gl_liu_v1/optimize_localization.py",
    "scripts/gl_liu_v1/two_stage_localization.py",
    "scripts/gl_liu_v1/localization/evidence_drop.py",
    "scripts/gl_liu_v1/localization/localization_metrics.py",
    "scripts/gl_liu_v1/localization/positional_views.py",
    "spectral_utils/adapted_dufs.py",
    "spectral_utils/dufs_liu_feature_contract.py",
    "spectral_utils/feature_contract.py",
    "spectral_utils/feature_utils.py",
    "spectral_utils/fusion_utils.py",
    "spectral_utils/laplacian_upcr.py",
    "spectral_utils/repgrid_scoring.py",
    "spectral_utils/selectors/a2_groupfs.py",
    "spectral_utils/streaming_utils.py",
    "spectral_utils/token_feature_views.py",
    "spectral_utils/upcr.py",
)

LOCAL_METHODS = (
    "local_iu_core",
    "local_temporal_core",
    "local_dufs_core",
    "local_hmm_reversible_core_iu",
    "local_hmm_absorbing_core_iu",
)
HMM_METHODS = {
    "reversible": "local_hmm_reversible_core_iu",
    "absorbing": "local_hmm_absorbing_core_iu",
}

# Only these values are copied into the fitting view.  In particular, the fit
# functions cannot see labels, final-answer correctness, text, or step spans.
TELEMETRY_KEYS = (
    "gen_token_ids",
    "token_entropies",
    "token_spilled_energies",
    "token_logsumexp",
    "top_k_logprobs",
)
FORBIDDEN_FIT_KEYS = {
    "label", "labels", "final_answer_correct", "step_token_spans",
    "steps", "problem", "solution", "gold",
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(_jsonable(value), handle, indent=2, sort_keys=True)


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_float(values) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _hash_int(values) -> str:
    array = np.asarray(values, dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _hash_text(values) -> str:
    payload = "\n".join(str(value) for value in values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def flatten_curves(curves):
    curves = [np.asarray(curve, dtype=float) for curve in curves]
    lengths = np.asarray([len(curve) for curve in curves], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    flat = np.concatenate(curves).astype(float) if curves else np.empty(0, dtype=float)
    return flat, offsets


def unflatten_curves(flat, offsets):
    flat = np.asarray(flat, dtype=float)
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 1 or offsets[0] != 0 or offsets[-1] != len(flat):
        raise ValueError("invalid flattened-curve offsets")
    if np.any(np.diff(offsets) <= 0):
        raise ValueError("every trace must contain at least one token")
    return [flat[offsets[index]:offsets[index + 1]] for index in range(len(offsets) - 1)]


def load_raw_rows(path):
    """Load trusted local cache rows without accessing label values."""
    with Path(path).open("rb") as handle:
        cache = pickle.load(handle)
    output = []
    for key in sorted(cache):
        row = cache[key]
        if row.get("align_diag", {}).get("problems"):
            continue
        output.append((key, row))
    if not output:
        raise ValueError(f"no aligned rows in {path}")
    return output


def telemetry_view(raw_rows):
    """Copy only registered telemetry keys; labels are structurally absent."""
    telemetry, row_ids = [], []
    for cache_key, row in raw_rows:
        missing = [key for key in TELEMETRY_KEYS if row.get(key) is None]
        if missing:
            raise KeyError(f"row {cache_key!r} is missing telemetry: {missing}")
        fit_row = {key: row[key] for key in TELEMETRY_KEYS}
        if FORBIDDEN_FIT_KEYS.intersection(fit_row):
            raise RuntimeError("evaluation-only field leaked into a fitting row")
        length = len(fit_row["token_entropies"])
        if length < 2:
            raise ValueError(f"row {cache_key!r} has fewer than two token observations")
        for key in ("gen_token_ids", "token_spilled_energies", "token_logsumexp"):
            if len(fit_row[key]) != length:
                raise ValueError(f"row {cache_key!r} has a misaligned {key}")
        topk = fit_row["top_k_logprobs"]
        if np.asarray(topk["logprobs"]).shape[0] != length:
            raise ValueError(f"row {cache_key!r} has misaligned top-k telemetry")
        telemetry.append(fit_row)
        row_ids.append(str(row.get("id", cache_key)))
    return telemetry, row_ids


def _ordinary_iu_arm(prepared):
    """Exact lambda=0 anchor used by the frozen LIU implementation."""
    fitted = upcr_fit(prepared["F"], **IU_FIT_DEFAULTS)
    anchor_index = prepared["names"].index("entropy_series")
    anchor = prepared["V"][:, anchor_index]
    _, flipped = anchor_orient(fitted.w @ prepared["F"], anchor)
    return {
        "names": prepared["names"],
        "mu": prepared["mu"],
        "sd": prepared["sd"],
        "derived": prepared["derived"],
        "weights": np.asarray(fitted.w, dtype=float),
        "flipped": bool(flipped),
    }, fitted


def _effective_rank(matrix):
    singular = np.linalg.svd(np.asarray(matrix, dtype=float), compute_uv=False)
    mass = singular * singular
    return float((mass.sum() ** 2) / (np.sum(mass * mass) + 1e-12))


def _mindgap_label_free(telemetry):
    evidence, detector = [], []
    for row in telemetry:
        curve = np.asarray(EVIDENCE_FNS["shannon"](row, 20), dtype=float)
        evidence.append(curve)
        detector.append(evidence_drop_risk(curve, M=5, ema_span=5))
    return evidence, np.asarray(detector, dtype=float)


def _reference_hash_check(model, subset, score_hashes, reference_root):
    path = Path(reference_root) / f"{model}__{subset}.json"
    if not path.exists():
        return {"available": False, "path": str(path), "passed": None}
    reference = json.load(path.open())["hashes_before_labels"]
    expected = {
        "global_mixed_v2_dufs": reference["detectors"]["global_dufs"],
        "local_temporal_core": reference["token_curves"]["local_temporal_core"],
        "local_dufs_core": reference["token_curves"]["local_dufs_core"],
        "mindgap_detector": reference["mindgap_detector"],
    }
    mismatches = {
        key: {"expected": expected[key], "observed": score_hashes[key]}
        for key in expected if score_hashes.get(key) != expected[key]
    }
    if mismatches:
        raise RuntimeError(
            f"frozen baseline reproduction failed for {model}/{subset}: {mismatches}"
        )
    return {
        "available": True,
        "path": str(path),
        "path_sha256": sha256_file(path),
        "passed": True,
        "expected": expected,
        "evaluation_expected": {
            "mindgap_locator": reference["mindgap_locator"],
        },
    }


def fit_cell(model, subset, telemetry, row_ids, reference_root):
    """Fit one cell from the already-sanitized, telemetry-only view."""
    if len(telemetry) != len(row_ids):
        raise ValueError("telemetry rows and row IDs must have equal length")
    for row in telemetry:
        if set(row) != set(TELEMETRY_KEYS) or FORBIDDEN_FIT_KEYS.intersection(row):
            raise RuntimeError("fit_cell received a non-canonical telemetry row")

    global_scores, global_diag = fit_answer_mixed(telemetry)
    global_detector = np.asarray(global_scores["answer_dufs_liu_mixed"], dtype=float)

    channels = build_token_channels(telemetry)
    fit_rows = _fit_row_order(channels)
    prepared = _prepare_token_matrix(channels, fit_rows, CORE_TOKEN_VIEWS)

    iu_arm, iu_result = _ordinary_iu_arm(prepared)
    temporal_arm, temporal_diag = _fit_one_arm(prepared, "temporal")
    dufs_arm, dufs_diag = _fit_one_arm(prepared, "dufs")
    curves = {
        "local_iu_core": _apply_arm(iu_arm, channels),
        "local_temporal_core": _apply_arm(temporal_arm, channels),
        "local_dufs_core": _apply_arm(dufs_arm, channels),
    }

    fit_risk = [
        np.asarray(curves["local_iu_core"][row_index], dtype=float)[:take]
        for row_index, take in prepared["chunks"]
    ]
    hmm_diag = {}
    for kind, method in HMM_METHODS.items():
        fitted = fit_upcr_initialized_hmm(
            fit_risk,
            kind=kind,
            seeds=DEFAULT_SEEDS,
        )
        output, _, apply_diag = apply_latent_state_fit(
            fitted, curves["local_iu_core"]
        )
        curves[method] = output
        hmm_diag[kind] = {
            "fit": fitted.diagnostics,
            "selected": model_to_dict(fitted.selected),
            "candidates": [model_to_dict(candidate) for candidate in fitted.candidates],
            "fallback": fitted.fallback,
            "fallback_reason": fitted.fallback_reason,
            "apply": apply_diag,
        }

    mindgap_evidence, mindgap_detector = _mindgap_label_free(telemetry)
    offsets = None
    flattened = {}
    for method in LOCAL_METHODS:
        flat, current_offsets = flatten_curves(curves[method])
        if offsets is None:
            offsets = current_offsets
        elif not np.array_equal(offsets, current_offsets):
            raise RuntimeError("local methods disagree on token-grid offsets")
        flattened[method] = flat
    mindgap_flat, mindgap_offsets = flatten_curves(mindgap_evidence)
    if not np.array_equal(offsets, mindgap_offsets):
        raise RuntimeError("Mind-the-Gap evidence does not match the token grid")

    locators = {
        method: np.asarray([int(np.argmax(curve)) for curve in curves[method]], dtype=np.int64)
        for method in LOCAL_METHODS
    }
    score_hashes = {
        "row_ids": _hash_text(row_ids),
        "offsets": _hash_int(offsets),
        "global_mixed_v2_dufs": _hash_float(global_detector),
        "mindgap_detector": _hash_float(mindgap_detector),
        "mindgap_evidence": _hash_float(mindgap_flat),
    }
    for method in LOCAL_METHODS:
        score_hashes[method] = _hash_float(flattened[method])
        score_hashes[method + "__token_locator"] = _hash_int(locators[method])

    reference = _reference_hash_check(
        model, subset, score_hashes, reference_root
    )
    arrays = {
        "row_ids": np.asarray(row_ids, dtype=str),
        "offsets": offsets,
        "global_mixed_v2_dufs": global_detector,
        "mindgap_detector": mindgap_detector,
        "mindgap_evidence": mindgap_flat,
    }
    for method in LOCAL_METHODS:
        arrays[method] = flattened[method]
        arrays[method + "__token_locator"] = locators[method]

    diagnostics = {
        "model": model,
        "subset": subset,
        "n_rows": len(telemetry),
        "labels_or_step_spans_read": False,
        "fit_api_forbidden_keys": sorted(FORBIDDEN_FIT_KEYS),
        "feature_contract": list(CORE_TOKEN_VIEWS),
        "n_fit_tokens": int(prepared["F"].shape[1]),
        "n_fit_sequences": len(prepared["chunks"]),
        "fit_rows": fit_rows,
        "fit_chunks": prepared["chunks"],
        "feature_effective_rank": _effective_rank(prepared["F"]),
        "ordinary_iu": {
            "weights": iu_arm["weights"],
            "derived_orientation": iu_arm["derived"],
            "anchor_flipped": iu_arm["flipped"],
            "rho_hat": iu_result.rho_hat,
            "n_components": iu_result.n_components_used,
        },
        "temporal_liu": temporal_diag,
        "dufs_liu": dufs_diag,
        "hmm": hmm_diag,
        "global": global_diag,
        "score_hashes_before_evaluation": score_hashes,
        "frozen_baseline_reproduction": reference,
    }
    return arrays, diagnostics


def write_frozen_cell(out_dir, model, subset, arrays, diagnostics):
    out_dir = Path(out_dir)
    scores = out_dir / "label_free_scores" / f"{model}__{subset}.npz"
    diag = out_dir / "label_free_diagnostics" / f"{model}__{subset}.json"
    scores.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(scores, **arrays)
    _write_json(diag, diagnostics)
    return scores, diag


def load_and_verify_frozen(out_dir, model, subset, manifest_entry):
    out_dir = Path(out_dir)
    if (manifest_entry.get("model"), manifest_entry.get("subset")) != (model, subset):
        raise RuntimeError("manifest cell identity mismatch")
    scores_path = out_dir / manifest_entry["scores"]
    diag_path = out_dir / manifest_entry["diagnostics"]
    if not scores_path.exists() or not diag_path.exists():
        raise FileNotFoundError(f"missing frozen scores for {model}/{subset}; run fit first")
    if sha256_file(scores_path) != manifest_entry["scores_file_sha256"]:
        raise RuntimeError(f"frozen NPZ file hash changed: {model}/{subset}")
    if sha256_file(diag_path) != manifest_entry["diagnostics_file_sha256"]:
        raise RuntimeError(f"frozen diagnostic file hash changed: {model}/{subset}")
    with np.load(scores_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    diagnostics = json.load(diag_path.open())
    expected = manifest_entry["score_hashes"]
    if diagnostics.get("score_hashes_before_evaluation") != expected:
        raise RuntimeError(f"diagnostic and manifest hashes disagree: {model}/{subset}")
    observed = {
        "row_ids": _hash_text(arrays["row_ids"]),
        "offsets": _hash_int(arrays["offsets"]),
        "global_mixed_v2_dufs": _hash_float(arrays["global_mixed_v2_dufs"]),
        "mindgap_detector": _hash_float(arrays["mindgap_detector"]),
        "mindgap_evidence": _hash_float(arrays["mindgap_evidence"]),
    }
    for method in LOCAL_METHODS:
        observed[method] = _hash_float(arrays[method])
        observed[method + "__token_locator"] = _hash_int(
            arrays[method + "__token_locator"]
        )
    mismatches = {
        key: {"expected": expected.get(key), "observed": value}
        for key, value in observed.items() if expected.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"frozen score verification failed: {mismatches}")
    return arrays, diagnostics


def _open_evaluation_payload(raw_rows, frozen_row_ids):
    """The only function in this file that reads labels or step spans."""
    row_ids, labels, spans = [], [], []
    for cache_key, row in raw_rows:
        row_ids.append(str(row.get("id", cache_key)))
        labels.append(int(row["label"]))
        spans.append(row["step_token_spans"])
    if list(frozen_row_ids) != row_ids:
        raise RuntimeError("evaluation rows do not match the frozen score order")
    return np.asarray(labels, dtype=int), spans, row_ids


def _token_to_step(token_index, spans):
    for step_index, span in enumerate(spans):
        if span is not None and int(span[0]) <= int(token_index) < int(span[1]):
            return int(step_index)
    return NO_ERROR


def _mindgap_step_locator(evidence_curves, spans_by_row):
    output = []
    for evidence, spans in zip(evidence_curves, spans_by_row):
        scores = step_drop_scores(evidence, spans, ema_span=5)
        output.append(int(np.nanargmax(scores)) if np.isfinite(scores).any() else NO_ERROR)
    return np.asarray(output, dtype=int)


def _component_metrics(token_locator, step_locator, labels, spans_by_row):
    labels = np.asarray(labels, dtype=int)
    erroneous = labels != NO_ERROR
    predicted = np.asarray(step_locator, dtype=int)
    exact = float(np.mean(predicted[erroneous] == labels[erroneous]))
    tol1 = float(np.mean(np.abs(predicted[erroneous] - labels[erroneous]) <= 1))
    mapped = erroneous & (predicted != NO_ERROR)
    signed = predicted[mapped] - labels[mapped]
    token_distances = []
    for token, label, spans in zip(np.asarray(token_locator), labels, spans_by_row):
        if label == NO_ERROR or not (0 <= label < len(spans)) or spans[label] is None:
            continue
        gold_start = int(spans[label][0])
        trace_end = max(int(span[1]) for span in spans if span is not None)
        token_distances.append(abs(int(token) - gold_start) / max(trace_end, 1))
    return {
        "exact": exact,
        "tol1": tol1,
        "mapped_fraction": float(np.mean(predicted[erroneous] != NO_ERROR)),
        "mean_signed_step_error": float(np.mean(signed)) if len(signed) else float("nan"),
        "median_absolute_step_error": float(np.median(np.abs(signed))) if len(signed) else float("nan"),
        "mean_normalized_token_distance": float(np.mean(token_distances)) if token_distances else float("nan"),
    }


def _split_metrics(risk, locator, labels, method):
    labels = np.asarray(labels, dtype=int)
    risk = np.asarray(risk, dtype=float)
    locator = np.asarray(locator, dtype=int)
    rng = np.random.default_rng(SPLIT_SEED)
    rows = []
    for split_index in range(N_SPLITS):
        permutation = rng.permutation(len(labels))
        calibration = permutation[:len(labels) // 2]
        evaluation = permutation[len(labels) // 2:]
        threshold, calibration_f1 = best_threshold(
            risk, locator, labels, calibration
        )
        prediction = np.where(risk[evaluation] > threshold, locator[evaluation], NO_ERROR)
        scored = processbench_f1(prediction, labels[evaluation])
        rows.append({
            "method": method,
            "split_index": split_index,
            "f1": scored["f1"],
            "acc_erroneous": scored["acc_erroneous"],
            "acc_correct": scored["acc_correct"],
            "sla": sla(prediction, labels[evaluation], 0),
            "sla_tol1": sla(prediction, labels[evaluation], 1),
            "tau": threshold,
            "cal_f1": calibration_f1,
        })
    return rows


def evaluate_cell(
    model,
    subset,
    raw_rows,
    arrays,
    evaluation_reference=None,
    hmm_metadata=None,
):
    labels, spans, row_ids = _open_evaluation_payload(raw_rows, arrays["row_ids"].tolist())
    offsets = arrays["offsets"]
    curves = {
        method: unflatten_curves(arrays[method], offsets) for method in LOCAL_METHODS
    }
    token_locators = {
        method: np.asarray(arrays[method + "__token_locator"], dtype=int)
        for method in LOCAL_METHODS
    }
    step_locators = {
        method: np.asarray([
            _token_to_step(token, row_spans)
            for token, row_spans in zip(token_locators[method], spans)
        ], dtype=int)
        for method in LOCAL_METHODS
    }
    mindgap_evidence = unflatten_curves(arrays["mindgap_evidence"], offsets)
    mindgap_locator = _mindgap_step_locator(mindgap_evidence, spans)
    if evaluation_reference is not None:
        expected = evaluation_reference.get("mindgap_locator")
        # The frozen factorial artifact used its generic score hash, which
        # casts locator arrays to little-endian float64. Preserve that encoding
        # only for historical reproduction; our own locator hashes stay int64.
        observed = _hash_float(mindgap_locator)
        if expected != observed:
            raise RuntimeError(
                f"Mind-the-Gap locator reproduction failed for {model}/{subset}: "
                f"expected {expected}, observed {observed}"
            )

    systems = {
        "global_dufs__local_iu_core": (
            arrays["global_mixed_v2_dufs"], step_locators["local_iu_core"]
        ),
        "global_dufs__local_temporal_core": (
            arrays["global_mixed_v2_dufs"], step_locators["local_temporal_core"]
        ),
        "global_dufs__local_dufs_core": (
            arrays["global_mixed_v2_dufs"], step_locators["local_dufs_core"]
        ),
        "global_dufs__hmm_reversible": (
            arrays["global_mixed_v2_dufs"], step_locators["local_hmm_reversible_core_iu"]
        ),
        "global_dufs__hmm_absorbing": (
            arrays["global_mixed_v2_dufs"], step_locators["local_hmm_absorbing_core_iu"]
        ),
        "mindgap_control": (arrays["mindgap_detector"], mindgap_locator),
    }

    system_rows, split_rows = [], []
    for method, (risk, locator) in systems.items():
        canonical = evaluate_two_stage(
            risk, locator, labels, n_splits=N_SPLITS, seed=SPLIT_SEED
        )
        detailed = _split_metrics(risk, locator, labels, method)
        for key in ("f1", "acc_erroneous", "acc_correct", "sla", "sla_tol1"):
            observed = float(np.mean([row[key] for row in detailed]))
            if not np.isclose(observed, canonical[key], atol=1e-14):
                raise RuntimeError(f"split replay disagrees with evaluate_two_stage for {method}/{key}")
        system_rows.append({
            "model": model,
            "subset": subset,
            "split": "development" if (model, subset) in DEV else "nonselection",
            "system": method,
            **canonical,
        })
        split_rows.extend({"model": model, "subset": subset, **row} for row in detailed)

    component_rows = []
    for method in LOCAL_METHODS:
        component_rows.append({
            "model": model,
            "subset": subset,
            "split": "development" if (model, subset) in DEV else "nonselection",
            "candidate": method,
            **_component_metrics(token_locators[method], step_locators[method], labels, spans),
        })
    # Mind-the-Gap's output is step-native; token-distance metrics are undefined.
    erroneous = labels != NO_ERROR
    component_rows.append({
        "model": model,
        "subset": subset,
        "split": "development" if (model, subset) in DEV else "nonselection",
        "candidate": "mindgap_locator",
        "exact": float(np.mean(mindgap_locator[erroneous] == labels[erroneous])),
        "tol1": float(np.mean(np.abs(mindgap_locator[erroneous] - labels[erroneous]) <= 1)),
        "mapped_fraction": 1.0,
        "mean_signed_step_error": float(np.mean(mindgap_locator[erroneous] - labels[erroneous])),
        "median_absolute_step_error": float(np.median(np.abs(
            mindgap_locator[erroneous] - labels[erroneous]
        ))),
        "mean_normalized_token_distance": float("nan"),
    })

    prediction_rows = []
    for row_index, (row_id, label) in enumerate(zip(row_ids, labels)):
        for method in LOCAL_METHODS:
            prediction_rows.append({
                "model": model,
                "subset": subset,
                "row_id": row_id,
                "candidate": method,
                "gold_step": int(label),
                "predicted_step": int(step_locators[method][row_index]),
                "predicted_token": int(token_locators[method][row_index]),
                "trace_tokens": int(offsets[row_index + 1] - offsets[row_index]),
            })
        prediction_rows.append({
            "model": model,
            "subset": subset,
            "row_id": row_id,
            "candidate": "mindgap_locator",
            "gold_step": int(label),
            "predicted_step": int(mindgap_locator[row_index]),
            "predicted_token": "",
            "trace_tokens": int(offsets[row_index + 1] - offsets[row_index]),
        })

    alignment_rows = []
    for kind, method in HMM_METHODS.items():
        curve_kind = (
            (hmm_metadata or {}).get(kind, {}).get("apply", {}).get(
                "output_curve_kind", "unknown"
            )
        )
        buckets = {offset: [] for offset in range(
            -ERROR_ALIGNMENT_RADIUS, ERROR_ALIGNMENT_RADIUS + 1
        )}
        if curve_kind == "posterior_state_entry_probability":
            for curve, label, row_spans in zip(curves[method], labels, spans):
                if label == NO_ERROR or not (0 <= label < len(row_spans)):
                    continue
                span = row_spans[label]
                if span is None:
                    continue
                gold_start = int(span[0])
                for offset in buckets:
                    token = gold_start + offset
                    if 0 <= token < len(curve):
                        buckets[offset].append(float(curve[token]))
        for offset, values in buckets.items():
            alignment_rows.append({
                "model": model,
                "subset": subset,
                "candidate": method,
                "curve_kind": curve_kind,
                "relative_token": offset,
                "mean_entry_probability": (
                    float(np.mean(values)) if values else float("nan")
                ),
                "sum_entry_probability": float(np.sum(values)),
                "n": len(values),
            })
    return system_rows, component_rows, split_rows, prediction_rows, alignment_rows


def selected_values(args):
    models = tuple(args.models.split(",")) if args.models else MODELS
    subsets = tuple(args.subsets.split(",")) if args.subsets else SUBSETS
    unknown_models = sorted(set(models) - set(MODELS))
    unknown_subsets = sorted(set(subsets) - set(SUBSETS))
    if unknown_models or unknown_subsets:
        raise ValueError(f"unknown models={unknown_models}, subsets={unknown_subsets}")
    return models, subsets


def cache_path(cache_root, model, subset):
    return Path(cache_root) / ("pb_" + model) / f"processbench_{subset}.pkl"


def _registered_cells(models, subsets):
    return [(model, subset) for model in models for subset in subsets]


def build_run_definition(cache_root, reference_root, models, subsets):
    inputs = []
    references = []
    for model, subset in _registered_cells(models, subsets):
        data_path = cache_path(cache_root, model, subset).resolve()
        if not data_path.exists():
            raise FileNotFoundError(data_path)
        inputs.append({
            "model": model,
            "subset": subset,
            "path": str(data_path),
            "sha256": sha256_file(data_path),
        })
        reference_path = (Path(reference_root) / f"{model}__{subset}.json").resolve()
        if not reference_path.exists():
            raise FileNotFoundError(
                f"frozen baseline reference is required: {reference_path}"
            )
        references.append({
            "model": model,
            "subset": subset,
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        })
    payload = {
        "version": VERSION,
        "models": list(models),
        "subsets": list(subsets),
        "cells": [
            {"model": model, "subset": subset}
            for model, subset in _registered_cells(models, subsets)
        ],
        "inputs": inputs,
        "baseline_references": references,
        "source_sha256": {
            relative: sha256_file(ROOT / relative) for relative in SOURCE_FILES
        },
        "feature_contract": list(CORE_TOKEN_VIEWS),
        "global_detector": "mixed-v2 DUFS-LIU, lambda=0.1, k=7",
        "local_initializer": "ordinary full-pool two-component IU-PCR, lambda=0",
        "hmm_seeds": list(DEFAULT_SEEDS),
        "minimum_seed_locator_agreement": DEFAULT_MIN_SEED_LOCATOR_AGREEMENT,
        "n_splits": N_SPLITS,
        "split_seed": SPLIT_SEED,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["run_fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
    return payload


def verify_run_definition(out_dir, cache_root, models, subsets):
    out_dir = Path(out_dir)
    path = out_dir / "RUN_DEFINITION.json"
    if not path.exists():
        raise FileNotFoundError("RUN_DEFINITION.json is missing")
    definition = json.load(path.open())
    if definition.get("version") != VERSION:
        raise RuntimeError("run-definition version mismatch")
    expected_cells = [
        {"model": model, "subset": subset}
        for model, subset in _registered_cells(models, subsets)
    ]
    if definition.get("cells") != expected_cells:
        raise RuntimeError("requested cells differ from the frozen run definition")
    for relative, expected in definition.get("source_sha256", {}).items():
        if sha256_file(ROOT / relative) != expected:
            raise RuntimeError(f"source changed after score freeze: {relative}")
    expected_inputs = {
        (item["model"], item["subset"]): item for item in definition["inputs"]
    }
    for model, subset in _registered_cells(models, subsets):
        item = expected_inputs[(model, subset)]
        observed_path = cache_path(cache_root, model, subset).resolve()
        if str(observed_path) != item["path"] or sha256_file(observed_path) != item["sha256"]:
            raise RuntimeError(f"input pickle changed after score freeze: {model}/{subset}")
    for item in definition.get("baseline_references", []):
        path = Path(item["path"])
        if sha256_file(path) != item["sha256"]:
            raise RuntimeError(
                "frozen baseline reference changed after score freeze: "
                f"{item['model']}/{item['subset']}"
            )
    canonical = dict(definition)
    fingerprint = canonical.pop("run_fingerprint")
    observed = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if observed != fingerprint:
        raise RuntimeError("run-definition fingerprint mismatch")
    return definition


def _manifest_cell_map(manifest):
    output = {}
    for item in manifest.get("cells", []):
        key = (item["model"], item["subset"])
        if key in output:
            raise RuntimeError(f"duplicate frozen cell: {key}")
        output[key] = item
    return output


def run_fit(args):
    models, subsets = selected_values(args)
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"refusing to overwrite non-empty experiment directory: {out_dir}"
        )
    definition = build_run_definition(
        args.cache_root, args.reference_root, models, subsets
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    definition_path = out_dir / "RUN_DEFINITION.json"
    _write_json(definition_path, definition)
    cells = []
    for model in models:
        for subset in subsets:
            path = cache_path(args.cache_root, model, subset)
            print(f"LABEL-FREE FIT {model}/{subset}", flush=True)
            raw_rows = load_raw_rows(path)
            telemetry, row_ids = telemetry_view(raw_rows)
            del raw_rows
            arrays, diagnostics = fit_cell(
                model, subset, telemetry, row_ids, args.reference_root
            )
            scores, diag = write_frozen_cell(
                out_dir, model, subset, arrays, diagnostics
            )
            cells.append({
                "model": model,
                "subset": subset,
                "scores": str(scores.relative_to(out_dir)),
                "scores_file_sha256": sha256_file(scores),
                "diagnostics": str(diag.relative_to(out_dir)),
                "diagnostics_file_sha256": sha256_file(diag),
                "score_hashes": diagnostics["score_hashes_before_evaluation"],
            })
    manifest = {
        "version": VERSION,
        "stage": "label_free_scores_frozen",
        "run_fingerprint": definition["run_fingerprint"],
        "run_definition_sha256": sha256_file(definition_path),
        "labels_or_step_spans_read": False,
        "feature_contract": list(CORE_TOKEN_VIEWS),
        "global_detector": "mixed-v2 DUFS-LIU, lambda=0.1, k=7",
        "local_initializer": "ordinary full-pool two-component IU-PCR, lambda=0",
        "primary": "reversible shared-variance two-state Gaussian HMM",
        "falsification_control": "absorbing shared-variance two-state Gaussian HMM",
        "hmm_seeds": list(DEFAULT_SEEDS),
        "minimum_seed_locator_agreement": DEFAULT_MIN_SEED_LOCATOR_AGREEMENT,
        "selection": "maximum label-free likelihood among guard-valid starts",
        "evaluation_requires_separate_command": True,
        "cells": cells,
    }
    _write_json(out_dir / "FREEZE_MANIFEST.json", manifest)
    print(out_dir / "FREEZE_MANIFEST.json")


def run_evaluate(args):
    out_dir = Path(args.out_dir).resolve()
    manifest_path = out_dir / "FREEZE_MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError("FREEZE_MANIFEST.json is required; run fit first")
    manifest = json.load(manifest_path.open())
    if manifest.get("version") != VERSION or manifest.get("stage") != "label_free_scores_frozen":
        raise RuntimeError("score manifest is not in the frozen pre-evaluation state")
    models, subsets = selected_values(args)
    definition = verify_run_definition(
        out_dir, args.cache_root, models, subsets
    )
    definition_path = out_dir / "RUN_DEFINITION.json"
    if manifest.get("run_definition_sha256") != sha256_file(definition_path):
        raise RuntimeError("run definition changed after score freeze")
    if manifest.get("run_fingerprint") != definition["run_fingerprint"]:
        raise RuntimeError("score manifest and run definition disagree")
    cell_map = _manifest_cell_map(manifest)
    expected_keys = set(_registered_cells(models, subsets))
    if set(cell_map) != expected_keys:
        raise RuntimeError("score manifest does not contain the exact requested cell roster")
    evaluation = out_dir / "evaluation"
    if evaluation.exists() and any(evaluation.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing evaluation: {evaluation}")
    systems, components, splits, predictions, aligned = [], [], [], [], []
    for model in models:
        for subset in subsets:
            arrays, diagnostics = load_and_verify_frozen(
                out_dir, model, subset, cell_map[(model, subset)]
            )
            # Evaluation-only values are first accessed below, after persistent
            # score artifacts and hashes have both been verified.
            raw_rows = load_raw_rows(cache_path(args.cache_root, model, subset))
            reference = diagnostics["frozen_baseline_reproduction"]
            cell = evaluate_cell(
                model,
                subset,
                raw_rows,
                arrays,
                reference.get("evaluation_expected"),
                diagnostics.get("hmm"),
            )
            systems.extend(cell[0]); components.extend(cell[1])
            splits.extend(cell[2]); predictions.extend(cell[3])
            aligned.extend(cell[4])
    output_files = {
        "systems_per_cell.csv": systems,
        "components_per_cell.csv": components,
        "split_metrics.csv": splits,
        "localization_rows.csv": predictions,
        "error_aligned_entry.csv": aligned,
    }
    for name, rows in output_files.items():
        _write_csv(evaluation / name, rows)
    _write_json(evaluation / "EVALUATION_MANIFEST.json", {
        "version": VERSION,
        "score_manifest": str(manifest_path),
        "score_manifest_sha256": sha256_file(manifest_path),
        "run_fingerprint": definition["run_fingerprint"],
        "labels_opened_after_score_verification": True,
        "protocol": "evaluate_two_stage; 100 repeated 50/50 calibration/evaluation splits; seed 0",
        "exploratory": True,
        "no_variant_selection_performed": True,
        "cells": [
            {"model": model, "subset": subset}
            for model, subset in _registered_cells(models, subsets)
        ],
        "files_sha256": {
            name: sha256_file(evaluation / name) for name in output_files
        },
    })
    print(evaluation)


def parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--cache-root", required=True)
    common.add_argument("--out-dir", required=True)
    common.add_argument("--models", default=None, help="comma-separated subset of qwen3_4b,qwen3_8b")
    common.add_argument("--subsets", default=None, help="comma-separated ProcessBench subsets")
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", parents=[common], help="fit and freeze label-free scores")
    fit.add_argument(
        "--reference-root",
        default=str(ROOT / "results" / "gl_liu_factorial_v2" / "diagnostics"),
        help="frozen baseline hashes; if a cell file exists, equality is mandatory",
    )
    commands.add_parser("evaluate", parents=[common], help="open labels only after score verification")
    return root


def main():
    args = parser().parse_args()
    if args.command == "fit":
        run_fit(args)
    else:
        run_evaluate(args)


if __name__ == "__main__":
    main()
