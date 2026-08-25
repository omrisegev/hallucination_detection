"""Post-freeze evaluation for incomparable RAG evidence benchmark panels."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
import csv
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .io import canonical_json_bytes, sha256_bytes
from .rag_evidence_contract import (
    AtomicRagDirectory,
    EVALUATION_MANIFEST_FILENAME,
    EVALUATION_SCHEMA,
    PANEL_IDS,
    PREPARATION_MANIFEST_FILENAME,
    PRIVATE_LABEL_FILENAME,
    REFCHECKER_SETTINGS,
    SCORE_AB_SCHEMA,
    SCORE_MANIFEST_FILENAME,
    RagEvidenceContractError,
    add_payload_sha256,
    load_private_labels,
    load_registry,
    payload_sha256,
    read_bound_file_bytes,
    validate_artifact_identifier,
    validate_source_binding,
    verify_payload,
)
from .rag_evidence_score_reader import load_scores, validate_score_arrays


EVALUATION_SOURCE_FILES = (
    "configs/reconstruction_benchmark_v1/rag_evidence.json",
    "spectral_utils/reconstruction_benchmark/io.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_contract.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_score_reader.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_evaluation.py",
    "spectral_utils/reconstruction_benchmark/rag_evidence_evaluation_ab.py",
    "scripts/reconstruction_benchmark/evaluate_rag_evidence.py",
    "scripts/reconstruction_benchmark/verify_rag_evidence_evaluation_ab.py",
)
MINIMUM_EVALUATION_GIT_HEAD = "4a8e29452c06b706777f9b50ad7feda87ac67ba3"
SCORE_VERIFIER_GIT_HEAD = "409900332854c0586c4abc7dbc33f10b565b59af"
SCORE_CERTIFICATE_FILENAME = "SCORE_AB_VERIFICATION.json"
SCORE_VERIFIER_SOURCE_SHA256 = {
    "configs/reconstruction_benchmark_v1/rag_evidence.json": "a835ef223fe0c84ce3729277ffb0c90bd5681b705b40eaa3d4d47cdc038379af",
    "spectral_utils/__init__.py": "8b2dd0c1eb39fc0832a6d25167ca628da71366dccdc0f58907b2f5adb5d42e19",
    "spectral_utils/reconstruction_benchmark/__init__.py": "2c4fc6447df74b9952b2481adb3eec7570149ad03dc0959935f4e77cae596582",
    "spectral_utils/dufs_liu_feature_contract.py": "c674f7b6cfd1a82dcc63cd7cc26bbd3d53ef5f4d289dbcbb7eabe933670fdf2d",
    "spectral_utils/feature_contract.py": "65ffba0bfc9bf4ded859d9ebffce9bd563a28652c47f5ed8d80ce91b64d4385d",
    "spectral_utils/feature_utils.py": "b68bc5e1647742667fc20908321485bf9229954713b5da7215bf0f2629c74362",
    "spectral_utils/fixed_application_pipelines.py": "0d2c6dae58530654f3d4ee18d4c5fa278ef41a61634c3e11caf1c86ca8816203",
    "spectral_utils/fusion_utils.py": "1f47ed8f1d16a41532d85dfbacb0c0d727645aaf6921223e9e8e9f6ccc6fadd7",
    "spectral_utils/repeated_measurement_reliability.py": "9e4b975e0f890e06f7194e1bd81012253329093acdb664756fa60cdab4de0868",
    "spectral_utils/token_feature_views.py": "3e86166ef75cd222ffbeedfa394b829739e05e4efd56b80eb6fe8f237b0752e7",
    "spectral_utils/upcr.py": "587d4b4699e8cd49d28315f70ffdabb4eb7ef9fd48e4df5436e121a71536c071",
    "spectral_utils/reconstruction_benchmark/fit_firewall.py": "fd472073bc972f4d45b9a4cd09f89d9fa831c0e542093cb6df0d316602d13770",
    "spectral_utils/reconstruction_benchmark/io.py": "97eccfe07463a5e5ff3470e2155f0d9c5a506631687c941e2841046efb57142b",
    "spectral_utils/reconstruction_benchmark/rag_evidence_contract.py": "db761f5330688c95bfc94057a96433a326baa59df3dfbea3339261f9465877b7",
    "spectral_utils/reconstruction_benchmark/rag_evidence_fit.py": "65352d1b85c72af389b924bb9d8cb81ae70f63e9922e667927ac5d830b1b54f1",
    "spectral_utils/reconstruction_benchmark/rag_evidence_ab.py": "9b6a3f0ebb0ba02fbe0704802b1ecccbefb967819b83d49e4427c69cc96a629a",
    "spectral_utils/reconstruction_benchmark/rag_evidence_runner.py": "c7e79a7c2d36de85c6782409c1b9a2ce09082d555645bd76ace11957b4b1fa2b",
    "scripts/reconstruction_benchmark/rag_evidence_fit_worker.py": "cb68c861e0d52c5602cad29844fef07ebbc2635a3dd4768aa6ccf8f0a774c186",
    "spectral_utils/ragtruth_evidence_contrast.py": "1b0c67010825062da832302d7efcca3cc7db3b667f5edcaac749f689905b4b3c",
    "spectral_utils/reconstruction_benchmark/rag_evidence_preparation.py": "8743b5dfd242a6f2d421e5ff0bdf938727f6d8fc5a6e64cecf1cabdfdcc9c140",
    "spectral_utils/repgrid_scoring.py": "b25e25f5c08fde40c9cb47b0f63f1915bf438362b97f2b68e482ae31c41cde79",
    "spectral_utils/streaming_utils.py": "03b57d452a5408a5f801db3cb13dbf2a91c3e7d809db95675c13d682d3616529",
    "spectral_utils/reconstruction_benchmark/rag_evidence_evaluation.py": "56a5a146e8f85a99d6288231ee83a32d88fe35203e35dfc4ea6439391dc7426d",
}
PREDICTION_COLUMNS = (
    "panel_id", "split", "subgroup", "method_id", "unit_id", "parent_id",
    "score", "prediction", "label", "bootstrap_group",
)
METRIC_COLUMNS = (
    "panel_id", "dataset", "unit", "access", "estimand", "split", "subgroup",
    "method_id", "metric", "value", "ci_low", "ci_high", "n", "n_groups",
    "positive_rate", "bootstrap_draws", "status",
)
CONTRAST_COLUMNS = (
    "panel_id", "split", "subgroup", "left_method", "right_method", "metric",
    "delta", "ci_low", "ci_high", "n", "n_groups", "bootstrap_draws", "status",
)

_FAST_METRIC_NAMES = frozenset({
    "auroc", "auprc", "f1", "precision", "recall", "accuracy", "macro_f1",
})
_RANKING_METRIC_NAMES = frozenset({"auroc", "auprc"})
_BINARY_HARD_METRIC_NAMES = frozenset({"f1", "precision", "recall"})
_THREEWAY_LABELS = ("Entailment", "Neutral", "Contradiction")
_BOOTSTRAP_MULTIPLICITY_BATCH_SIZE = 64

_SCORE_AUTHENTICATOR_CODE = r'''import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve(strict=True)
sys.path.insert(0, str(repo))
from spectral_utils.reconstruction_benchmark.io import canonical_json_bytes
from spectral_utils.reconstruction_benchmark.rag_evidence_ab import (
    authenticate_rag_evidence_score_certificate,
)

certificate = authenticate_rag_evidence_score_certificate(
    repo=repo,
    registry_path=Path(sys.argv[2]),
    source_root=Path(sys.argv[3]),
    release_root=Path(sys.argv[4]),
    private_root=Path(sys.argv[5]),
    release_id=sys.argv[6],
    require_scientific_full=sys.argv[7] == "1",
)
sys.stdout.buffer.write(canonical_json_bytes(certificate) + b"\n")
'''


def _git(
    repo: Path, *arguments: str, check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments], cwd=repo, capture_output=True, check=check,
    )


def _repository_snapshot(
    repo: str | Path,
    source_files: Sequence[str],
    *,
    expected_head: str | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    require_clean: bool,
    minimum_head: str | None = None,
) -> dict[str, Any]:
    repo_path = Path(repo).resolve(strict=True)
    head = _git(repo_path, "rev-parse", "HEAD").stdout.decode("ascii").strip()
    status = _git(
        repo_path, "status", "--porcelain=v1", "--untracked-files=all",
    ).stdout
    if expected_head is not None and head != expected_head:
        raise RagEvidenceContractError(
            f"RAG repository HEAD {head} differs from required {expected_head}"
        )
    if require_clean and status:
        raise RagEvidenceContractError("RAG repository must be clean")
    if minimum_head is not None and _git(
        repo_path, "merge-base", "--is-ancestor", minimum_head, head,
        check=False,
    ).returncode != 0:
        raise RagEvidenceContractError(
            "RAG evaluator does not descend from the optimized release"
        )
    rows = []
    for relative in source_files:
        digest = sha256_bytes(read_bound_file_bytes(
            repo_path / relative, name=f"RAG repository source {relative}",
        ))
        if expected_hashes is not None and expected_hashes.get(relative) != digest:
            raise RagEvidenceContractError(
                f"RAG repository source hash differs: {relative}"
            )
        rows.append({"path": relative, "sha256": digest})
    value = {
        "git_head": head,
        "git_clean": not bool(status),
        "git_status_sha256": sha256_bytes(status),
        "source_files": rows,
    }
    value["snapshot_sha256"] = payload_sha256(value)
    return value


def capture_evaluation_repository_snapshot(
    repo: str | Path, *, require_scientific_full: bool,
) -> dict[str, Any]:
    return _repository_snapshot(
        repo, EVALUATION_SOURCE_FILES,
        require_clean=require_scientific_full,
        minimum_head=MINIMUM_EVALUATION_GIT_HEAD if require_scientific_full else None,
    )


def capture_score_verifier_repository_snapshot(
    score_verifier_repo: str | Path,
) -> dict[str, Any]:
    return _repository_snapshot(
        score_verifier_repo, tuple(SCORE_VERIFIER_SOURCE_SHA256),
        expected_head=SCORE_VERIFIER_GIT_HEAD,
        expected_hashes=SCORE_VERIFIER_SOURCE_SHA256,
        require_clean=True,
    )


def authenticate_rag_evidence_score_certificate_from_repo(
    *, evaluation_repo: str | Path, score_verifier_repo: str | Path,
    registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path, release_id: str,
    require_scientific_full: bool,
) -> dict[str, Any]:
    """Run the exact frozen score authenticator before any private-label open."""

    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    evaluation_repo_path = Path(evaluation_repo).resolve(strict=True)
    score_repo_path = Path(score_verifier_repo).resolve(strict=True)
    if evaluation_repo_path == score_repo_path:
        raise RagEvidenceContractError(
            "RAG evaluation and score-verifier repositories must be distinct"
        )
    evaluation_snapshot = capture_evaluation_repository_snapshot(
        evaluation_repo_path,
        require_scientific_full=require_scientific_full,
    )
    score_snapshot = capture_score_verifier_repository_snapshot(score_repo_path)
    evaluation_registry = read_bound_file_bytes(
        Path(registry_path).resolve(strict=True), name="RAG evaluation registry",
    )
    score_registry_path = (
        score_repo_path / "configs/reconstruction_benchmark_v1/rag_evidence.json"
    )
    score_registry = read_bound_file_bytes(
        score_registry_path, name="RAG score-verifier registry",
    )
    if evaluation_registry != score_registry:
        raise RagEvidenceContractError(
            "RAG evaluation and score-verifier registries differ"
        )
    lane_root = Path(release_root) / release_id / "rag_evidence"
    certificate_path = lane_root / SCORE_CERTIFICATE_FILENAME
    frozen_payload = read_bound_file_bytes(
        certificate_path, name="RAG frozen score A/B certificate",
    )
    environment = dict(os.environ)
    for variable in (
        "PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE",
    ):
        environment.pop(variable, None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [
            sys.executable, "-I", "-B", "-c", _SCORE_AUTHENTICATOR_CODE,
            str(score_repo_path), str(score_registry_path),
            str(Path(source_root).resolve(strict=True)),
            str(Path(release_root).resolve(strict=True)),
            str(Path(private_root).resolve(strict=True)), release_id,
            # Scores remain bound to the frozen scientific/full certificate
            # even when current reporting uses debug bootstrap draws.
            "1",
        ],
        cwd=score_repo_path, env=environment, capture_output=True, check=False,
    )
    if (
        completed.returncode != 0
        or completed.stderr
        or completed.stdout != frozen_payload
    ):
        raise RagEvidenceContractError(
            "old-checkout RAG score authentication did not reproduce the frozen certificate"
        )
    try:
        certificate = json.loads(frozen_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagEvidenceContractError(
            "RAG frozen score certificate is not JSON"
        ) from error
    verify_payload(certificate, name="RAG frozen score A/B certificate")
    if (
        certificate.get("schema_version") != SCORE_AB_SCHEMA
        or certificate.get("status") != "PASS"
    ):
        raise RagEvidenceContractError("RAG frozen score certificate does not pass")
    if capture_score_verifier_repository_snapshot(score_repo_path) != score_snapshot:
        raise RagEvidenceContractError(
            "RAG score-verifier repository changed during authentication"
        )
    if capture_evaluation_repository_snapshot(
        evaluation_repo_path,
        require_scientific_full=require_scientific_full,
    ) != evaluation_snapshot:
        raise RagEvidenceContractError(
            "RAG evaluation repository changed during score authentication"
        )
    return {
        "certificate": certificate,
        "certificate_payload": frozen_payload,
        "evaluation_repo_snapshot": evaluation_snapshot,
        "score_verifier_repo_snapshot": score_snapshot,
    }


def _csv_bytes(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> bytes:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name, "") for name in columns})
    return stream.getvalue().encode("utf-8")


def _binary_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "auroc":
        return lambda y, s: float(roc_auc_score(y, s))
    if name == "auprc":
        return lambda y, s: float(average_precision_score(y, s))
    raise KeyError(name)


def _threeway_metric(name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    if name == "accuracy":
        return lambda y, p: float(np.mean(y == p))
    if name == "macro_f1":
        return lambda y, p: float(f1_score(
            y, p, labels=_THREEWAY_LABELS, average="macro", zero_division=0
        ))
    raise KeyError(name)


def _group_count(mask: np.ndarray, group_codes: np.ndarray, n_groups: int) -> np.ndarray:
    """Return exact integer row counts per group for a boolean row mask."""

    return np.bincount(group_codes[mask], minlength=n_groups).astype(np.int64, copy=False)


def _prepare_fast_metric(
    target: np.ndarray,
    value: np.ndarray,
    group_codes: np.ndarray,
    *,
    n_groups: int,
    metric_name: str,
) -> dict[str, Any]:
    """Precompute label/score-order state without changing metric semantics.

    The slow reference materializes every selected group, then sklearn sorts
    the resulting rows for every bootstrap draw.  A group selected ``k``
    times is exactly equivalent to assigning integer sample weight ``k`` to
    each of its rows.  We retain sklearn's stable descending score order and
    threshold arithmetic, but compute that order once per metric call.
    """

    target = np.asarray(target)
    value = np.asarray(value)
    group_codes = np.asarray(group_codes, dtype=np.int64)
    group_sizes = np.bincount(group_codes, minlength=n_groups).astype(
        np.int64, copy=False
    )
    positive_by_group = _group_count(target == 1, group_codes, n_groups)
    state: dict[str, Any] = {
        "metric_name": metric_name,
        "group_sizes": group_sizes,
        "positive_by_group": positive_by_group,
    }
    if metric_name in _RANKING_METRIC_NAMES:
        score = np.asarray(value, dtype=np.float64)
        # sklearn 1.9's confusion_matrix_at_thresholds uses a stable
        # descending sort.  Tied-row order does not change threshold totals,
        # while retaining the same order keeps the floating path literal.
        order = np.argsort(score, kind="stable")[::-1]
        state.update({
            "sorted_positive": np.asarray(target[order] == 1, dtype=np.float64),
            "sorted_score": score[order],
            "sorted_group_codes": group_codes[order],
        })
        return state
    if metric_name in _BINARY_HARD_METRIC_NAMES:
        prediction = np.asarray(value)
        state.update({
            "true_positive_by_group": _group_count(
                (target == 1) & (prediction == 1), group_codes, n_groups
            ),
            "false_positive_by_group": _group_count(
                (target != 1) & (prediction == 1), group_codes, n_groups
            ),
            "false_negative_by_group": _group_count(
                (target == 1) & (prediction != 1), group_codes, n_groups
            ),
        })
        return state
    if metric_name == "accuracy":
        state["correct_by_group"] = _group_count(
            target == value, group_codes, n_groups
        )
        return state
    if metric_name == "macro_f1":
        prediction = np.asarray(value)
        statistics = np.empty((n_groups, len(_THREEWAY_LABELS), 3), dtype=np.int64)
        for label_index, label in enumerate(_THREEWAY_LABELS):
            statistics[:, label_index, 0] = _group_count(
                (target == label) & (prediction == label), group_codes, n_groups
            )
            statistics[:, label_index, 1] = _group_count(
                (target != label) & (prediction == label), group_codes, n_groups
            )
            statistics[:, label_index, 2] = _group_count(
                (target == label) & (prediction != label), group_codes, n_groups
            )
        state["threeway_f1_statistics"] = statistics
        return state
    raise RagEvidenceContractError(f"unsupported fast RAG metric: {metric_name}")


def _resample_has_two_classes(
    state: Mapping[str, Any], multiplicities: np.ndarray
) -> bool:
    positives = int(np.dot(multiplicities, state["positive_by_group"]))
    total = int(np.dot(multiplicities, state["group_sizes"]))
    return 0 < positives < total


def _fast_ranking_metric(
    state: Mapping[str, Any], multiplicities: np.ndarray
) -> float:
    """Reproduce sklearn AUROC/AP on integer-weighted, presorted rows."""

    row_weights = multiplicities[state["sorted_group_codes"]]
    active = row_weights != 0
    positive = state["sorted_positive"][active]
    score = state["sorted_score"][active]
    weight = row_weights[active]
    threshold_indexes = np.concatenate((
        np.flatnonzero(np.diff(score)), np.asarray([len(score) - 1], dtype=np.int64)
    ))
    true_positive = np.cumsum(positive * weight, dtype=np.float64)[threshold_indexes]
    false_positive = np.cumsum(
        (1.0 - positive) * weight, dtype=np.float64
    )[threshold_indexes]
    if state["metric_name"] == "auroc":
        # Match roc_curve(drop_intermediate=True) before trapezoidal AUC.
        if len(false_positive) > 2:
            keep = np.where(np.concatenate((
                np.asarray([True]),
                np.logical_or(
                    np.diff(false_positive, 2), np.diff(true_positive, 2)
                ),
                np.asarray([True]),
            )))[0]
            false_positive = false_positive[keep]
            true_positive = true_positive[keep]
        fpr = np.concatenate((np.asarray([0.0]), false_positive)) / false_positive[-1]
        tpr = np.concatenate((np.asarray([0.0]), true_positive)) / true_positive[-1]
        return float(np.trapezoid(tpr, fpr))
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / true_positive[-1]
    # Match average_precision_score's non-interpolated step integral.
    precision = np.concatenate((precision[::-1], np.asarray([1.0])))
    recall = np.concatenate((recall[::-1], np.asarray([0.0])))
    return float(max(0.0, -np.sum(np.diff(recall) * precision[:-1])))


def _fast_metric_value(
    state: Mapping[str, Any], multiplicities: np.ndarray
) -> float:
    metric_name = str(state["metric_name"])
    if metric_name in _RANKING_METRIC_NAMES:
        return _fast_ranking_metric(state, multiplicities)
    if metric_name in _BINARY_HARD_METRIC_NAMES:
        true_positive = int(np.dot(
            multiplicities, state["true_positive_by_group"]
        ))
        false_positive = int(np.dot(
            multiplicities, state["false_positive_by_group"]
        ))
        false_negative = int(np.dot(
            multiplicities, state["false_negative_by_group"]
        ))
        if metric_name == "precision":
            denominator = true_positive + false_positive
            return float(true_positive / denominator) if denominator else 0.0
        if metric_name == "recall":
            denominator = true_positive + false_negative
            return float(true_positive / denominator) if denominator else 0.0
        denominator = 2 * true_positive + false_positive + false_negative
        return float(2 * true_positive / denominator) if denominator else 0.0
    if metric_name == "accuracy":
        correct = int(np.dot(multiplicities, state["correct_by_group"]))
        total = int(np.dot(multiplicities, state["group_sizes"]))
        return float(correct / total)
    if metric_name == "macro_f1":
        statistics = np.tensordot(
            multiplicities, state["threeway_f1_statistics"], axes=(0, 0)
        )
        true_positive = statistics[:, 0]
        false_positive = statistics[:, 1]
        false_negative = statistics[:, 2]
        denominator = 2 * true_positive + false_positive + false_negative
        per_class = np.divide(
            2 * true_positive,
            denominator,
            out=np.zeros(len(_THREEWAY_LABELS), dtype=np.float64),
            where=denominator != 0,
        )
        return float(np.mean(per_class))
    raise RagEvidenceContractError(f"unsupported fast RAG metric: {metric_name}")


def _iter_group_multiplicity_batches(
    rng: np.random.Generator,
    unique_groups: np.ndarray,
    *,
    draws: int,
) -> Iterator[np.ndarray]:
    """Yield small count batches while retaining one literal choice per draw."""

    n_groups = len(unique_groups)
    remaining = int(draws)
    while remaining:
        batch_rows = min(_BOOTSTRAP_MULTIPLICITY_BATCH_SIZE, remaining)
        batch = np.empty((batch_rows, n_groups), dtype=np.int64)
        for row_index in range(batch_rows):
            selected = rng.choice(unique_groups, size=n_groups, replace=True)
            batch[row_index] = np.bincount(
                np.searchsorted(unique_groups, selected), minlength=n_groups
            )
        yield batch
        remaining -= batch_rows


def _fast_grouped_samples(
    target: np.ndarray,
    value: np.ndarray,
    group_values: np.ndarray,
    *,
    draws: int,
    seed: int,
    require_two_classes: bool,
    metric_name: str,
) -> list[float]:
    unique_groups = np.unique(group_values)
    group_codes = np.searchsorted(unique_groups, group_values)
    state = _prepare_fast_metric(
        target, value, group_codes,
        n_groups=len(unique_groups), metric_name=metric_name,
    )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for batch in _iter_group_multiplicity_batches(
        rng, unique_groups, draws=int(draws)
    ):
        for multiplicities in batch:
            if require_two_classes and not _resample_has_two_classes(
                state, multiplicities
            ):
                continue
            result = _fast_metric_value(state, multiplicities)
            if np.isfinite(result):
                samples.append(result)
    return samples


def _fast_grouped_paired_samples(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    group_values: np.ndarray,
    *,
    draws: int,
    seed: int,
    metric_name: str,
) -> list[float]:
    unique_groups = np.unique(group_values)
    group_codes = np.searchsorted(unique_groups, group_values)
    left_state = _prepare_fast_metric(
        target, left, group_codes,
        n_groups=len(unique_groups), metric_name=metric_name,
    )
    right_state = _prepare_fast_metric(
        target, right, group_codes,
        n_groups=len(unique_groups), metric_name=metric_name,
    )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for batch in _iter_group_multiplicity_batches(
        rng, unique_groups, draws=int(draws)
    ):
        for multiplicities in batch:
            if not _resample_has_two_classes(left_state, multiplicities):
                continue
            samples.append(
                _fast_metric_value(left_state, multiplicities)
                - _fast_metric_value(right_state, multiplicities)
            )
    return samples


def grouped_interval(
    target: np.ndarray,
    value: np.ndarray,
    groups: Sequence[str],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    draws: int,
    seed: int,
    require_two_classes: bool,
    metric_name: str | None = None,
) -> dict[str, Any]:
    target = np.asarray(target)
    value = np.asarray(value)
    group_values = np.asarray(groups).astype(str)
    if len(target) != len(value) or len(target) != len(group_values) or not len(target):
        raise RagEvidenceContractError("grouped bootstrap input alignment failed")
    point_status = "OK"
    if require_two_classes and len(np.unique(target)) < 2:
        return {
            "value": "", "ci_low": "", "ci_high": "", "draws": 0,
            "status": "METRIC_UNDEFINED_SINGLE_CLASS",
        }
    point = metric(target, value)
    if metric_name is not None:
        if metric_name not in _FAST_METRIC_NAMES:
            raise RagEvidenceContractError(
                f"unsupported fast RAG metric: {metric_name}"
            )
        samples = _fast_grouped_samples(
            target, value, group_values, draws=draws, seed=seed,
            require_two_classes=require_two_classes, metric_name=metric_name,
        )
    else:
        # Compatibility/reference path for arbitrary metrics.  Production RAG
        # metrics always use the explicit, audited fast dispatch above.
        unique = np.unique(group_values)
        lookup = {group: np.flatnonzero(group_values == group) for group in unique}
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(int(draws)):
            selected = rng.choice(unique, size=len(unique), replace=True)
            indexes = np.concatenate([lookup[group] for group in selected])
            if require_two_classes and len(np.unique(target[indexes])) < 2:
                continue
            result = float(metric(target[indexes], value[indexes]))
            if np.isfinite(result):
                samples.append(result)
    if not samples:
        return {
            "value": point, "ci_low": "", "ci_high": "", "draws": 0,
            "status": "BOOTSTRAP_UNDEFINED",
        }
    array = np.asarray(samples, dtype=float)
    return {
        "value": point,
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "draws": len(array),
        "status": point_status,
    }


def grouped_paired_delta(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: Sequence[str],
    metric: Callable[[np.ndarray, np.ndarray], float],
    *, draws: int, seed: int, metric_name: str | None = None,
) -> dict[str, Any]:
    target = np.asarray(target)
    left, right = np.asarray(left, float), np.asarray(right, float)
    group_values = np.asarray(groups).astype(str)
    if not (len(target) == len(left) == len(right) == len(group_values)):
        raise RagEvidenceContractError("paired RAG contrast alignment failed")
    if len(np.unique(target)) < 2:
        return {"delta": "", "ci_low": "", "ci_high": "", "draws": 0, "status": "METRIC_UNDEFINED_SINGLE_CLASS"}
    if metric_name is not None:
        if metric_name not in _RANKING_METRIC_NAMES:
            raise RagEvidenceContractError(
                f"unsupported fast paired RAG metric: {metric_name}"
            )
        samples = _fast_grouped_paired_samples(
            target, left, right, group_values,
            draws=draws, seed=seed, metric_name=metric_name,
        )
    else:
        unique = np.unique(group_values)
        lookup = {group: np.flatnonzero(group_values == group) for group in unique}
        rng = np.random.default_rng(seed)
        samples = []
        for _ in range(int(draws)):
            selected = rng.choice(unique, size=len(unique), replace=True)
            indexes = np.concatenate([lookup[group] for group in selected])
            if len(np.unique(target[indexes])) < 2:
                continue
            samples.append(
                metric(target[indexes], left[indexes])
                - metric(target[indexes], right[indexes])
            )
    array = np.asarray(samples, dtype=float)
    return {
        "delta": float(metric(target, left) - metric(target, right)),
        "ci_low": float(np.quantile(array, 0.025)) if len(array) else "",
        "ci_high": float(np.quantile(array, 0.975)) if len(array) else "",
        "draws": len(array),
        "status": "OK" if len(array) else "BOOTSTRAP_UNDEFINED",
    }


def _panel(registry: Mapping[str, Any], panel_id: str) -> Mapping[str, Any]:
    return next(row for row in registry["panels"] if row["panel_id"] == panel_id)


def _metric_row(
    *, registry: Mapping[str, Any], panel_id: str, split: str, subgroup: str,
    method_id: str, metric: str, summary: Mapping[str, Any], n: int,
    groups: Sequence[str], positive_rate: str | float,
) -> dict[str, Any]:
    panel = _panel(registry, panel_id)
    return {
        "panel_id": panel_id,
        "dataset": panel["dataset"],
        "unit": panel["unit"],
        "access": panel["access"],
        "estimand": panel["estimand"],
        "split": split,
        "subgroup": subgroup,
        "method_id": method_id,
        "metric": metric,
        "value": summary["value"],
        "ci_low": summary["ci_low"],
        "ci_high": summary["ci_high"],
        "n": n,
        "n_groups": len(set(map(str, groups))),
        "positive_rate": positive_rate,
        "bootstrap_draws": summary["draws"],
        "status": summary["status"],
    }


def _binary_panel(
    *, registry: Mapping[str, Any], panel_id: str, split: str, subgroup: str,
    method_id: str, ids: np.ndarray, labels: np.ndarray, scores: np.ndarray,
    groups: np.ndarray, parent_ids: np.ndarray | None, draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    ids, groups = np.asarray(ids).astype(str), np.asarray(groups).astype(str)
    if not (len(ids) == len(labels) == len(scores) == len(groups)):
        raise RagEvidenceContractError(f"{panel_id}: score/label/group alignment failed")
    metrics = []
    for offset, metric_name in enumerate(("auroc", "auprc")):
        summary = grouped_interval(
            labels, scores, groups, _binary_metric(metric_name), draws=draws,
            seed=seed + offset, require_two_classes=True,
            metric_name=metric_name,
        )
        metrics.append(_metric_row(
            registry=registry, panel_id=panel_id, split=split, subgroup=subgroup,
            method_id=method_id, metric=metric_name, summary=summary, n=len(labels),
            groups=groups, positive_rate=float(labels.mean()),
        ))
    parents = ids if parent_ids is None else np.asarray(parent_ids).astype(str)
    predictions = [{
        "panel_id": panel_id, "split": split, "subgroup": subgroup,
        "method_id": method_id, "unit_id": unit_id, "parent_id": parent,
        "score": float(score), "prediction": "", "label": int(label),
        "bootstrap_group": group,
    } for unit_id, parent, score, label, group in zip(ids, parents, scores, labels, groups, strict=True)]
    return metrics, predictions


def _private_lookup(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output = {str(row["unit_id"]): row for row in rows}
    if len(output) != len(rows):
        raise RagEvidenceContractError("duplicate private RAG unit ID")
    return output


def _evaluate_ragtruth(
    *, registry: Mapping[str, Any], labels: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, predictions = [], []
    panel_by_unit = {
        "response": "ragtruth_evidence_contrast_answer",
        "sentence": "ragtruth_evidence_contrast_sentence",
        "token": "ragtruth_evidence_contrast_token",
    }
    for split_index, split in enumerate(("dev", "test")):
        private_rows = labels["splits"][split]
        response = _private_lookup(private_rows)
        response_ids = np.asarray(scores[f"rag_{split}_response_id"]).astype(str)
        if set(response_ids) != set(response):
            raise RagEvidenceContractError(f"RAGTruth {split} response roster drifted")
        response_y = np.asarray([response[item]["response_label"] for item in response_ids], int)
        response_groups = np.asarray([response[item]["source_id"] for item in response_ids], str)
        response_tasks = np.asarray(scores[f"rag_{split}_response_task"]).astype(str)
        expected_response_tasks = np.asarray(
            [response[item]["task_type"] for item in response_ids], str
        )
        if not np.array_equal(response_tasks, expected_response_tasks):
            raise RagEvidenceContractError(
                f"RAGTruth {split} response task/private binding drifted"
            )
        response_scores = np.asarray(scores[f"rag_{split}_response_score"], float)

        sentence_private = {
            sentence["unit_id"]: {**sentence, "source_id": row["source_id"], "task_type": row["task_type"]}
            for row in private_rows for sentence in row["sentence_labels"]
        }
        sentence_ids = np.asarray(scores[f"rag_{split}_sentence_id"]).astype(str)
        if set(sentence_ids) != set(sentence_private):
            raise RagEvidenceContractError(f"RAGTruth {split} sentence roster drifted")
        sentence_y = np.asarray([sentence_private[item]["label"] for item in sentence_ids], int)
        sentence_groups = np.asarray([sentence_private[item]["source_id"] for item in sentence_ids], str)
        sentence_tasks = np.asarray([sentence_private[item]["task_type"] for item in sentence_ids], str)
        sentence_scores = np.asarray(scores[f"rag_{split}_sentence_score"], float)

        token_parents = np.asarray(scores[f"rag_{split}_token_parent_id"]).astype(str)
        token_indexes = np.asarray(scores[f"rag_{split}_token_index"], int)
        observed_token_lattice = list(zip(
            token_parents.tolist(), token_indexes.tolist(), strict=True
        ))
        expected_token_lattice = [
            (str(row["unit_id"]), index)
            for row in private_rows
            for index in range(len(row["token_labels"]))
        ]
        if observed_token_lattice != expected_token_lattice:
            raise RagEvidenceContractError(
                f"RAGTruth {split} scorer-token lattice/private binding drifted"
            )
        token_ids = np.asarray([
            f"{parent}_t{index:05d}" for parent, index in zip(token_parents, token_indexes, strict=True)
        ], dtype="U")
        token_y = np.asarray([
            int(response[parent]["token_labels"][index])
            for parent, index in zip(token_parents, token_indexes, strict=True)
        ], int)
        token_groups = np.asarray([response[parent]["source_id"] for parent in token_parents], str)
        token_tasks = np.asarray([response[parent]["task_type"] for parent in token_parents], str)
        token_scores = np.asarray(scores[f"rag_{split}_token_score"], float)

        bundles = {
            "response": (response_ids, response_y, response_scores, response_groups, response_tasks, None),
            "sentence": (sentence_ids, sentence_y, sentence_scores, sentence_groups, sentence_tasks, None),
            "token": (token_ids, token_y, token_scores, token_groups, token_tasks, token_parents),
        }
        for unit_offset, (unit, bundle) in enumerate(bundles.items()):
            ids, target, values, groups, tasks, parents = bundle
            for subgroup_index, subgroup in enumerate(("all", *sorted(set(tasks.tolist())))):
                mask = np.ones(len(ids), bool) if subgroup == "all" else tasks == subgroup
                rows, preds = _binary_panel(
                    registry=registry, panel_id=panel_by_unit[unit], split=split,
                    subgroup=subgroup, method_id="fixed_rag_iu_pcr", ids=ids[mask],
                    labels=target[mask], scores=values[mask], groups=groups[mask],
                    parent_ids=None if parents is None else parents[mask], draws=draws,
                    seed=seed + split_index * 100 + unit_offset * 10 + subgroup_index,
                )
                metrics.extend(rows)
                # Keep a single tidy prediction roster; subgroup rows are
                # deterministic views of these same predictions.
                if subgroup == "all":
                    predictions.extend(preds)
    return metrics, predictions


def _evaluate_gasp(
    *, registry: Mapping[str, Any], labels: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    private = _private_lookup(labels["sentences"])
    ids = np.asarray(scores["gasp_sentence_id"]).astype(str)
    if set(ids) != set(private):
        raise RagEvidenceContractError("GASP score/private sentence roster drifted")
    target = np.asarray([private[item]["label"] for item in ids], int)
    groups = np.asarray([private[item]["source_id"] for item in ids], str)
    tasks = np.asarray(scores["gasp_task"]).astype(str)
    expected_tasks = np.asarray([private[item]["task_type"] for item in ids], str)
    if not np.array_equal(tasks, expected_tasks):
        raise RagEvidenceContractError("GASP task/private binding drifted")
    methods = {
        "gasp_threshold": np.asarray(scores["gasp_threshold_score"], float),
        "fixed_rag_iu_pcr_matched": np.asarray(scores["gasp_fixed_rag_score"], float),
    }
    metrics, predictions, contrasts = [], [], []
    for subgroup_index, subgroup in enumerate(("all", *sorted(set(tasks.tolist())))):
        mask = np.ones(len(ids), bool) if subgroup == "all" else tasks == subgroup
        for method_index, (method_id, values) in enumerate(methods.items()):
            rows, preds = _binary_panel(
                registry=registry, panel_id="gasp_protocol_sentence",
                split="local_400_response_sample", subgroup=subgroup,
                method_id=method_id, ids=ids[mask], labels=target[mask],
                scores=values[mask], groups=groups[mask], parent_ids=None,
                draws=draws, seed=seed + subgroup_index * 10 + method_index,
            )
            metrics.extend(rows)
            if subgroup == "all":
                predictions.extend(preds)
        for metric_index, metric_name in enumerate(("auroc", "auprc")):
            result = grouped_paired_delta(
                target[mask], methods["gasp_threshold"][mask],
                methods["fixed_rag_iu_pcr_matched"][mask], groups[mask],
                _binary_metric(metric_name), draws=draws,
                seed=seed + 100 + subgroup_index * 10 + metric_index,
                metric_name=metric_name,
            )
            contrasts.append({
                "panel_id": "gasp_protocol_sentence",
                "split": "local_400_response_sample", "subgroup": subgroup,
                "left_method": "gasp_threshold", "right_method": "fixed_rag_iu_pcr_matched",
                "metric": metric_name, "delta": result["delta"],
                "ci_low": result["ci_low"], "ci_high": result["ci_high"],
                "n": int(mask.sum()), "n_groups": len(set(groups[mask].tolist())),
                "bootstrap_draws": result["draws"], "status": result["status"],
            })
    return metrics, predictions, contrasts


def _evaluate_lettuce(
    *, registry: Mapping[str, Any], labels: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    private = _private_lookup(labels["rows"])
    ids = np.asarray(scores["lettuce_unit_id"]).astype(str)
    if set(ids) != set(private):
        raise RagEvidenceContractError("Lettuce score/private example roster drifted")
    target = np.asarray([private[item]["label"] for item in ids], int)
    groups = np.asarray([private[item]["source_id"] for item in ids], str)
    tasks = np.asarray([private[item]["task_type"] for item in ids], str)
    prediction = np.asarray(scores["lettuce_prediction"], int)
    metrics, predictions = [], []
    functions = {
        "f1": lambda y, p: float(f1_score(y, p, zero_division=0)),
        "precision": lambda y, p: float(precision_score(y, p, zero_division=0)),
        "recall": lambda y, p: float(recall_score(y, p, zero_division=0)),
    }
    for subgroup_index, subgroup in enumerate(("all", *sorted(set(tasks.tolist())))):
        mask = np.ones(len(ids), bool) if subgroup == "all" else tasks == subgroup
        for metric_index, (metric_name, function) in enumerate(functions.items()):
            summary = grouped_interval(
                target[mask], prediction[mask], groups[mask], function, draws=draws,
                seed=seed + subgroup_index * 10 + metric_index, require_two_classes=False,
                metric_name=metric_name,
            )
            metrics.append(_metric_row(
                registry=registry, panel_id="lettucedetect_example", split="test",
                subgroup=subgroup, method_id="lettucedetect_large_modernbert",
                metric=metric_name, summary=summary, n=int(mask.sum()), groups=groups[mask],
                positive_rate=float(target[mask].mean()),
            ))
        if subgroup == "all":
            probabilities = np.asarray(scores["lettuce_max_probability"], float)
            predictions.extend({
                "panel_id": "lettucedetect_example", "split": "test", "subgroup": "all",
                "method_id": "lettucedetect_large_modernbert", "unit_id": unit_id,
                "parent_id": unit_id, "score": float(probability),
                "prediction": int(pred), "label": int(label), "bootstrap_group": group,
            } for unit_id, probability, pred, label, group in zip(
                ids, probabilities, prediction, target, groups, strict=True
            ))
    return metrics, predictions


def _evaluate_refchecker(
    *, registry: Mapping[str, Any], labels: Mapping[str, Any], scores: Mapping[str, np.ndarray],
    draws: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    private = _private_lookup(labels["rows"])
    ids = np.asarray(scores["refchecker_unit_id"]).astype(str)
    if set(ids) != set(private):
        raise RagEvidenceContractError("RefChecker score/private claim roster drifted")
    setting = np.asarray(scores["refchecker_setting"]).astype(str)
    expected_setting = np.asarray([private[item]["setting"] for item in ids], str)
    if not np.array_equal(setting, expected_setting):
        raise RagEvidenceContractError("RefChecker setting binding drifted")
    groups = np.asarray([private[item]["example_id"] for item in ids], str)
    gold_threeway = np.asarray([private[item]["human_label"] for item in ids], str)
    gold_binary = np.asarray([private[item]["label_unsupported"] for item in ids], int)
    nli = np.asarray(scores["refchecker_nli_prediction"]).astype(str)
    binary_score = np.asarray(scores["refchecker_binary_score"], float)
    metrics, predictions = [], []
    if set(setting.tolist()) != set(REFCHECKER_SETTINGS):
        raise RagEvidenceContractError("RefChecker required setting coverage failed")
    for setting_index, subgroup in enumerate(REFCHECKER_SETTINGS):
        mask = setting == subgroup
        for metric_index, metric_name in enumerate(("accuracy", "macro_f1")):
            summary = grouped_interval(
                gold_threeway[mask], nli[mask], groups[mask], _threeway_metric(metric_name),
                draws=draws, seed=seed + setting_index * 20 + metric_index,
                require_two_classes=False,
                metric_name=metric_name,
            )
            metrics.append(_metric_row(
                registry=registry, panel_id="refchecker_threeway",
                split="official_fixed_claims", subgroup=subgroup,
                method_id="refchecker_nli", metric=metric_name, summary=summary,
                n=int(mask.sum()), groups=groups[mask], positive_rate="",
            ))
        rows, _ = _binary_panel(
            registry=registry, panel_id="refchecker_binary_claim",
            split="official_fixed_claims", subgroup=subgroup,
            method_id="fixed_rag_iu_pcr_transfer", ids=ids[mask],
            labels=gold_binary[mask], scores=binary_score[mask], groups=groups[mask],
            parent_ids=None, draws=draws, seed=seed + setting_index * 20 + 10,
        )
        metrics.extend(rows)
    # One prediction row per method/claim.  There is deliberately no pooled
    # RefChecker metric row; setting is carried as the subgroup.
    for index, unit_id in enumerate(ids):
        predictions.append({
            "panel_id": "refchecker_threeway", "split": "official_fixed_claims",
            "subgroup": setting[index], "method_id": "refchecker_nli",
            "unit_id": unit_id, "parent_id": unit_id, "score": "",
            "prediction": nli[index], "label": gold_threeway[index],
            "bootstrap_group": groups[index],
        })
        predictions.append({
            "panel_id": "refchecker_binary_claim", "split": "official_fixed_claims",
            "subgroup": setting[index], "method_id": "fixed_rag_iu_pcr_transfer",
            "unit_id": unit_id, "parent_id": unit_id, "score": float(binary_score[index]),
            "prediction": "", "label": int(gold_binary[index]),
            "bootstrap_group": groups[index],
        })
    return metrics, predictions


def compute_rag_evidence_evaluation_tables(
    *, registry: Mapping[str, Any], private: Mapping[str, Any],
    scores: Mapping[str, np.ndarray], draws: int, seed: int,
) -> dict[str, Any]:
    """Deterministically re-evaluate frozen scores from isolated private labels."""

    if draws <= 0:
        raise RagEvidenceContractError("RAG evaluation needs positive bootstrap draws")
    validate_score_arrays(scores)
    metrics, predictions = _evaluate_ragtruth(
        registry=registry, labels=private["ragtruth"], scores=scores,
        draws=draws, seed=seed,
    )
    gasp_metrics, gasp_predictions, contrasts = _evaluate_gasp(
        registry=registry, labels=private["gasp"], scores=scores,
        draws=draws, seed=seed + 1000,
    )
    lettuce_metrics, lettuce_predictions = _evaluate_lettuce(
        registry=registry, labels=private["lettuce"], scores=scores,
        draws=draws, seed=seed + 2000,
    )
    ref_metrics, ref_predictions = _evaluate_refchecker(
        registry=registry, labels=private["refchecker"], scores=scores,
        draws=draws, seed=seed + 3000,
    )
    metrics.extend(gasp_metrics + lettuce_metrics + ref_metrics)
    predictions.extend(gasp_predictions + lettuce_predictions + ref_predictions)
    if any(
        row["panel_id"].startswith("refchecker_") and row["subgroup"] == "all"
        for row in metrics
    ):
        raise RagEvidenceContractError("RefChecker pooled metric escaped the evaluator")
    if any(row.get("panel_id") not in PANEL_IDS for row in metrics):
        raise RagEvidenceContractError("unregistered panel escaped the RAG evaluator")
    if {row["panel_id"] for row in metrics} != set(PANEL_IDS):
        raise RagEvidenceContractError("RAG evaluation did not cover every registered panel")

    panel_status = []
    for panel_id in PANEL_IDS:
        rows = [row for row in metrics if row["panel_id"] == panel_id]
        status = "PASS" if rows and all(row["status"] in {"OK", "METRIC_UNDEFINED_SINGLE_CLASS"} for row in rows) else "FAIL"
        panel_status.append({
            "panel_id": panel_id,
            "status": status,
            "metric_rows": len(rows),
            "prediction_rows": sum(row["panel_id"] == panel_id for row in predictions),
            "cross_panel_macro_contribution": "FORBIDDEN",
        })
    if any(row["status"] != "PASS" for row in panel_status):
        raise RagEvidenceContractError("one or more RAG panel status gates failed")

    file_payloads = {
        "metrics.csv": _csv_bytes(metrics, METRIC_COLUMNS),
        "predictions.csv": _csv_bytes(predictions, PREDICTION_COLUMNS),
        "contrasts.csv": _csv_bytes(contrasts, CONTRAST_COLUMNS),
        "panel_status.csv": _csv_bytes(
            panel_status,
            (
                "panel_id", "status", "metric_rows", "prediction_rows",
                "cross_panel_macro_contribution",
            ),
        ),
    }
    return {
        "file_payloads": file_payloads,
        "metrics": metrics,
        "predictions": predictions,
        "contrasts": contrasts,
        "panel_status": panel_status,
    }


def evaluate_rag_evidence_build(
    *, repo: str | Path, score_verifier_repo: str | Path,
    registry_path: str | Path, source_root: str | Path,
    release_root: str | Path, private_root: str | Path,
    release_id: str, build_id: str, draws_override: int | None = None,
) -> dict[str, Any]:
    release_id = validate_artifact_identifier(release_id, name="RAG release ID")
    if build_id not in {"A", "B"}:
        raise RagEvidenceContractError("RAG evaluation build must be A or B")
    repo_path = Path(repo).resolve(strict=True)
    registry = load_registry(registry_path)
    lane_root = Path(release_root) / release_id / "rag_evidence"
    build_root = lane_root / build_id
    preparation_path = build_root / PREPARATION_MANIFEST_FILENAME
    score_manifest_path = build_root / "fit" / SCORE_MANIFEST_FILENAME
    preparation_payload = read_bound_file_bytes(
        preparation_path, name="RAG preparation manifest"
    )
    score_manifest_payload = read_bound_file_bytes(
        score_manifest_path, name="RAG score freeze"
    )
    preparation = json.loads(preparation_payload.decode("utf-8"))
    score_manifest = json.loads(score_manifest_payload.decode("utf-8"))
    verify_payload(preparation, name="RAG preparation manifest")
    verify_payload(score_manifest, name="RAG score freeze")
    # This gate must complete before the first private-label file open.
    score_authentication = authenticate_rag_evidence_score_certificate_from_repo(
        evaluation_repo=repo_path, score_verifier_repo=score_verifier_repo,
        registry_path=registry_path, source_root=source_root,
        release_root=release_root, private_root=private_root, release_id=release_id,
        require_scientific_full=draws_override is None,
    )
    score_certificate = score_authentication["certificate"]
    if (
        score_certificate["score_sha256"] != score_manifest["scores"]["sha256"]
        or score_certificate["private_label_sha256"]
        != preparation["private_labels"]["sha256"]
    ):
        raise RagEvidenceContractError("RAG evaluation is underbound to score A/B")
    score_certificate_payload = score_authentication["certificate_payload"]
    validate_source_binding(
        preparation["source_binding"], source_root=source_root, registry=registry
    )
    private_path = Path(preparation["private_labels"]["path"])
    private = load_private_labels(
        private_path,
        registry,
        expected_sha256=score_certificate["private_label_sha256"],
    )
    score_path = build_root / "fit" / score_manifest["scores"]["path"]
    scores = load_scores(
        score_path,
        expected_sha256=score_certificate["score_sha256"],
    )
    configured_draws = int(registry["evaluation"]["bootstrap"]["draws"])
    draws = configured_draws if draws_override is None else int(draws_override)
    if draws <= 0 or (draws_override is None and draws != 20_000):
        raise RagEvidenceContractError("RAG evaluation bootstrap draw contract failed")
    seed = int(registry["evaluation"]["bootstrap"]["seed"])
    derived = compute_rag_evidence_evaluation_tables(
        registry=registry, private=private, scores=scores, draws=draws, seed=seed
    )

    evaluation_final = build_root / "evaluation"
    stage = AtomicRagDirectory(evaluation_final)
    try:
        files = []
        for name, payload in derived["file_payloads"].items():
            digest = stage.write_bytes(name, payload)
            files.append({"path": name, "sha256": digest, "size_bytes": len(payload)})
        evaluation_repo_snapshot = score_authentication[
            "evaluation_repo_snapshot"
        ]
        score_verifier_repo_snapshot = score_authentication[
            "score_verifier_repo_snapshot"
        ]
        source_snapshot = {
            "files": evaluation_repo_snapshot["source_files"],
        }
        source_snapshot["snapshot_sha256"] = payload_sha256(source_snapshot)
        manifest = add_payload_sha256({
            "schema_version": EVALUATION_SCHEMA,
            "release_id": release_id,
            "build_id": build_id,
            "lane_id": registry["lane_id"],
            "scientific_full": draws_override is None,
            "score_ab_certificate_sha256": sha256_bytes(score_certificate_payload),
            "score_manifest_sha256": sha256_bytes(score_manifest_payload),
            "score_sha256": score_manifest["scores"]["sha256"],
            "private_label_sha256": preparation["private_labels"]["sha256"],
            "source_binding_sha256": preparation["source_binding_sha256"],
            "source_snapshot": source_snapshot,
            "evaluation_repo_snapshot": evaluation_repo_snapshot,
            "score_verifier_repo_snapshot": score_verifier_repo_snapshot,
            "bootstrap": {
                "draws_requested": draws,
                "group": "panel-registered source group",
                "paired_contrasts": True,
                "seed": seed,
            },
            "files": files,
            "panel_status": derived["panel_status"],
            "cross_panel_macro_computed": False,
            "refchecker_settings_pooled": False,
            "historical_scores_copied": False,
        })
        stage.write_json(EVALUATION_MANIFEST_FILENAME, manifest)
        if capture_evaluation_repository_snapshot(
            repo_path, require_scientific_full=draws_override is None,
        ) != evaluation_repo_snapshot:
            raise RagEvidenceContractError(
                "RAG evaluation repository changed before publication"
            )
        if capture_score_verifier_repository_snapshot(
            score_verifier_repo
        ) != score_verifier_repo_snapshot:
            raise RagEvidenceContractError(
                "RAG score-verifier repository changed before publication"
            )
        stage.commit()
        return manifest
    finally:
        stage.cleanup()


__all__ = [
    "CONTRAST_COLUMNS", "EVALUATION_SOURCE_FILES", "METRIC_COLUMNS",
    "PREDICTION_COLUMNS", "compute_rag_evidence_evaluation_tables",
    "SCORE_VERIFIER_GIT_HEAD", "SCORE_VERIFIER_SOURCE_SHA256",
    "authenticate_rag_evidence_score_certificate_from_repo",
    "capture_evaluation_repository_snapshot",
    "capture_score_verifier_repository_snapshot", "evaluate_rag_evidence_build",
    "grouped_interval", "grouped_paired_delta",
]
