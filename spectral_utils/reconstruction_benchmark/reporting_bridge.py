"""Strict bridge from frozen-24 evaluation artifacts to reporting schema v1.2.

The scientific evaluator is the only component allowed to open the historical
label bundle.  This module deliberately accepts only its immutable
``EVALUATION.json`` and ``PREDICTION_SNAPSHOT.npz`` outputs, plus the three
frozen registries that describe methods, cells, and preprocessing.  It never
parses a score file, a fit artifact, or a label bundle.  When a signed graph-
diagnostic package is supplied, its bound fit artifacts are re-hashed as opaque
provenance inputs, but their contents are not reused to compute scores.  The
historical raw label bundle is never opened.

The bridge is intentionally more than a format converter.  It re-hashes the
evaluation publication, checks the exact 24 x 13 x 2 ledgers, independently
recomputes row-level point metrics from the snapshot, binds every cohort to its
row/group identities, and produces comparison groups that cannot mix AUROC and
AUPRC.  Unknown scientific statuses or incomplete provenance fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import canonical_sha256 as method_config_sha256
from .io import canonical_json_bytes as evaluator_json_bytes
from .io import sha256_bytes, sha256_file
from .methods import PRIMARY_METHOD_IDS, PRIMARY_METHOD_SPECS
from ..reconstruction_reporting.io import (
    write_canonical_json,
    write_tidy_csv,
)
from ..reconstruction_reporting.registry import (
    build_registry,
    expected_coverage_rows,
    make_system_id,
    validate_result_references,
)
from ..reconstruction_reporting.schemas import (
    RANKABLE_STATUSES,
    canonical_json_bytes as reporting_json_bytes,
    canonical_sha256,
    derive_aggregate_cohort_id,
    derive_cohort_id,
    derive_comparison_group_id,
    record_sort_key,
    table_sha256,
    validate_comparison_groups,
    validate_equal_unit_aggregates,
    validate_expected_coverage,
    validate_records,
)


EVALUATION_MANIFEST_SCHEMA = "reconstruction-evaluation-manifest-v1"
EVALUATION_SCHEMA = "reconstruction-24cell-evaluation-v1"
PREDICTION_SNAPSHOT_SCHEMA = "reconstruction-prediction-snapshot-v1"
CELL_REGISTRY_SCHEMA = "reconstruction-frozen24-cell-registry-v1"
METHOD_REGISTRY_SCHEMA = "reconstruction-method-registry-v1"
FEATURE_REGISTRY_SCHEMA = "reconstruction-feature-contract-v1"
BRIDGE_SCHEMA = "reconstruction-24cell-reporting-bridge-v1"
GRAPH_MANIFEST_SCHEMA = "reconstruction-graph-diagnostics-manifest-v2"
GRAPH_PAYLOAD_SCHEMA = "reconstruction-graph-assumption-diagnostics-v2"
GRAPH_DIAGNOSTIC_VERSION = "frozen24-graph-assumption-diagnostics-v2"
GRAPH_PLOT_SCHEMA = "reconstruction-graph-diagnostic-plot-data-v3"
GRAPH_EXAMPLE_SCHEMA = "reconstruction-example-graph-data-v2"
GRAPH_EXAMPLE_RULE = "nuisance_available_then_connected_then_no_isolates_then_max_gap_then_min_degree_cv_then_cell_id_v2"
GRAPH_NODE_PERMUTATIONS = 32
GRAPH_METHOD_IDS = ("dufs_liu", "ca_specrage_atomic", "pgrd_a")
NONGRAPH_DIAGNOSTIC_METHOD_IDS = ("continuous_lsml", "family_nrm_a", "su_pcr")
DIAGNOSTIC_METHOD_IDS = GRAPH_METHOD_IDS + NONGRAPH_DIAGNOSTIC_METHOD_IDS
CA_CONTROL_SERIES = ("learned", "equal_view", "provenance_prior", "global_mean_alpha", "permuted")
_COMMON_GRAPH_PANELS = (
    "graph_health",
    "target_vs_nuisance_roughness",
    "node_permutation_null",
    "roughness_null_summary",
    "alignment_vs_improvement",
    "fixed_graph_group_bootstrap_stability",
    "length_only_graph_control",
    "random_family_graph_control",
)
REQUIRED_GRAPH_PANELS_BY_METHOD: Mapping[str, tuple[str, ...]] = {
    "dufs_liu": _COMMON_GRAPH_PANELS + (
        "dufs_gate_weights",
        "dufs_gate_weights_per_seed",
        "dufs_gate_stability",
        "dufs_seed_graph_stability",
    ),
    "ca_specrage_atomic": _COMMON_GRAPH_PANELS + (
        "ca_view_weights",
        "ca_alpha_stability",
        "ca_seed_graph_stability",
        "ca_alpha_controls",
    ),
    "pgrd_a": _COMMON_GRAPH_PANELS + (
        "pgrd_seed_graph_stability",
        "pgrd_cross_gradient",
        "pgrd_cross_gradient_null",
    ),
    "continuous_lsml": (
        "continuous_lsml_cluster_boundaries",
        "continuous_lsml_correlation_clusters",
    ),
    "family_nrm_a": (
        "family_nrm_residual_covariance",
        "family_nrm_residual_eigenspectrum",
        "family_nrm_family_contributions",
    ),
    "su_pcr": (
        "su_pcr_decomposition",
        "su_pcr_low_rank_eigenspectrum",
        "su_pcr_sparse_support",
        "su_pcr_sparse_support_stability",
    ),
}

TASK_ID = "final_answer_detection"
LANE_ID = "static_fusion_frozen24"
ADAPTER_ID = "frozen24_response_score_adapter_v1"
ACCESS_CONTRACT_ID = "gray_box_single_pass_saved_telemetry_v1"
EVALUATOR_ID = "reconstruction_24cell_grouped_evaluator_v1"
EVIDENCE_GRADE = "D0"
FIDELITY = "retrospective-common-matrix"
SOURCE_POPULATION_ID = "frozen24_response_v1"
SUITE_DATASET_ID = "frozen24_suite"
REFERENCE_METHOD_ID = "iu_pcr"
METRICS = ("auroc", "auprc")
BOOTSTRAP_DRAWS = 20_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")


class ReportingBridgeError(RuntimeError):
    """The evaluator-to-reporting boundary is incomplete or inconsistent."""


@dataclass(frozen=True)
class AuxiliaryArtifact:
    """One verified producer artifact copied alongside reporting inputs."""

    relative_path: str
    source_path: Path
    file_sha256: str
    kind: str


@dataclass(frozen=True)
class BridgeInputs:
    """Validated reporting inputs ready for the generic release builder."""

    registry: Mapping[str, Any]
    rows: Mapping[str, tuple[dict[str, Any], ...]]
    source_provenance: Mapping[str, Any]
    auxiliary_artifacts: tuple[AuxiliaryArtifact, ...] = ()


@dataclass(frozen=True)
class VerifiedGraphPackage:
    """Hash- and source-verified graph diagnostic publication."""

    manifest: Mapping[str, Any]
    payload: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]
    root: Path
    release_root: Path
    plot_path: Path
    example_path: Path
    example_arrays: Mapping[str, np.ndarray]
    auxiliary_artifacts: tuple[AuxiliaryArtifact, ...]
    provenance: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportingBridgeError(message)


def _load_json(path: Path, *, canonical: bool = True) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ReportingBridgeError(f"duplicate JSON key {key!r} in {path}")
            output[key] = value
        return output

    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=no_duplicates)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportingBridgeError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"expected a JSON object: {path}")
    if canonical:
        _require(
            raw == evaluator_json_bytes(value) + b"\n",
            f"evaluation artifact is not canonical JSON: {path}",
        )
    return value


def _payload_hash(value: Mapping[str, Any], field: str, *, context: str) -> str:
    declared = value.get(field)
    _require(
        isinstance(declared, str) and _SHA256.fullmatch(declared) is not None,
        f"{context}: missing or invalid {field}",
    )
    body = dict(value)
    body.pop(field, None)
    observed = sha256_bytes(evaluator_json_bytes(body))
    _require(observed == declared, f"{context}: {field} mismatch")
    return observed


def _resolve_artifact(root: Path, value: Any, *, context: str) -> Path:
    _require(isinstance(value, str) and value.strip(), f"{context}: invalid path")
    relative = Path(value)
    _require(not relative.is_absolute(), f"{context}: path must be relative")
    base = root.resolve()
    resolved = (root / relative).resolve()
    _require(base in resolved.parents, f"{context}: path escapes evaluation directory")
    _require(resolved.is_file(), f"{context}: missing artifact {resolved}")
    return resolved


def _hash_array(values: np.ndarray, dtype: str) -> str:
    payload = np.ascontiguousarray(np.asarray(values, dtype=dtype)).tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ReportingBridgeError(f"method configuration contains unsupported {type(value).__name__}")


def _token(value: str) -> str:
    rendered = _SAFE_TOKEN.sub("-", str(value)).strip("-")
    _require(bool(rendered), f"cannot derive an ID token from {value!r}")
    return rendered


def _float_equal(left: Any, right: Any, *, tolerance: float = 1e-12) -> bool:
    if left is None or right is None:
        return left is right
    return math.isfinite(float(left)) and math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance


def _auroc(y_error: np.ndarray, score: np.ndarray) -> float:
    """Exact unweighted Mann-Whitney AUROC with average ranks for ties."""

    y = np.asarray(y_error, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    n_positive = int(y.sum())
    n_negative = int(len(y) - n_positive)
    _require(n_positive > 0 and n_negative > 0, "AUROC requires both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    start = 0
    while start < len(s):
        end = start + 1
        while end < len(s) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[y == 1].sum())
    return (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _auprc(y_error: np.ndarray, score: np.ndarray) -> float:
    """Non-interpolated average precision with threshold ties grouped."""

    y = np.asarray(y_error, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    n_positive = int(y.sum())
    _require(n_positive > 0, "AUPRC requires at least one positive")
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    seen = 0
    true_positive = 0
    average_precision = 0.0
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and s_sorted[end] == s_sorted[start]:
            end += 1
        positives_here = int(y_sorted[start:end].sum())
        seen += end - start
        true_positive += positives_here
        average_precision += (positives_here / n_positive) * (true_positive / seen)
        start = end
    return float(average_precision)


_METHOD_GUIDE: Mapping[str, Mapping[str, Any]] = {
    "equal_feature_mean": {
        "acronym": "Equal-feature mean (descriptive baseline; no acronym)",
        "family": "equal_weight_baselines",
        "origin": ("project control", "Equal-feature reference floor", 2026, "Project baseline, not a published detector."),
        "history": "Introduced as the simplest check on whether learned weighting adds value under the same mixed-v2 matrix.",
        "references": (),
    },
    "equal_family_mean": {
        "acronym": "Equal-family mean (descriptive baseline; no acronym)",
        "family": "equal_weight_baselines",
        "origin": ("project control", "Equal-family reference floor", 2026, "Project baseline using the frozen six-family provenance map."),
        "history": "Added to prevent a provenance family with many columns from receiving more prior weight solely because it is larger.",
        "references": (),
    },
    "continuous_lsml": {
        "acronym": "Continuous L-SML (continuous Latent Spectral Meta-Learner adaptation)",
        "family": "spectral_meta_learning",
        "origin": ("project adaptation", "Unsupervised Ensemble Learning with Dependent Classifiers", 2016, "Adapts the dependent-classifier L-SML hierarchy to continuous measurements."),
        "history": "The project retained score magnitude rather than median-binarizing each measurement, then applied the two-level dependence grouping to the full prepared pool.",
        "references": (("L-SML", "Jaffe, Fetaya, Nadler, Jiang, and Kluger, 2016", "https://arxiv.org/abs/1510.05830"),),
    },
    "dufs_pf_lsml": {
        "acronym": "DUFS = Differentiable Unsupervised Feature Selection; L-SML = continuous latent spectral fusion",
        "family": "dufs_then_lsml",
        "origin": ("project integration", "Differentiable Unsupervised Feature Selection based on a Gated Laplacian", 2021, "Uses DUFS Eq. 7 selection before the project's continuous L-SML scorer."),
        "history": "Built after the advisors requested a label-free replacement for hand-picked feature subsets; this arm removes the tunable sparsity choice.",
        "references": (("DUFS", "Lindenbaum, Shaham, Svirsky, Peterfreund, and Kluger, 2021", "https://arxiv.org/abs/2007.04728"),),
    },
    "dufs_stability_lsml": {
        "acronym": "DUFS = Differentiable Unsupervised Feature Selection; L-SML = continuous latent spectral fusion",
        "family": "dufs_then_lsml",
        "origin": ("project integration", "DUFS stability selection", 2026, "Adds a label-free seed-stability rule around DUFS, then uses continuous L-SML."),
        "history": "Developed to choose among DUFS penalties without correctness labels; the selected columns must agree across five seeds.",
        "references": (("DUFS", "Lindenbaum, Shaham, Svirsky, Peterfreund, and Kluger, 2021", "https://arxiv.org/abs/2007.04728"),),
    },
    "upcr": {
        "acronym": "U-PCR = Unsupervised Principal Component Regression",
        "family": "unsupervised_ensemble_regression",
        "origin": ("published method with project deployment choices", "Unsupervised Ensemble Regression", 2017, "Implements U-PCR's covariance estimator with the project's documented exclusion/refit path."),
        "history": "Reimplemented after the feature-selection study showed that U-PCR's weak-view exclusion is itself a label-free selector; mixed-v2 now supplies the only feature orientation.",
        "references": (("U-PCR", "Dror, Nadler, Bilal, and Kluger, 2017", "https://arxiv.org/abs/1703.02965"),),
    },
    "iu_pcr": {
        "acronym": "IU-PCR = independent-error U-PCR variant using the full feature pool",
        "family": "unsupervised_ensemble_regression",
        "origin": ("published-line project control", "Crowdsourcing Regression: A Spectral Approach", 2022, "Uses the independent-error covariance model and the project's full-pool two-component head."),
        "history": "Frozen as the central non-graph control so every selector, dependence, or graph increment can be measured against the same PCR base.",
        "references": (("IU-PCR and SU-PCR", "Tenzer, Dror, Nadler, Bilal, and Kluger, AISTATS 2022", "https://proceedings.mlr.press/v151/tenzer22a.html"),),
    },
    "dufs_liu": {
        "acronym": "DUFS-LIU = DUFS-gated Laplacian-regularized IU-PCR",
        "family": "laplacian_regularized_pcr",
        "origin": ("project method", "DUFS-LIU mixed-v2", 2026, "Integrates DUFS sample geometry with a Laplacian penalty inside the IU-PCR solve."),
        "history": "Developed as the graph-based extension of the full-pool IU-PCR control; later audits showed that graph stability alone does not prove correctness alignment.",
        "references": (("DUFS", "Lindenbaum et al., NeurIPS 2021", "https://arxiv.org/abs/2007.04728"), ("IU-PCR", "Tenzer et al., AISTATS 2022", "https://proceedings.mlr.press/v151/tenzer22a.html")),
    },
    "su_pcr": {
        "acronym": "SU-PCR = sparse-error U-PCR",
        "family": "unsupervised_ensemble_regression",
        "origin": ("published method reproduction", "Crowdsourcing Regression: A Spectral Approach", 2022, "Reproduces the low-rank-plus-sparse covariance mechanism before PCR weighting."),
        "history": "Added after the literature audit found that the 2022 continuation of U-PCR had already published the sparse-dependence extension.",
        "references": (("SU-PCR", "Tenzer, Dror, Nadler, Bilal, and Kluger, AISTATS 2022", "https://proceedings.mlr.press/v151/tenzer22a.html"),),
    },
    "ca_specrage_atomic": {
        "acronym": "CA-SpecRaGE = conditional-agreement spectral representation graph ensemble, atomic-view adaptation",
        "family": "agreement_graph_pcr",
        "origin": ("project adaptation", "CA-SpecRaGE atomic", 2026, "A provenance-balanced atomic conditional-agreement adaptation, not the historical LOCO-micro arm."),
        "history": "Built after graph studies suggested comparing several definitions of sample geometry; it learns view-agreement weights before the same LIU head.",
        "references": (),
    },
    "deem_b3": {
        "acronym": "DEEM = Deep Energy-based Ensemble Method; B3 is the project's continuous additive adapter",
        "family": "deep_energy_ensemble",
        "origin": ("project adaptation", "Unsupervised Ensemble Learning Through Deep Energy-based Models", 2026, "Changes the paper's hard multinomial observation model to a continuous additive, graph-free model; not a direct reproduction."),
        "history": "Separated from Residual-Graph DEEM after that graph extension stopped at its synthetic gate; B3 remains the continuous graph-free candidate.",
        "references": (("DEEM", "Maymon, Buznah, and Shaham, AISTATS 2026", "https://arxiv.org/abs/2601.20556"),),
    },
    "family_nrm_a": {
        "acronym": "Family-NRM-A = within-cell Family Neutral Residual Mode, regime A",
        "family": "family_residual_correction",
        "origin": ("new project ablation", "Family Neutral Residual Mode", 2026, "New donor-free within-cell ablation of the earlier Family-NRM direction."),
        "history": "The earlier Family-NRM result used transfer across populations. Regime A asks whether the family residual signal itself survives when every representation and rule is derived inside the target cell without donors or labels.",
        "references": (),
    },
    "pgrd_a": {
        "acronym": "PGRD-A = within-cell Pooled Graph-Roughness Direction, regime A",
        "family": "residual_graph_correction",
        "origin": ("new project ablation", "Pooled Graph-Roughness Direction", 2026, "New donor-free within-cell residual-graph diagnostic and correction."),
        "history": "Developed from the Family-NRM residual representation to test whether a residual-space graph contains a useful direction without donor datasets or correctness labels.",
        "references": (),
    },
}


_DATASETS: Mapping[str, Mapping[str, str]] = {
    "triviaqa": {"name": "TriviaQA", "description": "Open-domain factual question answering; the prediction unit is one generated answer.", "family": "factual_qa", "source": "Joshi et al., TriviaQA, 2017"},
    "hotpotqa": {"name": "HotpotQA", "description": "Multi-hop question answering, retained here as a difficult QA cell rather than a separate RAG claim.", "family": "multi_hop_qa", "source": "Yang et al., HotpotQA, 2018"},
    "sciq": {"name": "SciQ", "description": "Science question answering; the detector predicts whether one generated answer is incorrect.", "family": "science_qa", "source": "Welbl et al., SciQ, 2017"},
    "nq_open": {"name": "Natural Questions Open", "description": "Open-domain short-answer questions evaluated on saved candidate generations.", "family": "factual_qa", "source": "Kwiatkowski et al., Natural Questions, 2019"},
    "squad_v2": {"name": "SQuAD v2", "description": "Reading-comprehension questions including unanswerable items; the unit is one saved generated answer.", "family": "reading_comprehension", "source": "Rajpurkar, Jia, and Liang, SQuAD 2.0, 2018"},
    "truthfulqa": {"name": "TruthfulQA", "description": "Questions designed to elicit common falsehoods; the unit is one saved candidate answer.", "family": "truthfulness_qa", "source": "Lin, Hilton, and Evans, TruthfulQA, 2022"},
    "gsm8k": {"name": "GSM8K", "description": "Grade-school mathematical reasoning; the unit is one complete generated reasoning trace.", "family": "mathematical_reasoning", "source": "Cobbe et al., GSM8K, 2021"},
    "math500": {"name": "MATH-500", "description": "A 500-problem subset of Hendrycks MATH; the unit is one complete generated solution trace.", "family": "mathematical_reasoning", "source": "Hendrycks et al., MATH, 2021"},
}


def _verify_publication(evaluation_dir: Path) -> tuple[dict[str, Any], dict[str, Any], Path, dict[str, str]]:
    manifest_path = evaluation_dir / "EVALUATION_MANIFEST.json"
    manifest = _load_json(manifest_path)
    _payload_hash(manifest, "payload_sha256", context="evaluation manifest")
    _require(manifest.get("schema_version") == EVALUATION_MANIFEST_SCHEMA, "evaluation manifest schema drift")

    evaluation_path = _resolve_artifact(evaluation_dir, manifest.get("evaluation_path"), context="evaluation")
    bootstrap_path = _resolve_artifact(evaluation_dir, manifest.get("bootstrap_path"), context="bootstrap")
    prediction_path = _resolve_artifact(evaluation_dir, manifest.get("prediction_snapshot_path"), context="prediction snapshot")
    expected_hashes = {
        "evaluation": manifest.get("evaluation_sha256"),
        "bootstrap": manifest.get("bootstrap_sha256"),
        "prediction_snapshot": manifest.get("prediction_snapshot_sha256"),
    }
    for name, digest in expected_hashes.items():
        _require(isinstance(digest, str) and _SHA256.fullmatch(digest) is not None, f"manifest has invalid {name} hash")
    _require(sha256_file(evaluation_path) == expected_hashes["evaluation"], "EVALUATION.json file hash drift")
    _require(sha256_file(bootstrap_path) == expected_hashes["bootstrap"], "BOOTSTRAP_DRAWS.npz file hash drift")
    _require(sha256_file(prediction_path) == expected_hashes["prediction_snapshot"], "PREDICTION_SNAPSHOT.npz file hash drift")
    _require(manifest.get("prediction_snapshot_schema") == PREDICTION_SNAPSHOT_SCHEMA, "prediction snapshot schema drift")

    evaluation = _load_json(evaluation_path)
    _payload_hash(evaluation, "payload_sha256", context="evaluation")
    _require(evaluation.get("schema_version") == EVALUATION_SCHEMA, "evaluation schema drift")
    for field in ("status", "headline_status", "population_id", "n_cells", "n_methods"):
        _require(manifest.get(field) == evaluation.get(field), f"manifest/evaluation {field} drift")
    _require(manifest.get("bootstrap_draws") == evaluation.get("bootstrap", {}).get("draws"), "bootstrap draw declaration drift")
    _require(manifest.get("canonical_bootstrap_draws") == BOOTSTRAP_DRAWS, "canonical bootstrap declaration drift")
    _require(manifest.get("input_provenance") == evaluation.get("provenance"), "manifest/evaluation provenance drift")
    return manifest, evaluation, prediction_path, {
        "evaluation_manifest_sha256": sha256_file(manifest_path),
        "evaluation_sha256": expected_hashes["evaluation"],
        "bootstrap_sha256": expected_hashes["bootstrap"],
        "prediction_snapshot_sha256": expected_hashes["prediction_snapshot"],
    }


def _load_registries(
    evaluation: Mapping[str, Any],
    cell_registry_path: Path,
    method_registry_path: Path,
    feature_registry_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    cells = _load_json(cell_registry_path, canonical=False)
    methods = _load_json(method_registry_path, canonical=False)
    feature = _load_json(feature_registry_path, canonical=False)
    _require(cells.get("schema_version") == CELL_REGISTRY_SCHEMA, "cell registry schema drift")
    _require(methods.get("schema_version") == METHOD_REGISTRY_SCHEMA, "method registry schema drift")
    _require(feature.get("schema_version") == FEATURE_REGISTRY_SCHEMA, "feature registry schema drift")
    _require(cells.get("population_id") == SOURCE_POPULATION_ID, "source population drift")
    _require(methods.get("score_semantics") == "higher_is_incorrect", "method score semantics drift")
    _require(feature.get("score_semantics") == "every system output is converted once to higher_is_incorrect", "feature score semantics drift")
    _require(feature.get("preprocessing_count") == 1, "feature preprocessing count drift")

    cell_rows = cells.get("cells")
    method_rows = methods.get("methods")
    _require(isinstance(cell_rows, list) and len(cell_rows) == 24, "cell registry must contain exactly 24 cells")
    _require(isinstance(method_rows, list) and len(method_rows) == 13, "method registry must contain exactly 13 methods")
    cell_ids = [row.get("cell_id") for row in cell_rows if isinstance(row, dict)]
    method_ids = [row.get("method_id") for row in method_rows if isinstance(row, dict)]
    _require(len(cell_ids) == len(set(cell_ids)) == 24, "cell registry IDs are not unique")
    _require(tuple(method_ids) == tuple(PRIMARY_METHOD_IDS), "method registry order/roster drift")
    _require(set(_METHOD_GUIDE) == set(method_ids), "method guide does not cover the exact roster")

    hashes = {
        "cell_registry_sha256": sha256_file(cell_registry_path),
        "method_registry_sha256": sha256_file(method_registry_path),
        "feature_registry_sha256": sha256_file(feature_registry_path),
    }
    provenance = evaluation.get("provenance")
    _require(isinstance(provenance, Mapping), "evaluation lacks provenance")
    _require(provenance.get("cell_registry_sha256") == hashes["cell_registry_sha256"], "evaluation/cell registry hash drift")
    _require(provenance.get("method_registry_sha256") == hashes["method_registry_sha256"], "evaluation/method registry hash drift")
    _require(provenance.get("labels_opened") is True, "evaluation did not cross the explicit label gate")
    _require(provenance.get("verified_cell_method_pairs") == 312, "evaluation did not verify all 312 fits")
    for field in (
        "label_bundle_sha256", "group_manifest_sha256", "score_ab_verification_sha256",
        "freeze_A_sha256", "freeze_B_sha256", "input_manifest_A_sha256",
        "input_manifest_B_sha256", "evaluation_module_sha256",
    ):
        _require(isinstance(provenance.get(field), str) and _SHA256.fullmatch(provenance[field]) is not None, f"evaluation provenance lacks {field}")
    return cells, methods, feature, hashes


def _load_snapshot(
    path: Path,
    cell_ids: Sequence[str],
    method_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    expected = set()
    for cell_id in cell_ids:
        expected.update((f"{cell_id}__row_ids", f"{cell_id}__group_ids", f"{cell_id}__y_error"))
        expected.update(f"{cell_id}__{method_id}__score" for method_id in method_ids)
    try:
        with np.load(path, allow_pickle=False) as bundle:
            _require(set(bundle.files) == expected, "prediction snapshot member roster drift")
            output: dict[str, dict[str, Any]] = {}
            for cell_id in cell_ids:
                row_ids = tuple(str(value) for value in np.asarray(bundle[f"{cell_id}__row_ids"]).tolist())
                group_ids = tuple(str(value) for value in np.asarray(bundle[f"{cell_id}__group_ids"]).tolist())
                y_error = np.asarray(bundle[f"{cell_id}__y_error"], dtype=np.int8)
                _require(row_ids and len(row_ids) == len(set(row_ids)), f"invalid/duplicate row IDs: {cell_id}")
                _require(all(value and value == value.strip() for value in row_ids), f"invalid row ID: {cell_id}")
                _require(len(group_ids) == len(row_ids) and all(value and value == value.strip() for value in group_ids), f"invalid group IDs: {cell_id}")
                _require(y_error.shape == (len(row_ids),) and np.isin(y_error, (0, 1)).all(), f"invalid error labels: {cell_id}")
                scores: dict[str, np.ndarray] = {}
                for method_id in method_ids:
                    values = np.asarray(bundle[f"{cell_id}__{method_id}__score"], dtype=np.float64)
                    _require(values.shape == y_error.shape and np.isfinite(values).all(), f"invalid score array: {cell_id}/{method_id}")
                    scores[method_id] = values
                output[cell_id] = {
                    "row_ids": row_ids,
                    "group_ids": group_ids,
                    "y_error": y_error,
                    "scores": scores,
                }
    except ReportingBridgeError:
        raise
    except Exception as exc:
        raise ReportingBridgeError(f"cannot read prediction snapshot {path}: {exc}") from exc
    return output


def _load_npz(path: Path, *, context: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as bundle:
            return {name: np.asarray(bundle[name]) for name in bundle.files}
    except Exception as exc:
        raise ReportingBridgeError(f"cannot read {context} {path}: {exc}") from exc


def _verify_release_file(
    release_root: Path,
    relative_path: Any,
    declared_hash: Any,
    *,
    context: str,
) -> tuple[Path, str]:
    path = _resolve_artifact(release_root, relative_path, context=context)
    _require(
        isinstance(declared_hash, str) and _SHA256.fullmatch(declared_hash) is not None,
        f"{context}: invalid declared SHA-256",
    )
    observed = sha256_file(path)
    _require(observed == declared_hash, f"{context}: file SHA-256 mismatch")
    return path, observed


def _verify_graph_tree(root: Path) -> Path:
    tree_path = root / "TREE_MANIFEST.json"
    tree = _load_json(tree_path)
    _require(tree.get("schema_version") == "canonical-tree-manifest-v1", "graph tree manifest schema drift")
    rows = tree.get("files")
    _require(isinstance(rows, list), "graph tree manifest files must be a list")
    expected_names = {
        "GRAPH_DIAGNOSTICS.json",
        "GRAPH_DIAGNOSTICS_MANIFEST.json",
        "PLOT_DATA.npz",
        "EXAMPLE_GRAPH_DATA.npz",
    }
    _require({row.get("path") for row in rows if isinstance(row, Mapping)} == expected_names, "graph tree file roster drift")
    _require(len(rows) == len(expected_names), "graph tree contains duplicate or non-object rows")
    for row in rows:
        _require(isinstance(row, Mapping), "graph tree contains a non-object row")
        path = _resolve_artifact(root, row.get("path"), context="graph tree artifact")
        _require(row.get("bytes") == path.stat().st_size, f"graph tree size drift: {row.get('path')}")
        _require(sha256_file(path) == row.get("sha256"), f"graph tree hash drift: {row.get('path')}")
    _require(
        tree.get("tree_sha256") == sha256_bytes(evaluator_json_bytes(rows)),
        "graph tree payload hash drift",
    )
    return tree_path


def _verify_graph_source_snapshot(
    manifest: Mapping[str, Any],
    payload: Mapping[str, Any],
    repo_root: Path,
) -> str:
    """Verify the producer's signed source snapshot without rerunning it."""

    snapshot = manifest.get("source_environment_snapshot")
    _require(isinstance(snapshot, Mapping), "graph producer source snapshot is missing")
    _require(
        payload.get("producer_source_environment_snapshot") == snapshot,
        "graph producer source snapshot manifest/payload drift",
    )
    _require(
        snapshot.get("schema_version") == "graph-diagnostics-source-environment-snapshot-v1",
        "graph producer source snapshot schema drift",
    )
    snapshot_body = dict(snapshot)
    declared_snapshot_hash = snapshot_body.pop("snapshot_sha256", None)
    _require(
        isinstance(declared_snapshot_hash, str)
        and _SHA256.fullmatch(declared_snapshot_hash) is not None
        and sha256_bytes(evaluator_json_bytes(snapshot_body)) == declared_snapshot_hash,
        "graph producer source snapshot hash drift",
    )
    _require(
        manifest.get("source_environment_snapshot_sha256") == declared_snapshot_hash,
        "graph producer snapshot pointer drift",
    )
    _require(snapshot.get("git_status_porcelain") == "", "graph diagnostics were not produced from a clean worktree")
    git_head = snapshot.get("git_head")
    _require(isinstance(git_head, str) and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", git_head) is not None, "graph producer git HEAD is invalid")
    _require(isinstance(snapshot.get("environment"), Mapping), "graph producer numerical environment is missing")

    source_rows = snapshot.get("source_files")
    _require(isinstance(source_rows, list) and source_rows, "graph producer source-file ledger is missing")
    source_by_path: dict[str, str] = {}
    for row in source_rows:
        _require(isinstance(row, Mapping), "graph producer source ledger contains a non-object")
        relative = row.get("path")
        declared = row.get("sha256")
        _require(isinstance(relative, str) and relative not in source_by_path, "graph producer source ledger has an invalid or duplicate path")
        _verify_release_file(repo_root, relative, declared, context=f"graph producer source {relative}")
        source_by_path[relative] = str(declared)
    _require(list(source_by_path) == sorted(source_by_path, key=lambda value: value.encode("utf-8")), "graph producer source ledger is not canonically ordered")
    for path_field, hash_field in (
        ("producer_path", "producer_sha256"),
        ("diagnostics_module_path", "diagnostics_module_sha256"),
    ):
        relative = manifest.get(path_field)
        declared = manifest.get(hash_field)
        _require(source_by_path.get(relative) == declared, f"graph {path_field} is not bound into the producer snapshot")
    return declared_snapshot_hash


def _verify_graph_plot_projection(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    arrays = _load_npz(path, context="graph plot projection")
    text_fields = (
        "diagnostic_id", "scope_type", "scope_value", "cell_id",
        "method_id", "method_version_id", "stage",
        "compared_method_id", "compared_method_version_id", "panel_id", "metric_id",
        "series_id", "null_id", "feature_matrix_sha256", "graph_sha256",
        "operator_sha256", "compared_graph_sha256", "compared_operator_sha256",
        "source_binding_id", "seed",
    )
    integer_fields = ("x_index", "draw_index")
    float_fields = ("x_value", "y_value")
    expected_members = set(text_fields + integer_fields + float_fields + ("schema_version", "diagnostic_version"))
    _require(set(arrays) == expected_members, "graph plot projection member roster drift")
    ok_records = sorted(
        (row for row in records if row["status"] == "OK"),
        key=lambda row: str(row["diagnostic_id"]).encode("utf-8"),
    )
    n = len(ok_records)
    _require(all(np.asarray(value).shape == (n,) for value in arrays.values()), "graph plot projection column length drift")
    expected_text: dict[str, list[str]] = {}
    for field in text_fields:
        expected_text[field] = []
        for row in ok_records:
            value = row.get(field)
            if field in ("compared_method_id", "compared_method_version_id", "graph_sha256", "operator_sha256", "compared_graph_sha256", "compared_operator_sha256"):
                value = value or "not_applicable"
            elif field == "null_id":
                value = value or "observed"
            elif field == "seed":
                value = int(value) if value is not None else -1
            expected_text[field].append(str(value))
    for field, expected in expected_text.items():
        _require([str(value) for value in arrays[field].tolist()] == expected, f"graph plot projection drift: {field}")
    for field in integer_fields:
        expected = [int(row[field]) if row.get(field) is not None else -1 for row in ok_records]
        _require(np.array_equal(np.asarray(arrays[field], dtype=np.int64), np.asarray(expected, dtype=np.int64)), f"graph plot projection drift: {field}")
    for field, source_field in (("x_value", "x_value"), ("y_value", "value")):
        expected = np.asarray([float(row[source_field]) for row in ok_records], dtype=np.float64)
        _require(np.array_equal(np.asarray(arrays[field], dtype=np.float64), expected), f"graph plot projection drift: {field}")
    _require(set(str(value) for value in arrays["schema_version"].tolist()) <= {GRAPH_PLOT_SCHEMA}, "graph plot schema drift")
    _require(set(str(value) for value in arrays["diagnostic_version"].tolist()) <= {GRAPH_DIAGNOSTIC_VERSION}, "graph plot diagnostic-version drift")


def _verify_graph_examples(
    path: Path,
    selected: Mapping[str, Any],
    snapshot: Mapping[str, Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    arrays = _load_npz(path, context="example graph data")
    _require(set(selected) == set(GRAPH_METHOD_IDS), "example graph method roster drift")
    for field, expected in (
        ("schema_version", GRAPH_EXAMPLE_SCHEMA),
        ("diagnostic_version", GRAPH_DIAGNOSTIC_VERSION),
        ("selection_rule_id", GRAPH_EXAMPLE_RULE),
    ):
        _require(field in arrays and arrays[field].shape == (1,) and str(arrays[field][0]) == expected, f"example graph {field} drift")
    expected_members = {"schema_version", "diagnostic_version", "selection_rule_id"}
    for method_id in GRAPH_METHOD_IDS:
        prefix = method_id
        cell_id = selected[method_id]
        if cell_id is None:
            _require(
                not any(member.startswith(f"{prefix}__") for member in arrays),
                f"unselected graph method has example members: {method_id}",
            )
            continue
        _require(isinstance(cell_id, str) and cell_id in snapshot, f"invalid selected example: {method_id}")
        required = (
            "cell_id", "row_ids", "embedding_x", "embedding_y", "y_error",
            "trace_length_available", "edge_source", "edge_target", "edge_weight",
            "feature_matrix_sha256", "graph_sha256", "operator_sha256",
        )
        expected_members.update(f"{prefix}__{field}" for field in required)
        if method_id == "pgrd_a" and f"{prefix}__residual_coordinates" in arrays:
            expected_members.add(f"{prefix}__residual_coordinates")
        _require(str(arrays[f"{prefix}__cell_id"][0]) == cell_id, f"example graph cell drift: {method_id}")
        row_ids = tuple(str(value) for value in arrays[f"{prefix}__row_ids"].tolist())
        _require(row_ids == snapshot[cell_id]["row_ids"], f"example graph row identity drift: {method_id}")
        n = len(row_ids)
        for field in ("embedding_x", "embedding_y"):
            value = np.asarray(arrays[f"{prefix}__{field}"], dtype=np.float64)
            _require(value.shape == (n,) and np.isfinite(value).all(), f"example graph invalid {field}: {method_id}")
        nuisance_available_array = np.asarray(arrays[f"{prefix}__trace_length_available"])
        _require(
            nuisance_available_array.shape == (1,) and nuisance_available_array.dtype.kind == "b",
            f"example graph invalid trace-length availability: {method_id}",
        )
        nuisance_available = bool(nuisance_available_array[0])
        nuisance_key = f"{prefix}__trace_length_coordinate"
        if nuisance_available:
            _require(nuisance_key in arrays, f"available example nuisance is missing: {method_id}")
            nuisance = np.asarray(arrays[nuisance_key], dtype=np.float64)
            _require(nuisance.shape == (n,) and np.isfinite(nuisance).all(), f"example graph invalid trace_length_coordinate: {method_id}")
            expected_members.add(nuisance_key)
        else:
            _require(nuisance_key not in arrays, f"unavailable example nuisance must be omitted: {method_id}")
        y_error = np.asarray(arrays[f"{prefix}__y_error"], dtype=np.int8)
        _require(np.array_equal(y_error, snapshot[cell_id]["y_error"]), f"example graph target drift: {method_id}")
        edge_source = np.asarray(arrays[f"{prefix}__edge_source"], dtype=np.int64)
        edge_target = np.asarray(arrays[f"{prefix}__edge_target"], dtype=np.int64)
        edge_weight = np.asarray(arrays[f"{prefix}__edge_weight"], dtype=np.float64)
        _require(edge_source.shape == edge_target.shape == edge_weight.shape, f"example edge length drift: {method_id}")
        _require(np.isfinite(edge_weight).all() and (edge_weight >= 0).all(), f"invalid example edge weights: {method_id}")
        _require(((edge_source >= 0) & (edge_source < n) & (edge_target >= 0) & (edge_target < n)).all(), f"invalid example edge index: {method_id}")
        feature_hash = str(arrays[f"{prefix}__feature_matrix_sha256"][0])
        graph_hash = str(arrays[f"{prefix}__graph_sha256"][0])
        operator_hash = str(arrays[f"{prefix}__operator_sha256"][0])
        _require(all(_SHA256.fullmatch(value) is not None for value in (feature_hash, graph_hash, operator_hash)), f"invalid example hashes: {method_id}")
        candidates = [row for row in records if row["cell_id"] == cell_id and row["method_id"] == method_id]
        _require(any(row["feature_matrix_sha256"] == feature_hash for row in candidates), f"example feature hash is not record-bound: {method_id}")
        _require(any(row.get("graph_sha256") == graph_hash and row.get("operator_sha256") == operator_hash for row in candidates), f"example graph/operator hashes are not record-bound: {method_id}")
        residual_key = f"{prefix}__residual_coordinates"
        if residual_key in arrays:
            residual = np.asarray(arrays[residual_key], dtype=np.float64)
            _require(residual.ndim == 2 and residual.shape[0] == n and np.isfinite(residual).all(), "invalid PGRD residual coordinates")
    _require(set(arrays) == expected_members, "example graph member roster drift")
    return arrays


def _verify_graph_package(
    graph_dir: Path,
    *,
    evaluation_dir: Path,
    release_id: str,
    publication_hashes: Mapping[str, str],
    evaluation: Mapping[str, Any],
    cell_ids: Sequence[str],
    method_versions: Mapping[str, str],
    snapshot: Mapping[str, Mapping[str, Any]],
    fit: Mapping[tuple[str, str], Mapping[str, Any]],
) -> VerifiedGraphPackage:
    graph_dir = graph_dir.resolve()
    _require(graph_dir.is_dir(), f"graph diagnostics directory is missing: {graph_dir}")
    release_root = evaluation_dir.parent.resolve()
    _require(graph_dir.parent.resolve() == release_root, "graph diagnostics are not in the evaluator's scientific release")
    tree_path = _verify_graph_tree(graph_dir)
    manifest_path = graph_dir / "GRAPH_DIAGNOSTICS_MANIFEST.json"
    manifest = _load_json(manifest_path)
    _payload_hash(manifest, "payload_sha256", context="graph diagnostics manifest")
    _require(manifest.get("schema_version") == GRAPH_MANIFEST_SCHEMA, "graph diagnostics manifest schema drift")
    _require(manifest.get("diagnostic_version") == GRAPH_DIAGNOSTIC_VERSION, "graph diagnostic version drift")
    _require(manifest.get("release_id") == release_id and manifest.get("status") == "OK", "graph diagnostics are not bound to this publishable release")
    _require(manifest.get("node_permutation_draws_per_cell_method") == GRAPH_NODE_PERMUTATIONS, "graph permutation contract drift")
    diagnostics_path = _resolve_artifact(graph_dir, manifest.get("diagnostics_path"), context="graph diagnostics payload")
    plot_path = _resolve_artifact(graph_dir, manifest.get("plot_data_path"), context="graph plot data")
    example_path = _resolve_artifact(graph_dir, manifest.get("example_graph_data_path"), context="example graph data")
    for path, field, context in (
        (diagnostics_path, "diagnostics_sha256", "graph diagnostics payload"),
        (plot_path, "plot_data_sha256", "graph plot data"),
        (example_path, "example_graph_data_sha256", "example graph data"),
    ):
        _require(sha256_file(path) == manifest.get(field), f"{context} file hash drift")
    payload = _load_json(diagnostics_path)
    _payload_hash(payload, "payload_sha256", context="graph diagnostics payload")
    _require(payload.get("schema_version") == GRAPH_PAYLOAD_SCHEMA, "graph diagnostics payload schema drift")
    _require(payload.get("diagnostic_version") == GRAPH_DIAGNOSTIC_VERSION, "graph diagnostics payload version drift")
    _require(payload.get("release_id") == release_id and payload.get("status") == "OK", "graph diagnostics payload is not publishable")
    _require(manifest.get("diagnostics_payload_sha256") == payload.get("payload_sha256"), "graph payload/manifest logical hash drift")
    _require(manifest.get("n_records") == len(payload.get("records", ())), "graph diagnostic record-count drift")
    _require(manifest.get("n_source_bindings") == len(payload.get("source_bindings", ())), "graph source-binding count drift")
    scope = payload.get("scope")
    _require(isinstance(scope, Mapping), "graph diagnostic scope missing")
    _require(scope.get("population_id") == SOURCE_POPULATION_ID and scope.get("n_cells") == 24, "graph diagnostic population drift")
    _require(tuple(scope.get("graph_methods", ())) == GRAPH_METHOD_IDS, "graph method scope drift")
    _require(tuple(scope.get("non_graph_methods", ())) == NONGRAPH_DIAGNOSTIC_METHOD_IDS, "non-graph method scope drift")
    _require(scope.get("performance_metrics_recomputed") is False and scope.get("raw_label_bundle_opened") is False, "graph package crossed its scientific boundary")
    null_registry = payload.get("null_registry")
    _require(isinstance(null_registry, Mapping), "graph diagnostic null registry is missing")
    _require(
        isinstance(null_registry.get("node_permutation"), Mapping)
        and null_registry["node_permutation"].get("draws_per_cell_method") == GRAPH_NODE_PERMUTATIONS,
        "graph node-permutation registry drift",
    )
    _require(
        isinstance(null_registry.get("ca_alpha_controls"), Mapping)
        and tuple(null_registry["ca_alpha_controls"].get("controls", ())) == CA_CONTROL_SERIES,
        "CA graph-control registry drift",
    )

    provenance = payload.get("provenance")
    _require(isinstance(provenance, Mapping) and manifest.get("source_provenance") == provenance, "graph provenance manifest/payload drift")
    expected_provenance = {
        "evaluation_manifest_sha256": publication_hashes["evaluation_manifest_sha256"],
        "evaluation_sha256": publication_hashes["evaluation_sha256"],
        "prediction_snapshot_sha256": publication_hashes["prediction_snapshot_sha256"],
        "score_ab_verification_sha256": evaluation["provenance"]["score_ab_verification_sha256"],
        "score_freeze_A_sha256": evaluation["provenance"]["freeze_A_sha256"],
        "input_manifest_A_sha256": evaluation["provenance"]["input_manifest_A_sha256"],
    }
    for field, expected in expected_provenance.items():
        _require(provenance.get(field) == expected, f"graph/evaluation provenance drift: {field}")
    _require(provenance.get("raw_label_bundle_opened") is False and provenance.get("targets_source") == "hashed evaluator prediction snapshot only", "graph target-source boundary drift")
    for path_field, hash_field in (
        ("score_ab_verification_path", "score_ab_verification_sha256"),
        ("score_freeze_A_path", "score_freeze_A_sha256"),
        ("input_manifest_A_path", "input_manifest_A_sha256"),
        ("evaluation_manifest_path", "evaluation_manifest_sha256"),
        ("evaluation_path", "evaluation_sha256"),
        ("prediction_snapshot_path", "prediction_snapshot_sha256"),
    ):
        _verify_release_file(release_root, provenance.get(path_field), provenance.get(hash_field), context=f"graph provenance {path_field}")

    repo_root = Path(__file__).resolve().parents[2]
    for path_field, hash_field in (("producer_path", "producer_sha256"), ("diagnostics_module_path", "diagnostics_module_sha256")):
        _verify_release_file(repo_root, manifest.get(path_field), manifest.get(hash_field), context=f"graph {path_field}")
    producer_snapshot_sha256 = _verify_graph_source_snapshot(manifest, payload, repo_root)

    binding_rows = payload.get("source_bindings")
    _require(isinstance(binding_rows, list), "graph source bindings must be a list")
    bindings: dict[str, Mapping[str, Any]] = {}
    for binding in binding_rows:
        _require(isinstance(binding, Mapping), "graph source binding contains a non-object")
        binding_id = binding.get("source_binding_id")
        body = dict(binding)
        body.pop("source_binding_id", None)
        _require(binding_id == "binding_" + sha256_bytes(evaluator_json_bytes(body))[:20], "graph source-binding ID drift")
        _require(binding_id not in bindings, "duplicate graph source binding")
        bindings[str(binding_id)] = binding
    _require(list(bindings) == sorted(bindings, key=lambda value: value.encode("utf-8")), "graph source bindings are not canonically ordered")
    for binding_id, binding in bindings.items():
        kind = binding.get("binding_type")
        if kind == "single_method_artifact":
            cell_id = binding.get("cell_id")
            _require(cell_id in cell_ids, f"invalid graph source-binding cell: {binding_id}")
            method_id = binding.get("method_id")
            _require(method_id in DIAGNOSTIC_METHOD_IDS and binding.get("method_version_id") == method_versions[method_id], f"graph source-binding method drift: {binding_id}")
            _require(binding.get("prepared_matrix_sha256") == fit[(cell_id, method_id)]["prepared_matrix_sha256"], f"graph source-binding matrix drift: {binding_id}")
            _require(binding.get("score_sha256") == fit[(cell_id, method_id)]["score_file_sha256"], f"graph source-binding score drift: {binding_id}")
            _require(binding.get("producer_snapshot_sha256") == producer_snapshot_sha256, f"graph source-binding producer snapshot drift: {binding_id}")
            for hash_field, expected in (
                ("score_freeze_A_sha256", evaluation["provenance"]["freeze_A_sha256"]),
                ("score_ab_verification_sha256", evaluation["provenance"]["score_ab_verification_sha256"]),
                ("evaluation_manifest_sha256", publication_hashes["evaluation_manifest_sha256"]),
                ("prediction_snapshot_sha256", publication_hashes["prediction_snapshot_sha256"]),
            ):
                _require(binding.get(hash_field) == expected, f"graph source-binding provenance drift: {binding_id}/{hash_field}")
            for path_field, hash_field in (
                ("prepared_artifact_path", "prepared_artifact_sha256"),
                ("score_record_path", "score_record_sha256"),
                ("score_path", "score_sha256"),
                ("artifact_index_path", "artifact_index_sha256"),
                ("score_freeze_A_path", "score_freeze_A_sha256"),
                ("score_ab_verification_path", "score_ab_verification_sha256"),
                ("evaluation_manifest_path", "evaluation_manifest_sha256"),
                ("prediction_snapshot_path", "prediction_snapshot_sha256"),
            ):
                _verify_release_file(release_root, binding.get(path_field), binding.get(hash_field), context=f"graph binding {binding_id}/{path_field}")
            artifact_path = binding.get("method_artifact_path")
            artifact_hash = binding.get("method_artifact_sha256")
            _require((artifact_path is None) == (artifact_hash is None), f"graph method artifact nullability drift: {binding_id}")
            if artifact_path is not None:
                _verify_release_file(release_root, artifact_path, artifact_hash, context=f"graph binding {binding_id}/method_artifact")
        elif kind == "paired_method_artifacts":
            cell_id = binding.get("cell_id")
            _require(cell_id in cell_ids, f"invalid paired graph source-binding cell: {binding_id}")
            for field in ("left_source_binding_id", "right_source_binding_id"):
                _require(binding.get(field) in bindings, f"graph paired binding has unknown {field}: {binding_id}")
                _require(bindings[binding[field]].get("binding_type") == "single_method_artifact", f"graph paired binding does not reference a single binding: {binding_id}")
                _require(bindings[binding[field]].get("cell_id") == cell_id, f"graph paired binding crosses cells: {binding_id}")
            _require(binding.get("left_source_binding_id") != binding.get("right_source_binding_id"), f"graph paired binding repeats one artifact: {binding_id}")
        elif kind == "multi_cell_method_artifacts":
            method_id = binding.get("method_id")
            _require(
                binding.get("release_id") == release_id
                and method_id in GRAPH_METHOD_IDS
                and binding.get("method_version_id") == method_versions[method_id],
                f"graph multi-cell binding release/method drift: {binding_id}",
            )
            children = binding.get("cell_source_bindings")
            _require(isinstance(children, list), f"graph multi-cell binding children are missing: {binding_id}")
            child_cells = [child.get("cell_id") for child in children if isinstance(child, Mapping)]
            _require(
                len(child_cells) == len(children)
                and len(child_cells) >= 3
                and len(set(child_cells)) == len(child_cells)
                and set(child_cells) <= set(cell_ids)
                and child_cells == sorted(child_cells, key=lambda value: str(value).encode("utf-8")),
                f"graph multi-cell binding coverage/order drift: {binding_id}",
            )
            for child in children:
                _require(isinstance(child, Mapping), f"graph multi-cell binding has a non-object child: {binding_id}")
                child_binding = bindings.get(child.get("source_binding_id"))
                _require(
                    child_binding is not None
                    and child_binding.get("binding_type") == "single_method_artifact"
                    and child_binding.get("cell_id") == child.get("cell_id")
                    and child_binding.get("method_id") == method_id,
                    f"graph multi-cell binding child drift: {binding_id}",
                )
            _require(binding.get("evaluation_manifest_sha256") == publication_hashes["evaluation_manifest_sha256"], f"graph multi-cell evaluation binding drift: {binding_id}")
            _require(binding.get("prediction_snapshot_sha256") == publication_hashes["prediction_snapshot_sha256"], f"graph multi-cell snapshot binding drift: {binding_id}")
            _require(binding.get("producer_snapshot_sha256") == producer_snapshot_sha256, f"graph multi-cell producer binding drift: {binding_id}")
        else:
            raise ReportingBridgeError(f"unknown graph source-binding type: {kind!r}")

    record_rows = payload.get("records")
    _require(isinstance(record_rows, list), "graph diagnostic records must be a list")
    records: list[Mapping[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    seen_panels: set[tuple[str, str, str]] = set()
    release_records: list[Mapping[str, Any]] = []
    for record in record_rows:
        _require(isinstance(record, Mapping), "graph diagnostic contains a non-object row")
        identity_fields = (
            "diagnostic_version", "scope_type", "scope_value", "cell_id",
            "method_id", "method_version_id",
            "compared_method_id", "compared_method_version_id", "stage", "panel_id",
            "metric_id", "series_id", "x_index", "x_value", "null_id", "seed",
            "draw_index", "feature_matrix_sha256", "graph_sha256", "operator_sha256",
            "compared_graph_sha256", "compared_operator_sha256", "source_binding_id",
        )
        identity = {field: record.get(field) for field in identity_fields}
        _require(record.get("diagnostic_id") == "diag_" + sha256_bytes(evaluator_json_bytes(identity))[:24], "graph diagnostic ID drift")
        _require(record.get("diagnostic_version") == GRAPH_DIAGNOSTIC_VERSION, "graph record version drift")
        cell_id = record.get("cell_id")
        method_id = record.get("method_id")
        _require(method_id in DIAGNOSTIC_METHOD_IDS, "graph diagnostic method drift")
        _require(record.get("method_version_id") == method_versions[method_id], "graph diagnostic method-version drift")
        scope_type = record.get("scope_type")
        scope_value = record.get("scope_value")
        if scope_type == "cell":
            _require(cell_id in cell_ids and scope_value == cell_id, "cell-scoped graph diagnostic scope drift")
            seen_pairs.add((str(cell_id), str(method_id)))
            seen_panels.add((str(cell_id), str(method_id), str(record.get("panel_id"))))
        elif scope_type == "release":
            _require(cell_id == "__release__" and scope_value == release_id and method_id in GRAPH_METHOD_IDS, "release-scoped graph diagnostic scope drift")
            _require(record.get("panel_id") == "alignment_vs_improvement_summary", "unexpected release-scoped graph diagnostic panel")
            release_records.append(record)
        else:
            raise ReportingBridgeError(f"unknown graph diagnostic scope: {scope_type!r}")
        compared = record.get("compared_method_id")
        compared_version = record.get("compared_method_version_id")
        _require((compared is None) == (compared_version is None), "graph compared-method nullability drift")
        if compared is not None:
            _require(compared in DIAGNOSTIC_METHOD_IDS and compared_version == method_versions[compared], "graph compared-method drift")
        _require(record.get("stage") in ("target_free", "post_freeze"), "graph diagnostic stage drift")
        source_status = record.get("status")
        _require(
            source_status == "OK"
            or (isinstance(source_status, str) and source_status.startswith("NOT_AVAILABLE_"))
            or (isinstance(source_status, str) and source_status.startswith("METRIC_UNDEFINED_")),
            "unknown graph diagnostic scientific status",
        )
        value = record.get("value")
        if source_status == "OK":
            _require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)), "OK graph diagnostic lacks finite value")
        else:
            _require(value is None, "unavailable graph diagnostic carries a value")
        _require(isinstance(record.get("panel_id"), str) and record.get("panel_id"), "graph diagnostic panel is invalid")
        _require(isinstance(record.get("metric_id"), str) and record.get("metric_id"), "graph diagnostic metric is invalid")
        _require(isinstance(record.get("series_id"), str) and record.get("series_id"), "graph diagnostic series is invalid")
        _require(type(record.get("x_index")) is int and record["x_index"] >= 0, "graph diagnostic x-index drift")
        _require(isinstance(record.get("x_value"), (int, float)) and not isinstance(record.get("x_value"), bool) and math.isfinite(float(record["x_value"])), "graph diagnostic x-value drift")
        for optional_integer in ("seed", "draw_index"):
            _require(record.get(optional_integer) is None or type(record[optional_integer]) is int, f"graph diagnostic {optional_integer} drift")
        binding = bindings.get(record.get("source_binding_id"))
        _require(binding is not None, "graph diagnostic has unknown source binding")
        if scope_type == "cell":
            if compared is None:
                _require(
                    binding.get("binding_type") == "single_method_artifact"
                    and binding.get("cell_id") == cell_id
                    and binding.get("method_id") == method_id,
                    "cell graph diagnostic/source binding drift",
                )
                _require(record.get("feature_matrix_sha256") == binding.get("feature_matrix_sha256"), "cell graph diagnostic feature binding drift")
            else:
                _require(binding.get("binding_type") == "paired_method_artifacts" and binding.get("cell_id") == cell_id, "paired graph diagnostic/source binding drift")
                left = bindings[binding["left_source_binding_id"]]
                right = bindings[binding["right_source_binding_id"]]
                _require(left.get("method_id") == method_id and right.get("method_id") == compared, "paired graph diagnostic method order drift")
                _require(left.get("feature_matrix_sha256") == right.get("feature_matrix_sha256") == record.get("feature_matrix_sha256"), "paired graph diagnostic feature binding drift")
        else:
            _require(binding.get("binding_type") == "multi_cell_method_artifacts" and binding.get("method_id") == method_id, "release graph diagnostic/source binding drift")
        for field in ("feature_matrix_sha256", "graph_sha256", "operator_sha256", "compared_graph_sha256", "compared_operator_sha256"):
            value_hash = record.get(field)
            _require(value_hash is None or (isinstance(value_hash, str) and _SHA256.fullmatch(value_hash) is not None), f"invalid graph diagnostic hash: {field}")
        records.append(record)
    _require(len({record["diagnostic_id"] for record in records}) == len(records), "duplicate graph diagnostic ID")
    _require([record["diagnostic_id"] for record in records] == sorted((record["diagnostic_id"] for record in records), key=lambda value: value.encode("utf-8")), "graph diagnostic records are not canonically ordered")
    _require(seen_pairs == {(cell_id, method_id) for cell_id in cell_ids for method_id in DIAGNOSTIC_METHOD_IDS}, "graph diagnostic cell/method coverage drift")
    coverage = payload.get("coverage")
    expected_panel_slots = len(cell_ids) * sum(len(panels) for panels in REQUIRED_GRAPH_PANELS_BY_METHOD.values())
    _require(isinstance(coverage, Mapping), "graph diagnostic coverage ledger is missing")
    _require(coverage.get("coverage_axis") == "cell_x_method_x_preregistered_panel", "graph diagnostic coverage axis drift")
    _require(coverage.get("complete") is True, "graph diagnostic coverage is incomplete")
    _require(coverage.get("expected_panel_slots") == expected_panel_slots and coverage.get("observed_panel_slots") == expected_panel_slots, "graph diagnostic panel-slot count drift")
    _require(coverage.get("required_panels_by_method") == {key: list(value) for key, value in REQUIRED_GRAPH_PANELS_BY_METHOD.items()}, "graph diagnostic preregistered panel roster drift")
    expected_panels = {
        (cell_id, method_id, panel_id)
        for cell_id in cell_ids
        for method_id, panels in REQUIRED_GRAPH_PANELS_BY_METHOD.items()
        for panel_id in panels
    }
    _require(expected_panels <= seen_panels, "graph diagnostic preregistered panel coverage drift")
    _require(
        len(release_records) == 2 * len(GRAPH_METHOD_IDS)
        and {(row["method_id"], row["metric_id"]) for row in release_records}
        == {
            (method_id, metric_id)
            for method_id in GRAPH_METHOD_IDS
            for metric_id in ("spearman_error_alignment_vs_auroc_delta", "pearson_error_alignment_vs_auroc_delta")
        },
        "graph diagnostic release-summary coverage drift",
    )
    for method_id in GRAPH_METHOD_IDS:
        method_release_rows = [row for row in release_records if row["method_id"] == method_id]
        binding_ids = {str(row["source_binding_id"]) for row in method_release_rows}
        _require(len(binding_ids) == 1, f"release-summary binding drift: {method_id}")
        release_binding = bindings[next(iter(binding_ids))]
        child_cells = [str(child["cell_id"]) for child in release_binding["cell_source_bindings"]]
        relation_cells = sorted(
            {
                str(row["cell_id"])
                for row in records
                if row["scope_type"] == "cell"
                and row["method_id"] == method_id
                and row["panel_id"] == "alignment_vs_improvement"
                and row["status"] == "OK"
            },
            key=lambda value: value.encode("utf-8"),
        )
        _require(child_cells == relation_cells, f"release-summary cell-binding drift: {method_id}")
        expected_feature_hash = sha256_bytes(evaluator_json_bytes([
            {
                "cell_id": child["cell_id"],
                "feature_matrix_sha256": bindings[child["source_binding_id"]]["feature_matrix_sha256"],
            }
            for child in release_binding["cell_source_bindings"]
        ]))
        graph_health_rows = {
            str(row["cell_id"]): row
            for row in records
            if row["scope_type"] == "cell"
            and row["method_id"] == method_id
            and row["panel_id"] == "graph_health"
            and row["metric_id"] == "n_edges"
            and row["status"] == "OK"
            and row["cell_id"] in child_cells
        }
        _require(set(graph_health_rows) == set(child_cells), f"release-summary graph-health binding drift: {method_id}")
        expected_graph_hash = sha256_bytes(evaluator_json_bytes([
            {
                "cell_id": cell_id,
                "graph_sha256": graph_health_rows[cell_id]["graph_sha256"],
            }
            for cell_id in child_cells
        ]))
        expected_operator_hash = sha256_bytes(evaluator_json_bytes([
            {
                "cell_id": cell_id,
                "operator_sha256": graph_health_rows[cell_id]["operator_sha256"],
            }
            for cell_id in child_cells
        ]))
        for row in method_release_rows:
            _require(row["feature_matrix_sha256"] == expected_feature_hash, f"release-summary feature hash drift: {method_id}")
            _require(row["graph_sha256"] == expected_graph_hash, f"release-summary graph hash drift: {method_id}")
            _require(row["operator_sha256"] == expected_operator_hash, f"release-summary operator hash drift: {method_id}")
    selected = payload.get("example_selection")
    _require(isinstance(selected, Mapping) and selected.get("rule_id") == GRAPH_EXAMPLE_RULE and selected.get("labels_used") is False, "graph example-selection contract drift")
    selected_cells = selected.get("selected_cell_by_method")
    _require(isinstance(selected_cells, Mapping) and manifest.get("selected_examples") == selected_cells, "graph selected-example manifest drift")
    _verify_graph_plot_projection(plot_path, records)
    example_arrays = _verify_graph_examples(example_path, selected_cells, snapshot, records)
    auxiliary = tuple(
        AuxiliaryArtifact(f"graph_sources/{path.name}", path, sha256_file(path), kind)
        for path, kind in (
            (manifest_path, "graph_diagnostics_manifest"),
            (diagnostics_path, "graph_diagnostics_source"),
            (plot_path, "validated_graph_plot_projection"),
            (example_path, "validated_example_graph_data"),
            (tree_path, "graph_diagnostics_tree_manifest"),
        )
    )
    return VerifiedGraphPackage(manifest, payload, tuple(records), graph_dir, release_root, plot_path, example_path, example_arrays, auxiliary, provenance)


def _expected_scopes(cell_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cell_ids = [str(row["cell_id"]) for row in cell_rows]
    scopes: list[dict[str, Any]] = [{"scope_type": "macro24", "scope_value": "all_24_cells", "cell_ids": cell_ids}]
    scopes.extend({"scope_type": "cell", "scope_value": cell_id, "cell_ids": [cell_id]} for cell_id in cell_ids)
    for field, scope_type in (("domain", "domain"), ("dataset_family", "dataset_family"), ("model_family", "model_family")):
        values = sorted({str(row[field]) for row in cell_rows}, key=lambda value: value.encode("utf-8"))
        for value in values:
            scopes.append({
                "scope_type": scope_type,
                "scope_value": value,
                "cell_ids": [str(row["cell_id"]) for row in cell_rows if row[field] == value],
            })
    return scopes


def _index_exact(rows: Any, fields: Sequence[str], expected: set[tuple[Any, ...]], *, context: str) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    _require(isinstance(rows, list), f"{context} must be a list")
    output: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, Mapping), f"{context} contains a non-object row")
        key = tuple(row.get(field) for field in fields)
        _require(key not in output, f"{context} repeats key {key!r}")
        output[key] = row
    missing = expected - set(output)
    extra = set(output) - expected
    _require(not missing and not extra, f"{context} roster drift: missing={sorted(missing)[:3]!r}, extra={sorted(extra)[:3]!r}")
    return output


def _validate_evaluation(
    evaluation: Mapping[str, Any],
    cell_registry: Mapping[str, Any],
    method_registry: Mapping[str, Any],
    snapshot: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    cell_rows = cell_registry["cells"]
    method_rows = method_registry["methods"]
    cell_ids = tuple(str(row["cell_id"]) for row in cell_rows)
    method_ids = tuple(str(row["method_id"]) for row in method_rows)
    versions = {str(row["method_id"]): str(row["method_version_id"]) for row in method_rows}
    _require(evaluation.get("population_id") == SOURCE_POPULATION_ID, "evaluation population drift")
    _require(evaluation.get("positive_class") == "incorrect", "evaluation positive class drift")
    _require(evaluation.get("label_conversion") == "y_error=1-y_correct", "evaluation label conversion drift")
    _require(evaluation.get("score_semantics") == "higher_is_incorrect", "evaluation score semantics drift")
    _require(evaluation.get("n_cells") == 24 and evaluation.get("n_methods") == 13, "evaluation is not exact 24x13")
    _require(tuple(evaluation.get("method_ids", ())) == method_ids, "evaluation method roster drift")
    _require(evaluation.get("reference_method_id") == REFERENCE_METHOD_ID, "evaluation reference method drift")
    _require(evaluation.get("status") in ("OK", "HEADLINE_BLOCKED"), "unknown evaluation status")
    _require(evaluation.get("headline_status") in ("OK", "HEADLINE_BLOCKED_INCOMPLETE_OR_NONCANONICAL"), "unknown headline status")
    _require((evaluation["status"] == "OK") == (evaluation["headline_status"] == "OK"), "evaluation/headline status contradiction")
    bootstrap = evaluation.get("bootstrap")
    _require(isinstance(bootstrap, Mapping), "evaluation lacks bootstrap contract")
    _require(bootstrap.get("draws") == BOOTSTRAP_DRAWS and bootstrap.get("canonical_draw_count") == BOOTSTRAP_DRAWS, "evaluation is not the canonical 20,000-draw run")
    minimum_fraction = bootstrap.get("minimum_valid_fraction")
    minimum_draws = bootstrap.get("minimum_valid_draws")
    _require(isinstance(minimum_fraction, (int, float)) and 0.0 < float(minimum_fraction) <= 1.0, "invalid minimum valid bootstrap fraction")
    _require(minimum_draws == math.ceil(float(minimum_fraction) * BOOTSTRAP_DRAWS), "minimum valid bootstrap count drift")
    _require(bootstrap.get("resampling_unit") == "verified_source_group_within_cell", "bootstrap unit drift")

    expected_fit = {(cell_id, method_id) for cell_id in cell_ids for method_id in method_ids}
    fit = _index_exact(evaluation.get("fit_outcomes"), ("cell_id", "method_id"), expected_fit, context="fit outcomes")
    matrix_hash_by_cell: dict[str, str] = {}
    for (cell_id, method_id), row in fit.items():
        _require(row.get("method_version_id") == versions[method_id], f"fit method-version drift: {cell_id}/{method_id}")
        status = row.get("fit_status")
        _require(status in ("OK", "OK_FALLBACK"), f"unusable fit status: {cell_id}/{method_id}")
        _require(row.get("fallback_used") is (status == "OK_FALLBACK"), f"fit fallback flag drift: {cell_id}/{method_id}")
        reason = row.get("fallback_reason")
        if status == "OK_FALLBACK":
            _require(isinstance(reason, str) and reason.strip(), f"fallback lacks reason: {cell_id}/{method_id}")
        else:
            _require(reason is None, f"non-fallback fit carries fallback reason: {cell_id}/{method_id}")
        for field in ("score_file_sha256", "prepared_matrix_sha256"):
            _require(isinstance(row.get(field), str) and _SHA256.fullmatch(row[field]) is not None, f"invalid {field}: {cell_id}/{method_id}")
        previous = matrix_hash_by_cell.setdefault(cell_id, row["prepared_matrix_sha256"])
        _require(previous == row["prepared_matrix_sha256"], f"methods use different prepared matrices: {cell_id}")

    expected_labels = {(cell_id,) for cell_id in cell_ids}
    labels = _index_exact(evaluation.get("label_provenance"), ("cell_id",), expected_labels, context="label provenance")
    for (cell_id,), row in labels.items():
        data = snapshot[cell_id]
        y_error = data["y_error"]
        n_error = int(y_error.sum())
        n_correct = int(len(y_error) - n_error)
        _require(row.get("n_rows") == len(y_error), f"label row count drift: {cell_id}")
        _require(row.get("n_correct") == n_correct and row.get("n_error") == n_error, f"label class-count drift: {cell_id}")
        _require(_float_equal(row.get("error_prevalence"), n_error / len(y_error)), f"label prevalence drift: {cell_id}")
        _require(row.get("y_error_sha256") == _hash_array(y_error, "<i1"), f"error-label hash drift: {cell_id}")
        _require(row.get("conversion") == "y_error=1-y_correct", f"label conversion drift: {cell_id}")

    expected_cell_metrics = {(cell_id, method_id, metric) for cell_id in cell_ids for method_id in method_ids for metric in METRICS}
    cell_metrics = _index_exact(evaluation.get("cell_metrics"), ("cell_id", "method_id", "metric"), expected_cell_metrics, context="cell metrics")
    cell_meta = {str(row["cell_id"]): row for row in cell_rows}
    allowed_cell_statuses = {"OK", "METRIC_UNDEFINED_SINGLE_CLASS", "BOOTSTRAP_INSUFFICIENT_VALID_DRAWS"}
    for (cell_id, method_id, metric), row in cell_metrics.items():
        data = snapshot[cell_id]
        y_error = data["y_error"]
        score = data["scores"][method_id]
        meta = cell_meta[cell_id]
        _require(row.get("status") in allowed_cell_statuses, f"unknown cell metric status: {cell_id}/{method_id}/{metric}")
        _require(row.get("population_id") == SOURCE_POPULATION_ID, f"cell metric population drift: {cell_id}")
        for field in ("domain", "dataset_id", "dataset_family", "model_id", "model_family"):
            _require(row.get(field) == meta[field], f"cell metric {field} drift: {cell_id}")
        _require(row.get("method_version_id") == versions[method_id], f"cell metric method version drift: {cell_id}/{method_id}")
        _require(row.get("n_rows") == len(y_error) and row.get("n_groups") == len(set(data["group_ids"])), f"cell metric coverage drift: {cell_id}/{method_id}")
        _require(row.get("positive_class") == "incorrect" and row.get("score_semantics") == "higher_is_incorrect", f"cell metric semantic drift: {cell_id}/{method_id}")
        two_class = set(np.unique(y_error)) == {0, 1}
        expected_point = (_auroc(y_error, score) if metric == "auroc" else _auprc(y_error, score)) if two_class else None
        _require(_float_equal(row.get("estimate"), expected_point), f"cell point metric disagrees with snapshot: {cell_id}/{method_id}/{metric}")
        requested = row.get("bootstrap_draws_requested")
        valid = row.get("bootstrap_draws_valid")
        _require(requested == BOOTSTRAP_DRAWS and isinstance(valid, int) and 0 <= valid <= requested, f"cell bootstrap count drift: {cell_id}/{method_id}/{metric}")
        if row["status"] == "OK":
            _require(valid >= minimum_draws and row.get("estimate") is not None, f"OK cell metric lacks sufficient draws: {cell_id}/{method_id}/{metric}")
        elif row["status"] == "BOOTSTRAP_INSUFFICIENT_VALID_DRAWS":
            _require(row.get("estimate") is not None and valid < minimum_draws, f"invalid insufficient-draw status: {cell_id}/{method_id}/{metric}")
        else:
            _require(not two_class and row.get("estimate") is None and valid == 0, f"invalid single-class status: {cell_id}/{method_id}/{metric}")

    scopes = _expected_scopes(cell_rows)
    scope_keys = {(scope["scope_type"], scope["scope_value"]): scope for scope in scopes}
    expected_aggregates = {(scope_type, scope_value, method_id, metric) for scope_type, scope_value in scope_keys for method_id in method_ids for metric in METRICS}
    aggregates = _index_exact(evaluation.get("aggregate_metrics"), ("scope_type", "scope_value", "method_id", "metric"), expected_aggregates, context="aggregate metrics")
    allowed_aggregate_statuses = {"OK", "INCOMPLETE_COMPONENT_CELLS", "BOOTSTRAP_UNDEFINED", "BOOTSTRAP_INSUFFICIENT_VALID_DRAWS", "HEADLINE_BLOCKED_INCOMPLETE_24"}
    for (scope_type, scope_value, method_id, metric), row in aggregates.items():
        scope = scope_keys[(scope_type, scope_value)]
        components = scope["cell_ids"]
        _require(row.get("status") in allowed_aggregate_statuses, f"unknown aggregate status: {scope_type}/{scope_value}")
        _require(row.get("cell_ids") == components and row.get("n_cells") == len(components), f"aggregate component drift: {scope_type}/{scope_value}")
        _require(row.get("aggregation") == "equal_cell_mean", f"aggregate rule drift: {scope_type}/{scope_value}")
        points = [cell_metrics[(cell_id, method_id, metric)]["estimate"] for cell_id in components]
        complete = all(value is not None for value in points)
        expected_point = float(np.mean(points)) if complete else None
        _require(_float_equal(row.get("estimate"), expected_point), f"aggregate point drift: {scope_type}/{scope_value}/{method_id}/{metric}")
        _require(row.get("bootstrap_draws_requested") == BOOTSTRAP_DRAWS, f"aggregate draw count drift: {scope_type}/{scope_value}")

    candidates = tuple(method_id for method_id in method_ids if method_id != REFERENCE_METHOD_ID)
    expected_contrasts = {(scope_type, scope_value, method_id, metric) for scope_type, scope_value in scope_keys for method_id in candidates for metric in METRICS}
    contrasts = _index_exact(evaluation.get("paired_contrasts_vs_iu_pcr"), ("scope_type", "scope_value", "candidate_method_id", "metric"), expected_contrasts, context="paired contrasts")
    allowed_contrast_statuses = {"OK", "INCOMPLETE_COMPONENT_CELLS", "BOOTSTRAP_INSUFFICIENT_VALID_DRAWS", "HEADLINE_BLOCKED_INCOMPLETE_24"}
    for (scope_type, scope_value, candidate, metric), row in contrasts.items():
        scope = scope_keys[(scope_type, scope_value)]
        components = scope["cell_ids"]
        _require(row.get("status") in allowed_contrast_statuses, f"unknown contrast status: {scope_type}/{scope_value}/{candidate}/{metric}")
        _require(row.get("reference_method_id") == REFERENCE_METHOD_ID, "contrast reference drift")
        _require(row.get("cell_ids") == components and row.get("n_cells") == len(components), "contrast component drift")
        candidate_point = aggregates[(scope_type, scope_value, candidate, metric)]["estimate"]
        reference_point = aggregates[(scope_type, scope_value, REFERENCE_METHOD_ID, metric)]["estimate"]
        expected_delta = None if candidate_point is None or reference_point is None else float(candidate_point) - float(reference_point)
        _require(_float_equal(row.get("delta"), expected_delta), f"contrast delta drift: {scope_type}/{scope_value}/{candidate}/{metric}")
        _require(row.get("bootstrap_draws_requested") == BOOTSTRAP_DRAWS, "contrast draw count drift")
        n_pairs = len(components) if expected_delta is not None else 0
        _require(row.get("wins", 0) + row.get("ties", 0) + row.get("losses", 0) in (0, n_pairs), "contrast W/T/L drift")

    headline = evaluation.get("headline_macro24_auroc")
    _require(isinstance(headline, list), "headline macro ledger must be a list")
    if evaluation["status"] == "OK":
        _require(len(headline) == 13, "OK evaluation lacks 13 headline rows")
        headline_keys = {(row.get("method_id"), row.get("metric")) for row in headline if isinstance(row, Mapping)}
        _require(headline_keys == {(method_id, "auroc") for method_id in method_ids}, "headline roster drift")
    else:
        _require(headline == [], "blocked evaluation must not publish headline rows")
    return {
        "fit": fit,
        "labels": labels,
        "cell_metrics": cell_metrics,
        "aggregates": aggregates,
        "contrasts": contrasts,
        "scopes": scopes,
    }


def _method_registry(
    method_registry: Mapping[str, Any],
    feature_contract_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    cards: list[dict[str, Any]] = []
    versions: list[dict[str, Any]] = []
    systems: list[dict[str, Any]] = []
    for source in method_registry["methods"]:
        method_id = str(source["method_id"])
        guide = _METHOD_GUIDE[method_id]
        origin_kind, origin_title, origin_year, relationship = guide["origin"]
        references = [
            {"title": title, "citation": citation, "url": url}
            for title, citation, url in guide["references"]
        ]
        stage = "new_unrun_ablation" if method_id in ("family_nrm_a", "pgrd_a") else "canonical"
        role = "control" if source["role"] in ("primary_floor", "primary_control") else "primary"
        cards.append({
            "method_id": method_id,
            "display_name": source["display_name"],
            "acronym_expansion": guide["acronym"],
            "family_id": guide["family"],
            "plain_summary": source["plain_description"],
            "input_operation_output": f"frozen mixed-v2 measurements -> {source['plain_description']} -> one continuous error-risk score",
            "formula": source["equation"],
            "formula_terms": source["symbols"],
            "origin": {"kind": origin_kind, "title": origin_title, "year": origin_year, "relationship": relationship},
            "development_history": guide["history"],
            "inputs": "One saved generation and its frozen, once-oriented mixed-v2 output-probability measurements; no label or donor data enter fitting.",
            "access_tier": "gray_box_single_pass",
            "supervision": "none",
            "donor_regime": "within_cell_fully_unsupervised",
            "model_passes": 1,
            "assumptions": [source["assumption"]],
            "fallbacks": [source["fallback"]],
            "limitations": [source["known_limit"]],
            "role": role,
            "research_stage": stage,
            "references": references,
            "style": {"color": source["color"], "marker": source["marker"]},
        })
        spec = PRIMARY_METHOD_SPECS[method_id]
        fixed_parameters = _jsonable(spec.config)
        _require(method_config_sha256(fixed_parameters) == source["config_sha256"], f"executable method configuration drift: {method_id}")
        _require(spec.method_version_id == source["method_version_id"], f"executable method version drift: {method_id}")
        versions.append({
            "method_version_id": source["method_version_id"],
            "method_id": method_id,
            "version_label": "frozen reconstruction benchmark v1",
            "definition_sha256": source["config_sha256"],
            "formula": source["equation"],
            "fixed_parameters": fixed_parameters,
            "source_paths": ["spectral_utils/reconstruction_benchmark/methods.py", "configs/reconstruction_benchmark_v1/methods.json"],
            "feature_contract_id": feature_contract_id,
            "research_stage": stage,
        })
        systems.append({
            "system_id": make_system_id(source["method_version_id"], ADAPTER_ID),
            "method_version_id": source["method_version_id"],
            "adapter_id": ADAPTER_ID,
            "access_contract_id": ACCESS_CONTRACT_ID,
            "display_name": source["display_name"],
            "enabled": True,
        })
    return cards, versions, systems


def _reporting_status(scientific_status: str, *, fallback: bool) -> tuple[str, str]:
    if scientific_status == "OK":
        return ("OK_FALLBACK" if fallback else "OK", "")
    if scientific_status == "METRIC_UNDEFINED_SINGLE_CLASS":
        return ("METRIC_UNDEFINED_SINGLE_CLASS", "Scientific evaluator found only one label class.")
    if scientific_status in ("BOOTSTRAP_INSUFFICIENT_VALID_DRAWS", "BOOTSTRAP_UNDEFINED"):
        return ("UNVERIFIED", f"Scientific evaluator status: {scientific_status}.")
    if scientific_status in ("INCOMPLETE_COMPONENT_CELLS", "HEADLINE_BLOCKED_INCOMPLETE_24"):
        return ("SCORE_INCOMPLETE", f"Scientific evaluator status: {scientific_status}.")
    raise ReportingBridgeError(f"unmapped scientific status {scientific_status!r}")


def _metric_label(metric: str) -> str:
    return "Area under the ROC curve" if metric == "auroc" else "Area under the precision-recall curve"


_GRAPH_PANEL_LABELS: Mapping[str, str] = {
    "graph_health": "Graph health",
    "target_vs_nuisance_roughness": "Correctness and trace-length smoothness",
    "roughness_null_summary": "Alignment relative to node permutations",
    "alignment_vs_improvement": "Cell alignment versus AUROC change",
    "alignment_vs_improvement_summary": "Across-cell alignment/performance association",
    "fixed_graph_group_bootstrap_stability": "Fixed fitted-graph weight sensitivity under source-group resampling",
    "length_only_graph_control": "Trace-length-only graph control",
    "random_family_graph_control": "Random feature-family graph control",
    "graph_operator_similarity": "Similarity between graph methods",
    "dufs_gate_weights": "DUFS feature weights",
    "dufs_gate_weights_per_seed": "DUFS feature weights by seed",
    "dufs_gate_stability": "DUFS weight stability",
    "dufs_seed_graph_stability": "DUFS graph stability across seeds",
    "ca_view_weights": "CA-SpecRaGE view weights",
    "ca_alpha_stability": "CA-SpecRaGE weight stability",
    "ca_seed_graph_stability": "CA-SpecRaGE graph stability across seeds",
    "ca_alpha_controls": "CA-SpecRaGE registered controls",
    "pgrd_seed_graph_stability": "PGRD graph stability across seeds",
    "pgrd_cross_gradient": "PGRD residual-gradient decomposition",
    "continuous_lsml_cluster_boundaries": "Continuous L-SML cluster boundaries",
    "continuous_lsml_correlation_clusters": "Continuous L-SML feature-correlation clusters",
    "family_nrm_residual_covariance": "Family-NRM residual covariance",
    "family_nrm_residual_eigenspectrum": "Family-NRM residual eigenspectrum",
    "family_nrm_family_contributions": "Family-NRM family contributions",
    "su_pcr_decomposition": "SU-PCR low-rank plus sparse decomposition",
    "su_pcr_low_rank_eigenspectrum": "SU-PCR low-rank eigenspectrum",
    "su_pcr_sparse_support": "SU-PCR sparse-dependence support",
    "su_pcr_sparse_support_stability": "SU-PCR support stability",
}


def _graph_diagnostic_label(panel_id: str, metric_id: str) -> str:
    panel = _GRAPH_PANEL_LABELS.get(panel_id, panel_id.replace("_", " ").capitalize())
    metric = metric_id.replace("_", " ")
    metric = metric.replace("auroc", "AUROC").replace("pcr", "PCR")
    return f"{panel} — {metric}"


def _convert_graph_diagnostics(
    package: VerifiedGraphPackage,
    *,
    base_common: Mapping[str, Any],
    cell_meta: Mapping[str, Mapping[str, Any]],
    population_by_cell: Mapping[str, str],
    slice_by_cell: Mapping[str, str],
    cohort_by_cell: Mapping[str, str],
    system_by_method: Mapping[str, str],
    method_version_by_id: Mapping[str, str],
    metric_by_scope_system: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    aggregate_context_by_scope: Mapping[tuple[str, str], Mapping[str, str]],
    snapshot: Mapping[str, Mapping[str, Any]],
    fit: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Project signed scalar diagnostics into the stable reporting schema.

    The source row remains losslessly embedded in ``notes``.  The dedicated
    fields contain only values that have a direct reporting-schema analogue;
    summary null/effect/p-values are never inferred from raw null draws here.
    """

    edge_counts: dict[tuple[str, str], int] = {}
    for source in package.records:
        if source["status"] == "OK" and source["panel_id"] == "graph_health" and source["metric_id"] == "n_edges":
            value = float(source["value"])
            _require(source.get("graph_sha256") is not None and value >= 0 and value.is_integer(), "graph-health n_edges is not an integer")
            edge_counts[(str(source["cell_id"]), str(source["method_id"]))] = int(value)

    raw_nulls: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for source in package.records:
        if source["status"] != "OK":
            continue
        panel = str(source["panel_id"])
        if panel == "node_permutation_null":
            series = str(source["series_id"])
            raw_nulls.setdefault((str(source["cell_id"]), str(source["method_id"]), series), []).append(source)
        elif panel == "pgrd_cross_gradient_null":
            raw_nulls.setdefault((str(source["cell_id"]), str(source["method_id"]), "pgrd_cross_gradient"), []).append(source)
    for key, null_rows in raw_nulls.items():
        _require(len(null_rows) == GRAPH_NODE_PERMUTATIONS, f"registered graph null draw count drift: {key}")
        _require({int(row["draw_index"]) for row in null_rows} == set(range(GRAPH_NODE_PERMUTATIONS)), f"registered graph null draw-index drift: {key}")

    roughness_effects: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for source in package.records:
        if source["panel_id"] == "roughness_null_summary" and str(source["metric_id"]).endswith("alignment_null_minus_real"):
            roughness_effects[(str(source["cell_id"]), str(source["method_id"]), str(source["series_id"]))] = source

    ca_learned: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for source in package.records:
        if source["panel_id"] == "ca_alpha_controls" and source["series_id"] == "learned" and source["status"] == "OK":
            ca_learned[(str(source["cell_id"]), str(source["method_id"]), str(source["metric_id"]))] = source

    source_bindings = {
        str(row["source_binding_id"]): row
        for row in package.payload["source_bindings"]
    }

    output: list[dict[str, Any]] = []
    for source in package.records:
        panel_id = str(source["panel_id"])
        metric_id = str(source["metric_id"])
        # The original signed JSON and PLOT_DATA preserve every atomic draw.
        # Reporting keeps one semantic null comparison instead of turning 32
        # draws per cell into 32 visually indistinguishable heatmap tiles.
        if panel_id in ("node_permutation_null", "pgrd_cross_gradient_null"):
            continue
        if panel_id == "roughness_null_summary" and metric_id.endswith("alignment_null_minus_real"):
            continue
        cell_id = str(source["cell_id"])
        method_id = str(source["method_id"])
        source_status = str(source["status"])
        is_release_relation = source.get("scope_type") == "release"
        if is_release_relation:
            source_binding = source_bindings[str(source["source_binding_id"])]
            bound_cells = [str(child["cell_id"]) for child in source_binding["cell_source_bindings"]]
            fallback = any(bool(fit[(bound_cell, method_id)]["fallback_used"]) for bound_cell in bound_cells)
        else:
            bound_cells = [cell_id]
            fallback = bool(fit[(cell_id, method_id)]["fallback_used"])
        if source_status == "OK":
            status = "OK_FALLBACK" if fallback else "OK"
            value = float(source["value"])
        elif source_status.startswith("NOT_AVAILABLE_"):
            status = "NOT_APPLICABLE"
            value = None
        elif source_status.startswith("METRIC_UNDEFINED_"):
            status = "UNVERIFIED"
            value = None
        else:  # Defensive even though package verification rejects this first.
            raise ReportingBridgeError(f"unmapped graph diagnostic status {source_status!r}")
        if is_release_relation:
            context = dict(aggregate_context_by_scope[("macro24", "all_24_cells")])
            primary_metric = metric_by_scope_system[("macro24", "all_24_cells", method_id, "auroc")]
            matrix_hash = str(source["feature_matrix_sha256"])
            n_nodes = sum(len(snapshot[key]["row_ids"]) for key in bound_cells)
            n_edges = sum(edge_counts.get((key, method_id), 0) for key in bound_cells)
        else:
            context = {
                "dataset_id": cell_meta[cell_id]["dataset_id"],
                "population_id": population_by_cell[cell_id],
                "cell_id": cell_id,
                "slice_id": slice_by_cell[cell_id],
            }
            primary_metric = metric_by_scope_system[("cell", cell_id, method_id, "auroc")]
            matrix_hash = source.get("feature_matrix_sha256") or "not_applicable"
            n_nodes = len(snapshot[cell_id]["row_ids"])
            n_edges = edge_counts.get((cell_id, method_id), 0)
        if source_status == "OK" and primary_metric["status"] == "OK_FALLBACK":
            status = "OK_FALLBACK"
        graph_hash = source.get("graph_sha256") or "not_applicable"
        compared = source.get("compared_method_id") or "not_applicable"
        null_id = source.get("null_id") or "observed"
        graph_variant = (
            f"panel={source['panel_id']};series={source['series_id']};"
            f"compared={compared};null={null_id};x={source['x_index']}"
        )
        graph_id = (
            f"graph::{method_id}::{context['cell_id']}::{str(graph_hash)[:16]}"
            if graph_hash != "not_applicable"
            else f"non_graph::{method_id}::{context['cell_id']}"
        )
        null_value: float | None = None
        effect: float | None = value
        permutation_count = 0
        semantic_mapping = "standalone signed scalar; effect equals value for display"
        if source_status == "OK" and panel_id == "target_vs_nuisance_roughness":
            series = "error_label" if metric_id == "error_label_roughness" else "trace_length_coordinate"
            null_rows = raw_nulls.get((cell_id, method_id, series), [])
            effect_row = roughness_effects.get((cell_id, method_id, series))
            _require(len(null_rows) == GRAPH_NODE_PERMUTATIONS and effect_row is not None, f"roughness diagnostic lacks its signed null summary: {cell_id}/{method_id}/{series}")
            null_value = float(np.median([float(row["value"]) for row in null_rows]))
            effect = float(effect_row["value"]) if effect_row["status"] == "OK" else None
            permutation_count = GRAPH_NODE_PERMUTATIONS
            semantic_mapping = "value=observed roughness; null_value=median of 32 signed node permutations; effect=null median minus observed"
        elif source_status == "OK" and panel_id == "pgrd_cross_gradient" and metric_id == "cross_gradient_norm":
            null_rows = raw_nulls.get((cell_id, method_id, "pgrd_cross_gradient"), [])
            _require(len(null_rows) == GRAPH_NODE_PERMUTATIONS, f"PGRD cross-gradient lacks its signed null draws: {cell_id}")
            null_value = float(np.median([float(row["value"]) for row in null_rows]))
            effect = null_value - float(value)
            permutation_count = GRAPH_NODE_PERMUTATIONS
            semantic_mapping = "value=observed cross-gradient norm; null_value=median of 32 signed node permutations; effect=null median minus observed"
        elif source_status == "OK" and panel_id == "ca_alpha_controls" and source["series_id"] != "learned":
            learned = ca_learned.get((cell_id, method_id, metric_id))
            _require(learned is not None, f"CA control lacks learned reference: {cell_id}/{metric_id}")
            control_value = float(source["value"])
            value = float(learned["value"])
            null_value = control_value
            effect = control_value - value
            permutation_count = 1 if source["series_id"] == "permuted" else 0
            semantic_mapping = "value=learned-graph roughness; null_value=registered control roughness; effect=control minus learned (positive favors learned for lower-is-smoother roughness)"
        elif source_status == "OK" and panel_id == "ca_alpha_controls" and source["series_id"] == "learned":
            # Learned is the reference copied into each registered control row.
            # Keeping it again would duplicate the same estimand in the report.
            continue
        if source_status != "OK":
            null_value = None
            effect = None
            permutation_count = 0

        note_payload = {
            "source_schema": GRAPH_PAYLOAD_SCHEMA,
            "source_diagnostic_id": source["diagnostic_id"],
            "source_panel_id": panel_id,
            "source_metric_id": metric_id,
            "source_status": source_status,
            "source_binding_id": source["source_binding_id"],
            "source_scope_type": source["scope_type"],
            "source_scope_value": source["scope_value"],
            "operator_sha256": source.get("operator_sha256"),
            "compared_method_id": source.get("compared_method_id"),
            "compared_method_version_id": source.get("compared_method_version_id"),
            "compared_graph_sha256": source.get("compared_graph_sha256"),
            "compared_operator_sha256": source.get("compared_operator_sha256"),
            "series_id": source["series_id"],
            "null_id": source.get("null_id"),
            "seed": source.get("seed"),
            "draw_index": source.get("draw_index"),
            "x_index": source["x_index"],
            "x_value": source["x_value"],
            "source_note": source.get("note"),
            "semantic_mapping": semantic_mapping,
            "edge_count_semantics": (
                f"sum of fitted cell-graph edges across {len(bound_cells)} explicitly bound components"
                if is_release_relation
                else "fitted graph edge count; node permutations preserve it and CA positive-weight controls preserve the same union support"
            ),
            "raw_null_draws_preserved_in_auxiliary": panel_id in ("target_vs_nuisance_roughness", "pgrd_cross_gradient"),
            "plot_source_policy": "reporting plots must derive from graph_diagnostics_long; the preserved PLOT_DATA projection is independently hash-verified auxiliary evidence",
        }
        status_detail = f"Signed graph-diagnostics source status: {source_status}."
        if source.get("note"):
            status_detail += f" {source['note']}"
        output.append({
            **base_common,
            **context,
            "cohort_id": primary_metric["cohort_id"] if is_release_relation else cohort_by_cell[cell_id],
            "method_id": method_id,
            "method_version_id": method_version_by_id[method_id],
            "adapter_id": ADAPTER_ID,
            "system_id": system_by_method[method_id],
            "comparison_group_id": primary_metric["comparison_group_id"],
            "status": status,
            "status_detail": status_detail,
            "graph_id": graph_id,
            "graph_variant": graph_variant,
            "graph_hash": graph_hash,
            "matrix_hash": matrix_hash,
            "diagnostic_id": str(source["diagnostic_id"]),
            "diagnostic_label": _graph_diagnostic_label(panel_id, metric_id),
            "diagnostic_unit": str(source["unit"]),
            "value": value,
            "null_value": null_value,
            "effect": effect,
            "p_value": None,
            "permutation_count": permutation_count,
            "label_stage": "label_free" if source["stage"] == "target_free" else "post_freeze_labels",
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "notes": reporting_json_bytes(note_payload).decode("utf-8"),
        })
    return output


def _convert_graph_examples(
    package: VerifiedGraphPackage,
    *,
    base_common: Mapping[str, Any],
    cell_meta: Mapping[str, Mapping[str, Any]],
    population_by_cell: Mapping[str, str],
    slice_by_cell: Mapping[str, str],
    cohort_by_cell: Mapping[str, str],
    system_by_method: Mapping[str, str],
    method_version_by_id: Mapping[str, str],
    metric_by_scope_system: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    fit: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    arrays = package.example_arrays
    selected = package.payload["example_selection"]["selected_cell_by_method"]
    rows: list[dict[str, Any]] = []
    for method_id in GRAPH_METHOD_IDS:
        prefix = method_id
        selected_cell = selected[method_id]
        if selected_cell is None:
            continue
        cell_id = str(selected_cell)
        row_ids = tuple(str(value) for value in arrays[f"{prefix}__row_ids"].tolist())
        embedding_x = np.asarray(arrays[f"{prefix}__embedding_x"], dtype=np.float64)
        embedding_y = np.asarray(arrays[f"{prefix}__embedding_y"], dtype=np.float64)
        y_error = np.asarray(arrays[f"{prefix}__y_error"], dtype=np.int8)
        nuisance_available = bool(np.asarray(arrays[f"{prefix}__trace_length_available"])[0])
        nuisance = (
            np.asarray(arrays[f"{prefix}__trace_length_coordinate"], dtype=np.float64)
            if nuisance_available
            else None
        )
        edge_source = np.asarray(arrays[f"{prefix}__edge_source"], dtype=np.int64)
        edge_target = np.asarray(arrays[f"{prefix}__edge_target"], dtype=np.int64)
        edge_weight = np.asarray(arrays[f"{prefix}__edge_weight"], dtype=np.float64)
        graph_hash = str(arrays[f"{prefix}__graph_sha256"][0])
        matrix_hash = str(arrays[f"{prefix}__feature_matrix_sha256"][0])
        operator_hash = str(arrays[f"{prefix}__operator_sha256"][0])
        example_id = f"graph-example::{method_id}::{cell_id}"
        outcome = fit[(cell_id, method_id)]
        fallback = bool(outcome["fallback_used"])
        status = "OK_FALLBACK" if fallback else "OK"
        status_detail = (
            f"Label-free-selected signed example; registered fit fallback: {outcome['fallback_reason']}"
            if fallback
            else "Label-free-selected signed example; correctness colors were opened only after score and graph freeze."
        )
        primary_metric = metric_by_scope_system[("cell", cell_id, method_id, "auroc")]
        common = {
            **base_common,
            "dataset_id": cell_meta[cell_id]["dataset_id"],
            "population_id": population_by_cell[cell_id],
            "cell_id": cell_id,
            "slice_id": slice_by_cell[cell_id],
            "cohort_id": cohort_by_cell[cell_id],
            "method_id": method_id,
            "method_version_id": method_version_by_id[method_id],
            "adapter_id": ADAPTER_ID,
            "system_id": system_by_method[method_id],
            "comparison_group_id": primary_metric["comparison_group_id"],
            "status": status,
            "status_detail": status_detail,
            "example_id": example_id,
            "selection_rule_id": GRAPH_EXAMPLE_RULE,
            "selection_label_free": True,
            "nuisance_name": "trace_length_coordinate",
            "nuisance_available": nuisance_available,
            "graph_hash": graph_hash,
            "matrix_hash": matrix_hash,
            "operator_hash": operator_hash,
            "label_stage": "post_freeze_labels",
            "notes": (
                "Same frozen spectral embedding and edge set are used for correctness and nuisance coloring; no performance-based example selection."
                if nuisance_available
                else "Same frozen spectral embedding and edge set are used for correctness; trace-length nuisance is unavailable and no substitute was used."
            ),
        }
        for index, row_id in enumerate(row_ids):
            rows.append({
                **common,
                "row_kind": "node",
                "source_row_id": row_id,
                "node_index": index,
                "embedding_x": float(embedding_x[index]),
                "embedding_y": float(embedding_y[index]),
                "y_error": bool(y_error[index]),
                "nuisance_value": float(nuisance[index]) if nuisance is not None else None,
                "edge_source_index": -1,
                "edge_target_index": -1,
                "edge_weight": None,
            })
        for index, (source, target, weight) in enumerate(zip(edge_source, edge_target, edge_weight)):
            rows.append({
                **common,
                "row_kind": "edge",
                "source_row_id": f"edge::{index:08d}",
                "node_index": -1,
                "embedding_x": None,
                "embedding_y": None,
                "y_error": None,
                "nuisance_value": None,
                "edge_source_index": int(source),
                "edge_target_index": int(target),
                "edge_weight": float(weight),
            })
    return rows


def build_bridge_inputs(
    *,
    evaluation_dir: str | Path,
    release_id: str,
    cell_registry_path: str | Path,
    method_registry_path: str | Path,
    feature_registry_path: str | Path,
    graph_diagnostics_dir: str | Path | None = None,
    allow_empty_graph_diagnostics: bool = False,
) -> BridgeInputs:
    """Validate one evaluator publication and construct reporting rows."""

    evaluation_dir = Path(evaluation_dir).resolve()
    _require(evaluation_dir.is_dir(), f"evaluation directory is missing: {evaluation_dir}")
    _require(isinstance(release_id, str) and release_id.strip() == release_id and release_id, "release_id must be nonempty and trimmed")
    manifest, evaluation, snapshot_path, publication_hashes = _verify_publication(evaluation_dir)
    cells_config, methods_config, feature_config, registry_hashes = _load_registries(
        evaluation,
        Path(cell_registry_path).resolve(),
        Path(method_registry_path).resolve(),
        Path(feature_registry_path).resolve(),
    )
    cell_rows = cells_config["cells"]
    method_rows = methods_config["methods"]
    cell_ids = tuple(str(row["cell_id"]) for row in cell_rows)
    method_ids = tuple(str(row["method_id"]) for row in method_rows)
    snapshot = _load_snapshot(snapshot_path, cell_ids, method_ids)
    ledgers = _validate_evaluation(evaluation, cells_config, methods_config, snapshot)
    _require(
        not (graph_diagnostics_dir is not None and allow_empty_graph_diagnostics),
        "graph diagnostics artifact and explicit empty opt-out are mutually exclusive",
    )
    if graph_diagnostics_dir is None:
        _require(
            allow_empty_graph_diagnostics,
            "scientific reporting requires --graph-diagnostics-dir; use the explicit empty opt-out only for non-publication bridge checks",
        )
        graph_package: VerifiedGraphPackage | None = None
    else:
        graph_package = _verify_graph_package(
            Path(graph_diagnostics_dir),
            evaluation_dir=evaluation_dir,
            release_id=release_id,
            publication_hashes=publication_hashes,
            evaluation=evaluation,
            cell_ids=cell_ids,
            method_versions={str(row["method_id"]): str(row["method_version_id"]) for row in method_rows},
            snapshot=snapshot,
            fit=ledgers["fit"],
        )

    feature_contract_id = str(feature_config["contract_id"])
    method_cards, method_versions, systems = _method_registry(methods_config, feature_contract_id)
    system_by_method = {
        version["method_id"]: make_system_id(version["method_version_id"], ADAPTER_ID)
        for version in method_versions
    }
    method_version_by_id = {version["method_id"]: version["method_version_id"] for version in method_versions}
    cell_meta = {str(row["cell_id"]): row for row in cell_rows}

    tasks = [{
        "task_id": TASK_ID,
        "display_name": "Final-answer hallucination detection",
        "description": "Rank complete saved answers or reasoning traces by the risk that their final answer is incorrect.",
        "prediction_unit": "one complete answer or reasoning trace",
        "primary_metric_id": "auroc",
        "positive_class": "incorrect final answer",
        "bootstrap_unit": "verified source question within cell",
    }]
    dataset_ids = sorted({str(row["dataset_id"]) for row in cell_rows}, key=lambda value: value.encode("utf-8"))
    _require(set(dataset_ids) <= set(_DATASETS), f"dataset guide missing IDs: {sorted(set(dataset_ids) - set(_DATASETS))}")
    datasets = []
    for dataset_id in dataset_ids:
        guide = _DATASETS[dataset_id]
        datasets.append({
            "dataset_id": dataset_id,
            "task_id": TASK_ID,
            "display_name": guide["name"],
            "description": guide["description"],
            "prediction_unit": "one complete saved answer or reasoning trace",
            "label_definition": "The frozen grader marks whether the final answer is incorrect; reporting converts correctness once to error-positive labels.",
            "positive_class": "incorrect final answer",
            "inclusion_reason": "One or more cells belong to the frozen corrected 24-cell development benchmark.",
            "dataset_family": guide["family"],
            "revision": "frozen consolidated-matrix revision recorded by the reconstruction provenance",
            "limitations": ["This is a D0 retrospective population: feature and method development inspected these cells."],
            "source": {"title": guide["name"], "citation": guide["source"], "url": ""},
        })
    datasets.append({
        "dataset_id": SUITE_DATASET_ID,
        "task_id": TASK_ID,
        "display_name": "Frozen 24-cell benchmark suite",
        "description": "A virtual reporting dataset used only to register equal-cell aggregates across the heterogeneous QA and mathematics cells.",
        "prediction_unit": "registered cell aggregate; no pooled prediction rows",
        "label_definition": "Inherited from the exact component cells; rows are never pooled into this virtual dataset.",
        "positive_class": "incorrect final answer",
        "inclusion_reason": "Allows macro, domain, dataset-family, and model-family summaries while preserving exact cell cohorts.",
        "dataset_family": "benchmark_suite",
        "revision": "reporting bridge v1",
        "limitations": ["This virtual dataset has no predictions and must never be interpreted as a pooled-row benchmark."],
        "source": {"title": "Frozen 24-cell reconstruction registry", "citation": "Project registry, 2026", "url": ""},
    })

    populations: list[dict[str, Any]] = []
    reporting_cells: list[dict[str, Any]] = []
    slices: list[dict[str, Any]] = []
    population_by_cell: dict[str, str] = {}
    slice_by_cell: dict[str, str] = {}
    cohort_by_cell: dict[str, str] = {}
    for cell_id in cell_ids:
        data = snapshot[cell_id]
        identities = [
            {"row_id": row_id, "group_id": group_id, "eligible": True, "status": "OK", "continuous_score": 0.0}
            for row_id, group_id in zip(data["row_ids"], data["group_ids"])
        ]
        cohort = derive_cohort_id(identities)
        population_id = f"{SOURCE_POPULATION_ID}::{cell_id}"
        slice_id = f"all::{cell_id}"
        population_by_cell[cell_id] = population_id
        slice_by_cell[cell_id] = slice_id
        cohort_by_cell[cell_id] = cohort
        meta = cell_meta[cell_id]
        populations.append({
            "population_id": population_id,
            "task_id": TASK_ID,
            "dataset_id": meta["dataset_id"],
            "display_name": f"{cell_id} exact frozen rows",
            "population_sha256": canonical_sha256({"schema": "frozen24-reporting-population-v1", "source_population_id": SOURCE_POPULATION_ID, "cell_id": cell_id, "row_ids": list(data["row_ids"]), "group_ids": list(data["group_ids"])}),
            "expected_n": len(data["row_ids"]),
            "group_unit": "verified source question or source item",
            "eligibility_rule": "Every row in the evaluator-certified prediction snapshot for this cell.",
        })
        reporting_cells.append({
            "cell_id": cell_id,
            "population_id": population_id,
            "task_id": TASK_ID,
            "dataset_id": meta["dataset_id"],
            "generation_model_id": meta["model_id"],
            "scorer_model_id": "same generation telemetry; no external scorer",
            "split_id": "frozen retrospective evaluation rows",
            "decoding_id": meta["generation_regime"],
            "dataset_family": meta["dataset_family"],
            "expected_n": len(data["row_ids"]),
            "status": "RETROSPECTIVE_D0",
        })
        slices.append({
            "slice_id": slice_id,
            "population_id": population_id,
            "cell_id": cell_id,
            "slice_dimension": "all_rows",
            "slice_value": "all",
            "display_name": f"All exact rows in {cell_id}",
            "expected_n": len(data["row_ids"]),
        })

    aggregation_records: list[dict[str, Any]] = []
    aggregation_id_by_scope: dict[tuple[str, str], str] = {}
    aggregate_context_by_scope: dict[tuple[str, str], dict[str, str]] = {}
    for cell_id in cell_ids:
        aggregation_id = f"frozen24::cell::{cell_id}"
        aggregation_id_by_scope[("cell", cell_id)] = aggregation_id
        aggregation_records.append({
            "aggregation_id": aggregation_id,
            "display_name": f"{cell_id} native cell metric",
            "rule": "native_metric",
            "unit_field": "native",
            "component_ids": [cell_id],
            "bootstrap_unit": "verified source question within cell",
            "weighting": "one row per exact evaluator prediction; grouped bootstrap for uncertainty",
        })
    # Cell rows must exist before an equal-cell aggregate can bind its exact
    # component cohorts.  The scientific evaluator lists macro24 first, so the
    # bridge uses an explicit dependency order instead of discovery order.
    ordered_metric_scopes = [
        scope for scope in ledgers["scopes"] if scope["scope_type"] == "cell"
    ] + [
        scope for scope in ledgers["scopes"] if scope["scope_type"] != "cell"
    ]
    for scope in ordered_metric_scopes:
        scope_type = str(scope["scope_type"])
        scope_value = str(scope["scope_value"])
        if scope_type == "cell":
            continue
        key = (scope_type, scope_value)
        suffix = f"{_token(scope_type)}-{_token(scope_value)}"
        aggregation_id = f"frozen24::equal-cell::{suffix}"
        pseudo_cell = f"aggregate::{suffix}"
        population_id = f"{SOURCE_POPULATION_ID}::aggregate::{suffix}"
        slice_id = f"aggregate::{suffix}::all"
        aggregation_id_by_scope[key] = aggregation_id
        aggregate_context_by_scope[key] = {"cell_id": pseudo_cell, "population_id": population_id, "slice_id": slice_id, "dataset_id": SUITE_DATASET_ID}
        aggregation_records.append({
            "aggregation_id": aggregation_id,
            "display_name": f"Equal-cell {scope_type}: {scope_value}",
            "rule": "equal_unit_mean",
            "unit_field": "cell_id",
            "component_ids": list(scope["cell_ids"]),
            "bootstrap_unit": "verified source groups within each component cell, then equal-cell mean",
            "weighting": "one equal weight per registered component cell; never pooled rows",
        })
        populations.append({
            "population_id": population_id,
            "task_id": TASK_ID,
            "dataset_id": SUITE_DATASET_ID,
            "display_name": f"Virtual {scope_type} aggregate: {scope_value}",
            "population_sha256": canonical_sha256({"schema": "frozen24-virtual-aggregate-v1", "scope_type": scope_type, "scope_value": scope_value, "component_ids": list(scope["cell_ids"])}),
            "expected_n": 0,
            "group_unit": "not applicable; aggregate only",
            "eligibility_rule": "No direct rows. The registered metric is the equal-cell mean of the listed component cohorts.",
        })
        reporting_cells.append({
            "cell_id": pseudo_cell,
            "population_id": population_id,
            "task_id": TASK_ID,
            "dataset_id": SUITE_DATASET_ID,
            "generation_model_id": "multiple registered generation models",
            "scorer_model_id": "not applicable; aggregate only",
            "split_id": "virtual equal-cell aggregate",
            "decoding_id": "multiple registered decoding contracts",
            "dataset_family": scope_type,
            "expected_n": 0,
            "status": "CONTEXT_ONLY_AGGREGATE",
        })
        slices.append({
            "slice_id": slice_id,
            "population_id": population_id,
            "cell_id": pseudo_cell,
            "slice_dimension": scope_type,
            "slice_value": scope_value,
            "display_name": f"{scope_type}: {scope_value}",
            "expected_n": 0,
        })

    adapter_definition = {
        "task": TASK_ID,
        "input": "one evaluator-certified continuous higher-is-incorrect response score",
        "output": "same continuous response-risk score",
        "threshold": None,
    }
    registry = build_registry(
        release_id=release_id,
        tasks=tasks,
        datasets=datasets,
        methods=method_cards,
        method_versions=method_versions,
        adapters=[{
            "adapter_id": ADAPTER_ID,
            "display_name": "Frozen response-score identity adapter",
            "task_id": TASK_ID,
            "plain_summary": "Copies each frozen continuous higher-is-incorrect score to the final-answer detection lane without thresholding or reorientation.",
            "input_unit": "complete answer or reasoning trace score",
            "output_unit": "complete answer or reasoning trace risk",
            "definition_sha256": canonical_sha256(adapter_definition),
            "source_paths": ["spectral_utils/reconstruction_benchmark/reporting_bridge.py"],
            "limitations": ["This adapter does not turn a response score into a localizer, early detector, or stopping policy."],
        }],
        systems=systems,
        access_contracts=[{
            "access_contract_id": ACCESS_CONTRACT_ID,
            "access_tier": "gray_box_single_pass",
            "input_type": "one saved generation plus output-probability telemetry transformed once by mixed-v2",
            "supervision": "none",
            "model_passes_per_question": 1,
            "traces_per_question": 1,
            "donor_regime": "within_cell_fully_unsupervised",
        }],
        feature_contracts=[{
            "feature_contract_id": feature_contract_id,
            "display_name": "Frozen mixed-v2 30-feature contract",
            "definition": json.dumps(feature_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            "sha256": registry_hashes["feature_registry_sha256"],
        }],
        evaluators=[{
            "evaluator_id": EVALUATOR_ID,
            "display_name": "Frozen-24 grouped paired evaluator",
            "definition": "Error-positive AUROC and AUPRC with 20,000 verified-source-group bootstrap draws per cell, shared across methods; equal-cell aggregates use the same draw index.",
            "sha256": evaluation["provenance"]["evaluation_module_sha256"],
        }],
        populations=populations,
        cells=reporting_cells,
        slices=slices,
        aggregations=aggregation_records,
    )

    run_id = f"evaluation::{publication_hashes['evaluation_sha256'][:20]}"
    base_common = {
        "release_id": release_id,
        "run_id": run_id,
        "lane_id": LANE_ID,
        "task_id": TASK_ID,
        "feature_contract_id": feature_contract_id,
        "access_contract_id": ACCESS_CONTRACT_ID,
        "evaluator_id": EVALUATOR_ID,
        "evidence_grade": EVIDENCE_GRADE,
    }
    fit = ledgers["fit"]
    labels = ledgers["labels"]
    scientific_cells = ledgers["cell_metrics"]
    scientific_aggregates = ledgers["aggregates"]
    scientific_contrasts = ledgers["contrasts"]

    metrics: list[dict[str, Any]] = []
    metric_by_scope_system: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    comparison_group_by_scope_system: dict[tuple[str, str, str, str], str] = {}
    for scope in ordered_metric_scopes:
        scope_type = str(scope["scope_type"])
        scope_value = str(scope["scope_value"])
        component_ids = list(scope["cell_ids"])
        if scope_type == "cell":
            cell_id = scope_value
            context = {
                "dataset_id": cell_meta[cell_id]["dataset_id"],
                "population_id": population_by_cell[cell_id],
                "cell_id": cell_id,
                "slice_id": slice_by_cell[cell_id],
            }
            aggregation_id = aggregation_id_by_scope[("cell", cell_id)]
            aggregation_level = "cell"
        else:
            context = aggregate_context_by_scope[(scope_type, scope_value)]
            aggregation_id = aggregation_id_by_scope[(scope_type, scope_value)]
            aggregation_level = "release" if scope_type == "macro24" else ("dataset" if scope_type == "dataset_family" else "task")
        for method_id in method_ids:
            system_id = system_by_method[method_id]
            fallback_components = [cell_id for cell_id in component_ids if fit[(cell_id, method_id)]["fallback_used"]]
            fallback = bool(fallback_components)
            for metric in METRICS:
                scientific = scientific_cells[(scope_value, method_id, metric)] if scope_type == "cell" else scientific_aggregates[(scope_type, scope_value, method_id, metric)]
                report_status, status_detail = _reporting_status(str(scientific["status"]), fallback=fallback)
                if fallback_components:
                    fallback_detail = f"Method used its registered fallback in {len(fallback_components)} component cell(s): {', '.join(fallback_components)}."
                    status_detail = (status_detail + " " + fallback_detail).strip()
                valid_draws = int(scientific["bootstrap_draws_valid"])
                status_detail = (status_detail + f" Scientific bootstrap retained {valid_draws}/{BOOTSTRAP_DRAWS} valid grouped draws.").strip()
                if scope_type == "cell":
                    cohort_id = cohort_by_cell[scope_value]
                    n_rows = int(scientific["n_rows"])
                    n_groups = int(scientific["n_groups"])
                    label_row = labels[(scope_value,)]
                    n_positive = int(label_row["n_error"])
                    n_negative = int(label_row["n_correct"])
                else:
                    components_for_cohort = [metric_by_scope_system[("cell", cell_id, method_id, metric)] for cell_id in component_ids]
                    cohort_id = derive_aggregate_cohort_id("cell_id", components_for_cohort)
                    n_rows = sum(int(row["n_rows"]) for row in components_for_cohort)
                    n_groups = sum(int(row["n_groups"]) for row in components_for_cohort)
                    n_positive = sum(int(row["n_positive"]) for row in components_for_cohort)
                    n_negative = sum(int(row["n_negative"]) for row in components_for_cohort)
                numeric_allowed = report_status in RANKABLE_STATUSES or report_status == "UNVERIFIED"
                row = {
                    **base_common,
                    **context,
                    "cohort_id": cohort_id,
                    "method_id": method_id,
                    "method_version_id": method_version_by_id[method_id],
                    "adapter_id": ADAPTER_ID,
                    "system_id": system_id,
                    "comparison_group_id": "pending",
                    "status": report_status,
                    "status_detail": status_detail,
                    "aggregation_id": aggregation_id,
                    "aggregation_level": aggregation_level,
                    "metric_id": metric,
                    "metric_label": _metric_label(metric),
                    "metric_unit": "fraction",
                    "positive_class": "incorrect final answer",
                    "better_direction": "higher",
                    "value": scientific["estimate"] if numeric_allowed else None,
                    "ci_low": scientific["ci_lower"] if numeric_allowed else None,
                    "ci_high": scientific["ci_upper"] if numeric_allowed else None,
                    "n_rows": n_rows,
                    "n_groups": n_groups,
                    "n_positive": n_positive,
                    "n_negative": n_negative,
                    "bootstrap_unit": "verified source group within cell" if scope_type == "cell" else "verified source groups within cells, then equal-cell mean",
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "is_primary": metric == "auroc",
                    "fidelity": FIDELITY,
                    "component_ids": component_ids,
                }
                row["comparison_group_id"] = derive_comparison_group_id(row)
                metrics.append(row)
                metric_by_scope_system[(scope_type, scope_value, method_id, metric)] = row
                comparison_group_by_scope_system[(scope_type, scope_value, method_id, metric)] = row["comparison_group_id"]

    predictions: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    for cell_id in cell_ids:
        data = snapshot[cell_id]
        for method_id in method_ids:
            outcome = fit[(cell_id, method_id)]
            fallback = bool(outcome["fallback_used"])
            status = "OK_FALLBACK" if fallback else "OK"
            detail = (
                f"Evaluator-certified fit fallback: {outcome['fallback_reason']}"
                if fallback
                else "Evaluator-certified A/B-identical fit; reporting applied no score transformation."
            )
            system_id = system_by_method[method_id]
            for metric in METRICS:
                metric_row = metric_by_scope_system[("cell", cell_id, method_id, metric)]
                for row_id, group_id, label, score in zip(data["row_ids"], data["group_ids"], data["y_error"], data["scores"][method_id]):
                    predictions.append({
                        **base_common,
                        "dataset_id": cell_meta[cell_id]["dataset_id"],
                        "population_id": population_by_cell[cell_id],
                        "cell_id": cell_id,
                        "slice_id": slice_by_cell[cell_id],
                        "cohort_id": cohort_by_cell[cell_id],
                        "method_id": method_id,
                        "method_version_id": method_version_by_id[method_id],
                        "adapter_id": ADAPTER_ID,
                        "system_id": system_id,
                        "comparison_group_id": metric_row["comparison_group_id"],
                        "status": status,
                        "status_detail": detail,
                        "row_id": row_id,
                        "group_id": group_id,
                        "continuous_score": float(score),
                        "discrete_prediction": None,
                        "label": bool(label),
                        "eligible": True,
                        "fallback_used": fallback,
                        "score_hash": outcome["score_file_sha256"],
                    })
            primary_metric = metric_by_scope_system[("cell", cell_id, method_id, "auroc")]
            n_rows = len(data["row_ids"])
            coverage.append({
                **base_common,
                "dataset_id": cell_meta[cell_id]["dataset_id"],
                "population_id": population_by_cell[cell_id],
                "cell_id": cell_id,
                "slice_id": slice_by_cell[cell_id],
                "cohort_id": cohort_by_cell[cell_id],
                "method_id": method_id,
                "method_version_id": method_version_by_id[method_id],
                "adapter_id": ADAPTER_ID,
                "system_id": system_id,
                "comparison_group_id": primary_metric["comparison_group_id"],
                "status": status,
                "status_detail": detail,
                "expected_n": n_rows,
                "eligible_n": n_rows,
                "scored_n": n_rows,
                "fallback_n": n_rows if fallback else 0,
                "excluded_n": 0,
                "failed_n": 0,
                "coverage_fraction": 1.0,
            })

    contrasts: list[dict[str, Any]] = []
    for scope in ledgers["scopes"]:
        scope_type = str(scope["scope_type"])
        scope_value = str(scope["scope_value"])
        for candidate in method_ids:
            if candidate == REFERENCE_METHOD_ID:
                continue
            for metric in METRICS:
                scientific = scientific_contrasts[(scope_type, scope_value, candidate, metric)]
                left = metric_by_scope_system[(scope_type, scope_value, candidate, metric)]
                right = metric_by_scope_system[(scope_type, scope_value, REFERENCE_METHOD_ID, metric)]
                fallback = left["status"] == "OK_FALLBACK" or right["status"] == "OK_FALLBACK"
                report_status, detail = _reporting_status(str(scientific["status"]), fallback=fallback)
                detail = (detail + f" Scientific paired bootstrap retained {scientific['bootstrap_draws_valid']}/{BOOTSTRAP_DRAWS} valid draws.").strip()
                numeric_allowed = report_status in RANKABLE_STATUSES or report_status == "UNVERIFIED"
                contrasts.append({
                    **{field: left[field] for field in (
                        "release_id", "run_id", "lane_id", "task_id", "dataset_id", "population_id",
                        "cell_id", "slice_id", "cohort_id", "method_id", "method_version_id", "adapter_id",
                        "system_id", "comparison_group_id", "feature_contract_id", "access_contract_id",
                        "evaluator_id", "evidence_grade",
                    )},
                    "status": report_status,
                    "status_detail": detail,
                    "aggregation_id": left["aggregation_id"],
                    "aggregation_level": left["aggregation_level"],
                    "metric_id": metric,
                    "metric_unit": "fraction",
                    "positive_class": "incorrect final answer",
                    "better_direction": "higher",
                    "left_system_id": left["system_id"],
                    "right_system_id": right["system_id"],
                    "delta": scientific["delta"] if numeric_allowed else None,
                    "ci_low": scientific["ci_lower"] if numeric_allowed else None,
                    "ci_high": scientific["ci_upper"] if numeric_allowed else None,
                    "wins": int(scientific["wins"]),
                    "ties": int(scientific["ties"]),
                    "losses": int(scientific["losses"]),
                    "n_pairs": int(scientific["n_cells"]),
                    "bootstrap_unit": left["bootstrap_unit"],
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "paired": True,
                    "fidelity": FIDELITY,
                })

    for scope_key, context in aggregate_context_by_scope.items():
        scope_type, scope_value = scope_key
        for method_id in method_ids:
            metric_row = metric_by_scope_system[(scope_type, scope_value, method_id, "auroc")]
            coverage.append({
                **base_common,
                **context,
                "cohort_id": metric_row["cohort_id"],
                "method_id": method_id,
                "method_version_id": method_version_by_id[method_id],
                "adapter_id": ADAPTER_ID,
                "system_id": system_by_method[method_id],
                "comparison_group_id": metric_row["comparison_group_id"],
                "status": "NOT_APPLICABLE",
                "status_detail": "Virtual equal-cell aggregate; it has component metrics but no pooled prediction rows.",
                "expected_n": 0,
                "eligible_n": 0,
                "scored_n": 0,
                "fallback_n": 0,
                "excluded_n": 0,
                "failed_n": 0,
                "coverage_fraction": 0.0,
            })

    graph_diagnostics = (
        []
        if graph_package is None
        else _convert_graph_diagnostics(
            graph_package,
            base_common=base_common,
            cell_meta=cell_meta,
            population_by_cell=population_by_cell,
            slice_by_cell=slice_by_cell,
            cohort_by_cell=cohort_by_cell,
            system_by_method=system_by_method,
            method_version_by_id=method_version_by_id,
            metric_by_scope_system=metric_by_scope_system,
            aggregate_context_by_scope=aggregate_context_by_scope,
            snapshot=snapshot,
            fit=fit,
        )
    )
    graph_examples = (
        []
        if graph_package is None
        else _convert_graph_examples(
            graph_package,
            base_common=base_common,
            cell_meta=cell_meta,
            population_by_cell=population_by_cell,
            slice_by_cell=slice_by_cell,
            cohort_by_cell=cohort_by_cell,
            system_by_method=system_by_method,
            method_version_by_id=method_version_by_id,
            metric_by_scope_system=metric_by_scope_system,
            fit=fit,
        )
    )
    rows: dict[str, tuple[dict[str, Any], ...]] = {
        "predictions": tuple(validate_records("predictions", predictions)),
        "metrics": tuple(validate_records("metrics", metrics)),
        "contrasts": tuple(validate_records("contrasts", contrasts)),
        "coverage": tuple(validate_records("coverage", coverage)),
        "graph_diagnostics": tuple(validate_records("graph_diagnostics", graph_diagnostics)),
        "graph_examples": tuple(validate_records("graph_examples", graph_examples)),
    }
    validate_result_references(registry, rows)
    validate_comparison_groups(rows["metrics"])
    validate_expected_coverage(expected_coverage_rows(registry), rows["coverage"])
    validate_equal_unit_aggregates(rows["metrics"], registry["aggregations"])

    source_provenance = {
        "schema": BRIDGE_SCHEMA,
        "release_id": release_id,
        "scientific_status": evaluation["status"],
        "headline_status": evaluation["headline_status"],
        "source_population_id": SOURCE_POPULATION_ID,
        "score_semantics": "higher_is_incorrect",
        "positive_class": "incorrect",
        "fidelity": FIDELITY,
        "evidence_grade": EVIDENCE_GRADE,
        "source_hashes": {**publication_hashes, **registry_hashes},
        "evaluation_manifest_payload_sha256": manifest["payload_sha256"],
        "evaluation_payload_sha256": evaluation["payload_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "row_counts": {table: len(table_rows) for table, table_rows in rows.items()},
        "graph_diagnostics_status": (
            "VERIFIED_SIGNED_SOURCE_CONVERTED"
            if graph_package is not None
            else "EXPLICIT_OPT_OUT_EMPTY_NON_PUBLICATION"
        ),
        "scientific_publication_eligible": graph_package is not None,
        "graph_source_hashes": (
            {
                "manifest_file_sha256": sha256_file(graph_package.root / "GRAPH_DIAGNOSTICS_MANIFEST.json"),
                "manifest_payload_sha256": graph_package.manifest["payload_sha256"],
                "diagnostics_file_sha256": graph_package.manifest["diagnostics_sha256"],
                "diagnostics_payload_sha256": graph_package.payload["payload_sha256"],
                "plot_data_sha256": graph_package.manifest["plot_data_sha256"],
                "example_graph_data_sha256": graph_package.manifest["example_graph_data_sha256"],
                "producer_source_environment_snapshot_sha256": graph_package.manifest["source_environment_snapshot_sha256"],
                "tree_manifest_sha256": sha256_file(graph_package.root / "TREE_MANIFEST.json"),
            }
            if graph_package is not None
            else {}
        ),
        "graph_plot_source_policy": (
            "PLOT_DATA.npz and EXAMPLE_GRAPH_DATA.npz were independently hash/projection checked and preserved under graph_sources; scalar plots derive from graph_diagnostics_long and example visuals derive from graph_examples_long, with canonical source-table hashes in every plot specification"
            if graph_package is not None
            else "not applicable: explicit empty non-publication mode"
        ),
        "semantic_inputs": [
            "EVALUATION.json",
            "PREDICTION_SNAPSHOT.npz",
            *(
                [
                    "graph_diagnostics/GRAPH_DIAGNOSTICS.json",
                    "graph_diagnostics/PLOT_DATA.npz",
                    "graph_diagnostics/EXAMPLE_GRAPH_DATA.npz",
                ]
                if graph_package is not None
                else []
            ),
        ],
        "opaque_provenance_inputs_rehashed_not_parsed": (
            ["prepared artifacts", "fit score files", "fit records", "method artifacts and indexes", "A/B and freeze manifests", "producer source files"]
            if graph_package is not None
            else []
        ),
        "forbidden_inputs_not_opened": ["historical raw label bundle"],
    }
    return BridgeInputs(
        registry=registry,
        rows=rows,
        source_provenance=source_provenance,
        auxiliary_artifacts=graph_package.auxiliary_artifacts if graph_package is not None else (),
    )


def _write_jsonl(path: Path, table: str, rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Write one deterministic JSONL file inside an unpublished staging tree.

    ``publish_bridge_inputs`` owns the atomic boundary: the entire unique staging
    directory is removed on any exception and renamed into place only after its
    signed manifest is complete.  Writing this potentially multi-gigabyte table
    directly inside that staging tree avoids both a second full payload copy and
    a redundant large-file rename before the actual publication boundary.
    """

    normalized = validate_records(table, rows)
    ordered = sorted(normalized, key=lambda row: record_sort_key(table, row))
    if path.exists():
        raise FileExistsError(f"staged JSONL target already exists: {path}")
    with path.open("wb") as handle:
        for row in ordered:
            handle.write(reporting_json_bytes(row))
            handle.write(b"\n")
    return {
        "table": table,
        "path": path.name,
        "row_count": len(ordered),
        "logical_sha256": table_sha256(table, ordered),
        "file_sha256": sha256_file(path),
    }


def publish_bridge_inputs(output_dir: str | Path, inputs: BridgeInputs) -> Path:
    """Atomically publish immutable bridge inputs for the report builder."""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"bridge output is immutable and already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.building-", dir=output_dir.parent))
    try:
        registry_path = staging / "research_registry.json"
        predictions_path = staging / "predictions.jsonl"
        write_canonical_json(registry_path, inputs.registry)
        artifacts = [{
            "path": registry_path.name,
            "row_count": 1,
            "file_sha256": sha256_file(registry_path),
            "logical_sha256": inputs.registry["registry_sha256"],
        }]
        artifacts.append(_write_jsonl(predictions_path, "predictions", inputs.rows["predictions"]))
        filenames = {
            "metrics": "metrics_long.csv",
            "contrasts": "contrasts_long.csv",
            "coverage": "coverage_long.csv",
            "graph_diagnostics": "graph_diagnostics_long.csv",
            "graph_examples": "graph_examples_long.csv",
        }
        for table, filename in filenames.items():
            record = write_tidy_csv(staging / filename, table, inputs.rows[table])
            record["path"] = filename
            artifacts.append(record)
        occupied = {record["path"] for record in artifacts}
        for auxiliary in inputs.auxiliary_artifacts:
            relative = Path(auxiliary.relative_path)
            _require(not relative.is_absolute(), "auxiliary artifact target must be relative")
            target = (staging / relative).resolve()
            _require(staging.resolve() in target.parents, "auxiliary artifact target escapes bridge publication")
            _require(relative.as_posix() not in occupied, f"duplicate bridge artifact target: {relative}")
            _require(sha256_file(auxiliary.source_path) == auxiliary.file_sha256, f"auxiliary source changed before publication: {auxiliary.source_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(auxiliary.source_path, target)
            _require(sha256_file(target) == auxiliary.file_sha256, f"auxiliary copy hash drift: {relative}")
            artifacts.append({
                "path": relative.as_posix(),
                "row_count": 0,
                "file_sha256": auxiliary.file_sha256,
                "logical_sha256": auxiliary.file_sha256,
                "kind": auxiliary.kind,
            })
            occupied.add(relative.as_posix())
        manifest = {
            **dict(inputs.source_provenance),
            "artifacts": sorted(artifacts, key=lambda row: row["path"]),
        }
        manifest["payload_sha256"] = canonical_sha256(manifest)
        write_canonical_json(staging / "BRIDGE_MANIFEST.json", manifest)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output_dir


__all__ = [
    "BRIDGE_SCHEMA",
    "AuxiliaryArtifact",
    "BridgeInputs",
    "ReportingBridgeError",
    "build_bridge_inputs",
    "publish_bridge_inputs",
]
