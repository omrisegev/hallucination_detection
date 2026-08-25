"""Known-answer and fail-closed tests for the RAG evidence benchmark lane."""

from __future__ import annotations

import ast
import json
from io import BytesIO
import os
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import spectral_utils.reconstruction_benchmark.rag_evidence_ab as ab_module
import spectral_utils.reconstruction_benchmark.rag_evidence_contract as contract_module
import spectral_utils.reconstruction_benchmark.rag_evidence_evaluation as evaluation_module
import spectral_utils.reconstruction_benchmark.rag_evidence_preparation as preparation_module
import spectral_utils.reconstruction_benchmark.rag_evidence_runner as runner_module
from spectral_utils.reconstruction_benchmark.rag_evidence_ab import (
    _assert_independent_evaluation_match,
    _assert_independent_score_match,
    _independently_reexecute_score_worker,
)
from spectral_utils.reconstruction_benchmark.rag_evidence_contract import (
    AtomicRagDirectory,
    FIT_INPUT_SCHEMA,
    PANEL_IDS,
    PREPARATION_MANIFEST_FILENAME,
    PRIVATE_LABEL_FILENAME,
    RagEvidenceContractError,
    SCORE_MANIFEST_FILENAME,
    add_payload_sha256,
    add_pickle_payload_sha256,
    load_registry,
    pickle_bytes,
    validate_fit_input,
    validate_fit_sanitization,
    validate_artifact_identifier,
    write_json_noreplace,
)
from spectral_utils.reconstruction_benchmark.rag_evidence_evaluation import (
    _evaluate_gasp,
    _evaluate_refchecker,
    compute_rag_evidence_evaluation_tables,
    grouped_interval,
    grouped_paired_delta,
)
from spectral_utils.reconstruction_benchmark.rag_evidence_fit import (
    SCORE_ARRAY_NAMES,
    compute_rag_evidence_scores,
    validate_score_arrays,
)
from spectral_utils.reconstruction_benchmark.rag_evidence_runner import (
    FIT_CAPSULE_CODE_ALLOWLIST,
    FIT_SOURCE_FILES,
    _copy_capsule,
    _launch_worker,
    _policy,
)
from spectral_utils.reconstruction_benchmark.io import (
    canonical_json_bytes,
    deterministic_npz_bytes,
    sha256_bytes,
    sha256_file,
)


REPO = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO / "configs/reconstruction_benchmark_v1/rag_evidence.json"


def _explicit_local_import_closure(paths: tuple[str, ...]) -> set[str]:
    """Return repo-local Python modules reached by module-level imports."""

    pending = [path for path in paths if path.endswith(".py")]
    observed: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        observed.add(relative)
        module_parts = relative.removesuffix(".py").split("/")
        package_parts = module_parts[:-1]
        tree = ast.parse((REPO / relative).read_text(encoding="utf-8"))
        for node in tree.body:
            imported_modules: list[str] = []
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    keep = len(package_parts) - (node.level - 1)
                    if keep < 0:
                        continue
                    prefix = package_parts[:keep]
                    suffix = node.module.split(".") if node.module else []
                    imported_modules.append(".".join((*prefix, *suffix)))
                elif node.module:
                    imported_modules.append(node.module)
            for module_name in imported_modules:
                if not module_name.startswith("spectral_utils"):
                    continue
                candidate = module_name.replace(".", "/") + ".py"
                if (REPO / candidate).is_file() and candidate not in observed:
                    pending.append(candidate)
    return observed


def test_rag_source_closures_bind_actual_shared_adapter_and_local_imports() -> None:
    preparation_sources = set(preparation_module.PREPARATION_SOURCE_FILES)
    assert {
        "spectral_utils/ragtruth_evidence_contrast.py",
        "spectral_utils/reconstruction_benchmark/io.py",
    } <= preparation_sources

    preparation_closure = _explicit_local_import_closure(
        preparation_module.PREPARATION_SOURCE_FILES
    )
    assert preparation_closure <= preparation_sources, sorted(
        preparation_closure - preparation_sources
    )

    capsule_sources = set(FIT_CAPSULE_CODE_ALLOWLIST)
    capsule_closure = _explicit_local_import_closure(FIT_CAPSULE_CODE_ALLOWLIST)
    assert capsule_closure <= capsule_sources, sorted(
        capsule_closure - capsule_sources
    )

    certified_chain = set(
        (*preparation_module.PREPARATION_SOURCE_FILES,
         *FIT_SOURCE_FILES,
         *evaluation_module.EVALUATION_SOURCE_FILES)
    )
    chain_closure = _explicit_local_import_closure(tuple(certified_chain))
    assert chain_closure <= certified_chain, sorted(chain_closure - certified_chain)


def test_all_raw_adapters_parse_only_authenticated_held_bytes() -> None:
    """Prevent a hash-then-stream ABA window from returning to any adapter."""

    tree = ast.parse(
        (REPO / "spectral_utils/reconstruction_benchmark/rag_evidence_preparation.py")
        .read_text(encoding="utf-8")
    )
    adapter_names = {
        "_prepare_ragtruth", "_prepare_gasp", "_prepare_lettuce", "_prepare_refchecker",
    }
    adapters = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in adapter_names
    }
    assert set(adapters) == adapter_names
    for name, node in adapters.items():
        calls = [candidate for candidate in ast.walk(node) if isinstance(candidate, ast.Call)]
        assert any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "read_bytes"
            for call in calls
        ), name
        assert not any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "open"
            for call in calls
        ), name


def test_score_capsules_reject_coordinated_copy_aba_across_ab_and_third(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    canonical_payloads: dict[str, bytes] = {}
    for relative in FIT_SOURCE_FILES:
        payload = f"canonical::{relative}".encode("utf-8")
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        canonical_payloads[relative] = payload
    monkeypatch.setattr(
        runner_module._BoundRagFitSourceClosure,
        "_git_state",
        lambda _self: ("synthetic-head", b""),
    )
    attacked_relative = FIT_CAPSULE_CODE_ALLOWLIST[0]
    attacked_path = repo / attacked_relative
    canonical = canonical_payloads[attacked_relative]
    attacker = b"X" * len(canonical)
    original_tree_manifest = runner_module.canonical_tree_manifest

    for phase in ("A", "B", "third-worker"):
        sources = runner_module._BoundRagFitSourceClosure(
            repo, allow_dirty_debug=False
        )
        restored = False

        def restore_after_capsule_copy(root: Path) -> dict:
            nonlocal restored
            observed = original_tree_manifest(root)
            attacked_path.write_bytes(canonical)
            restored = True
            return observed

        monkeypatch.setattr(
            runner_module, "canonical_tree_manifest", restore_after_capsule_copy
        )
        attacked_path.write_bytes(attacker)
        try:
            with pytest.raises(
                RagEvidenceContractError,
                match="held inode changed|held fit source bytes changed",
            ):
                _copy_capsule(
                    repo,
                    tmp_path / f"capsule-{phase}",
                    source_closure=sources,
                )
            assert restored
            assert (
                tmp_path
                / f"capsule-{phase}"
                / "code"
                / attacked_relative
            ).read_bytes() == canonical
            assert attacked_path.read_bytes() == canonical
        finally:
            sources.close(verify=False)


def _condition(seed: int, shift: float = 0.0, *, exact_jsd: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    tokens, width = 48, 8
    logits = rng.normal(size=(tokens, width))
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = 0.82 * probabilities / probabilities.sum(axis=1, keepdims=True)
    output = {
        "token_entropies": np.abs(rng.normal(2.0 + shift, 0.4, tokens)),
        "token_spilled_energies": np.abs(rng.normal(3.0 + shift, 0.5, tokens)),
        "token_logsumexp": np.abs(rng.normal(20.0 + shift, 0.5, tokens)),
        "top_k_logprobs": {
            "ids": np.tile(np.arange(width), (tokens, 1)),
            "logprobs": np.log(probabilities),
        },
    }
    if exact_jsd:
        output["token_jsd_vs_full"] = np.abs(
            rng.normal(0.05 + shift * 0.01, 0.01, tokens)
        )
    return output


def _fit_input() -> dict:
    dev = []
    for index in range(12):
        dev.append({
            "unit_id": f"dev_{index}",
            "task_type": "QA" if index % 2 else "Data2txt",
            "conditions": {
                "full": _condition(index * 10),
                "noctx": _condition(index * 10 + 1, 0.3),
                "loo_0": _condition(index * 10 + 2, 0.1),
                "loo_1": _condition(index * 10 + 3, 0.2),
            },
            "sentence_windows": [
                {"unit_id": f"dev_{index}_s0", "start": 0, "end": 24},
                {"unit_id": f"dev_{index}_s1", "start": 24, "end": 48},
            ],
        })
    test = []
    for index, row in enumerate(dev[:6]):
        test.append({
            **row,
            "unit_id": f"test_{index}",
            "sentence_windows": [
                {"unit_id": f"test_{index}_s0", "start": 0, "end": 24},
                {"unit_id": f"test_{index}_s1", "start": 24, "end": 48},
            ],
        })
    gasp = []
    for index in range(4):
        gasp.append({
            "response_unit_id": f"gasp_{index}",
            "task_type": "Summary" if index % 2 else "Data2txt",
            "conditions": {
                "full": _condition(200 + index * 10),
                "noctx": _condition(201 + index * 10, 0.3, exact_jsd=True),
                "loo_0": _condition(202 + index * 10, 0.1, exact_jsd=True),
                "loo_1": _condition(203 + index * 10, 0.2, exact_jsd=True),
            },
            "sentence_windows": [
                {"unit_id": f"gasp_{index}_s0", "start": 0, "end": 24},
                {"unit_id": f"gasp_{index}_s1", "start": 24, "end": 48},
            ],
        })
    refchecker = []
    settings = ("accurate_context", "noisy_context", "zero_context")
    verdicts = ("Entailment", "Neutral", "Contradiction")
    for index in range(9):
        refchecker.append({
            "unit_id": f"ref_{index}", "setting": settings[index % 3],
            "generator": "synthetic", "nli_prediction": verdicts[index % 3],
            "conditions": {
                "full": _condition(400 + index * 2),
                "noctx": _condition(401 + index * 2, 0.2),
            },
        })
    registry = load_registry(REGISTRY_PATH)
    value = add_pickle_payload_sha256({
        "schema_version": FIT_INPUT_SCHEMA,
        "lane_id": registry["lane_id"],
        "contract_version": registry["method_contract"]["fixed_rag_iu_pcr"]["feature_contract"],
        "panels": {
            "ragtruth": {"splits": {"dev": dev, "test": test}},
            "gasp": {"rows": gasp},
            "lettuce": {"rows": [
                {"unit_id": "lettuce_0", "task_type": "QA", "binary_prediction": 0,
                 "maximum_token_probability": 0.2, "truncated": 0},
                {"unit_id": "lettuce_1", "task_type": "QA", "binary_prediction": 1,
                 "maximum_token_probability": 0.9, "truncated": 0},
            ]},
            "refchecker": {"rows": refchecker},
        },
        "rosters": {"synthetic": True},
        "source_asset_roster_sha256": "0" * 64,
        "historical_scores_opened": False,
        "targets_opened_by_fit": False,
    })
    validate_fit_input(value, registry)
    return value


def test_registry_keeps_access_and_estimand_panels_separate() -> None:
    registry = load_registry(REGISTRY_PATH)
    assert tuple(row["panel_id"] for row in registry["panels"]) == PANEL_IDS
    assert registry["evaluation"]["cross_panel_macro"] == "FORBIDDEN"
    assert registry["evaluation"]["refchecker_setting_pooling"] == "FORBIDDEN"
    identities = {
        (row["dataset"], row["unit"], row["access"], row["estimand"])
        for row in registry["panels"]
    }
    assert len(identities) == len(PANEL_IDS)


@pytest.mark.parametrize(
    "forbidden",
    [
        {"rows": [{"label": 1}]},
        {"rows": [{"source_id": "secret"}]},
        {"rows": [{"nested": {"human_label": "Entailment"}}]},
        {"rows": [{"gold_spans": []}]},
        {"rosters": {"lettuce": {"gold_positive": 943}}},
    ],
)
def test_recursive_label_firewall_rejects_targets_and_groups(forbidden: dict) -> None:
    registry = load_registry(REGISTRY_PATH)
    with pytest.raises(RagEvidenceContractError, match="forbidden fit-visible field"):
        validate_fit_sanitization(
            forbidden, forbidden_fields=registry["fit_visibility"]["forbidden_fields"]
        )


def test_lettuce_gold_prevalence_is_private_only(
    tmp_path: Path,
) -> None:
    raw = {
        "later": {
            "response_id": 20,
            "task_type": "QA",
            "pred_hallucinated": False,
            "token_probs": [0.2, 0.3],
            "truncated": False,
            "source_id": "private_source_b",
            "gold_hallucinated": False,
        },
        "earlier": {
            "response_id": 10,
            "task_type": "Summary",
            "pred_hallucinated": True,
            "token_probs": [0.4, 0.9],
            "truncated": False,
            "source_id": "private_source_a",
            "gold_hallucinated": True,
        },
    }

    class SyntheticHeldSource:
        def read_bytes(self) -> bytes:
            return pickle.dumps(raw, protocol=5)

    sources = {"lettuce_cache": SyntheticHeldSource()}
    synthetic_registry = {
        "sources": {"lettuce_cache": {"path": "unused.pkl"}},
        "expected_rosters": {
            "lettuce": {"examples": 2, "gold_positive": 1, "truncated": 0}
        },
    }
    fit, private, public_roster = preparation_module._prepare_lettuce(
        sources=sources, registry=synthetic_registry
    )

    assert public_roster == {"examples": 2, "truncated": 0}
    assert private["target_audit"] == {
        "examples": 2, "gold_positive": 1, "truncated": 0
    }
    assert [row["unit_id"] for row in fit["rows"]] == [
        "lettuce_000000", "lettuce_000001"
    ]
    registry = load_registry(REGISTRY_PATH)
    validate_fit_sanitization(
        {"lettuce": fit, "rosters": {"lettuce": public_roster}},
        forbidden_fields=registry["fit_visibility"]["forbidden_fields"],
    )


def test_target_free_synthetic_fit_scores_every_panel_deterministically() -> None:
    fit_input = _fit_input()
    left, left_diagnostics = compute_rag_evidence_scores(fit_input)
    right, right_diagnostics = compute_rag_evidence_scores(fit_input)
    assert tuple(left) == SCORE_ARRAY_NAMES
    validate_score_arrays(left)
    for name in SCORE_ARRAY_NAMES:
        assert np.array_equal(left[name], right[name])
    assert left_diagnostics == right_diagnostics
    assert left_diagnostics["labels_seen_during_fit"] is False
    assert left_diagnostics["historical_scores_opened"] is False
    assert len(left["gasp_sentence_id"]) == 8
    assert len(left["refchecker_unit_id"]) == 9


def test_postfreeze_evaluator_regenerates_exact_tables_from_private_rosters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fit_input = _fit_input()
    scores, _ = compute_rag_evidence_scores(fit_input)
    ragtruth_splits = {}
    for split, fit_rows in fit_input["panels"]["ragtruth"]["splits"].items():
        private_rows = []
        for row_index, row in enumerate(fit_rows):
            token_count = len(row["conditions"]["full"]["token_entropies"])
            private_rows.append({
                "unit_id": row["unit_id"],
                "source_id": f"{split}_source_{row_index // 2}",
                "task_type": row["task_type"],
                "response_label": row_index % 2,
                "sentence_labels": [
                    {"unit_id": window["unit_id"], "label": (row_index + index) % 2}
                    for index, window in enumerate(row["sentence_windows"])
                ],
                "token_labels": np.asarray(
                    [(row_index + index) % 2 for index in range(token_count)],
                    dtype=np.uint8,
                ),
            })
        ragtruth_splits[split] = private_rows
    gasp_sentences = []
    for row_index, row in enumerate(fit_input["panels"]["gasp"]["rows"]):
        gasp_sentences.extend({
            "unit_id": window["unit_id"],
            "source_id": f"gasp_source_{row_index // 2}",
            "task_type": row["task_type"],
            "label": (row_index + sentence_index) % 2,
        } for sentence_index, window in enumerate(row["sentence_windows"]))
    lettuce_rows = [{
        "unit_id": row["unit_id"],
        "source_id": f"lettuce_source_{index}",
        "task_type": row["task_type"],
        "label": index % 2,
    } for index, row in enumerate(fit_input["panels"]["lettuce"]["rows"])]
    verdicts = ("Entailment", "Neutral", "Contradiction")
    refchecker_rows = [{
        "unit_id": row["unit_id"],
        "example_id": f"ref_example_{index // 2}",
        "setting": row["setting"],
        "human_label": verdicts[index % 3],
        "label_unsupported": index % 2,
    } for index, row in enumerate(fit_input["panels"]["refchecker"]["rows"])]
    private = {
        "ragtruth": {"splits": ragtruth_splits},
        "gasp": {"sentences": gasp_sentences},
        "lettuce": {"rows": lettuce_rows},
        "refchecker": {"rows": refchecker_rows},
    }
    registry = load_registry(REGISTRY_PATH)
    left = compute_rag_evidence_evaluation_tables(
        registry=registry, private=private, scores=scores, draws=5, seed=17
    )
    right = compute_rag_evidence_evaluation_tables(
        registry=registry, private=private, scores=scores, draws=5, seed=17
    )
    assert left["file_payloads"] == right["file_payloads"]
    assert tuple(left["file_payloads"]) == (
        "metrics.csv", "predictions.csv", "contrasts.csv", "panel_status.csv"
    )
    assert [row["panel_id"] for row in left["panel_status"]] == list(PANEL_IDS)
    assert all(row["status"] == "PASS" for row in left["panel_status"])

    original_interval = evaluation_module.grouped_interval
    original_paired = evaluation_module.grouped_paired_delta

    def slow_interval(*args: object, **kwargs: object) -> dict:
        kwargs.pop("metric_name", None)
        return original_interval(*args, **kwargs)  # type: ignore[arg-type]

    def slow_paired(*args: object, **kwargs: object) -> dict:
        kwargs.pop("metric_name", None)
        return original_paired(*args, **kwargs)  # type: ignore[arg-type]

    with monkeypatch.context() as context:
        context.setattr(evaluation_module, "grouped_interval", slow_interval)
        context.setattr(evaluation_module, "grouped_paired_delta", slow_paired)
        slow_reference = compute_rag_evidence_evaluation_tables(
            registry=registry, private=private, scores=scores, draws=5, seed=17
        )
    assert left["file_payloads"] == slow_reference["file_payloads"]

    duplicate_token = {
        name: np.asarray(values).copy() for name, values in scores.items()
    }
    duplicate_token["rag_test_token_parent_id"][-1] = (
        duplicate_token["rag_test_token_parent_id"][0]
    )
    duplicate_token["rag_test_token_index"][-1] = (
        duplicate_token["rag_test_token_index"][0]
    )
    with pytest.raises(
        RagEvidenceContractError, match="duplicate RAG test scorer-token lattice key"
    ):
        validate_score_arrays(duplicate_token, fit_input=fit_input)
    with pytest.raises(
        RagEvidenceContractError, match="duplicate RAG test scorer-token lattice key"
    ):
        compute_rag_evidence_evaluation_tables(
            registry=registry,
            private=private,
            scores=duplicate_token,
            draws=5,
            seed=17,
        )

    incomplete_token = {
        name: np.asarray(values).copy() for name, values in scores.items()
    }
    for name in (
        "rag_test_token_parent_id",
        "rag_test_token_index",
        "rag_test_token_score",
    ):
        incomplete_token[name] = incomplete_token[name][:-1]
    with pytest.raises(
        RagEvidenceContractError,
        match="score roster differs from registered fit input: rag_test_token_lattice",
    ):
        validate_score_arrays(incomplete_token, fit_input=fit_input)
    with pytest.raises(
        RagEvidenceContractError,
        match="scorer-token lattice/private binding drifted",
    ):
        compute_rag_evidence_evaluation_tables(
            registry=registry,
            private=private,
            scores=incomplete_token,
            draws=5,
            seed=17,
        )

    wrong_response_task = {
        name: np.asarray(values).copy() for name, values in scores.items()
    }
    wrong_response_task["rag_test_response_task"][0] = "attacker_subgroup"
    with pytest.raises(
        RagEvidenceContractError,
        match="response task/private binding drifted",
    ):
        compute_rag_evidence_evaluation_tables(
            registry=registry,
            private=private,
            scores=wrong_response_task,
            draws=5,
            seed=17,
        )

    wrong_gasp_task = {
        name: np.asarray(values).copy() for name, values in scores.items()
    }
    wrong_gasp_task["gasp_task"][0] = "attacker_subgroup"
    with pytest.raises(
        RagEvidenceContractError, match="GASP task/private binding drifted"
    ):
        compute_rag_evidence_evaluation_tables(
            registry=registry,
            private=private,
            scores=wrong_gasp_task,
            draws=5,
            seed=17,
        )


def test_registry_discloses_conditional_loo_cross_scorer_and_reporting_bridge() -> None:
    registry = load_registry(REGISTRY_PATH)
    ragtruth = [
        panel for panel in registry["panels"]
        if panel["panel_id"].startswith("ragtruth_evidence_contrast_")
    ]
    assert {panel["access"] for panel in ragtruth} == {
        "teacher_forced_full_noctx_loo_where_available"
    }
    assert "Summary rows have full/noctx only" in (
        registry["method_contract"]["fixed_rag_iu_pcr"]["loo_availability"]
    )
    transfer = registry["method_contract"]["fixed_rag_iu_pcr_transfer"]["adapter"]
    assert "cross-scorer adaptation" in transfer
    assert "Qwen2.5/RAGTruth" in transfer and "Qwen3 fixed-claim" in transfer
    assert registry["evaluation"]["lane_artifacts"] == "canonical_byte_stable_csv"
    assert registry["evaluation"]["downstream_reporting_bridge"] == (
        "typed_parquet_and_explicit_schema_required_before_integrated_"
        "dataset_cell_reporting"
    )


def test_preparation_source_adapter_reads_held_inode_across_aba_substitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "asset.bin"
    source.write_bytes(b"REGISTERED")
    digest = sha256_file(source)
    registry = {
        "sources": {
            "synthetic": {
                "path": "asset.bin",
                "sha256": digest,
                "size_bytes": source.stat().st_size,
            }
        }
    }
    roster = [{
        "asset_id": "synthetic",
        "path": "asset.bin",
        "size_bytes": source.stat().st_size,
        "sha256": digest,
    }]
    monkeypatch.setattr(
        contract_module,
        "EXPECTED_SOURCE_ASSET_ROSTER_SHA256",
        contract_module.payload_sha256(roster),
    )
    backup = tmp_path / "registered-backup.bin"
    with pytest.raises(RagEvidenceContractError, match="held inode changed"):
        with contract_module.BoundRagSourceAssets(tmp_path, registry) as sources:
            source.rename(backup)
            source.write_bytes(b"SUBSTITUTE")
            with sources["synthetic"].open() as handle:
                assert handle.read() == b"REGISTERED"
            source.unlink()
            backup.rename(source)
            sources.verify_stable()


def _synthetic_private_payload(*, marker: str) -> bytes:
    registry = load_registry(REGISTRY_PATH)
    value = add_pickle_payload_sha256({
        "schema_version": contract_module.PRIVATE_LABEL_SCHEMA,
        "lane_id": registry["lane_id"],
        "ragtruth": {},
        "gasp": {},
        "lettuce": {},
        "refchecker": {},
        "rosters": {},
        "private_target_audit": {"marker": marker},
        "source_asset_roster_sha256": "0" * 64,
    })
    return pickle_bytes(value)


def test_private_read_once_rejects_post_hash_pre_parse_inode_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    registry = load_registry(REGISTRY_PATH)
    target = tmp_path / PRIVATE_LABEL_FILENAME
    backup = tmp_path / "PRIVATE_LABELS.original.pkl"
    attacker = tmp_path / "PRIVATE_LABELS.attacker.pkl"
    original_payload = _synthetic_private_payload(marker="ORIGINAL")
    attacker_payload = _synthetic_private_payload(marker="ATTACKER")
    target.write_bytes(original_payload)
    attacker.write_bytes(attacker_payload)
    original_rebind = contract_module.BoundRagFile._assert_rebound
    swapped = False

    def swap_after_hash_before_parse(
        binding: contract_module.BoundRagFile,
    ) -> None:
        nonlocal swapped
        if binding.path == target.absolute() and not swapped:
            target.rename(backup)
            attacker.rename(target)
            swapped = True
        original_rebind(binding)

    monkeypatch.setattr(
        contract_module.BoundRagFile,
        "_assert_rebound",
        swap_after_hash_before_parse,
    )
    with pytest.raises(RagEvidenceContractError, match="inode was replaced"):
        contract_module.load_private_labels(
            target,
            registry,
            expected_sha256=sha256_bytes(original_payload),
        )
    assert swapped
    assert target.read_bytes() == attacker_payload


def test_bound_ab_trees_reject_symlink_alias_and_shared_hardlink(
    tmp_path: Path,
) -> None:
    external = tmp_path / "external"
    external.mkdir()
    (external / "artifact.bin").write_bytes(b"EXTERNAL")
    alias = tmp_path / "A"
    alias.symlink_to(external, target_is_directory=True)
    with pytest.raises((OSError, RagEvidenceContractError)):
        contract_module.BoundRagTree(alias, name="symlinked A tree")

    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "artifact.bin").write_bytes(b"SHARED")
    os.link(left / "artifact.bin", right / "artifact.bin")
    with contract_module.BoundRagTree(left) as left_tree, contract_module.BoundRagTree(
        right
    ) as right_tree:
        with pytest.raises(RagEvidenceContractError, match="share 1 regular-file"):
            contract_module.assert_physical_tree_independence(
                left_tree, right_tree
            )


def test_evaluation_certificate_rejects_ab_evaluation_aliases_outside_release(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release_id = "evaluation-alias"
    lane_root = tmp_path / "release" / release_id / "rag_evidence"
    private_lane_root = tmp_path / "private" / release_id / "rag_evidence"
    external = tmp_path / "external-evaluation"
    external.mkdir()
    (external / "metrics.csv").write_bytes(b"external")
    for build_id in ("A", "B"):
        build_root = lane_root / build_id
        build_root.mkdir(parents=True)
        (build_root / "evaluation").symlink_to(
            external, target_is_directory=True
        )
        (private_lane_root / build_id).mkdir(parents=True)
    reached_inner = False

    def forbidden_inner(**_kwargs: object) -> dict:
        nonlocal reached_inner
        reached_inner = True
        raise AssertionError("symlink aliases reached evaluation verification")

    monkeypatch.setattr(
        ab_module, "_derive_evaluation_certificate_bound", forbidden_inner
    )
    with pytest.raises((OSError, RagEvidenceContractError)):
        ab_module._derive_evaluation_certificate(
            repo=REPO,
            registry_path=REGISTRY_PATH,
            source_root=tmp_path,
            release_root=tmp_path / "release",
            private_root=tmp_path / "private",
            release_id=release_id,
            require_scientific_full=False,
        )
    assert not reached_inner


def test_score_contract_rejects_missing_panel_array() -> None:
    arrays, _ = compute_rag_evidence_scores(_fit_input())
    arrays.pop("lettuce_prediction")
    with pytest.raises(RagEvidenceContractError, match="array roster drifted"):
        validate_score_arrays(arrays)


def test_grouped_bootstrap_is_deterministic_and_resamples_whole_groups() -> None:
    target = np.asarray([0, 1, 0, 1, 0, 1])
    score = np.asarray([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
    groups = np.asarray(["a", "a", "b", "b", "c", "c"])
    metric = lambda y, s: float(np.mean((s >= 0.5) == y))
    left = grouped_interval(
        target, score, groups, metric, draws=50, seed=7, require_two_classes=True
    )
    right = grouped_interval(
        target, score, groups, metric, draws=50, seed=7, require_two_classes=True
    )
    assert left == right
    assert left["value"] == 1.0
    assert left["draws"] == 50


def _slow_grouped_samples(
    target: np.ndarray,
    value: np.ndarray,
    groups: np.ndarray,
    metric: object,
    *,
    draws: int,
    seed: int,
    require_two_classes: bool,
) -> np.ndarray:
    """Literal pre-acceleration bootstrap retained as the test oracle."""

    target = np.asarray(target)
    value = np.asarray(value)
    group_values = np.asarray(groups).astype(str)
    unique_groups = np.unique(group_values)
    lookup = {
        group: np.flatnonzero(group_values == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(draws)):
        selected = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        indexes = np.concatenate([lookup[group] for group in selected])
        if require_two_classes and len(np.unique(target[indexes])) < 2:
            continue
        result = float(metric(target[indexes], value[indexes]))  # type: ignore[operator]
        if np.isfinite(result):
            samples.append(result)
    return np.asarray(samples, dtype=np.float64)


def _slow_grouped_paired_samples(
    target: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    groups: np.ndarray,
    metric: object,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    target = np.asarray(target)
    left, right = np.asarray(left), np.asarray(right)
    group_values = np.asarray(groups).astype(str)
    unique_groups = np.unique(group_values)
    lookup = {
        group: np.flatnonzero(group_values == group) for group in unique_groups
    }
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(int(draws)):
        selected = rng.choice(
            unique_groups, size=len(unique_groups), replace=True
        )
        indexes = np.concatenate([lookup[group] for group in selected])
        if len(np.unique(target[indexes])) < 2:
            continue
        samples.append(
            float(metric(target[indexes], left[indexes]))  # type: ignore[operator]
            - float(metric(target[indexes], right[indexes]))  # type: ignore[operator]
        )
    return np.asarray(samples, dtype=np.float64)


def _unequal_binary_bootstrap_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sizes = np.asarray([1, 3, 8, 2, 11, 5, 4])
    classes = np.asarray([0, 0, 1, 1, 1, 1, 1])
    groups = np.concatenate([
        np.repeat(f"group-{index}", int(size))
        for index, size in enumerate(sizes)
    ])
    target = np.concatenate([
        np.repeat(int(label), int(size))
        for label, size in zip(classes, sizes, strict=True)
    ])
    rng = np.random.default_rng(2468)
    score = np.round(rng.normal(size=len(groups)), 1)
    order = rng.permutation(len(groups))
    return target[order], score[order], groups[order]


@pytest.mark.parametrize("draws", [63, 64, 65])
@pytest.mark.parametrize("metric_name", ["auroc", "auprc"])
def test_fast_grouped_ranking_matches_slow_reference_bytes_across_batches(
    draws: int, metric_name: str,
) -> None:
    target, score, groups = _unequal_binary_bootstrap_fixture()
    metric = {
        "auroc": lambda y, s: float(roc_auc_score(y, s)),
        "auprc": lambda y, s: float(average_precision_score(y, s)),
    }[metric_name]
    slow_samples = _slow_grouped_samples(
        target, score, groups, metric, draws=draws, seed=991,
        require_two_classes=True,
    )
    fast_samples = np.asarray(evaluation_module._fast_grouped_samples(
        target, score, groups.astype(str), draws=draws, seed=991,
        require_two_classes=True, metric_name=metric_name,
    ))
    assert fast_samples.tobytes() == slow_samples.tobytes()
    assert len(fast_samples) < draws  # pure-class group resamples are rejected
    slow_summary = grouped_interval(
        target, score, groups, metric, draws=draws, seed=991,
        require_two_classes=True,
    )
    fast_summary = grouped_interval(
        target, score, groups, metric, draws=draws, seed=991,
        require_two_classes=True, metric_name=metric_name,
    )
    assert canonical_json_bytes(fast_summary) == canonical_json_bytes(slow_summary)


@pytest.mark.parametrize("metric_name", ["f1", "precision", "recall"])
def test_fast_binary_hard_metrics_match_slow_reference_with_zero_division(
    metric_name: str,
) -> None:
    target, score, groups = _unequal_binary_bootstrap_fixture()
    prediction = (score > 0.8).astype(np.uint8)
    functions = {
        "f1": lambda y, p: float(f1_score(y, p, zero_division=0)),
        "precision": lambda y, p: float(precision_score(y, p, zero_division=0)),
        "recall": lambda y, p: float(recall_score(y, p, zero_division=0)),
    }
    metric = functions[metric_name]
    slow_samples = _slow_grouped_samples(
        target, prediction, groups, metric, draws=65, seed=313,
        require_two_classes=False,
    )
    fast_samples = np.asarray(evaluation_module._fast_grouped_samples(
        target, prediction, groups.astype(str), draws=65, seed=313,
        require_two_classes=False, metric_name=metric_name,
    ))
    assert fast_samples.tobytes() == slow_samples.tobytes()
    slow_summary = grouped_interval(
        target, prediction, groups, metric, draws=65, seed=313,
        require_two_classes=False,
    )
    fast_summary = grouped_interval(
        target, prediction, groups, metric, draws=65, seed=313,
        require_two_classes=False, metric_name=metric_name,
    )
    assert canonical_json_bytes(fast_summary) == canonical_json_bytes(slow_summary)


@pytest.mark.parametrize("metric_name", ["accuracy", "macro_f1"])
def test_fast_threeway_metrics_match_slow_reference_when_classes_disappear(
    metric_name: str,
) -> None:
    labels = np.asarray(["Entailment", "Neutral", "Contradiction"])
    sizes = np.asarray([1, 7, 2, 9, 3])
    groups = np.concatenate([
        np.repeat(f"claim-group-{index}", int(size))
        for index, size in enumerate(sizes)
    ])
    target = np.concatenate([
        np.repeat(labels[index % 3], int(size))
        for index, size in enumerate(sizes)
    ])
    prediction = np.roll(target, 3)
    metric = evaluation_module._threeway_metric(metric_name)
    slow_samples = _slow_grouped_samples(
        target, prediction, groups, metric, draws=65, seed=712,
        require_two_classes=False,
    )
    fast_samples = np.asarray(evaluation_module._fast_grouped_samples(
        target, prediction, groups.astype(str), draws=65, seed=712,
        require_two_classes=False, metric_name=metric_name,
    ))
    assert fast_samples.tobytes() == slow_samples.tobytes()
    slow_summary = grouped_interval(
        target, prediction, groups, metric, draws=65, seed=712,
        require_two_classes=False,
    )
    fast_summary = grouped_interval(
        target, prediction, groups, metric, draws=65, seed=712,
        require_two_classes=False, metric_name=metric_name,
    )
    assert canonical_json_bytes(fast_summary) == canonical_json_bytes(slow_summary)


@pytest.mark.parametrize("metric_name", ["auroc", "auprc"])
def test_fast_paired_delta_matches_slow_reference_bytes(metric_name: str) -> None:
    target, left, groups = _unequal_binary_bootstrap_fixture()
    right = np.round(np.cos(np.arange(len(left), dtype=float)), 1)
    metric = {
        "auroc": lambda y, s: float(roc_auc_score(y, s)),
        "auprc": lambda y, s: float(average_precision_score(y, s)),
    }[metric_name]
    slow_samples = _slow_grouped_paired_samples(
        target, left, right, groups, metric, draws=65, seed=411,
    )
    fast_samples = np.asarray(evaluation_module._fast_grouped_paired_samples(
        target, left, right, groups.astype(str), draws=65, seed=411,
        metric_name=metric_name,
    ))
    assert fast_samples.tobytes() == slow_samples.tobytes()
    slow_summary = grouped_paired_delta(
        target, left, right, groups, metric, draws=65, seed=411,
    )
    fast_summary = grouped_paired_delta(
        target, left, right, groups, metric, draws=65, seed=411,
        metric_name=metric_name,
    )
    assert canonical_json_bytes(fast_summary) == canonical_json_bytes(slow_summary)


def test_fast_grouped_single_class_status_matches_reference() -> None:
    target = np.zeros(9, dtype=np.uint8)
    score = np.linspace(0.0, 1.0, len(target))
    groups = np.asarray(["a"] * 2 + ["b"] * 3 + ["c"] * 4)
    metric = lambda y, s: float(roc_auc_score(y, s))
    slow = grouped_interval(
        target, score, groups, metric, draws=65, seed=1,
        require_two_classes=True,
    )
    fast = grouped_interval(
        target, score, groups, metric, draws=65, seed=1,
        require_two_classes=True, metric_name="auroc",
    )
    assert fast == slow == {
        "value": "", "ci_low": "", "ci_high": "", "draws": 0,
        "status": "METRIC_UNDEFINED_SINGLE_CLASS",
    }
    assert grouped_paired_delta(
        target, score, score[::-1], groups, metric, draws=65, seed=1,
        metric_name="auroc",
    ) == {
        "delta": "", "ci_low": "", "ci_high": "", "draws": 0,
        "status": "METRIC_UNDEFINED_SINGLE_CLASS",
    }


def test_multiplicity_batches_keep_one_literal_choice_call_per_draw() -> None:
    unique = np.asarray(["a", "b", "c"])

    class RecordingRng:
        def __init__(self) -> None:
            self.calls: list[tuple[np.ndarray, int, bool]] = []

        def choice(
            self, values: np.ndarray, *, size: int, replace: bool
        ) -> np.ndarray:
            self.calls.append((values.copy(), size, replace))
            offset = len(self.calls) % len(values)
            return np.roll(values, offset)

    rng = RecordingRng()
    batches = list(evaluation_module._iter_group_multiplicity_batches(
        rng, unique, draws=65,
    ))
    assert [len(batch) for batch in batches] == [64, 1]
    assert len(rng.calls) == 65
    assert all(
        np.array_equal(values, unique) and size == 3 and replace is True
        for values, size, replace in rng.calls
    )
    assert all(np.array_equal(row, np.ones(3, dtype=np.int64)) for batch in batches for row in batch)


def test_real_scale_token_bootstrap_uses_fast_metric_kernel() -> None:
    n_rows, n_groups, draws = 430_202, 450, 64
    index = np.arange(n_rows, dtype=np.int64)
    groups = index % n_groups
    target = ((index * 17 + 3) % 11 < 3).astype(np.uint8)
    score = ((index * 104_729) % 1_000_003).astype(np.float64)
    metric_calls = 0

    def counted_auc(y: np.ndarray, s: np.ndarray) -> float:
        nonlocal metric_calls
        metric_calls += 1
        return float(roc_auc_score(y, s))

    started = time.perf_counter()
    result = grouped_interval(
        target, score, groups, counted_auc, draws=draws, seed=2026082407,
        require_two_classes=True, metric_name="auroc",
    )
    elapsed = time.perf_counter() - started
    assert result["draws"] == draws
    assert metric_calls == 1  # point estimate only; no per-draw sklearn sort
    assert elapsed < 15.0


def test_only_within_gasp_panel_gets_a_paired_contrast() -> None:
    registry = load_registry(REGISTRY_PATH)
    ids = np.asarray([f"g{i}" for i in range(8)], dtype="U")
    labels = {"sentences": [
        {"unit_id": unit_id, "source_id": f"src{i // 2}",
         "task_type": "Summary" if i % 2 else "Data2txt", "label": i % 2}
        for i, unit_id in enumerate(ids)
    ]}
    scores = {
        "gasp_sentence_id": ids,
        "gasp_task": np.asarray(["Data2txt", "Summary"] * 4),
        "gasp_threshold_score": np.arange(8, dtype=float),
        "gasp_fixed_rag_score": np.arange(8, dtype=float)[::-1],
    }
    metrics, _, contrasts = _evaluate_gasp(
        registry=registry, labels=labels, scores=scores, draws=20, seed=3
    )
    assert metrics
    assert contrasts
    assert {row["panel_id"] for row in contrasts} == {"gasp_protocol_sentence"}
    assert all(row["left_method"] == "gasp_threshold" for row in contrasts)


def test_refchecker_never_emits_a_pooled_setting_metric() -> None:
    registry = load_registry(REGISTRY_PATH)
    settings = np.asarray(list(("accurate_context", "noisy_context", "zero_context")) * 4)
    ids = np.asarray([f"r{i}" for i in range(len(settings))], dtype="U")
    labels = {"rows": []}
    verdicts = ("Entailment", "Neutral", "Contradiction")
    for index, (unit_id, setting) in enumerate(zip(ids, settings, strict=True)):
        labels["rows"].append({
            "unit_id": unit_id, "example_id": f"e{index // 2}", "setting": setting,
            "human_label": verdicts[index % 3], "label_unsupported": int(index % 3 != 0),
        })
    scores = {
        "refchecker_unit_id": ids,
        "refchecker_setting": settings,
        "refchecker_nli_prediction": np.asarray([verdicts[index % 3] for index in range(len(ids))]),
        "refchecker_binary_score": np.linspace(0, 1, len(ids)),
    }
    metrics, predictions = _evaluate_refchecker(
        registry=registry, labels=labels, scores=scores, draws=20, seed=9
    )
    assert metrics and predictions
    assert {row["subgroup"] for row in metrics} == set(settings.tolist())
    assert not any(row["subgroup"] == "all" for row in metrics)
    assert {row["panel_id"] for row in metrics} == {
        "refchecker_threeway", "refchecker_binary_claim"
    }


def test_atomic_stage_does_not_replace_existing_output(tmp_path: Path) -> None:
    target = tmp_path / "published"
    first = AtomicRagDirectory(target)
    (first.path / "value.txt").write_text("first", encoding="utf-8")
    first.commit()
    with pytest.raises(FileExistsError):
        AtomicRagDirectory(target)
    assert (target / "value.txt").read_text(encoding="utf-8") == "first"
    first.cleanup()


def test_atomic_directory_parent_swap_fails_without_publishing_attacker_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = tmp_path / "stable-parent"
    parent.mkdir()
    displaced = tmp_path / "displaced-parent"
    target = parent / "published"
    stage = AtomicRagDirectory(target)
    (stage.path / "value.txt").write_text("GOOD", encoding="utf-8")
    stage_name = stage._stage_name
    original = contract_module._rename_directory_noreplace_at
    swapped = False

    def swap_parent_then_rename(
        descriptor: int, source_name: str, target_name: str, **kwargs: object
    ) -> tuple[int, int]:
        nonlocal swapped
        if not swapped and source_name == stage_name and target_name == "published":
            parent.rename(displaced)
            parent.mkdir()
            attacker_stage = parent / stage_name
            attacker_stage.mkdir()
            (attacker_stage / "value.txt").write_text("EVIL", encoding="utf-8")
            swapped = True
        return original(descriptor, source_name, target_name, **kwargs)

    monkeypatch.setattr(
        contract_module, "_rename_directory_noreplace_at", swap_parent_then_rename
    )
    with pytest.raises(RagEvidenceContractError, match="parent path inode was replaced"):
        stage.commit()
    assert not (parent / "published").exists()
    assert not (displaced / "published").exists()
    assert stage.quarantine_name is not None
    assert (
        displaced / stage.quarantine_name / "value.txt"
    ).read_text(encoding="utf-8") == "GOOD"
    assert (parent / stage_name / "value.txt").read_text(encoding="utf-8") == "EVIL"
    stage.cleanup()
    assert (displaced / stage.quarantine_name / "value.txt").exists()
    assert (parent / stage_name / "value.txt").read_text(encoding="utf-8") == "EVIL"


def test_atomic_directory_quarantines_wrong_inode_moved_inside_rename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "published"
    stage = AtomicRagDirectory(target)
    (stage.path / "value.txt").write_text("GOOD", encoding="utf-8")
    original = contract_module._rename_directory_noreplace_at
    backup_name = ".held-good-stage"
    substituted = False

    def substitute_source_inside_rename(
        descriptor: int, source_name: str, target_name: str, **kwargs: object
    ) -> tuple[int, int]:
        nonlocal substituted
        if not substituted and target_name == "published":
            original(descriptor, source_name, backup_name, **kwargs)
            os.mkdir(source_name, 0o700, dir_fd=descriptor)
            evil_directory = os.open(
                source_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            try:
                evil_file = os.open(
                    "value.txt",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=evil_directory,
                )
                try:
                    os.write(evil_file, b"EVIL")
                finally:
                    os.close(evil_file)
            finally:
                os.close(evil_directory)
            substituted = True
        return original(descriptor, source_name, target_name, **kwargs)

    monkeypatch.setattr(
        contract_module,
        "_rename_directory_noreplace_at",
        substitute_source_inside_rename,
    )
    with pytest.raises(RagEvidenceContractError, match="substituted staging inode"):
        stage.commit()
    assert not target.exists()
    assert stage.quarantine_name is not None
    assert (tmp_path / stage.quarantine_name / "value.txt").read_text() == "EVIL"
    assert (tmp_path / backup_name / "value.txt").read_text() == "GOOD"
    stage.cleanup()
    assert (tmp_path / stage.quarantine_name / "value.txt").exists()


def test_atomic_directory_quarantines_wrong_type_inserted_after_rename_syscall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "published"
    stage = AtomicRagDirectory(target)
    (stage.path / "value.txt").write_text("GOOD", encoding="utf-8")
    original_entry_identity = contract_module._entry_identity
    backup_name = ".held-good-published"
    substituted = False

    def substitute_final_before_post_stat(
        descriptor: int,
        name: str,
        *,
        require_directory: bool | None = None,
    ) -> tuple[int, int]:
        nonlocal substituted
        if not substituted and name == target.name:
            try:
                os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                os.rename(
                    name,
                    backup_name,
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                )
                evil = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=descriptor,
                )
                try:
                    os.write(evil, b"EVIL WRONG-TYPE FINAL")
                finally:
                    os.close(evil)
                substituted = True
        return original_entry_identity(
            descriptor, name, require_directory=require_directory
        )

    monkeypatch.setattr(
        contract_module, "_entry_identity", substitute_final_before_post_stat
    )
    with pytest.raises(RagEvidenceContractError, match="substituted staging inode"):
        stage.commit()
    assert not target.exists()
    assert (tmp_path / backup_name / "value.txt").read_text() == "GOOD"
    assert stage.quarantine_name is not None
    assert (tmp_path / stage.quarantine_name).read_bytes() == b"EVIL WRONG-TYPE FINAL"
    stage.cleanup()


def test_atomic_directory_rejects_replaced_staging_inode(tmp_path: Path) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    stage = AtomicRagDirectory(parent / "published")
    (stage.path / "value.txt").write_text("GOOD", encoding="utf-8")
    original_stage = parent / f"{stage._stage_name}.original"
    stage.path.rename(original_stage)
    stage.path.mkdir()
    (stage.path / "value.txt").write_text("EVIL", encoding="utf-8")
    with pytest.raises(RagEvidenceContractError, match="entry inode was replaced"):
        stage.commit()
    shutil.rmtree(stage.path)
    original_stage.rename(stage.path)
    stage.cleanup()
    assert not (parent / "published").exists()


def test_cleanup_quarantines_substituted_sibling_without_deleting_either_inode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    stage = AtomicRagDirectory(tmp_path / "published")
    (stage.path / "value.txt").write_text("GOOD", encoding="utf-8")
    original = contract_module._rename_directory_noreplace_at
    backup_name = ".held-good-before-cleanup"
    substituted = False

    def substitute_during_cleanup_rename(
        descriptor: int, source_name: str, target_name: str, **kwargs: object
    ) -> tuple[int, int]:
        nonlocal substituted
        if not substituted and source_name == stage._stage_name:
            original(descriptor, source_name, backup_name, **kwargs)
            os.mkdir(source_name, 0o700, dir_fd=descriptor)
            (tmp_path / source_name / "value.txt").write_text("EVIL", encoding="utf-8")
            substituted = True
        return original(descriptor, source_name, target_name, **kwargs)

    monkeypatch.setattr(
        contract_module,
        "_rename_directory_noreplace_at",
        substitute_during_cleanup_rename,
    )
    stage.cleanup()
    assert stage.quarantine_name is not None
    assert (tmp_path / stage.quarantine_name / "value.txt").read_text() == "EVIL"
    assert (tmp_path / backup_name / "value.txt").read_text() == "GOOD"


def test_stage_sensitive_write_stays_on_held_inode_after_parent_swap(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "private-parent"
    parent.mkdir()
    displaced = tmp_path / "private-parent-displaced"
    stage = AtomicRagDirectory(parent / "private-build")
    stage_name = stage._stage_name
    parent.rename(displaced)
    parent.mkdir()
    (parent / stage_name).mkdir()

    stage.write_bytes("PRIVATE_LABELS.pkl", b"SECRET TARGETS")
    assert not (parent / stage_name / "PRIVATE_LABELS.pkl").exists()
    assert (
        displaced / stage_name / "PRIVATE_LABELS.pkl"
    ).read_bytes() == b"SECRET TARGETS"
    with pytest.raises(RagEvidenceContractError, match="parent path inode was replaced"):
        stage.commit()
    stage.cleanup()
    assert stage.quarantine_name is not None
    assert (
        displaced / stage.quarantine_name / "PRIVATE_LABELS.pkl"
    ).read_bytes() == b"SECRET TARGETS"
    assert (parent / stage_name).is_dir()


def test_atomic_directory_target_injection_is_no_replace(tmp_path: Path) -> None:
    target = tmp_path / "published"
    stage = AtomicRagDirectory(target)
    (stage.path / "value.txt").write_text("GOOD", encoding="utf-8")
    target.mkdir()
    (target / "value.txt").write_text("INCUMBENT", encoding="utf-8")
    with pytest.raises(FileExistsError, match="output already exists"):
        stage.commit()
    assert (target / "value.txt").read_text(encoding="utf-8") == "INCUMBENT"
    stage.cleanup()


def test_certificate_parent_swap_preserves_replacement_parent_incumbent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = tmp_path / "cert-parent"
    parent.mkdir()
    displaced = tmp_path / "cert-parent-displaced"
    target = parent / "CERTIFICATE.json"
    original = contract_module._rename_directory_noreplace_at
    swapped = False

    def swap_parent_then_rename(
        descriptor: int, source_name: str, target_name: str, **kwargs: object
    ) -> tuple[int, int]:
        nonlocal swapped
        if not swapped:
            parent.rename(displaced)
            parent.mkdir()
            (parent / source_name).write_bytes(b"EVIL TEMP")
            (parent / target_name).write_bytes(b"EVIL INCUMBENT")
            swapped = True
        return original(descriptor, source_name, target_name, **kwargs)

    monkeypatch.setattr(
        contract_module, "_rename_directory_noreplace_at", swap_parent_then_rename
    )
    with pytest.raises(RagEvidenceContractError, match="parent path inode was replaced"):
        write_json_noreplace(target, {"status": "PASS"})
    assert target.read_bytes() == b"EVIL INCUMBENT"
    assert not (displaced / target.name).exists()
    quarantines = list(displaced.iterdir())
    assert len(quarantines) == 1
    assert json.loads(quarantines[0].read_text(encoding="utf-8")) == {
        "status": "PASS"
    }


def test_certificate_rename_is_no_replace_and_quarantines_temporary(tmp_path: Path) -> None:
    target = tmp_path / "CERTIFICATE.json"
    digest = write_json_noreplace(target, {"status": "PASS"})
    incumbent = target.read_bytes()
    assert digest == sha256_bytes(incumbent)
    with pytest.raises(FileExistsError, match="certificate already exists"):
        write_json_noreplace(target, {"status": "FAIL"})
    assert target.read_bytes() == incumbent
    quarantines = [
        path for path in tmp_path.iterdir()
        if "rag-evidence-quarantine" in path.name
    ]
    assert len(quarantines) == 1
    assert json.loads(quarantines[0].read_text(encoding="utf-8")) == {
        "status": "FAIL"
    }


def test_certificate_quarantines_wrong_inode_moved_inside_rename(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "CERTIFICATE.json"
    original = contract_module._rename_directory_noreplace_at
    backup_name = ".held-good-certificate"
    substituted = False

    def substitute_temporary_inside_rename(
        descriptor: int, source_name: str, target_name: str, **kwargs: object
    ) -> tuple[int, int]:
        nonlocal substituted
        if not substituted and target_name == target.name:
            original(descriptor, source_name, backup_name, **kwargs)
            evil = os.open(
                source_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=descriptor,
            )
            try:
                os.write(evil, b"EVIL CERTIFICATE")
            finally:
                os.close(evil)
            substituted = True
        return original(descriptor, source_name, target_name, **kwargs)

    monkeypatch.setattr(
        contract_module,
        "_rename_directory_noreplace_at",
        substitute_temporary_inside_rename,
    )
    with pytest.raises(RagEvidenceContractError, match="substituted temporary inode"):
        write_json_noreplace(target, {"status": "PASS"})
    assert not target.exists()
    assert (tmp_path / backup_name).is_file()
    quarantines = [
        path for path in tmp_path.iterdir()
        if "rag-evidence-quarantine" in path.name
    ]
    assert len(quarantines) == 1
    assert quarantines[0].read_bytes() == b"EVIL CERTIFICATE"


def test_certificate_quarantines_wrong_type_inserted_after_rename_syscall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "CERTIFICATE.json"
    original_entry_identity = contract_module._entry_identity
    backup_name = ".held-good-certificate-after-rename"
    substituted = False

    def substitute_final_before_post_stat(
        descriptor: int,
        name: str,
        *,
        require_directory: bool | None = None,
    ) -> tuple[int, int]:
        nonlocal substituted
        if not substituted and name == target.name:
            try:
                os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                os.rename(
                    name,
                    backup_name,
                    src_dir_fd=descriptor,
                    dst_dir_fd=descriptor,
                )
                os.mkdir(name, 0o700, dir_fd=descriptor)
                evil_directory = os.open(
                    name,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
                try:
                    evil = os.open(
                        "evil.txt",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                        dir_fd=evil_directory,
                    )
                    try:
                        os.write(evil, b"EVIL WRONG-TYPE FINAL")
                    finally:
                        os.close(evil)
                finally:
                    os.close(evil_directory)
                substituted = True
        return original_entry_identity(
            descriptor, name, require_directory=require_directory
        )

    monkeypatch.setattr(
        contract_module, "_entry_identity", substitute_final_before_post_stat
    )
    with pytest.raises(RagEvidenceContractError, match="substituted temporary inode"):
        write_json_noreplace(target, {"status": "PASS"})
    assert not target.exists()
    assert json.loads((tmp_path / backup_name).read_text()) == {"status": "PASS"}
    quarantines = [
        path for path in tmp_path.iterdir()
        if "rag-evidence-quarantine" in path.name
    ]
    assert len(quarantines) == 1
    assert (quarantines[0] / "evil.txt").read_bytes() == b"EVIL WRONG-TYPE FINAL"


def test_atomic_publication_rejects_symlink_parent_and_target(tmp_path: Path) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(OSError):
        AtomicRagDirectory(alias / "published")
    with pytest.raises(OSError):
        write_json_noreplace(alias / "CERTIFICATE.json", {"status": "PASS"})
    (real_parent / "nested").mkdir()
    with pytest.raises(OSError):
        AtomicRagDirectory(alias / "nested" / "published")
    with pytest.raises(OSError):
        write_json_noreplace(
            alias / "nested" / "CERTIFICATE.json", {"status": "PASS"}
        )

    victim = tmp_path / "victim.json"
    victim.write_bytes(b"INCUMBENT")
    target = real_parent / "CERTIFICATE.json"
    target.symlink_to(victim)
    with pytest.raises(FileExistsError, match="certificate already exists"):
        write_json_noreplace(target, {"status": "PASS"})
    assert target.is_symlink()
    assert victim.read_bytes() == b"INCUMBENT"


@pytest.mark.parametrize(
    "unsafe", ("../escaped", "nested/release", "nested\\release", ".hidden", "x y")
)
def test_release_identifier_rejects_traversal_before_source_open(
    unsafe: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(RagEvidenceContractError, match="unsafe RAG release ID"):
        validate_artifact_identifier(unsafe, name="RAG release ID")

    def forbidden_reconstruction(**_kwargs: object) -> dict:
        raise AssertionError("unsafe release ID reached registered-source reconstruction")

    monkeypatch.setattr(
        preparation_module,
        "reconstruct_rag_evidence_preparation",
        forbidden_reconstruction,
    )
    with pytest.raises(RagEvidenceContractError, match="unsafe RAG release ID"):
        preparation_module.prepare_rag_evidence_build(
            repo=REPO,
            registry_path=REGISTRY_PATH,
            source_root=tmp_path,
            release_root=tmp_path / "release",
            private_root=tmp_path / "private",
            release_id=unsafe,
            build_id="A",
            scientific_full=False,
        )
    assert not (tmp_path / "escaped").exists()


def test_preparation_pair_recovers_private_final_after_public_commit_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fit_bytes = b"target-free-fit-input"
    private_bytes = b"private-label-payload"
    reconstruction = {
        "registry": {"lane_id": "rag_evidence_benchmark_v1"},
        "fit_input_bytes": fit_bytes,
        "private_label_bytes": private_bytes,
        "fit_input_sha256": sha256_bytes(fit_bytes),
        "private_label_sha256": sha256_bytes(private_bytes),
        "source_binding": {"binding_sha256": "b" * 64},
        "source_snapshot": {"snapshot_sha256": "s" * 64},
        "rosters": {"synthetic": True},
    }
    monkeypatch.setattr(
        preparation_module,
        "reconstruct_rag_evidence_preparation",
        lambda **_kwargs: reconstruction,
    )
    release_id = "pair-crash-recovery"
    release_root = tmp_path / "release"
    private_root = tmp_path / "private"
    public_final = release_root / release_id / "rag_evidence" / "A"
    private_final = private_root / release_id / "rag_evidence" / "A"
    original_commit = AtomicRagDirectory.commit
    crashed = False

    def crash_at_public_commit(self: AtomicRagDirectory) -> None:
        nonlocal crashed
        if self.final_path == public_final.absolute() and not crashed:
            crashed = True
            raise SystemExit("deterministic crash after private commit")
        original_commit(self)

    monkeypatch.setattr(AtomicRagDirectory, "commit", crash_at_public_commit)
    with pytest.raises(SystemExit, match="deterministic crash"):
        preparation_module.prepare_rag_evidence_build(
            repo=REPO,
            registry_path=REGISTRY_PATH,
            source_root=tmp_path,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            build_id="A",
            scientific_full=False,
        )
    assert private_final.is_dir()
    assert not public_final.exists()
    leftover = public_final.parent / ".A.rag-staging-crash-evidence"
    leftover.mkdir()

    monkeypatch.setattr(AtomicRagDirectory, "commit", original_commit)
    manifest = preparation_module.prepare_rag_evidence_build(
        repo=REPO,
        registry_path=REGISTRY_PATH,
        source_root=tmp_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        build_id="A",
        scientific_full=False,
    )
    assert public_final.is_dir() and private_final.is_dir()
    assert leftover.is_dir()
    assert manifest["pair_transaction_id"]
    public_marker = public_final / preparation_module.PAIR_TRANSACTION_FILENAME
    private_marker = private_final / preparation_module.PAIR_TRANSACTION_FILENAME
    assert public_marker.read_bytes() == private_marker.read_bytes()
    orphan_quarantines = [
        path for path in private_final.parent.iterdir()
        if "recovered-private-orphan" in path.name
    ]
    assert len(orphan_quarantines) == 1
    assert (
        orphan_quarantines[0] / PRIVATE_LABEL_FILENAME
    ).read_bytes() == private_bytes
    retry_manifest = preparation_module.prepare_rag_evidence_build(
        repo=REPO,
        registry_path=REGISTRY_PATH,
        source_root=tmp_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        build_id="A",
        scientific_full=False,
    )
    assert retry_manifest == manifest
    assert len([
        path for path in private_final.parent.iterdir()
        if "recovered-private-orphan" in path.name
    ]) == 1
    tampered = dict(manifest)
    tampered["fit_input"] = {
        **tampered["fit_input"],
        "path": "../../attacker.pkl",
        "size_bytes": 999,
        "target_fields_present": True,
    }
    tampered["rosters"] = {"attacker": True}
    tampered.pop("payload_sha256")
    tampered = add_payload_sha256(tampered)
    (public_final / PREPARATION_MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(tampered) + b"\n"
    )
    with pytest.raises(
        RagEvidenceContractError,
        match="not the exact registered-source reconstruction",
    ):
        preparation_module.prepare_rag_evidence_build(
            repo=REPO,
            registry_path=REGISTRY_PATH,
            source_root=tmp_path,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            build_id="A",
            scientific_full=False,
        )


def test_complete_pair_recovery_rejects_public_swap_between_member_validations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fit_bytes = b"target-free-fit-input"
    private_bytes = b"private-label-payload"
    reconstruction = {
        "registry": {"lane_id": "rag_evidence_benchmark_v1"},
        "fit_input_bytes": fit_bytes,
        "private_label_bytes": private_bytes,
        "fit_input_sha256": sha256_bytes(fit_bytes),
        "private_label_sha256": sha256_bytes(private_bytes),
        "source_binding": {"binding_sha256": "b" * 64},
        "source_snapshot": {"snapshot_sha256": "s" * 64},
        "rosters": {"synthetic": True},
    }
    monkeypatch.setattr(
        preparation_module,
        "reconstruct_rag_evidence_preparation",
        lambda **_kwargs: reconstruction,
    )
    release_id = "pair-validation-swap"
    release_root = tmp_path / "release"
    private_root = tmp_path / "private"
    public_final = release_root / release_id / "rag_evidence" / "A"
    private_final = private_root / release_id / "rag_evidence" / "A"
    preparation_module.prepare_rag_evidence_build(
        repo=REPO,
        registry_path=REGISTRY_PATH,
        source_root=tmp_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        build_id="A",
        scientific_full=False,
    )

    original_validate = preparation_module._validate_recovery_artifact
    validated_backup = public_final.parent / ".validated-public-backup"
    attacker_fit = b"ATTACKER-FIT-WITH-COORDINATED-MANIFEST"
    swapped = False

    def swap_public_after_its_held_fd_validation(
        binding: object, **kwargs: object
    ) -> dict | None:
        nonlocal swapped
        result = original_validate(binding, **kwargs)
        if kwargs["kind"] == "public" and not swapped:
            public_final.rename(validated_backup)
            shutil.copytree(validated_backup, public_final)
            attacker_fit_path = (
                public_final / "inputs" / contract_module.FIT_INPUT_FILENAME
            )
            attacker_fit_path.write_bytes(attacker_fit)
            attacker_manifest_path = public_final / PREPARATION_MANIFEST_FILENAME
            attacker_manifest = json.loads(attacker_manifest_path.read_text())
            attacker_manifest["fit_input"] = {
                **attacker_manifest["fit_input"],
                "sha256": sha256_bytes(attacker_fit),
                "size_bytes": len(attacker_fit),
                "target_fields_present": True,
            }
            attacker_manifest["source_binding"] = {
                "binding_sha256": "a" * 64,
                "attacker": True,
            }
            attacker_manifest["source_binding_sha256"] = "a" * 64
            attacker_manifest["rosters"] = {"attacker": True}
            attacker_manifest.pop("payload_sha256")
            attacker_manifest = add_payload_sha256(attacker_manifest)
            attacker_manifest_path.write_bytes(
                canonical_json_bytes(attacker_manifest) + b"\n"
            )
            swapped = True
        return result

    monkeypatch.setattr(
        preparation_module,
        "_validate_recovery_artifact",
        swap_public_after_its_held_fd_validation,
    )
    with pytest.raises(
        RagEvidenceContractError,
        match="pair binding drifted during complete-pair validation",
    ):
        preparation_module.prepare_rag_evidence_build(
            repo=REPO,
            registry_path=REGISTRY_PATH,
            source_root=tmp_path,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            build_id="A",
            scientific_full=False,
        )
    assert not public_final.exists()
    assert not private_final.exists()
    assert (
        validated_backup / "inputs" / contract_module.FIT_INPUT_FILENAME
    ).read_bytes() == fit_bytes
    public_quarantines = [
        path for path in public_final.parent.iterdir()
        if "recovery-binding-drift" in path.name
    ]
    private_quarantines = [
        path for path in private_final.parent.iterdir()
        if "recovery-binding-drift" in path.name
    ]
    assert len(public_quarantines) == 1
    assert len(private_quarantines) == 1
    assert (
        public_quarantines[0] / "inputs" / contract_module.FIT_INPUT_FILENAME
    ).read_bytes() == attacker_fit
    assert (
        private_quarantines[0] / PRIVATE_LABEL_FILENAME
    ).read_bytes() == private_bytes


@pytest.mark.parametrize("attack", ("same_inode_rewrite", "name_inode_substitution"))
def test_complete_pair_recovery_revalidates_every_public_descendant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, attack: str
) -> None:
    fit_bytes = b"target-free-fit-input"
    private_bytes = b"private-label-payload"
    reconstruction = {
        "registry": {"lane_id": "rag_evidence_benchmark_v1"},
        "fit_input_bytes": fit_bytes,
        "private_label_bytes": private_bytes,
        "fit_input_sha256": sha256_bytes(fit_bytes),
        "private_label_sha256": sha256_bytes(private_bytes),
        "source_binding": {"binding_sha256": "b" * 64},
        "source_snapshot": {"snapshot_sha256": "s" * 64},
        "rosters": {"synthetic": True},
    }
    monkeypatch.setattr(
        preparation_module,
        "reconstruct_rag_evidence_preparation",
        lambda **_kwargs: reconstruction,
    )
    release_id = f"pair-descendant-{attack}"
    release_root = tmp_path / "release"
    private_root = tmp_path / "private"
    public_final = release_root / release_id / "rag_evidence" / "A"
    private_final = private_root / release_id / "rag_evidence" / "A"
    preparation_module.prepare_rag_evidence_build(
        repo=REPO,
        registry_path=REGISTRY_PATH,
        source_root=tmp_path,
        release_root=release_root,
        private_root=private_root,
        release_id=release_id,
        build_id="A",
        scientific_full=False,
    )

    original_validate = preparation_module._validate_recovery_artifact
    attacker_fit = b"X" * len(fit_bytes)
    displaced_good = tmp_path / f"{attack}-displaced-good.pkl"
    attacked = False

    def attack_after_public_validation(
        binding: object, **kwargs: object
    ) -> dict | None:
        nonlocal attacked
        result = original_validate(binding, **kwargs)
        if kwargs["kind"] == "public" and not attacked:
            fit_path = public_final / "inputs" / contract_module.FIT_INPUT_FILENAME
            if attack == "same_inode_rewrite":
                original_identity = (fit_path.stat().st_dev, fit_path.stat().st_ino)
                fit_path.write_bytes(attacker_fit)
                assert (fit_path.stat().st_dev, fit_path.stat().st_ino) == original_identity
            else:
                fit_path.rename(displaced_good)
                fit_path.write_bytes(attacker_fit)
            attacked = True
        return result

    monkeypatch.setattr(
        preparation_module,
        "_validate_recovery_artifact",
        attack_after_public_validation,
    )
    with pytest.raises(
        RagEvidenceContractError,
        match="pair descendant drifted during complete-pair validation",
    ):
        preparation_module.prepare_rag_evidence_build(
            repo=REPO,
            registry_path=REGISTRY_PATH,
            source_root=tmp_path,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            build_id="A",
            scientific_full=False,
        )
    assert attacked
    assert not public_final.exists()
    assert not private_final.exists()
    if attack == "name_inode_substitution":
        assert displaced_good.read_bytes() == fit_bytes
    public_quarantines = [
        path for path in public_final.parent.iterdir()
        if "recovery-binding-drift" in path.name
    ]
    private_quarantines = [
        path for path in private_final.parent.iterdir()
        if "recovery-binding-drift" in path.name
    ]
    assert len(public_quarantines) == 1
    assert len(private_quarantines) == 1
    assert (
        public_quarantines[0] / "inputs" / contract_module.FIT_INPUT_FILENAME
    ).read_bytes() == attacker_fit
    assert (
        private_quarantines[0] / PRIVATE_LABEL_FILENAME
    ).read_bytes() == private_bytes


def test_restricted_worker_scores_without_reading_private_probe(tmp_path: Path) -> None:
    registry = load_registry(REGISTRY_PATH)
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    fit_path = input_root / "FIT_INPUT.pkl"
    fit_path.write_bytes(pickle_bytes(_fit_input()))
    private_probe = tmp_path / "PRIVATE_LABELS.pkl"
    private_probe.write_bytes(b"controller-only target sentinel")
    code_root = _copy_capsule(REPO, tmp_path / "capsule")
    temp_root = tmp_path / "worker_tmp"
    temp_root.mkdir()
    output_root = tmp_path / "candidate"
    policy = _policy(
        code_root=code_root, input_root=input_root, output_root=output_root,
        temp_root=temp_root,
        forbidden=[("private_labels", private_probe), ("raw_registry", REGISTRY_PATH)],
    )
    _launch_worker(
        code_root=code_root, input_path=fit_path,
        input_sha256=sha256_file(fit_path), output_root=output_root,
        temp_root=temp_root, release_id="synthetic", build_id="A",
        lane_id=registry["lane_id"],
        forbidden_fields=list(registry["fit_visibility"]["forbidden_fields"]),
        policy=policy,
    )
    result = json.loads((output_root / "WORKER_RESULT.json").read_text(encoding="utf-8"))
    assert result["labels_opened_by_fit"] is False
    assert result["historical_scores_opened"] is False
    assert result["firewall_violations"] == []
    assert result["denial_probes"] == [
        {"probe_id": "private_labels", "read_denied": True},
        {"probe_id": "raw_registry", "read_denied": True},
    ]


def test_score_certificate_third_worker_reexecutes_exact_score_bytes(
    tmp_path: Path,
) -> None:
    registry = load_registry(REGISTRY_PATH)
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    fit_path = input_root / "FIT_INPUT.pkl"
    fit_input = _fit_input()
    fit_path.write_bytes(pickle_bytes(fit_input))
    private_probe = tmp_path / "PRIVATE_LABELS.pkl"
    private_probe.write_bytes(b"controller-only target sentinel")

    independent = _independently_reexecute_score_worker(
        repo=REPO,
        registry_path=REGISTRY_PATH,
        source_root=tmp_path,
        private_label_path=private_probe,
        fit_input_path=fit_path,
        fit_input_sha256=sha256_file(fit_path),
        registry=registry,
        release_id="synthetic-third-worker",
    )
    expected_arrays, _ = compute_rag_evidence_scores(fit_input)
    expected_bytes = deterministic_npz_bytes(expected_arrays)
    assert independent["score_bytes"] == expected_bytes
    assert independent["score_sha256"] == sha256_bytes(expected_bytes)
    assert independent["capsule_tree"]["files"]
    assert independent["denial_probes"]
    assert all(row["read_denied"] is True for row in independent["denial_probes"])


def test_registry_has_no_historical_result_source() -> None:
    raw = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert all(not item["path"].startswith("results/") for item in raw["sources"].values())
    assert raw["method_contract"]["lettucedetect_large_modernbert"]["score_source"].startswith("freshly rederived")


def _independent_score_fixture(payload: bytes = b"canonical-score") -> dict:
    capsule = {"tree_sha256": "c" * 64}
    diagnostics = {"fit_rows": 17, "labels_seen_during_fit": False}
    return {
        "independent": {
            "score_bytes": payload,
            "score_sha256": sha256_bytes(payload),
            "diagnostics": diagnostics,
            "capsule_tree": capsule,
        },
        "scores": {"A": payload, "B": payload},
        "manifests": {
            "A": {"scores": {"sha256": sha256_bytes(payload)}},
            "B": {"scores": {"sha256": sha256_bytes(payload)}},
        },
        "worker_diagnostics": {"A": diagnostics, "B": diagnostics},
        "capsule_trees": {"A": capsule, "B": capsule},
    }


def test_independent_score_reexecution_accepts_exact_bytes_and_capsule() -> None:
    checks = _assert_independent_score_match(**_independent_score_fixture())
    assert checks and all(checks.values())


def test_independent_score_reexecution_rejects_coordinated_ab_rewrite() -> None:
    fixture = _independent_score_fixture()
    tampered = b"coordinated-fabricated-score"
    fixture["scores"] = {"A": tampered, "B": tampered}
    fixture["manifests"] = {
        "A": {"scores": {"sha256": sha256_bytes(tampered)}},
        "B": {"scores": {"sha256": sha256_bytes(tampered)}},
    }
    with pytest.raises(RagEvidenceContractError, match="coordinated score/capsule rewrite"):
        _assert_independent_score_match(**fixture)


def _evaluation_match_fixture() -> dict:
    expected_payloads = {
        "metrics.csv": b"panel_id,value\nragtruth_answer,0.5\n",
        "predictions.csv": b"panel_id,unit_id\nragtruth_answer,u0\n",
        "contrasts.csv": b"panel_id,delta\ngasp_protocol_sentence,0.1\n",
        "panel_status.csv": b"panel_id,status\nragtruth_answer,PASS\n",
    }
    panel_status = [{"panel_id": panel_id, "status": "PASS"} for panel_id in PANEL_IDS]
    return {
        "expected_payloads": expected_payloads,
        "observed_payloads": {
            "A": dict(expected_payloads), "B": dict(expected_payloads)
        },
        "manifests": {
            "A": {"panel_status": panel_status},
            "B": {"panel_status": panel_status},
        },
        "expected_panel_status": panel_status,
    }


def test_independent_evaluation_accepts_exact_regenerated_tables() -> None:
    checks = _assert_independent_evaluation_match(**_evaluation_match_fixture())
    assert checks and all(checks.values())


def test_independent_evaluation_rejects_coordinated_allowed_csv_rewrite() -> None:
    fixture = _evaluation_match_fixture()
    fabricated = {
        name: b"panel_id,status\nragtruth_answer,PASS\n"
        for name in fixture["expected_payloads"]
    }
    fixture["observed_payloads"] = {"A": fabricated, "B": dict(fabricated)}
    with pytest.raises(RagEvidenceContractError, match="coordinated evaluation rewrite"):
        _assert_independent_evaluation_match(**fixture)


def test_full_evaluation_certificate_rejects_private_post_hash_swap_before_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the former cert forgery window through the complete A/B gate."""

    release_id = "private-cert-race"
    release_root = tmp_path / "release"
    private_root = tmp_path / "private"
    lane_root = release_root / release_id / "rag_evidence"
    private_lane_root = private_root / release_id / "rag_evidence"
    registry = {
        "lane_id": "rag_evidence_benchmark_v1",
        "evaluation": {"bootstrap": {"draws": 3, "seed": 17}},
        "panels": [
            {
                "panel_id": panel_id,
                "dataset": "synthetic",
                "unit": "claim",
                "access": "synthetic",
                "estimand": "synthetic",
                "metrics": ["metric"],
                "methods": ["method"],
            }
            for panel_id in PANEL_IDS
        ],
    }
    monkeypatch.setattr(ab_module, "load_registry", lambda _path: registry)

    original_private = _synthetic_private_payload(marker="ORIGINAL")
    attacker_private = _synthetic_private_payload(marker="ATTACKER")
    private_sha = sha256_bytes(original_private)
    score_payload = b"synthetic-score-archive"
    score_sha = sha256_bytes(score_payload)
    score_certificate = add_payload_sha256({
        "schema_version": contract_module.SCORE_AB_SCHEMA,
        "status": "PASS",
        "score_sha256": score_sha,
        "private_label_sha256": private_sha,
        "source_binding_sha256": "b" * 64,
    })
    lane_root.mkdir(parents=True)
    (lane_root / ab_module.SCORE_CERTIFICATE).write_bytes(
        canonical_json_bytes(score_certificate) + b"\n"
    )
    monkeypatch.setattr(
        ab_module,
        "authenticate_rag_evidence_score_certificate",
        lambda **_kwargs: score_certificate,
    )

    snapshot = {
        "files": [
            {"path": relative, "sha256": sha256_file(REPO / relative)}
            for relative in evaluation_module.EVALUATION_SOURCE_FILES
        ]
    }
    snapshot["snapshot_sha256"] = contract_module.payload_sha256(snapshot)
    panel_status = [
        {"panel_id": panel_id, "status": "PASS"}
        for panel_id in PANEL_IDS
    ]
    metrics = "panel_id,metric,method_id,subgroup\n" + "".join(
        f"{panel_id},metric,method,"
        f"{'accurate_context' if panel_id.startswith('refchecker_') else 'all'}\n"
        for panel_id in PANEL_IDS
    )
    reporting_payloads = {
        "metrics.csv": metrics.encode("utf-8"),
        "predictions.csv": (
            f"panel_id,subgroup\n{PANEL_IDS[0]},all\n"
        ).encode("utf-8"),
        "contrasts.csv": (
            "panel_id,left_method,right_method\n"
            "gasp_protocol_sentence,gasp_threshold,fixed_rag_iu_pcr_matched\n"
        ).encode("utf-8"),
        "panel_status.csv": (
            "panel_id,status\n"
            + "".join(f"{row['panel_id']},PASS\n" for row in panel_status)
        ).encode("utf-8"),
    }

    private_paths: dict[str, Path] = {}
    for build_id in ("A", "B"):
        build_root = lane_root / build_id
        fit_root = build_root / "fit"
        evaluation_root = build_root / "evaluation"
        private_build = private_lane_root / build_id
        (fit_root / "candidate").mkdir(parents=True)
        evaluation_root.mkdir()
        private_build.mkdir(parents=True)
        private_path = private_build / PRIVATE_LABEL_FILENAME
        private_path.write_bytes(original_private)
        private_paths[build_id] = private_path
        preparation = add_payload_sha256({
            "schema_version": contract_module.PREPARATION_SCHEMA,
            "private_labels": {
                "path": str(private_path.absolute()),
                "sha256": private_sha,
                "size_bytes": len(original_private),
            },
        })
        (build_root / PREPARATION_MANIFEST_FILENAME).write_bytes(
            canonical_json_bytes(preparation) + b"\n"
        )
        (fit_root / "candidate" / contract_module.SCORES_FILENAME).write_bytes(
            score_payload
        )
        score_manifest = add_payload_sha256({
            "schema_version": contract_module.SCORE_FREEZE_SCHEMA,
            "scores": {
                "path": f"candidate/{contract_module.SCORES_FILENAME}",
                "sha256": score_sha,
                "size_bytes": len(score_payload),
            },
        })
        score_manifest_payload = canonical_json_bytes(score_manifest) + b"\n"
        (fit_root / SCORE_MANIFEST_FILENAME).write_bytes(score_manifest_payload)
        declared_files = []
        for name, payload in reporting_payloads.items():
            (evaluation_root / name).write_bytes(payload)
            declared_files.append({
                "path": name,
                "sha256": sha256_bytes(payload),
                "size_bytes": len(payload),
            })
        evaluation_manifest = add_payload_sha256({
            "schema_version": contract_module.EVALUATION_SCHEMA,
            "release_id": release_id,
            "build_id": build_id,
            "lane_id": registry["lane_id"],
            "scientific_full": False,
            "score_sha256": score_sha,
            "score_manifest_sha256": sha256_bytes(score_manifest_payload),
            "private_label_sha256": private_sha,
            "source_binding_sha256": "b" * 64,
            "score_ab_certificate_sha256": sha256_file(
                lane_root / ab_module.SCORE_CERTIFICATE
            ),
            "source_snapshot": snapshot,
            "bootstrap": {
                "draws_requested": 3,
                "group": "panel-registered source group",
                "paired_contrasts": True,
                "seed": 17,
            },
            "files": declared_files,
            "panel_status": panel_status,
            "cross_panel_macro_computed": False,
            "refchecker_settings_pooled": False,
            "historical_scores_copied": False,
        })
        (evaluation_root / contract_module.EVALUATION_MANIFEST_FILENAME).write_bytes(
            canonical_json_bytes(evaluation_manifest) + b"\n"
        )

    monkeypatch.setattr(ab_module, "load_scores_bytes", lambda _payload: {})
    compute_called = False

    def forbidden_compute(**_kwargs: object) -> dict:
        nonlocal compute_called
        compute_called = True
        raise AssertionError("evaluation ran after private inode substitution")

    monkeypatch.setattr(
        evaluation_module,
        "compute_rag_evidence_evaluation_tables",
        forbidden_compute,
    )
    attacker_path = tmp_path / "attacker-private.pkl"
    attacker_path.write_bytes(attacker_private)
    backup_path = tmp_path / "original-private.pkl"
    original_rebind = contract_module.BoundRagFile._assert_rebound
    swapped = False

    def swap_a_private_after_hash(
        binding: contract_module.BoundRagFile,
    ) -> None:
        nonlocal swapped
        if binding.path == private_paths["A"].absolute() and not swapped:
            private_paths["A"].rename(backup_path)
            attacker_path.rename(private_paths["A"])
            swapped = True
        original_rebind(binding)

    monkeypatch.setattr(
        contract_module.BoundRagFile,
        "_assert_rebound",
        swap_a_private_after_hash,
    )
    with pytest.raises(RagEvidenceContractError):
        ab_module._derive_evaluation_certificate(
            repo=REPO,
            registry_path=REGISTRY_PATH,
            source_root=tmp_path,
            release_root=release_root,
            private_root=private_root,
            release_id=release_id,
            require_scientific_full=False,
        )
    assert swapped
    assert not compute_called
    assert private_paths["A"].read_bytes() == attacker_private


def test_evaluation_authenticates_score_before_private_open_or_compute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release_id = "auth-order"
    build_root = tmp_path / "release" / release_id / "rag_evidence" / "A"
    (build_root / "fit").mkdir(parents=True)
    preparation = add_payload_sha256({"schema_version": "test-preparation"})
    score = add_payload_sha256({"schema_version": "test-score"})
    (build_root / PREPARATION_MANIFEST_FILENAME).write_text(
        json.dumps(preparation), encoding="utf-8"
    )
    (build_root / "fit" / SCORE_MANIFEST_FILENAME).write_text(
        json.dumps(score), encoding="utf-8"
    )
    events: list[str] = []

    def fail_score_authentication(**_kwargs: object) -> dict:
        events.append("score_authentication")
        raise RagEvidenceContractError("score authentication sentinel")

    def forbidden_private_open(*_args: object, **_kwargs: object) -> dict:
        events.append("private_open")
        raise AssertionError("private labels opened before score authentication")

    def forbidden_evaluation_compute(*_args: object, **_kwargs: object) -> dict:
        events.append("evaluation_compute")
        raise AssertionError("evaluation computed before score authentication")

    monkeypatch.setattr(
        ab_module,
        "authenticate_rag_evidence_score_certificate",
        fail_score_authentication,
    )
    monkeypatch.setattr(evaluation_module, "load_private_labels", forbidden_private_open)
    monkeypatch.setattr(
        evaluation_module,
        "compute_rag_evidence_evaluation_tables",
        forbidden_evaluation_compute,
    )
    with pytest.raises(RagEvidenceContractError, match="score authentication sentinel"):
        evaluation_module.evaluate_rag_evidence_build(
            repo=REPO,
            registry_path=REGISTRY_PATH,
            source_root=tmp_path,
            release_root=tmp_path / "release",
            private_root=tmp_path / "private",
            release_id=release_id,
            build_id="A",
            draws_override=5,
        )
    assert events == ["score_authentication"]
