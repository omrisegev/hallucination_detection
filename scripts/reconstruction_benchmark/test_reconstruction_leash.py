#!/usr/bin/env python3
"""Synthetic end-to-end contract tests for the actual LEASH stopping lane."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.paper_exact.manifest import sha256_order  # noqa: E402
from spectral_utils.paper_exact.leash import LeashConfig, LeashStopper  # noqa: E402
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    canonical_json_bytes,
    canonical_tree_manifest,
    sha256_file,
)
from spectral_utils.reconstruction_benchmark import leash_contract as leash_contract_module  # noqa: E402
from spectral_utils.reconstruction_benchmark import leash_evaluation as leash_evaluation_module  # noqa: E402
from spectral_utils.reconstruction_benchmark import leash_fit as leash_fit_module  # noqa: E402
from spectral_utils.reconstruction_benchmark import leash_preparation as leash_preparation_module  # noqa: E402
from spectral_utils.reconstruction_benchmark.leash_ab import (  # noqa: E402
    verify_leash_fit_ab,
    verify_leash_preparation_ab,
)
from spectral_utils.reconstruction_benchmark.leash_contract import (  # noqa: E402
    AtomicLeashDirectory,
    FIT_FORBIDDEN_FIELDS,
    LeashContractError,
    add_payload_sha256,
    assert_no_forbidden_keys,
    canonical_jsonl_bytes,
    load_jsonl,
    load_registry,
    write_json_noreplace,
)
from spectral_utils.reconstruction_benchmark.leash_evaluation import (  # noqa: E402
    derive_leash_evaluation,
    evaluate_leash_build,
    verify_leash_evaluation_ab,
)
from spectral_utils.reconstruction_benchmark.leash_fit import (  # noqa: E402
    derive_policy_ledger,
    run_leash_fit,
)
from spectral_utils.reconstruction_benchmark.leash_preparation import (  # noqa: E402
    FIT_INPUT_FILENAME,
    audit_leash_sources,
    prepare_leash_build,
)


PRODUCTION_REGISTRY = REPO / "configs/reconstruction_benchmark_v1/leash_stopping.json"
TEST_TEMP_ROOT = Path(tempfile.gettempdir()).resolve(strict=True)
MODELS = ("fixture/model-a", "fixture/model-b", "fixture/model-c")


def _raw_record(question_id: str, arm: str, answer: str, gold: str, index: int) -> dict:
    stopped = arm == "leash" and index != 2
    reasoning = 0 if arm == "nocot" else (80 if stopped else 320)
    closure = 2
    channels = {}
    leash = None
    if arm == "leash":
        entropy = ([1.0] * 8 + [0.8] * (reasoning - 8)) if stopped else [1.0] * reasoning
        margin = [0.1] * reasoning
        pmax = [0.5] * reasoning
        channels = {"raw_entropy": entropy, "raw_margin": margin, "raw_pmax": pmax}
        replay = LeashStopper(LeashConfig())
        for values in zip(entropy, margin, pmax, strict=True):
            replay.push(*values)
        leash = replay.diagnostics()
    return {
        "trace_key": f"{arm}:central:{question_id}",
        "question_id": question_id,
        "arm": arm,
        "setting_label": "central",
        "prompt_text": "private fixture prompt",
        "prompt_token_ids": [1],
        "gen_token_ids": list(range(reasoning)),
        "full_text": "private fixture response",
        "channels": channels,
        "answer_text": answer,
        "answer_token_ids": [2, 3],
        "n_reasoning_tokens": reasoning,
        "n_closure_tokens": closure,
        "n_total_tokens": reasoning + closure,
        "stop_reason": "policy" if stopped else ("n/a" if arm == "nocot" else "length"),
        "stopped_early": stopped,
        "closure_generated": True,
        "leash": leash,
        "gold_answer": gold,
        "correct": False,
        "pred_answer": "17",
        "parse_status": "fallback_number",
        "wall_s": {"cot": 3.0, "leash": 2.0, "nocot": 1.0}[arm] + index / 10,
    }


def _write_ready_run(
    source: Path, *, dataset: str, model: str, omit_leash_channels: bool = False
) -> tuple[Path, dict]:
    slug = model.replace("/", "-")
    run = source / "ready" / f"s2_leash_{slug}_{dataset}"
    (run / "shards").mkdir(parents=True)
    question_ids = [f"{dataset}:{index}" for index in range(3)]
    if dataset == "aqua":
        source_name, golds = "deepmind/aqua_rat", ["A", "B", "C"]
        answers = {
            "cot": ["A) alpha", "The answer is B.", r"\boxed{C}"],
            "leash": ["The correct option is A.", "The answer is C.", "choice C)"],
            "nocot": ["17", "B)", "numeric only: 3"],
        }
    else:
        source_name, golds = "openai/gsm8k", ["1", "2", "3"]
        answers = {
            "cot": [r"\boxed{1}", r"\boxed{2}", r"\boxed{3}"],
            "leash": ["1", "2", "0"],
            "nocot": ["1", "0", "3"],
        }
    records = [
        _raw_record(question, arm, answers[arm][index], golds[index], index)
        for arm in ("leash", "cot", "nocot")
        for index, question in enumerate(question_ids)
    ]
    if omit_leash_channels:
        first_leash = next(record for record in records if record["arm"] == "leash")
        first_leash.pop("channels")
    shard = run / "shards/shard_00000.pkl"
    with shard.open("wb") as handle:
        pickle.dump(records, handle, protocol=pickle.HIGHEST_PROTOCOL)
    shard_sha = sha256_file(shard)
    index = {
        "shard": 0, "path": "shards/shard_00000.pkl", "n_traces": len(records),
        "bytes": shard.stat().st_size, "sha256": shard_sha,
        "keys": [record["trace_key"] for record in records],
        "question_ids": sorted(question_ids),
    }
    (run / "INDEX.jsonl").write_text(json.dumps(index) + "\n", encoding="utf-8")
    manifest = {
        "schema": "paper_exact_acquisition_v1", "run_id": run.name,
        "fidelity": "paper-specified-partial", "dataset_source": source_name,
        "dataset_revision": "test", "dataset_example_ids": question_ids,
        "dataset_order_sha256": sha256_order(question_ids), "model_id": model,
        "model_revision": "fixture-revision", "evaluator_revision": "paper_exact_evaluator_v1.0.0",
        "repo_commit": "4b6b81015971fc332db603468ff69c2925cc3084", "repo_dirty": False,
        "stop_behavior": {
            "rationale_cap": 320, "policy": "LEASH Alg. 1 (leash arm only)",
            "second_stage": "\n\nTherefore, the final answer is",
        },
        "expected_traces": len(records),
        "extra": {
            "arms": ["leash", "cot", "nocot"], "sweep": False,
            "leash_config": {
                "published": {"k": 8, "L": 5, "eps_H": 0.005, "delta_M": 0.05, "m": 64, "M": 320},
                "declared_by_us": {"B": 30.0, "tau_p": 0.95, "w": 16, "gamma": 0.1},
                "setting_label": "central", "t_min": 80,
                "fidelity": "paper-specified-partial",
            },
        },
    }
    (run / "RUN_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    status = {
        "n_expected": len(records), "n_finished": len(records), "n_failed": 0,
        "n_shards": 1, "bytes_total": shard.stat().st_size, "failures": [], "complete": True,
    }
    (run / "STATUS.json").write_text(json.dumps(status), encoding="utf-8")
    (run / "SUMMARY.json").write_text(json.dumps({"run_id": run.name}), encoding="utf-8")
    (run / "GATE_S2-leash-full.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    tree = canonical_tree_manifest(run)
    spec = {
        "run_id": run.name, "path": run.relative_to(source).as_posix(),
        "dataset": dataset, "model": model, "expected_questions": 3, "expected_traces": 9,
        "expected_leash_policy_stops": 2,
        "file_count": len(tree["files"]),
        "bytes_total": sum(item["bytes"] for item in tree["files"]),
        "tree_sha256": tree["tree_sha256"],
        "manifest_sha256": sha256_file(run / "RUN_MANIFEST.json"),
        "index_sha256": sha256_file(run / "INDEX.jsonl"),
        "status_sha256": sha256_file(run / "STATUS.json"),
        "summary_sha256": sha256_file(run / "SUMMARY.json"),
    }
    return run, spec


def _write_blocked_run(source: Path, *, dataset: str) -> dict:
    run = source / "blocked" / f"s2_leash_Mistral-7B-v0.1_{dataset}"
    run.mkdir(parents=True)
    manifest = {
        "run_id": run.name, "model_id": "mistralai/Mistral-7B-v0.1",
        "fidelity": "paper-specified-partial", "expected_traces": 9,
    }
    status = {
        "complete": True, "n_expected": 9, "n_finished": 0, "n_failed": 9,
        "n_shards": 0, "failures": [{
            "trace_key": "fixture", "reason": "ValueError: tokenizer.chat_template is not set"
        }],
    }
    gate = {"passed": True, "checks": []}
    values = {
        "RUN_MANIFEST.json": manifest,
        "STATUS.json": status,
        "GATE_S2-leash-full.json": gate,
    }
    for name, value in values.items():
        (run / name).write_text(json.dumps(value), encoding="utf-8")
    return {
        "run_id": run.name, "path": run.relative_to(source).as_posix(),
        "dataset": dataset, "model": "mistralai/Mistral-7B-v0.1",
        "expected_traces": 9, "expected_failed": 9,
        "coverage_status": "PROTOCOL_GATE_FAILED",
        "failure_signature": "tokenizer.chat_template is not set",
        "files": {name: sha256_file(run / name) for name in values},
    }


def _fixture_registry(
    root: Path, *, omit_leash_channels: bool = False
) -> tuple[Path, Path]:
    source = root / "source"
    source.mkdir()
    registry = deepcopy(load_registry(PRODUCTION_REGISTRY))
    ready = []
    first = True
    for model in MODELS:
        for dataset in ("aqua", "gsm8k"):
            _, spec = _write_ready_run(
                source, dataset=dataset, model=model,
                omit_leash_channels=omit_leash_channels and first,
            )
            ready.append(spec)
            first = False
    blocked = [_write_blocked_run(source, dataset=dataset) for dataset in ("aqua", "gsm8k")]
    implementation = {}
    for name in registry["source_contract"]["implementation_files"]:
        path = source / "implementation" / f"{name}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# synthetic {name}\n", encoding="utf-8")
        implementation[name] = {
            "path": path.relative_to(source).as_posix(), "sha256": sha256_file(path),
        }
    source_guard_code = {}
    for name, spec in registry["source_contract"]["source_guard_code_files"].items():
        relative = str(spec["path"])
        original = REPO / relative
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(original.read_bytes())
        source_guard_code[name] = {
            "path": relative,
            "sha256": sha256_file(path),
        }
    registry["source_contract"] = {
        "implementation_files": implementation,
        "source_guard_code_files": source_guard_code,
        "ready_runs": ready,
        "blocked_runs": blocked,
    }
    registry["population"].update(
        {
            "expected_ready_traces": 54,
            "expected_ready_questions_by_dataset": {"aqua": 3, "gsm8k": 3},
            "expected_models": list(MODELS),
        }
    )
    registry["evaluation"]["bootstrap"].update({"draws": 20, "seed": 41})
    path = root / "leash_registry.json"
    path.write_text(json.dumps(registry, sort_keys=True), encoding="utf-8")
    load_registry(path)
    return source, path


def _run_pipeline(root: Path) -> dict:
    source, registry = _fixture_registry(root)
    release = root / "release"
    private = root / "private"
    lane = release / "leash"
    for build in ("A", "B"):
        prepare_leash_build(
            source_root=source, registry_path=registry,
            public_output=lane / build / "preparation",
            private_output=private / build / "outcomes",
        )
    prep_cert = lane / "PREPARATION_AB_VERIFICATION.json"
    verify_leash_preparation_ab(
        source_root=source, registry_path=registry,
        public_a=lane / "A/preparation", private_a=private / "A/outcomes",
        public_b=lane / "B/preparation", private_b=private / "B/outcomes",
        certificate_path=prep_cert,
    )
    for build in ("A", "B"):
        run_leash_fit(
            source_root=source,
            preparation_dir=lane / build / "preparation",
            preparation_ab_certificate=prep_cert, registry_path=registry,
            output_dir=lane / build / "fit",
        )
    fit_cert = lane / "FIT_AB_VERIFICATION.json"
    verify_leash_fit_ab(
        source_root=source,
        registry_path=registry,
        preparation_a=lane / "A/preparation", preparation_b=lane / "B/preparation",
        preparation_ab_certificate=prep_cert,
        fit_a=lane / "A/fit", fit_b=lane / "B/fit", certificate_path=fit_cert,
    )
    for build in ("A", "B"):
        evaluate_leash_build(
            source_root=source, preparation_dir=lane / build / "preparation",
            fit_dir=lane / build / "fit", private_dir=private / build / "outcomes",
            preparation_ab_certificate=prep_cert, fit_ab_certificate=fit_cert,
            registry_path=registry, output_dir=lane / build / "evaluation",
        )
    eval_cert = lane / "EVALUATION_AB_VERIFICATION.json"
    verify_leash_evaluation_ab(
        source_root=source,
        preparation_a=lane / "A/preparation", fit_a=lane / "A/fit",
        private_a=private / "A/outcomes", evaluation_a=lane / "A/evaluation",
        preparation_b=lane / "B/preparation", fit_b=lane / "B/fit",
        private_b=private / "B/outcomes", evaluation_b=lane / "B/evaluation",
        preparation_ab_certificate=prep_cert, fit_ab_certificate=fit_cert,
        registry_path=registry, certificate_path=eval_cert,
    )
    return {
        "source": source, "registry": registry, "release": release, "private": private,
        "lane": lane, "prep_cert": prep_cert, "fit_cert": fit_cert, "eval_cert": eval_cert,
    }


class LeashContractTests(unittest.TestCase):
    def test_production_registry_preserves_scientific_boundary(self) -> None:
        registry = load_registry(PRODUCTION_REGISTRY)
        self.assertEqual(registry["fidelity"], "paper-specified-partial")
        self.assertEqual(len(registry["source_contract"]["ready_runs"]), 6)
        self.assertEqual(len(registry["source_contract"]["blocked_runs"]), 2)
        self.assertEqual(
            registry["claim_boundary"]["conceptual_objective_status"],
            "CONCEPTUAL_ONLY_NOT_REPRODUCED_EQUATION",
        )

    def test_all_leash_clis_reject_traversal_and_absolute_release_ids(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            release_root = root / "release"
            private_root = root / "private"
            source_root = root / "source"
            escaped = root / "escaped"
            absolute_escaped = root / "absolute-escaped"
            scripts = (
                ("prepare_leash_stopping.py", ["--build", "A", "--private-root", str(private_root)]),
                ("run_leash_stopping.py", ["--build", "A"]),
                ("evaluate_leash_stopping.py", ["--build", "A", "--private-root", str(private_root)]),
                ("verify_leash_preparation_ab.py", ["--private-root", str(private_root)]),
                ("verify_leash_ab.py", []),
                ("verify_leash_evaluation_ab.py", ["--private-root", str(private_root)]),
            )
            for release_id in ("../escaped", str(absolute_escaped)):
                for script_name, extras in scripts:
                    command = [
                        sys.executable,
                        str(REPO / "scripts/reconstruction_benchmark" / script_name),
                        "--release-id",
                        release_id,
                        "--source-root",
                        str(source_root),
                        "--release-root",
                        str(release_root),
                        "--registry",
                        str(PRODUCTION_REGISTRY),
                        *extras,
                    ]
                    completed = subprocess.run(command, capture_output=True, text=True)
                    self.assertNotEqual(completed.returncode, 0, script_name)
                    self.assertIn("unsafe LEASH release ID", completed.stderr)
            self.assertFalse(escaped.exists())
            self.assertFalse(absolute_escaped.exists())

    def test_atomic_no_clobber_and_cleanup(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            target = Path(temporary) / "tree"
            stage = AtomicLeashDirectory(target)
            stage.write_bytes("partial", b"partial")
            stage.cleanup()
            self.assertFalse(target.exists())
            target.mkdir()
            (target / "incumbent").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                AtomicLeashDirectory(target)
            certificate = Path(temporary) / "CERT.json"
            write_json_noreplace(certificate, {"incumbent": True})
            with self.assertRaises(FileExistsError):
                write_json_noreplace(certificate, {"candidate": True})
            self.assertIn("incumbent", certificate.read_text())

    def test_output_parent_symlink_is_rejected_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)

            with self.assertRaisesRegex(LeashContractError, "symlink component"):
                AtomicLeashDirectory(alias / "published")
            self.assertFalse((real / "published").exists())

            with self.assertRaisesRegex(LeashContractError, "symlink component"):
                write_json_noreplace(alias / "CERT.json", {"forged": False})
            self.assertFalse((real / "CERT.json").exists())

    def test_directory_publication_parent_swap_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            parent = root / "publication-parent"
            victim = root / "victim"
            moved = root / "publication-parent-moved"
            parent.mkdir()
            victim.mkdir()
            stage = AtomicLeashDirectory(parent / "published")
            stage.write_bytes("payload.json", b"genuine")
            original = leash_contract_module._rename_directory_noreplace_at
            swapped = False

            def swap_parent_then_rename(
                parent_fd: int, source_name: str, target_name: str
            ) -> None:
                nonlocal swapped
                if not swapped:
                    parent.rename(moved)
                    parent.symlink_to(victim, target_is_directory=True)
                    swapped = True
                original(parent_fd, source_name, target_name)

            try:
                with patch.object(
                    leash_contract_module,
                    "_rename_directory_noreplace_at",
                    side_effect=swap_parent_then_rename,
                ):
                    with self.assertRaisesRegex(LeashContractError, "path binding changed"):
                        stage.commit()
            finally:
                stage.cleanup()

            self.assertFalse((victim / "published").exists())
            self.assertFalse((moved / "published").exists())
            recovery = list(moved.glob(".published.leash-recovery-*"))
            self.assertEqual(len(recovery), 1)

    def test_directory_post_stat_substitution_cannot_report_success(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            stage = AtomicLeashDirectory(parent / "published")
            stage.write_bytes("PAYLOAD", b"GENUINE")
            attacker = parent / "attacker"
            attacker.mkdir()
            (attacker / "PAYLOAD").write_bytes(b"FORGED")
            genuine_moved = parent / "genuine-moved-inside-stat"
            original = leash_contract_module._entry_stat
            swapped = False

            def stat_then_substitute(parent_fd: int, name: str):
                nonlocal swapped
                observed = original(parent_fd, name)
                if name == "published" and observed is not None and not swapped:
                    (parent / "published").rename(genuine_moved)
                    attacker.rename(parent / "published")
                    swapped = True
                return observed

            try:
                with patch.object(
                    leash_contract_module, "_entry_stat", side_effect=stat_then_substitute
                ):
                    with self.assertRaisesRegex(LeashContractError, "published directory"):
                        stage.commit()
            finally:
                stage.cleanup()

            self.assertFalse((parent / "published").exists())
            payloads = sorted(path.read_bytes() for path in parent.rglob("PAYLOAD"))
            self.assertEqual(payloads, [b"FORGED", b"GENUINE"])
            self.assertTrue(list(parent.glob(".published.leash-recovery-*")))
            self.assertTrue(list(parent.glob(".published.leash-foreign-recovery-*")))

    def test_build_writer_parent_swap_never_touches_victim_stage_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            source, registry = _fixture_registry(root)
            public_parent = root / "release"
            public_output = public_parent / "preparation"
            private_output = root / "private/outcomes"
            victim = root / "victim"
            moved = root / "release-moved"
            victim.mkdir()
            original = leash_preparation_module._write_preparation_trees
            victim_payload: Path | None = None

            def swap_parent_before_tree_write(*, public_stage, private_stage, **kwargs):
                nonlocal victim_payload
                self.assertIsInstance(public_stage, AtomicLeashDirectory)
                self.assertIsInstance(private_stage, AtomicLeashDirectory)
                public_parent.rename(moved)
                public_parent.symlink_to(victim, target_is_directory=True)
                victim_stage = victim / public_stage._stage_name
                victim_stage.mkdir()
                victim_payload = victim_stage / FIT_INPUT_FILENAME
                victim_payload.write_text("KEEP", encoding="utf-8")
                original(
                    public_stage=public_stage,
                    private_stage=private_stage,
                    **kwargs,
                )

            with patch.object(
                leash_preparation_module,
                "_write_preparation_trees",
                side_effect=swap_parent_before_tree_write,
            ):
                with self.assertRaisesRegex(LeashContractError, "path binding changed"):
                    prepare_leash_build(
                        source_root=source,
                        registry_path=registry,
                        public_output=public_output,
                        private_output=private_output,
                    )

            self.assertIsNotNone(victim_payload)
            self.assertEqual(victim_payload.read_text(encoding="utf-8"), "KEEP")
            self.assertFalse((victim / "preparation").exists())
            self.assertFalse((moved / "preparation").exists())

    def test_certificate_publication_parent_swap_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            parent = root / "certificate-parent"
            victim = root / "victim"
            moved = root / "certificate-parent-moved"
            parent.mkdir()
            victim.mkdir()
            original = leash_contract_module._rename_entry_noreplace_at
            swapped = False

            def swap_parent_then_rename(
                source_parent_fd: int,
                source_name: str,
                target_parent_fd: int,
                target_name: str,
            ) -> None:
                nonlocal swapped
                if not swapped:
                    parent.rename(moved)
                    parent.symlink_to(victim, target_is_directory=True)
                    swapped = True
                original(
                    source_parent_fd, source_name, target_parent_fd, target_name
                )

            with patch.object(
                leash_contract_module,
                "_rename_entry_noreplace_at",
                side_effect=swap_parent_then_rename,
            ):
                with self.assertRaisesRegex(LeashContractError, "path binding changed"):
                    write_json_noreplace(parent / "CERT.json", {"genuine": True})

            self.assertFalse((victim / "CERT.json").exists())
            self.assertFalse((moved / "CERT.json").exists())
            recovery = list(moved.glob(".CERT.json.leash-recovery-*"))
            self.assertEqual(len(recovery), 1)
            self.assertIn("genuine", recovery[0].read_text(encoding="utf-8"))

    def test_certificate_post_stat_substitution_cannot_report_success(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            target = parent / "CERT.json"
            attacker = parent / "attacker.json"
            attacker.write_text("FORGED", encoding="utf-8")
            genuine_moved = parent / "genuine-moved-inside-stat.json"
            original = leash_contract_module._entry_stat
            swapped = False

            def stat_then_substitute(parent_fd: int, name: str):
                nonlocal swapped
                observed = original(parent_fd, name)
                if name == target.name and observed is not None and not swapped:
                    target.rename(genuine_moved)
                    attacker.rename(target)
                    swapped = True
                return observed

            with patch.object(
                leash_contract_module, "_entry_stat", side_effect=stat_then_substitute
            ):
                with self.assertRaisesRegex(LeashContractError, "certificate"):
                    write_json_noreplace(target, {"genuine": True})

            self.assertFalse(target.exists())
            contents = sorted(path.read_text(encoding="utf-8") for path in parent.iterdir())
            self.assertTrue(any("genuine" in value for value in contents))
            self.assertIn("FORGED", contents)
            self.assertTrue(list(parent.glob(".CERT.json.leash-recovery-*")))
            self.assertTrue(list(parent.glob(".CERT.json.leash-foreign-recovery-*")))

    def test_empty_quarantine_name_substitution_never_deletes_unrelated_directory(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            parent = Path(temporary) / "parent"
            parent.mkdir()
            genuine = parent / "genuine-empty"
            genuine.mkdir()
            parent_fd = os.open(parent, leash_contract_module._directory_open_flags())
            identity = leash_contract_module._identity(os.stat(genuine, follow_symlinks=False))
            original = leash_contract_module._quarantine_entry_by_identity
            genuine_recovery: Path | None = None
            unrelated_at_quarantine: Path | None = None

            def substitute_after_quarantine(parent_descriptor: int, **kwargs) -> str:
                nonlocal genuine_recovery, unrelated_at_quarantine
                quarantine = original(parent_descriptor, **kwargs)
                unrelated_at_quarantine = parent / quarantine
                genuine_recovery = parent / f"{quarantine}.genuine"
                unrelated_at_quarantine.rename(genuine_recovery)
                unrelated_at_quarantine.mkdir()
                return quarantine

            try:
                with patch.object(
                    leash_contract_module,
                    "_quarantine_entry_by_identity",
                    side_effect=substitute_after_quarantine,
                ):
                    leash_contract_module._remove_known_empty_directory_by_identity(
                        parent_fd,
                        identity=identity,
                        preferred_name=genuine.name,
                        name="test empty directory",
                    )
            finally:
                os.close(parent_fd)

            self.assertIsNotNone(genuine_recovery)
            self.assertIsNotNone(unrelated_at_quarantine)
            self.assertTrue(genuine_recovery.is_dir())
            self.assertTrue(unrelated_at_quarantine.is_dir())

    def test_directory_cleanup_name_substitution_preserves_unrelated_tree(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            parent = root / "parent"
            parent.mkdir()
            stage = AtomicLeashDirectory(parent / "published")
            stage.write_bytes("genuine.json", b"genuine")
            stage_entry = parent / stage._stage_name
            unrelated = parent / "unrelated"
            unrelated.mkdir()
            (unrelated / "KEEP.txt").write_text("KEEP", encoding="utf-8")
            genuine_moved = parent / "genuine-stage-moved"
            original = leash_contract_module._quarantine_entry_by_identity
            swapped = False

            def substitute_stage_name(parent_fd: int, **kwargs) -> str:
                nonlocal swapped
                if not swapped:
                    stage_entry.rename(genuine_moved)
                    unrelated.rename(stage_entry)
                    swapped = True
                return original(parent_fd, **kwargs)

            with patch.object(
                leash_contract_module,
                "_quarantine_entry_by_identity",
                side_effect=substitute_stage_name,
            ):
                stage.cleanup()

            self.assertFalse(genuine_moved.exists())
            self.assertEqual((stage_entry / "KEEP.txt").read_text(encoding="utf-8"), "KEEP")
            recovery = list(parent.glob(".published.leash-recovery-*"))
            self.assertEqual(len(recovery), 1)
            self.assertEqual((recovery[0] / "genuine.json").read_bytes(), b"genuine")

    def test_certificate_cleanup_name_substitution_preserves_unrelated_file(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            parent = root / "parent"
            parent.mkdir()
            unrelated = parent / "UNRELATED.txt"
            unrelated.write_text("KEEP", encoding="utf-8")
            original_verify = leash_contract_module._verify_directory_binding
            original_quarantine = leash_contract_module._quarantine_entry_by_identity
            verify_calls = 0
            substituted = False

            def fail_after_publication(*args, **kwargs) -> None:
                nonlocal verify_calls
                verify_calls += 1
                if verify_calls == 3:
                    raise LeashContractError("injected post-publication binding failure")
                original_verify(*args, **kwargs)

            def substitute_temporary_name(parent_fd: int, **kwargs) -> str:
                nonlocal substituted
                if not substituted:
                    identity = kwargs["identity"]
                    genuine_name = leash_contract_module._entry_names_by_identity(
                        parent_fd, identity
                    )[0]
                    genuine_moved = "genuine-certificate-moved"
                    leash_contract_module._rename_directory_noreplace_at(
                        parent_fd, genuine_name, genuine_moved
                    )
                    unrelated.rename(parent / kwargs["preferred_name"])
                    substituted = True
                return original_quarantine(parent_fd, **kwargs)

            with patch.object(
                leash_contract_module,
                "_verify_directory_binding",
                side_effect=fail_after_publication,
            ), patch.object(
                leash_contract_module,
                "_quarantine_entry_by_identity",
                side_effect=substitute_temporary_name,
            ):
                with self.assertRaisesRegex(
                    LeashContractError, "injected post-publication binding failure"
                ):
                    write_json_noreplace(parent / "CERT.json", {"genuine": True})

            self.assertFalse((parent / "CERT.json").exists())
            keep_files = [
                path for path in parent.iterdir()
                if path.is_file() and path.read_text(encoding="utf-8") == "KEEP"
            ]
            self.assertEqual(len(keep_files), 1)
            recovery = list(parent.glob(".CERT.json.leash-recovery-*"))
            self.assertEqual(len(recovery), 1)
            self.assertIn("genuine", recovery[0].read_text(encoding="utf-8"))

    def test_directory_acquisition_swap_never_clears_foreign_stage_fd(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            parent = root / "parent"
            parent.mkdir()
            unrelated = parent / "unrelated"
            unrelated.mkdir()
            (unrelated / "KEEP.txt").write_text("KEEP", encoding="utf-8")
            genuine_moved = parent / "genuine-stage-moved"
            original_open = leash_contract_module.os.open
            swapped = False

            def substitute_between_stat_and_open(
                path, flags, mode=0o777, *, dir_fd=None
            ):
                nonlocal swapped
                if (
                    not swapped
                    and isinstance(path, str)
                    and path.startswith(".published.leash-staging-")
                    and dir_fd is not None
                ):
                    stage_entry = parent / path
                    stage_entry.rename(genuine_moved)
                    unrelated.rename(stage_entry)
                    swapped = True
                return original_open(path, flags, mode, dir_fd=dir_fd)

            with patch.object(
                leash_contract_module.os,
                "open",
                side_effect=substitute_between_stat_and_open,
            ):
                with self.assertRaisesRegex(
                    LeashContractError, "staging directory identity changed"
                ):
                    AtomicLeashDirectory(parent / "published")

            self.assertFalse(genuine_moved.exists())
            keep_files = list(parent.rglob("KEEP.txt"))
            self.assertEqual(len(keep_files), 1)
            self.assertEqual(keep_files[0].read_text(encoding="utf-8"), "KEEP")

    def test_certificate_acquisition_swap_cannot_publish_foreign_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            parent = root / "parent"
            parent.mkdir()
            malicious = parent / "MALICIOUS.txt"
            malicious.write_text("FORGED", encoding="utf-8")
            genuine_moved = parent / "genuine-certificate-moved"
            original_verify = leash_contract_module._verify_directory_binding
            verify_calls = 0

            def substitute_after_first_binding_check(*args, **kwargs) -> None:
                nonlocal verify_calls
                original_verify(*args, **kwargs)
                verify_calls += 1
                if verify_calls == 1:
                    temporary_name = next(
                        path
                        for path in parent.iterdir()
                        if path.name.startswith(".CERT.json.leash-tmp-")
                    )
                    temporary_name.rename(genuine_moved)
                    malicious.rename(temporary_name)

            with patch.object(
                leash_contract_module,
                "_verify_directory_binding",
                side_effect=substitute_after_first_binding_check,
            ):
                with self.assertRaisesRegex(
                    LeashContractError, "name-to-fd identity binding changed"
                ):
                    write_json_noreplace(parent / "CERT.json", {"genuine": True})

            self.assertFalse((parent / "CERT.json").exists())
            forged_files = [
                path for path in parent.iterdir()
                if path.is_file() and path.read_text(encoding="utf-8") == "FORGED"
            ]
            self.assertEqual(len(forged_files), 1)
            recovery = list(parent.glob(".CERT.json.leash-recovery-*"))
            self.assertEqual(len(recovery), 1)
            self.assertIn("genuine", recovery[0].read_text(encoding="utf-8"))

    def test_label_firewall_is_recursive(self) -> None:
        with self.assertRaisesRegex(LeashContractError, "outcome leak"):
            assert_no_forbidden_keys({"safe": [{"gold_answer": "secret"}]})
        self.assertIn("answer_text", FIT_FORBIDDEN_FIELDS)

    def test_policy_freeze_rejects_missing_closure(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            source, registry_path = _fixture_registry(Path(temporary))
            registry = load_registry(registry_path)
            audit = audit_leash_sources(source_root=source, registry_path=registry_path)
            self.assertEqual(audit["n_ready_rows"], 54)
            public = Path(temporary) / "prep"
            private = Path(temporary) / "outcomes"
            prepare_leash_build(
                source_root=source, registry_path=registry_path,
                public_output=public, private_output=private,
            )
            rows = load_jsonl(public / FIT_INPUT_FILENAME, name="fit rows")
            bad = deepcopy(rows)
            target = next(row for row in bad if row["arm"] == "leash" and row["stopped_early"])
            target["closure_generated"] = False
            target["n_closure_tokens"] = 0
            target["n_total_tokens"] = target["n_reasoning_tokens"]
            with self.assertRaisesRegex(LeashContractError, "lacks realized closure"):
                derive_policy_ledger(bad)
            self.assertEqual(registry["evaluation"]["bootstrap"]["unit"], "source question")

    def test_source_tree_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            source, registry = _fixture_registry(Path(temporary))
            first = load_registry(registry)["source_contract"]["ready_runs"][0]
            (source / first["path"] / "SUMMARY.json").write_text("tamper", encoding="utf-8")
            with self.assertRaisesRegex(LeashContractError, "source tree binding failed"):
                audit_leash_sources(source_root=source, registry_path=registry)

    def test_source_shard_post_read_swap_is_rejected_before_derivation(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            source, registry = _fixture_registry(root)
            first = load_registry(registry)["source_contract"]["ready_runs"][0]
            shard = source / first["path"] / "shards/shard_00000.pkl"
            with shard.open("rb") as handle:
                forged_records = pickle.load(handle)
            forged_records[0]["answer_text"] = "FORGED AFTER ACCEPTED READ"
            forged = root / "forged-shard.pkl"
            with forged.open("wb") as handle:
                pickle.dump(forged_records, handle, protocol=pickle.HIGHEST_PROTOCOL)
            genuine_moved = root / "genuine-shard.pkl"
            original = leash_preparation_module.read_bound_bytes
            swapped = False

            def read_then_substitute(path, **kwargs):
                nonlocal swapped
                payload = original(path, **kwargs)
                if Path(path) == shard and not swapped:
                    shard.rename(genuine_moved)
                    forged.rename(shard)
                    swapped = True
                return payload

            with patch.object(
                leash_preparation_module,
                "read_bound_bytes",
                side_effect=read_then_substitute,
            ):
                with self.assertRaisesRegex(LeashContractError, "changed during derivation"):
                    audit_leash_sources(source_root=source, registry_path=registry)
            self.assertTrue(swapped)
            self.assertTrue(genuine_moved.exists())

    def test_source_guard_contract_uses_canonical_pipe_not_swappable_path(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            source, registry = _fixture_registry(root)
            genuine_bytes = leash_fit_module._controller_source_preparation_bytes(
                source_root=source, registry_path=registry
            )
            genuine = json.loads(genuine_bytes)
            forged = deepcopy(genuine)
            forged["fit_rows"][0]["wall_s"] = 999999.0
            forged_fit_bytes = canonical_jsonl_bytes(forged["fit_rows"])
            forged_fit_sha = hashlib.sha256(forged_fit_bytes).hexdigest()
            forged_manifest = dict(forged["preparation_manifest"])
            forged_manifest.pop("payload_sha256")
            forged_manifest["files"]["FIT_INPUT.jsonl"] = forged_fit_sha
            forged_manifest = add_payload_sha256(forged_manifest)
            forged["preparation_manifest"] = forged_manifest
            for item in forged["public_tree"]["files"]:
                if item["path"] == "FIT_INPUT.jsonl":
                    item.update(bytes=len(forged_fit_bytes), sha256=forged_fit_sha)
                elif item["path"] == "PREPARATION_MANIFEST.json":
                    manifest_bytes = canonical_json_bytes(forged_manifest) + b"\n"
                    item.update(
                        bytes=len(manifest_bytes),
                        sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                    )
            forged_public_sha = hashlib.sha256(
                canonical_json_bytes(forged["public_tree"]["files"])
            ).hexdigest()
            forged["public_tree"]["tree_sha256"] = forged_public_sha
            forged_certificate = dict(forged["certificate"])
            forged_certificate.pop("payload_sha256")
            forged_certificate["public_tree_sha256"] = {
                "A": forged_public_sha,
                "B": forged_public_sha,
            }
            forged_certificate["rederived_public_tree_sha256"] = forged_public_sha
            forged["certificate"] = add_payload_sha256(forged_certificate)
            forged_path = root / "forged-contract.json"
            forged_path.write_bytes(canonical_json_bytes(forged) + b"\n")
            commands: list[list[str]] = []

            def complete_then_swap(command, **kwargs):
                commands.append(list(command))
                # Reproduce the old completion-to-load attack if pathname IPC
                # is reintroduced.  The pipe implementation supplies no output
                # name, so this forged filesystem object stays unreachable.
                if "--output" in command:
                    output = Path(command[command.index("--output") + 1])
                    output.symlink_to(forged_path)
                return subprocess.CompletedProcess(
                    command, 0, stdout=genuine_bytes, stderr=b""
                )

            with patch.object(
                leash_fit_module.subprocess, "run", side_effect=complete_then_swap
            ):
                observed = leash_fit_module._isolated_source_preparation_contract(
                    source_root=source, registry_path=registry
                )
            self.assertEqual(observed, genuine)
            self.assertEqual(len(commands), 1)
            self.assertNotIn("--output", commands[0])
            self.assertIn("-I", commands[0])
            self.assertNotEqual(
                Path(commands[0][2]).resolve(),
                (REPO / "scripts/reconstruction_benchmark/leash_source_guard_worker.py").resolve(),
            )

            with patch.object(
                leash_fit_module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    commands[0],
                    0,
                    stdout=canonical_json_bytes(forged) + b"\n",
                    stderr=b"",
                ),
            ):
                with self.assertRaisesRegex(
                    LeashContractError, "independent controller rederivation"
                ):
                    leash_fit_module._isolated_source_preparation_contract(
                        source_root=source, registry_path=registry
                    )

            with patch.object(
                leash_fit_module.subprocess,
                "run",
                return_value=subprocess.CompletedProcess(
                    commands[0], 0, stdout=b" " + genuine_bytes, stderr=b""
                ),
            ):
                with self.assertRaisesRegex(
                    LeashContractError,
                    "non-canonical|independent controller rederivation",
                ):
                    leash_fit_module._isolated_source_preparation_contract(
                        source_root=source, registry_path=registry
                    )

    def test_missing_raw_callback_channels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            source, registry = _fixture_registry(
                Path(temporary), omit_leash_channels=True
            )
            with self.assertRaisesRegex(LeashContractError, "raw callback channels"):
                audit_leash_sources(source_root=source, registry_path=registry)

    def test_source_symlink_is_rejected_even_with_identical_bytes(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            source, registry = _fixture_registry(Path(temporary))
            first = load_registry(registry)["source_contract"]["ready_runs"][0]
            summary = source / first["path"] / "SUMMARY.json"
            identical = source / "identical-summary.json"
            identical.write_bytes(summary.read_bytes())
            summary.unlink()
            summary.symlink_to(identical)
            with self.assertRaisesRegex(LeashContractError, "contains a symlink"):
                audit_leash_sources(source_root=source, registry_path=registry)

    def test_coordinated_prep_and_certificate_forgery_fails_before_fit(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            source, registry = _fixture_registry(root)
            lane, private = root / "release/leash", root / "private"
            for build in ("A", "B"):
                prepare_leash_build(
                    source_root=source, registry_path=registry,
                    public_output=lane / build / "preparation",
                    private_output=private / build / "outcomes",
                )
            certificate_path = lane / "PREPARATION_AB_VERIFICATION.json"
            verify_leash_preparation_ab(
                source_root=source, registry_path=registry,
                public_a=lane / "A/preparation", private_a=private / "A/outcomes",
                public_b=lane / "B/preparation", private_b=private / "B/outcomes",
                certificate_path=certificate_path,
            )

            forged_tree = None
            for build in ("A", "B"):
                preparation = lane / build / "preparation"
                fit_path = preparation / "FIT_INPUT.jsonl"
                rows = load_jsonl(fit_path, name="fit input to forge")
                target = next(
                    row for row in rows if row["arm"] == "leash" and row["stopped_early"]
                )
                target.update(
                    {
                        "stopped_early": False,
                        "stop_reason": "length",
                        "policy_replay_fired": False,
                        "policy_replay_stop_index": None,
                    }
                )
                fit_path.write_bytes(canonical_jsonl_bytes(rows))
                manifest_path = preparation / "PREPARATION_MANIFEST.json"
                manifest = json.loads(manifest_path.read_text())
                manifest.pop("payload_sha256")
                manifest["files"]["FIT_INPUT.jsonl"] = sha256_file(fit_path)
                manifest_path.write_bytes(
                    canonical_json_bytes(add_payload_sha256(manifest)) + b"\n"
                )
                observed = canonical_tree_manifest(preparation)
                if forged_tree is not None:
                    self.assertEqual(observed, forged_tree)
                forged_tree = observed

            certificate = json.loads(certificate_path.read_text())
            certificate.pop("payload_sha256")
            certificate["public_tree_sha256"] = {
                "A": forged_tree["tree_sha256"], "B": forged_tree["tree_sha256"]
            }
            certificate["rederived_public_tree_sha256"] = forged_tree["tree_sha256"]
            certificate_path.write_bytes(
                canonical_json_bytes(add_payload_sha256(certificate)) + b"\n"
            )
            output = lane / "A/fit"
            with patch(
                "spectral_utils.reconstruction_benchmark.leash_fit.load_verified_fit_input",
                side_effect=AssertionError("fit input opened before current-source gate"),
            ):
                with self.assertRaisesRegex(
                    LeashContractError, "exact current-source rederivation"
                ):
                    run_leash_fit(
                        source_root=source,
                        preparation_dir=lane / "A/preparation",
                        preparation_ab_certificate=certificate_path,
                        registry_path=registry,
                        output_dir=output,
                    )
            self.assertFalse(output.exists())

    def test_registry_cross_binding_rejects_a_different_valid_registry(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            source, registry = _fixture_registry(root)
            lane, private = root / "release/leash", root / "private"
            for build in ("A", "B"):
                prepare_leash_build(
                    source_root=source, registry_path=registry,
                    public_output=lane / build / "preparation",
                    private_output=private / build / "outcomes",
                )
            certificate = lane / "PREPARATION_AB_VERIFICATION.json"
            verify_leash_preparation_ab(
                source_root=source, registry_path=registry,
                public_a=lane / "A/preparation", private_a=private / "A/outcomes",
                public_b=lane / "B/preparation", private_b=private / "B/outcomes",
                certificate_path=certificate,
            )
            alternate = json.loads(registry.read_text())
            alternate["evaluation"]["bootstrap"]["seed"] += 1
            alternate_path = root / "alternate_registry.json"
            alternate_path.write_text(json.dumps(alternate), encoding="utf-8")
            load_registry(alternate_path)
            with self.assertRaisesRegex(
                LeashContractError, "current-source rederivation|certificate differs"
            ):
                run_leash_fit(
                    source_root=source, preparation_dir=lane / "A/preparation",
                    preparation_ab_certificate=certificate, registry_path=alternate_path,
                    output_dir=lane / "A/fit",
                )

    def test_full_ab_pipeline_is_searchable_and_claim_scoped(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            state = _run_pipeline(Path(temporary))
            eval_a = state["lane"] / "A/evaluation"
            for table in (
                "coverage", "per_question", "cell_metrics", "contrasts", "frontier",
                "aggregate_metrics", "bootstrap_intervals",
            ):
                for suffix in ("jsonl", "csv", "parquet"):
                    self.assertTrue((eval_a / f"{table}.{suffix}").is_file())
            coverage = load_jsonl(eval_a / "coverage.jsonl", name="coverage")
            self.assertEqual(len(coverage), 8)
            blocked = [row for row in coverage if row["coverage_status"] == "PROTOCOL_GATE_FAILED"]
            self.assertEqual(len(blocked), 2)
            self.assertTrue(all(not row["usable_for_evaluation"] for row in blocked))
            questions = load_jsonl(eval_a / "per_question.jsonl", name="questions")
            self.assertEqual(len(questions), 54)
            self.assertTrue(all("answer_text" not in row for row in questions))
            aqua = next(row for row in questions if row["dataset"] == "aqua" and row["arm"] == "cot")
            self.assertEqual(aqua["parser_revision"], "fair_aqua_option_parser_v1.0.0")
            intervals = load_jsonl(eval_a / "bootstrap_intervals.jsonl", name="intervals")
            self.assertTrue(any(row["scope"] == "equal_dataset_after_equal_model" for row in intervals))
            self.assertTrue(all(row["n_boot"] == 20 for row in intervals))
            manifest = json.loads((eval_a / "EVALUATION_MANIFEST.json").read_text())
            self.assertTrue(manifest["policy_execution_evaluated"])
            self.assertFalse(manifest["paper_exact_claim"])
            self.assertFalse(manifest["conceptual_objective_reproduced_as_equation"])
            self.assertFalse(manifest["matched_accuracy_claim"])
            schema = json.loads((eval_a / "TABLE_SCHEMA.json").read_text())
            dtypes = schema["tables"]["per_question"]["dtypes"]
            self.assertEqual(dtypes["correct"], "bool")
            self.assertEqual(dtypes["n_total_tokens"], "int64")
            self.assertEqual(dtypes["wall_s"], "float64")
            import pyarrow.parquet as pq
            parquet_schema = pq.read_schema(eval_a / "per_question.parquet")
            self.assertEqual(str(parquet_schema.field("correct").type), "bool")
            self.assertEqual(str(parquet_schema.field("n_total_tokens").type), "int64")
            self.assertEqual(str(parquet_schema.field("wall_s").type), "double")
            eval_cert = json.loads(state["eval_cert"].read_text())
            self.assertTrue(eval_cert["transitive_rederivation"])
            self.assertTrue(eval_cert["searchable_output_contract_verified"])

    def test_evaluation_ab_rejects_symlinked_evaluation_root(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            state = _run_pipeline(root)
            evaluation_a = state["lane"] / "A/evaluation"
            evaluation_b = state["lane"] / "B/evaluation"
            genuine_b = state["lane"] / "B/evaluation-genuine"
            evaluation_b.rename(genuine_b)
            evaluation_b.symlink_to(evaluation_a, target_is_directory=True)
            with self.assertRaisesRegex(LeashContractError, "symlink|cannot open"):
                verify_leash_evaluation_ab(
                    source_root=state["source"],
                    preparation_a=state["lane"] / "A/preparation",
                    fit_a=state["lane"] / "A/fit",
                    private_a=state["private"] / "A/outcomes",
                    evaluation_a=evaluation_a,
                    preparation_b=state["lane"] / "B/preparation",
                    fit_b=state["lane"] / "B/fit",
                    private_b=state["private"] / "B/outcomes",
                    evaluation_b=evaluation_b,
                    preparation_ab_certificate=state["prep_cert"],
                    fit_ab_certificate=state["fit_cert"],
                    registry_path=state["registry"],
                    certificate_path=state["lane"] / "symlink-retry.json",
                )
            self.assertFalse((state["lane"] / "symlink-retry.json").exists())

    def test_evaluation_ab_rejects_23_file_hardlink_alias(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            state = _run_pipeline(root)
            evaluation_a = state["lane"] / "A/evaluation"
            alias_b = state["lane"] / "B/evaluation-hardlink-alias"
            alias_b.mkdir()
            source_files = sorted(
                path for path in evaluation_a.rglob("*") if path.is_file()
            )
            self.assertEqual(len(source_files), 23)
            for source_file in source_files:
                relative = source_file.relative_to(evaluation_a)
                target = alias_b / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(source_file, target)
            certificate = state["lane"] / "hardlink-retry.json"
            with self.assertRaisesRegex(
                LeashContractError, "hardlinked|physically disjoint"
            ):
                verify_leash_evaluation_ab(
                    source_root=state["source"],
                    preparation_a=state["lane"] / "A/preparation",
                    fit_a=state["lane"] / "A/fit",
                    private_a=state["private"] / "A/outcomes",
                    evaluation_a=evaluation_a,
                    preparation_b=state["lane"] / "B/preparation",
                    fit_b=state["lane"] / "B/fit",
                    private_b=state["private"] / "B/outcomes",
                    evaluation_b=alias_b,
                    preparation_ab_certificate=state["prep_cert"],
                    fit_ab_certificate=state["fit_cert"],
                    registry_path=state["registry"],
                    certificate_path=certificate,
                )
            self.assertFalse(certificate.exists())

    def test_private_outcomes_post_read_swap_is_rejected_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            root = Path(temporary)
            state = _run_pipeline(root)
            private_dir = state["private"] / "A/outcomes"
            outcomes_path = private_dir / "OUTCOMES.jsonl"
            forged_rows = load_jsonl(outcomes_path, name="private outcomes fixture")
            for row in forged_rows:
                row["answer_text"] = "definitely incorrect forged response"
            forged = root / "forged-outcomes.jsonl"
            forged.write_bytes(canonical_jsonl_bytes(forged_rows))
            genuine_moved = root / "genuine-outcomes.jsonl"
            original = leash_evaluation_module.read_bound_bytes
            swapped = False

            def read_then_substitute(path, **kwargs):
                nonlocal swapped
                payload = original(path, **kwargs)
                if Path(path) == outcomes_path and not swapped:
                    outcomes_path.rename(genuine_moved)
                    forged.rename(outcomes_path)
                    swapped = True
                return payload

            with patch.object(
                leash_evaluation_module,
                "read_bound_bytes",
                side_effect=read_then_substitute,
            ):
                with self.assertRaisesRegex(
                    LeashContractError, "changed during bound label loading"
                ):
                    leash_evaluation_module.derive_leash_evaluation(
                        source_root=state["source"],
                        registry_path=state["registry"],
                        preparation_dir=state["lane"] / "A/preparation",
                        fit_dir=state["lane"] / "A/fit",
                        private_dir=private_dir,
                        preparation_ab_certificate=state["prep_cert"],
                        fit_ab_certificate=state["fit_cert"],
                    )
            self.assertTrue(swapped)
            self.assertTrue(genuine_moved.exists())

    def test_coordinated_fit_and_certificate_forgery_fails_before_labels(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            state = _run_pipeline(Path(temporary))
            forged_tree = None
            for build in ("A", "B"):
                fit = state["lane"] / build / "fit"
                ledger_path = fit / "POLICY_EXECUTION.jsonl"
                rows = load_jsonl(ledger_path, name="fit ledger to forge")
                rows[0]["wall_s"] = float(rows[0]["wall_s"]) + 0.125
                ledger_path.write_bytes(canonical_jsonl_bytes(rows))
                manifest_path = fit / "FIT_MANIFEST.json"
                manifest = json.loads(manifest_path.read_text())
                manifest.pop("payload_sha256")
                manifest["files"]["POLICY_EXECUTION.jsonl"] = sha256_file(ledger_path)
                manifest_path.write_bytes(
                    canonical_json_bytes(add_payload_sha256(manifest)) + b"\n"
                )
                observed = canonical_tree_manifest(fit)
                if forged_tree is not None:
                    self.assertEqual(observed, forged_tree)
                forged_tree = observed
            fit_certificate = json.loads(state["fit_cert"].read_text())
            fit_certificate.pop("payload_sha256")
            fit_certificate["fit_tree_sha256"] = {
                "A": forged_tree["tree_sha256"], "B": forged_tree["tree_sha256"]
            }
            fit_certificate["rederived_fit_tree_sha256"] = forged_tree["tree_sha256"]
            state["fit_cert"].write_bytes(
                canonical_json_bytes(add_payload_sha256(fit_certificate)) + b"\n"
            )
            with patch(
                "spectral_utils.reconstruction_benchmark.leash_evaluation._load_private_outcomes",
                side_effect=AssertionError("private labels opened before current-source gate"),
            ):
                with self.assertRaisesRegex(
                    LeashContractError, "exact current-source rederivation"
                ):
                    derive_leash_evaluation(
                        source_root=state["source"], registry_path=state["registry"],
                        preparation_dir=state["lane"] / "A/preparation",
                        fit_dir=state["lane"] / "A/fit",
                        private_dir=state["private"] / "A/outcomes",
                        preparation_ab_certificate=state["prep_cert"],
                        fit_ab_certificate=state["fit_cert"],
                    )

    def test_mutation_breaks_ab_and_private_binding(self) -> None:
        with tempfile.TemporaryDirectory(dir=TEST_TEMP_ROOT) as temporary:
            state = _run_pipeline(Path(temporary))
            target = state["lane"] / "B/evaluation/coverage.csv"
            target.write_bytes(target.read_bytes() + b"tamper\n")
            with self.assertRaisesRegex(LeashContractError, "differs from rederivation"):
                verify_leash_evaluation_ab(
                    source_root=state["source"],
                    preparation_a=state["lane"] / "A/preparation",
                    fit_a=state["lane"] / "A/fit", private_a=state["private"] / "A/outcomes",
                    evaluation_a=state["lane"] / "A/evaluation", fit_b=state["lane"] / "B/fit",
                    preparation_b=state["lane"] / "B/preparation",
                    private_b=state["private"] / "B/outcomes", evaluation_b=state["lane"] / "B/evaluation",
                    preparation_ab_certificate=state["prep_cert"], fit_ab_certificate=state["fit_cert"],
                    registry_path=state["registry"], certificate_path=state["lane"] / "retry.json",
                )
            outcome = state["private"] / "A/outcomes/OUTCOMES.jsonl"
            outcome.write_bytes(outcome.read_bytes() + b"{}\n")
            with self.assertRaisesRegex(LeashContractError, "private outcome tree"):
                derive_leash_evaluation(
                    source_root=state["source"], registry_path=state["registry"],
                    preparation_dir=state["lane"] / "A/preparation",
                    fit_dir=state["lane"] / "A/fit", private_dir=state["private"] / "A/outcomes",
                    preparation_ab_certificate=state["prep_cert"], fit_ab_certificate=state["fit_cert"],
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
