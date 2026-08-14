"""Boundary/provenance tests for the append-only A6-S0a runner."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import automatic_group_free_phase_a6_s0a as runner
from scripts.test_a6_s0a import FakeContextTokenizer
from scripts.test_a6_s0a import TOKENIZERS
from spectral_utils import a6_s0a as s0a_core


class A6S0aRunnerTests(unittest.TestCase):
    def test_tokenizer_audit_is_json_native_and_replays_after_json_roundtrip(self) -> None:
        tokenizer = FakeContextTokenizer("qwen")
        tokenizer.chat_template = "<user>{{ content }}</user><assistant>"
        identity = runner._identity_by_scorer()["qwen3-4b"]
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "config.json").write_text(
                '{"eos_token_id":0}\n', encoding="utf-8",
            )
            (snapshot / "generation_config.json").write_text(
                '{"eos_token_id":[0,1]}\n', encoding="utf-8",
            )
            audit = runner._tokenizer_boundary_audit(tokenizer, identity, snapshot)
        replayed = json.loads(runner.canonical_json_bytes(audit))
        self.assertEqual(audit, replayed)
        self.assertEqual(audit["effective_generation_eos_token_ids"], [0, 1])
        self.assertEqual(audit["effective_generation_pad_token_id"], 0)
        prompt = audit["audit_prompt_evidence"]
        self.assertIsInstance(prompt["input_ids"], list)
        self.assertIsInstance(prompt["generation_parameters"], list)

    def test_tokenizer_audit_roster_rejects_extra_sibling_object(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            (snapshot / "config.json").write_text(
                '{"eos_token_id":0}\n', encoding="utf-8",
            )
            audits = {}
            for scorer_id, identity in runner._identity_by_scorer().items():
                kind = "llama" if scorer_id == "llama31-8b" else "qwen"
                tokenizer = FakeContextTokenizer(kind)
                tokenizer.chat_template = "<user>{{ content }}</user><assistant>"
                audits[scorer_id] = runner._tokenizer_boundary_audit(
                    tokenizer, identity, snapshot,
                )
            runner._validate_tokenizer_audits(audits)
            audits["llama_responses"] = {"label": 1}
            with self.assertRaisesRegex(RuntimeError, "roster"):
                runner._validate_tokenizer_audits(audits)

    def test_source_manifest_binds_runner_core_and_tests(self) -> None:
        names = runner.source_files()
        self.assertIn("scripts/automatic_group_free_phase_a6_s0a.py", names)
        self.assertIn("scripts/test_automatic_group_free_phase_a6_s0a.py", names)
        self.assertIn("scripts/test_a6_s0a.py", names)
        self.assertIn("scripts/test_a6_tokenizer_restore.py", names)
        self.assertIn("scripts/automatic_group_free_phase_a6_tokenizer_restore.py", names)
        self.assertIn("spectral_utils/a6_tokenizer_restore.py", names)
        self.assertIn("spectral_utils/a6_s0a.py", names)
        self.assertIn("spectral_utils/a6_interventions.py", names)
        self.assertEqual(len(names), len(set(names)))

    def test_prepare_missing_access_closes_before_project_import_or_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "boundary"
            with patch.object(runner, "_core", side_effect=AssertionError("imported")):
                with self.assertRaisesRegex(RuntimeError, "BLOCKED_TOKENIZER_ACCESS"):
                    runner.prepare(
                        out,
                        tokenizer_restore_root=Path(temporary) / "missing-restore",
                    )
            self.assertTrue(out.is_dir())
            self.assertEqual(list(out.iterdir()), [])

    def test_stdlib_snapshot_follows_source_link_once_and_freezes_regular_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scorer_id = "qwen3-4b"
            revision = runner.IDENTITIES[scorer_id]["revision"]
            source = root / revision
            blobs = root / "blobs"
            source.mkdir()
            blobs.mkdir()
            blob = blobs / "tok"
            blob.write_text('{"v":1}', encoding="utf-8")
            (source / "tokenizer.json").symlink_to(blob)
            (source / "ignored.bin").write_bytes(b"weights")
            destination, manifest = runner._prepare_snapshot_stdlib(
                source, root / "boundary", scorer_id,
            )
            self.assertFalse((destination / "tokenizer.json").is_symlink())
            self.assertEqual((destination / "tokenizer.json").read_bytes(), blob.read_bytes())
            self.assertIn("ignored.bin", manifest["repository_tree"])
            self.assertEqual([row["path"] for row in manifest["files"]], ["tokenizer.json"])
            runner._verify_snapshot_stdlib(destination, manifest)
            (destination / "extra.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "path set"):
                runner._verify_snapshot_stdlib(destination, manifest)

    def test_arbitrary_wrong_repo_directory_named_revision_is_not_trusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            identity = runner.IDENTITIES["qwen3-4b"]
            spoof = (
                Path(temporary) / "models--Wrong--Repo" / "snapshots"
                / identity["revision"]
            )
            spoof.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "frozen repo snapshot"):
                runner._require_resolved_revision_directory(spoof, "qwen3-4b")

    def test_checkpoint_is_exclusive_idempotent_and_tamper_evident(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoints" / "unit.json"
            runner._checkpoint(path, {"unit": 1}, "b" * 64)
            original = path.read_bytes()
            runner._checkpoint(path, {"unit": 1}, "b" * 64)
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaisesRegex(RuntimeError, "checkpoint mismatch"):
                runner._checkpoint(path, {"unit": 2}, "b" * 64)

    def test_checkpoint_prefix_rejects_noncanonical_json_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            path = out / "checkpoints" / "quartet" / "0000.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps({"boundary_sha256": "a" * 64, "payload": {"x": 1}}, indent=2),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "canonical JSON"):
                runner._load_checkpoint_prefix(
                    out, "quartet", 1, "a" * 64, lambda value: value,
                )

    def test_checkpoint_paths_reject_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / "out"
            outside = root / "outside"
            out.mkdir()
            outside.mkdir()
            (out / "checkpoints").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                runner._assert_known_output_paths(out, completed=False)
            path = out / "checkpoints" / "quartet" / "0000.json"
            with self.assertRaisesRegex(RuntimeError, "real directory"):
                runner._checkpoint(path, {"unit": 1}, "b" * 64)
            self.assertEqual(list(outside.iterdir()), [])

    def test_interrupted_checkpoint_temporary_is_recovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "checkpoints" / "quartet" / "0000.json"
            path.parent.mkdir(parents=True)
            temporary_path = path.with_name(path.name + ".tmp")
            temporary_path.write_bytes(b"partial")
            runner._assert_known_output_paths(root, completed=False)
            runner._checkpoint(path, {"unit": 1}, "b" * 64)
            self.assertTrue(path.is_file())
            self.assertFalse(temporary_path.exists())

    def test_registered_construction_exhaustion_writes_replayable_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            boundary = out / "A6_S0A_BOUNDARY.json"
            boundary.write_text("{}\n", encoding="utf-8")
            (out / "inputs").mkdir()
            core = SimpleNamespace(
                build_full_s0a_population=lambda *args, **kwargs: (_ for _ in ()).throw(
                    RuntimeError("CLOSE_INVALID_INTERVENTION_BOUNDARY:slot-7")
                ),
            )
            with patch.object(
                runner, "load_and_verify_boundary",
                return_value=({"tokenizer_snapshots": {}}, {}),
            ), patch.object(runner, "_core", return_value=core):
                first = runner.run(out, resume=False)
                second = runner.run(out, resume=False)
            self.assertEqual(first, second)
            self.assertEqual(first["verdict"], "CLOSE_INVALID_INTERVENTION_BOUNDARY")
            self.assertEqual(first["checkpoint_manifest"], [])
            self.assertTrue((out / "S0A_CLOSED.json").is_file())

    def test_runner_resume_deserializes_prefix_and_skips_completed_unit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            boundary_path = out / "A6_S0A_BOUNDARY.json"
            boundary_path.write_text("{}\n", encoding="utf-8")
            (out / "inputs").mkdir()
            boundary_sha = runner.sha256_file(boundary_path)
            first = s0a_core.build_s0a_quartets(
                TOKENIZERS, slots=s0a_core.frozen_quartet_slots()[:1],
            )[0]
            runner._checkpoint(
                out / "checkpoints" / "quartet" / "0000.json",
                s0a_core.public_quartet_record(first), boundary_sha,
            )
            observed = {}

            def stop_after_resume(*args, **kwargs):
                observed["quartets"] = kwargs["existing_quartets"]
                observed["natural"] = kwargs["existing_natural"]
                raise RuntimeError("CLOSE_INVALID_INTERVENTION_BOUNDARY:resume-proof")

            core = SimpleNamespace(
                quartet_record_from_public=s0a_core.quartet_record_from_public,
                natural_record_from_public=s0a_core.natural_record_from_public,
                public_quartet_record=s0a_core.public_quartet_record,
                public_natural_prompt_record=s0a_core.public_natural_prompt_record,
                build_full_s0a_population=stop_after_resume,
            )
            with patch.object(
                runner, "load_and_verify_boundary",
                return_value=({"tokenizer_snapshots": {}}, {}),
            ), patch.object(runner, "_core", return_value=core):
                result = runner.run(out, resume=True)
            self.assertEqual(result["verdict"], "CLOSE_INVALID_INTERVENTION_BOUNDARY")
            self.assertEqual(len(observed["quartets"]), 1)
            self.assertEqual(observed["quartets"][0].slot, first.slot)
            self.assertEqual(observed["natural"], ())

    def test_exclusive_json_refuses_history_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            runner._exclusive_json(path, {"a": 1})
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                runner._exclusive_json(path, {"a": 1})

    def test_output_allowlist_rejects_unmanifested_top_level_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "A6_S0A_BOUNDARY.json").write_text("{}\n", encoding="utf-8")
            (out / "BOUNDARY_REPORT.md").write_text("report\n", encoding="utf-8")
            runner._assert_known_output_paths(out, completed=False)
            (out / "unregistered.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unmanifested"):
                runner._assert_known_output_paths(out, completed=False)

    def test_output_allowlist_rejects_nested_inputs_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "A6_S0A_BOUNDARY.json").write_text("{}\n", encoding="utf-8")
            inputs = out / "inputs"
            expected = {"q4", "q8", "llama"}
            for name in expected:
                (inputs / name).mkdir(parents=True)
            boundary = {
                "tokenizer_snapshots": {
                    key: {"relative_directory": f"inputs/{name}"}
                    for key, name in zip(runner.IDENTITIES, sorted(expected))
                }
            }
            runner._assert_known_output_paths(
                out, completed=False, boundary=boundary,
            )
            (inputs / "llama_responses").mkdir()
            (inputs / "llama_responses" / "payload.json").write_text(
                '{"label":1}\n', encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "inputs"):
                runner._assert_known_output_paths(
                    out, completed=False, boundary=boundary,
                )

    def test_boundary_loader_rejects_altered_report_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            boundary = {
                "version": runner.VERSION, "status": runner.STATUS,
                "execution_contract_sha256": "0" * 64,
                "parent_protocol_sha256": "0" * 64,
                "source_sha256": {}, "runtime_versions": {}, "git_head": "x",
                "tokenizer_snapshots": {}, "tokenizer_audits": {},
                "tokenizer_restore_provenance": None,
                "configuration": {},
            }
            (out / "A6_S0A_BOUNDARY.json").write_bytes(
                runner.canonical_json_bytes(boundary)
            )
            (out / "BOUNDARY_REPORT.md").write_text(
                "llama response correctness payload\n", encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "report"):
                runner.load_and_verify_boundary(out)

    def test_forbidden_llama_namespace_closes_before_population_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "A6_S0A_BOUNDARY.json").write_text("{}\n", encoding="utf-8")
            (out / "llama_responses").mkdir()
            with patch.object(
                runner, "load_and_verify_boundary", return_value=({}, {}),
            ), patch.object(runner, "_core", side_effect=AssertionError("built")):
                with self.assertRaisesRegex(RuntimeError, "forbidden"):
                    runner.run(out)

    def test_hash_only_verifier_cannot_report_or_authorize_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            boundary = out / "A6_S0A_BOUNDARY.json"
            aggregate_path = out / "S0A_AGGREGATE.json"
            complete_path = out / "S0A_COMPLETE.json"
            boundary.write_text("{}\n", encoding="utf-8")
            boundary_sha = runner.sha256_file(boundary)
            aggregate = {
                "boundary_sha256": boundary_sha,
                "checkpoint_manifest": [], "result_file_sha256": {},
            }
            aggregate_path.write_bytes(runner.canonical_json_bytes(aggregate))
            completion = {
                "version": runner.VERSION, "verdict": "PASS_S0A",
                "boundary_sha256": boundary_sha,
                "aggregate_sha256": runner.sha256_file(aggregate_path),
            }
            complete_path.write_bytes(runner.canonical_json_bytes(completion))
            with patch.object(
                runner, "load_and_verify_boundary", return_value=({}, {}),
            ), patch.object(runner, "_assert_known_output_paths"), patch.object(
                runner, "_checkpoint_manifest", return_value=[],
            ):
                result = runner.verify(out, replay=False)
            self.assertEqual(result["status"], "HASH_ONLY_DIAGNOSTIC_NOT_AUTHORIZING_PASS")
            self.assertFalse(result["authorizes_next_stage"])


if __name__ == "__main__":
    unittest.main()
