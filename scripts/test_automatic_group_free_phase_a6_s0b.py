from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import automatic_group_free_phase_a6_s0b as runner


ROOT = Path(__file__).resolve().parents[1]
S0A = ROOT / "results" / "automatic_group_free_phase_a6_s0a_v1"


@dataclass(frozen=True)
class _File:
    path: str


class _FakeInputSpec:
    SCHEMA_VERSION = "fake-input-v1"
    REPOSITORY = "fake/repo"
    REVISION = "a" * 40
    OFFICIAL_API_URL = "https://example.invalid/exact"
    SELECTED_FILES = (_File("config.json"), _File("model.safetensors"))
    payloads = {"config.json": b"{}", "model.safetensors": b"weights"}

    @staticmethod
    def validate_official_tree(raw: bytes):
        if raw != b'{"official":true}\n':
            raise RuntimeError("official tree changed")
        return {"official": True}

    @classmethod
    def verify_selected_bytes(cls, item, payload: bytes):
        if payload != cls.payloads[item.path]:
            raise RuntimeError("fake payload changed")
        return {
            "path": item.path, "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }


class TestAutomaticA6S0b(unittest.TestCase):
    def _patches(self):
        provenance = {
            "relative_path": "results/automatic_group_free_phase_a6_s0a_v1",
            "boundary_sha256": "1" * 64,
            "aggregate_sha256": "2" * 64,
            "completion_sha256": "3" * 64,
        }
        return (
            mock.patch.object(runner, "_load_input_spec_stdlib", return_value=_FakeInputSpec),
            mock.patch.object(runner, "_prior_s0a_provenance", return_value=provenance),
            mock.patch.object(runner, "_load_pythia", return_value=(object(), object())),
            mock.patch.object(runner, "_pythia_runtime_audit", return_value={"fake": True}),
        )

    def test_prepare_and_verify_fake_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for name, payload in _FakeInputSpec.payloads.items():
                (source / name).write_bytes(payload)
            official = root / "official.json"
            official.write_bytes(b'{"official":true}\n')
            out = root / "out"
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3]:
                boundary = runner.prepare(
                    out, pythia_source=source, pythia_official_tree=official,
                )
                verified = runner.load_and_verify_boundary(
                    out, load_model=False, verify_prior=False,
                )[0]
            self.assertEqual(boundary, verified)
            self.assertEqual(boundary["status"], runner.STATUS)
            self.assertEqual(
                set((out / boundary["pythia_input"]["relative_directory"]).iterdir()),
                {out / boundary["pythia_input"]["relative_directory"] / name
                 for name in _FakeInputSpec.payloads},
            )

    def test_materialized_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for name, payload in _FakeInputSpec.payloads.items():
                (source / name).write_bytes(payload)
            official = root / "official.json"
            official.write_bytes(b'{"official":true}\n')
            out = root / "out"
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3]:
                boundary = runner.prepare(
                    out, pythia_source=source, pythia_official_tree=official,
                )
            snapshot = out / boundary["pythia_input"]["relative_directory"]
            (snapshot / "config.json").write_bytes(b"tamper")
            with mock.patch.object(
                runner, "_load_input_spec_stdlib", return_value=_FakeInputSpec,
            ), mock.patch.object(
                runner, "_prior_s0a_provenance",
                return_value=boundary["prior_s0a"],
            ):
                with self.assertRaisesRegex(RuntimeError, "fake payload changed"):
                    runner.load_and_verify_boundary(
                        out, load_model=False, verify_prior=False,
                    )

    def test_unmanifested_output_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for name, payload in _FakeInputSpec.payloads.items():
                (source / name).write_bytes(payload)
            official = root / "official.json"
            official.write_bytes(b'{"official":true}\n')
            out = root / "out"
            patches = self._patches()
            with patches[0], patches[1], patches[2], patches[3]:
                boundary = runner.prepare(
                    out, pythia_source=source, pythia_official_tree=official,
                )
            (out / "llama_responses.json").write_text("forbidden", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "unmanifested file"):
                runner._assert_known_output_paths(out, boundary)

    def test_real_s0a_prompt_schedule_is_complete(self) -> None:
        if not S0A.exists():
            self.skipTest("canonical S0a artifact unavailable")
        payloads = runner._load_quartet_payloads(S0A)
        schedule = runner._prompt_schedule(payloads)
        self.assertEqual(len(schedule), 14_400)
        self.assertEqual(tuple(sorted(schedule)), schedule)
        self.assertEqual(len({prompt for _, prompt in schedule}), 14_400)


if __name__ == "__main__":
    unittest.main()
