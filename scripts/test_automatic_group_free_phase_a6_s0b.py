from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import automatic_group_free_phase_a6_s0b as runner
from spectral_utils import a6_s0b as core


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


# ---------------------------------------------------------------------------
# Adversarial tests for the verifier/resume/replay runner paths added at the
# end of the Codex S0b session (HANDOFF_A6_S0B_TO_CLAUDE_2026_08_15.md §7.3):
#   1. replay mismatch never repairs or creates a file;
#   2. authoritative verify cannot be downgraded to hash-only PASS;
#   3. bootstrap resume validates all stored draws/summary, skips recompute;
#   4. control resume validates partitions, bijection, strata/edges, seeds,
#      hashes;
#   5. terminal artifact exclusivity and exact stage-specific layout;
#   6. unexpected exceptions remain implementation-invalid, never closure.
# ---------------------------------------------------------------------------


def _synthetic_row(
    group: int, fold: int, target: int, value: float, *,
    population: str = "qwen-source", scorer: str = "qwen3-4b",
) -> core.ShortcutRow:
    return core.ShortcutRow(
        row_id=f"{population}:g{group}:{target}", population_id=population,
        group_id=f"g{group}", outer_fold=fold, scorer_id=scorer,
        rendering_family="canonical", prompt_world="A" if target == 0 else "B",
        response_world="A", prompt_sha256=f"{population}:p{group}:{target}",
        response_sha256=f"{population}:rA", target=target,
        continuous=tuple(
            value + index / 100 for index in range(len(core.CONTINUOUS_COLUMNS))
        ),
        categorical=(
            "arithmetic", "value_leaf", "short", "canonical", "record_value",
            f"bank{group % 5}", f"template{group}", f"donor{group}",
        ),
    )


def _synthetic_population(population: str, scorer: str) -> tuple[core.ShortcutRow, ...]:
    # 20 groups per fold: a 20,000-draw grouped bootstrap over one stratum
    # must never produce an empty resampled class (P ~= 0.8**100 per draw).
    rows = []
    for fold in range(5):
        for offset in range(20):
            group = 20 * fold + offset
            rows.append(_synthetic_row(
                group, fold, 0, float(group), population=population, scorer=scorer,
            ))
            rows.append(_synthetic_row(
                group, fold, 1, float(group) + 0.25, population=population,
                scorer=scorer,
            ))
    return tuple(rows)


def _synthetic_bundles(rows: tuple[core.ShortcutRow, ...]) -> tuple[core.OofBundle, ...]:
    scores = tuple(float(row.target) for row in rows)
    return tuple(core.OofBundle(
        population_id=rows[0].population_id, ridge=ridge, scores=scores,
        fold_auc=(1.0,) * 5, fits=(),
    ) for ridge in core.RIDGES)


def _control_fixture():
    group_ids = tuple(f"g{index}" for index in range(8))
    records = tuple(core.GroupMatchingRecord(
        group_id=group_id, outer_fold=0,
        null_stratum_id="s0" if index < 4 else "s1",
        source_record_id=f"source{index}", donor_id=f"donor{index}",
        template_bank_id=f"bank{index}",
    ) for index, group_id in enumerate(group_ids))
    partitions = (("outer:0:held", group_ids), ("outer:0:train", group_ids))
    edges = tuple(
        (left, right) for left in group_ids for right in group_ids
        if left != right and ((int(left[1:]) < 4) == (int(right[1:]) < 4))
    )
    return group_ids, records, partitions, edges


def _rewrite_canonical(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(runner.canonical_json_bytes(value))


def _fake_prepare(root: Path):
    source = root / "source"
    source.mkdir()
    for name, payload in _FakeInputSpec.payloads.items():
        (source / name).write_bytes(payload)
    official = root / "official.json"
    official.write_bytes(b'{"official":true}\n')
    out = root / "out"
    provenance = {
        "relative_path": "results/automatic_group_free_phase_a6_s0a_v1",
        "boundary_sha256": "1" * 64, "aggregate_sha256": "2" * 64,
        "completion_sha256": "3" * 64,
    }
    with mock.patch.object(
        runner, "_load_input_spec_stdlib", return_value=_FakeInputSpec,
    ), mock.patch.object(
        runner, "_prior_s0a_provenance", return_value=provenance,
    ), mock.patch.object(
        runner, "_load_pythia", return_value=(object(), object()),
    ), mock.patch.object(
        runner, "_pythia_runtime_audit", return_value={"fake": True},
    ):
        runner.prepare(out, pythia_source=source, pythia_official_tree=official)
    return out, provenance


def _boundary_patches(provenance):
    return (
        mock.patch.object(runner, "_load_input_spec_stdlib", return_value=_FakeInputSpec),
        mock.patch.object(runner, "_prior_s0a_provenance", return_value=provenance),
    )


class TestEmitJsonReplay(unittest.TestCase):
    """Handoff item 1: replay mismatch never repairs or creates a file."""

    def test_replay_missing_artifact_is_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            path = out / "SHORTCUT_OOF.json"
            with self.assertRaisesRegex(RuntimeError, "required replay artifact"):
                runner._emit_json(path, {"a": 1}, root=out, replay=True)
            self.assertEqual(list(out.iterdir()), [])

    def test_replay_mismatch_never_repairs_stored_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            path = out / "SHORTCUT_OOF.json"
            stored = runner.canonical_json_bytes({"a": 1})
            path.write_bytes(stored)
            with self.assertRaisesRegex(RuntimeError, "replay mismatch"):
                runner._emit_json(path, {"a": 2}, root=out, replay=True)
            self.assertEqual(path.read_bytes(), stored)
            self.assertEqual({item.name for item in out.iterdir()}, {"SHORTCUT_OOF.json"})

    def test_replay_match_is_pure_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            path = out / "SHORTCUT_OOF.json"
            stored = runner.canonical_json_bytes({"a": 1})
            path.write_bytes(stored)
            runner._emit_json(path, {"a": 1}, root=out, replay=True)
            self.assertEqual(path.read_bytes(), stored)
            self.assertEqual({item.name for item in out.iterdir()}, {"SHORTCUT_OOF.json"})


class TestVerifySemanticReplay(unittest.TestCase):
    """Handoff item 2: verify cannot be downgraded to hash-only PASS."""

    def test_semantic_mismatch_raises_and_writes_nothing(self) -> None:
        schedule = (("a" * 64, "prompt one"), ("b" * 64, "prompt two"))
        stored = {"a" * 64: 1.0, "b" * 64: 2.0}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            with mock.patch.object(
                runner, "_load_pythia_nll", return_value=stored,
            ), mock.patch(
                "spectral_utils.a6_s0b.pythia_prompt_mean_nll",
                side_effect=[1.0, 2.5],
            ):
                with self.assertRaisesRegex(RuntimeError, "semantic replay mismatch"):
                    runner._recompute_pythia_nll(
                        out, {}, object(), object(), schedule,
                    )
            self.assertEqual(list(out.iterdir()), [])

    def test_semantic_match_returns_replay_digest(self) -> None:
        schedule = (("a" * 64, "prompt one"), ("b" * 64, "prompt two"))
        stored = {"a" * 64: 1.0, "b" * 64: 2.0}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            with mock.patch.object(
                runner, "_load_pythia_nll", return_value=stored,
            ), mock.patch(
                "spectral_utils.a6_s0b.pythia_prompt_mean_nll",
                side_effect=[1.0, 2.0],
            ):
                digest = runner._recompute_pythia_nll(
                    out, {}, object(), object(), schedule,
                )
            expected = hashlib.sha256(runner.canonical_json_bytes(
                [["a" * 64, 1.0], ["b" * 64, 2.0]]
            )).hexdigest()
            self.assertEqual(digest, expected)
            self.assertEqual(list(out.iterdir()), [])

    def test_verify_runs_semantic_replay_before_analysis_replay(self) -> None:
        boundary = {"prior_s0a": {
            "relative_path": "results/automatic_group_free_phase_a6_s0a_v1",
        }}
        analysis = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            with mock.patch.object(
                runner, "load_and_verify_boundary",
                return_value=(boundary, object(), object()),
            ), mock.patch.object(
                runner, "_load_quartet_payloads", return_value=[],
            ), mock.patch.object(
                runner, "_prompt_schedule", return_value=(),
            ), mock.patch.object(
                runner, "_recompute_pythia_nll",
                side_effect=RuntimeError("SEMANTIC_REPLAY_SENTINEL"),
            ), mock.patch.object(runner, "run_analysis", analysis):
                with self.assertRaisesRegex(RuntimeError, "SEMANTIC_REPLAY_SENTINEL"):
                    runner.verify(out)
        self.assertFalse(analysis.called)

    def test_verify_replays_analysis_with_byte_comparison(self) -> None:
        boundary = {"prior_s0a": {
            "relative_path": "results/automatic_group_free_phase_a6_s0a_v1",
        }}
        analysis = mock.Mock(return_value={"verdict": "PASS_S0B", "authorizes_s1": True})
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "A6_S0B_BOUNDARY.json").write_bytes(b"boundary")
            (out / "S0B_COMPLETE.json").write_bytes(b"complete")
            with mock.patch.object(
                runner, "load_and_verify_boundary",
                return_value=(boundary, object(), object()),
            ), mock.patch.object(
                runner, "_load_quartet_payloads", return_value=[],
            ), mock.patch.object(
                runner, "_prompt_schedule", return_value=(),
            ), mock.patch.object(
                runner, "_recompute_pythia_nll", return_value="f" * 64,
            ), mock.patch.object(
                runner, "run_analysis", analysis,
            ), mock.patch.object(runner, "_assert_known_output_paths"):
                result = runner.verify(out)
            self.assertTrue(analysis.call_args.kwargs["replay"])
            self.assertEqual(result["status"], "S0B_VERIFIED")
            self.assertEqual(result["verdict"], "PASS_S0B")
            self.assertEqual(result["pythia_replay_sha256"], "f" * 64)
            self.assertTrue(result["prior_s0a_full_verification"])
            self.assertTrue(result["authorizes_s1"])


class TestBootstrapResume(unittest.TestCase):
    """Handoff item 3: bootstrap resume validates draws/summary, skips redraw."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = _synthetic_population("qwen-source", "qwen3-4b")
        cls.bundles = _synthetic_bundles(cls.rows)
        cls.result = core.shortcut_gate_bootstrap(
            cls.rows, cls.bundles, "overall", n_draws=20_000,
        )

    def _written(self, out: Path):
        payload = runner._bootstrap_checkpoint(
            out, "qwen-source", 0, self.result, "b" * 64, replay=False,
        )
        return payload, out / "checkpoints" / "bootstrap" / "qwen-source" / "00.json"

    def _load(self, out: Path, *, gate_name: str = "overall", sha: str = "b" * 64):
        return runner._load_bootstrap_checkpoint(
            out, "qwen-source", 0, gate_name, self.rows, self.bundles, sha,
        )

    def test_valid_checkpoint_resumes_without_recomputation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            payload, _ = self._written(out)
            with mock.patch.object(
                core, "shortcut_gate_bootstrap",
                side_effect=AssertionError("resume must not redraw"),
            ), mock.patch.object(
                core, "bootstrap_group_multiplicities",
                side_effect=AssertionError("resume must not redraw"),
            ):
                loaded = self._load(out)
            self.assertEqual(loaded, payload)

    def test_tampered_draw_vector_fails_closed(self) -> None:
        cases = (
            ("altered draw value", lambda value: value["draws"].__setitem__(0, 0.75),
             "summary changed"),
            ("truncated draws", lambda value: value["draws"].pop(), "draw vector changed"),
            ("non-float draw", lambda value: value["draws"].__setitem__(0, True),
             "draw vector changed"),
        )
        for label, mutate, message in cases:
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as temporary:
                    out = Path(temporary) / "out"
                    out.mkdir()
                    _, path = self._written(out)
                    _rewrite_canonical(path, mutate)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self._load(out)

    def test_tampered_summary_fails_closed(self) -> None:
        for field, forged in (
            ("observed_max_macro_auc", 0.5),
            ("selected_ridge", 0.01),
            ("upper_95", 0.599),
            ("gate_pass", True),
            ("draw_count", 19_999),
            ("draw_unique_count", 7),
            ("draw_min", 0.0),
            ("draw_max", 0.5),
            ("draw_sha256", "0" * 64),
        ):
            with self.subTest(field):
                with tempfile.TemporaryDirectory() as temporary:
                    out = Path(temporary) / "out"
                    out.mkdir()
                    _, path = self._written(out)
                    _rewrite_canonical(path, lambda value: value.__setitem__(field, forged))
                    with self.assertRaisesRegex(RuntimeError, "summary changed"):
                        self._load(out)

    def test_tampered_identity_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            _, path = self._written(out)
            _rewrite_canonical(
                path, lambda value: value.__setitem__("population_id", "llama-audit"),
            )
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                self._load(out)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            self._written(out)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                self._load(out, gate_name="domain:arithmetic")
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            self._written(out)
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                self._load(out, sha="c" * 64)


class TestControlResume(unittest.TestCase):
    """Handoff item 4: control resume validates structure without rematching."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.group_ids, cls.records, cls.partitions, cls.edges = _control_fixture()
        cls.schedules = {
            family: core.materialize_control_schedule(
                family, 3, cls.partitions, cls.records, cls.edges,
            )
            for family in (2, 3)
        }

    def _written(self, out: Path, family: int):
        payload = runner._schedule_checkpoint(
            out, self.schedules[family], self.group_ids, "b" * 64, replay=False,
        )
        return payload, out / "checkpoints" / "control" / str(family) / "003.json"

    def _load(self, out: Path, family: int, *, sha: str = "b" * 64):
        return runner._load_schedule_checkpoint(
            out, family, 3, self.group_ids, self.partitions, self.records,
            self.edges, sha,
        )

    def test_valid_schedule_resumes_without_rematching(self) -> None:
        for family in (2, 3):
            with self.subTest(family=family):
                with tempfile.TemporaryDirectory() as temporary:
                    out = Path(temporary) / "out"
                    out.mkdir()
                    payload, _ = self._written(out, family)
                    with mock.patch.object(
                        core, "control2_derangement",
                        side_effect=AssertionError("resume must not rematch"),
                    ), mock.patch.object(
                        core, "control3_matching",
                        side_effect=AssertionError("resume must not rematch"),
                    ), mock.patch.object(
                        core, "hungarian_exact",
                        side_effect=AssertionError("resume must not rematch"),
                    ):
                        loaded = self._load(out, family)
                    self.assertEqual(loaded, payload)

    def test_fixed_point_and_roster_break_fail_closed(self) -> None:
        def make_fixed_point(value) -> None:
            pairs = value["assignments"][0][1]
            left0, right0 = pairs[0]
            donor_position = next(
                position for position, pair in enumerate(pairs)
                if position != 0 and pair[1] == left0
            )
            pairs[0] = [left0, left0]
            pairs[donor_position] = [pairs[donor_position][0], right0]

        def break_roster(value) -> None:
            pairs = value["assignments"][0][1]
            pairs[0] = [pairs[0][0], pairs[1][1]]

        for family in (2, 3):
            for label, mutate in (("fixed point", make_fixed_point), ("roster", break_roster)):
                with self.subTest(family=family, case=label):
                    with tempfile.TemporaryDirectory() as temporary:
                        out = Path(temporary) / "out"
                        out.mkdir()
                        _, path = self._written(out, family)
                        _rewrite_canonical(path, mutate)
                        with self.assertRaisesRegex(RuntimeError, "not a derangement"):
                            self._load(out, family)

    def test_stratum_and_eligibility_violations_fail_closed(self) -> None:
        def rotate_across_strata(value) -> None:
            rotated = [
                [index, (index + 1) % len(self.group_ids)]
                for index in range(len(self.group_ids))
            ]
            value["assignments"] = [
                [partition_id, [list(pair) for pair in rotated]]
                for partition_id, _ in value["assignments"]
            ]

        for family, message in (
            (2, "crossed a frozen stratum"),
            (3, "crossed an ineligible edge"),
        ):
            with self.subTest(family=family):
                with tempfile.TemporaryDirectory() as temporary:
                    out = Path(temporary) / "out"
                    out.mkdir()
                    _, path = self._written(out, family)
                    _rewrite_canonical(path, rotate_across_strata)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self._load(out, family)

    def test_seed_hash_and_schema_tampers_fail_closed(self) -> None:
        cases = (
            ("seed", lambda value: value.__setitem__(
                "seed_u64", value["seed_u64"] + 1), "seed changed"),
            ("schedule hash", lambda value: value.__setitem__(
                "schedule_sha256", "0" * 64), "schedule hash changed"),
            ("outer-held hash", lambda value: value.__setitem__(
                "outer_held_sha256", "0" * 64), "schedule hash changed"),
            ("partition order", lambda value: value["assignments"].reverse(),
             "partition order changed"),
            ("unknown partition", lambda value: value["assignments"][0].__setitem__(
                0, "outer:9:held"), "assignment schema changed"),
            ("boolean index", lambda value: value["assignments"][0][1].__setitem__(
                0, [True, 2]), "assignment index changed"),
        )
        for label, mutate, message in cases:
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as temporary:
                    out = Path(temporary) / "out"
                    out.mkdir()
                    _, path = self._written(out, 2)
                    _rewrite_canonical(path, mutate)
                    with self.assertRaisesRegex(RuntimeError, message):
                        self._load(out, 2)

    def test_identity_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            _, path = self._written(out, 2)
            _rewrite_canonical(path, lambda value: value.__setitem__("draw", 4))
            with self.assertRaisesRegex(RuntimeError, "identity changed"):
                self._load(out, 2)


class TestPriorProvenanceManifest(unittest.TestCase):
    """full_verify authenticates the sealed S0a tree via its own manifests.

    The S0a freeze is environment-locked (source tree, git HEAD, and macOS
    runtime of commit ba983aa), so its own authoritative replay can never
    pass on a later commit or another machine.  The prior check therefore
    verifies the completion -> aggregate -> checkpoint/result-file hash
    chain, which pins every byte S0b consumes.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not S0A.exists():
            raise unittest.SkipTest("canonical S0a artifacts are unavailable")
        cls._tmp = tempfile.TemporaryDirectory()
        cls.prior = Path(cls._tmp.name) / "s0a"
        shutil.copytree(S0A, cls.prior)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _tampered(self, relative: str, data: bytes):
        target = self.prior.joinpath(*relative.split("/"))
        original = target.read_bytes()

        class _Restore:
            def __enter__(self_inner):
                target.write_bytes(data)

            def __exit__(self_inner, *args):
                target.write_bytes(original)

        return _Restore()

    def test_full_verify_authenticates_manifest_chain(self) -> None:
        provenance = runner._prior_s0a_provenance(self.prior, full_verify=True)
        self.assertEqual(set(provenance), {
            "relative_path", "boundary_sha256", "aggregate_sha256",
            "completion_sha256",
        })
        skip_mode = runner._prior_s0a_provenance(self.prior, full_verify=False)
        self.assertEqual(provenance, skip_mode)

    def test_tampered_checkpoint_fails_closed(self) -> None:
        path = "checkpoints/quartet/0000.json"
        original = self.prior.joinpath(*path.split("/")).read_bytes()
        with self._tampered(path, original + b"\n"):
            with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                runner._prior_s0a_provenance(self.prior, full_verify=True)

    def test_tampered_result_file_fails_closed(self) -> None:
        original = (self.prior / "NULL_STRATA.json").read_bytes()
        with self._tampered("NULL_STRATA.json", original + b"\n"):
            with self.assertRaisesRegex(RuntimeError, "result file changed"):
                runner._prior_s0a_provenance(self.prior, full_verify=True)

    def test_tampered_aggregate_is_not_the_sealed_tree(self) -> None:
        value = json.loads((self.prior / "S0A_AGGREGATE.json").read_text(encoding="utf-8"))
        value["checkpoint_manifest"] = value["checkpoint_manifest"][:-1]
        with self._tampered("S0A_AGGREGATE.json", runner.canonical_json_bytes(value)):
            with self.assertRaisesRegex(RuntimeError, "not the sealed Step-268"):
                runner._prior_s0a_provenance(self.prior, full_verify=True)

    def test_tampered_completion_hash_field_fails_closed(self) -> None:
        # The boundary/aggregate bytes stay sealed; only the completion's
        # recorded aggregate hash is forged, so the pin passes and the
        # completion-chain check must fire.
        value = json.loads((self.prior / "S0A_COMPLETE.json").read_text(encoding="utf-8"))
        value["aggregate_sha256"] = "0" * 64
        with self._tampered("S0A_COMPLETE.json", runner.canonical_json_bytes(value)):
            with self.assertRaisesRegex(RuntimeError, "completion hashes changed"):
                runner._prior_s0a_provenance(self.prior, full_verify=True)

    def test_flipped_completion_verdict_fails_closed(self) -> None:
        value = json.loads((self.prior / "S0A_COMPLETE.json").read_text(encoding="utf-8"))
        value["verdict"] = "CLOSE_INVALID_INTERVENTION_BOUNDARY"
        with self._tampered("S0A_COMPLETE.json", runner.canonical_json_bytes(value)):
            with self.assertRaisesRegex(RuntimeError, "completion verdict changed"):
                runner._prior_s0a_provenance(self.prior, full_verify=True)


class TestTerminalArtifactsAndLayout(unittest.TestCase):
    """Handoff item 5: terminal exclusivity and the exact output namespace."""

    def test_run_analysis_refuses_after_terminal_artifact(self) -> None:
        for terminal in ("S0B_COMPLETE.json", "S0B_CLOSED.json"):
            with self.subTest(terminal):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    out, provenance = _fake_prepare(root)
                    (out / terminal).write_bytes(b"terminal")
                    patches = _boundary_patches(provenance)
                    with patches[0], patches[1]:
                        with self.assertRaisesRegex(
                            RuntimeError, "immutable terminal artifact",
                        ):
                            runner.run_analysis(out, verify_prior=False)

    def test_verify_rejects_conflicting_terminals(self) -> None:
        boundary = {"prior_s0a": {
            "relative_path": "results/automatic_group_free_phase_a6_s0a_v1",
        }}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "A6_S0B_BOUNDARY.json").write_bytes(b"boundary")
            (out / "S0B_COMPLETE.json").write_bytes(b"complete")
            (out / "S0B_CLOSED.json").write_bytes(b"closed")
            with mock.patch.object(
                runner, "load_and_verify_boundary",
                return_value=(boundary, object(), object()),
            ), mock.patch.object(
                runner, "_load_quartet_payloads", return_value=[],
            ), mock.patch.object(
                runner, "_prompt_schedule", return_value=(),
            ), mock.patch.object(
                runner, "_recompute_pythia_nll", return_value="f" * 64,
            ), mock.patch.object(
                runner, "run_analysis",
                return_value={"verdict": "PASS_S0B", "authorizes_s1": True},
            ), mock.patch.object(runner, "_assert_known_output_paths"):
                with self.assertRaisesRegex(RuntimeError, "terminal artifact set"):
                    runner.verify(out)

    def test_verify_rejects_terminal_inconsistent_with_verdict(self) -> None:
        boundary = {"prior_s0a": {
            "relative_path": "results/automatic_group_free_phase_a6_s0a_v1",
        }}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            (out / "A6_S0B_BOUNDARY.json").write_bytes(b"boundary")
            (out / "S0B_COMPLETE.json").write_bytes(b"complete")
            with mock.patch.object(
                runner, "load_and_verify_boundary",
                return_value=(boundary, object(), object()),
            ), mock.patch.object(
                runner, "_load_quartet_payloads", return_value=[],
            ), mock.patch.object(
                runner, "_prompt_schedule", return_value=(),
            ), mock.patch.object(
                runner, "_recompute_pythia_nll", return_value="f" * 64,
            ), mock.patch.object(
                runner, "run_analysis",
                return_value={
                    "verdict": "CLOSE_S0B_SHORTCUT_CONFOUNDING",
                    "authorizes_s1": False,
                },
            ), mock.patch.object(runner, "_assert_known_output_paths"):
                with self.assertRaisesRegex(RuntimeError, "terminal artifact set"):
                    runner.verify(out)

    def test_checkpoint_namespaces_fail_closed(self) -> None:
        boundary = {"pythia_input": {
            "relative_directory": "inputs/pythia-x",
            "files": [{"path": "config.json"}],
        }}
        cases = (
            (("checkpoints", "pythia", "14400.json"), "Pythia checkpoint namespace"),
            (("checkpoints", "surprise", "00.json"), "unknown family"),
            (("checkpoints", "bootstrap", "qwen", "00.json"),
             "bootstrap population namespace"),
            (("checkpoints", "bootstrap", "qwen-source", "19.json"),
             "bootstrap checkpoint namespace"),
            (("checkpoints", "control", "4", "000.json"), "control family namespace"),
            (("checkpoints", "control", "2", "200.json"),
             "control checkpoint namespace"),
            (("scratch", "x.json"), "unmanifested directory"),
        )
        for parts, message in cases:
            with self.subTest("/".join(parts)):
                with tempfile.TemporaryDirectory() as temporary:
                    out = Path(temporary) / "out"
                    target = out.joinpath(*parts)
                    target.parent.mkdir(parents=True)
                    target.write_bytes(b"{}")
                    with self.assertRaisesRegex(RuntimeError, message):
                        runner._assert_known_output_paths(out, boundary)

    def test_symlink_in_output_root_fails_closed(self) -> None:
        boundary = {"pythia_input": {
            "relative_directory": "inputs/pythia-x",
            "files": [{"path": "config.json"}],
        }}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            target = out / "S0B_AGGREGATE.json"
            target.write_bytes(b"{}")
            try:
                os.symlink(target, out / "alias.json")
            except OSError:
                self.skipTest("symlink creation requires privileges on this platform")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                runner._assert_known_output_paths(out, boundary)

    def test_interrupted_temporary_recovery_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            data = runner.canonical_json_bytes({"a": 1})
            path = out / "S0B_AGGREGATE.json"
            (out / "S0B_AGGREGATE.json.tmp").write_bytes(data)
            runner._exclusive_bytes(path, data, root=out)
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse((out / "S0B_AGGREGATE.json.tmp").exists())
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            data = runner.canonical_json_bytes({"a": 1})
            path = out / "S0B_AGGREGATE.json"
            (out / "S0B_AGGREGATE.json.tmp").write_bytes(b"crash junk")
            runner._exclusive_bytes(path, data, root=out)
            self.assertEqual(path.read_bytes(), data)
            self.assertFalse((out / "S0B_AGGREGATE.json.tmp").exists())

    def test_immutable_artifact_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "out"
            out.mkdir()
            path = out / "S0B_AGGREGATE.json"
            stored = runner.canonical_json_bytes({"a": 1})
            path.write_bytes(stored)
            with self.assertRaisesRegex(RuntimeError, "immutable artifact mismatch"):
                runner._exclusive_bytes(
                    path, runner.canonical_json_bytes({"a": 2}), root=out,
                )
            self.assertEqual(path.read_bytes(), stored)


class TestUnexpectedExceptionRouting(unittest.TestCase):
    """Handoff item 6: unexpected exceptions are implementation-invalid."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.qwen_rows = _synthetic_population("qwen-source", "qwen3-4b")
        cls.llama_rows = _synthetic_population("llama-audit", "llama31-8b")
        cls.rows = cls.qwen_rows + cls.llama_rows
        cls.group_ids, cls.records, cls.partitions, cls.edges = _control_fixture()

    def _analysis_stack(self, stack: ExitStack, provenance) -> None:
        stack.enter_context(mock.patch.object(
            runner, "_load_input_spec_stdlib", return_value=_FakeInputSpec,
        ))
        stack.enter_context(mock.patch.object(
            runner, "_prior_s0a_provenance", return_value=provenance,
        ))
        stack.enter_context(mock.patch.object(
            runner, "_load_quartet_payloads", return_value=[],
        ))
        stack.enter_context(mock.patch.object(
            runner, "_prompt_schedule", return_value=(("a" * 64, "prompt"),),
        ))
        stack.enter_context(mock.patch.object(
            runner, "_load_pythia_nll", return_value={"a" * 64: 1.0},
        ))
        stack.enter_context(mock.patch.object(
            core, "build_shortcut_rows", return_value=self.rows,
        ))

    def test_unexpected_fit_exceptions_propagate_without_closure(self) -> None:
        for label, error in (
            ("value error", ValueError("implementation defect")),
            ("foreign runtime error", RuntimeError("disk error while fitting")),
        ):
            with self.subTest(label):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    out, provenance = _fake_prepare(root)
                    with ExitStack() as stack:
                        self._analysis_stack(stack, provenance)
                        stack.enter_context(mock.patch.object(
                            core, "fit_oof_bundles", side_effect=error,
                        ))
                        with self.assertRaises(type(error)):
                            runner.run_analysis(out, verify_prior=False)
                    self.assertFalse((out / "S0B_CLOSED.json").exists())
                    self.assertFalse((out / "S0B_COMPLETE.json").exists())

    def test_registered_nonconvergence_closes_scientifically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out, provenance = _fake_prepare(root)
            with ExitStack() as stack:
                self._analysis_stack(stack, provenance)
                stack.enter_context(mock.patch.object(
                    core, "fit_oof_bundles",
                    side_effect=RuntimeError(
                        "shortcut logistic is unusable: ABNORMAL; gradient=2.1e-07",
                    ),
                ))
                result = runner.run_analysis(out, verify_prior=False)
            self.assertEqual(result["verdict"], "CLOSE_S0B_NUMERICAL_NONCONVERGENCE")
            self.assertFalse(result["authorizes_s1"])
            closed = json.loads((out / "S0B_CLOSED.json").read_text(encoding="utf-8"))
            self.assertEqual(closed["verdict"], "CLOSE_S0B_NUMERICAL_NONCONVERGENCE")
            self.assertFalse((out / "S0B_COMPLETE.json").exists())

    def _control_phase_stack(self, stack: ExitStack, provenance) -> None:
        self._analysis_stack(stack, provenance)
        stack.enter_context(mock.patch.object(
            core, "fit_oof_bundles",
            side_effect=[
                _synthetic_bundles(self.qwen_rows),
                _synthetic_bundles(self.llama_rows),
            ],
        ))
        stack.enter_context(mock.patch.object(
            core, "gate_names", return_value=("overall",),
        ))
        stack.enter_context(mock.patch.object(
            core, "shortcut_gate_bootstrap",
            return_value=SimpleNamespace(
                gate_name="overall", observed_max_macro_auc=0.5,
                selected_ridge=10.0, upper_95=0.5, gate_pass=True,
                bootstrap_max_macro_auc=(0.5, 0.5),
            ),
        ))
        stack.enter_context(mock.patch.object(
            core, "marginal_prevalence_audit", return_value={"pass": True},
        ))
        stack.enter_context(mock.patch.object(
            core, "group_matching_records", return_value=self.records,
        ))
        stack.enter_context(mock.patch.object(
            core, "freeze_matching_graph",
            return_value=core.MatchingFreeze(
                group_ids=self.group_ids, vector_sha256="v" * 64, caliper=1.0,
                unordered_pool_size=3, directed_eligible_edges=self.edges,
            ),
        ))
        stack.enter_context(mock.patch.object(
            core, "canonical_partition_memberships", return_value=self.partitions,
        ))

    def test_registered_control_exhaustion_closes_scientifically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out, provenance = _fake_prepare(root)
            with ExitStack() as stack:
                self._control_phase_stack(stack, provenance)
                stack.enter_context(mock.patch.object(
                    core, "materialize_control_schedule",
                    side_effect=RuntimeError("CLOSE_S0B_CONTROL2_DERANGEMENT_EXHAUSTED"),
                ))
                result = runner.run_analysis(out, verify_prior=False)
            self.assertEqual(result["verdict"], "CLOSE_S0B_MATCHING_PREMISE")
            self.assertTrue((out / "S0B_CLOSED.json").exists())
            self.assertFalse((out / "S0B_COMPLETE.json").exists())

    def test_unexpected_control_exception_propagates_without_closure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out, provenance = _fake_prepare(root)
            with ExitStack() as stack:
                self._control_phase_stack(stack, provenance)
                stack.enter_context(mock.patch.object(
                    core, "materialize_control_schedule",
                    side_effect=KeyError("implementation defect"),
                ))
                with self.assertRaises(KeyError):
                    runner.run_analysis(out, verify_prior=False)
            self.assertFalse((out / "S0B_CLOSED.json").exists())
            self.assertFalse((out / "S0B_COMPLETE.json").exists())


if __name__ == "__main__":
    unittest.main()
