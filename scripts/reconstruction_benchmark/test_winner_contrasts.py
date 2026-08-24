#!/usr/bin/env python3
"""Tests for downstream all-pairs and point-winner contrast artifacts."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_utils.reconstruction_benchmark.external_evaluation import (  # noqa: E402
    grouped_paired_bootstrap,
    population_grouped_paired_bootstrap,
)
from spectral_utils.reconstruction_benchmark.io import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
)
from spectral_utils.reconstruction_benchmark.winner_contrasts import (  # noqa: E402
    EvaluatedScopeMetric,
    LoadedSource,
    WinnerContrastError,
    _external_cell_draws,
    _external_population_draws,
    _rename_directory_noreplace,
    derive_winner_contrasts,
    publish_winner_contrasts,
    verify_winner_contrast_artifact,
    verify_winner_contrasts_ab,
)


def _source(
    *, metric: str = "auroc", points: dict[str, float] | None = None,
    draws: dict[str, np.ndarray] | None = None,
    source_type: str = "synthetic_shared_draws",
    source_binding: dict[str, object] | None = None,
) -> LoadedSource:
    method_ids = ("zeta", "alpha", "middle")
    if points is None:
        points = {"zeta": 0.7, "alpha": 0.72, "middle": 0.71}
    if draws is None:
        base = np.linspace(0.45, 0.85, 20_000)
        draws = {
            "zeta": base, "alpha": base + 0.02, "middle": base + 0.01,
        }
    else:
        draws = {
            method_id: np.resize(np.asarray(values, dtype=float), 20_000)
            for method_id, values in draws.items()
        }
    scope = EvaluatedScopeMetric(
        lane_id="synthetic", population_id="population",
        scope_type="cell", scope_value="cell", record_level="cell",
        cell_id="cell", dataset_id="dataset", model_id="model",
        slice_id="slice", cell_ids=("cell",), aggregation="single_cell",
        source_comparison_group_id=f"source::{metric}",
        bootstrap_unit="source_group", bootstrap_seed=17,
        linked_resampling=False, stratified_by_label=False,
        bootstrap_draws_requested=20_000, points=points, draws=draws,
        method_statuses={method_id: "OK" for method_id in method_ids},
        metric=metric,
    )
    return LoadedSource(
        source_type=source_type, lane_id="synthetic",
        method_ids=method_ids, metric_ids=(metric,), scopes=(scope,),
        source_binding=source_binding or {"fixture_sha256": "0" * 64},
    )


def _csv_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))


class DerivationTests(unittest.TestCase):
    def test_direct_pairing_not_marginal_ci_overlap(self) -> None:
        common = np.linspace(0.2, 0.8, 401)
        source = _source(
            points={"zeta": 0.50, "alpha": 0.52, "middle": 0.51},
            draws={
                "zeta": common, "alpha": common + 0.02,
                "middle": common + 0.01,
            },
        )
        result = derive_winner_contrasts(source)
        pair = next(row for row in result["all_pairs"]
                    if row["method_a_id"] == "alpha" and row["method_b_id"] == "zeta")
        self.assertEqual(pair["relation"], "A_BETTER")
        self.assertGreater(pair["oriented_ci_low"], 0.0)
        # The two marginal 95% intervals overlap almost completely; only the
        # shared-draw difference identifies this deterministic separation.
        self.assertLess(np.quantile(common + 0.02, 0.025), np.quantile(common, 0.975))
        separated = next(row for row in result["winner_sets"] if row["method_id"] == "zeta")
        self.assertFalse(separated["in_winner_reference_set"])
        self.assertEqual(separated["membership_status"], "SEPARATED_FROM_POINT_WINNER_95CI")
        self.assertIn("membership is determined by whether", separated["interpretation"])

    def test_lower_is_better_aurc_orientation(self) -> None:
        base = np.linspace(10.0, 30.0, 101)
        result = derive_winner_contrasts(_source(
            metric="aurc_x1000",
            points={"zeta": 2.0, "alpha": 1.0, "middle": 3.0},
            draws={"zeta": base + 1.0, "alpha": base, "middle": base + 2.0},
        ))
        pair = next(row for row in result["all_pairs"]
                    if row["method_a_id"] == "alpha" and row["method_b_id"] == "zeta")
        self.assertEqual(pair["relation"], "A_BETTER")
        self.assertEqual(pair["higher_is_better"], False)
        self.assertGreater(pair["oriented_advantage_a_over_b"], 0.0)
        winner = {row["method_id"]: row for row in result["winner_sets"]}
        self.assertEqual(winner["alpha"]["membership_status"], "POINT_WINNER")

    def test_utf8_first_representative_for_numerical_point_tie(self) -> None:
        base = np.linspace(0.2, 0.8, 101)
        result = derive_winner_contrasts(_source(
            points={"zeta": 0.7, "alpha": 0.7, "middle": 0.6},
            draws={"zeta": base, "alpha": base, "middle": base - 0.1},
        ))
        rows = result["winner_sets"]
        self.assertEqual({row["winner_reference_method_id"] for row in rows}, {"alpha"})
        self.assertEqual(
            {row["method_id"] for row in rows if row["membership_status"] == "POINT_WINNER"},
            {"alpha", "zeta"},
        )

    def test_rejects_method_specific_valid_draw_masks(self) -> None:
        base = np.linspace(0.2, 0.8, 20_000)
        alpha = base + 0.02
        alpha[17] = np.nan
        with self.assertRaisesRegex(WinnerContrastError, "same accepted draw indexes"):
            derive_winner_contrasts(_source(draws={
                "zeta": base, "alpha": alpha, "middle": base + 0.01,
            }))

    def test_rejects_point_tie_separated_from_representative(self) -> None:
        base = np.linspace(0.2, 0.8, 20_000)
        with self.assertRaisesRegex(WinnerContrastError, "point-winner tie is separated"):
            derive_winner_contrasts(_source(
                points={"zeta": 0.7, "alpha": 0.7, "middle": 0.6},
                draws={"zeta": base + 0.1, "alpha": base, "middle": base - 0.1},
            ))


class ExactBootstrapCounterpartTests(unittest.TestCase):
    def test_cell_counterpart_matches_frozen_evaluator(self) -> None:
        rng = np.random.default_rng(2)
        labels = np.asarray([0, 1, 0, 1, 0, 1, 1, 0], dtype=np.int8)
        groups = tuple(f"g{index // 2}" for index in range(len(labels)))
        scores = {
            "iu_pcr": rng.normal(size=len(labels)),
            "candidate": rng.normal(size=len(labels)),
        }
        expected = grouped_paired_bootstrap(
            labels=labels, scores_by_method=scores, group_ids=groups,
            draws=500, seed=991, reference_method="iu_pcr",
        )
        points, draws, metadata = _external_cell_draws(
            labels=labels, scores_by_method=scores, group_ids=groups,
            draws=500, seed=991, stratify_by_label=False,
        )
        self.assertEqual(metadata["valid_draws"], expected["valid_draws"])
        for method_id in scores:
            for metric in ("auroc", "auprc", "aurc_x1000"):
                values = draws[method_id][metric]
                self.assertEqual(points[method_id][metric], expected["metrics"][method_id][metric]["value"])
                self.assertEqual(float(np.quantile(values, 0.025)), expected["metrics"][method_id][metric]["ci_low"])
                self.assertEqual(float(np.quantile(values, 0.975)), expected["metrics"][method_id][metric]["ci_high"])
        for metric in ("auroc", "auprc", "aurc_x1000"):
            delta = draws["candidate"][metric] - draws["iu_pcr"][metric]
            contrast = expected["contrasts"]["candidate"][metric]
            self.assertEqual(float(np.quantile(delta, 0.025)), contrast["ci_low"])
            self.assertEqual(float(np.quantile(delta, 0.975)), contrast["ci_high"])

    def test_linked_population_counterpart_matches_frozen_evaluator(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int8)
        groups = tuple(f"g{index}" for index in range(len(labels)))
        cells = {
            "c1": {
                "labels": labels, "group_ids": groups,
                "scores_by_method": {
                    "iu_pcr": np.asarray([0., 1., 2., 3., 4., 5.]),
                    "candidate": np.asarray([1., 0., 3., 2., 5., 4.]),
                },
            },
            "c2": {
                "labels": labels, "group_ids": groups,
                "scores_by_method": {
                    "iu_pcr": np.asarray([0.2, 1.2, 2.2, 3.2, 4.2, 5.2]),
                    "candidate": np.asarray([1.2, 0.2, 3.2, 2.2, 5.2, 4.2]),
                },
            },
        }
        links = {"c1": "shared", "c2": "shared"}
        expected = population_grouped_paired_bootstrap(
            cells=cells, link_keys=links, draws=400, seed=812,
            reference_method="iu_pcr", weighting="equal_cell",
        )
        points, draws, metadata = _external_population_draws(
            cells=cells, link_keys=links, draws=400, seed=812,
            weighting="equal_cell", stratify_by_label=False,
        )
        self.assertEqual(metadata["valid_draws"], expected["valid_draws"])
        self.assertTrue(metadata["linked_resampling"])
        for method_id in ("candidate", "iu_pcr"):
            for metric in ("auroc", "auprc", "aurc_x1000"):
                values = draws[method_id][metric]
                self.assertEqual(points[method_id][metric], expected["metrics"][method_id][metric]["value"])
                self.assertEqual(float(np.quantile(values, 0.025)), expected["metrics"][method_id][metric]["ci_low"])
                self.assertEqual(float(np.quantile(values, 0.975)), expected["metrics"][method_id][metric]["ci_high"])


class ArtifactTests(unittest.TestCase):
    def test_publish_verify_ab_and_mutation_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first"
            second = root / "second"
            source_a = _source()
            source_b = _source()
            publish_winner_contrasts(source_a, output_dir=first, replica_id="A")
            publish_winner_contrasts(source_b, output_dir=second, replica_id="B")
            verified = verify_winner_contrast_artifact(first, source=source_a)
            self.assertEqual(verified["manifest"]["row_counts"]["all_pairs_contrasts.csv"], 3)
            certificate = verify_winner_contrasts_ab(
                first, second, source_a=source_a, source_b=source_b,
                output_path=root / "AB.json",
            )
            self.assertEqual(certificate["status"], "PASS")
            self.assertTrue(certificate["exact_source_rederivation"]["performed"])
            with self.assertRaises(FileExistsError):
                verify_winner_contrasts_ab(
                    first, second, source_a=source_a, source_b=source_b,
                    output_path=root / "AB.json",
                )
            with (first / "winner_reference_sets.csv").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaises(WinnerContrastError):
                verify_winner_contrast_artifact(first, source=source_a)

    def test_refreshed_self_hashes_cannot_forge_source_rederivation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _source()
            artifact = root / "artifact"
            publish_winner_contrasts(source, output_dir=artifact, replica_id="A")
            table_path = artifact / "all_pairs_contrasts.csv"
            with table_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = tuple(reader.fieldnames or ())
                rows = [dict(row) for row in reader]
            rows[0]["method_a_value"] = str(float(rows[0]["method_a_value"]) + 0.125)
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            forged = stream.getvalue().encode("utf-8")
            table_path.write_bytes(forged)
            manifest_path = artifact / "MANIFEST.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"]["all_pairs_contrasts.csv"] = {
                "sha256": sha256_bytes(forged), "bytes": len(forged),
            }
            manifest.pop("payload_sha256")
            manifest["payload_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
            manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
            with self.assertRaisesRegex(WinnerContrastError, "exact source rederivation"):
                verify_winner_contrast_artifact(artifact, source=source)

    def test_directory_publish_primitive_does_not_replace_raced_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            target = root / "target"
            staging.mkdir()
            target.mkdir()
            (staging / "new").write_text("new", encoding="utf-8")
            (target / "incumbent").write_text("incumbent", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                _rename_directory_noreplace(staging, target)
            self.assertTrue((staging / "new").is_file())
            self.assertTrue((target / "incumbent").is_file())

    def test_external_build_normalization_is_enumerated(self) -> None:
        common = {
            "evaluation_ab_certificate_sha256": "1" * 64,
            "evaluation_ab_certificate_file_sha256": "2" * 64,
            "metrics_sha256": "3" * 64, "contrasts_sha256": "4" * 64,
            "score_artifact_roster_sha256": "5" * 64,
            "label_artifact_roster_sha256": "6" * 64,
        }
        def binding(build: str) -> dict[str, object]:
            return {
                **common, "build_id": build,
                "evaluation_manifest_sha256": build.lower() * 64,
                "evaluation_manifest_payload_sha256": ("c" if build == "A" else "d") * 64,
                "score_freeze_manifest_sha256": ("e" if build == "A" else "f") * 64,
                "score_freeze_manifest_payload_sha256": ("7" if build == "A" else "8") * 64,
            }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for build in ("A", "B"):
                sources[build] = _source(
                    source_type="external_v3_signed_score_label_recomputation",
                    source_binding=binding(build),
                )
                publish_winner_contrasts(
                    sources[build], output_dir=root / build, replica_id=build,
                )
            result = verify_winner_contrasts_ab(
                root / "A", root / "B", source_a=sources["A"],
                source_b=sources["B"], output_path=root / "AB.json",
            )
            self.assertEqual(result["status"], "PASS")
            self.assertIn("build_id", result["normalization_contract"]["source_binding_build_fields"])

            bad_sources = {}
            for replica in ("A", "B"):
                bad_sources[replica] = _source(
                    source_type="external_v3_signed_score_label_recomputation",
                    source_binding=binding("A"),
                )
                publish_winner_contrasts(
                    bad_sources[replica], output_dir=root / f"bad_{replica}",
                    replica_id=replica,
                )
            with self.assertRaisesRegex(WinnerContrastError, "source builds A then B"):
                verify_winner_contrasts_ab(
                    root / "bad_A", root / "bad_B",
                    source_a=bad_sources["A"], source_b=bad_sources["B"],
                    output_path=root / "BAD_AB.json",
                )


if __name__ == "__main__":
    unittest.main()
