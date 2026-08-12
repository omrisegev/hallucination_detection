#!/usr/bin/env python3
"""Sentinel tests for the white-box layer-fusion HTML report."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
import re
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.whitebox_layer_fusion_report import (  # noqa: E402
    FIGURES,
    build_report,
    derive_validation_status,
    sha256_file,
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _SemanticAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tables: list[dict[str, bool]] = []
        self._table: dict[str, bool] | None = None
        self.external_refs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "table":
            self._table = {"caption": False, "thead": False, "tbody": False, "scoped_th": False}
            self.tables.append(self._table)
        elif self._table is not None and tag in {"caption", "thead", "tbody"}:
            self._table[tag] = True
        elif self._table is not None and tag == "th" and attributes.get("scope") == "col":
            self._table["scoped_th"] = True
        for name in ("src", "href"):
            value = attributes.get(name) or ""
            if value.startswith(("http://", "https://", "//")):
                self.external_refs.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table = None


def _fixture(
    directory: Path,
    *,
    gate_b: bool = True,
    architecture: bool = True,
    raw_backfill_gate_b: bool = True,
) -> None:
    cells = ("sentinel_gsm8k", "sentinel_nq_open")
    methods = (
        ("final_layer_nll", 0.7000),
        ("iu_pcr", 0.7210),
        ("dufs_liu_pcr", 0.7312),
        ("token_length", 0.5900),
        ("dufs_liu_pcr_length_residualized", 0.7280),
    )
    per_cell = []
    for cell_index, cell in enumerate(cells):
        for method, value in methods:
            per_cell.append(
                {
                    "cell": cell,
                    "method": method,
                    "feature_contract": "resid-core-32",
                    "layer_subset": "all32",
                    "structured": "flat",
                    "auroc": f"{value - 0.01 * cell_index:.4f}",
                    "auprc": f"{value - 0.08 - 0.01 * cell_index:.4f}",
                    "prevalence": "0.3200",
                    "n_samples": "500",
                    "n_groups": "500",
                    "status": "ok",
                    "extra_runner_column": "tolerated",
                }
            )
        per_cell.extend(
            (
                {
                    "cell": cell,
                    "method": "balanced_logistic_regression",
                    "feature_contract": "resid-core-32",
                    "layer_subset": "all32",
                    "structured": "flat",
                    "auroc": f"{0.8800 - .01 * cell_index:.4f}",
                    "auprc": "0.8100",
                    "prevalence": "0.3200",
                    "n_samples": "500",
                    "n_groups": "500",
                    "status": "diagnostic_only",
                    "label_use": "supervised_ceiling",
                },
                {
                    "cell": cell,
                    "method": "best_single_layer",
                    "feature_contract": "resid-core-32",
                    "layer_subset": "oracle_layer",
                    "structured": "flat",
                    "auroc": f"{0.8400 - .01 * cell_index:.4f}",
                    "auprc": "0.7600",
                    "prevalence": "0.3200",
                    "n_samples": "500",
                    "n_groups": "500",
                    "status": "evaluation_only",
                    "label_use": "evaluation_only",
                },
            )
        )
    _write_csv(directory / "per_cell_metrics.csv", per_cell)
    _write_csv(
        directory / "headline_summary.csv",
        [
            {
                "method": method,
                "feature_contract": "resid-core-32",
                "layer_subset": "all32",
                "macro_auroc": f"{value:.4f}",
                "macro_auroc_ci_low": f"{value - .0100:.4f}",
                "macro_auroc_ci_high": f"{value + .0100:.4f}",
                "macro_auprc": f"{value - .0800:.4f}",
            }
            for method, value in methods[:3]
        ]
        + [
            {
                "method": "balanced_logistic_regression",
                "feature_contract": "resid-core-32",
                "layer_subset": "all32",
                "macro_auroc": "0.9999",
                "macro_auroc_ci_low": "0.9900",
                "macro_auroc_ci_high": "1.0000",
                "macro_auprc": "0.9990",
                "label_use": "supervised_ceiling",
            }
        ],
    )
    _write_csv(
        directory / "paired_comparisons.csv",
        [
            {
                "contrast": "dufs_liu_all32_minus_final_layer_nll",
                "lhs": "dufs_liu_pcr",
                "rhs": "final_layer_nll",
                "metric": "auroc",
                "delta": "0.0312",
                "ci_low": "0.0100",
                "ci_high": "0.0510",
                "wins": "2",
                "ties": "0",
                "losses": "0",
                "worst_cell_delta": "0.0200",
                "p_raw": "0.0500",
                "p_holm": "0.1000",
                "primary": "true",
            },
            {
                "contrast": "dufs_liu_all32_minus_iu_pcr_all32",
                "lhs": "dufs_liu_pcr",
                "rhs": "iu_pcr",
                "metric": "auroc",
                "delta": "0.0102",
                "ci_low": "0.0010",
                "ci_high": "0.0200",
                "wins": "2",
                "ties": "0",
                "losses": "0",
                "worst_cell_delta": "0.0040",
                "p_raw": "0.1000",
                "p_holm": "0.1000",
                "primary": "true",
            },
        ],
    )
    _write_csv(
        directory / "data_coverage.csv",
        [
            {
                "cell": cell,
                "cell_id": cell,
                "n_source_rows": "500",
                "n_samples": "500",
                "n_excluded_rows": "0",
                "n_groups": "500",
                "prevalence": "0.3200",
                "raw_backfill_gate_b_status": "pass" if raw_backfill_gate_b else "fail",
                "raw_backfill_gate_b_median": "0.0110",
                "raw_backfill_gate_b_first": "0.0210",
                "raw_backfill_gate_b_fraction": "0.9400",
                "corrected_layer_gate_b_status": "pass" if gate_b else "fail",
                "corrected_layer_gate_b_median": "0.0080",
                "corrected_layer_gate_b_first": "0.0140",
                "corrected_layer_gate_b_fraction": "0.9700",
                "gate_b_status": "pass" if gate_b else "fail",
                "architecture_status": "pass" if architecture else "fail",
                "status": "complete",
                "exclusion_reason": "",
            }
            for cell in cells
        ],
    )
    _write_csv(
        directory / "layer_diagnostics.csv",
        [
            {
                "cell": cells[index % 2],
                "layer": str(layer),
                "metric": metric,
                "module": "resid",
                "auroc": f"{0.51 + layer * .006 + index * .002:.4f}",
                "spearman_to_anchor": f"{0.10 + layer * .01:.4f}",
            }
            for index, metric in enumerate(("lens_H", "lens_logp_tgt"))
            for layer in (0, 8, 16, 24, 31)
        ],
    )
    _write_csv(
        directory / "dependence_diagnostics.csv",
        [
            {
                "cell": "sentinel_gsm8k",
                "contract": "resid-core-32",
                "diagnostic": "correlation_vs_layer_distance",
                "layer_distance": str(distance),
                "value": f"{0.80 - distance * .02:.4f}",
                "feature_a": "",
                "feature_b": "",
                "effective_rank": "7.2",
            }
            for distance in (1, 4, 8, 16, 24)
        ]
        + [
            {
                "cell": "sentinel_gsm8k",
                "contract": "resid-core-32",
                "diagnostic": "layer_correlation",
                "layer_distance": str(abs(first - second)),
                "value": f"{1.0 - abs(first-second) * .025:.4f}",
                "feature_a": f"layer_{first:02d}",
                "feature_b": f"layer_{second:02d}",
                "effective_rank": "",
            }
            for first in (0, 8, 16, 24, 31)
            for second in (0, 8, 16, 24, 31)
        ]
        + [
            {
                "cell": "sentinel_nq_open",
                "contract": "lens-96",
                "diagnostic": "effective_rank",
                "layer_distance": "",
                "value": "9.5000",
                "feature_a": "",
                "feature_b": "",
                "effective_rank": "9.5000",
            }
        ],
    )
    _write_csv(
        directory / "weights_diagnostics.csv",
        [
            {
                "cell": "sentinel_gsm8k",
                "method": method,
                "contract": "resid-core-32",
                "kind": "fusion_weight",
                "feature": f"layer_{layer:02d}",
                "value": f"{(layer - 12) / 40:.4f}",
                "seed": "",
                "epoch": "",
                "graph_components": "",
                "mean_degree": "",
                "spectral_gap": "",
                "converged": "",
            }
            for method in ("iu_pcr", "dufs_liu_pcr")
            for layer in (0, 8, 16, 24, 31)
        ]
        + [
            {
                "cell": "sentinel_gsm8k",
                "method": "dufs_liu_pcr",
                "contract": "resid-core-32",
                "kind": "dufs_gate",
                "feature": f"layer_{layer:02d}",
                "value": f"{0.4 + layer / 80:.4f}",
                "seed": "11",
                "epoch": "80",
                "graph_components": "",
                "mean_degree": "",
                "spectral_gap": "",
                "converged": "true",
            }
            for layer in (0, 8, 16, 24, 31)
        ]
        + [
            {
                "cell": "sentinel_gsm8k",
                "method": "dufs_liu_pcr",
                "contract": "resid-core-32",
                "kind": "graph_health",
                "feature": "",
                "value": "",
                "seed": "11",
                "epoch": "80",
                "graph_components": "1",
                "mean_degree": "6.75",
                "spectral_gap": "0.083",
                "converged": "true",
            }
        ],
    )
    _write_json(
        directory / "validation_status.json",
        {
            "status": "VALIDATED",  # never trusted without the two booleans below
            "gates": {
                "corrected_layer_gate_b_all_pass": gate_b,
                "architecture_pilot_pass": architecture,
            },
        },
    )
    _write_json(
        directory / "RUN_DEFINITION.json",
        {"protocol_signature": "sentinel-protocol-sha", "cells": list(cells)},
    )
    _write_json(directory / "SOURCE_FREEZE_MANIFEST.json", {"sources_frozen": True})
    _write_json(
        directory / "SCORE_FREEZE_MANIFEST.json",
        {"labels_seen_during_fit": False, "scores_frozen_before_labels": True},
    )


class WhiteboxLayerFusionReportTest(unittest.TestCase):
    def test_self_contained_report_and_manifest_trace_every_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _fixture(directory)
            manifest = build_report(directory)
            report = (directory / "REPORT.html").read_text(encoding="utf-8")

            self.assertEqual(manifest["status"], "VALIDATED")
            self.assertIn("0.7312", report)  # sentinel value came from headline/per-cell CSV
            self.assertIn("sentinel_gsm8k", report)
            self.assertIn("labels seen during fitting", report.lower())
            self.assertIn("Raw backfill Gate B versus corrected live Gate B", report)
            self.assertIn("Raw median error", report)
            self.assertIn("Live median error", report)
            self.assertIn("direct raw-<code>token_entropies</code> versus sidecar-<code>lens_H</code> comparison is invalid", report)
            self.assertIn("Label-using diagnostic ceilings", report)
            self.assertIn("balanced_logistic_regression", report)
            self.assertIn("Fusion weights", report)
            self.assertIn("DUFS gates", report)
            self.assertIn("Graph and convergence health", report)
            self.assertNotIn("<pre", report.lower())
            semantic = _SemanticAudit()
            semantic.feed(report)
            self.assertEqual(semantic.external_refs, [])
            self.assertNotRegex(report.lower(), r"url\(\s*[\"']?(?:https?:)?//")
            self.assertGreaterEqual(len(semantic.tables), 10)
            self.assertTrue(all(all(table.values()) for table in semantic.tables))

            # Responsive containment heuristics: the document declares a mobile
            # viewport, every figure/image is bounded, grids can shrink to zero,
            # and wide semantic tables scroll inside their own wrapper.
            self.assertIn('name="viewport"', report)
            self.assertIn("max-width:100%", report)
            self.assertIn("overflow-x:auto", report)
            self.assertIn("minmax(0,1fr)", report)
            self.assertIn(".grid>*{min-width:0}", report)
            self.assertIn("@media(max-width:680px)", report)
            self.assertNotIn("min-width:340px", report)
            self.assertNotIn("min-width:210px", report)

            embedded = re.findall(r'<img src="data:image/svg\+xml;base64,([A-Za-z0-9+/=]+)"', report)
            self.assertEqual(len(embedded), len(FIGURES))
            for name, encoded in zip(FIGURES, embedded):
                figure = directory / "figures" / name
                self.assertTrue(figure.is_file())
                self.assertEqual(base64.b64decode(encoded), figure.read_bytes())
                self.assertIn("<svg", figure.read_text(encoding="utf-8"))
            correlation = (directory / "figures" / "layer_correlation_heatmap.svg").read_text()
            self.assertIn("Layer-correlation heatmap", correlation)
            self.assertIn("layer_00", correlation)
            eligible_heatmap = (directory / "figures" / "per_cell_heatmap.svg").read_text()
            eligible_forest = (directory / "figures" / "macro_forest.svg").read_text()
            self.assertNotIn("balanced_logistic_regression", eligible_heatmap)
            self.assertNotIn("0.9999", eligible_forest)

            disk_manifest = json.loads((directory / "REPORT_MANIFEST.json").read_text())
            self.assertEqual(disk_manifest["generated_artifacts"]["REPORT.html"], sha256_file(directory / "REPORT.html"))
            for name in FIGURES:
                key = f"figures/{name}"
                self.assertEqual(disk_manifest["generated_artifacts"][key], sha256_file(directory / key))
            expected_csv_hash = hashlib.sha256((directory / "per_cell_metrics.csv").read_bytes()).hexdigest()
            self.assertEqual(disk_manifest["input_artifacts"]["per_cell_metrics.csv"], expected_csv_hash)

    def test_failed_or_missing_required_gate_forces_preliminary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _fixture(directory, gate_b=False, architecture=True)
            manifest = build_report(directory)
            report = (directory / "REPORT.html").read_text(encoding="utf-8")
            self.assertFalse(manifest["validated"])
            self.assertEqual(manifest["status"], "PRELIMINARY / VALIDATION BLOCKED")
            self.assertIn("PRELIMINARY / VALIDATION BLOCKED", report)
            self.assertIn('data-validation-status="blocked"', report)
            self.assertIn('role="alert"', report)
            self.assertIn("descriptive only", report)
            self.assertIn("Corrected live Gate B (all 14 cells)", manifest["blockers"])

            # A claimed status cannot bypass a missing explicit architecture result.
            _write_json(directory / "validation_status.json", {"status": "VALIDATED", "gate_b_all_pass": True})
            validation = derive_validation_status(
                json.loads((directory / "validation_status.json").read_text()), directory
            )
            self.assertFalse(validation["validated"])
            self.assertIsNone(validation["architecture_pilot"])

    def test_raw_backfill_gate_is_audited_but_corrected_layer_gate_controls_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _fixture(directory, gate_b=True, architecture=True, raw_backfill_gate_b=False)
            manifest = build_report(directory)
            report = (directory / "REPORT.html").read_text(encoding="utf-8")
            self.assertTrue(manifest["validated"])
            self.assertIn("Raw backfill Gate B", report)
            self.assertIn("Corrected live Gate B", report)
            self.assertIn("Only corrected live Gate B controls report promotion", report)

    def test_missing_freeze_record_also_blocks_promotion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _fixture(directory)
            (directory / "SCORE_FREEZE_MANIFEST.json").unlink()
            manifest = build_report(directory)
            self.assertFalse(manifest["validated"])
            self.assertIn("SCORE_FREEZE_MANIFEST.json", " ".join(manifest["blockers"]))

    def test_label_access_or_missing_prelabel_freeze_cannot_be_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _fixture(directory)
            _write_json(
                directory / "SCORE_FREEZE_MANIFEST.json",
                {"labels_seen_during_fit": True, "scores_frozen_before_labels": False},
            )
            manifest = build_report(directory)
            report = (directory / "REPORT.html").read_text(encoding="utf-8")
            self.assertFalse(manifest["validated"])
            self.assertIn("Leakage boundary", " ".join(manifest["blockers"]))
            self.assertIn("Score hashes frozen before labels opened", manifest["blockers"])
            self.assertIn('data-validation-status="blocked"', report)


if __name__ == "__main__":
    unittest.main()
