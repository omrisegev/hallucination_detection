#!/usr/bin/env python3
"""Contract tests for the Reasoning Localization 0.3662 living report."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO / "results" / "reasoning_localization_03662_v1"
MODULE_PATH = REPO / "spectral_utils" / "reasoning_localization_reporting.py"
SPEC = importlib.util.spec_from_file_location("reasoning_localization_reporting_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {MODULE_PATH}")
REPORTING = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = REPORTING
SPEC.loader.exec_module(REPORTING)


class SemanticAudit(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.internal_hrefs: list[str] = []
        self.external_refs: list[str] = []
        self.tables: list[dict[str, bool]] = []
        self._table: dict[str, bool] | None = None
        self.svgs = 0
        self.figures = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if "id" in attributes and attributes["id"] is not None:
            self.ids.append(attributes["id"])
        href = attributes.get("href") or ""
        if href.startswith("#"):
            self.internal_hrefs.append(href[1:])
        for name in ("src", "href"):
            value = attributes.get(name) or ""
            if value.startswith(("http://", "https://", "//")):
                self.external_refs.append(value)
        if tag == "table":
            self._table = {"caption": False, "thead": False, "tbody": False, "scoped_th": False}
            self.tables.append(self._table)
        elif self._table is not None and tag in {"caption", "thead", "tbody"}:
            self._table[tag] = True
        elif self._table is not None and tag == "th" and attributes.get("scope") == "col":
            self._table["scoped_th"] = True
        elif tag == "svg":
            self.svgs += 1
        elif tag == "figure":
            self.figures += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self._table = None


class ReportingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = REPORTING.load_bundle(REPORT_DIR)
        REPORTING.validate_bundle(cls.bundle, REPORT_DIR, REPO)
        cls.build = REPORTING.prepare_build(REPORT_DIR, REPO)
        cls.report = cls.build.html_bytes.decode("utf-8")

    def test_locked_roster_and_three_axis_status(self) -> None:
        registry = self.bundle["variant_registry"]
        variants = registry["variants"]
        ids = {row["variant_id"] for row in variants}
        self.assertEqual(47, len(variants))
        self.assertEqual(set(REPORTING.EXPECTED_NEW_VARIANTS), set(REPORTING.EXPECTED_NEW_VARIANTS) & ids)
        self.assertEqual(7, len(registry["allowed_execution_statuses"]))
        self.assertEqual(6, len(registry["allowed_decision_statuses"]))
        self.assertEqual(5, len(registry["allowed_evidence_statuses"]))
        for variant in variants:
            self.assertIn(variant["execution_status"], registry["allowed_execution_statuses"])
            self.assertIn(variant["decision_status"], registry["allowed_decision_statuses"])
            self.assertIn(variant["evidence_status"], registry["allowed_evidence_statuses"])

    def test_historical_rows_are_context_only_and_never_rankable(self) -> None:
        variants = {row["variant_id"]: row for row in self.bundle["variant_registry"]["variants"]}
        for metric in self.bundle["metrics"]:
            if metric["status"] == "CONTEXT_ONLY":
                self.assertFalse(variants[metric["variant_id"]]["rankable"])
                self.assertEqual("CONTEXT_ONLY", metric["evidence_status"])
        mutated = copy.deepcopy(self.bundle)
        context = next(row for row in mutated["variant_registry"]["variants"] if row["execution_status"] == "CONTEXT_ONLY")
        context["rankable"] = True
        with self.assertRaisesRegex(REPORTING.ReportingValidationError, "context variant"):
            REPORTING.validate_bundle(mutated, REPORT_DIR, REPO)

    def test_planned_variants_have_no_numeric_results(self) -> None:
        planned = {row["variant_id"] for row in self.bundle["variant_registry"]["variants"] if row["execution_status"] == "PLANNED"}
        numeric = {row["variant_id"] for row in self.bundle["metrics"] if row["value"] != ""}
        self.assertFalse(planned & numeric)

    def test_hierarchical_family_expert_template_is_bounded_and_unexecuted(self) -> None:
        variants = {row["variant_id"]: row for row in self.bundle["variant_registry"]["variants"]}
        branch = variants["P3_HIER_FAMILY_EXPERTS"]
        self.assertEqual("PLANNED", branch["execution_status"])
        self.assertEqual("PENDING", branch["decision_status"])
        self.assertFalse(branch["rankable"])
        self.assertEqual(["P3_EQUAL_FAMILY_OUTER_REFERENCE"], branch["parent_variant_ids"])
        contract = branch["family_expert_contract"]
        self.assertEqual(1, contract["entropy_level"]["member_count"])
        self.assertEqual(["passthrough"], contract["entropy_level"]["allowed_inner_fusions"])
        self.assertEqual(28, sum(row["member_count"] for row in contract.values()))
        for family_id, row in contract.items():
            if family_id != "entropy_level":
                self.assertIn("U-PCR", row["allowed_inner_fusions"])
                self.assertIn("IU-PCR", row["allowed_inner_fusions"])
        selectors = branch["selection_contract"]
        self.assertIn("audit/test ProcessBench F1", selectors["forbidden_selectors"])
        self.assertIn("any PRMBench label or metric", selectors["forbidden_selectors"])

    def test_reducer_study_is_compact_factorized_and_unexecuted(self) -> None:
        variants = {row["variant_id"]: row for row in self.bundle["variant_registry"]["variants"]}
        stage_a = [row for row in variants.values() if row.get("reducer_stage") == "A_IDENTITY_AGGREGATION"]
        stage_b = [row for row in variants.values() if row.get("reducer_stage") == "B_SURVIVOR_TRANSFORM"]
        self.assertEqual(12, len(stage_a))
        self.assertEqual(4, len(stage_b))
        self.assertTrue(all(row["execution_status"] == "PLANNED" for row in stage_a + stage_b))
        self.assertTrue(all(row["rankable"] for row in stage_a))
        self.assertTrue(all(not row["rankable"] for row in stage_b))
        self.assertTrue(all(row["task_ids"] == ["processbench_first_error"] for row in stage_a + stage_b))
        self.assertIn("min(5, |I_s|)", variants["P2R_A_TOPK5_REFERENCE"]["step_reducer"])
        self.assertIn("ceil(0.25 |I_s|)", variants["P2R_A_TOPQ25"]["step_reducer"])
        self.assertIn("max(0, c_{t-1} + z_t - kappa)", variants["P2R_B_POS_CUSUM_TEMPLATE"]["transforms"][0])
        self.assertIn("suffix-invariance", variants["P2R_B_DSP_CAUSAL_TEMPLATE"]["causal_validity"])
        experiments = {row["experiment_id"]: row for row in self.bundle["experiment_registry"]["experiments"]}
        contract = experiments["P2_REDUCER_STUDY"]["reducer_contract"]
        self.assertEqual(12, len(contract["stage_a_order"]))
        self.assertEqual(4, len(contract["stage_b_templates"]))
        self.assertIn("no candidate-specific rethresholding", contract["threshold_rule"])
        self.assertEqual("ProcessBench", REPORTING._task_for_variant(stage_a[0], self.bundle["experiment_registry"]["experiments"]))

    def test_comparison_group_cannot_mix_task_population_or_metric(self) -> None:
        mutated = copy.deepcopy(self.bundle)
        changed = mutated["metrics"][1]
        changed["population_id"] = "different_population"
        with self.assertRaisesRegex(REPORTING.ReportingValidationError, "mixes task/population/metric"):
            REPORTING.validate_bundle(mutated, REPORT_DIR, REPO)

    def test_historical_values_and_hashes_are_verified_from_sources(self) -> None:
        metric = self.bundle["metrics"][0]
        self.assertEqual("0.3662328341717007", metric["value"])
        self.assertEqual("8072dd180ece6a992221a87684fee196b0dbf4c502cb376a2a3e9c071637e3eb", metric["source_sha256"])
        contrast = self.bundle["contrasts"][0]
        self.assertEqual("0.004811475804772508", contrast["delta"])
        self.assertEqual("-0.02638710838275541", contrast["ci_low"])
        mutated = copy.deepcopy(self.bundle)
        mutated["metrics"][0]["value"] = "0.999"
        with self.assertRaisesRegex(REPORTING.ReportingValidationError, "copied metric does not equal source"):
            REPORTING.validate_bundle(mutated, REPORT_DIR, REPO)

    def test_phase0_s0_is_registered_as_a_nonpromoting_checksum_audit(self) -> None:
        variants = {row["variant_id"]: row for row in self.bundle["variant_registry"]["variants"]}
        r2 = variants["R2_HISTORICAL_FAMILY6_BRIDGE"]
        self.assertEqual("COMPLETE", r2["execution_status"])
        self.assertEqual("NO_PROMOTION", r2["decision_status"])
        self.assertFalse(r2["rankable"])
        metric = next(row for row in self.bundle["metrics"] if row["variant_id"] == r2["variant_id"] and row["status"] == "COMPLETE")
        self.assertEqual("0.3662328341717007", metric["value"])
        self.assertEqual("1270", metric["n_rows"])
        self.assertEqual("635", metric["n_groups"])
        self.assertEqual("d12d651c", metric["comparison_group_id"].rsplit("_", 1)[-1])
        self.assertIn("P0 RUNNING", self.report)
        p0_plot = next(row for row in self.build.resolved_plots if row["plot_id"] == "PLOT_P0_WATERFALL")
        self.assertEqual("RENDERED", p0_plot["render_status"])
        self.assertIn("family6 hybrid bridge", self.report)

    def test_phase0_s1_is_a_source_bound_one_factor_audit(self) -> None:
        variants = {row["variant_id"]: row for row in self.bundle["variant_registry"]["variants"]}
        s1 = variants["P0_S1_FAMILY6_STEP_MAX"]
        self.assertEqual("COMPLETE", s1["execution_status"])
        self.assertEqual("NO_PROMOTION", s1["decision_status"])
        self.assertFalse(s1["rankable"])
        metric = next(
            row for row in self.bundle["metrics"]
            if row["variant_id"] == s1["variant_id"] and row["metric_id"] == "macro_f1"
        )
        self.assertEqual("0.33007771561392063", metric["value"])
        contrast = next(
            row for row in self.bundle["contrasts"]
            if row["left_variant_id"] == s1["variant_id"] and row["metric_id"] == "macro_f1"
        )
        self.assertEqual("-0.03615511855778009", contrast["delta"])
        self.assertLess(float(contrast["ci_high"]), 0.0)
        waterfall = next(row for row in self.bundle["plot_manifest"]["plots"] if row["plot_id"] == "PLOT_P0_WATERFALL")
        bridge = next(row for row in self.bundle["plot_manifest"]["plots"] if row["plot_id"] == "PLOT_P0_BRIDGE_FOREST")
        self.assertEqual("macro_f1", waterfall["selection"]["metric_id"])
        self.assertEqual("macro_f1", bridge["selection"]["metric_id"])

    def test_phase0_gate_rows_are_bound_to_the_frozen_replay_artifact(self) -> None:
        self.assertEqual(9, len(self.bundle["gates"]))
        self.assertTrue(all(row["passed"] == "true" for row in self.bundle["gates"]))
        mutated = copy.deepcopy(self.bundle)
        mutated["gates"][0]["observed"] = "false"
        with self.assertRaisesRegex(REPORTING.ReportingValidationError, "copied gate observation does not equal source"):
            REPORTING.validate_bundle(mutated, REPORT_DIR, REPO)

    def test_two_builds_are_byte_identical_and_manifest_binds_output(self) -> None:
        second = REPORTING.prepare_build(REPORT_DIR, REPO)
        self.assertEqual(self.build.html_bytes, second.html_bytes)
        self.assertEqual(self.build.manifest, second.manifest)
        self.assertEqual(REPORTING.sha256_bytes(self.build.html_bytes), self.build.manifest["output"]["sha256"])
        manifest_projection = {key: value for key, value in self.build.manifest.items() if key != "report_manifest_sha256"}
        self.assertEqual(
            REPORTING.sha256_bytes(REPORTING.canonical_json_bytes(manifest_projection)),
            self.build.manifest["report_manifest_sha256"],
        )
        self.assertEqual(21, len(self.build.manifest["plots"]))
        self.assertTrue(all(row["source_sha256"] for row in self.build.manifest["plots"]))
        self.assertTrue(all(row["selection_rule"] for row in self.build.manifest["plots"]))
        self.assertTrue(all("comparison_group" in row and "bootstrap_definition" in row for row in self.build.manifest["plots"]))
        self.assertTrue(any(row["role"] == "registered_result_artifact" for row in self.build.manifest["inputs"]))

    def test_every_variant_is_in_cards_and_master_table(self) -> None:
        for variant in self.bundle["variant_registry"]["variants"]:
            self.assertIn(f'id="method-{variant["variant_id"].lower()}"', self.report)
            self.assertGreaterEqual(self.report.count(variant["variant_id"]), 3)
        self.assertLess(self.report.index("PART I"), self.report.index("table-variants"))
        self.assertIn("Historical context — not rankable", self.report)

    def test_task_results_are_separate_and_no_combined_overall(self) -> None:
        self.assertIn("ProcessBench", self.report)
        self.assertIn("PRMBench", self.report)
        self.assertIn("Early", self.report)
        self.assertNotIn("overall score</th>", self.report.lower())
        self.assertIn("no overall score exists", self.report)

    def test_missing_and_pending_are_not_rendered_as_zero(self) -> None:
        self.assertIn("PLANNED — no eligible registered rows", self.report)
        self.assertIn("Missing is not zero", self.report)
        self.assertNotIn("NO_ELIGIBLE_CASE</strong><p>0", self.report)

    def test_semantic_self_contained_html_and_internal_anchors(self) -> None:
        audit = SemanticAudit()
        audit.feed(self.report)
        self.assertFalse(audit.external_refs)
        self.assertEqual(len(audit.ids), len(set(audit.ids)))
        self.assertFalse(set(audit.internal_hrefs) - set(audit.ids))
        self.assertGreaterEqual(audit.svgs, 3)
        self.assertGreaterEqual(audit.figures, 21)
        self.assertGreaterEqual(len(audit.tables), 6)
        self.assertTrue(all(all(table.values()) for table in audit.tables))
        self.assertIn("@media(max-width:760px)", self.report)
        self.assertIn("@media print", self.report)
        self.assertIn("data-download=", self.report)
        self.assertIn("filter-query", self.report)

    def test_claim_references_resolve_and_cannot_be_invented(self) -> None:
        mutated = copy.deepcopy(self.bundle)
        mutated["claims"]["claims"][0]["evidence_refs"].append("PLOT_DOES_NOT_EXIST")
        with self.assertRaisesRegex(REPORTING.ReportingValidationError, "unknown plot"):
            REPORTING.validate_bundle(mutated, REPORT_DIR, REPO)

    def test_snapshot_is_idempotent_but_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as temp_name:
            report_dir = Path(temp_name) / "report"
            snapshot = REPORTING.create_immutable_snapshot(report_dir, "reporting", self.build, REPO)
            self.assertEqual(snapshot, REPORTING.create_immutable_snapshot(report_dir, "reporting", self.build, REPO))
            (snapshot / "REPORT.html").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(REPORTING.ReportingValidationError, "immutable snapshot differs"):
                REPORTING.create_immutable_snapshot(report_dir, "reporting", self.build, REPO)

    def test_check_mode_detects_stale_output(self) -> None:
        with tempfile.TemporaryDirectory(dir=REPO) as temp_name:
            target = Path(temp_name)
            REPORTING.write_build(target, self.build)
            REPORTING.check_build(target, self.build)
            (target / "REPORT.html").write_bytes(self.build.html_bytes + b"\n")
            with self.assertRaisesRegex(REPORTING.ReportingValidationError, "stale or nondeterministic"):
                REPORTING.check_build(target, self.build)

    def test_embedded_json_is_parseable_and_download_source_complete(self) -> None:
        marker = '<script id="report-data" type="application/json">'
        payload = self.report.split(marker, 1)[1].split("</script>", 1)[0].replace("<\\/", "</")
        embedded = json.loads(payload)
        self.assertEqual(47, len(embedded["variants"]))
        self.assertEqual(self.bundle["metrics"], embedded["metrics"])
        self.assertEqual(self.bundle["contrasts"], embedded["contrasts"])
        self.assertEqual(self.bundle["gates"], embedded["gates"])

    def test_all_registered_chart_kinds_render_from_synthetic_registered_rows(self) -> None:
        plots = {row["kind"]: row for row in self.bundle["plot_manifest"]["plots"]}
        variants = {row["variant_id"]: row for row in self.bundle["variant_registry"]["variants"]}
        waterfall = REPORTING._waterfall_svg(plots["waterfall"], [
            {"variant_id": "R0_ENTROPY_MAX", "value": "0.30", "display_order": "1"},
            {"variant_id": "R1_ENTROPY_TOP5", "value": "0.32", "display_order": "2"},
        ], variants)
        heatmap = REPORTING._heatmap_svg(plots["heatmap"], [
            {"variant_id": "R0_ENTROPY_MAX", "cell_id": "cell_a", "value": "0.30"},
            {"variant_id": "R0_ENTROPY_MAX", "cell_id": "cell_b", "value": "0.34"},
        ], variants)
        gate = REPORTING._gate_matrix_svg(plots["gate_matrix"], [
            {"variant_id": "C1_ENT_SW16", "gate_id": "delta", "passed": "true"},
            {"variant_id": "C1_ENT_SW16", "gate_id": "worst", "passed": "false"},
        ], variants)
        scatter_plot = next(row for row in self.bundle["plot_manifest"]["plots"] if row["plot_id"] == "PLOT_P2_EXACT_CLEAN")
        scatter = REPORTING._scatter_svg(scatter_plot, [
            {"variant_id": "C1_ENT_SW16", "metric_id": "first_error_exact_delta", "value": "0.01"},
            {"variant_id": "C1_ENT_SW16", "metric_id": "clean_abstention_delta", "value": "-0.002"},
        ], variants)
        line = REPORTING._line_svg(plots["line"], [
            {"variant_id": "P5_CAUSAL__C1", "metric_id": "auroc", "axis_value": "64", "value": "0.58"},
            {"variant_id": "P5_CAUSAL__C1", "metric_id": "auroc", "axis_value": "128", "value": "0.61"},
        ])
        for rendered in (waterfall, heatmap, gate, scatter, line):
            self.assertIn("<svg", rendered)
            self.assertNotIn("PLANNED —", rendered)

    def test_plot_selection_accepts_registered_stratum_lists(self) -> None:
        self.assertTrue(REPORTING._matches({"slice_id": "prm_train"}, {"slice_id": ["prm_train", "prm_test"]}))
        self.assertFalse(REPORTING._matches({"slice_id": "other"}, {"slice_id": ["prm_train", "prm_test"]}))


if __name__ == "__main__":
    unittest.main()
