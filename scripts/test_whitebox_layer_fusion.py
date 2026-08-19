#!/usr/bin/env python3
"""Synthetic, dataset-free gates for white-box layer fusion."""

from __future__ import annotations

import copy
import inspect
import os
import sys
import types
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
# Import the focused submodule without executing spectral_utils/__init__.py,
# whose model-loading exports require the optional local torch/transformers
# stack even though these CPU-only gates do not.
if "spectral_utils" not in sys.modules:
    package = types.ModuleType("spectral_utils")
    package.__path__ = [os.path.join(REPO, "spectral_utils")]
    sys.modules["spectral_utils"] = package

from spectral_utils.whitebox_layer_fusion import (  # noqa: E402
    ALL_LAYERS,
    LATE_LAYERS,
    SPACED_LAYERS,
    FeatureMatrix,
    apply_neutral_residual_calibration,
    assert_no_label_fitting_signatures,
    assert_same_protocol,
    all_layers,
    entropy_agreement_gate,
    extract_dola_kl_proxy,
    extract_geometry,
    extract_haloscope_projection,
    extract_lens96,
    extract_lens_grid,
    extract_resid_core,
    extract_trilens_entropy,
    fit_controls,
    fit_core_spectral,
    fit_dependency_methods,
    fit_hierarchical,
    fit_haloscope_direct_proxy,
    fit_group_contribution_space,
    fit_neutral_residual_calibration,
    fixed_bands,
    late_layers,
    load_evaluation_labels,
    residualize_token_length,
    spaced_layers,
    validate_and_join,
)


def synthetic_artifacts(n_problems=8, candidates_per_problem=3, seed=29, n_layers=32):
    rng = np.random.default_rng(seed)
    n_modules, projection_dim, covariance_rank = 3, 8, 4
    n_rows = n_problems * candidates_per_problem
    latent_risk = np.linspace(-1.8, 1.8, n_rows) + rng.normal(0.0, 0.06, n_rows)
    raw, sidecar = {}, {
        "_meta": {
            "version": "layer-lens-v1",
            "model": "synthetic/Llama-3.1-8B-Instruct",
            "n_layers": n_layers,
            "hidden_size": 16,
            "modules": ["attn", "mlp", "resid"],
            "quantities": ["lens_H", "lens_logp_tgt", "lens_logp_top1", "lens_kl_final"],
            "proj_seed": 20260811,
            "proj_dim": projection_dim,
            "cov_eigs_r": covariance_rank,
            "dtype": "float32",
            "complete": True,
        }
    }
    layer_axis = np.arange(n_layers, dtype=float)[None, :, None]
    module_axis = np.arange(n_modules, dtype=float)[:, None, None]
    row_index = 0
    for problem in range(n_problems):
        raw[problem] = {"question": f"q{problem}", "candidates": []}
        for candidate_index in range(candidates_per_problem):
            risk = float(latent_risk[row_index])
            n_tokens = 2 + row_index % 6
            token_axis = np.arange(n_tokens, dtype=float)[None, None, :]
            noise = rng.normal(0.0, 0.012, (n_modules, n_layers, n_tokens))
            entropy = 1.8 + 0.42 * risk + 0.005 * layer_axis + 0.008 * module_axis
            entropy = entropy + 0.002 * token_axis + noise
            target_nll = 2.2 + 0.70 * risk + 0.004 * layer_axis + 0.006 * module_axis
            target_nll = target_nll + 0.003 * token_axis + noise * 0.6
            top1_surprisal = 1.2 + 0.48 * risk + 0.003 * layer_axis + 0.004 * module_axis
            top1_surprisal = top1_surprisal + 0.002 * token_axis + noise * 0.5
            distance = ((n_layers - 1.0) - layer_axis) / max(n_layers - 1.0, 1.0) + (2.0 - module_axis) * 0.05
            kl = (0.7 + 0.16 * risk + noise * 0.2) * distance
            kl[2, -1, :] = 0.0

            residual_norm = (
                8.0 + 0.18 * risk + 0.025 * np.arange(n_layers)[:, None]
                + 0.004 * np.arange(n_tokens)[None, :]
                + rng.normal(0.0, 0.008, (n_layers, n_tokens))
            )
            hidden = rng.normal(0.0, 0.04, (n_layers, projection_dim))
            hidden += np.arange(n_layers)[:, None] * np.linspace(0.01, 0.025, projection_dim)
            hidden += risk * np.linspace(-0.03, 0.04, projection_dim)[None, :]
            cov = np.empty((n_layers, covariance_rank), dtype=float)
            base_eigs = np.exp(-np.arange(covariance_rank, dtype=float) * 0.7)
            for layer in range(n_layers):
                cov[layer] = base_eigs * (1.0 + 0.003 * layer)
                cov[layer, 0] *= 1.0 + 0.07 * (risk + 2.0)

            label = bool(risk < 0.0)
            raw_candidate = {
                "gen_token_ids": list(range(n_tokens)),
                "token_entropies": entropy[2, -1].astype(float).tolist(),
                "label": label,
            }
            raw[problem]["candidates"].append(raw_candidate)
            sidecar[f"{problem}:{candidate_index}"] = {
                "lens_H": entropy.astype(np.float32),
                "lens_logp_tgt": (-target_nll).astype(np.float32),
                "lens_logp_top1": (-top1_surprisal).astype(np.float32),
                "lens_kl_final": kl.astype(np.float32),
                "resid_norm": residual_norm.astype(np.float32),
                "cov_eigs": cov.astype(np.float32),
                "hid_proj": hidden.astype(np.float32),
                "n_gen_tokens": n_tokens,
                "label": label,
            }
            row_index += 1
    return raw, sidecar, latent_risk


def joined_fixture():
    raw, sidecar, risk = synthetic_artifacts()
    cell, audit = validate_and_join(
        raw,
        sidecar,
        cell_id="synthetic",
        expected_model="synthetic/Llama-3.1-8B-Instruct",
        expected_hidden_size=16,
        expected_projection_dim=8,
        expected_covariance_rank=4,
    )
    return raw, sidecar, risk, cell, audit


class WhiteboxLayerFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw, cls.sidecar, cls.risk, cls.cell, cls.audit = joined_fixture()

    def test_nested_join_is_exact_and_label_free(self):
        self.assertEqual(self.audit["n_rows"], 24)
        self.assertEqual(self.audit["n_problems"], 8)
        self.assertEqual(self.audit["n_excluded_rows"], 0)
        self.assertLess(self.audit["min_tokens"], self.audit["max_tokens"])
        self.assertEqual(self.cell.row_ids[0], "0:0")
        self.assertEqual(self.cell.row_ids[-1], "7:2")
        self.assertTrue(all("label" not in record for record in self.cell.records))
        blocked_gate = entropy_agreement_gate(self.raw, self.cell)
        self.assertFalse(blocked_gate["pass"])
        self.assertTrue(blocked_gate["numeric_thresholds_pass"])
        self.assertFalse(blocked_gate["comparable_domains"])
        gate = entropy_agreement_gate(
            self.raw, self.cell, assume_comparable_domains=True
        )
        self.assertTrue(gate["pass"], gate)
        self.assertEqual(gate["n_compared_rows"], self.cell.n_samples)
        evaluation = load_evaluation_labels(self.raw, self.cell.row_ids)
        self.assertEqual(evaluation.shape, (24,))
        self.assertEqual(set(evaluation.tolist()), {0, 1})

    def test_invalid_truncation_can_be_reported_and_excluded(self):
        raw = copy.deepcopy(self.raw)
        raw[0]["candidates"][0]["gen_token_ids"].append(999)
        raw[0]["candidates"][0]["token_entropies"].append(9.0)
        with self.assertRaisesRegex(ValueError, "generated-token length mismatch"):
            validate_and_join(
                raw,
                self.sidecar,
                cell_id="strict",
                expected_model="synthetic/Llama-3.1-8B-Instruct",
                expected_hidden_size=16,
                expected_projection_dim=8,
                expected_covariance_rank=4,
            )
        cell, audit = validate_and_join(
            raw,
            self.sidecar,
            cell_id="excluding",
            expected_model="synthetic/Llama-3.1-8B-Instruct",
            expected_hidden_size=16,
            expected_projection_dim=8,
            expected_covariance_rank=4,
            exclude_invalid=True,
        )
        self.assertEqual(cell.n_samples, 23)
        self.assertEqual(audit["n_excluded_rows"], 1)
        self.assertEqual(audit["excluded_rows"][0]["row_id"], "0:0")
        self.assertIn("sidecar=2, raw=3", audit["excluded_rows"][0]["reason"])

    def test_resid_core_directions_and_final_kl_degeneracy(self):
        matrix = extract_resid_core(self.cell)
        self.assertEqual(matrix.values.shape, (24, 32))
        self.assertEqual(len(set(matrix.groups)), 4)
        self.assertGreater(np.corrcoef(matrix.risk_anchor, self.risk)[0, 1], 0.98)
        self.assertGreater(np.corrcoef(matrix.values[:, -1], self.risk)[0, 1], 0.95)
        self.assertIn(
            "resid.lens_kl_final.layer_31",
            matrix.metadata["dropped_degenerate_components"],
        )
        self.assertNotIn(
            "lens_kl_final",
            matrix.metadata["components_by_feature"]["resid_core.layer_31"],
        )
        self.assertEqual(extract_resid_core(self.cell, SPACED_LAYERS).n_features, 8)
        self.assertEqual(extract_resid_core(self.cell, LATE_LAYERS).n_features, 8)

    def test_lens_grid_and_lens96_have_fixed_named_groups(self):
        full = extract_lens_grid(self.cell, ALL_LAYERS)
        spaced = extract_lens96(self.cell)
        self.assertEqual(full.metadata["nominal_feature_count"], 384)
        self.assertEqual(spaced.metadata["nominal_feature_count"], 96)
        self.assertEqual(spaced.n_features, 95)
        self.assertEqual(len(set(spaced.groups)), 12)
        self.assertIn("resid.lens_kl_final.layer_31", spaced.metadata["dropped_degenerate_features"])

    def test_architecture_relative_layers_and_comparator_contracts(self):
        self.assertEqual(spaced_layers(32), SPACED_LAYERS)
        self.assertEqual(spaced_layers(36), (0, 5, 10, 15, 20, 25, 30, 35))
        self.assertEqual(spaced_layers(40), (0, 6, 11, 17, 22, 28, 33, 39))
        self.assertEqual(late_layers(40), tuple(range(32, 40)))
        self.assertEqual(len(all_layers(36)), 36)
        self.assertEqual(tuple(map(len, fixed_bands(40))), (10, 10, 10, 10))

        raw, sidecar, _risk = synthetic_artifacts(n_layers=36)
        cell, _audit = validate_and_join(
            raw, sidecar, cell_id="synthetic-36",
            expected_model="synthetic/Llama-3.1-8B-Instruct",
            expected_n_layers=36, expected_hidden_size=16,
            expected_projection_dim=8, expected_covariance_rank=4,
        )
        self.assertEqual(extract_resid_core(cell).n_features, 36)
        trilens = extract_trilens_entropy(cell)
        self.assertEqual(trilens.metadata["nominal_feature_count"], 108)
        self.assertEqual(len(set(trilens.groups)), 3)
        dola = extract_dola_kl_proxy(cell)
        self.assertEqual(dola.metadata["nominal_feature_count"], 36)
        self.assertNotIn("dola_kl_proxy.resid.layer_35", dola.feature_names)

    def test_geometry_overflow_is_explicitly_audited_and_core_remains_valid(self):
        sidecar = copy.deepcopy(self.sidecar)
        sidecar["0:0"]["cov_eigs"][0, 0] = np.inf
        with self.assertRaisesRegex(ValueError, "cov_eigs contains non-finite"):
            validate_and_join(
                self.raw, sidecar, cell_id="strict-geometry",
                expected_model="synthetic/Llama-3.1-8B-Instruct",
                expected_hidden_size=16, expected_projection_dim=8,
                expected_covariance_rank=4,
            )
        cell, audit = validate_and_join(
            self.raw, sidecar, cell_id="blocked-geometry",
            expected_model="synthetic/Llama-3.1-8B-Instruct",
            expected_hidden_size=16, expected_projection_dim=8,
            expected_covariance_rank=4, require_geometry_finite=False,
        )
        self.assertEqual(audit["nonfinite_geometry_counts"]["cov_eigs"], 1)
        self.assertFalse(audit["geometry_tensors_finite"])
        self.assertTrue(np.isfinite(extract_resid_core(cell).values).all())

    def test_haloscope_direct_proxy_is_rotation_invariant(self):
        matrix = extract_haloscope_projection(self.cell)
        score, diagnostic = fit_haloscope_direct_proxy(matrix, k=4)
        rng = np.random.default_rng(811)
        q, _ = np.linalg.qr(rng.normal(size=(matrix.n_features, matrix.n_features)))
        rotated = replace(matrix, values=matrix.values @ q)
        rotated_score, _ = fit_haloscope_direct_proxy(rotated, k=4)
        np.testing.assert_allclose(score, rotated_score, rtol=2e-10, atol=2e-10)
        self.assertFalse(diagnostic["labels_seen_during_fit"])

    def test_geometry_is_invariant_to_common_orthogonal_rotation(self):
        original = extract_geometry(self.cell, SPACED_LAYERS)
        rng = np.random.default_rng(103)
        q, _ = np.linalg.qr(rng.normal(size=(self.cell.projection_dim, self.cell.projection_dim)))
        rotated_records = []
        for record in self.cell.records:
            copied = dict(record)
            copied["hid_proj"] = np.asarray(record["hid_proj"]) @ q
            rotated_records.append(copied)
        rotated_cell = replace(self.cell, records=tuple(rotated_records))
        rotated = extract_geometry(rotated_cell, SPACED_LAYERS)
        self.assertEqual(original.feature_names, rotated.feature_names)
        np.testing.assert_allclose(original.values, rotated.values, rtol=2e-6, atol=2e-7)

    def test_unlabeled_token_length_residualization(self):
        rng = np.random.default_rng(7)
        lengths = np.arange(4, 124, 4)
        log_length = np.log1p(lengths)
        values = np.column_stack([
            3.0 * log_length + rng.normal(0, 0.05, len(lengths)),
            -1.7 * log_length + rng.normal(0, 0.05, len(lengths)),
            0.8 * log_length + rng.normal(0, 0.05, len(lengths)),
        ])
        anchor = 2.2 * log_length + rng.normal(0, 0.05, len(lengths))
        matrix = FeatureMatrix(
            values=values,
            feature_names=("a", "b", "c"),
            risk_anchor=anchor,
            groups=("g0", "g1", "g2"),
            protocol_signature="length-fixture",
        )
        residual = residualize_token_length(matrix, lengths)
        for column in range(residual.n_features):
            self.assertLess(abs(np.corrcoef(residual.values[:, column], log_length)[0, 1]), 1e-10)
        self.assertLess(abs(np.corrcoef(residual.risk_anchor, log_length)[0, 1]), 1e-10)
        self.assertFalse(np.allclose(matrix.values, residual.values))

    def test_controls_and_three_core_solvers_are_finite(self):
        matrix = extract_resid_core(self.cell)
        controls, control_diag = fit_controls(matrix, n_gen_tokens=self.cell.n_gen_tokens)
        self.assertEqual(
            set(controls), {"final_layer_nll", "token_length", "equal_mean", "pc1"}
        )
        self.assertFalse(control_diag["labels_seen_during_fit"])
        scores, diagnostic = fit_core_spectral(
            matrix,
            dufs_gates=np.ones(matrix.n_features),
        )
        self.assertEqual(set(scores), {"upcr", "iu_pcr", "dufs_liu_pcr"})
        self.assertTrue(all(np.isfinite(score).all() for score in scores.values()))
        self.assertTrue(diagnostic["fits"]["dufs_liu_pcr"]["lambda_zero_exact"])

    def test_lambda_zero_is_exact_iu(self):
        matrix = extract_resid_core(self.cell, SPACED_LAYERS)
        scores, diagnostic = fit_core_spectral(
            matrix,
            methods=("iu_pcr", "dufs_liu_pcr"),
            lambda_=0.0,
            dufs_gates=np.ones(matrix.n_features),
        )
        np.testing.assert_array_equal(scores["iu_pcr"], scores["dufs_liu_pcr"])
        self.assertTrue(diagnostic["fits"]["dufs_liu_pcr"]["lambda_zero_exact"])

    def test_registered_core_path_delegates_to_canonical_fit(self):
        matrix = extract_resid_core(self.cell, SPACED_LAYERS)
        canonical = {
            "deployed_upcr": np.linspace(0.0, 1.0, matrix.n_samples),
            "iu_pcr": np.linspace(1.0, 2.0, matrix.n_samples),
            "dufs_liu_pcr": np.linspace(2.0, 3.0, matrix.n_samples),
        }
        canonical_diag = {
            "labels_seen_during_fit": False,
            "lambda_zero_exact": True,
        }
        with patch(
            "spectral_utils.whitebox_layer_fusion.canonical_fit_spectral_scores",
            return_value=(canonical, canonical_diag),
        ) as fit:
            scores, diagnostic = fit_core_spectral(matrix)
        fit.assert_called_once()
        np.testing.assert_array_equal(scores["upcr"], canonical["deployed_upcr"])
        self.assertEqual(
            diagnostic["canonical_implementation"],
            "spectral_utils.paper_benchmark_suite.fit_spectral_scores",
        )
        self.assertFalse(diagnostic["labels_seen_during_fit"])

    def test_fixed_hierarchy_supports_all_registered_solvers(self):
        matrix = extract_resid_core(self.cell)
        for solver in ("upcr", "iu_pcr"):
            score, diagnostic = fit_hierarchical(matrix, solver)
            self.assertEqual(score.shape, (matrix.n_samples,))
            self.assertTrue(np.isfinite(score).all())
            self.assertEqual(diagnostic["n_groups"], 4)
            self.assertEqual(len(diagnostic["folded_feature_weights"]), matrix.n_features)
            self.assertFalse(diagnostic["labels_seen_during_fit"])
        lens = extract_lens96(self.cell)
        score, diagnostic = fit_hierarchical(lens, "iu_pcr")
        self.assertTrue(np.isfinite(score).all())
        self.assertEqual(diagnostic["n_groups"], 12)
        # DUFS-LIU uses the same code path at both levels.  A one-epoch smoke
        # keeps this mechanical gate cheap while production stays frozen at 80.
        def unit_gates(features, **_kwargs):
            return np.ones(features.shape[0]), {"source": "unit-test"}

        with patch("spectral_utils.whitebox_layer_fusion.dufs_soft_gates", unit_gates):
            score, diagnostic = fit_hierarchical(matrix, "dufs_liu_pcr", dufs_epochs=1)
        self.assertTrue(np.isfinite(score).all())
        self.assertEqual(diagnostic["solver"], "dufs_liu_pcr")

    def test_dependency_helpers_use_registered_names(self):
        matrix = extract_resid_core(self.cell)
        scores, diagnostic = fit_dependency_methods(
            matrix,
            methods=("lsml_continuous", "clustered_upcr"),
            lsml_groups=matrix.groups,
        )
        self.assertEqual(set(scores), {"lsml_continuous", "clustered_upcr"})
        self.assertTrue(all(np.isfinite(score).all() for score in scores.values()))
        self.assertTrue(
            diagnostic["fits"]["clustered_upcr"]["identifiability"]["ok"]
        )

    def test_neutral_residual_mode_reconstructs_iu_and_is_label_free(self):
        matrix = extract_resid_core(self.cell)
        spaces = []
        for index, scale in enumerate((0.0, 0.015, -0.012)):
            layer_wave = np.sin(np.arange(matrix.n_features, dtype=float))[None, :]
            row_wave = np.linspace(-1.0, 1.0, matrix.n_samples)[:, None]
            shifted = replace(
                matrix,
                values=matrix.values + scale * row_wave * layer_wave,
                protocol_signature=f"nrm-source-{index}",
            )
            space = fit_group_contribution_space(shifted)
            np.testing.assert_allclose(
                space.contributions.sum(axis=1),
                space.baseline_score,
                rtol=1e-12,
                atol=1e-12,
            )
            self.assertEqual(space.families, tuple(dict.fromkeys(matrix.groups)))
            self.assertLess(space.diagnostics["reconstruction_error"], 1e-12)
            spaces.append(space)

        calibration = fit_neutral_residual_calibration(
            spaces[:2], source_ids=("source-a", "source-b")
        )
        fitted = apply_neutral_residual_calibration(spaces[2], calibration)
        self.assertEqual(fitted.score.shape, (matrix.n_samples,))
        self.assertTrue(np.isfinite(fitted.score).all())
        self.assertFalse(fitted.diagnostics["labels_seen_during_fit"])
        self.assertEqual(calibration.source_ids, ("source-a", "source-b"))
        self.assertAlmostEqual(np.linalg.norm(calibration.direction), 1.0, places=12)
        self.assertGreaterEqual(float(np.sum(calibration.direction)), 0.0)
        self.assertLess(
            abs(fitted.diagnostics["baseline_correction_covariance"]), 1e-10
        )

        iu_score, _ = fit_core_spectral(
            matrix, methods=("iu_pcr",), dufs_gates=np.ones(matrix.n_features)
        )
        direct = fit_group_contribution_space(matrix)
        np.testing.assert_allclose(
            direct.baseline_score, iu_score["iu_pcr"], rtol=1e-12, atol=1e-12
        )

    def test_protocol_and_no_label_signatures(self):
        assert_no_label_fitting_signatures()
        fitting = (
            fit_controls,
            fit_core_spectral,
            fit_dependency_methods,
            fit_hierarchical,
            fit_group_contribution_space,
            fit_neutral_residual_calibration,
        )
        forbidden = {"label", "labels", "y", "target", "targets"}
        for function in fitting:
            self.assertTrue(forbidden.isdisjoint(inspect.signature(function).parameters))
        matrix = extract_resid_core(self.cell, SPACED_LAYERS)
        assert_same_protocol(matrix, extract_lens96(self.cell))
        incompatible = replace(matrix, protocol_signature="different-cohort")
        with self.assertRaisesRegex(ValueError, "protocol signature mismatch"):
            assert_same_protocol(matrix, incompatible)
        with self.assertRaisesRegex(ValueError, "metadata must not contain label"):
            FeatureMatrix(
                values=matrix.values,
                feature_names=matrix.feature_names,
                risk_anchor=matrix.risk_anchor,
                groups=matrix.groups,
                protocol_signature=matrix.protocol_signature,
                metadata={"labels": [0] * matrix.n_samples},
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
    extract_dola_kl_proxy,
    extract_haloscope_projection,
    fit_haloscope_direct_proxy,
    fixed_bands,
    late_layers,
    spaced_layers,
