import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from spectral_utils.advisor_controls import metrics, top_flags
from spectral_utils.error_count_evaluation import (ARMS, ALPHA, CONTROLS, SourceBootstrap,
    combined_success, flags_from_shared_rank, partial_spearman, sha256,
    summarize, validate_prediction_seal, evaluate_package, load_predictions, build_report)


class ErrorCountEvaluationTests(unittest.TestCase):
    def test_topk_stable_ties_and_failed_count(self):
        score = np.array([2., 2., 1., 9., 4.])
        offsets = np.array([0, 3, 5])
        rank = np.array([0, 1, 2, 3, 4])
        flag, covered = flags_from_shared_rank(rank, offsets, np.array([1, -1]))
        np.testing.assert_array_equal(flag, [True, False, False, False, False])
        np.testing.assert_array_equal(covered, [True, False])
        full, _ = flags_from_shared_rank(rank, offsets, np.array([0, 2]))
        np.testing.assert_array_equal(full, top_flags(score, offsets, [0, 2]))

    def test_pooled_metrics_and_ms_valid_f1(self):
        y = np.array([1, 0, 0, 0], bool)
        flags = np.array([1, 1, 0, 0], bool)
        m = metrics(y, flags)
        self.assertAlmostEqual(m["prm_score"], ((2/3)+(4/5))/2)
        bs = SourceBootstrap(["a", "b"], weights=np.array([[1, 1], [2, 0], [0, 2]]))
        result, _ = summarize(y, flags, [0, 2, 4], np.array([2, 0]), np.array([1, 1], bool), np.array([1, 1], bool), bs)
        self.assertAlmostEqual(result["count_mae"], .5)
        self.assertAlmostEqual(result["count_bias"], .5)
        ms, _ = summarize(y, flags, [0, 2, 4], np.array([2, 0]), np.array([1, 1], bool), np.array([1, 1], bool), bs, valid_only=True)
        self.assertAlmostEqual(ms["prm_score"], 4/5)

    def test_whole_group_bootstrap_matches_explicit_resampling(self):
        groups = np.array(["a", "a", "b"])
        weights = np.array([[1, 1], [2, 0], [0, 2]])
        bootstrap = SourceBootstrap(groups, weights=weights)
        values = np.array([1., 5., 9.])
        result = bootstrap.means(values, np.ones(3, bool))[:, 0]
        expected = [np.repeat(values, w[bootstrap.inverse]).mean() for w in weights]
        np.testing.assert_allclose(result, expected)

    def test_failure_is_missing_and_not_zero_count(self):
        y = np.array([1, 0, 1, 0], bool)
        bs = SourceBootstrap(["a", "b"], weights=np.array([[1, 1], [2, 0], [0, 2]]))
        result, replicas = summarize(y, np.array([1, 0, 0, 0], bool), [0, 2, 4], np.array([1, -1]), np.array([1, 0], bool), np.ones(2, bool), bs)
        self.assertEqual(result["n_answers"], 1)
        self.assertEqual(result["n_failed_answers"], 1)
        self.assertEqual(result["count_mae"], 0)
        self.assertTrue(np.isnan(replicas["count_mae"][2]))

    def test_length_partial_constant_and_rank_equivalence(self):
        n = np.arange(1, 9)
        self.assertTrue(np.isnan(partial_spearman(n, n, n)))
        self.assertTrue(np.isnan(partial_spearman(np.ones(8), n, n)))
        truth = np.array([1, 3, 2, 5, 4, 7, 6, 8])
        self.assertAlmostEqual(partial_spearman(truth, truth, n), 1.)
        bs = SourceBootstrap(["a", "b", "c", "d"], weights=np.array([[1, 1, 1, 1]]))
        value = bs.correlation(truth[:4], truth[:4], n[:4], np.ones(4, bool))[0]
        self.assertAlmostEqual(value, partial_spearman(truth[:4], truth[:4], n[:4]))

    def test_seal_missing_changed_and_unlisted_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"predictions").mkdir()
            files = {}
            for bank in ("N16", "N56"):
                path = root/f"predictions/{bank}.npz"
                np.savez(path, sentinel=np.array([1]))
                files[f"predictions/{bank}.npz"] = {"sha256": sha256(path)}
            seal = {"status": "FROZEN_BEFORE_EVALUATION", "files": files}
            (root/"PREDICTION_SEAL.json").write_text(json.dumps(seal))
            self.assertEqual(len(validate_prediction_seal(root)["verified_files"]), 2)
            (root/"predictions/N16.npz").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                validate_prediction_seal(root)

    def test_combined_gate_rejects_zero_and_incomplete_arms(self):
        rows = []
        for baseline, metric, low, high in [("Q80", "count_mae", -.2, -.1), ("LEGACY_MOM", "count_mae", -.3, -.1),
                                          ("Q80", "prm_score", .001, .02), ("Q80", "error_recall", -.009, -.001)]:
            rows.append(dict(bank="N16", method="IU_PCR", arm=ARMS[0], population="ALL_NONCONTROL", baseline=baseline,
                             metric=metric, simultaneous_lower=low, simultaneous_upper=high, full_coverage=True))
        success = combined_success(rows, {ARMS[0]: True})
        self.assertTrue(success["arms"][0]["success"])
        self.assertEqual(success["M"], 7)
        self.assertEqual(success["one_sided_alpha"], ALPHA)
        self.assertFalse(combined_success(rows, {ARMS[0]: False})["any_success"])
        # An otherwise-passing complete candidate cannot claim a comparison to
        # a baseline that is only defined on its common-covered subset.
        rows[1]["full_coverage"] = False
        self.assertFalse(combined_success(rows, {ARMS[0]: True})["any_success"])
        rows[1]["full_coverage"] = True
        self.assertFalse(combined_success(rows, {ARMS[0]: True},
                                         promotable={ARMS[0]: True, "Q80": True, "LEGACY_MOM": False})["any_success"])
        rows[-1]["simultaneous_lower"] = -.02
        self.assertFalse(combined_success(rows, {ARMS[0]: True})["any_success"])

    def test_completely_failed_arm_has_undefined_quality(self):
        y = np.array([1, 0], bool)
        bs = SourceBootstrap(["a"], weights=np.array([[1], [1]]))
        result, samples = summarize(y, np.zeros(2, bool), [0, 2], np.array([-1]), np.array([0], bool), np.array([1], bool), bs)
        self.assertEqual(result["n_answers"], 0)
        self.assertTrue(np.isnan(result["prm_score"]))
        self.assertTrue(np.isnan(samples["prm_score"]).all())

    def test_sealed_end_to_end_method_specific_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            package, frozen = base/"package", base/"frozen"
            (package/"predictions").mkdir(parents=True)
            (frozen/"inputs").mkdir(parents=True)
            idx = np.array([f"u{i}" for i in range(5)])
            offsets = np.arange(0, 11, 2)
            score = np.tile([1., 0.], 5)
            prediction = {"idx": idx, "offsets": offsets, "IU_PCR_score": score, "LSML_CONT_score": score,
                          "rank_IU_PCR": np.arange(10), "rank_LSML_CONT": np.arange(10), "group": idx, "fold": np.arange(5)}
            for arm in ARMS + CONTROLS:
                prediction["count__"+arm] = np.zeros(5, dtype=int)
                for method in ("IU_PCR", "LSML_CONT"):
                    prediction[f"count__{method}__{arm}"] = np.zeros(5, dtype=int)
            prediction["count__LSML_CONT__Q80"] = np.ones(5, dtype=int)
            files = {}
            for bank in ("N16", "N56"):
                path = package/f"predictions/{bank}.npz"
                np.savez(path, **prediction)
                files[str(path.relative_to(package))] = {"sha256": sha256(path)}
                for fold in range(5):
                    model = package/f"checkpoints/{bank}_fold{fold}/models.json"
                    model.parent.mkdir(parents=True)
                    model.write_text(json.dumps({"status": {arm: {"status": "PASS"} for arm in ARMS + CONTROLS}}))
                    files[str(model.relative_to(package))] = {"sha256": sha256(model)}
            np.savez(frozen/"inputs/population.npz", idx=idx, offsets=offsets,
                     y=np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], bool),
                     classification=np.array(["redundency", "circular", "multi_solutions", "correct", "correct"]),
                     group=idx, fold=np.arange(5))
            label_path = frozen/"inputs/population.npz"
            (package/"INPUT_FREEZE.json").write_text(json.dumps({"inputs": {str(label_path): {"sha256": sha256(label_path)}}}))
            files["INPUT_FREEZE.json"] = {"sha256": sha256(package/"INPUT_FREEZE.json")}
            (package/"PREDICTION_SEAL.json").write_text(json.dumps({"status": "FROZEN_BEFORE_EVALUATION", "files": files}))
            parsed = load_predictions(package/"predictions/N16.npz", "N16")
            self.assertEqual(set(parsed.counts), set(ARMS + CONTROLS))
            self.assertEqual(parsed.method_counts["LSML_CONT"]["Q80"].sum(), 5)
            status = evaluate_package(package, frozen, draws=10)
            self.assertEqual(status["n_answers"], 5)
            data = json.loads((package/"evaluation/RESULTS.json").read_text())
            q80 = [row for row in data["metrics"] if row["bank"] == "N16" and row["arm"] == "Q80" and row["population"] == "ALL_NONCONTROL"]
            self.assertEqual(next(row for row in q80 if row["method"] == "IU_PCR")["count_mean"], 0)
            self.assertEqual(next(row for row in q80 if row["method"] == "LSML_CONT")["count_mean"], 1)
            self.assertFalse(data["success"]["any_success"])
            first = build_report(package)
            report_hash = sha256(first)
            figure_hash = sha256(package/"report/count_quality.svg")
            build_report(package)
            self.assertEqual(sha256(first), report_hash)
            self.assertEqual(sha256(package/"report/count_quality.svg"), figure_hash)
            # Changed labels with identical answer IDs still fail authentication.
            with np.load(label_path) as original:
                changed = {key: original[key].copy() for key in original.files}
            changed["y"][0] = False
            np.savez(label_path, **changed)
            with self.assertRaisesRegex(ValueError, "label source hash mismatch"):
                evaluate_package(package, frozen, draws=10)


if __name__ == "__main__":
    unittest.main()
