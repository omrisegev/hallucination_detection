"""Meaningful official-sentinel and paired-source-group metric checks."""
import ast
import itertools
from pathlib import Path
import unittest
import numpy as np
from spectral_utils.family_external_metrics import summary, bootstrap
from spectral_utils.external_generalization.evaluation import confusion


class FamilyExternalMetricsTests(unittest.TestCase):
    def test_prmscore_components(self):
        result = summary([12, 4, 3, 1], "socratic")
        self.assertAlmostEqual(result["f1_correct"], 24/29)
        self.assertAlmostEqual(result["f1_error"], 6/11)
        self.assertAlmostEqual(result["metric"], (24/29+6/11)/2)

    def test_hard2_uses_harmonic_recalls(self):
        result = summary([12, 4, 3, 1], "hard2verify")
        self.assertAlmostEqual(result["metric"], 2*(12/13)*(3/7)/((12/13)+(3/7)))
        self.assertNotAlmostEqual(result["metric"], summary([12,4,3,1], "socratic")["metric"])

    def test_official_zero_hit_sentinel(self):
        result = summary([0, 2, 0, 3], "socratic")
        self.assertEqual(result["f1_correct"], -1)
        self.assertEqual(result["f1_error"], -1)
        self.assertEqual(result["metric"], -1)

    def test_source_group_variants_resample_together(self):
        counts = {"arm": [[2,1,0,1], [3,0,1,2], [1,2,3,0]]}
        grouped = {"arm": [[5,1,1,3], [1,2,3,0]]}
        _, a, _ = bootstrap(counts, ["a","a","b"], "socratic", draws=256)
        _, b, _ = bootstrap(grouped, ["a","b"], "socratic", draws=256)
        np.testing.assert_array_equal(a,b)

    def test_identical_arms_use_identical_draws(self):
        c = [[2,1,3,1],[1,2,1,0]]
        _, samples, _ = bootstrap({"a":c,"b":c}, ["x","y"], "socratic", draws=256)
        np.testing.assert_array_equal(samples[:,0],samples[:,1])

    def test_exhaustive_pinned_official_components(self):
        root = Path(__file__).resolve().parents[1]
        path = root/"scratch/external_generalization_private/sources/prmeval_classified_task.py"
        if not path.exists():
            self.skipTest("private pinned evaluator unavailable")
        names = {"evaluate_function","eval_on_hallucination_step"}
        nodes = [n for n in ast.parse(path.read_text(encoding="utf8")).body
                 if isinstance(n,ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in nodes}, names)
        ns = {}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),"exec"),ns)
        for n in range(1,5):
            for y in itertools.product((0,1),repeat=n):
                for p in itertools.product((0,1),repeat=n):
                    result = summary(confusion(y,p),"socratic")
                    meta = [{"idx":"x","classification":"synthetic",
                             "error_steps":[i+1 for i,c in enumerate(y) if not c]}]
                    pred = [{"idx":"x","scores":{"step_level_validity_labels":p}}]
                    official = ns["evaluate_function"](pred,meta)["total_hallucination_results"]
                    for ours,theirs in (("f1_correct","f1"),("f1_error","negative_f1"),
                                        ("precision_correct","precision"),("precision_error","negative_precision"),
                                        ("recall_correct","recall"),("recall_error","negative_recall")):
                        self.assertAlmostEqual(result[ours],official[theirs],places=12)


if __name__ == "__main__":
    unittest.main()
