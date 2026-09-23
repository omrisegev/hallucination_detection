"""Contract and collector regression tests, including real CPU forward passes."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.external_generalization.contracts import Answer, tokenize_answer, adapt_hard2verify, overlap_manifest
from spectral_utils.external_generalization.artifacts import RecordStore, seal_predictions, require_seal
from spectral_utils.external_generalization.budget import smoke_indices, authorize
from spectral_utils.external_generalization.evaluation import confusion, metric, paired_bootstrap
from spectral_utils.external_generalization.fusion import local_scores, decisions


class CharTokenizer:
    chat_template = "synthetic_char_fixture"
    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 60 + 1 for c in text],
                "offset_mapping": [(i, i+1) for i in range(len(text))]}
    def apply_chat_template(self, messages, **kwargs):
        return "User:" + messages[0]["content"] + "Assistant:"


class Contracts(unittest.TestCase):
    def test_labels_do_not_change_scoring_input(self):
        row = dict(unique_id="x", question="Q", model_response_by_step=["a", "b"], human_labels=[1, 0])
        a, _ = adapt_hard2verify([row])
        row["human_labels"] = [0, 1]
        b, _ = adapt_hard2verify([row])
        self.assertEqual(a, b)
        with self.assertRaises(ValueError):
            adapt_hard2verify([row, row])

    def test_spans_empty_steps_and_no_truncation(self):
        a = Answer("x", "socratic", "s", "q", "q", ("a", "", "bc"))
        item = tokenize_answer(a, CharTokenizer(), [1], 100)
        self.assertEqual(item["step_char_spans"], [(0, 1), (3, 3), (5, 7)])
        self.assertEqual(item["step_token_spans"], [(0, 1), (3, 3), (5, 7)])
        with self.assertRaises(ValueError):
            tokenize_answer(a, CharTokenizer(), [1], 4)

    def test_overlap_transitive_source_group(self):
        a = Answer("a", "one", "1", "q1", "q", ("s",))
        b = Answer("b", "one", "1", "q2", "q3", ("s",))
        c = Answer("c", "two", "x", "q2", "q2", ("s",))
        result = overlap_manifest({"one": [a,b], "two": [c]})
        self.assertEqual(len(set(result["groups"].values())), 1)

    def test_resume_lock_and_changed_identity(self):
        with tempfile.TemporaryDirectory() as t:
            with RecordStore(t, {"config": 1}) as store:
                store.put("a", {"value": [1,2]})
                with self.assertRaises(FileExistsError):
                    with RecordStore(t, {"config": 1}):
                        pass
            with RecordStore(t, {"config": 1}) as store:
                self.assertEqual(store.get("a"), {"value": [1,2]})
                with self.assertRaises(ValueError):
                    store.put("a", {"value": [2,1]})
            with self.assertRaises(ValueError):
                with RecordStore(t, {"config": 2}):
                    pass

    def test_seal_and_budget_gate(self):
        predictions = {"x": {"arms": {"f": [1]}}}
        with tempfile.TemporaryDirectory() as t:
            seal = seal_predictions(predictions, ["x"], ["f"], Path(t)/"seal.json", "h")
            require_seal(predictions, seal, "h")
            predictions["x"]["arms"]["f"] = [0]
            with self.assertRaises(ValueError):
                require_seal(predictions, seal, "h")
        with self.assertRaises(ValueError):
            authorize("full", preflight={"verdict":"PASS", "protocol_hash":"h", "session_id":"s"}, protocol_hash="h")

    def test_official_metrics_and_parser_penalty(self):
        c = confusion([1,1,1,0,0], [1,0,1,0,-2])
        np.testing.assert_array_equal(c, [2,1,1,1])
        self.assertAlmostEqual(float(metric(c, "hard2verify")), 4/7)
        self.assertAlmostEqual(float(metric(c, "socratic")), (4/6+2/4)/2)
        res = paired_bootstrap([c,c], [c,c], ["q1","q2"], "socratic", 12, draws=1000)
        self.assertEqual(res["ci_bonferroni"], [0,0])

    def test_smoke_includes_longest_and_quartiles(self):
        lengths = list(range(1,101))
        indices = smoke_indices(lengths, list(map(str,lengths)))
        self.assertEqual(len(indices), 12)
        self.assertIn(99, indices)
        self.assertEqual({i//25 for i in indices}, {0,1,2,3})

    def test_deterministic_matched_fallback_and_threshold_tie(self):
        x = np.zeros((8,11));x[:,2] = np.arange(8)
        a, d = local_scores(x, [(0,4),(4,8)])
        self.assertFalse(d["native"])
        for v in a.values():
            np.testing.assert_array_equal(v, [-1,1])
        self.assertEqual(decisions({"x":[0,1]}, {"x":1}), {"x":[True,False]})


def cpu_collector_smoke(out):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    spec = importlib.util.spec_from_file_location("external_driver", ROOT/"cluster/run_external_telemetry.py")
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=64, n_positions=512, n_embd=16, n_layer=1,
                                      n_head=2, bos_token_id=1, eos_token_id=2)).eval()
    answers = [Answer(str(i), "fixture", str(i), "Q", "Q", ("abc", "defg")) for i in range(5)]
    items = driver.make_items(answers, CharTokenizer(), 512)
    gate = driver.alignment_gate(model, items[0])
    with RecordStore(out, {"cpu_smoke":1}) as store:
        complete, rows = driver.run(items, store, lambda i:driver.collect_quantities(model,i), range(5))
        assert complete and len(rows)==5
        r = store.get("0")["telemetry"]
        assert np.asarray(r["top_k_logprobs"]["ids"]).shape == (len(items[0]["gen_ids"]),50)
        np.testing.assert_allclose(np.negative(r["actual_token_logprobs"]), r["token_spilled_energies"])
        # Interruption/resume must neither rescore completed rows nor lose IDs.
        driver.STOP = True
        done, _ = driver.run(items, store, lambda _:None, range(5))
        assert not done
        driver.STOP = False
        done, resumed = driver.run(items, store, lambda _: (_ for _ in ()).throw(AssertionError("rescored")), range(5))
        assert done and len(resumed)==5
    print(json.dumps({"cpu_driver_smoke":"PASS", "answers":5, "alignment_gate":gate}))


if __name__ == "__main__":
    if len(sys.argv)>1 and sys.argv[1]=="--cpu-smoke":
        cpu_collector_smoke(Path(sys.argv[2]))
    else:
        unittest.main()
