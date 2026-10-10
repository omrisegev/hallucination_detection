"""CPU test: the teacher-forced prefix carries Qwen3's empty think block (2026-10-10 fix).

Renders the pinned official Qwen3-8B chat template (revision b968826d, snapshot in
results/automatic_group_free_phase_a6_tokenizers_v2/evidence/official/qwen3-8b.json) with
jinja2, so it needs no model download.
    python scripts/test_teacher_forced_think_prefix.py
"""
import json
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from spectral_utils.processbench import (  # noqa: E402
    EMPTY_THINK_BLOCK, NO_THINK_SUFFIX, strip_generated_think_block, teacher_forced_prompt,
)

SNAPSHOT = os.path.join(ROOT, "results/automatic_group_free_phase_a6_tokenizers_v2/evidence/"
                              "official/qwen3-8b.json")
THINK, END_THINK, NL2 = 151667, 151668, 271


class _Ids:
    def __init__(self, ids):
        self.input_ids = ids


class StubQwen3Tokenizer:
    """Just enough of a HF tokenizer: the real chat template, and the think-block token ids."""

    def __init__(self, template):
        import jinja2
        self.chat_template = template
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(ValueError(m))
        self._tpl = env.from_string(template)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        if not isinstance(messages[-1]["content"], str):
            raise TypeError("string content only")
        return self._tpl.render(messages=messages, add_generation_prompt=add_generation_prompt, **kw)

    def __call__(self, text, add_special_tokens=True):
        assert text == EMPTY_THINK_BLOCK
        return _Ids([THINK, NL2, END_THINK, NL2])


class TestThinkPrefix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = StubQwen3Tokenizer(json.load(open(SNAPSHOT))["config"]["tokenizer_config"]["chat_template"])

    def test_default_prefix_ends_with_empty_block(self):
        p = teacher_forced_prompt(self.tok, "Solve 1+1." + NO_THINK_SUFFIX)
        self.assertTrue(p.endswith("<|im_start|>assistant\n" + EMPTY_THINK_BLOCK), repr(p[-60:]))
        self.assertIn("/no_think<|im_end|>", p)

    def test_legacy_prefix_is_the_old_one(self):
        p = teacher_forced_prompt(self.tok, "Solve 1+1." + NO_THINK_SUFFIX, think_prefix="none")
        self.assertTrue(p.endswith("<|im_start|>assistant\n"), repr(p[-40:]))

    def test_thinking_arm_has_no_block(self):
        p = teacher_forced_prompt(self.tok, "Solve 1+1.", thinking_suffix="")
        self.assertTrue(p.endswith("<|im_start|>assistant\n"))

    def test_strip_generated_block(self):
        ids, vals, n = strip_generated_think_block(self.tok, [THINK, NL2, END_THINK, NL2, 7, 8],
                                                   [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual((ids, list(vals), n), ([7, 8], [0.5, 0.6], 4))
        ids, vals, n = strip_generated_think_block(self.tok, [7, 8], [0.5, 0.6])
        self.assertEqual((ids, n), ([7, 8], 0))

    def test_bad_value_raises(self):
        with self.assertRaises(ValueError):
            teacher_forced_prompt(self.tok, "x", think_prefix="yes")


if __name__ == "__main__":
    unittest.main()
