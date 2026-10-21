"""Standalone-model prompts and empty-step spans of cluster/run_prmbench_prm_telemetry.py (--mode instruct).
Plan: docs/experiments/PRM_STANDALONE_TELEMETRY_2026-10-22.md. Real tokenizers are smoke-tested on AIRCC (/preflight)."""
import importlib.util
import os
import sys
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'cluster'))
pytest.importorskip('torch')
spec = importlib.util.spec_from_file_location('tel', os.path.join(ROOT, 'cluster/run_prmbench_prm_telemetry.py'))
D = importlib.util.module_from_spec(spec); spec.loader.exec_module(D)


class FakeTok:
    """Characters -> ids; BOS = 1. chat_template marks the chat-capable case."""
    def __init__(self, bos=1, chat=True, double_bos=False):
        self.bos_token_id, self.chat_template, self.double = bos, ('x' if chat else None), double_bos

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        return '\x01' * (2 if self.double else 1) + '|'.join(m['role'] + ':' + m['content'] for m in msgs) + '|assistant:'

    def __call__(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        if add_special_tokens and self.bos_token_id is not None: ids = [self.bos_token_id] + ids
        return type('E', (), {'input_ids': ids})()


def test_chat_keeps_the_template_bos_once():
    ids = D.standalone_prompt_ids(FakeTok(), 'chat', 'q?')
    assert ids[0] == 1 and ids[1] != 1 and 'system' not in ''.join(map(chr, ids[1:]))


def test_plain_adds_bos_and_separator():
    ids = D.standalone_prompt_ids(FakeTok(chat=False), 'plain', 'q?')
    assert ids[0] == 1 and ''.join(map(chr, ids[1:])) == 'q?\n\n'


def test_qwen_style_is_the_historical_call():
    tok = FakeTok(bos=None)
    ids = D.standalone_prompt_ids(tok, 'qwen_cot_chat', 'q?')
    prompt = tok.apply_chat_template([{'role': 'system', 'content': D.PRM_SYSTEM_PROMPT}, {'role': 'user', 'content': 'q?'}])
    assert ids == tok(prompt).input_ids


def test_loud_failures():
    with pytest.raises(ValueError):
        D.standalone_prompt_ids(FakeTok(chat=False), 'chat', 'q?')        # no chat template
    with pytest.raises(ValueError):
        D.standalone_prompt_ids(FakeTok(double_bos=True), 'chat', 'q?')   # template + tokenizer would give two BOS
    with pytest.raises(ValueError):
        D.standalone_prompt_ids(FakeTok(), 'nope', 'q?')


def test_fill_empty_spans():
    assert D.fill_empty_spans([(0, 3), None, (5, 9)]) == [(0, 3), (3, 3), (5, 9)]
    assert D.fill_empty_spans([None, (0, 2)]) == [(0, 0), (0, 2)]
    assert D.fill_empty_spans([(0, 2), (2, 4)]) == [(0, 2), (2, 4)]
