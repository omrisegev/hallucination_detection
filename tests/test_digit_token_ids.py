"""Tokenizer-aware digit ids (spectral_utils.digit_token_ids) and their use in external_banks_v4."""
import json
from pathlib import Path

import numpy as np
import pytest

from spectral_utils import digit_token_ids as DT
from spectral_utils import external_banks_v4 as E
from tests.test_external_banks_v4 import nonempty, synthetic_row

ROOT = Path(__file__).resolve().parents[1]
LOCAL = {
    'qwen2.5': ROOT / 'local_cache/qwen25_15b_tokenizer/tokenizer.json',
    'qwen3-8b': ROOT / 'results/automatic_group_free_phase_a6_tokenizers_v1/materialized/qwen3-8b/tokenizer.json',
    'llama3.1-8b': ROOT / 'results/automatic_group_free_phase_a6_tokenizers_v1/materialized/llama31-8b/tokenizer.json',
}


def write_tokenizer(path, vocab):
    path.write_text(json.dumps({'model': {'type': 'BPE' if isinstance(vocab, dict) else 'Unigram', 'vocab': vocab}}))
    return path


def sentencepiece_like_vocab():
    """Llama-2/Mistral layout: specials 0..2, byte-fallback <0x00>..<0xFF> at 3..258, digits much later."""
    vocab = {'<unk>': 0, '<s>': 1, '</s>': 2}
    vocab.update({f'<0x{b:02X}>': 3 + b for b in range(256)})
    vocab.update({f'w{i}': 259 + i for i in range(500)})
    vocab.update({c: 28734 + k for k, c in enumerate('0123456789')})
    return vocab


@pytest.mark.parametrize('name', sorted(LOCAL))
def test_local_qwen_and_llama3_tokenizers_are_15_24(name):
    if not LOCAL[name].is_file():
        pytest.skip(f'{LOCAL[name]} not present')
    ids, record = DT.resolve_digit_ids(LOCAL[name], expected=DT.QWEN_DIGIT_IDS)
    assert ids == E.DIGIT_IDS and record['matches_qwen_15_24']


def test_sentencepiece_layout_resolves_and_rejects_15_24(tmp_path):
    path = write_tokenizer(tmp_path / 'tokenizer.json', sentencepiece_like_vocab())
    ids, record = DT.resolve_digit_ids(path)
    assert ids == tuple(range(28734, 28744)) and not record['matches_qwen_15_24']
    assert len(record['tokenizer_sha256']) == 64
    with pytest.raises(ValueError, match='do not match'):
        DT.resolve_digit_ids(path, expected=E.DIGIT_IDS)


def test_unigram_list_vocab(tmp_path):
    pieces = [['<unk>', 0.0], ['a', -1.0]] + [[c, -2.0] for c in '0123456789']
    ids, _ = DT.resolve_digit_ids(write_tokenizer(tmp_path / 'tokenizer.json', pieces))
    assert ids == tuple(range(2, 12))


def test_missing_digit_fails_loudly(tmp_path):
    vocab = sentencepiece_like_vocab(); del vocab['7']
    with pytest.raises(ValueError, match="'7'"):
        DT.resolve_digit_ids(write_tokenizer(tmp_path / 'tokenizer.json', vocab))
    with pytest.raises(FileNotFoundError):
        DT.resolve_digit_ids(tmp_path / 'absent.json')


def test_digit_ids_follow_the_vocabulary():
    """Relabelling every token id leaves the digit features unchanged when the digit ids move with them."""
    row = synthetic_row(); spans = nonempty(row)
    base, base_active = E.digit_step_features(row, spans)
    moved = synthetic_row()
    ids = moved['top_k_logprobs']['ids'].astype(np.int64) + 28719          # 15..24 -> 28734..28743
    moved['top_k_logprobs'] = {'ids': ids, 'logprobs': moved['top_k_logprobs']['logprobs']}
    wrong, _ = E.digit_step_features(moved, spans)                           # default 15..24: digits never seen
    assert np.all(wrong == 0.)
    right, right_active = E.digit_step_features(moved, spans, digit_ids=range(28734, 28744))
    assert np.array_equal(right, base) and np.array_equal(right_active, base_active)
    assert np.array_equal(E.digit_step_features(row, spans, digit_ids=E.DIGIT_IDS)[0], base)


def test_constant_digit_channel_is_caught():
    names = ['a', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']
    x = np.random.default_rng(0).normal(size=(40, 4))
    DT.check_digit_channels_vary(x, names, E.DIGIT_NAMES)
    x[:, 2] = 0.
    with pytest.raises(ValueError, match='digit_spread'):
        DT.check_digit_channels_vary(x, names, E.DIGIT_NAMES)
