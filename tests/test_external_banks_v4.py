"""Unit tests for spectral_utils.external_banks_v4 on a tiny synthetic record (no data files)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from spectral_utils import external_banks_v4 as E

MAIN = Path(__file__).resolve().parents[3]


def synthetic_row(spans=((0, 1), (1, 20), (20, 20), (20, 48)), tokens=48, vocab=120, seed=0):
    rng = np.random.default_rng(seed)
    logits = 2.0*rng.normal(size=(tokens, vocab))
    logits[:, 15:25] += 1.0                       # digits present in most top-50 lists
    logp = logits-np.log(np.exp(logits).sum(axis=1, keepdims=True))
    order = np.argsort(-logp, axis=1)[:, :50]
    top = np.take_along_axis(logp, order, axis=1)
    gen = order[np.arange(tokens), rng.integers(0, 5, tokens)]
    p = np.exp(top[:, :15]); q = p/p.sum(axis=1, keepdims=True)
    return {'gen_token_ids': gen.tolist(),
            'token_entropies': (-(q*np.log(q)).sum(axis=1)/np.log(15)).tolist(),
            'token_spilled_energies': (-logp[np.arange(tokens), gen]).tolist(),
            'token_logsumexp': (10+rng.normal(size=tokens)).tolist(),
            'top_k_logprobs': {'ids': order.astype(np.int32), 'logprobs': top.astype(np.float32)},
            'step_token_spans': [list(s) for s in spans]}


def nonempty(row):
    spans = np.asarray(row['step_token_spans'])
    return spans[spans[:, 1] > spans[:, 0]]


def test_shapes_and_names():
    row = synthetic_row()
    spans = nonempty(row)
    matrix, names = E.new_step_features(row, spans)
    assert names == E.NEW_NAMES and matrix.shape == (3, 5) and np.isfinite(matrix).all()
    values, active = E.digit_step_features(row, spans)
    assert values.shape == active.shape == (3, 3) and active.dtype == bool
    assert len(E.DIGIT_NAMES) == 3 and E.DIGIT_IDS == tuple(range(15, 25))


def test_empty_steps_are_rejected():
    row = synthetic_row()
    with pytest.raises(ValueError, match='empty'):
        E.new_step_features(row)                  # row spans contain the empty step (20, 20)
    with pytest.raises(ValueError, match='empty'):
        E.digit_step_features(row)
    with pytest.raises(ValueError):
        E.new_step_features(row, np.array([[0, 60]]))   # beyond the token trace


def test_digit_mask():
    row = synthetic_row()
    values, active = E.digit_step_features(row, nonempty(row))
    # Step 0 holds only token 0, where the prefix innovation is undefined.
    assert active[0].tolist() == [True, True, False] and values[0, 2] == 0.
    assert active[1:].all()
    # Without digit ids in the top-K lists every digit channel is zero.
    shifted = synthetic_row()
    ids = shifted['top_k_logprobs']['ids'].copy()
    ids[(ids >= 15) & (ids < 25)] += 1000
    shifted['top_k_logprobs'] = {'ids': ids, 'logprobs': shifted['top_k_logprobs']['logprobs']}
    zero, _ = E.digit_step_features(shifted, nonempty(shifted))
    assert np.all(zero == 0.)


def test_realized_z_is_despiked_and_standardized():
    row = synthetic_row(spans=((0, 8), (8, 16), (16, 30), (30, 48)))
    z = E.realized_z(row, nonempty(row))
    assert abs(z.mean()) < 1e-12 and abs(z.std()-1) < 1e-12
    single = E.realized_z(row, np.array([[0, 48]]))
    assert single.tolist() == [0.]


def test_realized_drv_matches_bank_column():
    row = synthetic_row()
    spans = nonempty(row)
    bank = E.validate_telemetry(row)
    expected = E.derivative_step_readout(bank, spans)[:, list(E.BANK11_NAMES).index('chosen_surprisal')]
    assert np.array_equal(E.realized_drv(row, spans), expected)


def _load(relpath, name):
    path = MAIN/relpath
    if not path.exists():
        pytest.skip('source module not present: '+relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_vendored_derivative_matches_source():
    source = _load(E.VENDORED_SOURCES['ema, derivative_step_readout']['path'], 'derivative_source')
    rng = np.random.default_rng(3)
    x = rng.normal(size=(90, 4)); x[:, 2] = 1.0   # one constant (dead) channel
    spans = np.array([[0, 1], [1, 40], [40, 90]])
    assert np.array_equal(E.derivative_step_readout(x, spans), source.derivative_step_readout(x, spans))
