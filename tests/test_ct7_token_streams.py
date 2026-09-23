"""Item 3 (2026-09-23): token-level CT7 stream builders on synthetic top-50 rows."""
from __future__ import annotations

import re

import numpy as np

from lever_imports import ensure_spectral_package

ROOT = ensure_spectral_package()
from spectral_utils import ct7_token_streams as ts  # noqa: E402


def test_self_test():
    assert ts.self_test(seed=0)["steps"] == 6


def test_three_answer_alignment_and_shapes():
    rng = np.random.default_rng(1)
    rows = [ts.synthetic_row(rng, n, s) for n, s in ((40, 2), (95, 7), (23, 1))]
    offsets = np.concatenate([[0], np.cumsum([len(r["gen_token_ids"]) for r in rows])])
    mats, valids = [], []
    for r in rows:
        p = r["top_k_logprobs"]
        x, v = ts.answer_streams(p["logprobs"], p["ids"], r["gen_token_ids"], r["token_spilled_energies"])
        assert r["step_token_spans"][0, 0] == 0 and r["step_token_spans"][-1, 1] == len(x)
        mats.append(x); valids.append(v)
    tokens = np.vstack(mats); valid = np.vstack(valids)
    assert tokens.shape == (offsets[-1], 7) and valid.shape == tokens.shape
    for i, r in enumerate(rows):
        a, b = offsets[i], offsets[i + 1]
        assert not valid[a, 4] and (valid[a + 1:b, 4].all() if b - a > 1 else True)
    # single-step answer: despike leaves it unchanged
    x1 = ts.despike_step0(mats[2], valids[2], rows[2]["step_token_spans"], [6])
    assert np.array_equal(x1, mats[2])


def test_ve_orientation_matches_bank():
    rng = np.random.default_rng(2)
    r = ts.synthetic_row(rng, 60, 3); p = r["top_k_logprobs"]
    x, _ = ts.bank_streams(p["logprobs"], p["ids"], r["gen_token_ids"], r["token_spilled_energies"])
    # ve0 / ve0.75 are oriented toward ve1 within the answer: non-negative centred dot products
    for j in (1, 2):
        assert np.dot(x[:, j] - x[:, j].mean(), x[:, 3] - x[:, 3].mean()) >= 0


def test_module_reads_no_labels():
    src = (ROOT / "spectral_utils" / "ct7_token_streams.py").read_text(encoding="utf8")
    code = re.sub(r'"""[\s\S]*?"""', "", src)
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("#"))
    assert re.search(r"\b(target|labels|error_steps|np\.load|pickle)\b", code) is None
