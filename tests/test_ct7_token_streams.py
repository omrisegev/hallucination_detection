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


def test_bocpd_temporal_recipe_follows_the_bundle_chain():
    """The raw-row rebuild follows baseline -> prepare_temporal_context_data -> _bocpd_one:
    float64 oriented features, prefix innovation with token 0 included, float64 mean/scale,
    float32 storage round trip, BOCPD prior-mean residual; and it enters column 5 unmasked."""
    from spectral_utils.aligned_context_predictors import bocpd_mean
    from spectral_utils.renyi_locator_feature_bank import feature_matrix
    rng = np.random.default_rng(3)
    r = ts.synthetic_row(rng, 80, 5); p = r["top_k_logprobs"]
    b = ts.bocpd_residual_temporal_recipe(p["logprobs"], r["token_entropies"])
    m = feature_matrix(np.asarray(p["logprobs"], float), np.asarray(r["token_entropies"], float))["matrix"][:, :4]
    inn = np.r_[0.0, m[1:, 0] - np.cumsum(m[:-1, 0]) / np.arange(1, len(m))]
    aug = np.column_stack([m, inn]); feats = aug.astype(np.float32)
    z = (feats.astype(float) - aug.mean(0)) / np.maximum(aug.std(0), 1e-8)
    assert np.array_equal(b, (z - bocpd_mean(z, hazard=1 / 32)).mean(1))
    # float32 storage matters at the 1e-8 replay tolerance: without it the result moves
    z64 = (aug - aug.mean(0)) / np.maximum(aug.std(0), 1e-8)
    assert not np.array_equal(b, (z64 - bocpd_mean(z64, hazard=1 / 32)).mean(1))
    x, v = ts.answer_streams(p["logprobs"], p["ids"], r["gen_token_ids"], r["token_spilled_energies"], bocpd=b)
    assert np.array_equal(x[:, 5], b) and v[:, 5].all() and x.dtype == np.float64
