"""Mechanism tests for gate feature/readout selection."""
from __future__ import annotations

import inspect
import numpy as np

from spectral_utils import gate_feature_readout as model


def run():
    rng = np.random.default_rng(20260914)
    p = rng.dirichlet(np.linspace(2.0, 0.2, 60), size=14)
    p = np.sort(p, axis=1)[:, ::-1][:, :50]
    logprobs = np.log(p)
    entropy = np.linspace(0.1, 1.4, len(p))
    spans = np.asarray([[0, 4], [4, 14]])

    signals = model.token_signals(logprobs, entropy)
    assert tuple(signals) == model.SIGNAL_NAMES
    np.testing.assert_allclose(signals["raw_neglogp1"], -logprobs[:, 0], atol=1e-12, rtol=0)
    assert np.all(signals["q15_H1"] >= signals["q15_Hinf"] - 1e-12)
    assert np.all(signals["tail15_mass"] >= signals["tail50_mass"] - 1e-12)

    readouts = model.apply_readouts(entropy, spans)
    assert readouts["token_mean"] == float(entropy.mean())
    assert readouts["token_top10"] == float(np.sort(entropy)[-10:].mean())
    expected_steps = np.mean([entropy[:4].mean(), np.sort(entropy[4:])[-10:].mean()])
    np.testing.assert_allclose(readouts["mean_step_top10"], expected_steps, atol=1e-12, rtol=0)

    detectors = model.answer_detectors(logprobs, entropy, spans)
    assert tuple(detectors) == model.METHODS and len(detectors) == 33
    assert detectors[model.BASELINE] == float(entropy.mean())
    assert "label" not in inspect.signature(model.answer_detectors).parameters
    return {"status": "PASS", "checks": 9, "candidates": len(detectors)}


if __name__ == "__main__":
    print(run())
