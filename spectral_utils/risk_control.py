"""Risk control for a one-dimensional threshold: Learn-then-Test with Hoeffding-Bentkus p-values and fixed-sequence
testing (Angelopoulos, Bates, Candes, Jordan, Lei 2021; the testing step used by Laufer-Goldshtein et al., Pareto
Testing, ICLR 2023).

Used by the calibrated "clean answer" track (docs/research_notes/math_rigor_2026-10-24/
CALIBRATED_CLEAN_THRESHOLD_DESIGN.md). LABELS ARE REQUIRED here, on a declared calibration split: this is a
calibrated tool, not part of the label-free method.

Guarantee: if the candidate thresholds are tested in a fixed order (most conservative first) and the losses are
in [0, 1] and the calibration and test units are exchangeable, the returned threshold satisfies
P(risk > alpha) <= delta over the draw of the calibration set.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binom


def _kl_bernoulli(a, b):
    a = min(max(a, 1e-12), 1 - 1e-12); b = min(max(b, 1e-12), 1 - 1e-12)
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def hb_pvalue(risk_hat: float, n: int, alpha: float) -> float:
    """Hoeffding-Bentkus p-value for H0: risk > alpha, given the empirical risk of n i.i.d. losses in [0, 1].

    p = min( exp(-n * KL(min(r, alpha) || alpha)),  e * P(Binomial(n, alpha) <= ceil(n r)) ).
    """
    if n <= 0:
        return 1.
    hoeff = float(np.exp(-n * _kl_bernoulli(min(risk_hat, alpha), alpha)))
    bentkus = float(np.e * binom.cdf(np.ceil(n * risk_hat), n, alpha))
    return min(1., hoeff, bentkus)


def fixed_sequence_threshold(thresholds, risks, n: int, alpha: float, delta: float):
    """Fixed-sequence Learn-then-Test over thresholds ordered from most to least conservative.

    thresholds: candidate values in testing order; risks: empirical risk at each (same order); n: number of
    calibration units the risks average over. Rejects H0_j: risk_j > alpha while p_j <= delta, and stops at the
    first non-rejection. Returns (index of the last rejected threshold or None, p-values).
    """
    pvals = np.array([hb_pvalue(float(r), n, alpha) for r in risks])
    last = None
    for j, p in enumerate(pvals):
        if p > delta:
            break
        last = j
    return last, pvals
