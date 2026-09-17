"""FROZEN development candidate CT7 (Step 418, 2026-09-17): six entropy-family streams plus the
de-spiked entropy-free chosen-token test, equal weight. Do not edit; any change is a new candidate.

Views (all per-step values, answer-standardized; higher = more likely the first error):
  1. H0lim                     Top10 step mean of the limit Renyi order-0 shape of the top-15 distribution
  2. ve0                       Top10 step mean of escort varentropy, order 0
  3. ve0.75                    Top10 step mean of escort varentropy, order .75
  4. ve1                       Top10 step mean of escort varentropy, order 1
       (varentropy channels oriented per answer toward ve1, as in digitfree_broad50.token_bank)
  5. H0lim_prefix_innovation   Top10 step mean of H0lim minus its strict prefix mean
  6. bocpd_residual            answer-standardized BOCPD signed residual channel: mean of the five
                               innovation5 features' residuals against a reset-before-observation
                               Gaussian BOCPD predictor, hazard 1/32 (aligned_context_predictors.bocpd_mean,
                               branch codex/temporal-research-20260915 @ ff3b02852)
  7. chosen_token_z            pooled step test sum(-log q(x) - H(q)) / sqrt(sum VE(q) + n*.01) over the
                               renormalized top-50 distribution q, answer-standardized, with step 0
                               set to the mean of the answer's other steps before standardization
Fusion: equal weight 1/7, no fitted parameter, no sign anchor (all views carry a fixed orientation).
Readout: argmax step within the answer; ProcessBench no-error decision from the frozen non-digit
tail15 gate (within-cell midrank >= .33), which is not part of this module.
No labels, no answers other than the scored answer enter any quantity here.
"""
from __future__ import annotations

import numpy as np

from .chosen_token_calibration import SUFFICIENT, step_z_readouts
from .digitfree_broad50 import NAMES as BANK50, masked_answer_standardize

CANDIDATE_ID = "CT7-six-entropy-plus-despiked-chosen-token-z-equal-v1"
SIX = ("H0lim", "ve0", "ve0.75", "ve1", "H0lim_prefix_innovation")  # + bocpd_residual
N_VIEWS = 7


def despiked_chosen_token_z(sufficient_stats: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Pooled chosen-token z-test per step, step 0 neutralized, answer-standardized."""
    stats = np.asarray(sufficient_stats, float)
    if stats.ndim != 2 or stats.shape[1] != len(SUFFICIENT) or stats.shape[0] != offsets[-1]:
        raise ValueError("sufficient statistics do not match the step offsets")
    z = step_z_readouts(stats)[:, 2]
    z = masked_answer_standardize(z[:, None], np.isfinite(z)[:, None], offsets)[:, 0]
    out = z.copy()
    for a, b in zip(offsets[:-1], offsets[1:]):
        if b - a > 1:
            out[a] = z[a + 1:b].mean()
    return masked_answer_standardize(out[:, None], np.isfinite(out)[:, None], offsets)[:, 0]


def candidate_views(bank50_standardized: np.ndarray, bocpd_residual: np.ndarray,
                    sufficient_stats: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """The seven frozen views, shape [steps, 7]."""
    x = np.asarray(bank50_standardized, float)
    if x.ndim != 2 or x.shape[1] != len(BANK50) or x.shape[0] != offsets[-1]:
        raise ValueError("bank matrix does not match the 50-stream digit-free contract")
    r = np.asarray(bocpd_residual, float)
    if r.shape != (offsets[-1],):
        raise ValueError("BOCPD residual does not match the step offsets")
    cols = [x[:, list(BANK50).index(n)] for n in SIX] + [r, despiked_chosen_token_z(sufficient_stats, offsets)]
    views = np.column_stack(cols)
    if not np.isfinite(views).all():
        raise ValueError("nonfinite view")
    return views


def candidate_step_scores(bank50_standardized, bocpd_residual, sufficient_stats, offsets) -> np.ndarray:
    """Equal-weight fused step score of the frozen candidate."""
    return candidate_views(bank50_standardized, bocpd_residual, sufficient_stats, offsets).mean(axis=1)
