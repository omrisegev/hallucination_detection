"""Chosen-token calibration statistics that are distribution-free under the model's own belief.

The existing chosen-token channels (surprisal, top-1 log gap, rank, mass above) mix two things:
how surprising the provided token is, and how flat the distribution was. In a flat distribution
every token is surprising, so those channels inherit entropy mechanically (Step 414: conditional
correlation .56-.58 with the entropy family).

Null hypothesis H0 at token t: the provided token was drawn from the model's own next-token
distribution p_t. Two statistics whose null distribution does not depend on p_t:

* ``pit_mid``   probability integral transform from the top: mass of strictly more probable
                tokens plus half the provided token's own mass. Under H0 it has mean 1/2 for
                every p_t (the randomized version is exactly Uniform(0,1)). High = the token
                sits in the model's tail.
* ``pit_normal`` Phi^-1(pit_mid), the same evidence on a Gaussian scale for Top-k readout.
* ``excess_surprisal`` surprisal minus the entropy: -log q(x) - H(q). Mean 0 under H0 for every q.
* ``std_excess_surprisal`` excess divided by sqrt(varentropy + floor). Mean 0 under H0 for every
                q and variance 1 except in the near-deterministic band, where the floor that keeps
                it finite deflates the variance (self-test: sd .43 in the lowest entropy quintile).
                PIT is therefore the primary distribution-free statistic.

The model's full vocabulary is not saved; only the top-50 log-probabilities are. All four are
computed on the renormalized top-50 distribution q. A provided token outside the top 50 is
censored: its q-mass is taken as the unobserved remainder of the full distribution, capped below
the smallest listed mass, and it is flagged. No labels enter this module.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri

NAMES = ("pit_mid", "pit_normal", "excess_surprisal", "std_excess_surprisal")
VARENTROPY_FLOOR = 1e-2
PIT_CLIP = 1e-4


def token_calibration(logp, ids, provided, surprisal):
    """Per-token calibration statistics. Returns (matrix [T,4], censored [T], diagnostics [T,3])."""
    lp = np.asarray(logp, float); ids = np.asarray(ids); provided = np.asarray(provided)
    s_full = np.asarray(surprisal, float)
    if lp.ndim != 2 or lp.shape != ids.shape or provided.shape != (len(lp),) or s_full.shape != provided.shape:
        raise ValueError("requires aligned top-k log-probabilities, ids, provided ids and surprisal")
    if not (np.isfinite(lp).all() and np.isfinite(s_full).all()):
        raise ValueError("nonfinite inputs")
    p = np.exp(lp)
    listed = p.sum(axis=1)
    q = p / listed[:, None]
    logq = lp - np.log(listed)[:, None]
    H = -(q * logq).sum(axis=1)
    VE = np.maximum((q * logq ** 2).sum(axis=1) - H ** 2, 0.0)

    hits = ids == provided[:, None]
    inside = hits.any(axis=1)
    rank = np.where(inside, hits.argmax(axis=1), lp.shape[1])
    # provided token mass on the q scale
    q_tok = np.where(inside, q[np.arange(len(q)), np.minimum(rank, lp.shape[1] - 1)], 0.0)
    # censored: the token's full-softmax mass exp(-s) is known exactly; place it on the q scale
    # but never above the smallest listed q, so the ordering claim "outside the top-k" holds.
    q_cens = np.minimum(np.exp(-s_full) / listed, q[:, -1])
    q_tok = np.where(inside, q_tok, q_cens)
    above = (q * (np.arange(lp.shape[1])[None, :] < rank[:, None])).sum(axis=1)
    pit = np.clip(above + 0.5 * q_tok, PIT_CLIP, 1 - PIT_CLIP)
    s_q = -np.log(np.maximum(q_tok, 1e-300))
    excess = s_q - H
    std_excess = excess / np.sqrt(VE + VARENTROPY_FLOOR)
    x = np.column_stack([pit, ndtri(pit), excess, std_excess])
    if not np.isfinite(x).all():
        raise ValueError("nonfinite calibration statistic")
    return x, ~inside, np.column_stack([H, VE, s_full])


def step_top_readout(x, spans, k=10):
    """Mean of the largest min(k, n) token values in each step: the bank's fixed Top10 readout."""
    out = np.empty((len(spans), x.shape[1]))
    for i, (a, b) in enumerate(np.asarray(spans, int)):
        if not 0 <= a < b <= len(x):
            raise ValueError("invalid span")
        seg = x[a:b]; kk = min(k, b - a)
        out[i] = np.partition(seg, b - a - kk, axis=0)[-kk:].mean(axis=0)
    return out


def self_test(seed=0):
    """Under H0 (tokens sampled from their own distribution), every statistic is flat in entropy;
    raw surprisal is not. Also checks censoring and shapes."""
    rng = np.random.default_rng(seed)
    T, K, V = 60000, 50, 400
    temps = np.exp(rng.uniform(np.log(.05), np.log(5.0), T))
    logits = rng.standard_normal((T, V)) * 3 / temps[:, None]
    logits -= logits.max(axis=1, keepdims=True)
    full = np.exp(logits); full /= full.sum(axis=1, keepdims=True)
    order = np.argsort(-full, axis=1)[:, :K]
    top = np.take_along_axis(full, order, axis=1)
    # H0: sample the provided token from the TOP-K renormalized distribution (what we can test)
    qk = top / top.sum(axis=1, keepdims=True)
    c = qk.cumsum(axis=1); u = rng.random(T)[:, None]
    pick = (c < u).sum(axis=1).clip(0, K - 1)
    provided = order[np.arange(T), pick]
    surprisal = -np.log(full[np.arange(T), provided])
    x, cens, diag = token_calibration(np.log(top), order, provided, surprisal)
    H = diag[:, 0]
    assert not cens.any()
    bins = np.quantile(H, np.linspace(0, 1, 6))
    idx = np.clip(np.searchsorted(bins, H, side="right") - 1, 0, 4)
    means = np.array([[x[idx == b, j].mean() for b in range(5)] for j in range(4)])
    raw = np.array([surprisal[idx == b].mean() for b in range(5)])
    stds = np.array([x[idx == b, 3].std() for b in range(5)])
    # pit: mean 1/2 in every entropy quintile; excess: mean 0; standardized: mean 0, sd ~1
    assert np.all(np.abs(means[0] - .5) < .02), means[0]
    assert np.all(np.abs(means[2]) < .05), means[2]
    assert np.all(np.abs(means[3]) < .05), means[3]
    # The varentropy floor deflates the spread only in the near-deterministic band (lowest entropy
    # quintile), where sqrt(VE) -> 0 would otherwise explode. It never inflates it. PIT has no floor.
    assert np.all(np.abs(stds[1:] - 1) < .15) and stds[0] <= 1.05, stds
    assert raw[-1] - raw[0] > 1.0, raw          # raw surprisal climbs with entropy
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    # censoring: a token outside the top-k is flagged and gets the maximal PIT region
    x2, cens2, _ = token_calibration(np.log(top[:5]), order[:5], np.full(5, -1), np.full(5, 30.0))
    assert cens2.all() and np.all(x2[:, 0] > .99 - 1e-9)
    return {"pit_mean_by_entropy_quintile": means[0].round(3).tolist(),
            "std_excess_mean_by_quintile": means[3].round(3).tolist(),
            "std_excess_sd_by_quintile": stds.round(3).tolist(),
            "raw_surprisal_by_quintile": raw.round(3).tolist(),
            "corr_with_entropy": {"surprisal": corr(surprisal, H), "pit_mid": corr(x[:, 0], H),
                                  "excess": corr(x[:, 2], H), "std_excess": corr(x[:, 3], H)}}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), indent=1))


SUFFICIENT = ("n_tokens", "sum_std_excess", "sum_pit_normal", "sum_excess", "sum_varentropy", "n_censored")


def step_sufficient_stats(x, censored, diag, spans):
    """Per-step sums from which every step readout is derived without re-reading tokens.

    Columns: token count, sum of standardized excess, sum of PIT normal scores, sum of excess
    surprisal, sum of varentropy, censored token count.
    """
    out = np.empty((len(spans), len(SUFFICIENT)))
    for i, (a, b) in enumerate(np.asarray(spans, int)):
        if not 0 <= a < b <= len(x):
            raise ValueError("invalid span")
        out[i] = (b - a, x[a:b, 3].sum(), x[a:b, 1].sum(), x[a:b, 2].sum(), diag[a:b, 1].sum(),
                  censored[a:b].sum())
    return out


def step_z_readouts(stats, floor=VARENTROPY_FLOOR):
    """Step-level tests whose null is N(0,1) whatever the entropy and the step length.

    * ``z_std_excess``  sum of per-token standardized excess / sqrt(n)
    * ``z_pit``         sum of PIT normal scores / sqrt(n). Drifts upward with step length because
                        the mid-PIT normal score is not centred for discrete distributions (see the
                        self-test); reported, not primary.
    * ``z_pooled``      sum of excess surprisal / sqrt(sum of varentropy + n * floor): the exact
                        one-sample test of "this step is more surprising than the model expected",
                        pooling the variance over the step instead of per token.
    """
    n = stats[:, 0]
    return np.column_stack([stats[:, 1] / np.sqrt(n), stats[:, 2] / np.sqrt(n),
                            stats[:, 3] / np.sqrt(stats[:, 4] + n * floor)])


def self_test_step_readouts(seed=1):
    """Under H0 the step z readouts have mean 0 and sd ~1 at every step length; Top10 does not."""
    rng = np.random.default_rng(seed)
    T, K, V = 40000, 50, 300
    temps = np.exp(rng.uniform(np.log(.1), np.log(4.0), T))
    logits = rng.standard_normal((T, V)) * 3 / temps[:, None]
    logits -= logits.max(axis=1, keepdims=True)
    full = np.exp(logits); full /= full.sum(axis=1, keepdims=True)
    order = np.argsort(-full, axis=1)[:, :K]; top = np.take_along_axis(full, order, axis=1)
    qk = top / top.sum(axis=1, keepdims=True); c = qk.cumsum(axis=1)
    pick = (c < rng.random(T)[:, None]).sum(axis=1).clip(0, K - 1)
    provided = order[np.arange(T), pick]
    x, cens, diag = token_calibration(np.log(top), order, provided, -np.log(full[np.arange(T), provided]))
    lengths = rng.choice([4, 16, 64], size=4000)
    ends = np.cumsum(lengths); keep = ends <= T; lengths = lengths[keep]; ends = ends[keep]
    spans = np.column_stack([ends - lengths, ends])
    z = step_z_readouts(step_sufficient_stats(x, cens, diag, spans))
    top10 = step_top_readout(x, spans)[:, 3]
    res = {}
    for n in (4, 16, 64):
        m = lengths == n
        res[int(n)] = {"z_pooled_mean": float(z[m, 2].mean()), "z_pooled_sd": float(z[m, 2].std()),
                       "z_pit_mean": float(z[m, 1].mean()), "top10_std_excess_mean": float(top10[m].mean())}
        # Self-normalized sums are N(0,1) only asymptotically; surprisal is right-skewed, so very
        # short steps carry a small negative bias (about -0.1 at 4 tokens) that shrinks with n.
        assert abs(z[m, 2].mean()) < .15, res
        assert .75 < z[m, 2].std() < 1.2, res
    assert res[64]["top10_std_excess_mean"] - res[4]["top10_std_excess_mean"] > .5, res
    # Known failure, kept visible: the mid-PIT normal score has a small positive per-token bias for
    # discrete distributions (Phi^-1 of a non-uniform mid-PIT), so sum/sqrt(n) drifts upward with
    # step length. z_pit is therefore NOT length-free; z_pooled is the primary readout.
    assert res[64]["z_pit_mean"] > res[4]["z_pit_mean"], res
    return res
