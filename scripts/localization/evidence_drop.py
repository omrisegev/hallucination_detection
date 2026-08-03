"""
evidence_drop.py — the "Mind the Gap" (ICML 2026, PMLR 306) detector and its three baselines.

Digest with the full protocol and the five verified traps in the paper:
    papers/digests/mind-the-gap-catching-hallucinations-via-evidence-drop.md

Everything here is pure numpy over quantities already saved in our raw pkls, so this module
never touches a GPU and never needs the model. All functions take a per-token series and return
a scalar RISK score: **higher = more likely hallucinated**, matching the paper's phi(x, y).
(Note this is the OPPOSITE orientation from `spectral_utils.streaming_utils.FEATURE_SIGNS`,
where higher = more likely correct. Converting between the two is a single negation, done
explicitly at the call site rather than silently here.)

WHAT THE PAPER SPECIFIES, AND WHAT WE HAD TO CHOOSE
--------------------------------------------------
Specified (Sec. 3.3, Eq. 9-12, and Appendix D Eq. 46-48):
  - evidence proxy  Ehat_i = -H(Ptilde) over the RENORMALIZED top-K distribution, K = 20
  - EMA smoothing with span 5, then flux  Delta_j = Etilde_j - Etilde_{j-1}
  - risk  phi = -(1/M) * sum of the M most negative Delta, M = 5
  - the three baselines' *Avg* forms

NOT specified, and pre-registered here (see `DROP_SIGN_CONVENTION` below):
  - the *Drop* forms of LN-S and LogTokU. Appendix D gives only their Avg formulas, and the
    Sec. 3.3 Drop pipeline is written solely for Ehat_i = -H(Ptilde). The orientation of the
    per-step series before EMA + M-worst-drops is never stated, and it flips the metric.

CAVEAT CARRIED FROM THE DIGEST: Appendix E.1's Table 5 panels (a) and (b) are byte-identical,
so only one of the M / EMA-span ablations was actually run. M = 10 scores 90.75 against the
M = 5 default's 88.26 for Shannon Drop. `DEFAULTS` follows the paper's stated default; the
ablation sweep is the thing that tells us whether that default was ever justified.
"""
import numpy as np

# The paper's stated defaults (Sec. 3.3 + Appendix E.1).
DEFAULTS = {"M": 5, "ema_span": 5, "top_k": 20}

DROP_SIGN_CONVENTION = """
PRE-REGISTERED, ours not theirs.

Every baseline's per-token series is built so that HIGHER = MORE EVIDENCE, matching the sign of
the paper's own Ehat_i = -H(Ptilde) (a concentrated distribution -> low entropy -> high
evidence). Concretely:

    Shannon  ->  -H_i                      (the paper's own proxy, Eq. 10)
    LogTokU  ->  +M_i = +sum_{v in V_K} log P(v)   (their "evidence mass", Eq. 47, unnegated)
    LN-S     ->  +log P(sampled token)     (Eq. 48, unnegated)

The identical EMA + M-worst-negative-drop pipeline is then applied to all three. This is the
only reading under which "Drop" means the same operation for every baseline, and under which
each series' Avg form is the plain negated mean the paper prints. Any gap against their Table 1
/ Table 3 may therefore be a convention difference rather than a method difference — which is
exactly why it is written down before any number is produced.
""".strip()


# ── evidence proxies (per-token, higher = more evidence) ─────────────────────

def renormalized_top_k_logprobs(top_k_logprobs, k: int = 20) -> np.ndarray:
    """[T, k] log-probabilities renormalized over the top-k, from a saved `top_k_logprobs` dict.

    Our pkls store `{'ids': int32 [T, K], 'logprobs': float32 [T, K]}` with K = 50, and the
    entries are already sorted descending, so the paper's K = 20 is a strict prefix — Eq. 9's
    Ptilde is exactly reconstructible offline with no re-inference.

    At tau = 0 these logprobs are RAW log-softmax: `model_utils.generate_full` sets
    `sampling = temp > 1e-4` and passes `top_k=None, top_p=None` under greedy, so no warper ran.
    That matters — Eq. 9 renormalizes the raw distribution, not a warped one.
    """
    if not isinstance(top_k_logprobs, dict) or "logprobs" not in top_k_logprobs:
        raise TypeError("expected a {'ids':..., 'logprobs':...} dict from the raw pkl schema")
    lp = np.asarray(top_k_logprobs["logprobs"], dtype=np.float64)
    if lp.ndim != 2:
        raise ValueError(f"logprobs must be [T, K]; got shape {lp.shape}")
    if lp.shape[1] < k:
        raise ValueError(f"saved top-{lp.shape[1]} logprobs cannot supply the paper's top-{k}")
    lp = lp[:, :k]
    # Renormalize in log space: log Ptilde = lp - logsumexp(lp).
    m = lp.max(axis=1, keepdims=True)
    return lp - (m + np.log(np.exp(lp - m).sum(axis=1, keepdims=True)))


def shannon_evidence(top_k_logprobs, k: int = 20) -> np.ndarray:
    """Ehat_i = -H(Ptilde) per token (Eq. 10). Higher = more evidence."""
    log_p = renormalized_top_k_logprobs(top_k_logprobs, k)
    p = np.exp(log_p)
    return (p * log_p).sum(axis=1)          # = -H, since H = -sum p log p


def logtoku_evidence(top_k_logprobs, k: int = 20) -> np.ndarray:
    """M_i = sum_{v in V_K} log P(v) per token (Eq. 47). Higher = more evidence.

    Note this is the RAW top-k log-probability mass, deliberately NOT renormalized — Eq. 47
    sums log P(v), and renormalizing would make it a constant-shifted entropy and destroy the
    distinction from the Shannon proxy.
    """
    lp = np.asarray(top_k_logprobs["logprobs"], dtype=np.float64)[:, :k]
    return lp.sum(axis=1)


def ln_s_evidence(token_spilled_energies) -> np.ndarray:
    """log P(sampled token) per token (Eq. 48). Higher = more evidence.

    Our pkls save `token_spilled_energies` = -log p(sampled token), so this is a plain negation
    and needs no logprob lookup.
    """
    return -np.asarray(token_spilled_energies, dtype=np.float64)


# ── the Drop pipeline ────────────────────────────────────────────────────────

def ema(x, span: int = 5) -> np.ndarray:
    """Exponential moving average with pandas' `span` convention: alpha = 2 / (span + 1).

    Adjusted form, i.e. `pandas.Series.ewm(span=..., adjust=True)`:

        y_t = sum_{i<=t} (1-a)^(t-i) x_i  /  sum_{i<=t} (1-a)^(t-i)

    Implemented as a first-order IIR recursion (`y_t = x_t + (1-a) y_{t-1}`) rather than the
    obvious `cumsum(x / w) * w` trick. That trick divides by (1-a)^t, which **underflows to
    zero after a few hundred tokens** and yields NaN — fatal here, since these traces are
    thousands of tokens long. The denominator has the closed form (1 - (1-a)^(t+1)) / a, whose
    (1-a)^(t+1) term underflows harmlessly toward the correct limit 1/a.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return a
    alpha = 2.0 / (span + 1.0)
    decay = 1.0 - alpha
    try:
        from scipy.signal import lfilter
        num = lfilter([1.0], [1.0, -decay], a)
    except ImportError:                                  # pragma: no cover - scipy is a dep
        num = np.empty_like(a)
        acc = 0.0
        for i, v in enumerate(a):
            acc = v + decay * acc
            num[i] = acc
    with np.errstate(under="ignore"):
        den = (1.0 - decay ** np.arange(1, a.size + 1)) / alpha
    return num / den


def _flux_tol(evidence) -> float:
    """Magnitude below which a negative flux is float noise, not an Evidence Drop.

    The EMA of a perfectly flat trace is the constant back again only up to rounding, so
    `d < 0` on its own turns ~1e-16 jitter into a stream of spurious "drops" and makes the
    detector report a risk score for a trace where nothing happened. Scale-relative and set
    twelve orders of magnitude below the data: evidence values are O(0.1-5) nats and the
    smallest drop the paper cares about is O(0.01), so this can only ever remove noise.
    """
    e = np.asarray(evidence, dtype=np.float64)
    return 1e-12 * max(1.0, float(np.abs(e).max()) if e.size else 1.0)


def evidence_drop_risk(evidence, M: int = 5, ema_span: int = 5) -> float:
    """phi = -(1/M) * sum of the M most negative fluxes (Eq. 11-12). Higher = more risk.

    `evidence` is a per-token series oriented higher = more evidence. Only NEGATIVE fluxes are
    considered ("we only track positions in the reasoning trajectory where evidence drops
    appear"); if a trace has fewer than M negative fluxes, the mean is taken over however many
    exist, and a trace with none scores 0.0 (no drop anywhere = no risk).
    """
    e = np.asarray(evidence, dtype=np.float64)
    if e.size < 2:
        return 0.0
    d = np.diff(ema(e, ema_span))
    neg = d[d < -_flux_tol(e)]
    if neg.size == 0:
        return 0.0
    worst = np.sort(neg)[:M]                    # ascending -> the M most negative
    return float(-worst.mean())


def average_risk(evidence) -> float:
    """The 'Avg' baseline: plain negated mean of the evidence series. Higher = more risk.

    Reproduces each of Appendix D's sequence scores exactly under `DROP_SIGN_CONVENTION`:
    Shannon -> mean H_i; LogTokU -> -(1/T) sum M_i; LN-S -> -(1/T) sum log P.
    """
    e = np.asarray(evidence, dtype=np.float64)
    return float(-e.mean()) if e.size else 0.0


# ── candidate-level entry points ─────────────────────────────────────────────

EVIDENCE_FNS = {
    "shannon": lambda cand, k: shannon_evidence(cand["top_k_logprobs"], k),
    "logtoku": lambda cand, k: logtoku_evidence(cand["top_k_logprobs"], k),
    "ln_s":    lambda cand, k: ln_s_evidence(cand["token_spilled_energies"]),
}

METHODS = [f"{b}_{agg}" for b in ("shannon", "logtoku", "ln_s") for agg in ("avg", "drop")]


def candidate_risks(cand, M: int = None, ema_span: int = None, top_k: int = None) -> dict:
    """All six {baseline} x {avg, drop} risk scores for one candidate dict from a raw pkl.

    Returns NaN for any score whose input series is missing, so a cell with partial coverage
    still yields the scores it can support rather than failing the whole run.
    """
    M = DEFAULTS["M"] if M is None else M
    ema_span = DEFAULTS["ema_span"] if ema_span is None else ema_span
    top_k = DEFAULTS["top_k"] if top_k is None else top_k

    out = {}
    for base, fn in EVIDENCE_FNS.items():
        try:
            ev = fn(cand, top_k)
        except (KeyError, TypeError, ValueError, IndexError):
            out[f"{base}_avg"] = np.nan
            out[f"{base}_drop"] = np.nan
            continue
        out[f"{base}_avg"] = average_risk(ev)
        out[f"{base}_drop"] = evidence_drop_risk(ev, M, ema_span)
    return out


# ── known-answer tests ───────────────────────────────────────────────────────

def smoke() -> None:
    """Synthetic traces with planted drops at known indices. CPU-only, sub-second."""
    rng = np.random.default_rng(0)

    # 1. EMA matches pandas' adjusted ewm on a known series (hand-computed for span=1,
    #    where alpha = 1 and the EMA is the identity).
    x = np.array([1.0, 5.0, 2.0, 8.0])
    assert np.allclose(ema(x, span=1), x), ema(x, span=1)
    # span -> large makes the EMA approach the running mean.
    assert np.allclose(ema(x, span=10**6), np.cumsum(x) / np.arange(1, 5), atol=1e-3)

    # 2. A flat trace has no drops at all -> risk exactly 0. This is the float-noise guard:
    #    without `_flux_tol` the EMA's ~1e-16 jitter registers as a stream of drops and a
    #    trace where nothing happened gets a nonzero risk score.
    assert evidence_drop_risk(np.ones(50)) == 0.0
    assert evidence_drop_risk(np.full(4000, 3.7)) == 0.0, "long flat trace must also be 0"
    # A monotonically RISING evidence trace also has no negative flux -> risk 0.
    assert evidence_drop_risk(np.arange(50, dtype=float)) == 0.0
    # But a real drop far above the tolerance is still caught.
    assert evidence_drop_risk(np.concatenate([np.ones(20), np.full(20, 0.999)])) > 0.0

    # 3. A planted cliff must dominate: same mean evidence, one sharp drop vs none.
    flat = np.concatenate([np.zeros(40), np.zeros(40)])
    cliff = np.concatenate([np.full(40, 0.5), np.full(40, -0.5)])   # mean 0, one big drop
    assert average_risk(flat) == average_risk(cliff) == 0.0, "means must match by construction"
    assert evidence_drop_risk(cliff) > evidence_drop_risk(flat), "the cliff must score riskier"

    # 4. Deeper cliff -> strictly higher risk (monotone in drop magnitude).
    deep = np.concatenate([np.full(40, 2.0), np.full(40, -2.0)])
    assert evidence_drop_risk(deep) > evidence_drop_risk(cliff)

    # 5. M controls how many drops are averaged: with ONE planted drop and noise elsewhere,
    #    a larger M dilutes the score (the paper's own "M -> All dilutes local signals").
    noisy = rng.normal(0, 0.01, 200)
    noisy[100:] -= 3.0
    assert evidence_drop_risk(noisy, M=1) > evidence_drop_risk(noisy, M=50), "M must dilute"

    # 6. Renormalized top-k is a proper log-distribution (rows sum to 1) and is a PREFIX
    #    operation — asking for more than was saved must raise, never silently pad.
    lp = np.log(np.array([[0.5, 0.3, 0.2], [0.8, 0.1, 0.1]]))
    log_p = renormalized_top_k_logprobs({"logprobs": lp}, k=3)
    assert np.allclose(np.exp(log_p).sum(axis=1), 1.0), np.exp(log_p).sum(axis=1)
    assert np.allclose(np.exp(log_p), np.exp(lp)), "already-normalized input must be unchanged"
    try:
        renormalized_top_k_logprobs({"logprobs": lp}, k=20)
    except ValueError:
        pass
    else:
        raise AssertionError("asking for top-20 from saved top-3 must raise")

    # 7. Shannon evidence = -H, so a peaked distribution scores HIGHER than a flat one.
    peaked = np.log(np.array([[0.97, 0.02, 0.01]]))
    flat3 = np.log(np.array([[1 / 3, 1 / 3, 1 / 3]]))
    assert shannon_evidence({"logprobs": peaked}, 3)[0] > shannon_evidence({"logprobs": flat3}, 3)[0]
    # and -H of a uniform 3-way distribution is exactly -log(3).
    assert np.isclose(shannon_evidence({"logprobs": flat3}, 3)[0], -np.log(3))

    # 8. LN-S evidence is the negation of the saved spilled energy.
    assert np.allclose(ln_s_evidence([0.1, 2.0]), [-0.1, -2.0])

    # 9. `average_risk` reproduces Appendix D exactly: Shannon Avg == mean H.
    ev = shannon_evidence({"logprobs": np.vstack([peaked, flat3])}, 3)
    assert np.isclose(average_risk(ev), -ev.mean())

    # 10. A candidate missing a series yields NaN for that baseline only, not an exception.
    cand = {"token_spilled_energies": [0.1, 0.2, 0.3, 0.4]}      # no top_k_logprobs
    r = candidate_risks(cand)
    assert np.isnan(r["shannon_drop"]) and np.isnan(r["logtoku_avg"]), r
    assert np.isfinite(r["ln_s_avg"]) and np.isfinite(r["ln_s_drop"]), r

    print("evidence_drop.smoke: PASS (10 checks)")


if __name__ == "__main__":
    smoke()
