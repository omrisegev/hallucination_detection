"""
DeepConf — "Deep Think with Confidence" (Fu et al., arXiv:2508.15260), native protocol.

Handoff §M1/§M2. Nothing in this module may be called "exact DeepConf" until
`equality_audit` has shown that our saved telemetry reproduces the pinned official
function row for row on raw logits. Until then every DeepConf-derived number in this
project is a **named proxy**, and the manifest says so.

Three implementations exist in the wild and they are NOT interchangeable
-----------------------------------------------------------------------
1. **Paper Eq. 2**: `C_i = -(1/k) sum_{j=1..k} log P_i(j)` over the top-k tokens.
2. **Paper Appendix G.4 step 5** (the actual released snippet):
   `new_conf = -sum(logprobs[1:]) / len(logprobs[1:])` — vLLM puts the *sampled* token at
   index 0 and the top-k candidates after it, so this **excludes the sampled token**.
3. **Repository `main`'s `processors.py`**: softmaxes the logits, takes the `conf_topk`
   largest probabilities and averages their negative log probabilities, with no
   sampled-token special case.

Including or excluding index 0 shifts every confidence value and therefore every
percentile threshold, so `CONF_VARIANTS` exposes all three by name and the P0 audit pins
exactly one from the cloned repository at the recorded commit. Choosing after seeing the
results is forbidden; the pin is recorded in the manifest before generation starts.

Raw vs post-warper
------------------
`out.scores` from HF `generate` are temperature-scaled and top-k/top-p masked; `out.logits`
are raw. DeepConf's confidence is defined on the model's distribution, so it must be
computed from RAW logits. Our older caches stored post-warper values, which is why the
phase-1 checkpoint labels every historical DeepConf number a proxy. `logits_stage` in the
manifest records which one a run used, and `equality_audit` refuses to pass on
post-warper input.
"""
import numpy as np

#: Pinned by the P0 audit; recorded in every DeepConf manifest.
OFFICIAL_REPO = "https://github.com/facebookresearch/deepconf"
#: vLLM commit the paper pins in Appendix G.1 (Python 3.12.0, CUDA 12.8).
PINNED_VLLM_COMMIT = "31f09c615f4f067dba765ce5fe7d00d880212a6d"

#: Paper Appendix G.3 requests `top_logprobs=20` from vLLM.
DEFAULT_CONF_TOPK = 20
#: Paper §3.1 / Alg. 2 group window.
DEFAULT_GROUP_WINDOW = 2048
#: Online warm-up trace count (Alg. 2).
DEFAULT_N_INIT = 16
#: Consensus stop for online sampling (Alg. 2).
DEFAULT_BETA = 0.95
#: Budgets reported in the paper's online table.
BUDGETS = (32, 64, 128, 256, 512)
#: DeepConf-low / DeepConf-high filtering percentages.
ETA_LOW, ETA_HIGH = 10, 90


# ── token confidence ────────────────────────────────────────────────────────────

def _as_desc(logprobs):
    a = np.asarray(logprobs, dtype=np.float64)
    return a


def conf_paper_eq2(top_logprobs_desc, conf_topk: int = DEFAULT_CONF_TOPK) -> float:
    """Paper Eq. 2 literally: mean of -log p over the top-k tokens, sampled token included
    if it is among them. `top_logprobs_desc` is the descending-sorted top-k logprob vector
    of ONE token position."""
    lp = _as_desc(top_logprobs_desc)[:conf_topk]
    return float(-np.mean(lp)) if lp.size else float("nan")


def conf_appendix_g4(sampled_then_topk_logprobs, conf_topk: int = DEFAULT_CONF_TOPK) -> float:
    """Appendix G.4 snippet: drop index 0 (the sampled token), average the rest.

    Expects the vLLM layout — sampled token first, then the top-k candidates.
    """
    lp = _as_desc(sampled_then_topk_logprobs)[1:conf_topk + 1]
    return float(-np.mean(lp)) if lp.size else float("nan")


def conf_repo_main(top_logprobs_desc, conf_topk: int = DEFAULT_CONF_TOPK) -> float:
    """Repository `main` `processors.py`: same arithmetic as Eq. 2, no sampled-token case.

    Kept as a separate name even though the formula coincides with `conf_paper_eq2`,
    because the *input* differs: `processors.py` re-softmaxes raw logits itself and takes
    its own top-k, so it never sees a sampled-token-first vector. Two names that currently
    compute the same thing from different inputs is the honest encoding of the situation;
    collapsing them would hide which one a run actually used.
    """
    lp = _as_desc(top_logprobs_desc)[:conf_topk]
    return float(-np.mean(lp)) if lp.size else float("nan")


CONF_VARIANTS = {
    "paper_eq2": conf_paper_eq2,
    "appendix_g4_exclude_sampled": conf_appendix_g4,
    "repo_main_processors": conf_repo_main,
}


def trace_token_confidence(top_k_logprobs, variant: str = "appendix_g4_exclude_sampled",
                           conf_topk: int = DEFAULT_CONF_TOPK,
                           sampled_first: bool = None) -> np.ndarray:
    """Per-token confidence C_t for a whole trace.

    `top_k_logprobs` is the [T, K] logprob matrix (descending within each row, or
    sampled-token-first when `variant='appendix_g4_exclude_sampled'` and the capture came
    from vLLM). `sampled_first` documents the layout explicitly; leaving it None uses the
    variant's own convention.
    """
    if variant not in CONF_VARIANTS:
        raise ValueError(f"unknown DeepConf variant {variant!r}; have {sorted(CONF_VARIANTS)}")
    mat = np.asarray(top_k_logprobs, dtype=np.float64)
    if mat.ndim != 2:
        raise ValueError(f"expected [T, K] logprobs, got shape {mat.shape}")
    if sampled_first is None:
        sampled_first = (variant == "appendix_g4_exclude_sampled")
    if variant == "appendix_g4_exclude_sampled" and not sampled_first:
        raise ValueError(
            "appendix_g4_exclude_sampled drops index 0 as the sampled token, but the "
            "capture is not sampled-token-first. Passing a descending top-k matrix here "
            "would silently drop the most probable token instead."
        )
    fn = CONF_VARIANTS[variant]
    return np.array([fn(row, conf_topk) for row in mat], dtype=np.float64)


# ── group / trace aggregation (paper §3.1, Eq. 4-7) ─────────────────────────────

def group_confidences(conf, window: int = DEFAULT_GROUP_WINDOW) -> np.ndarray:
    """Overlapping sliding-window group confidences C_{G_i} (Eq. 4), stride 1.

    Windows are the *complete* windows only. A trace shorter than one window yields a
    single group over the whole trace, which matches the official behaviour of never
    letting a short trace be judged on a partial window it never filled.
    """
    c = np.asarray(conf, dtype=np.float64)
    T = len(c)
    if T == 0:
        return np.array([], dtype=np.float64)
    w = int(min(window, T))
    csum = np.concatenate([[0.0], np.cumsum(c)])
    return (csum[w:] - csum[:-w]) / w


def lowest_group_conf(conf, window: int = DEFAULT_GROUP_WINDOW) -> float:
    """Eq. 6 — min over groups. The statistic DeepConf's online rule uses."""
    g = group_confidences(conf, window)
    return float(np.min(g)) if g.size else float("nan")


def bottom_pct_group_conf(conf, pct: float = 10.0,
                          window: int = DEFAULT_GROUP_WINDOW) -> float:
    """Eq. 5 — mean of the lowest `pct`% of group confidences."""
    g = group_confidences(conf, window)
    if not g.size:
        return float("nan")
    k = max(1, int(round(len(g) * pct / 100.0)))
    return float(np.mean(np.sort(g)[:k]))


def tail_conf(conf, tail_tokens: int = 2048) -> float:
    """Eq. 7 — mean confidence over the final `tail_tokens` tokens."""
    c = np.asarray(conf, dtype=np.float64)
    return float(np.mean(c[-int(tail_tokens):])) if c.size else float("nan")


def mean_conf(conf) -> float:
    """Global average trace confidence — the aggregation DeepConf argues against, kept as
    the paper's own comparison point (App. B.4 finds local signals only 0.7pp better)."""
    c = np.asarray(conf, dtype=np.float64)
    return float(np.mean(c)) if c.size else float("nan")


TRACE_STATISTICS = {
    "lowest_group_2k": lambda c: lowest_group_conf(c, 2048),
    "lowest_group_512": lambda c: lowest_group_conf(c, 512),
    "lowest_group_1k": lambda c: lowest_group_conf(c, 1024),
    "bottom_10pct": lambda c: bottom_pct_group_conf(c, 10.0, 2048),
    "bottom_50pct": lambda c: bottom_pct_group_conf(c, 50.0, 2048),
    "tail_2k": lambda c: tail_conf(c, 2048),
    "tail_10pct": lambda c: tail_conf(c, max(1, int(0.10 * len(c)))),
    "mean": mean_conf,
}


# ── offline filtering + voting (paper §3.2) ─────────────────────────────────────

def filter_and_vote(answers, trace_conf, eta: float = None, weighted: bool = True) -> dict:
    """One offline DeepConf decision over a working set of traces.

    `eta` keeps the top `eta`% most confident traces; None means no filtering (plain
    majority / mean-weighted vote). The cutoff is recomputed **within the working set on
    every resampling run**, as the paper specifies — computing it once on the full 4,096
    pool would leak pool-level information into every K-sized run.

    Confidence weights must be strictly positive for `V(a) = sum_t C_t * I(answer(t)=a)`
    to behave; DeepConf confidences are `-log p` averages and so are naturally positive.
    A signed, z-scored score (like ours) needs a declared positivity mapping before it can
    enter this harness — that mapping is an adaptation, not DeepConf.
    """
    ans = list(answers)
    conf = np.asarray(trace_conf, dtype=np.float64)
    keep = np.isfinite(conf) & np.array([a is not None for a in ans])
    if eta is not None and keep.sum() > 0:
        cutoff = np.percentile(conf[keep], 100.0 - float(eta))
        keep = keep & (conf >= cutoff)
    idx = np.flatnonzero(keep)
    if idx.size == 0:
        return {"answer": None, "votes": {}, "n_voting": 0, "beta": float("nan")}

    votes = {}
    for i in idx:
        w = float(conf[i]) if weighted else 1.0
        if weighted and w <= 0:
            raise ValueError("confidence-weighted voting needs strictly positive weights")
        votes[ans[i]] = votes.get(ans[i], 0.0) + w
    best = max(votes, key=votes.get)
    total = sum(votes.values())
    return {
        "answer": best,
        "votes": votes,
        "n_voting": int(idx.size),
        "beta": float(votes[best] / total) if total > 0 else float("nan"),
    }


def offline_resample(pool, K: int, n_runs: int = 64, eta: float = None,
                     weighted: bool = True, statistic: str = "lowest_group_2k",
                     seed: int = 42) -> dict:
    """Resample working sets of size K from a per-question trace pool, `n_runs` times.

    `pool` is a list of {'answer', 'conf' (per-token array or precomputed float),
    'is_correct', 'n_tokens'}. Returns mean accuracy and mean sampled tokens over runs —
    the paper's own table unit (all metrics averaged over 64 independent resamplings).
    """
    rng = np.random.default_rng(seed)
    stat_fn = TRACE_STATISTICS[statistic]
    scores = np.array([t["conf"] if np.isscalar(t["conf"]) else stat_fn(t["conf"])
                       for t in pool], dtype=np.float64)
    answers = [t["answer"] for t in pool]
    correct_of = {}
    for t in pool:
        correct_of.setdefault(t["answer"], bool(t["is_correct"]))
    ntok = np.array([t.get("n_tokens", 0) for t in pool], dtype=np.float64)

    accs, toks = [], []
    for _ in range(int(n_runs)):
        pick = rng.choice(len(pool), size=min(K, len(pool)), replace=False)
        res = filter_and_vote([answers[i] for i in pick], scores[pick],
                              eta=eta, weighted=weighted)
        accs.append(1.0 if correct_of.get(res["answer"], False) else 0.0)
        toks.append(float(ntok[pick].sum()))
    return {
        "K": int(K), "eta": eta, "weighted": bool(weighted), "statistic": statistic,
        "accuracy": float(np.mean(accs)), "tokens": float(np.mean(toks)),
        "n_runs": int(n_runs), "pool_size": len(pool),
    }


# ── online early termination (Alg. 2) ───────────────────────────────────────────

def online_threshold(warmup_trace_confs, eta: float = ETA_LOW) -> float:
    """s = Percentile_{100-eta} over the warm-up traces' lowest-group confidences.

    eta=10 (DeepConf-low) keeps only the most confident tenth, so the threshold is the
    90th percentile. Getting the percentile direction backwards inverts the whole method
    and still produces a plausible-looking accuracy number, which is why the P1 regression
    suite asserts the direction rather than trusting it.
    """
    c = np.asarray([x for x in warmup_trace_confs if np.isfinite(x)], dtype=np.float64)
    if c.size == 0:
        return float("nan")
    return float(np.percentile(c, 100.0 - float(eta)))


def online_should_terminate(running_conf, threshold: float,
                            window: int = DEFAULT_GROUP_WINDOW) -> bool:
    """Terminate as soon as the *current* group confidence falls below the threshold.

    Only complete windows count: before the trace has produced `window` tokens there is no
    group yet, and terminating on a partial window would kill short traces on noise.
    """
    c = np.asarray(running_conf, dtype=np.float64)
    if c.size < window:
        return False
    return bool(np.mean(c[-window:]) < threshold)


def consensus_reached(votes: dict, beta: float = DEFAULT_BETA) -> bool:
    if not votes:
        return False
    total = sum(votes.values())
    return total > 0 and (max(votes.values()) / total) >= beta


# ── equality audit ──────────────────────────────────────────────────────────────

def equality_audit(our_conf, official_conf, logits_stage: str,
                   atol: float = 1e-6, rtol: float = 1e-5) -> dict:
    """Prove row-level equality of our confidence with the pinned official function.

    Handoff §M1: "First prove row-level equality of our saved confidence with the pinned
    function using raw logits." A post-warper input fails outright rather than being
    compared with a loosened tolerance — the two are different quantities, and a
    near-match on a low-temperature trace would be coincidence, not equality.
    """
    if logits_stage != "raw":
        return {
            "passed": False,
            "reason": f"logits_stage={logits_stage!r}; DeepConf confidence is defined on "
                      f"RAW logits. Post-warper telemetry may only be reported as a named "
                      f"proxy, never as exact DeepConf.",
            "n": 0,
        }
    a = np.asarray(our_conf, dtype=np.float64).ravel()
    b = np.asarray(official_conf, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return {"passed": False, "reason": f"shape mismatch {a.shape} vs {b.shape}", "n": 0}
    ok = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(a[ok] - b[ok])
    tol = atol + rtol * np.abs(b[ok])
    n_bad = int(np.sum(diff > tol))
    return {
        "passed": bool(n_bad == 0 and ok.all()),
        "n": int(a.size),
        "n_compared": int(ok.sum()),
        "n_nonfinite": int((~ok).sum()),
        "n_exceeding_tol": n_bad,
        "max_abs_diff": float(diff.max()) if diff.size else 0.0,
        "reason": "row-level equality on raw logits" if n_bad == 0 else
                  f"{n_bad}/{ok.sum()} rows exceed tolerance",
    }
