"""
Per-token telemetry capture with a live stopping hook.

Handoff §3.2 requires, for every generated token: full-vocabulary entropy, log-sum-exp,
sampled-token probability/logprob, spilled energy, pmax, top-two logprob margin, the exact
DeepConf confidence, and raw-versus-post-warper top-50 IDs/logprobs. The stopping lanes
(REFRAIN, LEASH, DeepConf-online) additionally need to *act* on that telemetry mid-trace,
which `generate_full` cannot do — it is a one-shot call that returns after the trace is
finished.

So this module owns one incremental decode loop that all four lanes share. Sharing it is
the point: vanilla, REFRAIN and LEASH must differ **only** in their stopping rule, never in
their prompt, sampler, tokenizer, EOS set, or telemetry definition, or the accuracy-token
frontier they are compared on is not a controlled comparison.

Raw vs post-warper, explicitly
------------------------------
Each step computes both distributions and keeps them apart by name:

    raw_*        from the model's untouched logits — the definition every published
                 confidence/entropy statistic uses
    sampled_*    from the temperature/top-p/top-k-warped distribution actually sampled from

`token_entropies` in this project's older caches are post-warper top-K=15 values. They are
kept (as `sampled_entropy`) for continuity with the frozen feature contract, but no
published baseline is ever computed from them here.
"""
from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class DecodeConfig:
    """The generation contract. Every field is pinned in the run manifest."""
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_new_tokens: int = 16384
    seed: int = 42
    logprob_top_k: int = 50          # retained top-k for offline recomputation
    conf_topk: int = 20              # DeepConf's own top-k (Appendix G.3)
    eos_token_ids: tuple = ()
    keep_top_k_arrays: bool = True   # False = scalar channels only (the M2 storage contract)


@dataclass
class TokenChannels:
    """The frozen scalar channels, one array per trace."""
    raw_entropy: list = field(default_factory=list)       # full-vocab H from raw logits
    raw_logsumexp: list = field(default_factory=list)     # log Z from raw logits
    raw_logprob_sampled: list = field(default_factory=list)
    raw_pmax: list = field(default_factory=list)
    raw_margin: list = field(default_factory=list)        # top1 - top2 logprob, raw
    spilled_energy: list = field(default_factory=list)    # -log p(sampled), raw
    deepconf_conf: list = field(default_factory=list)     # pinned DeepConf C_t
    sampled_entropy: list = field(default_factory=list)   # post-warper top-15 H (legacy)

    def as_dict(self) -> dict:
        return {k: list(v) for k, v in self.__dict__.items()}

    def __len__(self):
        return len(self.raw_entropy)


def _warp(logits: torch.Tensor, cfg: DecodeConfig) -> torch.Tensor:
    """Apply temperature, then top-k, then top-p — the HF `generate` order.

    Order matters: top-p over an already-top-k-truncated distribution is not the same set
    as top-k over a top-p set. Matching HF's order is what makes a vanilla run here
    comparable to the rest of this project's caches.
    """
    if cfg.temperature and cfg.temperature > 1e-4:
        logits = logits / cfg.temperature
    if cfg.top_k and cfg.top_k > 0:
        kth = torch.topk(logits, min(cfg.top_k, logits.shape[-1]))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if cfg.top_p is not None and 0 < cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        drop = cum - probs > cfg.top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return logits


class IncrementalDetokenizer:
    """Maintain the decoded text of a growing token list in O(window) per token.

    A naive `tok.decode(all_ids)` after every token is O(T^2) — at the 16,384-token cap
    that is ~134M token-decodes per trace and dominates the GPU time. Decoding only a
    trailing window and splicing keeps it linear, while still going through the real
    tokenizer so multi-byte and byte-BPE pieces resolve correctly.

    Correctness rests on the anchor never moving into the middle of a multi-token
    character: the anchor advances only past tokens that are already committed to `head`,
    and `window` (64) is far larger than any single UTF-8 sequence's token span.
    """

    def __init__(self, tok, window: int = 64):
        self.tok = tok
        self.window = int(window)
        self.ids = []
        self.anchor = 0
        self.head = ""

    def append(self, token_id: int) -> str:
        self.ids.append(int(token_id))
        if len(self.ids) - self.anchor > 2 * self.window:
            keep = len(self.ids) - self.window
            self.head += self.tok.decode(self.ids[self.anchor:keep],
                                         skip_special_tokens=False)
            self.anchor = keep
        return self.text

    @property
    def text(self) -> str:
        return self.head + self.tok.decode(self.ids[self.anchor:], skip_special_tokens=False)


@torch.no_grad()
def stream_generate(mdl, tok, prompt_ids, cfg: DecodeConfig,
                    on_token=None, stop_check=None, generator=None):
    """Decode incrementally, capturing telemetry and consulting a stopping hook.

    Args:
        prompt_ids:  1-D LongTensor of prompt token IDs (already chat-templated).
        on_token:    optional callback(step_idx, token_id, channels) for streaming consumers.
        stop_check:  optional callback(text_so_far, channels) -> bool, consulted after each
                     token. Returning True ends generation with `stop_reason='policy'`.
        generator:   torch.Generator for reproducible sampling.

    Returns a dict with the decoded text, token IDs, `TokenChannels`, optional retained
    top-k arrays, and an explicit `stop_reason` in {'eos', 'length', 'policy'}.

    `stop_reason` is a first-class field because the stopping lane's accounting depends on
    it: a trace that hit the 16,384-token cap did not "choose" to stop, and counting it as
    a natural completion would credit vanilla with savings it never made.
    """
    from .deepconf import conf_paper_eq2

    device = mdl.device
    ids = prompt_ids.to(device).unsqueeze(0)
    past = None
    cur = ids
    ch = TokenChannels()
    out_ids, raw_topk_ids, raw_topk_lps, warp_topk_ids, warp_topk_lps = [], [], [], [], []
    eos = set(int(e) for e in cfg.eos_token_ids)
    stop_reason = "length"
    detok = IncrementalDetokenizer(tok) if stop_check is not None else None

    for step in range(int(cfg.max_new_tokens)):
        out = mdl(input_ids=cur, past_key_values=past, use_cache=True)
        past = out.past_key_values
        raw_logits = out.logits[0, -1, :].float()

        # ── raw-distribution channels (the published definitions) ──
        raw_lse = torch.logsumexp(raw_logits, dim=-1)
        raw_logprobs = raw_logits - raw_lse
        raw_probs = raw_logprobs.exp()
        raw_H = float(-(raw_probs * raw_logprobs).sum())
        top_raw = torch.topk(raw_logprobs, max(cfg.logprob_top_k, cfg.conf_topk + 1))

        # ── sample from the warped distribution ──
        warped = _warp(raw_logits.clone(), cfg)
        if cfg.temperature and cfg.temperature > 1e-4:
            wprobs = torch.softmax(warped, dim=-1)
            nxt = int(torch.multinomial(wprobs, 1, generator=generator).item())
        else:
            nxt = int(torch.argmax(warped).item())
        wlogprobs = torch.log_softmax(warped, dim=-1)

        ch.raw_entropy.append(raw_H)
        ch.raw_logsumexp.append(float(raw_lse))
        ch.raw_logprob_sampled.append(float(raw_logprobs[nxt]))
        ch.raw_pmax.append(float(raw_probs.max()))
        ch.raw_margin.append(float(top_raw.values[0] - top_raw.values[1]))
        ch.spilled_energy.append(float(-raw_logprobs[nxt]))
        # DeepConf on RAW logprobs, descending top-k, sampled token NOT special-cased —
        # the `repo_main_processors` layout. The appendix variant needs a sampled-first
        # vector, which is a vLLM artefact we do not have here; the M1 equality audit is
        # what decides which variant a DeepConf run may claim.
        ch.deepconf_conf.append(conf_paper_eq2(top_raw.values[:cfg.conf_topk].cpu().numpy(),
                                               cfg.conf_topk))
        w15 = torch.topk(wlogprobs, min(15, wlogprobs.shape[-1]))
        p15 = w15.values.exp()
        ch.sampled_entropy.append(float(-(p15 * w15.values).sum()))

        if cfg.keep_top_k_arrays:
            raw_topk_ids.append(top_raw.indices[:cfg.logprob_top_k].to(torch.int32).cpu().numpy())
            raw_topk_lps.append(top_raw.values[:cfg.logprob_top_k].to(torch.float32).cpu().numpy())
            wt = torch.topk(wlogprobs, min(cfg.logprob_top_k, wlogprobs.shape[-1]))
            warp_topk_ids.append(wt.indices.to(torch.int32).cpu().numpy())
            warp_topk_lps.append(wt.values.to(torch.float32).cpu().numpy())

        out_ids.append(nxt)
        if on_token is not None:
            on_token(step, nxt, ch)

        if nxt in eos:
            stop_reason = "eos"
            break

        if stop_check is not None:
            if stop_check(detok.append(nxt), ch):
                stop_reason = "policy"
                break

        cur = torch.tensor([[nxt]], device=device)

    result = {
        "gen_token_ids": out_ids,
        "full_text": tok.decode(out_ids, skip_special_tokens=True),
        "raw_text": tok.decode(out_ids, skip_special_tokens=False),
        "channels": ch.as_dict(),
        "n_tokens": len(out_ids),
        "stop_reason": stop_reason,
    }
    # The KV cache is deliberately NOT returned: a driver that pickled this dict would
    # serialise gigabytes of cache into a shard. A forced closure re-prefills instead —
    # one forward pass over a few thousand tokens, negligible against the trace itself.
    del past
    if cfg.keep_top_k_arrays and raw_topk_ids:
        result["raw_top_k_logprobs"] = {
            "ids": np.stack(raw_topk_ids), "logprobs": np.stack(raw_topk_lps)}
        result["sampled_top_k_logprobs"] = {
            "ids": np.stack(warp_topk_ids), "logprobs": np.stack(warp_topk_lps)}
    return result


@torch.no_grad()
def score_continuation(mdl, tok, context_ids, answer_ids) -> dict:
    """Length-normalised geometric-mean likelihood of `answer_ids` given `context_ids`.

    REFRAIN Eq. 6: `Score(y|x) = exp( (1/|y|) * sum_i log p(y_i | x, y_<i) )`, evaluated on
    the answer tokens only. Teacher-forced, one forward pass, raw logits — the reward must
    not depend on the sampler that produced the answer, or the bandit would be chasing its
    own sampling noise.
    """
    if len(answer_ids) == 0:
        return {"score": float("nan"), "mean_logprob": float("nan"), "n": 0}
    device = mdl.device
    ctx = torch.as_tensor(list(context_ids), device=device)
    ans = torch.as_tensor(list(answer_ids), device=device)
    seq = torch.cat([ctx, ans]).unsqueeze(0)
    logits = mdl(input_ids=seq).logits[0].float()
    # position i predicts token i+1, so the answer's first token is predicted at len(ctx)-1
    lp = torch.log_softmax(logits[len(ctx) - 1: len(ctx) + len(ans) - 1], dim=-1)
    tok_lp = lp.gather(-1, ans.unsqueeze(-1)).squeeze(-1)
    mean_lp = float(tok_lp.mean())
    return {"score": float(np.exp(mean_lp)), "mean_logprob": mean_lp, "n": int(len(ans))}


def causal_prefix_channels(channels: dict, t: int) -> dict:
    """Truncate every channel to the first `t` tokens.

    The only sanctioned way to build a prefix feature matrix. Handoff §P1 and phase-1
    checkpoint §4.4: building the completed-trace matrix and slicing it later leaks the
    future through full-trace centering, final length, centered STFT frames and prefix
    backfill. Callers must rebuild features from this truncated dict, not slice features
    computed on the whole trace.
    """
    return {k: list(v)[:int(t)] for k, v in channels.items()}
