"""
Model loading and token-level entropy generation.

Supports arbitrary HuggingFace causal LMs via device_map='auto'.
Pass quantize_4bit=True for 70B-class models on a single GPU.
"""
import gc
import numpy as np
import torch
import torch.nn.functional as F


def free_memory() -> None:
    """Release Python GC memory and empty the CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(model_id: str, quantize_4bit: bool = False):
    """
    Load a HuggingFace causal LM and its tokenizer.

    Args:
        model_id:       HuggingFace model ID (e.g. 'meta-llama/Llama-3.1-8B-Instruct').
        quantize_4bit:  Use bitsandbytes 4-bit quantization. Required for 70B models on
                        a single 40 GB GPU. Slightly affects logit values but preserves
                        relative entropy ordering.
                        Ignored for pre-quantized AWQ/GPTQ model IDs — those are loaded
                        as-is since bitsandbytes and AWQ/GPTQ configs conflict.

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    is_prequantized = any(tag in model_id.lower() for tag in ("awq", "gptq"))

    kwargs = dict(attn_implementation="eager", trust_remote_code=False)

    if quantize_4bit and not is_prequantized:
        # device_map="auto" reads the pre-quantization FP16 size and may dispatch layers
        # to CPU before BNB quantizes them, causing a ValueError on 72B models.
        # device_map={"": 0} forces all layers onto GPU 0 so BNB can quantize them in-place.
        kwargs["device_map"] = {"": 0}
        # Do NOT pass dtype alongside quantization_config — bitsandbytes owns dtype
        # internally. Passing torch_dtype causes newer transformers to bypass BNB and
        # load full FP16 weights → OOM on 72B models.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        # AWQ/GPTQ weights are already quantized on disk; auto sees the true (small) size.
        kwargs["device_map"] = "auto"
        # "dtype" is the current kwarg name (transformers ≥4.50 deprecated "torch_dtype")
        kwargs["dtype"] = torch.bfloat16

    try:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        mm_tag = ""
    except (ValueError, KeyError) as e:
        # Multimodal checkpoints (Mistral3, Gemma3, ...) aren't AutoModelForCausalLM-mappable.
        # Load the image-text-to-text class and drive it text-only (generate with input_ids
        # only, no pixel_values) — the LM head still yields output_scores/output_logits, so
        # entropy/logsumexp capture is unaffected.
        from transformers import AutoModelForImageTextToText
        mm_kwargs = dict(attn_implementation="eager", trust_remote_code=False,
                         device_map="auto", dtype=torch.bfloat16)
        print(f"AutoModelForCausalLM failed ({type(e).__name__}); loading {model_id} via "
              f"AutoModelForImageTextToText (text-only)")
        mdl = AutoModelForImageTextToText.from_pretrained(model_id, **mm_kwargs)
        mm_tag = " [multimodal/text-only]"
    mdl.eval()
    quant_tag = " [AWQ]" if is_prequantized else (" [4-bit NF4]" if quantize_4bit else "")
    print(f"Loaded {model_id}{quant_tag}{mm_tag}")
    return mdl, tok


def fmt_prompt(tok, msg: str) -> str:
    """
    Apply the model's chat template to a single user message.
    Falls back to a plain <|user|>/<|assistant|> format if the tokenizer
    does not have a chat template defined.
    """
    # Try plain-string content (most models), then multimodal list-content (Gemma-3 /
    # Mistral-3 processors expect content as a list of typed parts), then a raw fallback.
    for content in (msg, [{"type": "text", "text": msg}]):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            continue
    return f"<|user|>\n{msg}\n<|assistant|>\n"


_CHAT_TURN_END_MARKERS = ("<|im_end|>", "<end_of_turn>", "<|eot_id|>")


def chat_turn_end_token_ids(tok) -> list:
    """
    Token id(s) that end an assistant turn under a ChatML-style template, beyond
    whatever the model's own generation_config.json lists as eos_token_id.

    Some "base" checkpoints (e.g. Qwen2.5-Math-1.5B) ship a full chat_template in
    tokenizer_config.json for convenience, but their generation_config.json was never
    updated to match — it still lists only the bare completion EOS (e.g. <|endoftext|>).
    generate() stops on generation_config.eos_token_id, not on whatever the chat
    template happens to close a turn with, so a model that correctly finishes its
    answer and emits <|im_end|> is not recognized as done: it keeps sampling past the
    end of its own turn into territory it was never trained to continue, which shows
    up as degenerate repetition ("Assistant\nAssistant\n...") or garbled text. Verified
    for Qwen/Qwen2.5-Math-1.5B directly against the HF Hub configs: generation_config.json
    eos_token_id=151643 (<|endoftext|> only) while tokenizer_config.json's chat_template
    ends every turn with <|im_end|> (id 151645) -- never registered as a stop condition.
    """
    ids = []
    for marker in _CHAT_TURN_END_MARKERS:
        tid = tok.convert_tokens_to_ids(marker)
        if tid is not None and tid != tok.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids


def token_entropies_from_scores(scores, K: int = 15) -> list:
    """
    Convert a sequence of HuggingFace generation scores to per-token entropy values.

    Each element of `scores` is the logit tensor for one generated token.
    Entropy is computed over the top-K probability mass to avoid log(0) issues.

    Args:
        scores: tuple of (1, vocab_size) tensors returned by model.generate with
                output_scores=True.
        K:      Number of top tokens to use for entropy estimation.

    Returns:
        List of float entropy values, one per generated token.
    """
    ents = []
    for s in scores:
        lp   = F.log_softmax(s[0], dim=-1)
        topk = torch.topk(lp, min(K, lp.shape[-1])).values
        p    = torch.exp(topk)
        p    = p / (p.sum() + 1e-12)
        ents.append(-(p * torch.log(p + 1e-12)).sum().item())
    return ents


def token_entropies_and_spilled(scores, gen_ids, K: int = 15):
    """
    Compute per-token Shannon entropy H(n) and Spilled Energy ΔE(n) simultaneously.

    ΔE(n) = -log p(x_n | x_{<n}) — the neg log-prob of the ACTUALLY SAMPLED token.
    Unlike H(n) which averages over all tokens, ΔE decouples from H when the model
    is uncertain but samples a common token (high H, low ΔE) or confident but
    generates a rare token (low H, high ΔE).

    Args:
        scores:  tuple of (1, vocab_size) tensors from model.generate output_scores=True.
        gen_ids: 1D tensor of generated token IDs (out.sequences[0][prompt_len:]).
        K:       Top-K for entropy estimation.

    Returns:
        (ents, spilled) — two lists of float, one per generated token.
    """
    ents, spilled = [], []
    for s, token_id in zip(scores, gen_ids):
        lp   = F.log_softmax(s[0], dim=-1)
        topk = torch.topk(lp, min(K, lp.shape[-1])).values
        p    = torch.exp(topk)
        p    = p / (p.sum() + 1e-12)
        ents.append(-(p * torch.log(p + 1e-12)).sum().item())
        spilled.append(-lp[token_id.item()].item())
    return ents, spilled


def extract_top_k_logprobs(scores, top_k: int = 50):
    """
    Extract the top-K log-probabilities per generated token as compact numpy arrays.

    Saving the raw top-50 lets any future feature (entropy at arbitrary K, probability
    mass, token-level confidence) be recomputed offline without re-running the model.
    The numpy-pair form is ~3.5x smaller on disk than list[list[(id, logprob)]].

    Args:
        scores: tuple of (1, vocab_size) tensors from model.generate output_scores=True.
        top_k:  Number of top entries to keep per token.

    Returns:
        {'ids': int32 array [T, K], 'logprobs': float32 array [T, K]} — row t sorted
        by descending log-prob. None if scores is empty.
    """
    if not scores:
        return None
    ids_rows, lp_rows = [], []
    for s in scores:
        lp   = F.log_softmax(s[0], dim=-1)
        topk = torch.topk(lp, min(top_k, lp.shape[-1]))
        ids_rows.append(topk.indices.cpu().numpy().astype(np.int32))
        lp_rows.append(topk.values.cpu().numpy().astype(np.float32))
    return {"ids": np.stack(ids_rows), "logprobs": np.stack(lp_rows)}


def token_entropy_full_from_logits(logits) -> list:
    """
    Compute per-token Shannon entropy over the FULL vocabulary from raw (pre-warper) logits.

    Unlike token_entropies_from_scores (a top-K estimate over the post-warp sampling
    distribution), this is the exact H_t = -sum_v p_v log p_v over the entire vocabulary of
    the model's raw predictive distribution — the definition EDIS (arXiv:2602.01288) Eq. 1
    uses. Requires out.logits (generate(output_logits=True)), not out.scores, for the same
    reason token_logsumexp_from_scores does: out.scores are temperature-scaled and
    top-k/top-p masked, so a full-vocab entropy over them would be at the wrong scale and
    missing mass.

    Args:
        logits: tuple of (1, vocab_size) RAW-logit tensors from generate(output_logits=True).

    Returns:
        List of float entropy values, one per generated token.
    """
    ents = []
    for s in logits:
        lp = F.log_softmax(s[0], dim=-1)
        p = torch.exp(lp)
        ents.append(-(p * lp).sum().item())
    return ents


def token_logsumexp_from_scores(logits) -> list:
    """
    Compute per-token logsumexp of the FULL-vocab logits (the log-partition Z_n).

    This is the blocking capture field for the energy baselines (EPR, Semantic Energy,
    Spilled Energy), which all define energy via the partition function over the ENTIRE
    vocabulary of the RAW logits (confirmed in all three papers). The raw logit of any
    token decomposes as
        logit_i = logprob_i + logsumexp(logits)
    so saving Z_n = logsumexp(logits[n]) alongside the raw top-K logprobs lets every raw
    logit — and therefore every energy quantity — be reconstructed offline without
    re-running the model.

    IMPORTANT: pass `out.logits` (from generate(output_logits=True)), NOT `out.scores`.
    `out.scores` are the post-processed generation logits — already divided by the
    sampling temperature and masked to top-k / top-p — so logsumexp(out.scores) is a
    partition over only the surviving tokens at the wrong scale, NOT the true Z_n.
    `out.logits` are the raw, pre-warper, full-vocab logits, which is what the energy
    papers use.

    Args:
        logits: tuple of (1, vocab_size) RAW-logit tensors from generate(output_logits=True).

    Returns:
        List of float log-partition values, one per generated token.
    """
    return [torch.logsumexp(s[0], dim=-1).item() for s in logits]


def _middle_last_hidden(hidden_states, layer=None):
    """
    Extract the last-token hidden state at a middle layer — the INSIDE/EigenScore
    sentence embedding (arXiv 2402.03744 uses the int(L/2) layer, last token).

    `hidden_states` is model.generate's output_hidden_states tuple: one entry per
    generated step, each a tuple over (embeddings + L transformer layers), each of
    shape (batch, seq_len_at_that_step, hidden). We take the final step's chosen layer
    at its last position — the representation of the last generated token.

    Args:
        hidden_states: out.hidden_states from generate(output_hidden_states=True).
        layer:         layer index into the per-step tuple. None -> int(L_total/2),
                       where L_total = len(per-step tuple) = num_layers + 1.

    Returns:
        float16 numpy vector [hidden_dim], or None if hidden_states is empty.
    """
    if not hidden_states:
        return None
    last_step = hidden_states[-1]            # tuple over layers for the final token
    idx = len(last_step) // 2 if layer is None else layer
    vec = last_step[idx][0, -1, :]
    return vec.float().cpu().numpy().astype(np.float16)


def generate_full(mdl, tok, prompt_msg: str, temperature: float = 1.0,
                  K: int = 15, max_new_tokens: int = 512,
                  logprob_top_k: int = 50,
                  gen_top_p: float = None, gen_top_k: int = 50,
                  repetition_penalty: float = None, no_repeat_ngram_size: int = None,
                  capture_logsumexp: bool = False,
                  capture_hidden: bool = False, hidden_layer: int = None,
                  capture_attention: bool = False, attention_top_k: int = 100,
                  capture_layer_fft: bool = False,
                  capture_full_entropy: bool = False,
                  **kwargs):
    """
    Generate a response and return text, per-token entropies, spilled energies, and char offsets.

    Args:
        mdl:            Loaded causal LM.
        tok:            Corresponding tokenizer.
        prompt_msg:     The user message (plain text, not yet chat-templated).
        temperature:    Sampling temperature. (Also accepts 'T' as kwarg.)
        K:              Top-K for entropy estimation.
        max_new_tokens: Maximum response length. (Also accepts 'max_new' as kwarg.)
        logprob_top_k:  Top-K entries to keep in 'top_k_logprobs' (0 disables — saves
                        memory/disk when raw logprobs are not needed).
        repetition_penalty:   HF generate() repetition penalty (None -> HF default/off).
                        NOT the fix for degenerate-loop generation on a "base checkpoint
                        with a chat_template" model (see chat_turn_end_token_ids) — that
                        turned out to be a missing eos_token_id, not a sampling problem.
                        This is a LogitsProcessor applied before sampling, so it is baked
                        into 'token_entropies' / 'top_k_logprobs' like temperature/top_p/
                        top_k already are (see token_logsumexp_from_scores' out.scores-vs
                        -out.logits note) — avoid it unless you specifically want a
                        distorted entropy trace; always check eos_token_id coverage first.
        no_repeat_ngram_size: HF generate() hard ban on repeating an n-gram (None -> off).
                        Same caveat as repetition_penalty. Concretely dangerous for math/
                        code generation: a hard n-gram ban forces the model off legitimate
                        repeated substrings (variable names, digits) into garbled
                        substitutes — this is what happened when it was tried as an EDIS-
                        grid fix (see cluster/presets.py PILOT FINDING), before the real
                        cause (missing eos_token_id) was found and this was reverted.

    Replication-grid capture flags (all default OFF — the GOOD_5 / spectral path is
    unchanged when every flag is False; only the listed extra key is added when True):
        capture_logsumexp:  add 'token_logsumexp' (list[float], the true full-vocab
                            log-partition Z_n per token, from RAW logits via
                            output_logits=True) AND 'top_k_logprobs_raw' (raw-logit
                            top-K, distinct from the sampling-distribution
                            'top_k_logprobs') — the blocking fields for the energy
                            baselines (EPR / Semantic Energy / Spilled Energy), which
                            need the partition over the whole vocabulary of raw logits.
        capture_hidden:     add 'hidden_middle_last' (float16 [hidden_dim]) — the INSIDE
                            /EigenScore last-token middle-layer sentence embedding.
        hidden_layer:       explicit layer index for the hidden capture (default int(L/2)).
        capture_attention:  add 'attn_lap_eigvals' (float16 [L, H, attention_top_k]) +
                            'attn_diag_logmean' (float32 [L, H]) + 'attn_lap_meta' via
                            attn_laplacian_capture() — the LapEigvals (2502.17598)
                            attention-Laplacian reducer, computed on-GPU from ONE extra
                            teacher-forced forward pass over prompt+generation (never
                            stores the raw [L,H,T,T] maps).
        attention_top_k:    eigenvalues kept per (layer, head) — 100 covers the paper's
                            whole k in {5,10,20,50,100} sweep offline.
        capture_layer_fft:  reserved for HSAD (layer-axis FFT scalars) — not yet
                            implemented; raises so a preset never silently no-ops.
        capture_full_entropy: add 'token_entropies_full' (list[float], exact full-vocab
                            Shannon entropy per token from RAW logits via
                            output_logits=True) — the EDIS paper's H_t definition, as
                            opposed to the top-K=15 'token_entropies' our own spectral
                            features use.

    Returns:
        dict with keys:
            'full_text':              str
            'token_entropies':        list[float] — Shannon entropy H(n) per token
            'token_spilled_energies': list[float] — Spilled Energy ΔE(n) per token
            'token_offsets':          list[(start_char, end_char)] from re-tokenizing full_text
                                      (length may differ from token_entropies by 1–2 tokens; callers
                                       should trim both to min(len) before slicing).
            'top_k_logprobs':         {'ids': int32 [T, K], 'logprobs': float32 [T, K]}
                                      or None when logprob_top_k=0
            'gen_token_ids':          list[int] — sampled token IDs
            'token_logsumexp':        list[float] — true full-vocab Z_n from raw logits,
                                      only when capture_logsumexp=True
            'top_k_logprobs_raw':     {'ids':int32[T,K], 'logprobs':float32[T,K]} from the
                                      RAW (pre-warper) distribution — only when
                                      capture_logsumexp=True and logprob_top_k>0
            'hidden_middle_last':     float16 [hidden_dim] — only when capture_hidden=True
            'token_entropies_full':   list[float] — full-vocab H(n), only when
                                      capture_full_entropy=True
    """
    if capture_layer_fft:
        raise NotImplementedError(
            "capture_layer_fft (HSAD layer-axis FFT reducer) is not yet implemented — "
            "see replication-grid plan U-follow-up.")

    temp = kwargs.get("T", temperature)
    max_tokens = kwargs.get("max_new", max_new_tokens)

    # Base LMs (e.g. OPT-30B) have no chat template — a few-shot prompt must go in raw,
    # not wrapped in a synthetic <|user|>/<|assistant|> frame the model never saw.
    raw_prompt = kwargs.get("raw_prompt", False)
    prompt = prompt_msg if raw_prompt else fmt_prompt(tok, prompt_msg)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    # Union the model's own generation_config eos_token_id(s) with the chat template's
    # turn-end marker(s) -- see chat_turn_end_token_ids docstring. Only additive: never
    # removes a stop condition the model already had, just plugs the common gap where a
    # base checkpoint ships a chat_template without updating generation_config.json.
    cfg_eos = getattr(mdl.generation_config, "eos_token_id", None)
    if cfg_eos is None:
        cfg_eos = tok.eos_token_id
    eos_ids = list(cfg_eos) if isinstance(cfg_eos, (list, tuple)) else [cfg_eos]
    if not raw_prompt:
        for tid in chat_turn_end_token_ids(tok):
            if tid not in eos_ids:
                eos_ids.append(tid)

    sampling = temp > 1e-4
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=sampling,
            temperature=temp if sampling else None,
            top_k=gen_top_k if sampling else None,
            top_p=gen_top_p if (sampling and gen_top_p is not None) else None,
            repetition_penalty=repetition_penalty if repetition_penalty is not None else None,
            no_repeat_ngram_size=no_repeat_ngram_size if no_repeat_ngram_size is not None else None,
            output_scores=True,
            output_logits=capture_logsumexp or capture_full_entropy,  # RAW full-vocab logits
            output_hidden_states=capture_hidden,
            return_dict_in_generate=True,
            eos_token_id=eos_ids,
            pad_token_id=tok.eos_token_id,
        )

    gen_ids   = out.sequences[0][inputs.input_ids.shape[1]:]
    full_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    all_ents, all_spilled = token_entropies_and_spilled(out.scores, gen_ids, K)
    top_k_lp  = extract_top_k_logprobs(out.scores, logprob_top_k) if logprob_top_k > 0 else None

    encoding = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets  = encoding.offset_mapping

    result = {
        "full_text":              full_text,
        "token_entropies":        all_ents,
        "token_spilled_energies": all_spilled,
        "token_offsets":          offsets,
        "top_k_logprobs":         top_k_lp,
        "gen_token_ids":          gen_ids.tolist(),
    }
    if capture_logsumexp:
        # out.logits are the RAW, pre-warper, full-vocab logits (unlike out.scores, which
        # are temperature-scaled + top-k/top-p masked). The energy baselines define energy
        # via the partition over the whole vocabulary of raw logits, so Z_n and the raw
        # top-K logprobs must come from out.logits.
        result["token_logsumexp"] = token_logsumexp_from_scores(out.logits)
        if logprob_top_k > 0:
            result["top_k_logprobs_raw"] = extract_top_k_logprobs(out.logits, logprob_top_k)
    if capture_hidden:
        result["hidden_middle_last"] = _middle_last_hidden(out.hidden_states, hidden_layer)
    if capture_full_entropy:
        result["token_entropies_full"] = token_entropy_full_from_logits(out.logits)
    if capture_attention:
        result.update(attn_laplacian_capture(mdl, out.sequences[0], top_k=attention_top_k))
        result["attn_lap_meta"] = {
            "total_len": int(out.sequences.shape[1]),
            "prompt_len": int(inputs.input_ids.shape[1]),
            "top_k": int(attention_top_k),
        }
    return result


# ── LapEigvals attention-Laplacian reducer (arXiv 2502.17598) ────────────────────
#
# Their pipeline: causal attention map A(l,h) (row-stochastic, LOWER-TRIANGULAR) ->
# out-degree-style diagonal d_ii = (sum_u a_ui) / (T - i)  (0-based i, divisor T..1 —
# their length-independence normalization) -> Laplacian L = D - A -> top-k LARGEST
# eigenvalues per (layer, head), concatenated -> PCA-512 -> logistic-regression probe
# (class_weight='balanced'). KEY SIMPLIFICATION: L inherits A's lower-triangularity,
# so eig(L) = diag(L) = d_ii - a_ii exactly (the paper's own z~ = sort(diag(L))) — no
# eigendecomposition is ever needed, which is what makes on-GPU capture cheap.
#
# We additionally store the per-(layer, head) mean of log(a_ii) — the sufficient
# statistic for LLM-Check-style unsupervised Attention Scores (the paper's unsupervised
# baseline family), so both a supervised probe AND an unsupervised attention score can
# be computed offline from the same capture.

def _attn_lap_diag_stats(layer_att, top_k: int):
    """LapEigvals eigenvalues + diag-log stat for ONE layer's attention probs.

    layer_att: [1, H, T, T] row-stochastic causal attention (any device/dtype).
    Returns (eigvals float16 [H, top_k] CPU — NaN-padded when T < top_k,
             diag_logmean float32 [H] CPU).
    """
    att = layer_att[0].float()                              # [H, T, T]
    H, T, _ = att.shape
    colsum = att.sum(dim=1)                                 # [H, T]: sum_u a_ui
    denom = (T - torch.arange(T, device=att.device, dtype=att.dtype))   # T..1
    diag = torch.diagonal(att, dim1=1, dim2=2)              # [H, T]: a_ii
    eig = colsum / denom - diag                             # diag(L) = eig(L)
    eig = torch.sort(eig, dim=1, descending=True).values
    k = min(top_k, T)
    out = torch.full((H, top_k), float("nan"), dtype=torch.float16)
    out[:, :k] = eig[:, :k].to(torch.float16).cpu()
    diag_logmean = torch.log(diag.clamp_min(1e-12)).mean(dim=1).float().cpu()
    return out, diag_logmean


def attn_laplacian_capture(mdl, full_ids, top_k: int = 100) -> dict:
    """One teacher-forced forward pass over the full sequence (prompt + generation)
    with output_attentions=True, reduced on-GPU to the LapEigvals features.

    SDPA attention falls back to eager for this call (transformers does this
    automatically when output_attentions=True); flash-attention-2 loads would raise —
    the project's load_model uses the default (sdpa) implementation. Peak memory is
    the eager maps for all layers at once (~[L,H,T,T] float32: ~20 GB at T≈2200 for an
    8B model) — sized for the B200 cluster nodes, not Colab T4s.

    Returns {'attn_lap_eigvals': float16 [L, H, top_k],
             'attn_diag_logmean': float32 [L, H]}.
    """
    import numpy as _np
    ids = full_ids if full_ids.dim() == 2 else full_ids.unsqueeze(0)
    with torch.no_grad():
        fwd = mdl(input_ids=ids.to(mdl.device), output_attentions=True, use_cache=False)
    eig_list, dlm_list = [], []
    for layer_att in fwd.attentions:
        e, d = _attn_lap_diag_stats(layer_att, top_k)
        eig_list.append(e.numpy())
        dlm_list.append(d.numpy())
    del fwd
    return {"attn_lap_eigvals": _np.stack(eig_list),        # [L, H, top_k] float16
            "attn_diag_logmean": _np.stack(dlm_list)}       # [L, H] float32
