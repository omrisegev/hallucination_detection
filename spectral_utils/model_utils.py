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

    mdl = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    mdl.eval()
    quant_tag = " [AWQ]" if is_prequantized else (" [4-bit NF4]" if quantize_4bit else "")
    print(f"Loaded {model_id}{quant_tag}")
    return mdl, tok


def fmt_prompt(tok, msg: str) -> str:
    """
    Apply the model's chat template to a single user message.
    Falls back to a plain <|user|>/<|assistant|> format if the tokenizer
    does not have a chat template defined.
    """
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        return f"<|user|>\n{msg}\n<|assistant|>\n"


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


def topk_logprobs_from_scores(scores, K: int = 50):
    """
    Compact top-K log-probabilities per generated token.

    Args:
        scores: tuple of (1, vocab_size) tensors from model.generate output_scores=True.
        K:      number of top tokens to keep per step.

    Returns:
        {'ids': np.int32 [T, K], 'logprobs': np.float16 [T, K]}

    Two dense arrays instead of the list-of-(token_id, logprob) tuples form:
    ~10x smaller pickles at K=50 over long traces.  float16 keeps ~3 decimal
    digits — more than enough to recompute entropy at any K' ≤ K or top-K
    probability-mass features.
    """
    ids_rows, lp_rows = [], []
    for s in scores:
        lp = F.log_softmax(s[0].float(), dim=-1)
        top = torch.topk(lp, min(K, lp.shape[-1]))
        ids_rows.append(top.indices.to(torch.int32).cpu().numpy())
        lp_rows.append(top.values.cpu().numpy().astype(np.float16))
    return {"ids": np.stack(ids_rows), "logprobs": np.stack(lp_rows)}


def generate_full(mdl, tok, prompt_msg: str, temperature: float = 1.0,
                  K: int = 15, max_new_tokens: int = 512,
                  top_k_logprobs: int = 0, **kwargs):
    """
    Generate a response and return text, per-token entropies, spilled energies, and char offsets.

    Args:
        mdl:            Loaded causal LM.
        tok:            Corresponding tokenizer.
        prompt_msg:     The user message (plain text, not yet chat-templated).
        temperature:    Sampling temperature. (Also accepts 'T' as kwarg.)
        K:              Top-K for entropy estimation.
        max_new_tokens: Maximum response length. (Also accepts 'max_new' as kwarg.)
        top_k_logprobs: When > 0, also save the top-K log-probabilities per token
                        (compact arrays, see topk_logprobs_from_scores). Use 50
                        for raw-data-rule caches; 0 (default) keeps the old
                        return schema and cost.

    Returns:
        dict with keys:
            'full_text':              str
            'token_entropies':        list[float] — Shannon entropy H(n) per token
            'token_spilled_energies': list[float] — Spilled Energy ΔE(n) per token
            'token_offsets':          list[(start_char, end_char)] from re-tokenizing full_text
                                      (length may differ from token_entropies by 1–2 tokens; callers
                                       should trim both to min(len) before slicing).
            'gen_token_ids':          list[int] — sampled token IDs (needed to recompute
                                      ΔE or attention features later).
            'top_k_logprobs':         only when top_k_logprobs > 0 —
                                      {'ids': int32 [T,K], 'logprobs': float16 [T,K]}.
    """
    temp = kwargs.get("T", temperature)
    max_tokens = kwargs.get("max_new", max_new_tokens)

    prompt = fmt_prompt(tok, prompt_msg)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True if temp > 1e-4 else False,
            temperature=temp if temp > 1e-4 else None,
            top_k=50 if temp > 1e-4 else None,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )

    gen_ids   = out.sequences[0][inputs.input_ids.shape[1]:]
    full_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    all_ents, all_spilled = token_entropies_and_spilled(out.scores, gen_ids, K)

    encoding = tok(full_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets  = encoding.offset_mapping

    result = {
        "full_text":              full_text,
        "token_entropies":        all_ents,
        "token_spilled_energies": all_spilled,
        "token_offsets":          offsets,
        "gen_token_ids":          gen_ids.tolist(),
    }
    if top_k_logprobs > 0:
        result["top_k_logprobs"] = topk_logprobs_from_scores(out.scores, top_k_logprobs)
    return result
