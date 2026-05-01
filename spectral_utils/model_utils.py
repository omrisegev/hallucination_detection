"""
Model loading and token-level entropy generation.

Supports arbitrary HuggingFace causal LMs via device_map='auto'.
Pass quantize_4bit=True for 70B-class models on a single GPU.
"""
import gc
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

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb_cfg = None
    if quantize_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
        trust_remote_code=True,
    )
    mdl.eval()
    quant_tag = " [4-bit]" if quantize_4bit else ""
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


def generate_full(mdl, tok, prompt_msg: str, temperature: float = 1.0,
                  K: int = 15, max_new_tokens: int = 512):
    """
    Generate a response and return both the decoded text and the per-token entropy trace.

    The full response (no trace/answer split) is returned. Entropy of every generated
    token is extracted from the model's internal scores.

    Args:
        mdl:            Loaded causal LM.
        tok:            Corresponding tokenizer.
        prompt_msg:     The user message (plain text, not yet chat-templated).
        temperature:    Sampling temperature.
        K:              Top-K for entropy estimation.
        max_new_tokens: Maximum response length in tokens.

    Returns:
        (full_text: str, all_entropies: list[float])
    """
    prompt = fmt_prompt(tok, prompt_msg)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tok.eos_token_id,
        )

    gen_ids   = out.sequences[0][inputs.input_ids.shape[1]:]
    full_text = tok.decode(gen_ids, skip_special_tokens=True).strip()
    all_ents  = token_entropies_from_scores(out.scores, K)
    return full_text, all_ents
