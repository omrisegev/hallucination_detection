"""
prm_scorer.py — Qwen2.5-Math-PRM-7B step-reward scoring (Reasoning localization: supervised
PRM ceiling, docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md item 3).

Paper: "Towards Effective Process Supervision in Mathematical Reasoning" (Qwen team; model card
and blog at https://qwenlm.github.io/blog/qwen2.5-math-prm/, checkpoint
`Qwen/Qwen2.5-Math-PRM-7B`). This is a PUBLISHED, HUMAN-LABEL-TRAINED process reward model — a
supervised ceiling, not a peer of our label-free gray-box scoring. Report it in its own category
(docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md's "Fair comparison
categories" table), never beside U-PCR/DUFS-LIU as if it used the same information/supervision.

USAGE REPRODUCED VERBATIM FROM THE MODEL CARD
-----------------------------------------------
`AutoModel` (NOT `AutoModelForCausalLM`) with `trust_remote_code=True` — this is a custom
classification head over the Qwen2.5-Math-7B-Instruct backbone, fetched 2026-08-10 from
huggingface.co/Qwen/Qwen2.5-Math-PRM-7B's README.md. Steps are joined with the literal string
`"<extra_0>"` (including a TRAILING one), so a `len(steps)`-step response has exactly
`len(steps)` occurrences of that token — one reward per step, in order. `make_step_rewards` is
reproduced unchanged (do not "clean up" the boolean-mask reshape; a different reduction order
here silently permutes which score belongs to which step).
"""
import numpy as np

PRM_MODEL_ID = "Qwen/Qwen2.5-Math-PRM-7B"

# The model card's own system prompt — part of the published protocol, not our choice.
PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def _patch_cache_compat():
    """The checkpoint's own `modeling_qwen2_rm.py` (trust_remote_code) calls
    `past_key_value.get_usable_length(kv_seq_len, layer_idx)` — an API from an older
    transformers version. This cluster's transformers renamed it to `get_seq_length(layer_idx)`
    (job 176541 failed on exactly this: `AttributeError: 'DynamicCache' object has no attribute
    'get_usable_length'`). We always call this model with a single, cache-free forward pass, so
    the old method only ever needs to report "how many tokens are already cached for this layer"
    for an empty cache — which is exactly what `get_seq_length` already computes. A narrow alias,
    not a version pin: the third-party checkpoint's code is frozen, we cannot edit it, and the
    project's own convention (see the `check_torch_load_is_safe` shim in every cluster driver) is
    a targeted monkeypatch over touching pinned NGC-container package versions."""
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = lambda self, new_seq_length, layer_idx=0: (
            self.get_seq_length(layer_idx)
        )


def load_prm_model(model_id: str = PRM_MODEL_ID, attn_impl: str = "sdpa"):
    """Load the PRM checkpoint + tokenizer. `AutoModel`, not `AutoModelForCausalLM` — this
    model's forward pass returns per-token 2-way classification logits, not vocab logits."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    _patch_cache_compat()
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    try:
        mdl = AutoModel.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16,
            trust_remote_code=True, attn_implementation=attn_impl,
        ).eval()
    except TypeError:
        # Some trust_remote_code classes don't forward attn_implementation to their backbone —
        # fall back to the model card's own exact call rather than fail the load over this.
        mdl = AutoModel.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).eval()
    return mdl, tok


def make_step_rewards(logits, token_masks):
    """Verbatim from the model card (variable names kept identical for auditability)."""
    import torch.nn.functional as F
    probabilities = F.softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1)  # bs, seq_len, num_labels

    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i]  # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1]  # valid_tokens, num_labels
        all_scores_res.append(positive_probs.cpu().tolist())
    return all_scores_res


def score_steps(mdl, tok, problem: str, steps: list) -> list:
    """One row -> one reward per step, via the model card's exact 3-turn chat template."""
    import torch

    response = "<extra_0>".join(steps) + "<extra_0>"
    messages = [
        {"role": "system", "content": PRM_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
        {"role": "assistant", "content": response},
    ]
    conversation_str = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    input_ids = tok.encode(conversation_str, return_tensors="pt").to(mdl.device)

    with torch.no_grad():
        outputs = mdl(input_ids=input_ids)

    step_sep_id = tok.encode("<extra_0>")[0]
    token_masks = (input_ids == step_sep_id)
    rewards = make_step_rewards(outputs[0], token_masks)[0]
    if len(rewards) != len(steps):
        raise ValueError(
            f"expected {len(steps)} step rewards (one per '<extra_0>'), got {len(rewards)} — "
            "a tokenizer/template change likely broke the one-token-per-separator assumption"
        )
    return rewards


def localize_first_error(rewards: list, threshold: float = 0.5) -> int:
    """First step whose positive-class probability drops below `threshold`, else NO_ERROR
    (-1). This is the standard PRM decision rule ProcessBench itself uses for PRM baselines:
    a step is "wrong" once the model's own step-correctness probability crosses 0.5."""
    for i, r in enumerate(rewards):
        if r < threshold:
            return i
    return -1


def smoke() -> None:
    # make_step_rewards is exercised indirectly via score_steps in the real driver (needs a GPU
    # + the real checkpoint); this smoke test covers the pure-Python decision rule, which is
    # exactly the piece a refactor could silently break without any GPU involved.
    assert localize_first_error([1.0, 0.9, 0.98, 1.0]) == -1
    assert localize_first_error([1.0, 0.19, 0.97, 1.0]) == 1
    assert localize_first_error([0.4, 0.9, 0.9]) == 0
    assert localize_first_error([0.5, 0.9]) == -1, "exactly 0.5 must NOT count as below threshold"
    assert localize_first_error([]) == -1
    assert localize_first_error([1.0, 0.9, 0.98, 1.0], threshold=0.95) == 1
    print("prm_scorer.smoke: PASS (6 checks)")


if __name__ == "__main__":
    smoke()
