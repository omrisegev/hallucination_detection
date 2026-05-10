"""
Baseline hallucination detectors used for comparison against spectral Nadler fusion.

These are simplified, single-file implementations meant for in-notebook benchmarking.
For published-quality baseline implementations, see the original papers.
"""
import math
import re
from typing import List, Optional

import numpy as np
import torch


# ── Lite Semantic Entropy ─────────────────────────────────────────────────────
#
# Reference: Kuhn, Gal & Farquhar (2023) "Semantic uncertainty: Linguistic
# invariances for uncertainty estimation in natural language generation".
#
# The original protocol uses bidirectional NLI entailment to cluster k MC samples
# into semantic equivalence classes, then takes entropy over the cluster sizes.
# The "lite" variant here replaces NLI with a token-overlap (Jaccard) clustering
# threshold. This is faster (no second model needed) and reasonable for short
# statement-level continuations where lexical overlap correlates with semantic
# equivalence. Kuhn et al. report this as a useful weak baseline.

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize_words(text: str) -> set:
    return {w.lower() for w in _TOKEN_RE.findall(text)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _cluster_by_jaccard(samples: List[str], threshold: float = 0.5) -> List[int]:
    """
    Greedy single-link clustering of strings by Jaccard token overlap.
    Two samples join the same cluster if their Jaccard >= threshold.
    Returns a cluster id per sample (in input order).
    """
    word_sets = [_tokenize_words(s) for s in samples]
    cluster_ids = [-1] * len(samples)
    next_id = 0
    for i, ws in enumerate(word_sets):
        for j in range(i):
            if cluster_ids[j] != -1 and _jaccard(ws, word_sets[j]) >= threshold:
                cluster_ids[i] = cluster_ids[j]
                break
        if cluster_ids[i] == -1:
            cluster_ids[i] = next_id
            next_id += 1
    return cluster_ids


def _entropy_from_cluster_ids(ids: List[int]) -> float:
    """Shannon entropy (nats) over the cluster size distribution."""
    if not ids:
        return 0.0
    counts: dict = {}
    for c in ids:
        counts[c] = counts.get(c, 0) + 1
    total = sum(counts.values())
    h = 0.0
    for n in counts.values():
        p = n / total
        if p > 0:
            h -= p * math.log(p + 1e-12)
    return h


def lite_semantic_entropy_for_statement(
    mdl,
    tok,
    prompt_text: str,
    statement_token_start_global: int,
    n_continuation_tokens: int = 32,
    k: int = 10,
    temperature: float = 1.0,
    jaccard_threshold: float = 0.5,
    seed: Optional[int] = None,
) -> float:
    """
    Compute lite (token-overlap) semantic entropy for a single statement.

    The model is re-prompted with the original prompt + the prefix of generation
    up to where the statement begins. From that point, k continuations are
    sampled. The continuations are clustered by Jaccard overlap and the entropy
    over cluster sizes is returned.

    Higher entropy → more divergent regenerations → higher hallucination signal.

    Args:
        mdl:                          Loaded causal LM (must be on a device).
        tok:                          Corresponding tokenizer.
        prompt_text:                  The full input prompt + the generation prefix
                                      up to (but not including) the statement to test.
                                      Caller is responsible for slicing the prefix.
        statement_token_start_global: Unused — kept in signature for symmetry with
                                      richer SE protocols. The caller already trimmed
                                      `prompt_text` to the regeneration point.
        n_continuation_tokens:        How many tokens to generate per MC sample.
        k:                            Number of MC samples.
        temperature:                  Sampling temperature.
        jaccard_threshold:            Cluster two samples together if their token
                                      Jaccard overlap >= threshold.
        seed:                         If given, set torch + numpy seed for
                                      reproducible MC sampling.

    Returns:
        Semantic entropy in nats. 0.0 if all k samples cluster together.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    inputs = tok(prompt_text, return_tensors="pt").to(mdl.device)
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]

    completions: List[str] = []
    with torch.no_grad():
        for _ in range(k):
            out = mdl.generate(
                **inputs,
                max_new_tokens=n_continuation_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=50,
                pad_token_id=tok.eos_token_id,
            )
            gen_ids = out[0][inputs.input_ids.shape[1]:]
            completions.append(tok.decode(gen_ids, skip_special_tokens=True).strip())

    cluster_ids = _cluster_by_jaccard(completions, threshold=jaccard_threshold)
    return _entropy_from_cluster_ids(cluster_ids)


# ── Mean negative log-probability (perplexity-style) ─────────────────────────
#
# Cheap baseline: average -log P(token) across the statement's tokens.
# Higher = more uncertain = more likely hallucinated. This is what most early
# hallucination-detection papers used as the "perplexity" reference.

def mean_neg_logprob_baseline(token_entropies: List[float]) -> float:
    """
    Mean per-token entropy across the statement's tokens.

    Note: this is *entropy* (already computed in our pipeline), not the
    log-probability of the realised tokens. They differ in direction but
    correlate strongly. For a strict perplexity baseline you would need
    the realised-token log-probs (see HuggingFace `compute_transition_scores`).
    """
    if not token_entropies:
        return float("nan")
    return float(np.mean(token_entropies))
