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


# ── Official Semantic Entropy (Farquhar et al., Nature 2024) ──────────────────
#
# Reference: Farquhar, Kossen, Kuhn & Gal (2024) "Detecting hallucinations in
# large language models using semantic consistency", Nature.
#
# Protocol: generate K samples, cluster by bidirectional NLI entailment
# (i entails j AND j entails i → same cluster), compute Shannon entropy over
# cluster sizes. Higher entropy = more semantic diversity = higher hallucination risk.
# NLI model: cross-encoder/nli-deberta-v3-base (or -large for paper accuracy).

_NLI_LABEL_ORDERS: dict = {}  # cached per model_name


def nli_load_model(model_name: str = "cross-encoder/nli-deberta-v3-base",
                   device: str = "cuda",
                   cache_dir: str = None):
    """
    Load an NLI model and tokenizer for semantic entropy and SelfCheckGPT.

    Returns (model, tokenizer, device) tuple.
    Caches label-order metadata in module-level dict.

    Args:
        cache_dir: Optional local directory for model weights.  Pass a path on
                   fast local storage (e.g. '/content/nli_cache') when HF_HOME
                   points to a Drive FUSE mount — Drive does not support
                   os.sendfile(), which HuggingFace uses during the initial copy,
                   causing OSError [Errno 5] Input/output error.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    kw = {"cache_dir": cache_dir} if cache_dir is not None else {}
    tok = AutoTokenizer.from_pretrained(model_name, **kw)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name, **kw)
    mdl.to(device).eval()

    # Build label→index mapping so nli_classify works for any HF NLI model.
    id2label = getattr(mdl.config, "id2label", {})
    label2id = {v.lower(): int(k) for k, v in id2label.items()}
    # Normalise: handle 'entailment'/'ENTAILMENT', 'contradiction'/'CONTRADICTION'
    _NLI_LABEL_ORDERS[model_name] = label2id
    return mdl, tok, device


def nli_classify(
    premise: str,
    hypothesis: str,
    nli_model,
    nli_tokenizer,
    device: str = "cuda",
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> str:
    """
    Classify NLI relationship between premise and hypothesis.
    Returns 'entailment', 'neutral', or 'contradiction'.
    """
    inputs = nli_tokenizer(
        premise, hypothesis,
        return_tensors="pt", truncation=True, max_length=512,
    ).to(device)
    with torch.no_grad():
        logits = nli_model(**inputs).logits[0]  # (num_labels,)

    label_idx = int(logits.argmax().item())
    id2label = getattr(nli_model.config, "id2label", {})
    label_raw = id2label.get(label_idx, str(label_idx)).lower()
    # Normalise variant spellings
    if "entail" in label_raw:
        return "entailment"
    if "contradict" in label_raw:
        return "contradiction"
    return "neutral"


def _build_nli_clusters(
    samples: List[str],
    nli_model,
    nli_tokenizer,
    device: str = "cuda",
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> List[int]:
    """
    Build semantic equivalence classes by bidirectional NLI entailment.
    Uses greedy union-find: merge cluster of j into cluster of i if i↔j both entail.
    """
    n = len(samples)
    cluster_ids = list(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if cluster_ids[i] == cluster_ids[j]:
                continue
            rel_ij = nli_classify(samples[i], samples[j], nli_model, nli_tokenizer, device, model_name)
            rel_ji = nli_classify(samples[j], samples[i], nli_model, nli_tokenizer, device, model_name)
            if rel_ij == "entailment" and rel_ji == "entailment":
                old_id = cluster_ids[j]
                new_id = cluster_ids[i]
                for k in range(n):
                    if cluster_ids[k] == old_id:
                        cluster_ids[k] = new_id
    return cluster_ids


def official_semantic_entropy(
    samples: List[str],
    nli_model,
    nli_tokenizer,
    device: str = "cuda",
    model_name: str = "cross-encoder/nli-deberta-v3-base",
) -> float:
    """
    Compute Official Semantic Entropy (Farquhar et al., Nature 2024).

    Args:
        samples: K text completions for the same question.
        nli_model: AutoModelForSequenceClassification for NLI (e.g. DeBERTa-v3).
        nli_tokenizer: Matching tokenizer.
        device: torch device string.
        model_name: Used for label-order lookup.

    Returns:
        Semantic entropy in nats. Higher → more semantic uncertainty → more likely hallucinated.
    """
    if len(samples) < 2:
        return 0.0
    cluster_ids = _build_nli_clusters(samples, nli_model, nli_tokenizer, device, model_name)
    return _entropy_from_cluster_ids(cluster_ids)


# ── Self-Consistency (Wang et al., 2023) ─────────────────────────────────────
#
# Reference: Wang et al. (2023) "Self-Consistency Improves Chain of Thought
# Reasoning in Language Models". ICLR 2023.
#
# Black-box method (text only). Generate K samples, take majority vote answer,
# return fraction of samples agreeing with majority. Higher = more consistent =
# less likely hallucinated.

def self_consistency_score(
    answers: List[Optional[str]],
    normalize_fn=None,
) -> float:
    """
    Self-Consistency score: fraction of K answers agreeing with the majority.

    Args:
        answers: K extracted answer strings (None for extraction failures).
        normalize_fn: Optional function mapping raw answer → canonical form.
                      If None, strips and lowercases.

    Returns:
        Fraction in [0, 1]. Higher = more consistent = less likely hallucinated.
        Returns NaN if fewer than 2 non-None answers.
    """
    valid = [a for a in answers if a is not None]
    if len(valid) < 2:
        return float("nan")

    if normalize_fn is not None:
        normalized = [normalize_fn(a) for a in valid]
    else:
        normalized = [str(a).strip().lower() for a in valid]

    counts: dict = {}
    for a in normalized:
        counts[a] = counts.get(a, 0) + 1

    if not counts:
        return float("nan")

    majority_count = max(counts.values())
    return majority_count / len(answers)  # denominator = total K (including None)


# ── SelfCheckGPT (Manakul et al., ICLR 2023) ─────────────────────────────────
#
# Reference: Manakul, Liusie & Gales (2023) "SelfCheckGPT: Zero-Resource
# Black-Box Hallucination Detection for Generative Large Language Models".
# EMNLP 2023.
#
# Checks consistency of each sentence in the main response against K alternative
# samples via NLI. Contradiction fraction = proxy for hallucination probability.

_SENT_RE = re.compile(r"[^.!?]+[.!?]+")


def _split_sentences(text: str) -> List[str]:
    """Simple sentence splitter; falls back to the full text as one sentence."""
    sents = [s.strip() for s in _SENT_RE.findall(text) if len(s.strip()) > 10]
    return sents if sents else [text.strip()]


def selfcheck_nli_score(
    main_text: str,
    sample_responses: List[str],
    nli_model,
    nli_tokenizer,
    device: str = "cuda",
    model_name: str = "cross-encoder/nli-deberta-v3-base",
    sentence_splitter=None,
) -> float:
    """
    SelfCheckGPT (NLI variant, Manakul et al. 2023).

    For each sentence in main_text, checks how many of the K sample_responses
    *contradict* it. The response-level score = mean over sentences.

    Higher score → more contradictions → more likely hallucinated.

    Args:
        main_text: Primary generated response.
        sample_responses: K alternative samples from the same prompt.
        nli_model / nli_tokenizer: NLI model (DeBERTa or equivalent).
        device: torch device.
        model_name: Used for label lookup.
        sentence_splitter: Optional callable(str) → List[str].

    Returns:
        Mean per-sentence contradiction fraction. NaN if no sentences or no samples.
    """
    if not main_text or not sample_responses:
        return float("nan")

    splitter = sentence_splitter if sentence_splitter is not None else _split_sentences
    sentences = splitter(main_text)

    if not sentences:
        return float("nan")

    sentence_scores = []
    for sent in sentences:
        contradiction_count = 0
        for sample in sample_responses:
            rel = nli_classify(sample, sent, nli_model, nli_tokenizer, device, model_name)
            if rel == "contradiction":
                contradiction_count += 1
        sentence_scores.append(contradiction_count / len(sample_responses))

    return float(np.mean(sentence_scores))


# ── Verbalized Confidence ─────────────────────────────────────────────────────
#
# Black-box 1-pass baseline: prompt model for a 0-100 confidence score after
# generating its answer. Calibration may be poor due to RLHF alignment (models
# tend to overstate confidence). Lower score = more likely hallucinated.

VERBALIZED_CONF_SUFFIX = (
    "\n\nOn a scale from 0 to 100, how confident are you that "
    "your previous answer is correct? Answer with a single integer only."
)

_CONF_LABEL_RE = re.compile(r"[Cc]onfidence\s*:?\s*(\d{1,3})\b")
_CONF_ANY_RE   = re.compile(r"\b([0-9]{1,3})\b")


def parse_verbalized_confidence(text: str) -> float:
    """
    Extract confidence score (0-100) from model text output.

    Strategy (in order):
    1. Look for explicit label "Confidence: X" (handles 1-pass where the
       response contains math numbers before the confidence line).
    2. Fall back to the LAST standalone integer in [0, 100] — confidence
       is always at the end of the response, math numbers come first.
    3. Return NaN if nothing found.

    Returns value / 100.0 so the result is in [0, 1].
    Higher = more confident = less likely hallucinated.
    """
    # 1. Explicit label match
    m = _CONF_LABEL_RE.search(text)
    if m:
        v = int(m.group(1))
        if 0 <= v <= 100:
            return v / 100.0

    # 2. Last integer in [0, 100]
    last_val = None
    for m in _CONF_ANY_RE.finditer(text):
        v = int(m.group(1))
        if 0 <= v <= 100:
            last_val = v
    if last_val is not None:
        return last_val / 100.0

    return float("nan")
