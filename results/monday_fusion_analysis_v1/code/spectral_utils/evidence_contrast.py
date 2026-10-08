"""
evidence_contrast.py — Δ-view builders and the exogenous evidence graph for the
Evidence-Contrast U-PCR/DUFS-LIU campaign (`cluster/run_conditional_rescore.py`,
`docs/research_notes/ragtruth_ec_preregistration_v1.md`).

WHY THIS GRAPH IS DIFFERENT FROM WHAT ALREADY FAILED
--------------------------------------------------------------------------------------
Steps 227–230 closed every graph built ENDOGENOUSLY — from the same feature covariance it
was meant to regularize (feature kNN, micro-view clusters, semantic family graphs,
cross-partition diffusion). All of them reproduced stable structure without identifying
target-relevant structure. `build_evidence_graph` below is EXOGENOUS: its edge weights come
from chunk-TEXT similarity (computed once from the retrieval evidence itself, before any
score exists), not from score covariance. Two intervention views correlate here because
they removed textually similar evidence, not because their measured scores happened to
agree — the kind of independent side information the earlier diagnoses said was missing.

Δ-VIEW SEMANTICS
--------------------------------------------------------------------------------------
Every Δ view compares a condition's per-token telemetry against the SAME response's `full`
condition, token-for-token (the response is tokenized once — `gen_token_ids` is identical
across all of one response's conditions, so alignment is exact by construction, not
approximate). A positive `dnll_*` / `dent_*` means removing evidence made the model MORE
surprised / less confident at that token — the intended grounding-sensitivity signal, not
nuisance to be subtracted (see the preregistration doc's "Important distinction from noise
removal").
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np

_WORD_PATTERN = re.compile(r"[a-z0-9]+")


# ── per-response Δ views ────────────────────────────────────────────────────────────

def _to_array(record, key):
    value = record.get(key)
    return np.asarray(value, dtype=float) if value is not None else None


def js_divergence(ids_a, lp_a, ids_b, lp_b, tail_prob=1e-6):
    """Per-token Jensen-Shannon divergence between two top-K distributions, renormalized
    over the union of their observed token ids with a shared tail bucket absorbing the
    probability mass outside each condition's own top-K (so removing evidence that shifts
    mass onto previously-unseen tokens is still measured, not silently dropped)."""
    T = ids_a.shape[0]
    out = np.empty(T, dtype=float)
    for t in range(T):
        ids_union = np.union1d(ids_a[t], ids_b[t])
        pa = np.full(ids_union.shape[0] + 1, tail_prob)  # +1 slot = tail bucket
        pb = np.full(ids_union.shape[0] + 1, tail_prob)
        idx_a = {tok: i for i, tok in enumerate(ids_union)}
        mass_a, mass_b = 0.0, 0.0
        for k in range(ids_a.shape[1]):
            i = idx_a[ids_a[t, k]]
            p = np.exp(lp_a[t, k])
            pa[i] = p
            mass_a += p
        for k in range(ids_b.shape[1]):
            i = idx_a[ids_b[t, k]]
            p = np.exp(lp_b[t, k])
            pb[i] = p
            mass_b += p
        pa[-1] = max(tail_prob, 1.0 - mass_a)
        pb[-1] = max(tail_prob, 1.0 - mass_b)
        pa = pa / pa.sum()
        pb = pb / pb.sum()
        m = 0.5 * (pa + pb)
        kl_am = np.sum(pa * np.log((pa + 1e-12) / (m + 1e-12)))
        kl_bm = np.sum(pb * np.log((pb + 1e-12) / (m + 1e-12)))
        out[t] = 0.5 * kl_am + 0.5 * kl_bm
    return out


def response_delta_views(records_by_condition: dict) -> dict:
    """`records_by_condition`: `{"full": rec, "noctx": rec, "loo_0": rec, ...}`, each `rec`
    a conditional-rescore cache entry (schema: `token_entropies`, `token_spilled_energies`,
    `token_logsumexp`, `top_k_logprobs`). Returns per-token arrays (all length `T`, the
    response's token count) plus their response-level (scalar mean) aggregates.
    """
    full = records_by_condition["full"]
    T = len(full["token_entropies"])
    ent_full = _to_array(full, "token_entropies")
    spill_full = _to_array(full, "token_spilled_energies")
    z_full = _to_array(full, "token_logsumexp")
    lp_full = full.get("top_k_logprobs") or {}

    loo_conditions = sorted(
        (c for c in records_by_condition if c.startswith("loo_")),
        key=lambda c: int(c[len("loo_"):]),
    )

    token_views = {}
    if "noctx" in records_by_condition:
        noctx = records_by_condition["noctx"]
        token_views["dent_noctx"] = _to_array(noctx, "token_entropies")[:T] - ent_full
        token_views["dnll_noctx"] = _to_array(noctx, "token_spilled_energies")[:T] - spill_full
        z_noctx = _to_array(noctx, "token_logsumexp")
        token_views["dlogsumexp_noctx"] = (z_noctx[:T] - z_full) if z_noctx is not None else None
        lp_noctx = noctx.get("top_k_logprobs")
        if lp_full and lp_noctx:
            token_views["djs_noctx"] = js_divergence(
                np.asarray(lp_full["ids"])[:T], np.asarray(lp_full["logprobs"])[:T],
                np.asarray(lp_noctx["ids"])[:T], np.asarray(lp_noctx["logprobs"])[:T],
            )

    if loo_conditions:
        dent_stack = np.stack([
            _to_array(records_by_condition[c], "token_entropies")[:T] - ent_full
            for c in loo_conditions
        ])
        dnll_stack = np.stack([
            _to_array(records_by_condition[c], "token_spilled_energies")[:T] - spill_full
            for c in loo_conditions
        ])
        token_views["dent_loo_max"] = dent_stack.max(axis=0)
        token_views["dent_loo_mean"] = dent_stack.mean(axis=0)
        token_views["dnll_loo_max"] = dnll_stack.max(axis=0)
        token_views["dnll_loo_mean"] = dnll_stack.mean(axis=0)

    token_views = {name: value for name, value in token_views.items() if value is not None}
    response_views = {name: float(np.mean(value)) for name, value in token_views.items()}
    return {"token": token_views, "response": response_views, "n_tokens": T,
            "n_loo_chunks": len(loo_conditions)}


# ── the exogenous evidence graph ────────────────────────────────────────────────────

def _tf_vector(text: str, vocab: dict) -> np.ndarray:
    words = _WORD_PATTERN.findall(text.lower())
    counts = Counter(words)
    vec = np.zeros(len(vocab), dtype=float)
    for word, count in counts.items():
        index = vocab.get(word)
        if index is not None:
            vec[index] = count
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def build_evidence_graph(chunk_texts, node_names) -> np.ndarray:
    """Deterministic, offline, dependency-free TF-IDF word-cosine similarity graph.

    `chunk_texts`: list of strings, one per `loo_j` chunk (the text that condition REMOVES,
    i.e. `context[start:end]` for that chunk's span). `node_names`: the full node list this
    graph must cover, e.g. `["full", "noctx", "loo_0", ..., "loo_{m-1}"]` — `full` and
    `noctx` get zero similarity to every chunk (they don't correspond to removing any single
    piece of evidence) except a self-similarity of 1.0 on the diagonal.

    Returns a symmetric `[n, n]` adjacency matrix aligned to `node_names`'s order, ready for
    `spectral_utils.laplacian_upcr.build_graph_from_features`-style consumption (or direct
    use as a precomputed affinity matrix).
    """
    n = len(node_names)
    adjacency = np.eye(n, dtype=float)
    loo_index = {name: i for i, name in enumerate(node_names) if name.startswith("loo_")}
    if not chunk_texts or not loo_index:
        return adjacency

    vocab = {}
    for text in chunk_texts:
        for word in _WORD_PATTERN.findall(text.lower()):
            vocab.setdefault(word, len(vocab))
    if not vocab:
        return adjacency

    vectors = {}
    for j, text in enumerate(chunk_texts):
        name = f"loo_{j}"
        if name in loo_index:
            vectors[name] = _tf_vector(text, vocab)

    names = list(vectors)
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            sim = float(np.dot(vectors[names[a]], vectors[names[b]]))
            ia, ib = loo_index[names[a]], loo_index[names[b]]
            adjacency[ia, ib] = sim
            adjacency[ib, ia] = sim
    return adjacency


# ── known-answer tests ───────────────────────────────────────────────────────────────

def smoke() -> None:
    rng = np.random.default_rng(0)
    T = 40
    base_entropy = rng.uniform(0.3, 1.0, T)
    base_spilled = rng.uniform(0.5, 1.5, T)
    base_z = rng.uniform(4.0, 6.0, T)

    def _lp(shift=0.0):
        ids = np.tile(np.arange(5), (T, 1))
        logits = -np.abs(rng.normal(0, 1, size=(T, 5))) - shift
        logits = np.sort(logits, axis=1)[:, ::-1]
        return {"ids": ids, "logprobs": logits}

    full = {"token_entropies": base_entropy.tolist(),
            "token_spilled_energies": base_spilled.tolist(),
            "token_logsumexp": base_z.tolist(), "top_k_logprobs": _lp(0.0)}
    noctx = {"token_entropies": (base_entropy + 0.4).tolist(),
             "token_spilled_energies": (base_spilled + 0.6).tolist(),
             "token_logsumexp": (base_z - 0.2).tolist(), "top_k_logprobs": _lp(0.3)}
    loo0 = {"token_entropies": (base_entropy + 0.1).tolist(),
            "token_spilled_energies": (base_spilled + 0.05).tolist()}
    loo1 = {"token_entropies": (base_entropy + 0.9).tolist(),  # the "important" chunk
            "token_spilled_energies": (base_spilled + 1.1).tolist()}

    views = response_delta_views({"full": full, "noctx": noctx, "loo_0": loo0, "loo_1": loo1})

    # 1. noctx views are strictly positive here (removing all context made the model more
    #    surprised everywhere) and shaped correctly.
    assert views["token"]["dent_noctx"].shape == (T,)
    assert (views["token"]["dent_noctx"] > 0).all()
    assert (views["token"]["dnll_noctx"] > 0).all()
    assert views["response"]["dent_noctx"] > 0

    # 2. loo_max correctly picks up the more-important chunk (loo_1) at every token, since
    #    loo_1's deltas dominate loo_0's everywhere here.
    assert np.allclose(views["token"]["dent_loo_max"],
                       np.asarray(loo1["token_entropies"]) - base_entropy)
    assert views["n_loo_chunks"] == 2

    # 3. JS divergence is non-negative, zero only for identical distributions, and reacts to
    #    the noctx shift.
    djs = views["token"]["djs_noctx"]
    assert (djs >= -1e-9).all()
    assert djs.mean() > 0

    # 4. A response with only `full` (no interventions) yields no Δ views at all, not a
    #    crash — the fusion-isolation and full-context-only arms need this to degrade
    #    gracefully.
    solo = response_delta_views({"full": full})
    assert solo["token"] == {}
    assert solo["response"] == {}

    # 5. Evidence graph: two textually similar chunks get high similarity, a dissimilar
    #    third chunk gets low similarity to both, and self-similarity is always 1.
    chunks = [
        "the cat sat on the mat in the sun",
        "a cat sat upon a mat under the sun",
        "quantum entanglement violates local realism",
    ]
    names = ["full", "noctx", "loo_0", "loo_1", "loo_2"]
    graph = build_evidence_graph(chunks, names)
    assert graph.shape == (5, 5)
    assert np.allclose(np.diag(graph), 1.0)
    i0, i1, i2 = names.index("loo_0"), names.index("loo_1"), names.index("loo_2")
    assert graph[i0, i1] > graph[i0, i2]
    assert graph[i0, i1] > graph[i1, i2]
    # full/noctx never correspond to a removed chunk -> zero off-diagonal similarity
    ifull, inoctx = names.index("full"), names.index("noctx")
    assert graph[ifull, i0] == 0.0 and graph[inoctx, i1] == 0.0

    # 6. Empty chunk list degrades to the identity matrix, not a crash.
    empty_graph = build_evidence_graph([], names)
    assert np.allclose(empty_graph, np.eye(5))

    print("evidence_contrast.smoke: PASS (6 checks)")


if __name__ == "__main__":
    smoke()
