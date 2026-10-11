#!/usr/bin/env python
"""
Exact-protocol GASP rescoring on RAGTruth with **full-vocabulary** Jensen-Shannon divergence —
the published competitor for Benchmark 2a (unsupported-sentence localization) of
docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md.

GASP: "Detecting Hallucinations in Retrieval-Augmented Generation through Grounding-Aware
Sensitivity by Perturbation" (Bouke, arXiv:2607.04223). It holds the answer FIXED and measures
how the answer's own predictive distribution moves when evidence is removed — the closest
published relative of our evidence-contrast fusion, which is exactly why it has to be run
under its own protocol rather than approximated.

WHY THIS JOB EXISTS WHEN WE ALREADY HAVE A GASP ARM
--------------------------------------------------------------------------------------
`scripts/rag_ec_v1/gasp.py` already reproduces Eqs. (8)-(11), but on the preregistered cache,
which differs from the paper in two ways that no amount of rescoring can fix offline:

1. **The JSD is a top-50 + shared-tail APPROXIMATION.** `spectral_utils/evidence_contrast.py::
   js_divergence` unions two top-50 id sets and lumps all remaining mass into one bucket,
   because dense distributions were never saved. The paper's Eqs. (9)/(11) are full-vocabulary.
   A dense [T, V] tensor cannot be stored (200 tokens x ~152k vocab x 4 B = 122 MB per response
   per condition), so the exact quantity must be computed ONLINE, during the forward pass, and
   only the resulting per-token scalar kept. That is what this driver does.

2. **The chunk unit, the sample, and the caps are all different.** Ours: per task type (QA=3
   passages, Data2txt=9 JSON fields, Summary=1 document), full 2,700-response test split, no
   caps. The paper's: K=5 SENTENCE-GROUPED chunks, 400 class-balanced responses, Summary and
   Data2txt only (QA excluded as "not long enough for stable estimation"), context capped at
   700 tokens and the answer at 200.

FIDELITY DECLARATION — LEVEL 2 (protocol reproduction, our own sampling)
--------------------------------------------------------------------------------------
The paper is arXiv-only and single-author, and no official code release or published response
ID list was located, so the paper's exact 400 sample IDs cannot be reused. This job reproduces
the protocol as specified in the text with OUR OWN declared seed, recorded in the manifest.
The sentence splitter (`spectral_utils.ragtruth.split_sentences`) is likewise our own
disclosed implementation, since the paper does not publish one.

The number to check against is the Qwen2.5-1.5B-specific **0.713 response AUC / 0.673 span
AUC** — not the rounder ~0.73/~0.67 cross-scorer average that older manifests in this repo
quote, which mixes in the 0.5B and SmolLM2 scorers we do not run.

WHAT IS SAVED
--------------------------------------------------------------------------------------
Per (response, condition), keyed `f"{response_id}::{condition}"`. In addition to the usual
telemetry, every non-`full` condition carries `token_jsd_vs_full` — the EXACT full-vocabulary
JSD against the `full` condition at each answer token. `top_k_logprobs` is still saved so the
exact and approximate JSD can be compared on identical rows, which turns "does the top-50
approximation matter?" into a measurement instead of an assumption.

Usage:
    python cluster/run_gasp_exact.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --out $SHARED/results/gasp_ragtruth_exact_qwen15b_full
"""
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from backfill_views import forward_batch, candidate_quantities
from spectral_utils import load_model, load_cache, save_cache_atomic, free_memory
from spectral_utils.ragtruth import (
    load_ragtruth, condition_prompt, sentence_grouped_chunks, split_sentences,
    response_token_spans,
)

EXIT_INCOMPLETE = 85
STOP = {"flag": False}

# The paper's own settings (Section 5.2/5.4), reproduced verbatim.
GASP_K = 5
GASP_CONTEXT_CAP = 700
GASP_ANSWER_CAP = 200
GASP_TASK_TYPES = ("Summary", "Data2txt")   # QA excluded BY THE PAPER, not by us
GASP_N_PER_CLASS = 200                      # 200 hallucinated + 200 clean = 400 responses


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[gasp] SIGTERM received — will checkpoint after the current response", flush=True)


def balanced_sample(rows, n_per_class: int, seed: int = 0):
    """The paper's class-balanced 200/200 draw. Grouped by nothing on purpose — the paper
    samples responses, and its own evaluation groups by response only."""
    rng = np.random.default_rng(seed)
    hallucinated = [r for r in rows if r["labels"]]
    clean = [r for r in rows if not r["labels"]]

    def _take(pool):
        if len(pool) <= n_per_class:
            return list(pool)
        idx = np.sort(rng.choice(len(pool), size=n_per_class, replace=False))
        return [pool[i] for i in idx]

    picked = _take(hallucinated) + _take(clean)
    picked.sort(key=lambda r: str(r["response_id"]))
    return picked, {"n_hallucinated_pool": len(hallucinated), "n_clean_pool": len(clean),
                    "n_hallucinated_taken": min(len(hallucinated), n_per_class),
                    "n_clean_taken": min(len(clean), n_per_class)}


def cap_context(tok, row, cap_tokens: int):
    """Truncate the evidence to `cap_tokens` and splice the truncated text back into the
    ORIGINAL prompt, keeping the prompt-surgery contract: every character outside the evidence
    substring stays byte-identical. Returns a new row dict; the input is never mutated."""
    context = row["context"]
    enc = tok(context, add_special_tokens=False, return_offsets_mapping=True)
    n_tokens = len(enc["input_ids"])
    if n_tokens <= cap_tokens:
        capped = context
    else:
        # End of the last retained token, so the cut never lands mid-token.
        capped = context[:enc["offset_mapping"][cap_tokens - 1][1]]

    prompt = row["prompt"]
    idx = prompt.index(context)   # guaranteed by load_ragtruth's integrity gate
    new_row = dict(row)
    new_row["context"] = capped
    new_row["prompt"] = prompt[:idx] + capped + prompt[idx + len(context):]
    new_row["_context_tokens_before_cap"] = n_tokens
    new_row["_context_truncated"] = n_tokens > cap_tokens
    return new_row


def cap_answer(tok, response: str, labels, cap_tokens: int):
    """Truncate the answer to `cap_tokens`. Returns `(gen_ids, offsets, kept_text, span_token_
    spans, align_diag, truncated)`. Gold spans are re-mapped against the RETAINED prefix, and
    any gold span that falls entirely past the cap is dropped and counted — never silently
    carried as if it were still scoreable."""
    gen_ids, offsets, span_token_spans, align_diag = response_token_spans(tok, response, labels)
    truncated = len(gen_ids) > cap_tokens
    if not truncated:
        return gen_ids, offsets, response, span_token_spans, align_diag, False

    gen_ids = gen_ids[:cap_tokens]
    offsets = offsets[:cap_tokens]
    kept_chars = offsets[-1][1]
    kept_text = response[:kept_chars]
    kept_labels = [lab for lab in labels if lab["start"] < kept_chars]
    _, _, span_token_spans, align_diag = response_token_spans(tok, kept_text, kept_labels)
    align_diag["n_gold_spans_lost_to_answer_cap"] = len(labels) - len(kept_labels)
    return gen_ids, offsets, kept_text, span_token_spans, align_diag, True


def full_vocab_jsd(logits_a, logits_b) -> np.ndarray:
    """Exact per-token Jensen-Shannon divergence over the ENTIRE vocabulary, in nats.

    JSD(P||Q) = 0.5*KL(P||M) + 0.5*KL(Q||M) with M = (P+Q)/2. Computed in float32 from the raw
    logits — no top-k truncation, no tail bucket, which is the whole point of this job. The
    `logsumexp` route (rather than exp-then-normalize) keeps it stable at the small
    probabilities that dominate a 152k-token vocabulary.
    """
    lp_a = torch.log_softmax(logits_a.float(), dim=-1)
    lp_b = torch.log_softmax(logits_b.float(), dim=-1)
    lp_m = torch.logaddexp(lp_a, lp_b) - float(np.log(2.0))
    kl_a = (lp_a.exp() * (lp_a - lp_m)).sum(dim=-1)
    kl_b = (lp_b.exp() * (lp_b - lp_m)).sum(dim=-1)
    jsd = 0.5 * (kl_a + kl_b)
    # Numerically JSD is in [0, ln 2]; clamp only to kill -0.0/1e-9 float dust.
    return jsd.clamp_(min=0.0).cpu().numpy().astype("float32")


def process_response(mdl, tok, row, cfg, cache):
    """Run every condition of ONE response together so the dense distributions needed for an
    exact JSD exist simultaneously — then discard them. Peak memory is
    (n_conditions x answer_tokens x vocab), i.e. ~7 x 200 x 152k, which is a few hundred MB in
    bf16 and never touches disk."""
    capped = cap_context(tok, row, cfg.context_cap)
    gen_ids, offsets, kept_text, span_token_spans, align_diag, ans_trunc = cap_answer(
        tok, row["response"], row["labels"], cfg.answer_cap)
    if not gen_ids:
        return 0

    chunks = sentence_grouped_chunks(capped["context"], k=cfg.k)
    conditions = ["full", "noctx"] + [f"loo_{j}" for j in range(len(chunks))]

    items = []
    for condition in conditions:
        prompt = condition_prompt(capped, condition, spans=chunks)
        items.append({"prompt_ids": tok(prompt).input_ids, "gen_ids": gen_ids,
                      "condition": condition})

    slices = forward_batch(mdl, items)
    by_condition = {it["condition"]: lg for it, lg in zip(items, slices)}
    full_logits = by_condition["full"]

    sentence_spans = split_sentences(kept_text)
    n_written = 0
    for condition in conditions:
        logits = by_condition[condition]
        q = candidate_quantities(logits, gen_ids, warpers=None,
                                 raw_top_k=cfg.logprob_top_k, post_top_k=cfg.logprob_top_k)
        jsd = None if condition == "full" else full_vocab_jsd(full_logits, logits)
        cache[f"{row['response_id']}::{condition}"] = {
            "response_id": row["response_id"],
            "source_id": row["source_id"],
            "task_type": row["task_type"],
            "model": row["model"],
            "condition": condition,
            "response": kept_text,
            "sentence_spans": sentence_spans,
            "span_labels": row["labels"],
            "span_token_spans": span_token_spans,
            "response_label": bool(row["labels"]),
            "align_diag": align_diag,
            "n_chunks": len(chunks),
            "chunk_spans": chunks,
            "context_tokens_before_cap": capped["_context_tokens_before_cap"],
            "context_truncated": capped["_context_truncated"],
            "answer_truncated": ans_trunc,
            "gen_token_ids": gen_ids,
            "token_entropies": q["token_entropies_recomputed"],
            "token_spilled_energies": q["token_spilled_energies"],
            "token_logsumexp": q["token_logsumexp"],
            "top_k_logprobs": q["top_k_logprobs_raw"],
            # THE point of this job: exact, full-vocabulary, computed online and never stored dense.
            "token_jsd_vs_full": jsd,
        }
        n_written += 1

    del slices, by_condition, full_logits
    return n_written


def run(mdl, tok, rows, cfg, out_path):
    cache = load_cache(out_path)
    done = {k.split("::")[0] for k in cache}
    todo = [r for r in rows if str(r["response_id"]) not in {str(d) for d in done}]
    print(f"[gasp] {len(rows) - len(todo)}/{len(rows)} responses already done -> {out_path}",
          flush=True)

    for i, row in enumerate(todo):
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} items", flush=True)
            return False
        t0 = time.time()
        n = process_response(mdl, tok, row, cfg, cache)
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            print(f"[gasp] {i + 1}/{len(todo)} responses ({n} conditions, "
                  f"{time.time() - t0:.2f}s last)", flush=True)
        if (i + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    return True


def cell_stats(cache):
    responses = {e["response_id"] for e in cache.values()}
    per_response = {}
    for e in cache.values():
        per_response.setdefault(e["response_id"], []).append(e["condition"])
    n_cond = [len(v) for v in per_response.values()]
    by_task, by_label = {}, {"hallucinated": 0, "clean": 0}
    for rid, _ in per_response.items():
        entry = next(e for e in cache.values() if e["response_id"] == rid)
        by_task[entry["task_type"]] = by_task.get(entry["task_type"], 0) + 1
        by_label["hallucinated" if entry["response_label"] else "clean"] += 1
    jsd_vals = [float(np.mean(e["token_jsd_vs_full"])) for e in cache.values()
                if e.get("token_jsd_vs_full") is not None and len(e["token_jsd_vs_full"])]
    return {
        "n_items": len(cache), "n_responses": len(responses),
        "conditions_per_response_min": int(min(n_cond)) if n_cond else 0,
        "conditions_per_response_max": int(max(n_cond)) if n_cond else 0,
        "by_task_type": by_task, "by_response_label": by_label,
        "n_context_truncated": sum(1 for e in cache.values() if e["context_truncated"]),
        "n_answer_truncated": sum(1 for e in cache.values() if e["answer_truncated"]),
        "mean_full_vocab_jsd": float(np.mean(jsd_vals)) if jsd_vals else None,
        "max_full_vocab_jsd": float(np.max(jsd_vals)) if jsd_vals else None,
        "ln2_upper_bound": float(np.log(2.0)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=GASP_K)
    ap.add_argument("--context-cap", type=int, default=GASP_CONTEXT_CAP)
    ap.add_argument("--answer-cap", type=int, default=GASP_ANSWER_CAP)
    ap.add_argument("--n-per-class", type=int, default=GASP_N_PER_CLASS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=25)
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_gasp_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[gasp] model={cfg.model} K={cfg.k} context_cap={cfg.context_cap} "
          f"answer_cap={cfg.answer_cap} n_per_class={cfg.n_per_class}", flush=True)

    rows, diag = load_ragtruth(split=cfg.split, task_types=GASP_TASK_TYPES)
    rows, sample_diag = balanced_sample(rows, cfg.n_per_class, seed=cfg.seed)
    print(f"[gasp] {len(rows)} responses after the class-balanced draw: {sample_diag}", flush=True)

    mdl, tok = load_model(cfg.model, quantize_4bit=False)

    out_path = os.path.join(out_dir, "gasp_exact.pkl")
    t0 = time.time()
    if not run(mdl, tok, rows, cfg, out_path):
        print("[gasp] INCOMPLETE — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)

    stats = cell_stats(load_cache(out_path))
    manifest = {
        "driver": "run_gasp_exact.py",
        "paper": "GASP — Detecting Hallucinations in Retrieval-Augmented Generation through "
                 "Grounding-Aware Sensitivity by Perturbation (Bouke, arXiv:2607.04223)",
        "fidelity_level": 2,
        "fidelity_note": "protocol reproduction with OUR OWN seed and sample IDs — the paper is "
                         "arXiv-only, no official code release or response-ID list was located, "
                         "so its exact 400 IDs cannot be reused. The sentence splitter is also "
                         "ours (spectral_utils.ragtruth.split_sentences); the paper publishes none.",
        "target_numbers": {"response_auc": 0.713, "span_auc": 0.673,
                           "scorer": "Qwen2.5-1.5B-Instruct",
                           "note": "the scorer-specific figures, NOT the ~0.73/~0.67 "
                                   "cross-scorer average quoted in older manifests here"},
        "model": cfg.model,
        "protocol": {
            "k_chunks": cfg.k, "chunk_unit": "sentence-grouped, character-mass balanced",
            "context_cap_tokens": cfg.context_cap, "answer_cap_tokens": cfg.answer_cap,
            "task_types": list(GASP_TASK_TYPES),
            "qa_excluded_by": "the paper ('not long enough for stable estimation')",
            "n_per_class": cfg.n_per_class, "seed": cfg.seed,
            "jsd": "EXACT full-vocabulary, computed online during the forward pass; no dense "
                   "distribution is ever stored",
        },
        "ragtruth_diagnostics": diag,
        "sample_diagnostics": sample_diag,
        "cell_stats": stats,
        "elapsed_sec": time.time() - t0,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)

    print("\n=== CELL STATS ===")
    print(json.dumps(stats, indent=2, default=str))
    print(f"\nCOMPLETE -> {out_dir}", flush=True)
    free_memory()


if __name__ == "__main__":
    main()
