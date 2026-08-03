#!/usr/bin/env python
"""
Teacher-forced ProcessBench scoring driver for the AIRCC cluster (Extension F / Track A2).

Replicates "Mind the Gap" (ICML 2026) §4: "For the step-level diagnostics on ProcessBench, we
adopt a teacher-forcing (or forced decoding) approach to extract the predictive distributions
along the reasoning traces. Specifically, given a pre-defined reasoning chain, we perform a
single forward pass to compute the logit distributions for each token transition."

NOTHING IS GENERATED HERE. Each row's reasoning chain already exists (ProcessBench solutions are
written by other models and annotated by experts); we only measure our model's per-token
distributions over that fixed text. One forward pass per row, so 3,400 rows x 2 models is cheap
next to any generation job.

REUSE, NOT REIMPLEMENTATION
---------------------------
The forward pass and the per-token quantities come from `cluster/backfill_views.py`, which is
already validated by its own Gate B on real cells:
    forward_batch(mdl, items)         -> right-padded batched logits sliced to [T_gen, V]
    candidate_quantities(...)         -> token_logsumexp / top_k_logprobs_raw / spilled / H
Writing a second forward pass would mean maintaining two definitions of the same quantity.

GATE B (--validate) IS NOT OPTIONAL
-----------------------------------
A wrong prompt or chat template produces a plausible-looking but systematically shifted entropy
trace, which is invisible in the output and fatal downstream. ProcessBench has no saved trace to
compare against, so `--validate` runs this driver's own machinery over a slice of an existing
GENERATED cell (where `token_entropies` *is* saved) and requires agreement. Run it before
trusting any ProcessBench output from a new model.

Preemption-safe on the `run_inference.py` contract: SIGTERM sets a flag, the loop checkpoints via
`save_cache_atomic` and exits EXIT_INCOMPLETE (85) so a chained `--dependency=afterany` job
resumes; presence of a row's key in the cache is the resume marker.

Usage:
    python cluster/run_teacher_forced.py --model Qwen/Qwen3-8B --subsets gsm8k,math \\
        --out $SHARED/results/evdrop_processbench_qwen3_8b
    python cluster/run_teacher_forced.py --model Qwen/Qwen3-8B --validate \\
        --validate-pkl $SHARED/results/ars_gsm8k_qwen3_8b/raw_gsm8k_T0.0.pkl
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

# Same NGC/transformers guard as run_inference.py — torch 2.6.0a0 parses as < 2.6 and blocks
# torch.load of .bin checkpoints for models with no safetensors.
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from backfill_views import forward_batch, candidate_quantities
from spectral_utils import load_model, load_cache, save_cache_atomic, free_memory
from spectral_utils.model_utils import fmt_prompt
from spectral_utils.processbench import (
    SUBSETS, NO_ERROR, load_processbench, processbench_prompt,
    build_chain, step_token_spans, assert_alignment,
)

EXIT_INCOMPLETE = 85
STOP = {"flag": False}

# Gate B threshold. `backfill_views` measures bf16 decode-vs-forward kernel noise at well under
# this; a wrong prompt/template produces O(0.1-1 nat) systematic divergence, cleanly separable.
GATE_B_MEDIAN_MAX = 0.05
GATE_B_MIN_CORR = 0.999


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[teacher-forced] SIGTERM received — will checkpoint after the current batch", flush=True)


def build_items(tok, rows):
    """One work item per row: prompt ids, chain token ids, and the step spans over those ids."""
    items = []
    for i, row in enumerate(rows):
        text, char_spans = build_chain(row["steps"])
        gen_ids, spans = step_token_spans(tok, text, char_spans)
        # The alignment gate runs per row, at build time, BEFORE any GPU work — a misaligned
        # row is a data problem and should not consume a forward pass.
        diag = assert_alignment(gen_ids, spans, row["steps"], strict=False)
        prompt = fmt_prompt(tok, processbench_prompt(row))
        items.append({
            "idx": i,
            "prompt_ids": tok(prompt).input_ids,
            "gen_ids": gen_ids,
            "step_token_spans": spans,
            "align_diag": diag,
            "row": row,
        })
    return items


def batched(items, max_batch_tokens: int, max_batch: int = 8):
    """Length-sorted batches capped by total padded tokens.

    Right padding is exact for causal LMs, but a batch mixing a 100-token chain with a
    4,000-token one wastes most of the compute on pad. Sorting by length first keeps batches
    near-uniform; the token cap is what actually bounds memory, since a [B, T, V] logit tensor
    at V ~ 150k dominates everything else.
    """
    order = sorted(items, key=lambda it: len(it["prompt_ids"]) + len(it["gen_ids"]))
    batch, longest = [], 0
    for it in order:
        n = len(it["prompt_ids"]) + len(it["gen_ids"])
        longest_if_added = max(longest, n)
        if batch and (longest_if_added * (len(batch) + 1) > max_batch_tokens
                      or len(batch) >= max_batch):
            yield batch
            batch, longest = [it], n
        else:
            batch.append(it)
            longest = longest_if_added
    if batch:
        yield batch


def score_rows(mdl, items, cache, out_path, cfg):
    """Teacher-force every not-yet-done row and append its quantities to `cache`."""
    todo = [it for it in items if it["idx"] not in cache]
    print(f"[teacher-forced] {len(cache)} rows already done, {len(todo)} to go", flush=True)
    n_done = 0

    for batch in batched(todo, cfg.max_batch_tokens, cfg.max_batch):
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            return False
        t0 = time.time()
        slices = forward_batch(mdl, batch)
        for it, logits in zip(batch, slices):
            # warpers=None: the paper's protocol is greedy (tau = 0), so there is no
            # temperature/top-k/top-p warp to mirror and post-warp == raw by construction.
            q = candidate_quantities(
                logits, it["gen_ids"], warpers=None,
                raw_top_k=cfg.logprob_top_k, post_top_k=cfg.logprob_top_k,
            )
            row = it["row"]
            cache[it["idx"]] = {
                "id": row.get("id"),
                "generator": row.get("generator"),
                "problem": row["problem"],
                "steps": row["steps"],
                "label": int(row["label"]),                       # first erroneous step, or -1
                "final_answer_correct": bool(row.get("final_answer_correct", False)),
                "step_token_spans": it["step_token_spans"],
                "align_diag": it["align_diag"],
                "gen_token_ids": it["gen_ids"],
                # `candidate_quantities` names its gate output `token_entropies_recomputed`;
                # here there is no prior trace to gate against, so it IS the trace.
                "token_entropies": q["token_entropies_recomputed"],
                "token_spilled_energies": q["token_spilled_energies"],
                "token_logsumexp": q["token_logsumexp"],
                "top_k_logprobs": q["top_k_logprobs_raw"],
            }
            n_done += 1
        del slices
        if n_done and n_done % cfg.checkpoint_every < len(batch):
            save_cache_atomic(cache, out_path)
        print(f"[teacher-forced] batch of {len(batch)} in {time.time()-t0:.1f}s "
              f"({len(cache)}/{len(items)} rows)", flush=True)

    save_cache_atomic(cache, out_path)
    return True


def cell_stats(cache):
    """Reported per subset. `n_with_error` is the SLA-eligible population — rows labelled -1
    have no erroneous step to localize and are used only by ProcessBench's official F1."""
    labels = [e["label"] for e in cache.values()]
    n_err = sum(1 for x in labels if x != NO_ERROR)
    steps = [len(e["steps"]) for e in cache.values()]
    toks = [len(e["token_entropies"]) for e in cache.values()]
    unmapped = sum(e["align_diag"]["n_unmapped_steps"] for e in cache.values())
    misaligned = sum(1 for e in cache.values() if e["align_diag"]["problems"])
    step_lens = [n for e in cache.values() for n in e["align_diag"]["step_token_lengths"]]
    return {
        "n_rows": len(cache),
        "n_with_error": n_err,
        "n_all_correct": len(cache) - n_err,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "mean_tokens": float(np.mean(toks)) if toks else 0.0,
        "n_unmapped_steps": unmapped,
        "n_rows_misaligned": misaligned,
        # The A4 constraint, measured rather than assumed: compute_spectral_features needs >= 8
        # tokens and compute_stft_features >= 32, so these two fractions bound how much of the
        # feature pool is even available at step level.
        "frac_steps_lt_8_tokens": float(np.mean([n < 8 for n in step_lens])) if step_lens else 0.0,
        "frac_steps_lt_32_tokens": float(np.mean([n < 32 for n in step_lens])) if step_lens else 0.0,
        "median_step_tokens": float(np.median(step_lens)) if step_lens else 0.0,
    }


def run_gate_b(mdl, tok, cfg):
    """Gate B: reproduce a GENERATED cell's saved `token_entropies` with this driver's own path.

    Passing means the prompt reconstruction, chat template and entropy formula all agree with
    what actually produced that cell. Failing means the ProcessBench numbers would be measuring
    a different conditioning than intended.
    """
    print(f"[gate-b] validating against {cfg.validate_pkl}", flush=True)
    cache = load_cache(cfg.validate_pkl)
    from run_inference import DATASETS
    _, prompt_fn, _ = DATASETS[cfg.validate_dataset]

    items, saved = [], []
    for idx in sorted(cache)[: cfg.validate_n]:
        entry = cache[idx]
        for cand in entry["candidates"][:1]:
            ids = cand.get("gen_token_ids")
            ents = cand.get("token_entropies")
            if not ids or not ents:
                continue
            msg = prompt_fn(entry["gold_row"])
            if cfg.validate_prompt_suffix:
                msg = f"{msg}{cfg.validate_prompt_suffix}"
            items.append({"prompt_ids": tok(fmt_prompt(tok, msg)).input_ids,
                          "gen_ids": list(ids)})
            saved.append(np.asarray(ents, dtype=np.float64))
    if not items:
        raise SystemExit("[gate-b] no usable candidates (need gen_token_ids + token_entropies)")

    diffs, corrs = [], []
    for batch_start in range(0, len(items), cfg.max_batch):
        chunk = items[batch_start: batch_start + cfg.max_batch]
        for k, logits in enumerate(forward_batch(mdl, chunk)):
            q = candidate_quantities(logits, chunk[k]["gen_ids"], warpers=None,
                                     raw_top_k=cfg.logprob_top_k, post_top_k=cfg.logprob_top_k)
            a = saved[batch_start + k]
            b = np.asarray(q["token_entropies_recomputed"], dtype=np.float64)
            n = min(len(a), len(b))
            if n < 3:
                continue
            diffs.append(np.abs(a[:n] - b[:n]))
            if a[:n].std() > 1e-12 and b[:n].std() > 1e-12:
                corrs.append(float(np.corrcoef(a[:n], b[:n])[0, 1]))

    d = np.concatenate(diffs)
    med, p99 = float(np.median(d)), float(np.percentile(d, 99))
    med_r = float(np.median(corrs)) if corrs else float("nan")
    ok = med <= GATE_B_MEDIAN_MAX and med_r >= GATE_B_MIN_CORR
    print(f"[gate-b] traces={len(diffs)} tokens={d.size} median|d|={med:.5f} "
          f"p99|d|={p99:.5f} median_r={med_r:.6f}", flush=True)
    print(f"[gate-b] {'PASS' if ok else 'FAIL'} "
          f"(need median|d| <= {GATE_B_MEDIAN_MAX} and median_r >= {GATE_B_MIN_CORR})", flush=True)
    if not ok:
        raise SystemExit("[gate-b] FAILED — the prompt/template/warp does not reproduce the "
                         "saved trace. Do NOT trust ProcessBench output from this config.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--subsets", default=",".join(SUBSETS),
                    help=f"comma-separated ProcessBench subsets (of {SUBSETS})")
    ap.add_argument("--n-samples", type=int, default=None, help="cap rows per subset (pilot)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384,
                    help="cap on padded tokens per batch; the [B,T,V] logit tensor dominates memory")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--validate", action="store_true", help="run Gate B and exit")
    ap.add_argument("--validate-pkl", default=None)
    ap.add_argument("--validate-dataset", default="gsm8k")
    ap.add_argument("--validate-prompt-suffix", default="")
    ap.add_argument("--validate-n", type=int, default=20)
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_processbench_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[teacher-forced] model={cfg.model} subsets={cfg.subsets} n={cfg.n_samples}", flush=True)
    if torch.cuda.is_available():
        print(f"[teacher-forced] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[teacher-forced] WARNING: no CUDA device — running on CPU", flush=True)

    mdl, tok = load_model(cfg.model)
    mdl.eval()

    if cfg.validate:
        if not cfg.validate_pkl:
            raise SystemExit("--validate needs --validate-pkl (a GENERATED cell with "
                             "gen_token_ids + token_entropies saved)")
        run_gate_b(mdl, tok, cfg)
        return

    cells = {}
    for subset in [s.strip() for s in cfg.subsets.split(",") if s.strip()]:
        out_path = os.path.join(out_dir, f"processbench_{subset}.pkl")
        rows = load_processbench(subset, cfg.n_samples)
        items = build_items(tok, rows)

        bad = [it["idx"] for it in items if it["align_diag"]["problems"]]
        if bad:
            print(f"[teacher-forced] WARNING: {len(bad)} row(s) failed the step-alignment gate "
                  f"in {subset} (first: {bad[:5]}); they are scored but flagged in align_diag "
                  f"and MUST be excluded from step-level metrics.", flush=True)

        cache = load_cache(out_path)
        if not score_rows(mdl, items, cache, out_path, cfg):
            print("[teacher-forced] INCOMPLETE — resubmit with the same --out to resume", flush=True)
            sys.exit(EXIT_INCOMPLETE)
        st = cell_stats(cache)
        cells[subset] = st
        print(f"=== {subset} DONE: {st['n_rows']} rows | {st['n_with_error']} with an error step | "
              f"mean {st['mean_steps']:.1f} steps / {st['mean_tokens']:.0f} tokens | "
              f"steps <8 tok {st['frac_steps_lt_8_tokens']:.1%}, <32 tok "
              f"{st['frac_steps_lt_32_tokens']:.1%} ===", flush=True)

    manifest = {
        "driver": "run_teacher_forced.py",
        "paper": "Mind the Gap / Evidence Drop (ICML 2026, PMLR 306) — ProcessBench SLA arm",
        "model": cfg.model,
        "protocol": "teacher-forced single forward pass over the provided reasoning chain",
        "prompt": "data_loaders.math_prompt on the problem (OUR choice — the paper states none)",
        "step_separator": "\\n\\n",
        "logprob_top_k": cfg.logprob_top_k,
        "n_samples_per_subset": cfg.n_samples,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cells": cells,
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    print(f"\nALL SUBSETS COMPLETE -> {out_dir}", flush=True)
    free_memory()


if __name__ == "__main__":
    main()
