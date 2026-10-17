#!/usr/bin/env python
"""
ProcessBench critic-model ceiling driver for the AIRCC cluster.

Reproduces ProcessBench's OWN published evaluation category (paper section on critic models,
`code/run_eval.py` in github.com/QwenLM/ProcessBench, fetched 2026-08-10): prompt a general LLM
to critique a step-by-step math solution paragraph by paragraph and name the earliest erroneous
paragraph (or -1). This is a genuine GENERATION job (the critique is new text), unlike
`run_teacher_forced.py`'s forward-only pass — the two drivers measure different things and are
not interchangeable.

Runs over the SAME `Qwen/ProcessBench` rows already teacher-forced for the GL-LIU study, so the
critic-model number lands as a real competing column on identical inputs (Comparison Level 1 for
the dataset; the critique itself is new model output, as ProcessBench's own protocol requires).

Named competitor (mandatory competitor gate, docs/research_notes/external_data_collection_plan_2026.md):
    ProcessBench: Identifying Process Errors in Mathematical Reasoning
    Zheng et al., arXiv:2412.06559

Usage:
    python cluster/run_processbench_critic.py --model Qwen/Qwen2.5-72B-Instruct \\
        --subsets gsm8k,math --n-samples 30 --out $SHARED/results/pb_critic_qwen72b_pilot
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

# Same NGC/transformers guard as run_inference.py / run_teacher_forced.py — torch 2.6.0a0 parses
# as < 2.6 and blocks torch.load of .bin checkpoints for models with no safetensors.
try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from spectral_utils import load_model, generate_full, load_cache, save_cache_atomic, free_memory
from spectral_utils.processbench import (
    SUBSETS, NO_ERROR, load_processbench, critic_prompt,
    is_correct_processbench_critic, extract_critic_prediction, first_error_f1,
)

EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[pb-critic] SIGTERM received — will checkpoint after the current row", flush=True)


def run_subset(mdl, tok, rows, cfg, out_path):
    """Generate one critique per row, checkpointed and resumable. Returns False if preempted."""
    cache = load_cache(out_path)
    n_done = sum(1 for e in cache.values() if "generated_critique" in e)
    print(f"[pb-critic] {n_done}/{len(rows)} rows already done -> {out_path}", flush=True)

    for idx, row in enumerate(rows):
        if idx in cache and "generated_critique" in cache[idx]:
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            return False
        t0 = time.time()
        msg = critic_prompt(row)
        r = generate_full(
            mdl, tok, msg, temperature=0.0,          # ProcessBench's own SamplingParams(temperature=0.)
            max_new_tokens=cfg.max_new,               # 8192, matching run_eval.py (non-QwQ)
            logprob_top_k=cfg.logprob_top_k,
            system_message=None,                      # single user turn, matches their `messages=[...]`
        )
        pred = extract_critic_prediction(r["full_text"])
        match = pred is not None and pred == int(row["label"])
        cache[idx] = {
            "id": row.get("id"),
            "generator": row.get("generator"),
            "problem": row["problem"],
            "steps": row["steps"],
            "label": int(row["label"]),
            "full_text": r["full_text"],
            "generated_critique": r["full_text"],     # named to match ProcessBench's own field
            "prediction": pred,
            "match": bool(match),
            "token_entropies": r["token_entropies"],
            "gen_token_ids": r["gen_token_ids"],
            "truncated": len(r["gen_token_ids"]) >= cfg.max_new,
        }
        print(f"[pb-critic] row {idx + 1}/{len(rows)}: pred={pred} label={row['label']} "
              f"match={match} {len(r['gen_token_ids'])} tok {time.time() - t0:.1f}s", flush=True)
        if (idx + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--subsets", default=",".join(SUBSETS),
                    help=f"comma-separated ProcessBench subsets (of {SUBSETS})")
    ap.add_argument("--n-samples", type=int, default=None, help="cap rows per subset (pilot)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new", type=int, default=8192,
                    help="ProcessBench's own run_eval.py default (32768 only for QwQ, N/A here)")
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    # sdpa, not the load_model default of eager: a 72B model doing long (up to 8192-token)
    # single-sequence generation hit exactly this OOM on the HLE pilot this same session
    # (eager materializes a full [H,T,T] fp32 attention-score tensor) — fixed there by switching
    # to sdpa, which this driver never needs eager's capture_attention for anyway.
    ap.add_argument("--attn-impl", default="sdpa")
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_pb_critic_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[pb-critic] model={cfg.model} subsets={cfg.subsets} n={cfg.n_samples} "
          f"max_new={cfg.max_new} attn_impl={cfg.attn_impl}", flush=True)

    mdl, tok = load_model(cfg.model, attn_impl=cfg.attn_impl)
    mdl.eval()

    cells = {}
    for subset in [s.strip() for s in cfg.subsets.split(",") if s.strip()]:
        out_path = os.path.join(out_dir, f"pb_critic_{subset}.pkl")
        rows = load_processbench(subset, cfg.n_samples)
        cache = load_cache(out_path)
        if not run_subset(mdl, tok, rows, cfg, out_path):
            print("[pb-critic] INCOMPLETE — resubmit with the same --out to resume", flush=True)
            sys.exit(EXIT_INCOMPLETE)
        cache = load_cache(out_path)
        stats = first_error_f1(cache)
        n_trunc = sum(1 for e in cache.values() if e.get("truncated"))
        n_unparsed = sum(1 for e in cache.values() if e.get("prediction") is None)
        stats["n_truncated"] = n_trunc
        stats["n_unparsed"] = n_unparsed
        cells[subset] = stats
        print(f"=== {subset} DONE: n_error={stats['n_error']} n_correct={stats['n_correct']} "
              f"error_acc={stats['error_acc']} correct_acc={stats['correct_acc']} "
              f"F1={stats['f1']} | truncated={n_trunc} unparsed={n_unparsed} ===", flush=True)

    manifest = {
        "driver": "run_processbench_critic.py",
        "paper": "ProcessBench: Identifying Process Errors in Mathematical Reasoning "
                 "(Zheng et al., arXiv:2412.06559)",
        "model": cfg.model,
        "protocol": "ProcessBench's own critic-model prompt (code/templates/critique_template.txt "
                    "+ run_eval.py's prepare_input_boxed/extract_answer), greedy decoding, "
                    "single user turn, no system message",
        "max_new_tokens": cfg.max_new,
        "logprob_top_k": cfg.logprob_top_k,
        "attn_impl": cfg.attn_impl,
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
