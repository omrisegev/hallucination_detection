#!/usr/bin/env python
"""
Teacher-forced **PRMBench** telemetry driver — our own arm for Benchmark 3 (every-step
correctness) of docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md.

A fork of `cluster/run_teacher_forced.py` in which **only `build_items` changes**. Everything
that carries risk — `forward_batch` / `candidate_quantities` from `cluster/backfill_views.py`,
the length-sorted token-capped batching, the SIGTERM/`save_cache_atomic`/exit-85 contract, and
Gate B — is imported from the ProcessBench driver unchanged, so there is exactly one definition
of each per-token quantity in this repository.

NOTHING IS GENERATED. PRMBench traces are fixed text (`modified_process`); we only measure our
model's predictive distribution over them, one forward pass per row.

GATE B IS STILL NOT OPTIONAL. A wrong chat template produces a plausible-looking but
systematically shifted entropy trace that is invisible downstream. PRMBench has no generated
cell of its own to validate against, so `--validate` reuses the ProcessBench driver's gate over
an existing GENERATED cell — it is checking the model+template+entropy path, which is shared,
not the benchmark.

Usage:
    python cluster/run_prmbench_teacher_forced.py --model Qwen/Qwen3-8B \\
        --out $SHARED/results/prmbench_qwen3_8b_telemetry_full
    python cluster/run_prmbench_teacher_forced.py --model Qwen/Qwen3-8B --validate \\
        --validate-pkl $SHARED/results/pb_qwen3_8b/... --validate-dataset gsm8k
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

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from backfill_views import forward_batch, candidate_quantities
from spectral_utils import load_model, load_cache, save_cache_atomic, free_memory
from spectral_utils.model_utils import fmt_prompt
from spectral_utils.processbench import NO_THINK_SUFFIX
from spectral_utils.prmbench import (
    PRMBENCH_DATASET_ID, load_prmbench, prmbench_prompt,
    build_chain, step_token_spans, assert_alignment,
)

# Reused verbatim from the ProcessBench driver — same batching, same gate, same thresholds.
from run_teacher_forced import (  # noqa: E402
    batched, run_gate_b, GATE_B_MEDIAN_MAX, GATE_B_MIN_CORR,
)

EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[prmbench-tf] SIGTERM received — will checkpoint after the current batch", flush=True)


def build_items(tok, meta, thinking_suffix=NO_THINK_SUFFIX):
    """The ONLY function that differs from `run_teacher_forced.build_items`.

    PRMBench rows carry `question` + `steps` (already the official `modified_question` /
    `modified_process`, resolved by `load_prmbench`) instead of ProcessBench's `problem` /
    `steps` + integer `label`. The chain construction, span mapping, and pre-GPU alignment gate
    are identical calls into the same shared helpers.
    """
    items = []
    for i, row in enumerate(meta):
        text, char_spans = build_chain(row["steps"])
        gen_ids, spans = step_token_spans(tok, text, char_spans)
        diag = assert_alignment(gen_ids, spans, row["steps"], strict=False)
        prompt = fmt_prompt(tok, prmbench_prompt(row, thinking_suffix))
        items.append({
            "idx": i,
            "prompt_ids": tok(prompt).input_ids,
            "gen_ids": gen_ids,
            "step_token_spans": spans,
            "align_diag": diag,
            "row": row,
        })
    return items


def score_rows(mdl, items, cache, out_path, cfg):
    todo = [it for it in items if it["idx"] not in cache]
    print(f"[prmbench-tf] {len(cache)} rows already done, {len(todo)} to go", flush=True)
    n_done = 0

    for batch in batched(todo, cfg.max_batch_tokens, cfg.max_batch):
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            return False
        t0 = time.time()
        slices = forward_batch(mdl, batch)
        for it, logits in zip(batch, slices):
            q = candidate_quantities(
                logits, it["gen_ids"], warpers=None,
                raw_top_k=cfg.logprob_top_k, post_top_k=cfg.logprob_top_k,
            )
            row = it["row"]
            cache[it["idx"]] = {
                "idx": row["idx"],
                "source_idx": row["source_idx"],
                "classification": row["classification"],
                "category": row["category"],
                "question": row["question"],
                "steps": row["steps"],
                # Per-step ground truth, 1-indexed exactly as PRMBench ships it. Carried through
                # so the scorer never re-joins the Hub, but NEVER read by any fit.
                "error_steps": row["error_steps"],
                "step_token_spans": it["step_token_spans"],
                "align_diag": it["align_diag"],
                "gen_token_ids": it["gen_ids"],
                "token_entropies": q["token_entropies_recomputed"],
                "token_spilled_energies": q["token_spilled_energies"],
                "token_logsumexp": q["token_logsumexp"],
                "top_k_logprobs": q["top_k_logprobs_raw"],
            }
            n_done += 1
        del slices
        if n_done and n_done % cfg.checkpoint_every < len(batch):
            save_cache_atomic(cache, out_path)
        print(f"[prmbench-tf] batch of {len(batch)} in {time.time() - t0:.1f}s "
              f"({len(cache)}/{len(items)} rows)", flush=True)

    save_cache_atomic(cache, out_path)
    return True


def cell_stats(cache):
    """Availability diagnostics. `frac_steps_lt_8_tokens` is the binding constraint: the five
    core local curves need a usable window inside each step, and PRMBench steps are shorter than
    ProcessBench's on average, so this fraction bounds how much of the panel is scoreable."""
    steps = [len(e["steps"]) for e in cache.values()]
    toks = [len(e["token_entropies"]) for e in cache.values()]
    unmapped = sum(e["align_diag"]["n_unmapped_steps"] for e in cache.values())
    misaligned = sum(1 for e in cache.values() if e["align_diag"]["problems"])
    step_lens = [n for e in cache.values() for n in e["align_diag"]["step_token_lengths"]]
    by_class = {}
    for e in cache.values():
        by_class[e["classification"]] = by_class.get(e["classification"], 0) + 1
    return {
        "n_rows": len(cache),
        "n_steps": int(np.sum(steps)) if steps else 0,
        "mean_steps": float(np.mean(steps)) if steps else 0.0,
        "mean_tokens": float(np.mean(toks)) if toks else 0.0,
        "n_unmapped_steps": unmapped,
        "n_rows_misaligned": misaligned,
        "frac_steps_lt_8_tokens": float(np.mean([n < 8 for n in step_lens])) if step_lens else 0.0,
        "frac_steps_lt_32_tokens": float(np.mean([n < 32 for n in step_lens])) if step_lens else 0.0,
        "median_step_tokens": float(np.median(step_lens)) if step_lens else 0.0,
        "counts_by_classification": by_class,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dataset", default=PRMBENCH_DATASET_ID)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-samples", type=int, default=None, help="cap RAW rows (debug only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--checkpoint-every", type=int, default=100)
    ap.add_argument("--prompt-suffix", default=NO_THINK_SUFFIX)
    ap.add_argument("--validate", action="store_true", help="run Gate B and exit")
    ap.add_argument("--validate-pkl", default=None)
    ap.add_argument("--validate-dataset", default="gsm8k")
    ap.add_argument("--validate-prompt-suffix", default=NO_THINK_SUFFIX)
    ap.add_argument("--validate-n", type=int, default=20)
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_prmbench_tf_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[prmbench-tf] model={cfg.model} dataset={cfg.dataset} n={cfg.n_samples}", flush=True)
    mdl, tok = load_model(cfg.model, quantize_4bit=False)

    if cfg.validate:
        run_gate_b(mdl, tok, cfg)
        return

    meta, diag = load_prmbench(n_samples=cfg.n_samples, seed=cfg.seed,
                               dataset_id=cfg.dataset, revision=cfg.revision)
    print(f"[prmbench-tf] loaded {len(meta)} meta rows: {json.dumps(diag)}", flush=True)

    items = build_items(tok, meta, cfg.prompt_suffix)
    n_misaligned = sum(1 for it in items if it["align_diag"]["problems"])
    print(f"[prmbench-tf] built {len(items)} items, {n_misaligned} with alignment problems",
          flush=True)

    out_path = os.path.join(out_dir, "prmbench_telemetry.pkl")
    cache = load_cache(out_path)
    t0 = time.time()
    if not score_rows(mdl, items, cache, out_path, cfg):
        print("[prmbench-tf] INCOMPLETE — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)

    stats = cell_stats(load_cache(out_path))
    manifest = {
        "driver": "run_prmbench_teacher_forced.py",
        "paper": "PRMBench (Song et al., arXiv:2501.03124, ACL 2025)",
        "model": cfg.model,
        "dataset": cfg.dataset,
        "dataset_revision": cfg.revision,
        "dataset_diagnostics": diag,
        "protocol": "teacher-forced single forward pass over the fixed official trace; no "
                    "generation; conditioning prompt = spectral_utils.prmbench.prmbench_prompt "
                    f"(math_prompt + {cfg.prompt_suffix!r})",
        "logprob_top_k": cfg.logprob_top_k,
        "gate_b_thresholds": {"median_max": GATE_B_MEDIAN_MAX, "min_corr": GATE_B_MIN_CORR},
        "cell_stats": stats,
        "n_items_misaligned": n_misaligned,
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
