#!/usr/bin/env python
"""
Standalone GPU inference driver for the AIRCC cluster (Slurm + rootless Docker, B200).

Runs K-candidate sampling per problem and saves the richest raw form per candidate
(full_text, token_entropies, token_spilled_energies, token_offsets, top_k_logprobs,
gen_token_ids, label) so every future feature/baseline can be derived offline —
including offline re-grading, since full_text is kept.

Preemption-safe: Slurm sends SIGTERM 15 minutes before SIGKILL (--signal=B:TERM@900
in the sbatch template, forwarded into the container). The handler sets a flag that
is checked after every single generation; the driver checkpoints atomically and
exits 0. The job is auto-requeued and resumes exactly where it stopped — completed
problems are skipped, partially-done problems are filled from the missing candidate.

Example (AIME24, the missing EDIS-paper dataset):
    python cluster/run_inference.py --dataset aime24 --temps 0.2,0.6,1.0 \
        --k 8 --n-samples 30 --max-new 1024 --out /shared/.../results/edis_aime24
"""
import argparse
import os
import signal
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

from spectral_utils import load_model, generate_full, load_cache, save_cache_atomic
from spectral_utils.data_loaders import (
    load_gsm8k, gsm8k_prompt, is_correct_gsm8k,
    load_math500, math_prompt, is_correct_math,
    load_amc23, amc23_prompt, is_correct_amc23,
    load_aime24, aime24_prompt, is_correct_aime24,
)

# dataset -> (loader(n) -> list[row dict], prompt_fn(row) -> str, grader(gen, row) -> bool)
DATASETS = {
    "gsm8k":   (lambda n: load_gsm8k()[:n], gsm8k_prompt,  is_correct_gsm8k),
    "math500": (load_math500,               math_prompt,   is_correct_math),
    "amc23":   (load_amc23,                 amc23_prompt,  is_correct_amc23),
    "aime24":  (load_aime24,                aime24_prompt, is_correct_aime24),
}

STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[driver] SIGTERM received — will checkpoint after current generation", flush=True)


def question_text(row) -> str:
    if isinstance(row, dict):
        return row.get("question") or row.get("problem") or row.get("query") or ""
    return str(row)


def run_temp(mdl, tok, rows, prompt_fn, grader, temp, args, out_path) -> bool:
    """Run one (dataset, temperature) cell. Returns False if preempted mid-run."""
    cache = load_cache(out_path)
    n_done = sum(1 for e in cache.values() if len(e["candidates"]) >= args.k)
    print(f"\n=== T={temp} -> {out_path} ({n_done}/{len(rows)} problems already complete) ===",
          flush=True)

    for idx, row in enumerate(rows):
        entry = cache.setdefault(idx, {
            "question": question_text(row),
            "gold_row": row,
            "candidates": [],
        })
        dirty = False
        while len(entry["candidates"]) < args.k:
            if STOP["flag"]:
                save_cache_atomic(cache, out_path)
                print(f"PREEMPTED — checkpoint saved at T={temp} problem={idx} "
                      f"candidate={len(entry['candidates'])}", flush=True)
                return False
            t0 = time.time()
            r = generate_full(mdl, tok, prompt_fn(row), temperature=temp,
                              max_new_tokens=args.max_new,
                              logprob_top_k=args.logprob_top_k)
            r["label"] = bool(grader(r["full_text"], row))
            entry["candidates"].append(r)
            print(f"[T={temp}] problem {idx + 1}/{len(rows)} cand "
                  f"{len(entry['candidates'])}/{args.k}: {len(r['token_entropies'])} tok, "
                  f"label={int(r['label'])}, {time.time() - t0:.1f}s", flush=True)
            dirty = True
        if dirty and (idx + 1) % args.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    labels = [c["label"] for e in cache.values() for c in e["candidates"]]
    lens = [len(c["token_entropies"]) for e in cache.values() for c in e["candidates"]]
    print(f"=== T={temp} DONE: {len(cache)} problems x {args.k} candidates | "
          f"accuracy {sum(labels) / max(len(labels), 1):.3f} | "
          f"mean trace {sum(lens) / max(len(lens), 1):.0f} tok ===", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--model", default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    ap.add_argument("--temps", default="0.2,0.6,1.0",
                    help="comma-separated sampling temperatures")
    ap.add_argument("--k", type=int, default=8, help="candidates per problem")
    ap.add_argument("--n-samples", type=int, default=30, help="number of problems")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--out", default=None,
                    help="output dir (default: results_<dataset> under CWD)")
    ap.add_argument("--checkpoint-every", type=int, default=1,
                    help="save cache every N completed problems")
    ap.add_argument("--logprob-top-k", type=int, default=50,
                    help="top-K logprobs saved per token (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    torch.manual_seed(args.seed)

    out_dir = args.out or f"results_{args.dataset}"
    os.makedirs(out_dir, exist_ok=True)
    temps = [float(t) for t in args.temps.split(",")]

    if torch.cuda.is_available():
        print(f"[driver] GPU: {torch.cuda.get_device_name(0)} "
              f"capability={torch.cuda.get_device_capability(0)}", flush=True)
    else:
        print("[driver] WARNING: no CUDA device — running on CPU", flush=True)

    loader, prompt_fn, grader = DATASETS[args.dataset]
    rows = loader(args.n_samples)
    mdl, tok = load_model(args.model)

    for temp in temps:
        out_path = os.path.join(out_dir, f"raw_{args.dataset}_T{temp}.pkl")
        if not run_temp(mdl, tok, rows, prompt_fn, grader, temp, args, out_path):
            sys.exit(0)  # preempted: clean exit, Slurm requeues and we resume

    print(f"\nALL TEMPS COMPLETE for {args.dataset} -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
