#!/usr/bin/env python
"""
uPRM's own "LLM-as-a-Judge" control driver for the AIRCC cluster.

Reproduces the cheap, no-training control from "Unsupervised Process Reward Models"
(Gadetsky et al., EPFL, arXiv:2605.10158) that uPRM itself is measured against in their own
Table 1 — NOT uPRM, which requires training a new LoRA-tuned model via RL (~44 GPU-hours on
8xH200, see the paper's Appendix B). One forward pass per row (no generation), like
run_teacher_forced.py and run_processbench_prm.py, but reading two specific marker-token
probabilities instead of full-vocab entropy or a reward head. See
spectral_utils/uprm_baseline.py's module docstring for the marker/tokenization details and the
real bug (BPE-merging the marker with its following separator) this module's own checks catch.

Named competitor (mandatory competitor gate):
    "LLM-as-a-Judge" control from Gadetsky et al., arXiv:2605.10158 (uPRM paper), Eq. (6) —
    reported as OUR reproduction of THEIR baseline, never as uPRM itself.

Usage:
    python cluster/run_processbench_uprm_baseline.py --model Qwen/Qwen3-8B \\
        --subsets gsm8k,math --n-samples 30 --out $SHARED/results/pb_uprm_baseline_qwen3_8b_pilot
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

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

from spectral_utils import load_model, load_cache, save_cache_atomic, free_memory
from spectral_utils.uprm_baseline import score_candidates, localize_first_error
from spectral_utils.processbench import SUBSETS, NO_ERROR, load_processbench, first_error_f1

EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[pb-uprm-base] SIGTERM received — will checkpoint after the current row", flush=True)


def run_subset(mdl, tok, rows, cfg, out_path):
    cache = load_cache(out_path)
    n_done = sum(1 for e in cache.values() if "scores" in e)
    print(f"[pb-uprm-base] {n_done}/{len(rows)} rows already done -> {out_path}", flush=True)

    for idx, row in enumerate(rows):
        if idx in cache and "scores" in cache[idx]:
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            return False
        t0 = time.time()
        try:
            scores = score_candidates(mdl, tok, row["problem"], row["steps"])
            pred = localize_first_error(scores)
            failed = None
        except ValueError as e:
            # a tokenization-assumption break (see module docstring) — record and move on
            # rather than crash the whole subset over one row's BPE quirk.
            scores, pred, failed = {}, None, str(e)
        match = (pred is not None) and (pred == int(row["label"]))
        cache[idx] = {
            "id": row.get("id"),
            "generator": row.get("generator"),
            "problem": row["problem"],
            "steps": row["steps"],
            "label": int(row["label"]),
            "scores": scores,
            "prediction": pred,
            "match": bool(match),
            "failed": failed,
        }
        print(f"[pb-uprm-base] row {idx + 1}/{len(rows)}: pred={pred} label={row['label']} "
              f"match={match} failed={failed is not None} {time.time() - t0:.2f}s", flush=True)
        if (idx + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    return True


def cell_stats(cache: dict) -> dict:
    scored = {i: e for i, e in cache.items() if e["prediction"] is not None}
    stats = first_error_f1(scored)
    stats["n_rows"] = len(cache)
    stats["n_failed_tokenization"] = len(cache) - len(scored)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="matches the model already used for the other 8 teacher-forced "
                         "ProcessBench cells in this project")
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--n-samples", type=int, default=None, help="cap rows per subset (pilot)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--attn-impl", default="sdpa")
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_pb_uprm_base_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[pb-uprm-base] model={cfg.model} subsets={cfg.subsets} n={cfg.n_samples}", flush=True)
    mdl, tok = load_model(cfg.model, attn_impl=cfg.attn_impl)
    mdl.eval()

    cells = {}
    for subset in [s.strip() for s in cfg.subsets.split(",") if s.strip()]:
        out_path = os.path.join(out_dir, f"pb_uprm_base_{subset}.pkl")
        rows = load_processbench(subset, cfg.n_samples)
        if not run_subset(mdl, tok, rows, cfg, out_path):
            print("[pb-uprm-base] INCOMPLETE — resubmit with the same --out to resume", flush=True)
            sys.exit(EXIT_INCOMPLETE)
        cache = load_cache(out_path)
        stats = cell_stats(cache)
        cells[subset] = stats
        print(f"=== {subset} DONE: n_error={stats['n_error']} n_correct={stats['n_correct']} "
              f"error_acc={stats['error_acc']} correct_acc={stats['correct_acc']} "
              f"F1={stats['f1']} | n_failed_tokenization={stats['n_failed_tokenization']} ===",
              flush=True)

    manifest = {
        "driver": "run_processbench_uprm_baseline.py",
        "paper": "\"LLM-as-a-Judge\" control from Unsupervised Process Reward Models "
                 "(Gadetsky et al., EPFL, arXiv:2605.10158), Eq. (6) — NOT uPRM itself",
        "model": cfg.model,
        "protocol": "our marker/prompt reconstruction (paper publishes no code) — see "
                    "spectral_utils/uprm_baseline.py module docstring",
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
