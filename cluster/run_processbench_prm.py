#!/usr/bin/env python
"""
Qwen2.5-Math-PRM-7B supervised ceiling driver for the AIRCC cluster.

Scores the SAME `Qwen/ProcessBench` rows already teacher-forced for the GL-LIU study with a
PUBLISHED, human-label-trained process reward model, giving reasoning localization its first
supervised PRM ceiling (docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md
item 3). One forward pass per row (no generation) — cheap, like run_teacher_forced.py, but a
DIFFERENT model class (`AutoModel` classification head, not a causal LM) and a different score
(a published, human-supervision-trained per-step reward, not our own gray-box telemetry).

Named competitor (mandatory competitor gate):
    Qwen2.5-Math-PRM-7B — "Towards Effective Process Supervision in Mathematical Reasoning"
    (Qwen team, https://qwenlm.github.io/blog/qwen2.5-math-prm/)

Usage:
    python cluster/run_processbench_prm.py --subsets gsm8k,math \\
        --n-samples 30 --out $SHARED/results/pb_prm_qwen25math7b_pilot
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

from spectral_utils import load_cache, save_cache_atomic, free_memory
from spectral_utils.prm_scorer import PRM_MODEL_ID, load_prm_model, score_steps, localize_first_error
from spectral_utils.processbench import SUBSETS, NO_ERROR, load_processbench, first_error_f1

EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[pb-prm] SIGTERM received — will checkpoint after the current row", flush=True)


def run_subset(mdl, tok, rows, cfg, out_path):
    cache = load_cache(out_path)
    n_done = sum(1 for e in cache.values() if "rewards" in e)
    print(f"[pb-prm] {n_done}/{len(rows)} rows already done -> {out_path}", flush=True)

    for idx, row in enumerate(rows):
        if idx in cache and "rewards" in cache[idx]:
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            return False
        t0 = time.time()
        rewards = score_steps(mdl, tok, row["problem"], row["steps"])
        pred = localize_first_error(rewards, threshold=cfg.threshold)
        match = pred == int(row["label"])
        cache[idx] = {
            "id": row.get("id"),
            "generator": row.get("generator"),
            "problem": row["problem"],
            "steps": row["steps"],
            "label": int(row["label"]),
            "rewards": rewards,
            "prediction": pred,
            "match": bool(match),
        }
        print(f"[pb-prm] row {idx + 1}/{len(rows)}: pred={pred} label={row['label']} "
              f"match={match} n_steps={len(rewards)} {time.time() - t0:.2f}s", flush=True)
        if (idx + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=PRM_MODEL_ID)
    ap.add_argument("--subsets", default=",".join(SUBSETS))
    ap.add_argument("--n-samples", type=int, default=None, help="cap rows per subset (pilot)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="step-reward decision boundary; ProcessBench's own PRM baseline rule")
    ap.add_argument("--checkpoint-every", type=int, default=10)
    ap.add_argument("--attn-impl", default="sdpa")
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_pb_prm_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[pb-prm] model={cfg.model} subsets={cfg.subsets} n={cfg.n_samples} "
          f"threshold={cfg.threshold}", flush=True)
    mdl, tok = load_prm_model(cfg.model, attn_impl=cfg.attn_impl)

    cells = {}
    for subset in [s.strip() for s in cfg.subsets.split(",") if s.strip()]:
        out_path = os.path.join(out_dir, f"pb_prm_{subset}.pkl")
        rows = load_processbench(subset, cfg.n_samples)
        if not run_subset(mdl, tok, rows, cfg, out_path):
            print("[pb-prm] INCOMPLETE — resubmit with the same --out to resume", flush=True)
            sys.exit(EXIT_INCOMPLETE)
        cache = load_cache(out_path)
        stats = first_error_f1(cache)
        cells[subset] = stats
        print(f"=== {subset} DONE: n_error={stats['n_error']} n_correct={stats['n_correct']} "
              f"error_acc={stats['error_acc']} correct_acc={stats['correct_acc']} "
              f"F1={stats['f1']} ===", flush=True)

    manifest = {
        "driver": "run_processbench_prm.py",
        "paper": "Qwen2.5-Math-PRM-7B — Towards Effective Process Supervision in Mathematical "
                 "Reasoning (Qwen team, https://qwenlm.github.io/blog/qwen2.5-math-prm/)",
        "model": cfg.model,
        "protocol": "AutoModel classification head, system+user+assistant chat template, steps "
                    "joined by <extra_0> (+ trailing), positive-class softmax probability per "
                    "step, first step below --threshold = predicted first error",
        "threshold": cfg.threshold,
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
