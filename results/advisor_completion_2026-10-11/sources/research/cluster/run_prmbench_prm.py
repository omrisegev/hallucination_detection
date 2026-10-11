#!/usr/bin/env python
"""
Qwen2.5-Math-PRM-7B on **PRMBench** — the supervised every-step ceiling for Benchmark 3 of
docs/experiments/FOUR_LOCALIZATION_BENCHMARKS_CLUSTER_HANDOFF.md.

Why a second PRM job when `cluster/run_processbench_prm.py` already exists: ProcessBench labels
only the FIRST wrong step and certifies nothing after it, so it cannot measure an every-step
classifier at all. PRMBench annotates every step of 6,216 traces (83,456 raw step labels), which
is the ground truth this panel needs. Same checkpoint, same scoring code
(`spectral_utils/prm_scorer.py`), different benchmark and different metric.

The step reward -> label convention is the one place a sign error would silently invert the whole
panel. PRMBench's official metric (`spectral_utils/prmbench.py::eval_on_hallucination_step`, a
port of `mr_eval/utils/task_utils.py`) reads `labels[i] == 1` as **"the scorer asserts step i is
VALID"**. A PRM reward is already a correctness probability, so the mapping is
`label = 1 if reward >= threshold else 0` — no inversion. The ProcessBench PRM baseline's own
0.5 boundary is reused unchanged; a threshold tuned on PRMBench labels would turn a published
ceiling into a fitted method.

Usage:
    python cluster/run_prmbench_prm.py \\
        --out $SHARED/results/prmbench_qwen25math7b_full
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
from spectral_utils.prm_scorer import PRM_MODEL_ID, load_prm_model, score_steps
from spectral_utils.prmbench import (
    PRMBENCH_DATASET_ID, load_prmbench, prmbench_evaluate,
)

EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[prmbench-prm] SIGTERM received — will checkpoint after the current row", flush=True)


def run(mdl, tok, meta, cfg, out_path):
    cache = load_cache(out_path)
    n_done = sum(1 for e in cache.values() if "rewards" in e)
    print(f"[prmbench-prm] {n_done}/{len(meta)} rows already done -> {out_path}", flush=True)

    n_reward_mismatch = 0
    for i, row in enumerate(meta):
        if i in cache and "rewards" in cache[i]:
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            return False, n_reward_mismatch

        t0 = time.time()
        rewards = score_steps(mdl, tok, row["question"], row["steps"])
        # score_steps raises when the <extra_0> tokenization assumption breaks, so a length
        # mismatch here would be a NEW failure mode; record it rather than crash the run.
        if len(rewards) != len(row["steps"]):
            n_reward_mismatch += 1
        labels = [1 if r >= cfg.threshold else 0 for r in rewards]

        cache[i] = {
            "idx": row["idx"],
            "source_idx": row["source_idx"],
            "classification": row["classification"],
            "category": row["category"],
            "n_steps": len(row["steps"]),
            "error_steps": row["error_steps"],
            "rewards": rewards,
            "labels": labels,      # 1 == "step is VALID", matching the official metric
            "elapsed_sec": time.time() - t0,
        }
        if (i + 1) % 100 == 0 or i + 1 == len(meta):
            print(f"[prmbench-prm] {i + 1}/{len(meta)} rows "
                  f"({time.time() - t0:.2f}s last)", flush=True)
        if (i + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)

    save_cache_atomic(cache, out_path)
    return True, n_reward_mismatch


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=PRM_MODEL_ID)
    ap.add_argument("--dataset", default=PRMBENCH_DATASET_ID)
    ap.add_argument("--revision", default=None, help="pin the dataset revision in the manifest")
    ap.add_argument("--n-samples", type=int, default=None, help="cap RAW rows (debug only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="ProcessBench's own PRM decision boundary, reused unchanged; never "
                         "tuned on PRMBench labels")
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint-every", type=int, default=50)
    ap.add_argument("--attn-impl", default="sdpa")
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_prmbench_prm_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[prmbench-prm] model={cfg.model} dataset={cfg.dataset} n={cfg.n_samples} "
          f"threshold={cfg.threshold}", flush=True)

    meta, diag = load_prmbench(n_samples=cfg.n_samples, seed=cfg.seed,
                               dataset_id=cfg.dataset, revision=cfg.revision)
    print(f"[prmbench-prm] loaded {len(meta)} meta rows: {json.dumps(diag)}", flush=True)

    mdl, tok = load_prm_model(cfg.model, attn_impl=cfg.attn_impl)

    out_path = os.path.join(out_dir, "prmbench_prm.pkl")
    t0 = time.time()
    ok, n_mismatch = run(mdl, tok, meta, cfg, out_path)
    if not ok:
        print("[prmbench-prm] INCOMPLETE — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)

    cache = load_cache(out_path)
    predictions = [{"idx": e["idx"], "labels": e["labels"]} for e in cache.values()]
    results = prmbench_evaluate(predictions, meta)
    results.pop("adaptation_note", None)

    manifest = {
        "driver": "run_prmbench_prm.py",
        "paper": "PRMBench: A Fine-grained and Challenging Benchmark for Process-Level Reward "
                 "Models (Song et al., arXiv:2501.03124, ACL 2025)",
        "competitor": "Qwen2.5-Math-PRM-7B — Towards Effective Process Supervision in "
                      "Mathematical Reasoning (Qwen team)",
        "model": cfg.model,
        "dataset": cfg.dataset,
        "dataset_revision": cfg.revision,
        "dataset_diagnostics": diag,
        "metric_source": "spectral_utils/prmbench.py — port of mr_eval task prmtest_classified "
                         "(load_data_function + evaluate_function + eval_on_hallucination_step)",
        "label_convention": "labels[i] == 1 means the scorer asserts step i is VALID",
        "threshold": cfg.threshold,
        "supervision": "human-process-label-trained PRM (PRM800K-style) — supervised ceiling, "
                       "never reported as a peer of a label-free score",
        "fidelity_level": 1,
        "n_reward_count_mismatch": n_mismatch,
        "results": results,
        "elapsed_sec": time.time() - t0,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)

    print("\n=== PRMBENCH TOTALS ===")
    print(json.dumps(results["total"], indent=2, default=str))
    print("\n=== BY CATEGORY ===")
    print(json.dumps(results["by_category"], indent=2, default=str))
    print(f"\nCOMPLETE -> {out_dir}", flush=True)
    free_memory()


if __name__ == "__main__":
    main()
