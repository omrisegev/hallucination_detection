#!/usr/bin/env python
"""
Qwen2.5-Math-PRM-7B on the external step benchmarks (Socratic-PRMBench, Hard2Verify).

Same checkpoint and scoring code as cluster/run_prmbench_prm.py (`spectral_utils/prm_scorer.py`,
the model card's 3-turn template with one `<extra_0>` after every step). The input is the
label-free `answers.json` already used for our Qwen3 telemetry (same uids, same `steps` lists,
so the PRM sees exactly the step segmentation our methods are scored on). No gold labels are
read here; evaluation happens offline with the frozen external evaluator.

Saves one record per uid: rewards (validity probability per step), n_steps, input token count.

Usage:
    python cluster/run_external_prm.py --answers .../socratic/answers.json --out $SHARED/results/external_prm_v1/socratic
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
from spectral_utils.prm_scorer import PRM_MODEL_ID, PRM_SYSTEM_PROMPT, load_prm_model, score_steps

EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[external-prm] SIGTERM received — will checkpoint after the current row", flush=True)


def n_tokens(tok, question, steps):
    msgs = [{"role": "system", "content": PRM_SYSTEM_PROMPT}, {"role": "user", "content": question},
            {"role": "assistant", "content": "<extra_0>".join(steps) + "<extra_0>"}]
    return len(tok.encode(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=PRM_MODEL_ID)
    ap.add_argument("--n-samples", type=int, default=None, help="debug cap")
    ap.add_argument("--checkpoint-every", type=int, default=50)
    cfg = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_sigterm)
    os.makedirs(cfg.out, exist_ok=True)

    rows = json.load(open(cfg.answers, encoding="utf8"))
    if cfg.n_samples:
        rows = rows[:cfg.n_samples]
    assert len({r["uid"] for r in rows}) == len(rows)
    out_path = os.path.join(cfg.out, "external_prm.pkl")
    cache = load_cache(out_path)
    print(f"[external-prm] {len(cache)}/{len(rows)} done -> {out_path}", flush=True)

    mdl, tok = load_prm_model(cfg.model)
    t_all = time.time()
    for i, r in enumerate(rows):
        if r["uid"] in cache:
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            sys.exit(EXIT_INCOMPLETE)
        t0 = time.time()
        steps = list(r["steps"])
        cache[r["uid"]] = {"uid": r["uid"], "n_steps": len(steps),
                           "n_input_tokens": n_tokens(tok, r["question"], steps),
                           "rewards": score_steps(mdl, tok, r["question"], steps),
                           "elapsed_sec": time.time() - t0}
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"[external-prm] {i + 1}/{len(rows)} ({time.time() - t0:.2f}s last)", flush=True)
        if (i + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)
    save_cache_atomic(cache, out_path)

    manifest = {"driver": "run_external_prm.py", "model": cfg.model, "answers": os.path.abspath(cfg.answers),
                "n_rows": len(cache), "n_input_tokens_max": max(e["n_input_tokens"] for e in cache.values()),
                "n_rows_over_4096_tokens": sum(e["n_input_tokens"] > 4096 for e in cache.values()),
                "label_convention": "reward = probability the step is VALID; native rule reward >= 0.5 -> VALID",
                "labels_read": False, "elapsed_sec": time.time() - t_all,
                "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(cfg.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"COMPLETE -> {cfg.out}", flush=True)
    free_memory()


if __name__ == "__main__":
    main()
