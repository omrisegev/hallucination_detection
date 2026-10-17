#!/usr/bin/env python
"""
PRMBench step voters: the uPRM LLM-as-a-Judge control and the ProcessBench critic, on PRMBench.

Purpose: put the same non-PRM step scorers we already have on ProcessBench (Qwen3-8B and
Qwen2.5-14B judges, Qwen2.5-72B critic) on PRMBench, where every step is labelled, so they can
serve as label-free voters next to Qwen2.5-Math-PRM-7B (and any further PRM run through
cluster/run_prmbench_prm_telemetry.py) for MoM error counting and L-SML fusion.

Scoring code is reused unchanged:
  --kind uprm_judge  spectral_utils.uprm_baseline.score_candidates (one forward pass per row).
                     Also stores the per-step P(step incorrect) recovered exactly from the
                     candidate scores: S(j) = log p-_j + sum_{t<j} log p+_t, and the two marker
                     probabilities are renormalized over {+,-}, so log p+_t = log1p(-exp(log p-_t)).
  --kind critic      spectral_utils.processbench.critic_prompt + generate_full (greedy, 8192),
                     i.e. ProcessBench's own critic protocol; yields one first-error index.

Rows come from spectral_utils.prmbench.load_prmbench (the same loader as run_prmbench_prm.py),
keyed by its `idx`, so outputs join 1:1 with the PRM-7B cache. Gold `error_steps` are 1-based;
`first_error` stored here is 0-based (-1 = no error) and is for evaluation only — nothing in this
driver reads it.

Sharding (--n-shards K --shard i) takes rows i, i+K, ... so a long critic run can be split over
parallel GPUs; each shard writes its own pkl. Resume/SIGTERM/exit-85 follow run_inference.py.

Usage (pilot):
    python cluster/run_prmbench_step_voters.py --kind uprm_judge --model Qwen/Qwen3-8B \\
        --n-samples 30 --out $SHARED/results/prmbench_voters_v1/uprm_qwen3_8b_pilot
"""
import argparse
import json
import math
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
from spectral_utils.prmbench import PRMBENCH_DATASET_ID, load_prmbench

EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[prmbench-voters] SIGTERM received — will checkpoint after the current row", flush=True)


def step_error_logprobs(scores: dict) -> list:
    """Per-step log P(step incorrect) from uprm_baseline candidate scores (keys 1..T+1)."""
    T = len(scores) - 1
    out, cum_pos = [], 0.0
    for j in range(1, T + 1):
        lpn = scores[j] - cum_pos
        out.append(lpn)
        cum_pos += math.log1p(-math.exp(min(lpn, -1e-12)))
    if abs(cum_pos - scores[T + 1]) > 1e-4 * max(1.0, abs(cum_pos)):
        raise ValueError(f"per-step recovery does not reproduce S(T+1): {cum_pos} vs {scores[T + 1]}")
    return out


def gold_first_error(row) -> int:
    return min(row["error_steps"]) - 1 if row["error_steps"] else -1


def score_row(kind, mdl, tok, row, cfg):
    if kind == "uprm_judge":
        from spectral_utils.uprm_baseline import score_candidates, localize_first_error
        try:
            scores = score_candidates(mdl, tok, row["question"], row["steps"])
        except ValueError as e:
            # tokenization-assumption break (see uprm_baseline docstring): record, don't crash
            return {"scores": {}, "step_logp_err": None, "prediction": None, "failed": str(e)}
        try:
            step_err, recovery_error = step_error_logprobs(scores), None
        except ValueError as e:
            # the candidate scores stay valid; only the convenience per-step view is missing
            step_err, recovery_error = None, str(e)
        return {"scores": scores, "step_logp_err": step_err, "recovery_error": recovery_error,
                "prediction": localize_first_error(scores), "failed": None}
    from spectral_utils import generate_full
    from spectral_utils.processbench import critic_prompt, extract_critic_prediction
    r = generate_full(mdl, tok, critic_prompt({"problem": row["question"], "steps": row["steps"]}),
                      temperature=0.0, max_new_tokens=cfg.max_new,
                      logprob_top_k=cfg.logprob_top_k, system_message=None)
    return {"full_text": r["full_text"], "generated_critique": r["full_text"],
            "prediction": extract_critic_prediction(r["full_text"]),
            "token_entropies": r["token_entropies"], "gen_token_ids": r["gen_token_ids"],
            "top_k_logprobs": r.get("top_k_logprobs"),
            "truncated": len(r["gen_token_ids"]) >= cfg.max_new}


def run(kind, mdl, tok, rows, cfg, out_path):
    cache = load_cache(out_path)
    print(f"[prmbench-voters] {len(cache)}/{len(rows)} rows already done -> {out_path}", flush=True)
    for k, row in enumerate(rows):
        if row["idx"] in cache:
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} rows", flush=True)
            return False
        t0 = time.time()
        rec = score_row(kind, mdl, tok, row, cfg)
        rec.update(idx=row["idx"], source_idx=row["source_idx"], classification=row["classification"],
                   category=row["category"], n_steps=len(row["steps"]), error_steps=row["error_steps"],
                   first_error=gold_first_error(row), elapsed_sec=time.time() - t0)
        cache[row["idx"]] = rec
        print(f"[prmbench-voters] {k + 1}/{len(rows)} {row['idx']}: pred={rec['prediction']} "
              f"n_steps={rec['n_steps']} {rec['elapsed_sec']:.1f}s", flush=True)
        if (k + 1) % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)
    save_cache_atomic(cache, out_path)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--kind", choices=["uprm_judge", "critic"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default=PRMBENCH_DATASET_ID)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--n-samples", type=int, default=None, help="cap RAW rows (pilot only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new", type=int, default=8192, help="ProcessBench run_eval.py default")
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--attn-impl", default="sdpa")
    cfg = ap.parse_args()
    if not 0 <= cfg.shard < cfg.n_shards:
        ap.error("--shard must be in [0, n_shards)")
    if cfg.checkpoint_every is None:
        cfg.checkpoint_every = 5 if cfg.kind == "critic" else 25

    signal.signal(signal.SIGTERM, _on_sigterm)
    os.makedirs(cfg.out, exist_ok=True)
    meta, diag = load_prmbench(n_samples=cfg.n_samples, seed=cfg.seed,
                               dataset_id=cfg.dataset, revision=cfg.revision)
    rows = meta[cfg.shard::cfg.n_shards]
    print(f"[prmbench-voters] kind={cfg.kind} model={cfg.model} rows={len(rows)}/{len(meta)} "
          f"shard={cfg.shard}/{cfg.n_shards} diag={json.dumps(diag)}", flush=True)

    from spectral_utils import load_model
    mdl, tok = load_model(cfg.model, attn_impl=cfg.attn_impl)
    mdl.eval()

    tag = f"_shard{cfg.shard}of{cfg.n_shards}" if cfg.n_shards > 1 else ""
    out_path = os.path.join(cfg.out, f"prmbench_{cfg.kind}{tag}.pkl")
    t0 = time.time()
    if not run(cfg.kind, mdl, tok, rows, cfg, out_path):
        print("[prmbench-voters] INCOMPLETE — resubmit with the same args to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)

    cache = load_cache(out_path)
    stats = {"n_rows": len(cache),
             "n_unscored": sum(e["prediction"] is None for e in cache.values()),
             "n_failed": sum(bool(e.get("failed")) for e in cache.values()),
             "n_truncated": sum(bool(e.get("truncated")) for e in cache.values())}
    manifest = {
        "driver": "run_prmbench_step_voters.py", "kind": cfg.kind, "model": cfg.model,
        "dataset": cfg.dataset, "dataset_revision": cfg.revision, "dataset_diagnostics": diag,
        "protocol": ("uPRM paper Eq.6 LLM-as-a-Judge control, our marker reconstruction "
                     "(spectral_utils/uprm_baseline.py)") if cfg.kind == "uprm_judge" else
                    "ProcessBench critic prompt, greedy, single user turn, no system message",
        "n_samples": cfg.n_samples, "shard": cfg.shard, "n_shards": cfg.n_shards,
        "max_new_tokens": cfg.max_new if cfg.kind == "critic" else None,
        "attn_impl": cfg.attn_impl, "stats": stats, "elapsed_sec": time.time() - t0,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = os.path.join(cfg.out, f"manifest_{cfg.kind}{tag}.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    print(f"COMPLETE {json.dumps(stats)} -> {cfg.out}", flush=True)
    free_memory()


if __name__ == "__main__":
    main()
