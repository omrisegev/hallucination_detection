#!/usr/bin/env python
"""
Evidence-condition teacher-forced rescoring driver for RAGTruth (AIRCC cluster).

For each fixed, already-published RAGTruth response, rescore it under several evidence
conditions (full context, no context, leave-one-chunk-out) via teacher-forcing. NOTHING IS
GENERATED. The response text is fixed; only the CONTEXT inside its prompt changes between
conditions, via `spectral_utils.ragtruth.condition_prompt`'s prompt-surgery (edit the
evidence substring in place, keep every other character of the prompt identical).

This is the RAGTruth analogue of `cluster/run_teacher_forced.py`'s ProcessBench driver, and
reuses the SAME forward-pass primitives on purpose:
    forward_batch(mdl, items)   -> right-padded batched logits sliced to [T_gen, V]
    candidate_quantities(...)   -> token_logsumexp / top_k_logprobs_raw / spilled / H
A work item here is a (response, condition) PAIR, not a row: `gen_ids` (the tokenized
response) is IDENTICAL across every condition of one response and computed once; only
`prompt_ids` changes per condition. Cache key is `f"{response_id}::{condition}"`, so resume
after preemption is exact at the (response, condition) granularity, not the response.

GATE B (--validate) IS NOT OPTIONAL — see run_teacher_forced.py's docstring for why. Here it
is repointed at a GENERATED cell of the SAME scorer model (Qwen2.5-1.5B-Instruct), because
Qwen2.5-Instruct's default `repetition_penalty=1.05` changes the entropy trace even under
greedy decoding; the validation preset generates with `repetition_penalty=1.0` explicitly so
Gate B validates the exact penalty-free code path this driver's `warpers=None` call assumes.
See `docs/research_notes/ragtruth_ec_preregistration_v1.md`.

Preemption-safe on the `run_inference.py` contract: SIGTERM sets a flag, the loop checkpoints
via `save_cache_atomic` and exits EXIT_INCOMPLETE (85) so a chained `--dependency=afterany`
job resumes; presence of a (response_id, condition) key in the cache is the resume marker.

Usage:
    python cluster/run_conditional_rescore.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --split test --out $SHARED/results/ragtruth_ec_qwen25_15b
    python cluster/run_conditional_rescore.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --validate --validate-pkl $SHARED/results/gateb_gsm8k_qwen25_15b/raw_gsm8k_T0.0.pkl
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
from spectral_utils.model_utils import fmt_prompt
from spectral_utils.ragtruth import (
    TASK_TYPES, load_ragtruth, distinct_conditions, condition_prompt,
    response_token_spans,
)

EXIT_INCOMPLETE = 85
STOP = {"flag": False}

# Same thresholds as run_teacher_forced.py's Gate B (backfill_views' measured bf16
# batched-decode noise floor).
GATE_B_MEDIAN_MAX = 0.05
GATE_B_MIN_CORR = 0.999

# Qwen2.5-1.5B-Instruct's context window is 32768; guard well under it so a pathological
# RAGTruth prompt (long Data2txt structured record) can never silently truncate.
MAX_PROMPT_PLUS_RESPONSE_TOKENS = 30_000


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[conditional-rescore] SIGTERM received — will checkpoint after the current batch",
          flush=True)


def build_items(tok, rows):
    """One work item per (response, condition) pair sharing one response tokenization.

    Rows whose prompt+response would exceed `MAX_PROMPT_PLUS_RESPONSE_TOKENS` are skipped
    entirely (all their conditions) and counted, never silently truncated.
    """
    items, skipped_long = [], 0
    for row in rows:
        gen_ids, offsets, span_token_spans, align_diag = response_token_spans(
            tok, row["response"], row["labels"]
        )
        conditions = distinct_conditions(row)
        prompt_token_lens = {}
        too_long = False
        for condition in conditions:
            prompt = condition_prompt(row, condition)
            prompt_ids = tok(prompt).input_ids
            prompt_token_lens[condition] = len(prompt_ids)
            if len(prompt_ids) + len(gen_ids) > MAX_PROMPT_PLUS_RESPONSE_TOKENS:
                too_long = True
        if too_long:
            skipped_long += 1
            continue
        for condition in conditions:
            prompt = condition_prompt(row, condition)
            items.append({
                "key": f"{row['response_id']}::{condition}",
                "response_id": row["response_id"],
                "source_id": row["source_id"],
                "condition": condition,
                "prompt_ids": tok(prompt).input_ids,
                "gen_ids": gen_ids,
                "span_token_spans": span_token_spans,
                "align_diag": align_diag,
                "row": row,
            })
    return items, skipped_long


def batched(items, max_batch_tokens: int, max_batch: int = 8):
    """Length-sorted batches capped by total padded tokens (identical policy to
    `run_teacher_forced.batched` — see its docstring)."""
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


def score_items(mdl, items, cache, out_path, cfg):
    """Teacher-force every not-yet-done (response, condition) item."""
    todo = [it for it in items if it["key"] not in cache]
    print(f"[conditional-rescore] {len(cache)} items already done, {len(todo)} to go", flush=True)
    n_done = 0

    for batch in batched(todo, cfg.max_batch_tokens, cfg.max_batch):
        if STOP["flag"]:
            save_cache_atomic(cache, out_path)
            print(f"PREEMPTED — checkpoint saved with {len(cache)} items", flush=True)
            return False
        t0 = time.time()
        slices = forward_batch(mdl, batch)
        for it, logits in zip(batch, slices):
            # warpers=None: pure measurement, nothing was generated by this scorer. See the
            # module docstring for why repetition_penalty must be handled at the Gate-B
            # validation preset rather than mirrored here.
            q = candidate_quantities(
                logits, it["gen_ids"], warpers=None,
                raw_top_k=cfg.logprob_top_k, post_top_k=cfg.logprob_top_k,
            )
            row = it["row"]
            cache[it["key"]] = {
                "response_id": it["response_id"],
                "source_id": it["source_id"],
                "condition": it["condition"],
                "task_type": row["task_type"],
                "source": row["source"],
                "generator_model": row["model"],
                "quality": row["quality"],
                "response_label": bool(row["labels"]),
                "span_labels": row["labels"],
                "span_token_spans": it["span_token_spans"],
                "align_diag": it["align_diag"],
                "prompt_len": len(it["prompt_ids"]),
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
        print(f"[conditional-rescore] batch of {len(batch)} in {time.time()-t0:.1f}s "
              f"({len(cache)}/{len(items)} items)", flush=True)

    save_cache_atomic(cache, out_path)
    return True


def cell_stats(cache):
    by_condition, by_task = {}, {}
    for entry in cache.values():
        by_condition[entry["condition"]] = by_condition.get(entry["condition"], 0) + 1
        by_task[entry["task_type"]] = by_task.get(entry["task_type"], 0) + 1
    n_responses = len({entry["response_id"] for entry in cache.values()})
    n_unmapped = sum(entry["align_diag"]["n_unmapped"] for entry in cache.values())
    return {
        "n_items": len(cache), "n_responses": n_responses,
        "by_condition": by_condition, "by_task_type": by_task,
        "n_span_labels_unmapped": n_unmapped,
    }


def run_gate_b(mdl, tok, cfg):
    """Gate B: reproduce a GENERATED Qwen2.5-1.5B cell's saved `token_entropies` with this
    driver's own teacher-forced path. See the module docstring for the repetition-penalty
    trap this specifically guards against."""
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

    floor = None
    if not ok:
        print("[gate-b] threshold missed — measuring the self-consistency floor "
              "(same driver, batch=1 vs batch=%d)" % cfg.max_batch, flush=True)
        floor = _self_consistency(mdl, items, cfg)
        if floor:
            print(f"[gate-b] FLOOR: median|d|={floor['median']:.5f} "
                  f"p99|d|={floor['p99']:.5f} median_r={floor['median_r']:.6f}", flush=True)
            ok = (med <= GATE_B_MEDIAN_MAX
                  and med <= max(10 * floor["median"], 1e-3)
                  and med_r >= min(GATE_B_MIN_CORR, floor["median_r"] - 0.002))
            print(f"[gate-b] vs floor -> {'PASS' if ok else 'FAIL'}", flush=True)

    print(f"[gate-b] {'PASS' if ok else 'FAIL'} "
          f"(need median|d| <= {GATE_B_MEDIAN_MAX} and median_r >= {GATE_B_MIN_CORR}, "
          f"or within the measured self-consistency floor)", flush=True)
    if not ok:
        raise SystemExit(
            "[gate-b] FAILED — the prompt/template/repetition-penalty handling does not "
            "reproduce the saved trace. Do NOT trust RAGTruth output from this config. "
            "Check the validation preset's repetition_penalty=1.0 setting first."
        )


def _self_consistency(mdl, items, cfg):
    def run(batch_size):
        out = []
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            for k, logits in enumerate(forward_batch(mdl, chunk)):
                q = candidate_quantities(logits, chunk[k]["gen_ids"], warpers=None,
                                         raw_top_k=cfg.logprob_top_k,
                                         post_top_k=cfg.logprob_top_k)
                out.append(np.asarray(q["token_entropies_recomputed"], dtype=np.float64))
        return out

    a_all, b_all = run(cfg.max_batch), run(1)
    diffs, corrs = [], []
    for a, b in zip(a_all, b_all):
        n = min(len(a), len(b))
        if n < 3:
            continue
        diffs.append(np.abs(a[:n] - b[:n]))
        if a[:n].std() > 1e-12 and b[:n].std() > 1e-12:
            corrs.append(float(np.corrcoef(a[:n], b[:n])[0, 1]))
    if not diffs:
        return None
    d = np.concatenate(diffs)
    return {"median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
            "median_r": float(np.median(corrs)) if corrs else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--task-types", default=",".join(TASK_TYPES),
                    help=f"comma-separated RAGTruth task types (of {TASK_TYPES})")
    ap.add_argument("--n-samples", type=int, default=None, help="cap responses (pilot)")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for --n-samples/--source-ids-file")
    ap.add_argument("--source-ids-file", default=None,
                    help="restrict to these source_ids (one per line) — for a dev slice "
                         "drawn from the train split via spectral_utils.ragtruth.group_split")
    ap.add_argument("--out", default=None)
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--validate", action="store_true", help="run Gate B and exit")
    ap.add_argument("--validate-pkl", default=None)
    ap.add_argument("--validate-dataset", default="gsm8k")
    ap.add_argument("--validate-n", type=int, default=20)
    cfg = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    out_dir = cfg.out or f"results_ragtruth_ec_{cfg.model.split('/')[-1]}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"[conditional-rescore] model={cfg.model} split={cfg.split} n={cfg.n_samples}",
          flush=True)
    if torch.cuda.is_available():
        print(f"[conditional-rescore] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("[conditional-rescore] WARNING: no CUDA device — running on CPU", flush=True)

    mdl, tok = load_model(cfg.model)
    mdl.eval()

    if cfg.validate:
        if not cfg.validate_pkl:
            raise SystemExit("--validate needs --validate-pkl (a GENERATED cell with "
                             "gen_token_ids + token_entropies saved, ideally generated "
                             "with repetition_penalty=1.0 — see module docstring)")
        run_gate_b(mdl, tok, cfg)
        return

    task_types = [t.strip() for t in cfg.task_types.split(",") if t.strip()]
    rows, load_diag = load_ragtruth(split=cfg.split, task_types=task_types,
                                    n=cfg.n_samples, seed=cfg.seed)
    print(f"[conditional-rescore] loaded {load_diag}", flush=True)

    if cfg.source_ids_file:
        with open(cfg.source_ids_file) as handle:
            wanted = {line.strip() for line in handle if line.strip()}
        rows = [row for row in rows if row["source_id"] in wanted]
        print(f"[conditional-rescore] restricted to {len(wanted)} source_ids -> "
              f"{len(rows)} responses", flush=True)

    items, skipped_long = build_items(tok, rows)
    print(f"[conditional-rescore] {len(rows)} responses -> {len(items)} (response, condition) "
          f"items; {skipped_long} response(s) skipped for exceeding "
          f"{MAX_PROMPT_PLUS_RESPONSE_TOKENS} prompt+response tokens", flush=True)

    out_path = os.path.join(out_dir, f"ragtruth_ec_{cfg.split}.pkl")
    cache = load_cache(out_path)
    if not score_items(mdl, items, cache, out_path, cfg):
        print("[conditional-rescore] INCOMPLETE — resubmit with the same --out to resume",
              flush=True)
        sys.exit(EXIT_INCOMPLETE)

    st = cell_stats(cache)
    print(f"=== DONE: {st['n_items']} items / {st['n_responses']} responses | "
          f"by_condition={st['by_condition']} | by_task_type={st['by_task_type']} | "
          f"unmapped span labels={st['n_span_labels_unmapped']} ===", flush=True)

    manifest = {
        "driver": "run_conditional_rescore.py",
        "corpus": "RAGTruth (Niu et al. 2024, ACL) — data/ragtruth_protocol/PROVENANCE.md",
        "model": cfg.model,
        "protocol": "teacher-forced rescoring of a FIXED published response under multiple "
                    "evidence conditions; nothing is generated",
        "split": cfg.split, "task_types": task_types,
        "n_samples_requested": cfg.n_samples, "seed": cfg.seed,
        "source_ids_file": cfg.source_ids_file,
        "logprob_top_k": cfg.logprob_top_k,
        "max_prompt_plus_response_tokens": MAX_PROMPT_PLUS_RESPONSE_TOKENS,
        "n_skipped_too_long": skipped_long,
        "job_id": os.environ.get("SLURM_JOB_ID", ""),
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stats": st, "load_diagnostics": load_diag,
    }
    path = os.path.join(out_dir, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    print(f"\nALL ITEMS COMPLETE -> {out_dir}", flush=True)
    free_memory()


if __name__ == "__main__":
    main()
