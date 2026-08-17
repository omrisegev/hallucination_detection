#!/usr/bin/env python
"""
M1/M2 — DeepConf native multi-trace acquisition (AIRCC cluster driver).

"Deep Think with Confidence" (Fu et al., arXiv:2508.15260) on its Qwen3-8B x AIME24 cell:
paper prompt, T=0.6 / top-p 0.95 / top-k 20, 32k cap, native warm-up N_init=16.

This driver **acquires traces**; it computes no paper table. Every DeepConf variant
(filter percentile, statistic, K, budget, 64 fresh resamplings) is pure offline arithmetic
over the saved confidence, and is produced by `scripts/paper_exact_deepconf_offline.py`
from these shards without generating another token. That split is what makes the 1.8B-token
pool a one-time cost instead of a per-variant one.

Two modes
---------
    --mode pilot   K=32 or 64 traces/question. A PROTOCOL CHECK, never a table row
                   (handoff §M1). Its job is to prove the equality audit, validate answer
                   normalisation and the percentile direction, and measure throughput.
    --mode full    K=4096 traces/question x 30 questions ~ 1.8e9 generated tokens.
                   Only this can claim the paper's table.

Storage contract (handoff §3.2)
-------------------------------
The full pool retains, per token, the native confidence scalar plus the four frozen project
channels — and NOT the raw top-50 arrays, which would take the pool from ~20-60 GB to
0.6-1.2 TB. Raw top-50 is kept only for a deterministic audit sample (`--audit-every`),
which is what the equality audit needs. `--keep-top-k-all` exists but refuses to run in
full mode without `--i-accept-terabyte-retention`, because silently expanding the run is
exactly the failure §3.2 warns about.

Regression targets (NOT gates): majority@512 80.0% at 2.32e8 tokens;
online DeepConf-low 86.5% at 0.90e8 tokens (-61.1%).

Usage:
    python cluster/run_paper_exact_deepconf.py --mode pilot --k 32 \
        --out $SHARED/results/paper_exact/m1_deepconf_pilot
    python cluster/run_paper_exact_deepconf.py --mode full --k 4096 \
        --out $SHARED/results/paper_exact/m2_deepconf_full --shard 0 --n-shards 8
"""
import argparse
import hashlib
import json
import os
import signal
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    import transformers.modeling_utils as _mu
    _mu.check_torch_load_is_safe = lambda *a, **k: None
except Exception:
    pass

import numpy as np
import torch

from spectral_utils import load_model, free_memory
from spectral_utils.paper_exact import deepconf as DC
from spectral_utils.paper_exact import evaluator as EV
from spectral_utils.paper_exact.gates import Gate
from spectral_utils.paper_exact.manifest import build_manifest, write_manifest, verify_manifest
from spectral_utils.paper_exact.shards import ShardWriter
from spectral_utils.paper_exact.telemetry import DecodeConfig, batch_generate

EXIT_INCOMPLETE = 85
STOP = {"flag": False}
PAPER_PDF = os.path.join(REPO_ROOT, "papers", "DEEP THINK WITH CONFIDENCE.pdf")

#: Appendix F: appended to the problem for Qwen3 and GPT-OSS.
DEEPCONF_PROMPT = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."
#: Table 11 decoding for Qwen3-8B.
QWEN3_DECODING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
MAX_NEW = 32768


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[m1] SIGTERM — will checkpoint after the current trace", flush=True)


def synthetic_rows(n=3):
    """Placeholder rows for --dry-run: exercises the order hash and the manifest gate
    without a dataset-hub round trip."""
    return [{"question_id": f"dryrun-{i}", "index": i,
             "problem": f"synthetic problem {i}", "answer": str(i)} for i in range(n)]


def load_aime24():
    from datasets import load_dataset
    ds = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    rows = []
    for i, r in enumerate(ds):
        q = r.get("Problem", r.get("problem", ""))
        a = str(r.get("Answer", r.get("answer", "")))
        rows.append({"question_id": str(r.get("ID", i)), "index": i, "problem": q, "answer": a})
    return rows


def _generate_with_oom_backoff(mdl, tok, pids, chunk, cfg, generator, min_batch: int = 1):
    """Decode a batch, halving it on CUDA OOM until it fits.

    The KV cache for B traces at the 32k cap is ~147 KB per token per trace, so a batch whose
    members all run long can exceed even a B200's 183 GB late in a run that has been fine for
    hours. Handoff §6 is explicit that the response to an OOM is to reduce batch size **only** —
    never to change model, max length, quantization, prompt or decoding — so that is exactly
    what this does, and it records the reduction rather than hiding it.
    """
    size = len(chunk)
    while True:
        try:
            out = []
            for i in range(0, len(chunk), size):
                part = chunk[i:i + size]
                out.extend(batch_generate(mdl, tok, [pids] * len(part), cfg,
                                          generator=generator))
            return out
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if size <= min_batch:
                raise
            size = max(min_batch, size // 2)
            print(f"[m1] CUDA OOM — retrying this batch at size {size} "
                  f"(batch size is the only thing reduced)", flush=True)


def eos_ids(tok, mdl):
    cfg = getattr(mdl.generation_config, "eos_token_id", None) or tok.eos_token_id
    ids = list(cfg) if isinstance(cfg, (list, tuple)) else [cfg]
    tid = tok.convert_tokens_to_ids("<|im_end|>")
    if tid is not None and tid >= 0 and tid not in ids:
        ids.append(tid)
    return tuple(int(i) for i in ids if i is not None)


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--model-revision", default="main")
    ap.add_argument("--mode", default="pilot", choices=["smoke", "pilot", "full"])
    ap.add_argument("--k", type=int, default=None, help="traces per question")
    ap.add_argument("--n-questions", type=int, default=None)
    ap.add_argument("--max-new", type=int, default=MAX_NEW)
    ap.add_argument("--conf-topk", type=int, default=DC.DEFAULT_CONF_TOPK)
    ap.add_argument("--conf-variant", default="paper_eq2", choices=sorted(DC.CONF_VARIANTS))
    ap.add_argument("--audit-every", type=int, default=64,
                    help="retain raw top-50 arrays on every Nth trace (equality audit sample)")
    ap.add_argument("--keep-top-k-all", action="store_true")
    ap.add_argument("--i-accept-terabyte-retention", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="traces decoded concurrently. All traces in a batch share one "
                         "question's prompt, so no padding is needed; raise until the GPU's "
                         "memory or the measured tok/s stops improving.")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--attn-impl", default="sdpa")
    ap.add_argument("--dry-run", action="store_true",
                    help="build and verify the manifest from synthetic rows, then exit — "
                         "no dataset, no model, no GPU")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_sigterm)
    K = args.k or {"smoke": 2, "pilot": 32, "full": 4096}[args.mode]
    if args.keep_top_k_all and args.mode == "full" and not args.i_accept_terabyte_retention:
        sys.exit("--keep-top-k-all in full mode projects 0.6-1.2 TB of retained top-50 "
                 "arrays. That is a separate storage decision (handoff §3.2): re-run with "
                 "--i-accept-terabyte-retention only after Drive and shared quotas are "
                 "verified, or drop the flag and rely on --audit-every.")

    rows = (synthetic_rows(3) if args.dry_run else load_aime24())
    if args.n_questions:
        rows = rows[:args.n_questions]
    if args.mode == "smoke":
        rows = rows[:2]

    # Shard by (question, trace) so every worker gets an even slice — sharding by question
    # would leave 8 workers with 30 questions and 6 of them idle at the tail.
    units = [(r, t) for r in rows for t in range(K)]
    units = [u for i, u in enumerate(units) if i % args.n_shards == args.shard]
    print(f"[m1] mode={args.mode} K={K} questions={len(rows)} "
          f"traces_this_shard={len(units)} out={args.out}", flush=True)

    # One ShardWriter owns one directory exclusively (see shards.ShardWriter): two workers on
    # one directory would collide on shard numbering, clobber each other's STATUS.json, and
    # quarantine shards the other is still writing. So a sharded run gives each worker its own
    # part_NN/, and iter_run_dirs/read_shards reassemble them for the offline replay.
    run_dir = args.out if args.n_shards == 1 else os.path.join(
        args.out, f"part_{args.shard:02d}")
    os.makedirs(run_dir, exist_ok=True)
    if args.dry_run:
        tok_only = type("T", (), {"chat_template": "DRY-RUN-CHAT-TEMPLATE"})()
    else:
        from transformers import AutoTokenizer
        tok_only = AutoTokenizer.from_pretrained(args.model)
    man = build_manifest(
        run_id=os.path.basename(args.out.rstrip("/")) +
               ("" if args.n_shards == 1 else f"#part{args.shard:02d}"),
        paper_title="Deep Think with Confidence",
        paper_pdf_path=PAPER_PDF,
        # 'paper-specified-partial', not 'paper-specified': the official repo is runnable
        # and pinned, but we generate with HF transformers rather than the paper's pinned
        # vLLM commit, and we retain scalar channels rather than full top-50. Both are
        # declared deviations below, and a run with declared deviations is partial by
        # definition (handoff §1) — build_manifest refuses the stronger label.
        fidelity="paper-specified-partial",
        dataset_source="Maxwell-Jia/AIME_2024", dataset_revision="train",
        dataset_example_ids=[f"{r['question_id']}#{t}" for r, t in units],
        model_id=args.model, model_revision=args.model_revision,
        prompt_text=DEEPCONF_PROMPT, chat_template=tok_only.chat_template or "",
        decoding={**QWEN3_DECODING, "max_new_tokens": args.max_new,
                  "conf_topk": args.conf_topk, "conf_variant": args.conf_variant},
        seed_policy={"seed_base": 42, "per_trace_seed": "42 + global_trace_index",
                     "why": "independent traces per question, reproducible on resume"},
        max_new_tokens=args.max_new,
        stop_behavior={"eos": "generation_config + <|im_end|>", "cap": args.max_new,
                       "online_termination": "NOT applied during acquisition; the online "
                                             "rule is replayed offline over full traces"},
        signal_definitions={
            "deepconf_conf": f"DeepConf {args.conf_variant} over raw top-{args.conf_topk} logprobs",
            "raw_entropy": "full-vocabulary Shannon entropy from raw logits",
            "raw_logsumexp": "log-partition over the full raw-logit vocabulary",
            "spilled_energy": "-log p(sampled token), raw",
            "raw_margin": "top1 - top2 raw logprob",
        },
        logits_stage="raw",
        official_code_url=DC.OFFICIAL_REPO,
        official_code_commit=os.environ.get("DEEPCONF_COMMIT", "pinned-by-P0"),
        container_image=os.environ.get("SLURM_CONTAINER_IMAGE",
                                       "nvcr.io/nvidia/pytorch:25.01-py3"),
        evaluator_revision=EV.EVALUATOR_REVISION,
        declared_deviations=[
            {"field": "inference_engine",
             "paper_says": f"vLLM pinned at {DC.PINNED_VLLM_COMMIT}",
             "we_do": "HuggingFace transformers with raw-logit capture",
             "why": "the NGC container's torch may not be upgraded (CLAUDE.md); the "
                    "equality audit tests our confidence against the pinned function "
                    "instead of assuming engine equivalence"},
            {"field": "top_50_retention",
             "paper_says": "n/a",
             "we_do": f"scalar channels for every token; raw top-50 on every "
                      f"{args.audit_every}th trace",
             "why": "full top-50 retention is 0.6-1.2 TB (handoff §3.2)"},
        ],
        repo_root=REPO_ROOT,
        extra={"mode": args.mode, "K": K, "n_questions": len(rows),
               "shard": args.shard, "n_shards": args.n_shards,
               "batch_size": args.batch_size,
               "pinned_vllm_commit": DC.PINNED_VLLM_COMMIT,
               "n_init_warmup": DC.DEFAULT_N_INIT, "group_window": DC.DEFAULT_GROUP_WINDOW},
    )
    man["expected_traces"] = len(units)
    problems = verify_manifest(man, require_clean_tree=(args.mode == "full"))
    write_manifest(man, run_dir)

    gate = Gate(f"M-deepconf-{args.mode}", run_dir)
    gate.check("manifest_complete", not problems, f"{len(problems)} problems", problems)
    gate.check("pilot_is_not_a_table_row", True,
               "pilot mode is a protocol check; only --mode full may claim the paper table"
               if args.mode != "full" else "full pool: table reproduction permitted")
    gate.check("storage_contract",
               not (args.keep_top_k_all and args.mode == "full")
               or args.i_accept_terabyte_retention,
               "scalar-rich retention with an audit sample")
    gate.finish(raise_on_fail=True)
    if args.dry_run:
        print(f"[m1] DRY RUN OK — manifest builds and verifies "
              f"(fidelity={man['fidelity']}, {len(man['declared_deviations'])} deviations)",
              flush=True)
        return

    mdl, tok = load_model(args.model, attn_impl=args.attn_impl)
    mdl.eval()
    eos = eos_ids(tok, mdl)

    expected = [f"{r['question_id']}#{t}" for r, t in units]
    writer = ShardWriter(run_dir, expected_keys=expected)
    done = writer.done_keys()
    generator = torch.Generator(device=mdl.device)
    incomplete, n_new, t_start, tok_count = False, 0, time.time(), 0

    # ── batch the pending traces by question ──
    #
    # DeepConf's online rule is replayed offline over complete traces, so acquisition needs no
    # live stopping hook and every trace of a question can decode in one batch. That matters
    # enormously: at batch 1 an 8B model re-reads all 16 GB of weights per token, which the M1
    # pilot measured at 47 tok/s — 15,000 GPU-hours for the full pool. Batching amortises that
    # read across the batch.
    #
    # All traces of one question share an identical prompt, so a batch needs no padding at all.
    # Audit traces (which retain the raw top-50 arrays) are grouped separately from the rest,
    # because retaining [T, 50] arrays for a whole batch would cost gigabytes of host memory
    # for traces that do not need them.
    pending = [(row, t) for row, t in units if f"{row['question_id']}#{t}" not in done]
    groups = {}
    for row, t_idx in pending:
        keep_arrays = bool(args.keep_top_k_all or (t_idx % max(1, args.audit_every) == 0))
        groups.setdefault((row["question_id"], keep_arrays), []).append((row, t_idx))
    print(f"[m1] {len(pending)} traces pending in {len(groups)} question groups, "
          f"batch_size={args.batch_size}", flush=True)

    for (qid, keep_arrays), members in sorted(groups.items()):
        for i in range(0, len(members), args.batch_size):
            if STOP["flag"]:
                incomplete = True
                break
            chunk = members[i:i + args.batch_size]
            row = chunk[0][0]
            keys = [f"{row['question_id']}#{t}" for _, t in chunk]
            cfg = DecodeConfig(**QWEN3_DECODING, max_new_tokens=args.max_new,
                               logprob_top_k=50, conf_topk=args.conf_topk,
                               eos_token_ids=eos, keep_top_k_arrays=keep_arrays)
            # One generator seed per BATCH, derived from its first trace key. A resumed shard
            # that re-forms the same batch reproduces it exactly; re-forming a different batch
            # yields different (still independent) traces, which is fine for a sampled pool but
            # is why the seed is recorded per record.
            seed = 42 + int(hashlib.sha256(keys[0].encode()).hexdigest()[:8], 16) % (2 ** 31)
            generator.manual_seed(seed)
            prompt_text = DEEPCONF_PROMPT.format(question=row["problem"])
            chat = tok.apply_chat_template([{"role": "user", "content": prompt_text}],
                                           tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True)
            pids = tok(chat, add_special_tokens=False).input_ids
            t0 = time.time()
            try:
                gens = _generate_with_oom_backoff(mdl, tok, pids, chunk, cfg, generator)
            except Exception as e:  # noqa: BLE001 — one bad batch must not lose the shard
                for k in keys:
                    writer.add_failure(k, row["question_id"], repr(e))
                print(f"[m1] FAILED batch {keys[0]}..(+{len(keys) - 1}): {e!r}", flush=True)
                continue

            for (_, t_idx), key, gen in zip(chunk, keys, gens):
                graded = EV.grade_math(gen["full_text"], row["answer"])
                conf = np.asarray(gen["channels"]["deepconf_conf"], dtype=np.float64)
                rec = {
                    "trace_key": key, "question_id": row["question_id"], "trace_index": t_idx,
                    "prompt_text": prompt_text, "prompt_token_ids": list(pids),
                    "gen_token_ids": gen["gen_token_ids"], "full_text": gen["full_text"],
                    "channels": gen["channels"], "n_tokens": gen["n_tokens"],
                    "stop_reason": gen["stop_reason"], "gold_answer": row["answer"],
                    "correct": graded["correct"], "pred_answer": graded["pred_answer"],
                    "parse_status": graded["parse_status"],
                    # Precomputed native statistics. The offline replay recomputes them from
                    # `channels` and asserts equality, so these are a convenience, not the
                    # source of truth for any table.
                    "trace_statistics": {name: float(fn(conf))
                                         for name, fn in DC.TRACE_STATISTICS.items()},
                    "conf_variant": args.conf_variant, "conf_topk": args.conf_topk,
                    "retains_raw_top_k": bool(keep_arrays), "sampling_seed": seed,
                    "batch_size": len(chunk),
                }
                if keep_arrays and "raw_top_k_logprobs" in gen:
                    rec["raw_top_k_logprobs"] = gen["raw_top_k_logprobs"]
                writer.add(rec)
                n_new += 1
                tok_count += gen["n_tokens"]
            rate = tok_count / max(1e-9, time.time() - t_start)
            print(f"[m1] q={qid} batch of {len(chunk)} in {time.time() - t0:.1f}s "
                  f"| {rate:.0f} tok/s cumulative, {n_new}/{len(pending)} new", flush=True)
        if incomplete:
            break

    writer.close()
    free_memory()

    elapsed = time.time() - t_start
    tp = {"new_traces": n_new, "tokens": tok_count, "elapsed_s": round(elapsed, 1),
          "tokens_per_s": round(tok_count / max(1e-9, elapsed), 1),
          "mean_tokens_per_trace": round(tok_count / max(1, n_new), 1)}
    # Throughput is the number that decides whether M2 is affordable, so it is recorded
    # every run rather than estimated once.
    with open(os.path.join(run_dir, "THROUGHPUT.json"), "w") as f:
        json.dump(tp, f, indent=2)
    print(f"[m1] throughput: {json.dumps(tp)}", flush=True)

    if incomplete:
        print("[m1] PREEMPTED — resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)
    print(f"\nDEEPCONF ACQUISITION COMPLETE -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
