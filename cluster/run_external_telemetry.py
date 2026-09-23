"""Label-blind, resumable teacher-forced collection for two external benchmarks.

No fusion, feature fitting, benchmark answer generation or quality evaluation.
Each record contains sufficient distribution telemetry for later CPU extraction.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "cluster"))
from spectral_utils.external_generalization.artifacts import RecordStore, atomic_json, file_hash
from spectral_utils.external_generalization.contracts import Answer, tokenize_answer, digest
from spectral_utils.external_generalization.budget import smoke_indices, authorize

STOP = False


def stop(*_):
    global STOP
    STOP = True


def make_items(answers, tokenizer, max_context):
    # Exact source telemetry conditioning: math_prompt + /no_think, followed by
    # the tokenizer's default chat template. QwQ's template differs; pin/hash it.
    from spectral_utils.data_loaders import math_prompt
    items = []
    for a in answers:
        prompt = math_prompt({"problem": a.question}) + " /no_think"
        rendered = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(rendered)["input_ids"]
        items.append(tokenize_answer(a, tokenizer, prompt_ids, max_context))
    return items


def collect_quantities(model, item):
    import numpy as np
    import torch
    from backfill_views import forward_batch, candidate_quantities
    logits = forward_batch(model, [item])[0]
    q = candidate_quantities(logits, item["gen_ids"], warpers=None, raw_top_k=50,
                             post_top_k=50, chunk=128)
    full_entropy = []
    for begin in range(0, len(logits), 128):
        lp = logits[begin:begin+128].float().log_softmax(-1)
        full_entropy.extend((-(lp.exp()*lp).sum(-1)).cpu().tolist())
    top = q["top_k_logprobs_raw"]
    del logits
    # q15 entropy is preserved separately from full-vocabulary entropy; the
    # historical token_entropies field uses q15, not full-vocabulary entropy.
    return {"gen_token_ids": item["gen_ids"], "token_offsets": item["token_offsets"],
            "step_char_spans": item["step_char_spans"], "step_token_spans": item["step_token_spans"],
            "prompt_ids": item["prompt_ids"], "token_entropies": q["token_entropies_recomputed"],
            "token_entropy_full": full_entropy, "token_logsumexp": q["token_logsumexp"],
            "token_spilled_energies": q["token_spilled_energies"],
            "actual_token_logprobs": (-np.asarray(q["token_spilled_energies"])).tolist(),
            "top_k_logprobs": {"ids": top["ids"].tolist(), "logprobs": top["logprobs"].tolist()}}


def alignment_gate(model, item):
    """Fixed-token serial-prefix vs batched teacher-forcing consistency; no generation.

    Complements (does not claim to replace) historical generated-cache Gate B.
    Tests exact target offsets at first/middle/last positions of a short trace.
    """
    import torch
    from backfill_views import forward_batch
    short = dict(item, gen_ids=item["gen_ids"][:16])
    with torch.no_grad():
        batch = forward_batch(model, [short])[0].float().log_softmax(-1)
        differences = []
        for j in sorted({0, len(short["gen_ids"])//2, len(short["gen_ids"])-1}):
            prefix = short["prompt_ids"] + short["gen_ids"][:j]
            ids = torch.tensor([prefix], device=model.device)
            lp = model(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1)
            target = short["gen_ids"][j]
            differences.append(abs(float(lp[target] - batch[j, target])))
    if max(differences) > .05:
        raise ValueError("fixed-token alignment gate failed: " + str(differences))
    return {"status": "PASS", "actual_logprob_abs_errors": differences, "max_allowed": .05}


def run(items, store, scorer, selected, *, memory=lambda: 0):
    measurements = []
    for index in selected:
        if STOP:
            return False, measurements
        item = items[index]
        previous = store.get(item["uid"])
        if previous is not None:
            measurements.append(previous["measurement"])
            continue
        start, cpu = time.perf_counter(), time.process_time()
        payload = scorer(item)
        measurement = {"uid": item["uid"], "context_tokens": len(item["prompt_ids"])+len(item["gen_ids"]),
                       "answer_tokens": len(item["gen_ids"]), "seconds": time.perf_counter()-start,
                       "cpu_seconds": time.process_time()-cpu, "peak_gpu_bytes": memory(),
                       "generation_measured": False}
        measurement["bytes"] = len(json.dumps(payload).encode())
        store.put(item["uid"], {"telemetry": payload, "measurement": measurement})
        measurements.append(measurement)
        print(json.dumps(measurement), flush=True)
    return True, measurements


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--answers", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--revision", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    p.add_argument("--preflight", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--budget-decision", type=Path)
    p.add_argument("--estimate", type=Path)
    p.add_argument("--max-context", type=int, default=32768)
    p.add_argument("--validate-pkl", type=Path)
    args = p.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    protocol = json.loads(args.protocol.read_text())
    if protocol["models"].get(args.model) != args.revision:
        raise ValueError("checkpoint is not pinned in the collection protocol")
    authorize(args.mode, preflight=json.loads(args.preflight.read_text()),
              protocol_hash=file_hash(args.protocol),
              estimate_hash=file_hash(args.estimate) if args.estimate else None,
              decision=json.loads(args.budget_decision.read_text()) if args.budget_decision else None)
    if args.mode == "smoke" and (int(os.environ.get("SLURM_GPUS_ON_NODE", "1")) != 1
                                 or int(os.environ.get("SLURM_TIMELIMIT", "60")) > 60):
        raise ValueError("smoke requires at most one allocated GPU-hour")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    answers = [Answer(**row) for row in json.loads(args.answers.read_text(encoding="utf8"))]
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    items = make_items(answers, tok, args.max_context)
    lengths = [len(i["prompt_ids"])+len(i["gen_ids"]) for i in items]
    selected = smoke_indices(lengths, [a.uid for a in answers]) if args.mode == "smoke" else list(range(len(items)))
    identity = {"model": args.model, "revision": args.revision, "dtype": "bfloat16", "attn": "sdpa",
                "answers_sha256": file_hash(args.answers), "protocol_sha256": file_hash(args.protocol),
                "chat_template_sha256": digest(tok.chat_template), "prompt_suffix": " /no_think",
                "thinking_mode": "tokenizer_default_with_source_no_think_suffix",
                "max_context": args.max_context, "schema": 1, "mode": args.mode,
                "selected_uids": [items[i]["uid"] for i in selected]}
    atomic_json(args.out / "TOKENIZATION.json", {"identity": identity, "lengths": lengths,
                "uids": [a.uid for a in answers], "selected_indices": selected,
                "total_answer_tokens": sum(len(i["gen_ids"]) for i in items), "truncated": 0})
    loaded_at = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    load_seconds = time.perf_counter()-loaded_at
    torch.cuda.reset_peak_memory_stats()
    gate = alignment_gate(model, items[min(range(len(items)), key=lambda i:lengths[i])])
    if args.validate_pkl:
        from types import SimpleNamespace
        from run_teacher_forced import run_gate_b
        run_gate_b(model, tok, SimpleNamespace(validate_pkl=str(args.validate_pkl),
                   validate_dataset="gsm8k", validate_n=5, validate_prompt_suffix=" /no_think",
                   max_batch=1, logprob_top_k=50))
    atomic_json(args.out / "ALIGNMENT_GATE.json", gate)
    with RecordStore(args.out / "records", identity) as store:
        complete, measurements = run(items, store, lambda i: collect_quantities(model, i), selected,
                                      memory=torch.cuda.max_memory_allocated)
    atomic_json(args.out / "TIMING.json", {"identity": identity, "complete": complete,
                "load_seconds": load_seconds, "measurements": measurements,
                "job_id": os.environ.get("SLURM_JOB_ID"), "quality_evaluated": False})
    return 0 if complete else 85


if __name__ == "__main__":
    raise SystemExit(main())
