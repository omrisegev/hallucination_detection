"""G7 diagnostic for agent step telemetry v1: record, never raise, never change the gate.

Repeats the alignment_gate comparisons (serial prefix vs batched teacher forcing, causal
perturbation, target offset) on the shortest item of each benchmark plus three longer items,
for all 16 gate positions and for 8 positions inside step spans of the full trace. Then repeats
the shortest item per benchmark under the SDPA math backend (bf16) and in fp32, to separate
bf16 kernel noise from an indexing/alignment bug. Label-blind: reads answers only.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/"cluster"))
from run_agent_telemetry import MODEL, REVISION, make_items
from backfill_views import forward_batch
from spectral_utils.external_generalization.artifacts import atomic_json, file_hash


def step_positions(item):
    return {i for a, b in item["step_token_spans"] for i in range(a, b)}


def compare(model, item, positions, torch):
    """Serial-prefix vs batched log-softmax at the given gen positions of item (no truncation here)."""
    batch = forward_batch(model, [item])[0].float().log_softmax(-1)
    steps = step_positions(item)
    rows = []
    for j in positions:
        prefix = item["prompt_ids"] + item["gen_ids"][:j]
        ids = torch.tensor([prefix], device=model.device)
        with torch.no_grad():
            lp = model(input_ids=ids, use_cache=False).logits[0, -1].float().log_softmax(-1)
        t = item["gen_ids"][j]
        rows.append({"j": j, "context_tokens": len(prefix) + 1, "on_step_token": j in steps,
                     "target_logprob_batch": float(batch[j, t]), "target_logprob_serial": float(lp[t]),
                     "abs_diff_target": abs(float(lp[t] - batch[j, t])),
                     "max_abs_diff_full_vocab": float((lp - batch[j]).abs().max()),
                     "argmax_equal": int(lp.argmax()) == int(batch[j].argmax())})
    return rows, batch


def gate_like(model, item, torch):
    """Exact alignment_gate inputs (first 16 gen tokens) but every position, without raising."""
    short = dict(item, gen_ids=item["gen_ids"][:16])
    rows, batch = compare(model, short, range(len(short["gen_ids"])), torch)
    full_ids = torch.tensor([short["prompt_ids"] + short["gen_ids"]], device=model.device)
    with torch.no_grad():
        direct = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids),
                       use_cache=False).logits[0].float().log_softmax(-1)
    for r in rows:
        j = r["j"]
        r["target_offset_err"] = float((direct[len(short["prompt_ids"]) - 1 + j] - batch[j]).abs().max())
        changed = dict(short, gen_ids=short["gen_ids"][:j] + [(t + 1) % batch.shape[-1] for t in short["gen_ids"][j:]])
        with torch.no_grad():
            causal = forward_batch(model, [changed])[0][j].float().log_softmax(-1)
        r["causal_err"] = float((causal - batch[j]).abs().max())
    return rows


def summary(rows, key="abs_diff_target"):
    import numpy as np
    v = np.asarray([r[key] for r in rows], float)
    k = int(v.argmax())
    gate_js = sorted({0, len(rows) // 2, len(rows) - 1})
    return {"n": len(v), "max": float(v.max()), "mean": float(v.mean()), "p99": float(np.percentile(v, 99)),
            "argmax_j": rows[k]["j"], "argmax_on_step_token": rows[k]["on_step_token"],
            "argmax_target_logprob": rows[k]["target_logprob_batch"],
            "gate_positions": gate_js, "gate_positions_max": max(rows[i][key] for i in gate_js if i < len(rows)),
            "n_over_0p05": int((v > .05).sum())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-diag-tokens", type=int, default=12000)
    args = p.parse_args()
    if args.model_path.name != REVISION:
        raise ValueError("model-path must be the pinned snapshot")
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    args.out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, use_fast=True)
    chosen = {}
    for b in ("apb", "tprm", "aft"):
        f = args.inputs / f"{b}_answers.jsonl"
        answers = [json.loads(l) for l in f.read_text(encoding="utf8").splitlines()]
        items = [i for i in make_items(answers, tok) if not i["overlength"]]
        items.sort(key=lambda i: (len(i["prompt_ids"]) + len(i["gen_ids"]), i["uid"]))
        lengths = [len(i["prompt_ids"]) + len(i["gen_ids"]) for i in items]
        # shortest (= the item alignment_gate uses) + three longer items at length quartiles, capped.
        cap = [k for k, n in enumerate(lengths) if n <= args.max_diag_tokens]
        picks = [0] + [cap[int(q * (len(cap) - 1))] for q in (.25, .5, 1.0)]
        chosen[b] = {"answers_sha256": file_hash(f), "items": [items[k] for k in dict.fromkeys(picks)]}
        print(json.dumps({"benchmark": b, "lengths": [lengths[k] for k in dict.fromkeys(picks)]}), flush=True)

    env = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
           "model": MODEL, "revision": REVISION, "job_id": os.environ.get("SLURM_JOB_ID"),
           "code_commit": (ROOT / "SOURCE_COMMIT").read_text().strip() if (ROOT / "SOURCE_COMMIT").exists() else None}
    report = {"schema": 1, "env": env, "gate_unchanged": True, "exception_added": False, "runs": []}

    def run_config(name, dtype, attn, items_by_b, sdpa_math=False):
        t0 = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True, dtype=dtype,
                                                     device_map="auto", attn_implementation=attn).eval()
        ctx = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH) if sdpa_math else None
        if ctx is not None:
            ctx.__enter__()
        try:
            for b, its in items_by_b.items():
                for item in its:
                    rec = {"config": name, "dtype": str(dtype), "attn": attn, "sdpa_math": sdpa_math,
                           "benchmark": b, "uid": item["uid"], "prompt_tokens": len(item["prompt_ids"]),
                           "gen_tokens": len(item["gen_ids"]),
                           "steps": len(item["step_token_spans"])}
                    g = gate_like(model, item, torch)
                    rec["gate16"] = {"rows": g, "target": summary(g), "full_vocab": summary(g, "max_abs_diff_full_vocab"),
                                     "max_causal_err": max(r["causal_err"] for r in g),
                                     "max_target_offset_err": max(r["target_offset_err"] for r in g),
                                     "would_raise_fixed_shape": max(max(r["causal_err"], r["target_offset_err"]) for r in g) > 1e-5}
                    steps = sorted(step_positions(item))
                    if steps:
                        sel = [steps[int(q)] for q in np.linspace(0, len(steps) - 1, min(8, len(steps)))]
                        s, _ = compare(model, item, list(dict.fromkeys(sel)), torch)
                        rec["step_positions"] = {"rows": s, "target": summary(s), "full_vocab": summary(s, "max_abs_diff_full_vocab")}
                    report["runs"].append(rec)
                    atomic_json(args.out / "G7_DIAGNOSTIC.json", report)
                    print(json.dumps({"config": name, "benchmark": b, "uid": item["uid"],
                                      "prompt_tokens": rec["prompt_tokens"], "gate16_max": rec["gate16"]["target"]["max"],
                                      "step_max": rec.get("step_positions", {}).get("target", {}).get("max")}), flush=True)
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
            del model
            torch.cuda.empty_cache()
        return time.perf_counter() - t0

    shortest = {b: v["items"][:1] for b, v in chosen.items()}
    report["seconds"] = {
        "bf16_sdpa": run_config("bf16_sdpa", torch.bfloat16, "sdpa", {b: v["items"] for b, v in chosen.items()}),
        "bf16_sdpa_math": run_config("bf16_sdpa_math", torch.bfloat16, "sdpa", shortest, sdpa_math=True),
        "fp32_sdpa": run_config("fp32_sdpa", torch.float32, "sdpa", shortest),
    }
    report["answers_sha256"] = {b: v["answers_sha256"] for b, v in chosen.items()}
    report["complete"] = True
    atomic_json(args.out / "G7_DIAGNOSTIC.json", report)
    print("DONE", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
