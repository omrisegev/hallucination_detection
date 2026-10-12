#!/usr/bin/env python
"""
PRMBench: Qwen2.5-Math-PRM-7B step rewards AND token-probability telemetry, for label-free
"PRM boosting" (fusing the PRM step score with our N16 telemetry bank). Teacher-forced, no
generation, one forward pass per row. Output rows have the run_prmbench_teacher_forced schema
(gen_token_ids / token_entropies / token_spilled_energies / token_logsumexp / top_k_logprobs /
step_token_spans), so the frozen feature-bank code reads them unchanged.

Two modes:

--mode prm (primary, "same pass"):
    Input = the PRM model card's exact 3-turn template with one <extra_0> after every step
    (spectral_utils.prm_scorer). One backbone forward gives the last hidden state h. The reward
    head gives the step rewards (checked against the model's own forward on the first rows), and
    the checkpoint's own lm_head.weight, loaded from the safetensors shard because the PRM class
    does not use it, gives vocabulary logits from the same h. Telemetry covers the step tokens only;
    <extra_0> tokens are not scored. The first token of step k>0 is predicted from the <extra_0>
    position, as in that forward pass. Caveat recorded in the manifest: the PRM class never trains
    lm_head, while the backbone was fine-tuned, so these probabilities come from a head that was
    not trained together with these hidden states. The manifest records how lm_head compares with
    Qwen2.5-Math-7B-Instruct's.

--mode instruct (reference): Qwen/Qwen2.5-Math-7B-Instruct as a causal LM. Prompt = the same
    system prompt (Qwen2.5-Math CoT, = PRM_SYSTEM_PROMPT) + question; chain = steps joined by
    "\\n\\n" with spectral_utils.prmbench.build_chain/step_token_spans, exactly as the Qwen3 cells.

Nothing is masked here; the mask1 rule (drop forced token 0) is applied downstream like the Qwen3 cells.

    python cluster/run_prmbench_prm_telemetry.py --mode prm --out $SHARED/results/prmbench_prm7b_telemetry
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
from spectral_utils import load_cache, save_cache_atomic, free_memory
from spectral_utils.prm_scorer import PRM_MODEL_ID, PRM_SYSTEM_PROMPT, load_prm_model, make_step_rewards
from spectral_utils.prmbench import load_prmbench, build_chain, step_token_spans, assert_alignment
from run_teacher_forced import batched  # noqa: E402

INSTRUCT_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[prm-tel] SIGTERM received; will checkpoint after the current row", flush=True)


def load_head_weight(model_id, name="lm_head.weight"):
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    d = snapshot_download(model_id, local_files_only=True,
                          allow_patterns=["*.json", "*.safetensors"])
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    with safe_open(os.path.join(d, idx[name]), framework="pt") as f:
        return f.get_tensor(name)


def row_record(row, extra):
    rec = {k: row[k] for k in ("idx", "source_idx", "classification", "category", "question", "steps")}
    rec["error_steps"] = row["error_steps"]       # carried for evaluation only, never read by a fit
    rec.update(extra)
    return rec


def quantities(logits, gen_ids, k):
    q = candidate_quantities(logits, gen_ids, warpers=None, raw_top_k=k, post_top_k=k)
    return {"token_entropies": q["token_entropies_recomputed"],
            "token_spilled_energies": q["token_spilled_energies"],
            "token_logsumexp": q["token_logsumexp"],
            "top_k_logprobs": q["top_k_logprobs_raw"]}


def prm_inputs(tok, row):
    response = "<extra_0>".join(row["steps"]) + "<extra_0>"
    msgs = [{"role": "system", "content": PRM_SYSTEM_PROMPT}, {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": response}]
    conv = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    cut = conv.rindex(response)
    ids = tok.encode(conv); pre = tok.encode(conv[:cut])
    if ids[:len(pre)] != pre:
        raise ValueError("prefix tokenization differs inside the full conversation")
    sep = tok.encode("<extra_0>")[0]
    seps = [p for p in range(len(pre), len(ids)) if ids[p] == sep]
    if len(seps) != len(row["steps"]):
        raise ValueError(f"{len(seps)} separators for {len(row['steps'])} steps")
    pos, spans, start, n_text_mismatch = [], [], len(pre), 0
    for k, s in enumerate(seps):
        seg = list(range(start, s))
        if tok.decode([ids[p] for p in seg]) != row["steps"][k]:
            n_text_mismatch += 1
        spans.append((len(pos), len(pos) + len(seg))); pos += seg; start = s + 1
    return ids, seps, pos, spans, n_text_mismatch


def run_prm(cfg, meta, out_path):
    mdl, tok = load_prm_model(cfg.model)
    W = load_head_weight(cfg.model).to(mdl.device, dtype=torch.bfloat16)
    head_check = {"prm_lm_head_shape": list(W.shape)}
    try:
        Wi = load_head_weight(INSTRUCT_ID).to(W.device, dtype=torch.bfloat16)
        head_check.update({"instruct_lm_head_identical": bool(torch.equal(W, Wi)),
                           "max_abs_diff_vs_instruct": float((W.float() - Wi.float()).abs().max())})
        del Wi
    except Exception as exc:
        head_check["instruct_compare_error"] = str(exc)[:200]
    print("[prm-tel] lm_head check", head_check, flush=True)
    backbone, score = mdl.model, mdl.score
    cache = load_cache(out_path); n_done = 0; worst_reward_gap = 0.0
    for i, row in enumerate(meta):
        if i in cache:
            continue
        if STOP["flag"]:
            save_cache_atomic(cache, out_path); return False, head_check, worst_reward_gap
        t0 = time.time()
        ids, seps, pos, spans, mism = prm_inputs(tok, row)
        x = torch.tensor([ids], device=mdl.device)
        with torch.no_grad():
            h = backbone(input_ids=x, use_cache=False)[0]
            cls = score(h)
            rewards = make_step_rewards(cls, x == ids[seps[0]])[0]
            if n_done < cfg.verify_rows:   # reward path must equal the model's own forward
                ref = make_step_rewards(mdl(input_ids=x)[0], x == ids[seps[0]])[0]
                worst_reward_gap = max(worst_reward_gap, float(np.max(np.abs(np.subtract(rewards, ref)))))
                if worst_reward_gap > 1e-3:
                    raise SystemExit(f"reward path mismatch {worst_reward_gap}")
            pidx = torch.tensor(pos, device=mdl.device) - 1
            logits = h[0, pidx] @ W.T
        gen_ids = [ids[p] for p in pos]
        rec = row_record(row, {"gen_token_ids": gen_ids, "step_token_spans": spans, "rewards": rewards,
                               "n_input_tokens": len(ids), "n_step_text_mismatch": mism,
                               **quantities(logits, gen_ids, cfg.logprob_top_k)})
        cache[i] = rec; n_done += 1
        del h, cls, logits
        if n_done % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)
            print(f"[prm-tel] {len(cache)}/{len(meta)} rows ({time.time() - t0:.2f}s last)", flush=True)
    save_cache_atomic(cache, out_path)
    return True, head_check, worst_reward_gap


def run_instruct(cfg, meta, out_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model)
    mdl = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map={"": 0},
                                               attn_implementation="sdpa").eval()
    items = []
    for i, row in enumerate(meta):
        text, char_spans = build_chain(row["steps"])
        gen_ids, spans = step_token_spans(tok, text, char_spans)
        diag = assert_alignment(gen_ids, spans, row["steps"], strict=False)
        prompt = tok.apply_chat_template([{"role": "system", "content": PRM_SYSTEM_PROMPT},
                                          {"role": "user", "content": row["question"]}],
                                         tokenize=False, add_generation_prompt=True)
        items.append({"idx": i, "prompt_ids": tok(prompt).input_ids, "gen_ids": gen_ids,
                      "step_token_spans": spans, "align_diag": diag, "row": row})
    cache = load_cache(out_path); todo = [it for it in items if it["idx"] not in cache]; n_done = 0
    for batch in batched(todo, cfg.max_batch_tokens, cfg.max_batch):
        if STOP["flag"]:
            save_cache_atomic(cache, out_path); return False
        for it, logits in zip(batch, forward_batch(mdl, batch)):
            cache[it["idx"]] = row_record(it["row"], {"gen_token_ids": it["gen_ids"], "step_token_spans": it["step_token_spans"],
                                                      "align_diag": it["align_diag"],
                                                      "n_input_tokens": len(it["prompt_ids"]) + len(it["gen_ids"]),
                                                      **quantities(logits, it["gen_ids"], cfg.logprob_top_k)})
            n_done += 1
        if n_done and n_done % cfg.checkpoint_every < len(batch):
            save_cache_atomic(cache, out_path)
            print(f"[prm-tel] {len(cache)}/{len(items)} rows", flush=True)
    save_cache_atomic(cache, out_path)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prm", "instruct"], required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=None, help="cap RAW rows (pilot only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--meta-pkl", default=None, help="take PRMBench rows from this telemetry pkl (no datasets import)")
    ap.add_argument("--verify-rows", type=int, default=20)
    cfg = ap.parse_args()
    cfg.model = cfg.model or (PRM_MODEL_ID if cfg.mode == "prm" else INSTRUCT_ID)
    signal.signal(signal.SIGTERM, _on_sigterm)
    os.makedirs(cfg.out, exist_ok=True)
    if cfg.meta_pkl:   # rows (question, steps, ids) of an existing PRMBench telemetry cell: same rows, same order
        src = load_cache(cfg.meta_pkl)
        meta = [{k: e[k] for k in ("idx", "source_idx", "classification", "category", "question", "steps", "error_steps")}
                for _, e in sorted(src.items())][:cfg.n_samples]
        diag = {"meta_source": cfg.meta_pkl, "n_rows": len(meta)}
        del src
    else:
        meta, diag = load_prmbench(n_samples=cfg.n_samples, seed=cfg.seed)
    print(f"[prm-tel] mode={cfg.mode} model={cfg.model} rows={len(meta)}", flush=True)
    out_path = os.path.join(cfg.out, "prmbench_telemetry.pkl"); t0 = time.time(); extra = {}
    if cfg.mode == "prm":
        ok, head_check, gap = run_prm(cfg, meta, out_path); extra = {"lm_head_check": head_check, "reward_path_max_abs_gap": gap}
    else:
        ok = run_instruct(cfg, meta, out_path)
    if not ok:
        print("[prm-tel] INCOMPLETE; resubmit with the same --out to resume", flush=True)
        sys.exit(EXIT_INCOMPLETE)
    cache = load_cache(out_path)
    surpr = np.concatenate([np.asarray(e["token_spilled_energies"]) for e in cache.values()])
    ent = np.concatenate([np.asarray(e["token_entropies"]) for e in cache.values()])
    manifest = {"driver": "run_prmbench_prm_telemetry.py", "mode": cfg.mode, "model": cfg.model,
                "dataset_diagnostics": diag, "n_rows": len(cache), "logprob_top_k": cfg.logprob_top_k,
                "system_prompt": PRM_SYSTEM_PROMPT,
                "token_surprisal_median": float(np.median(surpr)), "token_surprisal_mean": float(surpr.mean()),
                "token_entropy15_median": float(np.median(ent)),
                "n_rows_step_text_mismatch": int(sum(e.get("n_step_text_mismatch", 0) > 0 for e in cache.values())),
                "max_input_tokens": int(max(e["n_input_tokens"] for e in cache.values())),
                "n_rows_over_4096_tokens": int(sum(e["n_input_tokens"] > 4096 for e in cache.values())),
                **extra, "elapsed_sec": time.time() - t0, "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = os.path.join(cfg.out, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    print(json.dumps(manifest, indent=2, default=str), flush=True)
    print(f"COMPLETE -> {cfg.out}", flush=True)
    free_memory()


if __name__ == "__main__":
    main()
