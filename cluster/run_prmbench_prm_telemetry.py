#!/usr/bin/env python
"""
PRMBench: Qwen2.5-Math-PRM-7B step rewards AND token-probability telemetry, for label-free
"PRM boosting" (fusing the PRM step score with our N16 telemetry bank). Teacher-forced, no
generation, one forward pass per row. Output rows have the run_prmbench_teacher_forced schema
(gen_token_ids / token_entropies / token_spilled_energies / token_logsumexp / top_k_logprobs /
step_token_spans), so the frozen feature-bank code reads them unchanged.

Adapters: qwen_prm (historical 7B), qwen_prm_72b (literal upstream
wrapper), rlhflow_prm, pure_prm, math_shepherd. Skywork's templates are
inspectable but inference is blocked pending its n+1 reward clarification.
No separate base-model forward is performed; only cached vocabulary weights
may be borrowed. --compare-to checks the first 40 rows for GPU regression T1.

Two compatibility modes:

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

--mode instruct (reference): a standalone causal LM, as released, on the same rows. Default model
    Qwen/Qwen2.5-Math-7B-Instruct. Chain = steps joined by "\\n\\n" with
    spectral_utils.prmbench.build_chain/step_token_spans, exactly as the Qwen3 cells. --prompt-style:
      qwen_cot_chat (default, historical): chat template, system = Qwen2.5-Math CoT (= PRM_SYSTEM_PROMPT),
                    user = question. Unchanged code path for the Qwen-7B reference (job 282803).
      chat:         chat template, no system message, user = question (Llama-3.1-8B-Instruct).
      plain:        no template: BOS + question + "\\n\\n" (base models without a chat template, Mistral-7B).
    Standalone base models of the PRMs (plan docs/experiments/PRM_STANDALONE_TELEMETRY_2026-10-22.md):
    Qwen2.5-Math-72B-Instruct (qwen_cot_chat), Qwen2.5-Math-7B (qwen_cot_chat), Llama-3.1-8B-Instruct (chat),
    Mistral-7B-v0.1 (plain).

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
from spectral_utils.prm_scorer import PRM_MODEL_ID, PRM_SYSTEM_PROMPT, make_step_rewards
from spectral_utils.prmbench import load_prmbench, build_chain, step_token_spans, assert_alignment
from spectral_utils.prm_adapters import ADAPTERS, get_adapter, OFFICIAL_WRAPPER_COMMIT, OFFICIAL_QWEN_SYSTEM_PROMPT
from run_teacher_forced import batched  # noqa: E402

INSTRUCT_ID = "Qwen/Qwen2.5-Math-7B-Instruct"
EXIT_INCOMPLETE = 85
STOP = {"flag": False}


def _on_sigterm(signum, frame):
    STOP["flag"] = True
    print("[prm-tel] SIGTERM received; will checkpoint after the current row", flush=True)


def answers_rows(path, n_samples=None):
    """Rows of a label-free external answers.json (uid, question, steps): same uids and step lists as
    our Qwen3 external telemetry. classification/error_steps stay None; labels live in the evaluator."""
    with open(path, encoding="utf8") as f:
        rows = json.load(f)
    rows = rows[:n_samples] if n_samples else rows
    if len({r["uid"] for r in rows}) != len(rows):
        raise ValueError("duplicate uid in answers.json")
    meta = [{"idx": r["uid"], "source_idx": r.get("source_id"), "classification": None,
             "category": r.get("benchmark"), "question": r["question"], "steps": list(r["steps"]),
             "error_steps": None} for r in rows]
    return meta, {"answers": os.path.abspath(path), "n_rows": len(meta)}


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
    """Compatibility entry point: historical Qwen-7B tokenization, unchanged."""
    batch = get_adapter('qwen_prm').build(tok, row['question'], row['steps'])
    pos, spans = compressed_positions(batch)
    return batch['input_ids'], batch['score_positions'], pos, spans, batch['n_step_text_mismatch']


def compressed_positions(batch):
    pos, spans = [], []
    for segment in batch['step_token_positions']:
        spans.append((len(pos), len(pos) + len(segment))); pos.extend(segment)
    return pos, spans


def compare_caches(cache, reference_path, n_rows=40):
    """T1: exact IDs/spans/reward order and <=1e-5 floating telemetry differences."""
    reference = load_cache(reference_path)
    ref = {str(e['idx']): e for e in reference.values()}
    order = [str(e['idx']) for _, e in sorted(reference.items())][:n_rows]
    current = [str(e['idx']) for _, e in sorted(cache.items())][:n_rows]
    if current != order: raise ValueError('T1 first-row identity/order mismatch')
    checked = 0; gaps = {'rewards': 0.}
    for _, row in sorted(cache.items()):
        if checked == n_rows: break
        if 'failure' in row: raise ValueError('T1 contains a counted inference failure')
        old = ref[str(row['idx'])]
        for key in ('gen_token_ids', 'step_token_spans', 'rewards'):
            if not np.array_equal(row[key], old[key]): raise ValueError('T1 mismatch: ' + key)
        for key in ('token_entropies', 'token_spilled_energies', 'token_logsumexp'):
            a, b = np.asarray(row[key]), np.asarray(old[key])
            if a.shape != b.shape or not np.allclose(a, b, atol=1e-5, rtol=0):
                raise ValueError('T1 mismatch: ' + key)
            gaps[key] = max(gaps.get(key, 0.), float(np.max(np.abs(a-b), initial=0)))
        if not np.array_equal(row['top_k_logprobs']['ids'], old['top_k_logprobs']['ids']):
            raise ValueError('T1 mismatch: top-k IDs')
        if not np.allclose(row['top_k_logprobs']['logprobs'], old['top_k_logprobs']['logprobs'], atol=1e-5, rtol=0):
            raise ValueError('T1 mismatch: top-k log probabilities')
        checked += 1
    if checked != n_rows: raise ValueError(f'T1 requires {n_rows} rows; checked {checked}')
    return {'status': 'PASS', 'n_checked': checked, 'max_abs_gaps': gaps}


def chunk_quantities(adapter, h, pos, ids, k, chunk_size):
    """Bound vocabulary memory; concatenate compact telemetry, not vocabulary logits."""
    pieces = []
    for start in range(0, len(pos), chunk_size):
        pp = pos[start:start+chunk_size]; gi = [ids[p] for p in pp]
        logits = adapter.vocab_logits(h, [p-1 for p in pp])
        pieces.append(quantities(logits, gi, k)); del logits
    if not pieces:
        return {'token_entropies': [], 'token_spilled_energies': [], 'token_logsumexp': [],
                'top_k_logprobs': {'ids': np.empty((0, k), np.int32), 'logprobs': np.empty((0, k), np.float32)}}
    out = {key: np.concatenate([q[key] for q in pieces]).tolist()
           for key in ('token_entropies', 'token_spilled_energies', 'token_logsumexp')}
    out['top_k_logprobs'] = {key: np.concatenate([q['top_k_logprobs'][key] for q in pieces])
                           for key in ('ids', 'logprobs')}
    return out


def run_prm(cfg, meta, out_path):
    adapter = get_adapter(cfg.adapter)
    mdl, tok, head_check = adapter.load(cfg.model)
    head_check['prm_lm_head_shape'] = list(adapter.head_weight.shape)
    if cfg.adapter == 'qwen_prm':
        gap = head_check['lm_head_vs_base_max_abs_diff']
        head_check.update(instruct_lm_head_identical=gap == 0 if gap is not None else None,
                          max_abs_diff_vs_instruct=gap)
    print('[prm-tel] lm_head check', head_check, flush=True)
    cache = load_cache(out_path); n_done = 0; worst_reward_gap = 0.0
    context = int(mdl.config.max_position_embeddings)
    device = next(adapter.backbone.parameters()).device
    for i, row in enumerate(meta):
        if i in cache: continue
        if STOP['flag']:
            save_cache_atomic(cache, out_path); return False, head_check, worst_reward_gap
        t0 = time.time()
        batch = adapter.build(tok, row['question'], row['steps'])
        ids = batch['input_ids']; pos, spans = compressed_positions(batch)
        if len(ids) > context:
            cache[i] = row_record(row, {'failure': 'context_overflow', 'n_input_tokens': len(ids),
                                       'context_limit': context})
            save_cache_atomic(cache, out_path)
            print(f'[prm-tel] counted context failure {row["idx"]}: {len(ids)} > {context}', flush=True)
            continue
        x = torch.tensor([ids], device=device)
        with torch.no_grad():
            h, outputs = adapter.forward(x, batch)
            rewards = adapter.rewards(outputs, batch)
            if len(rewards) != len(row['steps']): raise ValueError('reward/step count mismatch')
            if cfg.adapter.startswith('qwen_prm') and n_done < cfg.verify_rows:
                ref = make_step_rewards(mdl(input_ids=x)[0], x == ids[batch['score_positions'][0]])[0]
                worst_reward_gap = max(worst_reward_gap, float(np.max(np.abs(np.subtract(rewards, ref)), initial=0)))
                if worst_reward_gap > 1e-3: raise SystemExit(f'reward path mismatch {worst_reward_gap}')
            # Keep historical 7B GEMM/telemetry reduction bit-identical. 72B is chunked.
            chunk = (len(pos) or 1) if cfg.adapter == 'qwen_prm' else 1024
            q = chunk_quantities(adapter, h, pos, ids, cfg.logprob_top_k, chunk)
        extra = {'gen_token_ids': [ids[p] for p in pos], 'step_token_spans': spans, 'rewards': rewards,
                 'native_labels': adapter.native_labels(rewards), 'n_input_tokens': len(ids),
                 'n_step_text_mismatch': batch['n_step_text_mismatch'],
                 'n_boundary_whitespace': batch['n_boundary_whitespace'], **q}
        if 'terminal_score_position' in batch:
            extra['terminal_reward'] = float(outputs['logits'][0, batch['terminal_score_position']])
        cache[i] = row_record(row, extra); n_done += 1
        del h, outputs
        if n_done % cfg.checkpoint_every == 0:
            save_cache_atomic(cache, out_path)
            print(f'[prm-tel] {len(cache)}/{len(meta)} rows ({time.time()-t0:.2f}s last)', flush=True)
    save_cache_atomic(cache, out_path)
    return True, head_check, worst_reward_gap


PROMPT_STYLES = ("qwen_cot_chat", "chat", "plain")


def standalone_prompt_ids(tok, style, question):
    """Prompt token ids for --mode instruct. qwen_cot_chat keeps the historical call exactly."""
    if style == "qwen_cot_chat":
        prompt = tok.apply_chat_template([{"role": "system", "content": PRM_SYSTEM_PROMPT},
                                          {"role": "user", "content": question}],
                                         tokenize=False, add_generation_prompt=True)
        return tok(prompt).input_ids
    if style == "chat":
        if not getattr(tok, "chat_template", None):
            raise ValueError("prompt style 'chat' needs a tokenizer with a chat template")
        prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                         add_generation_prompt=True)
        ids = tok(prompt, add_special_tokens=False).input_ids     # the template already carries BOS
    elif style == "plain":
        ids = tok(question + "\n\n").input_ids                   # tokenizer adds its BOS
    else:
        raise ValueError(f"unknown prompt style {style}")
    if tok.bos_token_id is not None and (ids[:1] != [tok.bos_token_id] or ids[1:2] == [tok.bos_token_id]):
        raise ValueError(f"prompt style {style}: expected exactly one leading BOS, got {ids[:3]}")
    return ids


def fill_empty_spans(spans):
    """An empty step can map to no token (SentencePiece/Mistral). Write it as an empty span at the previous end, as the
    PRM path does, so the bank builder fills it. Qwen and Llama tokenizers never return None (smoke test, 2026-10-22)."""
    out, hi = [], 0
    for sp in spans:
        sp = (hi, hi) if sp is None else tuple(sp)
        out.append(sp); hi = sp[1]
    return out


def run_instruct(cfg, meta, out_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model)
    device_map = {"": 0} if torch.cuda.is_available() else None   # CPU only for the local smoke test
    mdl = AutoModelForCausalLM.from_pretrained(cfg.model, torch_dtype=torch.bfloat16, device_map=device_map,
                                               attn_implementation="sdpa").eval()
    cfg.model_revision = getattr(mdl.config, "_commit_hash", None)
    items = []
    cache = load_cache(out_path)
    for i, row in enumerate(meta):
        text, char_spans = build_chain(row["steps"])
        gen_ids, spans = step_token_spans(tok, text, char_spans)
        diag = assert_alignment(gen_ids, spans, row["steps"], strict=False)
        spans = fill_empty_spans(spans)
        prompt_ids = standalone_prompt_ids(tok, cfg.prompt_style, row["question"])
        n_input = len(prompt_ids) + len(gen_ids)
        if n_input > mdl.config.max_position_embeddings:
            cache[i] = row_record(row, {'failure': 'context_overflow', 'n_input_tokens': n_input,
                                       'context_limit': mdl.config.max_position_embeddings})
            continue
        items.append({"idx": i, "prompt_ids": prompt_ids, "gen_ids": gen_ids,
                      "step_token_spans": spans, "align_diag": diag, "row": row})
    save_cache_atomic(cache, out_path)
    todo = [it for it in items if it["idx"] not in cache]; n_done = 0
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
    ap.add_argument("--mode", choices=["prm", "instruct"], default="prm")
    ap.add_argument("--model", default=None)
    ap.add_argument("--adapter", choices=list(ADAPTERS), default="qwen_prm")
    ap.add_argument("--compare-to", help="T1 reference telemetry pkl; compare first 40 rows")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-samples", type=int, default=None, help="cap RAW rows (pilot only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--logprob-top-k", type=int, default=50)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--max-batch-tokens", type=int, default=16384)
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--meta-pkl", default=None, help="take PRMBench rows from this telemetry pkl (no datasets import)")
    ap.add_argument("--verify-rows", type=int, default=20)
    ap.add_argument("--prompt-style", choices=PROMPT_STYLES, default="qwen_cot_chat",
                    help="--mode instruct only: prompt for the standalone model (see the module docstring)")
    ap.add_argument("--answers", default=None, help="external label-free answers.json (Socratic, Hard2Verify); "
                    "output is external_telemetry.pkl")
    cfg = ap.parse_args()
    if cfg.mode == 'prm' and cfg.adapter == 'qwen_prm' and cfg.model and '72B' in cfg.model:
        ap.error('Use --adapter qwen_prm_72b for the 72B official template and chunked vocabulary projection')
    if cfg.mode == 'prm' and getattr(ADAPTERS[cfg.adapter], 'blocked_reason', None):
        ap.error(ADAPTERS[cfg.adapter].blocked_reason)
    cfg.model = cfg.model or (ADAPTERS[cfg.adapter].model_id if cfg.mode == "prm" else INSTRUCT_ID)
    if cfg.mode == 'instruct' and cfg.adapter != 'qwen_prm':
        ap.error('--adapter applies only to --mode prm')
    if cfg.mode == 'prm' and cfg.prompt_style != 'qwen_cot_chat':
        ap.error('--prompt-style applies only to --mode instruct')
    signal.signal(signal.SIGTERM, _on_sigterm)
    os.makedirs(cfg.out, exist_ok=True)
    if cfg.answers and cfg.meta_pkl:
        ap.error('--answers and --meta-pkl are exclusive')
    if cfg.answers:
        meta, diag = answers_rows(cfg.answers, cfg.n_samples)
    elif cfg.meta_pkl:   # rows (question, steps, ids) of an existing PRMBench telemetry cell: same rows, same order
        src = load_cache(cfg.meta_pkl)
        meta = [{k: e[k] for k in ("idx", "source_idx", "classification", "category", "question", "steps", "error_steps")}
                for _, e in sorted(src.items())][:cfg.n_samples]
        diag = {"meta_source": cfg.meta_pkl, "n_rows": len(meta)}
        del src
    else:
        meta, diag = load_prmbench(n_samples=cfg.n_samples, seed=cfg.seed)
    print(f"[prm-tel] mode={cfg.mode} model={cfg.model} rows={len(meta)}", flush=True)
    out_path = os.path.join(cfg.out, "external_telemetry.pkl" if cfg.answers else "prmbench_telemetry.pkl"); t0 = time.time(); extra = {}
    identity_path = os.path.join(cfg.out, 'run_identity.json')
    identity = {'adapter': cfg.adapter, 'mode': cfg.mode, 'model': cfg.model,
                **({'prompt_style': cfg.prompt_style} if cfg.mode == 'instruct' and cfg.prompt_style != 'qwen_cot_chat' else {}),
                'logprob_top_k': cfg.logprob_top_k, 'row_ids': [str(r['idx']) for r in meta],
                'questions_steps': [(r['question'], r['steps']) for r in meta]}
    identity = json.loads(json.dumps(identity))
    if os.path.exists(identity_path):
        with open(identity_path) as f: previous = json.load(f)
        n_previous = len(previous['row_ids'])
        fixed = ('adapter', 'mode', 'model', 'logprob_top_k', 'prompt_style')
        same_prefix = (previous['row_ids'] == identity['row_ids'][:n_previous] and
                       previous['questions_steps'] == identity['questions_steps'][:n_previous])
        if not same_prefix or any(previous.get(k) != identity.get(k) for k in fixed):
            raise ValueError('resume identity differs: model, adapter, rows, text, or top-k')
    elif os.path.exists(out_path):
        legacy = load_cache(out_path)
        for i, r in legacy.items():
            if i >= len(meta) or any(r[k] != meta[i][k] for k in ('idx', 'question', 'steps')):
                raise ValueError('legacy resume row/text mismatch')
        old_manifest = os.path.join(cfg.out, 'manifest.json')
        if not os.path.exists(old_manifest): raise ValueError('legacy cache has no model provenance; use a fresh --out')
        with open(old_manifest) as f: previous = json.load(f)
        if any(previous.get(k, 'qwen_prm' if k == 'adapter' else None) != identity[k]
               for k in ('mode', 'model', 'adapter', 'logprob_top_k')):
            raise ValueError('legacy resume model/adapter mismatch')
    with open(identity_path+'.tmp', 'w') as f: json.dump(identity, f)
    os.replace(identity_path+'.tmp', identity_path)
    if cfg.mode == "prm":
        ok, head_check, gap = run_prm(cfg, meta, out_path); extra = {"lm_head_check": head_check, "reward_path_max_abs_gap": gap}
    else:
        ok = run_instruct(cfg, meta, out_path)
    cache = load_cache(out_path)
    failures = {str(i): e for i, e in cache.items() if 'failure' in e}
    good = [e for e in cache.values() if 'failure' not in e]
    if cfg.compare_to and ok:
        extra['regression'] = compare_caches(cache, cfg.compare_to)
    extra.update(status=('COMPLETE_WITH_FAILURES' if failures else 'COMPLETE') if ok else 'INCOMPLETE', n_failed_rows=len(failures),
                 n_successful_rows=len(good), failed_rows=failures,
                 n_rows_context_overflow=sum(e.get('failure') == 'context_overflow' for e in failures.values()))
    if cfg.mode == 'prm':
        extra.update({k: head_check.get(k) for k in ('adapter', 'lm_head_source', 'lm_head_vs_base_max_abs_diff',
                                                   'hidden_after_final_norm', 'official_wrapper_commit')})
    surpr = np.concatenate([np.asarray(e["token_spilled_energies"]) for e in good]) if good else np.array([])
    ent = np.concatenate([np.asarray(e["token_entropies"]) for e in good]) if good else np.array([])
    manifest = {"driver": "run_prmbench_prm_telemetry.py", "mode": cfg.mode, "model": cfg.model,
                "dataset_diagnostics": diag, "n_rows": len(cache), "logprob_top_k": cfg.logprob_top_k,
                "system_prompt": None if cfg.mode == 'instruct' and cfg.prompt_style != 'qwen_cot_chat' else
                                 (OFFICIAL_QWEN_SYSTEM_PROMPT if cfg.adapter == 'qwen_prm_72b' and cfg.mode == 'prm' else PRM_SYSTEM_PROMPT)
                                 if cfg.mode == 'instruct' or cfg.adapter.startswith('qwen_prm') else None,
                **({"prompt_style": cfg.prompt_style, "model_revision": getattr(cfg, 'model_revision', None)}
                   if cfg.mode == 'instruct' else {}),
                "n_boundary_whitespace": int(sum(e.get('n_boundary_whitespace', 0) for e in good)),
                "token_surprisal_median": float(np.median(surpr)) if len(surpr) else None, "token_surprisal_mean": float(surpr.mean()) if len(surpr) else None,
                "token_entropy15_median": float(np.median(ent)) if len(ent) else None,
                "n_step_text_mismatch": int(sum(e.get("n_step_text_mismatch", 0) for e in good)),
                "n_rows_step_text_mismatch": int(sum(e.get("n_step_text_mismatch", 0) > 0 for e in cache.values())),
                "max_input_tokens": int(max((e["n_input_tokens"] for e in cache.values()), default=0)),
                "n_rows_over_4096_tokens": int(sum(e["n_input_tokens"] > 4096 for e in cache.values())),
                **extra, "elapsed_sec": time.time() - t0, "job_id": os.environ.get("SLURM_JOB_ID", ""),
                "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path = os.path.join(cfg.out, "manifest.json")
    with open(path + ".tmp", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    os.replace(path + ".tmp", path)
    print(json.dumps(manifest, indent=2, default=str), flush=True)
    if not ok:
        print('[prm-tel] INCOMPLETE; resubmit with the same --out to resume', flush=True)
        sys.exit(EXIT_INCOMPLETE)
    if failures:
        print(f'[prm-tel] COMPLETE_WITH_FAILURES: {len(failures)} counted rows; fusion blocked', flush=True)
        sys.exit(2)
    print(f'COMPLETE -> {cfg.out}', flush=True)
    free_memory()


if __name__ == "__main__":
    main()
